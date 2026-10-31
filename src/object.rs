//! The performance-era object: a [`Shape`] (hidden class) paired with a flat
//! vector of [`NanBox`] value slots (`ROADMAP.md` §3).
//!
//! This composes the three object-model pillars. An object holds:
//! - a shared, immutable [`Shape`] describing *where* each property lives, and
//! - a dense `Vec<NanBox>` holding the property *values* by slot index.
//!
//! Reading a property is a shape lookup (cacheable on the shape pointer) plus a
//! slot load; adding one transitions the shape and pushes a slot. Objects of the
//! same structure share a shape, so the per-object cost is just the value
//! vector — and a handle into a [`Heap`] is how other values point at it.
//!
//! An object that accumulates more own properties than the realm's
//! `object_dictionary_threshold` switches, behind this same method API, from the
//! shaped representation to a **dictionary** (`Mode::Dict`): an
//! insertion-ordered map that creates no further shape transitions. This bounds
//! the shape transition-tree's growth for programs that pile up unbounded unique
//! keys (MEM-3), at the cost of the per-shape inline-cache fast path for those
//! (now-atypical) objects.
//!
//! [`Shape`]: crate::shape::Shape
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Heap`]: crate::heap::Heap
//!
//! Pure, safe `alloc`-only Rust; this is the representation the bytecode VM will
//! migrate onto once the GC that manages the heap lands.

use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec::Vec;

/// The numeric value of `k` if it is a canonical array-index key (a non-negative
/// integer `< 2^32 - 1` whose decimal form is exactly `k`), else `None`.
fn array_index(k: &str) -> Option<u32> {
    k.parse::<u32>()
        .ok()
        .filter(|n| *n < u32::MAX && n.to_string() == k)
}

/// How an [`Object`] stores its data properties.
///
/// Objects start `Shaped` — a shared hidden-class [`Shape`] plus value slots
/// indexed by the shape's slot numbers — which is the zero-overhead
/// representation for normal objects whose structure is shared across many
/// instances. An object that accumulates more own properties than the realm's
/// `object_dictionary_threshold` converts in place to `Dict` mode, an
/// insertion-ordered map (see [`DictData`]) that adds **no** shape
/// transitions: this bounds the shape transition-tree's growth for programs
/// that pile up unbounded unique keys (MEM-3).
///
/// The mode is the object's first byte, and since only two of its 256 values
/// are valid, the rest are a niche the enclosing `Cell` encodes its other
/// variants in: a `Cell::Object` needs no tag of its own, so a heap cell is no
/// larger than an object. (Compiled code reads the cell's tag and the mode at
/// the same, probed, offset.)
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// A shape plus slots.
    Shaped = 0,
    /// A dictionary ([`ObjectExtra::dict`]).
    Dict = 1,
}

/// A dictionary-mode object's properties: values keyed by name, with a
/// parallel list preserving insertion order.
///
/// Each map entry records its key's index in `order`. `delete` leaves a
/// `None` **tombstone** at that index instead of removing it (an eager
/// removal is a linear `retain`, so deleting n keys was O(n²) — 20 000
/// deletes took 9.7 s against node's 2 ms); a re-added key is appended,
/// which is exactly the spec's "a re-created property is chronologically
/// new". Enumeration skips tombstones without consulting the map, and
/// `compact_order` drops them once they outnumber the live keys.
#[derive(Default)]
struct DictData {
    order: Vec<Option<Box<str>>>,
    map: BTreeMap<Box<str>, DictSlot>,
}

/// How many property values an object stores inline, in its own cell, before
/// spilling to a separate allocation. Most objects are small (a point, a
/// literal, a class instance with a few fields), so they need no allocation
/// beyond their heap cell — and dropping a dead one frees nothing.
pub(crate) const INLINE_SLOTS: usize = 4;

/// How many spilled slot values (past [`INLINE_SLOTS`]) the out-of-line part
/// holds in its own allocation, before the rest go to a further `Vec`: slot
/// `i` in `INLINE_SLOTS..INLINE_SLOTS + SPILL_NEAR` is one load away from
/// the object (its `extra` pointer) rather than two, and an object of up to
/// `INLINE_SLOTS + SPILL_NEAR` properties needs no allocation beyond its
/// `extra` part.
pub(crate) const SPILL_NEAR: usize = 8;

/// The most slots a shaped object holds (its count is a `u16`); an object
/// that would grow past it becomes a dictionary, whatever the realm's
/// threshold.
const MAX_SHAPED_LEN: usize = u16::MAX as usize;

/// A dictionary-mode property: its value and its key's index in the insertion
/// `order` (see [`DictData`]).
#[derive(Clone, Copy)]
struct DictSlot {
    value: NanBox,
    pos: usize,
}

/// [`Object`]'s `proto` word for "no prototype" (no handle packs to it: a
/// raw handle uses only the low 48 bits).
pub(crate) const NO_PROTO: u64 = u64::MAX;

/// [`Object::jit_flags`] bit: a host-exotic object (see
/// `Realm::mark_host_exotic`) — compiled code reads none of its properties
/// inline.
pub const JIT_FLAG_EXOTIC: u8 = 1;
/// [`Object::jit_flags`] bit: an object whose property writes need more than
/// the ordinary store (a global object, whose writes reach the global
/// bindings; a built-in prototype whose method overwrites are tracked) —
/// compiled code writes none of its properties inline.
pub const JIT_FLAG_NO_INLINE_SET: u8 = 2;
/// [`Object::jit_flags`] bit: the object has (or once had) a non-writable own
/// property, so compiled code's in-place store must go to the helper (which
/// checks the key). Set by [`Object::set_readonly`], never cleared.
pub const JIT_FLAG_READONLY: u8 = 4;
/// [`Object::jit_flags`] bit: a realm's global object. A change to its own
/// keys also bumps the global binding epoch (`crate::env`), so compiled code
/// that validated "the global object has no such own property" for a global
/// write stops relying on it. Set once, when the global object is made.
pub const JIT_FLAG_GLOBAL: u8 = 8;
/// [`Object::jit_flags`] bit: an object some prototype-chain cache recorded
/// (see the prototype epoch, `proto_epoch`). A change to its layout — its shape, key epoch,
/// `[[Prototype]]`, representation or other `jit_flags` — bumps the process-
/// wide prototype epoch, which retires every such cache's "validated at
/// epoch E" stamp. Set when a cache records the object, never cleared;
/// compiled code adds no property to such an object inline (its slow path
/// does the bump), but may still store to its existing data properties.
pub const JIT_FLAG_WATCHED: u8 = 16;
/// The [`JIT_FLAG_EXOTIC`]..[`JIT_FLAG_WATCHED`] bits of an object's flags
/// byte (what [`Object::jit_flags`] reports).
const JIT_FLAGS_MASK: u8 = 0x1f;
/// Flags-byte bit: the object is frozen (`Object.freeze`): no new properties
/// and no writes to existing ones. Always set together with
/// [`OBJ_FLAG_SEALED`] and [`OBJ_FLAG_NON_EXTENSIBLE`].
pub(crate) const OBJ_FLAG_FROZEN: u8 = 0x20;
/// Flags-byte bit: the object is sealed (`Object.seal`): no new properties
/// and no deletions, but existing writable properties may still change.
/// Always set together with [`OBJ_FLAG_NON_EXTENSIBLE`].
pub(crate) const OBJ_FLAG_SEALED: u8 = 0x40;
/// Flags-byte bit: new properties may not be added
/// (`Object.preventExtensions`). Stored inverted so an ordinary object's
/// flags byte is `0`.
pub(crate) const OBJ_FLAG_NON_EXTENSIBLE: u8 = 0x80;

/// The prototype epoch: bumped whenever an object flagged
/// [`JIT_FLAG_WATCHED`] changes layout (see [`Object`]'s layout hooks) and
/// whenever two heap cells swap contents. A prototype-chain cache that
/// validated its whole recorded chain while the epoch was `E` — every object
/// on it being watched — may skip re-walking the chain for as long as the
/// epoch is still `E`. Process-wide, so a change on any thread merely
/// retires every stamp. Starts at 1; `0` is the "never validated" stamp.
static PROTO_EPOCH: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// The current [`PROTO_EPOCH`] value.
#[inline]
#[must_use]
pub(crate) fn proto_epoch() -> u64 {
    PROTO_EPOCH.load(core::sync::atomic::Ordering::Relaxed)
}

/// Retires every prototype-chain stamp (see [`PROTO_EPOCH`]).
#[inline]
pub(crate) fn bump_proto_epoch() {
    PROTO_EPOCH.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// The address of [`PROTO_EPOCH`], which compiled code reads with a plain
/// load (an `AtomicU64` has `u64`'s layout).
#[must_use]
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
pub(crate) fn proto_epoch_addr() -> usize {
    PROTO_EPOCH.as_ptr() as usize
}

/// A property-bearing object: a hidden-class shape plus its value slots (or a
/// dictionary once it grows past the threshold), with an optional side list of
/// accessor (getter/setter) properties.
///
/// 64 bytes: a header word (mode, flags, slot count, key epoch), the shape,
/// the prototype, the out-of-line part and four inline values. Slot
/// `i < INLINE_SLOTS` is `inline[i]`; the rest live in the out-of-line part's
/// `spill` (its near part, then a far `Vec`). `len` counts the slots
/// (always the shape's length in shaped mode, `0` in dictionary mode);
/// inline entries at or past it are unused (never read or traced).
#[repr(C)] // compiled code reads the fields at probed offsets; `mode` must be first (the niche)
pub struct Object {
    /// Shaped or dictionary storage. First, so `Cell` can use its niche.
    mode: Mode,
    /// [`JIT_FLAG_EXOTIC`] and the other generic-JIT bits (tested by the
    /// JIT's inline property paths before touching the object; only ever set,
    /// never cleared, so a stale bit merely sends an access to the always
    /// correct helper), plus the integrity bits ([`OBJ_FLAG_FROZEN`],
    /// [`OBJ_FLAG_SEALED`], [`OBJ_FLAG_NON_EXTENSIBLE`]). `0` for an
    /// ordinary, extensible object.
    flags: u8,
    /// The number of slots (shaped mode).
    len: u16,
    /// Bumped whenever the set of own keys (data or accessor) may have changed,
    /// so a cache of "does this object own `k`" can be revalidated by a compare
    /// (see [`key_epoch`](Object::key_epoch)).
    key_epoch: u32,
    /// The hidden class (shaped mode), or an empty sentinel shape (dictionary
    /// mode, so inline caches keyed on it always miss).
    shape: Rc<Shape>,
    /// The `[[Prototype]]` link (`Object.create`/`getPrototypeOf`) as a raw
    /// handle ([`Handle::to_raw`](crate::heap::Handle::to_raw)), or
    /// [`NO_PROTO`] for none. A property miss walks this chain. Stored raw (one
    /// word) so compiled code can compare it with a cached handle.
    proto: u64,
    /// The rarely-used parts of an object (spilled slots, the dictionary,
    /// accessors, attribute marks, the unified key order, the class tag),
    /// boxed so an ordinary data object — the overwhelmingly common case —
    /// stays small. `None` means all empty.
    extra: Option<Box<ObjectExtra>>,
    /// The first [`INLINE_SLOTS`] slot values.
    inline: [NanBox; INLINE_SLOTS],
}

// A heap cell is an object (no separate tag; see [`Mode`]), and both fit a
// cache line.
const _: () = assert!(core::mem::size_of::<Object>() == 64);

/// An object's slot values past the inline room (shaped mode), in slot
/// order: spilled value `i` (slot `INLINE_SLOTS + i`) is `near[i]` for
/// `i < SPILL_NEAR`, else `far[i - SPILL_NEAR]`. `near` entries at or past
/// `n` are unused (never read or traced).
struct Spill {
    /// The first [`SPILL_NEAR`] spilled values, in the `extra` allocation
    /// itself (compiled code reads them at a fixed offset from it).
    near: [NanBox; SPILL_NEAR],
    /// The spilled values past `near`.
    far: Vec<NanBox>,
    /// How many values are spilled.
    n: u32,
}

impl Spill {
    /// No spilled values.
    const EMPTY: Self = Self {
        near: [NanBox::undefined(); SPILL_NEAR],
        far: Vec::new(),
        n: 0,
    };

    /// Whether no value is spilled.
    #[inline]
    fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The spilled values in use in `near`.
    #[inline]
    fn near_live(&self) -> &[NanBox] {
        &self.near[..(self.n as usize).min(SPILL_NEAR)]
    }

    /// Spilled value `i`, if in range.
    #[inline]
    fn get(&self, i: usize) -> Option<&NanBox> {
        if i >= self.n as usize {
            None
        } else if i < SPILL_NEAR {
            Some(&self.near[i])
        } else {
            self.far.get(i - SPILL_NEAR)
        }
    }

    /// Spilled value `i`, mutably, if in range.
    #[inline]
    fn get_mut(&mut self, i: usize) -> Option<&mut NanBox> {
        if i >= self.n as usize {
            None
        } else if i < SPILL_NEAR {
            Some(&mut self.near[i])
        } else {
            self.far.get_mut(i - SPILL_NEAR)
        }
    }

    /// Appends `v` as the next spilled value.
    #[inline]
    fn push(&mut self, v: NanBox) {
        let i = self.n as usize;
        if i < SPILL_NEAR {
            self.near[i] = v;
        } else {
            self.far.push(v);
        }
        self.n += 1;
    }

    /// Drops every spilled value (keeping `far`'s buffer).
    fn clear(&mut self) {
        self.n = 0;
        self.far.clear();
    }

    /// The spilled values, in order.
    #[inline]
    fn iter(&self) -> impl Iterator<Item = &NanBox> {
        self.near_live().iter().chain(self.far.iter())
    }

    /// The spilled values, mutably, in order.
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut NanBox> {
        let n = (self.n as usize).min(SPILL_NEAR);
        self.near[..n].iter_mut().chain(self.far.iter_mut())
    }
}

impl Default for Spill {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// The out-of-line part of an [`Object`]: everything an ordinary data object
/// leaves empty. Allocated on first use (see `Object::extra_mut`).
#[derive(Default)]
struct ObjectExtra {
    /// Slot values past the inline room (shaped mode).
    spill: Spill,
    /// The properties of a dictionary-mode object.
    dict: Option<Box<DictData>>,
    /// Own keys that are **non-writable** (`defineProperty` with
    /// `writable: false`): writes are silently ignored.
    readonly: Vec<alloc::boxed::Box<str>>,
    /// Accessor properties: `(name, getter, setter)`, both held as value
    /// handles (`undefined` when absent). Kept out of the shape's slot layout.
    accessors: Vec<(alloc::boxed::Box<str>, NanBox, NanBox)>,
    /// Own keys that are **non-enumerable** (e.g. class methods): present in the
    /// slots and readable, but hidden from `Object.keys`/spread/`for-in`/JSON.
    hidden: Vec<alloc::boxed::Box<str>>,
    /// Own keys that are **non-configurable** (`defineProperty` with
    /// `configurable: false`): they cannot be deleted.
    non_configurable: Vec<alloc::boxed::Box<str>>,
    /// The class this object was instantiated from (for `instanceof`), if any.
    class_tag: Option<u32>,
    /// How many of the first slots (shaped mode) hold **non-enumerable**
    /// keys not listed in `hidden`: a built-in prototype or namespace marks
    /// each method hidden right after adding it, which only moves this count
    /// along (no list entry, no scan). Written out to `hidden` (and reset)
    /// before anything renumbers the slots (a delete, dictionary mode) or
    /// clears one of these marks.
    hidden_prefix: u32,
    /// Unified own-key insertion order across **both** data and accessor
    /// properties. `None` on the fast path (a pure data object, or one that has
    /// never held an accessor): enumeration then derives its order from the shape
    /// (data keys) followed by the accessor side-list, which is correct when the
    /// two never interleave. The first `define_accessor` seeds this to the current
    /// data keys and every subsequent new key (data or accessor) is appended, so
    /// `[[OwnPropertyKeys]]` reports the true chronological creation order even when
    /// a getter/setter is defined between two data properties (or an accessor is
    /// later redefined as data, keeping its original slot). Boxed, so
    /// `None` is a null word (compiled code's inline adds test it).
    #[allow(clippy::box_collection)] // for the null word
    key_order: Option<Box<Vec<Box<str>>>>,
    /// A by-name index over the lists above (see [`AttrIndex`]), built once
    /// one holds more than [`ATTR_INDEX_MIN`] entries and kept in step
    /// with them from then on; `None` while they are short enough to search.
    index: Option<Box<AttrIndex>>,
}

/// Which attribute-mark list of an [`ObjectExtra`] (an index into
/// [`AttrEntry::marks`]).
#[derive(Clone, Copy)]
enum Mark {
    Readonly = 0,
    Hidden = 1,
    NonConfigurable = 2,
}

/// The length of one of an object's attribute-mark or accessor lists past
/// which the object gets an [`AttrIndex`]: below it a linear search is about
/// as fast, and costs no memory (built-in prototypes, with a few dozen
/// hidden methods each, stay unindexed).
const ATTR_INDEX_MIN: usize = 128;

/// The key-order log length past which the object gets an [`AttrIndex`]
/// (the log is only scanned when a key is added).
const ORDER_INDEX_MIN: usize = 256;

/// A single attribute list this short is scanned even when indexed: a few
/// string compares beat a tree walk (a built-in prototype with many hidden
/// methods but one accessor is indexed, and its accessor probed per miss).
const ATTR_SCAN_MAX: usize = 8;

/// Where each key stands in an [`ObjectExtra`]'s attribute lists, so that
/// "is `k` non-writable / non-enumerable / non-configurable", "`k`'s
/// accessor" and "is `k` in the key-order log" are a map lookup rather than
/// a scan: an object with thousands of `defineProperty`'d or accessor
/// properties (a bundled module's export object) made every one of those
/// O(n), and defining, reading or enumerating them O(n^2).
///
/// The lists stay the source of truth (compiled code reads their lengths,
/// and the accessor list by position); this mirrors them.
#[derive(Default)]
struct AttrIndex {
    map: BTreeMap<Box<str>, AttrEntry>,
}

/// A key's positions in its object's attribute lists (see [`AttrIndex`]).
#[derive(Clone, Copy, Default)]
struct AttrEntry {
    /// One more than the key's position in the readonly / hidden /
    /// non-configurable list ([`Mark`] order); `0` when not marked.
    marks: [u32; 3],
    /// One more than the key's position in the accessor list; `0` for none.
    acc: u32,
    /// Whether the key-order log records the key.
    ordered: bool,
}

impl AttrIndex {
    /// The entry for `key`, created empty if absent.
    fn entry(&mut self, key: &str) -> &mut AttrEntry {
        if !self.map.contains_key(key) {
            self.map.insert(Box::from(key), AttrEntry::default());
        }
        self.map.get_mut(key).expect("just inserted")
    }

    /// The entry for `key`, if any.
    fn get(&self, key: &str) -> Option<&AttrEntry> {
        self.map.get(key)
    }
}

impl ObjectExtra {
    /// Whether nothing is recorded (the object could drop its box).
    fn is_empty(&self) -> bool {
        self.spill.is_empty() && self.only_spill()
    }

    /// Whether nothing but spilled slot values is recorded.
    fn only_spill(&self) -> bool {
        self.dict.is_none()
            && self.readonly.is_empty()
            && self.accessors.is_empty()
            && self.hidden.is_empty()
            && self.non_configurable.is_empty()
            && self.class_tag.is_none()
            && self.hidden_prefix == 0
            && self.key_order.is_none()
            && self.index.is_none()
    }

    /// The mark list `m`.
    fn marks(&self, m: Mark) -> &Vec<Box<str>> {
        match m {
            Mark::Readonly => &self.readonly,
            Mark::Hidden => &self.hidden,
            Mark::NonConfigurable => &self.non_configurable,
        }
    }

    /// The mark list `m`, mutably.
    fn marks_mut(&mut self, m: Mark) -> &mut Vec<Box<str>> {
        match m {
            Mark::Readonly => &mut self.readonly,
            Mark::Hidden => &mut self.hidden,
            Mark::NonConfigurable => &mut self.non_configurable,
        }
    }

    /// Whether `key` carries mark `m`.
    #[inline]
    fn has_mark(&self, m: Mark, key: &str) -> bool {
        let list = self.marks(m);
        match &self.index {
            Some(ix) if list.len() > ATTR_SCAN_MAX => {
                ix.get(key).is_some_and(|e| e.marks[m as usize] != 0)
            }
            _ => list.iter().any(|k| k.as_ref() == key),
        }
    }

    /// Gives `key` (unmarked) mark `m`.
    #[inline]
    fn add_mark(&mut self, m: Mark, key: &str) {
        let list = self.marks_mut(m);
        list.push(Box::from(key));
        let n = list.len() as u32;
        match self.index.as_deref_mut() {
            Some(ix) => ix.entry(key).marks[m as usize] = n,
            None if n as usize > ATTR_INDEX_MIN => self.build_index(),
            None => {}
        }
    }

    /// Removes mark `m` from `key`, if it has it.
    fn remove_mark(&mut self, m: Mark, key: &str) {
        let Some(ix) = self.index.as_deref_mut() else {
            self.marks_mut(m).retain(|k| k.as_ref() != key);
            return;
        };
        let Some(e) = ix.map.get_mut(key) else {
            return;
        };
        let pos = core::mem::take(&mut e.marks[m as usize]);
        if pos == 0 {
            return;
        }
        // Unordered: the last entry fills the gap.
        let list = match m {
            Mark::Readonly => &mut self.readonly,
            Mark::Hidden => &mut self.hidden,
            Mark::NonConfigurable => &mut self.non_configurable,
        };
        let i = pos as usize - 1;
        list.swap_remove(i);
        if let Some(moved) = list.get(i)
            && let Some(e) = ix.map.get_mut(moved.as_ref())
        {
            e.marks[m as usize] = pos;
        }
    }

    /// The position of `key`'s accessor in the accessor list.
    #[inline]
    fn acc_pos(&self, key: &str) -> Option<usize> {
        match &self.index {
            _ if self.accessors.len() <= ATTR_SCAN_MAX => self
                .accessors
                .iter()
                .position(|(k, _, _)| k.as_ref() == key),
            Some(ix) => ix
                .get(key)
                .and_then(|e| (e.acc != 0).then(|| e.acc as usize - 1)),
            None => self
                .accessors
                .iter()
                .position(|(k, _, _)| k.as_ref() == key),
        }
    }

    /// Appends an accessor for `key` (which has none).
    fn add_acc(&mut self, key: &str, getter: NanBox, setter: NanBox) {
        self.accessors.push((Box::from(key), getter, setter));
        let n = self.accessors.len() as u32;
        match self.index.as_deref_mut() {
            Some(ix) => ix.entry(key).acc = n,
            None if n as usize > ATTR_INDEX_MIN => self.build_index(),
            None => {}
        }
    }

    /// Removes `key`'s accessor, keeping the others in order; whether it had
    /// one.
    fn remove_acc(&mut self, key: &str) -> bool {
        let Some(i) = self.acc_pos(key) else {
            return false;
        };
        self.accessors.remove(i);
        if let Some(ix) = self.index.as_deref_mut() {
            // Every later accessor moves down one place (O(n), like the
            // removal itself).
            for e in ix.map.values_mut() {
                if e.acc as usize > i + 1 {
                    e.acc -= 1;
                } else if e.acc as usize == i + 1 {
                    e.acc = 0;
                }
            }
        }
        true
    }

    /// Whether the key-order log records `key` (`false` without a log).
    fn order_has(&self, key: &str) -> bool {
        match (&self.index, &self.key_order) {
            (_, None) => false,
            (Some(ix), Some(_)) => ix.get(key).is_some_and(|e| e.ordered),
            (None, Some(order)) => order.iter().any(|k| k.as_ref() == key),
        }
    }

    /// Appends `key` (unrecorded) to the key-order log, which must exist.
    fn order_push(&mut self, key: &str) {
        let Some(order) = self.key_order.as_mut() else {
            return;
        };
        order.push(Box::from(key));
        let n = order.len();
        match self.index.as_deref_mut() {
            Some(ix) => ix.entry(key).ordered = true,
            None if n > ORDER_INDEX_MIN => self.build_index(),
            None => {}
        }
    }

    /// Drops `key` from the key-order log, if there is one.
    fn order_remove(&mut self, key: &str) {
        let Some(order) = self.key_order.as_mut() else {
            return;
        };
        if let Some(ix) = self.index.as_deref_mut() {
            match ix.map.get_mut(key) {
                Some(e) if e.ordered => e.ordered = false,
                _ => return,
            }
        }
        order.retain(|k| k.as_ref() != key);
    }

    /// Starts the key-order log as `order` (there is none yet).
    fn set_order(&mut self, order: Vec<Box<str>>) {
        if let Some(ix) = self.index.as_deref_mut() {
            for k in &order {
                ix.entry(k).ordered = true;
            }
        }
        let n = order.len();
        self.key_order = Some(Box::new(order));
        if self.index.is_none() && n > ORDER_INDEX_MIN {
            self.build_index();
        }
    }

    /// Builds the [`AttrIndex`] from the lists.
    #[cold]
    #[inline(never)]
    fn build_index(&mut self) {
        let mut ix = AttrIndex::default();
        for m in [Mark::Readonly, Mark::Hidden, Mark::NonConfigurable] {
            for (i, k) in self.marks(m).iter().enumerate() {
                ix.entry(k).marks[m as usize] = i as u32 + 1;
            }
        }
        for (i, (k, _, _)) in self.accessors.iter().enumerate() {
            ix.entry(k).acc = i as u32 + 1;
        }
        for k in self.key_order.iter().flat_map(|o| o.iter()) {
            ix.entry(k).ordered = true;
        }
        self.index = Some(Box::new(ix));
    }
}

/// The all-empty [`ObjectExtra`] that reads see when an object has none.
static EMPTY_EXTRA: ObjectExtra = ObjectExtra {
    spill: Spill::EMPTY,
    dict: None,
    readonly: Vec::new(),
    accessors: Vec::new(),
    hidden: Vec::new(),
    non_configurable: Vec::new(),
    class_tag: None,
    hidden_prefix: 0,
    key_order: None,
    index: None,
};

/// An empty dictionary, for reads of a dictionary-mode object's map.
static EMPTY_DICT: DictData = DictData {
    order: Vec::new(),
    map: BTreeMap::new(),
};

/// Drops an object's `extra` part — off the sweep's common path.
#[cold]
#[inline(never)]
fn drop_extra(extra: Box<ObjectExtra>) {
    drop(extra);
}

impl ObjectExtra {
    /// Whether this part, once its spill count and class tag are reset,
    /// is exactly a fresh one owning no memory but itself: no dictionary,
    /// no attribute-mark, accessor or key-order storage, no far spill.
    fn recyclable(&self) -> bool {
        self.dict.is_none()
            && self.key_order.is_none()
            && self.index.is_none()
            && self.spill.far.capacity() == 0
            && self.readonly.capacity() == 0
            && self.accessors.capacity() == 0
            && self.hidden.capacity() == 0
            && self.non_configurable.capacity() == 0
    }
}

/// Emptied `extra` parts of swept objects, kept for reuse so that an object
/// growing past its inline slots does not pay an allocation (nor its death a
/// free). Every part here reads as `ObjectExtra::default()` does: nothing
/// recorded, no spilled value (the near spill's stale words are past the
/// count, so never read or traced), owning no memory but itself. Compiled
/// code's inline property add pops one for an object's first spilled slot
/// (see `ExtraPool::jit_offsets`); the collector's sweep refills it.
#[derive(Default)]
pub struct ExtraPool {
    // Boxed: compiled code moves a part's pointer straight into an object's
    // `extra` field (an `Option<Box<ObjectExtra>>`).
    #[allow(clippy::vec_box)]
    boxes: Vec<Box<ObjectExtra>>,
}

impl core::fmt::Debug for ExtraPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExtraPool")
            .field("len", &self.boxes.len())
            .finish()
    }
}

impl ExtraPool {
    /// The most parts kept.
    pub const CAP: usize = 16_384;

    /// The number of parts kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    /// Whether no part is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    /// Keeps a dying object's `extra` part when it can be reset to a fresh
    /// one in place (and there is room); drops it otherwise. Out of line,
    /// so the sweep's common case (no `extra` part) stays small.
    #[inline(never)]
    fn give(&mut self, mut x: Box<ObjectExtra>) {
        if self.boxes.len() < Self::CAP && x.recyclable() {
            x.spill.n = 0;
            x.class_tag = None;
            x.hidden_prefix = 0;
            self.boxes.push(x);
        } else {
            drop_extra(x);
        }
    }

    /// Byte offsets, within an `ExtraPool`, of its `Vec`'s data pointer and
    /// length, for compiled code's inline pop: `(ptr, len)`. `vec_ptr_off`
    /// and `vec_len_off` are a `Vec`'s probed word offsets
    /// ([`ObjectLayout`]), the same whatever its element type.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn jit_offsets(vec_ptr_off: usize, vec_len_off: usize) -> (usize, usize) {
        let f = core::mem::offset_of!(ExtraPool, boxes);
        (f + vec_ptr_off, f + vec_len_off)
    }

    /// The pool's `Vec` data pointer and length, for the layout test.
    #[cfg(all(test, feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn raw_parts(&self) -> (usize, usize) {
        (self.boxes.as_ptr() as usize, self.boxes.len())
    }
}

impl Object {
    /// Drops the object, the way the collector's sweep does it: the common
    /// case (no `extra` part) is one shape-count decrement, inline; an
    /// `extra` part goes back to `pool` when it can.
    #[inline(always)]
    pub(crate) fn release(self, pool: &mut ExtraPool) {
        let Self { shape, extra, .. } = self;
        drop(shape);
        if let Some(x) = extra {
            pool.give(x);
        }
    }

    /// Creates an empty object whose layout starts at `root` (the shared root
    /// shape of the owning realm/heap, so identically-structured objects share
    /// shapes).
    #[must_use]
    pub fn new(root: Rc<Shape>) -> Self {
        Self {
            mode: Mode::Shaped,
            flags: 0,
            len: 0,
            key_epoch: 0,
            shape: root,
            proto: NO_PROTO,
            extra: None,
            inline: [NanBox::undefined(); INLINE_SLOTS],
        }
    }

    /// The object that adding each key of `shape` in turn, as a plain data
    /// property, to `Object::new(root)` leaves (`values` in slot order) — for
    /// a caller that builds many objects with the same keys in one step. The
    /// values past the inline room go to a spill sized for exactly them.
    /// `None` unless `shape` holds exactly `values.len()` keys.
    #[must_use]
    #[inline(always)]
    pub fn with_data(shape: Rc<Shape>, values: &[NanBox]) -> Option<Self> {
        if values.len() > MAX_SHAPED_LEN || shape.len() as usize != values.len() {
            return None;
        }
        // The layout's record of which slots may hold a non-Number: only
        // the slots it still has as Number-only need a look.
        let mut pending = shape.number_only_slots();
        while pending != 0 {
            let i = pending.trailing_zeros() as usize;
            pending &= pending - 1;
            if values.get(i).is_some_and(|v| !v.is_number()) {
                shape.note_non_number(i as u32);
            }
        }
        Some(Self {
            mode: Mode::Shaped,
            flags: 0,
            len: values.len() as u16,
            key_epoch: values.len() as u32,
            shape,
            proto: NO_PROTO,
            extra: values.get(INLINE_SLOTS..).and_then(Self::spill_part),
            // Word by word: a variable-length slice copy into the (moved)
            // temporary would cost a store-forwarding stall per word.
            inline: core::array::from_fn(|i| values.get(i).copied().unwrap_or(NanBox::undefined())),
        })
    }

    /// [`with_data`](Self::with_data) with the first `hidden` slots marked
    /// non-enumerable — what marking each of those keys hidden right after
    /// adding it leaves (the hidden prefix; no mark lists).
    #[must_use]
    pub fn with_data_hidden_prefix(
        shape: Rc<Shape>,
        values: &[NanBox],
        hidden: u32,
    ) -> Option<Self> {
        if hidden as usize > values.len() {
            return None;
        }
        let mut o = Self::with_data(shape, values)?;
        if hidden != 0 {
            o.extra_mut().hidden_prefix = hidden;
        }
        Some(o)
    }

    /// A built-in function's own-property object: `length` then `name`, both
    /// `{ writable: false, enumerable: false, configurable: true }`, the
    /// attributes held by `shape` (a [`Shape::builtin_fn_layout`]) rather
    /// than by attribute-mark lists, so making one allocates nothing past the
    /// cell. The first change that the layout alone cannot describe (a new
    /// key, a delete, a cleared mark, an accessor, dictionary mode) first
    /// writes the marks out and moves to the ordinary layout (see
    /// `materialize_implied_attrs`); until then every attribute query answers
    /// as though the marks were listed.
    #[must_use]
    pub fn builtin_fn_props(shape: Rc<Shape>, length: NanBox, name: NanBox) -> Self {
        debug_assert!(shape.has_implied_fn_attrs());
        let mut o = Self::with_data(shape, &[length, name]).expect("a {length, name} layout");
        // What `set_readonly` leaves: compiled code's in-place store goes to
        // the helper, which checks the key.
        o.flags |= JIT_FLAG_READONLY;
        o
    }

    /// Whether the object's layout implies its `length`/`name` marks (see
    /// [`builtin_fn_props`](Self::builtin_fn_props)).
    #[inline]
    fn implied_attrs(&self) -> bool {
        !self.is_dict() && self.shape.has_implied_fn_attrs()
    }

    /// Whether `key` is non-writable and non-enumerable by layout alone.
    #[inline]
    fn implied_mark(&self, key: &str) -> bool {
        self.implied_attrs() && (key == "length" || key == "name")
    }

    /// Writes the marks a [`builtin_fn_props`](Self::builtin_fn_props)
    /// layout implies into the attribute lists and moves the object to the
    /// ordinary layout with the same slots, before a change the layout cannot
    /// describe. A no-op for any other object.
    #[inline]
    fn materialize_implied_attrs(&mut self) {
        if self.implied_attrs() {
            self.write_out_implied_attrs();
        }
    }

    #[cold]
    #[inline(never)]
    fn write_out_implied_attrs(&mut self) {
        let x = self.extra_mut();
        for k in ["length", "name"] {
            x.add_mark(Mark::Readonly, k);
        }
        // Slots 0 and 1: the two keys are the start of a hidden prefix,
        // which a constructor's `prototype` and static methods extend.
        debug_assert_eq!(x.hidden_prefix, 0);
        x.hidden_prefix = 2;
        self.layout_changed();
        self.shape = self.shape.unmarked_twin();
        // Another layout than one this object grew into: it learns the
        // object's values.
        self.note_all_slots();
    }

    /// Records on the object's layout which of its slots hold a value that
    /// is not a Number (`Shape::note_non_number`) — for a layout the object
    /// takes other than by adding a key.
    fn note_all_slots(&self) {
        if self.is_dict() {
            return;
        }
        let mut non_numbers = 0u64;
        for (i, v) in self.slots_iter().take(64).enumerate() {
            non_numbers |= u64::from(!v.is_number()) << i;
        }
        self.shape.note_non_numbers(non_numbers);
    }

    /// Records a write of `v` into slot `slot` on the object's layout
    /// (`Shape::note_non_number`): every slot write goes through here.
    #[inline]
    fn note_slot(&self, slot: u32, v: NanBox) {
        if !v.is_number() {
            self.shape.note_non_number(slot);
        }
    }

    /// The out-of-line part holding just `rest` as the spilled slot values
    /// (none for no values).
    #[inline(never)]
    fn spill_part(rest: &[NanBox]) -> Option<Box<ObjectExtra>> {
        if rest.is_empty() {
            return None;
        }
        let mut x = Box::<ObjectExtra>::default();
        for (d, &v) in x.spill.near.iter_mut().zip(rest) {
            *d = v;
        }
        if let Some(far) = rest.get(SPILL_NEAR..) {
            x.spill.far = far.to_vec();
        }
        x.spill.n = rest.len() as u32;
        Some(x)
    }

    /// [`is_plain_shaped`](Self::is_plain_shaped), with a key set that has
    /// changed once per key: what adding each key of its shape in turn to an
    /// empty object leaves, so [`with_data`](Self::with_data) of its shape
    /// and slot values makes the same object.
    #[must_use]
    pub fn is_plain_fresh_data(&self) -> bool {
        self.is_plain_shaped() && self.key_epoch == u32::from(self.len)
    }

    /// Whether this is an ordinary shaped object holding only plain data
    /// properties (the shape alone describes its keys and slots; its `extra`
    /// part, if any, holds only spilled slot values).
    #[must_use]
    pub fn is_plain_shaped(&self) -> bool {
        !self.is_dict()
            && self.extra.as_deref().is_none_or(ObjectExtra::only_spill)
            && self.flags == 0
            && self.shape.len() as usize == self.len as usize
    }

    /// A counter that changes whenever this object's set of own keys (data or
    /// accessor) may have changed: equal values at two moments mean the same
    /// own keys. (A `u32` that would need 2³² key changes between two reads to
    /// alias.)
    #[must_use]
    pub fn key_epoch(&self) -> u32 {
        self.key_epoch
    }

    /// Bumps [`key_epoch`](Self::key_epoch) — and, for a global object
    /// ([`JIT_FLAG_GLOBAL`]), the global binding epoch, since whether it owns a
    /// like-named property decides how a global write behaves.
    #[inline]
    fn bump_key_epoch(&mut self) {
        self.key_epoch = self.key_epoch.wrapping_add(1);
        if self.flags & (JIT_FLAG_GLOBAL | JIT_FLAG_WATCHED) != 0 {
            if self.flags & JIT_FLAG_GLOBAL != 0 {
                crate::env::bump_global_binding_epoch();
            }
            self.layout_changed();
        }
    }

    /// The layout hook: called on every change to what a prototype-chain
    /// cache validates of an object — its shape (every shape change also
    /// bumps the key epoch), key epoch, `[[Prototype]]`, representation
    /// (shaped or dictionary) and `jit_flags` — and bumps the prototype epoch
    /// when the object is [`JIT_FLAG_WATCHED`].
    #[inline]
    fn layout_changed(&self) {
        if self.flags & JIT_FLAG_WATCHED != 0 {
            bump_proto_epoch();
        }
    }

    /// Flags the object [`JIT_FLAG_WATCHED`] (a prototype-chain cache is
    /// recording it). Not itself a layout change.
    pub(crate) fn watch(&mut self) {
        self.flags |= JIT_FLAG_WATCHED;
    }

    /// Whether the object is in dictionary mode.
    #[inline]
    fn is_dict(&self) -> bool {
        self.mode == Mode::Dict
    }

    /// The dictionary (dictionary mode; else an empty one).
    #[inline]
    fn dict(&self) -> &DictData {
        self.ex().dict.as_deref().unwrap_or(&EMPTY_DICT)
    }

    /// The dictionary, mutably (dictionary mode only).
    #[inline]
    fn dict_mut(&mut self) -> &mut DictData {
        debug_assert!(self.is_dict());
        self.extra_mut().dict.get_or_insert_with(Default::default)
    }

    /// The spilled slot values (slot `INLINE_SLOTS + i` at `get(i)`).
    #[inline]
    fn spill(&self) -> &Spill {
        &self.ex().spill
    }

    /// Slot `i`, if in range (shaped mode; a dictionary has no slots).
    #[inline]
    fn slot_get(&self, i: u32) -> Option<NanBox> {
        if i >= u32::from(self.len) {
            return None;
        }
        let i = i as usize;
        if i < INLINE_SLOTS {
            Some(self.inline[i])
        } else {
            self.spill().get(i - INLINE_SLOTS).copied()
        }
    }

    /// Slot `i`, mutably, if in range.
    #[inline]
    fn slot_get_mut(&mut self, i: u32) -> Option<&mut NanBox> {
        if i >= u32::from(self.len) {
            return None;
        }
        let i = i as usize;
        if i < INLINE_SLOTS {
            Some(&mut self.inline[i])
        } else {
            self.extra
                .as_deref_mut()
                .and_then(|x| x.spill.get_mut(i - INLINE_SLOTS))
        }
    }

    /// Slot `i`, which must be in range.
    #[inline]
    fn slot_at(&self, i: u32) -> NanBox {
        self.slot_get(i).expect("slot index in range")
    }

    /// Appends `v` as the next slot (shaped mode; `len < MAX_SHAPED_LEN`).
    #[inline(always)]
    fn slot_push(&mut self, v: NanBox) {
        let i = self.len as usize;
        debug_assert!(i < MAX_SHAPED_LEN);
        if i < INLINE_SLOTS {
            self.inline[i] = v;
        } else {
            self.spill_push(v);
        }
        self.len += 1;
        self.note_slot(i as u32, v);
    }

    /// [`slot_push`](Self::slot_push) past the inline room.
    #[inline(never)]
    fn spill_push(&mut self, v: NanBox) {
        self.extra_mut().spill.push(v);
    }

    /// The slot values, in slot order.
    #[inline]
    fn slots_iter(&self) -> impl Iterator<Item = &NanBox> {
        let spill = self.spill();
        self.inline[..(self.len as usize).min(INLINE_SLOTS)]
            .iter()
            .chain(spill.iter())
    }

    /// Replaces the slots with `values`, in order.
    fn set_slots(&mut self, values: Vec<NanBox>) {
        self.len = 0;
        if let Some(x) = self.extra.as_deref_mut() {
            x.spill.clear();
        }
        for v in values {
            self.slot_push(v);
        }
    }

    /// Drops the out-of-line part once it records nothing, so the object
    /// qualifies for compiled code's inline adds again.
    fn tidy_extra(&mut self) {
        if self.extra.as_deref().is_some_and(ObjectExtra::is_empty) {
            self.extra = None;
        }
    }

    /// Adds `key` (absent) with `value`: a shape transition and a new slot, or
    /// a dictionary entry. A shaped object already at [`MAX_SHAPED_LEN`]
    /// becomes a dictionary first.
    fn add_new(&mut self, key: &str, value: NanBox) {
        self.materialize_implied_attrs();
        if !self.is_dict() && self.len as usize >= MAX_SHAPED_LEN {
            self.convert_to_dict();
        }
        if self.is_dict() {
            let d = self.dict_mut();
            d.map.insert(
                Box::from(key),
                DictSlot {
                    value,
                    pos: d.order.len(),
                },
            );
            d.order.push(Some(Box::from(key)));
        } else {
            self.shape = self.shape.add(key);
            self.slot_push(value);
        }
    }

    /// The out-of-line part, or the shared empty one.
    fn ex(&self) -> &ObjectExtra {
        self.extra.as_deref().unwrap_or(&EMPTY_EXTRA)
    }

    /// The out-of-line part, allocated on first use.
    #[inline]
    fn extra_mut(&mut self) -> &mut ObjectExtra {
        self.extra.get_or_insert_with(Default::default)
    }

    /// Appends `key` to the unified insertion-order log iff the object is tracking
    /// order (it has held an accessor) and the key is not already recorded. A no-op
    /// on the fast path (`key_order` is `None`). Called after a *new* own key — data
    /// or accessor — is materialized.
    fn track_key(&mut self, key: &str) {
        self.bump_key_epoch();
        if let Some(x) = self.extra.as_deref_mut()
            && x.key_order.is_some()
            && !x.order_has(key)
        {
            x.order_push(key);
        }
    }

    /// Begins tracking unified own-key insertion order if not already doing so,
    /// seeding the log with the current data keys in their present order. Idempotent
    /// and a no-op once tracking has started. Callers invoke this *before* a
    /// structural change that would otherwise lose a key's chronological position —
    /// e.g. converting an existing data property into an accessor, where the data
    /// slot is deleted (dropping it from the shape) before the accessor is
    /// materialized: seeding first records the key at its real slot so the accessor
    /// re-slots into that same position rather than being appended at the end.
    pub fn ensure_key_order(&mut self) {
        if self.ex().key_order.is_none() {
            let order = self.keys().into_iter().map(Box::from).collect();
            self.extra_mut().set_order(order);
        }
    }

    /// The unified own-key order (data + accessor) reconstructed from the
    /// insertion log, filtered to keys still present, then self-healed by appending
    /// any present key the log missed (data keys before accessor keys). Only called
    /// when `key_order` is `Some`.
    fn merged_key_order(&self) -> Vec<&str> {
        let x = self.ex();
        let order = x
            .key_order
            .as_deref()
            .expect("merged_key_order needs a log");
        let present = |k: &str| self.contains(k) || x.acc_pos(k).is_some();
        let data = self.keys();
        let mut result: Vec<&str> = Vec::with_capacity(data.len() + x.accessors.len());
        // The log holds each key once (`track_key` checks).
        result.extend(order.iter().map(AsRef::as_ref).filter(|k| present(k)));
        // Self-heal: any live key the log never recorded (a data key added through a
        // path that predates tracking) still enumerates — appended in the legacy
        // data-then-accessor order.
        let accessors = x.accessors.iter().map(|(k, _, _)| k.as_ref());
        for k in data
            .iter()
            .copied()
            .chain(accessors.filter(|k| !self.contains(k)))
        {
            if !x.order_has(k) && !result.contains(&k) {
                result.push(k);
            }
        }
        result
    }

    /// The own-key base sequence for `[[OwnPropertyKeys]]`-style enumeration: the
    /// unified insertion order when tracking, else the legacy data-then-accessor
    /// concatenation (correct when data and accessors never interleave).
    fn enum_base(&self) -> Vec<&str> {
        if self.ex().key_order.is_some() {
            self.merged_key_order()
        } else {
            let mut keys = self.keys();
            keys.extend(self.ex().accessors.iter().map(|(k, _, _)| k.as_ref()));
            keys
        }
    }

    /// The `[[Prototype]]` handle, if any.
    #[must_use]
    pub fn proto(&self) -> Option<crate::heap::Handle> {
        (self.proto != NO_PROTO).then(|| crate::heap::Handle::from_raw(self.proto))
    }

    /// Sets the `[[Prototype]]` link (`None` clears it to a null prototype).
    pub fn set_proto(&mut self, proto: Option<crate::heap::Handle>) {
        self.proto = proto.map_or(NO_PROTO, crate::heap::Handle::to_raw);
        self.layout_changed();
    }

    /// The generic-JIT flags ([`JIT_FLAG_EXOTIC`], [`JIT_FLAG_NO_INLINE_SET`]).
    #[must_use]
    pub fn jit_flags(&self) -> u8 {
        self.flags & JIT_FLAGS_MASK
    }

    /// Whether a new data property may be added to this object by adopting a
    /// cached child shape (the generic JIT's add cache): extensible, neither
    /// frozen nor sealed, without accessors and not tracking a unified key
    /// order (so the add is the shape transition alone).
    #[must_use]
    pub fn jit_add_ok(&self) -> bool {
        self.flags & (OBJ_FLAG_FROZEN | OBJ_FLAG_SEALED | OBJ_FLAG_NON_EXTENSIBLE) == 0
            && self
                .extra
                .as_deref()
                .is_none_or(|x| x.accessors.is_empty() && x.key_order.is_none())
    }

    /// Adds a data property by adopting `to`, a child of the object's current
    /// shape (its new key in the next slot), holding `value` — the add half of
    /// [`init_data_cached`](Self::init_data_cached). `false`, having changed
    /// nothing, when the object is not shaped or `to` is not such a child.
    pub fn jit_adopt_child(&mut self, to: Rc<Shape>, value: NanBox) -> bool {
        if self.is_dict()
            || !to.parent().is_some_and(|p| Rc::ptr_eq(p, &self.shape))
            || u32::from(self.len) != self.shape.len()
            || self.len as usize >= MAX_SHAPED_LEN
        {
            return false;
        }
        self.shape = to;
        self.slot_push(value);
        self.bump_key_epoch();
        true
    }

    /// The position of the accessor named `name` in the object's accessor
    /// list (stable while [`key_epoch`](Self::key_epoch) is: every change to
    /// the list bumps it).
    #[must_use]
    pub fn accessor_index(&self, name: &str) -> Option<usize> {
        self.ex().acc_pos(name)
    }

    /// Sets generic-JIT flag bits (they are never cleared).
    pub fn add_jit_flags(&mut self, bits: u8) {
        let bits = bits & JIT_FLAGS_MASK;
        if self.flags | bits != self.flags {
            self.layout_changed();
        }
        self.flags |= bits;
    }

    /// Tags this object with the class it was constructed from.
    pub fn set_class_tag(&mut self, class_id: u32) {
        self.extra_mut().class_tag = Some(class_id);
    }

    /// The class this object was constructed from, if any.
    #[must_use]
    pub fn class_tag(&self) -> Option<u32> {
        self.ex().class_tag
    }

    /// Defines an accessor property `name` with `getter`/`setter` (either may be
    /// `undefined`). Replaces an existing accessor of the same name.
    pub fn define_accessor(&mut self, name: &str, getter: NanBox, setter: NanBox) {
        self.global_attrs_changed();
        self.materialize_implied_attrs();
        // Start tracking unified key order the moment an accessor is involved: seed
        // the log with the current data keys (their chronological order) so this and
        // every later new key slot into the correct interleaved position.
        self.ensure_key_order();
        let x = self.extra_mut();
        if let Some(i) = x.acc_pos(name) {
            let a = &mut x.accessors[i];
            if !matches!(getter.unpack(), crate::nanbox::Unpacked::Undefined) {
                a.1 = getter;
            }
            if !matches!(setter.unpack(), crate::nanbox::Unpacked::Undefined) {
                a.2 = setter;
            }
        } else {
            x.add_acc(name, getter, setter);
        }
        self.track_key(name);
    }

    /// The names of this object's accessor (getter/setter) properties.
    #[must_use]
    pub fn accessor_keys(&self) -> Vec<&str> {
        self.ex()
            .accessors
            .iter()
            .map(|(k, _, _)| k.as_ref())
            .collect()
    }

    /// The `(getter, setter)` of accessor `name`, if defined.
    #[must_use]
    pub fn accessor(&self, name: &str) -> Option<(NanBox, NanBox)> {
        let x = self.ex();
        if x.accessors.is_empty() {
            return None;
        }
        x.acc_pos(name).map(|i| {
            let (_, g, s) = &x.accessors[i];
            (*g, *s)
        })
    }

    /// The object's shape when it is in shaped (non-dictionary) mode.
    #[inline]
    #[must_use]
    pub fn shaped_shape(&self) -> Option<&Rc<Shape>> {
        (!self.is_dict()).then_some(&self.shape)
    }

    /// Whether the object holds the internal primitive-wrapper key
    /// ([`crate::shape::PRIM_WRAP_KEY`]) — answered from the shape without a
    /// lookup in shaped mode.
    #[inline]
    #[must_use]
    pub fn is_prim_wrapper(&self) -> bool {
        if self.is_dict() {
            self.dict().map.contains_key(crate::shape::PRIM_WRAP_KEY)
        } else {
            self.shape.is_wrapper()
        }
    }

    /// Whether the object holds the internal brand key `key` (one of
    /// [`crate::shape::BRAND_KEYS`]) — a flag test in shaped mode.
    #[inline]
    #[must_use]
    pub fn has_brand_key(&self, key: &str) -> bool {
        if self.is_dict() {
            self.dict().map.contains_key(key)
        } else {
            self.shape.has_brand(crate::shape::brand_bit(key))
        }
    }

    /// Whether the object has no accessor properties at all.
    #[inline]
    #[must_use]
    pub fn has_no_accessors(&self) -> bool {
        self.extra.as_deref().is_none_or(|e| e.accessors.is_empty())
    }

    /// Whether the object's own properties are exactly its shape's data keys,
    /// every one enumerable, in the shape's order: a shaped object with no
    /// accessors, no non-enumerable keys and no separate key-order log. Two
    /// such objects with the same shape then have the same
    /// [`enumerable_keys`](Object::enumerable_keys) at the same slots.
    #[inline]
    #[must_use]
    pub fn keys_follow_shape(&self) -> bool {
        !self.is_dict()
            && self.extra.as_deref().is_none_or(|e| {
                e.accessors.is_empty()
                    && e.hidden.is_empty()
                    && e.hidden_prefix == 0
                    && e.key_order.is_none()
            })
            && !self.shape.has_implied_fn_attrs()
    }

    /// The object's current shape (its hidden class). In dictionary mode this is
    /// an empty sentinel shape, so any inline cache keyed on the returned pointer
    /// resolves no key and misses — a dictionary object never binds an IC.
    #[must_use]
    pub fn shape(&self) -> &Rc<Shape> {
        &self.shape
    }

    /// The number of own data properties (excludes side-list accessors).
    #[must_use]
    pub fn len(&self) -> u32 {
        if self.is_dict() {
            self.dict().map.len() as u32
        } else {
            self.shape.len()
        }
    }

    /// Whether the object has no own data properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        if self.is_dict() {
            self.dict().map.is_empty()
        } else {
            self.shape.is_empty()
        }
    }

    /// The value of own property `key`, or `None` if absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<NanBox> {
        if self.is_dict() {
            self.dict().map.get(key).map(|d| d.value)
        } else {
            self.slot_get(self.shape.lookup(key)?)
        }
    }

    /// Reads slot `slot` directly, without a name lookup — the inline-cache fast
    /// path. Returns `None` if the object is in dictionary mode (which has no slot
    /// vector) or the index is out of range. The caller must have obtained `slot`
    /// from a cache armed on *this object's current shape* (see
    /// [`cached_get`](Object::cached_get)); a slot from a stale shape would read a
    /// different property.
    #[must_use]
    pub fn slot(&self, slot: u32) -> Option<NanBox> {
        self.slot_get(slot)
    }

    /// The address of slot `slot`'s value (inline, or in the spill array),
    /// for compiled code that reads it while the prototype epoch shows the
    /// object watched and unchanged: the object's cell moves only when the
    /// heap arena grows, is compacted or swaps cells, and its spill array only
    /// when a slot is added or removed — each of which bumps the epoch.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn jit_slot_ptr(&self, slot: u32) -> Option<*const NanBox> {
        if self.is_dict() || slot >= u32::from(self.len) {
            return None;
        }
        let i = slot as usize;
        if i < INLINE_SLOTS {
            Some(core::ptr::from_ref(&self.inline[i]))
        } else {
            self.spill().get(i - INLINE_SLOTS).map(core::ptr::from_ref)
        }
    }

    /// The address of accessor `i`'s getter, under the same terms as
    /// [`jit_slot_ptr`](Self::jit_slot_ptr) (the accessor list changes only
    /// with the object's key epoch, a layout change).
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn jit_getter_ptr(&self, i: usize) -> Option<*const NanBox> {
        self.ex()
            .accessors
            .get(i)
            .map(|(_, g, _)| core::ptr::from_ref(g))
    }

    /// Writes slot `slot` directly, without a name lookup or the integrity
    /// checks of [`set`](Object::set): the caller must have resolved `slot`
    /// under this object's current shape and ruled out a frozen object and a
    /// read-only key. Returns `false` (nothing written) in dictionary mode or
    /// for an out-of-range slot.
    pub fn set_slot(&mut self, slot: u32, value: NanBox) -> bool {
        match self.slot_get_mut(slot) {
            Some(cell) => {
                *cell = value;
                self.note_slot(slot, value);
                true
            }
            None => false,
        }
    }

    /// Reads own data property `key`, consulting `cache` so a repeat access on the
    /// same shape skips the name lookup (a shape-pointer compare plus a slot
    /// load). Returns `None` if the property is absent (a cache miss).
    ///
    /// This is the bytecode VM's `GetProp` fast path for a plain shaped object. A
    /// dictionary-mode object resolves through its empty sentinel shape, so the
    /// cache always misses and this falls back to a normal lookup — never binding
    /// a dictionary to the cache. Accessors live outside the slot layout and are
    /// the caller's responsibility (checked before this is reached), so a hit here
    /// is always a genuine own data slot.
    pub fn cached_get(&self, key: &str, cache: &mut crate::ic::PropertyCache) -> Option<NanBox> {
        if self.is_dict() {
            // Dictionary mode carries an empty sentinel shape; route around the
            // cache entirely so it never binds (and never goes stale on a later
            // dictionary mutation that does not change the sentinel pointer).
            return self.dict().map.get(key).map(|d| d.value);
        }
        let slot = cache.lookup(&self.shape, key)?;
        self.slot_get(slot)
    }

    /// Writes `value` to own data property `key` *in place*, consulting `cache`,
    /// and reports whether the write happened. Returns `false` when `key` is not
    /// an existing own data property (a new property is a shape transition and must
    /// go through [`set`](Object::set)'s slow path), when the object is in
    /// dictionary mode, or when the property is frozen/read-only — in every such
    /// case the caller falls back to the slow path.
    ///
    /// This only ever rewrites a slot that already exists on the current shape, so
    /// it performs no transition and the cache stays valid; a transition would
    /// produce a new shape pointer that simply misses next time.
    pub fn cached_set(
        &mut self,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) -> bool {
        // A private element (`#x`) is stored as a property but is not one: it
        // is [[PrivateElements]], which `Object.freeze` does not reach, so the
        // integrity flags must not silently drop the write.
        if !crate::realm::is_private_key(key) && (self.is_frozen() || self.is_readonly(key)) {
            return false;
        }
        if self.is_dict() {
            return false;
        }
        // Absent: a new property (transition) — let the slow path add it.
        let Some(slot) = cache.lookup(&self.shape, key) else {
            return false;
        };
        self.set_slot(slot, value)
    }

    /// Whether the object has own property `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        if self.is_dict() {
            self.dict().map.contains_key(key)
        } else {
            self.shape.contains(key)
        }
    }

    /// Sets own property `key` to `value`: updates the value in place if the
    /// property exists, otherwise adds it. Works transparently in either storage
    /// mode (shaped or dictionary). A no-op on a frozen object (matching
    /// `Object.freeze` semantics in non-strict code).
    ///
    /// Conversion to dictionary mode is driven by the realm via
    /// [`maybe_convert_to_dict`](Object::maybe_convert_to_dict), called just
    /// before adding a new property; once converted, this method appends into the
    /// dictionary and creates no shape transitions.
    pub fn set(&mut self, key: &str, value: NanBox) {
        // A private element (`#x`) is stored as a property but is not one: it
        // is [[PrivateElements]], which `Object.freeze` does not reach, so the
        // integrity flags must not silently drop the write.
        if !crate::realm::is_private_key(key) && (self.is_frozen() || self.is_readonly(key)) {
            return;
        }
        if self.write_existing(key, value) {
            return;
        }
        // A non-extensible object silently ignores new keys.
        if self.is_extensible() {
            self.add_new(key, value);
            self.track_key(key);
        }
    }

    /// Overwrites own data property `key` with `value` if it exists; `false`
    /// (nothing written) if it does not.
    fn write_existing(&mut self, key: &str, value: NanBox) -> bool {
        if self.is_dict() {
            match self.dict_mut().map.get_mut(key) {
                Some(d) => {
                    d.value = value;
                    true
                }
                None => false,
            }
        } else {
            match self.shape.lookup(key) {
                Some(slot) => {
                    *self.slot_get_mut(slot).expect("slot index in range") = value;
                    self.note_slot(slot, value);
                    true
                }
                None => false,
            }
        }
    }

    /// Like [`set`](Object::set) but bypasses the extensibility / frozen /
    /// readonly guards: it always stores (creating a new slot if needed). Used by
    /// `[[DefineOwnProperty]]`, which validates configurability/extensibility
    /// itself and must be able to materialize a data slot even on a non-extensible
    /// object (e.g. converting an existing accessor property to a data property, or
    /// changing the value of a configurable-but-non-extensible property).
    pub fn force_set(&mut self, key: &str, value: NanBox) {
        if !self.write_existing(key, value) {
            self.add_new(key, value);
            self.track_key(key);
        }
    }

    /// If adding `key` would be a *new* own property that pushes the own-data
    /// count past `threshold`, converts the object to dictionary mode in place so
    /// the subsequent [`set`](Object::set) — and every later add — creates no
    /// shape transitions. A no-op if `key` already exists, the object is already a
    /// dictionary, the object is non-extensible/frozen, or the count is still
    /// within the threshold. The realm calls this immediately before
    /// [`set`](Object::set) on a property add (MEM-3: bounds shape-tree growth).
    pub fn maybe_convert_to_dict(&mut self, key: &str, threshold: usize) {
        if !self.is_dict()
            && self.shape.len() as usize >= threshold
            && self.is_extensible()
            && !self.is_frozen()
            && !self.is_readonly(key)
            && self.shape.lookup(key).is_none()
        {
            self.convert_to_dict();
        }
    }

    /// Converts a shaped object to dictionary mode in place, preserving the
    /// current properties in insertion order and dropping the live `Rc<Shape>`
    /// so no further transitions are created. A no-op if already a dictionary.
    /// Rebuilds `Dict.order` from the live keys when it has accumulated more
    /// slack than live entries.
    ///
    /// Deletion leaves tombstones behind (see [`DictData`]), so
    /// without this a long-lived object that is repeatedly added to and deleted
    /// from would grow `order` without bound and slow every enumeration.
    /// Dropping the tombstones is one linear pass over `order` plus one over the
    /// map (renumbering each entry's position — no key lookups), and runs only
    /// after as many deletions as there are live keys, so deletion stays
    /// amortized O(log n).
    fn compact_order(&mut self) {
        if !self.is_dict() {
            return;
        }
        let DictData { order, map } = self.dict_mut();
        if order.len() <= 2 * map.len() + 16 {
            return;
        }
        // `renumber[i]`: the new index of the live key at old index `i`.
        let mut renumber: Vec<usize> = Vec::with_capacity(order.len());
        let mut next = 0;
        for k in order.iter() {
            renumber.push(next);
            if k.is_some() {
                next += 1;
            }
        }
        order.retain(Option::is_some);
        for d in map.values_mut() {
            d.pos = renumber[d.pos];
        }
    }

    fn convert_to_dict(&mut self) {
        if self.is_dict() {
            return;
        }
        self.materialize_implied_attrs();
        self.write_out_hidden_prefix();
        let keys = self.shape.keys();
        let mut order: Vec<Option<Box<str>>> = Vec::with_capacity(keys.len());
        let mut map: BTreeMap<Box<str>, DictSlot> = BTreeMap::new();
        for k in keys {
            let slot = self.shape.lookup(k).expect("shape key resolves");
            let value = self.slot_at(slot);
            map.insert(
                Box::from(k),
                DictSlot {
                    value,
                    pos: order.len(),
                },
            );
            order.push(Some(Box::from(k)));
        }
        self.len = 0;
        self.inline = [NanBox::undefined(); INLINE_SLOTS];
        let x = self.extra_mut();
        x.spill = Spill::EMPTY;
        x.dict = Some(Box::new(DictData { order, map }));
        // A fresh empty shape so `shape()` returns a pointer that resolves no
        // key (inline caches keyed on it always miss for this object).
        self.layout_changed();
        self.shape = Shape::root();
        self.mode = Mode::Dict;
    }

    /// The own property names, in insertion order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        if self.is_dict() {
            self.dict()
                .order
                .iter()
                .flatten()
                .map(AsRef::as_ref)
                .collect()
        } else {
            self.shape.keys()
        }
    }

    /// How many own properties (data and accessor) this object stores.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        let data = if self.is_dict() {
            self.dict().map.len()
        } else {
            self.len as usize
        };
        data + self.ex().accessors.len()
    }

    /// Calls `f` with every own key that encodes a Symbol (the `\0sym:{id}`
    /// form), data or accessor, in no particular order and possibly more than
    /// once. The collector's symbol-liveness scan: unlike [`Self::all_keys`] it
    /// neither allocates nor orders, and a dictionary object answers from its
    /// sorted map in O(log n + symbols) — `\0` sorts first, so the symbol keys
    /// are its smallest — rather than walking every key on every collection.
    pub fn for_each_symbol_key(&self, mut f: impl FnMut(&str)) {
        const SYM: &str = crate::shape::SYMBOL_KEY_PREFIX;
        if self.is_dict() {
            use core::ops::Bound;
            for (k, _) in self
                .dict()
                .map
                .range::<str, _>((Bound::Included(SYM), Bound::Unbounded))
            {
                if !k.starts_with(SYM) {
                    break;
                }
                f(k);
            }
        } else if self.shape.has_symbol_keys() {
            // The layout records whether it holds any Symbol key at all, so the
            // common symbol-free object costs one bit test.
            self.shape.for_each_key_unordered(|k| {
                if k.starts_with(SYM) {
                    f(k);
                }
            });
        }
        for (k, _, _) in &self.ex().accessors {
            if k.starts_with(SYM) {
                f(k);
            }
        }
    }

    /// All own property names — data and accessor, including non-enumerable — in
    /// insertion order. For reflection that ignores enumerability (e.g.
    /// `Object.getOwnPropertySymbols`).
    #[must_use]
    pub fn all_keys(&self) -> Vec<&str> {
        self.enum_base()
    }

    /// All own property names (data + accessor, **including** non-enumerable) in spec
    /// `[[OwnPropertyKeys]]` order: integer-index keys ascending, then the rest in
    /// insertion order. Used by `getOwnPropertyNames` / `Reflect.ownKeys`.
    #[must_use]
    pub fn ordered_keys(&self) -> Vec<&str> {
        let base = self.enum_base();
        let mut ints: Vec<&str> = base
            .iter()
            .copied()
            .filter(|k| array_index(k).is_some())
            .collect();
        ints.sort_by_key(|k| array_index(k).unwrap());
        let rest = base.into_iter().filter(|k| array_index(k).is_none());
        ints.into_iter().chain(rest).collect()
    }

    /// The own **enumerable** property names (excludes keys marked hidden), in
    /// spec order: integer-index keys ascending, then the rest in insertion order.
    #[must_use]
    pub fn enumerable_keys(&self) -> Vec<&str> {
        let base: Vec<&str> = self
            .enum_base()
            .into_iter()
            .filter(|k| !self.is_hidden(k))
            .collect();
        let mut ints: Vec<&str> = base
            .iter()
            .copied()
            .filter(|k| array_index(k).is_some())
            .collect();
        ints.sort_by_key(|k| array_index(k).unwrap());
        let rest = base.into_iter().filter(|k| array_index(k).is_none());
        ints.into_iter().chain(rest).collect()
    }

    /// Whether some own key (data or accessor) is an array index.
    #[must_use]
    pub fn has_index_key(&self) -> bool {
        let x = self.ex();
        let digit = |k: &str| k.as_bytes().first().is_some_and(u8::is_ascii_digit);
        let data = if self.is_dict() {
            self.dict()
                .map
                .keys()
                .any(|k| digit(k) && array_index(k).is_some())
        } else {
            self.shape
                .keys()
                .iter()
                .any(|k| digit(k) && array_index(k).is_some())
        };
        data || x
            .accessors
            .iter()
            .any(|(k, _, _)| digit(k) && array_index(k).is_some())
    }

    /// The keys carrying an attribute mark (non-writable, non-enumerable or
    /// non-configurable) — possibly with no data slot or accessor of their own:
    /// an array's attribute override for an element its dense store holds.
    /// A key marked twice is listed twice.
    #[must_use]
    pub fn marked_keys(&self) -> Vec<&str> {
        let x = self.ex();
        x.readonly
            .iter()
            .chain(&x.hidden)
            .chain(&x.non_configurable)
            .map(AsRef::as_ref)
            .collect()
    }

    /// Marks own property `key` non-enumerable (idempotent).
    pub fn set_hidden(&mut self, key: &str) {
        // The newest key, right past the hidden prefix (a property marked
        // right after it was added, all earlier ones hidden too), extends the
        // prefix. Found without a lookup: building a big prototype key by key
        // must not make every intermediate layout build its hashed index.
        let prefix = self.ex().hidden_prefix;
        if !self.is_dict()
            && self.shape.last_key() == Some(key)
            && self.shape.len() == prefix + 1
            && !self.implied_mark(key)
        {
            if !self.ex().has_mark(Mark::Hidden, key) {
                self.extra_mut().hidden_prefix = prefix + 1;
            }
            return;
        }
        if !self.is_hidden(key) {
            self.extra_mut().add_mark(Mark::Hidden, key);
        }
    }

    /// Whether own property `key` is non-enumerable.
    #[must_use]
    #[inline]
    pub fn is_hidden(&self, key: &str) -> bool {
        let x = self.ex();
        self.implied_mark(key)
            || self.prefix_hidden(key)
            || (!x.hidden.is_empty() && x.has_mark(Mark::Hidden, key))
    }

    /// Whether `key` is a data property in the hidden prefix (see
    /// `ObjectExtra::hidden_prefix`).
    #[inline]
    fn prefix_hidden(&self, key: &str) -> bool {
        let prefix = self.ex().hidden_prefix;
        prefix != 0 && !self.is_dict() && self.shape.lookup(key).is_some_and(|s| s < prefix)
    }

    /// Lists the hidden prefix's keys in `hidden` and empties the prefix,
    /// before a change that renumbers the slots or clears one of its marks.
    #[inline]
    fn write_out_hidden_prefix(&mut self) {
        if self.ex().hidden_prefix != 0 {
            self.write_out_hidden_prefix_slow();
        }
    }

    #[cold]
    #[inline(never)]
    fn write_out_hidden_prefix_slow(&mut self) {
        let prefix = self.ex().hidden_prefix as usize;
        let keys: Vec<Box<str>> = self
            .shape
            .keys()
            .into_iter()
            .take(prefix)
            .map(Box::from)
            .collect();
        let x = self.extra_mut();
        x.hidden_prefix = 0;
        for k in keys {
            if !x.has_mark(Mark::Hidden, &k) {
                x.add_mark(Mark::Hidden, &k);
            }
        }
    }

    /// Clears the non-enumerable mark for `key` (a `defineProperty` redefining a
    /// configurable property to `enumerable: true`).
    pub fn clear_hidden(&mut self, key: &str) {
        self.materialize_implied_attrs();
        if self.prefix_hidden(key) {
            self.write_out_hidden_prefix();
        }
        if let Some(x) = self.extra.as_deref_mut() {
            x.remove_mark(Mark::Hidden, key);
        }
    }

    /// Marks own property `key` non-writable (idempotent).
    pub fn set_readonly(&mut self, key: &str) {
        if !self.is_readonly(key) {
            self.global_attrs_changed();
            if self.flags & JIT_FLAG_READONLY == 0 {
                self.layout_changed();
            }
            self.flags |= JIT_FLAG_READONLY;
            self.extra_mut().add_mark(Mark::Readonly, key);
        }
    }

    /// Whether own property `key` is non-writable.
    #[must_use]
    #[inline]
    pub fn is_readonly(&self, key: &str) -> bool {
        let x = self.ex();
        self.implied_mark(key) || (!x.readonly.is_empty() && x.has_mark(Mark::Readonly, key))
    }

    /// Clears the non-writable mark for `key` (a `defineProperty` redefines
    /// attributes from scratch, so the prior `writable: false` is dropped first).
    pub fn clear_readonly(&mut self, key: &str) {
        self.materialize_implied_attrs();
        if let Some(x) = self.extra.as_deref_mut() {
            x.remove_mark(Mark::Readonly, key);
        }
    }

    /// Marks own property `key` non-configurable (idempotent).
    pub fn set_non_configurable(&mut self, key: &str) {
        if !self.is_non_configurable(key) {
            self.extra_mut().add_mark(Mark::NonConfigurable, key);
        }
    }

    /// Whether own property `key` is non-configurable (cannot be deleted).
    #[must_use]
    #[inline]
    pub fn is_non_configurable(&self, key: &str) -> bool {
        let x = self.ex();
        !x.non_configurable.is_empty() && x.has_mark(Mark::NonConfigurable, key)
    }

    /// Clears the non-configurable mark for `key` (a `defineProperty` redefining a
    /// configurable property whose new descriptor keeps it configurable).
    pub fn clear_non_configurable(&mut self, key: &str) {
        if let Some(x) = self.extra.as_deref_mut() {
            x.remove_mark(Mark::NonConfigurable, key);
        }
    }

    /// Whether *no* own property is marked non-configurable — a cheap probe used to
    /// skip the per-index ArraySetLength stop scan on the common (unrestricted) array.
    #[must_use]
    pub fn non_configurable_is_empty(&self) -> bool {
        self.ex().non_configurable.is_empty()
    }

    /// Whether a `CreateDataProperty(key)` on this object is a plain store:
    /// the object accepts new keys and `key` is either absent or an existing
    /// writable, enumerable, configurable data property (no accessor, no
    /// attribute marks). Internal (`\0`-prefixed) keys never qualify.
    #[must_use]
    pub fn is_plain_data_slot_or_absent(&self, key: &str) -> bool {
        self.flags & (OBJ_FLAG_FROZEN | OBJ_FLAG_SEALED | OBJ_FLAG_NON_EXTENSIBLE) == 0
            && !key.starts_with('\0')
            && !self.implied_mark(key)
            && self.extra.as_deref().is_none_or(|x| {
                (x.readonly.is_empty() || !self.is_readonly(key))
                    && (x.accessors.is_empty() || self.accessor(key).is_none())
                    && (x.hidden.is_empty() && x.hidden_prefix == 0 || !self.is_hidden(key))
                    && (x.non_configurable.is_empty() || !self.is_non_configurable(key))
            })
    }

    /// `CreateDataProperty(key, value)` through a transition cache, for a
    /// site that always adds the same `key` (an object literal). Returns
    /// `false`, having changed nothing, unless the define is a plain store
    /// (see [`is_plain_data_slot_or_absent`](Self::is_plain_data_slot_or_absent)).
    /// A cache hit adds the property by adopting the cached child shape; a
    /// miss goes through [`set`](Self::set) (dictionary conversion included)
    /// and re-arms the cache when the add was a single transition.
    pub fn init_data_cached(
        &mut self,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
        dict_threshold: usize,
    ) -> bool {
        if !self.is_plain_data_slot_or_absent(key) {
            return false;
        }
        if !self.is_dict()
            && (self.shape.len() as usize) < dict_threshold.min(MAX_SHAPED_LEN)
            && self.extra.as_deref().is_none_or(|x| x.key_order.is_none())
            && let Some(to) = cache.transition_from(&self.shape)
        {
            self.shape = to;
            self.slot_push(value);
            self.bump_key_epoch();
            return true;
        }
        self.maybe_convert_to_dict(key, dict_threshold);
        let before = (!self.is_dict()).then(|| Rc::clone(&self.shape));
        self.set(key, value);
        if let Some(before) = before
            && !self.is_dict()
            && self.shape.parent().is_some_and(|p| Rc::ptr_eq(p, &before))
        {
            cache.arm_transition(Rc::clone(&self.shape));
        }
        true
    }

    /// Whether `key` is an own property (data slot or accessor).
    #[must_use]
    pub fn has_own_key(&self, key: &str) -> bool {
        self.contains(key) || self.accessor(key).is_some()
    }

    /// Marks the object frozen (`Object.freeze`) — implies sealed + non-extensible.
    pub fn freeze(&mut self) {
        self.global_attrs_changed();
        self.flags |= OBJ_FLAG_FROZEN | OBJ_FLAG_SEALED | OBJ_FLAG_NON_EXTENSIBLE;
    }

    /// For a global object ([`JIT_FLAG_GLOBAL`]): a change to whether a write
    /// to one of its existing data properties stores (`set`'s frozen and
    /// non-writable checks, or an accessor taking the key) — which compiled
    /// code's global-object mirror holds answered under the global binding
    /// epoch (`nbvm::global_mirror`), so it bumps that.
    #[inline]
    fn global_attrs_changed(&self) {
        if self.flags & JIT_FLAG_GLOBAL != 0 {
            crate::env::bump_global_binding_epoch();
        }
    }

    /// The address of own data property `key`'s value when [`set`](Self::set)
    /// of it would store exactly there and do nothing else: a dictionary-mode
    /// object, not frozen, the key a writable data property without a
    /// same-named accessor. For compiled code's global-object mirror
    /// (`nbvm::global_mirror`): the address stays put until the dictionary
    /// gains or loses a key (a key epoch bump), and the facts hold until a
    /// [`global_attrs_changed`](Self::global_attrs_changed) one.
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn jit_data_ptr(&mut self, key: &str) -> Option<*mut NanBox> {
        if !self.is_dict()
            || self.is_frozen()
            || crate::realm::is_private_key(key)
            || self.is_readonly(key)
            || self.accessor(key).is_some()
        {
            return None;
        }
        self.dict_mut()
            .map
            .get_mut(key)
            .map(|d| core::ptr::from_mut(&mut d.value))
    }

    /// Prevents new properties (`Object.preventExtensions`).
    pub fn prevent_extensions(&mut self) {
        self.flags |= OBJ_FLAG_NON_EXTENSIBLE;
    }

    /// Seals the object (`Object.seal`): no new props, no deletions.
    pub fn seal(&mut self) {
        self.flags |= OBJ_FLAG_SEALED | OBJ_FLAG_NON_EXTENSIBLE;
    }

    /// Whether new properties may be added.
    #[must_use]
    pub fn is_extensible(&self) -> bool {
        self.flags & OBJ_FLAG_NON_EXTENSIBLE == 0
    }

    /// Whether the object is sealed (or frozen).
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.flags & (OBJ_FLAG_SEALED | OBJ_FLAG_FROZEN) != 0
    }

    /// Whether the object is frozen.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.flags & OBJ_FLAG_FROZEN != 0
    }

    /// Removes any accessor (getter/setter) for `key`, leaving data properties
    /// intact — used when a `defineProperty` replaces an accessor with data.
    pub fn clear_accessor(&mut self, key: &str) {
        self.bump_key_epoch();
        if let Some(x) = self.extra.as_deref_mut() {
            x.remove_acc(key);
        }
    }

    /// Deletes only the *data slot* for `key`, rebuilding the shape/slots from
    /// `root` without it, and leaving any same-named accessor untouched. Used when
    /// `defineProperty` converts a data property into an accessor.
    pub fn delete_data(&mut self, root: Rc<Shape>, key: &str) {
        self.materialize_implied_attrs();
        self.write_out_hidden_prefix();
        self.bump_key_epoch();
        if self.is_dict() {
            let d = self.dict_mut();
            if let Some(s) = d.map.remove(key) {
                d.order[s.pos] = None;
            }
        } else if self.shape.contains(key) {
            self.rebuild_without(root, key);
        }
    }

    /// Rebuilds the shape and slots from `root` without `key` (shaped mode,
    /// `key` present).
    fn rebuild_without(&mut self, root: Rc<Shape>, key: &str) {
        let kept: Vec<(alloc::string::String, NanBox)> = self
            .shape
            .keys()
            .into_iter()
            .filter(|k| *k != key)
            .map(|k| {
                let slot = self.shape.lookup(k).expect("shape key resolves");
                (alloc::string::String::from(k), self.slot_at(slot))
            })
            .collect();
        let mut new_shape = root;
        let mut values = Vec::with_capacity(kept.len());
        for (k, v) in kept {
            new_shape = new_shape.transition(&k);
            values.push(v);
        }
        self.shape = new_shape;
        self.set_slots(values);
        self.tidy_extra();
    }

    /// Deletes own property `key`, rebuilding the shape/slots from `root` without
    /// it (also drops a same-named accessor). Returns whether anything was
    /// removed.
    pub fn delete(&mut self, root: Rc<Shape>, key: &str) -> bool {
        self.materialize_implied_attrs();
        self.write_out_hidden_prefix();
        self.bump_key_epoch();
        let had_accessor = self.extra.as_deref_mut().is_some_and(|x| x.remove_acc(key));
        // Clear the property's attribute flags so a later re-add of the same key
        // starts from defaults — otherwise a stale `readonly` flag makes `set` a
        // silent no-op, and a stale `hidden`/`non_configurable` flag wrongly
        // carries over to the new property (e.g. re-defining a deleted, formerly
        // non-writable `name`/`length`, or NamedEvaluation re-installing `name`).
        if let Some(x) = self.extra.as_deref_mut() {
            x.remove_mark(Mark::Hidden, key);
            x.remove_mark(Mark::Readonly, key);
            x.remove_mark(Mark::NonConfigurable, key);
        }
        // Drop from the unified insertion log so a later re-add of the same key
        // appends at the end (a deleted-then-recreated property is chronologically
        // new), rather than reclaiming its former slot.
        if let Some(x) = self.extra.as_deref_mut() {
            x.order_remove(key);
        }
        if self.is_dict() {
            let d = self.dict_mut();
            let Some(s) = d.map.remove(key) else {
                return had_accessor;
            };
            d.order[s.pos] = None;
            self.compact_order();
            true
        } else {
            if !self.shape.contains(key) {
                return had_accessor;
            }
            self.rebuild_without(root, key);
            true
        }
    }

    /// Rewrites every outgoing handle through `forward` — the mutating mirror of
    /// [`trace_handles`](Object::trace_handles), used by a moving collector to
    /// fix up references after relocation.
    pub fn relocate_handles(
        &mut self,
        forward: &dyn Fn(crate::heap::Handle) -> crate::heap::Handle,
    ) {
        let fwd = |v: &mut NanBox| {
            if let Some(raw) = v.as_handle() {
                *v = NanBox::handle(forward(crate::heap::Handle::from_raw(raw)).to_raw());
            }
        };
        let n = (self.len as usize).min(INLINE_SLOTS);
        self.inline[..n].iter_mut().for_each(fwd);
        if let Some(x) = self.extra.as_deref_mut() {
            x.spill.iter_mut().for_each(fwd);
            if let Some(d) = x.dict.as_deref_mut() {
                for d in d.map.values_mut() {
                    fwd(&mut d.value);
                }
            }
            for (_, g, s) in &mut x.accessors {
                fwd(g);
                fwd(s);
            }
        }
        if let Some(p) = self.proto() {
            self.set_proto(Some(forward(p)));
        }
    }

    /// Calls `visit` for every heap [`Handle`](crate::heap::Handle) this object
    /// references through a slot — the outgoing edges a tracing collector
    /// follows.
    pub fn trace_handles(&self, mut visit: impl FnMut(crate::heap::Handle)) {
        let mut trace_value = |v: &NanBox| {
            if let Some(raw) = v.as_handle() {
                visit(crate::heap::Handle::from_raw(raw));
            }
        };
        self.slots_iter().for_each(&mut trace_value);
        if let Some(x) = self.extra.as_deref() {
            if let Some(d) = x.dict.as_deref() {
                for d in d.map.values() {
                    trace_value(&d.value);
                }
            }
            for (_, g, s) in &x.accessors {
                trace_value(g);
                trace_value(s);
            }
        }
        if let Some(p) = self.proto() {
            visit(p);
        }
    }
}

impl crate::gc::Trace for Object {
    fn trace(&self, visit: &mut dyn FnMut(crate::heap::Handle)) {
        self.trace_handles(visit);
    }
}

/// Runtime-probed byte layout of [`Object`] (and the inner `Vec<NanBox>`
/// storage) for the generic-JIT inline property paths. Every field is derived
/// from a real, populated shaped object (never hand-baked); the JIT layout
/// verification tests cross-check each against a safe read, so a layout drift
/// fails loudly instead of corrupting a raw memory read. Offsets are relative
/// to the `Object` base unless noted.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ObjectLayout {
    /// Byte offset of the storage-mode byte.
    pub mode_off: usize,
    /// The shaped mode's byte value.
    pub shaped_disc: u8,
    /// Byte offset of the `shape: Rc<Shape>` field. The 8-byte word there is
    /// the `Rc`'s stored `NonNull<RcBox>` (the shape identity the IC compares
    /// against).
    pub shape_off: usize,
    /// Byte offset of the slot count (`len: u16`).
    pub len_off: usize,
    /// Byte offset of the first inline slot (slot `i < INLINE_SLOTS` at
    /// `+ 8 * i`).
    pub inline_off: usize,
    /// Byte offset, within `ObjectExtra`, of the near spilled values (slot
    /// `i` in `INLINE_SLOTS..INLINE_SLOTS + SPILL_NEAR` at `+ 8 * (i -
    /// INLINE_SLOTS)`; an object with spilled slots always has its `extra`
    /// part).
    pub extra_near_off: usize,
    /// Byte offset, within `ObjectExtra`, of the far spill `Vec`'s data
    /// pointer (slot `i >= INLINE_SLOTS + SPILL_NEAR` at that pointer `+ 8 *
    /// (i - INLINE_SLOTS - SPILL_NEAR)`).
    pub extra_far_ptr_off: usize,
    /// Byte offset of the data pointer within a `Vec<NanBox>`.
    pub vec_ptr_off: usize,
    /// Byte offset of the length within a `Vec<NanBox>`.
    pub vec_len_off: usize,
    /// Byte offset of the flags byte (the `JIT_FLAG_*` and `OBJ_FLAG_*` bits).
    pub jit_flags_off: usize,
    /// Byte offset of the raw `proto` word.
    pub proto_off: usize,
    /// Byte offset of `key_epoch: u32`.
    pub epoch_off: usize,
    /// Byte offset of the `extra` box pointer (null for none).
    pub extra_off: usize,
    /// Byte offset of the accessor list's length word within `ObjectExtra`.
    pub extra_acc_len_off: usize,
    /// Byte offset of the accessor list's data pointer within `ObjectExtra`.
    pub extra_acc_ptr_off: usize,
    /// Byte offsets of the length words of the attribute-mark lists
    /// (non-writable, non-enumerable, non-configurable keys) within
    /// `ObjectExtra`.
    pub extra_marks_len_off: [usize; 3],
    /// Byte offset of the key-order log's word within `ObjectExtra` (null
    /// for none).
    pub extra_key_order_off: usize,
    /// Byte offset of the spilled-value count (`u32`) within `ObjectExtra`.
    pub extra_spill_n_off: usize,
    /// The size of one accessor entry (`(name, getter, setter)`).
    pub acc_stride: usize,
    /// Byte offset of the getter within an accessor entry.
    pub acc_getter_off: usize,
    /// Offset from the shape word (the `Rc`'s stored pointer) to the shape's
    /// `brands` byte (bit 0: a primitive wrapper).
    pub shape_brands_off: isize,
}

/// Derives [`ObjectLayout`] from a real shaped object with enough own data
/// properties to fill both the near and the far spill (so the far `Vec` is
/// populated).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn jit_object_layout() -> ObjectLayout {
    // Two properties past the near spill, so the far `Vec` is populated.
    let mut obj = Object::new(Shape::root());
    let count = INLINE_SLOTS + SPILL_NEAR + 2;
    for i in 0..count {
        obj.set(&alloc::format!("k{i}"), NanBox::number(i as f64));
    }

    let obj_base = core::ptr::addr_of!(obj) as usize;
    let mode_off = core::ptr::addr_of!(obj.mode) as usize - obj_base;
    // SAFETY: reading the live object's repr(u8) mode byte, in bounds.
    #[allow(unsafe_code)]
    let shaped_disc = unsafe { *((obj_base + mode_off) as *const u8) };
    assert_eq!(obj.mode, Mode::Shaped);
    let shape_off = core::ptr::addr_of!(obj.shape) as usize - obj_base;
    let len_off = core::ptr::addr_of!(obj.len) as usize - obj_base;
    let inline_off = core::ptr::addr_of!(obj.inline) as usize - obj_base;

    // Locate the data-pointer and length words inside a `Vec<NanBox>` by
    // matching their known values against each 8-byte word of the Vec — a pure
    // derivation from a real instance (no assumed field order). A `Vec`'s
    // words sit at the same offsets whatever its element type.
    let probe: Vec<NanBox> = alloc::vec![NanBox::undefined(); 3];
    let mut probe = probe;
    probe.reserve(8);
    let vec_base = core::ptr::addr_of!(probe) as usize;
    let want_ptr = probe.as_ptr() as usize;
    let want_len = probe.len(); // == 3, capacity >= 11
    let words = core::mem::size_of::<Vec<NanBox>>() / core::mem::size_of::<usize>();
    let mut vec_ptr_off = usize::MAX;
    let mut vec_len_off = usize::MAX;
    for i in 0..words {
        // SAFETY: reading the i-th usize-word of a live `Vec<NanBox>`, in bounds.
        #[allow(unsafe_code)]
        let w = unsafe { *((vec_base + i * core::mem::size_of::<usize>()) as *const usize) };
        if w == want_ptr {
            vec_ptr_off = i * core::mem::size_of::<usize>();
        } else if w == want_len {
            vec_len_off = i * core::mem::size_of::<usize>();
        }
    }
    assert!(
        vec_ptr_off != usize::MAX && vec_len_off != usize::MAX && vec_ptr_off != vec_len_off,
        "could not derive Vec ptr/len offsets"
    );

    // The shape's brand byte, relative to the stored `Rc` word.
    // SAFETY: reading the 8-byte `Rc<Shape>` word of a live object, in bounds.
    #[allow(unsafe_code)]
    let shape_word = unsafe { *((obj_base + shape_off) as *const usize) };
    let brands_addr = Rc::as_ptr(&obj.shape) as usize + crate::shape::Shape::brands_offset();
    let shape_brands_off = brands_addr.wrapping_sub(shape_word).cast_signed();
    // `extra`: an `Option<Box<_>>` is one nullable pointer word (allocated:
    // the object has spilled slots).
    let extra_off = core::ptr::addr_of!(obj.extra) as usize - obj_base;
    // SAFETY: reading the pointer word of the live `Option<Box<ObjectExtra>>`.
    #[allow(unsafe_code)]
    let extra_word = unsafe { *((obj_base + extra_off) as *const usize) };
    let extra_ref = obj.extra.as_deref().expect("spilled slots live in `extra`");
    assert_eq!(extra_word, core::ptr::addr_of!(*extra_ref) as usize);
    let spill_off = core::mem::offset_of!(ObjectExtra, spill);
    let extra_near_off = spill_off + core::mem::offset_of!(Spill, near);
    assert_eq!(
        extra_word + extra_near_off,
        extra_ref.spill.near.as_ptr() as usize
    );
    let extra_far_ptr_off = spill_off + core::mem::offset_of!(Spill, far) + vec_ptr_off;
    // SAFETY: reading the far spill Vec's pointer word of the live `ObjectExtra`.
    #[allow(unsafe_code)]
    let far_word = unsafe { *((extra_word + extra_far_ptr_off) as *const usize) };
    assert_eq!(far_word, extra_ref.spill.far.as_ptr() as usize);
    // Each spilled value where the offsets say it is.
    for i in INLINE_SLOTS..count {
        let j = i - INLINE_SLOTS;
        let at = if j < SPILL_NEAR {
            extra_word + extra_near_off + 8 * j
        } else {
            far_word + 8 * (j - SPILL_NEAR)
        };
        // SAFETY: a live spilled value of `obj` (checked against `slot_get`).
        #[allow(unsafe_code)]
        let v = unsafe { *(at as *const NanBox) };
        assert_eq!(v.to_bits(), obj.slot_at(i as u32).to_bits());
    }
    let acc_probe: (Box<str>, NanBox, NanBox) =
        (Box::from("k"), NanBox::undefined(), NanBox::null());
    let acc_getter_off =
        core::ptr::addr_of!(acc_probe.1) as usize - core::ptr::addr_of!(acc_probe) as usize;
    let extra_acc_ptr_off = core::mem::offset_of!(ObjectExtra, accessors) + vec_ptr_off;
    let extra_acc_len_off = core::mem::offset_of!(ObjectExtra, accessors) + vec_len_off;
    let extra_marks_len_off = [
        core::mem::offset_of!(ObjectExtra, readonly) + vec_len_off,
        core::mem::offset_of!(ObjectExtra, hidden) + vec_len_off,
        core::mem::offset_of!(ObjectExtra, non_configurable) + vec_len_off,
    ];
    let extra_key_order_off = core::mem::offset_of!(ObjectExtra, key_order);
    let extra_spill_n_off = spill_off + core::mem::offset_of!(Spill, n);
    // SAFETY: the count and the (unset) key-order word of the live `ObjectExtra`.
    #[allow(unsafe_code)]
    let (n_word, order_word) = unsafe {
        (
            *((extra_word + extra_spill_n_off) as *const u32),
            *((extra_word + extra_key_order_off) as *const usize),
        )
    };
    assert_eq!(n_word as usize, count - INLINE_SLOTS);
    assert!(extra_ref.key_order.is_none() && order_word == 0);
    ObjectLayout {
        mode_off,
        shaped_disc,
        shape_off,
        len_off,
        inline_off,
        extra_near_off,
        extra_far_ptr_off,
        vec_ptr_off,
        vec_len_off,
        jit_flags_off: core::mem::offset_of!(Object, flags),
        proto_off: core::mem::offset_of!(Object, proto),
        epoch_off: core::mem::offset_of!(Object, key_epoch),
        extra_off,
        extra_acc_len_off,
        extra_acc_ptr_off,
        extra_marks_len_off,
        extra_key_order_off,
        extra_spill_n_off,
        acc_stride: core::mem::size_of::<(Box<str>, NanBox, NanBox)>(),
        acc_getter_off,
        shape_brands_off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::Heap;
    use crate::nanbox::Unpacked;

    fn n(x: f64) -> NanBox {
        NanBox::number(x)
    }

    #[test]
    fn every_slot_write_keeps_the_layout_number_record() {
        let root = Shape::root();
        // Adds, in-place writes, slot writes and `with_data`.
        let mut a = Object::new(Rc::clone(&root));
        a.set("x", n(1.0));
        a.set("y", n(f64::NAN));
        a.set("z", n(-0.0));
        let s = Rc::clone(a.shape());
        assert!((0..3).all(|i| !s.may_be_non_number(i)));
        a.set("y", NanBox::undefined());
        assert!(s.may_be_non_number(1) && !s.may_be_non_number(0));
        assert!(a.set_slot(2, NanBox::boolean(true)));
        assert!(s.may_be_non_number(2));
        let t = root.transition("p").transition("q");
        assert!(Object::with_data(Rc::clone(&t), &[n(1.0), n(2.0)]).is_some());
        assert!(!t.may_be_non_number(0) && !t.may_be_non_number(1));
        assert!(Object::with_data(Rc::clone(&t), &[n(1.0), NanBox::null()]).is_some());
        assert!(!t.may_be_non_number(0) && t.may_be_non_number(1));
        // A delete rebuilds the layout from the root: the new one learns
        // the values it takes over.
        let mut b = Object::new(Rc::clone(&root));
        b.set("u", n(1.0));
        b.set("v", NanBox::null());
        b.set("w", n(3.0));
        b.delete(Rc::clone(&root), "u");
        assert!(b.shape().may_be_non_number(0) && !b.shape().may_be_non_number(1));
    }

    #[test]
    fn attribute_index_mirrors_the_lists() {
        let mut o = Object::new(Shape::root());
        let mut keys = Vec::new();
        for i in 0..300 {
            let k = alloc::format!("k{i}");
            o.set(&k, n(i as f64));
            keys.push(k);
        }
        for (i, k) in keys.iter().enumerate() {
            if i % 2 == 0 {
                o.set_hidden(k);
            }
            if i % 3 == 0 {
                o.set_readonly(k);
            }
            if i % 5 == 0 {
                o.set_non_configurable(k);
            }
        }
        for i in 0..30 {
            o.define_accessor(&alloc::format!("a{i}"), n(i as f64), NanBox::undefined());
        }
        assert!(o.ex().index.is_some());
        // Remove a scattering, then check every query against the lists.
        for (i, k) in keys.iter().enumerate() {
            if i % 4 == 0 {
                o.clear_hidden(k);
                o.clear_readonly(k);
            }
        }
        for i in (0..30).step_by(7) {
            o.clear_accessor(&alloc::format!("a{i}"));
        }
        o.delete(Shape::root(), "k9");
        o.define_accessor("k9", n(9.0), NanBox::undefined());
        let x = o.ex();
        for k in keys
            .iter()
            .cloned()
            .chain((0..30).map(|i| alloc::format!("a{i}")))
        {
            let scan = |l: &Vec<Box<str>>| l.iter().any(|e| e.as_ref() == k);
            assert_eq!(o.is_hidden(&k), scan(&x.hidden), "{k}");
            assert_eq!(o.is_readonly(&k), scan(&x.readonly), "{k}");
            assert_eq!(o.is_non_configurable(&k), scan(&x.non_configurable), "{k}");
            let pos = x.accessors.iter().position(|(a, _, _)| a.as_ref() == k);
            assert_eq!(o.accessor_index(&k), pos, "{k}");
        }
        let all = o.all_keys();
        assert_eq!(all.len(), 300 + 30 - 5);
        assert_eq!(all.last(), Some(&"k9"));
        assert!(!o.enumerable_keys().contains(&"k2"));
        assert!(o.enumerable_keys().contains(&"k4"));
    }

    #[test]
    fn builtin_fn_props_imply_their_marks_until_changed() {
        let root = Shape::root();
        let layout = Shape::builtin_fn_layout(&root);
        let mut o = Object::builtin_fn_props(Rc::clone(&layout), n(2.0), n(7.0));
        assert_eq!(o.keys(), ["length", "name"]);
        assert!(o.is_hidden("length") && o.is_readonly("name"));
        assert!(!o.is_non_configurable("name"));
        assert!(o.enumerable_keys().is_empty());
        assert!(!o.keys_follow_shape());
        // A write to a non-writable property is ignored.
        o.set("name", n(1.0));
        assert!(matches!(o.get("name").unwrap().unpack(), Unpacked::Number(x) if x == 7.0));
        // A new key moves the object to the ordinary layout, marks kept.
        o.set("prototype", n(3.0));
        assert!(!o.shape().has_implied_fn_attrs());
        assert_eq!(o.keys(), ["length", "name", "prototype"]);
        assert!(o.is_hidden("length") && o.is_hidden("name") && !o.is_hidden("prototype"));
        assert!(o.is_readonly("length") && o.is_readonly("name") && !o.is_readonly("prototype"));
        // The ordinary twin is the plain `+length +name` layout.
        let plain = root.add("length").add("name");
        assert!(Rc::ptr_eq(o.shape().parent().unwrap(), &plain));
        // Clearing a mark on a fresh one keeps the other.
        let mut p = Object::builtin_fn_props(layout, n(0.0), n(5.0));
        p.clear_readonly("name");
        assert!(!p.is_readonly("name") && p.is_readonly("length"));
        assert!(p.is_hidden("name"));
        p.set("name", n(6.0));
        assert!(matches!(p.get("name").unwrap().unpack(), Unpacked::Number(x) if x == 6.0));
    }

    #[test]
    fn hidden_prefix_marks_keys_added_then_hidden() {
        let root = Shape::root();
        let mut o = Object::new(Rc::clone(&root));
        for k in ["a", "b", "c"] {
            o.set(k, n(1.0));
            o.set_hidden(k);
        }
        o.set("d", n(1.0));
        assert_eq!(o.ex().hidden_prefix, 3);
        assert!(o.ex().hidden.is_empty());
        assert_eq!(o.enumerable_keys(), ["d"]);
        assert!(o.is_hidden("b") && !o.is_hidden("d") && !o.is_hidden("zz"));
        // Out of order: listed.
        o.set("e", n(1.0));
        o.set_hidden("d");
        assert_eq!(o.ex().hidden_prefix, 3);
        assert_eq!(o.enumerable_keys(), ["e"]);
        // Clearing one prefix mark writes the prefix out.
        o.clear_hidden("b");
        assert_eq!(o.ex().hidden_prefix, 0);
        assert_eq!(o.enumerable_keys(), ["b", "e"]);
        // A delete renumbers slots, so the prefix is written out first.
        let mut q = Object::new(Rc::clone(&root));
        for k in ["x", "y", "z"] {
            q.set(k, n(1.0));
            q.set_hidden(k);
        }
        q.set("w", n(1.0));
        assert!(q.delete(Rc::clone(&root), "x"));
        assert_eq!(q.enumerable_keys(), ["w"]);
        assert!(q.is_hidden("y") && q.is_hidden("z"));
        q.set("x", n(2.0));
        assert_eq!(q.enumerable_keys(), ["w", "x"]);
    }

    #[test]
    fn set_get_and_update() {
        let mut o = Object::new(Shape::root());
        assert!(o.is_empty());
        o.set("x", n(1.0));
        o.set("y", n(2.0));
        assert_eq!(o.len(), 2);
        assert_eq!(o.get("x").unwrap().unpack(), Unpacked::Number(1.0));
        assert_eq!(o.get("y").unwrap().unpack(), Unpacked::Number(2.0));
        assert_eq!(o.get("z"), None);
        // Updating an existing property keeps the shape and reuses the slot.
        let shape_before = Rc::clone(o.shape());
        o.set("x", n(9.0));
        assert!(Rc::ptr_eq(o.shape(), &shape_before));
        assert_eq!(o.get("x").unwrap().unpack(), Unpacked::Number(9.0));
        assert_eq!(o.len(), 2);
        assert_eq!(o.keys(), ["x", "y"]);
    }

    #[test]
    fn same_structure_objects_share_a_shape() {
        let root = Shape::root();
        let mut a = Object::new(Rc::clone(&root));
        let mut b = Object::new(Rc::clone(&root));
        a.set("p", n(1.0));
        a.set("q", n(2.0));
        b.set("p", n(10.0));
        b.set("q", n(20.0));
        // Distinct values, one shared hidden class.
        assert!(Rc::ptr_eq(a.shape(), b.shape()));
        assert_eq!(a.get("p").unwrap().unpack(), Unpacked::Number(1.0));
        assert_eq!(b.get("p").unwrap().unpack(), Unpacked::Number(10.0));
    }

    #[test]
    fn inline_slots_keep_the_object_small() {
        // Four inline values and the header fill one 64-byte line, and a heap
        // cell is no larger than an object (it uses the mode byte's niche).
        assert_eq!(core::mem::size_of::<Object>(), 64);
        assert_eq!(
            core::mem::size_of::<crate::cell::Cell>(),
            core::mem::size_of::<Object>()
        );
    }

    #[test]
    fn values_spill_past_the_inline_room_and_survive_deletes() {
        let mut o = Object::new(Shape::root());
        let keys = ["a", "b", "c", "d", "e", "f", "g"];
        for (i, k) in keys.iter().enumerate() {
            o.set(k, n(i as f64));
        }
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(o.get(k), Some(n(i as f64)));
        }
        assert_eq!(o.slot_count(), keys.len());
        // Deleting an inline key moves a spilled value inline.
        assert!(o.delete(Shape::root(), "b"));
        for (i, k) in keys.iter().enumerate().filter(|(_, k)| **k != "b") {
            assert_eq!(o.get(k), Some(n(i as f64)));
        }
        assert_eq!(o.get("b"), None);
        assert_eq!(o.slot_count(), keys.len() - 1);
        o.set("e", n(40.0));
        assert_eq!(o.get("e"), Some(n(40.0)));
        // The dictionary form keeps every value.
        o.convert_to_dict();
        assert_eq!(o.get("g"), Some(n(6.0)));
        assert_eq!(o.get("e"), Some(n(40.0)));
    }

    #[test]
    fn mixed_value_kinds_in_slots() {
        let mut o = Object::new(Shape::root());
        o.set("a", NanBox::number(3.5));
        o.set("b", NanBox::boolean(true));
        o.set("c", NanBox::null());
        o.set("d", NanBox::handle(42));
        assert_eq!(o.get("a").unwrap().unpack(), Unpacked::Number(3.5));
        assert_eq!(o.get("b").unwrap().unpack(), Unpacked::Bool(true));
        assert_eq!(o.get("c").unwrap().unpack(), Unpacked::Null);
        assert_eq!(o.get("d").unwrap().unpack(), Unpacked::Handle(42));
    }

    #[test]
    fn objects_live_in_the_heap_and_reference_each_other() {
        // An object graph: `parent.child` holds a handle to another object in
        // the same heap — the value representation the GC will manage.
        let root = Shape::root();
        let mut heap: Heap<Object> = Heap::new();

        let mut child = Object::new(Rc::clone(&root));
        child.set("value", n(7.0));
        let child_handle = heap.alloc(child);

        let mut parent = Object::new(Rc::clone(&root));
        parent.set("child", NanBox::handle(child_handle.to_raw()));
        let parent_handle = heap.alloc(parent);

        // Walk parent -> child by resolving the handle stored in the slot.
        let parent_ref = heap.get(parent_handle).unwrap();
        let raw = parent_ref.get("child").unwrap().as_handle().unwrap();
        let resolved = crate::heap::Handle::from_raw(raw);
        let child_ref = heap.get(resolved).unwrap();
        assert_eq!(
            child_ref.get("value").unwrap().unpack(),
            Unpacked::Number(7.0)
        );
    }

    /// Adds `key=value` the way the realm does: convert-if-needed, then set.
    fn add(o: &mut Object, key: &str, value: NanBox, threshold: usize) {
        o.maybe_convert_to_dict(key, threshold);
        o.set(key, value);
    }

    /// `delete` leaves a tombstone in `Dict.order` so that it need not scan it
    /// (deleting n keys was O(n²)). Enumeration must still report each live key
    /// exactly once, at the position of its *last* insertion — a
    /// deleted-then-re-added property is chronologically new — and `order` must
    /// not grow without bound under churn.
    #[test]
    fn dictionary_delete_keeps_enumeration_order_and_is_bounded() {
        let threshold = 4;
        let mut o = Object::new(Shape::root());
        for i in 0..10 {
            add(&mut o, &alloc::format!("k{i}"), n(f64::from(i)), threshold);
        }
        assert_eq!(o.len(), 10);

        // A delete removes the key from enumeration and from the count.
        o.delete(Shape::root(), "k3");
        assert_eq!(o.len(), 9);
        assert!(!o.keys().contains(&"k3"));

        // Re-adding puts it back at the END, and exactly once.
        add(&mut o, "k3", n(99.0), threshold);
        assert_eq!(o.len(), 10);
        assert_eq!(*o.keys().last().unwrap(), "k3");
        assert_eq!(o.keys().iter().filter(|k| **k == "k3").count(), 1);
        assert_eq!(o.get("k3").unwrap().unpack(), Unpacked::Number(99.0));

        // Heavy churn on one key: order stays correct and `order` stays bounded
        // (compaction), rather than accumulating one stale entry per delete.
        for r in 0..500 {
            o.delete(Shape::root(), "k0");
            add(&mut o, "k0", n(f64::from(r)), threshold);
        }
        assert_eq!(o.len(), 10);
        assert_eq!(o.keys().iter().filter(|k| **k == "k0").count(), 1);
        assert_eq!(*o.keys().last().unwrap(), "k0");
        assert_eq!(o.keys().len(), 10);
        if o.is_dict() {
            assert!(
                o.dict().order.len() <= 2 * 10 + 16,
                "tombstones were never compacted"
            );
        }
        // Compaction renumbered every survivor: each still deletes cleanly.
        assert_eq!(
            o.keys(),
            ["k1", "k2", "k4", "k5", "k6", "k7", "k8", "k9", "k3", "k0"]
        );

        // Deleting everything leaves an empty, self-consistent object.
        for i in 0..10 {
            o.delete(Shape::root(), &alloc::format!("k{i}"));
        }
        assert_eq!(o.len(), 0);
        assert!(o.keys().is_empty());
        assert!(o.is_empty());
    }

    /// The collector's symbol-key scan sees exactly the `\0sym:` keys, data
    /// and accessor, in both storage modes.
    #[test]
    fn symbol_key_scan_finds_only_symbol_keys() {
        for threshold in [2, 64] {
            let mut o = Object::new(Shape::root());
            for k in [
                "a",
                "\u{0}sym:7",
                "\u{0}hidden",
                "z",
                "\u{0}sym:3",
                "\u{1}x",
            ] {
                add(&mut o, k, n(1.0), threshold);
            }
            o.define_accessor("\u{0}sym:9", NanBox::undefined(), NanBox::undefined());
            let mut seen = Vec::new();
            o.for_each_symbol_key(|k| seen.push(alloc::string::String::from(k)));
            seen.sort();
            assert_eq!(seen, ["\u{0}sym:3", "\u{0}sym:7", "\u{0}sym:9"]);
        }
    }

    #[test]
    fn converts_to_dictionary_past_threshold_and_preserves_semantics() {
        // Threshold 4: the 5th distinct key triggers conversion.
        let mut o = Object::new(Shape::root());
        let threshold = 4;
        for i in 0..4 {
            add(&mut o, &alloc::format!("k{i}"), n(f64::from(i)), threshold);
        }
        // Still shaped: the shape resolves a real key.
        assert_eq!(o.shape().lookup("k0"), Some(0));
        assert_eq!(o.len(), 4);

        // The 5th add converts to dictionary mode.
        add(&mut o, "k4", n(4.0), threshold);
        assert_eq!(o.len(), 5);
        // In dictionary mode the sentinel shape resolves nothing (ICs miss).
        assert_eq!(o.shape().lookup("k0"), None);
        assert_eq!(o.shape().lookup("k4"), None);

        // Keep adding well past the threshold.
        for i in 5..300 {
            add(&mut o, &alloc::format!("k{i}"), n(f64::from(i)), threshold);
        }
        assert_eq!(o.len(), 300);

        // get across the conversion boundary.
        assert_eq!(o.get("k0").unwrap().unpack(), Unpacked::Number(0.0));
        assert_eq!(o.get("k3").unwrap().unpack(), Unpacked::Number(3.0));
        assert_eq!(o.get("k150").unwrap().unpack(), Unpacked::Number(150.0));
        assert_eq!(o.get("k299").unwrap().unpack(), Unpacked::Number(299.0));
        assert_eq!(o.get("nope"), None);

        // Insertion order is preserved across the boundary.
        let keys = o.keys();
        assert_eq!(keys.len(), 300);
        assert_eq!(keys[0], "k0");
        assert_eq!(keys[299], "k299");
        assert!(o.contains("k200"));
        assert!(o.has_own_key("k200"));

        // Update-in-place in dictionary mode.
        add(&mut o, "k150", n(-1.0), threshold);
        assert_eq!(o.len(), 300, "updating an existing key does not grow");
        assert_eq!(o.get("k150").unwrap().unpack(), Unpacked::Number(-1.0));

        // Delete in dictionary mode.
        assert!(o.delete(Shape::root(), "k0"));
        assert_eq!(o.get("k0"), None);
        assert_eq!(o.len(), 299);
        assert_eq!(o.keys()[0], "k1");
        // Deleting an absent key is a no-op (returns false).
        assert!(!o.delete(Shape::root(), "k0"));
    }

    #[test]
    fn dictionary_ordered_keys_put_integer_indices_first() {
        // Mirror the `{b, 2, 1, a}` example but force dictionary mode with a low
        // threshold so the integer-index ordering is exercised in the dict path.
        let mut o = Object::new(Shape::root());
        let threshold = 0; // convert on the very first add
        add(&mut o, "b", n(1.0), threshold);
        add(&mut o, "2", n(1.0), threshold);
        add(&mut o, "1", n(1.0), threshold);
        add(&mut o, "a", n(1.0), threshold);
        // Confirm we are in dictionary mode.
        assert_eq!(o.shape().lookup("b"), None);
        // [[OwnPropertyKeys]] order: ascending integer indices, then strings in
        // insertion order.
        assert_eq!(o.ordered_keys(), ["1", "2", "b", "a"]);
        assert_eq!(o.enumerable_keys(), ["1", "2", "b", "a"]);
        // Raw insertion order is unchanged.
        assert_eq!(o.keys(), ["b", "2", "1", "a"]);
    }

    #[test]
    fn dictionary_traces_handle_values() {
        // GC must trace handle values stored in a dictionary-mode object.
        let mut o = Object::new(Shape::root());
        let threshold = 0;
        add(&mut o, "h1", NanBox::handle(7), threshold);
        add(&mut o, "n", n(1.0), threshold);
        add(&mut o, "h2", NanBox::handle(9), threshold);
        let mut seen: Vec<u64> = Vec::new();
        o.trace_handles(|h| seen.push(h.to_raw()));
        seen.sort_unstable();
        assert_eq!(seen, [7, 9]);
    }
}
