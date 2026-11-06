//! The `for-in` key-list cache.
//!
//! `for (k in o)` takes the list of `o`'s enumerable String keys, its own and
//! inherited ones (EnumerateObjectProperties, §14.7.5.9), every time the loop
//! is entered. Building it walks the whole prototype chain and every key of
//! every link (each own key shadows a like-named inherited one, enumerable or
//! not), so for a small object most of the work is `Object.prototype`'s
//! non-enumerable methods. That list depends on nothing but the receiver's
//! own keys and the chain's, so it is memoized here.
//!
//! An entry is keyed on the receiver's shape. A **plain shaped** ordinary
//! object (only data properties, all enumerable: see
//! [`Object::is_plain_shaped`](crate::object::Object::is_plain_shaped)) has
//! exactly the keys its shape lists, in its order, so every receiver with
//! that shape has the same own share of the list. Each prototype-chain link
//! is recorded by handle (a handle carries its slot's generation, so a freed
//! and reused slot never matches) and key epoch, which changes whenever the
//! link's own keys, or their enumerability, may have changed; a different
//! `[[Prototype]]` shows as a different next handle. Only ordinary,
//! non-wrapper, non-host-exotic objects take part (never a proxy, an array,
//! a typed array, a String wrapper, the global object or a function), and a
//! moved cell (`Realm::swap_cell_state`, compaction) retires every entry.
//! The links are watched (`crate::object::JIT_FLAG_WATCHED`), so once the
//! chain has been walked and found unchanged at prototype epoch `E`, a
//! lookup while the epoch is still `E` checks only the receiver's own
//! `[[Prototype]]`.
//!
//! The cached key Strings are handed to the loop as they are (a String is
//! immutable, and its identity is not observable), so they are GC roots.
//!
//! An entry also records, per key, the receiver shape's slot of it (an own
//! key; inherited ones have none), so the loop body's `o[k]` and the loop's
//! per-key liveness check `k in o` can answer for a key String of the list
//! on an object of the entry's shape without reading the key's text or
//! searching the layout ([`Realm::for_in_key_slot`]). Such a loop presents
//! the keys in order, so a per-entry hint (the last key found) finds the
//! next one by its handle in one or two compares; anything else (another
//! String, another layout, a key out of order) is a miss after at most three
//! compares, and the ordinary lookup decides. Nothing is recorded for
//! lookups that are not of a listed key, so other key patterns cannot evict
//! or thrash anything.
//!
//! Compiled code goes further (`crate::jit::forin`): an entry's per-key
//! slots are also a [`KeyTable`], which a compiled `for-in` loop's
//! [`ForInSite`] holds while the loop runs, so its `o[k]` and `k in o` read
//! the slot with no call. A table names the entry's shared key array, which
//! it keeps a GC root for as long as anything holds the table: so while a
//! site holds it, no other array (and no other key String) can take that
//! array's handle, and "the loop's key array is the table's" proves that the
//! key Strings in the loop's register are the table's.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;

/// How many receiver shapes the cache remembers. Each layout's entry is found
/// through the layout's own hint (`Shape::for_in_hint`), confirmed by the
/// entry's shape, so any `CAP` layouts in use stay cached together; a new
/// one replaces the entries round-robin.
const CAP: usize = 16;

/// The longest prototype chain an entry records.
const MAX_LINKS: usize = 8;

/// One memoized key list (see the module docs). (Its key slots serve the
/// compiled code's helpers only.)
#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
struct Entry {
    /// The receiver's shape (held, so its address is not reused).
    shape: Rc<Shape>,
    /// `(handle, key epoch)` per prototype-chain link, nearest first; the
    /// last link's `[[Prototype]]` is `null`.
    links: Vec<(u64, u32)>,
    /// `Realm::cell_swaps` when the entry was made.
    swaps: u32,
    /// The prototype epoch (`crate::object::proto_epoch`) at which `links`
    /// last matched the chain, all of them watched (`0`: never).
    stamp: core::cell::Cell<u64>,
    /// The `for-in` key list.
    keys: Vec<NanBox>,
    /// Per key: its bits and its slot in `shape` (an own key whose String is
    /// its own storage key, see `Realm::string_key_in_place`), else
    /// [`NO_SLOT`]; and the array of `keys` handed to every loop over this
    /// entry's layout (see [`Realm::for_in_cached_array`]), once made.
    table: Rc<KeyTable>,
    /// The index of the key [`Realm::for_in_key_slot`] last found.
    hint: core::cell::Cell<u32>,
    /// The raw handle of the object [`Realm::for_in_key_slot`] last found
    /// not to be host-exotic (`0`: none yet).
    plain_obj: core::cell::Cell<u64>,
}

/// An entry's keys and their slots, as compiled code reads them (see the
/// module docs): `bits[i]` is key `i`'s bits and `slots[i]` its slot in the
/// layout `shape` ([`NO_SLOT`] for none). While anything holds the table,
/// `array` (once made) and the key Strings are GC roots.
#[repr(C)]
#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
pub(crate) struct KeyTable {
    /// The bits of the entry's key array (`0`: not made yet, or retired
    /// before a compaction: then no loop's array is this table's).
    pub(crate) array: core::cell::Cell<u64>,
    /// The layout's shape word as an object's shape field holds it
    /// (`crate::jit::shape_word`; `0` until compiled code first needs it).
    pub(crate) shape_word: core::cell::Cell<u64>,
    /// How many keys.
    pub(crate) len: u64,
    /// The first of `bits`.
    pub(crate) bits_ptr: *const u64,
    /// The first of `slots`.
    pub(crate) slots_ptr: *const u64,
    bits: Box<[u64]>,
    slots: Box<[u64]>,
    /// The layout (held, so its address is not reused).
    shape: Rc<Shape>,
    /// The prototype epoch at which the entry's chain was last found
    /// unchanged for a receiver whose `[[Prototype]]` is `proto` (`0`:
    /// never): see [`Realm::for_in_table_hit`].
    pub(crate) stamp: core::cell::Cell<u64>,
    /// See `stamp`.
    proto: core::cell::Cell<Option<u64>>,
    /// `proto` as an object's raw `[[Prototype]]` word holds it
    /// (`crate::object::NO_PROTO` for `null`), for compiled code's hit
    /// (`crate::jit::forin`).
    pub(crate) proto_raw: core::cell::Cell<u64>,
    /// The layout's slot count, for compiled code's hit.
    pub(crate) shape_len: u64,
}

#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
impl KeyTable {
    fn new(shape: Rc<Shape>, bits: Box<[u64]>, slots: Box<[u64]>) -> Self {
        Self {
            array: core::cell::Cell::new(0),
            shape_word: core::cell::Cell::new(0),
            len: bits.len() as u64,
            bits_ptr: bits.as_ptr(),
            slots_ptr: slots.as_ptr(),
            bits,
            slots,
            shape_len: u64::from(shape.len()),
            shape,
            stamp: core::cell::Cell::new(0),
            proto: core::cell::Cell::new(None),
            proto_raw: core::cell::Cell::new(crate::object::NO_PROTO),
        }
    }

    /// Key `i`'s `(bits, slot)`.
    fn key(&self, i: usize) -> Option<(u64, u64)> {
        Some((*self.bits.get(i)?, *self.slots.get(i)?))
    }

    /// The layout the slots are of.
    pub(crate) fn shape(&self) -> &Rc<Shape> {
        &self.shape
    }

    /// The table's GC roots: its key Strings and its array.
    fn roots(&self, out: &mut Vec<Handle>) {
        let array = Some(self.array.get()).filter(|&a| a != 0);
        out.extend(
            self.bits
                .iter()
                .chain(array.as_ref())
                .filter_map(|&b| NanBox::from_bits(b).as_handle().map(Handle::from_raw)),
        );
    }
}

/// What a compiled `for-in` loop's `EnumKeys` last found (see the module
/// docs): the key table of its array, or an empty one. Compiled code reads
/// the table through `table`, never null.
#[repr(C)]
#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
pub(crate) struct ForInSite {
    /// The held table's address (else `empty`'s).
    pub(crate) table: core::cell::Cell<*const KeyTable>,
    held: core::cell::RefCell<Option<Rc<KeyTable>>>,
    empty: Box<KeyTable>,
}

#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
impl ForInSite {
    /// A site holding no table.
    #[must_use]
    pub(crate) fn new() -> Self {
        let empty = Box::new(KeyTable::new(Shape::root(), Box::new([]), Box::new([])));
        Self {
            table: core::cell::Cell::new(core::ptr::from_ref(&*empty)),
            held: core::cell::RefCell::new(None),
            empty,
        }
    }

    /// The held table.
    pub(crate) fn held(&self) -> core::cell::Ref<'_, Option<Rc<KeyTable>>> {
        self.held.borrow()
    }

    /// Holds `t` (or nothing).
    pub(crate) fn hold(&self, t: Option<Rc<KeyTable>>) {
        self.table.set(
            t.as_deref()
                .map_or(core::ptr::from_ref(&*self.empty), core::ptr::from_ref),
        );
        *self.held.borrow_mut() = t;
    }
}

impl Default for ForInSite {
    fn default() -> Self {
        Self::new()
    }
}

/// The handle bits of a `NanBox` handle (`NanBox::as_handle`).
#[cfg_attr(
    not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
const HANDLE_BITS: u64 = 0x0000_ffff_ffff_ffff;

/// A [`KeyTable`] slot for a key with no slot of its own there.
pub(crate) const NO_SLOT: u64 = u32::MAX as u64;

/// The realm's `for-in` key-list cache (see the module docs).
#[derive(Default)]
pub(crate) struct ForInCache {
    entries: [Option<Entry>; CAP],
    /// The entry a new layout replaces next.
    next: usize,
    /// Bit `i` set: some entry has (or had) a key String in heap slot `i`.
    /// A key lookup tests it before anything else, so a String that is no
    /// entry's key costs [`Realm::for_in_key_slot`] one bit test. (Never
    /// cleared: a stale bit only sends a lookup to the entry check.)
    key_bits: Vec<u64>,
    /// The tables of entries replaced since, which a compiled loop may still
    /// hold (their roots stay roots while one does).
    retired: Vec<Weak<KeyTable>>,
}

/// The heap slot index of a handle (its low half; the high half is the
/// slot's generation).
fn key_bit(raw: u64) -> (usize, u64) {
    let i = (raw & 0xffff_ffff) as usize;
    (i / 64, 1 << (i % 64))
}

impl ForInCache {
    /// The index of the entry for `shape`: the one its hint names, if that
    /// entry is `shape`'s.
    #[inline]
    fn find(&self, shape: &Rc<Shape>) -> Option<usize> {
        let i = usize::from(shape.for_in_hint()).checked_sub(1)?;
        self.entries
            .get(i)?
            .as_ref()
            .is_some_and(|e| Rc::ptr_eq(&e.shape, shape))
            .then_some(i)
    }

    /// The index a new entry for `shape` takes (and its hint names): the
    /// entry `shape` has, else an empty one, else the next one round.
    fn victim(&mut self, shape: &Rc<Shape>) -> usize {
        let i = self
            .find(shape)
            .or_else(|| self.entries.iter().position(Option::is_none))
            .unwrap_or_else(|| {
                let i = self.next % CAP;
                self.next = (i + 1) % CAP;
                i
            });
        shape.set_for_in_hint(u16::try_from(i + 1).unwrap_or(0));
        i
    }
}

impl Realm {
    /// The receiver's shape, when `h` is a plain shaped ordinary object
    /// whose `for-in` share is decided by that shape alone.
    fn for_in_receiver_shape(&self, h: Handle) -> Option<&Rc<Shape>> {
        self.for_in_receiver(h).map(|(_, s)| s)
    }

    /// [`for_in_receiver_shape`](Self::for_in_receiver_shape), with the
    /// object.
    #[inline]
    fn for_in_receiver(&self, h: Handle) -> Option<(&crate::object::Object, &Rc<Shape>)> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if !o.is_plain_shaped() || o.is_prim_wrapper() || self.is_host_exotic(h) {
            return None;
        }
        Some((o, o.shaped_shape()?))
    }

    /// One prototype-chain link's `(key epoch, [[Prototype]])`, when it is an
    /// ordinary object whose `for-in` share its own keys decide.
    fn for_in_link(&self, h: Handle) -> Option<(u32, Option<Handle>)> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if o.is_prim_wrapper()
            || o.jit_flags() & crate::object::JIT_FLAG_GLOBAL != 0
            || self.is_host_exotic(h)
        {
            return None;
        }
        Some((o.key_epoch(), o.proto()))
    }

    /// The memoized `for-in` key list of `h`, when an entry still describes
    /// it exactly.
    #[must_use]
    pub(crate) fn for_in_cached(&self, h: Handle) -> Option<&[NanBox]> {
        self.for_in_cached_entry(h).map(|(_, e)| e.keys.as_slice())
    }

    /// The entry (and its index) whose key list is `h`'s
    /// ([`for_in_cached`](Self::for_in_cached)).
    fn for_in_cached_entry(&self, h: Handle) -> Option<(usize, &Entry)> {
        let cache = self.for_in_cache.as_deref()?;
        let (o, shape) = self.for_in_receiver(h)?;
        let w = cache.find(shape)?;
        let e = cache.entries[w].as_ref()?;
        if e.swaps != self.cell_swaps {
            return None;
        }
        let mut cur = o.proto();
        // Every link is watched, so while the prototype epoch is the one the
        // whole chain was last found unchanged at, only the receiver's own
        // `[[Prototype]]` can differ.
        let now = crate::object::proto_epoch();
        if e.stamp.get() == now && e.links.first().map(|l| l.0) == cur.map(Handle::to_raw) {
            return Some((w, e));
        }
        for &(raw, epoch) in &e.links {
            let link = cur?;
            if link.to_raw() != raw {
                return None;
            }
            let (now, next) = self.for_in_link(link)?;
            if now != epoch {
                return None;
            }
            cur = next;
        }
        if cur.is_some() {
            return None;
        }
        e.stamp.set(now);
        Some((w, e))
    }

    /// An array of the memoized `for-in` key list of `h` (see
    /// [`for_in_cached`](Self::for_in_cached)), whose liveness object is `h`
    /// itself. `None` on a miss.
    ///
    /// The array is the entry's own, made once: a `for-in` loop's key list
    /// lives in a register of the loop that only its element reads see
    /// (`Op::EnumKeys`, then `Op::GetElem`), so no code can tell one array
    /// from another or change it, and every loop over the same layout may
    /// share it.
    pub(crate) fn for_in_cached_array(&mut self, h: Handle) -> Option<NanBox> {
        self.for_in_cached_table(h).map(|(a, _)| a)
    }

    /// [`for_in_cached_array`](Self::for_in_cached_array), with the key
    /// table whose array it is (`None` when making the array retired the
    /// entry: then the array is only this loop's).
    pub(crate) fn for_in_cached_table(
        &mut self,
        h: Handle,
    ) -> Option<(NanBox, Option<Rc<KeyTable>>)> {
        let (w, e) = self.for_in_cached_entry(h)?;
        // (The chain was just found unchanged, for `h`'s `[[Prototype]]`.)
        e.table.stamp.set(crate::object::proto_epoch());
        let proto = self.object_proto(h).map(Handle::to_raw);
        e.table.proto.set(proto);
        e.table
            .proto_raw
            .set(proto.unwrap_or(crate::object::NO_PROTO));
        let (w, e) = (w, self.for_in_cache.as_deref()?.entries[w].as_ref()?);
        if e.table.array.get() != 0 {
            let t = Rc::clone(&e.table);
            return Some((NanBox::from_bits(t.array.get()), Some(t)));
        }
        let keys = e.keys.to_vec();
        let a = NanBox::handle(self.new_array(keys).to_raw());
        if let Some(e) = self
            .for_in_cache
            .as_deref()
            .and_then(|c| c.entries[w].as_ref())
            && e.swaps == self.cell_swaps
        {
            e.table.array.set(a.to_bits());
            return Some((a, Some(Rc::clone(&e.table))));
        }
        Some((a, None))
    }

    /// The array of `t`, a key table some loop over `h` was given before,
    /// when it is `h`'s `for-in` key list still
    /// ([`for_in_cached_table`](Self::for_in_cached_table) without the
    /// lookup): `h` is a plain shaped ordinary object of `t`'s layout
    /// whose `[[Prototype]]` is the one the table's chain was last found
    /// unchanged for, at the prototype epoch that still is current (every
    /// link is watched).
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn for_in_table_hit(&self, h: Handle, t: &KeyTable) -> Option<NanBox> {
        let a = t.array.get();
        if a == 0 || t.stamp.get() != crate::object::proto_epoch() {
            return None;
        }
        let (o, shape) = self.for_in_receiver(h)?;
        (Rc::ptr_eq(shape, &t.shape) && o.proto().map(Handle::to_raw) == t.proto.get())
            .then(|| NanBox::from_bits(a))
    }

    /// Records `keys` as the `for-in` key list of `h`, when `h` and its
    /// prototype chain are ones an entry can describe (see the module docs).
    pub(crate) fn for_in_remember(&mut self, h: Handle, keys: &[NanBox]) {
        let Some(shape) = self.for_in_receiver_shape(h).cloned() else {
            return;
        };
        let mut links = Vec::new();
        let mut cur = self.object_proto(h);
        while let Some(link) = cur {
            if links.len() == MAX_LINKS {
                return;
            }
            let Some((epoch, next)) = self.for_in_link(link) else {
                return;
            };
            links.push((link.to_raw(), epoch));
            cur = next;
        }
        let slots = keys
            .iter()
            .map(|&k| {
                self.string_key_in_place(k)
                    .and_then(|t| shape.lookup(t))
                    .map_or(NO_SLOT, u64::from)
            })
            .collect();
        let bits = keys.iter().map(|k| k.to_bits()).collect();
        let table = Rc::new(KeyTable::new(Rc::clone(&shape), bits, slots));
        // Each link is watched (`crate::object::JIT_FLAG_WATCHED`), so a
        // change to its keys or `[[Prototype]]` bumps the prototype epoch and
        // an unchanged epoch proves the chain (`for_in_cached_entry`).
        for &(raw, _) in &links {
            if let Some(Cell::Object(o)) = self.heap.get_mut(Handle::from_raw(raw)) {
                o.watch();
            }
        }
        let entry = Entry {
            links,
            swaps: self.cell_swaps,
            stamp: core::cell::Cell::new(0),
            keys: keys.to_vec(),
            table,
            hint: core::cell::Cell::new(0),
            plain_obj: core::cell::Cell::new(0),
            shape,
        };
        let cache = self.for_in_cache.get_or_insert_with(Default::default);
        let w = cache.victim(&entry.shape);
        // A replaced entry's table stays rooted while a compiled loop holds
        // it (and is forgotten once nothing does).
        cache.retired.retain(|t| t.strong_count() > 0);
        if let Some(old) = cache.entries[w].take()
            && Rc::strong_count(&old.table) > 1
        {
            cache.retired.push(Rc::downgrade(&old.table));
        }
        for k in &entry.keys {
            if let Some(raw) = k.as_handle() {
                let (word, bit) = key_bit(raw);
                if word >= cache.key_bits.len() {
                    cache.key_bits.resize(word + 1, 0);
                }
                cache.key_bits[word] |= bit;
            }
        }
        cache.entries[w] = Some(entry);
    }

    /// The ordinary object `obj` and the slot of the key String `key` in it,
    /// when `key` is one of the memoized key Strings of an entry for `obj`'s
    /// shape and an own key there, and `obj` has no accessors and is not
    /// host-exotic: `obj[key]` is then that slot's value (what
    /// `own_data_by_string` reads) and `key in obj` is true. Found by handle
    /// near the entry's hint only (see the module docs); `None` otherwise
    /// (the ordinary lookup decides).
    #[inline]
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn for_in_key_slot(
        &self,
        obj: NanBox,
        key: NanBox,
    ) -> Option<(&crate::object::Object, u32)> {
        let cache = self.for_in_cache.as_deref()?;
        let (word, bit) = key_bit(key.as_handle()?);
        if cache.key_bits.get(word).is_none_or(|w| w & bit == 0) {
            return None;
        }
        self.for_in_key_slot_of(cache, obj, key)
    }

    /// [`for_in_key_slot`](Self::for_in_key_slot) once the key's bit is set.
    #[inline(never)]
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    fn for_in_key_slot_of<'a>(
        &'a self,
        cache: &'a ForInCache,
        obj: NanBox,
        key: NanBox,
    ) -> Option<(&'a crate::object::Object, u32)> {
        let h = Handle::from_raw(obj.as_handle()?);
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        let e = cache.entries[cache.find(shape)?].as_ref()?;
        if e.swaps != self.cell_swaps {
            return None;
        }
        let bits = key.to_bits();
        let at = e.hint.get() as usize;
        let hit = |i: usize| e.table.key(i).filter(|k| k.0 == bits).map(|k| (i, k.1));
        let (i, slot) = hit(at).or_else(|| hit(at + 1)).or_else(|| hit(0))?;
        e.hint.set(i as u32);
        let slot = u32::try_from(slot).ok()?;
        // `o[key]` is the slot's value when no accessor shares its name
        // (none at all, here): what `Object::get` reads in shaped mode. A
        // wrapper's or a mapped `arguments` object's layout is never an
        // entry's. (A String with an auxiliary object is the general path's,
        // as in `string_key_in_place`.)
        if u64::from(slot) == NO_SLOT
            || !o.has_no_accessors()
            || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0
            || (self.str_special && self.aux_props.contains_key(&(bits & HANDLE_BITS)))
        {
            return None;
        }
        // (Whether an object is host-exotic is fixed for its handle, whose
        // generation a reused slot changes: the last one found not to be is
        // remembered.)
        if e.plain_obj.get() != h.to_raw() {
            if self.is_host_exotic(h) {
                return None;
            }
            e.plain_obj.set(h.to_raw());
        }
        Some((o, slot))
    }

    /// The cached key Strings and arrays, which are GC roots (a replaced
    /// entry's too, while a compiled loop holds its table).
    pub(crate) fn for_in_cache_roots(&self, out: &mut Vec<Handle>) {
        if let Some(c) = self.for_in_cache.as_deref() {
            for e in c.entries.iter().flatten() {
                e.table.roots(out);
            }
            for t in c.retired.iter().filter_map(Weak::upgrade) {
                t.roots(out);
            }
        }
    }

    /// Drops the cache before a compaction moves every handle: a table a
    /// compiled loop may still hold then names no array (so no loop's array
    /// is its), and nothing is rooted for it.
    pub(crate) fn for_in_cache_retire(&mut self) {
        if let Some(c) = self.for_in_cache.take() {
            for e in c.entries.iter().flatten() {
                e.table.array.set(0);
            }
            for t in c.retired.iter().filter_map(Weak::upgrade) {
                t.array.set(0);
            }
        }
    }
}
