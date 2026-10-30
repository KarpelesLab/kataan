//! Hidden classes ("shapes") — shared property-layout descriptors
//! (`ROADMAP.md` §3, the object model).
//!
//! In the interpreter era an object stores its own `Vec<(name, value)>` and
//! looks properties up by linear scan. The performance object model instead
//! separates an object's **layout** from its **values**: objects with the same
//! set of properties added in the same order share one immutable `Shape` that
//! maps each property name to a dense slot index, and the object stores only a
//! flat `Vec` of values indexed by that slot. This makes property access a slot
//! load (cacheable by an inline cache keyed on the shape) and shrinks objects.
//!
//! Shapes form a **transition tree**: the root (`Shape::root`) is the empty
//! layout, and adding a property transitions to a child shape (creating it once
//! and caching it, so every `{x, y}` built `x`-then-`y` ends up at the same
//! shape). The chain of parents records the property order; each node adds
//! exactly one property at the next slot.
//!
//! Pure, safe `alloc`-only Rust (`Rc` + `RefCell`); the object that pairs a
//! shape with a value vector, and inline caches over shapes, build on top.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::{Rc, Weak};
use alloc::vec::Vec;
use core::cell::{Cell, OnceCell, RefCell};

/// Layouts with fewer properties than this are always searched by walking the
/// parent chain: a handful of length-checked compares beats hashing the key.
const INDEX_MIN_LEN: u32 = 8;

/// How many chain-walk lookups a large layout serves before it builds its
/// hashed index. Most interior shapes of a transition chain are looked up
/// only while an object is being built (once by `set`, once by `transition`),
/// so they never pay for an index; a layout that is actually read from (a
/// built-in prototype, a shared record layout) builds it on its third probe.
const INDEX_AFTER_PROBES: u32 = 3;

/// A 32-bit hash of a property key (word-at-a-time multiply-rotate; never 0
/// so the index can use 0 as its empty marker).
#[inline]
pub(crate) fn key_hash(key: &str) -> u32 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    let bytes = key.as_bytes();
    let mut h: u64 = bytes.len() as u64;
    let (chunks, rest) = bytes.as_chunks::<8>();
    for c in chunks {
        h = (h.rotate_left(5) ^ u64::from_le_bytes(*c)).wrapping_mul(K);
    }
    if !rest.is_empty() {
        let mut buf = [0u8; 8];
        buf[..rest.len()].copy_from_slice(rest);
        h = (h.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(K);
    }
    let h = (h ^ (h >> 32)) as u32;
    if h == 0 { 1 } else { h }
}

/// The hashed key table of one (large) layout: an open-addressed table of
/// `(hash << 32) | (slot + 1)` words (0 = empty) plus the keys in slot order.
/// Built once, lazily, and shared by every object with that shape (shapes are
/// immutable, so it never needs invalidating).
struct ShapeIndex {
    table: Box<[u64]>,
    keys: Box<[Rc<str>]>,
}

impl ShapeIndex {
    fn build(shape: &Shape) -> Self {
        let mut keys: Vec<Rc<str>> = Vec::with_capacity(shape.len as usize);
        let mut node = shape;
        while let Some(k) = &node.key {
            keys.push(Rc::clone(k));
            match node.parent.as_deref() {
                Some(p) => node = p,
                None => break,
            }
        }
        keys.reverse();
        let cap = (keys.len() * 2).next_power_of_two().max(16);
        let mask = cap - 1;
        let mut table = alloc::vec![0u64; cap].into_boxed_slice();
        for (slot, k) in keys.iter().enumerate() {
            let h = key_hash(k);
            let mut i = h as usize & mask;
            while table[i] != 0 {
                i = (i + 1) & mask;
            }
            table[i] = (u64::from(h) << 32) | (slot as u64 + 1);
        }
        Self {
            table,
            keys: keys.into_boxed_slice(),
        }
    }

    #[inline]
    fn find(&self, key: &str) -> Option<u32> {
        let h = key_hash(key);
        let mask = self.table.len() - 1;
        let mut i = h as usize & mask;
        loop {
            let e = self.table[i];
            if e == 0 {
                return None;
            }
            if (e >> 32) as u32 == h {
                let slot = (e as u32) - 1;
                if &*self.keys[slot as usize] == key {
                    return Some(slot);
                }
            }
            i = (i + 1) & mask;
        }
    }
}

/// An immutable description of an object's property layout: which names map to
/// which slot indices, shared by every object with that structure. Build shapes
/// from [`Shape::root`] via [`Shape::transition`].
pub struct Shape {
    /// The shape this one extends (the layout without the last property), or
    /// `None` for the root.
    parent: Option<Rc<Shape>>,
    /// The property this shape adds over its parent (`None` for the root).
    /// Shared (`Rc`) so a [`ShapeIndex`] can list a layout's keys without
    /// copying them.
    key: Option<Rc<str>>,
    /// The slot index of `key` (and `= parent.len()`); meaningless for the root.
    slot: u32,
    /// The number of properties in this layout.
    len: u32,
    /// Cached child shapes, one per property added from here — so a repeated
    /// `+key` reuses the same child rather than forking the tree.
    ///
    /// **Weak** on purpose. A child holds its parent strongly (the parent chain
    /// is the layout), so a strong cache entry here would close a cycle and no
    /// shape would ever be freed — leaking the entire transition tree of every
    /// realm. A cached child that no object still uses is simply dropped and
    /// rebuilt on next use; shape identity is only ever a cache-hit test
    /// ([`crate::ic`]), so a rebuilt shape costs a miss, never correctness.
    transitions: RefCell<Transitions>,
    /// The hashed key table, built after [`INDEX_AFTER_PROBES`] lookups of a
    /// layout with at least [`INDEX_MIN_LEN`] properties (see [`ShapeIndex`]).
    index: OnceCell<Box<ShapeIndex>>,
    /// Chain-walk lookups served so far (drives building `index`).
    probes: Cell<u32>,
    /// Which internal brand keys ([`BRAND_KEYS`]) the layout holds, one bit
    /// each, so "is this a wrapper / `[[IsHTMLDDA]]` object" needs no lookup.
    brands: u8,
    /// The slots (bit `i` for slot `i < 64`) that some object with this
    /// layout may hold a value other than a Number in: see
    /// [`Shape::note_non_number`]. A clear bit is a promise that every
    /// object whose shape is this one holds a Number in that slot; compiled
    /// code reads such a slot without a type check. Bits are only ever set.
    /// Slots from 64 on are not tracked (always "may be anything").
    any: Cell<u64>,
}

/// The most cached children a [`Transitions`] list holds before it becomes a
/// map: a shape usually has a handful, which a short scan finds fastest.
const TRANSITION_LIST_MAX: usize = 16;

/// A shape's cached child shapes, one per property added from it (each a
/// [`Weak`] link; see [`Shape::transitions`]).
///
/// Most shapes have one child (an object built key by key walks a chain),
/// held without a list allocation, or a few, kept in a short list. Each
/// entry shares its child's key string. A shape that many
/// different first keys hang off (the root, when objects each start with a
/// distinct key: `{["id" + i]: …}`, a JSON array of single-key objects, or a
/// rebuild-from-root after `delete`) switches to an ordered map, so finding a
/// child stays logarithmic rather than a scan of every key ever added there,
/// which made building `n` such objects quadratic. Entries whose child shape
/// has been freed are swept out whenever the map has doubled since the last
/// sweep, so the cache cannot grow without bound either.
#[derive(Default)]
enum Transitions {
    /// No child yet.
    #[default]
    None,
    /// One child.
    One(Rc<str>, Weak<Shape>),
    /// A few children: scanned linearly.
    List(Vec<(Rc<str>, Weak<Shape>)>),
    /// Many children: looked up by key. `sweep_at` is the size at which dead
    /// entries are next swept.
    Map {
        map: BTreeMap<Rc<str>, Weak<Shape>>,
        sweep_at: usize,
    },
}

impl Transitions {
    /// The live child shape cached for `key`, if any.
    fn get(&self, key: &str) -> Option<Rc<Shape>> {
        match self {
            Transitions::None => None,
            Transitions::One(k, child) => {
                if k.as_ref() == key {
                    child.upgrade()
                } else {
                    None
                }
            }
            Transitions::List(list) => list
                .iter()
                .find(|(k, _)| k.as_ref() == key)
                .and_then(|(_, child)| child.upgrade()),
            Transitions::Map { map, .. } => map.get(key).and_then(Weak::upgrade),
        }
    }

    /// Calls `f` on every live cached child shape.
    fn for_each_live(&self, mut f: impl FnMut(Rc<Shape>)) {
        match self {
            Transitions::None => {}
            Transitions::One(_, child) => {
                if let Some(c) = child.upgrade() {
                    f(c);
                }
            }
            Transitions::List(list) => list.iter().filter_map(|(_, c)| c.upgrade()).for_each(f),
            Transitions::Map { map, .. } => map.values().filter_map(Weak::upgrade).for_each(f),
        }
    }

    /// Caches `child` as the shape reached by adding `key` (the child's own
    /// key string), replacing a dead entry for the same key rather than
    /// adding a second one.
    fn insert(&mut self, key: &Rc<str>, child: &Rc<Shape>) {
        let weak = Rc::downgrade(child);
        match self {
            Transitions::None => *self = Transitions::One(Rc::clone(key), weak),
            Transitions::One(k, c) => {
                if k == key || c.strong_count() == 0 {
                    *k = Rc::clone(key);
                    *c = weak;
                } else {
                    let first = (Rc::clone(k), core::mem::take(c));
                    *self = Transitions::List(alloc::vec![first, (Rc::clone(key), weak)]);
                }
            }
            Transitions::List(list) => {
                if let Some(entry) = list.iter_mut().find(|(k, _)| k == key) {
                    entry.1 = weak;
                    return;
                }
                if list.len() >= TRANSITION_LIST_MAX {
                    list.retain(|(_, c)| c.strong_count() > 0);
                }
                if list.len() < TRANSITION_LIST_MAX {
                    list.push((Rc::clone(key), weak));
                    return;
                }
                let mut map: BTreeMap<Rc<str>, Weak<Shape>> = list.drain(..).collect();
                map.insert(Rc::clone(key), weak);
                let sweep_at = 2 * map.len();
                *self = Transitions::Map { map, sweep_at };
            }
            Transitions::Map { map, sweep_at } => {
                map.insert(Rc::clone(key), weak);
                if map.len() >= *sweep_at {
                    map.retain(|_, c| c.strong_count() > 0);
                    *sweep_at = (2 * map.len()).max(2 * TRANSITION_LIST_MAX);
                }
            }
        }
    }
}

/// The internal key under which a primitive wrapper object boxes its value.
pub const PRIM_WRAP_KEY: &str = "\u{0}prim";

/// Internal brand keys whose presence a [`Shape`] records as a flag bit (bit
/// `i` for `BRAND_KEYS[i]`): the primitive-wrapper slot and the
/// `[[IsHTMLDDA]]` brand.
pub const BRAND_KEYS: [&str; 2] = [PRIM_WRAP_KEY, "\u{0}htmldda"];

/// The prefix of a property key that encodes a Symbol (`\0sym:{id}`).
pub const SYMBOL_KEY_PREFIX: &str = "\u{0}sym:";

/// The `brands` bit recording that a layout holds at least one Symbol key
/// ([`SYMBOL_KEY_PREFIX`]), so the collector's symbol-liveness scan can skip
/// every object whose layout has none without walking its keys. Kept clear of
/// the [`BRAND_KEYS`] bits.
const SYMBOL_KEYS_BIT: u8 = 0x80;

/// The `brands` bit of the one layout made by
/// [`Shape::builtin_fn_layout`]: an object with exactly this layout holds
/// `length` and `name` as non-writable, non-enumerable data properties
/// without listing them in its attribute marks (see
/// `Object::builtin_fn_props`). Never inherited by a child layout.
const IMPLIED_FN_ATTRS_BIT: u8 = 0x40;

/// The [`BRAND_KEYS`] bit of `key` (0 for any other key).
#[inline]
#[must_use]
pub fn brand_bit(key: &str) -> u8 {
    if !key.starts_with('\0') {
        return 0;
    }
    BRAND_KEYS
        .iter()
        .position(|b| *b == key)
        .map_or(0, |i| 1 << i)
}

/// Every `brands` bit `key` sets: its [`brand_bit`], plus [`SYMBOL_KEYS_BIT`]
/// for a Symbol key.
#[inline]
fn key_bits(key: &str) -> u8 {
    if !key.starts_with('\0') {
        return 0;
    }
    brand_bit(key)
        | if key.starts_with(SYMBOL_KEY_PREFIX) {
            SYMBOL_KEYS_BIT
        } else {
            0
        }
}

impl Shape {
    /// The root (empty) shape. Every object's layout descends from it.
    #[must_use]
    pub fn root() -> Rc<Shape> {
        Rc::new(Shape {
            parent: None,
            key: None,
            slot: 0,
            len: 0,
            transitions: RefCell::new(Transitions::default()),
            index: OnceCell::new(),
            probes: Cell::new(0),
            brands: 0,
            any: Cell::new(0),
        })
    }

    /// The number of properties in this layout.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// The layout this one extends (without its last property), or `None`
    /// for the root.
    #[must_use]
    pub fn parent(&self) -> Option<&Rc<Shape>> {
        self.parent.as_ref()
    }

    /// The byte offset of the brand bits within a `Shape` (read by compiled
    /// code: bit 0 is [`is_wrapper`](Self::is_wrapper)).
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn brands_offset() -> usize {
        core::mem::offset_of!(Shape, brands)
    }

    /// The byte offset of the [`any`](Self::may_be_non_number) slot mask (a
    /// `u64`) within a `Shape`, read by compiled code.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn any_offset() -> usize {
        core::mem::offset_of!(Shape, any)
    }

    /// Whether some object with this layout may hold something other than a
    /// Number in slot `slot` (always `true` from slot 64 on, which is not
    /// tracked). `false` is a promise for as long as it stays `false`: every
    /// object with this layout holds a Number there.
    #[inline]
    #[must_use]
    pub fn may_be_non_number(&self, slot: u32) -> bool {
        slot >= 64 || self.any.get() & (1u64 << slot) != 0
    }

    /// The slots `i < min(len, 64)` this layout still has as Number-only
    /// (see [`may_be_non_number`](Self::may_be_non_number)), as a bit mask.
    #[inline]
    #[must_use]
    pub fn number_only_slots(&self) -> u64 {
        let all = if self.len >= 64 {
            !0
        } else {
            (1u64 << self.len) - 1
        };
        all & !self.any.get()
    }

    /// Records that an object with this layout now holds a value that is not
    /// a Number in slot `slot` (or is about to). Must be called by every
    /// write of such a value into a slot, and for every slot of an object
    /// that takes this layout other than by adding a key to its parent's
    /// (whose mask a child always covers).
    ///
    /// The mark is one-way, and it is made on every live layout descending
    /// from this one too: an object of this layout keeps the value when it
    /// adds a key, and a layout created later starts from its parent's
    /// mask. (A child whose own bit is already set has, by the same rule,
    /// marked all of its own descendants.)
    #[inline]
    pub fn note_non_number(&self, slot: u32) {
        if slot < 64 {
            self.note_non_numbers(1u64 << slot);
        }
    }

    /// [`note_non_number`](Self::note_non_number) for every slot `i < 64`
    /// whose bit is set in `mask`.
    #[inline]
    pub fn note_non_numbers(&self, mask: u64) {
        if mask & !self.any.get() != 0 {
            self.downgrade(mask);
        }
    }

    /// The slow half of [`note_non_numbers`](Self::note_non_numbers): marks
    /// `mask` here and on every live descendant that lacks it.
    #[cold]
    #[inline(never)]
    fn downgrade(&self, mask: u64) {
        self.any.set(self.any.get() | mask);
        let mut stack: Vec<Rc<Shape>> = Vec::new();
        self.transitions.borrow().for_each_live(|c| stack.push(c));
        while let Some(s) = stack.pop() {
            let missing = mask & !s.any.get();
            if missing == 0 {
                continue;
            }
            s.any.set(s.any.get() | missing);
            s.transitions.borrow().for_each_live(|c| stack.push(c));
        }
    }

    /// The byte offset of the `parent` link (an `Option<Rc<Shape>>`, one
    /// nullable pointer word) within a [`Shape`], for compiled code's
    /// transition check.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn parent_offset() -> usize {
        core::mem::offset_of!(Shape, parent)
    }

    /// Whether the layout holds the primitive-wrapper key [`PRIM_WRAP_KEY`].
    #[inline]
    #[must_use]
    pub fn is_wrapper(&self) -> bool {
        self.brands & 1 != 0
    }

    /// Whether the layout holds the brand key with [`brand_bit`] `bit`.
    #[inline]
    #[must_use]
    pub fn has_brand(&self, bit: u8) -> bool {
        self.brands & bit != 0
    }

    /// Whether the layout holds any Symbol key ([`SYMBOL_KEY_PREFIX`]).
    #[inline]
    #[must_use]
    pub fn has_symbol_keys(&self) -> bool {
        self.brands & SYMBOL_KEYS_BIT != 0
    }

    /// Calls `f` with every key of this layout, newest first, without
    /// allocating (unlike [`keys`](Self::keys)).
    pub fn for_each_key_unordered(&self, mut f: impl FnMut(&str)) {
        if let Some(ix) = self.index.get() {
            ix.keys.iter().for_each(|k| f(k));
            return;
        }
        let mut node = self;
        loop {
            if let Some(k) = &node.key {
                f(k);
            }
            match node.parent.as_deref() {
                Some(p) => node = p,
                None => break,
            }
        }
    }

    /// The key this layout adds over its parent (the newest one, in the last
    /// slot), or `None` for the root.
    #[inline]
    #[must_use]
    pub fn last_key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// Whether this is the empty (root) layout.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The shape reached by adding property `key` to this one. If `key` is
    /// already present this shape is returned unchanged; otherwise the child
    /// shape is created once and cached, so the same structure always maps to
    /// the same shape (structural sharing). The new property takes the next
    /// slot (`self.len()`).
    pub fn transition(self: &Rc<Self>, key: &str) -> Rc<Shape> {
        // Already present: no layout change.
        if self.lookup(key).is_some() {
            return Rc::clone(self);
        }
        self.add(key)
    }

    /// [`transition`](Self::transition) for a `key` the caller has just found
    /// absent from this layout (so the presence check is not repeated). Adding
    /// a key that *is* present would corrupt the layout, so callers must have
    /// looked it up on this very shape.
    pub fn add(self: &Rc<Self>, key: &str) -> Rc<Shape> {
        debug_assert!(self.lookup(key).is_none(), "Shape::add of a present key");
        // Reuse the cached transition if this `+key` was taken before and the
        // child is still in use. (The `let` ends the borrow before the
        // `borrow_mut` below.)
        let cached = self.transitions.borrow().get(key);
        if let Some(child) = cached {
            return child;
        }
        let key: Rc<str> = Rc::from(key);
        let child = Rc::new(Shape {
            parent: Some(Rc::clone(self)),
            key: Some(Rc::clone(&key)),
            slot: self.len,
            len: self.len + 1,
            transitions: RefCell::new(Transitions::default()),
            index: OnceCell::new(),
            probes: Cell::new(0),
            brands: (self.brands & !IMPLIED_FN_ATTRS_BIT) | key_bits(&key),
            // What may be anything in the parent's slots may be here too:
            // an object that adds a key keeps its other values.
            any: Cell::new(self.any.get()),
        });
        self.transitions.borrow_mut().insert(&key, &child);
        child
    }

    /// The `{length, name}` layout of a built-in function's own properties,
    /// grown from `root`, marked so that an object with it has both keys
    /// non-writable and non-enumerable by layout alone (see
    /// [`has_implied_fn_attrs`](Self::has_implied_fn_attrs)). Its `name` node
    /// is not cached as `root + length`'s `name` child: that child stays the
    /// ordinary layout, which [`unmarked_twin`](Self::unmarked_twin) finds.
    #[must_use]
    pub fn builtin_fn_layout(root: &Rc<Self>) -> Rc<Shape> {
        debug_assert!(root.is_empty());
        let length = root.transition("length");
        Rc::new(Shape {
            parent: Some(Rc::clone(&length)),
            key: Some(Rc::from("name")),
            slot: length.len,
            len: length.len + 1,
            transitions: RefCell::new(Transitions::default()),
            index: OnceCell::new(),
            probes: Cell::new(0),
            brands: (length.brands & !IMPLIED_FN_ATTRS_BIT) | IMPLIED_FN_ATTRS_BIT,
            any: Cell::new(length.any.get()),
        })
    }

    /// Whether this is a [`builtin_fn_layout`](Self::builtin_fn_layout).
    #[inline]
    #[must_use]
    pub fn has_implied_fn_attrs(&self) -> bool {
        self.brands & IMPLIED_FN_ATTRS_BIT != 0
    }

    /// The ordinary layout with the same keys in the same slots as this
    /// [`builtin_fn_layout`](Self::builtin_fn_layout) (`self` for any other).
    #[must_use]
    pub fn unmarked_twin(self: &Rc<Self>) -> Rc<Shape> {
        match (&self.parent, &self.key) {
            (Some(parent), Some(key)) if self.has_implied_fn_attrs() => parent.transition(key),
            _ => Rc::clone(self),
        }
    }

    /// The slot index of `key` in this layout, or `None` if absent. A small
    /// layout walks the parent chain (newest property first); a large one that
    /// is looked up repeatedly answers from its hashed index.
    #[must_use]
    #[inline]
    pub fn lookup(&self, key: &str) -> Option<u32> {
        if self.len >= INDEX_MIN_LEN {
            return self.indexed_lookup(key);
        }
        self.walk_lookup(key)
    }

    /// [`lookup`](Self::lookup) for a large layout: through the index, built
    /// on the first lookup past [`INDEX_AFTER_PROBES`].
    fn indexed_lookup(&self, key: &str) -> Option<u32> {
        if let Some(ix) = self.index.get() {
            return ix.find(key);
        }
        let n = self.probes.get() + 1;
        self.probes.set(n);
        if n >= INDEX_AFTER_PROBES {
            return self
                .index
                .get_or_init(|| Box::new(ShapeIndex::build(self)))
                .find(key);
        }
        self.walk_lookup(key)
    }

    /// The parent-chain walk behind [`lookup`](Self::lookup).
    fn walk_lookup(&self, key: &str) -> Option<u32> {
        let mut node = self;
        loop {
            if let Some(k) = &node.key
                && k.as_ref() == key
            {
                return Some(node.slot);
            }
            node = node.parent.as_deref()?;
        }
    }

    /// Whether `key` is present in this layout.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.lookup(key).is_some()
    }

    /// The property names in slot order (slot 0 first).
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        if let Some(ix) = self.index.get() {
            return ix.keys.iter().map(|k| &**k).collect();
        }
        let mut out: Vec<&str> = Vec::with_capacity(self.len as usize);
        let mut node = self;
        loop {
            if let Some(k) = &node.key {
                out.push(k.as_ref());
            }
            match node.parent.as_deref() {
                Some(p) => node = p,
                None => break,
            }
        }
        out.reverse(); // collected newest-first; present in insertion order
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_then_many_children_are_each_reused() {
        let root = Shape::root();
        let a = root.add("a");
        assert!(Rc::ptr_eq(&a, &root.add("a")));
        let b = root.add("b");
        assert!(Rc::ptr_eq(&a, &root.add("a")) && Rc::ptr_eq(&b, &root.add("b")));
        // A dead only child is replaced, not kept beside a new one.
        let x = a.add("x");
        drop(x);
        let y = a.add("y");
        assert!(matches!(&*a.transitions.borrow(), Transitions::One(k, _) if &**k == "y"));
        assert!(Rc::ptr_eq(&y, &a.add("y")));
        let x2 = a.add("x");
        assert_eq!(x2.lookup("x"), Some(1));
        // Past the list size, children still resolve.
        let kids: Vec<Rc<Shape>> = (0..40).map(|i| b.add(&alloc::format!("k{i}"))).collect();
        for (i, k) in kids.iter().enumerate() {
            assert!(Rc::ptr_eq(k, &b.add(&alloc::format!("k{i}"))));
        }
    }

    #[test]
    fn builtin_fn_layout_is_marked_but_its_children_are_not() {
        let root = Shape::root();
        let l = Shape::builtin_fn_layout(&root);
        assert!(l.has_implied_fn_attrs());
        assert_eq!(l.keys(), ["length", "name"]);
        let plain = l.unmarked_twin();
        assert!(!plain.has_implied_fn_attrs());
        assert!(Rc::ptr_eq(&plain, &root.add("length").add("name")));
        assert!(!l.add("prototype").has_implied_fn_attrs());
    }

    #[test]
    fn root_is_empty() {
        let root = Shape::root();
        assert!(root.is_empty());
        assert_eq!(root.len(), 0);
        assert_eq!(root.lookup("x"), None);
        assert!(root.keys().is_empty());
    }

    #[test]
    fn transition_assigns_sequential_slots() {
        let s = Shape::root();
        let s = s.transition("x");
        let s = s.transition("y");
        let s = s.transition("z");
        assert_eq!(s.len(), 3);
        assert_eq!(s.lookup("x"), Some(0));
        assert_eq!(s.lookup("y"), Some(1));
        assert_eq!(s.lookup("z"), Some(2));
        assert_eq!(s.lookup("w"), None);
        assert_eq!(s.keys(), ["x", "y", "z"]);
    }

    #[test]
    fn same_structure_shares_one_shape() {
        // Two objects built {a, b} from the same root reach the *same* shape
        // (structural sharing holds within a root's transition tree).
        let root = Shape::root();
        let o1 = root.transition("a").transition("b");
        let o2 = root.transition("a").transition("b");
        assert!(Rc::ptr_eq(&o1, &o2));
    }

    #[test]
    fn divergent_structure_forks_the_tree() {
        let base = Shape::root().transition("a");
        let with_b = base.transition("b");
        let with_c = base.transition("c");
        assert!(!Rc::ptr_eq(&with_b, &with_c));
        assert_eq!(with_b.lookup("b"), Some(1));
        assert_eq!(with_b.lookup("c"), None);
        assert_eq!(with_c.lookup("c"), Some(1));
        // The shared prefix is the same node.
        assert_eq!(with_b.lookup("a"), Some(0));
        assert_eq!(with_c.lookup("a"), Some(0));
    }

    #[test]
    fn re_adding_an_existing_key_is_a_no_op() {
        let s = Shape::root().transition("x").transition("y");
        let again = s.transition("x");
        assert!(Rc::ptr_eq(&s, &again));
        assert_eq!(again.len(), 2);
    }

    #[test]
    fn a_shape_subtree_is_freed_with_its_last_user() {
        // The transition cache holds children *weakly*, so a shape nobody uses is
        // released. A strong cache would close a parent↔child cycle and leak the
        // whole transition tree of every realm (~465 kB per realm, which is what
        // made a long-lived process running many scripts grow without bound).
        let root = Shape::root();
        {
            let child = root.transition("a").transition("b");
            assert_eq!(child.len(), 2);
            assert!(Rc::strong_count(&root) > 1, "the child chain roots here");
        }
        assert_eq!(
            Rc::strong_count(&root),
            1,
            "dropping the last user must free the subtree"
        );
        // Rebuilding after the drop yields an equivalent layout.
        let again = root.transition("a").transition("b");
        assert_eq!(again.keys(), ["a", "b"]);
        assert_eq!(again.lookup("b"), Some(1));
    }

    #[test]
    fn large_layouts_answer_from_their_index() {
        let names: Vec<alloc::string::String> = (0..100).map(|i| alloc::format!("k{i}")).collect();
        let mut s = Shape::root();
        for n in &names {
            s = s.transition(n);
        }
        // Enough probes to build the index, then every key and some misses.
        for _ in 0..(INDEX_AFTER_PROBES + 1) {
            assert_eq!(s.lookup("k42"), Some(42));
        }
        assert!(s.index.get().is_some(), "a hot large layout is indexed");
        for (i, n) in names.iter().enumerate() {
            assert_eq!(s.lookup(n), Some(i as u32));
        }
        assert_eq!(s.lookup("k100"), None);
        assert_eq!(s.lookup(""), None);
        assert_eq!(s.lookup("k4"), Some(4));
        let keys: Vec<&str> = s.keys();
        assert_eq!(keys.len(), 100);
        assert_eq!(keys[0], "k0");
        assert_eq!(keys[99], "k99");
        assert_ne!(key_hash(""), 0);
        assert_ne!(key_hash("k1"), key_hash("k2"));
    }

    #[test]
    fn brand_keys_are_flagged_through_transitions() {
        assert_eq!(BRAND_KEYS[1], crate::realm::HTMLDDA_SLOT);
        let plain = Shape::root().transition("a").transition("b");
        assert!(!plain.is_wrapper());
        let w = Shape::root()
            .transition("a")
            .transition(PRIM_WRAP_KEY)
            .transition("z");
        assert!(w.is_wrapper());
        assert!(!w.has_brand(brand_bit(BRAND_KEYS[1])));
        let d = plain.transition(BRAND_KEYS[1]);
        assert!(d.has_brand(brand_bit(BRAND_KEYS[1])) && !d.is_wrapper());
        assert_eq!(brand_bit("prim"), 0);
    }

    #[test]
    fn symbol_keys_are_flagged_through_transitions() {
        let plain = Shape::root().transition("a").transition("\u{0}hidden");
        assert!(!plain.has_symbol_keys());
        let sym = plain.transition("\u{0}sym:12").transition("z");
        assert!(sym.has_symbol_keys() && !sym.is_wrapper());
        // The flag is not a brand bit.
        assert!(
            !sym.has_brand(brand_bit(BRAND_KEYS[0])) && !sym.has_brand(brand_bit(BRAND_KEYS[1]))
        );
        let mut seen = Vec::new();
        sym.for_each_key_unordered(|k| seen.push(alloc::string::String::from(k)));
        seen.sort();
        assert_eq!(seen, ["\u{0}hidden", "\u{0}sym:12", "a", "z"]);
    }

    #[test]
    fn property_order_differs_by_insertion_order() {
        let xy = Shape::root().transition("x").transition("y");
        let yx = Shape::root().transition("y").transition("x");
        // Different layouts (different slot assignment), so different shapes.
        assert!(!Rc::ptr_eq(&xy, &yx));
        assert_eq!(xy.lookup("x"), Some(0));
        assert_eq!(yx.lookup("x"), Some(1));
        assert_eq!(xy.keys(), ["x", "y"]);
        assert_eq!(yx.keys(), ["y", "x"]);
    }

    #[test]
    fn non_number_marks_reach_every_live_descendant() {
        let root = Shape::root();
        let x = root.transition("x");
        let xy = x.transition("y");
        let xyz = xy.transition("z");
        let xw = x.transition("w");
        assert!(!xyz.may_be_non_number(0) && !xw.may_be_non_number(0));
        // Marking `x`'s slot 0 marks every layout grown from it.
        x.note_non_number(0);
        for s in [&x, &xy, &xyz, &xw] {
            assert!(s.may_be_non_number(0));
        }
        assert!(!xyz.may_be_non_number(1) && !xyz.may_be_non_number(2));
        // A child made later starts from its parent's marks.
        let xv = x.transition("v");
        assert!(xv.may_be_non_number(0) && !xv.may_be_non_number(1));
        // A deeper mark leaves the ancestors alone.
        xy.note_non_numbers(0b10);
        assert!(xyz.may_be_non_number(1) && !x.may_be_non_number(1) && !xw.may_be_non_number(1));
        // Untracked slots always may be anything.
        assert!(root.may_be_non_number(64) && !root.may_be_non_number(63));
        root.note_non_number(200);
    }
}
