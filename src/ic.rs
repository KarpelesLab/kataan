//! Inline caches for property access (`ROADMAP.md` §3, item 2).
//!
//! A property-access bytecode site (e.g. `obj.x`) resolves a name to a slot via
//! its object's [`Shape`](crate::shape::Shape). Doing the shape lookup every time is wasteful: most
//! sites see the *same* shape repeatedly (objects built the same way share one).
//! An **inline cache** memoizes the last shape→slot resolution at the site, so a
//! repeat access on a matching shape is a pointer compare plus a slot load — no
//! name lookup. This is, per the roadmap, the single largest lever for
//! real-world JS speed.
//!
//! `PropertyCache` is **monomorphic**: it remembers one shape. The hit/miss
//! counters it keeps are also the **type feedback** an optimizing JIT consumes —
//! a site that is overwhelmingly monomorphic is a prime inlining/specialization
//! candidate. (Polymorphic caches — a small set of shapes — and megamorphic
//! fallback layer on top.)
//!
//! Pure, safe `alloc`-only Rust: shape identity is an `Rc` pointer comparison,
//! no `unsafe`.

use crate::shape::Shape;
use alloc::rc::Rc;

/// A monomorphic inline cache for one property-access site: it remembers the
/// last shape it resolved and the slot the property lived in.
#[derive(Default)]
#[repr(C)] // JIT inline fast path reads `shape`/`slot` offsets; verified in jit layout test
pub struct PropertyCache {
    /// The cached shape, if the site has been warmed.
    shape: Option<Rc<Shape>>,
    /// The slot `key` resolved to under `shape`.
    slot: u32,
    /// Times the cached shape matched (fast path taken).
    hits: u32,
    /// Times it did not (cold, shape change, or absent property).
    misses: u32,
    /// The site's prototype-chain entry (see [`ChainCache`]), armed by a lookup
    /// that had to look past the receiver's own data slots.
    pub chain: Option<alloc::boxed::Box<ChainCache>>,
    /// How many times `chain` was (re)armed: a site that keeps missing is
    /// polymorphic, and stops paying for re-arming after a few.
    pub chain_fills: u16,
    /// The site's built-in method entry, for a method-call site (see
    /// `crate::nbvm::native_call`): which intrinsic the call resolved to and
    /// how to tell it still does.
    pub(crate) native: Option<alloc::boxed::Box<crate::nbvm::native_call::NativeSite>>,
    /// Whether `shape` was last armed by an add (see
    /// [`arm_transition`](Self::arm_transition)) rather than a lookup.
    by_add: bool,
}

/// How many times a site's [`ChainCache`] may be (re)armed.
pub const CHAIN_FILL_LIMIT: u16 = 16;

/// A prototype-chain inline cache: where a lookup of the site's key found its
/// property, as the shapes of every object it visited, receiver first and
/// holder last, plus the holder's slot (or the fact that the holder has an
/// accessor of that name).
///
/// It is validated, never trusted: a hit re-walks the *live* `[[Prototype]]`
/// links from the receiver and requires each object to be an ordinary shaped
/// object whose shape is pointer-identical to the recorded one (the `Rc`s here
/// keep those shapes alive, so identity cannot be faked by address reuse).
/// A shape fixes an object's data keys and their slots, so an equal shape
/// proves an intermediate object still lacks the key as a data property and
/// the holder still has it at `slot`. Accessors live outside the shape, so the
/// walk re-checks them on every visited object. A cached handle is never
/// dereferenced; only links read from live objects are.
pub struct ChainCache {
    /// The shapes along the chain, receiver first, holder last.
    pub shapes: alloc::vec::Vec<Rc<Shape>>,
    /// The holder's slot for a data property.
    pub slot: u32,
    /// Whether the holder's property is an accessor (re-read on each hit).
    pub accessor: bool,
    /// Whether no object on the chain has the key: `shapes` is then the whole
    /// chain, and its last object's `[[Prototype]]` is null.
    pub absent: bool,
}

impl PropertyCache {
    /// A cold cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves `key` for an object of shape `object_shape`, using the cache.
    /// On a hit (same shape as last time) it returns the cached slot without a
    /// name lookup; on a miss it resolves through the shape and re-arms the
    /// cache. Returns `None` if the property is absent (counted as a miss).
    pub fn lookup(&mut self, object_shape: &Rc<Shape>, key: &str) -> Option<u32> {
        if let Some(cached) = &self.shape
            && Rc::ptr_eq(cached, object_shape)
        {
            self.hits += 1;
            return Some(self.slot);
        }
        self.misses += 1;
        let slot = object_shape.lookup(key)?;
        self.shape = Some(Rc::clone(object_shape));
        self.slot = slot;
        self.by_add = false;
        Some(slot)
    }

    /// The **transition** use of a cache, for a site that adds one fixed
    /// property to objects (an object literal's `key: value`): the cached
    /// shape is the *result* of the add, so a hit — `from` is exactly its
    /// parent — hands back the child layout with no name lookup and no walk
    /// of the transition table. `None` on a miss.
    pub fn transition_from(&mut self, from: &Rc<Shape>) -> Option<Rc<Shape>> {
        if let Some(to) = &self.shape
            && to.parent().is_some_and(|p| Rc::ptr_eq(p, from))
        {
            self.hits += 1;
            return Some(Rc::clone(to));
        }
        self.misses += 1;
        None
    }

    /// Arms a transition cache (see [`transition_from`](Self::transition_from))
    /// with the shape an add produced.
    pub fn arm_transition(&mut self, to: Rc<Shape>) {
        self.slot = to.len().saturating_sub(1);
        self.shape = Some(to);
        self.by_add = true;
    }

    /// Whether the cache is warmed (has resolved at least one shape).
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.shape.is_some()
    }

    /// The number of fast-path hits so far (type feedback).
    #[must_use]
    pub fn hits(&self) -> u32 {
        self.hits
    }

    /// The number of misses so far (type feedback).
    #[must_use]
    pub fn misses(&self) -> u32 {
        self.misses
    }

    /// Discards the cached shape (e.g. when the site goes megamorphic).
    pub fn clear(&mut self) {
        self.shape = None;
    }

    /// The cache's `(shape, slot)` memo and its counters, for a compiler
    /// that starts its own copy of the site warm (see [`WarmEntry`]).
    #[must_use]
    pub fn warm_entry(&self) -> Option<WarmEntry> {
        self.shape.as_ref().map(|s| WarmEntry {
            shape: Rc::clone(s),
            slot: self.slot,
            hits: self.hits,
            misses: self.misses,
            by_add: self.by_add,
        })
    }

    /// The memo of a cache whose counters say it is settled (as
    /// [`WarmEntry::is_stable`]): `(shape, slot, by_add)`, without taking a
    /// reference to the shape.
    #[must_use]
    pub fn stable_memo(&self) -> Option<(&Rc<Shape>, u32, bool)> {
        if self.hits < 4 || self.hits < self.misses.saturating_mul(4) {
            return None;
        }
        self.shape.as_ref().map(|s| (s, self.slot, self.by_add))
    }

    /// Arms the cache with `warm`'s memo (counters start afresh).
    pub fn seed(&mut self, warm: &WarmEntry) {
        self.shape = Some(Rc::clone(&warm.shape));
        self.slot = warm.slot;
        self.by_add = warm.by_add;
    }
}

/// A warm site's memo, as [`PropertyCache::warm_entry`] reports it: the shape
/// last resolved and its slot for the site's key (a fact of that immutable
/// layout), and how often the site hit or missed while it was interpreted —
/// the type feedback that says whether the site is monomorphic.
#[derive(Clone)]
pub struct WarmEntry {
    /// The cached shape (kept alive by this `Rc`).
    pub shape: Rc<Shape>,
    /// The slot the site's key has under `shape`.
    pub slot: u32,
    /// Fast-path hits so far.
    pub hits: u32,
    /// Misses so far (the cold one included).
    pub misses: u32,
    /// Whether the memo was recorded by an add (a write site whose objects
    /// arrive *without* the key: `shape` is the add's result).
    pub by_add: bool,
}

impl WarmEntry {
    /// Whether the site looks monomorphic and settled: it hit several times
    /// as often as it missed (a site cycling through layouts misses about
    /// as often as it hits).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.hits >= 4 && self.hits >= self.misses.saturating_mul(4)
    }
}

/// Runtime-probed byte layout of [`PropertyCache`] for the generic-JIT inline
/// property-get fast path. Derived by pointer arithmetic on a real instance
/// (never hand-baked); verified by the JIT layout test. `PropertyCache` is
/// `#[repr(C)]`, so the field order is stable, but the offsets are still probed
/// rather than assumed.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheLayout {
    /// Byte offset of the `shape: Option<Rc<Shape>>` field (the 8-byte word the
    /// inline guard compares against the object's shape pointer). `None` stores a
    /// null pointer (niche), which never matches a live shape → the guard misses.
    pub shape_off: usize,
    /// Byte offset of the cached `slot: u32`.
    pub slot_off: usize,
    /// Byte offset of the `hits: u32` counter (bumped in place on an inline hit).
    pub hits_off: usize,
}

/// Derives [`CacheLayout`] from a real [`PropertyCache`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn jit_cache_layout() -> CacheLayout {
    let pc = PropertyCache::new();
    let base = core::ptr::addr_of!(pc) as usize;
    CacheLayout {
        shape_off: core::ptr::addr_of!(pc.shape) as usize - base,
        slot_off: core::ptr::addr_of!(pc.slot) as usize - base,
        hits_off: core::ptr::addr_of!(pc.hits) as usize - base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::Object;

    #[test]
    fn first_lookup_misses_then_repeats_hit() {
        let root = Shape::root();
        let mut obj = Object::new(Rc::clone(&root));
        obj.set("x", crate::nanbox::NanBox::number(1.0));
        obj.set("y", crate::nanbox::NanBox::number(2.0));

        let mut ic = PropertyCache::new();
        assert!(!ic.is_warm());

        // Cold: a miss that warms the cache.
        assert_eq!(ic.lookup(obj.shape(), "y"), Some(1));
        assert!(ic.is_warm());
        assert_eq!((ic.hits(), ic.misses()), (0, 1));

        // Same shape again: fast-path hits.
        assert_eq!(ic.lookup(obj.shape(), "y"), Some(1));
        assert_eq!(ic.lookup(obj.shape(), "y"), Some(1));
        assert_eq!((ic.hits(), ic.misses()), (2, 1));
    }

    #[test]
    fn shape_change_re_arms_the_cache() {
        let root = Shape::root();
        let mut a = Object::new(Rc::clone(&root));
        a.set("p", crate::nanbox::NanBox::number(1.0));
        // `b` has a different shape (extra property → different layout).
        let mut b = Object::new(Rc::clone(&root));
        b.set("p", crate::nanbox::NanBox::number(1.0));
        b.set("q", crate::nanbox::NanBox::number(2.0));

        let mut ic = PropertyCache::new();
        assert_eq!(ic.lookup(a.shape(), "p"), Some(0)); // miss, arm on a's shape
        assert_eq!(ic.lookup(a.shape(), "p"), Some(0)); // hit
        assert_eq!(ic.lookup(b.shape(), "p"), Some(0)); // miss: different shape
        assert_eq!(ic.lookup(b.shape(), "p"), Some(0)); // hit on b's shape now
        assert_eq!((ic.hits(), ic.misses()), (2, 2));
    }

    #[test]
    fn two_objects_one_shape_share_the_fast_path() {
        // Same structure → same shape → the cache armed by one hits for the
        // other (the whole point of shapes + ICs).
        let root = Shape::root();
        let mut a = Object::new(Rc::clone(&root));
        a.set("k", crate::nanbox::NanBox::number(1.0));
        let mut b = Object::new(Rc::clone(&root));
        b.set("k", crate::nanbox::NanBox::number(9.0));
        assert!(Rc::ptr_eq(a.shape(), b.shape()));

        let mut ic = PropertyCache::new();
        assert_eq!(ic.lookup(a.shape(), "k"), Some(0)); // miss (arm)
        assert_eq!(ic.lookup(b.shape(), "k"), Some(0)); // hit: shared shape
        assert_eq!((ic.hits(), ic.misses()), (1, 1));
    }

    #[test]
    fn absent_property_is_a_miss() {
        let root = Shape::root();
        let mut obj = Object::new(root);
        obj.set("x", crate::nanbox::NanBox::number(1.0));
        let mut ic = PropertyCache::new();
        assert_eq!(ic.lookup(obj.shape(), "nope"), None);
        assert_eq!((ic.hits(), ic.misses()), (0, 1));
        assert!(!ic.is_warm());
    }

    #[test]
    fn clear_cools_the_cache() {
        let root = Shape::root();
        let mut obj = Object::new(root);
        obj.set("x", crate::nanbox::NanBox::number(1.0));
        let mut ic = PropertyCache::new();
        ic.lookup(obj.shape(), "x");
        assert!(ic.is_warm());
        ic.clear();
        assert!(!ic.is_warm());
    }
}
