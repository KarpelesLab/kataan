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
        Some(slot)
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
