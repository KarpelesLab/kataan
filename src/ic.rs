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
    /// For an element access site (`obj[key]`, `obj[key] = v`): what the
    /// interpreter saw it access ([`elem_fb`]), the feedback the
    /// machine-code tier orders its element kind tests by.
    elem_kind: u8,
    /// For a call site: which VM closure it calls ([`CalleeHint`]), the
    /// feedback the machine-code tier inlines a callee from.
    callee: CalleeHint,
    /// The layouts the cache held before its current one ([`PolyHist`]),
    /// once it has held two: the feedback a polymorphic site's compiled
    /// code compares a short chain of layouts from.
    hist: Option<alloc::boxed::Box<PolyHist>>,
}

/// The layouts a [`PropertyCache`] was armed with before its current one,
/// most recent first and none of them the current one ([`Seen`]); how many
/// distinct layouts fell off the end (a site that keeps evicting is
/// megamorphic); and the current layout's hits before its present run (it
/// may have been current before) and the hit count its run started at.
#[derive(Default)]
pub struct PolyHist {
    seen: [Option<Seen>; POLY_HIST],
    evictions: u32,
    cur_tally: u32,
    armed_at: u32,
}

/// A layout a [`PropertyCache`] held: the slot the site's key has there (a
/// fact of that immutable layout), the hits the cache counted while it held
/// it, and whether it was last armed by an add (its result: no object
/// arrives with it at the add).
#[derive(Clone)]
pub struct Seen {
    pub shape: Rc<Shape>,
    pub slot: u32,
    pub hits: u32,
    pub by_add: bool,
}

/// How many earlier layouts a [`PolyHist`] keeps.
pub const POLY_HIST: usize = 3;

impl PolyHist {
    /// Notes that the cache leaves `old` for `next`; returns the hits `next`
    /// had counted when it was current before.
    fn note(&mut self, old: Seen, next: &Rc<Shape>) -> u32 {
        // (The one it goes to is current, not earlier.)
        let mut next_hits = 0;
        if let Some(i) = self
            .seen
            .iter()
            .position(|e| e.as_ref().is_some_and(|e| Rc::ptr_eq(&e.shape, next)))
        {
            next_hits = self.seen[i].as_ref().map_or(0, |e| e.hits);
            self.remove(i);
        }
        if self.seen[POLY_HIST - 1].is_some() {
            self.evictions = self.evictions.saturating_add(1);
        }
        for i in (1..POLY_HIST).rev() {
            self.seen[i] = self.seen[i - 1].take();
        }
        self.seen[0] = Some(old);
        next_hits
    }

    /// Removes entry `i`, closing the gap.
    fn remove(&mut self, i: usize) {
        for j in i..POLY_HIST - 1 {
            self.seen[j] = self.seen[j + 1].take();
        }
        self.seen[POLY_HIST - 1] = None;
    }
}

/// What a call site has called: nothing yet, always the one VM function
/// whose id and capture count are the key (`id | n_captures << 32`), a few
/// VM functions (a polymorphic site, `task.run()` over several classes), or
/// more than that.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CalleeHint {
    /// No VM closure called yet.
    #[default]
    None,
    /// Always this key.
    One(u64),
    /// Two to [`CalleeHint::POLY_MAX`] VM functions, by function id in the
    /// order first seen (unused entries [`CalleeHint::NO_FN`]); their keys'
    /// capture counts are the functions' own (a guard compares the whole key
    /// anyway).
    Poly([u32; CalleeHint::POLY_MAX]),
    /// More than [`CalleeHint::Poly`] keeps, or not only VM functions.
    Mixed,
}

impl CalleeHint {
    /// An unused [`CalleeHint::Poly`] entry.
    pub const NO_FN: u32 = u32::MAX;

    /// The most VM functions a [`CalleeHint::Poly`] names.
    pub const POLY_MAX: usize = 6;

    /// The hint after the site also called the VM closure with key `key`.
    #[must_use]
    pub fn with(self, key: u64) -> Self {
        let fid = key as u32;
        match self {
            Self::None => Self::One(key),
            Self::One(k) if k == key => self,
            Self::One(k) if k as u32 == fid || fid == Self::NO_FN || k as u32 == Self::NO_FN => {
                Self::Mixed
            }
            Self::One(k) => {
                let mut f = [Self::NO_FN; Self::POLY_MAX];
                f[0] = k as u32;
                f[1] = fid;
                Self::Poly(f)
            }
            Self::Poly(mut f) => {
                if f.contains(&fid) {
                    return self;
                }
                match f.iter_mut().find(|x| **x == Self::NO_FN) {
                    Some(x) => {
                        *x = fid;
                        Self::Poly(f)
                    }
                    None => Self::Mixed,
                }
            }
            Self::Mixed => Self::Mixed,
        }
    }

    /// The hint of a site that called exactly the closures `keys` (in that
    /// order, without repeats).
    #[must_use]
    pub fn of_keys(keys: &[u64]) -> Self {
        keys.iter().fold(Self::None, |h, &k| h.with(k))
    }

    /// The function ids of a [`CalleeHint::Poly`] (empty for any other).
    #[must_use]
    pub fn poly_fns(&self) -> alloc::vec::Vec<u32> {
        match self {
            Self::Poly(f) => f.iter().copied().filter(|&x| x != Self::NO_FN).collect(),
            _ => alloc::vec::Vec::new(),
        }
    }
}

/// Element access feedback (`PropertyCache::elem_kind`): what an
/// `obj[key]` site's receiver (and, for a plain array, its element) was
/// each time the interpreter ran it, merged into one byte. A hint only:
/// the compiled code tests every case anyway, the feedback only orders the
/// tests (and keeps an integer version from speculating that a site whose
/// elements were never int32s reads int32s).
pub mod elem_fb {
    /// Nothing seen yet.
    pub const NONE: u8 = 0;
    /// A typed-array view of kind `k` (`TYPED_ARRAY_KINDS` index `k`, one
    /// the compiled code serves) is `TYPED + k`.
    pub const TYPED: u8 = 1;
    /// The last typed kind (Float64).
    pub const TYPED_LAST: u8 = TYPED + 8;
    /// A plain array, the element an int32 Number.
    pub const ARR_INT: u8 = 10;
    /// A plain array, the element another integral Number below 2^53 in
    /// magnitude (a uint32 hash word, say: its `ToInt32` bits are exact).
    pub const ARR_NUM: u8 = 11;
    /// A plain array, the element another Number (a fraction, `-0`, NaN,
    /// an infinity, a huge one).
    pub const ARR_FRAC: u8 = 12;
    /// A plain array, the element anything else (a hole, out of bounds, an
    /// object, a string).
    pub const ARR_ANY: u8 = 13;
    /// Anything else (an object, a string, another typed array kind).
    pub const OTHER: u8 = 14;
    /// More than one of the above (other than plain arrays' element types,
    /// which widen).
    pub const MIXED: u8 = 15;

    /// The feedback of a site that saw `a`, then `b`.
    #[must_use]
    pub const fn merge(a: u8, b: u8) -> u8 {
        if a == b || b == NONE {
            a
        } else if a == NONE {
            b
        } else if a >= ARR_INT && a <= ARR_ANY && b >= ARR_INT && b <= ARR_ANY {
            if a > b { a } else { b }
        } else {
            MIXED
        }
    }

    /// The typed-array kind a site's feedback `f` names, if it names one.
    #[must_use]
    pub const fn typed_kind(f: u8) -> Option<u8> {
        if f >= TYPED && f <= TYPED_LAST {
            Some(f - TYPED)
        } else {
            None
        }
    }

    /// Whether a site with feedback `f` was never seen to give an element
    /// an integer version could hold (an int32, or the `ToInt32` bits of an
    /// integral one that only bitwise ops read): a float view's, a plain
    /// array's fraction or non-Number, or anything else's. (No feedback, or
    /// mixed, says nothing.)
    #[must_use]
    pub const fn never_int(f: u8) -> bool {
        matches!(f, ARR_FRAC | ARR_ANY | OTHER) || f == TYPED + 7 || f == TYPED + 8
    }
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
    /// For a chain that starts at a VM closure (a static class member read,
    /// `C.kind` with `kind` on a parent class), which `shapes` cannot describe:
    /// the links instead (see `Realm::fn_chain_get`). `None` otherwise.
    pub fn_links: Option<alloc::boxed::Box<crate::realm::FnChain>>,
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
        self.leave_for(object_shape);
        self.shape = Some(Rc::clone(object_shape));
        self.slot = slot;
        self.by_add = false;
        Some(slot)
    }

    /// Records the current layout (if any) in the history before the cache
    /// is armed with `next` (see [`PolyHist`]).
    fn leave_for(&mut self, next: &Rc<Shape>) {
        let Some(old) = self.shape.take() else {
            return;
        };
        if Rc::ptr_eq(&old, next) {
            self.shape = Some(old);
            return;
        }
        let (slot, by_add, hits) = (self.slot, self.by_add, self.hits);
        let h = self.hist.get_or_insert_with(alloc::boxed::Box::default);
        let seen = Seen {
            shape: old,
            slot,
            hits: h.cur_tally.saturating_add(hits.wrapping_sub(h.armed_at)),
            by_add,
        };
        h.cur_tally = h.note(seen, next);
        h.armed_at = hits;
    }

    /// The layouts the cache held, the current one first (if any), each
    /// with the hits counted while it was current; and how many the
    /// history evicted (see [`PolyHist`]).
    #[must_use]
    pub fn seen(&self) -> (alloc::vec::Vec<Seen>, u32) {
        let mut out = alloc::vec::Vec::new();
        let (tally, evictions) = match &self.hist {
            Some(h) => (
                h.cur_tally
                    .saturating_add(self.hits.wrapping_sub(h.armed_at)),
                h.evictions,
            ),
            None => (self.hits, 0),
        };
        if let Some(s) = &self.shape {
            out.push(Seen {
                shape: Rc::clone(s),
                slot: self.slot,
                hits: tally,
                by_add: self.by_add,
            });
        }
        if let Some(h) = &self.hist {
            out.extend(h.seen.iter().flatten().cloned());
        }
        (out, evictions)
    }

    /// The cached slot when `object_shape` is the cached shape, without
    /// counting the hit or arming anything: the probe of a caller that
    /// falls back to [`lookup`](Self::lookup) on `None`.
    #[inline]
    #[must_use]
    pub fn hit_slot(&self, object_shape: &Rc<Shape>) -> Option<u32> {
        match &self.shape {
            Some(cached) if Rc::ptr_eq(cached, object_shape) => Some(self.slot),
            _ => None,
        }
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
        self.leave_for(&to);
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
        let (mut seen, evictions) = self.seen();
        self.shape.as_ref()?;
        let cur = seen.remove(0);
        Some(WarmEntry {
            shape: cur.shape,
            slot: cur.slot,
            hits: self.hits,
            misses: self.misses,
            by_add: cur.by_add,
            tally: cur.hits,
            others: seen,
            evictions,
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

    /// Records that the call site called the VM closure with key `key`
    /// (see [`CalleeHint`]).
    pub fn note_callee(&mut self, key: u64) {
        self.callee = self.callee.with(key);
    }

    /// The call site's callee feedback (see [`CalleeHint`]).
    #[must_use]
    pub fn callee_hint(&self) -> CalleeHint {
        self.callee
    }

    /// The element access site's feedback ([`elem_fb`]).
    #[must_use]
    pub fn elem_kind(&self) -> u8 {
        self.elem_kind
    }

    /// Notes that the element access site accessed `kind` ([`elem_fb`]).
    #[inline]
    pub fn note_elem(&mut self, kind: u8) {
        if self.elem_kind != kind {
            self.elem_kind = elem_fb::merge(self.elem_kind, kind);
        }
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
    /// The hits counted while the cache held `shape` (all of `hits` for a
    /// cache that never held another).
    pub tally: u32,
    /// The other layouts the cache held, most recent first, with the hits
    /// counted while it held each (`PropertyCache::seen`): a polymorphic
    /// site's other layouts.
    pub others: alloc::vec::Vec<Seen>,
    /// How many distinct layouts the site's history evicted (a site that
    /// keeps evicting is megamorphic).
    pub evictions: u32,
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
    fn element_feedback_merges() {
        use elem_fb::*;
        let mut ic = PropertyCache::new();
        assert_eq!(ic.elem_kind(), NONE);
        ic.note_elem(TYPED + 8);
        ic.note_elem(TYPED + 8);
        assert_eq!(elem_fb::typed_kind(ic.elem_kind()), Some(8));
        assert!(never_int(ic.elem_kind()));
        ic.note_elem(TYPED + 5);
        assert_eq!(ic.elem_kind(), MIXED);
        assert!(!never_int(MIXED));
        // A plain array's element types widen.
        assert_eq!(merge(ARR_INT, ARR_NUM), ARR_NUM);
        assert_eq!(merge(ARR_ANY, ARR_INT), ARR_ANY);
        assert_eq!(merge(ARR_INT, TYPED + 1), MIXED);
        assert_eq!(merge(ARR_NUM, ARR_FRAC), ARR_FRAC);
        assert!(!never_int(ARR_INT) && !never_int(ARR_NUM) && !never_int(TYPED + 1));
        assert!(never_int(ARR_FRAC) && never_int(ARR_ANY) && never_int(TYPED + 8));
    }

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
    fn polymorphic_memo_is_the_most_used_layout() {
        let root = Shape::root();
        let mut a = Object::new(Rc::clone(&root));
        a.set("p", crate::nanbox::NanBox::number(1.0));
        let mut b = Object::new(Rc::clone(&root));
        b.set("q", crate::nanbox::NanBox::number(1.0));
        b.set("p", crate::nanbox::NanBox::number(1.0));
        let mut c = Object::new(Rc::clone(&root));
        c.set("r", crate::nanbox::NanBox::number(1.0));
        c.set("s", crate::nanbox::NanBox::number(1.0));
        c.set("p", crate::nanbox::NanBox::number(1.0));
        let mut ic = PropertyCache::new();
        // `b` mostly, `a` and `c` now and then; the cache ends on `a`.
        for _ in 0..10 {
            ic.lookup(a.shape(), "p");
            for _ in 0..20 {
                ic.lookup(b.shape(), "p");
            }
            ic.lookup(c.shape(), "p");
            ic.lookup(c.shape(), "p");
        }
        ic.lookup(a.shape(), "p");
        let w = ic.warm_entry().expect("warm");
        // The memo is the current layout; the history says which is used most.
        assert!(Rc::ptr_eq(&w.shape, a.shape()));
        assert_eq!(w.tally, 0);
        let hits: alloc::vec::Vec<(u32, u32)> = w.others.iter().map(|o| (o.slot, o.hits)).collect();
        assert_eq!(hits, alloc::vec![(2, 10), (1, 190)]);
        assert_eq!(w.evictions, 0);
        // A fourth and fifth layout evict the oldest.
        let mut d = Object::new(Rc::clone(&root));
        d.set("t", crate::nanbox::NanBox::number(1.0));
        d.set("p", crate::nanbox::NanBox::number(1.0));
        let mut e = Object::new(Rc::clone(&root));
        e.set("u", crate::nanbox::NanBox::number(1.0));
        e.set("v", crate::nanbox::NanBox::number(1.0));
        e.set("w", crate::nanbox::NanBox::number(1.0));
        e.set("p", crate::nanbox::NanBox::number(1.0));
        ic.lookup(d.shape(), "p");
        ic.lookup(e.shape(), "p");
        ic.lookup(a.shape(), "p");
        let w = ic.warm_entry().expect("warm");
        assert_eq!(w.others.len(), POLY_HIST);
        assert!(w.evictions >= 1);
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
