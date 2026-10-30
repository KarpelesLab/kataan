//! A mark-and-sweep tracing collector over [`Heap`](crate::heap::Heap)
//! (`ROADMAP.md` §3, the GC).
//!
//! Reference counting (the interpreter era's `Rc`) cannot reclaim cycles —
//! `a.b = b; b.a = a` leaks. A tracing collector instead starts from a **root
//! set** (the values currently reachable from the running program — stack
//! slots, globals) and marks everything reachable by following each object's
//! outgoing handle edges; whatever is left unmarked is unreachable and is swept
//! (freed). Cycles among unreachable objects are collected correctly because
//! reachability, not in-degree, decides liveness.
//!
//! This is the simplest correct policy (stop-the-world mark-sweep); the
//! generational/incremental refinements layer on top. The collector is generic
//! over any heap element that can enumerate its outgoing handles via the
//! `Trace` trait,
//! so it is exercised independently of the interpreter's object type.
//!
//! Pure, safe `alloc`-only Rust.

use crate::heap::{Handle, Heap};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// A heap object that can report the handles it references, so the collector can
/// follow its outgoing edges during the mark phase.
pub trait Trace {
    /// Calls `visit` once for every handle this object refers to.
    fn trace(&self, visit: &mut dyn FnMut(Handle));

    /// [`trace`](Self::trace) with a statically dispatched visitor — what the
    /// mark loop calls, so an implementation that overrides it lets the
    /// per-edge callback inline instead of going through a `dyn` call. Must
    /// visit exactly what `trace` visits.
    fn trace_each<F: FnMut(Handle)>(&self, mut visit: F)
    where
        Self: Sized,
    {
        self.trace(&mut visit);
    }

    /// How much tracing work this object costs — roughly its slot count. A
    /// collection's cost is the *weight* of the live set, not its object
    /// count, so the allocation trigger is scaled by weight: one object with
    /// a million properties must not be re-traced every few thousand
    /// allocations.
    fn weight(&self) -> usize {
        1
    }

    /// Whether this object can change the handles it references **without a
    /// mutable borrow of its heap slot** (state shared behind an `Rc`, say),
    /// which the generational write barrier — the mutable borrow
    /// ([`Heap::get_mut`]) — cannot see. A minor collection rescans every
    /// such old object, as though the barrier had fired on it.
    fn interior_mutable(&self) -> bool {
        false
    }

    /// Drops a swept object. The default is plain `drop`; an implementation
    /// may special-case its common shapes so the sweep's per-object cost is not
    /// the full drop glue, and hand reusable parts to the heap's `pool`.
    fn release(self, pool: &mut crate::object::ExtraPool)
    where
        Self: Sized,
    {
        let _ = pool;
        drop(self);
    }

    /// [`release`](Self::release)s the object held in `place` (a heap slot
    /// being freed), leaving `None`. The default takes it out first; an
    /// implementation can test the kind in place, so that only the parts of
    /// the common kinds that own memory are moved out, not the whole value.
    #[inline(always)]
    fn release_in(place: &mut Option<Self>, pool: &mut crate::object::ExtraPool)
    where
        Self: Sized,
    {
        if let Some(value) = place.take() {
            value.release(pool);
        }
    }
}

/// What a collection knows to be alive, as handed to the weak-key
/// (ephemeron) hooks: the [`MarkSet`] of a full collection, or, in a minor
/// one, [`MinorLive`] — every old object plus the young ones marked.
pub trait LiveSet {
    /// Whether `handle` is (as far as this collection is concerned) alive.
    fn contains(&self, handle: &Handle) -> bool;
}

impl LiveSet for MarkSet {
    fn contains(&self, handle: &Handle) -> bool {
        MarkSet::contains(self, handle)
    }
}

/// The [`LiveSet`] of a minor collection: a handle is alive if its object is
/// old, or young and reached by the marking so far (which promotes it, so both
/// are "age at least [`OLD_AGE`]").
pub struct MinorLive<'a, T>(pub &'a Heap<T>);

impl<T> LiveSet for MinorLive<'_, T> {
    fn contains(&self, handle: &Handle) -> bool {
        self.0.age(*handle).is_some_and(|a| a >= OLD_AGE)
    }
}

/// A heap object whose outgoing handles can be **rewritten** — needed by a
/// moving (compacting) collector, which relocates objects and then fixes up
/// every reference to point at the new location.
pub trait Relocate {
    /// Replaces each outgoing handle `h` with `forward(h)`.
    fn relocate(&mut self, forward: &dyn Fn(Handle) -> Handle);
}

/// Statistics from a [`collect`] cycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    /// Objects reachable from the roots (kept).
    pub marked: usize,
    /// Objects swept (freed) this cycle.
    pub swept: usize,
    /// The summed [`Trace::weight`] of the kept objects (computed by
    /// [`sweep_full`]; the object count elsewhere).
    pub weight: usize,
}

/// The age at which an object is considered part of the **old** generation
/// (has survived at least one collection). Tunable; `1` means "survived once".
pub const OLD_AGE: u8 = 1;

/// The set of objects a mark phase reached: one entry per heap slot (the
/// marked handle's generation + 1, `0` for unmarked), so membership is an
/// index load instead of a tree search. A stale handle — an older generation
/// of a slot whose current object was marked — is correctly *not* a member.
#[derive(Clone, Debug, Default)]
pub struct MarkSet {
    marks: Vec<u32>,
    count: usize,
}

impl MarkSet {
    /// An empty set sized for a heap of `slots` slots (it grows on demand).
    #[must_use]
    pub fn with_slots(slots: usize) -> Self {
        Self {
            marks: alloc::vec![0; slots],
            count: 0,
        }
    }

    /// Adds `handle`; returns whether it was newly added.
    pub fn insert(&mut self, handle: Handle) -> bool {
        let i = handle.index() as usize;
        if i >= self.marks.len() {
            self.marks.resize(i + 1, 0);
        }
        let tag = u32::from(handle.generation()) + 1;
        if self.marks[i] == tag {
            return false;
        }
        if self.marks[i] == 0 {
            self.count += 1;
        }
        self.marks[i] = tag;
        true
    }

    /// Whether `handle` (this exact generation) is marked.
    #[must_use]
    pub fn contains(&self, handle: &Handle) -> bool {
        self.marks
            .get(handle.index() as usize)
            .is_some_and(|&t| t == u32::from(handle.generation()) + 1)
    }

    /// The number of marked objects.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether nothing is marked.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The marked handles, in slot-index order.
    pub fn iter(&self) -> impl Iterator<Item = Handle> + '_ {
        self.marks
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t != 0)
            .map(|(i, &t)| Handle::from_parts(i as u32, (t - 1) as u16))
    }
}

/// Marks everything reachable from `roots` (depth-first over outgoing handle
/// edges) and returns the marked set.
fn mark<T: Trace>(heap: &Heap<T>, roots: impl IntoIterator<Item = Handle>) -> MarkSet {
    let mut marked = MarkSet::with_slots(heap.slot_capacity());
    let mut work: Vec<Handle> = Vec::new();
    for root in roots {
        if heap.is_live(root) && marked.insert(root) {
            work.push(root);
        }
    }
    drain(
        heap,
        &mut marked,
        &mut work,
        &mut |_, _| {},
        &mut Vec::new(),
    );
    marked
}

/// Drains the grey `work` list into `marked`, following each object's outgoing
/// edges, plus whatever extra edges `scan(object, &mut side)` pushes into
/// `side` (empty between calls). Shared by [`mark`] and the ephemeron fixpoint.
fn drain<T: Trace>(
    heap: &Heap<T>,
    marked: &mut MarkSet,
    work: &mut Vec<Handle>,
    scan: &mut impl FnMut(&T, &mut Vec<Handle>),
    side: &mut Vec<Handle>,
) {
    while let Some(handle) = work.pop() {
        if let Some(obj) = heap.get(handle) {
            obj.trace_each(|edge| {
                // Only follow live, not-yet-marked targets (stale handles are
                // ignored, which also breaks cycles).
                if heap.is_live(edge) && marked.insert(edge) {
                    work.push(edge);
                }
            });
            scan(obj, side);
            for edge in side.drain(..) {
                if heap.is_live(edge) && marked.insert(edge) {
                    work.push(edge);
                }
            }
        }
    }
}

/// Marks everything reachable from `roots`, then applies **ephemeron** (weak-key)
/// expansion to a fixpoint and returns the marked set.
///
/// An ephemeron — a weak-key-map entry — keeps its *value* alive **only if its
/// *key* is independently reachable**; an entry must never keep its own key
/// alive. This is exactly the rule a realm's value-reachable side-tables need:
/// `key → value` survives iff `key` is live, and a surviving entry's `value`
/// (e.g. a function's lazily-built `.prototype`) is then itself a root.
///
/// Because a value kept alive by a live key may *itself* be the key of another
/// entry, one pass is not enough — `expand` is re-run until the marked set
/// reaches a fixpoint. `expand(heap, &marked, &mut extra)` must push, into
/// `extra`, the value handles of every entry whose key is currently in `marked`
/// (it is handed `heap` so it can inspect live objects); the collector marks
/// those (transitively) and repeats. Keys are *never* added by `expand`, so a
/// dead key stays dead and its entry can be pruned afterward.
pub fn mark_with_ephemerons<T: Trace>(
    heap: &Heap<T>,
    roots: impl IntoIterator<Item = Handle>,
    expand: impl FnMut(&Heap<T>, &MarkSet, &mut Vec<Handle>),
) -> MarkSet {
    mark_with_ephemerons_scan(heap, roots, |_, _| {}, expand)
}

/// [`mark_with_ephemerons`] with **extra strong edges** that the objects'
/// own [`Trace`] cannot report: `scan(object, &mut out)` is called once for
/// every marked object, right after its traced edges, and every handle it
/// pushes into `out` is marked as though the object referenced it (e.g. the
/// symbols a live object uses as property keys, which it stores by id). The
/// object is still in cache at that point, so this costs far less than a
/// separate walk over the marked set.
pub fn mark_with_ephemerons_scan<T: Trace>(
    heap: &Heap<T>,
    roots: impl IntoIterator<Item = Handle>,
    mut scan: impl FnMut(&T, &mut Vec<Handle>),
    mut expand: impl FnMut(&Heap<T>, &MarkSet, &mut Vec<Handle>),
) -> MarkSet {
    let mut marked = MarkSet::with_slots(heap.slot_capacity());
    let mut work: Vec<Handle> = Vec::new();
    let mut side: Vec<Handle> = Vec::new();
    for root in roots {
        if heap.is_live(root) && marked.insert(root) {
            work.push(root);
        }
    }
    drain(heap, &mut marked, &mut work, &mut scan, &mut side);

    // Ephemeron fixpoint: surface the values of entries whose keys are now live,
    // mark them transitively, and repeat until nothing new is marked.
    let mut extra: Vec<Handle> = Vec::new();
    loop {
        extra.clear();
        expand(heap, &marked, &mut extra);
        let mut grew = false;
        for h in extra.drain(..) {
            if heap.is_live(h) && marked.insert(h) {
                work.push(h);
                grew = true;
            }
        }
        if !grew {
            break;
        }
        drain(heap, &mut marked, &mut work, &mut scan, &mut side);
    }
    marked
}

/// Runs one stop-the-world **major** mark-and-sweep cycle: marks everything
/// reachable from `roots`, frees everything else, promotes survivors one
/// generation, and returns what it kept/swept.
pub fn collect<T: Trace>(heap: &mut Heap<T>, roots: &[Handle]) -> Stats {
    let marked = mark(heap, roots.iter().copied());

    // --- sweep: free every live object the mark phase did not reach (a
    // survivor ages toward the old generation; the remembered set is rebuilt) ---
    let swept = heap.sweep(|h, _| marked.contains(&h), T::interior_mutable);

    Stats {
        marked: marked.len(),
        swept,
        weight: marked.len(),
    }
}

/// The **full sweep** half of a major collection, applied to an already-computed
/// `marked` set (see [`mark_with_ephemerons`]): frees every live object the mark
/// phase did not reach, promotes the survivors one generation, clears the
/// remembered set, and returns the [`Stats`].
///
/// Exposed separately from [`collect`] so a caller can interpose between marking
/// and sweeping — e.g. to prune its own weak-key side-tables against the marked
/// set — without forcing two overlapping mutable borrows. Pruning a dead key
/// before this runs does not change the sweep set (the key is already unmarked);
/// it only stops the table from holding a now-dangling handle.
pub fn sweep_full<T: Trace>(heap: &mut Heap<T>, marked: &MarkSet) -> Stats {
    let mut weight = 0usize;
    let swept = heap.sweep(
        |h, value| {
            let keep = marked.contains(&h);
            if keep {
                weight = weight.saturating_add(value.weight());
            }
            keep
        },
        T::interior_mutable,
    );
    Stats {
        marked: marked.len(),
        swept,
        weight,
    }
}

/// Runs a **minor** (generational) collection: it reclaims only short-lived
/// objects in the **young** generation (allocated since the last collection).
/// Because most objects die young, marking and sweeping just the nursery is
/// cheap. See [`collect_minor_with`].
pub fn collect_minor<T: Trace>(heap: &mut Heap<T>, roots: &[Handle]) -> Stats {
    collect_minor_with(heap, roots, |_, _| {}, |_, _, _| {})
}

/// A minor collection with the hooks of [`mark_with_ephemerons_scan`]
/// (`scan` for extra strong edges, `expand` for weak-key tables, the latter
/// given a [`MinorLive`] view) — the generational counterpart of a full
/// collection's mark and sweep.
///
/// Soundness rests on one invariant, kept by the write barrier: **an old
/// object outside the remembered set references no young object.** The old
/// generation is not traced at all — every old object is presumed alive (a
/// later full collection decides) — so the young survivors are exactly those
/// reachable from the roots and from the remembered old objects through
/// young objects only. Marking a young object promotes it (its age is the
/// mark), and [`Heap::sweep_young`] frees the rest; afterwards no young
/// object remains, so the remembered set restarts empty but for the
/// interior-mutable old objects ([`Trace::interior_mutable`]), which a minor
/// collection always rescans.
pub fn collect_minor_with<T: Trace>(
    heap: &mut Heap<T>,
    roots: &[Handle],
    scan: impl FnMut(&T, &mut Vec<Handle>),
    expand: impl FnMut(&Heap<T>, &MinorLive<'_, T>, &mut Vec<Handle>),
) -> Stats {
    let mark = mark_minor(heap, roots, scan, expand);
    sweep_minor(heap, mark)
}

/// The outcome of [`mark_minor`], for [`sweep_minor`]: between the two, the
/// young survivors are exactly the young objects a [`MinorLive`] reports
/// alive, so a caller can prune its weak-key tables against it.
#[derive(Debug)]
pub struct MinorMark {
    rescanned: Vec<u32>,
    marked: usize,
}

/// The mark half of [`collect_minor_with`].
pub fn mark_minor<T: Trace>(
    heap: &mut Heap<T>,
    roots: &[Handle],
    mut scan: impl FnMut(&T, &mut Vec<Handle>),
    mut expand: impl FnMut(&Heap<T>, &MinorLive<'_, T>, &mut Vec<Handle>),
) -> MinorMark {
    let mut work: Vec<Handle> = Vec::new();
    let mut edges: Vec<Handle> = Vec::new();
    let mut marked = 0usize;
    // An old root needs no tracing: whatever young object it reaches, it
    // reaches through a remembered object or a young one.
    for &root in roots {
        if heap.promote_young(root) {
            marked += 1;
            work.push(root);
        }
    }
    // The remembered old objects are scanned like roots' referents.
    let rescanned = heap.take_remembered();
    for &index in &rescanned {
        if let Some(obj) = heap.live_handle_at(index).and_then(|h| heap.peek(h)) {
            obj.trace_each(|e| edges.push(e));
            scan(obj, &mut edges);
        }
        for e in edges.drain(..) {
            if heap.promote_young(e) {
                marked += 1;
                work.push(e);
            }
        }
    }
    let mut extra: Vec<Handle> = Vec::new();
    loop {
        while let Some(h) = work.pop() {
            if let Some(obj) = heap.get(h) {
                obj.trace_each(|e| edges.push(e));
                scan(obj, &mut edges);
            }
            for e in edges.drain(..) {
                if heap.promote_young(e) {
                    marked += 1;
                    work.push(e);
                }
            }
        }
        // Ephemeron fixpoint, as in a full collection.
        expand(heap, &MinorLive(heap), &mut extra);
        for h in extra.drain(..) {
            if heap.promote_young(h) {
                marked += 1;
                work.push(h);
            }
        }
        if work.is_empty() {
            break;
        }
    }
    MinorMark { rescanned, marked }
}

/// The sweep half of [`collect_minor_with`]: frees the young objects
/// [`mark_minor`] did not reach.
pub fn sweep_minor<T: Trace>(heap: &mut Heap<T>, mark: MinorMark) -> Stats {
    let swept = heap.sweep_young(mark.rescanned, T::interior_mutable);
    Stats {
        marked: mark.marked,
        swept,
        weight: mark.marked,
    }
}

/// **GC verification** for [`collect_minor_with`]: the first old object
/// outside the remembered set that references a young one (`scan` adds the
/// extra edges a collection's `scan` would), as `(container, target)`, or
/// `(h, h)` for an untracked young object or an interior-mutable old one
/// missing from the remembered set. `None` when the invariants hold.
pub fn remembered_violation<T: Trace>(
    heap: &Heap<T>,
    mut scan: impl FnMut(&T, &mut Vec<Handle>),
) -> Option<(Handle, Handle)> {
    heap.remembered_violation(
        |obj, out| {
            obj.trace_each(|e| out.push(e));
            scan(obj, out);
        },
        T::interior_mutable,
    )
}

/// Runs a **moving (compacting)** collection: marks from `roots`, relocates the
/// live objects to the front of the heap's slot table (eliminating the gaps that
/// sweeping leaves), and rewrites every reference — inside surviving objects and
/// in `roots` (updated in place) — to the new locations.
///
/// Compaction restores allocation locality and lets the slot table shrink, at
/// the cost of rewriting pointers; it pairs with the tracing collector above.
pub fn compact<T: Trace + Relocate>(heap: &mut Heap<T>, roots: &mut [Handle]) -> Stats {
    compact_with(heap, roots, &mut |_| {})
}

/// Like [`compact`], but also calls `fixup` with the old→new forwarding function after the
/// heap and roots have been relocated, so the caller can repair handles it holds *outside*
/// the heap graph — e.g. a realm's handle-keyed side-tables — keeping them sound across a
/// moving collection.
#[allow(clippy::type_complexity)] // the `fixup` callback receives the forwarding function
pub fn compact_with<T: Trace + Relocate>(
    heap: &mut Heap<T>,
    roots: &mut [Handle],
    fixup: &mut dyn FnMut(&dyn Fn(Handle) -> Handle),
) -> Stats {
    let before = heap.len();
    let marked = mark(heap, roots.iter().copied());

    // Relocate live objects densely; `map` forwards old handle → new handle.
    let map: alloc::collections::BTreeMap<Handle, Handle> =
        heap.compact_to(&marked).into_iter().collect();
    let forward = |h: Handle| map.get(&h).copied().unwrap_or(h);

    // Fix up every reference inside surviving objects, then the roots, then the caller's
    // external (out-of-heap) tables.
    relocate(heap, &forward);
    for r in roots.iter_mut() {
        *r = forward(*r);
    }
    fixup(&forward);

    Stats {
        marked: marked.len(),
        swept: before - marked.len(),
        weight: marked.len(),
    }
}

/// Rewrites every outgoing reference of every live heap object via `forward` —
/// the **pointer-relocation pass**. It is the moving collector's fix-up step,
/// and equally the load step of a **heap snapshot** (`ROADMAP.md` §2.2 "heap
/// snapshots"): when a serialized object graph is reloaded, the freshly-allocated
/// objects carry the *old* (serialized) handles, and one relocation pass with a
/// `old → new` forwarding map repairs the whole graph in place — no per-edge
/// special-casing, exactly as the moving GC relocates after compaction.
pub fn relocate<T: Relocate>(heap: &mut Heap<T>, forward: &dyn Fn(Handle) -> Handle) {
    for handle in heap.live_handles() {
        if let Some(obj) = heap.get_mut(handle) {
            obj.relocate(forward);
        }
    }
}

/// An **incremental** (step-bounded) tri-color marker. Where [`collect`] marks
/// the whole heap in one stop-the-world pass, this splits marking across many
/// bounded [`step`](IncrementalMarker::step)s that can be interleaved with the
/// mutator — bounding pause time.
///
/// Tri-color invariant: an object is *white* (unmarked — a collection candidate),
/// *grey* (marked but its children not yet scanned — held in the worklist), or
/// *black* (marked and scanned). Marking is done when no grey objects remain.
///
/// Soundness under concurrent mutation needs a **write barrier**: if the mutator
/// stores a pointer to a white object into an already-black object, that white
/// object could be missed — so the barrier ([`mark_grey`](IncrementalMarker::mark_grey),
/// a Dijkstra-style shade-on-write) greys the stored target, keeping it live.
pub struct IncrementalMarker {
    /// Black ∪ grey — everything marked so far.
    marked: BTreeSet<Handle>,
    /// The grey worklist (marked, awaiting child scan).
    grey: Vec<Handle>,
}

impl IncrementalMarker {
    /// Begins marking from `roots` (greying each).
    #[must_use]
    pub fn new(roots: &[Handle]) -> Self {
        let mut m = Self {
            marked: BTreeSet::new(),
            grey: Vec::new(),
        };
        for &r in roots {
            m.mark_grey(r);
        }
        m
    }

    /// Shades `handle` grey if still white — the write-barrier hook the mutator
    /// calls when it stores a reference during marking, and how children are
    /// enqueued during a scan.
    pub fn mark_grey(&mut self, handle: Handle) {
        if self.marked.insert(handle) {
            self.grey.push(handle);
        }
    }

    /// Scans up to `budget` grey objects, greying their (live) children.
    /// Returns `true` once marking is complete (the grey set is empty).
    pub fn step<T: Trace>(&mut self, heap: &Heap<T>, budget: usize) -> bool {
        for _ in 0..budget {
            let Some(handle) = self.grey.pop() else {
                return true;
            };
            let mut edges: Vec<Handle> = Vec::new();
            if let Some(obj) = heap.get(handle) {
                obj.trace(&mut |h| edges.push(h));
            }
            for edge in edges {
                if heap.is_live(edge) {
                    self.mark_grey(edge);
                }
            }
        }
        self.grey.is_empty()
    }

    /// Whether marking has finished (no grey objects remain to scan).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.grey.is_empty()
    }

    /// Frees every live object that marking did not reach. Call only once
    /// [`is_complete`](IncrementalMarker::is_complete) holds. Returns the count
    /// swept.
    pub fn sweep<T>(&self, heap: &mut Heap<T>) -> usize {
        let mut swept = 0;
        for handle in heap.live_handles() {
            if !self.marked.contains(&handle) {
                heap.free(handle);
                swept += 1;
            }
        }
        swept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal traceable node: a tag plus a list of outgoing handles.
    struct Node {
        tag: u32,
        edges: Vec<Handle>,
        /// Edges changed through a shared borrow (declared by `interior`).
        shared: core::cell::RefCell<Vec<Handle>>,
        interior: bool,
    }

    impl Node {
        fn new(tag: u32) -> Self {
            Self {
                tag,
                edges: Vec::new(),
                shared: core::cell::RefCell::new(Vec::new()),
                interior: false,
            }
        }
    }

    impl Trace for Node {
        fn trace(&self, visit: &mut dyn FnMut(Handle)) {
            for &e in self.edges.iter().chain(self.shared.borrow().iter()) {
                visit(e);
            }
        }

        fn interior_mutable(&self) -> bool {
            self.interior
        }
    }

    impl Relocate for Node {
        fn relocate(&mut self, forward: &dyn Fn(Handle) -> Handle) {
            for e in &mut self.edges {
                *e = forward(*e);
            }
        }
    }

    #[test]
    fn unreachable_objects_are_swept() {
        let mut heap: Heap<Node> = Heap::new();
        let keep = heap.alloc(Node::new(1));
        let drop = heap.alloc(Node::new(2));
        assert_eq!(heap.len(), 2);

        let stats = collect(&mut heap, &[keep]);
        assert_eq!(stats.marked, 1);
        assert_eq!(stats.swept, 1);
        assert!(heap.is_live(keep));
        assert!(!heap.is_live(drop));
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.get(keep).unwrap().tag, 1);
    }

    #[test]
    fn reachable_chain_is_kept() {
        // root -> a -> b -> c, plus an unreferenced d.
        let mut heap: Heap<Node> = Heap::new();
        let c = heap.alloc(Node::new(3));
        let mut b_node = Node::new(2);
        b_node.edges.push(c);
        let b = heap.alloc(b_node);
        let mut a_node = Node::new(1);
        a_node.edges.push(b);
        let a = heap.alloc(a_node);
        let _d = heap.alloc(Node::new(4));

        let stats = collect(&mut heap, &[a]);
        assert_eq!(stats.marked, 3);
        assert_eq!(stats.swept, 1);
        assert!(heap.is_live(a) && heap.is_live(b) && heap.is_live(c));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn cycles_among_garbage_are_collected() {
        // Two nodes referencing each other, neither reachable from a root —
        // reference counting would leak them; tracing reclaims both.
        let mut heap: Heap<Node> = Heap::new();
        let x = heap.alloc(Node::new(1));
        let y = heap.alloc(Node::new(2));
        heap.get_mut(x).unwrap().edges.push(y);
        heap.get_mut(y).unwrap().edges.push(x);
        let survivor = heap.alloc(Node::new(3));

        let stats = collect(&mut heap, &[survivor]);
        assert_eq!(stats.swept, 2);
        assert_eq!(stats.marked, 1);
        assert!(!heap.is_live(x) && !heap.is_live(y));
        assert!(heap.is_live(survivor));
    }

    #[test]
    fn major_collection_promotes_survivors() {
        // A survivor of a major collection ages into the old generation.
        let mut heap: Heap<Node> = Heap::new();
        let keep = heap.alloc(Node::new(1));
        assert_eq!(heap.age(keep), Some(0)); // young
        collect(&mut heap, &[keep]);
        assert_eq!(heap.age(keep), Some(OLD_AGE)); // promoted
    }

    #[test]
    fn minor_collection_sweeps_only_the_young() {
        let mut heap: Heap<Node> = Heap::new();
        // `old` survives a major collection → promoted to the old generation.
        let old = heap.alloc(Node::new(1));
        collect(&mut heap, &[old]);
        assert_eq!(heap.age(old), Some(OLD_AGE));

        // Now allocate young objects: one kept by a root, one garbage.
        let young_keep = heap.alloc(Node::new(2));
        let young_garbage = heap.alloc(Node::new(3));

        // A minor collection sweeps only the young garbage; `old` is untouched
        // (not even considered for sweeping) and `young_keep` is promoted.
        let stats = collect_minor(&mut heap, &[old, young_keep]);
        assert_eq!(stats.swept, 1);
        assert!(heap.is_live(old) && heap.is_live(young_keep));
        assert!(!heap.is_live(young_garbage));
        assert_eq!(heap.age(young_keep), Some(OLD_AGE)); // promoted
    }

    #[test]
    fn minor_collection_keeps_young_referenced_by_old() {
        // A young object reachable ONLY through an old object must survive a
        // minor collection (the old generation acts as roots).
        let mut heap: Heap<Node> = Heap::new();
        let old = heap.alloc(Node::new(1));
        collect(&mut heap, &[old]); // promote `old`

        let young = heap.alloc(Node::new(2));
        heap.get_mut(old).unwrap().edges.push(young); // old -> young edge
        heap.record_edge(old, young, OLD_AGE); // the write barrier remembers `old`

        // `young` is not a direct root, but `old` is in the remembered set and
        // points at it, so it survives the minor collection.
        let stats = collect_minor(&mut heap, &[old]);
        assert_eq!(stats.swept, 0);
        assert!(heap.is_live(young));
    }

    #[test]
    fn a_mutable_borrow_of_an_old_object_is_the_write_barrier() {
        // No explicit `record_edge`: `get_mut` itself remembers `old`.
        let mut heap: Heap<Node> = Heap::new();
        let old = heap.alloc(Node::new(1));
        collect(&mut heap, &[old]);
        let young = heap.alloc(Node::new(2));
        heap.get_mut(old).unwrap().edges.push(young);
        assert_eq!(remembered_violation(&heap, |_, _| {}), None);
        let stats = collect_minor(&mut heap, &[]);
        assert_eq!(stats.swept, 0);
        assert!(heap.is_live(young));
        assert_eq!(heap.age(young), Some(OLD_AGE));
        // Settled: nothing is young, and `old` left the remembered set.
        assert!(heap.remembered_roots().is_empty());
        let later = heap.alloc(Node::new(3));
        assert_eq!(collect_minor(&mut heap, &[]).swept, 1);
        assert!(!heap.is_live(later));
    }

    #[test]
    fn an_interior_mutable_old_object_is_rescanned_by_every_minor_collection() {
        let mut heap: Heap<Node> = Heap::new();
        let mut node = Node::new(1);
        node.interior = true;
        let old = heap.alloc(node);
        collect(&mut heap, &[old]);
        assert_eq!(heap.remembered_roots(), [old]);
        for round in 0..3 {
            // A store through a shared borrow: no barrier can see it.
            let young = heap.alloc(Node::new(10 + round));
            heap.get(old).unwrap().shared.borrow_mut().push(young);
            let garbage = heap.alloc(Node::new(20 + round));
            assert_eq!(remembered_violation(&heap, |_, _| {}), None);
            collect_minor(&mut heap, &[]);
            assert!(heap.is_live(young) && !heap.is_live(garbage));
            assert_eq!(heap.remembered_roots(), [old]);
        }
    }

    #[test]
    fn the_verifier_reports_an_unremembered_old_to_young_edge() {
        // A shared-borrow store into an object that does not declare its
        // interior mutability is exactly the hole the verifier exists for.
        let mut heap: Heap<Node> = Heap::new();
        let old = heap.alloc(Node::new(1));
        collect(&mut heap, &[old]);
        let young = heap.alloc(Node::new(2));
        heap.get(old).unwrap().shared.borrow_mut().push(young);
        assert_eq!(remembered_violation(&heap, |_, _| {}), Some((old, young)));
        // … and an interior-mutable old object outside the remembered set.
        let mut heap: Heap<Node> = Heap::new();
        let h = heap.alloc(Node::new(1));
        collect(&mut heap, &[h]);
        heap.get_mut(h).unwrap().interior = true;
        collect_minor(&mut heap, &[h]);
        assert_eq!(remembered_violation(&heap, |_, _| {}), None);
    }

    #[test]
    fn young_objects_in_reused_slots_are_swept_by_a_minor_collection() {
        let mut heap: Heap<Node> = Heap::new();
        let root = heap.alloc(Node::new(0));
        let dead: Vec<Handle> = (1..=4).map(|t| heap.alloc(Node::new(t))).collect();
        collect(&mut heap, &[root]);
        assert!(dead.iter().all(|&h| !heap.is_live(h)));
        // These reuse the freed slots (below the young floor).
        let reused: Vec<Handle> = (5..=8).map(|t| heap.alloc(Node::new(t))).collect();
        assert_eq!(heap.slot_capacity(), 5);
        heap.get_mut(root).unwrap().edges.push(reused[1]);
        // An explicit free between collections keeps the bookkeeping straight.
        heap.free(reused[3]);
        let fresh = heap.alloc(Node::new(9));
        let stats = collect_minor(&mut heap, &[root]);
        assert_eq!(stats.swept, 3);
        assert!(heap.is_live(reused[1]));
        assert!(!heap.is_live(reused[0]) && !heap.is_live(reused[2]) && !heap.is_live(fresh));
        assert_eq!(heap.len(), 2);
        assert_eq!(remembered_violation(&heap, |_, _| {}), None);
    }

    #[test]
    fn minor_collection_frees_young_when_no_barrier_recorded() {
        // Without the barrier, an old object's stale view doesn't keep young
        // garbage alive — the remembered set is the sole old-roots source.
        let mut heap: Heap<Node> = Heap::new();
        let old = heap.alloc(Node::new(1));
        collect(&mut heap, &[old]);
        let young = heap.alloc(Node::new(2)); // unreferenced, no barrier
        let stats = collect_minor(&mut heap, &[old]);
        assert_eq!(stats.swept, 1);
        assert!(!heap.is_live(young));
    }

    #[test]
    fn compaction_relocates_survivors_and_fixes_references() {
        // a -> b, and a gap (c) between them; d is unreachable garbage.
        let mut heap: Heap<Node> = Heap::new();
        let b = heap.alloc(Node::new(2));
        let _c = heap.alloc(Node::new(3)); // becomes garbage (a gap)
        let mut a_node = Node::new(1);
        a_node.edges.push(b);
        let a = heap.alloc(a_node);
        let _d = heap.alloc(Node::new(4)); // garbage

        let mut roots = [a];
        let stats = compact(&mut heap, &mut roots);
        assert_eq!(stats.marked, 2);
        assert_eq!(stats.swept, 2);

        // The roots were rewritten to the new locations; the graph is intact.
        let a2 = roots[0];
        assert_eq!(heap.get(a2).unwrap().tag, 1);
        let b2 = heap.get(a2).unwrap().edges[0];
        assert_eq!(heap.get(b2).unwrap().tag, 2);
        // The slot table is dense (only the 2 survivors remain).
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.live_handles().len(), 2);
    }

    #[test]
    fn compaction_preserves_a_reachable_cycle() {
        let mut heap: Heap<Node> = Heap::new();
        let x = heap.alloc(Node::new(1));
        let y = heap.alloc(Node::new(2));
        heap.get_mut(x).unwrap().edges.push(y);
        heap.get_mut(y).unwrap().edges.push(x);
        let _garbage = heap.alloc(Node::new(9));

        let mut roots = [x];
        let stats = compact(&mut heap, &mut roots);
        assert_eq!(stats.marked, 2);
        assert_eq!(stats.swept, 1);
        // The cycle survives with references fixed up.
        let x2 = roots[0];
        let y2 = heap.get(x2).unwrap().edges[0];
        let back = heap.get(y2).unwrap().edges[0];
        assert_eq!(back, x2);
    }

    #[test]
    fn incremental_marking_matches_a_full_collection() {
        // root -> a -> b -> c (a chain), plus unreachable garbage d, e.
        let mut heap: Heap<Node> = Heap::new();
        let c = heap.alloc(Node::new(3));
        let mut b = Node::new(2);
        b.edges.push(c);
        let b = heap.alloc(b);
        let mut a = Node::new(1);
        a.edges.push(b);
        let a = heap.alloc(a);
        let _d = heap.alloc(Node::new(4));
        let _e = heap.alloc(Node::new(5));

        // Mark one object per step (the finest granularity).
        let mut marker = IncrementalMarker::new(&[a]);
        let mut steps = 0;
        while !marker.step(&heap, 1) {
            steps += 1;
            assert!(steps < 100, "marking should terminate");
        }
        let swept = marker.sweep(&mut heap);
        assert_eq!(swept, 2); // d, e
        assert!(heap.is_live(a) && heap.is_live(b) && heap.is_live(c));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn incremental_write_barrier_keeps_a_late_stored_reference() {
        // Simulate the hazard the barrier guards: during marking, the mutator
        // stores a reference to a *white* object into an *already-scanned* one.
        let mut heap: Heap<Node> = Heap::new();
        let root = heap.alloc(Node::new(1)); // becomes black early
        let late = heap.alloc(Node::new(2)); // allocated white, not yet referenced

        let mut marker = IncrementalMarker::new(&[root]);
        // Drive marking to completion of the root (root becomes black, grey
        // empties — `late` is still white and would be swept).
        while !marker.step(&heap, 1) {}
        assert!(marker.is_complete());

        // The mutator now links root -> late and fires the write barrier.
        heap.get_mut(root).unwrap().edges.push(late);
        marker.mark_grey(late); // Dijkstra shade-on-write
        // Re-run to drain the re-greyed work.
        while !marker.step(&heap, 4) {}

        let swept = marker.sweep(&mut heap);
        assert_eq!(swept, 0, "the barrier-shaded object must survive");
        assert!(heap.is_live(late));
    }

    #[test]
    fn reachable_cycle_survives() {
        // A cycle that *is* reachable from a root must be kept (and not loop
        // forever during marking).
        let mut heap: Heap<Node> = Heap::new();
        let x = heap.alloc(Node::new(1));
        let y = heap.alloc(Node::new(2));
        heap.get_mut(x).unwrap().edges.push(y);
        heap.get_mut(y).unwrap().edges.push(x);

        let stats = collect(&mut heap, &[x]);
        assert_eq!(stats.marked, 2);
        assert_eq!(stats.swept, 0);
        assert!(heap.is_live(x) && heap.is_live(y));
    }

    #[test]
    fn snapshot_reload_relocates_pointers() {
        use alloc::collections::BTreeMap;
        // Build a graph in heap A: a 2-cycle n0 <-> n1, plus n1 -> n2.
        let mut a: Heap<Node> = Heap::new();
        let n0 = a.alloc(Node::new(10));
        let n1 = a.alloc(Node::new(11));
        let n2 = a.alloc(Node::new(12));
        a.get_mut(n0).unwrap().edges.push(n1);
        a.get_mut(n1).unwrap().edges.push(n0);
        a.get_mut(n1).unwrap().edges.push(n2);

        // "Snapshot": capture each live object's payload + its edges as the
        // *original* handles (what a serialized graph would record).
        let live = a.live_handles();
        let snap: Vec<(Handle, u32, Vec<Handle>)> = live
            .iter()
            .map(|h| {
                let mut edges = Vec::new();
                a.get(*h).unwrap().trace(&mut |e| edges.push(e));
                (*h, a.get(*h).unwrap().tag, edges)
            })
            .collect();

        // "Reload" into a fresh heap B. Allocate a decoy first so B's handles do
        // NOT coincide with A's — relocation must do real work. Each reloaded node
        // keeps the OLD handles in its edges, to be repaired by the relocation pass.
        let mut b: Heap<Node> = Heap::new();
        let _decoy = b.alloc(Node::new(99));
        let mut map: BTreeMap<Handle, Handle> = BTreeMap::new();
        let mut new_handles = Vec::new();
        for (old, tag, edges) in &snap {
            let mut node = Node::new(*tag);
            node.edges = edges.clone(); // still the serialized (old) handles
            let nh = b.alloc(node);
            map.insert(*old, nh);
            new_handles.push(nh);
        }

        // The pointer-relocation pass: forward every old handle to its new one.
        let forward = |h: Handle| map.get(&h).copied().unwrap_or(h);
        relocate(&mut b, &forward);

        // The graph in B is intact under the new handles: n0<->n1, n1->n2.
        let (b0, b1, b2) = (new_handles[0], new_handles[1], new_handles[2]);
        assert_eq!(b.get(b0).unwrap().tag, 10);
        assert_eq!(b.get(b0).unwrap().edges, alloc::vec![b1]);
        assert_eq!(b.get(b1).unwrap().edges, alloc::vec![b0, b2]);
        assert_eq!(b.get(b2).unwrap().tag, 12);
        // And the relocated handles genuinely differ from the originals.
        assert_ne!(b0, n0);
    }

    #[test]
    fn ephemeron_drops_dead_key_entries_but_keeps_live_ones() {
        // Two weak-key entries: key0 -> val0, key1 -> val1. key0 is a real root,
        // key1 is not. After ephemeron marking, val0 is kept (live key) and
        // key1/val1 are not marked (dead key — must be prunable).
        let mut heap: Heap<Node> = Heap::new();
        let key0 = heap.alloc(Node::new(0));
        let val0 = heap.alloc(Node::new(10));
        let key1 = heap.alloc(Node::new(1));
        let val1 = heap.alloc(Node::new(11));
        let table = [(key0, val0), (key1, val1)];

        let marked = mark_with_ephemerons(&heap, [key0], |_heap, marked, extra| {
            for (k, v) in &table {
                if marked.contains(k) {
                    extra.push(*v);
                }
            }
        });
        assert!(marked.contains(&key0) && marked.contains(&val0));
        assert!(
            !marked.contains(&key1) && !marked.contains(&val1),
            "a dead key and its value must stay unmarked",
        );
    }

    #[test]
    fn ephemeron_reaches_a_fixpoint_through_chained_entries() {
        // entry A: keyA -> valA, where valA points at keyB.
        // entry B: keyB -> valB. Only keyA is a root, so valA keeps keyB alive,
        // which must in turn keep valB alive (the fixpoint).
        let mut heap: Heap<Node> = Heap::new();
        let val_b = heap.alloc(Node::new(22));
        let key_b = heap.alloc(Node::new(2));
        let mut va = Node::new(21);
        va.edges.push(key_b); // valA -> keyB
        let val_a = heap.alloc(va);
        let key_a = heap.alloc(Node::new(1));
        let table = [(key_a, val_a), (key_b, val_b)];

        let marked = mark_with_ephemerons(&heap, [key_a], |_heap, marked, extra| {
            for (k, v) in &table {
                if marked.contains(k) {
                    extra.push(*v);
                }
            }
        });
        assert!(marked.contains(&key_a) && marked.contains(&val_a));
        assert!(
            marked.contains(&key_b) && marked.contains(&val_b),
            "A's value keeps B's key live, so B's value survives (fixpoint)",
        );
    }

    #[test]
    fn empty_roots_sweeps_everything() {
        let mut heap: Heap<Node> = Heap::new();
        heap.alloc(Node::new(1));
        heap.alloc(Node::new(2));
        let stats = collect(&mut heap, &[]);
        assert_eq!(stats.swept, 2);
        assert_eq!(stats.marked, 0);
        assert!(heap.is_empty());
    }
}
