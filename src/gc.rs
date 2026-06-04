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
}

/// Statistics from a [`collect`] cycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stats {
    /// Objects reachable from the roots (kept).
    pub marked: usize,
    /// Objects swept (freed) this cycle.
    pub swept: usize,
}

/// The age at which an object is considered part of the **old** generation
/// (has survived at least one collection). Tunable; `1` means "survived once".
pub const OLD_AGE: u8 = 1;

/// Marks everything reachable from `roots` (depth-first over outgoing handle
/// edges) and returns the marked set.
fn mark<T: Trace>(heap: &Heap<T>, roots: impl IntoIterator<Item = Handle>) -> BTreeSet<Handle> {
    let mut marked: BTreeSet<Handle> = BTreeSet::new();
    let mut work: Vec<Handle> = Vec::new();
    for root in roots {
        if heap.is_live(root) && marked.insert(root) {
            work.push(root);
        }
    }
    while let Some(handle) = work.pop() {
        let mut edges: Vec<Handle> = Vec::new();
        if let Some(obj) = heap.get(handle) {
            obj.trace(&mut |h| edges.push(h));
        }
        for edge in edges {
            // Only follow live, not-yet-marked targets (stale handles are
            // ignored, which also breaks cycles).
            if heap.is_live(edge) && marked.insert(edge) {
                work.push(edge);
            }
        }
    }
    marked
}

/// Runs one stop-the-world **major** mark-and-sweep cycle: marks everything
/// reachable from `roots`, frees everything else, promotes survivors one
/// generation, and returns what it kept/swept.
pub fn collect<T: Trace>(heap: &mut Heap<T>, roots: &[Handle]) -> Stats {
    let marked = mark(heap, roots.iter().copied());

    // --- sweep: free every live object the mark phase did not reach ---
    let mut swept = 0;
    for handle in heap.live_handles() {
        if marked.contains(&handle) {
            heap.tenure(handle); // a survivor ages toward the old generation
        } else {
            heap.free(handle);
            swept += 1;
        }
    }
    // A full collection re-establishes the generation boundary from scratch.
    heap.clear_remembered();

    Stats {
        marked: marked.len(),
        swept,
    }
}

/// Runs a **minor** (generational) collection: it reclaims only short-lived
/// objects in the **young** generation. Because most objects die young, sweeping
/// just the nursery is cheap.
///
/// Correctness without a write barrier: the entire **old** generation is treated
/// as part of the root set, so a young object kept alive solely by an old
/// referent survives. (A later refinement adds a remembered set so only mutated
/// old objects need scanning.) Surviving young objects are promoted.
pub fn collect_minor<T: Trace>(heap: &mut Heap<T>, roots: &[Handle]) -> Stats {
    // Roots = the program roots ∪ the **remembered set** (old objects written
    // with a young pointer), rather than the entire old generation.
    let remembered = heap.remembered_roots();
    let marked = mark(heap, roots.iter().copied().chain(remembered));

    // Sweep only the young generation; promote the young survivors.
    let mut swept = 0;
    for handle in heap.handles_where(|a| a < OLD_AGE) {
        if marked.contains(&handle) {
            heap.tenure(handle);
        } else {
            heap.free(handle);
            swept += 1;
        }
    }
    // The surviving young are now old; the recorded old→young edges are old→old.
    heap.clear_remembered();

    Stats {
        marked: marked.len(),
        swept,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal traceable node: a tag plus a list of outgoing handles.
    struct Node {
        tag: u32,
        edges: Vec<Handle>,
    }

    impl Node {
        fn new(tag: u32) -> Self {
            Self {
                tag,
                edges: Vec::new(),
            }
        }
    }

    impl Trace for Node {
        fn trace(&self, visit: &mut dyn FnMut(Handle)) {
            for &e in &self.edges {
                visit(e);
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
