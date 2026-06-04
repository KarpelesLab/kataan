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

/// Runs one stop-the-world mark-and-sweep cycle: marks everything reachable from
/// `roots`, frees everything else, and returns what it kept/swept.
pub fn collect<T: Trace>(heap: &mut Heap<T>, roots: &[Handle]) -> Stats {
    // --- mark: depth-first from the roots over outgoing handle edges ---
    let mut marked: BTreeSet<Handle> = BTreeSet::new();
    let mut work: Vec<Handle> = Vec::new();
    for &root in roots {
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

    // --- sweep: free every live object the mark phase did not reach ---
    let mut swept = 0;
    for handle in heap.live_handles() {
        if !marked.contains(&handle) {
            heap.free(handle);
            swept += 1;
        }
    }

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
