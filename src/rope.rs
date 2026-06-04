//! Rope strings — lazy-concatenation string values (`ROADMAP.md` §3, item 4).
//!
//! Building a string by repeated concatenation (`s += chunk` in a loop, a
//! `join`, a template assembled piecewise) is quadratic with a flat buffer:
//! each `+` copies the whole left side. A **rope** stores a concatenation as a
//! small tree of shared segments, so joining two ropes is O(1) — it allocates
//! one interior node pointing at both sides — and the characters are only
//! copied once, when the final string is materialized.
//!
//! `Rope` is a cheap-to-clone handle (`Rc` inside). Concatenation is O(1) and
//! length is O(1) (cached per node); materializing flattens to a `String`
//! in O(n) with an explicit work stack, so even a deeply left-nested rope — the
//! exact shape `s += x` in a loop produces — flattens without recursing.
//!
//! Pure, safe `alloc`-only Rust. (Small-string inlining is a later refinement;
//! the tree already fixes the asymptotics that bite real code.)

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

/// A string value represented as a tree of concatenated segments. Cloning is a
/// reference-count bump; concatenation and length are O(1).
#[derive(Clone)]
pub struct Rope(Rc<Node>);

enum Node {
    /// A contiguous run of text.
    Leaf(Box<str>),
    /// The concatenation of two ropes, with the total byte length cached.
    Concat { left: Rope, right: Rope, len: usize },
}

impl Rope {
    /// An empty rope.
    #[must_use]
    pub fn new() -> Self {
        Self::leaf("")
    }

    /// A leaf rope holding `s`.
    #[must_use]
    pub fn leaf(s: &str) -> Self {
        Rope(Rc::new(Node::Leaf(Box::from(s))))
    }

    /// The total length in bytes (O(1) — cached).
    #[must_use]
    pub fn len(&self) -> usize {
        match &*self.0 {
            Node::Leaf(s) => s.len(),
            Node::Concat { len, .. } => *len,
        }
    }

    /// Whether the rope is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Concatenates `self` and `other` in O(1), sharing both sides.
    #[must_use]
    pub fn concat(&self, other: &Rope) -> Rope {
        // Concatenating with empty just shares the non-empty side.
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let len = self.len() + other.len();
        Rope(Rc::new(Node::Concat {
            left: self.clone(),
            right: other.clone(),
            len,
        }))
    }

    /// Appends `s` to the rope (O(1)), returning the new rope.
    #[must_use]
    pub fn push_str(&self, s: &str) -> Rope {
        self.concat(&Rope::leaf(s))
    }

    /// Materializes the rope into a flat `String` in O(n), iteratively (so a
    /// deeply nested rope cannot overflow the stack). (`Display`/`ToString` are
    /// also implemented; this form pre-allocates the exact capacity.)
    #[must_use]
    pub fn materialize(&self) -> String {
        let mut out = String::with_capacity(self.len());
        // Depth-first, left-to-right, using an explicit stack of pending nodes.
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Leaf(s) => out.push_str(s),
                Node::Concat { left, right, .. } => {
                    // Push right first so left is processed first (LIFO).
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        out
    }
}

impl Default for Rope {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for Rope {
    fn from(s: &str) -> Self {
        Rope::leaf(s)
    }
}

impl core::fmt::Display for Rope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Stream segments without a temporary allocation.
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Leaf(s) => f.write_str(s)?,
                Node::Concat { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn leaf_basics() {
        let r = Rope::leaf("hello");
        assert_eq!(r.len(), 5);
        assert!(!r.is_empty());
        assert_eq!(r.materialize(), "hello");
        assert!(Rope::new().is_empty());
        assert_eq!(Rope::new().materialize(), "");
    }

    #[test]
    fn concat_is_o1_and_flattens() {
        let a = Rope::leaf("foo");
        let b = Rope::leaf("bar");
        let ab = a.concat(&b);
        assert_eq!(ab.len(), 6);
        assert_eq!(ab.materialize(), "foobar");
        // Originals are untouched (persistent).
        assert_eq!(a.materialize(), "foo");
        assert_eq!(b.materialize(), "bar");
    }

    #[test]
    fn concat_with_empty_shares_the_other_side() {
        let a = Rope::leaf("x");
        let empty = Rope::new();
        assert_eq!(a.concat(&empty).materialize(), "x");
        assert_eq!(empty.concat(&a).materialize(), "x");
    }

    #[test]
    fn repeated_append_builds_correctly() {
        // The `s += part` loop shape: a deeply left-nested rope.
        let mut r = Rope::new();
        let mut expected = String::new();
        for i in 0..50 {
            let part = i.to_string();
            r = r.push_str(&part);
            expected.push_str(&part);
        }
        assert_eq!(r.len(), expected.len());
        assert_eq!(r.materialize(), expected);
    }

    #[test]
    fn deeply_nested_rope_flattens_without_recursion() {
        // A pathologically deep rope (10k deep) must flatten iteratively.
        let mut r = Rope::leaf("a");
        for _ in 0..10_000 {
            r = r.push_str("b");
        }
        let s = r.materialize();
        assert_eq!(s.len(), 10_001);
        assert!(s.starts_with("ab"));
        assert!(s.ends_with("bb"));
    }

    #[test]
    fn display_matches_to_string() {
        let r = Rope::leaf("a").push_str("b").concat(&Rope::leaf("cd"));
        assert_eq!(alloc::format!("{r}"), r.materialize());
        assert_eq!(r.materialize(), "abcd");
    }

    #[test]
    fn clone_is_cheap_and_shares() {
        let a = Rope::leaf("shared");
        let b = a.clone();
        // Same underlying allocation.
        assert!(Rc::ptr_eq(&a.0, &b.0));
        assert_eq!(b.materialize(), "shared");
    }
}
