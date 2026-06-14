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
//!
//! # Storage: WTF-8 bytes
//!
//! Leaves hold **WTF-8 bytes** (`Box<[u8]>`), not `Box<str>`, so a string may
//! carry lone UTF-16 surrogates (DOMString semantics — see [`crate::wtf8`]). A
//! string with no surrogates is byte-identical to its UTF-8, so the common case
//! is unchanged. `Rope::from`/`Rope::leaf` store a `&str`'s (valid-WTF-8)
//! bytes; `Rope::from_wtf8`/`Rope::from_bytes` take surrogate-bearing bytes.
//! `Rope::materialize_bytes` returns the WTF-8 bytes; `Rope::materialize`
//! and `Display` stay `String`-typed and are **lossy** (surrogates → U+FFFD),
//! so existing `&str`/`String` callers compile and behave unchanged for
//! non-surrogate strings. Surrogate-correct accessors land in a later milestone.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wtf8;

/// Maximum length, in bytes, of a string the engine will build by
/// concatenation. A `+` whose result would exceed this is a `RangeError`
/// ("Invalid string length") rather than a `usize` overflow / OOM bomb.
pub const MAX_STRING_LEN: usize = crate::limits::DEFAULT_MAX_STRING_LEN;

/// A string value represented as a tree of concatenated segments. Cloning is a
/// reference-count bump; concatenation and length are O(1).
#[derive(Clone)]
pub struct Rope(Rc<Node>);

enum Node {
    /// A contiguous run of WTF-8 bytes.
    Leaf(Box<[u8]>),
    /// The concatenation of two ropes, with the total byte length cached.
    Concat { left: Rope, right: Rope, len: usize },
}

impl Rope {
    /// An empty rope.
    #[must_use]
    pub fn new() -> Self {
        Self::leaf("")
    }

    /// A leaf rope holding `s` (stored as its UTF-8 bytes, which are valid
    /// WTF-8).
    #[must_use]
    pub fn leaf(s: &str) -> Self {
        Rope(Rc::new(Node::Leaf(Box::from(s.as_bytes()))))
    }

    /// A leaf rope holding raw WTF-8 `bytes` — the surrogate-bearing path. The
    /// bytes are taken as-is (assumed already well-formed WTF-8, e.g. from
    /// [`crate::wtf8::from_utf16`] or another rope's `Rope::materialize_bytes`).
    #[must_use]
    pub fn from_wtf8(bytes: Vec<u8>) -> Self {
        Rope(Rc::new(Node::Leaf(bytes.into_boxed_slice())))
    }

    /// A leaf rope copying raw WTF-8 `bytes`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Rope(Rc::new(Node::Leaf(Box::from(bytes))))
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

    /// Concatenates `self` and `other` in O(1) when the combined length is
    /// representable (≤ [`MAX_STRING_LEN`]), sharing both sides. Returns `None`
    /// when the result would overflow `usize` or exceed the cap — the caller
    /// turns that into a `RangeError("Invalid string length")` rather than
    /// corrupting the cached length or OOM-ing on materialization.
    #[must_use]
    pub fn try_concat(&self, other: &Rope) -> Option<Rope> {
        // Concatenating with empty just shares the non-empty side.
        if self.is_empty() {
            return Some(other.clone());
        }
        if other.is_empty() {
            return Some(self.clone());
        }
        let len = self.len().checked_add(other.len())?;
        if len > MAX_STRING_LEN {
            return None;
        }
        Some(Rope(Rc::new(Node::Concat {
            left: self.clone(),
            right: other.clone(),
            len,
        })))
    }

    /// Concatenates `self` and `other` in O(1), sharing both sides. A result
    /// past [`MAX_STRING_LEN`] (or `usize` overflow) is clamped to sharing only
    /// the left side — a bounded, panic-free degradation. Callers that need to
    /// surface the spec's `RangeError` use [`Rope::try_concat`] instead.
    #[must_use]
    pub fn concat(&self, other: &Rope) -> Rope {
        self.try_concat(other).unwrap_or_else(|| self.clone())
    }

    /// Appends `s` to the rope (O(1)), returning the new rope.
    #[must_use]
    pub fn push_str(&self, s: &str) -> Rope {
        self.concat(&Rope::leaf(s))
    }

    /// Borrows the rope's WTF-8 bytes directly when it is a single, unconcatenated
    /// `Leaf` (the common case — a string that has never been `+`-joined), returning
    /// `None` for a `Concat` tree. This is a zero-copy fast path: callers that only
    /// *read* the bytes (equality, ordering, emptiness) avoid the owned `Vec` that
    /// [`Rope::materialize_bytes`] allocates. Fall back to `materialize_bytes` when
    /// this returns `None`.
    #[must_use]
    pub fn as_leaf_bytes(&self) -> Option<&[u8]> {
        match &*self.0 {
            Node::Leaf(s) => Some(s),
            Node::Concat { .. } => None,
        }
    }

    /// Materializes the rope into a flat WTF-8 byte buffer in O(n), iteratively
    /// (so a deeply nested rope cannot overflow the stack). This is the
    /// lossless form — lone surrogates are preserved. Callers wanting a real
    /// `&str`/`String` use `Rope::materialize` (lossy).
    #[must_use]
    pub fn materialize_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len());
        // Depth-first, left-to-right, using an explicit stack of pending nodes.
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Leaf(s) => out.extend_from_slice(s),
                Node::Concat { left, right, .. } => {
                    // Push right first so left is processed first (LIFO).
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        out
    }

    /// Materializes the rope into a flat `String` in O(n). **Lossy**: any stored
    /// lone surrogate is replaced with U+FFFD (via [`crate::wtf8::to_string_lossy`]),
    /// so the signature stays `String` for existing `&str`/`String` callers. A
    /// non-surrogate string is unchanged. Use `Rope::materialize_bytes` for the
    /// lossless WTF-8 bytes.
    #[must_use]
    pub fn materialize(&self) -> String {
        wtf8::to_string_lossy(&self.materialize_bytes())
    }
}

impl Drop for Node {
    /// Dismantles a `Concat` chain iteratively. A deeply nested rope would
    /// otherwise free recursively (each `Node` drops its child `Rc<Node>`, which
    /// drops the next…), overflowing the stack on platforms with a small default
    /// (e.g. Windows' 1 MB test threads) for the pathological left-/right-deep
    /// shape that O(1) concatenation builds.
    fn drop(&mut self) {
        // A cheap leaf to swap into a child slot so we can take ownership of the
        // real `Rc<Node>` without recursing through its `Drop`.
        let sentinel = || Rc::new(Node::Leaf(Box::from(&[][..])));
        let Node::Concat { left, right, .. } = self else {
            return; // a leaf owns nothing recursive
        };
        let mut stack = alloc::vec![
            core::mem::replace(&mut left.0, sentinel()),
            core::mem::replace(&mut right.0, sentinel()),
        ];
        while let Some(rc) = stack.pop() {
            // Only the last owner can free; otherwise just drop the `Rc` (a
            // refcount decrement, no recursion).
            if let Ok(mut node) = Rc::try_unwrap(rc)
                && let Node::Concat { left, right, .. } = &mut node
            {
                stack.push(core::mem::replace(&mut left.0, sentinel()));
                stack.push(core::mem::replace(&mut right.0, sentinel()));
                // `node` now holds sentinel leaves, so its own drop is shallow.
            }
        }
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
        // Stream segments. Leaves are usually plain UTF-8 (the `as_str` fast
        // path), written without allocation; a leaf bearing lone surrogates is
        // written lossily (surrogates → U+FFFD), matching `materialize`.
        let mut stack: Vec<&Rope> = alloc::vec![self];
        while let Some(node) = stack.pop() {
            match &*node.0 {
                Node::Leaf(s) => match wtf8::as_str(s) {
                    Some(valid) => f.write_str(valid)?,
                    None => f.write_str(&wtf8::to_string_lossy(s))?,
                },
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

    /// A deeply nested rope must also *drop* iteratively — a recursive free
    /// overflows small stacks (this regressed CI on Windows' 1 MB test
    /// threads). Run on a deliberately tiny stack to catch it everywhere.
    #[cfg(feature = "std")]
    #[test]
    fn deeply_nested_rope_drops_without_overflow() {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut r = Rope::leaf("a");
                for _ in 0..100_000 {
                    r = r.push_str("b");
                }
                assert_eq!(r.materialize().len(), 100_001);
                // `r` drops here, on the 256 KB stack.
            })
            .expect("spawn")
            .join()
            .expect("the deep rope dropped without overflowing the stack");
    }

    #[test]
    fn display_matches_to_string() {
        let r = Rope::leaf("a").push_str("b").concat(&Rope::leaf("cd"));
        assert_eq!(alloc::format!("{r}"), r.materialize());
        assert_eq!(r.materialize(), "abcd");
    }

    #[test]
    fn non_surrogate_string_is_byte_identical() {
        // The common case: a normal string round-trips byte-for-byte through the
        // rope, exactly as before the WTF-8 storage change.
        for s in ["", "hello", "héllo 中 😀 mix"] {
            let r = Rope::from(s);
            assert_eq!(r.materialize_bytes(), s.as_bytes());
            assert_eq!(r.materialize(), s);
            assert_eq!(r.len(), s.len());
        }
    }

    #[test]
    fn lone_surrogate_round_trips_through_rope() {
        // "a\uD800b" stored via from_wtf8 survives losslessly in the bytes,
        // while the lossy String form replaces the surrogate with U+FFFD.
        let bytes = crate::wtf8::from_utf16(&[0x61, 0xD800, 0x62]);
        let r = Rope::from_wtf8(bytes.clone());
        assert_eq!(r.materialize_bytes(), bytes);
        assert_eq!(r.materialize(), "a\u{FFFD}b");
        assert_eq!(alloc::format!("{r}"), "a\u{FFFD}b");
        assert_eq!(r.len(), bytes.len()); // byte length, not unit length
    }

    #[test]
    fn concat_preserves_surrogate_bytes() {
        let left = Rope::from("x");
        let right = Rope::from_wtf8(crate::wtf8::from_utf16(&[0xD800]));
        let joined = left.concat(&right);
        let mut expected = alloc::vec![b'x'];
        expected.extend_from_slice(&crate::wtf8::from_utf16(&[0xD800]));
        assert_eq!(joined.materialize_bytes(), expected);
    }

    #[test]
    fn from_bytes_matches_from_wtf8() {
        let bytes = crate::wtf8::from_utf16(&[0xDC00, 0x41]);
        assert_eq!(
            Rope::from_bytes(&bytes).materialize_bytes(),
            Rope::from_wtf8(bytes.clone()).materialize_bytes()
        );
    }

    #[test]
    fn as_leaf_bytes_borrows_unconcatenated_leaf() {
        // A plain leaf exposes its bytes zero-copy.
        let leaf = Rope::leaf("hello");
        assert_eq!(leaf.as_leaf_bytes(), Some(&b"hello"[..]));
        // A surrogate-bearing leaf is exposed losslessly.
        let bytes = crate::wtf8::from_utf16(&[0xD800]);
        let surr = Rope::from_wtf8(bytes.clone());
        assert_eq!(surr.as_leaf_bytes(), Some(&bytes[..]));
        // A concatenation has no single backing slice.
        let joined = Rope::leaf("foo").concat(&Rope::leaf("bar"));
        assert_eq!(joined.as_leaf_bytes(), None);
        // …but its materialized bytes still agree with the borrow-or-copy path.
        assert_eq!(joined.materialize_bytes(), b"foobar");
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
