//! Interned strings — "atoms" (`ROADMAP.md` §3, item 4).
//!
//! Identifiers and property keys are compared constantly: every property
//! lookup, every shape transition, every inline-cache probe. Comparing them as
//! strings is O(length); interning each distinct string to a small integer
//! **atom** makes the comparison a single `u32` equality, and gives shapes and
//! inline caches a cheap, `Copy` key.
//!
//! An `AtomTable` hands out a stable `Atom` per distinct string and resolves
//! an atom back to its text. Interning is deduplicated, so equal strings always
//! map to equal atoms — which is what makes `atom_a == atom_b` equivalent to
//! `string_a == string_b`.
//!
//! Pure, safe `alloc`-only Rust. (Rope/slice string *values* — the other half
//! of item 4 — are a separate concern from these key atoms.)
//!
//! # Storage: WTF-8 bytes
//!
//! Keys are stored as **WTF-8 bytes** (`Box<[u8]>`), so a property key may carry
//! lone UTF-16 surrogates (a JS property key is a DOMString — see
//! [`crate::wtf8`]). [`AtomTable::intern`]/[`AtomTable::get`] keep their `&str`
//! signatures (the common case), and [`AtomTable::intern_bytes`]/
//! [`AtomTable::get_bytes`] take surrogate-bearing WTF-8. [`AtomTable::resolve`]
//! still returns `Option<&str>` for the non-surrogate keys existing callers use
//! (it declines a surrogate-bearing key — those use
//! [`AtomTable::resolve_bytes`]); [`AtomTable::resolve_lossy`] always yields a
//! `String` (surrogates → U+FFFD).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::wtf8;

/// An interned string: a small, `Copy` handle that compares in O(1). Only
/// meaningful within the [`AtomTable`] that produced it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Atom(u32);

impl Atom {
    /// The atom's raw index (e.g. to store compactly in bytecode).
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Rebuilds an atom from a raw index previously obtained via
    /// [`Atom::index`]. The index must come from the same table to be
    /// meaningful.
    #[must_use]
    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

/// A deduplicating string interner: distinct strings get distinct atoms, equal
/// strings share one.
#[derive(Default)]
pub struct AtomTable {
    /// Atom index → its WTF-8 key bytes.
    strings: Vec<Box<[u8]>>,
    /// Key bytes → atom, for deduplication on intern.
    lookup: BTreeMap<Box<[u8]>, Atom>,
}

impl AtomTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns `s`, returning its atom — the same atom every time for equal
    /// strings.
    pub fn intern(&mut self, s: &str) -> Atom {
        self.intern_bytes(s.as_bytes())
    }

    /// Interns raw WTF-8 `bytes` (the surrogate-bearing path), returning its
    /// atom — the same atom every time for equal byte sequences.
    pub fn intern_bytes(&mut self, bytes: &[u8]) -> Atom {
        if let Some(&atom) = self.lookup.get(bytes) {
            return atom;
        }
        let atom = Atom(self.strings.len() as u32);
        let boxed: Box<[u8]> = Box::from(bytes);
        self.strings.push(boxed.clone());
        self.lookup.insert(boxed, atom);
        atom
    }

    /// The atom for `s` if it has been interned, without interning it.
    #[must_use]
    pub fn get(&self, s: &str) -> Option<Atom> {
        self.get_bytes(s.as_bytes())
    }

    /// The atom for raw WTF-8 `bytes` if interned, without interning it.
    #[must_use]
    pub fn get_bytes(&self, bytes: &[u8]) -> Option<Atom> {
        self.lookup.get(bytes).copied()
    }

    /// The text of `atom` as `&str`, or `None` if it does not belong to this
    /// table **or** its key bears a lone surrogate (use [`Self::resolve_bytes`]
    /// or [`Self::resolve_lossy`] for those). Non-surrogate keys — the common
    /// case — resolve for free.
    #[must_use]
    pub fn resolve(&self, atom: Atom) -> Option<&str> {
        self.resolve_bytes(atom).and_then(wtf8::as_str)
    }

    /// The raw WTF-8 key bytes of `atom`, or `None` if it does not belong to
    /// this table. Lossless — preserves lone surrogates.
    #[must_use]
    pub fn resolve_bytes(&self, atom: Atom) -> Option<&[u8]> {
        self.strings.get(atom.0 as usize).map(AsRef::as_ref)
    }

    /// The text of `atom` as a `String`, lossily (lone surrogates → U+FFFD), or
    /// `None` if it does not belong to this table.
    #[must_use]
    pub fn resolve_lossy(&self, atom: Atom) -> Option<String> {
        self.resolve_bytes(atom).map(wtf8::to_string_lossy)
    }

    /// The number of distinct interned strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether nothing has been interned yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_deduplicates() {
        let mut t = AtomTable::new();
        let a = t.intern("hello");
        let b = t.intern("hello");
        let c = t.intern("world");
        assert_eq!(a, b); // same string → same atom
        assert_ne!(a, c); // different string → different atom
        assert_eq!(t.len(), 2); // only two distinct strings stored
    }

    #[test]
    fn atom_equality_is_string_equality() {
        let mut t = AtomTable::new();
        let keys = ["x", "y", "length", "x", "constructor", "y"];
        let atoms: Vec<Atom> = keys.iter().map(|k| t.intern(k)).collect();
        for (i, ki) in keys.iter().enumerate() {
            for (j, kj) in keys.iter().enumerate() {
                assert_eq!(atoms[i] == atoms[j], ki == kj);
            }
        }
        assert_eq!(t.len(), 4); // x, y, length, constructor
    }

    #[test]
    fn resolve_round_trips() {
        let mut t = AtomTable::new();
        let a = t.intern("foo");
        let b = t.intern("bar");
        assert_eq!(t.resolve(a), Some("foo"));
        assert_eq!(t.resolve(b), Some("bar"));
        // An out-of-range atom does not resolve.
        assert_eq!(t.resolve(Atom::from_index(999)), None);
    }

    #[test]
    fn get_does_not_intern() {
        let mut t = AtomTable::new();
        assert_eq!(t.get("nope"), None);
        let a = t.intern("yep");
        assert_eq!(t.get("yep"), Some(a));
        assert_eq!(t.get("nope"), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn index_round_trips() {
        let mut t = AtomTable::new();
        let a = t.intern("z");
        assert_eq!(Atom::from_index(a.index()), a);
        assert_eq!(t.resolve(Atom::from_index(a.index())), Some("z"));
    }

    #[test]
    fn surrogate_keys_round_trip_via_bytes() {
        let mut t = AtomTable::new();
        // A property key bearing a lone high surrogate.
        let key = crate::wtf8::from_utf16(&[0x6B, 0xD800]); // "k\uD800"
        let a = t.intern_bytes(&key);
        // Interning the same bytes dedups.
        assert_eq!(t.intern_bytes(&key), a);
        assert_eq!(t.get_bytes(&key), Some(a));
        // Lossless byte resolution.
        assert_eq!(t.resolve_bytes(a), Some(key.as_slice()));
        // `resolve` (str) declines a surrogate-bearing key.
        assert_eq!(t.resolve(a), None);
        // Lossy resolution replaces the surrogate.
        assert_eq!(t.resolve_lossy(a).as_deref(), Some("k\u{FFFD}"));
        // A normal key still resolves as &str.
        let b = t.intern("plain");
        assert_eq!(t.resolve(b), Some("plain"));
        // The surrogate key and a near-identical plain key are distinct atoms.
        let c = t.intern("k");
        assert_ne!(a, c);
    }

    #[test]
    fn intern_str_and_bytes_agree() {
        let mut t = AtomTable::new();
        let a = t.intern("hi");
        let b = t.intern_bytes("hi".as_bytes());
        assert_eq!(a, b);
        assert_eq!(t.get("hi"), Some(a));
        assert_eq!(t.get_bytes("hi".as_bytes()), Some(a));
    }

    #[test]
    fn atoms_are_assigned_in_intern_order() {
        let mut t = AtomTable::new();
        assert_eq!(t.intern("a").index(), 0);
        assert_eq!(t.intern("b").index(), 1);
        assert_eq!(t.intern("a").index(), 0); // dedup, no new index
        assert_eq!(t.intern("c").index(), 2);
    }
}
