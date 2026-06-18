//! NaN-boxed value representation — groundwork for the performance object model
//! (`ROADMAP.md` §3).
//!
//! The interpreter era uses a tagged `enum Value` (one machine word for the
//! discriminant plus a payload word, reference-counted). The performance engine
//! instead packs every value into a **single 64-bit word** using the unused
//! bit-patterns of IEEE-754 double NaNs — the technique V8/JSC/SpiderMonkey all
//! use. A [`NanBox`] is either:
//!
//! - a **double** (any non-NaN `f64`, stored as its raw bits); or
//! - one of the **immediate singletons** `undefined`/`null`/`true`/`false`; or
//! - a **heap handle**: a 48-bit index into the (future) managed heap. Storing a
//!   relocatable *handle* rather than a raw pointer is deliberate — it lets a
//!   moving/generational collector relocate objects without rewriting every
//!   value (the GC updates the handle table instead).
//!
//! The encoding is canonical: a `f64` NaN is normalized to the canonical
//! quiet-NaN bit pattern on the way in, so a genuine NaN number is never
//! confused with a boxed immediate or handle.
//!
//! This module is **pure, safe, `no_std` `core`-only** Rust: the float↔bits
//! reinterpretation goes through [`f64::to_bits`]/[`f64::from_bits`] (no
//! `transmute`), and handles are plain integers (no pointer provenance), so the
//! whole representation is exercised and validated long before any `unsafe`
//! heap plumbing is written.

/// A JavaScript value packed into a single 64-bit word (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NanBox(u64);

/// The quiet-NaN base pattern. A word is a *boxed* (non-double) value exactly
/// when all of these bits are set; any other pattern is a genuine `f64`.
const QNAN: u64 = 0x7ffc_0000_0000_0000;

/// The sign bit, used to mark a boxed word as a heap handle (set) rather than an
/// immediate singleton (clear).
const SIGN: u64 = 0x8000_0000_0000_0000;

// Immediate singletons: `QNAN` with a small tag in the low bits.
const TAG_UNDEFINED: u64 = QNAN | 0x01;
const TAG_NULL: u64 = QNAN | 0x02;
const TAG_FALSE: u64 = QNAN | 0x03;
const TAG_TRUE: u64 = QNAN | 0x04;
/// An array *hole* (an elision / absent index in a sparse `Cell::Array`). It is
/// an internal sentinel that never escapes to user code: a `[[Get]]` of a hole
/// index walks the prototype chain (and yields `undefined` if nothing is found),
/// `[[HasProperty]]`/`in`/`Object.keys`/`for-in`/the iteration built-ins all
/// treat it as absent. It decodes (`unpack`/`is_undefined`/`to_boolean`/…) as
/// `undefined` so any accidental leak is harmless.
const TAG_HOLE: u64 = QNAN | 0x05;

/// The canonical quiet-NaN payload a `f64` NaN is normalized to, chosen so that
/// `(bits & QNAN) != QNAN` (i.e. it still reads back as a number).
const CANONICAL_NAN: u64 = 0x7ff8_0000_0000_0000;

/// A handle's payload occupies the low 48 bits.
const HANDLE_MASK: u64 = 0x0000_ffff_ffff_ffff;

/// The decoded view of a [`NanBox`], for ergonomic matching.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Unpacked {
    /// `undefined`.
    Undefined,
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// An IEEE-754 double.
    Number(f64),
    /// A heap handle (a 48-bit index into the managed heap).
    Handle(u64),
}

impl NanBox {
    /// `undefined`.
    #[must_use]
    pub const fn undefined() -> Self {
        Self(TAG_UNDEFINED)
    }

    /// `null`.
    #[must_use]
    pub const fn null() -> Self {
        Self(TAG_NULL)
    }

    /// An array hole — the internal absent-index sentinel for a sparse
    /// `Cell::Array` slot. It never escapes to user code: it decodes as
    /// `undefined`, while [`is_hole`](NanBox::is_hole) distinguishes it so a
    /// `[[Get]]` of the slot consults the prototype and `[[HasProperty]]` /
    /// enumeration / the iteration built-ins treat it as absent.
    #[must_use]
    pub const fn hole() -> Self {
        Self(TAG_HOLE)
    }

    /// Whether this is the array-hole sentinel.
    #[must_use]
    pub const fn is_hole(self) -> bool {
        self.0 == TAG_HOLE
    }

    /// A boolean.
    #[must_use]
    pub const fn boolean(b: bool) -> Self {
        Self(if b { TAG_TRUE } else { TAG_FALSE })
    }

    /// A number. A `f64` NaN is normalized to the canonical quiet NaN so it can
    /// never alias a boxed immediate or handle.
    #[must_use]
    pub fn number(n: f64) -> Self {
        if n.is_nan() {
            Self(CANONICAL_NAN)
        } else {
            Self(n.to_bits())
        }
    }

    /// A heap handle. Only the low 48 bits are significant; higher bits are
    /// ignored (the managed heap is addressed by index, well within 2^48).
    #[must_use]
    pub const fn handle(index: u64) -> Self {
        Self(SIGN | QNAN | (index & HANDLE_MASK))
    }

    /// The raw 64-bit encoding (e.g. to store in a register file).
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Reconstructs a box from its raw encoding (the inverse of [`to_bits`]).
    ///
    /// [`to_bits`]: NanBox::to_bits
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Whether this word encodes a double.
    #[must_use]
    pub const fn is_number(self) -> bool {
        (self.0 & QNAN) != QNAN
    }

    /// Whether this word is a heap handle.
    #[must_use]
    pub const fn is_handle(self) -> bool {
        !self.is_number() && (self.0 & SIGN) == SIGN
    }

    /// Whether this is `undefined`. A leaked array hole also reports `true`
    /// (it decodes as `undefined`), so callers never observe the sentinel.
    #[must_use]
    pub const fn is_undefined(self) -> bool {
        self.0 == TAG_UNDEFINED || self.0 == TAG_HOLE
    }

    /// Whether this is `null`.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == TAG_NULL
    }

    /// Whether this is a boolean.
    #[must_use]
    pub const fn is_boolean(self) -> bool {
        self.0 == TAG_TRUE || self.0 == TAG_FALSE
    }

    /// The number, if this is one.
    #[must_use]
    pub fn as_number(self) -> Option<f64> {
        if self.is_number() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    /// The boolean, if this is one.
    #[must_use]
    pub const fn as_boolean(self) -> Option<bool> {
        match self.0 {
            TAG_TRUE => Some(true),
            TAG_FALSE => Some(false),
            _ => None,
        }
    }

    /// The heap-handle index, if this is a handle.
    #[must_use]
    pub const fn as_handle(self) -> Option<u64> {
        if self.is_handle() {
            Some(self.0 & HANDLE_MASK)
        } else {
            None
        }
    }

    /// Decodes into the [`Unpacked`] view for matching.
    #[must_use]
    pub fn unpack(self) -> Unpacked {
        if self.is_number() {
            return Unpacked::Number(f64::from_bits(self.0));
        }
        match self.0 {
            TAG_UNDEFINED | TAG_HOLE => Unpacked::Undefined,
            TAG_NULL => Unpacked::Null,
            TAG_TRUE => Unpacked::Bool(true),
            TAG_FALSE => Unpacked::Bool(false),
            // Any remaining boxed word is a handle (sign bit set by construction).
            _ => Unpacked::Handle(self.0 & HANDLE_MASK),
        }
    }

    // --- ECMAScript abstract operations (the primitive fast paths a VM needs
    //     on every conditional branch and comparison) ---

    /// ECMAScript `ToBoolean`. Heap objects are always truthy; `undefined`,
    /// `null`, `false`, `+0`/`-0`, and `NaN` are falsy, everything else truthy.
    #[must_use]
    pub fn to_boolean(self) -> bool {
        match self.unpack() {
            Unpacked::Undefined | Unpacked::Null => false,
            Unpacked::Bool(b) => b,
            Unpacked::Number(n) => n != 0.0 && !n.is_nan(),
            Unpacked::Handle(_) => true,
        }
    }

    /// ECMAScript strict equality (`===`) for the cases decidable without the
    /// heap: numbers compare by IEEE-754 value (so `NaN !== NaN` and
    /// `+0 === -0`); everything else compares by encoding, which is identity for
    /// handles and equality for the singletons. (String/object value equality
    /// that needs heap contents layers on top.)
    #[must_use]
    pub fn strict_equals(self, other: Self) -> bool {
        if self.is_number() && other.is_number() {
            f64::from_bits(self.0) == f64::from_bits(other.0)
        } else {
            self.0 == other.0
        }
    }

    /// ECMAScript `SameValue` (`Object.is`): like [`strict_equals`] but
    /// `NaN` is the same as `NaN` and `+0` is *not* the same as `-0`.
    ///
    /// [`strict_equals`]: NanBox::strict_equals
    #[must_use]
    pub fn same_value(self, other: Self) -> bool {
        if self.is_number() && other.is_number() {
            let a = f64::from_bits(self.0);
            let b = f64::from_bits(other.0);
            if a.is_nan() && b.is_nan() {
                return true;
            }
            if a == 0.0 && b == 0.0 {
                return a.is_sign_negative() == b.is_sign_negative();
            }
            a == b
        } else {
            self.0 == other.0
        }
    }

    /// The `typeof` string for the cases decidable without the heap. Note
    /// `typeof null === "object"`. A handle is reported as `"object"` here —
    /// distinguishing a callable (`"function"`) requires inspecting the heap
    /// object, so that refinement lives with the heap, not the box.
    #[must_use]
    pub fn type_of(self) -> &'static str {
        match self.unpack() {
            Unpacked::Undefined => "undefined",
            Unpacked::Null => "object",
            Unpacked::Bool(_) => "boolean",
            Unpacked::Number(_) => "number",
            Unpacked::Handle(_) => "object",
        }
    }
}

impl core::fmt::Debug for NanBox {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.unpack() {
            Unpacked::Undefined => write!(f, "undefined"),
            Unpacked::Null => write!(f, "null"),
            Unpacked::Bool(b) => write!(f, "{b}"),
            Unpacked::Number(n) => write!(f, "{n}"),
            Unpacked::Handle(h) => write!(f, "handle({h})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singletons_are_distinct_and_typed() {
        assert!(NanBox::undefined().is_undefined());
        assert!(NanBox::null().is_null());
        assert!(NanBox::boolean(true).is_boolean());
        assert!(NanBox::boolean(false).is_boolean());

        // Mutually exclusive.
        assert!(!NanBox::undefined().is_null());
        assert!(!NanBox::null().is_undefined());
        assert!(!NanBox::undefined().is_number());
        assert!(!NanBox::undefined().is_handle());
        assert_ne!(NanBox::undefined(), NanBox::null());
        assert_ne!(NanBox::boolean(true), NanBox::boolean(false));

        assert_eq!(NanBox::boolean(true).as_boolean(), Some(true));
        assert_eq!(NanBox::boolean(false).as_boolean(), Some(false));
        assert_eq!(NanBox::undefined().as_boolean(), None);
    }

    #[test]
    fn numbers_round_trip() {
        for &n in &[
            0.0_f64,
            -0.0,
            1.0,
            -1.0,
            123.456,
            42.0,
            -273.15,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e308,
            -1e-308,
        ] {
            let b = NanBox::number(n);
            assert!(b.is_number(), "{n} should be a number");
            assert!(!b.is_handle());
            assert!(!b.is_undefined());
            assert_eq!(b.as_number(), Some(n));
            assert_eq!(b.unpack(), Unpacked::Number(n));
        }
        // Signed zero is preserved bit-exactly.
        assert!(NanBox::number(-0.0).as_number().unwrap().is_sign_negative());
    }

    #[test]
    fn nan_is_a_number_not_a_box() {
        let b = NanBox::number(f64::NAN);
        assert!(b.is_number());
        assert!(!b.is_handle() && !b.is_undefined() && !b.is_null() && !b.is_boolean());
        assert!(b.as_number().unwrap().is_nan());
        // A NaN produced by arithmetic (a different bit pattern) is also boxed
        // as a number, not mistaken for an immediate/handle.
        let arith_nan = NanBox::number(f64::INFINITY - f64::INFINITY);
        assert!(arith_nan.is_number());
        assert!(arith_nan.as_number().unwrap().is_nan());
    }

    #[test]
    fn handles_round_trip() {
        for &h in &[
            0_u64,
            1,
            2,
            1000,
            0xffff,
            0x1_0000,
            HANDLE_MASK,
            HANDLE_MASK - 1,
        ] {
            let b = NanBox::handle(h);
            assert!(b.is_handle(), "handle {h} should be a handle");
            assert!(!b.is_number());
            assert!(!b.is_undefined() && !b.is_null() && !b.is_boolean());
            assert_eq!(b.as_handle(), Some(h));
            assert_eq!(b.unpack(), Unpacked::Handle(h));
        }
    }

    #[test]
    fn bits_round_trip() {
        // The raw encoding round-trips bit-exactly (including NaN, whose
        // `Unpacked` view would not self-compare since `NaN != NaN`).
        for b in [
            NanBox::undefined(),
            NanBox::null(),
            NanBox::boolean(true),
            NanBox::boolean(false),
            NanBox::number(123.456),
            NanBox::number(f64::NAN),
            NanBox::handle(0x1234_5678),
        ] {
            assert_eq!(NanBox::from_bits(b.to_bits()), b);
            assert_eq!(NanBox::from_bits(b.to_bits()).to_bits(), b.to_bits());
        }
    }

    #[test]
    fn to_boolean_follows_spec() {
        assert!(!NanBox::undefined().to_boolean());
        assert!(!NanBox::null().to_boolean());
        assert!(NanBox::boolean(true).to_boolean());
        assert!(!NanBox::boolean(false).to_boolean());
        assert!(!NanBox::number(0.0).to_boolean());
        assert!(!NanBox::number(-0.0).to_boolean());
        assert!(!NanBox::number(f64::NAN).to_boolean());
        assert!(NanBox::number(1.0).to_boolean());
        assert!(NanBox::number(-1.0).to_boolean());
        assert!(NanBox::number(f64::INFINITY).to_boolean());
        assert!(NanBox::handle(0).to_boolean()); // objects are truthy
    }

    #[test]
    fn strict_equals_follows_spec() {
        // Numbers: value equality, NaN != NaN, +0 === -0.
        assert!(NanBox::number(1.0).strict_equals(NanBox::number(1.0)));
        assert!(!NanBox::number(1.0).strict_equals(NanBox::number(2.0)));
        assert!(!NanBox::number(f64::NAN).strict_equals(NanBox::number(f64::NAN)));
        assert!(NanBox::number(0.0).strict_equals(NanBox::number(-0.0)));
        // Singletons.
        assert!(NanBox::undefined().strict_equals(NanBox::undefined()));
        assert!(NanBox::null().strict_equals(NanBox::null()));
        assert!(!NanBox::undefined().strict_equals(NanBox::null()));
        assert!(NanBox::boolean(true).strict_equals(NanBox::boolean(true)));
        assert!(!NanBox::boolean(true).strict_equals(NanBox::boolean(false)));
        // Handles: identity.
        assert!(NanBox::handle(7).strict_equals(NanBox::handle(7)));
        assert!(!NanBox::handle(7).strict_equals(NanBox::handle(8)));
        // Cross-kind.
        assert!(!NanBox::number(0.0).strict_equals(NanBox::null()));
        assert!(!NanBox::handle(0).strict_equals(NanBox::number(0.0)));
    }

    #[test]
    fn same_value_differs_from_strict_on_nan_and_zero() {
        // Object.is: NaN is the same as NaN, +0 is not the same as -0.
        assert!(NanBox::number(f64::NAN).same_value(NanBox::number(f64::NAN)));
        assert!(!NanBox::number(0.0).same_value(NanBox::number(-0.0)));
        assert!(NanBox::number(0.0).same_value(NanBox::number(0.0)));
        assert!(NanBox::number(-0.0).same_value(NanBox::number(-0.0)));
        assert!(NanBox::number(3.0).same_value(NanBox::number(3.0)));
        assert!(NanBox::handle(1).same_value(NanBox::handle(1)));
        assert!(!NanBox::handle(1).same_value(NanBox::handle(2)));
    }

    #[test]
    fn type_of_follows_spec() {
        assert_eq!(NanBox::undefined().type_of(), "undefined");
        assert_eq!(NanBox::null().type_of(), "object"); // the classic quirk
        assert_eq!(NanBox::boolean(true).type_of(), "boolean");
        assert_eq!(NanBox::number(1.0).type_of(), "number");
        assert_eq!(NanBox::number(f64::NAN).type_of(), "number");
        assert_eq!(NanBox::handle(0).type_of(), "object");
    }

    #[test]
    fn unpack_matches_predicates() {
        let cases = [
            (NanBox::undefined(), Unpacked::Undefined),
            (NanBox::null(), Unpacked::Null),
            (NanBox::boolean(true), Unpacked::Bool(true)),
            (NanBox::boolean(false), Unpacked::Bool(false)),
            (NanBox::number(2.5), Unpacked::Number(2.5)),
            (NanBox::handle(7), Unpacked::Handle(7)),
        ];
        for (box_, expected) in cases {
            assert_eq!(box_.unpack(), expected);
        }
    }
}
