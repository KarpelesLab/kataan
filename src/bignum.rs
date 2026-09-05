//! The engine's arbitrary-precision signed integer — the foundation for a
//! conformant `BigInt`.
//!
//! This is a thin wrapper over [`puremp::Int`] (the Karpelès Lab pure-Rust
//! multi-precision maths crate), preserving a small, `BigInt`-focused API so the
//! rest of the engine is decoupled from the backend. `puremp::Int` provides the
//! semantically-critical operations directly: truncated (toward-zero) division
//! (`div_rem`), two's-complement bitwise ops, and radix I/O. No `unsafe`, no
//! foreign code; `no_std` + `alloc`.

use alloc::string::String;
use core::cmp::Ordering;
use puremp::Int;

/// An arbitrary-precision signed integer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BigInt(Int);

impl BigInt {
    /// Zero.
    #[must_use]
    pub fn zero() -> Self {
        Self(Int::from(0i32))
    }

    /// Whether this is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Whether this is strictly negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    /// Builds from an `i128`.
    #[must_use]
    pub fn from_i128(v: i128) -> Self {
        Self(Int::from_i128(v))
    }

    /// Converts to the nearest `f64` (overflowing to ±∞ for huge magnitudes),
    /// with correct round-to-nearest-even. For any value that fits in an `i128`
    /// (magnitude < 2^127) we route through Rust's `i128 as f64`, which is IEEE
    /// correctly-rounded; the backend's own conversion can round the wrong way at
    /// a tie's neighbour (e.g. `Number(8692288669465520373761n)`), so it is only a
    /// fallback for the rare > 2^127 magnitudes.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        if let Some(v) = self.to_i128() {
            return v as f64;
        }
        // Beyond `i128`: assemble the IEEE-754 double by hand from the top 54
        // magnitude bits plus a sticky bit, rounding half-to-even. (The backend's
        // own `to_f64` rounds the wrong way on some ties' neighbours, which is
        // observable as `BigInt(Number(x)) !== x` for exactly-representable `x`.)
        let neg = self.is_negative();
        let mag = if neg { self.neg() } else { self.clone() };
        let bits = mag.bit_len();
        debug_assert!(bits > 127);
        // A magnitude of more than 1024 bits is at least 2^1024 — always ±∞. Taking
        // it first keeps the `2^shift` divisor below 2^970, so an astronomically
        // large BigInt costs no big-integer work here.
        if bits > 1024 {
            return if neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        // Split off all but the top 54 bits; `rem != 0` is the sticky bit.
        let shift = bits - 54;
        let (q, rem) = match mag.divmod(&Self::from_i128(2).pow(shift)) {
            Some(qr) => qr,
            None => return f64::NAN,
        };
        // `q` has exactly 54 bits, so it fits an `i128` comfortably.
        let mut m = q.to_i128().unwrap_or(0) as u64;
        let sticky = !rem.is_zero();
        let round_bit = m & 1;
        m >>= 1; // the 53-bit significand
        let mut exp = shift + 1; // value ≈ m · 2^exp
        if round_bit == 1 && (sticky || (m & 1) == 1) {
            m += 1;
            if m == 1 << 53 {
                m >>= 1;
                exp += 1;
            }
        }
        // IEEE-754 binary64: `m` is normalized in [2^52, 2^53), so the biased
        // exponent is `exp + 52 + 1023`; at or past 2047 the value overflows.
        let biased = exp + 52 + 1023;
        if biased >= 2047 {
            return if neg {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        let raw = (u64::from(neg) << 63) | (biased << 52) | (m & 0x000f_ffff_ffff_ffff);
        f64::from_bits(raw)
    }

    /// Converts to an `i128` if it fits, else `None`.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        i128::try_from(&self.0).ok()
    }

    /// The low 64 bits of the value in two's-complement, regardless of magnitude
    /// (the `BigInt64Array`/`BigUint64Array` element encoding: `ToBigInt64` /
    /// `ToBigUint64` keep only the low 64 bits). For a negative value this is the
    /// wrapped two's-complement bit pattern (`(-1n) -> 0xFFFF_FFFF_FFFF_FFFF`).
    #[must_use]
    pub fn to_u64_wrapping(&self) -> u64 {
        // `mod_2k(64)` is the non-negative residue mod 2^64 — exactly the
        // two's-complement low 64 bits — so it always fits a `u64`.
        u64::try_from(&self.0.mod_2k(64)).unwrap_or(0)
    }

    /// Returns `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self(self.0.add(&other.0))
    }

    /// Returns `-self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        Self(self.0.neg())
    }

    /// Returns `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        Self(self.0.sub(&other.0))
    }

    /// Returns `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Self(self.0.mul(&other.0))
    }

    /// Returns `(quotient, remainder)` with truncated (toward-zero) division;
    /// the remainder takes the dividend's sign. Returns `None` on divide-by-zero.
    #[must_use]
    pub fn divmod(&self, other: &Self) -> Option<(Self, Self)> {
        // `div_rem` is truncated (toward-zero): the remainder takes the
        // dividend's sign, matching BigInt `/` and `%`. `None` on divide-by-zero.
        self.0.div_rem(&other.0).map(|(q, r)| (Self(q), Self(r)))
    }

    /// Bitwise AND (two's-complement).
    #[must_use]
    pub fn bitand(&self, other: &Self) -> Self {
        Self(self.0.bitand(&other.0))
    }

    /// Bitwise OR (two's-complement).
    #[must_use]
    pub fn bitor(&self, other: &Self) -> Self {
        Self(self.0.bitor(&other.0))
    }

    /// Bitwise XOR (two's-complement).
    #[must_use]
    pub fn bitxor(&self, other: &Self) -> Self {
        Self(self.0.bitxor(&other.0))
    }

    /// The number of bits in the magnitude (0 for zero) — i.e.
    /// `floor(log2(|n|)) + 1`. Used to bound the projected size of a `pow`/shift
    /// result before growing it, so an attacker exponent cannot drive an OOM.
    #[must_use]
    pub fn bit_len(&self) -> u64 {
        u64::from(self.0.bit_len())
    }

    /// Like [`pow`](Self::pow), but refuses to build a result larger than
    /// `max_bits` bits, returning `None` instead. The result of `self ** exp`
    /// has roughly `bit_len(self) * exp` bits, so this rejects the allocation
    /// up front — a defense-in-depth guard so no caller can trigger a
    /// multi-gigabyte allocation bomb (MEM-6). Computes `pow(exp)` when within
    /// bounds.
    #[must_use]
    pub fn try_pow(&self, exp: u64, max_bits: u64) -> Option<Self> {
        if self.bit_len().saturating_mul(exp) > max_bits {
            return None;
        }
        Some(self.pow(exp))
    }

    /// Returns `self ** exp` (non-negative exponent) by binary exponentiation.
    #[must_use]
    pub fn pow(&self, exp: u64) -> Self {
        // `puremp::Int::pow` takes a `u32`; a `u64` exponent large enough to
        // overflow it is always rejected by `try_pow`'s `max_bits` guard first,
        // but fall back to binary exponentiation so any exponent stays correct.
        if let Ok(e) = u32::try_from(exp) {
            return Self(self.0.pow(e));
        }
        let mut result = Self::from_i128(1);
        let mut base = self.clone();
        let mut exp = exp;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base);
            }
        }
        result
    }

    /// Renders in `radix` (2..=36), with a leading `-` when negative.
    #[must_use]
    pub fn to_str_radix(&self, radix: u32) -> String {
        debug_assert!((2..=36).contains(&radix));
        let mut s = String::new();
        // `write_radix` into a `String` is infallible.
        let _ = self.0.write_radix(&mut s, radix);
        s
    }

    /// Parses `s` in `radix` (2..=36); an optional leading `-`/`+` is allowed.
    /// Returns `None` on an invalid digit.
    #[must_use]
    pub fn from_str_radix(s: &str, radix: u32) -> Option<Self> {
        Int::from_str_radix(s.trim(), radix).ok().map(Self)
    }
}

impl Ord for BigInt {
    /// Signed comparison.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl core::fmt::Display for BigInt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::BigInt;
    use alloc::string::ToString;

    fn b(s: &str) -> BigInt {
        BigInt::from_str_radix(s, 10).unwrap()
    }

    #[test]
    fn roundtrips_decimal() {
        for s in [
            "0",
            "1",
            "-1",
            "42",
            "-42",
            "1000000000000000000000000000000",
        ] {
            assert_eq!(b(s).to_string(), s);
        }
    }

    #[test]
    fn add_and_sub() {
        assert_eq!(b("10").add(&b("20")).to_string(), "30");
        assert_eq!(b("-5").add(&b("3")).to_string(), "-2");
        assert_eq!(b("5").add(&b("-5")).to_string(), "0");
        assert_eq!(b("100").sub(&b("250")).to_string(), "-150");
        // Carries across a limb boundary.
        assert_eq!(b("4294967295").add(&b("1")).to_string(), "4294967296");
    }

    #[test]
    fn multiply_beyond_i128() {
        // 2^127 * 2^127 = 2^254, which overflows i128.
        let big = b("170141183460469231731687303715884105728"); // 2^127
        assert_eq!(
            big.mul(&big).to_string(),
            "28948022309329048855892746252171976963317496166410141009864396001978282409984"
        );
    }

    #[test]
    fn divmod_truncates_toward_zero() {
        assert_eq!(b("17").divmod(&b("5")).unwrap().0.to_string(), "3");
        assert_eq!(b("17").divmod(&b("5")).unwrap().1.to_string(), "2");
        assert_eq!(b("-17").divmod(&b("5")).unwrap().1.to_string(), "-2");
        assert_eq!(
            b("1000000000000000000000")
                .divmod(&b("7"))
                .unwrap()
                .0
                .to_string(),
            "142857142857142857142"
        );
        assert!(b("1").divmod(&b("0")).is_none());
    }

    #[test]
    fn pow_and_radix() {
        assert_eq!(
            b("2").pow(100).to_string(),
            "1267650600228229401496703205376"
        );
        assert_eq!(b("255").to_str_radix(16), "ff");
        assert_eq!(BigInt::from_str_radix("ff", 16).unwrap().to_string(), "255");
        assert_eq!(
            BigInt::from_str_radix("-1010", 2).unwrap().to_string(),
            "-10"
        );
    }

    #[test]
    fn bitwise_twos_complement() {
        assert_eq!(b("12").bitand(&b("10")).to_string(), "8");
        assert_eq!(b("12").bitor(&b("10")).to_string(), "14");
        assert_eq!(b("12").bitxor(&b("10")).to_string(), "6");
        // Negative operands follow two's-complement semantics.
        assert_eq!(b("-1").bitand(&b("12")).to_string(), "12"); // -1 is all ones
        assert_eq!(b("-12").bitor(&b("10")).to_string(), "-2");
        assert_eq!(b("-5").bitxor(&b("3")).to_string(), "-8");
        // Beyond i128 width.
        let big = b("2").pow(200);
        assert_eq!(big.bitor(&BigInt::from_i128(1)).sub(&big).to_string(), "1");
    }

    #[test]
    fn i128_roundtrip() {
        for v in [0i128, 1, -1, i128::MAX, i128::MIN, 123456789] {
            assert_eq!(BigInt::from_i128(v).to_i128(), Some(v));
        }
        // A value beyond i128 has no i128 form.
        assert_eq!(b("2").pow(200).to_i128(), None);
    }

    #[test]
    fn to_u64_wrapping_twos_complement() {
        assert_eq!(BigInt::from_i128(0).to_u64_wrapping(), 0);
        assert_eq!(BigInt::from_i128(1).to_u64_wrapping(), 1);
        assert_eq!(BigInt::from_i128(-1).to_u64_wrapping(), u64::MAX);
        assert_eq!(BigInt::from_i128(255).to_u64_wrapping(), 255);
        // Only the low 64 bits survive.
        assert_eq!(b("18446744073709551617").to_u64_wrapping(), 1); // 2^64 + 1
    }
}
