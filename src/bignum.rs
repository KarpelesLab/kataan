//! A pure, `alloc`-only arbitrary-precision signed integer — the foundation for
//! a conformant `BigInt` (replacing the bounded `i128` approximation).
//!
//! The magnitude is stored little-endian in base-2^32 limbs with no leading
//! zeros (an empty limb vector is the canonical zero). Arithmetic is schoolbook:
//! O(n) add/sub, O(n·m) multiply, and a simple bit-at-a-time long division that
//! is correct for any divisor. Base conversion (to/from decimal and other
//! radixes) rides on division by a single limb. No `unsafe`, no foreign code.

use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// An arbitrary-precision signed integer.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BigInt {
    /// Sign: `true` iff the value is strictly negative. Zero is non-negative.
    negative: bool,
    /// Little-endian base-2^32 magnitude, normalized (no trailing zero limbs);
    /// an empty vector denotes zero.
    mag: Vec<u32>,
}

impl BigInt {
    /// Zero.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            negative: false,
            mag: Vec::new(),
        }
    }

    /// Whether this is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    /// Whether this is strictly negative.
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Builds from an `i128` (the previous representation, for migration).
    #[must_use]
    pub fn from_i128(v: i128) -> Self {
        if v == 0 {
            return Self::zero();
        }
        let negative = v < 0;
        // Use the unsigned magnitude (handles i128::MIN without overflow).
        let mut u = v.unsigned_abs();
        let mut mag = Vec::new();
        while u != 0 {
            mag.push((u & 0xFFFF_FFFF) as u32);
            u >>= 32;
        }
        Self { negative, mag }.normalized()
    }

    /// Converts to the nearest `f64` (overflowing to ±∞ for huge magnitudes).
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        let mut result = 0.0_f64;
        // Horner over little-endian base-2^32 limbs, high limb first.
        for &limb in self.mag.iter().rev() {
            result = result * 4_294_967_296.0 + f64::from(limb);
        }
        if self.negative { -result } else { result }
    }

    /// Converts to an `i128` if it fits, else `None`.
    #[must_use]
    pub fn to_i128(&self) -> Option<i128> {
        if self.mag.len() > 4 {
            return None;
        }
        let mut acc: u128 = 0;
        for (i, &limb) in self.mag.iter().enumerate() {
            acc |= u128::from(limb) << (32 * i);
        }
        if self.negative {
            // -acc must fit in i128 (acc ≤ 2^127).
            if acc > (1u128 << 127) {
                return None;
            }
            Some((acc as i128).wrapping_neg())
        } else {
            if acc > i128::MAX as u128 {
                return None;
            }
            Some(acc as i128)
        }
    }

    fn normalized(mut self) -> Self {
        while self.mag.last() == Some(&0) {
            self.mag.pop();
        }
        if self.mag.is_empty() {
            self.negative = false;
        }
        self
    }

    /// Compares magnitudes (ignoring sign).
    fn abs_cmp(a: &[u32], b: &[u32]) -> Ordering {
        if a.len() != b.len() {
            return a.len().cmp(&b.len());
        }
        for i in (0..a.len()).rev() {
            if a[i] != b[i] {
                return a[i].cmp(&b[i]);
            }
        }
        Ordering::Equal
    }

    /// Magnitude add: `a + b`.
    fn mag_add(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
        let mut carry = 0u64;
        for i in 0..a.len().max(b.len()) {
            let av = u64::from(a.get(i).copied().unwrap_or(0));
            let bv = u64::from(b.get(i).copied().unwrap_or(0));
            let sum = av + bv + carry;
            out.push((sum & 0xFFFF_FFFF) as u32);
            carry = sum >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        out
    }

    /// Magnitude subtract: `a - b`, requiring `a >= b`.
    fn mag_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(a.len());
        let mut borrow = 0i64;
        for (i, &ai) in a.iter().enumerate() {
            let av = i64::from(ai);
            let bv = i64::from(b.get(i).copied().unwrap_or(0));
            let mut diff = av - bv - borrow;
            if diff < 0 {
                diff += 1 << 32;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(diff as u32);
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }

    /// Magnitude multiply (schoolbook).
    fn mag_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = alloc::vec![0u32; a.len() + b.len()];
        for (i, &av) in a.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &bv) in b.iter().enumerate() {
                let cur = u64::from(out[i + j]) + u64::from(av) * u64::from(bv) + carry;
                out[i + j] = (cur & 0xFFFF_FFFF) as u32;
                carry = cur >> 32;
            }
            out[i + b.len()] += carry as u32;
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }

    /// Returns `self + other`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        if self.negative == other.negative {
            Self {
                negative: self.negative,
                mag: Self::mag_add(&self.mag, &other.mag),
            }
            .normalized()
        } else {
            // Different signs → subtract the smaller magnitude from the larger.
            match Self::abs_cmp(&self.mag, &other.mag) {
                Ordering::Equal => Self::zero(),
                Ordering::Greater => Self {
                    negative: self.negative,
                    mag: Self::mag_sub(&self.mag, &other.mag),
                }
                .normalized(),
                Ordering::Less => Self {
                    negative: other.negative,
                    mag: Self::mag_sub(&other.mag, &self.mag),
                }
                .normalized(),
            }
        }
    }

    /// Returns `-self`.
    #[must_use]
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            Self {
                negative: !self.negative,
                mag: self.mag.clone(),
            }
        }
    }

    /// Returns `self - other`.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Returns `self * other`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            negative: self.negative != other.negative,
            mag: Self::mag_mul(&self.mag, &other.mag),
        }
        .normalized()
    }

    /// The bit at position `i` of a magnitude.
    fn mag_bit(m: &[u32], i: usize) -> u32 {
        (m[i / 32] >> (i % 32)) & 1
    }

    /// Magnitude long division (bit-at-a-time): returns `(quotient, remainder)`,
    /// requiring a non-empty divisor.
    fn mag_divmod(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
        if Self::abs_cmp(a, b) == Ordering::Less {
            return (Vec::new(), a.to_vec());
        }
        let bits = a.len() * 32;
        let mut quot = alloc::vec![0u32; a.len()];
        let mut rem: Vec<u32> = Vec::new();
        for i in (0..bits).rev() {
            // rem <<= 1; rem |= bit i of a.
            let mut carry = Self::mag_bit(a, i);
            for limb in &mut rem {
                let new = (u64::from(*limb) << 1) | u64::from(carry);
                *limb = (new & 0xFFFF_FFFF) as u32;
                carry = (new >> 32) as u32;
            }
            if carry != 0 {
                rem.push(carry);
            }
            if Self::abs_cmp(&rem, b) != Ordering::Less {
                rem = Self::mag_sub(&rem, b);
                quot[i / 32] |= 1 << (i % 32);
            }
        }
        while quot.last() == Some(&0) {
            quot.pop();
        }
        (quot, rem)
    }

    /// Returns `(quotient, remainder)` with truncated (toward-zero) division;
    /// the remainder takes the dividend's sign. Returns `None` on divide-by-zero.
    #[must_use]
    pub fn divmod(&self, other: &Self) -> Option<(Self, Self)> {
        if other.is_zero() {
            return None;
        }
        let (q, r) = Self::mag_divmod(&self.mag, &other.mag);
        let quot = Self {
            negative: self.negative != other.negative,
            mag: q,
        }
        .normalized();
        let rem = Self {
            negative: self.negative,
            mag: r,
        }
        .normalized();
        Some((quot, rem))
    }

    /// The two's-complement limb representation over `width` limbs (negative
    /// values are sign-extended with infinite leading 1s, truncated to `width`).
    fn to_twos(&self, width: usize) -> Vec<u32> {
        let mut v = self.mag.clone();
        v.resize(width, 0);
        if self.negative {
            for limb in &mut v {
                *limb = !*limb;
            }
            let mut carry = 1u64;
            for limb in &mut v {
                let s = u64::from(*limb) + carry;
                *limb = s as u32;
                carry = s >> 32;
            }
        }
        v
    }

    /// Reconstructs a signed value from a two's-complement limb vector.
    fn from_twos(mut v: Vec<u32>, negative: bool) -> Self {
        if negative {
            for limb in &mut v {
                *limb = !*limb;
            }
            let mut carry = 1u64;
            for limb in &mut v {
                let s = u64::from(*limb) + carry;
                *limb = s as u32;
                carry = s >> 32;
            }
        }
        Self { negative, mag: v }.normalized()
    }

    /// Applies a limb-wise bitwise op under arbitrary-precision two's-complement
    /// semantics (so negative operands behave as infinite-width).
    fn bit_op(&self, other: &Self, f: impl Fn(u32, u32) -> u32) -> Self {
        let width = self.mag.len().max(other.mag.len()) + 1;
        let a = self.to_twos(width);
        let b = other.to_twos(width);
        let out: Vec<u32> = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
        // The sign of the result is the op applied to each operand's sign bit.
        let negative = f(u32::from(self.negative), u32::from(other.negative)) & 1 == 1;
        Self::from_twos(out, negative)
    }

    /// Bitwise AND (two's-complement).
    #[must_use]
    pub fn bitand(&self, other: &Self) -> Self {
        self.bit_op(other, |a, b| a & b)
    }

    /// Bitwise OR (two's-complement).
    #[must_use]
    pub fn bitor(&self, other: &Self) -> Self {
        self.bit_op(other, |a, b| a | b)
    }

    /// Bitwise XOR (two's-complement).
    #[must_use]
    pub fn bitxor(&self, other: &Self) -> Self {
        self.bit_op(other, |a, b| a ^ b)
    }

    /// The number of bits in the magnitude (0 for zero) — i.e.
    /// `floor(log2(|n|)) + 1`. Used to bound the projected size of a `pow`/shift
    /// result before growing it, so an attacker exponent cannot drive an OOM.
    #[must_use]
    pub fn bit_len(&self) -> u64 {
        match self.mag.last() {
            None => 0,
            Some(&top) => {
                let full = (self.mag.len() as u64 - 1) * 32;
                full + (32 - u64::from(top.leading_zeros()))
            }
        }
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
    pub fn pow(&self, mut exp: u64) -> Self {
        let mut result = Self::from_i128(1);
        let mut base = self.clone();
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

    /// Divides the magnitude by a single small base, returning `(quotient, rem)`
    /// — the workhorse of base conversion.
    fn divmod_small(mag: &[u32], divisor: u32) -> (Vec<u32>, u32) {
        let mut out = alloc::vec![0u32; mag.len()];
        let mut rem = 0u64;
        for i in (0..mag.len()).rev() {
            let cur = (rem << 32) | u64::from(mag[i]);
            out[i] = (cur / u64::from(divisor)) as u32;
            rem = cur % u64::from(divisor);
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        (out, rem as u32)
    }

    /// Renders in `radix` (2..=36), with a leading `-` when negative.
    #[must_use]
    pub fn to_str_radix(&self, radix: u32) -> String {
        debug_assert!((2..=36).contains(&radix));
        if self.is_zero() {
            return String::from("0");
        }
        let mut digits = Vec::new();
        let mut mag = self.mag.clone();
        while !mag.is_empty() {
            let (q, r) = Self::divmod_small(&mag, radix);
            digits.push(core::char::from_digit(r, radix).unwrap());
            mag = q;
        }
        let mut s = String::with_capacity(digits.len() + 1);
        if self.negative {
            s.push('-');
        }
        s.extend(digits.iter().rev());
        s
    }

    /// Parses `s` in `radix` (2..=36); an optional leading `-`/`+` is allowed.
    /// Returns `None` on an invalid digit.
    #[must_use]
    pub fn from_str_radix(s: &str, radix: u32) -> Option<Self> {
        let s = s.trim();
        let (negative, body) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        if body.is_empty() {
            return None;
        }
        let mut acc = Self::zero();
        let base = Self::from_i128(i128::from(radix));
        for ch in body.chars() {
            let d = ch.to_digit(radix)?;
            acc = acc.mul(&base).add(&Self::from_i128(i128::from(d)));
        }
        acc.negative = negative;
        Some(acc.normalized())
    }
}

impl Ord for BigInt {
    /// Signed comparison.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => Self::abs_cmp(&self.mag, &other.mag),
            (true, true) => Self::abs_cmp(&other.mag, &self.mag),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl core::fmt::Display for BigInt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_str_radix(10))
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
}
