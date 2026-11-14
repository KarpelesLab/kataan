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
use alloc::vec::Vec;
use core::cmp::Ordering;
use puremp::Int;

/// Whether the engine's own fast paths over the backend run (a remainder by
/// one limb, bitwise ops of non-negatives; `KATAAN_BIGINT_FAST=0` turns them
/// off, read once per process).
#[cfg(feature = "std")]
fn fast_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_BIGINT_FAST").is_none_or(|v| v != "0"))
}

#[cfg(not(feature = "std"))]
fn fast_enabled() -> bool {
    true
}

/// The largest operand (in 64-bit limbs) the engine's own multiplication and
/// division take: values of a few hundred bits, the size of a modular
/// exponentiation's or a hash's, where the backend's general dispatch and
/// its scratch allocations cost more than the arithmetic. (A dividend may
/// have twice as many: a product of two such values.)
const SMALL_LIMBS: usize = 8;

/// The sign of a non-zero result that is negative when `neg`.
fn sign_of(neg: bool) -> puremp::Sign {
    if neg {
        puremp::Sign::Negative
    } else {
        puremp::Sign::Positive
    }
}

/// The schoolbook product of two magnitudes (little-endian limbs), as
/// `a.len() + b.len()` limbs (a top zero limb included when there is one).
fn mul_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = alloc::vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let (row, rest) = out[i..].split_at_mut(b.len());
        let mut carry = 0u64;
        for (o, &y) in row.iter_mut().zip(b) {
            // x*y + out + carry < 2^128: (2^64-1)^2 + 2(2^64-1) = 2^128 - 1.
            let t = u128::from(x) * u128::from(y) + u128::from(*o) + u128::from(carry);
            *o = t as u64;
            carry = (t >> 64) as u64;
        }
        rest[0] = carry;
    }
    out
}

/// `(u1·2^64 + u0) / d` and its remainder, for a normalized `d` (top bit
/// set) with reciprocal `v = ⌊(2^128 − 1) / d⌋ − 2^64` and `u1 < d` (so the
/// quotient fits a limb), by two multiplications ("Improved division by
/// invariant integers", Möller & Granlund, 2011, Algorithm 4).
#[inline]
fn div_2by1(u1: u64, u0: u64, d: u64, v: u64) -> (u64, u64) {
    let q = (u128::from(v) * u128::from(u1)).wrapping_add((u128::from(u1) << 64) | u128::from(u0));
    let mut q1 = ((q >> 64) as u64).wrapping_add(1);
    let q0 = q as u64;
    let mut r = u0.wrapping_sub(q1.wrapping_mul(d));
    if r > q0 {
        q1 = q1.wrapping_sub(1);
        r = r.wrapping_add(d);
    }
    if r >= d {
        q1 = q1.wrapping_add(1);
        r -= d;
    }
    (q1, r)
}

/// The magnitude `a` modulo the non-zero limb `d`: a pass from the top limb
/// down, scaled so the divisor is normalized (the remainder then scaled back).
fn rem_1(a: &[u64], d: u64) -> u64 {
    debug_assert!(d != 0);
    let s = d.leading_zeros();
    let dn = d << s;
    let v = (u128::MAX / u128::from(dn) - (1u128 << 64)) as u64;
    let Some(&top) = a.last() else {
        return 0;
    };
    let mut rem = if s == 0 { 0 } else { top >> (64 - s) };
    for i in (0..a.len()).rev() {
        let lo = if s == 0 || i == 0 {
            0
        } else {
            a[i - 1] >> (64 - s)
        };
        rem = div_2by1(rem, (a[i] << s) | lo, dn, v).1;
    }
    rem >> s
}

/// Knuth's long division (TAOCP vol. 2, §4.3.1, Algorithm D) of the
/// magnitude `a` by the magnitude `b` (2..=[`SMALL_LIMBS`] limbs, top limb
/// non-zero; `a` at most twice that), in place on stack buffers: the
/// quotient's limbs (only when `want_q`, else empty) and the remainder's.
fn div_limbs(a: &[u64], b: &[u64], want_q: bool) -> (Vec<u64>, Vec<u64>) {
    let n = b.len();
    debug_assert!((2..=SMALL_LIMBS).contains(&n) && b[n - 1] != 0);
    debug_assert!(a.len() <= SMALL_LIMBS * 2);
    if a.len() < n {
        return (Vec::new(), a.to_vec());
    }
    let m = a.len() - n;
    // Normalize: shift both left until the divisor's top bit is set; the
    // dividend gains a top limb for the bits shifted out.
    let s = b[n - 1].leading_zeros();
    let shl = |x: &[u64], out: &mut [u64]| {
        let mut prev = 0u64;
        for (o, &v) in out.iter_mut().zip(x) {
            *o = if s == 0 {
                v
            } else {
                (v << s) | (prev >> (64 - s))
            };
            prev = v;
        }
        if s != 0 { prev >> (64 - s) } else { 0 }
    };
    let mut d = [0u64; SMALL_LIMBS];
    shl(b, &mut d[..n]);
    let mut u = [0u64; SMALL_LIMBS * 2 + 1];
    u[a.len()] = shl(a, &mut u[..a.len()]);
    let mut q = [0u64; SMALL_LIMBS * 2 + 1];
    let (d1, d2) = (u128::from(d[n - 1]), u128::from(d[n - 2]));
    const B: u128 = 1 << 64;
    // The reciprocal of the normalized top limb, for two-by-one divisions by
    // multiplication (Möller–Granlund): one 128-bit division per call.
    let v = (u128::MAX / d1 - B) as u64;
    for j in (0..=m).rev() {
        // Estimate the quotient digit from the top two limbs (at most B - 1:
        // the top limb never exceeds the divisor's), then correct it with
        // the next one: it is then exact or one too large.
        let (mut qhat, mut rhat) = if u[j + n] >= d[n - 1] {
            (u64::MAX, u128::from(u[j + n - 1]) + d1)
        } else {
            let (q, r) = div_2by1(u[j + n], u[j + n - 1], d[n - 1], v);
            (q, u128::from(r))
        };
        while rhat < B && u128::from(qhat) * d2 > ((rhat << 64) | u128::from(u[j + n - 2])) {
            qhat -= 1;
            rhat += d1;
        }
        // u[j..=j+n] -= qhat * d. The running carry also takes the borrow:
        // qhat * d[i] + carry <= 2^128 - 2^64, so its high limb stays below
        // 2^64 - 1 and one more fits.
        let (window, top) = u[j..=j + n].split_at_mut(n);
        let mut carry = 0u64;
        for (ui, &di) in window.iter_mut().zip(&d[..n]) {
            let p = u128::from(qhat) * u128::from(di) + u128::from(carry);
            let (t, b) = ui.overflowing_sub(p as u64);
            *ui = t;
            carry = (p >> 64) as u64 + u64::from(b);
        }
        let (t, b) = top[0].overflowing_sub(carry);
        top[0] = t;
        if b {
            // One too large: add the divisor back.
            qhat -= 1;
            let mut c = 0u64;
            for (ui, &di) in window.iter_mut().zip(&d[..n]) {
                let (t, c1) = ui.overflowing_add(di);
                let (t, c2) = t.overflowing_add(c);
                *ui = t;
                c = u64::from(c1 | c2);
            }
            top[0] = top[0].wrapping_add(c);
        }
        q[j] = qhat;
    }
    // The remainder is u[..n], shifted back.
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        let hi = if s == 0 || i + 1 == n {
            0
        } else {
            u[i + 1] << (64 - s)
        };
        r.push(if s == 0 { u[i] } else { (u[i] >> s) | hi });
    }
    let q = if want_q { q[..=m].to_vec() } else { Vec::new() };
    (q, r)
}

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
        let (a, b) = (self.0.limbs(), other.0.limbs());
        if fast_enabled() && a.len() >= 2 && b.len() >= 2 && a.len() + b.len() <= SMALL_LIMBS * 2 {
            let neg = self.0.is_negative() != other.0.is_negative();
            return Self(Int::from_limb_vec(sign_of(neg), mul_limbs(a, b)));
        }
        Self(self.0.mul(&other.0))
    }

    /// Returns `(quotient, remainder)` with truncated (toward-zero) division;
    /// the remainder takes the dividend's sign. Returns `None` on divide-by-zero.
    #[must_use]
    pub fn divmod(&self, other: &Self) -> Option<(Self, Self)> {
        if let Some((q, r)) = self.small_div(other, true) {
            let qneg = self.0.is_negative() != other.0.is_negative();
            return Some((
                Self(Int::from_limb_vec(sign_of(qneg), q)),
                Self(Int::from_limb_vec(sign_of(self.0.is_negative()), r)),
            ));
        }
        // `div_rem` is truncated (toward-zero): the remainder takes the
        // dividend's sign, matching BigInt `/` and `%`. `None` on divide-by-zero.
        self.0.div_rem(&other.0).map(|(q, r)| (Self(q), Self(r)))
    }

    /// The magnitudes' quotient (empty unless `want_q`) and remainder by
    /// [`div_limbs`], when the divisor has 2..=[`SMALL_LIMBS`] limbs and the
    /// dividend at most twice that (and the fast paths are on); `None` for
    /// anything else, the backend's division then deciding.
    fn small_div(&self, other: &Self, want_q: bool) -> Option<(Vec<u64>, Vec<u64>)> {
        let (a, b) = (self.0.limbs(), other.0.limbs());
        (fast_enabled() && (2..=SMALL_LIMBS).contains(&b.len()) && a.len() <= SMALL_LIMBS * 2)
            .then(|| div_limbs(a, b, want_q))
    }

    /// `self % other` (truncated: the remainder takes the dividend's sign),
    /// `None` on divide-by-zero. A divisor of one limb is a Horner pass over
    /// the dividend's limbs, with no quotient built.
    #[must_use]
    pub fn rem(&self, other: &Self) -> Option<Self> {
        if fast_enabled()
            && let [d] = other.0.limbs()
        {
            // A zero remainder is the (unsigned) zero.
            let r = Int::from_u64(rem_1(self.0.limbs(), *d));
            return Some(Self(if self.0.is_negative() { r.neg() } else { r }));
        }
        if let Some((_, r)) = self.small_div(other, false) {
            return Some(Self(Int::from_limb_vec(sign_of(self.0.is_negative()), r)));
        }
        self.divmod(other).map(|(_, r)| r)
    }

    /// Bitwise AND (two's-complement).
    #[must_use]
    pub fn bitand(&self, other: &Self) -> Self {
        if fast_enabled() {
            let (a, b) = (&self.0, &other.0);
            match (a.is_negative(), b.is_negative()) {
                // Two non-negatives: the limbs ANDed, as long as the shorter.
                (false, false) => {
                    let v: alloc::vec::Vec<u64> = a
                        .limbs()
                        .iter()
                        .zip(b.limbs())
                        .map(|(x, y)| x & y)
                        .collect();
                    return Self(Int::from_limb_vec(puremp::Sign::Positive, v));
                }
                // A non-negative of one limb (or zero) and a negative: the
                // negative's low two's-complement limb (`-m mod 2^64`) masked.
                (false, true) | (true, false) => {
                    let (p, n) = if b.is_negative() { (a, b) } else { (b, a) };
                    if p.limbs().len() <= 1 {
                        let low = n.least_significant_limb().wrapping_neg();
                        return Self(Int::from_u64(low & p.least_significant_limb()));
                    }
                }
                (true, true) => {}
            }
        }
        Self(self.0.bitand(&other.0))
    }

    /// Bitwise OR (two's-complement).
    #[must_use]
    pub fn bitor(&self, other: &Self) -> Self {
        if let Some(v) = self.nonneg_limbwise(other, |x, y| x | y) {
            return v;
        }
        Self(self.0.bitor(&other.0))
    }

    /// Bitwise XOR (two's-complement).
    #[must_use]
    pub fn bitxor(&self, other: &Self) -> Self {
        if let Some(v) = self.nonneg_limbwise(other, |x, y| x ^ y) {
            return v;
        }
        Self(self.0.bitxor(&other.0))
    }

    /// `op` limb by limb over two non-negatives, the shorter one zero-extended
    /// (an OR or XOR, for which `op(x, 0) == x`); `None` when either operand
    /// is negative.
    fn nonneg_limbwise(&self, other: &Self, op: impl Fn(u64, u64) -> u64) -> Option<Self> {
        let (a, b) = (&self.0, &other.0);
        if !fast_enabled() || a.is_negative() || b.is_negative() {
            return None;
        }
        let (long, short) = if a.limbs().len() >= b.limbs().len() {
            (a.limbs(), b.limbs())
        } else {
            (b.limbs(), a.limbs())
        };
        let v: alloc::vec::Vec<u64> = long
            .iter()
            .enumerate()
            .map(|(i, &x)| op(x, short.get(i).copied().unwrap_or(0)))
            .collect();
        Some(Self(Int::from_limb_vec(puremp::Sign::Positive, v)))
    }

    /// Returns `self * 2^k` (a left shift by `k` bits).
    #[must_use]
    pub fn shl_bits(&self, k: u32) -> Self {
        Self(self.0.mul_2k(u64::from(k)))
    }

    /// Returns `floor(self / 2^k)` (an arithmetic right shift by `k` bits:
    /// a negative value with any of its low `k` bits set rounds down).
    #[must_use]
    pub fn shr_floor_bits(&self, k: u32) -> Self {
        let a = self.0.limbs();
        if fast_enabled() && a.len() >= 2 && (k as usize) < a.len() * 64 {
            // The magnitude shifted right by whole limbs then bits; a negative
            // value that loses a set bit rounds away from zero (floor).
            let (w, s) = ((k / 64) as usize, k % 64);
            let mut lost = a[..w].iter().any(|&l| l != 0) || (s != 0 && a[w] << (64 - s) != 0);
            let mut v: Vec<u64> = (w..a.len())
                .map(|i| {
                    let hi = if s == 0 || i + 1 == a.len() {
                        0
                    } else {
                        a[i + 1] << (64 - s)
                    };
                    (a[i] >> s) | hi
                })
                .collect();
            let neg = self.0.is_negative();
            if neg && lost {
                for l in &mut v {
                    let (t, c) = l.overflowing_add(1);
                    *l = t;
                    lost = c;
                    if !c {
                        break;
                    }
                }
                if lost {
                    v.push(1);
                }
            }
            return Self(Int::from_limb_vec(sign_of(neg), v));
        }
        let q = self.0.div_2k_trunc(u64::from(k));
        if self.0.is_negative() && self.0.trailing_zeros() < k {
            Self(q.sub(&Int::from(1i32)))
        } else {
            Self(q)
        }
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
            return Self(self.0.pow(u64::from(e)));
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
        Int::from_str_radix(s.trim_matches(crate::unicode::is_white_space), radix)
            .ok()
            .map(Self)
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

    /// The engine's own small-operand paths (multiplication, Knuth division,
    /// one-limb remainder, shifts, bitwise ops of non-negatives) agree with
    /// the backend's general algorithms on random and edge-pattern operands
    /// of every sign.
    #[test]
    fn fast_paths_match_backend() {
        use alloc::vec::Vec;
        use puremp::{Int, Sign};
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let gen_limb = |rnd: &mut dyn FnMut() -> u64| match rnd() % 6 {
            0 => 0,
            1 => u64::MAX,
            2 => 1u64 << (rnd() % 64),
            3 => u64::MAX >> (rnd() % 64),
            _ => rnd(),
        };
        for _ in 0..20000 {
            let make = |rnd: &mut dyn FnMut() -> u64| {
                let len = (rnd() % 18) as usize;
                let limbs: Vec<u64> = (0..len).map(|_| gen_limb(rnd)).collect();
                let sign = if rnd() % 2 == 0 {
                    Sign::Positive
                } else {
                    Sign::Negative
                };
                BigInt(Int::from_limbs(sign, &limbs))
            };
            let x = make(&mut rnd);
            let y = make(&mut rnd);
            assert_eq!(x.mul(&y).0, x.0.mul(&y.0), "{x} * {y}");
            if !y.is_zero() {
                let (q, r) = x.0.div_rem(&y.0).unwrap();
                let (fq, fr) = x.divmod(&y).unwrap();
                assert_eq!((fq.0, fr.0), (q, r.clone()), "{x} divmod {y}");
                assert_eq!(x.rem(&y).unwrap().0, r, "{x} % {y}");
            }
            assert_eq!(x.bitand(&y).0, x.0.bitand(&y.0), "{x} & {y}");
            assert_eq!(x.bitor(&y).0, x.0.bitor(&y.0), "{x} | {y}");
            assert_eq!(x.bitxor(&y).0, x.0.bitxor(&y.0), "{x} ^ {y}");
            let k = (rnd() % 1200) as u32;
            let q = x.0.div_2k_trunc(u64::from(k));
            let want = if x.0.is_negative() && x.0.trailing_zeros() < k {
                q.sub(&Int::from(1i32))
            } else {
                q
            };
            assert_eq!(x.shr_floor_bits(k).0, want, "{x} >> {k}");
        }
    }

    /// The reciprocal two-by-one division agrees with `u128` division.
    #[test]
    fn div_2by1_matches_u128() {
        let mut st = 0x2545_f491_4f6c_dd1du64;
        for i in 0..200_000u64 {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let d = (st | (1 << 63)) >> (i % 3 == 0) as u32 | (1 << 63);
            let u1 = if i % 5 == 0 {
                d - 1
            } else {
                st.rotate_left(17) % d
            };
            let u0 = if i % 7 == 0 {
                u64::MAX
            } else {
                st.rotate_left(31)
            };
            let v = (u128::MAX / u128::from(d) - (1u128 << 64)) as u64;
            let n = (u128::from(u1) << 64) | u128::from(u0);
            let want = ((n / u128::from(d)) as u64, (n % u128::from(d)) as u64);
            assert_eq!(super::div_2by1(u1, u0, d, v), want, "{u1:x}:{u0:x} / {d:x}");
        }
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
