//! The ECMAScript `Number::toString` digits of a double by exact integer
//! arithmetic, for the doubles whose scaled value fits in 128 bits.
//!
//! A positive finite double is `v = f × 2^e` (`f` the 53-bit significand).
//! Every decimal that converts back to `v` (round to nearest, ties to even)
//! lies in the rounding interval around `v`, whose ends are the midpoints to
//! the neighbouring doubles: `v ± 2^(e-1)`, except that the lower gap is
//! halved when `f` is the smallest significand of its binade. An end belongs
//! to the interval exactly when `f` is even (the tie then rounds to `v`).
//!
//! Scaling everything by `4 × 10^K × 2^-e` (`K ≥ 0` chosen so that
//! `floor(v × 10^K)` has 17 or 18 digits) turns the value and both ends into
//! integers `N`, `L`, `H` over the power-of-two denominator `2^(2-e)`, so their
//! integer parts in units of `10^-K` are exact shifts. Seventeen significant
//! digits always round-trip, so the interval holds at least one integer `c`
//! in those units; the shortest output is the largest power of ten `p` such
//! that some multiple of `p` lies in the interval, and among the (at most two)
//! multiples next to `v` the closer one, an exact tie going to the even
//! digit, which is the spec's choice. Doubles outside the covered range
//! (subnormals, very small or very large magnitudes) return `None`.

use super::Digits;

/// `10^k` for `k` in `0..=21`, as 128-bit integers.
const POW10_128: [u128; 22] = {
    let mut t = [1u128; 22];
    let mut i = 1;
    while i < 22 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// `10^k` for `k` in `0..=19`.
const POW10_64: [u64; 20] = {
    let mut t = [1u64; 20];
    let mut i = 1;
    while i < 20 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// The shortest round-tripping digits of the finite `abs > 0` and the
/// exponent `n` with `abs = 0.digits × 10^n`, or `None` when `abs` is outside
/// the range this computes exactly (see the module docs).
pub(super) fn shortest_digits_exact(abs: f64) -> Option<(Digits, i32)> {
    let bits = abs.to_bits();
    let be = ((bits >> 52) & 0x7ff) as i32;
    if be == 0 || be == 0x7ff {
        return None; // subnormal, zero, or not finite
    }
    let frac = bits & ((1u64 << 52) - 1);
    let f = frac | (1u64 << 52);
    let e = be - 1075;
    // The lower gap is half the upper one at the bottom of a binade (except
    // for the smallest normal binade, whose neighbour below is subnormal and
    // equally spaced).
    let lower_quarter = frac == 0 && be > 1;
    let even = f & 1 == 0;
    let (k, sh, n, lo, hi) = if e >= 0 {
        if e > 10 {
            return None;
        }
        let v4 = u128::from(f) << 2;
        let lo = if lower_quarter { v4 - 1 } else { v4 - 2 };
        (0i32, 2u32, v4 << e, lo << e, (v4 + 2) << e)
    } else {
        // floor(log2 v) = e + 52, and floor(x × log10 2) = (x × 78913) >> 18
        // over this range, so v ≥ 10^d with d = that floor, and K = 16 - d
        // puts floor(v × 10^K) in [10^16, 10^18).
        let d = ((e + 52) * 78913) >> 18;
        let k = 16 - d;
        if !(0..=21).contains(&k) {
            return None;
        }
        let p = POW10_128[k as usize];
        let v4 = u128::from(f) << 2;
        let lo = if lower_quarter { v4 - 1 } else { v4 - 2 };
        (k, (2 - e) as u32, v4 * p, lo * p, (v4 + 2) * p)
    };
    let mask = (1u128 << sh) - 1;
    let whole = u64::try_from(n >> sh).ok()?;
    if e < 0 && !(10_000_000_000_000_000..1_000_000_000_000_000_000).contains(&whole) {
        return None;
    }
    // The smallest and largest integers (units of 10^-K) in the interval.
    let lo_int = u64::try_from(lo >> sh).ok()?;
    let a = if lo & mask == 0 && even {
        lo_int
    } else {
        lo_int + 1
    };
    let hi_int = u64::try_from(hi >> sh).ok()?;
    let b = if hi & mask == 0 && !even {
        hi_int - 1
    } else {
        hi_int
    };
    if a > b {
        return None; // cannot happen in range; leave it to the general path
    }
    // Largest j such that a multiple of 10^j lies in [a, b]:
    // floor(b / 10^j) > floor((a - 1) / 10^j).
    let (mut bj, mut aj, mut vj, mut j) = (b, a - 1, whole, 0usize);
    while j < 19 {
        let (b2, a2) = (bj / 10, aj / 10);
        if b2 <= a2 {
            break;
        }
        (bj, aj, vj) = (b2, a2, vj / 10);
        j += 1;
    }
    // The multiples of 10^j at or below and above v: `vj` and `vj + 1`.
    let p = u128::from(POW10_64[j]);
    let below_ok = vj > aj;
    let above_ok = vj < bj;
    let c = match (below_ok, above_ok) {
        (true, false) => vj,
        (false, true) => vj + 1,
        (true, true) => {
            let dl = n - ((u128::from(vj) * p) << sh);
            let dh = ((u128::from(vj + 1) * p) << sh) - n;
            if dl < dh || (dl == dh && vj & 1 == 0) {
                vj
            } else {
                vj + 1
            }
        }
        (false, false) => return None,
    };
    if c == 0 {
        return None;
    }
    // The decimal digits of `c`, two at a time from the end of `d.buf`
    // (`c < 2^64` has at most 20), then moved to the front without the
    // trailing zeros.
    let mut d = Digits {
        buf: [0; 24],
        len: 0,
    };
    let mut pos = d.buf.len();
    let mut x = c;
    while x >= 100 {
        let r = (x % 100) as usize * 2;
        x /= 100;
        pos -= 2;
        d.buf[pos..pos + 2].copy_from_slice(&DIGIT_PAIRS[r..r + 2]);
    }
    if x >= 10 {
        let r = x as usize * 2;
        pos -= 2;
        d.buf[pos..pos + 2].copy_from_slice(&DIGIT_PAIRS[r..r + 2]);
    } else {
        pos -= 1;
        d.buf[pos] = b'0' + x as u8;
    }
    let len = d.buf.len() - pos;
    let mut end = d.buf.len();
    while end > pos + 1 && d.buf[end - 1] == b'0' {
        end -= 1;
    }
    d.buf.copy_within(pos..end, 0);
    d.len = end - pos;
    Some((d, len as i32 + j as i32 - k))
}

/// `"00" "01" … "99"`: the two decimal digits of each value below 100.
const DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::super::shortest_digits_fmt;
    use super::shortest_digits_exact;

    fn same(x: f64) -> bool {
        match shortest_digits_exact(x) {
            None => true,
            Some((d, e)) => {
                let (g, ge) = shortest_digits_fmt(x);
                d.as_bytes() == g.as_bytes() && e == ge
            }
        }
    }

    #[test]
    fn matches_the_formatter() {
        // A deterministic xorshift over raw bit patterns in the covered range,
        // plus quotients, binade bottoms and short decimals.
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        let mut covered = 0;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let exp = 1075 - 70 + (s >> 56) % 82; // e in [-70, 11]
            let x = f64::from_bits((exp << 52) | (s & ((1 << 52) - 1)));
            assert!(same(x), "{x:e}");
            covered += usize::from(shortest_digits_exact(x).is_some());
        }
        assert!(covered > 150_000, "{covered}");
        for i in 1..20_000u32 {
            for q in [3.0, 7.0, 10.0, 100.0, 1e3, 1e5, 9.0, 11.0] {
                let x = f64::from(i) / q;
                assert!(same(x), "{x:e}");
            }
        }
        let mut x = 1.0f64 / 1_099_511_627_776.0; // 2^-40
        for _ in 0..100 {
            assert!(same(x), "{x:e}");
            assert!(same(x * 1.5), "{x:e}");
            let y = f64::from_bits(x.to_bits() - 1);
            assert!(same(y), "{y:e}");
            let z = f64::from_bits(x.to_bits() + 1);
            assert!(same(z), "{z:e}");
            x *= 2.0;
        }
        for x in [
            0.1,
            0.2,
            0.3,
            1.0 / 3.0,
            123.456,
            5e-5,
            3e-5,
            9e18,
            1e17,
            1.0 / 1024.0,
        ] {
            assert!(same(x), "{x:e}");
            assert!(shortest_digits_exact(x).is_some(), "{x:e}");
        }
    }
}
