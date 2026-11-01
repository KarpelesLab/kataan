//! Shortest round-tripping decimal digits of a double, and the ECMAScript
//! `Number::toString(x, 10)` text built from them.
//!
//! The digit search is the algorithm of Ulf Adams, "Ryu: Fast Float-to-String
//! Conversion" (PLDI 2018), written from the paper's description. A positive
//! double is `v = m2 × 2^e2` (exponent shifted by 2, so the bounds below are
//! integers). Every decimal that reads back as `v` lies in the rounding
//! interval `[mm, mp] × 2^e2`, `mm = 4·m2 − 1 − s` (`s = 1` except at the
//! bottom of a binade, where the gap below is half as wide) and
//! `mp = 4·m2 + 2`; its ends belong to it exactly when `m2` is even
//! (round-half-even reads a tie back as the even significand).
//!
//! The three values are multiplied by a power of ten `10^-e10` held to 125
//! bits (the tables in [`super::pow5_tables`]), which the paper shows is
//! exact enough that the integer parts `vm`, `vr`, `vp` are those of the
//! exact products. Removing trailing decimal digits while `vp / 10 > vm / 10`
//! then yields the shortest length; the output is `vr` rounded to nearest on
//! the last removed digit (ties to even when everything removed below it was
//! zero and `vr` was exact), bumped past `vm` when the lower end is excluded.
//! That is ECMAScript's choice: the fewest digits, then the candidate closest
//! to `v`, then the even one (the note to `Number::toString` step 5).

use super::pow5_tables::{POW5, POW5_INV};

/// Bits held by a [`POW5`] / [`POW5_INV`] entry.
const POW5_BITS: i32 = 125;

/// `ceil(log2(5^e))` (`1` for `e = 0`), exact for `0 <= e <= 3528`.
#[inline]
fn pow5_bits(e: i32) -> i32 {
    ((e as u32 * 1_217_359) >> 19) as i32 + 1
}

/// `floor(log10(2^e))`, exact for `0 <= e <= 1650`.
#[inline]
fn log10_pow2(e: i32) -> u32 {
    (e as u32 * 78_913) >> 18
}

/// `floor(log10(5^e))`, exact for `0 <= e <= 2620`.
#[inline]
fn log10_pow5(e: i32) -> u32 {
    (e as u32 * 732_923) >> 20
}

/// Whether `5^p` divides `v` (`v > 0`).
#[inline]
fn multiple_of_pow5(mut v: u64, p: u32) -> bool {
    let mut n = 0;
    while v.is_multiple_of(5) {
        v /= 5;
        n += 1;
        if n >= p {
            return true;
        }
    }
    n >= p
}

/// `floor(m × mul / 2^j)` for a 125-bit table entry `mul` and `64 <= j`.
#[inline]
fn mul_shift(m: u64, mul: (u64, u64), j: i32) -> u64 {
    let lo = u128::from(m) * u128::from(mul.0);
    let hi = u128::from(m) * u128::from(mul.1);
    (((lo >> 64) + hi) >> (j - 64)) as u64
}

/// The shortest decimal `d × 10^e` that reads back as the finite double
/// `abs > 0`, and among those the closest, a tie going to the even `d`:
/// `(d, e)`, `d` with no trailing zeros... except that `d` may end in zeros
/// when they were not removable (callers strip them, see [`digits_of`]).
#[must_use]
pub(crate) fn shortest(abs: f64) -> (u64, i32) {
    let bits = abs.to_bits();
    let mant = bits & ((1u64 << 52) - 1);
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let (e2, m2) = if exp == 0 {
        (1 - 1023 - 52 - 2, mant)
    } else {
        (exp - 1023 - 52 - 2, mant | (1u64 << 52))
    };
    let accept = m2 & 1 == 0;
    let mv = 4 * m2;
    let mm_shift = u64::from(mant != 0 || exp <= 1);
    let (mm, mp) = (mv - 1 - mm_shift, mv + 2);

    let (mut vr, mut vp, mut vm, e10);
    let (mut vm_tz, mut vr_tz) = (false, false);
    if e2 >= 0 {
        // v × 10^-q = mv × 2^(e2-q) / 5^q: multiply by the reciprocal.
        let q = log10_pow2(e2) - u32::from(e2 > 3);
        e10 = q as i32;
        let k = POW5_BITS + pow5_bits(q as i32) - 1;
        let j = -e2 + q as i32 + k;
        let mul = POW5_INV[q as usize];
        vr = mul_shift(mv, mul, j);
        vp = mul_shift(mp, mul, j);
        vm = mul_shift(mm, mul, j);
        // Exactness: only one of mm, mv, mp can be a multiple of 5, and
        // none of a power of five past 5^23 (they are below 2^55).
        if q <= 21 {
            if mv.is_multiple_of(5) {
                vr_tz = multiple_of_pow5(mv, q);
            } else if accept {
                vm_tz = multiple_of_pow5(mm, q);
            } else {
                vp -= u64::from(multiple_of_pow5(mp, q));
            }
        }
    } else {
        // v × 10^-e10 = mv × 5^i / 2^q with i = -e2 - q.
        let q = log10_pow5(-e2) - u32::from(-e2 > 1);
        e10 = q as i32 + e2;
        let i = -e2 - q as i32;
        let k = pow5_bits(i) - POW5_BITS;
        let j = q as i32 - k;
        let mul = POW5[i as usize];
        vr = mul_shift(mv, mul, j);
        vp = mul_shift(mp, mul, j);
        vm = mul_shift(mm, mul, j);
        if q <= 1 {
            // mv has at least two trailing zero bits, so the products
            // are exact for q <= 1.
            vr_tz = true;
            if accept {
                vm_tz = mm_shift == 1;
            } else {
                vp -= 1;
            }
        } else if q < 63 {
            vr_tz = mv & ((1u64 << q) - 1) == 0;
        }
    }

    let mut removed = 0i32;
    let output = if vm_tz || vr_tz {
        // The general case: track whether everything removed was zero.
        let mut last = 0u64;
        loop {
            let (vpd, vmd) = (vp / 10, vm / 10);
            if vpd <= vmd {
                break;
            }
            let vrd = vr / 10;
            vm_tz &= vm - vmd * 10 == 0;
            vr_tz &= last == 0;
            last = vr - vrd * 10;
            (vr, vp, vm) = (vrd, vpd, vmd);
            removed += 1;
        }
        if vm_tz {
            loop {
                let vmd = vm / 10;
                if vm - vmd * 10 != 0 {
                    break;
                }
                let (vpd, vrd) = (vp / 10, vr / 10);
                vr_tz &= last == 0;
                last = vr - vrd * 10;
                (vr, vp, vm) = (vrd, vpd, vmd);
                removed += 1;
            }
        }
        if vr_tz && last == 5 && vr.is_multiple_of(2) {
            last = 4; // an exact tie: round to even
        }
        vr + u64::from((vr == vm && (!accept || !vm_tz)) || last >= 5)
    } else {
        // The common case: nothing is exact, so only the last removed
        // digit decides the rounding. Two digits at a time first.
        let mut round_up = false;
        let (vpd, vmd) = (vp / 100, vm / 100);
        if vpd > vmd {
            let vrd = vr / 100;
            round_up = vr - vrd * 100 >= 50;
            (vr, vp, vm) = (vrd, vpd, vmd);
            removed += 2;
        }
        loop {
            let (vpd, vmd) = (vp / 10, vm / 10);
            if vpd <= vmd {
                break;
            }
            let vrd = vr / 10;
            round_up = vr - vrd * 10 >= 5;
            (vr, vp, vm) = (vrd, vpd, vmd);
            removed += 1;
        }
        vr + u64::from(vr == vm || round_up)
    };
    (output, e10 + removed)
}

/// `"00" "01" … "99"`: the two decimal digits of each value below 100.
pub(crate) const DIGIT_PAIRS: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[2 * i] = b'0' + (i / 10) as u8;
        t[2 * i + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// `10^k` for `k` in `0..20`.
const POW10: [u64; 20] = {
    let mut t = [1u64; 20];
    let mut i = 1;
    while i < 20 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// The number of decimal digits of `v` (`1` for `0`): from its bit length,
/// `floor(bits × log10 2)` (`1233 / 4096` is close enough below 2^64) is
/// that count or one less, which one comparison settles.
#[inline]
fn decimal_len(v: u64) -> usize {
    let bits = 64 - (v | 1).leading_zeros();
    let t = (bits * 1233) >> 12;
    t as usize + usize::from((v | 1) >= POW10[t as usize])
}

/// Writes the decimal digits of `v` into `out`, which must be exactly as
/// long as `v` has digits.
#[inline]
pub(crate) fn write_digits(mut v: u64, out: &mut [u8]) {
    // Eight digits at a time in 32-bit arithmetic, from the end.
    let mut end = out.len();
    while v >= 100_000_000 {
        let lo = (v % 100_000_000) as u32;
        v /= 100_000_000;
        if let Some(o) = out.get_mut(end - 8..end) {
            write_8(lo, o);
        }
        end -= 8;
    }
    write_short(v as u32, &mut out[..end]);
}

/// The two digits of `r < 100` at `o[at..at + 2]`.
#[inline]
fn put_pair(o: &mut [u8], at: usize, r: u32) {
    let r = r as usize * 2;
    o[at] = DIGIT_PAIRS[r];
    o[at + 1] = DIGIT_PAIRS[r + 1];
}

/// `x < 10^8` as exactly eight digits (leading zeros kept) in `o`.
#[inline]
fn write_8(x: u32, o: &mut [u8]) {
    let (hi, lo) = (x / 10_000, x % 10_000);
    put_pair(o, 0, hi / 100);
    put_pair(o, 2, hi % 100);
    put_pair(o, 4, lo / 100);
    put_pair(o, 6, lo % 100);
}

/// The digits of `x` in `o`, which is exactly as long as `x` has digits.
#[inline]
fn write_short(mut x: u32, o: &mut [u8]) {
    let mut pos = o.len();
    while x >= 100 {
        let r = x % 100;
        x /= 100;
        pos -= 2;
        put_pair(o, pos, r);
    }
    if x >= 10 {
        put_pair(o, pos - 2, x);
    } else {
        o[pos - 1] = b'0' + x as u8;
    }
}

/// The shortest digits of the finite `abs > 0` as `(d, k, n)`: `d` with no
/// trailing zeros, `k` its digit count, and `n` with `abs = 0.d × 10^n` (the
/// spec's `s`, `k` and `n`).
#[must_use]
pub(crate) fn digits_of(abs: f64) -> (u64, usize, i32) {
    let (mut d, mut e) = shortest(abs);
    while d >= 10 && d.is_multiple_of(10) {
        d /= 10;
        e += 1;
    }
    let k = decimal_len(d);
    (d, k, e + k as i32)
}

/// Room for any [`write_js_number`] text (at most 25 bytes).
pub(crate) const NUMBER_TEXT_MAX: usize = 32;

/// Writes ECMAScript `Number::toString(x, 10)` into `buf` and returns its
/// length: `NaN`, `±Infinity`, `0` for both zeros, an integer below 2^53 as
/// its digits, and otherwise the shortest round-tripping digits laid out in
/// fixed or exponential notation by the spec's thresholds.
#[inline]
pub(crate) fn write_js_number(x: f64, buf: &mut [u8; NUMBER_TEXT_MAX]) -> usize {
    // The usual number, an integer below 2^32 in magnitude: 32-bit
    // conversion and digits, inline in the caller.
    let a = x.abs();
    if a < 4_294_967_296.0 {
        let u = a as u32;
        if f64::from(u) == a {
            let neg = usize::from(x < 0.0 && u != 0);
            let len = decimal_len(u64::from(u));
            buf[0] = b'-';
            write_short(u, &mut buf[neg..neg + len]);
            return neg + len;
        }
    }
    write_js_number_general(x, buf)
}

/// [`write_js_number`] for every number but an integer below 2^32 in
/// magnitude.
#[inline(never)]
fn write_js_number_general(x: f64, buf: &mut [u8; NUMBER_TEXT_MAX]) -> usize {
    let a = x.abs();
    // (Below 2^32, only a non-integer is left: the shortest digits below.)
    if (4_294_967_296.0..9_007_199_254_740_992.0).contains(&a) {
        let u = a as u64;
        if u as f64 == a {
            let neg = usize::from(x < 0.0 && u != 0);
            let len = decimal_len(u);
            buf[0] = b'-';
            write_digits(u, &mut buf[neg..neg + len]);
            return neg + len;
        }
    } else if !a.is_finite() {
        let s: &[u8] = if x.is_nan() {
            b"NaN"
        } else if x > 0.0 {
            b"Infinity"
        } else {
            b"-Infinity"
        };
        buf[..s.len()].copy_from_slice(s);
        return s.len();
    }
    let (d, k, n) = digits_of(a);
    let mut p = 0;
    if x < 0.0 {
        buf[0] = b'-';
        p = 1;
    }
    let ku = k as i32;
    if ku <= n && n <= 21 {
        // Digits, then n - k zeros.
        write_digits(d, &mut buf[p..p + k]);
        p += k;
        let z = (n - ku) as usize;
        buf[p..p + z].fill(b'0');
        p + z
    } else if 0 < n && n <= 21 {
        // d[..n] '.' d[n..]
        let n = n as usize;
        write_digits(d, &mut buf[p + 1..p + 1 + k]);
        buf.copy_within(p + 1..p + 1 + n, p);
        buf[p + n] = b'.';
        p + k + 1
    } else if -6 < n && n <= 0 {
        // "0." then -n zeros, then the digits.
        let z = (-n) as usize;
        buf[p] = b'0';
        buf[p + 1] = b'.';
        buf[p + 2..p + 2 + z].fill(b'0');
        p += 2 + z;
        write_digits(d, &mut buf[p..p + k]);
        p + k
    } else {
        // d[0] ('.' d[1..])? 'e' ('+' | '-') |n - 1|
        write_digits(d, &mut buf[p + 1..p + 1 + k]);
        buf[p] = buf[p + 1];
        if k > 1 {
            buf[p + 1] = b'.';
            p += k + 1;
        } else {
            p += 1;
        }
        buf[p] = b'e';
        buf[p + 1] = if n > 0 { b'+' } else { b'-' };
        p += 2;
        let e = (n - 1).unsigned_abs() as u64;
        let len = decimal_len(e);
        write_digits(e, &mut buf[p..p + len]);
        p + len
    }
}

#[cfg(test)]
mod tests {
    use super::{NUMBER_TEXT_MAX, digits_of, write_js_number};

    /// The pre-Ryu implementation (exact 128-bit path, then the formatter)
    /// as the oracle.
    fn oracle(x: f64) -> (alloc::vec::Vec<u8>, i32) {
        let (d, e) = super::super::shortest_digits_reference(x);
        (d.as_bytes().to_vec(), e)
    }

    fn check(x: f64) {
        if !(x.is_finite() && x > 0.0) {
            return;
        }
        let (d, k, n) = digits_of(x);
        let mut b = [0u8; 20];
        let s = alloc::format!("{d}");
        b[..k].copy_from_slice(s.as_bytes());
        let (od, oe) = oracle(x);
        assert_eq!((&b[..k], n), (&od[..], oe), "{x:e} bits {:#x}", x.to_bits());
        assert_eq!(text(x), reference_text(x), "{x:e}");
        assert_eq!(text(-x), reference_text(-x), "{x:e}");
    }

    /// `Number::toString(x)` spelled out from the reference digits, step by
    /// step as the spec lays them out.
    fn reference_text(x: f64) -> alloc::string::String {
        let (d, n) = oracle(x.abs());
        let d = alloc::string::String::from_utf8(d).unwrap();
        let k = d.len() as i32;
        let sign = if x < 0.0 { "-" } else { "" };
        let body = if k <= n && n <= 21 {
            alloc::format!("{d}{}", "0".repeat((n - k) as usize))
        } else if 0 < n && n <= 21 {
            alloc::format!("{}.{}", &d[..n as usize], &d[n as usize..])
        } else if -6 < n && n <= 0 {
            alloc::format!("0.{}{d}", "0".repeat((-n) as usize))
        } else {
            let e = n - 1;
            let es = if e >= 0 {
                alloc::format!("+{e}")
            } else {
                alloc::format!("-{}", -e)
            };
            if k == 1 {
                alloc::format!("{d}e{es}")
            } else {
                alloc::format!("{}.{}e{es}", &d[..1], &d[1..])
            }
        };
        alloc::format!("{sign}{body}")
    }

    #[test]
    fn matches_the_reference_digits() {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..300_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            check(f64::from_bits(s >> 1));
            // Short significands (decimal-looking values) and binade edges.
            check(f64::from_bits((s >> 1) & !0xffff_ffff));
            check(f64::from_bits((s >> 1) & !((1 << 52) - 1)));
        }
        for i in 1..5_000u32 {
            for q in [3.0, 7.0, 10.0, 100.0, 1e3, 1e7, 1e21, 1e-300] {
                check(f64::from(i) / q);
                check(f64::from(i) * q);
            }
        }
        // Subnormals, the smallest and largest doubles, powers of two/ten.
        for b in [
            1u64,
            2,
            3,
            0xf_ffff_ffff_ffff,
            0x10_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
        ] {
            check(f64::from_bits(b));
        }
        let mut p = 1.0f64;
        for _ in 0..1074 {
            check(p);
            p /= 2.0;
        }
        let mut p = 1.0f64;
        for _ in 0..1024 {
            check(p);
            check(f64::from_bits(p.to_bits() + 1));
            check(f64::from_bits(p.to_bits() - 1));
            p *= 2.0;
        }
        for e in -323..=308 {
            let x: f64 = alloc::format!("1e{e}").parse().unwrap();
            check(x);
            check(f64::from_bits(x.to_bits() + 1));
            check(f64::from_bits(x.to_bits() - 1));
        }
    }

    /// The long differential run: `KATAAN_RYU_N` (default 10 million)
    /// doubles per thread on 4 threads, drawn as raw bit patterns, short
    /// decimals at every exponent, integers around 2^53 and values around
    /// 1e21, each compared with the reference digits.
    #[cfg(feature = "std")]
    #[test]
    #[ignore = "long: run explicitly"]
    fn matches_the_reference_digits_at_scale() {
        let n: u64 = std::env::var("KATAAN_RYU_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000_000);
        let threads: alloc::vec::Vec<_> = (0..4u64)
            .map(|t| {
                std::thread::spawn(move || {
                    let mut s: u64 =
                        0x2545_F491_4F6C_DD1D ^ (t.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut next = move || {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        s
                    };
                    for i in 0..n {
                        let r = next();
                        let x = match i % 4 {
                            0 => f64::from_bits(r >> 1),
                            1 => {
                                // d × 10^e with 1..=17 digits.
                                let digits = 1 + (r % 17) as u32;
                                let d = next() % 10u64.pow(digits);
                                let e = (next() % 650) as i32 - 340;
                                let t = alloc::format!("{d}e{e}");
                                t.parse::<f64>().unwrap_or(1.0)
                            }
                            2 => {
                                let base = if r & 1 == 0 {
                                    9_007_199_254_740_992.0f64
                                } else {
                                    1e21
                                };
                                let off = (next() % 4096) as i64 - 2048;
                                f64::from_bits((base.to_bits() as i64 + off) as u64)
                            }
                            _ => {
                                // Subnormals and the binade bottoms.
                                if r & 1 == 0 {
                                    f64::from_bits(next() & ((1u64 << 52) - 1))
                                } else {
                                    f64::from_bits(next() & !((1u64 << 52) - 1) & !(1u64 << 63))
                                }
                            }
                        };
                        check(x);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }

    fn text(x: f64) -> alloc::string::String {
        let mut b = [0u8; NUMBER_TEXT_MAX];
        let n = write_js_number(x, &mut b);
        alloc::string::String::from_utf8(b[..n].to_vec()).unwrap()
    }

    #[test]
    fn lays_out_like_number_to_string() {
        for (x, s) in [
            (0.0, "0"),
            (-0.0, "0"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (1.0, "1"),
            (-1.0, "-1"),
            (123.456, "123.456"),
            (-0.5, "-0.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e21, "1e+21"),
            (999_999_999_999_999_900_000.0, "999999999999999900000"),
            (1e20, "100000000000000000000"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (-5e-324, "-5e-324"),
            (1e-6, "0.000001"),
            (1.5e-7, "1.5e-7"),
            (1e-7, "1e-7"),
            (0.000_001_234_5, "0.0000012345"),
            (9_007_199_254_740_992.0, "9007199254740992"),
            (9_007_199_254_740_993.0, "9007199254740992"),
            (1_152_921_504_606_846_976.0, "1152921504606847000"),
            (f64::MAX, "1.7976931348623157e+308"),
            (-f64::MIN_POSITIVE, "-2.2250738585072014e-308"),
            (2f64.powi(-25), "2.9802322387695312e-8"),
            (4_294_967_295.0, "4294967295"),
            (-2_147_483_648.0, "-2147483648"),
            (1.0 / 7.0, "0.14285714285714285"),
        ] {
            assert_eq!(text(x), s, "{x:e}");
        }
    }
}
