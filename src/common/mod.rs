//! Cross-cutting building blocks shared by every layer of the engine: source
//! positions ([`Span`]) and, as the engine grows, the string interner, the
//! diagnostic types, and the bump arena.

mod span;

pub use span::Span;

/// no_std-safe float operations that mirror the `f64` inherent methods absent
/// from `core` (only these need shimming — `abs`/`signum`/`copysign` are already
/// in `core`). In a `std` build the inherent methods win by name resolution and
/// this trait is inert; import it only under `#[cfg(not(feature = "std"))]` so it
/// supplies `floor`/`trunc`/`fract`/`round`/`powi` for the `--no-default-features`
/// core build (no `std`, no `libm`).
pub trait FloatExt {
    /// Round toward negative infinity.
    fn floor(self) -> f64;
    /// Truncate toward zero.
    fn trunc(self) -> f64;
    /// The fractional part (`self - self.trunc()`).
    fn fract(self) -> f64;
    /// Round half away from zero (matches `std`'s `f64::round`).
    fn round(self) -> f64;
    /// `self` raised to an integer power (exponentiation by squaring).
    fn powi(self, n: i32) -> f64;
}

impl FloatExt for f64 {
    #[inline]
    fn trunc(self) -> f64 {
        if !self.is_finite() || self.abs() >= 9_223_372_036_854_775_808.0 {
            self
        } else {
            self as i64 as f64
        }
    }
    #[inline]
    fn floor(self) -> f64 {
        let t = FloatExt::trunc(self);
        if t > self { t - 1.0 } else { t }
    }
    #[inline]
    fn fract(self) -> f64 {
        self - FloatExt::trunc(self)
    }
    #[inline]
    fn round(self) -> f64 {
        if !self.is_finite() {
            return self;
        }
        let a = FloatExt::floor(self.abs() + 0.5);
        if self < 0.0 { -a } else { a }
    }
    #[inline]
    fn powi(self, n: i32) -> f64 {
        let mut base = if n < 0 { 1.0 / self } else { self };
        let mut exp = n.unsigned_abs();
        let mut acc = 1.0;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }
}

/// `Math.round` with ECMAScript semantics: round half **toward +∞** (so
/// `round(2.5) == 3` but `round(-2.5) == -2`), preserving `-0`, `NaN`, and `±∞`,
/// and returning `-0` for `x ∈ [-0.5, -0)`. This differs from Rust's
/// `f64::round` (half away from zero) and from a naive `(x + 0.5).floor()`
/// (which mis-rounds `0.49999999999999994` up to `1` and loses the sign of zero).
///
/// Needs `std`/`libm` for `floor`; callers gate accordingly.
#[cfg(feature = "std")]
#[must_use]
pub fn js_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if x > 0.0 {
        // `(0, 0.5)` rounds to +0 (this also fixes the `0.4999…` rounding edge).
        if x < 0.5 { 0.0 } else { (x + 0.5).floor() }
    } else if x >= -0.5 {
        -0.0 // `[-0.5, -0)` rounds to -0
    } else {
        (x + 0.5).floor()
    }
}

#[cfg(all(test, feature = "std"))]
mod float_ext_tests {
    use super::FloatExt;
    #[test]
    fn float_ext_matches_std() {
        for &x in &[
            0.0,
            -0.0,
            0.5,
            -0.5,
            2.5,
            -2.5,
            3.7,
            -3.7,
            1e18,
            -1e18,
            0.4999999999999999,
            123.456,
            -123.456,
            9.75,
            -9.75,
        ] {
            assert_eq!(FloatExt::trunc(x), x.trunc(), "trunc {x}");
            assert_eq!(FloatExt::floor(x), x.floor(), "floor {x}");
            assert_eq!(FloatExt::fract(x), x.fract(), "fract {x}");
            assert_eq!(FloatExt::round(x), x.round(), "round {x}");
        }
        for &(b, n) in &[(2.0, 10), (3.0, 0), (5.0, 3), (2.0, -2), (1.5, 4)] {
            assert_eq!(FloatExt::powi(b, n), b.powi(n), "powi {b}^{n}");
        }
    }
}
