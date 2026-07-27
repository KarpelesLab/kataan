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

/// Sine of an angle **in degrees**, to within ~1e-13 absolute.
///
/// The astronomical calendar algorithms (see
/// [`temporal_astro`](crate::nbexec::temporal_astro)) evaluate hundreds of
/// `sin`/`cos` terms whose arguments are degree-valued polynomials, and `core`
/// has no transcendental functions — so the `--no-default-features` build cannot
/// use `f64::sin`. This is a self-contained replacement: reduce modulo 360°,
/// fold into the first quadrant by symmetry, then evaluate a Taylor series on
/// `[0, π/2]` (through `x^17/17!`, whose truncation error there is ~1e-13).
#[must_use]
pub fn sin_deg(degrees: f64) -> f64 {
    if !degrees.is_finite() {
        return f64::NAN;
    }
    // Reduce to [0, 360). `f64::rem_euclid` is std-only, so use the floor form
    // (`FloatExt::floor` is the no_std shim and resolves in both builds).
    let d = degrees - 360.0 * FloatExt::floor(degrees / 360.0);
    // sin(180 + x) = -sin(x): fold the lower half-plane onto the upper.
    let (sign, d) = if d > 180.0 {
        (-1.0, d - 180.0)
    } else {
        (1.0, d)
    };
    // sin(180 - x) = sin(x): fold the second quadrant onto the first.
    let d = if d > 90.0 { 180.0 - d } else { d };
    sign * sin_series(d * (core::f64::consts::PI / 180.0))
}

/// Cosine of an angle **in degrees**. `cos(x) = sin(x + 90°)`, which reuses
/// [`sin_deg`]'s reduction exactly rather than repeating it.
#[must_use]
pub fn cos_deg(degrees: f64) -> f64 {
    sin_deg(degrees + 90.0)
}

/// Taylor series for `sin` on `[0, π/2]`, evaluated in Horner form over
/// `u = x²`. The coefficient of `u^k` is `(-1)^k / (2k+1)!`; truncating after
/// `u^8` (i.e. `x^17/17!`) leaves ~1e-13 of error at `x = π/2`.
fn sin_series(x: f64) -> f64 {
    let u = x * x;
    let mut acc = 1.0 / 355_687_428_096_000.0; // u^8 : +1/17!
    for c in [
        -1.0 / 1_307_674_368_000.0, // u^7 : -1/15!
        1.0 / 6_227_020_800.0,      // u^6 : +1/13!
        -1.0 / 39_916_800.0,        // u^5 : -1/11!
        1.0 / 362_880.0,            // u^4 : +1/9!
        -1.0 / 5_040.0,             // u^3 : -1/7!
        1.0 / 120.0,                // u^2 : +1/5!
        -1.0 / 6.0,                 // u^1 : -1/3!
        1.0,                        // u^0 : +1
    ] {
        acc = acc * u + c;
    }
    x * acc
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

#[cfg(test)]
mod trig_tests {
    use super::{cos_deg, sin_deg};

    /// Agreement with `std`'s `f64::sin`/`cos` across the full reduction range,
    /// including the arguments the astronomical algorithms actually produce
    /// (degree polynomials reaching ~1e8 for extreme years).
    #[test]
    fn sin_cos_deg_match_std() {
        let mut worst = 0.0f64;
        let mut d = -1440.0;
        while d <= 1440.0 {
            let r = d * core::f64::consts::PI / 180.0;
            worst = worst.max((sin_deg(d) - r.sin()).abs());
            worst = worst.max((cos_deg(d) - r.cos()).abs());
            d += 0.25;
        }
        assert!(worst < 1e-12, "worst |Δ| over ±1440° was {worst}");
        // Exact quadrant values.
        for (deg, want) in [(0.0, 0.0), (90.0, 1.0), (180.0, 0.0), (270.0, -1.0)] {
            assert!((sin_deg(deg) - want).abs() < 1e-12, "sin({deg})");
        }
        for (deg, want) in [(0.0, 1.0), (90.0, 0.0), (180.0, -1.0), (270.0, 0.0)] {
            assert!((cos_deg(deg) - want).abs() < 1e-12, "cos({deg})");
        }
        // Large arguments still reduce correctly (mod 360 is exact enough here).
        for d in [36_000.769_537_44, 1.0e6, 9.7e7, -4.2e7] {
            let expected = (d % 360.0) * core::f64::consts::PI / 180.0;
            assert!(
                (sin_deg(d) - expected.sin()).abs() < 1e-9,
                "sin({d}) reduction"
            );
        }
        assert!(sin_deg(f64::NAN).is_nan());
        assert!(sin_deg(f64::INFINITY).is_nan());
    }
}
