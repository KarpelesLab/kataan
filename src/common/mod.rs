//! Cross-cutting building blocks shared by every layer of the engine: source
//! positions ([`Span`]) and, as the engine grows, the string interner, the
//! diagnostic types, and the bump arena.

mod span;

pub use span::Span;

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
