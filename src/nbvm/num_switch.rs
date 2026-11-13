//! The case values of a `switch` whose case tests are all Number constants,
//! for its decision-tree dispatch (`Compiler::switch_dispatch`).
//!
//! `switch` compares with strict equality, case by case, until one matches.
//! Against a Number that is: the discriminant is a Number with the same value
//! — `1` matches `1.0`, `0` matches `-0`, NaN matches nothing, and no other
//! type (`"1"`, `1n`, an object) ever matches, so nothing is converted and
//! nothing runs. A constant test has no effect when evaluated, so when every
//! test is one, the dispatch only has to find the first case whose value
//! equals a Number discriminant: a binary search over the sorted distinct
//! values (compiled as `<` and `===` on the discriminant), not a compare per
//! case.
//!
//! Pure, safe `alloc`-only Rust.

use crate::ast::{BinaryOp, Expr, UnaryOp};
use alloc::vec::Vec;

/// The dispatch keys of a switch's case values `tests` (each with its case's
/// index, in case order): each distinct value with the index of the first
/// case that has it (the one a match selects), ascending by value. NaN is left
/// out (it matches nothing) and `-0` becomes `+0` (which it equals).
#[must_use]
pub(super) fn switch_keys(tests: &[(f64, u32)]) -> Vec<(f64, u32)> {
    let mut keys: Vec<(f64, u32)> = tests
        .iter()
        .filter(|(k, _)| !k.is_nan())
        .map(|&(k, i)| (if k == 0.0 { 0.0 } else { k }, i))
        .collect();
    // By value, then by case index, so the first of equal values is the
    // first case; then one entry per value.
    keys.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    keys.dedup_by(|b, a| a.0 == b.0);
    keys
}

/// The value of a case test that is a Number constant: a numeric literal, a
/// sign applied to one, or an arithmetic or bitwise combination of such —
/// expressions whose evaluation has no effect and always gives this Number.
/// `None` for anything else (an identifier such as `NaN` may be shadowed).
#[must_use]
pub(super) fn static_number(e: &Expr) -> Option<f64> {
    match e {
        Expr::Number { value, .. } => Some(*value),
        Expr::Unary { op, argument, .. } => {
            let a = static_number(argument)?;
            match op {
                UnaryOp::Minus => Some(-a),
                UnaryOp::Plus => Some(a),
                UnaryOp::BitNot => Some(f64::from(!to_int32(a))),
                _ => None,
            }
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            let (a, b) = (static_number(left)?, static_number(right)?);
            let shift = |b: f64| to_int32(b) as u32 & 31;
            Some(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::BitAnd => f64::from(to_int32(a) & to_int32(b)),
                BinaryOp::BitOr => f64::from(to_int32(a) | to_int32(b)),
                BinaryOp::BitXor => f64::from(to_int32(a) ^ to_int32(b)),
                BinaryOp::Shl => f64::from(to_int32(a).wrapping_shl(shift(b))),
                BinaryOp::Shr => f64::from(to_int32(a).wrapping_shr(shift(b))),
                BinaryOp::Ushr => f64::from((to_int32(a) as u32).wrapping_shr(shift(b))),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// ECMAScript `ToInt32` of a Number.
fn to_int32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    const TWO32: f64 = 4_294_967_296.0;
    let m = crate::common::FloatExt::trunc(n) % TWO32;
    let m = if m < 0.0 { m + TWO32 } else { m };
    m as u32 as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    #[test]
    fn keys_follow_strict_equality() {
        let k = switch_keys(&[
            (1.0, 0),
            (2.5, 1),
            (-0.0, 2),
            (1.0, 3),
            (f64::NAN, 4),
            (0.0, 5),
            (-7.0, 6),
        ]);
        // The first case of each value, ascending; NaN gone; -0 is 0.
        assert_eq!(k, [(-7.0, 6), (0.0, 2), (1.0, 0), (2.5, 1)]);
        assert!(k[1].0.is_sign_positive());
    }

    #[test]
    fn constant_tests_fold() {
        let span = crate::common::Span::point(0);
        let num = |v: f64| Expr::Number {
            value: v,
            span,
            legacy_octal: false,
        };
        let neg = Expr::Unary {
            op: UnaryOp::Minus,
            argument: Box::new(num(3.0)),
            span,
        };
        assert_eq!(static_number(&neg), Some(-3.0));
        let shl = Expr::Binary {
            op: BinaryOp::Shl,
            left: Box::new(num(1.0)),
            right: Box::new(num(36.0)),
            span,
        };
        assert_eq!(static_number(&shl), Some(16.0));
        let not = Expr::Unary {
            op: UnaryOp::BitNot,
            argument: Box::new(num(4_294_967_296.5)),
            span,
        };
        assert_eq!(static_number(&not), Some(-1.0));
        let s = Expr::Str {
            value: Box::from(&b"1"[..]),
            legacy_octal: false,
            span,
        };
        assert_eq!(static_number(&s), None);
    }
}
