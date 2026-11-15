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

/// The case whose body the dispatch to each case lands on: the first case at
/// or after it with a statement in its body (cases with empty bodies share the
/// entry of the next one, so `case 1: case 2: f();` is one target), else the
/// last case (whose entry is the end of the bodies).
#[must_use]
pub(super) fn landing_cases(cases: &[crate::ast::SwitchCase]) -> Vec<u32> {
    let mut out = alloc::vec![0u32; cases.len()];
    let mut next = cases.len().saturating_sub(1) as u32;
    for (i, case) in cases.iter().enumerate().rev() {
        if !case.body.is_empty() {
            next = i as u32;
        }
        out[i] = next;
    }
    out
}

/// The longest span of integer case values [`dense_index`] covers with a
/// direct table (in entries, and relative to the number of values).
const DENSE_MAX_SPAN: usize = 1 << 16;
const DENSE_MAX_SPREAD: usize = 4;

/// `Op::SwitchNum`'s direct table, when every key of `table` (ascending, no
/// NaN, no `-0`) is an integer and they span at most [`DENSE_MAX_SPREAD`]
/// times as many integers as there are keys (and at most
/// [`DENSE_MAX_SPAN`]): entry `x - table[0].0` holds the pick of `x`,
/// `u32::MAX` where no key is. Empty otherwise (the lookup then searches).
#[must_use]
pub(crate) fn dense_index(table: &[(f64, u32)]) -> alloc::boxed::Box<[u32]> {
    let (Some(first), Some(last)) = (table.first(), table.last()) else {
        return alloc::boxed::Box::default();
    };
    let integral = |k: f64| {
        k > -9.007_199_254_740_992e15 && k < 9.007_199_254_740_992e15 && (k as i64) as f64 == k
    };
    if !table.iter().all(|&(k, _)| integral(k)) {
        return alloc::boxed::Box::default();
    }
    let span = (last.0 as i64 - first.0 as i64) as u64 + 1;
    let limit = (table.len().saturating_mul(DENSE_MAX_SPREAD)).min(DENSE_MAX_SPAN) as u64;
    if span > limit {
        return alloc::boxed::Box::default();
    }
    let mut dense = alloc::vec![u32::MAX; span as usize];
    for &(k, pick) in table {
        dense[(k as i64 - first.0 as i64) as usize] = pick;
    }
    dense.into_boxed_slice()
}

/// `Op::SwitchNum`'s answer: the pick of the entry of `table` (ascending by
/// key, no NaN, no `-0`) whose key `v` strictly equals, else `-1`. Only a
/// Number equals one, and `-0` the `+0` key; NaN equals none. `dense` is
/// [`dense_index`]`(table)`.
#[must_use]
pub(crate) fn lookup(table: &[(f64, u32)], dense: &[u32], v: crate::nanbox::NanBox) -> f64 {
    let Some(x) = v.as_number() else {
        return -1.0;
    };
    let Some(first) = table.first() else {
        return -1.0;
    };
    if !dense.is_empty() {
        // (NaN fails both bounds; `-0 - 0` is `-0`, which is `>= 0` and
        // indexes entry 0 like `+0`; a fraction is not its truncation.)
        let d = x - first.0;
        if d >= 0.0 && d < dense.len() as f64 {
            let i = d as usize;
            if i as f64 == d && dense[i] != u32::MAX {
                return f64::from(dense[i]);
            }
        }
        return -1.0;
    }
    // The last key `<= x`, by halving without a data-dependent branch (the
    // keys a random discriminant meets are unpredictable); NaN is `<=` none
    // and ends on the first key, which it does not equal.
    let (mut base, mut n) = (0usize, table.len());
    while n > 1 {
        let half = n / 2;
        if table[base + half].0 <= x {
            base += half;
        }
        n -= half;
    }
    if table[base].0 == x {
        f64::from(table[base].1)
    } else {
        -1.0
    }
}

/// Whether a long Number-constant `switch` (and a string one) finds its case
/// with one lookup op whose answer is the case its body lands on
/// (`Op::SwitchNum`, `landing_cases`), dispatching on that: unless
/// `KATAAN_JIT_SWITCHNUM=0`. Read once per process.
pub(super) fn lookup_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SWITCHNUM").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

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

    /// The direct table and the search answer alike: by strict equality,
    /// `-0` as `+0`, nothing for NaN, a fraction or a non-Number.
    #[test]
    fn lookup_is_strict_equality() {
        use crate::nanbox::NanBox;
        let close = switch_keys(&[(-3.0, 0), (0.0, 1), (1.0, 2), (5.0, 3)]);
        let far = switch_keys(&[(-3.0, 0), (0.0, 1), (1.0, 2), (5e9, 3), (0.5, 4)]);
        assert_eq!(dense_index(&close).len(), 9);
        assert!(dense_index(&far).is_empty());
        for t in [&close, &far] {
            let d = dense_index(t);
            let at = |v: NanBox| lookup(t, &d, v);
            assert_eq!(at(NanBox::number(-3.0)), 0.0);
            assert_eq!(at(NanBox::number(-0.0)), 1.0);
            assert_eq!(at(NanBox::number(0.0)), 1.0);
            assert_eq!(at(NanBox::number(1.0)), 2.0);
            for miss in [-4.0, -2.0, 2.0, 0.25, 1.5, 6.0, f64::NAN, f64::INFINITY] {
                assert_eq!(at(NanBox::number(miss)), -1.0, "{miss}");
            }
            assert_eq!(at(NanBox::undefined()), -1.0);
            assert_eq!(at(NanBox::boolean(true)), -1.0);
        }
        let d = dense_index(&far);
        assert_eq!(lookup(&far, &d, NanBox::number(5e9)), 3.0);
        assert_eq!(lookup(&far, &d, NanBox::number(0.5)), 4.0);
        assert_eq!(lookup(&[], &[], NanBox::number(0.0)), -1.0);
    }
}
