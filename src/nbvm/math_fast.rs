//! `Math.f(x, …)` computed in the VM.
//!
//! A hosted `Math.sqrt(d)` is an [`Op::CallMethod`](super::Op::CallMethod) on
//! whatever `Math` currently is. When the receiver's own data property `key` is
//! still one of the built-in `Math` natives (an identity check on the function
//! cell, so `Math.sqrt = f`, a getter, a replaced or proxied `Math` all miss)
//! and every argument is already a Number (or `undefined`), the call has no
//! observable effect besides its result: `ToNumber` of those runs no code. The
//! result is then computed here, with the same arithmetic the built-in uses,
//! instead of a round trip through the host.

use super::{Realm, Reg};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::nbexec::{
    N_MATH_ABS, N_MATH_ATAN, N_MATH_ATAN2, N_MATH_CBRT, N_MATH_CEIL, N_MATH_COS, N_MATH_EXP,
    N_MATH_FLOOR, N_MATH_HYPOT, N_MATH_LOG, N_MATH_MAX, N_MATH_MIN, N_MATH_POW, N_MATH_ROUND,
    N_MATH_SIGN, N_MATH_SIN, N_MATH_SQRT, N_MATH_TAN, N_MATH_TRUNC,
};

/// The native a `Math` member of this name is installed as.
fn native_for(key: &str) -> Option<u16> {
    Some(match key {
        "sqrt" => N_MATH_SQRT,
        "abs" => N_MATH_ABS,
        "floor" => N_MATH_FLOOR,
        "ceil" => N_MATH_CEIL,
        "round" => N_MATH_ROUND,
        "trunc" => N_MATH_TRUNC,
        "sign" => N_MATH_SIGN,
        "min" => N_MATH_MIN,
        "max" => N_MATH_MAX,
        "pow" => N_MATH_POW,
        "hypot" => N_MATH_HYPOT,
        "cbrt" => N_MATH_CBRT,
        "exp" => N_MATH_EXP,
        "log" => N_MATH_LOG,
        "sin" => N_MATH_SIN,
        "cos" => N_MATH_COS,
        "tan" => N_MATH_TAN,
        "atan" => N_MATH_ATAN,
        "atan2" => N_MATH_ATAN2,
        _ => return None,
    })
}

/// `recv[key](...args)` when it is a built-in `Math` function over Numbers;
/// `None` to take the ordinary call path.
pub(super) fn call(
    realm: &Realm,
    recv: NanBox,
    key: &str,
    args: &[Reg],
    regs: &[NanBox],
) -> Option<NanBox> {
    let id = native_for(key)?;
    let h = Handle::from_raw(recv.as_handle()?);
    if !realm.is_ordinary_get_cell(h) || realm.accessor(h, key).is_some() {
        return None;
    }
    let f = realm.get_property(h, key)?.as_handle()?;
    if realm.native_at(Handle::from_raw(f)) != Some(id) {
        return None;
    }
    compute(realm, id, args, regs)
}

/// Whether `id` is a built-in `Math` function [`compute`] serves.
pub(super) fn is_math_native(id: u16) -> bool {
    matches!(
        id,
        N_MATH_SQRT
            | N_MATH_ABS
            | N_MATH_FLOOR
            | N_MATH_CEIL
            | N_MATH_ROUND
            | N_MATH_TRUNC
            | N_MATH_SIGN
            | N_MATH_MIN
            | N_MATH_MAX
            | N_MATH_POW
            | N_MATH_HYPOT
            | N_MATH_CBRT
            | N_MATH_EXP
            | N_MATH_LOG
            | N_MATH_SIN
            | N_MATH_COS
            | N_MATH_TAN
            | N_MATH_ATAN
            | N_MATH_ATAN2
    )
}

/// The built-in `Math` function `id` called on the argument registers `args`
/// of `regs`, when every argument is a Number or `undefined` (so `ToNumber`
/// runs no code); `None` to take the ordinary call path.
pub(super) fn compute(realm: &Realm, id: u16, args: &[Reg], regs: &[NanBox]) -> Option<NanBox> {
    // Every argument must convert without running code.
    let mut nums = [0.0f64; 4];
    if args.len() > nums.len() {
        return None;
    }
    for (slot, r) in nums.iter_mut().zip(args) {
        let v = regs[*r as usize];
        *slot = match v.as_number() {
            Some(n) => n,
            None if v.is_undefined() => f64::NAN,
            None => return None,
        };
    }
    let given = &nums[..args.len()];
    let arg = |i: usize| given.get(i).copied().unwrap_or(f64::NAN);
    let r = match id {
        N_MATH_SQRT => arg(0).sqrt(),
        N_MATH_ABS => arg(0).abs(),
        N_MATH_FLOOR => arg(0).floor(),
        N_MATH_CEIL => arg(0).ceil(),
        N_MATH_TRUNC => arg(0).trunc(),
        N_MATH_ROUND => {
            let n = arg(0);
            // A magnitude ≥ 2^52 is already an integer.
            if n.abs() >= 4_503_599_627_370_496.0 || !n.is_finite() {
                n
            } else {
                crate::common::js_round(n)
            }
        }
        N_MATH_SIGN => {
            let n = arg(0);
            if n.is_nan() {
                f64::NAN
            } else if n > 0.0 {
                1.0
            } else if n < 0.0 {
                -1.0
            } else {
                n
            }
        }
        N_MATH_MAX => {
            let mut m = f64::NEG_INFINITY;
            for &n in given {
                if n.is_nan() {
                    m = f64::NAN;
                } else if !m.is_nan() && (n > m || (n == 0.0 && m == 0.0 && n.is_sign_positive())) {
                    m = n;
                }
            }
            m
        }
        N_MATH_MIN => {
            let mut m = f64::INFINITY;
            for &n in given {
                if n.is_nan() {
                    m = f64::NAN;
                } else if !m.is_nan() && (n < m || (n == 0.0 && m == 0.0 && n.is_sign_negative())) {
                    m = n;
                }
            }
            m
        }
        N_MATH_POW => return Some(realm.pow(NanBox::number(arg(0)), NanBox::number(arg(1)))),
        N_MATH_HYPOT => {
            let (mut any_inf, mut any_nan, mut sum) = (false, false, 0.0);
            for &n in given {
                if n.is_infinite() {
                    any_inf = true;
                } else if n.is_nan() {
                    any_nan = true;
                } else {
                    sum += n * n;
                }
            }
            if any_inf {
                f64::INFINITY
            } else if any_nan {
                f64::NAN
            } else {
                sum.sqrt()
            }
        }
        N_MATH_CBRT => arg(0).cbrt(),
        N_MATH_EXP => arg(0).exp(),
        N_MATH_LOG => arg(0).ln(),
        N_MATH_SIN => arg(0).sin(),
        N_MATH_COS => arg(0).cos(),
        N_MATH_TAN => arg(0).tan(),
        N_MATH_ATAN => arg(0).atan(),
        N_MATH_ATAN2 => arg(0).atan2(arg(1)),
        _ => return None,
    };
    Some(NanBox::number(r))
}
