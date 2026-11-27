//! **Jump threading on a proven boolean.** An inlined call's value reaches
//! the caller's test through a join:
//!
//! ```text
//!   StrictEq   d = …            (the callee's `return a === b`)
//!   Move       c = d
//!   InlineLeave
//!   Jump       L                (every inlined return)
//!   …          c = <the call>   (the guard's miss: any value)
//! L: JumpIfFalse c, F           (or `Not n = c; JumpIfFalse n, F`)
//! ```
//!
//! At `L`, `c` may hold anything (the miss path's call), so the test there
//! runs the general truthiness sequence (handle floor, immediate class,
//! number) — eight instructions for a boolean. On the `Jump` itself the
//! type facts often prove `c` a boolean: then that `Jump` branches on `c`
//! directly (one byte compare of the boolean's tag) to where the test would
//! send it, past the test. A `Not` in between is folded the same way (its
//! result still stored, flipped from `c`'s bits).
//!
//! Only plain generic code is threaded: no promoted region on either side,
//! no receiver-cache site at the skipped ops (their walks may fill a way),
//! no test fused into a guard, and only forward branches (a back-edge's
//! poll stays where it is). `KATAAN_JIT_BOOLTHREAD=0` turns it off.

use super::GenericOp;

/// Whether threading is on (`KATAAN_JIT_BOOLTHREAD`, default on).
pub(super) fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("KATAAN_JIT_BOOLTHREAD").map_or(true, |v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// A `Jump` threaded past the test at its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Thread {
    /// The boolean register the branch tests.
    pub(super) cond: u16,
    /// With a folded `Not`: its destination, stored as `!cond`.
    pub(super) not_dst: Option<u16>,
    /// Where a `false` `cond` goes.
    pub(super) on_false: usize,
    /// Where a `true` `cond` goes.
    pub(super) on_true: usize,
}

/// The threading of the `Jump` at `pc` to `target`, if any: `plain(q)`
/// says op `q` may be skipped (plain generic code, no receiver-cache site),
/// `is_bool(r)` that register `r` provably holds a boolean at `pc`.
pub(super) fn plan(
    ops: &[GenericOp],
    pc: usize,
    target: usize,
    fused_test: &[bool],
    plain: &dyn Fn(usize) -> bool,
    is_bool: &dyn Fn(u16) -> bool,
) -> Option<Thread> {
    if !enabled() || target <= pc || !plain(pc) {
        return None;
    }
    let test = |q: usize| match ops.get(q) {
        Some(&GenericOp::JumpIfFalse { cond, target: f })
            if !fused_test.get(q).copied().unwrap_or(true)
                && plain(q)
                && f > pc
                && f <= ops.len()
                && q + 1 < ops.len() =>
        {
            Some((cond, f))
        }
        _ => None,
    };
    match ops.get(target) {
        Some(&GenericOp::JumpIfFalse { .. }) => {
            let (cond, f) = test(target)?;
            is_bool(cond).then_some(Thread {
                cond,
                not_dst: None,
                on_false: f,
                on_true: target + 1,
            })
        }
        Some(&GenericOp::Not { dst, a }) if plain(target) && dst != a => {
            let (cond, f) = test(target + 1)?;
            (cond == dst && is_bool(a)).then_some(Thread {
                cond: a,
                not_dst: Some(dst),
                on_false: target + 2,
                on_true: f,
            })
        }
        _ => None,
    }
}
