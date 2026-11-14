//! Which compiled `g.next(v)` sites may hand back one reused iterator result
//! (`jit_tier::jit_helper_gen_call_reuse`, `CallSite::iter_reuse`) instead
//! of a fresh object per call.
//!
//! A site qualifies when no result it returns can ever be told apart from a
//! fresh one: on every path from the call, while any register holding the
//! result is still live, the ops are only
//!
//! - `Move` (a copy of the result joins the set; an overwritten register
//!   leaves it), `LoadConst`, `Not`, `Jump`, `JumpIfFalse` (`ToBoolean` of
//!   anything runs no code), `StrictEq` of registers not holding it, and
//! - `GetProp` of `done` or `value` from a register holding it — own data
//!   properties of the result's layout, read with no code run and no throw.
//!
//! So the result never reaches anything that could keep it, compare it or
//! give it a property, and no code runs while it is live — in particular no
//! other run of the same site (this activation's or another's), which would
//! store the next values into it. Liveness is the JIT's own backward
//! analysis (`numeric::liveness`, conservative for any op it does not model:
//! "reads everything it names"); a path that runs past `MAX_STEPS` ops with
//! the result still live, or off the end of the body, disqualifies the site.
//! An exit to the interpreter inside such a region runs the same ops on the
//! same registers, with the same outcome.

use crate::nbvm::Op;
use alloc::vec::Vec;

/// The most ops one site's paths may run while its result is live.
const MAX_STEPS: usize = 48;

/// Whether reused iterator results are enabled (`KATAAN_JIT_ITERREUSE=0`
/// turns them off, read once per process).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_ITERREUSE").is_none_or(|v| v != "0"))
}

/// For each op of `bytecode`: whether it is a `g.next(v)` method call whose
/// iterator results never escape (see the module docs). `catches` are the
/// body's catch entries (as `numeric::liveness` takes them).
pub(super) fn plan(bytecode: &[Op], n_regs: usize, catches: &[usize]) -> Vec<bool> {
    let mut out = alloc::vec![false; bytecode.len()];
    let is_site = |op: &Op| matches!(op, Op::CallMethod { key, args, .. } if key == "next" && args.len() <= 1);
    if !enabled() || !bytecode.iter().any(is_site) {
        return out;
    }
    let Some((words, live)) = super::numeric::liveness(bytecode, n_regs, catches) else {
        return out;
    };
    let live_in = |pc: usize, r: u16| {
        (r as usize) >= n_regs
            || live
                .get(pc * words + r as usize / 64)
                .is_none_or(|w| w & (1u64 << (r % 64)) != 0)
    };
    for (p, op) in bytecode.iter().enumerate() {
        if let Op::CallMethod { dst, .. } = op
            && is_site(op)
        {
            out[p] = no_escape(bytecode, p, *dst, &live_in);
        }
    }
    out
}

/// Whether the result the call at `p` writes to `dst` never escapes.
fn no_escape(bytecode: &[Op], p: usize, dst: u16, live_in: &dyn Fn(usize, u16) -> bool) -> bool {
    let mut work: Vec<(usize, Vec<u16>)> = alloc::vec![(p + 1, alloc::vec![dst])];
    let mut steps = 0;
    while let Some((pc, mut held)) = work.pop() {
        held.retain(|&r| live_in(pc, r));
        if held.is_empty() {
            continue;
        }
        steps += 1;
        if steps > MAX_STEPS {
            return false;
        }
        let Some(op) = bytecode.get(pc) else {
            return false;
        };
        let next = match op {
            Op::Move { dst, src } => {
                if held.contains(src) {
                    if !held.contains(dst) {
                        held.push(*dst);
                    }
                } else {
                    held.retain(|r| r != dst);
                }
                pc + 1
            }
            Op::LoadConst { dst, .. } | Op::Not { dst, .. } => {
                held.retain(|r| r != dst);
                pc + 1
            }
            Op::StrictEq { dst, a, b } if !held.contains(a) && !held.contains(b) => {
                held.retain(|r| r != dst);
                pc + 1
            }
            Op::GetProp { dst, obj, key }
                if held.contains(obj) && (key == "done" || key == "value") =>
            {
                held.retain(|r| r != dst);
                pc + 1
            }
            Op::JumpIfFalse { target, .. } => {
                work.push((*target, held.clone()));
                pc + 1
            }
            Op::Jump { target } => *target,
            _ => return false,
        };
        work.push((next, held));
    }
    true
}
