//! Pointer-chasing loops: `while (q.link) q = q.link;`, walking a linked
//! structure to its end. Each hop of the plain code reads the field, tests the
//! value, walks its handle, checks its layout, records it in the receiver
//! cache, and moves it through two temporaries: about 28 instructions.
//!
//! [`plan`] finds the loop; the emitter puts an **accelerator** in front of
//! its first op. While the next object is a plain object of the layout the
//! receiver cache proves (so truthy, and the field read on it is the same
//! read), the accelerator steps `q` to it in about 12 instructions, keeping
//! the handle and the heap slot in registers, and then falls into the ordinary
//! loop, which does the rest: the last hop, the exit, any other shape.
//!
//! Running the accelerator is running iterations of the loop. What those
//! iterations would have changed is `q`, the receiver cache's way (updated on
//! the way out) and two temporaries: the condition's, which the ordinary loop
//! writes first thing, and the moved value's, which [`plan`] requires no later
//! op to read before writing. Nothing else can observe them: the hops run no
//! Rust code, no getter (the layout has a data property there) and allocate
//! nothing. They do not count down the loop's polls; instead at most
//! [`MAX_HOPS`] of them run between two ordinary iterations, which do, so an
//! interrupt still reaches a walk around a cycle, a bounded factor later.

use crate::nbvm::Op;
use alloc::vec::Vec;

/// The most hops between two ordinary iterations of the loop.
pub(super) const MAX_HOPS: u64 = 256;

/// Per op: whether it is the first op of such a loop.
#[must_use]
pub(super) fn plan(ops: &[Op]) -> Vec<bool> {
    let mut at = alloc::vec![false; ops.len()];
    // (A handler could read the temporaries at a throw: nothing here throws,
    // but the liveness below follows no catch edge, so such a body has none.)
    if ops.iter().any(|o| matches!(o, Op::PushHandler { .. })) {
        return at;
    }
    let n = ops.len().saturating_sub(4);
    for (l, x) in at.iter_mut().enumerate().take(n) {
        *x = is_chase(ops, l);
    }
    at
}

/// Whether `ops[l..=l + 4]` is exactly the loop, entered only at `l`, its
/// moved value dead after it.
fn is_chase(ops: &[Op], l: usize) -> bool {
    let (
        Op::GetProp {
            dst: t,
            obj: q,
            key,
        },
        Op::JumpIfFalse { cond, target: exit },
        Op::GetProp {
            dst: u,
            obj: q2,
            key: key2,
        },
        Op::Move { dst: q3, src: u2 },
        Op::Jump { target: back },
    ) = (&ops[l], &ops[l + 1], &ops[l + 2], &ops[l + 3], &ops[l + 4])
    else {
        return false;
    };
    if cond != t || q2 != q || key2 != key || q3 != q || u2 != u || *back != l {
        return false;
    }
    if t == q || u == q || t == u || *exit <= l + 4 {
        return false;
    }
    // Entered only at its first op.
    let inner = l + 1..=l + 4;
    let entered = ops.iter().any(|o| match o {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => inner.contains(target),
        Op::InlineGuard { call, .. } => inner.contains(call),
        _ => false,
    });
    if entered {
        return false;
    }
    !read_before_written(ops, *exit, *u)
}

/// Whether some path from op `from` reads register `r` before writing it.
fn read_before_written(ops: &[Op], from: usize, r: u16) -> bool {
    let mut seen = alloc::vec![false; ops.len() + 1];
    let mut todo = alloc::vec![from];
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    while let Some(pc) = todo.pop() {
        if pc >= ops.len() || core::mem::replace(&mut seen[pc], true) {
            continue;
        }
        let op = &ops[pc];
        if crate::bytecode::op_mentions_reg(op, r) {
            super::numeric::def_use(op, &mut defs, &mut uses);
            if !defs.contains(&r) || uses.contains(&r) {
                return true;
            }
            continue;
        }
        match op {
            Op::Jump { target } => todo.push(*target),
            Op::JumpIfFalse { target, .. } => {
                todo.push(*target);
                todo.push(pc + 1);
            }
            Op::Return { .. }
            | Op::Throw { .. }
            | Op::ThrowTypeError { .. }
            | Op::ThrowReferenceError { .. } => {}
            Op::InlineGuard { call, .. } => {
                todo.push(*call);
                todo.push(pc + 1);
            }
            _ => todo.push(pc + 1),
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn get(dst: u16, obj: u16) -> Op {
        Op::GetProp {
            dst,
            obj,
            key: "link".to_string(),
        }
    }

    fn the_loop() -> Vec<Op> {
        alloc::vec![
            get(1, 0),
            Op::JumpIfFalse { cond: 1, target: 5 },
            get(2, 0),
            Op::Move { dst: 0, src: 2 },
            Op::Jump { target: 0 },
            Op::Return { src: 0 },
        ]
    }

    #[test]
    fn finds_the_loop() {
        assert!(plan(&the_loop())[0]);
    }

    #[test]
    fn refuses_a_moved_value_read_afterwards() {
        let mut ops = the_loop();
        ops[5] = Op::Return { src: 2 };
        assert!(!plan(&ops)[0]);
    }

    #[test]
    fn refuses_a_branch_into_the_loop() {
        let mut ops = the_loop();
        ops.push(Op::Jump { target: 2 });
        assert!(!plan(&ops)[0]);
    }

    #[test]
    fn refuses_another_key_or_a_handler() {
        let mut ops = the_loop();
        ops[2] = Op::GetProp {
            dst: 2,
            obj: 0,
            key: "next".to_string(),
        };
        assert!(!plan(&ops)[0]);
        let mut ops = the_loop();
        ops.push(Op::PushHandler { target: 0, reg: 3 });
        assert!(!plan(&ops)[0]);
    }
}
