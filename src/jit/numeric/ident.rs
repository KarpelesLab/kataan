//! **Identity-compared registers** for `===` (`ROADMAP.md` §2.1).
//!
//! `x === y` is a compare of the two NaN-boxed words exactly when neither
//! value is a Number (`NaN !== NaN`, `0 === -0`), a String or a BigInt (equal
//! by contents): every other value — `undefined`, `null`, a boolean, a
//! Symbol, any object — is `===` only to itself, one word. So when one
//! operand's register can never hold any of those three kinds, the words
//! decide, whatever the other operand holds.
//!
//! A register qualifies when every op of the (spliced) body that names it
//! either only reads it, or writes it a fresh object (an object or array
//! literal, `new`, a function), a qualifying register's value (a move), or
//! `undefined` / `null` / a boolean; and nothing writes it outside the ops
//! (a parameter, a capture or `this`, which the call fills, never does). A
//! register starts as `undefined`, and every other way a value reaches it —
//! the interpreter running the same ops, an on-stack entry, a deoptimization
//! exit writing back what the compiled code held — goes through those ops.
//! Any op the list below does not know, naming the register in any role,
//! disqualifies it.

use crate::nanbox::NanBox;
use crate::nbvm::Op;
use alloc::vec::Vec;

/// Per register (`0..n_regs`): whether it never holds a Number, a String or
/// a BigInt (see the module docs). `first_local`: the first register no
/// call fills (past the parameters, the captures and `this`).
pub(super) fn identity_regs(bytecode: &[Op], n_regs: usize, first_local: usize) -> Vec<bool> {
    let mut ok: Vec<bool> = (0..n_regs).map(|r| r >= first_local).collect();
    // (No register of a body that reaches its scope dynamically.)
    if bytecode
        .iter()
        .any(|op| matches!(op, Op::DirectEval { .. } | Op::Env { .. }))
    {
        return alloc::vec![false; n_regs];
    }
    let imm = |v: NanBox| {
        let b = v.to_bits();
        b == NanBox::undefined().to_bits()
            || b == NanBox::null().to_bits()
            || b == NanBox::boolean(false).to_bits()
            || b == NanBox::boolean(true).to_bits()
    };
    let mut regs: Vec<u16> = Vec::new();
    // A fixpoint (a move's source may be disqualified later).
    loop {
        let mut changed = false;
        for op in bytecode {
            regs.clear();
            regs.extend(super::op_regs(op));
            regs.sort_unstable();
            regs.dedup();
            for &r in &regs {
                let i = r as usize;
                if !ok.get(i).copied().unwrap_or(false) {
                    continue;
                }
                let fine = fresh_def(op, r, &ok, &imm) || only_reads(op, r);
                if !fine {
                    ok[i] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            return ok;
        }
    }
}

/// Whether `op` writes register `r` a fresh object, a qualifying register's
/// value or an immediate `imm` accepts (reading `r` nowhere else).
fn fresh_def(op: &Op, r: u16, ok: &[bool], imm: &dyn Fn(NanBox) -> bool) -> bool {
    let q = |s: u16| ok.get(s as usize).copied().unwrap_or(false);
    match op {
        Op::NewObject { dst } | Op::NewArray { dst, .. } | Op::LoadFunc { dst, .. } => *dst == r,
        Op::NewArrayCtor { dst, arg } => *dst == r && *arg != r,
        Op::NewObjectLit { dst, lit } => *dst == r && !lit.srcs.contains(&r),
        Op::NewArrayFrom { dst, elems } => *dst == r && !elems.contains(&r),
        Op::MakeClosure { dst, captures, .. } => *dst == r && !captures.contains(&r),
        Op::Construct { dst, ctor, args } => *dst == r && *ctor != r && !args.contains(&r),
        Op::Move { dst, src } => *dst == r && (*src == r || q(*src)),
        Op::LoadConst { dst, value } => *dst == r && imm(*value),
        _ => false,
    }
}

/// Whether `op` only reads register `r` (writes no register it names as `r`).
fn only_reads(op: &Op, r: u16) -> bool {
    match op {
        Op::StrictEq { dst, .. }
        | Op::GetProp { dst, .. }
        | Op::GetMethod { dst, .. }
        | Op::GetKey { dst, .. }
        | Op::ArrayLen { dst, .. }
        | Op::NewObjectLit { dst, .. }
        | Op::NewArrayFrom { dst, .. }
        | Op::MakeClosure { dst, .. }
        | Op::Construct { dst, .. }
        | Op::CallMethod { dst, .. }
        | Op::CallValue { dst, .. }
        | Op::CallWithMethod { dst, .. } => *dst != r,
        Op::Move { dst, src } => *src == r && *dst != r,
        Op::SetProp { .. }
        | Op::SetPropStrict { .. }
        | Op::SetKey { .. }
        | Op::SetKeyStrict { .. }
        | Op::JumpIfFalse { .. }
        | Op::Return { .. }
        | Op::CheckTdz { .. }
        | Op::CellSet { .. }
        | Op::StoreGlobal { .. } => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nbvm::Op;

    #[test]
    fn fresh_objects_qualify_and_anything_else_disqualifies() {
        let ops = alloc::vec![
            Op::NewObject { dst: 2 },
            Op::Move { dst: 3, src: 2 },
            Op::LoadConst {
                dst: 4,
                value: NanBox::number(1.0),
            },
            Op::Move { dst: 5, src: 4 },
            Op::NewObject { dst: 6 },
            Op::Add { dst: 6, a: 6, b: 4 },
            Op::StrictEq { dst: 7, a: 3, b: 2 },
            Op::LoadConst {
                dst: 8,
                value: NanBox::null(),
            },
            Op::NewObject { dst: 0 },
        ];
        let ok = identity_regs(&ops, 9, 1);
        assert!(ok[2] && ok[3] && ok[8], "objects, a move of one, null");
        assert!(!ok[4] && !ok[5], "a Number, a move of one");
        assert!(!ok[6], "an object later written by an add");
        assert!(!ok[7], "a boolean result of an op not in the list");
        assert!(!ok[0], "a register the call fills");
    }
}
