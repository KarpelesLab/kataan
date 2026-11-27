//! An `arguments` object that is never made: scalar replacement of
//! `arguments.length` and `arguments[i]` in the generic tier.
//!
//! A body whose only uses of its `arguments` object (`Op::MakeArguments`,
//! its first op) are `arguments.length` (`Op::ArrayLen`) and
//! `arguments[k]` (`Op::GetKey`) — of its register or of copies of it, each
//! written once — never lets the object escape: nothing else reads them, so nobody can tell whether the object exists until
//! one of those reads would see more than the arguments themselves. Its
//! native-call entry (`call_seq::emit_native_entry`) then stores the
//! argument count, a Number, in the register instead of an object (when the
//! arguments lie right below the window, as the native-call header lays
//! them out; else it makes the object as usual). The two reads test the
//! register first:
//!
//! * a Number is the count: `length` is that count (the object's own,
//!   still-writable `length`, which nothing could have written), and an
//!   index `i` below it is argument `i`, read from below the window (the
//!   object's own data property, which nothing could have redefined; an
//!   unmapped one, or a mapped one with no parameter to alias: a body with
//!   `Op::MapArguments` is not taken);
//! * any other index — a string key, an index past the count (which the
//!   prototype chain answers, and whose getter would see the object as its
//!   receiver), `-1`, a fraction — first makes the object exactly as the
//!   entry would have (from the arguments still below the window and the
//!   header's callee) into the register and every copy (a copy made later
//!   copies it), then reads it as usual; so does every later read, which
//!   finds an object there.
//!
//! An activation the interpreter starts (its register file filled by
//! `call_with_inner`) has the object in the register from the start, and the
//! reads take their ordinary paths. The body must not be able to hand its
//! frame to the interpreter mid-run with a count in the register: no
//! deoptimization exits (the caller checks), no loop (a back-edge may leave
//! for the interpreter), no direct `eval`; and none of the reads may lie in
//! a promoted loop region, whose emitter owns its ops.

use super::GenericOp;
use crate::nbvm::Op;
use alloc::vec::Vec;

/// The plan: the `arguments` register and the reads of it.
pub(crate) struct ScalarArgs {
    /// `Op::MakeArguments`'s register.
    pub reg: u16,
    /// It and its copies (each written once, by an `Op::Move` from another).
    pub regs: Vec<u16>,
    /// Whether the object is a mapped one.
    pub mapped: bool,
    /// The pcs of the `length` and index reads.
    pub uses: Vec<usize>,
}

impl ScalarArgs {
    /// The plan for a body (`ops`, lowered from `bytecode` op for op), when
    /// its `arguments` object can go unmade (see the module docs);
    /// `in_region` tells whether a pc lies in a promoted loop region.
    pub(crate) fn plan(
        ops: &[GenericOp],
        bytecode: &[Op],
        in_region: impl Fn(usize) -> bool,
    ) -> Option<Self> {
        if !enabled() || ops.len() != bytecode.len() {
            return None;
        }
        let Some(&Op::MakeArguments { dst: reg, mapped }) = bytecode.first() else {
            return None;
        };
        let backward = ops.iter().enumerate().any(|(i, o)| match *o {
            GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. } => target <= i,
            _ => false,
        });
        if backward {
            return None;
        }
        // The register and its copies (`Op::Move`s of it: the binding a
        // body's `arguments` reads go through).
        let mut regs = alloc::vec![reg];
        loop {
            let before = regs.len();
            for op in bytecode {
                if let Op::Move { dst, src } = *op
                    && regs.contains(&src)
                    && !regs.contains(&dst)
                {
                    regs.push(dst);
                }
            }
            if regs.len() == before {
                break;
            }
        }
        let mut uses = Vec::new();
        let mut copies = alloc::vec![0usize; regs.len()];
        for (pc, op) in bytecode.iter().enumerate().skip(1) {
            if matches!(op, Op::DirectEval { .. } | Op::MapArguments { .. }) {
                return None;
            }
            if !regs
                .iter()
                .any(|&r| crate::bytecode::op_mentions_reg(op, r))
            {
                continue;
            }
            if in_region(pc) {
                return None;
            }
            let ours = |r: u16| regs.contains(&r);
            match (op, &ops[pc]) {
                // A copy, the only write of its destination (counted below).
                (Op::Move { dst, src }, _) if ours(*dst) && ours(*src) && *dst != reg => {
                    if let Some(i) = regs.iter().position(|&r| r == *dst) {
                        copies[i] += 1;
                    }
                }
                (Op::ArrayLen { dst, arr }, GenericOp::ArrayLen { .. })
                    if ours(*arr) && !ours(*dst) =>
                {
                    uses.push(pc);
                }
                (Op::GetKey { dst, obj, key }, GenericOp::GetElem { .. })
                    if ours(*obj) && !ours(*key) && !ours(*dst) =>
                {
                    uses.push(pc);
                }
                _ => return None,
            }
        }
        if copies.iter().skip(1).any(|&n| n != 1) {
            return None;
        }
        if uses.is_empty() {
            return None;
        }
        Some(Self {
            reg,
            regs,
            mapped,
            uses,
        })
    }
}

/// Whether `arguments` objects may go unmade (`KATAAN_JIT_SCALAR_ARGS=0`
/// turns it off, read once per process).
fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SCALAR_ARGS").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}
