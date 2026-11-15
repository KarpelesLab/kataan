//! A string `switch`'s case pick by a compare tree (`KATAAN_JIT_SWITCHTREE`).
//!
//! `Op::SwitchStr` leaves the index of the first matching case (a Number,
//! `-1` for none) in its destination, and the bytecode compiler then picks
//! the case body by a chain of `index === k` tests, one per distinct case
//! (`Compiler::switch_dispatch` below its tree threshold): each a
//! `LoadConst` of `k`, a `StrictEq`, a `JumpIfFalse` to the next test and a
//! `Jump` to the body, the last test's miss a `Jump` to the default. A
//! case near the end of a long chain pays for every test before it.
//!
//! Compiled code goes from the index to the case's jump by a binary search
//! instead: the index is a small non-negative integer Number or `-1`, and
//! the bits of non-negative Numbers order as the Numbers do (as unsigned
//! words; `-1`'s, with the sign bit, above them all), so the search compares
//! the index's bits with each case's, equal bits being exactly `===` for
//! them. The chain's own ops stay compiled (whatever else reaches them runs
//! them as before); this path skips them, which only leaves their
//! temporaries unwritten: [`pick_chain`] takes a chain only when no other op
//! reads one.

use super::call_seq::{cc, r};
use super::{Label, X64Assembler};
use crate::nbvm::Op;
use alloc::vec::Vec;

/// Whether string-switch picks compile to a compare tree
/// (`KATAAN_JIT_SWITCHTREE=0` turns it off; read once per process).
pub(super) fn enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_SWITCHTREE").is_none_or(|v| v != "0"))
}

/// The pick chain after an `Op::SwitchStr` (see the module docs).
pub(super) struct PickChain {
    /// Per test: the case index's bits and the pc of the test's `Jump` to
    /// its body.
    pub(super) picks: Vec<(u64, usize)>,
    /// The pc of the last test's miss `Jump`.
    pub(super) default: usize,
}

/// The fewest tests a chain must have to be searched.
const MIN_PICKS: usize = 3;

/// The chain right after the `Op::SwitchStr` at `pc` of `bytecode`, when it
/// is exactly the compiler's (each test on the switch's destination, its
/// miss going to the next test, every jump forward), and no op outside it
/// reads a register its tests write.
pub(super) fn pick_chain(bytecode: &[Op], pc: usize) -> Option<PickChain> {
    let Some(Op::SwitchStr { dst: idx, .. }) = bytecode.get(pc) else {
        return None;
    };
    let idx = *idx;
    let mut picks = Vec::new();
    let mut temps: Vec<u16> = Vec::new();
    let mut i = pc + 1;
    let default = loop {
        match bytecode.get(i..i + 4) {
            Some(
                [
                    Op::LoadConst { dst: k, value },
                    Op::StrictEq { dst: t, a, b },
                    Op::JumpIfFalse { cond, target: next },
                    Op::Jump { target },
                ],
            ) if *a == idx
                && b == k
                && cond == t
                && *k != idx
                && *t != idx
                && *next == i + 4
                && *target > i + 3
                && value
                    .as_number()
                    .is_some_and(|n| !n.is_sign_negative() && f64::from(n as u32) == n) =>
            {
                picks.push((value.to_bits(), i + 3));
                temps.extend([*k, *t]);
                i += 4;
            }
            _ => match bytecode.get(i) {
                Some(Op::Jump { target }) if *target > i => break i,
                _ => return None,
            },
        }
    };
    if picks.len() < MIN_PICKS {
        return None;
    }
    let (mut defs, mut uses) = (Vec::new(), Vec::new());
    for (j, op) in bytecode.iter().enumerate() {
        if (pc + 1..=default).contains(&j) {
            continue;
        }
        super::numeric::def_use(op, &mut defs, &mut uses);
        if uses.iter().any(|u| temps.contains(u)) {
            return None;
        }
    }
    picks.sort_unstable_by_key(|&(bits, _)| bits);
    Some(PickChain { picks, default })
}

/// The most tests a leaf of the tree makes one by one.
const LEAF: usize = 3;

impl X64Assembler {
    /// To the label of `keys` (sorted by their bits) whose bits are `rax`'s,
    /// else to `miss`. Clobbers `r11`.
    pub(super) fn pick_tree(&mut self, keys: &[(u64, Label)], miss: Label) {
        if keys.len() <= LEAF {
            for &(bits, to) in keys {
                self.g_movabs(r::R11, bits);
                self.g_cmp(r::RAX, r::R11);
                self.g_jcc(cc::E, to);
            }
            self.jmp(miss);
            return;
        }
        let mid = keys.len() / 2;
        let lower = self.new_label();
        self.g_movabs(r::R11, keys[mid].0);
        self.g_cmp(r::RAX, r::R11);
        self.g_jcc(cc::B, lower);
        self.pick_tree(&keys[mid..], miss);
        self.bind(lower);
        self.pick_tree(&keys[..mid], miss);
    }
}
