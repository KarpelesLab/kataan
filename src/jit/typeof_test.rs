//! **`typeof v === 'lit'` as a type test.** The bytecode spells it
//!
//! ```text
//! TypeOf     t = typeof v
//! StrEqConst d = t === 'lit'     (or !==, ==, !=: `t` is a String either way)
//! JumpIfFalse d, L               (often)
//! ```
//!
//! and `TypeOf` alone is a call (the interpreter's arm builds the name's
//! String). When `t` is read by nothing but the compare, the pair is one
//! test of `v`'s representation: a Number is a word outside the boxed
//! space, a boolean / `undefined` / `null` is one of a few words, and a
//! heap value is never a Number or a boolean. What the word alone cannot
//! decide (a heap value's `"object"` / `"string"` / `"function"`, which
//! depend on its cell, its marker properties, a proxy's target, the
//! `[[IsHTMLDDA]]` brand) runs both ops in the interpreter, as before.
//! A literal no `typeof` returns is never equal.

use crate::nbvm::{Op, STR_EQ_NEGATE};
use alloc::vec;
use alloc::vec::Vec;

/// What `typeof v` is compared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TypeLit {
    Number,
    Boolean,
    Undefined,
    Object,
    Function,
    String,
    /// `"symbol"`, `"bigint"`: only ever a heap value's.
    OtherHeap,
    /// Nothing `typeof` returns.
    Never,
}

impl TypeLit {
    fn of(lit: &[u8]) -> TypeLit {
        match lit {
            b"number" => TypeLit::Number,
            b"boolean" => TypeLit::Boolean,
            b"undefined" => TypeLit::Undefined,
            b"object" => TypeLit::Object,
            b"function" => TypeLit::Function,
            b"string" => TypeLit::String,
            b"symbol" | b"bigint" => TypeLit::OtherHeap,
            _ => TypeLit::Never,
        }
    }
}

/// A fusable `typeof v <op> 'lit'` at a `TypeOf`.
#[derive(Clone, Copy, Debug)]
pub(super) struct TypeofTest {
    /// The tested register.
    pub(super) v: u16,
    /// The compare's boolean register.
    pub(super) d: u16,
    pub(super) lit: TypeLit,
    /// `!==` / `!=`.
    pub(super) neg: bool,
}

/// Per op, the [`TypeofTest`] its `TypeOf` begins: the `StrEqConst` right
/// after reads its result, which nothing else names, and nothing branches
/// to the compare (`target`: per op, whether something branches there or
/// enters there).
pub(super) fn typeof_tests(bytecode: &[Op], target: &[bool]) -> Vec<Option<TypeofTest>> {
    let mut out = vec![None; bytecode.len()];
    for i in 0..bytecode.len().saturating_sub(1) {
        let (
            Op::TypeOf { dst: t, a: v },
            Op::StrEqConst {
                dst: d,
                src,
                lit,
                kind,
            },
        ) = (&bytecode[i], &bytecode[i + 1])
        else {
            continue;
        };
        if src != t || t == v || t == d || target.get(i + 1).copied().unwrap_or(true) {
            continue;
        }
        let named = bytecode
            .iter()
            .filter(|o| crate::bytecode::op_mentions_reg(o, *t))
            .count();
        if named != 2 {
            continue;
        }
        out[i] = Some(TypeofTest {
            v: *v,
            d: *d,
            lit: TypeLit::of(lit),
            neg: kind & STR_EQ_NEGATE != 0,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_classify() {
        assert_eq!(TypeLit::of(b"number"), TypeLit::Number);
        assert_eq!(TypeLit::of(b"symbol"), TypeLit::OtherHeap);
        assert_eq!(TypeLit::of(b"Number"), TypeLit::Never);
        assert_eq!(TypeLit::of(b""), TypeLit::Never);
    }

    #[test]
    fn only_a_private_result_fuses() {
        let ops = vec![
            Op::TypeOf { dst: 1, a: 0 },
            Op::StrEqConst {
                dst: 2,
                src: 1,
                lit: b"number".to_vec(),
                kind: STR_EQ_NEGATE,
            },
            Op::Return { src: 2 },
        ];
        let t = typeof_tests(&ops, &[false; 4]);
        let f = t[0].expect("fuses");
        assert_eq!((f.v, f.d, f.lit, f.neg), (0, 2, TypeLit::Number, true));
        // Something else reads `t`: no.
        let mut ops2 = ops.clone();
        ops2[2] = Op::Return { src: 1 };
        assert!(typeof_tests(&ops2, &[false; 4])[0].is_none());
        // A branch to the compare: no.
        assert!(typeof_tests(&ops, &[false, true, false, false])[0].is_none());
    }
}
