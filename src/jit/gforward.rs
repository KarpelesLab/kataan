//! **Global read forwarding** for the generic tier: a read of a global
//! lexical binding (`let` / `const` / `class` at a script's top level) that
//! an op of the same straight run just wrote or read is a copy of that
//! op's register.
//!
//! A top-level loop over script bindings (`x ^= x << 13; x ^= x >>> 17;`)
//! reads each binding again after every write: the compiler loads the name
//! once per mention. Rewritten as `Move`s on the tier's own copy of the
//! bytecode, those reads drop out of every analysis' way — the types and
//! integer facts flow through the copy, the binding has one read per
//! iteration (which a promoted loop then carries in a register,
//! `numeric::plan_carried_globals`), and the int32 version keeps the value
//! in a GPR from one statement to the next.
//!
//! # Why the copy reads the same value
//!
//! The names are those the body itself initializes as global lexical
//! bindings (`Op::InitGlobal`). Such a binding is a declarative one of the
//! global scope: it is never an accessor, never deleted or made read-only,
//! and no global object property or later script shadows it (a later
//! script's redeclaration is an early error; `globalThis.x = …` makes a
//! property the binding hides). `Op::LoadGlobal` reads the global scope's
//! own binding first, and an `Op::StoreGlobal` of the name either writes
//! it or throws (TDZ, `const`), so right after either op the binding holds
//! its register's value. Between that op and the read, every op is one
//! that cannot run JavaScript or Rust code that writes bindings
//! ([`pure`]): a copy, a constant, arithmetic over operands the type
//! inference proves Numbers, a test, or another such binding's read or
//! write. Nothing between them writes the register, and no join lies in
//! between (a branch target, a handler's entry, an inlined call's
//! fallback), so every path to the read came through the op. A throw in
//! between never reaches the read.
//!
//! The rewrite keeps every op at its pc and writes the same register, so
//! the interpreter — which runs the original bytecode at a deoptimization
//! exit or before an on-stack replacement — sees the same frame.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::nbvm::{Op, Reg};

/// Whether `KATAAN_JIT_GFWD=0` turns the rewrite off (read once per
/// process).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GFWD").is_none_or(|v| v != "0"))
}

/// Whether `KATAAN_JIT_GFWD=log`: report the forwarded reads.
fn log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GFWD").is_some_and(|v| v == "log"))
}

/// The ops between a binding's write or read and a later read of it that
/// keep the binding's value and run no code: their operands are proven
/// Numbers where an object operand could run a conversion (`num`).
fn pure(op: &Op, lex: &[&str], num: &dyn Fn(Reg) -> bool) -> bool {
    match op {
        Op::LoadConst { .. }
        | Op::Move { .. }
        | Op::StrictEq { .. }
        | Op::Not { .. }
        | Op::TypeOf { .. }
        | Op::CheckTdz { .. } => true,
        Op::LoadGlobal { name, .. } | Op::StoreGlobal { name, .. } => lex.contains(&name.as_str()),
        Op::Add { a, b, .. }
        | Op::AddValue { a, b, .. }
        | Op::Sub { a, b, .. }
        | Op::Mul { a, b, .. }
        | Op::Div { a, b, .. }
        | Op::Mod { a, b, .. }
        | Op::Lt { a, b, .. }
        | Op::ValueBin { a, b, .. }
        | Op::Compare { a, b, .. } => num(*a) && num(*b),
        Op::Neg { a, .. } | Op::BitNot { a, .. } => num(*a),
        Op::ToNumeric { src, .. } | Op::Step { src, .. } => num(*src),
        _ => false,
    }
}

/// The rewritten bytecode (see the module docs), or `None` when no read is
/// forwarded. `n_regs` sizes the type inference.
pub(super) fn plan(ops: &[Op], n_regs: usize) -> Option<Vec<Op>> {
    if !enabled() {
        return None;
    }
    let mut lex: Vec<&str> = Vec::new();
    for op in ops {
        if let Op::InitGlobal { name, .. } = op
            && !name.starts_with('\u{1}')
            && !lex.contains(&name.as_str())
        {
            lex.push(name);
        }
    }
    if lex.is_empty()
        || !ops
            .iter()
            .any(|o| matches!(o, Op::LoadGlobal { name, .. } if lex.contains(&name.as_str())))
    {
        return None;
    }
    let n = ops.len();
    let mut join = vec![false; n + 1];
    for op in ops {
        let t = match op {
            Op::Jump { target } | Op::JumpIfFalse { target, .. } => *target,
            Op::PushHandler { target, .. } => *target,
            Op::InlineGuard { call, .. } => *call,
            _ => continue,
        };
        if let Some(j) = join.get_mut(t) {
            *j = true;
        }
    }
    let types = super::numeric::infer(ops, n_regs);
    let mut out: Option<Vec<Op>> = None;
    let mut moved: Vec<usize> = Vec::new();
    // The bindings whose value a register holds: `(name, register)`.
    let mut known: Vec<(String, Reg)> = Vec::new();
    let (mut d, mut u) = (Vec::new(), Vec::new());
    for (pc, op) in ops.iter().enumerate() {
        if join[pc] {
            known.clear();
        }
        match op {
            Op::LoadGlobal { dst, name, .. } if lex.contains(&name.as_str()) => {
                let held = known.iter().find(|k| &k.0 == name).map(|k| k.1);
                if let Some(src) = held {
                    let o = out.get_or_insert_with(|| ops.to_vec());
                    o[pc] = Op::Move { dst: *dst, src };
                    moved.push(pc);
                }
                known.retain(|k| k.1 != *dst && &k.0 != name);
                known.push((name.clone(), *dst));
                continue;
            }
            Op::StoreGlobal { src, name, .. } if lex.contains(&name.as_str()) => {
                known.retain(|k| &k.0 != name);
                known.push((name.clone(), *src));
                continue;
            }
            _ => {}
        }
        let num = |r: Reg| types.get(pc, r) == super::numeric::T_NUM;
        if !pure(op, &lex, &num) {
            known.clear();
            continue;
        }
        super::numeric::def_use(op, &mut d, &mut u);
        known.retain(|k| !d.contains(&k.1));
    }
    if log() && !moved.is_empty() {
        std::eprintln!("[jit] gforward: reads {moved:?}");
    }
    out
}
