//! The generic (NanBox) tier on AArch64: a **baseline** body over the frame
//! ABI, `extern "C" fn(ctx, regs, entry) -> u64` (see
//! [`JitFunction::compile_generic`](super::JitFunction::compile_generic) for
//! the contract the interpreter relies on), compiled straight from the
//! function's bytecode.
//!
//! What it lowers natively: constants and moves, the branches (with the
//! back-edge poll on every backward one), `return`, and the Number fast
//! paths of `+ - * /`, `<` and the relational compares and `===` (both
//! operands Numbers: the FP instruction, its NaN made the canonical one).
//! Every other op — and every slow path — is the interpreter's own,
//! run on the same register file by `jit_helper_exec_op`, so the body has
//! the interpreter's semantics by construction. `try` regions keep their
//! handlers on the activation's handler stack in the context (as the x86-64
//! body does): a throw anywhere asks `jit_helper_unwind` for the catch pc
//! and continues there. Ops whose meaning is tied
//! to the interpreter's frame (suspensions, proper tail calls,
//! inlined activations, direct `eval`) make the function ineligible:
//! [`eligible`] says so before anything is emitted. A generator's or async
//! function's suspensions leave the body through `jit_helper_suspend`, and
//! its resumptions re-enter at the pcs after them ([`resume_pcs`]), as on
//! x86-64.
//!
//! Registers: `x19` = `ctx`, `x20` = the register file, `x21` = the poll
//! countdown, `x22` = the NanBox `QNAN` mask (all callee-saved, so they
//! survive the helpers); `x9..x12`, `d0`/`d1` are scratch. The native frame
//! holds no values (everything lives in the register file the collector
//! roots), so a collection in any helper is safe at any point.

// Only an aarch64 build runs this tier (the emitter is built and tested
// everywhere).
#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use super::Label;
use super::a64::{A64Assembler, Cond, FP, LR, SP};
use crate::nanbox::NanBox;
use crate::nbvm::Op;
use alloc::vec::Vec;

/// The runtime helpers the baseline body calls, as code addresses.
#[derive(Clone, Copy)]
pub(crate) struct Helpers {
    /// `jit_helper_exec_op(ctx, regs, n_regs, op) -> 0 | throw`.
    pub exec_op: u64,
    /// `jit_helper_truthy(ctx, v) -> 0 | 1`.
    pub truthy: u64,
    /// `jit_helper_backedge(ctx, regs, n_regs, ops, n_ops, header) ->
    /// countdown | throw | exit`.
    pub backedge: u64,
    /// `jit_helper_push_handler(ctx, catch pc, reg) -> 0 | throw`.
    pub push_handler: u64,
    /// `jit_helper_pop_handler(ctx, _) -> 0`.
    pub pop_handler: u64,
    /// `jit_helper_unwind(ctx, regs, n_regs) -> catch pc | NO_HANDLER`.
    pub unwind: u64,
    /// `jit_helper_suspend(ctx, regs, n_regs, op, pc) -> 0 | throw | exit`.
    pub suspend: u64,
}

/// `ctx`.
const CTX: u8 = 19;
/// The register file.
const REGS: u8 = 20;
/// The back-edge poll countdown.
const COUNT: u8 = 21;
/// `QNAN` (a word is a Number iff `bits & QNAN != QNAN`).
const QNAN_R: u8 = 22;

/// The NanBox `QNAN` mask (`nanbox::QNAN`).
const QNAN: u64 = 0x7ffc_0000_0000_0000;
/// The canonical Number NaN (`nanbox::CANONICAL_NAN`): AArch64's default
/// NaN is the positive one, so every FP result that is a NaN is stored as
/// this instead.
const CANONICAL_NAN: u64 = 0xfff8_0000_0000_0000;

/// Why `ops` cannot run in the baseline body, or `None` when they can.
#[must_use]
pub(crate) fn refusal(ops: &[Op], is_async: bool, is_generator: bool) -> Option<&'static str> {
    for op in ops {
        let why = match op {
            // The suspensions the body leaves at (`suspends`).
            op if suspends(op, is_async, is_generator) => continue,
            Op::Yield { .. }
            | Op::Await { .. }
            | Op::GeneratorStart
            | Op::YieldDelegate { .. }
            | Op::YieldResume { .. }
            | Op::AsyncDelegateCall { .. }
            | Op::AddAsyncDisposable { .. }
            | Op::AsyncDisposeStep { .. } => "suspension",
            Op::TailCall { .. }
            | Op::TailCallValue { .. }
            | Op::TailCallThis { .. }
            | Op::TailCallGuard { .. }
            | Op::TailSpreadGuard { .. }
            | Op::TailMethodGuard { .. } => "tail call",
            Op::InlineGuard { .. } | Op::InlineLeave => "inlined activation",
            Op::DirectEval { .. } | Op::Env { .. } => "direct eval / environment",
            Op::Jump { target }
            | Op::JumpIfFalse { target, .. }
            | Op::PushHandler { target, .. }
                if *target > ops.len() =>
            {
                "branch out of range"
            }
            _ => continue,
        };
        return Some(why);
    }
    None
}

/// Whether `op` is a suspension the body of a function of this kind leaves
/// at, through `jit_helper_suspend` (as the x86-64 body does): a sync
/// generator's start, `yield` and `yield*` steps; an async function's
/// `await`; an async generator's start, `yield` and `await`.
fn suspends(op: &Op, is_async: bool, is_generator: bool) -> bool {
    match op {
        Op::GeneratorStart | Op::YieldResume { .. } => is_generator,
        Op::YieldDelegate { .. } => is_generator && !is_async,
        Op::Await { .. } => is_async,
        _ => false,
    }
}

/// Where a generator's or an async function's body is re-entered on
/// resumption: after each start, `yield` and `await`, and at each sync
/// `yield*` step (which runs again).
#[must_use]
pub(crate) fn resume_pcs(ops: &[Op], is_async: bool, is_generator: bool) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(pc, op)| match op {
            Op::YieldDelegate { .. } if suspends(op, is_async, is_generator) => Some(pc),
            op if suspends(op, is_async, is_generator) => Some(pc + 1),
            _ => None,
        })
        .filter(|&pc| pc < ops.len())
        .collect()
}

/// Whether the baseline body compiles `ops` (see [`refusal`]).
#[must_use]
pub(crate) fn eligible(ops: &[Op], is_async: bool, is_generator: bool) -> bool {
    refusal(ops, is_async, is_generator).is_none()
}

/// The body's entry points besides pc 0: every loop header (the target of
/// a backward branch), sorted — where on-stack replacement enters.
#[must_use]
pub(crate) fn osr_entries(ops: &[Op]) -> Vec<usize> {
    let mut v: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match op {
            Op::Jump { target } | Op::JumpIfFalse { target, .. } if *target <= pc => Some(*target),
            _ => None,
        })
        .filter(|&t| t != 0)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// The byte offset of register `r` in the register file.
fn slot(r: u16) -> i64 {
    i64::from(r) * 8
}

/// Compiles `ops` (an [`eligible`] body over `n_regs` registers of a
/// function that is async / a generator as `kind` says; `ops` must stay at
/// this address while the code lives: the code points into it) to the
/// frame ABI. `entries` are its entry pcs besides 0 ([`osr_entries`] and
/// [`resume_pcs`]). `None` when an operand does not encode.
#[must_use]
pub(crate) fn compile(
    ops: &[Op],
    n_regs: usize,
    (is_async, is_generator): (bool, bool),
    entries: &[usize],
    h: &Helpers,
) -> Option<Vec<u8>> {
    if n_regs > 0xffff || !eligible(ops, is_async, is_generator) {
        return None;
    }
    let n = ops.len();
    let mut a = A64Assembler::new();
    // One label per op, plus the end (falling off it).
    let labels: Vec<Label> = (0..=n).map(|_| a.new_label()).collect();
    let epilogue = a.new_label();
    // Where a throw goes (the sentinel in x0): the unwinding stub when the
    // body has `try` regions, else straight out.
    let mut catches: Vec<usize> = ops
        .iter()
        .filter_map(|op| match op {
            Op::PushHandler { target, .. } => Some(*target),
            _ => None,
        })
        .collect();
    catches.sort_unstable();
    catches.dedup();
    let thrown = if catches.is_empty() {
        epilogue
    } else {
        a.new_label()
    };

    // Prologue: the frame record and the four callee-saved registers.
    a.stp_pre(FP, LR, SP, -48);
    a.mov_sp(FP, SP);
    a.stp(CTX, REGS, SP, 16);
    a.stp(COUNT, QNAN_R, SP, 32);
    a.mov(CTX, 0);
    a.mov(REGS, 1);
    a.mov_imm(COUNT, 1); // poll at the first back-edge
    a.mov_imm(QNAN_R, QNAN);
    // Entry dispatch: pc 0 or a loop header; anything else is refused with
    // the throw sentinel, the register file untouched.
    a.cbz(2, labels[0]);
    for &e in entries {
        if e == 0 || e >= n {
            continue;
        }
        a.mov_imm(9, e as u64);
        a.cmp(2, 9);
        a.b_cond(Cond::Eq, labels[e]);
    }
    a.mov_imm(0, NanBox::jit_throw_bits());
    a.b(epilogue);

    let throw_bits = NanBox::jit_throw_bits();
    let exit_bits = NanBox::jit_exit_bits();
    let (false_bits, true_bits) = (
        NanBox::boolean(false).to_bits(),
        NanBox::boolean(true).to_bits(),
    );
    let ops_ptr = ops.as_ptr() as u64;

    // Runs op `pc` through the interpreter; a throw leaves with the sentinel
    // (already in x0).
    let exec_op = |a: &mut A64Assembler, pc: usize| {
        a.mov(0, CTX);
        a.mov(1, REGS);
        a.mov_imm(2, n_regs as u64);
        a.mov_imm(3, core::ptr::from_ref(&ops[pc]) as u64);
        a.call_abs(h.exec_op);
        a.cbnz(0, thrown);
    };
    // The back-edge poll before a backward branch to `header`: every
    // `countdown` iterations, the helper (interrupts, agents, the GC
    // safepoint), which may also ask the body to leave at `header`.
    let poll = |a: &mut A64Assembler, header: usize| {
        let skip = a.new_label();
        a.sub_imm(COUNT, COUNT, 1);
        a.cbnz(COUNT, skip);
        a.mov(0, CTX);
        a.mov(1, REGS);
        a.mov_imm(2, n_regs as u64);
        a.mov_imm(3, ops_ptr);
        a.mov_imm(4, n as u64);
        a.mov_imm(5, header as u64);
        a.call_abs(h.backedge);
        a.mov_imm(9, throw_bits);
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, thrown);
        a.mov_imm(9, exit_bits);
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, epilogue);
        a.mov(COUNT, 0);
        a.bind(skip);
    };
    // Branches to `slow` unless x9 and x10 both hold Numbers.
    let both_numbers = |a: &mut A64Assembler, slow: Label| {
        a.and(11, 9, QNAN_R);
        a.cmp(11, QNAN_R);
        a.b_cond(Cond::Eq, slow);
        a.and(11, 10, QNAN_R);
        a.cmp(11, QNAN_R);
        a.b_cond(Cond::Eq, slow);
        a.fmov_from_x(0, 9);
        a.fmov_from_x(1, 10);
    };

    for (pc, op) in ops.iter().enumerate() {
        a.bind(labels[pc]);
        match *op {
            Op::LoadConst { dst, value } => {
                a.mov_imm(9, value.to_bits());
                a.str(9, REGS, slot(dst));
            }
            Op::Move { dst, src } => {
                a.ldr(9, REGS, slot(src));
                a.str(9, REGS, slot(dst));
            }
            // No-ops in the interpreter too (their values are placed by the
            // caller).
            Op::MakeArguments { .. } | Op::LoadNewTarget { .. } => {}
            Op::Jump { target } => {
                if target <= pc {
                    poll(&mut a, target);
                }
                a.b(labels[target]);
            }
            Op::JumpIfFalse { cond, target } => {
                // false / true inline, else ToBoolean by the helper.
                let taken = a.new_label();
                let fall = labels[pc + 1];
                a.ldr(1, REGS, slot(cond));
                a.mov_imm(9, false_bits);
                a.cmp(1, 9);
                a.b_cond(Cond::Eq, taken);
                a.mov_imm(9, true_bits);
                a.cmp(1, 9);
                a.b_cond(Cond::Eq, fall);
                a.mov(0, CTX);
                a.call_abs(h.truthy);
                a.cbnz(0, fall);
                a.bind(taken);
                if target <= pc {
                    poll(&mut a, target);
                }
                a.b(labels[target]);
            }
            Op::Return { src } => {
                a.ldr(0, REGS, slot(src));
                a.b(epilogue);
            }
            Op::PushHandler { target, reg } => {
                a.mov(0, CTX);
                a.mov_imm(1, target as u64);
                a.mov_imm(2, u64::from(reg));
                a.call_abs(h.push_handler);
                a.cbnz(0, thrown);
            }
            Op::PopHandler => {
                a.mov(0, CTX);
                a.call_abs(h.pop_handler);
            }
            // A suspension: the helper records what it yields and where the
            // resumption re-enters, and the body leaves (everything is in
            // the register file); a `yield*` step that finished falls
            // through.
            ref op if suspends(op, is_async, is_generator) => {
                a.mov(0, CTX);
                a.mov(1, REGS);
                a.mov_imm(2, n_regs as u64);
                a.mov_imm(3, core::ptr::from_ref(op) as u64);
                a.mov_imm(4, pc as u64);
                a.call_abs(h.suspend);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.cbnz(0, epilogue);
            }
            Op::Add { dst, a: ra, b: rb }
            | Op::AddValue { dst, a: ra, b: rb }
            | Op::Sub { dst, a: ra, b: rb }
            | Op::Mul { dst, a: ra, b: rb }
            | Op::Div { dst, a: ra, b: rb } => {
                let slow = a.new_label();
                let nan = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(ra));
                a.ldr(10, REGS, slot(rb));
                both_numbers(&mut a, slow);
                match op {
                    Op::Sub { .. } => a.fsub(0, 0, 1),
                    Op::Mul { .. } => a.fmul(0, 0, 1),
                    Op::Div { .. } => a.fdiv(0, 0, 1),
                    _ => a.fadd(0, 0, 1),
                }
                a.fcmp(0, 0);
                a.b_cond(Cond::Vs, nan);
                a.str_d(0, REGS, slot(dst));
                a.b(done);
                a.bind(nan);
                a.mov_imm(9, CANONICAL_NAN);
                a.str(9, REGS, slot(dst));
                a.b(done);
                a.bind(slow);
                exec_op(&mut a, pc);
                a.bind(done);
            }
            Op::Lt { dst, a: ra, b: rb }
            | Op::Compare {
                dst,
                op: 0..=3,
                a: ra,
                b: rb,
            }
            | Op::StrictEq { dst, a: ra, b: rb } => {
                // On two Numbers: the ordered comparison (a NaN operand
                // makes every one false), as a boolean.
                let cond = match op {
                    Op::Compare { op: 1, .. } => Cond::Gt,
                    Op::Compare { op: 2, .. } => Cond::Ls,
                    Op::Compare { op: 3, .. } => Cond::Ge,
                    Op::StrictEq { .. } => Cond::Eq,
                    _ => Cond::Mi,
                };
                let slow = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(ra));
                a.ldr(10, REGS, slot(rb));
                both_numbers(&mut a, slow);
                a.fcmp(0, 1);
                // false + (cond ? 1 : 0) is the boolean's word (`TAG_TRUE`
                // is `TAG_FALSE + 1`).
                a.cset(11, cond);
                a.mov_imm(9, false_bits);
                a.add(9, 9, 11);
                a.str(9, REGS, slot(dst));
                a.b(done);
                a.bind(slow);
                exec_op(&mut a, pc);
                a.bind(done);
            }
            _ => exec_op(&mut a, pc),
        }
    }
    // Falling off the end.
    a.bind(labels[n]);
    a.mov_imm(0, NanBox::jit_falloff_bits());
    if !catches.is_empty() {
        a.b(epilogue);
        // A throw: the innermost handler's catch pc (the value bound in
        // its register), else out with the sentinel.
        a.bind(thrown);
        a.mov(0, CTX);
        a.mov(1, REGS);
        a.mov_imm(2, n_regs as u64);
        a.call_abs(h.unwind);
        for &t in &catches {
            a.mov_imm(9, t as u64);
            a.cmp(0, 9);
            a.b_cond(Cond::Eq, labels[t]);
        }
        a.mov_imm(0, throw_bits);
    }
    a.bind(epilogue);
    a.ldp(COUNT, QNAN_R, SP, 32);
    a.ldp(CTX, REGS, SP, 16);
    a.ldp_post(FP, LR, SP, 48);
    a.ret();
    a.finish()
}

/// Whether the baseline generic tier runs on this target: aarch64, opted in
/// with `KATAAN_JIT_A64GEN=1` (read once per process; always on in the
/// crate's own tests, so the whole suite exercises it there). Off by
/// default while it hands most ops to the interpreter one at a time, which
/// can cost more than interpreting them.
#[must_use]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(target_arch = "aarch64")
        && (cfg!(test)
            || *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_A64GEN").is_some_and(|v| v != "0")))
}

#[cfg(all(feature = "alloc", jit_native))]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
impl super::JitProto {
    /// Compiles `proto` to the baseline generic body (see the module docs),
    /// or `None` when it is not [`eligible`] or its code does not encode.
    pub(crate) fn compile_a64_baseline(
        proto: &crate::nbvm::FnProto,
        helpers: &super::GenericHelpers,
    ) -> Option<Self> {
        if !eligible(&proto.ops, proto.is_async, proto.is_generator) {
            return None;
        }
        // The body's own copy of the bytecode: the code points into it.
        let bytecode: alloc::boxed::Box<[Op]> = proto.ops.clone().into_boxed_slice();
        let kind = (proto.is_async, proto.is_generator);
        let mut entries = osr_entries(&bytecode);
        entries.extend(resume_pcs(&bytecode, kind.0, kind.1));
        entries.sort_unstable();
        entries.dedup();
        let h = Helpers {
            exec_op: helpers.exec_op as usize as u64,
            truthy: helpers.truthy as usize as u64,
            backedge: helpers.backedge as usize as u64,
            push_handler: helpers.push_handler as usize as u64,
            pop_handler: helpers.pop_handler as usize as u64,
            unwind: helpers.unwind as usize as u64,
            suspend: helpers.suspend as usize as u64,
        };
        let code = compile(&bytecode, proto.n_regs, kind, &entries, &h)?;
        let func = super::JitFunction::from_code(&code)?;
        Some(Self {
            func,
            n_params: proto.n_params,
            kind: super::JitKind::Generic,
            native_entry: None,
            n_regs: proto.n_regs,
            osr_entries: entries,
            bytecode: Some(bytecode),
            sites: None,
            pc_map: None,
            returns_bool: false,
            gen_resume: Vec::new(),
        })
    }
}

/// The emitter on any host (running the code is the aarch64 suite's job,
/// where every lib test runs with this tier on).
#[cfg(test)]
mod tests {
    use super::*;

    fn protos(src: &str) -> Vec<crate::nbvm::FnProto> {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        crate::nbvm::compile_program(&program).expect("compile")
    }

    const H: Helpers = Helpers {
        exec_op: 0x1000,
        truthy: 0x2000,
        backedge: 0x3000,
        push_handler: 0x4000,
        pop_handler: 0x5000,
        unwind: 0x6000,
        suspend: 0x7000,
    };

    #[test]
    fn eligible_bodies_encode() {
        let funcs = protos(
            "function f(n) { let s = 0; for (let i = 0; i < n; i++) { s += i * 2.5; } return s; }
             function g(o) { try { return o.x.y; } catch (e) { return 'caught ' + e; } }
             function h(a, b) { if (a <= b && a !== b) return [a, b].map(x => x / 3); return {a, b}; }
             f(10); g({}); h(1, 2);",
        );
        let mut compiled = 0;
        for p in &funcs {
            if eligible(&p.ops, p.is_async, p.is_generator) {
                let entries = osr_entries(&p.ops);
                let code = compile(&p.ops, p.n_regs, (false, false), &entries, &H);
                assert!(code.is_some(), "{} encodes", p.name);
                compiled += 1;
            }
        }
        assert!(compiled >= 4, "most bodies are eligible ({compiled})");
    }

    #[test]
    fn frame_bound_ops_are_refused() {
        let start = [Op::GeneratorStart];
        assert_eq!(refusal(&start, false, false), Some("suspension"));
        assert_eq!(refusal(&start, false, true), None);
        assert_eq!(resume_pcs(&start, false, true), Vec::<usize>::new());
        let ops = [Op::TailCall {
            func: 0,
            args: Vec::new(),
        }];
        assert_eq!(refusal(&ops, false, false), Some("tail call"));
        let ops = [Op::Jump { target: 2 }];
        assert_eq!(refusal(&ops, false, false), Some("branch out of range"));
        assert_eq!(refusal(&[Op::PopHandler], false, false), None);
    }

    #[test]
    fn loop_headers_are_entries() {
        let funcs = protos("function f(n) { let s = 0; while (s < n) { s++; } return s; } f;");
        let f = funcs.iter().find(|p| p.name == "f").expect("f");
        let entries = osr_entries(&f.ops);
        assert_eq!(entries.len(), 1, "one loop: {:?}", f.ops);
        let back = f.ops.iter().enumerate().any(|(pc, op)| {
            matches!(op, Op::Jump { target } | Op::JumpIfFalse { target, .. }
                if *target <= pc && *target == entries[0])
        });
        assert!(back, "the entry is a backward branch's target");
    }
}
