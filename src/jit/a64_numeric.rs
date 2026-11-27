//! The AArch64 lowerings of the JIT's numeric IRs: the Int tier
//! ([`RegOp`]), the Float tier ([`FloatOp`]) and the small demonstration
//! programs ([`ArithOp`], [`StackOp`], the counted loop). Each returns the
//! finished machine code (AAPCS64), which `JitFunction` maps executable; the
//! x86-64 versions of the same functions live with `JitFunction`.
//!
//! Frame (Int and Float tiers): `stp x29, x30, [sp, #-16]!; mov x29, sp;
//! sub sp, sp, #frame`, with virtual register `r` homed at `[sp + 8*r]`
//! (`frame` is `8 * n_regs` rounded up to 16, at most 512 bytes, so every
//! slot is in reach of the scaled-offset load/store). `x0..x5` / `d0..d3`
//! bring the arguments, `x0` / `d0` return the result. Scratch registers
//! are `x0..x3`, `d0..d1` and, in the Int tier, `x9`/`x10` holding the
//! ±2^53 safe-integer bounds (caller-saved: reloaded after a call).
//!
//! The semantics are the x86 tiers' exactly (they are both checked against
//! the same interpreter oracles, [`eval_reg`](super::eval_reg) and
//! [`eval_float`](super::eval_float)): the Int tier keeps only values
//! within ±2^53 and leaves for the interpreter (returns the `i64::MAX`
//! sentinel) on overflow, a result out of that range, a zero divisor or a
//! negative dividend of `%`.

use super::a64::{A64Assembler, Cond, FP, LR, SP, Shift, ZR};
use super::{
    ArithOp, BinOp, BinOp2, FBinOp, FloatOp, Label, RegOp, SAFE_INT_MAX, ShiftOp, StackOp,
};
use alloc::vec::Vec;

/// The safe-integer upper bound's register (Int tier).
const HI: u8 = 9;
/// The safe-integer lower bound's register (Int tier).
const LO: u8 = 10;

/// `x0..x5`: where the Int tier's arguments arrive, in order.
const MAX_INT_ARGS: usize = 6;
/// `d0..d3`: the Float tier's arguments.
const MAX_FLOAT_ARGS: usize = 4;

/// The slot of virtual register `r`: `[sp + 8*r]`.
fn slot(r: u8) -> i64 {
    i64::from(r) * 8
}

/// The frame prologue: saves `x29`/`x30`, sets the frame pointer and
/// reserves `frame` bytes for the slots.
fn prologue(a: &mut A64Assembler, frame: u32) {
    a.stp_pre(FP, LR, SP, -16);
    a.mov_sp(FP, SP);
    if frame != 0 {
        a.sub_imm(SP, SP, i64::from(frame));
    }
}

/// The frame epilogue and return (the result already in `x0`/`d0`).
fn epilogue(a: &mut A64Assembler) {
    a.mov_sp(SP, FP);
    a.ldp_post(FP, LR, SP, 16);
    a.ret();
}

/// The slot area's size: `8 * n_regs` rounded up to 16 bytes.
fn frame_of(n_regs: usize) -> u32 {
    ((n_regs as u32 * 8) + 15) & !15
}

/// [`JitFunction::compile_arith`](super::JitFunction::compile_arith):
/// `fn(i64) -> i64`, the accumulator in `x0`.
pub(crate) fn arith(ops: &[ArithOp]) -> Option<Vec<u8>> {
    let mut a = A64Assembler::new();
    for op in ops {
        let imm = |a: &mut A64Assembler, n: i32| a.mov_imm(1, i64::from(n) as u64);
        match *op {
            ArithOp::AddImm(n) => {
                imm(&mut a, n);
                a.add(0, 0, 1);
            }
            ArithOp::SubImm(n) => {
                imm(&mut a, n);
                a.sub(0, 0, 1);
            }
            ArithOp::MulImm(n) => {
                imm(&mut a, n);
                a.mul(0, 0, 1);
            }
            ArithOp::AndImm(n) => {
                imm(&mut a, n);
                a.and(0, 0, 1);
            }
            ArithOp::OrImm(n) => {
                imm(&mut a, n);
                a.orr(0, 0, 1);
            }
            ArithOp::XorImm(n) => {
                imm(&mut a, n);
                a.eor(0, 0, 1);
            }
            ArithOp::ShlImm(n) => a.lsl_imm(0, 0, n),
            ArithOp::SarImm(n) => a.asr_imm(0, 0, n),
            ArithOp::Neg => a.neg(0, 0),
        }
    }
    a.ret();
    a.finish()
}

/// [`JitFunction::compile_binary`](super::JitFunction::compile_binary).
pub(crate) fn binary(op: BinOp) -> Option<Vec<u8>> {
    let mut a = A64Assembler::new();
    match op {
        BinOp::Add => a.add(0, 0, 1),
        BinOp::Sub => a.sub(0, 0, 1),
        BinOp::Mul => a.mul(0, 0, 1),
    }
    a.ret();
    a.finish()
}

/// [`JitFunction::compile_sum_1_to_n`](super::JitFunction::compile_sum_1_to_n).
pub(crate) fn sum_1_to_n() -> Option<Vec<u8>> {
    let mut a = A64Assembler::new();
    let top = a.new_label();
    let done = a.new_label();
    a.mov_imm(2, 0); // acc
    a.mov(1, 0); // i = n
    a.bind(top);
    a.cmp_imm(1, 0);
    a.b_cond(Cond::Le, done);
    a.add(2, 2, 1);
    a.sub_imm(1, 1, 1);
    a.b(top);
    a.bind(done);
    a.mov(0, 2);
    a.ret();
    a.finish()
}

/// [`JitFunction::compile_stack`](super::JitFunction::compile_stack) (the
/// program already checked to leave exactly one value): one 16-byte stack
/// slot per operand, keeping `sp` aligned.
pub(crate) fn stack(ops: &[StackOp]) -> Option<Vec<u8>> {
    let mut a = A64Assembler::new();
    for op in ops {
        match *op {
            // (As the oracle `eval_stack`: argument `i & 1`.)
            StackOp::Arg(i) => a.push(i & 1),
            StackOp::Const(n) => {
                a.mov_imm(2, n as u64);
                a.push(2);
            }
            StackOp::Add | StackOp::Sub | StackOp::Mul => {
                a.pop(3);
                a.pop(2);
                match op {
                    StackOp::Add => a.add(2, 2, 3),
                    StackOp::Sub => a.sub(2, 2, 3),
                    _ => a.mul(2, 2, 3),
                }
                a.push(2);
            }
        }
    }
    a.pop(0);
    a.ret();
    a.finish()
}

/// [`JitFunction::compile_reg`](super::JitFunction::compile_reg): the Int
/// tier, `extern "C" fn(i64 x6) -> i64`. `None` for a malformed program
/// (as the x86 version), or code that does not encode.
pub(crate) fn reg(n_regs: usize, n_args: usize, ops: &[RegOp]) -> Option<Vec<u8>> {
    if n_regs > 64 || n_args > MAX_INT_ARGS {
        return None;
    }
    let ok = |r: u8| (r as usize) < n_regs;
    let mut a = A64Assembler::new();
    let labels: Vec<Label> = (0..ops.len()).map(|_| a.new_label()).collect();
    let deopt = a.new_label();
    prologue(&mut a, frame_of(n_regs));
    // Every slot starts at 0 (as the x86 tier: a read of an unwritten slot
    // is never stack garbage).
    for r in 0..n_regs {
        a.str(ZR, SP, slot(r as u8));
    }
    let bounds = |a: &mut A64Assembler| {
        a.mov_imm(HI, SAFE_INT_MAX as u64);
        a.mov_imm(LO, (-SAFE_INT_MAX) as u64);
    };
    bounds(&mut a);
    // The result in `x0` must be within ±2^53 (else leave).
    let range = |a: &mut A64Assembler| {
        a.cmp(0, HI);
        a.b_cond(Cond::Gt, deopt);
        a.cmp(0, LO);
        a.b_cond(Cond::Lt, deopt);
    };
    let mut has_ret = false;
    for (i, op) in ops.iter().enumerate() {
        a.bind(labels[i]);
        match *op {
            RegOp::Arg { dst, index } => {
                if !ok(dst) || index as usize >= n_args {
                    return None;
                }
                a.str(index, SP, slot(dst));
            }
            RegOp::Const { dst, imm } => {
                if !ok(dst) {
                    return None;
                }
                a.mov_imm(0, imm as u64);
                a.str(0, SP, slot(dst));
            }
            RegOp::Bin {
                dst,
                a: ra,
                b: rb,
                op,
            } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                a.ldr(0, SP, slot(ra));
                a.ldr(1, SP, slot(rb));
                match op {
                    BinOp2::Add => {
                        a.adds(0, 0, 1);
                        a.b_cond(Cond::Vs, deopt);
                    }
                    BinOp2::Sub => {
                        a.subs(0, 0, 1);
                        a.b_cond(Cond::Vs, deopt);
                    }
                    BinOp2::Mul => {
                        // The 128-bit product fits 64 bits iff its high half
                        // is the low half's sign extension.
                        a.mul(2, 0, 1);
                        a.smulh(3, 0, 1);
                        a.cmp_shifted(3, 2, Shift::Asr, 63);
                        a.b_cond(Cond::Ne, deopt);
                        a.mov(0, 2);
                    }
                    BinOp2::And => a.and(0, 0, 1),
                    BinOp2::Or => a.orr(0, 0, 1),
                    BinOp2::Xor => a.eor(0, 0, 1),
                }
                range(&mut a);
                a.str(0, SP, slot(dst));
            }
            RegOp::Move { dst, src } => {
                if !ok(dst) || !ok(src) {
                    return None;
                }
                a.ldr(0, SP, slot(src));
                a.str(0, SP, slot(dst));
            }
            RegOp::Lt { dst, a: ra, b: rb } | RegOp::Eq { dst, a: ra, b: rb } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                a.ldr(0, SP, slot(ra));
                a.ldr(1, SP, slot(rb));
                a.cmp(0, 1);
                let cond = if matches!(op, RegOp::Lt { .. }) {
                    Cond::Lt
                } else {
                    Cond::Eq
                };
                a.cset(0, cond);
                a.str(0, SP, slot(dst));
            }
            RegOp::BitNot32 { dst, a: ra } => {
                if !ok(dst) || !ok(ra) {
                    return None;
                }
                a.ldr(0, SP, slot(ra));
                a.mvn_w(0, 0);
                a.sxtw(0, 0);
                a.str(0, SP, slot(dst));
            }
            RegOp::Mod { dst, a: ra, b: rb } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // A zero divisor (NaN) and a negative dividend (whose zero
                // remainder is `-0`) leave; else the truncated remainder.
                a.ldr(1, SP, slot(rb));
                a.cbz(1, deopt);
                a.ldr(0, SP, slot(ra));
                a.tbnz(0, 63, deopt);
                a.sdiv(2, 0, 1);
                a.msub(0, 2, 1, 0);
                a.str(0, SP, slot(dst));
            }
            RegOp::Eqz { dst, a: ra } => {
                if !ok(dst) || !ok(ra) {
                    return None;
                }
                a.ldr(0, SP, slot(ra));
                a.cmp_imm(0, 0);
                a.cset(0, Cond::Eq);
                a.str(0, SP, slot(dst));
            }
            RegOp::Neg { dst, a: ra } => {
                if !ok(dst) || !ok(ra) {
                    return None;
                }
                a.ldr(0, SP, slot(ra));
                a.negs(0, 0);
                a.b_cond(Cond::Vs, deopt);
                range(&mut a);
                a.str(0, SP, slot(dst));
            }
            RegOp::Bit32 {
                dst,
                a: ra,
                b: rb,
                op,
            } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // ToInt32 of both is their low words; the op on the words,
                // sign-extended, is the exact i32 result.
                a.ldr(0, SP, slot(ra));
                a.ldr(1, SP, slot(rb));
                match op {
                    BinOp2::And => a.and_w(0, 0, 1),
                    BinOp2::Or => a.orr_w(0, 0, 1),
                    BinOp2::Xor => a.eor_w(0, 0, 1),
                    // The lowering only makes bitwise `Bit32`s.
                    BinOp2::Add | BinOp2::Sub | BinOp2::Mul => return None,
                }
                a.sxtw(0, 0);
                a.str(0, SP, slot(dst));
            }
            RegOp::Shift32 {
                dst,
                a: ra,
                b: rb,
                op,
            } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // The 32-bit shifts take the count mod 32 (JS masks it to 5
                // bits) and zero-extend; `<<`/`>>` are then sign-extended,
                // `>>>` stays the unsigned u32.
                a.ldr(0, SP, slot(ra));
                a.ldr(1, SP, slot(rb));
                match op {
                    ShiftOp::Shl => a.lsl_w(0, 0, 1),
                    ShiftOp::Sar => a.asr_w(0, 0, 1),
                    ShiftOp::Shr => a.lsr_w(0, 0, 1),
                }
                if matches!(op, ShiftOp::Shl | ShiftOp::Sar) {
                    a.sxtw(0, 0);
                }
                a.str(0, SP, slot(dst));
            }
            RegOp::JumpIfFalse { cond, target } => {
                if !ok(cond) || target >= ops.len() {
                    return None;
                }
                a.ldr(0, SP, slot(cond));
                a.cbz(0, labels[target]);
            }
            RegOp::Jump { target } => {
                if target >= ops.len() {
                    return None;
                }
                a.b(labels[target]);
            }
            RegOp::Ret { src } => {
                if !ok(src) {
                    return None;
                }
                a.ldr(0, SP, slot(src));
                epilogue(&mut a);
                has_ret = true;
            }
            RegOp::Call {
                dst,
                code_ptr,
                n_args: argc,
                args,
            } => {
                if !ok(dst) || argc as usize > MAX_INT_ARGS {
                    return None;
                }
                for (i, &arg) in args[..argc as usize].iter().enumerate() {
                    if !ok(arg) {
                        return None;
                    }
                    a.ldr(i as u8, SP, slot(arg));
                }
                // `sp` is 16-aligned (the frame is a multiple of 16).
                a.call_abs(code_ptr);
                bounds(&mut a);
                // A callee's leave returns the out-of-range sentinel, which
                // the range check passes on as this body's own leave.
                range(&mut a);
                a.str(0, SP, slot(dst));
            }
        }
    }
    if !has_ret {
        return None;
    }
    a.bind(deopt);
    a.mov_imm(0, i64::MAX as u64);
    epilogue(&mut a);
    a.finish()
}

/// [`JitFunction::compile_float`](super::JitFunction::compile_float): the
/// Float tier, `extern "C" fn(f64 x4) -> f64`.
pub(crate) fn float(n_regs: usize, n_args: usize, ops: &[FloatOp]) -> Option<Vec<u8>> {
    if n_regs > 64 || n_args > MAX_FLOAT_ARGS {
        return None;
    }
    let ok = |r: u8| (r as usize) < n_regs;
    let mut a = A64Assembler::new();
    let labels: Vec<Label> = (0..ops.len()).map(|_| a.new_label()).collect();
    prologue(&mut a, frame_of(n_regs));
    let mut has_ret = false;
    // `d0 = op(slot a, slot b)` for the binary ops below.
    let load2 = |a: &mut A64Assembler, ra: u8, rb: u8| {
        a.ldr_d(0, SP, slot(ra));
        a.ldr_d(1, SP, slot(rb));
    };
    for (i, op) in ops.iter().enumerate() {
        a.bind(labels[i]);
        match *op {
            FloatOp::Arg { dst, index } => {
                if !ok(dst) || index as usize >= n_args {
                    return None;
                }
                a.str_d(index, SP, slot(dst));
            }
            FloatOp::Const { dst, imm } => {
                if !ok(dst) {
                    return None;
                }
                a.mov_imm(0, imm.to_bits());
                a.str(0, SP, slot(dst));
            }
            FloatOp::Bin {
                dst,
                a: ra,
                b: rb,
                op,
            } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                load2(&mut a, ra, rb);
                match op {
                    FBinOp::Add => a.fadd(0, 0, 1),
                    FBinOp::Sub => a.fsub(0, 0, 1),
                    FBinOp::Mul => a.fmul(0, 0, 1),
                    FBinOp::Div => a.fdiv(0, 0, 1),
                }
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::Move { dst, src } => {
                if !ok(dst) || !ok(src) {
                    return None;
                }
                a.ldr(0, SP, slot(src));
                a.str(0, SP, slot(dst));
            }
            FloatOp::Lt { dst, a: ra, b: rb } | FloatOp::Eq { dst, a: ra, b: rb } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // After `fcmp`, `mi` is "ordered and less", `eq` "ordered and
                // equal" (unordered sets only `C`/`V`). Stored as 0.0/1.0.
                load2(&mut a, ra, rb);
                a.fcmp(0, 1);
                let cond = if matches!(op, FloatOp::Lt { .. }) {
                    Cond::Mi
                } else {
                    Cond::Eq
                };
                a.cset(0, cond);
                a.scvtf(0, 0);
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::JumpIfFalse { cond, target } => {
                if !ok(cond) || target >= ops.len() {
                    return None;
                }
                // Falsy: ±0 or NaN (as the x86 tier's `ucomisd`/`je`).
                a.ldr_d(0, SP, slot(cond));
                a.fcmp_zero(0);
                a.b_cond(Cond::Eq, labels[target]);
                a.b_cond(Cond::Vs, labels[target]);
            }
            FloatOp::Jump { target } => {
                if target >= ops.len() {
                    return None;
                }
                a.b(labels[target]);
            }
            FloatOp::Neg { dst, a: ra }
            | FloatOp::Sqrt { dst, a: ra }
            | FloatOp::Abs { dst, a: ra }
            | FloatOp::Floor { dst, a: ra }
            | FloatOp::Ceil { dst, a: ra }
            | FloatOp::Trunc { dst, a: ra } => {
                if !ok(dst) || !ok(ra) {
                    return None;
                }
                a.ldr_d(0, SP, slot(ra));
                match op {
                    FloatOp::Neg { .. } => a.fneg(0, 0),
                    FloatOp::Sqrt { .. } => a.fsqrt(0, 0),
                    FloatOp::Abs { .. } => a.fabs(0, 0),
                    FloatOp::Floor { .. } => a.frintm(0, 0),
                    FloatOp::Ceil { .. } => a.frintp(0, 0),
                    _ => a.frintz(0, 0),
                }
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::Mod { dst, a: ra, b: rb } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // a - trunc(a/b)*b, rounded at each step as the x86 tier.
                load2(&mut a, ra, rb);
                a.fdiv(2, 0, 1);
                a.frintz(2, 2);
                a.fmul(2, 2, 1);
                a.fsub(0, 0, 2);
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::Max { dst, a: ra, b: rb } | FloatOp::Min { dst, a: ra, b: rb } => {
                if !ok(dst) || !ok(ra) || !ok(rb) {
                    return None;
                }
                // `fmax`/`fmin` are `Math.max`/`Math.min` on two numbers:
                // NaN propagates, and +0 is above -0.
                load2(&mut a, ra, rb);
                if matches!(op, FloatOp::Max { .. }) {
                    a.fmax(0, 0, 1);
                } else {
                    a.fmin(0, 0, 1);
                }
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::Eqz { dst, a: ra } => {
                if !ok(dst) || !ok(ra) {
                    return None;
                }
                // `!x`: x is ±0 (`eq`) or NaN (`vs`).
                a.ldr_d(0, SP, slot(ra));
                a.fcmp_zero(0);
                a.cset(0, Cond::Eq);
                a.cset(1, Cond::Vs);
                a.orr(0, 0, 1);
                a.scvtf(0, 0);
                a.str_d(0, SP, slot(dst));
            }
            FloatOp::Ret { src } => {
                if !ok(src) {
                    return None;
                }
                a.ldr_d(0, SP, slot(src));
                epilogue(&mut a);
                has_ret = true;
            }
        }
    }
    if !has_ret {
        return None;
    }
    a.finish()
}
