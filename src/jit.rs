//! A baseline machine-code JIT (Phase G).
//!
//! This is the genuine native-code path the roadmap calls for: an x86-64
//! assembler (`X64Assembler`) that lowers a small arithmetic IR to machine
//! code, and an executable-memory region (`ExecBuffer`) that maps the code
//! W^X (write, then flip to execute) and hands back a callable function pointer.
//!
//! **No foreign code.** Executable memory needs the OS, but we never link a C
//! library: the `mmap`/`mprotect`/`munmap` calls are issued directly through the
//! Linux x86-64 `syscall` instruction via `core::arch::asm!` — pure Rust over the
//! kernel ABI. The whole thing is gated to `target_os = "linux"` +
//! `target_arch = "x86_64"`; on every other target `available()` is `false`
//! and compilation returns `None`, so callers transparently fall back to the
//! interpreter.
//!
//! `unsafe` is used — and *only* used — for the three irreducibly-unsafe steps a
//! JIT requires: the raw syscalls, writing/executing mapped memory, and the
//! transmute of a code pointer to a function pointer. Each carries a safety
//! comment. This is exactly the "audited VM hot-path primitives" carve-out the
//! crate's `unsafe_code = "deny"` policy leaves open.

use alloc::vec::Vec;

#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod call_seq;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod global_persist;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub mod inline;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod props;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod recv_cache;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod tierup;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub use call_seq::{
    CallLayout, CallMemo, CallSite, CallTarget, CalleeHelper, CalleeInfo, ConstructAllocHelper,
    ConstructFast, ConstructHelper, ConstructResultHelper, FRAME_HEADER, IdentMemo, JitStack,
    MethodHelper, SuperTwiceHelper,
};
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub use props::MethodSite;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub(crate) use props::shape_word;
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub use tierup::MAX_GENERATION;

/// `2^53` — the largest magnitude an `f64` represents every integer below
/// exactly. The integer JIT keeps a value only while it stays within ±this; a
/// result outside the range deopts, since `i64` and `f64` arithmetic diverge
/// beyond it.
const SAFE_INT_MAX: i64 = 9_007_199_254_740_992;

/// An arithmetic operation in the JIT's tiny IR, applied left-to-right to a
/// running accumulator seeded with the function's `i64` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithOp {
    /// `acc += imm`
    AddImm(i32),
    /// `acc -= imm`
    SubImm(i32),
    /// `acc *= imm`
    MulImm(i32),
    /// `acc &= imm`
    AndImm(i32),
    /// `acc |= imm`
    OrImm(i32),
    /// `acc ^= imm`
    XorImm(i32),
    /// `acc <<= imm` (0..=63)
    ShlImm(u8),
    /// `acc >>= imm` (arithmetic, 0..=63)
    SarImm(u8),
    /// `acc = -acc`
    Neg,
}

impl ArithOp {
    /// Evaluates the op on `acc` with wrapping `i64` arithmetic — the reference
    /// the JIT-compiled code must match.
    #[must_use]
    pub fn eval(self, acc: i64) -> i64 {
        match self {
            ArithOp::AddImm(n) => acc.wrapping_add(i64::from(n)),
            ArithOp::SubImm(n) => acc.wrapping_sub(i64::from(n)),
            ArithOp::MulImm(n) => acc.wrapping_mul(i64::from(n)),
            ArithOp::AndImm(n) => acc & i64::from(n),
            ArithOp::OrImm(n) => acc | i64::from(n),
            ArithOp::XorImm(n) => acc ^ i64::from(n),
            ArithOp::ShlImm(n) => acc.wrapping_shl(u32::from(n)),
            ArithOp::SarImm(n) => acc.wrapping_shr(u32::from(n)),
            ArithOp::Neg => acc.wrapping_neg(),
        }
    }
}

/// Evaluates a whole op sequence — the interpreter mirror of the compiled code.
#[must_use]
pub fn eval_arith(ops: &[ArithOp], arg: i64) -> i64 {
    ops.iter().fold(arg, |acc, op| op.eval(acc))
}

/// A stack-machine instruction — the shape the register VM lowers integer
/// expressions to. Compiled to native code over the hardware stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StackOp {
    /// Push argument `0` or `1`.
    Arg(u8),
    /// Push an `i64` constant.
    Const(i64),
    /// Pop `b`, pop `a`, push `a + b`.
    Add,
    /// Pop `b`, pop `a`, push `a - b`.
    Sub,
    /// Pop `b`, pop `a`, push `a * b`.
    Mul,
}

/// Evaluates a [`StackOp`] program over two `i64` arguments — the interpreter
/// oracle for the JIT-compiled stack machine. Returns the value left on top.
#[must_use]
pub fn eval_stack(ops: &[StackOp], args: [i64; 2]) -> i64 {
    let mut stack: Vec<i64> = Vec::new();
    for op in ops {
        match *op {
            StackOp::Arg(i) => stack.push(args[i as usize & 1]),
            StackOp::Const(n) => stack.push(n),
            StackOp::Add | StackOp::Sub | StackOp::Mul => {
                let b = stack.pop().unwrap_or(0);
                let a = stack.pop().unwrap_or(0);
                stack.push(match op {
                    StackOp::Add => a.wrapping_add(b),
                    StackOp::Sub => a.wrapping_sub(b),
                    _ => a.wrapping_mul(b),
                });
            }
        }
    }
    stack.pop().unwrap_or(0)
}

/// A binary op for the register compiler (`op_rax_mem`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp2 {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Xor,
}

/// A JS 32-bit shift op (the count is masked to 5 bits; operand is `ToInt32`d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftOp {
    /// `<<` — left shift (result reinterpreted as a signed i32)
    Shl,
    /// `>>` — sign-propagating (arithmetic) right shift
    Sar,
    /// `>>>` — zero-filling (logical) right shift (result is an unsigned u32)
    Shr,
}

/// A binary floating-point op for the float compiler. Unlike the integer path,
/// `f64` arithmetic matches JS number semantics exactly, so no overflow/range
/// guard is needed — and it supports division.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FBinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
}

/// A floating-point register-machine instruction — straight-line `f64`
/// arithmetic over a virtual-register file (each register an `f64` frame slot).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)] // field names mirror the IR
pub enum FloatOp {
    Arg {
        dst: u8,
        index: u8,
    },
    Const {
        dst: u8,
        imm: f64,
    },
    Bin {
        dst: u8,
        a: u8,
        b: u8,
        op: FBinOp,
    },
    /// `reg[dst] = reg[a] % reg[b]` — float remainder (`a - trunc(a/b)*b`, JS `%`).
    /// Emitted via `roundsd` (SSE4.1), so lowering gates on its presence.
    Mod {
        dst: u8,
        a: u8,
        b: u8,
    },
    Move {
        dst: u8,
        src: u8,
    },
    /// `reg[dst] = (reg[a] < reg[b]) ? 1.0 : 0.0` (ordered; NaN → 0.0)
    Lt {
        dst: u8,
        a: u8,
        b: u8,
    },
    /// if `reg[cond] == 0.0`, jump to op index `target`
    JumpIfFalse {
        cond: u8,
        target: usize,
    },
    /// unconditional jump to op index `target`
    Jump {
        target: usize,
    },
    /// `reg[dst] = -reg[a]` — float unary minus (`0.0 - x`).
    Neg {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = sqrt(reg[a])` — `Math.sqrt` (the SSE2 `sqrtsd`).
    Sqrt {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = |reg[a]|` — `Math.abs` (clears the f64 sign bit).
    Abs {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = Math.max(reg[a], reg[b])` — JS max: NaN if either is NaN,
    /// `+0 > -0`.
    Max {
        dst: u8,
        a: u8,
        b: u8,
    },
    /// `reg[dst] = Math.min(reg[a], reg[b])` — JS min: NaN if either is NaN,
    /// `-0 < +0`.
    Min {
        dst: u8,
        a: u8,
        b: u8,
    },
    /// `reg[dst] = floor(reg[a])` — `Math.floor` (SSE4.1 `roundsd`, gated).
    Floor {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = ceil(reg[a])` — `Math.ceil` (SSE4.1 `roundsd`, gated).
    Ceil {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = trunc(reg[a])` — `Math.trunc` (SSE4.1 `roundsd`, gated).
    Trunc {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = !reg[a]` — JS logical-not: `1.0` if `reg[a]` is falsy (`±0.0`
    /// or `NaN`), else `0.0`. (`ucomisd x, 0` sets ZF for both equal and NaN.)
    Eqz {
        dst: u8,
        a: u8,
    },
    /// `reg[dst] = (reg[a] === reg[b]) ? 1.0 : 0.0` — JS numeric strict equality:
    /// ordered *and* equal, so `NaN === NaN` is `0.0` and `+0.0 === -0.0` is `1.0`.
    Eq {
        dst: u8,
        a: u8,
        b: u8,
    },
    Ret {
        src: u8,
    },
}

/// Interprets a [`FloatOp`] program — the oracle for [`JitFunction::compile_float`].
#[must_use]
pub fn eval_float(ops: &[FloatOp], n_regs: usize, args: &[f64]) -> f64 {
    let mut regs = alloc::vec![0.0f64; n_regs];
    let mut pc = 0usize;
    while pc < ops.len() {
        match ops[pc] {
            FloatOp::Arg { dst, index } => {
                regs[dst as usize] = args.get(index as usize).copied().unwrap_or(0.0);
                pc += 1;
            }
            FloatOp::Const { dst, imm } => {
                regs[dst as usize] = imm;
                pc += 1;
            }
            FloatOp::Bin { dst, a, b, op } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                regs[dst as usize] = match op {
                    FBinOp::Add => x + y,
                    FBinOp::Sub => x - y,
                    FBinOp::Mul => x * y,
                    FBinOp::Div => x / y,
                };
                pc += 1;
            }
            FloatOp::Mod { dst, a, b } => {
                // Matches the emitted `a - trunc(a/b)*b`; Rust's `%` on f64 is the
                // same IEEE remainder (sign of the dividend), as is JS `%`.
                let (x, y) = (regs[a as usize], regs[b as usize]);
                regs[dst as usize] = x - (x / y).trunc() * y;
                pc += 1;
            }
            FloatOp::Move { dst, src } => {
                regs[dst as usize] = regs[src as usize];
                pc += 1;
            }
            FloatOp::Lt { dst, a, b } => {
                regs[dst as usize] = f64::from(u8::from(regs[a as usize] < regs[b as usize]));
                pc += 1;
            }
            FloatOp::JumpIfFalse { cond, target } => {
                if regs[cond as usize] == 0.0 {
                    pc = target;
                } else {
                    pc += 1;
                }
            }
            FloatOp::Jump { target } => pc = target,
            FloatOp::Neg { dst, a } => {
                regs[dst as usize] = -regs[a as usize];
                pc += 1;
            }
            FloatOp::Sqrt { dst, a } => {
                regs[dst as usize] = regs[a as usize].sqrt();
                pc += 1;
            }
            FloatOp::Abs { dst, a } => {
                regs[dst as usize] = regs[a as usize].abs();
                pc += 1;
            }
            FloatOp::Max { dst, a, b } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                regs[dst as usize] = if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else if x > y {
                    x
                } else if y > x {
                    y
                } else if x.is_sign_positive() {
                    x // equal: +0 wins over -0
                } else {
                    y
                };
                pc += 1;
            }
            FloatOp::Min { dst, a, b } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                regs[dst as usize] = if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else if x < y {
                    x
                } else if y < x {
                    y
                } else if x.is_sign_negative() {
                    x // equal: -0 wins over +0
                } else {
                    y
                };
                pc += 1;
            }
            FloatOp::Floor { dst, a } => {
                regs[dst as usize] = regs[a as usize].floor();
                pc += 1;
            }
            FloatOp::Trunc { dst, a } => {
                regs[dst as usize] = regs[a as usize].trunc();
                pc += 1;
            }
            FloatOp::Ceil { dst, a } => {
                regs[dst as usize] = regs[a as usize].ceil();
                pc += 1;
            }
            FloatOp::Eqz { dst, a } => {
                let x = regs[a as usize];
                regs[dst as usize] = f64::from(u8::from(x == 0.0 || x.is_nan()));
                pc += 1;
            }
            FloatOp::Eq { dst, a, b } => {
                // Rust f64 `==` is IEEE ordered equality: NaN != NaN, +0.0 == -0.0.
                regs[dst as usize] = f64::from(u8::from(regs[a as usize] == regs[b as usize]));
                pc += 1;
            }
            FloatOp::Ret { src } => return regs[src as usize],
        }
    }
    0.0
}

/// A register-machine instruction over a flat virtual-register file — the model
/// the bytecode VM uses. Each register is an `i64` slot in the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)] // field names (dst/a/b/op/imm/index/src) mirror the IR
pub enum RegOp {
    /// `reg[dst] = arg[index]`
    Arg { dst: u8, index: u8 },
    /// `reg[dst] = imm`
    Const { dst: u8, imm: i64 },
    /// `reg[dst] = reg[a] <op> reg[b]`
    Bin { dst: u8, a: u8, b: u8, op: BinOp2 },
    /// `reg[dst] = reg[src]`
    Move { dst: u8, src: u8 },
    /// `reg[dst] = (reg[a] < reg[b]) as 0/1` (signed)
    Lt { dst: u8, a: u8, b: u8 },
    /// `reg[dst] = (reg[a] == 0) as 0/1` — JS logical-not of an integer (`!x`).
    Eqz { dst: u8, a: u8 },
    /// `reg[dst] = (reg[a] == reg[b]) as 0/1` — strict equality of integers.
    Eq { dst: u8, a: u8, b: u8 },
    /// `reg[dst] = -reg[a]` — JS unary minus of an integer (`-x`).
    Neg { dst: u8, a: u8 },
    /// `reg[dst] = ~ToInt32(reg[a])` — JS bitwise-not (`~x`); result is an i32.
    BitNot32 { dst: u8, a: u8 },
    /// `reg[dst] = reg[a] % reg[b]` — integer remainder (JS `%` on integers). A
    /// zero divisor deopts (JS yields `NaN`, which isn't an integer).
    Mod { dst: u8, a: u8, b: u8 },
    /// `reg[dst] = ToInt32(reg[a]) <op> ToInt32(reg[b])` — a JS 32-bit bitwise op
    /// (`&`/`|`/`^`). Both operands are truncated to `i32` first, so the result is
    /// the exact 32-bit value (always within ±2^53), needing no range guard. `op`
    /// is one of `And`/`Or`/`Xor`.
    Bit32 { dst: u8, a: u8, b: u8, op: BinOp2 },
    /// `reg[dst] = reg[a] <op> (reg[b] & 31)` — a JS 32-bit shift (`<<`/`>>`/`>>>`).
    /// The operand is taken as 32 bits and the result re-extended (signed for
    /// `<<`/`>>`, unsigned for `>>>`); always within ±2^53, so no range guard.
    Shift32 { dst: u8, a: u8, b: u8, op: ShiftOp },
    /// if `reg[cond] == 0`, jump to op index `target`
    JumpIfFalse { cond: u8, target: usize },
    /// unconditional jump to op index `target`
    Jump { target: usize },
    /// return `reg[src]`
    Ret { src: u8 },
    /// `reg[dst] = call(code_ptr)(reg[args[0]], …, reg[args[n_args-1]])` — a
    /// native call to another compiled function at absolute `code_ptr`, System V
    /// ABI (integer args in rdi/rsi/rdx/rcx/r8/r9, result in rax). Programs with a
    /// `Call` skip the constant/allocation passes (which assume pure register ops)
    /// and are compiled directly.
    Call {
        /// destination register for the result
        dst: u8,
        /// absolute address of the callee's native code
        code_ptr: u64,
        /// number of arguments (≤ 6)
        n_args: u8,
        /// the argument registers (first `n_args` used)
        args: [u8; 6],
    },
}

/// Interprets a [`RegOp`] program (the oracle for [`JitFunction::compile_reg`]),
/// with a program counter so branches/loops are handled. `n_regs` registers;
/// `args` are the function arguments.
#[must_use]
pub fn eval_reg(ops: &[RegOp], n_regs: usize, args: &[i64]) -> i64 {
    let mut regs = alloc::vec![0i64; n_regs];
    let mut pc = 0usize;
    while pc < ops.len() {
        match ops[pc] {
            RegOp::Arg { dst, index } => {
                regs[dst as usize] = args.get(index as usize).copied().unwrap_or(0);
                pc += 1;
            }
            RegOp::Const { dst, imm } => {
                regs[dst as usize] = imm;
                pc += 1;
            }
            RegOp::Bin { dst, a, b, op } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                regs[dst as usize] = match op {
                    BinOp2::Add => x.wrapping_add(y),
                    BinOp2::Sub => x.wrapping_sub(y),
                    BinOp2::Mul => x.wrapping_mul(y),
                    BinOp2::And => x & y,
                    BinOp2::Or => x | y,
                    BinOp2::Xor => x ^ y,
                };
                pc += 1;
            }
            RegOp::Move { dst, src } => {
                regs[dst as usize] = regs[src as usize];
                pc += 1;
            }
            RegOp::Lt { dst, a, b } => {
                regs[dst as usize] = i64::from(regs[a as usize] < regs[b as usize]);
                pc += 1;
            }
            RegOp::Eqz { dst, a } => {
                regs[dst as usize] = i64::from(regs[a as usize] == 0);
                pc += 1;
            }
            RegOp::Eq { dst, a, b } => {
                regs[dst as usize] = i64::from(regs[a as usize] == regs[b as usize]);
                pc += 1;
            }
            RegOp::Neg { dst, a } => {
                regs[dst as usize] = regs[a as usize].wrapping_neg();
                pc += 1;
            }
            RegOp::BitNot32 { dst, a } => {
                regs[dst as usize] = i64::from(!(regs[a as usize] as i32));
                pc += 1;
            }
            RegOp::Mod { dst, a, b } => {
                let (x, y) = (regs[a as usize], regs[b as usize]);
                // The JIT deopts on y == 0; the oracle is only consulted for y != 0.
                regs[dst as usize] = if y == 0 { 0 } else { x.wrapping_rem(y) };
                pc += 1;
            }
            RegOp::Bit32 { dst, a, b, op } => {
                let (x, y) = (regs[a as usize] as i32, regs[b as usize] as i32);
                let v = match op {
                    BinOp2::And => x & y,
                    BinOp2::Or => x | y,
                    BinOp2::Xor => x ^ y,
                    _ => x & y,
                };
                regs[dst as usize] = i64::from(v);
                pc += 1;
            }
            RegOp::Shift32 { dst, a, b, op } => {
                let count = (regs[b as usize] as u32) & 31;
                regs[dst as usize] = match op {
                    ShiftOp::Shl => i64::from((regs[a as usize] as i32).wrapping_shl(count)),
                    ShiftOp::Sar => i64::from((regs[a as usize] as i32).wrapping_shr(count)),
                    ShiftOp::Shr => i64::from((regs[a as usize] as u32).wrapping_shr(count)),
                };
                pc += 1;
            }
            RegOp::JumpIfFalse { cond, target } => {
                if regs[cond as usize] == 0 {
                    pc = target;
                } else {
                    pc += 1;
                }
            }
            RegOp::Jump { target } => pc = target,
            RegOp::Ret { src } => return regs[src as usize],
            // `Call` invokes native code; the interpreter oracle is only used for
            // pure register programs (which never contain a Call).
            RegOp::Call { .. } => unreachable!("eval_reg does not evaluate Call ops"),
        }
    }
    0
}

/// The optimizing-JIT pass over the integer register IR: **constant folding**
/// (with propagation), then **copy propagation**, then **dead-code elimination**.
/// The result is observationally identical to the input — same [`eval_reg`] for
/// all inputs — but constant subexpressions are pre-computed, register copies are
/// forwarded to their source, and never-read computations are removed.
#[must_use]
pub fn optimize_reg(ops: &[RegOp], n_regs: usize) -> Vec<RegOp> {
    dce_reg(&copy_propagate(&fold_constants(ops, n_regs), n_regs))
}

/// Copy propagation: after `Move dst, src`, later reads of `dst` are rewritten to
/// read `src` directly (so the `Move` becomes dead and DCE removes it). Each
/// `Move` target tracks the *root* register it copies, so chains collapse in one
/// step. Sound across control flow: all copy relationships are cleared at any
/// jump target and after a branch, and an entry is invalidated when its source
/// register is overwritten.
#[must_use]
pub fn copy_propagate(ops: &[RegOp], n_regs: usize) -> Vec<RegOp> {
    let mut is_target = alloc::vec![false; ops.len()];
    for op in ops {
        if let RegOp::JumpIfFalse { target, .. } | RegOp::Jump { target } = op
            && *target < is_target.len()
        {
            is_target[*target] = true;
        }
    }
    // `copy_of[r] = Some(root)` means `r` currently equals register `root`.
    let mut copy_of: Vec<Option<u8>> = alloc::vec![None; n_regs];
    let resolve = |copy_of: &[Option<u8>], r: u8| copy_of[r as usize].unwrap_or(r);
    // Invalidate any copy whose source is `w` (it is about to change), and clear
    // `w`'s own copy status.
    let invalidate = |copy_of: &mut [Option<u8>], w: u8| {
        for c in copy_of.iter_mut() {
            if *c == Some(w) {
                *c = None;
            }
        }
        copy_of[w as usize] = None;
    };
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if is_target[i] {
            copy_of.iter_mut().for_each(|c| *c = None);
        }
        // Rewrite source operands to their roots.
        let rewritten = match *op {
            RegOp::Bin { dst, a, b, op } => RegOp::Bin {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
                op,
            },
            RegOp::Lt { dst, a, b } => RegOp::Lt {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
            },
            RegOp::Eqz { dst, a } => RegOp::Eqz {
                dst,
                a: resolve(&copy_of, a),
            },
            RegOp::Eq { dst, a, b } => RegOp::Eq {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
            },
            RegOp::Neg { dst, a } => RegOp::Neg {
                dst,
                a: resolve(&copy_of, a),
            },
            RegOp::BitNot32 { dst, a } => RegOp::BitNot32 {
                dst,
                a: resolve(&copy_of, a),
            },
            RegOp::Mod { dst, a, b } => RegOp::Mod {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
            },
            RegOp::Bit32 { dst, a, b, op } => RegOp::Bit32 {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
                op,
            },
            RegOp::Shift32 { dst, a, b, op } => RegOp::Shift32 {
                dst,
                a: resolve(&copy_of, a),
                b: resolve(&copy_of, b),
                op,
            },
            RegOp::Move { dst, src } => RegOp::Move {
                dst,
                src: resolve(&copy_of, src),
            },
            RegOp::JumpIfFalse { cond, target } => RegOp::JumpIfFalse {
                cond: resolve(&copy_of, cond),
                target,
            },
            RegOp::Ret { src } => RegOp::Ret {
                src: resolve(&copy_of, src),
            },
            other => other,
        };
        // Update copy state from the (rewritten) op's destination.
        match rewritten {
            RegOp::Move { dst, src } => {
                invalidate(&mut copy_of, dst);
                if src != dst {
                    copy_of[dst as usize] = Some(src);
                }
            }
            RegOp::Const { dst, .. }
            | RegOp::Bin { dst, .. }
            | RegOp::Lt { dst, .. }
            | RegOp::Eqz { dst, .. }
            | RegOp::Eq { dst, .. }
            | RegOp::Neg { dst, .. }
            | RegOp::BitNot32 { dst, .. }
            | RegOp::Mod { dst, .. }
            | RegOp::Bit32 { dst, .. }
            | RegOp::Shift32 { dst, .. }
            | RegOp::Call { dst, .. }
            | RegOp::Arg { dst, .. } => invalidate(&mut copy_of, dst),
            RegOp::JumpIfFalse { .. } | RegOp::Jump { .. } => {
                copy_of.iter_mut().for_each(|c| *c = None);
            }
            RegOp::Ret { .. } => {}
        }
        out.push(rewritten);
    }
    out
}

/// Removes ops whose destination register is never read (anywhere), then remaps
/// branch targets to the surviving instructions. Sound: a never-read result
/// cannot affect the function's value, and a dropped arithmetic op only means the
/// native code no longer deopts on its (unused) overflow — the final value is
/// unchanged. `Ret`/`Jump`/`JumpIfFalse` are always kept; a jump to a removed op
/// is retargeted to the next surviving op (the removed op had no effect).
#[must_use]
pub fn dce_reg(ops: &[RegOp]) -> Vec<RegOp> {
    use alloc::collections::BTreeSet;
    let mut used: BTreeSet<u8> = BTreeSet::new();
    for op in ops {
        match *op {
            RegOp::Bin { a, b, .. }
            | RegOp::Lt { a, b, .. }
            | RegOp::Eq { a, b, .. }
            | RegOp::Mod { a, b, .. }
            | RegOp::Bit32 { a, b, .. }
            | RegOp::Shift32 { a, b, .. } => {
                used.insert(a);
                used.insert(b);
            }
            RegOp::Move { src, .. }
            | RegOp::Eqz { a: src, .. }
            | RegOp::Neg { a: src, .. }
            | RegOp::BitNot32 { a: src, .. } => {
                used.insert(src);
            }
            RegOp::JumpIfFalse { cond, .. } => {
                used.insert(cond);
            }
            RegOp::Ret { src } => {
                used.insert(src);
            }
            RegOp::Call { n_args, args, .. } => {
                for a in &args[..n_args as usize] {
                    used.insert(*a);
                }
            }
            RegOp::Const { .. } | RegOp::Arg { .. } | RegOp::Jump { .. } => {}
        }
    }
    let keep = |op: &RegOp| match *op {
        // A `Call` has side effects (it invokes a function), so always keep it.
        RegOp::Ret { .. } | RegOp::Jump { .. } | RegOp::JumpIfFalse { .. } | RegOp::Call { .. } => {
            true
        }
        RegOp::Const { dst, .. }
        | RegOp::Bin { dst, .. }
        | RegOp::Move { dst, .. }
        | RegOp::Lt { dst, .. }
        | RegOp::Eqz { dst, .. }
        | RegOp::Eq { dst, .. }
        | RegOp::Neg { dst, .. }
        | RegOp::BitNot32 { dst, .. }
        | RegOp::Mod { dst, .. }
        | RegOp::Bit32 { dst, .. }
        | RegOp::Shift32 { dst, .. }
        | RegOp::Arg { dst, .. } => used.contains(&dst),
    };
    // `newpos[old]` = new index of the first surviving op at-or-after `old`.
    let mut newpos = alloc::vec![0usize; ops.len() + 1];
    let mut n = 0;
    for (i, op) in ops.iter().enumerate() {
        newpos[i] = n;
        if keep(op) {
            n += 1;
        }
    }
    newpos[ops.len()] = n;
    let mut out = Vec::with_capacity(n);
    for op in ops.iter().filter(|o| keep(o)) {
        out.push(match *op {
            RegOp::JumpIfFalse { cond, target } => RegOp::JumpIfFalse {
                cond,
                target: newpos.get(target).copied().unwrap_or(n),
            },
            RegOp::Jump { target } => RegOp::Jump {
                target: newpos.get(target).copied().unwrap_or(n),
            },
            other => other,
        });
    }
    out
}

/// The registers an op reads or writes (`dst` first if it has one).
fn op_regs(op: &RegOp) -> (Option<u8>, [Option<u8>; 2]) {
    match *op {
        RegOp::Arg { dst, .. } | RegOp::Const { dst, .. } => (Some(dst), [None, None]),
        RegOp::Move { dst, src } => (Some(dst), [Some(src), None]),
        RegOp::Bin { dst, a, b, .. }
        | RegOp::Lt { dst, a, b }
        | RegOp::Eq { dst, a, b }
        | RegOp::Mod { dst, a, b }
        | RegOp::Bit32 { dst, a, b, .. }
        | RegOp::Shift32 { dst, a, b, .. } => (Some(dst), [Some(a), Some(b)]),
        RegOp::Eqz { dst, a } | RegOp::Neg { dst, a } | RegOp::BitNot32 { dst, a } => {
            (Some(dst), [Some(a), None])
        }
        RegOp::JumpIfFalse { cond, .. } => (None, [Some(cond), None]),
        RegOp::Ret { src } => (None, [Some(src), None]),
        RegOp::Jump { .. } => (None, [None, None]),
        // Call programs skip allocation, so this is only for exhaustiveness.
        RegOp::Call { dst, .. } => (Some(dst), [None, None]),
    }
}

/// A **linear-scan register allocator**: computes each virtual register's live
/// interval (first..=last instruction it appears in) and reassigns registers so
/// that values whose intervals don't overlap share storage. Returns the rewritten
/// program and the reduced register count. The interval is taken over linear
/// instruction order — conservative across branches/loops (it never aliases two
/// values that are simultaneously live), so the result is observationally
/// identical with a smaller frame. This is the classic allocator algorithm; here
/// the "registers" it colors are the frame slots `compile_reg` emits.
#[must_use]
pub fn allocate_reg(ops: &[RegOp], n_regs: usize) -> (Vec<RegOp>, usize) {
    // Live interval [first, last] (in linear order) for each used register.
    let mut first = alloc::vec![usize::MAX; n_regs];
    let mut last = alloc::vec![0usize; n_regs];
    for (i, op) in ops.iter().enumerate() {
        let (dst, srcs) = op_regs(op);
        for r in dst.into_iter().chain(srcs.into_iter().flatten()) {
            let r = r as usize;
            first[r] = first[r].min(i);
            last[r] = last[r].max(i);
        }
    }
    // Loop-aware extension: a register live across a loop body `[target, j]` (a
    // backward branch at `j` to `target <= j`) must hold its slot for the whole
    // loop — otherwise a register defined inside the loop could reuse its slot and
    // clobber the value the next iteration reads. Extend such intervals to span
    // the loop, iterating to a fixpoint to handle nesting.
    loop {
        let mut changed = false;
        for (j, op) in ops.iter().enumerate() {
            let (RegOp::Jump { target } | RegOp::JumpIfFalse { target, .. }) = *op else {
                continue;
            };
            if target > j {
                continue; // forward branch — not a loop back-edge
            }
            for r in 0..n_regs {
                if first[r] != usize::MAX && first[r] <= j && last[r] >= target {
                    let (nf, nl) = (first[r].min(target), last[r].max(j));
                    if nf != first[r] || nl != last[r] {
                        first[r] = nf;
                        last[r] = nl;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Intervals of registers that actually appear, sorted by start.
    let mut intervals: Vec<(usize, usize, u8)> = (0..n_regs)
        .filter(|&r| first[r] != usize::MAX)
        .map(|r| (first[r], last[r], r as u8))
        .collect();
    intervals.sort_unstable();

    // Linear scan: `slot_free_at[s]` is the instruction index after which slot `s`
    // is free again (one past its current occupant's last use).
    let mut mapping = alloc::vec![0u8; n_regs];
    let mut slot_end: Vec<usize> = Vec::new(); // slot -> last index it's busy until
    for (start, end, vreg) in intervals {
        // Find a slot free at `start` (its occupant's interval ended before).
        let slot = slot_end.iter().position(|&e| e < start).unwrap_or_else(|| {
            slot_end.push(0);
            slot_end.len() - 1
        });
        slot_end[slot] = end;
        mapping[vreg as usize] = slot as u8;
    }
    let new_n = slot_end.len().max(1);

    // Rewrite the program with the slot assignment.
    let m = |r: u8| mapping[r as usize];
    let out = ops
        .iter()
        .map(|op| match *op {
            RegOp::Arg { dst, index } => RegOp::Arg { dst: m(dst), index },
            RegOp::Const { dst, imm } => RegOp::Const { dst: m(dst), imm },
            RegOp::Move { dst, src } => RegOp::Move {
                dst: m(dst),
                src: m(src),
            },
            RegOp::Bin { dst, a, b, op } => RegOp::Bin {
                dst: m(dst),
                a: m(a),
                b: m(b),
                op,
            },
            RegOp::Lt { dst, a, b } => RegOp::Lt {
                dst: m(dst),
                a: m(a),
                b: m(b),
            },
            RegOp::Eqz { dst, a } => RegOp::Eqz {
                dst: m(dst),
                a: m(a),
            },
            RegOp::Eq { dst, a, b } => RegOp::Eq {
                dst: m(dst),
                a: m(a),
                b: m(b),
            },
            RegOp::Neg { dst, a } => RegOp::Neg {
                dst: m(dst),
                a: m(a),
            },
            RegOp::BitNot32 { dst, a } => RegOp::BitNot32 {
                dst: m(dst),
                a: m(a),
            },
            RegOp::Mod { dst, a, b } => RegOp::Mod {
                dst: m(dst),
                a: m(a),
                b: m(b),
            },
            RegOp::Bit32 { dst, a, b, op } => RegOp::Bit32 {
                dst: m(dst),
                a: m(a),
                b: m(b),
                op,
            },
            RegOp::Shift32 { dst, a, b, op } => RegOp::Shift32 {
                dst: m(dst),
                a: m(a),
                b: m(b),
                op,
            },
            RegOp::JumpIfFalse { cond, target } => RegOp::JumpIfFalse {
                cond: m(cond),
                target,
            },
            RegOp::Jump { target } => RegOp::Jump { target },
            RegOp::Ret { src } => RegOp::Ret { src: m(src) },
            RegOp::Call {
                dst,
                code_ptr,
                n_args,
                args,
            } => RegOp::Call {
                dst: m(dst),
                code_ptr,
                n_args,
                args: args.map(m),
            },
        })
        .collect();
    (out, new_n)
}

/// Constant folding with propagation: a `Bin`/`Lt`/`Move` over known-constant
/// operands becomes a `Const`. Stays sound across control flow by clearing all
/// known constants at any jump target and after a branch.
#[must_use]
pub fn fold_constants(ops: &[RegOp], n_regs: usize) -> Vec<RegOp> {
    // Every op index that is a branch target: constants can't be assumed there.
    let mut is_target = alloc::vec![false; ops.len()];
    for op in ops {
        if let RegOp::JumpIfFalse { target, .. } | RegOp::Jump { target } = op
            && *target < is_target.len()
        {
            is_target[*target] = true;
        }
    }
    let mut known: Vec<Option<i64>> = alloc::vec![None; n_regs];
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if is_target[i] {
            known.iter_mut().for_each(|k| *k = None);
        }
        let lowered = match *op {
            RegOp::Const { dst, imm } => {
                known[dst as usize] = Some(imm);
                RegOp::Const { dst, imm }
            }
            RegOp::Move { dst, src } => {
                if let Some(v) = known[src as usize] {
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Move { dst, src }
                }
            }
            RegOp::Bin { dst, a, b, op } => {
                let (ka, kb) = (known[a as usize], known[b as usize]);
                if let (Some(x), Some(y)) = (ka, kb) {
                    // Both constant: fold, but only when the result stays in the
                    // exact-integer range so the native overflow/range deopt is
                    // still reachable for genuinely out-of-range arithmetic.
                    let v = match op {
                        BinOp2::Add => x.wrapping_add(y),
                        BinOp2::Sub => x.wrapping_sub(y),
                        BinOp2::Mul => x.wrapping_mul(y),
                        BinOp2::And => x & y,
                        BinOp2::Or => x | y,
                        BinOp2::Xor => x ^ y,
                    };
                    if (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&v) {
                        known[dst as usize] = Some(v);
                        RegOp::Const { dst, imm: v }
                    } else {
                        known[dst as usize] = None;
                        RegOp::Bin { dst, a, b, op }
                    }
                } else if let Some(simplified) = simplify_bin(dst, a, b, op, ka, kb, &mut known) {
                    // Algebraic identity / strength reduction (x+0, x*1, x*0,
                    // x-x, x^x, …). All operands are already in ±2^53, so the
                    // simplified form needs no extra guard.
                    simplified
                } else {
                    known[dst as usize] = None;
                    RegOp::Bin { dst, a, b, op }
                }
            }
            RegOp::Lt { dst, a, b } => {
                if let (Some(x), Some(y)) = (known[a as usize], known[b as usize]) {
                    let v = i64::from(x < y);
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Lt { dst, a, b }
                }
            }
            RegOp::Eqz { dst, a } => {
                if let Some(x) = known[a as usize] {
                    let v = i64::from(x == 0);
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Eqz { dst, a }
                }
            }
            RegOp::Eq { dst, a, b } => {
                if let (Some(x), Some(y)) = (known[a as usize], known[b as usize]) {
                    let v = i64::from(x == y);
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Eq { dst, a, b }
                }
            }
            RegOp::Neg { dst, a } => {
                // Fold only when the negation stays in the exact-integer range, so
                // the native overflow guard remains reachable otherwise.
                match known[a as usize] {
                    Some(x) if (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&x.wrapping_neg()) => {
                        let v = x.wrapping_neg();
                        known[dst as usize] = Some(v);
                        RegOp::Const { dst, imm: v }
                    }
                    _ => {
                        known[dst as usize] = None;
                        RegOp::Neg { dst, a }
                    }
                }
            }
            RegOp::BitNot32 { dst, a } => {
                if let Some(x) = known[a as usize] {
                    let v = i64::from(!(x as i32));
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::BitNot32 { dst, a }
                }
            }
            RegOp::Mod { dst, a, b } => {
                // Fold only when the divisor is a known non-zero constant (a zero
                // divisor must still reach the native deopt, not be folded).
                match (known[a as usize], known[b as usize]) {
                    (Some(x), Some(y)) if y != 0 && x >= 0 => {
                        let v = x.wrapping_rem(y);
                        known[dst as usize] = Some(v);
                        RegOp::Const { dst, imm: v }
                    }
                    _ => {
                        known[dst as usize] = None;
                        RegOp::Mod { dst, a, b }
                    }
                }
            }
            RegOp::Bit32 { dst, a, b, op } => {
                if let (Some(x), Some(y)) = (known[a as usize], known[b as usize]) {
                    let (x, y) = (x as i32, y as i32);
                    let v = i64::from(match op {
                        BinOp2::And => x & y,
                        BinOp2::Or => x | y,
                        BinOp2::Xor => x ^ y,
                        _ => x & y,
                    });
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Bit32 { dst, a, b, op }
                }
            }
            RegOp::Shift32 { dst, a, b, op } => {
                if let (Some(x), Some(y)) = (known[a as usize], known[b as usize]) {
                    let count = (y as u32) & 31;
                    let v = match op {
                        ShiftOp::Shl => i64::from((x as i32).wrapping_shl(count)),
                        ShiftOp::Sar => i64::from((x as i32).wrapping_shr(count)),
                        ShiftOp::Shr => i64::from((x as u32).wrapping_shr(count)),
                    };
                    known[dst as usize] = Some(v);
                    RegOp::Const { dst, imm: v }
                } else {
                    known[dst as usize] = None;
                    RegOp::Shift32 { dst, a, b, op }
                }
            }
            RegOp::Arg { dst, .. } => {
                known[dst as usize] = None;
                *op
            }
            // Branches end a straight-line region; clear constants conservatively.
            RegOp::JumpIfFalse { .. } | RegOp::Jump { .. } => {
                known.iter_mut().for_each(|k| *k = None);
                *op
            }
            RegOp::Ret { .. } => *op,
            RegOp::Call { dst, .. } => {
                known[dst as usize] = None;
                *op
            }
        };
        out.push(lowered);
    }
    out
}

/// Algebraic identities / strength reduction for `dst = a <op> b`, given which
/// operands are known constants (`ka`/`kb`). Returns the simplified op (a `Move`
/// or `Const`) and updates `known[dst]`, or `None` if no identity applies. Every
/// register value is already within ±2^53, so the simplified forms are exact.
fn simplify_bin(
    dst: u8,
    a: u8,
    b: u8,
    op: BinOp2,
    ka: Option<i64>,
    kb: Option<i64>,
    known: &mut [Option<i64>],
) -> Option<RegOp> {
    use BinOp2::{Add, And, Mul, Or, Sub, Xor};
    // `dst = a` (a Move forwarding the value of register `r`).
    let mov = |known: &mut [Option<i64>], r: u8| {
        known[dst as usize] = known[r as usize];
        Some(RegOp::Move { dst, src: r })
    };
    // `dst = c` (a constant).
    let con = |known: &mut [Option<i64>], c: i64| {
        known[dst as usize] = Some(c);
        Some(RegOp::Const { dst, imm: c })
    };
    // Identities with a constant right operand.
    if let Some(y) = kb {
        match (op, y) {
            (Add, 0) | (Sub, 0) | (Or, 0) | (Xor, 0) | (Mul, 1) | (And, -1) => {
                return mov(known, a);
            }
            (Mul, 0) | (And, 0) => return con(known, 0),
            (Or, -1) => return con(known, -1),
            _ => {}
        }
    }
    // Identities with a constant left operand (commutative ops).
    if let Some(x) = ka {
        match (op, x) {
            (Add, 0) | (Or, 0) | (Xor, 0) | (Mul, 1) | (And, -1) => return mov(known, b),
            (Mul, 0) | (And, 0) => return con(known, 0),
            (Or, -1) => return con(known, -1),
            _ => {}
        }
    }
    // Identities on the same register: x-x = 0, x^x = 0, x&x = x, x|x = x.
    if a == b {
        match op {
            Sub | Xor => return con(known, 0),
            And | Or => return mov(known, a),
            _ => {}
        }
    }
    None
}

/// If `v` is a `NanBox` whole number that fits an `i64` losslessly, its integer
/// value — the JIT integer fast path only applies to such constants. `-0` is
/// not one: an `i64` has no negative zero, so it would come back as `+0`.
#[cfg(feature = "alloc")]
#[inline(always)]
fn nanbox_int(v: crate::nanbox::NanBox) -> Option<i64> {
    let n = v.as_number()?;
    // Within the exact-integer range (±2^53; false for NaN and the
    // infinities) first, so the conversion below needs no saturation.
    if !(-9.007_199_254_740_992e15..=9.007_199_254_740_992e15).contains(&n) {
        return None;
    }
    let i = n as i64;
    // A lossless round-trip, and not -0.
    if (i as f64) != n || (i == 0 && n.is_sign_negative()) {
        return None;
    }
    Some(i)
}

/// The register an op the Int tier lowers writes (`None` for the others).
#[cfg(feature = "alloc")]
fn int_tier_dst(op: &crate::nbvm::Op) -> Option<crate::nbvm::Reg> {
    use crate::nbvm::Op;
    match op {
        Op::LoadConst { dst, .. }
        | Op::Add { dst, .. }
        | Op::AddValue { dst, .. }
        | Op::Sub { dst, .. }
        | Op::Mul { dst, .. }
        | Op::Mod { dst, .. }
        | Op::Move { dst, .. }
        | Op::Lt { dst, .. }
        | Op::Not { dst, .. }
        | Op::StrictEq { dst, .. }
        | Op::ValueBin { dst, .. }
        | Op::Neg { dst, .. }
        | Op::BitNot { dst, .. }
        | Op::Call { dst, .. } => Some(*dst),
        _ => None,
    }
}

/// Lowers a real bytecode-VM function (`nbvm::FnProto`) to the JIT's register IR,
/// **iff** it is a straight-line integer function the baseline JIT can handle:
/// no captures, ≤ 64 registers, ≤ 6 params, and only `LoadConst` (of an integer),
/// `Add`/`Sub`/`Mul`, `Move`, and a terminating `Return`. Any other op (calls,
/// branches, property access, non-integer constants, …) makes it return `None`,
/// so the caller falls back to the interpreter.
///
/// This is the bridge from the VM's instruction stream to native code: a `proto`
/// produced by `nbvm::compile_program` over real JS source can be compiled with
/// [`JitFunction::compile_reg`] and run natively.
#[cfg(feature = "alloc")]
#[must_use]
pub fn lower_nbvm(proto: &crate::nbvm::FnProto) -> Option<Vec<RegOp>> {
    lower_nbvm_with(proto, &alloc::collections::BTreeMap::new())
}

/// Whether the unboxed numeric tiers (Int/Float) can represent `proto`'s
/// booleans. Those tiers compute a comparison (`<`, `===`, `==`, `!`) as a
/// native `0`/`1`, which is exact only while the result is consumed as a
/// *condition* (`JumpIfFalse`, `!`) or copied between such registers; boxed as a
/// return value it would become the number `0`/`1`, and fed to arithmetic or a
/// comparison it would compare as a number (`(a < b) === 1` is `false` in JS).
/// So every register a comparison writes (closed over `Move`) must never also
/// hold a number and must only be read by a branch, a `!` or a `Move`;
/// otherwise the function is left to the generic tier, which boxes real
/// booleans.
///
/// With `bool_return`, a body whose *every* `return` returns such a register
/// (a predicate, `x => x % 3 === 0`) is sound too: its result is then always
/// a native `0`/`1` that the caller boxes as `false`/`true`
/// ([`JitProto::returns_bool`]). `Some(returns_bool)` when sound.
#[cfg(feature = "alloc")]
fn numeric_tier_booleans(proto: &crate::nbvm::FnProto, bool_return: bool) -> Option<bool> {
    use crate::nbvm::Op;
    let mut returns = (false, false); // (a boolean, a number)
    let ok = numeric_tier_booleans_check(proto, &mut |is_bool| {
        if is_bool {
            returns.0 = true;
        } else {
            returns.1 = true;
        }
    });
    if !ok {
        return None;
    }
    match returns {
        (false, _) => Some(false),
        (true, false) if bool_return => {
            // Every exit must be a `Return` (no fall-off `undefined`).
            proto
                .ops
                .iter()
                .any(|op| matches!(op, Op::Return { .. }))
                .then_some(true)
        }
        _ => None,
    }
}

/// [`numeric_tier_booleans`] without the return rule: `on_return(is_bool)` is
/// told of each `return`'s register kind.
#[cfg(feature = "alloc")]
fn numeric_tier_booleans_check(
    proto: &crate::nbvm::FnProto,
    on_return: &mut dyn FnMut(bool),
) -> bool {
    use crate::nbvm::Op;
    let n = proto.n_regs;
    let mut boolish = alloc::vec![false; n];
    let is_bool_op = |op: &Op| match op {
        Op::Lt { dst, .. } | Op::Not { dst, .. } | Op::StrictEq { dst, .. } => Some(*dst),
        Op::ValueBin { dst, op, .. } if *op == crate::nbvm::VB_LOOSE_EQ => Some(*dst),
        _ => None,
    };
    for op in &proto.ops {
        if let Some(d) = is_bool_op(op)
            && let Some(b) = boolish.get_mut(d as usize)
        {
            *b = true;
        }
    }
    // Close over moves (a bounded fixpoint: each round adds a register or stops).
    loop {
        let mut changed = false;
        for op in &proto.ops {
            if let Op::Move { dst, src } = op
                && boolish.get(*src as usize) == Some(&true)
                && boolish.get(*dst as usize) == Some(&false)
            {
                boolish[*dst as usize] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let is_b = |r: crate::nbvm::Reg| boolish.get(r as usize) == Some(&true);
    // A parameter is a number on these tiers.
    if (0..proto.n_params.min(n)).any(|r| boolish[r]) {
        return false;
    }
    for op in &proto.ops {
        match op {
            Op::JumpIfFalse { .. } => {}
            Op::Not { .. } => {}
            Op::Return { src } => on_return(is_b(*src)),
            Op::Move { dst, src } => {
                if is_b(*dst) != is_b(*src) {
                    return false; // a boolean register also receives a number
                }
            }
            _ if is_bool_op(op).is_some() => {
                // `<`/`===`/`==` read their operands as numbers.
                let reads_bool = match op {
                    Op::Lt { a, b, .. } | Op::StrictEq { a, b, .. } | Op::ValueBin { a, b, .. } => {
                        is_b(*a) || is_b(*b)
                    }
                    _ => false,
                };
                if reads_bool {
                    return false;
                }
            }
            _ => {
                if (0..n).any(|r| {
                    is_b(r as crate::nbvm::Reg)
                        && crate::bytecode::op_mentions_reg(op, r as crate::nbvm::Reg)
                }) {
                    return false;
                }
            }
        }
    }
    true
}

/// Like [`lower_nbvm`], but `registry` maps a callee function-table index to the
/// absolute address of its already-compiled native code, so an `Op::Call` to a
/// registered (JIT-compiled) function lowers to a native [`RegOp::Call`]. A call
/// to an unregistered function still bails (returns `None`).
#[must_use]
pub fn lower_nbvm_with(
    proto: &crate::nbvm::FnProto,
    registry: &alloc::collections::BTreeMap<u32, u64>,
) -> Option<Vec<RegOp>> {
    use crate::nbvm::Op;
    if proto.n_regs > 64
        || proto.n_params > 6
        || proto.n_captures != 0
        || numeric_tier_booleans(proto, true).is_none()
    {
        return None;
    }
    let reg8 = |r: crate::nbvm::Reg| -> Option<u8> {
        if (r as usize) < proto.n_regs {
            u8::try_from(r).ok()
        } else {
            None
        }
    };
    let mut out = Vec::new();
    // Def-use safety: a register may be read only after it has been written
    // (params count as written). This rejects any function that reads an
    // uninitialized slot — `this`, a capture, a hoisted/TDZ binding — which the
    // native frame does not hold, so the JIT can't silently diverge from the
    // interpreter's `undefined`.
    let mut written = alloc::vec![false; proto.n_regs];
    for w in written.iter_mut().take(proto.n_params) {
        *w = true;
    }
    let read = |w: &[bool], r: crate::nbvm::Reg| -> Option<u8> {
        let r8 = reg8(r)?;
        if *w.get(r as usize)? { Some(r8) } else { None }
    };
    // The registers only ever written by `LoadConst` of a positive integer. An
    // integer product is a JS `-0` exactly when it is zero with a negative
    // factor, which a positive constant factor rules out (the other factor is
    // never `-0`: the guards and constants reject it) — so only such a `*` is
    // lowered; any other one is left to a tier that keeps the sign of zero.
    let pos_const: Vec<bool> = (0..proto.n_regs)
        .map(|r| {
            let mut writes = proto
                .ops
                .iter()
                .filter(|o| int_tier_dst(o).is_some_and(|d| d as usize == r))
                .peekable();
            r >= proto.n_params
                && writes.peek().is_some()
                && writes.all(|o| {
                    matches!(o, Op::LoadConst { value, .. }
                        if nanbox_int(*value).is_some_and(|c| c > 0))
                })
        })
        .collect();
    // Parameters arrive in registers `0..n_params`; seed them from the args.
    for i in 0..proto.n_params {
        out.push(RegOp::Arg {
            dst: u8::try_from(i).ok()?,
            index: u8::try_from(i).ok()?,
        });
    }
    for op in &proto.ops {
        let lowered = match op {
            Op::LoadConst { dst, value } => {
                let imm = nanbox_int(*value)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Const { dst: d, imm }
            }
            // `Add` is the numeric-typed add; `AddValue` is the general `+`
            // (string-or-number). The integer fast path treats both as integer
            // addition — the range/overflow guards keep it sound.
            Op::Add { dst, a, b } | Op::AddValue { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Bin {
                    dst: d,
                    a,
                    b,
                    op: BinOp2::Add,
                }
            }
            Op::Sub { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Bin {
                    dst: d,
                    a,
                    b,
                    op: BinOp2::Sub,
                }
            }
            Op::Mul { dst, a, b }
                if pos_const.get(*a as usize) == Some(&true)
                    || pos_const.get(*b as usize) == Some(&true) =>
            {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Bin {
                    dst: d,
                    a,
                    b,
                    op: BinOp2::Mul,
                }
            }
            // Integer remainder `%`. The native code deopts on a zero divisor
            // (JS yields NaN) and on a non-integer/overflowing result via guards.
            Op::Mod { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Mod { dst: d, a, b }
            }
            Op::Move { dst, src } => {
                let s = read(&written, *src)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Move { dst: d, src: s }
            }
            Op::Lt { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Lt { dst: d, a, b }
            }
            // JS logical-not of an integer: `!x == (x == 0)`. Sound on the integer
            // path, where every register value is an exact guarded integer — so
            // `<=`/`>=`/`!==` (compiled to `Lt`/`StrictEq` + `Not`) now JIT.
            Op::Not { dst, a } => {
                let a = read(&written, *a)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Eqz { dst: d, a }
            }
            // `===` of two integers is numeric equality (both are guarded exact
            // integers on this path); `!==` is this followed by `Not` (→ `Eqz`).
            Op::StrictEq { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Eq { dst: d, a, b }
            }
            // Loose `==` between two integers coerces to numeric equality — the
            // same as `===` on this all-integer path.
            Op::ValueBin { dst, op, a, b } if *op == crate::nbvm::VB_LOOSE_EQ => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Eq { dst: d, a, b }
            }
            // JS unary minus is not lowered: `-0` (of `0`) has no `i64` form.
            // JS bitwise-not `~x` = `~ToInt32(x)`; the native code truncates to i32.
            Op::BitNot { dst, a } => {
                let a = read(&written, *a)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::BitNot32 { dst: d, a }
            }
            // JS 32-bit bitwise `&`/`|`/`^`: ToInt32 both operands then combine.
            // The native code truncates to i32 first, so it matches JS semantics
            // for any in-range integer operand (i64 bitwise would not).
            Op::ValueBin { dst, op, a, b }
                if matches!(
                    *op,
                    crate::nbvm::VB_BIT_AND | crate::nbvm::VB_BIT_OR | crate::nbvm::VB_BIT_XOR
                ) =>
            {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                let bop = match *op {
                    crate::nbvm::VB_BIT_AND => BinOp2::And,
                    crate::nbvm::VB_BIT_OR => BinOp2::Or,
                    _ => BinOp2::Xor,
                };
                RegOp::Bit32 {
                    dst: d,
                    a,
                    b,
                    op: bop,
                }
            }
            // JS 32-bit shifts `<<`/`>>`/`>>>` (count masked to 5 bits).
            Op::ValueBin { dst, op, a, b }
                if matches!(
                    *op,
                    crate::nbvm::VB_SHL | crate::nbvm::VB_SHR | crate::nbvm::VB_USHR
                ) =>
            {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                let sop = match *op {
                    crate::nbvm::VB_SHL => ShiftOp::Shl,
                    crate::nbvm::VB_SHR => ShiftOp::Sar,
                    _ => ShiftOp::Shr,
                };
                RegOp::Shift32 {
                    dst: d,
                    a,
                    b,
                    op: sop,
                }
            }
            // Branch targets are nbvm op indices; the lowered stream prepends one
            // `Arg` per parameter, so every target shifts by `n_params`.
            Op::JumpIfFalse { cond, target } => RegOp::JumpIfFalse {
                cond: read(&written, *cond)?,
                target: target.checked_add(proto.n_params)?,
            },
            Op::Jump { target } => RegOp::Jump {
                target: target.checked_add(proto.n_params)?,
            },
            Op::Return { src } => RegOp::Ret {
                src: read(&written, *src)?,
            },
            // A static call to an already-compiled function lowers to a native
            // call; an unregistered callee bails the whole compilation.
            Op::Call { dst, func, args } => {
                if args.len() > 6 {
                    return None;
                }
                let code_ptr = registry.get(func).copied()?;
                let mut argregs = [0u8; 6];
                for (i, r) in args.iter().enumerate() {
                    argregs[i] = read(&written, *r)?;
                }
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Call {
                    dst: d,
                    code_ptr,
                    n_args: args.len() as u8,
                    args: argregs,
                }
            }
            _ => return None,
        };
        out.push(lowered);
    }
    // Eligible only if it terminates (the last op returns, so control never falls
    // off the end of the emitted code) and every branch target is in range.
    if !matches!(out.last(), Some(RegOp::Ret { .. })) {
        return None;
    }
    for op in &out {
        if let RegOp::JumpIfFalse { target, .. } | RegOp::Jump { target } = op
            && *target >= out.len()
        {
            return None;
        }
    }
    Some(out)
}

/// `f64` value of a numeric `NanBox` constant (any finite number — the float
/// path, unlike the integer path, has no range restriction).
#[cfg(feature = "alloc")]
fn nanbox_f64(v: crate::nanbox::NanBox) -> Option<f64> {
    match v.unpack() {
        crate::nanbox::Unpacked::Number(n) if n.is_finite() => Some(n),
        _ => None,
    }
}

/// Lowers a real bytecode-VM function to the **float** register IR — the `f64`
/// fast path. Eligible iff it is straight-line numeric arithmetic: no captures,
/// ≤ 64 registers, ≤ 4 params, only `Const` (any finite number),
/// `Add`/`AddValue`/`Sub`/`Mul`/`Div`, `Move`, and a terminating `Return`. This
/// covers `/` and non-integer values, which the integer path
/// ([`lower_nbvm`]) rejects, but not branches/loops (the float compiler is
/// Whether this CPU supports SSE4.1 (needed for `roundsd`). `is_x86_feature_detected!`
/// only exists on x86 targets, so it is `cfg`-gated; everywhere else this is `false`
/// (the JIT only emits machine code on x86-64 Linux anyway).
#[cfg(all(feature = "alloc", target_arch = "x86_64"))]
fn has_sse41() -> bool {
    std::is_x86_feature_detected!("sse4.1")
}
#[cfg(all(feature = "alloc", not(target_arch = "x86_64")))]
fn has_sse41() -> bool {
    false
}

/// straight-line). Returns `None` otherwise, with the same def-use safety check.
#[cfg(feature = "alloc")]
#[must_use]
pub fn lower_nbvm_float(proto: &crate::nbvm::FnProto) -> Option<Vec<FloatOp>> {
    use crate::nbvm::Op;
    if proto.n_regs > 64
        || proto.n_params > 4
        || proto.n_captures != 0
        || numeric_tier_booleans(proto, false).is_none()
    {
        return None;
    }
    let reg8 = |r: crate::nbvm::Reg| -> Option<u8> {
        ((r as usize) < proto.n_regs).then(|| u8::try_from(r).ok())?
    };
    let mut written = alloc::vec![false; proto.n_regs];
    for w in written.iter_mut().take(proto.n_params) {
        *w = true;
    }
    let read = |w: &[bool], r: crate::nbvm::Reg| -> Option<u8> {
        let r8 = reg8(r)?;
        if *w.get(r as usize)? { Some(r8) } else { None }
    };
    let mut out = Vec::new();
    for i in 0..proto.n_params {
        out.push(FloatOp::Arg {
            dst: u8::try_from(i).ok()?,
            index: u8::try_from(i).ok()?,
        });
    }
    let bin = |w: &mut [bool], dst, a, b, op| -> Option<FloatOp> {
        let (a, b) = (read(w, a)?, read(w, b)?);
        let d = reg8(dst)?;
        w[dst as usize] = true;
        Some(FloatOp::Bin { dst: d, a, b, op })
    };
    for op in &proto.ops {
        let lowered = match op {
            Op::LoadConst { dst, value } => {
                let imm = nanbox_f64(*value)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Const { dst: d, imm }
            }
            Op::Add { dst, a, b } | Op::AddValue { dst, a, b } => {
                bin(&mut written, *dst, *a, *b, FBinOp::Add)?
            }
            Op::Sub { dst, a, b } => bin(&mut written, *dst, *a, *b, FBinOp::Sub)?,
            Op::Mul { dst, a, b } => bin(&mut written, *dst, *a, *b, FBinOp::Mul)?,
            Op::Div { dst, a, b } => bin(&mut written, *dst, *a, *b, FBinOp::Div)?,
            // Float `%` (`a - trunc(a/b)*b`) needs `roundsd` — SSE4.1 only.
            // `%` is not lowered: the emitted `a - trunc(a/b)*b` is not the exact
            // IEEE remainder (it rounds, and loses the sign of a zero result).
            Op::Move { dst, src } => {
                let s = read(&written, *src)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Move { dst: d, src: s }
            }
            Op::Lt { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Lt { dst: d, a, b }
            }
            // Float unary minus is not lowered: the emitted `0.0 - x` is `+0` for
            // `-(+0)`.
            // Float logical-not `!x` — so `<=`/`>=`/`!==` (Lt/StrictEq + Not) JIT.
            Op::Not { dst, a } => {
                let a = read(&written, *a)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Eqz { dst: d, a }
            }
            // Float strict equality `===` (numeric, NaN-aware); `!==` adds `Not`.
            Op::StrictEq { dst, a, b } => {
                let (a, b) = (read(&written, *a)?, read(&written, *b)?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Eq { dst: d, a, b }
            }
            // `Math.sqrt(x)` / `Math.abs(x)` — pure single-argument f64 intrinsics
            // (one SSE instruction; no realm access, so JIT-safe on the float path).
            Op::CallNative { dst, native, args }
                if (*native == crate::nbvm::NB_MATH_SQRT
                    || *native == crate::nbvm::NB_MATH_ABS)
                    && args.len() == 1 =>
            {
                let a = read(&written, args[0])?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                if *native == crate::nbvm::NB_MATH_SQRT {
                    FloatOp::Sqrt { dst: d, a }
                } else {
                    FloatOp::Abs { dst: d, a }
                }
            }
            // `Math.floor(x)` / `Math.ceil(x)` / `Math.trunc(x)` — one SSE4.1
            // `roundsd` each, but only if this CPU has SSE4.1; otherwise bail so the
            // whole function stays in the interpreter (the baseline is only SSE2).
            Op::CallNative { dst, native, args }
                if (*native == crate::nbvm::NB_MATH_FLOOR
                    || *native == crate::nbvm::NB_MATH_CEIL
                    || *native == crate::nbvm::NB_MATH_TRUNC)
                    && args.len() == 1
                    && has_sse41() =>
            {
                let a = read(&written, args[0])?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                if *native == crate::nbvm::NB_MATH_FLOOR {
                    FloatOp::Floor { dst: d, a }
                } else if *native == crate::nbvm::NB_MATH_CEIL {
                    FloatOp::Ceil { dst: d, a }
                } else {
                    FloatOp::Trunc { dst: d, a }
                }
            }
            // `Math.max(a, b)` / `Math.min(a, b)` — two-argument f64 intrinsics with
            // JS NaN/±0 semantics (handled in codegen). Also pure → JIT-safe.
            Op::CallNative { dst, native, args }
                if (*native == crate::nbvm::NB_MATH_MAX || *native == crate::nbvm::NB_MATH_MIN)
                    && args.len() == 2 =>
            {
                let (a, b) = (read(&written, args[0])?, read(&written, args[1])?);
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                if *native == crate::nbvm::NB_MATH_MAX {
                    FloatOp::Max { dst: d, a, b }
                } else {
                    FloatOp::Min { dst: d, a, b }
                }
            }
            // Targets are nbvm op indices; the lowered stream prepends one `Arg`
            // per parameter, so each target shifts by `n_params`.
            Op::JumpIfFalse { cond, target } => FloatOp::JumpIfFalse {
                cond: read(&written, *cond)?,
                target: target.checked_add(proto.n_params)?,
            },
            Op::Jump { target } => FloatOp::Jump {
                target: target.checked_add(proto.n_params)?,
            },
            Op::Return { src } => FloatOp::Ret {
                src: read(&written, *src)?,
            },
            _ => return None,
        };
        out.push(lowered);
    }
    // Eligible only if it terminates with a `Ret` and every target is in range.
    if !matches!(out.last(), Some(FloatOp::Ret { .. })) {
        return None;
    }
    for op in &out {
        if let FloatOp::JumpIfFalse { target, .. } | FloatOp::Jump { target } = op
            && *target >= out.len()
        {
            return None;
        }
    }
    Some(out)
}

/// A single op in the **generic (NanBox) tier** IR. Unlike the Int/Float tiers
/// (which unbox to machine `i64`/`f64` at the boundary and deopt on any
/// non-numeric argument), the generic tier operates directly on `NanBox` words
/// held in the **interpreter's own register file** (the frame ABI, see
/// [`JitFunction::compile_generic`]): the op stream is the function's bytecode
/// one-for-one (op `i` here is bytecode op `i`, so branch targets and OSR entry
/// points are bytecode pcs), and every op either has an inline fast path with a
/// runtime-helper slow path, or is executed by the interpreter itself
/// ([`GenericOp::Interp`]) on that same register file. Register operands are
/// bytecode registers (`u16`).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenericOp {
    /// `reg[dst] = imm` — load an immediate NanBox word.
    Const {
        /// Destination register.
        dst: u16,
        /// The raw NanBox bits to load.
        imm: u64,
    },
    /// `reg[dst] = reg[a] + reg[b]` — generic `+` with full ECMAScript semantics.
    /// Both-number operands take an inline SSE fast path; anything else calls
    /// `jit_helper_add`.
    Add {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
    },
    /// `reg[dst] = reg[src]` — register copy.
    Move {
        /// Destination register.
        dst: u16,
        /// Source register.
        src: u16,
    },
    /// `reg[dst] = reg[obj].key` — a plain (static-string-key) property get, with
    /// an inline monomorphic shape-guarded slot load and a `jit_helper_get_prop`
    /// slow path (the interpreter's real `vm_get_prop` through the site's
    /// persistent inline cache). `site` indexes the per-site key + cache table.
    GetProp {
        /// Destination register.
        dst: u16,
        /// Register holding the receiver object.
        obj: u16,
        /// Index into the per-site key/IC table.
        site: u16,
        /// Index into the per-site call caches (an inherited getter's call).
        call: u16,
    },
    /// An `Op::GetMethod`'s read (the tier's bytecode holds it as a
    /// `GetProp`): looked up as `CallMethod` looks its method up — the
    /// site's chain cache inline, else `jit_helper_get_method_site`, which
    /// first asks the site's built-in memo.
    GetMethod {
        /// Destination register.
        dst: u16,
        /// Register holding the receiver.
        obj: u16,
        /// Index into the per-site key/IC table.
        site: u16,
        /// Index into the per-site call caches (the helper's getter cache).
        call: u16,
    },
    /// `reg[obj].key = reg[val]` — a plain (static-string-key) sloppy-mode
    /// property set, inline in-place store on a shape hit, else
    /// `jit_helper_set_prop` (the interpreter's `vm_set_prop_mode(.., false)`).
    SetProp {
        /// Register holding the receiver object.
        obj: u16,
        /// Register holding the value to store.
        val: u16,
        /// Index into the per-site key/IC table.
        site: u16,
        /// Strict-mode code (`Op::SetPropStrict`): a failed write throws.
        strict: bool,
    },
    /// `reg[dst] = reg[obj][reg[index]]` — a computed element read (`Op::GetKey`):
    /// inline dense-array fast path, else `jit_helper_get_elem` (`vm_get_elem`).
    GetElem {
        /// Destination register.
        dst: u16,
        /// Register holding the receiver.
        obj: u16,
        /// Register holding the index / key value.
        index: u16,
    },
    /// `reg[obj][reg[index]] = reg[val]` — a computed element write
    /// (`Op::SetKey`): inline in-bounds dense-array store, else
    /// `jit_helper_set_elem` (`vm_set_elem`).
    SetElem {
        /// Register holding the receiver.
        obj: u16,
        /// Register holding the index / key value.
        index: u16,
        /// Register holding the value to store.
        val: u16,
        /// Strict-mode code (`Op::SetKeyStrict`): a failed write throws.
        strict: bool,
    },
    /// `reg[dst] = reg[arr].length` (`Op::ArrayLen`): inline dense-array length,
    /// else `jit_helper_array_len` (`vm_array_len`).
    ArrayLen {
        /// Destination register.
        dst: u16,
        /// Register holding the receiver.
        arr: u16,
    },
    /// `reg[dst] = reg[a] <op> reg[b]` for `-`/`*`/`/`/`%` (`op` is a `GA_*` code).
    /// `Sub`/`Mul`/`Div` take an inline both-numbers SSE fast path; `Mod` (and any
    /// non-number operand) calls `jit_helper_arith`.
    Arith {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
        /// The `GA_*` operation discriminant.
        op: u8,
    },
    /// `reg[dst] = reg[a] < reg[b]` — relational `<`. Both-numbers inline; else
    /// `jit_helper_lt`.
    Lt {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
    },
    /// `Op::Compare` (`<`/`>`/`<=`/`>=` by `op` = 0..=3): both-numbers inline,
    /// else the interpreter runs the op (bytecode pc `pc`).
    Compare {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
        /// 0 `<`, 1 `>`, 2 `<=`, 3 `>=`.
        op: u8,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::Add` (the numeric-only `+`): both-numbers inline, else the
    /// interpreter runs the op (bytecode pc `pc`), which faults as it always did.
    NumAdd {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `reg[dst] = reg[a] === reg[b]` — strict equality. Both-numbers inline;
    /// else `jit_helper_strict_eq` (never throws).
    StrictEq {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
    },
    /// `reg[dst] = reg[a] <op> reg[b]` for `Op::ValueBin` (`**`, bitwise, shifts,
    /// loose `==`/`!=`; `op` is a `VB_*` code). Loose `==`/`!=` take an inline
    /// both-numbers fast path; everything else calls `jit_helper_value_bin`.
    ValueBin {
        /// Destination register.
        dst: u16,
        /// Left operand register.
        a: u16,
        /// Right operand register.
        b: u16,
        /// The `VB_*` operation discriminant.
        op: u8,
    },
    /// `reg[dst] = !truthy(reg[a])` — logical not. Inlines the number/boolean
    /// truthiness test; else calls `jit_helper_truthy`.
    Not {
        /// Destination register.
        dst: u16,
        /// Source register.
        a: u16,
    },
    /// `pc = target` — unconditional branch to bytecode pc `target` (a backward
    /// one is a loop back-edge: it polls `jit_helper_backedge`).
    Jump {
        /// Target op index (`== len` is the fall-off-the-end exit).
        target: usize,
    },
    /// `if !truthy(reg[cond]) { pc = target }` — conditional branch. Inlines the
    /// number/boolean truthiness test; else calls `jit_helper_truthy`.
    JumpIfFalse {
        /// Register holding the condition value.
        cond: u16,
        /// Target op index taken when the condition is falsy.
        target: usize,
    },
    /// `reg[dst] = <builtin native>(reg[args…])` (`Op::CallNative`), routed
    /// through `jit_helper_call_native`. The argument register list is at
    /// per-call-site table index `site`.
    CallNative {
        /// Destination register.
        dst: u16,
        /// Native builtin discriminant.
        native: u16,
        /// Index into the per-call-site argument-register table.
        site: u16,
    },
    /// `Op::InitProp`: define a literal's own data property through the site's
    /// persistent inline cache (`jit_helper_init_prop`).
    InitProp {
        /// Register holding the literal object.
        obj: u16,
        /// Register holding the value.
        val: u16,
        /// Index into the per-site key/IC table.
        site: u16,
        /// Whether the key is an ordinary one (not `\0`-internal), so a
        /// transition-cache hit may be made inline.
        plain: bool,
    },
    /// `Op::CheckTdz`: inline "not the TDZ sentinel" test; the interpreter runs
    /// the op (and throws its `ReferenceError`) otherwise.
    CheckTdz {
        /// Register to test.
        src: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::ToNumeric`: a number is copied through inline; anything else is the
    /// interpreter's.
    ToNumeric {
        /// Destination register.
        dst: u16,
        /// Source register.
        src: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::Step` (`++`/`--` on a numeric value): a number is stepped inline;
    /// a BigInt (or anything else) is the interpreter's.
    Step {
        /// Destination register.
        dst: u16,
        /// Source register.
        src: u16,
        /// `--` rather than `++`.
        dec: bool,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::GetElem` (an array read by numeric index — capture cells,
    /// destructuring): inline dense-array load, else the interpreter's.
    ArrGet {
        /// Destination register.
        dst: u16,
        /// Register holding the array.
        arr: u16,
        /// Register holding the index.
        index: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::SetElem` (an array write by numeric index): inline in-bounds dense
    /// store, else the interpreter's.
    ArrSet {
        /// Register holding the array.
        arr: u16,
        /// Register holding the index.
        index: u16,
        /// Register holding the value.
        src: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::CellGet` (a captured variable's read): inline for a `Cell::Box`
    /// (and, out of line, a one-element array cell), else the interpreter's.
    CellGet {
        /// Destination register.
        dst: u16,
        /// Register holding the binding cell.
        cell: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// `Op::CellSet` (a captured variable's write): inline for a `Cell::Box`
    /// when the write barrier has nothing to do, else the interpreter's.
    CellSet {
        /// Register holding the binding cell.
        cell: u16,
        /// Register holding the value.
        src: u16,
        /// The bytecode pc of this op (its slow path).
        pc: u32,
    },
    /// Any other op: the interpreter executes bytecode op `pc` itself, on this
    /// same register file (`jit_helper_exec_op`) — the one semantics, by
    /// construction.
    Interp {
        /// The bytecode pc of the op.
        pc: u32,
    },
    /// An op proven to have no effect (a `BindThis` whose register nothing else
    /// names).
    Nop,
    /// `Op::PushHandler`: enter a `try` region whose catch code is at bytecode pc
    /// `target`, binding the thrown value in register `reg`. The handler stack is
    /// the activation's, kept in the context (`jit_helper_push_handler`), so a
    /// throw anywhere in the body can unwind to it (`jit_helper_unwind`).
    PushHandler {
        /// The catch entry (a bytecode pc).
        target: usize,
        /// The register the thrown value is bound in.
        reg: u16,
    },
    /// `Op::PopHandler`: leave the innermost `try` region.
    PopHandler,
    /// `Op::InlineGuard` (see `inline`): whether the call op at `call` calls
    /// the function `key` names — its callee identified as the call
    /// sequence's, the same limits checked and the same activation header
    /// pushed — with its captures loaded into `caps..caps + n_caps`.
    InlineGuard {
        /// The boolean result register.
        dst: u16,
        /// The guarded call op's pc.
        call: u32,
        /// The expected callee key (`CalleeInfo::key`).
        key: u64,
        /// The first capture register.
        caps: u16,
        /// How many captures.
        n_caps: u16,
    },
    /// `Op::InlineLeave`: pop the inlined activation's header.
    InlineLeave,
    /// `Op::CallMethod` (`recv.key(args)`): `jit_helper_call_method` runs the
    /// interpreter's method call with a persistent per-site method cache (the
    /// one-op fallback would start every call with a cold one), entering a
    /// compiled callee directly.
    CallMethod {
        /// The bytecode pc of the op.
        pc: u32,
        /// The site index (its method-lookup cache).
        site: u16,
        /// Destination register.
        dst: u16,
        /// The receiver register.
        recv: u16,
        /// Index into the per-call-site argument-register table.
        args: u16,
        /// Index into the per-site call caches (the native call sequence's).
        cache: u16,
    },
    /// `Op::CallValue` (`callee(args)`, `this` undefined): the native call
    /// sequence into a compiled callee (`call_seq::emit_call`), guarded on the
    /// site's one-entry cache `cache`; anything else is
    /// `jit_helper_call_value` (the interpreter's call, which refills the
    /// cache). The argument registers are at call-site table index `site`.
    CallValue {
        /// Destination register.
        dst: u16,
        /// Register holding the callee value.
        callee: u16,
        /// Index into the per-call-site argument-register table.
        site: u16,
        /// Index into the per-site call caches.
        cache: u16,
        /// The bytecode pc of the op (its slow path).
        pc: u32,
    },
    /// `Op::CallWithMethod` (`callee(args)` with `this` the receiver, the
    /// method read before the arguments): the native call sequence into a
    /// compiled callee, guarded on the site's cache `cache`, as `CallValue`;
    /// anything else is `jit_helper_call_with_method` (the interpreter's call
    /// half, a built-in served through the site's memo, refilling the cache).
    CallWithMethod {
        /// Destination register.
        dst: u16,
        /// Register holding the method.
        callee: u16,
        /// The receiver register (`this`).
        recv: u16,
        /// Index into the per-call-site argument-register table.
        site: u16,
        /// Index into the per-site call caches.
        cache: u16,
        /// The per-site key/IC table index (the built-in memo's cache).
        ic: u16,
        /// The bytecode pc of the op (its slow path).
        pc: u32,
    },
    /// `Op::PlainPrims`: `true` inline when no checked register holds a heap
    /// value (or is known to hold a Number), else the interpreter's op (a
    /// String is plain too).
    PlainPrims {
        /// Destination register.
        dst: u16,
        /// Index into the per-call-site argument-register table (the
        /// checked registers).
        site: u16,
        /// The bytecode pc of the op (its slow path).
        pc: u32,
    },
    /// `Op::CallKey` (`recv[key](args)`): the element read is
    /// `jit_helper_get_elem` (the interpreter's `vm_get_elem`), then the call
    /// of the value with `this` the receiver is the native call sequence, or
    /// `jit_helper_call_this` (the interpreter's call half, refilling the cache).
    CallKey {
        /// Destination register.
        dst: u16,
        /// The receiver register.
        recv: u16,
        /// The key register.
        key: u16,
        /// Index into the per-call-site argument-register table.
        site: u16,
        /// Index into the per-site call caches.
        cache: u16,
        /// The bytecode pc of the op (its slow path).
        pc: u32,
    },
    /// `Op::Construct` (`new ctor(args)`): the native call sequence for an
    /// ordinary VM constructor (`call_seq::CalleeSrc::Construct`), else
    /// `jit_helper_construct` (the interpreter's, refilling the cache).
    Construct {
        /// Destination register.
        dst: u16,
        /// Register holding the constructor.
        ctor: u16,
        /// Index into the per-call-site argument-register table.
        site: u16,
        /// Index into the per-site call caches.
        cache: u16,
        /// The bytecode pc of the op (its slow path).
        pc: u32,
    },
    /// `return reg[src]`.
    Ret {
        /// Register holding the return value.
        src: u16,
    },
}

/// How the generic tier computes a one-argument built-in `Math` call inline.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
enum MathInline {
    /// `sqrtsd`.
    Sqrt,
    /// The sign bit cleared.
    Abs,
    /// `roundsd` with this rounding-mode immediate (SSE4.1).
    Round(u8),
    /// `Math.max` (`true`) / `Math.min` (`false`) of its arguments.
    MaxMin(bool),
}

/// The built-in id and inline form of a `CallMethod` of one argument (for
/// `max`/`min`, one to four) whose key names a `Math` function the generic
/// tier computes inline.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
fn inline_math(op: &crate::nbvm::Op) -> Option<(u16, MathInline)> {
    use crate::nbexec::{
        N_MATH_ABS, N_MATH_CEIL, N_MATH_FLOOR, N_MATH_MAX, N_MATH_MIN, N_MATH_SQRT, N_MATH_TRUNC,
    };
    let crate::nbvm::Op::CallMethod { key, args, .. } = op else {
        return None;
    };
    match key.as_str() {
        "max" if (1..=4).contains(&args.len()) => {
            return Some((N_MATH_MAX, MathInline::MaxMin(true)));
        }
        "min" if (1..=4).contains(&args.len()) => {
            return Some((N_MATH_MIN, MathInline::MaxMin(false)));
        }
        _ => {}
    }
    if args.len() != 1 {
        return None;
    }
    let sse41 = has_sse41();
    Some(match key.as_str() {
        "sqrt" => (N_MATH_SQRT, MathInline::Sqrt),
        "abs" => (N_MATH_ABS, MathInline::Abs),
        "floor" if sse41 => (N_MATH_FLOOR, MathInline::Round(0x09)),
        "ceil" if sse41 => (N_MATH_CEIL, MathInline::Round(0x0a)),
        "trunc" if sse41 => (N_MATH_TRUNC, MathInline::Round(0x0b)),
        _ => return None,
    })
}

/// The result of [`lower_nbvm_generic`]: the generic-tier op stream (one per
/// bytecode op), the per-site property-key strings (owned so the emitted code can
/// hold stable pointers into them for the inline caches), and the per-call-site
/// argument-register lists.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
type GenericLowering = (
    Vec<GenericOp>,
    Vec<alloc::boxed::Box<str>>,
    Vec<Vec<u16>>,
    usize,
);

/// Why the generic tier cannot run `op` at all — `None` when it can (inline, via
/// a helper, or by handing the op to the interpreter). The refused ops are the
/// ones whose meaning is tied to the interpreter's *frame* rather than to the
/// register file: a suspension (`Yield`/`Await`/generator and async-iteration steps), a proper
/// tail call (which replaces the frame), and direct `eval`/environment ops (which
/// can reach the frame's scope by name).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
fn generic_refuses(op: &crate::nbvm::Op) -> Option<&'static str> {
    use crate::nbvm::Op;
    match op {
        Op::Yield { .. }
        | Op::Await { .. }
        | Op::GeneratorStart
        | Op::InitGenerator { .. }
        | Op::YieldDelegate { .. }
        | Op::YieldResume { .. }
        | Op::AsyncDelegateCall { .. }
        | Op::AsyncIterOpen { .. }
        | Op::AsyncIterStep { .. }
        | Op::AsyncIterClose { .. }
        | Op::AddAsyncDisposable { .. }
        | Op::AsyncDisposeStep { .. } => Some("suspension"),
        Op::TailCall { .. } | Op::TailCallValue { .. } | Op::TailCallThis { .. } => {
            Some("tail call")
        }
        Op::DirectEval { .. } | Op::Env { .. } => Some("direct eval / environment"),
        _ => None,
    }
}

/// Lowers a `nbvm::FnProto` to the generic-tier IR, one [`GenericOp`] per
/// bytecode op. Ops with a dedicated lowering (constants, moves, `+`/`-`/`*`/`/`/
/// `%`, the comparisons, `!`, branches, static property get/set, computed element
/// get/set, `.length`, native calls, `return`) get inline fast paths; every other
/// op becomes [`GenericOp::Interp`] (the interpreter runs it on the same register
/// file). `None` if any op is one `generic_refuses` names, or a register/branch is
/// out of range (the verifier's invariants, re-checked so the emitter can index
/// the register file without bounds checks).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub fn lower_nbvm_generic(proto: &crate::nbvm::FnProto) -> Option<GenericLowering> {
    use crate::nbvm::Op;
    if proto.is_generator || proto.ops.len() > u32::MAX as usize {
        return None;
    }
    let n_regs = proto.n_regs;
    let r = |x: crate::nbvm::Reg| -> Option<u16> { ((x as usize) < n_regs).then_some(x) };
    let mut out = Vec::with_capacity(proto.ops.len());
    // Per-site key strings for `GetProp`/`SetProp` (the site index into this Vec is
    // stored in the op); the caller boxes each for a stable pointer + persistent IC.
    let mut keys: Vec<alloc::boxed::Box<str>> = Vec::new();
    // Per-call-site argument-register lists (`CallNative`, `CallValue`).
    let mut call_args: Vec<Vec<u16>> = Vec::new();
    // Call caches (`CallValue`).
    let mut n_calls = 0usize;
    for (pc, op) in proto.ops.iter().enumerate() {
        if generic_refuses(op).is_some() {
            return None;
        }
        let pc32 = u32::try_from(pc).ok()?;
        let lowered = match op {
            Op::LoadConst { dst, value } => GenericOp::Const {
                dst: r(*dst)?,
                imm: value.to_bits(),
            },
            Op::AddValue { dst, a, b } => GenericOp::Add {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
            },
            Op::Add { dst, a, b } => GenericOp::NumAdd {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
                pc: pc32,
            },
            Op::Move { dst, src } => GenericOp::Move {
                dst: r(*dst)?,
                src: r(*src)?,
            },
            Op::Sub { dst, a, b }
            | Op::Mul { dst, a, b }
            | Op::Div { dst, a, b }
            | Op::Mod { dst, a, b } => GenericOp::Arith {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
                op: match op {
                    Op::Sub { .. } => crate::nbvm::GA_SUB,
                    Op::Mul { .. } => crate::nbvm::GA_MUL,
                    Op::Div { .. } => crate::nbvm::GA_DIV,
                    _ => crate::nbvm::GA_MOD,
                },
            },
            Op::Lt { dst, a, b } => GenericOp::Lt {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
            },
            Op::Compare { dst, op, a, b } if *op <= 3 => GenericOp::Compare {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
                op: *op,
                pc: pc32,
            },
            Op::StrictEq { dst, a, b } => GenericOp::StrictEq {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
            },
            Op::ValueBin { dst, op, a, b } => GenericOp::ValueBin {
                dst: r(*dst)?,
                a: r(*a)?,
                b: r(*b)?,
                op: *op,
            },
            Op::Not { dst, a } => GenericOp::Not {
                dst: r(*dst)?,
                a: r(*a)?,
            },
            Op::Jump { target } => GenericOp::Jump {
                target: (*target <= proto.ops.len()).then_some(*target)?,
            },
            Op::JumpIfFalse { cond, target } => GenericOp::JumpIfFalse {
                cond: r(*cond)?,
                target: (*target <= proto.ops.len()).then_some(*target)?,
            },
            Op::GetProp { dst, obj, key } => {
                let site = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                let call = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                GenericOp::GetProp {
                    dst: r(*dst)?,
                    obj: r(*obj)?,
                    site,
                    call,
                }
            }
            Op::SetProp { obj, key, src } | Op::SetPropStrict { obj, key, src } => {
                let site = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                GenericOp::SetProp {
                    obj: r(*obj)?,
                    val: r(*src)?,
                    site,
                    strict: matches!(op, Op::SetPropStrict { .. }),
                }
            }
            Op::GetKey { dst, obj, key } => GenericOp::GetElem {
                dst: r(*dst)?,
                obj: r(*obj)?,
                index: r(*key)?,
            },
            Op::SetKey { obj, key, src } | Op::SetKeyStrict { obj, key, src } => {
                GenericOp::SetElem {
                    obj: r(*obj)?,
                    index: r(*key)?,
                    val: r(*src)?,
                    strict: matches!(op, Op::SetKeyStrict { .. }),
                }
            }
            Op::ArrayLen { dst, arr } => GenericOp::ArrayLen {
                dst: r(*dst)?,
                arr: r(*arr)?,
            },
            Op::CallNative { dst, native, args } if args.len() <= 32 => {
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                GenericOp::CallNative {
                    dst: r(*dst)?,
                    native: *native,
                    site,
                }
            }
            // `OrdinaryCallBindThis` with no reader: computing the bound `this`
            // (the global object, or a `ToObject` wrapper) has no observable effect
            // when no other op names the register, so it is skipped.
            Op::BindThis { this }
                if !proto
                    .ops
                    .iter()
                    .enumerate()
                    .any(|(i, o)| i != pc && crate::bytecode::op_mentions_reg(o, *this)) =>
            {
                GenericOp::Nop
            }
            Op::Return { src } => GenericOp::Ret { src: r(*src)? },
            // Filled in on entry (`call_with_inner`, or the native-call entry);
            // the interpreter runs them as no-ops.
            Op::MakeArguments { .. } | Op::LoadNewTarget { .. } => GenericOp::Nop,
            Op::CallKey {
                dst,
                recv,
                key,
                args,
            } if args.len() <= 32 => {
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                let cache = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                GenericOp::CallKey {
                    dst: r(*dst)?,
                    recv: r(*recv)?,
                    key: r(*key)?,
                    site,
                    cache,
                    pc: pc32,
                }
            }
            Op::Construct { dst, ctor, args } if args.len() <= 32 => {
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                let cache = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                GenericOp::Construct {
                    dst: r(*dst)?,
                    ctor: r(*ctor)?,
                    site,
                    cache,
                    pc: pc32,
                }
            }
            Op::CallWithMethod {
                dst,
                callee,
                recv,
                key,
                args,
            } if args.len() <= 32 => {
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                let cache = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                let ic = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                GenericOp::CallWithMethod {
                    dst: r(*dst)?,
                    callee: r(*callee)?,
                    recv: r(*recv)?,
                    site,
                    cache,
                    ic,
                    pc: pc32,
                }
            }
            Op::PlainPrims { dst, regs } if regs.len() <= 32 => {
                let mut rs: Vec<u16> = Vec::with_capacity(regs.len());
                for a in regs {
                    rs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(rs);
                GenericOp::PlainPrims {
                    dst: r(*dst)?,
                    site,
                    pc: pc32,
                }
            }
            Op::CallValue { dst, callee, args } if args.len() <= 32 => {
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let site = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                let cache = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                GenericOp::CallValue {
                    dst: r(*dst)?,
                    callee: r(*callee)?,
                    site,
                    cache,
                    pc: pc32,
                }
            }
            Op::PushHandler { target, reg } => GenericOp::PushHandler {
                target: (*target <= proto.ops.len()).then_some(*target)?,
                reg: r(*reg)?,
            },
            Op::PopHandler => GenericOp::PopHandler,
            Op::InlineGuard {
                dst,
                call,
                key,
                callee,
                args,
                caps,
                n_caps,
            } => {
                // The guarded call must be a call op with a native sequence
                // (its argument table entry), reading the same registers.
                let same = match proto.ops.get(*call) {
                    Some(Op::CallMethod {
                        recv,
                        args: a,
                        key: k,
                        ..
                    }) => recv == callee && a == args && k != "caller",
                    Some(Op::CallValue {
                        callee: c, args: a, ..
                    })
                    | Some(Op::CallWithMethod {
                        callee: c, args: a, ..
                    }) => c == callee && a == args,
                    Some(Op::GetProp { obj, .. }) => obj == callee && args.is_empty(),
                    _ => false,
                };
                if !same || args.len() > 32 || *call <= pc {
                    return None;
                }
                for i in 0..*n_caps {
                    r(caps.checked_add(i)?)?;
                }
                r(*callee)?;
                GenericOp::InlineGuard {
                    dst: r(*dst)?,
                    call: u32::try_from(*call).ok()?,
                    key: *key,
                    caps: *caps,
                    n_caps: *n_caps,
                }
            }
            Op::InlineLeave => GenericOp::InlineLeave,
            Op::CallMethod {
                dst,
                recv,
                key,
                args,
            } if args.len() <= 32 => {
                let site = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                let mut argregs: Vec<u16> = Vec::with_capacity(args.len());
                for a in args {
                    argregs.push(r(*a)?);
                }
                let args = u16::try_from(call_args.len()).ok()?;
                call_args.push(argregs);
                let cache = u16::try_from(n_calls).ok()?;
                n_calls += 1;
                GenericOp::CallMethod {
                    pc: pc32,
                    site,
                    dst: r(*dst)?,
                    recv: r(*recv)?,
                    args,
                    cache,
                }
            }
            Op::CallMethod { key, .. } => {
                let site = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                // No native sequence (`args` past the cap): `cache` is unused.
                GenericOp::CallMethod {
                    pc: pc32,
                    site,
                    dst: 0,
                    recv: 0,
                    args: u16::MAX,
                    cache: u16::MAX,
                }
            }
            Op::InitProp { obj, key, src } => {
                let site = u16::try_from(keys.len()).ok()?;
                keys.push(alloc::boxed::Box::from(key.as_str()));
                GenericOp::InitProp {
                    obj: r(*obj)?,
                    val: r(*src)?,
                    site,
                    plain: !key.starts_with('\0'),
                }
            }
            Op::CheckTdz { src } => GenericOp::CheckTdz {
                src: r(*src)?,
                pc: pc32,
            },
            Op::ToNumeric { dst, src } => GenericOp::ToNumeric {
                dst: r(*dst)?,
                src: r(*src)?,
                pc: pc32,
            },
            Op::Step { dst, src, dec } => GenericOp::Step {
                dst: r(*dst)?,
                src: r(*src)?,
                dec: *dec,
                pc: pc32,
            },
            Op::GetElem { dst, arr, index } => GenericOp::ArrGet {
                dst: r(*dst)?,
                arr: r(*arr)?,
                index: r(*index)?,
                pc: pc32,
            },
            Op::SetElem { arr, index, src } => GenericOp::ArrSet {
                arr: r(*arr)?,
                index: r(*index)?,
                src: r(*src)?,
                pc: pc32,
            },
            Op::CellGet { dst, cell } => GenericOp::CellGet {
                dst: r(*dst)?,
                cell: r(*cell)?,
                pc: pc32,
            },
            Op::CellSet { cell, src } => GenericOp::CellSet {
                cell: r(*cell)?,
                src: r(*src)?,
                pc: pc32,
            },
            _ => GenericOp::Interp { pc: pc32 },
        };
        out.push(lowered);
    }
    // The interpreter ops compiled with a call site of their own, in pc order
    // after every other site (see `interp_sites`).
    for op in &proto.ops {
        match op {
            Op::SuperCallArgs { args, .. } => {
                call_args.push(args.clone());
                n_calls += 1;
            }
            Op::SuperCtor { .. } | Op::InitFields { .. } => n_calls += 1,
            _ => {}
        }
    }
    drop_dead_stores(
        &mut out,
        &proto.ops,
        n_regs,
        proto.n_params + proto.n_captures + 1,
    );
    Some((out, keys, call_args, n_calls))
}

/// Per bytecode pc, the `(argument-list index, call-site index)` of an
/// interpreter op [`lower_nbvm_generic`] gave a call site of its own
/// (`Op::SuperCallArgs`, `Op::SuperCtor`, `Op::InitFields`): those sites
/// come after every other one, in pc order, so they are the last of the
/// `n_args` argument lists and `n_sites` sites.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
fn interp_sites(
    bytecode: &[crate::nbvm::Op],
    n_args: usize,
    n_sites: usize,
) -> Option<Vec<Option<(u16, u16)>>> {
    use crate::nbvm::Op;
    let (mut extra_args, mut extra_sites) = (0usize, 0usize);
    for op in bytecode {
        match op {
            Op::SuperCallArgs { .. } => {
                extra_args += 1;
                extra_sites += 1;
            }
            Op::SuperCtor { .. } | Op::InitFields { .. } => extra_sites += 1,
            _ => {}
        }
    }
    let mut args = n_args.checked_sub(extra_args)?;
    let mut site = n_sites.checked_sub(extra_sites)?;
    let mut out = Vec::with_capacity(bytecode.len());
    for op in bytecode {
        let here = match op {
            Op::SuperCallArgs { .. } => {
                let v = (u16::try_from(args).ok()?, u16::try_from(site).ok()?);
                args += 1;
                site += 1;
                Some(v)
            }
            Op::SuperCtor { .. } | Op::InitFields { .. } => {
                let v = (0, u16::try_from(site).ok()?);
                site += 1;
                Some(v)
            }
            _ => None,
        };
        out.push(here);
    }
    Some(out)
}

/// Turns into `Nop`s the `LoadConst`s and `Move`s (`ops`, lowered from
/// `bytecode`) whose destination nothing ever reads: a register of at least
/// `first` (below it are the activation's inputs, which the entry code reads)
/// that no op names except such a pure write of it. That is the binding
/// initialization a block's `let`/`const` gets when the binding itself ends
/// up in another register (`LoadConst undefined` then a `Move` of it).
/// Removing a `Move` can leave its source unread in turn, so it iterates.
/// No other code reads a register by number: an op names every register it
/// touches (the verifier's operand list), and a catch binding is named by its
/// `PushHandler`.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
fn drop_dead_stores(
    ops: &mut [GenericOp],
    bytecode: &[crate::nbvm::Op],
    n_regs: usize,
    first: usize,
) {
    use crate::nbvm::Op;
    // The pure write of `reg` that op `i` is, if it is one.
    let pure_dst = |i: usize| match bytecode.get(i) {
        Some(Op::LoadConst { dst, .. }) => Some(*dst),
        Some(Op::Move { dst, src }) if dst != src => Some(*dst),
        _ => None,
    };
    let mut mentions: Vec<Vec<usize>> = alloc::vec![Vec::new(); n_regs];
    for (i, op) in bytecode.iter().enumerate() {
        let seen = core::cell::RefCell::new(Vec::new());
        crate::bytecode::op_for_each_reg(op, &|r| seen.borrow_mut().push(r));
        let mut seen = seen.into_inner();
        seen.sort_unstable();
        seen.dedup();
        for r in seen {
            if let Some(m) = mentions.get_mut(r as usize) {
                m.push(i);
            }
        }
    }
    let mut dead = alloc::vec![false; ops.len()];
    let mut work: Vec<usize> = (first..n_regs).collect();
    while let Some(reg) = work.pop() {
        let Some(m) = mentions.get(reg) else { continue };
        let unread = m
            .iter()
            .all(|&i| dead[i] || pure_dst(i).is_some_and(|d| d as usize == reg));
        if !unread {
            continue;
        }
        for &i in m {
            if dead[i] {
                continue;
            }
            // Only ops lowered as the plain write they are.
            if !matches!(ops[i], GenericOp::Const { .. } | GenericOp::Move { .. }) {
                continue;
            }
            dead[i] = true;
            ops[i] = GenericOp::Nop;
            if let Some(Op::Move { src, .. }) = bytecode.get(i)
                && (*src as usize) >= first
            {
                work.push(*src as usize);
            }
        }
    }
}

/// Why [`lower_nbvm_generic`] rejects `proto` — a human-readable diagnostic for
/// the `KATAAN_JIT_LOG` knob (never consulted for a compile decision).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub fn generic_reject_reason(proto: &crate::nbvm::FnProto) -> alloc::string::String {
    use alloc::format;
    if proto.is_generator {
        return alloc::string::String::from("generator");
    }
    for (pc, op) in proto.ops.iter().enumerate() {
        if let Some(why) = generic_refuses(op) {
            return format!("{why} at pc {pc}: {op:?}");
        }
    }
    alloc::string::String::from("register / branch target out of range, or emit failure")
}

/// A bytecode-VM function compiled to native code, callable from the VM with
/// `NanBox` values. This is the end-to-end fast path: it owns the compiled
/// machine code and performs the unbox→native→rebox round-trip with an integer
/// **type guard** at the boundary (a non-integer argument deopts to the
/// interpreter, exactly as the optimizing tier's guards will).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub struct JitProto {
    func: JitFunction,
    n_params: usize,
    kind: JitKind,
    /// Generic tier: the byte offset of the native-call entry
    /// (`call_seq::emit_native_entry`) — `None` for a body with `try` regions,
    /// whose handler stack needs the activation set-up the ordinary entry
    /// (`jit_tier::run_compiled`) does, or one whose `new.target` register the
    /// entry cannot fill.
    native_entry: Option<usize>,
    /// Generic tier: the register-file size the body addresses (the bytecode's
    /// `n_regs`); a caller's register file must be at least this long.
    n_regs: usize,
    /// Generic tier: the bytecode pcs the body can be entered at besides `0`
    /// (loop headers — the on-stack-replacement entries), ascending.
    osr_entries: Vec<usize>,
    /// Generic tier: the bytecode the body was lowered from. `Interp` sites bake
    /// pointers to these ops into the machine code, so it lives as long as it.
    #[allow(dead_code)]
    bytecode: Option<alloc::boxed::Box<[crate::nbvm::Op]>>,
    /// Per-property-access-site storage for the generic tier (`None` for the
    /// Int/Float tiers). Owns the key strings and inline caches the emitted code
    /// holds raw pointers into, so those pointers stay valid for the code's life.
    sites: Option<GenericSites>,
    /// Generic tier, a body that inlined calls (`inline`): its pc maps — per
    /// own pc its pc in the body, and per body pc the function's own (or
    /// `u32::MAX` inside an inlined body).
    pc_map: Option<alloc::boxed::Box<(Vec<u32>, Vec<u32>)>>,
    /// Int tier: the body returns a comparison's native `0`/`1` (a predicate;
    /// see [`numeric_tier_booleans`]), which [`Self::call_guarded`] boxes as
    /// a boolean. Such a body is never a native callee of another Int body.
    returns_bool: bool,
}

/// Per-site key + inline-cache storage for a generic-tier body, owned by the
/// [`JitProto`] alongside the compiled code. The emitted machine code holds raw
/// pointers into `keys` (the UTF-8 bytes) and `caches` (each `PropertyCache`),
/// baked as immediates at compile time; boxing keeps every address stable across
/// the move into the `JitProto` (only the owning pointers move; the heap
/// allocations do not). The caches are interior-mutable because a runtime helper
/// re-arms them through the raw pointer while the (shared) `JitProto` is borrowed.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
struct GenericSites {
    /// Stable-address key strings, one per site.
    #[allow(dead_code)]
    keys: Vec<alloc::boxed::Box<str>>,
    /// Persistent monomorphic inline caches, one per site.
    caches: alloc::boxed::Box<[core::cell::UnsafeCell<crate::ic::PropertyCache>]>,
    /// `(pc, call cache)` of every `CallMethod`/`CallValue` with a native
    /// call sequence, by pc.
    call_pcs: Vec<(u32, u16)>,
    /// The `CallValue` sites' one-entry callee caches (the emitted code holds
    /// their addresses; the site helper refills them).
    #[allow(dead_code)]
    calls: alloc::boxed::Box<[core::cell::UnsafeCell<CallSite>]>,
    /// The method-call chain caches, one per site (the emitted code holds
    /// their addresses; the method helper refills them).
    #[allow(dead_code)]
    methods: Vec<Option<alloc::boxed::Box<core::cell::UnsafeCell<props::MethodSite>>>>,
    /// The shapes the code compares with as immediates (see [`Baked`]): held
    /// for as long as the code exists, so no other shape can be allocated at
    /// one of their addresses and pass for it.
    #[allow(dead_code)]
    baked: Vec<alloc::rc::Rc<crate::shape::Shape>>,
    /// What the back-edge poll's recompilation test reads (see [`tierup`]).
    tier_up: tierup::TierUp,
    /// The persistent global binding cache the code reads and writes
    /// (`global_persist`).
    #[allow(dead_code)]
    persist: global_persist::Persist,
}

/// The generic-tier runtime-helper table: the `extern "C"` slow paths the emitted
/// code calls for `+`, property get, and property set. Passed to
/// [`JitProto::compile_generic`].
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub struct GenericHelpers {
    /// The `+` helper (`jit_helper_add`).
    pub add: JitAddHelper,
    /// The property-get helper (`jit_helper_get_prop`).
    pub get: JitGetPropHelper,
    /// The property-get helper through the site's chain and call caches
    /// (`jit_helper_get_prop_site`).
    pub get_site: JitGetSiteHelper,
    /// [`get_site`](Self::get_site) for an `Op::GetMethod`'s read, which
    /// first asks the site's built-in memo (`jit_helper_get_method_site`).
    pub get_method_site: JitGetSiteHelper,
    /// The property-set helper (`jit_helper_set_prop`), sloppy mode.
    pub set: JitSetPropHelper,
    /// The strict-mode property-set helper (`jit_helper_set_prop_strict`).
    pub set_strict: JitSetPropHelper,
    /// The property-set helper through the site's add cache
    /// (`jit_helper_set_prop_site`), both modes.
    pub set_site: JitSetSiteHelper,
    /// The literal-property definition helper (`jit_helper_init_prop`).
    pub init_prop: JitSetPropHelper,
    /// The computed element-get helper (`jit_helper_get_elem`) — `obj[key]`.
    pub get_elem: JitAddHelper,
    /// The computed element-set helper (`jit_helper_set_elem`) — `obj[key] = v`.
    pub set_elem: JitSetElemHelper,
    /// The `.length` helper (`jit_helper_array_len`) — `Op::ArrayLen`.
    pub array_len: JitTruthyHelper,
    /// The `-`/`*`/`/`/`%` helper (`jit_helper_arith`), dispatched by a `GA_*`
    /// discriminant passed as the fourth argument.
    pub arith: JitBinHelper,
    /// The `Op::ValueBin` helper (`jit_helper_value_bin`) — `**`, bitwise, shifts,
    /// loose `==`/`!=`; dispatched by a `VB_*` discriminant.
    pub value_bin: JitBinHelper,
    /// The relational `<` helper (`jit_helper_lt`).
    pub lt: JitAddHelper,
    /// The strict-equality `===` helper (`jit_helper_strict_eq`) — never throws.
    pub strict_eq: JitAddHelper,
    /// The truthiness helper (`jit_helper_truthy`), returning `0`/`1` — never throws.
    pub truthy: JitTruthyHelper,
    /// The interpreter-op helper (`jit_helper_exec_op`) — runs one bytecode op
    /// (any op without a native lowering) on the body's register file.
    pub exec_op: JitExecOpHelper,
    /// The loop back-edge poll (`jit_helper_backedge`) — host interrupt, agent
    /// scheduling and the GC safepoint; returns the next poll countdown.
    pub backedge: JitBackedgeHelper,
    /// The native-builtin-call helper (`jit_helper_call_native`) — runs the
    /// interpreter's `Op::CallNative` dispatch for a builtin (Pass 4).
    pub call_native: JitCallNativeHelper,
    /// The heap-arena accessor (`jit_arena(ctx, recv, mode)`) — returns the object
    /// heap's slot-array base + length, or an empty arena (so the inline path
    /// misses) unless `recv` qualifies for the inline path `mode` (an `ARENA_*`
    /// code): a receiver the interpreter would route to the host, or an array whose
    /// side tables make the raw element store differ from the spec, never does.
    /// Otherwise as before: it
    /// base pointer + length so the inline monomorphic property-get fast path
    /// (Pass 6) can read an object's shape/slot without re-entering the
    /// interpreter. Re-invoked on every fast-path entry so a reallocated arena
    /// (`slots.push`) is always addressed at its current base.
    pub arena: JitArenaHelper,
    /// `Op::PushHandler` (`jit_helper_push_handler(ctx, target, reg)`): push onto
    /// the activation's handler stack; `0`, or the throw sentinel on overflow.
    pub push_handler: JitAddHelper,
    /// `Op::PopHandler` (`jit_helper_pop_handler(ctx, _)`): never throws.
    pub pop_handler: JitTruthyHelper,
    /// The handler-stack segment of a natively entered body with `try`
    /// regions (`jit_helper_handler_frame(ctx, leave, saved)`, see
    /// `call_seq::emit_handler_entry`).
    pub handler_frame: JitAddHelper,
    /// `Op::CallMethod` (`jit_helper_call_method(ctx, regs, n_regs, op, cache,
    /// call_site)`, `call_site` the native sequence's `CallSite` or null).
    pub call_method: JitCallMethodHelper,
    /// A method call's callee lookup (`jit_method_info`).
    pub method_info: MethodHelper,
    /// `Op::Construct`'s slow path (`jit_helper_construct(ctx, regs, n_regs,
    /// op, site, _)`): the interpreter's, refilling the site's cache.
    pub construct: JitCallMethodHelper,
    /// `Op::CallKey`'s call half (`jit_helper_call_this(ctx, regs, n_regs, op,
    /// site, method)`): call `method` (already read) with `this` the
    /// receiver, as the interpreter does, refilling the site's cache.
    pub call_this: JitCallKeyHelper,
    /// A `new`'s callee check and object creation (`jit_construct_info`).
    pub construct_info: ConstructHelper,
    /// A `new`'s result when the body returned a heap value
    /// (`jit_construct_result`).
    pub construct_result: ConstructResultHelper,
    /// A `new` served by its site's inline constructor cache: its `this`
    /// (`jit_construct_alloc`).
    pub construct_alloc: ConstructAllocHelper,
    /// `super(…)`'s "already bound" `ReferenceError` (`jit_super_twice`).
    pub super_twice: SuperTwiceHelper,
    /// `Op::SuperCallArgs`'s slow path (`jit_helper_super_call(ctx, regs,
    /// n_regs, op, site, _)`): the interpreter's, refilling the site's cache.
    pub super_call: JitCallMethodHelper,
    /// `Op::SuperCtor`'s slow path (`jit_helper_ident_op`, same ABI): the
    /// interpreter's, refilling the site's identity cache.
    pub super_ctor: JitCallMethodHelper,
    /// `Op::InitFields`'s slow path (`jit_helper_ident_op`, same ABI): the
    /// interpreter's, refilling the site's identity cache.
    pub init_fields: JitCallMethodHelper,
    /// `Op::CallValue`'s slow path (`jit_helper_call_value(ctx, regs, n_regs,
    /// op, site)`, `site` the `CallSite`): the interpreter's call, which also
    /// refills the site's cache.
    pub call_value: JitCallMethodHelper,
    /// `Op::CallWithMethod`'s slow path (`jit_helper_call_with_method(ctx,
    /// regs, n_regs, op, cache, site)`, `cache` the site's `PropertyCache`,
    /// `site` its `CallSite`): the interpreter's call half, refilling the
    /// site's cache.
    pub call_with_method: JitCallMethodHelper,
    /// [`call_with_method`](Self::call_with_method) for the built-in the
    /// site's identity memo names (`jit_helper_call_native_got`, same ABI).
    pub call_native_got: JitCallMethodHelper,
    /// The call sequence's callee identification (`jit_callee_info`).
    pub callee_info: CalleeHelper,
    /// Where the call sequence finds the interpreter state it maintains.
    pub call_layout: CallLayout,
    /// The throw-exit unwinder (`jit_helper_unwind(ctx, regs, n_regs)`): if the
    /// pending fault is a thrown value and the activation has a handler, pop it,
    /// bind the value and return the catch pc; else [`NO_HANDLER`].
    pub unwind: JitUnwindHelper,
    /// `jit_helper_global_ref(ctx, op) -> *mut NanBox | 0`: validates a global
    /// binding for direct access from a compiled loop (`numeric`).
    pub global_ref: JitTruthyHelper,
    /// `jit_helper_math1(ctx, recv, x, op) -> bits | throw sentinel`: a
    /// one-argument built-in `Math.f(x)` call (`numeric`).
    pub math1: JitBinHelper,
    /// `jit_helper_site_fill(ctx, recv, site, _) -> 0`: records where a
    /// method site's key is found on `recv` (`MethodSite::fill`), for the
    /// inline `Math.f(x)` lookup.
    pub site_fill: JitBinHelper,
    /// `jit_helper_new_object(ctx, _) -> bits`: `Op::NewObject`'s fresh
    /// ordinary object (never throws).
    pub new_object: JitTruthyHelper,
    /// `jit_helper_new_array(ctx, regs, n_regs, op) -> 0`: `Op::NewArrayFrom`
    /// / `Op::NewArray`, written to the op's destination (the exec-op ABI).
    pub new_array: JitExecOpHelper,
}

/// The runtime-helper ABI for a method call: `(ctx, regs, n_regs, op, cache) ->
/// 0` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitCallMethodHelper = extern "C" fn(
    *mut core::ffi::c_void,
    *mut u64,
    usize,
    *const crate::nbvm::Op,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> u64;

/// The runtime-helper ABI for `Op::CallKey`'s call half: `(ctx, regs, n_regs,
/// op, site, method) -> 0` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitCallKeyHelper = extern "C" fn(
    *mut core::ffi::c_void,
    *mut u64,
    usize,
    *const crate::nbvm::Op,
    *mut core::ffi::c_void,
    u64,
) -> u64;

/// The [`JitUnwindHelper`] result meaning "no handler caught it: propagate".
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub const NO_HANDLER: u64 = u64::MAX;

/// The runtime-helper ABI for the throw-exit unwinder: `(ctx, regs, n_regs) ->
/// catch pc` (or [`NO_HANDLER`]).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitUnwindHelper = extern "C" fn(*mut core::ffi::c_void, *mut u64, usize) -> u64;

/// The base pointer + length of the object heap's slot array, returned by the
/// [`JitArenaHelper`]. `#[repr(C)]` with two integer-class eightbytes, so the
/// System V ABI returns it in `rax:rdx` (base in `rax`, `len` in `rdx`) — which
/// the emitted fast path reads directly after the `call`.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[repr(C)]
pub struct JitArena {
    /// `slots.as_ptr()` — the arena base (`base + index * SLOT_STRIDE` addresses a
    /// slot). Never cached across a call: the `Vec` may reallocate.
    pub base: *const u8,
    /// `slots.len()` — the slot-index bound (used for the fast path's bounds check).
    pub len: usize,
}

/// The runtime-helper ABI for the heap-arena accessor (`JIT_DESIGN.md` Pass 6):
/// `(ctx) -> JitArena`. A leaf, non-allocating read that never triggers GC, so the
/// inline fast path stays GC-safe and realloc-safe.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitArenaHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64) -> JitArena;

/// `jit_arena` mode: a static-key property read on a shaped object.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub const ARENA_GET: u64 = 0;
/// `jit_arena` mode: a static-key property write on a shaped object.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub const ARENA_SET: u64 = 1;
/// `jit_arena` mode: a dense-array element / `.length` read.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub const ARENA_ELEM_GET: u64 = 2;
/// `jit_arena` mode: an in-bounds dense-array element write.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub const ARENA_ELEM_SET: u64 = 3;

/// The byte offsets + discriminant values the inline monomorphic property-get
/// fast path bakes into emitted machine code, all combined **relative to a slot's
/// base address** (`arena_base + index * slot_stride`). Assembled by
/// [`compute_jit_layout`] from the per-type runtime probes
/// ([`crate::heap::jit_slot_layout`], [`crate::cell::jit_cell_layout`],
/// [`crate::object::jit_object_layout`], [`crate::ic::jit_cache_layout`]) — never
/// hand-written. If any offset is wrong the JIT layout verification test
/// (`jit_layout_matches_safe_reads`) fails loudly rather than the JIT silently
/// corrupting a raw read.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct JitLayout {
    /// `size_of::<Slot<Cell>>()` — the arena stride.
    pub slot_stride: i32,
    /// A live slot's occupancy byte.
    pub slot_occupied_disc: u8,
    /// Slot-relative offset of the slot's occupancy byte.
    pub off_slot_occ: i32,
    /// `Cell::Object` discriminant byte.
    pub cell_object_disc: u8,
    /// The shaped storage mode's byte (an object's first byte, which is also
    /// the `Cell` tag: the `Cell::Object` tag test and this mode test read the
    /// same byte).
    pub obj_shaped_disc: u8,
    /// Slot-relative offset of the slot's `generation: u16`.
    pub off_slot_gen: i32,
    /// Slot-relative offset of the `Cell` tag byte.
    pub off_cell_tag: i32,
    /// Slot-relative offset of the object's storage-mode byte.
    pub off_od_tag: i32,
    /// Slot-relative offset of the object's shape-`Rc` word (8 bytes).
    pub off_shape: i32,
    /// Slot-relative offset of the object's slot count (`u16`).
    pub off_slots_len: i32,
    /// Slot-relative offset of the object's first inline slot (slot
    /// `i < INLINE_SLOTS` at `+ 8 * i`).
    pub off_inline: i32,
    /// `ObjectExtra`-relative offset of the object's near spilled values
    /// (slot `i` in `INLINE_SLOTS..INLINE_SLOTS + SPILL_NEAR` at `+ 8 * (i -
    /// INLINE_SLOTS)`; the `extra` pointer is at `props.off_extra`, and is
    /// non-null whenever the object has a spilled slot).
    pub extra_near: i32,
    /// `ObjectExtra`-relative offset of the object's far spill `Vec` data
    /// pointer (slot `i >= INLINE_SLOTS + SPILL_NEAR` at that pointer `+ 8 *
    /// (i - INLINE_SLOTS - SPILL_NEAR)`).
    pub extra_far_ptr: i32,
    /// `PropertyCache`-relative offset of the cached shape word (8 bytes).
    pub cache_shape_off: i32,
    /// `PropertyCache`-relative offset of the cached `slot: u32`.
    pub cache_slot_off: i32,
    /// `PropertyCache`-relative offset of the `hits: u32` counter.
    pub cache_hits_off: i32,
    /// `Cell::Array` discriminant byte — the inline array-element fast paths guard
    /// the receiver cell tag against this.
    pub cell_array_disc: u8,
    /// Slot-relative offset of a `Cell::Array`'s element `Vec` data pointer.
    pub off_arr_ptr: i32,
    /// Slot-relative offset of a `Cell::Array`'s element `Vec` length (the dense
    /// bound for the inline element GET/SET bounds check).
    pub off_arr_len: i32,
    /// Slot-relative offset of a `Cell::Array`'s side-state flag byte
    /// (`crate::cell::ArrayElems`): non-zero routes an element access to the
    /// helper.
    pub off_arr_side: i32,
    /// Slot-relative offset of a `Cell::Array`'s prototype-set flag byte
    /// (`crate::cell::ArrayElems`).
    pub off_arr_proto_set: i32,
    /// Slot-relative offset of a `Cell::Array`'s inline element count byte
    /// (`crate::cell::ARRAY_ON_HEAP` when the elements are in the `Vec`, whose
    /// pointer and length are `off_arr_ptr` / `off_arr_len`). Read through
    /// `p_arr_len` / `p_arr_len_ptr`.
    pub off_arr_inline_len: i32,
    /// Slot-relative offset of a `Cell::Array`'s first inline element.
    pub off_arr_inline: i32,
    /// Slot-relative offset of a `Cell::Array`'s element `Vec` (three words).
    pub off_arr_vec: i32,
    /// The words of an inline array's empty element `Vec`.
    pub arr_empty_vec: [u64; 3],
    /// The inline property paths' extra offsets (`props.rs`).
    pub props: props::PropsLayout,
}

/// Assembles the [`JitLayout`] from the four per-type probes, combining the
/// nested field offsets into slot-relative offsets. Computed once (memoized) and
/// baked into every emitted property-get site.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn compute_jit_layout() -> &'static JitLayout {
    use std::sync::OnceLock;
    static LAYOUT: OnceLock<JitLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let slot = crate::heap::jit_slot_layout(crate::cell::Cell::Object(
            crate::object::Object::new(crate::shape::Shape::root()),
        ));
        let cell = crate::cell::jit_cell_layout();
        let obj = crate::object::jit_object_layout();
        let cache = crate::ic::jit_cache_layout();
        // From a slot base S: the Cell payload sits at S + slot.value_off; the
        // Object at + cell.object_off; then the Object/Vec fields.
        let cell_base = slot.value_off;
        let obj_base = cell_base + cell.object_off;
        let to_i32 = |v: usize| i32::try_from(v).expect("jit layout offset fits in i32");
        JitLayout {
            slot_stride: to_i32(slot.stride),
            slot_occupied_disc: slot.occupied_disc,
            off_slot_occ: to_i32(slot.occupied_off),
            cell_object_disc: cell.object_disc,
            obj_shaped_disc: obj.shaped_disc,
            off_slot_gen: to_i32(slot.generation_off),
            off_cell_tag: to_i32(cell_base),
            off_od_tag: to_i32(obj_base + obj.mode_off),
            off_shape: to_i32(obj_base + obj.shape_off),
            off_slots_len: to_i32(obj_base + obj.len_off),
            off_inline: to_i32(obj_base + obj.inline_off),
            extra_near: to_i32(obj.extra_near_off),
            extra_far_ptr: to_i32(obj.extra_far_ptr_off),
            cache_shape_off: to_i32(cache.shape_off),
            cache_slot_off: to_i32(cache.slot_off),
            cache_hits_off: to_i32(cache.hits_off),
            cell_array_disc: cell.array_disc,
            // An array's element Vec sits at `cell_base + array_vec_off`; its data
            // ptr / len are the generic `Vec<NanBox>` offsets within that Vec.
            off_arr_ptr: to_i32(cell_base + cell.array_vec_off + obj.vec_ptr_off),
            off_arr_len: to_i32(cell_base + cell.array_vec_off + obj.vec_len_off),
            off_arr_side: to_i32(cell_base + cell.array_side_off),
            off_arr_proto_set: to_i32(cell_base + cell.array_proto_set_off),
            off_arr_inline_len: to_i32(cell_base + cell.array_inline_len_off),
            off_arr_inline: to_i32(cell_base + cell.array_inline_off),
            off_arr_vec: to_i32(cell_base + cell.array_vec_off),
            arr_empty_vec: cell.array_empty_vec,
            props: props::PropsLayout::new(&slot, cell_base, obj_base, &obj),
        }
    })
}

/// The runtime-helper ABI for a discriminated binary op: `(ctx, a, b, op) -> NanBox
/// bits` (or the throw sentinel). `op` selects the concrete operation (a `GA_*` or
/// `VB_*` code). Used by both the arithmetic (`jit_helper_arith`) and the
/// `Op::ValueBin` (`jit_helper_value_bin`) slow paths.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitBinHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64, u64) -> u64;

/// The runtime-helper ABI for a static-key property read through its site:
/// `(ctx, obj, site, call site) -> NanBox bits` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitGetSiteHelper =
    extern "C" fn(*mut core::ffi::c_void, u64, *mut MethodSite, *const CallSite) -> u64;

/// The runtime-helper ABI for a static-key property write through its site:
/// `(ctx, obj, site, val, strict) -> undefined bits` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitSetSiteHelper =
    extern "C" fn(*mut core::ffi::c_void, u64, *mut MethodSite, u64, u64) -> u64;

/// The runtime-helper ABI for a computed element write: `(ctx, recv, key, val,
/// strict) -> undefined bits` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitSetElemHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64, u64, u64) -> u64;

/// The runtime-helper ABI for a truthiness test: `(ctx, v) -> 0 | 1`. Never throws.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitTruthyHelper = extern "C" fn(*mut core::ffi::c_void, u64) -> u64;

/// The runtime-helper ABI for a property **get** (`JIT_DESIGN.md` Pass 2):
/// `(ctx, obj, key_ptr, key_len, cache) -> NanBox bits` (or the throw sentinel).
/// `cache` is an opaque pointer to the site's `PropertyCache`.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitGetPropHelper =
    extern "C" fn(*mut core::ffi::c_void, u64, *const u8, usize, *mut core::ffi::c_void) -> u64;

/// The runtime-helper ABI for a property **set**: `(ctx, obj, key_ptr, key_len,
/// cache, val) -> undefined bits` (or the throw sentinel).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitSetPropHelper = extern "C" fn(
    *mut core::ffi::c_void,
    u64,
    *const u8,
    usize,
    *mut core::ffi::c_void,
    u64,
) -> u64;

/// Resolved per-site call data the emitter bakes into a property-access site: the
/// key string's raw pointer/length and the site's inline-cache pointer.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct SiteInfo {
    key_ptr: *const u8,
    key_len: usize,
    cache_ptr: *mut core::ffi::c_void,
    /// The site's method-call chain cache (see [`props::MethodSite`]).
    method_ptr: *mut props::MethodSite,
    /// The site's compiled-in layout, for a site that was monomorphic while
    /// interpreted.
    baked: Option<Baked>,
    /// For a write site that was a monomorphic add while interpreted: that
    /// add, compiled in (its shapes kept alive by the owning `JitProto`).
    add: Option<props::BakedAdd>,
}

/// A property site's layout compiled into its code: a receiver whose shape
/// word is `shape` has the site's key as a data property in slot `slot`
/// (a fact of that immutable shape, which the owning [`JitProto`] keeps
/// alive). Any other receiver takes the site's cache path.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct Baked {
    shape: u64,
    slot: u32,
}

/// The calling-convention / signature a compiled [`JitProto`] exposes to a
/// *caller's* native code — the public mirror of the private `JitKind`, recorded
/// per callee so a JIT→JIT direct call is only ever emitted at the matching ABI (a
/// wrong-ABI direct call would mis-dispatch). Used to build the per-tier call
/// registries in `nbvm::ensure_jit`.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AbiKind {
    /// Integer ABI: `extern "C" fn(i64, …) -> i64` (an Int-tier body).
    Int,
    /// Float ABI: `extern "C" fn(f64, …) -> f64` (a Float-tier body). No intra-JIT
    /// direct-call path targets it (the Float tier never emits calls).
    Float,
    /// Generic (NanBox) ABI: `extern "C" fn(ctx, *const u64, usize) -> u64` (a
    /// generic-tier body). A generic caller may direct-call another generic body.
    Generic,
}

/// Which native fast path a [`JitProto`] holds.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum JitKind {
    /// Integer arithmetic + branches/loops; guards args to exact integers and
    /// deopts on overflow/range.
    Int,
    /// Straight-line `f64` arithmetic (incl. division); guards args to numbers.
    Float,
    /// The generic (NanBox) tier: operates on boxed values, with an inline
    /// number fast path and a runtime-helper slow path that re-enters the
    /// interpreter (`JIT_DESIGN.md` Pass 1). Invoked via
    /// [`JitProto::call_frame_raw`], not `call_guarded`.
    Generic,
}

/// The runtime-helper ABI the generic tier calls into. `ctx` is an opaque
/// pointer to the live interpreter context (`nbvm::Ctx`), which the helper
/// reconstructs; the operands and result are raw `NanBox` words. On a thrown
/// exception the helper stashes the value in the context and returns
/// `NanBox::jit_throw_bits`(crate::nanbox::NanBox::jit_throw_bits).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitAddHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u64;

/// The runtime-helper ABI for running one bytecode op on the body's register
/// file: `(ctx, regs, n_regs, op) -> 0` (or the throw sentinel, with the fault in
/// `ctx.jit_pending_fault`). `op` points at the `nbvm::Op`, owned by the
/// compiled [`JitProto`].
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitExecOpHelper =
    extern "C" fn(*mut core::ffi::c_void, *mut u64, usize, *const crate::nbvm::Op) -> u64;

/// The runtime-helper ABI for a loop back-edge poll: `(ctx, regs, n_regs, ops,
/// n_ops, header) -> next countdown` (or the throw sentinel, or
/// `NanBox::jit_exit_bits`: leave for the interpreter at loop header pc
/// `header`, the branch's target). `ops`/`n_ops` is the running bytecode (the
/// GC safepoint roots its constants).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitBackedgeHelper = extern "C" fn(
    *mut core::ffi::c_void,
    *mut u64,
    usize,
    *const crate::nbvm::Op,
    usize,
    usize,
) -> u64;

/// The runtime-helper ABI for a **native builtin call** (`Op::CallNative`):
/// `(ctx, native_id, argv_ptr, argc) -> NanBox bits` (or the throw sentinel, with
/// the fault in `ctx.jit_pending_fault`). `native_id` is the builtin discriminant.
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub type JitCallNativeHelper = extern "C" fn(*mut core::ffi::c_void, u64, *const u64, usize) -> u64;

#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
impl JitProto {
    /// Compiles a `nbvm::FnProto` to native code if it is JIT-eligible.
    /// Prefers the integer path ([`lower_nbvm`], which handles branches/loops);
    /// otherwise tries the straight-line float path ([`lower_nbvm_float`], which
    /// handles `/` and non-integer values). `None` if neither applies.
    #[must_use]
    pub fn compile(proto: &crate::nbvm::FnProto) -> Option<Self> {
        Self::compile_with_registry(proto, &alloc::collections::BTreeMap::new())
    }

    /// Like [`compile`](Self::compile) but with a `registry` of already-compiled
    /// callee code addresses, so a function that statically calls another JIT'd
    /// function lowers to native calls (see [`lower_nbvm_with`]).
    #[must_use]
    pub fn compile_with_registry(
        proto: &crate::nbvm::FnProto,
        registry: &alloc::collections::BTreeMap<u32, u64>,
    ) -> Option<Self> {
        if let Some(ops) = lower_nbvm_with(proto, registry) {
            let returns_bool = numeric_tier_booleans(proto, true) == Some(true);
            let has_call = ops.iter().any(|o| matches!(o, RegOp::Call { .. }));
            let func = if has_call {
                // Call programs skip the constant/allocation passes (which assume
                // pure register ops) and compile directly.
                JitFunction::compile_reg(proto.n_regs, proto.n_params, &ops)?
            } else {
                // Optimizing tier: fold/simplify/copy-prop/DCE, then register-
                // allocate (shrinking the frame) before native codegen.
                let ops = optimize_reg(&ops, proto.n_regs);
                let (ops, n_regs) = allocate_reg(&ops, proto.n_regs);
                JitFunction::compile_reg(n_regs, proto.n_params, &ops)?
            };
            return Some(Self {
                func,
                n_params: proto.n_params,
                kind: JitKind::Int,
                native_entry: None,
                sites: None,
                n_regs: proto.n_regs,
                osr_entries: Vec::new(),
                bytecode: None,
                pc_map: None,
                returns_bool,
            });
        }
        let ops = lower_nbvm_float(proto)?;
        let func = JitFunction::compile_float(proto.n_regs, proto.n_params, &ops)?;
        Some(Self {
            func,
            n_params: proto.n_params,
            kind: JitKind::Float,
            native_entry: None,
            sites: None,
            n_regs: proto.n_regs,
            osr_entries: Vec::new(),
            bytecode: None,
            pc_map: None,
            returns_bool: false,
        })
    }

    /// The native entry address of this compiled function — register it so other
    /// JIT'd functions can call it (see [`compile_with_registry`](Self::compile_with_registry)).
    #[must_use]
    pub fn code_ptr(&self) -> usize {
        self.func.code_ptr()
    }

    /// Calls the native code with `NanBox` arguments, or `None` to **deopt** to
    /// the interpreter (wrong arity; an argument that isn't the kind this path
    /// requires; or, for the integer path, an overflow/out-of-range result).
    #[must_use]
    pub fn call_guarded(&self, args: &[crate::nanbox::NanBox]) -> Option<crate::nanbox::NanBox> {
        if args.len() != self.n_params {
            return None;
        }
        match self.kind {
            JitKind::Int => {
                // Guard: every argument must be an exact integer. (Straight-line
                // per-slot reads: a zip into a stack array costs a `memset` and
                // a `memcpy` call on this per-callback path.)
                let int = |v: &crate::nanbox::NanBox| nanbox_int(*v);
                let a = match args {
                    [] => [0; 6],
                    [a] => [int(a)?, 0, 0, 0, 0, 0],
                    [a, b] => [int(a)?, int(b)?, 0, 0, 0, 0],
                    [a, b, c] => [int(a)?, int(b)?, int(c)?, 0, 0, 0],
                    [a, b, c, d] => [int(a)?, int(b)?, int(c)?, int(d)?, 0, 0],
                    [a, b, c, d, e] => [int(a)?, int(b)?, int(c)?, int(d)?, int(e)?, 0],
                    [a, b, c, d, e, f] => [int(a)?, int(b)?, int(c)?, int(d)?, int(e)?, int(f)?],
                    _ => return None,
                };
                let r = self.func.call_int6(a);
                // A result outside ±2^53 is the overflow/range deopt sentinel.
                if (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&r) {
                    Some(if self.returns_bool {
                        crate::nanbox::NanBox::boolean(r != 0)
                    } else {
                        crate::nanbox::NanBox::number(r as f64)
                    })
                } else {
                    None
                }
            }
            JitKind::Float => {
                // Guard: every argument must be a (finite) JS number.
                let num = |v: &crate::nanbox::NanBox| v.as_number();
                let a = match args {
                    [] => [0.0; 4],
                    [a] => [num(a)?, 0.0, 0.0, 0.0],
                    [a, b] => [num(a)?, num(b)?, 0.0, 0.0],
                    [a, b, c] => [num(a)?, num(b)?, num(c)?, 0.0],
                    [a, b, c, d] => [num(a)?, num(b)?, num(c)?, num(d)?],
                    _ => return None,
                };
                let r = self.func.call_f64x4(a);
                Some(crate::nanbox::NanBox::number(r))
            }
            // The generic tier is invoked via `call_frame_raw` (it needs the
            // live context pointer), never through the value-only `call_guarded`.
            JitKind::Generic => None,
        }
    }

    /// Compiles `proto` on the **generic (NanBox) tier** (the frame ABI, see
    /// [`JitFunction::compile_generic`]) if [`lower_nbvm_generic`] accepts it,
    /// wiring `helpers` as the runtime slow paths. Returns `None` if the function
    /// isn't generic-eligible.
    ///
    /// Per-property-access-site key strings and inline caches are allocated here
    /// (in `GenericSites`, owned by the returned `JitProto`) so their addresses
    /// are stable; the emitted code holds raw pointers into them, and into the
    /// owned copy of the bytecode.
    #[must_use]
    pub fn compile_generic(proto: &crate::nbvm::FnProto, helpers: &GenericHelpers) -> Option<Self> {
        Self::compile_generic_warm(proto, helpers, &[])
    }

    /// [`compile_generic`](Self::compile_generic), starting from the
    /// interpreter's warm inline caches: `warm[pc]` is the memo of the
    /// property site at bytecode `pc` (if any). Each site's cache starts
    /// with that memo, and a site the interpreter saw stay monomorphic gets
    /// its shape and slot compiled in as constants: its fast
    /// path compares the receiver's shape with an immediate and reads or
    /// writes the slot at a fixed offset, with the cache path behind it.
    #[must_use]
    pub fn compile_generic_warm(
        proto: &crate::nbvm::FnProto,
        helpers: &GenericHelpers,
        warm: &[Option<crate::ic::WarmEntry>],
    ) -> Option<Self> {
        Self::compile_generic_closures(proto, helpers, warm, &|_| false)
    }

    /// [`compile_generic_warm`](Self::compile_generic_warm), told which
    /// functions' closures (`Op::MakeClosure { func, .. }`) need nothing at
    /// creation but their cell (`plain(func)`: not an async function, whose
    /// closure links `%AsyncFunction.prototype%`): those with few captures
    /// are allocated inline.
    #[must_use]
    pub fn compile_generic_closures(
        proto: &crate::nbvm::FnProto,
        helpers: &GenericHelpers,
        warm: &[Option<crate::ic::WarmEntry>],
        plain: &dyn Fn(u32) -> bool,
    ) -> Option<Self> {
        Self::compile_generic_inlining(proto, helpers, warm, plain, &|_, _| None)
    }

    /// [`compile_generic_closures`](Self::compile_generic_closures), inlining
    /// the callees `pick` names at call sites (see [`inline`]): `pick(pc,
    /// op)` for the call op `op` at the function's own pc `pc`.
    #[must_use]
    pub fn compile_generic_inlining<'a>(
        proto: &crate::nbvm::FnProto,
        helpers: &GenericHelpers,
        warm: &[Option<crate::ic::WarmEntry>],
        plain: &dyn Fn(u32) -> bool,
        pick: &dyn Fn(usize, &crate::nbvm::Op) -> Option<inline::Candidate<'a>>,
    ) -> Option<Self> {
        // The register `call_with_inner` puts `new.target` in, if any (a
        // native-call entry fills it from the call header; one at or below the
        // `this` slot would collide with an input, so such a body gets none).
        let new_target = proto.ops.iter().take(2).find_map(|o| match o {
            crate::nbvm::Op::LoadNewTarget { dst } => Some(*dst as usize),
            _ => None,
        });
        let nt_ok = new_target.is_none_or(|nt| nt > proto.n_params + proto.n_captures);
        let expanded = inline::expand(proto, warm, pick);
        let spliced;
        let (proto, warm, pc_map) = match expanded {
            Some(e) => {
                let mut p = proto.clone();
                p.ops = e.ops;
                p.n_regs = e.n_regs;
                spliced = p;
                (
                    &spliced,
                    e.warm,
                    Some(alloc::boxed::Box::new((e.new_of, e.orig_of))),
                )
            }
            None => (proto, warm.to_vec(), None),
        };
        // An `Op::GetMethod` is a `GetProp` here (the same `[[Get]]`, also in
        // every analysis and in the interpreter's one-op fallback), whose slow
        // path first asks the site's built-in memo.
        let method_gets: Vec<bool> = proto
            .ops
            .iter()
            .map(|o| matches!(o, crate::nbvm::Op::GetMethod { .. }))
            .collect();
        let as_gets;
        let proto = if method_gets.contains(&true) {
            let mut p = proto.clone();
            for op in &mut p.ops {
                if let crate::nbvm::Op::GetMethod { dst, obj, key } = op {
                    *op = crate::nbvm::Op::GetProp {
                        dst: *dst,
                        obj: *obj,
                        key: core::mem::take(key),
                    };
                }
            }
            as_gets = p;
            &as_gets
        } else {
            proto
        };
        let warm = &warm[..];
        let inline_closures: Vec<bool> = proto
            .ops
            .iter()
            .map(|op| match op {
                crate::nbvm::Op::MakeClosure { func, captures, .. } => {
                    captures.len() <= crate::cell::CAPTURES_INLINE_CAP && plain(*func)
                }
                _ => false,
            })
            .collect();
        let (mut ops, keys, call_args, n_calls) = lower_nbvm_generic(proto)?;
        for (op, &m) in ops.iter_mut().zip(&method_gets) {
            if let (
                true,
                GenericOp::GetProp {
                    dst,
                    obj,
                    site,
                    call,
                },
            ) = (m, *op)
            {
                *op = GenericOp::GetMethod {
                    dst,
                    obj,
                    site,
                    call,
                };
            }
        }
        // One persistent cache per `CallValue` site (stable addresses, as below).
        let calls: alloc::boxed::Box<[core::cell::UnsafeCell<CallSite>]> = (0..n_calls)
            .map(|_| core::cell::UnsafeCell::new(CallSite::new()))
            .collect();
        let call_sites: Vec<*const CallSite> = calls.iter().map(|c| c.get().cast_const()).collect();
        let bytecode: alloc::boxed::Box<[crate::nbvm::Op]> = proto.ops.clone().into_boxed_slice();
        // One persistent inline cache per property-access site.
        let mut caches: alloc::boxed::Box<[core::cell::UnsafeCell<crate::ic::PropertyCache>]> = (0
            ..keys.len())
            .map(|_| core::cell::UnsafeCell::new(crate::ic::PropertyCache::new()))
            .collect();
        // The interpreter's memos: every `GetProp`/`SetProp` site starts warm,
        // and a stable one is compiled in (its shape kept alive by `baked`).
        let mut baked_at: Vec<Option<Baked>> = alloc::vec![None; keys.len()];
        let mut add_at: Vec<Option<props::BakedAdd>> = alloc::vec![None; keys.len()];
        let mut baked: Vec<alloc::rc::Rc<crate::shape::Shape>> = Vec::new();
        let mut tier_baked: Vec<Option<(alloc::rc::Rc<crate::shape::Shape>, u32)>> =
            alloc::vec![None; keys.len()];
        for (pc, op) in ops.iter().enumerate() {
            let (site, write) = match *op {
                GenericOp::GetProp { site, .. } => (site as usize, false),
                GenericOp::SetProp { site, .. } => (site as usize, true),
                _ => continue,
            };
            let (Some(Some(w)), Some(key), Some(cache)) =
                (warm.get(pc), keys.get(site), caches.get_mut(site))
            else {
                continue;
            };
            // A layout's keys and slots never change: re-derive the memo
            // from the shape itself rather than trust it.
            if w.shape.lookup(key) != Some(w.slot) {
                continue;
            }
            cache.get_mut().seed(w);
            if w.is_stable() && !(write && (w.by_add || w.shape.is_wrapper())) {
                baked_at[site] = Some(Baked {
                    shape: props::shape_word(&w.shape),
                    slot: w.slot,
                });
                baked.push(alloc::rc::Rc::clone(&w.shape));
                tier_baked[site] = Some((alloc::rc::Rc::clone(&w.shape), w.slot));
            } else if write
                && w.by_add
                && w.hits >= 4
                && let Some(b) = props::BakedAdd::of(&w.shape, w.slot)
            {
                // An add site also counts a miss per add (its in-place
                // lookup), so its transition's hits alone say it settled.
                // (`w.shape` holds its parent, the add's `from`.)
                add_at[site] = Some(b);
                baked.push(alloc::rc::Rc::clone(&w.shape));
            }
        }
        // A chain cache for each method-call and property-write site.
        let mut wants = alloc::vec![false; keys.len()];
        for op in &ops {
            if let GenericOp::CallMethod { site, .. }
            | GenericOp::SetProp { site, .. }
            | GenericOp::GetProp { site, .. }
            | GenericOp::GetMethod { site, .. } = *op
                && let Some(w) = wants.get_mut(site as usize)
            {
                *w = true;
            }
        }
        let methods: Vec<Option<alloc::boxed::Box<core::cell::UnsafeCell<props::MethodSite>>>> =
            keys.iter()
                .zip(caches.iter())
                .zip(wants)
                .map(|((k, c), want)| {
                    want.then(|| {
                        alloc::boxed::Box::new(core::cell::UnsafeCell::new(props::MethodSite::new(
                            k,
                            c.get().cast::<core::ffi::c_void>(),
                        )))
                    })
                })
                .collect();
        // Resolve each site's raw key + cache pointers (stable: `keys`/`caches` heap
        // allocations do not move when the owners are moved into the `JitProto`).
        let site_info: Vec<SiteInfo> = keys
            .iter()
            .zip(caches.iter())
            .zip(methods.iter())
            .map(|((k, c), m)| SiteInfo {
                key_ptr: k.as_ptr(),
                key_len: k.len(),
                cache_ptr: c.get().cast::<core::ffi::c_void>(),
                method_ptr: m.as_ref().map_or(core::ptr::null_mut(), |m| m.get()),
                baked: None,
                add: None,
            })
            .collect();
        let tier_up = tierup::TierUp::plan(&ops, tier_baked);
        let call_pcs: Vec<(u32, u16)> = ops
            .iter()
            .enumerate()
            .filter_map(|(at, op)| match *op {
                GenericOp::CallMethod { pc, cache, .. }
                | GenericOp::CallValue { pc, cache, .. }
                | GenericOp::CallWithMethod { pc, cache, .. }
                    if cache != u16::MAX =>
                {
                    Some((pc, cache))
                }
                GenericOp::GetProp { call, .. } => Some((u32::try_from(at).ok()?, call)),
                _ => None,
            })
            .collect();
        let site_info: Vec<SiteInfo> = site_info
            .into_iter()
            .zip(baked_at.into_iter().zip(add_at))
            .map(|(si, (b, add))| SiteInfo {
                baked: b,
                add,
                ..si
            })
            .collect();
        // Loop headers: the targets of backward branches.
        let mut osr_entries: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter_map(|(i, op)| match *op {
                GenericOp::Jump { target } | GenericOp::JumpIfFalse { target, .. }
                    if target <= i =>
                {
                    Some(target)
                }
                _ => None,
            })
            .collect();
        osr_entries.sort_unstable();
        osr_entries.dedup();
        let persist = global_persist::Persist::new();
        let (func, native_entry) = JitFunction::compile_generic(
            persist.addr(),
            proto.n_regs,
            &ops,
            &bytecode,
            helpers,
            &site_info,
            &call_args,
            &osr_entries,
            &call_sites,
            (proto.n_params, proto.n_captures, new_target),
            &inline_closures,
            pc_map.as_ref().map_or(&[][..], |m| &m.1[..]),
        )?;
        Some(Self {
            func,
            n_params: proto.n_params,
            kind: JitKind::Generic,
            native_entry: nt_ok.then_some(native_entry),
            sites: Some(GenericSites {
                keys,
                caches,
                call_pcs,
                calls,
                methods,
                baked,
                tier_up,
                persist,
            }),
            n_regs: proto.n_regs,
            osr_entries,
            bytecode: Some(bytecode),
            pc_map,
            returns_bool: false,
        })
    }

    /// What this body's call sites learned of their callees, by the
    /// function's own pc (the callee feedback of its recompilation): a site
    /// whose cache holds one callee names it, one with several is mixed; an
    /// inlined site names its inlined callee unless its call took another.
    #[must_use]
    pub fn call_hints(&self) -> Vec<crate::ic::CalleeHint> {
        use crate::ic::CalleeHint;
        let (Some(s), Some(bc)) = (self.sites.as_ref(), self.bytecode.as_ref()) else {
            return Vec::new();
        };
        let site_of = |pc: usize| -> Option<&CallSite> {
            let i = s
                .call_pcs
                .binary_search_by_key(&pc, |&(p, _)| p as usize)
                .ok()?;
            let c = s.calls.get(s.call_pcs[i].1 as usize)?;
            // SAFETY: the body is not running (no other borrow is live).
            #[allow(unsafe_code)]
            Some(unsafe { &*c.get() })
        };
        let keys = |c: &CallSite| -> Vec<u64> {
            c.targets
                .iter()
                .map(|t| t.key)
                .filter(|&k| k != CallSite::EMPTY)
                .collect()
        };
        let mut out = alloc::vec![CalleeHint::None; self.orig_pc(bc.len()).min(bc.len()) + 1];
        for (pc, op) in bc.iter().enumerate() {
            let orig = self.orig_pc(pc);
            let Some(slot) = out.get_mut(orig) else {
                continue;
            };
            match op {
                crate::nbvm::Op::InlineGuard { call, key, .. } => {
                    let others = site_of(*call).map(keys).unwrap_or_default();
                    *slot = if others.iter().all(|k| k == key) {
                        CalleeHint::One(*key)
                    } else {
                        CalleeHint::Mixed
                    };
                }
                crate::nbvm::Op::CallMethod { .. }
                | crate::nbvm::Op::CallValue { .. }
                | crate::nbvm::Op::CallWithMethod { .. }
                | crate::nbvm::Op::GetProp { .. } => {
                    if let Some(k) = site_of(pc).map(keys) {
                        *slot = match k[..] {
                            [] => CalleeHint::None,
                            [one] => CalleeHint::One(one),
                            _ => CalleeHint::Mixed,
                        };
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// How many call sites this body inlined (`inline`).
    #[must_use]
    pub fn inlined(&self) -> usize {
        self.inlined_sites().len()
    }

    /// The call sites this body inlined, as `(own pc, callee function id)`.
    #[must_use]
    pub fn inlined_sites(&self) -> Vec<(usize, u32)> {
        self.bytecode.as_ref().map_or_else(Vec::new, |b| {
            b.iter()
                .enumerate()
                .filter_map(|(pc, o)| match o {
                    crate::nbvm::Op::InlineGuard { key, .. } => {
                        Some((self.orig_pc(pc), (*key & 0xffff_ffff) as u32))
                    }
                    _ => None,
                })
                .collect()
        })
    }

    /// The body's pc of the function's own pc `pc` (they differ in a body
    /// that inlined calls).
    #[must_use]
    pub fn body_pc(&self, pc: usize) -> usize {
        match &self.pc_map {
            Some(m) => m.0.get(pc).map_or(usize::MAX, |&p| p as usize),
            None => pc,
        }
    }

    /// The function's own pc of the body's pc `pc` (`usize::MAX` inside an
    /// inlined body).
    #[must_use]
    pub fn orig_pc(&self, pc: usize) -> usize {
        match &self.pc_map {
            Some(m) => match m.1.get(pc) {
                Some(&p) if p != u32::MAX => p as usize,
                _ => usize::MAX,
            },
            None => pc,
        }
    }

    /// The total inline-cache hit count across this generic body's property sites
    /// (0 for a non-generic body). Type feedback + the differential-test proof that
    /// the monomorphic shape fast path is being taken: a repeat access on a
    /// same-shape object hits without a name lookup.
    #[must_use]
    pub fn ic_hits(&self) -> u32 {
        self.sites.as_ref().map_or(0, |s| {
            s.caches
                .iter()
                // SAFETY: no other borrow of any cache is live here — the native
                // body (the only writer, via the raw pointer) is not running.
                .map(|c| {
                    #[allow(unsafe_code)]
                    unsafe {
                        (*c.get()).hits()
                    }
                })
                .sum()
        })
    }

    /// The total inline-cache miss count across this generic body's property sites
    /// (0 for a non-generic body).
    #[must_use]
    pub fn ic_misses(&self) -> u32 {
        self.sites.as_ref().map_or(0, |s| {
            s.caches
                .iter()
                // SAFETY: as `ic_hits`.
                .map(|c| {
                    #[allow(unsafe_code)]
                    unsafe {
                        (*c.get()).misses()
                    }
                })
                .sum()
        })
    }

    /// Whether this compiled function is a generic-tier body (dispatched via
    /// [`call_frame_raw`](Self::call_frame_raw)).
    #[must_use]
    pub fn is_generic(&self) -> bool {
        self.kind == JitKind::Generic
    }

    /// Whether this Int-tier body returns a boolean as a native `0`/`1` (see
    /// [`Self::call_guarded`]): a native caller would read it as a number, so
    /// it must not be a JIT→JIT direct-call target.
    #[must_use]
    pub fn returns_bool(&self) -> bool {
        self.returns_bool
    }

    /// The [`AbiKind`] this body exposes to a caller's native code — the tier it
    /// compiled to. A caller consults this to decide whether a JIT→JIT direct call
    /// is ABI-compatible (see `nbvm::ensure_jit`).
    #[must_use]
    pub fn abi_kind(&self) -> AbiKind {
        match self.kind {
            JitKind::Int => AbiKind::Int,
            JitKind::Float => AbiKind::Float,
            JitKind::Generic => AbiKind::Generic,
        }
    }

    /// Whether a generic-tier body can be entered at bytecode pc `pc` (`0`, or a
    /// loop header — an on-stack-replacement entry).
    #[must_use]
    pub fn has_entry(&self, pc: usize) -> bool {
        let pc = self.body_pc(pc);
        self.kind == JitKind::Generic && (pc == 0 || self.osr_entries.binary_search(&pc).is_ok())
    }

    /// Invokes a generic-tier body on the interpreter register file `regs`,
    /// starting at bytecode pc `entry`, with the live context `ctx` (an opaque
    /// pointer the runtime helpers reconstruct). Returns the raw result word, or
    /// `None` (nothing ran) if this is not a generic body, `regs` is shorter than
    /// the body's register count, or `entry` is not an entry point. The word is a
    /// `NanBox`'s bits, `NanBox::jit_throw_bits` (a fault is pending in `ctx`), or
    /// `NanBox::jit_falloff_bits` (control fell off the end) — the caller
    /// interprets the sentinels (see `nbvm::jit_tier`).
    ///
    /// The caller must pass a live `nbvm::Ctx` as `ctx` for the duration of the
    /// call (the helpers dereference it as `&mut Ctx`). The call is synchronous
    /// and single-threaded — the ordinary interpreter reentrancy pattern.
    #[must_use]
    pub fn call_frame_raw(
        &self,
        ctx: *mut core::ffi::c_void,
        regs: &mut [crate::nanbox::NanBox],
        entry: usize,
    ) -> Option<u64> {
        if regs.len() < self.n_regs || !self.has_entry(entry) {
            return None;
        }
        let entry = self.body_pc(entry);
        Some(
            self.func
                .call_generic(ctx, regs.as_mut_ptr().cast::<u64>(), entry as u64),
        )
    }

    /// Enters a generic body at its native-call entry (see
    /// [`Self::native_entry`]) exactly as a compiled caller's call sequence
    /// does: `window` is the callee's register window on the register stack,
    /// laid out just above the native-call header and the `argc` arguments at
    /// `argv`; `caps` the closure's captured values and `this` the receiver.
    /// `None` if the body has no such entry.
    ///
    /// # Safety
    /// `ctx` is the live, otherwise-unborrowed `Ctx`; `window` points at
    /// `n_regs` writable words at the top of that context's register stack,
    /// directly above a complete native-call header (`FRAME_HEADER` words)
    /// and the arguments (which `argv` points at); `caps` at the closure's
    /// `n_captures` captured values.
    #[allow(unsafe_code)]
    #[must_use]
    pub unsafe fn call_native_entry(
        &self,
        ctx: *mut core::ffi::c_void,
        window: *mut u64,
        argc: u64,
        argv: *const u64,
        caps: *const u64,
        this: u64,
    ) -> Option<u64> {
        let code = self.native_entry()?;
        // SAFETY: `code` is this body's native-call entry, emitted to the
        // System V signature below (see `call_seq::emit_native_entry`); the
        // pointers satisfy its contract per this function's.
        let f: extern "C" fn(
            *mut core::ffi::c_void,
            *mut u64,
            u64,
            *const u64,
            *const u64,
            u64,
        ) -> u64 = unsafe { core::mem::transmute(code as usize) };
        Some(f(ctx, window, argc, argv, caps, this))
    }

    /// The parameter count of the compiled function.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// The register-file size a generic body needs.
    #[must_use]
    pub fn n_regs(&self) -> usize {
        self.n_regs
    }

    /// The native-call entry address (`call_seq::emit_native_entry`) a call
    /// sequence may enter this body at, if it has one.
    #[must_use]
    pub fn native_entry(&self) -> Option<u64> {
        self.native_entry
            .map(|off| (self.func.code_ptr() + off) as u64)
    }
}

/// The reference for [`JitFunction::compile_sum_1_to_n`]: `sum(1..=n)` for
/// `n >= 0`, else `0`.
#[must_use]
pub fn eval_sum_1_to_n(n: i64) -> i64 {
    let mut acc = 0i64;
    let mut i = n;
    while i > 0 {
        acc = acc.wrapping_add(i);
        i -= 1;
    }
    acc
}

/// A minimal x86-64 machine-code emitter (System V AMD64 ABI).
///
/// Only the instructions the arithmetic IR needs are encoded. The accumulator
/// lives in `rax`; the first integer argument arrives in `rdi`, the second in
/// `rsi`; the result is returned in `rax`.
#[derive(Default)]
pub struct X64Assembler {
    code: Vec<u8>,
    /// Per-label byte offset (`usize::MAX` until [`bind`](Self::bind)).
    labels: Vec<usize>,
    /// Pending `rel32` jump fixups: `(operand offset in `code`, target label)`.
    fixups: Vec<(usize, usize)>,
    /// **Frame mode** (the generic tier's frame ABI): a *non-negative* `[rbp+disp]`
    /// operand of the frame-slot methods is redirected to `[rbx+disp]` — the
    /// interpreter-owned register file whose base the body pins in `rbx` — while a
    /// negative one still addresses the native frame (the call scratch buffer).
    /// Off for the Int/Float tiers, whose register slots are all `rbp`-negative.
    rbx_frame: bool,
    /// The generic tier's receiver-cache key slots (`[rbp+disp]`, see
    /// `compile_generic`): each is set to [`RECV_CACHE_EMPTY`] by the frame
    /// prologue and after every `call` this assembler emits, since Rust code
    /// (which alone can move, free or retype a heap cell) runs only behind a
    /// call.
    recv_cache_keys: Vec<i32>,
    /// Frame slots (`[rbp+disp]`) the frame prologue alone sets to
    /// [`RECV_CACHE_EMPTY`]: the generic tier's global binding cache, which
    /// a call does not empty (it is guarded by the global binding epoch).
    entry_clear: Vec<i32>,
    /// Where the most recent [`bind`](Self::bind) bound its label (no label
    /// is bound past it, as `bind` binds at the end of the code).
    last_bind: usize,
    /// While the generic tier emits an op inside a promoted numeric loop
    /// (`numeric`): the SSE registers every `call` stores to the register
    /// file first (when the flag is set) and reloads after, as `(xmm,
    /// register-file displacement, store)`.
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    call_spill: Vec<(u8, i32, bool)>,
    /// In a body that inlines calls (`inline`): the native-frame slot of a
    /// pending inlined activation's materialization stub, which every
    /// `call` first runs when set (`call_seq::emit_inline_guard`).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    inline_pend: Option<i32>,
    /// The inlined calls' state slots only the prologue empties.
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    inline_clear: Vec<i32>,
    /// Out-of-line code moved off the hot path ([`cold_end`](Self::cold_end)),
    /// placed after everything else by [`finish`](Self::finish).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    cold: Vec<ColdChunk>,
    /// While a cold region is open: the labels bound since it began.
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    cold_binds: Option<Vec<usize>>,
}

/// A run of code [`X64Assembler::cold_end`] cut out of the main stream: its
/// bytes, the labels bound in it and the jump fixups in it (both at offsets
/// relative to the chunk's start).
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
struct ColdChunk {
    code: Vec<u8>,
    labels: Vec<(usize, usize)>,
    fixups: Vec<(usize, usize)>,
}

/// Where a cold region began ([`X64Assembler::cold_begin`]).
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64")),
    allow(dead_code)
)]
pub(crate) struct ColdMark {
    start: usize,
    fixups: usize,
    last_bind: usize,
}

/// [`X64Assembler::labels`]' entry for a label bound in a cold chunk that
/// [`X64Assembler::finish`] has not placed yet.
const LABEL_COLD: usize = usize::MAX - 1;

/// The value a receiver-cache key slot holds when it caches nothing: the
/// handle bits of slot index `u32::MAX`, which `Heap::alloc` never hands out,
/// so no register ever holds them.
const RECV_CACHE_EMPTY: i32 = -1;

/// A branch target in an [`X64Assembler`], resolved by [`X64Assembler::bind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

impl X64Assembler {
    /// A new, empty assembler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The emitted machine code (call [`finish`](Self::finish) first to resolve
    /// jumps).
    #[must_use]
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Enables frame mode (see the `rbx_frame` field).
    pub fn set_rbx_frame(&mut self, on: bool) {
        self.rbx_frame = on;
    }

    /// Emits the `disp32` of a `[rbp+disp32]` frame-slot operand whose ModRM byte
    /// was just pushed, rewriting that ModRM's base from `rbp` (`101`) to `rbx`
    /// (`011`) for a register-file slot in frame mode (both encode mod=10 with no
    /// SIB byte, so only the r/m bits change).
    fn rbp_disp(&mut self, disp: i32) {
        if self.rbx_frame
            && disp >= 0
            && let Some(m) = self.code.last_mut()
        {
            *m = (*m & !7) | 3;
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// Sets the receiver-cache key slots (see the `recv_cache_keys` field).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn set_recv_cache_keys(&mut self, keys: Vec<i32>) {
        self.recv_cache_keys = keys;
    }

    /// Sets the promoted registers every `call` brackets (see the
    /// `call_spill` field).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn set_call_spill(&mut self, spill: Vec<(u8, i32, bool)>) {
        self.call_spill = spill;
    }

    /// The promoted registers every `call` currently brackets.
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn call_spill_now(&self) -> Vec<(u8, i32, bool)> {
        self.call_spill.clone()
    }

    /// Sets the slots only the frame prologue empties (see the `entry_clear`
    /// field).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn set_entry_clear(&mut self, slots: Vec<i32>) {
        self.entry_clear = slots;
    }

    /// Empties every receiver-cache key slot (after a `call`, in the prologue).
    fn clear_recv_cache(&mut self) {
        for i in 0..self.recv_cache_keys.len() {
            let d = self.recv_cache_keys[i];
            self.store_empty(d);
        }
    }

    /// `mov qword [rbp+d], RECV_CACHE_EMPTY` (sign-extended imm32).
    fn store_empty(&mut self, d: i32) {
        self.code.extend_from_slice(&[0x48, 0xc7, 0x85]);
        self.code.extend_from_slice(&d.to_le_bytes());
        self.code.extend_from_slice(&RECV_CACHE_EMPTY.to_le_bytes());
    }

    /// Empties every [`entry_clear`](Self::set_entry_clear) slot (the frame
    /// prologue; and the generic tier's global cache reset).
    pub(crate) fn clear_entry_slots(&mut self) {
        for i in 0..self.entry_clear.len() {
            let d = self.entry_clear[i];
            self.store_empty(d);
        }
    }

    /// Allocates a fresh, unbound label.
    pub fn new_label(&mut self) -> Label {
        self.labels.push(usize::MAX);
        Label(self.labels.len() - 1)
    }

    /// Binds `label` to the current emission point.
    pub fn bind(&mut self, label: Label) {
        // A `jmp label` just before (the 5-byte `E9 rel32` whose fixup is the
        // last one) jumps to the very next instruction: drop it — unless a
        // label is bound right after it, which would then point past the end.
        let n = self.code.len();
        if n >= 5
            && self.last_bind != n
            && self.code[n - 5] == 0xe9
            && self.fixups.last() == Some(&(n - 4, label.0))
        {
            self.code.truncate(n - 5);
            self.fixups.pop();
        }
        self.labels[label.0] = self.code.len();
        self.last_bind = self.code.len();
        if let Some(b) = self.cold_binds.as_mut() {
            b.push(label.0);
        }
    }

    /// Resolves every recorded `rel32` jump fixup; call once after emission.
    /// Returns the finished machine code.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        // The cold chunks, after everything else (they are position
        // independent: every jump is a `rel32` fixup).
        for chunk in core::mem::take(&mut self.cold) {
            let base = self.code.len();
            self.code.extend_from_slice(&chunk.code);
            for (label, at) in chunk.labels {
                self.labels[label] = base + at;
            }
            for (at, label) in chunk.fixups {
                self.fixups.push((base + at, label));
            }
        }
        for (at, label) in core::mem::take(&mut self.fixups) {
            let target = self.labels[label];
            debug_assert_ne!(target, usize::MAX, "unbound label in jump");
            // `rel32` is relative to the instruction *after* the 4-byte operand.
            let rel = (target as i64) - (at as i64 + 4);
            let bytes = (rel as i32).to_le_bytes();
            self.code[at..at + 4].copy_from_slice(&bytes);
        }
        self.code
    }

    /// Starts a **cold region**: the code emitted from here to the matching
    /// [`cold_end`](Self::cold_end) is moved out of the main stream, after
    /// all of it (see [`finish`](Self::finish)), so the hot path around it
    /// runs straight through. The code just before must not fall through
    /// into the region (an unconditional jump or a return), and the region's
    /// own last instruction must not fall through out of it; its code is
    /// entered only by jumps to the labels bound in it. Labels bound at the
    /// region's end belong to the code after it.
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn cold_begin(&mut self) -> ColdMark {
        let start = self.code.len();
        let mark = ColdMark {
            start,
            fixups: self.fixups.len(),
            last_bind: self.last_bind,
        };
        // No `bind` in the region may drop the jump just before it (which
        // would then fall through to whatever follows the region).
        self.last_bind = start;
        debug_assert!(self.cold_binds.is_none(), "cold regions do not nest");
        self.cold_binds = Some(Vec::new());
        mark
    }

    /// Ends the cold region `mark` began (see [`cold_begin`](Self::cold_begin)).
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn cold_end(&mut self, mark: ColdMark) {
        let (start, end) = (mark.start, self.code.len());
        let code = self.code.split_off(start);
        // A label bound in the region, or just before it (at its first
        // instruction), moves with it — unless bound at its end.
        let mut bound = self.cold_binds.take().unwrap_or_default();
        if mark.last_bind == start {
            for (i, &l) in self.labels.iter().enumerate() {
                if l == start {
                    bound.push(i);
                }
            }
        }
        let mut labels = Vec::new();
        let mut at_start = false;
        for i in bound {
            let l = &mut self.labels[i];
            if *l >= start && *l < end {
                labels.push((i, *l - start));
                *l = LABEL_COLD;
            } else if *l == end {
                *l = start;
                at_start = true;
            }
        }
        let fixups = self
            .fixups
            .split_off(mark.fixups)
            .into_iter()
            .map(|(at, label)| (at - start, label))
            .collect();
        // The region's end is now `start`: a label bound there keeps a
        // `bind` from dropping the jump before it.
        self.last_bind = if at_start {
            start
        } else if mark.last_bind < start {
            mark.last_bind
        } else {
            usize::MAX
        };
        self.cold.push(ColdChunk {
            code,
            labels,
            fixups,
        });
    }

    /// Emits a `rel32` jump operand placeholder targeting `label`.
    fn emit_rel32(&mut self, label: Label) {
        self.fixups.push((self.code.len(), label.0));
        self.code.extend_from_slice(&[0, 0, 0, 0]);
    }

    /// `cmp rax, imm32`.
    pub fn cmp_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x3d]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `test rcx, rcx` (sets flags from `rcx`).
    pub fn test_rcx_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x85, 0xc9]);
    }

    /// `mov rcx, rdi` — copy the first argument into the loop counter.
    pub fn mov_rcx_rdi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xf9]);
    }

    /// `xor rax, rax` (`rax = 0`).
    pub fn zero_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]);
    }

    /// `add rax, rcx`.
    pub fn add_rax_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x01, 0xc8]);
    }

    /// `dec rcx`.
    pub fn dec_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xff, 0xc9]);
    }

    /// `jmp label` (32-bit relative).
    pub fn jmp(&mut self, label: Label) {
        self.code.push(0xe9);
        self.emit_rel32(label);
    }

    /// `jle label` — jump if signed `<=` (after a `cmp`/`test`).
    pub fn jle(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x8e]);
        self.emit_rel32(label);
    }

    /// `jg label` — jump if signed `>`.
    pub fn jg(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x8f]);
        self.emit_rel32(label);
    }

    /// `je label` — jump if equal.
    pub fn je(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x84]);
        self.emit_rel32(label);
    }

    /// `push rdi` / `push rsi` / `push rax` — onto the native stack.
    pub fn push_rdi(&mut self) {
        self.code.push(0x57);
    }
    /// `push rsi`.
    pub fn push_rsi(&mut self) {
        self.code.push(0x56);
    }
    /// `push rax`.
    pub fn push_rax(&mut self) {
        self.code.push(0x50);
    }
    /// `pop rax`.
    pub fn pop_rax(&mut self) {
        self.code.push(0x58);
    }
    /// `pop rcx`.
    pub fn pop_rcx(&mut self) {
        self.code.push(0x59);
    }
    /// `movabs rax, imm64`.
    pub fn movabs_rax(&mut self, imm: i64) {
        self.code.extend_from_slice(&[0x48, 0xb8]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }
    /// `sub rax, rcx`.
    pub fn sub_rax_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x29, 0xc8]);
    }
    /// `imul rax, rcx`.
    pub fn imul_rax_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc1]);
    }

    // --- stack-frame (rbp-relative) addressing, for the register compiler ---

    /// `push rbp; mov rbp, rsp; sub rsp, frame` — a standard frame prologue.
    pub fn prologue(&mut self, frame: u32) {
        self.code.push(0x55); // push rbp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]); // sub rsp, imm32
        self.code.extend_from_slice(&frame.to_le_bytes());
    }

    /// `leave; ret` — restore `rsp`/`rbp` and return (`rax` holds the result).
    pub fn epilogue(&mut self) {
        self.code.push(0xc9); // leave
        self.code.push(0xc3); // ret
    }

    /// `mov [rbp+disp], <arg register>` — spill incoming integer arg `i` (0..=5,
    /// in `rdi/rsi/rdx/rcx/r8/r9`) to its frame slot.
    pub fn store_arg(&mut self, arg: usize, disp: i32) {
        match arg {
            0 => self.code.extend_from_slice(&[0x48, 0x89, 0xbd]), // rdi
            1 => self.code.extend_from_slice(&[0x48, 0x89, 0xb5]), // rsi
            2 => self.code.extend_from_slice(&[0x48, 0x89, 0x95]), // rdx
            3 => self.code.extend_from_slice(&[0x48, 0x89, 0x8d]), // rcx
            4 => self.code.extend_from_slice(&[0x4c, 0x89, 0x85]), // r8
            _ => self.code.extend_from_slice(&[0x4c, 0x89, 0x8d]), // r9
        }
        self.rbp_disp(disp);
    }

    /// `mov rax, [rbp+disp]`.
    pub fn load_rax(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x85]);
        self.rbp_disp(disp);
    }

    /// `mov [rbp+disp], rax`.
    pub fn store_rax(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x89, 0x85]);
        self.rbp_disp(disp);
    }

    /// `mov rcx, [rbp+disp]`.
    pub fn load_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x8d]);
        self.rbp_disp(disp);
    }

    /// `movsxd rax, eax` — truncate `rax` to 32 bits then sign-extend (JS ToInt32).
    pub fn to_int32_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x63, 0xc0]);
    }

    /// `movsxd rcx, ecx` — JS ToInt32 of `rcx`.
    pub fn to_int32_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x63, 0xc9]);
    }

    /// `<op> rax, rcx` for the bitwise ops `and`/`or`/`xor` (others unused here).
    pub fn bit_rax_rcx(&mut self, op: BinOp2) {
        let opc = match op {
            BinOp2::And => 0x21,
            BinOp2::Or => 0x09,
            BinOp2::Xor => 0x31,
            _ => 0x21,
        };
        self.code.extend_from_slice(&[0x48, opc, 0xc8]);
    }

    /// `not eax` — 32-bit bitwise complement (writing `eax` zeroes the upper 32).
    pub fn not_eax(&mut self) {
        self.code.extend_from_slice(&[0xf7, 0xd0]);
    }

    /// `cqo` — sign-extend `rax` into `rdx:rax` (for a signed `idiv`).
    pub fn cqo(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x99]);
    }
    /// `idiv rcx` — signed divide `rdx:rax` by `rcx` (quotient→rax, remainder→rdx).
    pub fn idiv_rcx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xf7, 0xf9]);
    }
    /// `mov rax, rdx` — move the `idiv` remainder into `rax`.
    pub fn mov_rax_rdx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xd0]);
    }

    /// `cmp rcx, imm8` (sign-extended).
    pub fn cmp_rcx_imm8(&mut self, imm: i8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xf9, imm as u8]);
    }

    /// `shl/sar/shr eax, cl` — a 32-bit shift of `eax` by `cl` (masked to 5 bits;
    /// writing `eax` zeroes the upper 32 bits of `rax`, i.e. zero-extends).
    pub fn shift_eax_cl(&mut self, op: ShiftOp) {
        let modrm = match op {
            ShiftOp::Shl => 0xe0, // /4
            ShiftOp::Sar => 0xf8, // /7
            ShiftOp::Shr => 0xe8, // /5
        };
        self.code.extend_from_slice(&[0xd3, modrm]);
    }

    /// `cmp rax, [rbp+disp]`.
    pub fn cmp_rax_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x3b, 0x85]);
        self.rbp_disp(disp);
    }

    /// `movabs r11, imm64` — load the upper safe-integer bound (`+2^53`).
    pub fn movabs_r11(&mut self, imm: i64) {
        self.code.extend_from_slice(&[0x49, 0xbb]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }
    /// `movabs r10, imm64` — load the lower safe-integer bound (`-2^53`).
    pub fn movabs_r10(&mut self, imm: i64) {
        self.code.extend_from_slice(&[0x49, 0xba]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }
    /// `cmp rax, r11`.
    pub fn cmp_rax_r11(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8]);
    }
    /// `cmp rax, r10`.
    pub fn cmp_rax_r10(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd0]);
    }
    /// `jo label` — jump if the last arithmetic op signed-overflowed.
    pub fn jo(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80]);
        self.emit_rel32(label);
    }
    /// `jl label` — jump if signed `<`.
    pub fn jl(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x8c]);
        self.emit_rel32(label);
    }

    /// `test rax, rax` (sets flags from `rax`).
    pub fn test_rax_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
    }

    /// `setl al; movzx rax, al` — `rax = (signed less-than flag) ? 1 : 0`, after a
    /// `cmp`.
    pub fn setl_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]); // setl al
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }

    /// `sete al; movzx rax, al` — `rax = (zero flag) ? 1 : 0`, after a `test`/`cmp`.
    pub fn sete_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x94, 0xc0]); // sete al
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }

    /// `<op> rax, [rbp+disp]` for `add`/`sub`/`imul`/`and`/`or`/`xor`.
    pub fn op_rax_mem(&mut self, op: BinOp2, disp: i32) {
        match op {
            BinOp2::Add => self.code.extend_from_slice(&[0x48, 0x03, 0x85]),
            BinOp2::Sub => self.code.extend_from_slice(&[0x48, 0x2b, 0x85]),
            BinOp2::Mul => self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0x85]),
            BinOp2::And => self.code.extend_from_slice(&[0x48, 0x23, 0x85]),
            BinOp2::Or => self.code.extend_from_slice(&[0x48, 0x0b, 0x85]),
            BinOp2::Xor => self.code.extend_from_slice(&[0x48, 0x33, 0x85]),
        }
        self.rbp_disp(disp);
    }

    // --- SSE2 scalar-double (f64) addressing, for the float compiler. `xmm0` is
    // the accumulator; f64 args arrive in `xmm0..xmm3`; the result returns in
    // `xmm0`. ModRM `0x85` selects `[rbp + disp32]` with reg field `xmm0`. ---

    /// `movsd xmm0, [rbp+disp]`.
    pub fn movsd_xmm0_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x85]);
        self.rbp_disp(disp);
    }
    /// `movsd [rbp+disp], xmm0`.
    pub fn movsd_mem_xmm0(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, 0x85]);
        self.rbp_disp(disp);
    }
    /// `movsd [rbp+disp], xmm<arg>` — spill incoming f64 arg `arg` (0..=3) to a
    /// frame slot.
    pub fn store_arg_f64(&mut self, arg: usize, disp: i32) {
        let modrm = match arg {
            0 => 0x85,
            1 => 0x8d,
            2 => 0x95,
            _ => 0x9d,
        };
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, modrm]);
        self.rbp_disp(disp);
    }
    /// `ucomisd xmm0, [rbp+disp]` — unordered compare, set EFLAGS like `cmp`.
    pub fn ucomisd_xmm0_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x2e, 0x85]);
        self.rbp_disp(disp);
    }
    /// `xorpd xmm1, xmm1` — set `xmm1` to `+0.0`.
    pub fn zero_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x57, 0xc9]);
    }
    /// `xorpd xmm0, xmm0` — set `xmm0` to `+0.0`.
    pub fn zero_xmm0(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x57, 0xc0]);
    }
    /// `sqrtsd xmm0, xmm0` — `xmm0 = sqrt(xmm0)`.
    pub fn sqrtsd_xmm0(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x51, 0xc0]);
    }
    /// `roundsd xmm0, xmm0, imm8` (SSE4.1). `mode`: 0x09 = floor, 0x0a = ceil,
    /// 0x0b = trunc, 0x08 = nearest (the `0x08` bit suppresses the precision
    /// exception). The caller must ensure SSE4.1 is present.
    pub fn roundsd_xmm0(&mut self, mode: u8) {
        self.code
            .extend_from_slice(&[0x66, 0x0f, 0x3a, 0x0b, 0xc0, mode]);
    }
    /// `movsd xmm1, [rbp+disp]`.
    pub fn movsd_xmm1_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x8d]);
        self.rbp_disp(disp);
    }
    /// `maxsd xmm0, xmm1` — SSE max (returns xmm1 on NaN/equal; callers handle those).
    pub fn maxsd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x5f, 0xc1]);
    }
    /// `minsd xmm0, xmm1` — SSE min (returns xmm1 on NaN/equal; callers handle those).
    pub fn minsd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x5d, 0xc1]);
    }
    /// `andpd xmm0, xmm1`.
    pub fn andpd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x54, 0xc1]);
    }
    /// `orpd xmm0, xmm1`.
    pub fn orpd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x56, 0xc1]);
    }
    /// `addsd xmm0, xmm1` — used to coalesce a NaN operand into a NaN result.
    pub fn addsd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x58, 0xc1]);
    }
    /// `subsd xmm1, xmm0` — `xmm1 = xmm1 - xmm0` (used by float `%`).
    pub fn subsd_xmm1_xmm0(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x5c, 0xc8]);
    }
    /// `movsd [rbp+disp], xmm1`.
    pub fn movsd_mem_xmm1(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, 0x8d]);
        self.rbp_disp(disp);
    }
    /// `jp label` — jump if parity (an unordered `ucomisd`, i.e. a NaN operand).
    pub fn jp(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x8a]);
        self.emit_rel32(label);
    }
    /// `jne label` — jump if not equal (ZF clear).
    pub fn jne(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x85]);
        self.emit_rel32(label);
    }
    /// `xmm0 = |xmm0|` — clear the f64 sign bit by `and`ing with `0x7fff…` built
    /// in `xmm1` (`pcmpeqd` → all-ones, `psrlq 1` → mask, `andpd`).
    pub fn abs_xmm0(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x76, 0xc9]); // pcmpeqd xmm1, xmm1
        self.code.extend_from_slice(&[0x66, 0x0f, 0x73, 0xd1, 0x01]); // psrlq xmm1, 1
        self.code.extend_from_slice(&[0x66, 0x0f, 0x54, 0xc1]); // andpd xmm0, xmm1
    }
    /// `sete al; setnp cl; and al, cl; movzx rax, al` — after a `ucomisd`, leaves
    /// `rax = (ordered && equal) ? 1 : 0` (false for NaN; true for `+0` vs `-0`).
    pub fn ordered_equal_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x94, 0xc0]); // sete al
        self.code.extend_from_slice(&[0x0f, 0x9b, 0xc1]); // setnp cl
        self.code.extend_from_slice(&[0x20, 0xc8]); // and al, cl
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }
    /// `ucomisd xmm0, xmm1`.
    pub fn ucomisd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x2e, 0xc1]);
    }
    /// `seta al; movzx rax, al` — `rax = (above: ordered and >) ? 1 : 0` after a
    /// `ucomisd` (`above` is false for NaN, matching JS ordered comparison).
    pub fn seta_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x97, 0xc0]); // seta al
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }
    /// `cvtsi2sd xmm0, rax` — convert the integer in `rax` to an `f64` in `xmm0`.
    pub fn cvtsi2sd_xmm0_rax(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x48, 0x0f, 0x2a, 0xc0]);
    }
    /// `<addsd|subsd|mulsd|divsd> xmm0, [rbp+disp]`.
    pub fn fbin_xmm0_mem(&mut self, op: FBinOp, disp: i32) {
        let opcode = match op {
            FBinOp::Add => 0x58,
            FBinOp::Sub => 0x5c,
            FBinOp::Mul => 0x59,
            FBinOp::Div => 0x5e,
        };
        self.code.extend_from_slice(&[0xf2, 0x0f, opcode, 0x85]);
        self.rbp_disp(disp);
    }

    /// `mov rax, rdi` — seed the accumulator with the first argument.
    pub fn mov_rax_rdi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xf8]);
    }

    /// `mov rax, rsi` — seed the accumulator with the second argument.
    pub fn mov_rax_rsi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xf0]);
    }

    /// `add rax, rsi`.
    pub fn add_rax_rsi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x01, 0xf0]);
    }

    /// `sub rax, rsi`.
    pub fn sub_rax_rsi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x29, 0xf0]);
    }

    /// `imul rax, rsi`.
    pub fn imul_rax_rsi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc6]);
    }

    /// `add rax, imm32` (sign-extended to 64 bits).
    pub fn add_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x05]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `sub rax, imm32`.
    pub fn sub_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x2d]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `and rax, imm32`.
    pub fn and_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x25]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `or rax, imm32`.
    pub fn or_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x0d]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `xor rax, imm32`.
    pub fn xor_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x35]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `imul rax, rax, imm32`.
    pub fn imul_rax_imm(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x69, 0xc0]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `shl rax, imm8`.
    pub fn shl_rax_imm(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe0, imm]);
    }

    /// `sar rax, imm8` (arithmetic right shift).
    pub fn sar_rax_imm(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0xc1, 0xf8, imm]);
    }

    /// `neg rax`.
    pub fn neg_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0xf7, 0xd8]);
    }

    /// `ret`.
    pub fn ret(&mut self) {
        self.code.push(0xc3);
    }

    /// `call rax` — an indirect native call through `rax` (holding the callee's
    /// code address). The System V ABI requires `rsp` 16-byte aligned here.
    pub fn call_rax(&mut self) {
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        self.inline_frame_hook();
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        self.spill_before_call();
        self.code.extend_from_slice(&[0xff, 0xd0]);
        self.clear_recv_cache();
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        self.reload_after_call();
    }

    /// `mov <argreg[i]>, [rbp+disp]` — load a frame slot into System V integer
    /// argument register `i` (rdi, rsi, rdx, rcx, r8, r9), for a native call.
    pub fn load_argreg(&mut self, i: usize, disp: i32) {
        // REX prefix + ModRM reg field per argument register.
        let (rex, modrm): (u8, u8) = match i {
            0 => (0x48, 0xbd), // rdi
            1 => (0x48, 0xb5), // rsi
            2 => (0x48, 0x95), // rdx
            3 => (0x48, 0x8d), // rcx
            4 => (0x4c, 0x85), // r8
            5 => (0x4c, 0x8d), // r9
            _ => return,
        };
        self.code.extend_from_slice(&[rex, 0x8b, modrm]);
        self.rbp_disp(disp);
    }

    /// `movabs <argreg[i]>, imm64` — load a 64-bit immediate into System V integer
    /// argument register `i` (rdi, rsi, rdx, rcx, r8, r9), for a native call (e.g.
    /// a baked-in key pointer / length / inline-cache pointer).
    pub fn movabs_argreg(&mut self, i: usize, imm: i64) {
        // REX.W (+ REX.B for r8/r9) + `B8+rd` opcode per argument register.
        let (rex, op): (u8, u8) = match i {
            0 => (0x48, 0xbf), // rdi
            1 => (0x48, 0xbe), // rsi
            2 => (0x48, 0xba), // rdx
            3 => (0x48, 0xb9), // rcx
            4 => (0x49, 0xb8), // r8
            5 => (0x49, 0xb9), // r9
            _ => return,
        };
        self.code.extend_from_slice(&[rex, op]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `lea <argreg[i]>, [rbp+disp32]` — load the *address* of a frame slot into
    /// System V integer argument register `i` (rdi, rsi, rdx, rcx, r8, r9), for
    /// passing a pointer to an in-frame scratch buffer (the generic-tier call
    /// argument marshaling).
    pub fn lea_argreg(&mut self, i: usize, disp: i32) {
        // REX.W (+ REX.R for r8/r9) + `8D /r`, ModRM mod=10 (disp32), rm=101 (rbp).
        let (rex, modrm): (u8, u8) = match i {
            0 => (0x48, 0xbd), // rdi
            1 => (0x48, 0xb5), // rsi
            2 => (0x48, 0x95), // rdx
            3 => (0x48, 0x8d), // rcx
            4 => (0x4c, 0x85), // r8
            5 => (0x4c, 0x8d), // r9
            _ => return,
        };
        self.code.extend_from_slice(&[rex, 0x8d, modrm]);
        self.rbp_disp(disp);
    }

    /// `sub rsp, imm8`.
    pub fn sub_rsp_imm8(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, imm]);
    }
    /// `add rsp, imm8`.
    pub fn add_rsp_imm8(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xc4, imm]);
    }

    // --- generic tier, frame ABI: `extern "C" fn(ctx, regs, entry_pc) -> u64` ---

    /// `push rbp; mov rbp, rsp; push r15; push rbx; sub rsp, frame` — the frame-ABI
    /// prologue. `r15` pins `ctx` and `rbx` the interpreter register file (both
    /// callee-saved). `frame` must be a multiple of 16 so `rsp` is 16-byte aligned
    /// before any `call`.
    pub fn frame_prologue(&mut self, frame: u32) {
        self.code.push(0x55); // push rbp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
        self.code.extend_from_slice(&[0x41, 0x57]); // push r15
        self.code.push(0x53); // push rbx
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]); // sub rsp, imm32
        self.code.extend_from_slice(&frame.to_le_bytes());
        self.clear_recv_cache();
        self.clear_entry_slots();
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        for i in 0..self.inline_clear.len() {
            let d = self.inline_clear[i];
            self.store_empty(d);
        }
    }

    /// `lea rsp, [rbp-16]; pop rbx; pop r15; pop rbp; ret` — the frame-ABI
    /// epilogue (the result is already in `rax`).
    pub fn frame_epilogue(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x8d, 0x65, 0xf0]); // lea rsp, [rbp-16]
        self.code.push(0x5b); // pop rbx
        self.code.extend_from_slice(&[0x41, 0x5f]); // pop r15
        self.code.push(0x5d); // pop rbp
        self.code.push(0xc3); // ret
    }

    /// `mov rbx, rsi` — pin the register-file argument.
    pub fn mov_rbx_rsi(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xf3]);
    }

    /// `mov rsi, rbx` — pass the register file as a helper's second argument.
    pub fn mov_rsi_rbx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xde]);
    }

    /// `cmp rdx, imm32` (sign-extended).
    pub fn cmp_rdx_imm32(&mut self, imm: i32) {
        self.code.extend_from_slice(&[0x48, 0x81, 0xfa]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `movq xmm1, rax` — move raw f64 bits into `xmm1`.
    pub fn cvt_rax_bits_to_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x48, 0x0f, 0x6e, 0xc8]);
    }

    /// `subsd xmm0, xmm1`.
    pub fn subsd_xmm0_xmm1(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x5c, 0xc1]);
    }

    /// `test rdx, rdx`.
    pub fn test_rdx_rdx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x85, 0xd2]);
    }

    /// `setae al; movzx rax, al` — `rax = (carry clear) ? 1 : 0`; after `ucomisd`
    /// this is "greater or equal, and ordered".
    pub fn setae_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x93, 0xc0]); // setae al
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }

    /// `mov qword [rbp+disp8], imm32` (sign-extended) — initialize a frame slot.
    pub fn mov_frame8_imm32(&mut self, disp8: i8, imm: i32) {
        self.code
            .extend_from_slice(&[0x48, 0xc7, 0x45, disp8.to_le_bytes()[0]]);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `dec qword [rbp+disp8]` (sets ZF when it reaches zero).
    pub fn dec_frame8(&mut self, disp8: i8) {
        self.code
            .extend_from_slice(&[0x48, 0xff, 0x4d, disp8.to_le_bytes()[0]]);
    }

    /// `mov qword [rbp+disp8], rax`.
    pub fn store_rax_frame8(&mut self, disp8: i8) {
        self.code
            .extend_from_slice(&[0x48, 0x89, 0x45, disp8.to_le_bytes()[0]]);
    }

    // --- generic (NanBox) tier: `extern "C" fn(ctx, args, n) -> u64` ---
    //
    // The entry pins `ctx` (rdi) in the callee-saved `r15`, homes the NanBox
    // register file in rbp-relative slots, and calls runtime helpers via the
    // System V integer ABI. These emit the extra instructions that convention
    // needs; the value-arithmetic reuses the SSE2/GPR helpers above.

    /// `push rbp; mov rbp, rsp; push r15; sub rsp, frame` — the generic-tier
    /// prologue. `r15` (callee-saved) is preserved so it can pin `ctx` across the
    /// body; `frame` must be chosen so `rsp` is 16-byte aligned before any `call`
    /// (see [`JitFunction::compile_generic`]).
    pub fn generic_prologue(&mut self, frame: u32) {
        self.code.push(0x55); // push rbp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]); // mov rbp, rsp
        self.code.extend_from_slice(&[0x41, 0x57]); // push r15
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]); // sub rsp, imm32
        self.code.extend_from_slice(&frame.to_le_bytes());
    }

    /// `lea rsp, [rbp-8]; pop r15; pop rbp; ret` — the generic-tier epilogue
    /// (restores the pinned `r15` and unwinds; the result is already in `rax`).
    pub fn generic_epilogue(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x8d, 0x65, 0xf8]); // lea rsp, [rbp-8]
        self.code.extend_from_slice(&[0x41, 0x5f]); // pop r15
        self.code.push(0x5d); // pop rbp
        self.code.push(0xc3); // ret
    }

    /// `mov r15, rdi` — pin the `ctx` argument in the callee-saved register.
    pub fn mov_r15_rdi(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x89, 0xff]);
    }

    /// `mov rdi, r15` — reload the pinned `ctx` into the first argument register
    /// ahead of a helper `call`.
    pub fn mov_rdi_r15(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x89, 0xff]);
    }

    /// `mov rax, [rsi + disp32]` — load an incoming argument from the `args`
    /// pointer (which arrives in `rsi`), for the generic-tier `Arg` op.
    pub fn mov_rax_rsi_disp(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x86]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `and rax, r11` — mask a NanBox word by the quiet-NaN pattern held in `r11`
    /// (the generic-tier `is_number` tag test).
    pub fn and_rax_r11(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x21, 0xd8]);
    }

    /// `ucomisd xmm0, xmm0` — unordered self-compare (sets PF iff `xmm0` is NaN),
    /// used to canonicalize a NaN sum before reboxing it as a `NanBox`.
    pub fn ucomisd_xmm0_xmm0(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x2e, 0xc0]);
    }

    /// `mov rcx, rax` — copy the accumulator into the scratch register.
    pub fn mov_rcx_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x89, 0xc1]);
    }

    /// `and rcx, r11` — mask `rcx` by the value held in `r11` (generic-tier
    /// `is_number` tag test, keeping the original value in `rax`).
    pub fn and_rcx_r11(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x21, 0xd9]);
    }

    /// `cmp rcx, r11`.
    pub fn cmp_rcx_r11(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd9]);
    }

    /// `add rax, r11` — add the value in `r11` (used to box a native `0`/`1` boolean
    /// into a `NanBox` by adding the `TAG_FALSE` base: `TAG_FALSE + 0/1`).
    pub fn add_rax_r11(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x01, 0xd8]);
    }

    /// `setne al; movzx rax, al` — `rax = (zero flag clear) ? 1 : 0` after a
    /// `test`/`cmp`/`ucomisd` (for `ucomisd`, this is 0 for a `+0`/`-0`/NaN operand).
    pub fn setne_rax(&mut self) {
        self.code.extend_from_slice(&[0x0f, 0x95, 0xc0]); // setne al
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]); // movzx rax, al
    }

    /// Emits code computing the JS truthiness of the `NanBox` in frame slot `disp`
    /// into `rax` as `0`/`1`. Inlines the two common cases — a number (truthy iff
    /// nonzero and not NaN) and a boolean (`tag_true`/`tag_false`) — and falls back
    /// to `jit_helper_truthy(ctx, v)` (the interpreter's `ToBoolean`) for anything
    /// else. Clobbers `rax`/`rcx`/`r11`/`xmm0`/`xmm1`; `ctx` must be pinned in `r15`
    /// (the generic-tier invariant), and `qnan` is the quiet-NaN tag mask.
    pub fn emit_truthy(
        &mut self,
        disp: i32,
        truthy_helper: u64,
        qnan: u64,
        tags: (u64, u64),
        realm_off: i32,
    ) {
        self.emit_truthy_or(disp, truthy_helper, qnan, tags, realm_off, None);
    }

    /// [`emit_truthy`](Self::emit_truthy), but with `slow` set, a value the
    /// inline cases do not decide jumps there instead of calling the helper.
    pub fn emit_truthy_or(
        &mut self,
        disp: i32,
        truthy_helper: u64,
        qnan: u64,
        (tag_true, tag_false): (u64, u64),
        realm_off: i32,
        slow: Option<Label>,
    ) {
        let numl = self.new_label();
        let set1 = self.new_label();
        let set0 = self.new_label();
        let have = self.new_label();
        let helper = self.new_label();
        self.load_rax(disp);
        // A heap handle first: the usual condition that is neither a Number
        // nor a boolean is an object (`if (node)`, `if (!p) return`), and a
        // proven Number or boolean never gets here (`numeric`).
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        let heap = self.new_label();
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        {
            self.mov_rcx_rax();
            self.raw(&[0x48, 0xc1, 0xe9, 50]); // shr rcx, 50
            self.raw(&[0x81, 0xf9, 0xff, 0x3f, 0x00, 0x00]); // cmp ecx, 0x3fff
            self.je(heap);
        }
        // is_number(v) == (v & QNAN) != QNAN — test on a copy, keeping v in rax.
        self.movabs_r11(qnan as i64);
        self.mov_rcx_rax();
        self.and_rcx_r11();
        self.cmp_rcx_r11();
        self.jne(numl); // a number → numeric truthiness
        // boolean? compare the raw word against the two boolean tags.
        self.movabs_r11(tag_true as i64);
        self.cmp_rax_r11();
        self.je(set1);
        self.movabs_r11(tag_false as i64);
        self.cmp_rax_r11();
        self.je(set0);
        // `null`/`undefined` inline; an ordinary object or a VM function
        // inline above.
        #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
        {
            self.p_truthy_nullish(set0);
            self.jmp(helper);
            self.bind(heap);
            self.p_truthy_handle(realm_off, set1, helper);
        }
        #[cfg(not(all(feature = "alloc", target_os = "linux", target_arch = "x86_64")))]
        let _ = realm_off;
        self.bind(helper);
        if let Some(slow) = slow {
            self.jmp(slow);
        }
        // otherwise: jit_helper_truthy(ctx, v) → 0/1 (runs ToBoolean; never throws).
        self.mov_rdi_r15();
        self.load_argreg(1, disp);
        self.movabs_rax(truthy_helper as i64);
        self.call_rax();
        self.jmp(have);
        // number path: truthy iff the f64 is ordered and != 0 (NaN and ±0 → falsy).
        self.bind(numl);
        self.movsd_xmm0_mem(disp);
        self.zero_xmm1();
        self.ucomisd_xmm0_xmm1();
        self.setne_rax();
        self.jmp(have);
        self.bind(set1);
        self.movabs_rax(1);
        self.jmp(have);
        self.bind(set0);
        self.zero_rax();
        self.bind(have);
    }

    // --- inline monomorphic property-GET fast path (generic tier, Pass 6).
    // These read the object heap at runtime-verified offsets (see
    // `compute_jit_layout`): a slot pointer is built in `rcx`, the arena base in
    // `r10`, the handle generation in `r8`, and the cached shape/slot come from a
    // `PropertyCache` pointer in `r11`. Every one is a plain read; on any guard
    // miss the emitter jumps to the existing helper slow path. ---

    /// `mov r10, rax` — stash the arena base returned by `jit_arena`.
    pub fn mov_r10_rax(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x89, 0xc2]);
    }
    /// `mov r8, rax` — copy the NanBox handle bits for generation extraction.
    pub fn mov_r8_rax(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x89, 0xc0]);
    }
    /// `shr r8, imm8` — logical right shift (isolate the handle generation).
    pub fn shr_r8_imm(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x49, 0xc1, 0xe8, imm]);
    }
    /// `cmp rax, rdx`.
    pub fn cmp_rax_rdx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x39, 0xd0]);
    }
    /// `jae label` (jump if unsigned above-or-equal, CF=0) — an out-of-bounds
    /// index/slot fails the check.
    pub fn jae(&mut self, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x83]);
        self.emit_rel32(label);
    }
    /// `add rax, r10` — `rax += arena base`.
    pub fn add_rax_r10(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x01, 0xd0]);
    }
    /// `cmp byte [rcx+disp32], imm8` — compare a tag byte at the slot pointer.
    pub fn cmp_byte_at_rcx_imm(&mut self, disp: i32, imm: u8) {
        self.code.extend_from_slice(&[0x80, 0xb9]);
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm);
    }
    /// `cmp word [rcx+disp32], r8w` — compare the slot's `generation: u16` against
    /// the handle generation (low 16 bits of `r8`).
    pub fn cmp_word_at_rcx_r8w(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x66, 0x44, 0x39, 0x81]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `mov rax, [rcx+disp32]` — load an 8-byte field at the slot pointer.
    pub fn mov_rax_at_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x81]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `cmp rax, [rcx+disp32]`.
    pub fn cmp_rax_at_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x3b, 0x81]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `mov r11, [rcx+disp32]` — load the slot `Vec`'s data pointer.
    pub fn mov_r11_at_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x4c, 0x8b, 0x99]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `cmp rax, [r11+disp32]` — compare against a `PropertyCache` field (the
    /// cached shape word).
    pub fn cmp_rax_at_r11(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x49, 0x3b, 0x83]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `mov eax, [r11+disp32]` — load a 32-bit `PropertyCache` field (the cached
    /// slot index), zero-extended into `rax`.
    pub fn mov_eax_at_r11(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x41, 0x8b, 0x83]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `mov rax, [r11 + rax*8]` — load the NanBox at slot index `rax` from the slot
    /// `Vec` data pointer in `r11`.
    pub fn mov_rax_at_r11_rax8(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x8b, 0x04, 0xc3]);
    }
    /// `add dword [r11+disp32], imm8` — bump a `PropertyCache` counter in place
    /// (the inline-hit `hits` increment; sign-extended imm8).
    pub fn add_dword_at_r11_imm(&mut self, disp: i32, imm: u8) {
        self.code.extend_from_slice(&[0x41, 0x83, 0x83]);
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm);
    }

    // --- inline property-SET + array-element GET/SET fast paths (generic tier).
    // These extend the Pass-6 property-GET primitives with in-place stores and the
    // array-cell reads at runtime-verified offsets (see `compute_jit_layout`). ---

    /// `mov rdx, [rbp+disp]` — load a register-file slot (the value to store) into
    /// `rdx`, the source for an in-place slot/element write.
    pub fn load_rdx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x95]);
        self.rbp_disp(disp);
    }
    /// `mov [r11 + rax*8], rdx` — store `rdx` at index `rax` of the `Vec` data
    /// pointer in `r11` (the inline slot / element write; mirror of
    /// [`mov_rax_at_r11_rax8`](Self::mov_rax_at_r11_rax8)).
    pub fn mov_at_r11_rax8_rdx(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x89, 0x14, 0xc3]);
    }
    /// `cmp qword [r11 + rax*8], rdx` — compare an element slot with `rdx`.
    pub fn cmp_at_r11_rax8_rdx(&mut self) {
        self.code.extend_from_slice(&[0x49, 0x39, 0x14, 0xc3]);
    }
    /// `cmp qword [rcx+disp32], imm8` — compare an 8-byte field at the slot pointer
    /// against a small sign-extended immediate (the `readonly` Vec length vs 0).
    pub fn cmp_qword_at_rcx_imm(&mut self, disp: i32, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xb9]);
        self.code.extend_from_slice(&disp.to_le_bytes());
        self.code.push(imm);
    }
    /// `mov r8, [rcx+disp32]` — load an 8-byte field at the slot pointer into `r8`
    /// (the array element `Vec`'s dense length).
    pub fn mov_r8_at_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x4c, 0x8b, 0x81]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `cmp rax, r8` — compare the (unsigned) element index in `rax` against the
    /// dense length in `r8`; a following `jae` bails on OOB or a negative index.
    pub fn cmp_rax_r8(&mut self) {
        self.code.extend_from_slice(&[0x4c, 0x39, 0xc0]);
    }
    /// `cvttsd2si rax, xmm0` — convert `xmm0` (an f64) to a signed i64 in `rax`
    /// with truncation toward zero (an out-of-range / NaN operand yields the x86
    /// "integer indefinite" `i64::MIN`, which the subsequent guards reject).
    pub fn cvttsd2si_rax_xmm0(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x48, 0x0f, 0x2c, 0xc0]);
    }
    /// `cvtsi2sd xmm1, rax` — convert the i64 in `rax` back to an f64 in `xmm1`, so
    /// a `ucomisd xmm0, xmm1` can verify the index was an exact integer (a
    /// fractional index like `a[1.5]` fails the round-trip → helper).
    pub fn cvtsi2sd_xmm1_rax(&mut self) {
        self.code.extend_from_slice(&[0xf2, 0x48, 0x0f, 0x2a, 0xc8]);
    }
}

/// Whether the JIT can emit and run native code on this target.
#[must_use]
pub const fn available() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
}

/// A compiled native function over `i64` arguments, owning the executable
/// memory it lives in. `f1`/`f2` call into the mapped code.
pub struct JitFunction {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    buf: exec::ExecBuffer,
}

impl JitFunction {
    /// Compiles `ops` into a native `fn(i64) -> i64` (accumulator seeded with the
    /// argument). Returns `None` when the JIT is unavailable on this target.
    #[must_use]
    pub fn compile_arith(ops: &[ArithOp]) -> Option<Self> {
        let mut a = X64Assembler::new();
        a.mov_rax_rdi();
        for op in ops {
            match *op {
                ArithOp::AddImm(n) => a.add_rax_imm(n),
                ArithOp::SubImm(n) => a.sub_rax_imm(n),
                ArithOp::MulImm(n) => a.imul_rax_imm(n),
                ArithOp::AndImm(n) => a.and_rax_imm(n),
                ArithOp::OrImm(n) => a.or_rax_imm(n),
                ArithOp::XorImm(n) => a.xor_rax_imm(n),
                ArithOp::ShlImm(n) => a.shl_rax_imm(n),
                ArithOp::SarImm(n) => a.sar_rax_imm(n),
                ArithOp::Neg => a.neg_rax(),
            }
        }
        a.ret();
        Self::from_code(&a.finish())
    }

    /// Compiles a native binary op `fn(i64, i64) -> i64` (`a <op> b`).
    #[must_use]
    pub fn compile_binary(op: BinOp) -> Option<Self> {
        let mut a = X64Assembler::new();
        a.mov_rax_rdi();
        match op {
            BinOp::Add => a.add_rax_rsi(),
            BinOp::Sub => a.sub_rax_rsi(),
            BinOp::Mul => a.imul_rax_rsi(),
        }
        a.ret();
        Self::from_code(&a.finish())
    }

    /// Compiles a native counted loop: `fn(n) -> sum(1..=n)` for `n >= 0`, else
    /// `0`. Demonstrates real native control flow (a backward branch). The
    /// emitted code is:
    ///
    /// ```text
    ///   xor   rax, rax        ; acc = 0
    ///   mov   rcx, rdi        ; i   = n
    /// loop:
    ///   test  rcx, rcx
    ///   jle   done            ; while i > 0
    ///   add   rax, rcx        ; acc += i
    ///   dec   rcx             ; i--
    ///   jmp   loop
    /// done:
    ///   ret
    /// ```
    #[must_use]
    pub fn compile_sum_1_to_n() -> Option<Self> {
        let mut a = X64Assembler::new();
        let loop_top = a.new_label();
        let done = a.new_label();
        a.zero_rax();
        a.mov_rcx_rdi();
        a.bind(loop_top);
        a.test_rcx_rcx();
        a.jle(done);
        a.add_rax_rcx();
        a.dec_rcx();
        a.jmp(loop_top);
        a.bind(done);
        a.ret();
        Self::from_code(&a.finish())
    }

    /// Compiles a [`StackOp`] program to a native `fn(i64, i64) -> i64`, using
    /// the hardware stack for operands: a push-instruction per `Arg`/`Const`, a
    /// `pop rcx; pop rax; <op> rax, rcx; push rax` per binary op, and a final
    /// `pop rax; ret`. Each program is stack-balanced, so `rsp` is restored and
    /// no calls occur, keeping the ABI happy without explicit alignment. Returns
    /// `None` on the unavailable target or a malformed (non-single-result)
    /// program.
    #[must_use]
    pub fn compile_stack(ops: &[StackOp]) -> Option<Self> {
        // Validate that the program leaves exactly one value (a quick verifier,
        // the spirit of §2.2's bytecode validation).
        let mut depth: i32 = 0;
        for op in ops {
            match op {
                StackOp::Arg(_) | StackOp::Const(_) => depth += 1,
                StackOp::Add | StackOp::Sub | StackOp::Mul => {
                    if depth < 2 {
                        return None;
                    }
                    depth -= 1;
                }
            }
        }
        if depth != 1 {
            return None;
        }
        let mut a = X64Assembler::new();
        for op in ops {
            match *op {
                StackOp::Arg(0) => a.push_rdi(),
                StackOp::Arg(_) => a.push_rsi(),
                StackOp::Const(n) => {
                    a.movabs_rax(n);
                    a.push_rax();
                }
                StackOp::Add | StackOp::Sub | StackOp::Mul => {
                    a.pop_rcx(); // b
                    a.pop_rax(); // a
                    match op {
                        StackOp::Add => a.add_rax_rcx(),
                        StackOp::Sub => a.sub_rax_rcx(),
                        _ => a.imul_rax_rcx(),
                    }
                    a.push_rax();
                }
            }
        }
        a.pop_rax();
        a.ret();
        Self::from_code(&a.finish())
    }

    /// Compiles a [`RegOp`] register-machine program (up to 6 integer args) to a
    /// native function, using a stack frame: each of `n_regs` virtual registers
    /// is homed to an `i64` slot at `[rbp - (r+1)*8]` (a spill-everything
    /// allocation, with `rax`/`rcx` as scratch). Returns `None` on the
    /// unavailable target, `>64` registers, or a malformed program (a register or
    /// arg index out of range, or no `Ret`).
    #[must_use]
    pub fn compile_reg(n_regs: usize, n_args: usize, ops: &[RegOp]) -> Option<Self> {
        if n_regs > 64 || n_args > 6 {
            return None;
        }
        let ok_reg = |r: u8| (r as usize) < n_regs;
        let disp = |r: u8| -((i32::from(r) + 1) * 8);
        // Frame: n_regs slots, rounded up to 16-byte alignment.
        let frame = ((n_regs as u32 * 8) + 15) & !15;
        let mut a = X64Assembler::new();
        // One label per op so any op can be a branch target; bound just before
        // that op's code is emitted. Plus a shared deopt trampoline.
        let labels: Vec<Label> = (0..ops.len()).map(|_| a.new_label()).collect();
        let deopt = a.new_label();
        a.prologue(frame);
        // Zero every register slot, so an (unexpected) read of an unwritten slot
        // yields 0 rather than stack garbage — defense in depth behind
        // `lower_nbvm`'s def-use check.
        a.zero_rax();
        for r in 0..n_regs {
            a.store_rax(-((r as i32 + 1) * 8));
        }
        // Hoist the safe-integer bounds (±2^53) into scratch regs r10/r11, so the
        // per-op range guard is two register compares. Loaded after the prologue;
        // r10/r11 are caller-saved and not argument registers, so no arg clobber.
        a.movabs_r11(SAFE_INT_MAX);
        a.movabs_r10(-SAFE_INT_MAX);
        // Emits the deopt guard for a result in `rax`: a signed-overflow check
        // (when `ovf`) and a ±2^53 range check, so every value the JIT keeps is a
        // value `f64` represents exactly — else it bails to the interpreter.
        macro_rules! guard {
            ($asm:expr, $ovf:expr) => {{
                if $ovf {
                    $asm.jo(deopt);
                }
                $asm.cmp_rax_r11();
                $asm.jg(deopt);
                $asm.cmp_rax_r10();
                $asm.jl(deopt);
            }};
        }
        let mut has_ret = false;
        for (i, op) in ops.iter().enumerate() {
            a.bind(labels[i]);
            match *op {
                RegOp::Arg { dst, index } => {
                    if !ok_reg(dst) || index as usize >= n_args {
                        return None;
                    }
                    a.store_arg(index as usize, disp(dst));
                }
                RegOp::Const { dst, imm } => {
                    if !ok_reg(dst) {
                        return None;
                    }
                    a.movabs_rax(imm);
                    a.store_rax(disp(dst));
                }
                RegOp::Bin {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    a.load_rax(disp(ra));
                    a.op_rax_mem(op, disp(rb));
                    // Add/Sub/Mul can overflow i64; And/Or/Xor cannot, but all can
                    // leave the exact-integer range, so range-check every result.
                    let can_overflow = matches!(op, BinOp2::Add | BinOp2::Sub | BinOp2::Mul);
                    guard!(a, can_overflow);
                    a.store_rax(disp(dst));
                }
                RegOp::Move { dst, src } => {
                    if !ok_reg(dst) || !ok_reg(src) {
                        return None;
                    }
                    a.load_rax(disp(src));
                    a.store_rax(disp(dst));
                }
                RegOp::Lt { dst, a: ra, b: rb } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    a.load_rax(disp(ra));
                    a.cmp_rax_mem(disp(rb));
                    a.setl_rax();
                    a.store_rax(disp(dst));
                }
                RegOp::BitNot32 { dst, a: ra } => {
                    if !ok_reg(dst) || !ok_reg(ra) {
                        return None;
                    }
                    // `not eax` complements the low 32 bits; sign-extend the i32.
                    a.load_rax(disp(ra));
                    a.not_eax();
                    a.to_int32_rax();
                    a.store_rax(disp(dst));
                }
                RegOp::Mod { dst, a: ra, b: rb } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    // Deopt on a zero divisor (JS `%0` is NaN); else signed idiv,
                    // keeping the remainder (rdx). |rem| < |b| ≤ 2^53 → in range.
                    a.load_rcx(disp(rb));
                    a.test_rcx_rcx();
                    a.je(deopt);
                    a.load_rax(disp(ra));
                    // A negative dividend can leave a zero remainder, which JS
                    // signs `-0`: the interpreter's.
                    a.test_rax_rax();
                    a.jl(deopt);
                    a.cqo();
                    a.idiv_rcx();
                    a.mov_rax_rdx();
                    a.store_rax(disp(dst));
                }
                RegOp::Eqz { dst, a: ra } => {
                    if !ok_reg(dst) || !ok_reg(ra) {
                        return None;
                    }
                    a.load_rax(disp(ra));
                    a.test_rax_rax();
                    a.sete_rax(); // rax = (reg[ra] == 0) ? 1 : 0
                    a.store_rax(disp(dst));
                }
                RegOp::Eq { dst, a: ra, b: rb } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    a.load_rax(disp(ra));
                    a.cmp_rax_mem(disp(rb));
                    a.sete_rax(); // rax = (reg[ra] == reg[rb]) ? 1 : 0
                    a.store_rax(disp(dst));
                }
                RegOp::Neg { dst, a: ra } => {
                    if !ok_reg(dst) || !ok_reg(ra) {
                        return None;
                    }
                    a.load_rax(disp(ra));
                    a.neg_rax();
                    guard!(a, true); // deopt on i64::MIN overflow or out-of-range
                    a.store_rax(disp(dst));
                }
                RegOp::Bit32 {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    // ToInt32 each operand (truncate to 32 bits, sign-extend), then
                    // the bitwise op. The result is an exact i32 → no range guard.
                    a.load_rax(disp(ra));
                    a.to_int32_rax();
                    a.load_rcx(disp(rb));
                    a.to_int32_rcx();
                    a.bit_rax_rcx(op);
                    a.store_rax(disp(dst));
                }
                RegOp::Shift32 {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                } => {
                    if !ok_reg(dst) || !ok_reg(ra) || !ok_reg(rb) {
                        return None;
                    }
                    // 32-bit shift of the operand by `cl` (count masked to 5 bits).
                    // The 32-bit op zero-extends; `<<`/`>>` then sign-extend the i32
                    // result, `>>>` keeps the zero-extended u32. Always in ±2^53.
                    a.load_rax(disp(ra));
                    a.load_rcx(disp(rb));
                    a.shift_eax_cl(op);
                    if matches!(op, ShiftOp::Shl | ShiftOp::Sar) {
                        a.to_int32_rax(); // sign-extend the signed i32 result
                    }
                    a.store_rax(disp(dst));
                }
                RegOp::JumpIfFalse { cond, target } => {
                    if !ok_reg(cond) || target >= ops.len() {
                        return None;
                    }
                    a.load_rax(disp(cond));
                    a.test_rax_rax();
                    a.je(labels[target]); // jump if rax == 0 (falsy)
                }
                RegOp::Jump { target } => {
                    if target >= ops.len() {
                        return None;
                    }
                    a.jmp(labels[target]);
                }
                RegOp::Ret { src } => {
                    if !ok_reg(src) {
                        return None;
                    }
                    a.load_rax(disp(src));
                    a.epilogue();
                    has_ret = true;
                    // Do NOT break: later ops may be branch targets.
                }
                RegOp::Call {
                    dst,
                    code_ptr,
                    n_args,
                    args,
                } => {
                    if !ok_reg(dst) || n_args as usize > 6 {
                        return None;
                    }
                    // Load each argument register from its slot into the System V
                    // integer arg register (rdi, rsi, rdx, rcx, r8, r9). The frame
                    // is already 16-byte aligned (frame size is a multiple of 16
                    // and rbp is aligned), so the `call` needs no rsp adjustment.
                    for (i, &arg) in args[..n_args as usize].iter().enumerate() {
                        if !ok_reg(arg) {
                            return None;
                        }
                        a.load_argreg(i, disp(arg));
                    }
                    a.movabs_rax(code_ptr as i64);
                    a.call_rax();
                    // The callee clobbered the caller-saved bounds regs; reload.
                    a.movabs_r11(SAFE_INT_MAX);
                    a.movabs_r10(-SAFE_INT_MAX);
                    // Range-guard the result: a legitimate callee result is in
                    // ±2^53, while a callee *deopt* returns the i64::MAX sentinel
                    // (out of range), so this propagates the callee's deopt to the
                    // caller instead of silently using the sentinel as a value.
                    guard!(a, false);
                    a.store_rax(disp(dst));
                }
            }
        }
        if !has_ret {
            return None;
        }
        // The deopt trampoline: return a sentinel outside the safe-integer range
        // (`i64::MAX`), which the caller recognizes as "bail to the interpreter".
        // Unreachable by fall-through (every `Ret` returns); reached only by the
        // guard jumps above.
        a.bind(deopt);
        a.movabs_rax(i64::MAX);
        a.epilogue();
        Self::from_code(&a.finish())
    }

    /// Compiles a straight-line [`FloatOp`] program to a native `fn(f64…) -> f64`
    /// (up to 4 `f64` args), homing each of `n_regs` registers to an `f64` frame
    /// slot and computing in `xmm0` with SSE2. Unlike the integer path this needs
    /// no overflow/range guard — `f64` arithmetic already matches JS numbers — and
    /// it supports division. Returns `None` on the unavailable target, `>64`
    /// registers, `>4` args, or a malformed program (bad index / no `Ret`).
    #[must_use]
    pub fn compile_float(n_regs: usize, n_args: usize, ops: &[FloatOp]) -> Option<Self> {
        if n_regs > 64 || n_args > 4 {
            return None;
        }
        let ok = |r: u8| (r as usize) < n_regs;
        let disp = |r: u8| -((i32::from(r) + 1) * 8);
        let frame = ((n_regs as u32 * 8) + 15) & !15;
        let mut a = X64Assembler::new();
        // One label per op so any op can be a branch target.
        let labels: Vec<Label> = (0..ops.len()).map(|_| a.new_label()).collect();
        a.prologue(frame);
        let mut has_ret = false;
        for (i, op) in ops.iter().enumerate() {
            a.bind(labels[i]);
            match *op {
                FloatOp::Arg { dst, index } => {
                    if !ok(dst) || index as usize >= n_args {
                        return None;
                    }
                    a.store_arg_f64(index as usize, disp(dst));
                }
                FloatOp::Const { dst, imm } => {
                    if !ok(dst) {
                        return None;
                    }
                    // Store the constant's bit pattern into the slot.
                    a.movabs_rax(imm.to_bits() as i64);
                    a.store_rax(disp(dst));
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
                    a.movsd_xmm0_mem(disp(ra));
                    a.fbin_xmm0_mem(op, disp(rb));
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Move { dst, src } => {
                    if !ok(dst) || !ok(src) {
                        return None;
                    }
                    a.movsd_xmm0_mem(disp(src));
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Lt { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    // `seta` after `ucomisd b, a` is `b > a` ordered = `a < b`
                    // (false for NaN). Convert the 0/1 to an f64 in the slot.
                    a.movsd_xmm0_mem(disp(rb));
                    a.ucomisd_xmm0_mem(disp(ra));
                    a.seta_rax();
                    a.cvtsi2sd_xmm0_rax();
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::JumpIfFalse { cond, target } => {
                    if !ok(cond) || target >= ops.len() {
                        return None;
                    }
                    // Jump when the cond slot is +0.0 (the only falsy value Lt
                    // produces); compare against a zeroed xmm1.
                    a.movsd_xmm0_mem(disp(cond));
                    a.zero_xmm1();
                    a.ucomisd_xmm0_xmm1();
                    a.je(labels[target]);
                }
                FloatOp::Jump { target } => {
                    if target >= ops.len() {
                        return None;
                    }
                    a.jmp(labels[target]);
                }
                FloatOp::Neg { dst, a: ra } => {
                    if !ok(dst) || !ok(ra) {
                        return None;
                    }
                    // -x == 0.0 - x (NaN/∞ propagate correctly).
                    a.zero_xmm0();
                    a.fbin_xmm0_mem(FBinOp::Sub, disp(ra));
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Sqrt { dst, a: ra } => {
                    if !ok(dst) || !ok(ra) {
                        return None;
                    }
                    a.movsd_xmm0_mem(disp(ra));
                    a.sqrtsd_xmm0();
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Floor { dst, a: ra }
                | FloatOp::Ceil { dst, a: ra }
                | FloatOp::Trunc { dst, a: ra } => {
                    // `roundsd` is SSE4.1; refuse to emit it on hardware without it.
                    if !ok(dst) || !ok(ra) || !has_sse41() {
                        return None;
                    }
                    // roundsd mode: 0x09 floor, 0x0a ceil, 0x0b truncate-toward-zero.
                    let mode = match op {
                        FloatOp::Floor { .. } => 0x09,
                        FloatOp::Ceil { .. } => 0x0a,
                        _ => 0x0b,
                    };
                    a.movsd_xmm0_mem(disp(ra));
                    a.roundsd_xmm0(mode);
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Mod { dst, a: ra, b: rb } => {
                    // dst = a - trunc(a/b)*b. `roundsd` (trunc) is SSE4.1. Reads
                    // [a]/[b] before writing [dst], so dst may alias a or b.
                    if !ok(dst) || !ok(ra) || !ok(rb) || !has_sse41() {
                        return None;
                    }
                    a.movsd_xmm1_mem(disp(ra)); // xmm1 = a (preserved)
                    a.movsd_xmm0_mem(disp(ra)); // xmm0 = a
                    a.fbin_xmm0_mem(FBinOp::Div, disp(rb)); // xmm0 = a/b
                    a.roundsd_xmm0(0x0b); // xmm0 = trunc(a/b)
                    a.fbin_xmm0_mem(FBinOp::Mul, disp(rb)); // xmm0 = trunc(a/b)*b
                    a.subsd_xmm1_xmm0(); // xmm1 = a - trunc(a/b)*b
                    a.movsd_mem_xmm1(disp(dst));
                }
                FloatOp::Abs { dst, a: ra } => {
                    if !ok(dst) || !ok(ra) {
                        return None;
                    }
                    a.movsd_xmm0_mem(disp(ra));
                    a.abs_xmm0();
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Max { dst, a: ra, b: rb } | FloatOp::Min { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let is_max = matches!(op, FloatOp::Max { .. });
                    // xmm0 = a, xmm1 = b; branch on the ucomisd flags:
                    //   NaN (parity)  → a + b  (propagates NaN)
                    //   equal         → andpd (max → +0) / orpd (min → -0)
                    //   otherwise     → maxsd / minsd
                    let nan = a.new_label();
                    let neq = a.new_label();
                    let done = a.new_label();
                    a.movsd_xmm0_mem(disp(ra));
                    a.movsd_xmm1_mem(disp(rb));
                    a.ucomisd_xmm0_xmm1();
                    a.jp(nan);
                    a.jne(neq);
                    if is_max {
                        a.andpd_xmm0_xmm1();
                    } else {
                        a.orpd_xmm0_xmm1();
                    }
                    a.jmp(done);
                    a.bind(neq);
                    if is_max {
                        a.maxsd_xmm0_xmm1();
                    } else {
                        a.minsd_xmm0_xmm1();
                    }
                    a.jmp(done);
                    a.bind(nan);
                    a.addsd_xmm0_xmm1();
                    a.bind(done);
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Eqz { dst, a: ra } => {
                    if !ok(dst) || !ok(ra) {
                        return None;
                    }
                    // `!x`: `ucomisd x, 0` sets ZF when x == 0 *or* x is NaN — both
                    // the JS-falsy cases — so `sete` is exactly `!x`. → 0.0/1.0.
                    a.movsd_xmm0_mem(disp(ra));
                    a.zero_xmm1();
                    a.ucomisd_xmm0_xmm1();
                    a.sete_rax();
                    a.cvtsi2sd_xmm0_rax();
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Eq { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    // Numeric `===`: ordered (ZF) and not-unordered (¬PF). `ucomisd`
                    // sets ZF for equal *and* NaN, so AND with ¬PF excludes NaN.
                    a.movsd_xmm0_mem(disp(ra));
                    a.ucomisd_xmm0_mem(disp(rb));
                    a.ordered_equal_rax();
                    a.cvtsi2sd_xmm0_rax();
                    a.movsd_mem_xmm0(disp(dst));
                }
                FloatOp::Ret { src } => {
                    if !ok(src) {
                        return None;
                    }
                    a.movsd_xmm0_mem(disp(src)); // result in xmm0
                    a.epilogue();
                    has_ret = true;
                    // Do NOT break: later ops may be branch targets.
                }
            }
        }
        if !has_ret {
            return None;
        }
        Self::from_code(&a.finish())
    }

    /// Compiles a [`GenericOp`] program (the **generic NanBox tier**) to an
    /// `extern "C" fn(ctx: *mut c_void, regs: *mut u64, entry: u64) -> u64` — the
    /// **frame ABI**.
    ///
    /// `regs` is the interpreter's own register file for the activation (`n_regs`
    /// NanBox words, already holding the parameters, captures and `this` exactly as
    /// the interpreter laid them out); the body pins it in `rbx` and `ctx` in `r15`,
    /// and every op reads and writes it in place. Because the native body and the
    /// interpreter share one register file and one pc numbering (op `i` is
    /// bytecode op `i`), an op without a native lowering is simply handed to the
    /// interpreter ([`GenericOp::Interp`] → `jit_helper_exec_op`), and the body can
    /// be entered mid-function: `entry` is the bytecode pc to start at — `0`, or a
    /// loop header in `osr_entries` (on-stack replacement from a hot interpreted
    /// loop). Value ops emit an inline both-numbers fast path plus a System-V
    /// slow-path `call` to the matching runtime helper. A helper that throws
    /// returns the `NanBox::jit_throw_bits` sentinel, on which the body returns
    /// that sentinel (the fault is pending in `ctx`); falling off the end of the
    /// op stream returns `NanBox::jit_falloff_bits`. A backward branch polls
    /// `jit_helper_backedge` (host interrupt, agent scheduling, GC safepoint)
    /// every few iterations, counted in a frame slot.
    ///
    /// `ops` is the bytecode the stream was lowered from; it must outlive the
    /// returned code (the `Interp` sites bake pointers into it).
    #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn compile_generic(
        persist: u64,
        n_regs: usize,
        ops: &[GenericOp],
        bytecode: &[crate::nbvm::Op],
        helpers: &GenericHelpers,
        sites: &[SiteInfo],
        call_args: &[Vec<u16>],
        osr_entries: &[usize],
        call_sites: &[*const CallSite],
        (n_params, n_caps, new_target): (usize, usize, Option<usize>),
        inline_closures: &[bool],
        pc_orig: &[u32],
    ) -> Option<(Self, usize)> {
        use crate::nanbox::NanBox;
        if ops.len() != bytecode.len() || n_regs > 0xffff {
            return None;
        }
        let add_helper = helpers.add as usize as u64;
        let get_site_helper = helpers.get_site as usize as u64;
        let get_method_site_helper = helpers.get_method_site as usize as u64;
        let call_with_method_helper = helpers.call_with_method as usize as u64;
        let call_native_got_helper = helpers.call_native_got as usize as u64;
        let set_site_helper = helpers.set_site as usize as u64;
        let init_prop_helper = helpers.init_prop as usize as u64;
        let get_elem_helper = helpers.get_elem as usize as u64;
        let set_elem_helper = helpers.set_elem as usize as u64;
        let array_len_helper = helpers.array_len as usize as u64;
        let arith_helper = helpers.arith as usize as u64;
        let value_bin_helper = helpers.value_bin as usize as u64;
        let lt_helper = helpers.lt as usize as u64;
        let strict_eq_helper = helpers.strict_eq as usize as u64;
        let truthy_helper = helpers.truthy as usize as u64;
        let call_native_helper = helpers.call_native as usize as u64;
        let exec_op_helper = helpers.exec_op as usize as u64;
        let backedge_helper = helpers.backedge as usize as u64;
        let push_handler_helper = helpers.push_handler as usize as u64;
        let pop_handler_helper = helpers.pop_handler as usize as u64;
        let unwind_helper = helpers.unwind as usize as u64;
        let call_method_helper = helpers.call_method as usize as u64;
        let call_value_helper = helpers.call_value as usize as u64;
        let method_helper = helpers.method_info as usize as u64;
        let construct_helper = helpers.construct_info as usize as u64;
        let construct_result_helper = helpers.construct_result as usize as u64;
        let construct_alloc_helper = helpers.construct_alloc as usize as u64;
        let super_twice_helper = helpers.super_twice as usize as u64;
        let interp_sites = interp_sites(bytecode, call_args.len(), call_sites.len())?;
        let construct_slow_helper = helpers.construct as usize as u64;
        let call_this_helper = helpers.call_this as usize as u64;
        // The catch entries of the body's `try` regions: where the throw exit
        // resumes after `jit_helper_unwind` binds a thrown value.
        let mut catch_targets: Vec<usize> = ops
            .iter()
            .filter_map(|op| match *op {
                GenericOp::PushHandler { target, .. } => Some(target),
                _ => None,
            })
            .collect();
        catch_targets.sort_unstable();
        catch_targets.dedup();
        // Runtime-verified heap layout for the inline monomorphic property-get fast
        // path (offsets/discriminants derived by pointer arithmetic on real
        // instances; see `compute_jit_layout` + the `jit_layout_matches_safe_reads`
        // verification test).
        let layout = *compute_jit_layout();
        // The widest argument list across all native-call sites; a scratch buffer
        // of this many NanBox slots is reserved in the native frame so each call
        // can marshal its arguments into a contiguous region.
        let max_argc = call_args.iter().map(Vec::len).max().unwrap_or(0);
        if max_argc > 32 {
            return None;
        }
        // `is_number(bits) == (bits & QNAN) != QNAN`.
        const QNAN: u64 = 0x7ffc_0000_0000_0000;
        // A heap handle is `SIGN | QNAN | payload`, so `(bits & HANDLE_TAG) ==
        // HANDLE_TAG` iff `bits` is a handle (both the sign and the QNAN pattern
        // set; a number clears QNAN bits, every other tag clears SIGN).
        const HANDLE_TAG: u64 = 0x8000_0000_0000_0000 | QNAN;
        // The back-edge poll countdown slot (`[rbp-24]`) and how many iterations a
        // poll covers when no agent is scheduling (the helper returns the reload).
        const COUNTDOWN: i8 = -24;
        // The canonical quiet NaN a `NanBox` normalizes every NaN to, so a NaN sum
        // reboxes bit-identically to the interpreter's `NanBox::number`.
        let canonical_nan = NanBox::number(f64::NAN).to_bits();
        // Boolean tag words. A native `0`/`1` boxes to a `NanBox` boolean by adding
        // `tag_false` (`tag_false + 0 = false`, `tag_false + 1 = true`).
        let tag_false = NanBox::boolean(false).to_bits();
        let tag_true = NanBox::boolean(true).to_bits();
        // Register `r` lives at `[rbx + 8r]` — the interpreter's register file
        // (frame mode routes non-negative displacements through `rbx`).
        let disp = |r: u16| i32::from(r) * 8;
        // The native-call scratch buffer sits below the saved `r15`/`rbx` and the
        // countdown slot. The helper reads `argv[i]` at ascending addresses, so slot
        // 0 is the deepest: slot `i` at `[rbp - 24 - (max_argc - i)*8]`.
        let argbuf_disp = |i: usize| -(24 + (max_argc as i32 - i as i32) * 8);
        // One more slot under the buffer: a method call's looked-up callee.
        let method_scratch = -(24 + (max_argc as i32 + 1) * 8);
        // Type-specialized arithmetic and loop register promotion (`numeric`),
        // whose frame slots sit under the method scratch slot.
        let mut num = numeric::NumGen::new(
            ops,
            bytecode,
            numeric::NumEnv {
                n_regs,
                bytecode: bytecode.as_ptr(),
                bytecode_len: bytecode.len(),
                add: add_helper,
                arith: arith_helper,
                value_bin: value_bin_helper,
                lt: lt_helper,
                strict_eq: strict_eq_helper,
                truthy: truthy_helper,
                exec_op: exec_op_helper,
                backedge: backedge_helper,
                global_ref: helpers.global_ref as usize as u64,
                math1: helpers.math1 as usize as u64,
                persist,
                realm: helpers.call_layout.realm,
            },
            osr_entries,
            &catch_targets,
        );
        num.set_frame_base(method_scratch);
        // Countdown slot + argument buffer + method scratch + the numeric slots,
        // rounded to keep `rsp` 16-byte aligned (after `push rbp; push r15; push
        // rbx`, rsp ≡ 0 mod 16).
        // The receiver cache (`recv_cache`), under the numeric slots: way `w`
        // keeps a receiver's handle bits at `[rbp + rc_key(w)]` and the address
        // of its heap slot, as a full walk proved it live, at
        // `[rbp + rc_cell(w)]`. The prologue and every call empty the keys.
        let gc_plan = recv_cache::plan_globals(ops, bytecode, |pc| num.in_region(pc));
        // The global binding cache's read slots persist across activations
        // (`global_persist`): word `1 + k` of the array holds slot `k`'s
        // address, word `0` the epoch they were validated under.
        let gc_reads: Vec<usize> = (0..gc_plan.slots.min(global_persist::BODY_WORDS - 1))
            .filter(|&k| {
                let mut sites = (0..ops.len()).filter(|&pc| gc_plan.site.get(pc) == Some(&Some(k)));
                sites.all(|pc| matches!(bytecode.get(pc), Some(crate::nbvm::Op::LoadGlobal { .. })))
            })
            .collect();
        let rc = recv_cache::plan(
            ops,
            bytecode,
            &gc_plan,
            |pc| num.in_region(pc),
            |pc| num.in_spec(pc),
            |pc| num.global_copy(pc),
            // A truthiness test this emitter makes (not a promoted loop's,
            // nor one of a proven Number or boolean, which `numeric` makes).
            |pc, r| !num.in_region(pc) && !num.is_number_at(pc, r) && !num.is_bool_at(pc, r),
            |pc| num.reuse_planned(pc).is_some(),
            osr_entries,
            &catch_targets,
        );
        // The ways whose every site compiles in one layout `S` (see
        // `recv_cache::shape_ways`): a hit there proves the receiver has `S`.
        let way_shape = recv_cache::shape_ways(&rc, ops, |op| match *op {
            GenericOp::GetProp { site, .. } | GenericOp::SetProp { site, .. } => {
                sites.get(site as usize)?.baked.map(|b| b.shape)
            }
            _ => None,
        });
        // Such a site's way hit skips its shape compare.
        let shape_known = |at: usize, shape: u64| {
            rc.sites[at].is_some_and(|p| p.check && way_shape[p.way] == Some(shape))
        };
        let shape_ways: Vec<usize> = (0..rc.ways).filter(|&w| way_shape[w].is_some()).collect();
        // The `InitProp`s of a fresh literal (`recv_cache::fresh_inits`).
        let fresh_init = recv_cache::fresh_inits(ops, bytecode, osr_entries, &catch_targets);
        let rc_base = method_scratch - 8 * num.frame_slots() as i32;
        let rc_key = |w: usize| rc_base - 8 * (3 * w as i32 + 1);
        let rc_cell = |w: usize| rc_base - 8 * (3 * w as i32 + 2);
        // Per way, its receiver's `extra` part pointer, as a speculative
        // loop's site last loaded it (see `extra_in_frame`).
        let rc_extra = |w: usize| rc_base - 8 * (3 * w as i32 + 3);
        // The global binding cache, under the receiver cache: slot `k` holds
        // the address `jit_helper_global_ref` validated for its name (a read
        // or a write binding). Unlike the receiver keys it stays filled
        // across calls, guarded by the global binding epoch it was validated
        // under (`gc_epoch`, one for all slots: a changed epoch empties them
        // all, as does a `catch` entry). The prologue empties both.
        let gc_base = rc_base - 24 * rc.ways as i32;
        let gc_slot = |k: usize| gc_base - 8 * (k as i32 + 1);
        let gc_epoch = gc_base - 8 * (gc_plan.slots as i32 + 1);
        let gc_frame_slots = if gc_plan.slots > 0 {
            gc_plan.slots + 1
        } else {
            0
        };
        let epoch_addr = crate::env::global_binding_epoch_addr() as i64;
        // The inline `Math.f(x)` sites' own caches, under the global binding
        // cache: per site, its receiver's handle bits (a key, emptied as the
        // receiver cache's are) and heap slot address, and the bits of the
        // method last proved the built-in (a key too).
        let math_sites: Vec<usize> = (0..ops.len())
            .filter(|&i| {
                matches!(ops[i], GenericOp::CallMethod { .. })
                    && bytecode.get(i).and_then(inline_math).is_some()
            })
            .collect();
        let m_base = gc_base - 8 * gc_frame_slots as i32;
        let m_slot = |site: usize, k: i32| m_base - 8 * (3 * site as i32 + k + 1);
        // An inlined call's active header (`inline`), under the `Math` sites'
        // slots: emptied on entry, set by a guard, popped by its leave — or
        // by the throw exit, for a throw out of the inlined body.
        let has_inline = ops
            .iter()
            .any(|o| matches!(o, GenericOp::InlineGuard { .. }));
        let inl_base = m_base - 8 * (3 * math_sites.len() as i32);
        let inl_slots = call_seq::InlineSlots {
            pend: inl_base - 8,
            hdr: inl_base - 16,
            callee: inl_base - 24,
            ok: inl_base - 32,
            max_argc: ops
                .iter()
                .filter_map(|o| match *o {
                    GenericOp::InlineGuard { call, .. } => match ops.get(call as usize)? {
                        GenericOp::CallMethod { args, .. }
                        | GenericOp::CallValue { site: args, .. }
                        | GenericOp::CallWithMethod { site: args, .. } => {
                            call_args.get(*args as usize).map(Vec::len)
                        }
                        _ => Some(0),
                    },
                    _ => None,
                })
                .max()
                .unwrap_or(0),
        };
        // A handler's catch pc as the interpreter knows it (the function's
        // own numbering: see `inline`).
        let orig_pc = |t: usize| -> usize {
            match pc_orig.get(t) {
                Some(&o) if o != u32::MAX => o as usize,
                Some(_) => usize::MAX,
                None => t,
            }
        };
        let frame = ((16
            + max_argc as u32 * 8
            + (num.frame_slots()
                + 3 * rc.ways
                + gc_frame_slots
                + 3 * math_sites.len()
                + 4 * usize::from(has_inline)) as u32
                * 8)
            + 15)
            & !15;
        let mut a = X64Assembler::new();
        a.set_rbx_frame(true);
        a.set_recv_cache_keys(
            (0..rc.ways)
                .map(rc_key)
                .chain((0..math_sites.len()).flat_map(|m| [m_slot(m, 0), m_slot(m, 2)]))
                .chain(has_inline.then_some(inl_slots.ok))
                .collect(),
        );
        // (Also the promoted loops' epoch slots: emptying them with the
        // global binding cache's merely makes the loops validate afresh.)
        let mut entry_clear = num.entry_clear_slots();
        if gc_plan.slots > 0 {
            entry_clear.extend((0..gc_plan.slots).map(gc_slot));
            entry_clear.push(gc_epoch);
        }
        if has_inline {
            // (Emptied by the prologue only: `entry_clear` is also the
            // global binding cache's reset, which may run inside an inlined
            // body.)
            a.set_inline_pend(
                Some(inl_slots.pend),
                alloc::vec![inl_slots.pend, inl_slots.hdr],
            );
        }
        if !entry_clear.is_empty() {
            a.set_entry_clear(entry_clear);
        }
        // The inlined calls' frame materialization stubs: (entry, the call's
        // argument registers).
        let mut inline_stubs: Vec<(Label, Vec<u16>)> = Vec::new();
        // The out-of-line full walks: (entry, resume, receiver, miss, way).
        let mut cold_walks: Vec<(Label, Label, u16, Label, usize, recv_cache::Family)> = Vec::new();
        // The global binding cache's refills: (entry, resume, done, op, slot).
        let mut cold_globals: Vec<(Label, Label, Label, usize, i32, usize)> = Vec::new();
        // The global binding cache's resets on a changed epoch: (entry, refill).
        let mut cold_epochs: Vec<(Label, Label, Label)> = Vec::new();
        // One label per op (any op can be a branch target or an OSR entry) plus
        // one for the end of the stream (a branch to `len` falls off the end).
        let labels: Vec<Label> = (0..=ops.len()).map(|_| a.new_label()).collect();
        let throw_exit = a.new_label();
        let refuse_exit = a.new_label();
        // Where a back-edge poll that asked the body to leave goes.
        let leave_exit = a.new_label();
        num.set_leave_exit(leave_exit);
        a.frame_prologue(frame);
        a.mov_r15_rdi(); // pin ctx
        a.mov_rbx_rsi(); // pin the register file
        a.mov_frame8_imm32(COUNTDOWN, 1); // poll at the first back-edge
        // Entry dispatch: `entry` (rdx) is 0 or a known loop header.
        for &t in osr_entries {
            if t == 0 || t >= ops.len() {
                continue;
            }
            a.cmp_rdx_imm32(i32::try_from(t).ok()?);
            a.je(labels[t]);
        }
        a.test_rdx_rdx();
        a.jne(refuse_exit); // an unknown entry is a caller bug: refuse, never misrun
        let ok = |r: u16| (r as usize) < n_regs;
        // After a helper `call`, `rax` holds the result or the throw sentinel; the
        // sentinel jumps to the shared throw exit (the caller then reads the pending
        // exception). Emitted after every throwing helper call.
        macro_rules! throw_check {
            () => {{
                a.movabs_r11(NanBox::jit_throw_bits() as i64);
                a.cmp_rax_r11();
                a.je(throw_exit);
            }};
        }
        // The interpreter runs bytecode op `pc` on the register file.
        macro_rules! exec_op {
            ($pc:expr) => {{
                let op_ptr: *const crate::nbvm::Op = bytecode.get($pc as usize)?;
                a.mov_rdi_r15(); // rdi = ctx
                a.mov_rsi_rbx(); // rsi = regs
                a.movabs_argreg(2, n_regs as i64); // rdx = n_regs
                a.movabs_argreg(3, op_ptr as i64); // rcx = &op
                a.movabs_rax(exec_op_helper as i64);
                a.call_rax();
                throw_check!();
            }};
        }
        // A loop back-edge poll: every `countdown` iterations, call the helper
        // (interrupt / agents / GC safepoint), which returns the next countdown.
        // (The helper may instead ask the body to leave for the interpreter
        // at the loop header `$header`: everything is in the register file.)
        macro_rules! backedge {
            ($header:expr) => {{
                let skip = a.new_label();
                a.dec_frame8(COUNTDOWN);
                a.jne(skip);
                a.mov_rdi_r15();
                a.mov_rsi_rbx();
                a.movabs_argreg(2, n_regs as i64);
                a.movabs_argreg(3, bytecode.as_ptr() as i64);
                a.movabs_argreg(4, bytecode.len() as i64);
                a.movabs_argreg(5, $header as i64);
                a.movabs_rax(backedge_helper as i64);
                a.call_rax();
                throw_check!();
                a.movabs_r11(NanBox::jit_exit_bits() as i64);
                a.cmp_rax_r11();
                a.je(leave_exit);
                a.store_rax_frame8(COUNTDOWN);
                a.bind(skip);
            }};
        }
        // Both operands numbers? (jumps to `$slow` otherwise).
        macro_rules! both_numbers {
            ($ra:expr, $rb:expr, $slow:expr) => {{
                a.movabs_r11(QNAN as i64);
                a.load_rax(disp($ra));
                a.and_rax_r11();
                a.cmp_rax_r11();
                a.je($slow);
                a.load_rax(disp($rb));
                a.and_rax_r11();
                a.cmp_rax_r11();
                a.je($slow);
            }};
        }
        // xmm0 = [ra] <fb> [rb], reboxed (NaN canonicalized) into dst.
        macro_rules! fbin_store {
            ($fb:expr, $ra:expr, $rb:expr, $dst:expr, $done:expr) => {{
                let nan_fix = a.new_label();
                a.movsd_xmm0_mem(disp($ra));
                a.fbin_xmm0_mem($fb, disp($rb));
                a.ucomisd_xmm0_xmm0();
                a.jp(nan_fix);
                a.movsd_mem_xmm0(disp($dst));
                a.jmp($done);
                a.bind(nan_fix);
                a.movabs_rax(canonical_nan as i64);
                a.store_rax(disp($dst));
                a.jmp($done);
            }};
        }
        // The receiver walk shared by the inline heap fast paths: from the handle in
        // `rax`, guard handle-ness / arena bounds / slot occupancy / generation and
        // leave `rcx = &Slot` (`r10` = arena base, `rdx` = arena len, `rax` = the
        // receiver's bits). Any miss jumps to `$miss`. Needs the arena just
        // loaded (`p_load_arena`: rax:rdx).
        macro_rules! walk_to_slot {
            ($recv:expr, $miss:expr) => {{
                a.mov_r10_rax(); // r10 = arena base
                a.load_rax(disp($recv)); // rax = recv bits
                // A heap handle has all of `HANDLE_TAG`'s bits (the top 14).
                const _: () = assert!(HANDLE_TAG == !0u64 << 50);
                a.mov_rcx_rax();
                a.raw(&[0x48, 0xc1, 0xe9, 50]); // shr rcx, 50
                a.raw(&[0x81, 0xf9, 0xff, 0x3f, 0x00, 0x00]); // cmp ecx, 0x3fff
                a.jne($miss); // not a heap handle
                a.mov_r8_rax();
                a.shr_r8_imm(32); // r8 = generation (low 16)
                a.raw(&[0x89, 0xc1]); // mov ecx, eax: rcx = handle index
                a.raw(&[0x48, 0x39, 0xd1]); // cmp rcx, rdx
                a.jae($miss); // index >= arena len
                if layout.slot_stride == 72 {
                    // rcx = r10 + rcx*72 in two `lea`s (shorter on the
                    // dependency chain than `imul` + `add`).
                    a.raw(&[0x48, 0x8d, 0x0c, 0xc9]); // lea rcx, [rcx + rcx*8]
                    a.raw(&[0x49, 0x8d, 0x0c, 0xca]); // lea rcx, [r10 + rcx*8]
                } else {
                    a.raw(&[0x48, 0x69, 0xc9]); // imul rcx, rcx, stride
                    a.raw(&layout.slot_stride.to_le_bytes());
                    a.raw(&[0x4c, 0x01, 0xd1]); // add rcx, r10: rcx = &Slot
                }
                a.cmp_byte_at_rcx_imm(layout.off_slot_occ, layout.slot_occupied_disc);
                a.jne($miss);
                a.cmp_word_at_rcx_r8w(layout.off_slot_gen);
                a.jne($miss);
            }};
        }
        // What a receiver of family `$fam` must be beyond a live slot (the
        // checks a receiver-cache fill proves): for `Obj` an ordinary shaped
        // object without the host-exotic flag, for `Arr` an array without
        // side-table state. Else to `$miss`.
        macro_rules! family_checks {
            ($fam:expr, $miss:expr) => {{
                match $fam {
                    recv_cache::Family::Obj => {
                        a.cmp_byte_at_rcx_imm(layout.off_cell_tag, layout.cell_object_disc);
                        a.jne($miss);
                        // (One compare when the object-data tag is the very
                        // byte and value the cell tag was just compared to.)
                        if (layout.off_od_tag, layout.obj_shaped_disc)
                            != (layout.off_cell_tag, layout.cell_object_disc)
                        {
                            a.cmp_byte_at_rcx_imm(layout.off_od_tag, layout.obj_shaped_disc);
                            a.jne($miss);
                        }
                        a.p_test_byte_imm(
                            call_seq::r::RCX,
                            layout.props.off_jit_flags,
                            props::FLAG_EXOTIC,
                        );
                        a.jne($miss);
                    }
                    recv_cache::Family::Arr => {
                        a.cmp_byte_at_rcx_imm(layout.off_cell_tag, layout.cell_array_disc);
                        a.jne($miss);
                        a.p_array_plain(&layout, helpers.call_layout.realm, false, $miss);
                    }
                    recv_cache::Family::Cell => {
                        a.cmp_byte_at_rcx_imm(layout.off_cell_tag, layout.props.box_disc);
                        a.jne($miss);
                    }
                }
            }};
        }
        // `rcx = &Slot` of the receiver in `$recv` — a live, occupied slot of
        // its generation, passing `family_checks!($fam)` — for op `$at`, else
        // to `$miss`: through the receiver cache when its plan says so (a hit
        // takes the way's address; a miss walks out of line and refills it),
        // else the full walk (then recorded in the way when a later op may
        // reuse it). Clobbers `rax`, `rdx`, `r8`, `r10`, `r11`; leaves no
        // arena in `r10`/`rdx`.
        macro_rules! recv_slot {
            ($at:expr, $recv:expr, $miss:expr, $fam:expr) => {{ recv_slot!($at, $recv, $miss, $fam, None) }};
            ($at:expr, $recv:expr, $miss:expr, $fam:expr, $walked:expr) => {{ recv_slot!($at, $recv, $miss, $fam, $walked, true) }};
            // `$walked`: where a way miss's full walk resumes instead of
            // the hit path's continuation. `$cell`: whether a known site's
            // fast path reads the heap slot (`rcx`) at all (a load through
            // the frame's `extra` pointer alone does not).
            ($at:expr, $recv:expr, $miss:expr, $fam:expr, $walked:expr, $cell:expr) => {{
                // A way holds one family's fills only (or a hit would skip
                // checks its fill never made).
                if rc.sites[$at].is_some_and(|p| p.family != $fam) {
                    return None;
                }
                match rc.sites[$at] {
                    // Still in the way since an earlier site of this
                    // iteration (`recv_cache::plan`; the loop's fast version
                    // only).
                    Some(recv_cache::SitePlan {
                        way, known: true, ..
                    }) if num.in_region($at) => {
                        if $cell {
                            a.g_load(call_seq::r::RCX, call_seq::r::RBP, rc_cell(way));
                        }
                    }
                    Some(recv_cache::SitePlan {
                        way, check: true, ..
                    }) => {
                        let walk = a.new_label();
                        let back = a.new_label();
                        a.load_rax(disp($recv));
                        a.g_cmp_mem(call_seq::r::RAX, call_seq::r::RBP, rc_key(way));
                        a.jne(walk);
                        a.g_load(call_seq::r::RCX, call_seq::r::RBP, rc_cell(way));
                        a.bind(back);
                        let walked: Option<Label> = $walked;
                        cold_walks.push((walk, walked.unwrap_or(back), $recv, $miss, way, $fam));
                    }
                    plan => {
                        a.p_load_arena(helpers.call_layout.realm, &layout.props);
                        walk_to_slot!($recv, $miss);
                        family_checks!($fam, $miss);
                        if let Some(recv_cache::SitePlan {
                            way, fill: true, ..
                        }) = plan
                        {
                            // (`rax`: the receiver's bits, from the walk.)
                            a.g_store(call_seq::r::RBP, rc_key(way), call_seq::r::RAX);
                            a.g_store(call_seq::r::RBP, rc_cell(way), call_seq::r::RCX);
                        }
                    }
                }
            }};
        }
        // `rax` = the truthiness (`0`/`1`) of register `$r` for op `$at`
        // (`emit_truthy_or`, slow path `$bail`). A receiver-cache site
        // (`recv_cache::plan`'s truthiness sites) first tries its way, or
        // walks and fills it: a later property op on the same object then
        // skips its walk. (A way whose fills prove a layout, see
        // `recv_cache::shape_ways`, is filled only with an object of that
        // layout.) A plain object without the `[[IsHTMLDDA]]` brand is
        // truthy; anything else takes the general test.
        macro_rules! truthy_rax {
            ($at:expr, $r:expr, $bail:expr) => {{
                let general = a.new_label();
                let have = a.new_label();
                if let Some(p) = rc.sites[$at] {
                    if p.family != recv_cache::Family::Obj {
                        return None;
                    }
                    let walk = a.new_label();
                    let proven = a.new_label();
                    if p.check {
                        a.load_rax(disp($r));
                        a.g_cmp_mem(call_seq::r::RAX, call_seq::r::RBP, rc_key(p.way));
                        a.jne(walk);
                        a.g_load(call_seq::r::RCX, call_seq::r::RBP, rc_cell(p.way));
                        a.jmp(proven);
                    }
                    a.bind(walk);
                    a.p_load_arena(helpers.call_layout.realm, &layout.props);
                    walk_to_slot!($r, general);
                    family_checks!(recv_cache::Family::Obj, general);
                    // (`rax`: the receiver's bits, from the walk.)
                    let skip_fill = a.new_label();
                    if p.fill {
                        if let Some(shape) = way_shape[p.way] {
                            a.g_load(call_seq::r::R11, call_seq::r::RCX, layout.off_shape);
                            a.g_movabs(call_seq::r::RDX, shape);
                            a.g_cmp(call_seq::r::R11, call_seq::r::RDX);
                            a.jne(skip_fill);
                        }
                        a.g_store(call_seq::r::RBP, rc_key(p.way), call_seq::r::RAX);
                        a.g_store(call_seq::r::RBP, rc_cell(p.way), call_seq::r::RCX);
                    }
                    a.bind(skip_fill);
                    a.bind(proven);
                    a.g_load(call_seq::r::R11, call_seq::r::RCX, layout.off_shape);
                    a.p_test_byte_imm(
                        call_seq::r::R11,
                        layout.props.shape_brands,
                        crate::shape::brand_bit(crate::shape::BRAND_KEYS[1]),
                    );
                    a.jne(general);
                    a.movabs_rax(1);
                    a.jmp(have);
                }
                a.bind(general);
                a.emit_truthy_or(
                    disp($r),
                    truthy_helper,
                    QNAN,
                    (tag_true, tag_false),
                    helpers.call_layout.realm,
                    $bail,
                );
                a.bind(have);
            }};
        }
        // An exact non-negative integer index in `reg` → `rax`, else `$miss`
        // (`$num`: `reg` is proven a Number at op `$at`).
        macro_rules! int_index {
            ($reg:expr, $miss:expr) => {{ int_index!($reg, $miss, false) }};
            ($reg:expr, $miss:expr, $num:expr) => {{
                if !$num {
                    a.load_rax(disp($reg));
                    a.movabs_r11(QNAN as i64);
                    a.and_rax_r11();
                    a.cmp_rax_r11();
                    a.je($miss); // boxed (non-number) index
                }
                a.movsd_xmm0_mem(disp($reg)); // xmm0 = index f64
                a.cvttsd2si_rax_xmm0(); // rax = trunc(index)
                a.cvtsi2sd_xmm1_rax(); // xmm1 = (f64)rax
                a.ucomisd_xmm0_xmm1();
                a.jne($miss); // fractional
                a.jp($miss); // NaN
            }};
        }
        // `GetProp` miss paths, emitted after the body: (miss, done, dst, obj,
        // method site, call site).
        // A promoted loop's `GetProp`s whose result goes straight into an SSE
        // register: (spill read, its return, the slow paths' resume, past
        // the op, the op).
        let mut cold_reloads: Vec<(Label, Label, Label, Label, usize)> = Vec::new();
        // The cache paths of `GetProp`s with a compiled-in layout: (entry,
        // return, miss, the promoted result's spill read, cache).
        let mut cold_caches: Vec<(Label, Label, Label, Option<Label>, *mut core::ffi::c_void)> =
            Vec::new();
        // The shape compares a layout-proving way's hit skips: (entry, a
        // write's store-guard miss, layout, mismatch, resume).
        let mut cold_verifies: Vec<(Label, Option<Label>, u64, Label, Label)> = Vec::new();
        // Per op: a `SetProp` in a speculative loop's fast version, every
        // continuation of whose fast path passed its flags-byte test (its
        // slow paths leave for the plain copy), so a later site of the
        // iteration on the same receiver (`recv_cache::SitePlan::flags_from`)
        // needs no such test.
        let mut set_proven = alloc::vec![false; ops.len()];
        // Per op: a site of a speculative loop's fast version after whose
        // fast path its way's `rc_extra` slot holds its receiver's `extra`
        // pointer (an inline add, which alone could change it without Rust
        // code, leaves that version), so a later known site on the receiver
        // (`recv_cache::SitePlan::pred`) reads its near spill one load
        // sooner.
        let mut extra_in_frame = alloc::vec![false; ops.len()];
        #[allow(clippy::type_complexity)]
        let mut cold_gets: Vec<(
            Label,
            Label,
            u16,
            u16,
            *mut MethodSite,
            *const CallSite,
            Vec<(u8, i32, bool)>,
        )> = Vec::new();
        // The slow paths of a speculative loop's fast version that leave for
        // its plain copy instead: (slow path entry, the leave).
        let mut cold_bails: Vec<(Label, Label)> = Vec::new();
        // A speculative region's plain copy is emitted right after it
        // (`numeric::NumGen::schedule`).
        num.alloc_copy_labels(&mut a);
        for (op_idx, copy) in num.schedule(ops.len()) {
            let op = &ops[op_idx];
            num.set_copy(copy);
            if copy {
                a.bind(num.copy_label(op_idx)?);
            } else {
                a.bind(labels[op_idx]);
            }
            // A global binding read or write through the global binding
            // cache: the cached address, else (out of line) a fresh one from
            // `jit_helper_global_ref`, else the interpreter.
            if let Some(k) = gc_plan.site.get(op_idx).copied().flatten() {
                let slot = gc_slot(k);
                let fetch = a.new_label();
                let back = a.new_label();
                let done = a.new_label();
                let reset = a.new_label();
                let recheck = a.new_label();
                // The epoch the slots were validated under must still be
                // current (else, out of line, they are all emptied).
                a.g_movabs(call_seq::r::R11, epoch_addr as u64);
                a.g_load(call_seq::r::RAX, call_seq::r::R11, 0);
                a.g_cmp_mem(call_seq::r::RAX, call_seq::r::RBP, gc_epoch);
                a.jne(reset);
                cold_epochs.push((reset, fetch, recheck));
                a.bind(recheck);
                a.g_load(call_seq::r::RAX, call_seq::r::RBP, slot);
                a.g_cmp_imm(call_seq::r::RAX, RECV_CACHE_EMPTY);
                a.je(fetch);
                a.bind(back);
                match *bytecode.get(op_idx)? {
                    crate::nbvm::Op::LoadGlobal { dst, .. } if ok(dst) => {
                        a.g_load(call_seq::r::RAX, call_seq::r::RAX, 0);
                        a.store_rax(disp(dst));
                    }
                    crate::nbvm::Op::StoreGlobal { src, .. } if ok(src) => {
                        a.g_load(call_seq::r::RCX, call_seq::r::RBX, disp(src));
                        a.g_store(call_seq::r::RAX, 0, call_seq::r::RCX);
                    }
                    _ => return None,
                }
                a.bind(done);
                cold_globals.push((fetch, back, done, op_idx, slot, k));
                continue;
            }
            if num.emit(&mut a, &labels, throw_exit, op_idx, ops)? {
                continue;
            }
            // An op of a promoted loop: its calls store and reload the
            // promoted registers, and a promoted register it writes is
            // reloaded after it (`numeric::NumGen::call_spill`).
            let in_region = num.in_region(op_idx);
            if in_region {
                a.set_call_spill(num.call_spill(op_idx));
            }
            // Where a fast path that left its result in the SSE register it
            // is promoted to resumes (past the slow paths' reload).
            let mut past_reload: Option<Label> = None;
            // Likewise, with that reload out of line (it binds the slow
            // paths' resume itself, `cold_reloads`).
            let mut ool_past: Option<Label> = None;
            match *op {
                GenericOp::Const { dst, imm } => {
                    if !ok(dst) {
                        return None;
                    }
                    a.movabs_rax(imm as i64);
                    a.store_rax(disp(dst));
                }
                GenericOp::Move { dst, src } => {
                    if !ok(dst) || !ok(src) {
                        return None;
                    }
                    a.load_rax(disp(src));
                    a.store_rax(disp(dst));
                }
                GenericOp::Nop => {}
                GenericOp::PushHandler { target, reg } => {
                    if !ok(reg) || target > ops.len() {
                        return None;
                    }
                    a.mov_rdi_r15();
                    a.movabs_argreg(1, i64::try_from(orig_pc(target)).ok()?);
                    a.movabs_argreg(2, i64::from(reg));
                    a.movabs_rax(push_handler_helper as i64);
                    a.call_rax();
                    throw_check!();
                }
                GenericOp::PopHandler => {
                    a.mov_rdi_r15();
                    a.movabs_rax(pop_handler_helper as i64);
                    a.call_rax();
                }
                GenericOp::InlineGuard {
                    dst,
                    call,
                    key,
                    caps,
                    n_caps,
                } => {
                    let no_args: Vec<u16> = Vec::new();
                    let mut getter = false;
                    let (callee, argv) = match ops.get(call as usize)? {
                        GenericOp::GetProp { site, obj, .. } => {
                            let si = sites.get(*site as usize)?;
                            if si.method_ptr.is_null() {
                                return None;
                            }
                            getter = true;
                            (
                                call_seq::CalleeSrc::Method {
                                    recv: *obj,
                                    site: si.method_ptr,
                                    scratch: method_scratch,
                                    helper: method_helper,
                                },
                                &no_args,
                            )
                        }
                        GenericOp::CallMethod {
                            site, recv, args, ..
                        } => {
                            let si = sites.get(*site as usize)?;
                            if si.method_ptr.is_null() {
                                return None;
                            }
                            (
                                call_seq::CalleeSrc::Method {
                                    recv: *recv,
                                    site: si.method_ptr,
                                    scratch: method_scratch,
                                    helper: method_helper,
                                },
                                call_args.get(*args as usize)?,
                            )
                        }
                        GenericOp::CallValue { callee, site, .. }
                        | GenericOp::CallWithMethod { callee, site, .. } => (
                            call_seq::CalleeSrc::Reg(*callee),
                            call_args.get(*site as usize)?,
                        ),
                        _ => return None,
                    };
                    if !ok(dst)
                        || argv.iter().any(|&r| !ok(r))
                        || (0..n_caps).any(|i| !ok(caps.saturating_add(i)))
                    {
                        return None;
                    }
                    // The receiver cache, for a method's receiver
                    // (`recv_cache::plan`'s guard sites).
                    let way = match rc.sites[op_idx] {
                        Some(p) if p.family == recv_cache::Family::Obj => Some(call_seq::RecvWay {
                            key: rc_key(p.way),
                            cell: rc_cell(p.way),
                            check: p.check,
                            fill: p.fill,
                            known: p.known && num.in_region(op_idx),
                        }),
                        Some(_) => return None,
                        None => None,
                    };
                    if way.is_some() && matches!(callee, call_seq::CalleeSrc::Reg(_)) {
                        return None;
                    }
                    let stub = a.new_label();
                    let memo = call_sites
                        .get(match ops.get(call as usize)? {
                            GenericOp::CallMethod { cache, .. }
                            | GenericOp::CallValue { cache, .. }
                            | GenericOp::CallWithMethod { cache, .. } => *cache as usize,
                            GenericOp::GetProp { call, .. } => *call as usize,
                            _ => return None,
                        })
                        .copied()?;
                    call_seq::emit_inline_guard(
                        &mut a,
                        &call_seq::InlineEmit {
                            dst,
                            key,
                            callee,
                            args: argv,
                            caps,
                            n_caps,
                            layout: &helpers.call_layout,
                            slots: inl_slots,
                            stub,
                            memo,
                            getter,
                            way,
                        },
                    )?;
                    inline_stubs.push((stub, argv.clone()));
                }
                GenericOp::InlineLeave => {
                    call_seq::emit_inline_leave(&mut a, &helpers.call_layout, &inl_slots);
                }
                GenericOp::CallMethod {
                    pc,
                    site,
                    dst,
                    recv,
                    args,
                    cache,
                } => {
                    if pc as usize != op_idx {
                        return None;
                    }
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    let si = sites.get(site as usize)?;
                    let done = a.new_label();
                    // `Math.f(x)` with the built-in `f`, inline: the site's
                    // chain cache finds the method, which must be that very
                    // built-in, and a Number argument is computed in place.
                    if let Some((native_id, f)) = inline_math(bytecode.get(op_idx)?)
                        && !si.method_ptr.is_null()
                    {
                        let crate::nbvm::Op::CallMethod { args: margs, .. } =
                            bytecode.get(op_idx)?
                        else {
                            return None;
                        };
                        let x = *margs.first()?;
                        if !ok(dst) || !ok(recv) || margs.iter().any(|&m| !ok(m)) {
                            return None;
                        }
                        let fill = a.new_label();
                        let general = a.new_label();
                        let nan = a.new_label();
                        let mi = math_sites.iter().position(|&p| p == op_idx)?;
                        let walk = a.new_label();
                        let have = a.new_label();
                        let proven = a.new_label();
                        a.g_cmp_byte_imm(call_seq::r::R15, helpers.call_layout.hosted, 0);
                        a.g_jcc(call_seq::cc::E, general); // not hosted: no chain cache
                        // The receiver's heap slot through the site's own
                        // one-entry cache (else walked, out of line).
                        a.load_rax(disp(recv));
                        a.g_cmp_mem(call_seq::r::RAX, call_seq::r::RBP, m_slot(mi, 0));
                        a.jne(walk);
                        // The same receiver, and a method proved the
                        // built-in since the last call: when nothing in this
                        // body can store the method's key without a call
                        // (no inline store or definition of it), the lookup
                        // would find that very value — only Rust code
                        // (behind a call, which empties both keys) can
                        // otherwise change, shadow or remove it. (The key is
                        // stored only once the lookup on this receiver found
                        // it; a walk to another receiver looks it up again.)
                        let crate::nbvm::Op::CallMethod { key, .. } = bytecode.get(op_idx)? else {
                            return None;
                        };
                        let key = key.as_str();
                        let stores_key = bytecode.iter().any(|o| {
                            matches!(o,
                                crate::nbvm::Op::SetProp { key: k, .. }
                                | crate::nbvm::Op::SetPropStrict { key: k, .. }
                                | crate::nbvm::Op::InitProp { key: k, .. } if k.as_str() == key)
                                || matches!(o, crate::nbvm::Op::NewObjectLit { lit, .. }
                                    if lit.keys.iter().any(|k| k.as_str() == key))
                        });
                        if !stores_key {
                            a.g_load(call_seq::r::RAX, call_seq::r::RBP, m_slot(mi, 2));
                            a.g_cmp_imm(call_seq::r::RAX, RECV_CACHE_EMPTY);
                            a.jne(proven);
                        }
                        a.g_load(call_seq::r::RCX, call_seq::r::RBP, m_slot(mi, 1));
                        a.bind(have);
                        a.p_method_lookup_walked(
                            &layout,
                            helpers.call_layout.realm,
                            si.method_ptr,
                            fill,
                            None,
                        );
                        // The method: the very value proved the built-in since
                        // the last call, else proved now.
                        a.g_cmp_mem(call_seq::r::RAX, call_seq::r::RBP, m_slot(mi, 2));
                        a.je(proven);
                        a.g_mov(call_seq::r::R9, call_seq::r::RAX);
                        a.p_is_native(&layout, helpers.call_layout.realm, native_id, general);
                        a.g_store(call_seq::r::RBP, m_slot(mi, 2), call_seq::r::R9);
                        a.bind(proven);
                        for &m in margs.iter() {
                            if !num.is_number_at(op_idx, m) {
                                a.movabs_r11(QNAN as i64);
                                a.load_rax(disp(m));
                                a.and_rax_r11();
                                a.cmp_rax_r11();
                                a.je(general); // not a Number: its conversion is the call's
                            }
                        }
                        if let MathInline::MaxMin(max) = f {
                            // Every argument is a Number (so no conversion
                            // runs): NaN if any is NaN, else the largest
                            // (smallest), `+0` above `-0` — for equal
                            // operands the AND (OR) of their bits.
                            a.movsd_xmm0_mem(disp(x));
                            for &m in &margs[1..] {
                                let unequal = a.new_label();
                                let next = a.new_label();
                                a.movsd_xmm1_mem(disp(m));
                                a.ucomisd_xmm0_xmm1();
                                a.jp(nan);
                                a.jne(unequal);
                                a.raw(if max {
                                    &[0x66, 0x0f, 0x54, 0xc1] // andpd xmm0, xmm1
                                } else {
                                    &[0x66, 0x0f, 0x56, 0xc1] // orpd xmm0, xmm1
                                });
                                a.jmp(next);
                                a.bind(unequal);
                                a.raw(if max {
                                    &[0xf2, 0x0f, 0x5f, 0xc1] // maxsd xmm0, xmm1
                                } else {
                                    &[0xf2, 0x0f, 0x5d, 0xc1] // minsd xmm0, xmm1
                                });
                                a.bind(next);
                            }
                            if let Some(xr) = num.result_xmm(op_idx, dst) {
                                let past = a.new_label();
                                numeric::NumGen::set_xmm_from_xmm0(&mut a, xr);
                                a.jmp(past);
                                past_reload = Some(past);
                            } else {
                                a.movsd_mem_xmm0(disp(dst));
                                a.jmp(done);
                            }
                            a.bind(nan);
                            a.movabs_rax(canonical_nan as i64);
                            a.store_rax(disp(dst));
                            a.jmp(done);
                        } else if let MathInline::Abs = f {
                            // The sign bit cleared (a NaN argument is already
                            // the canonical one, whose sign is clear).
                            a.load_rax(disp(x));
                            a.g_movabs(call_seq::r::R11, i64::MAX as u64);
                            a.g_and(call_seq::r::RAX, call_seq::r::R11);
                            a.store_rax(disp(dst));
                            a.jmp(done);
                        } else {
                            a.movsd_xmm0_mem(disp(x));
                            match f {
                                MathInline::Round(mode) => a.roundsd_xmm0(mode),
                                MathInline::MaxMin(_) => return None,
                                _ => a.sqrtsd_xmm0(),
                            }
                            a.ucomisd_xmm0_xmm0();
                            a.jp(nan);
                            if let Some(x) = num.result_xmm(op_idx, dst) {
                                let past = a.new_label();
                                numeric::NumGen::set_xmm_from_xmm0(&mut a, x);
                                a.jmp(past);
                                past_reload = Some(past);
                            } else {
                                a.movsd_mem_xmm0(disp(dst));
                                a.jmp(done);
                            }
                            a.bind(nan);
                            a.movabs_rax(canonical_nan as i64);
                            a.store_rax(disp(dst));
                            a.jmp(done);
                        }
                        // The receiver walk, recorded in the site's cache.
                        a.bind(walk);
                        a.p_load_arena(helpers.call_layout.realm, &layout.props);
                        walk_to_slot!(recv, general);
                        family_checks!(recv_cache::Family::Obj, general);
                        a.g_store(call_seq::r::RBP, m_slot(mi, 0), call_seq::r::RAX);
                        a.g_store(call_seq::r::RBP, m_slot(mi, 1), call_seq::r::RCX);
                        a.jmp(have);
                        // A chain-cache miss: refill it from the receiver
                        // (in a speculative loop's fast version, the copy
                        // does, as it makes every call — from the
                        // receiver's `RequireObjectCoercible` when this
                        // call checks for it, `NumGen::roc_folded`).
                        let bail = if op_idx > 0 && num.roc_folded(op_idx - 1) {
                            num.spec_bail(&mut a, op_idx - 1)
                        } else {
                            num.spec_bail(&mut a, op_idx)
                        };
                        a.bind(fill);
                        if let Some(b) = bail {
                            a.jmp(b);
                        }
                        a.mov_rdi_r15();
                        a.load_argreg(1, disp(recv));
                        a.movabs_argreg(2, si.method_ptr as i64);
                        a.movabs_rax(helpers.site_fill as usize as i64);
                        a.call_rax();
                        a.bind(general);
                        if let Some(b) = bail {
                            a.jmp(b);
                        }
                    }
                    // `Math.f(x)` with the built-in `f` is computed directly.
                    num.math_call(&mut a, op_idx, done);
                    // The native call sequence, when the site has one.
                    let native = call_args
                        .get(args as usize)
                        .zip(call_sites.get(cache as usize));
                    let site_ptr = native.map_or(core::ptr::null(), |(_, c)| *c);
                    if let Some((argv, &site_ptr)) = native {
                        if !ok(dst) || !ok(recv) || argv.iter().any(|&r| !ok(r)) {
                            return None;
                        }
                        let slow = a.new_label();
                        call_seq::emit_call(
                            &mut a,
                            &call_seq::CallEmit {
                                dst,
                                callee: call_seq::CalleeSrc::Method {
                                    recv,
                                    site: si.method_ptr,
                                    scratch: method_scratch,
                                    helper: method_helper,
                                },
                                this: Some(recv),
                                args: argv,
                                site: site_ptr,
                                layout: &helpers.call_layout,
                                throw_exit,
                                slow,
                                done,
                            },
                        )?;
                        a.bind(slow);
                    }
                    a.mov_rdi_r15(); // rdi = ctx
                    a.mov_rsi_rbx(); // rsi = regs
                    a.movabs_argreg(2, n_regs as i64); // rdx = n_regs
                    a.movabs_argreg(3, op_ptr as i64); // rcx = &op
                    a.movabs_argreg(4, si.cache_ptr as i64); // r8 = the site cache
                    a.movabs_argreg(5, site_ptr as i64); // r9 = the call cache (or null)
                    a.movabs_rax(call_method_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
                GenericOp::InitProp {
                    obj,
                    val,
                    site,
                    plain,
                } => {
                    if !ok(obj) || !ok(val) {
                        return None;
                    }
                    let s = sites.get(site as usize)?;
                    let helper = a.new_label();
                    let done = a.new_label();
                    if plain && fresh_init.get(op_idx).copied().unwrap_or(false) {
                        // A fresh literal's own property: no receiver walk.
                        a.p_init_prop_fresh(
                            &layout,
                            helpers.call_layout.realm,
                            (disp(obj), disp(val)),
                            s.cache_ptr as u64,
                            helper,
                        );
                        a.jmp(done);
                    } else if plain {
                        // ===== inline transition-cache add (see
                        // `p_init_prop_inline`) =====
                        recv_slot!(op_idx, obj, helper, recv_cache::Family::Obj);
                        a.p_init_prop_inline(
                            &layout,
                            helpers.call_layout.realm,
                            s.cache_ptr as u64,
                            disp(val),
                            helper,
                        );
                        a.jmp(done);
                    }
                    a.bind(helper);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(obj));
                    a.movabs_argreg(2, s.key_ptr as i64);
                    a.movabs_argreg(3, s.key_len as i64);
                    a.movabs_argreg(4, s.cache_ptr as i64);
                    a.load_argreg(5, disp(val));
                    a.movabs_rax(init_prop_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                    // An add: maybe to an object a layout-proving way holds.
                    for &w in &shape_ways {
                        a.store_empty(rc_key(w));
                    }
                }
                GenericOp::CheckTdz { src, pc } => {
                    if !ok(src) {
                        return None;
                    }
                    let done = a.new_label();
                    a.load_rax(disp(src));
                    a.movabs_r11(NanBox::tdz().to_bits() as i64);
                    a.cmp_rax_r11();
                    a.jne(done);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::ToNumeric { dst, src, pc } => {
                    if !ok(dst) || !ok(src) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    a.movabs_r11(QNAN as i64);
                    a.load_rax(disp(src));
                    a.mov_rcx_rax();
                    a.and_rcx_r11();
                    a.cmp_rcx_r11();
                    a.je(slow); // not a number
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::Step { dst, src, dec, pc } => {
                    if !ok(dst) || !ok(src) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    let nan_fix = a.new_label();
                    a.movabs_r11(QNAN as i64);
                    a.load_rax(disp(src));
                    a.and_rax_r11();
                    a.cmp_rax_r11();
                    a.je(slow); // not a number
                    a.movsd_xmm0_mem(disp(src));
                    a.movabs_rax(1.0f64.to_bits() as i64);
                    a.cvt_rax_bits_to_xmm1();
                    if dec {
                        a.subsd_xmm0_xmm1();
                    } else {
                        a.addsd_xmm0_xmm1();
                    }
                    a.ucomisd_xmm0_xmm0();
                    a.jp(nan_fix);
                    a.movsd_mem_xmm0(disp(dst));
                    a.jmp(done);
                    a.bind(nan_fix);
                    a.movabs_rax(canonical_nan as i64);
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::ArrGet {
                    dst,
                    arr,
                    index,
                    pc,
                } => {
                    if !ok(dst) || !ok(arr) || !ok(index) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    recv_slot!(op_idx, arr, slow, recv_cache::Family::Arr);
                    int_index!(index, slow);
                    a.p_arr_len_ptr(&layout);
                    a.cmp_rax_r8();
                    a.jae(slow);
                    a.mov_rax_at_r11_rax8();
                    a.movabs_r11(NanBox::hole().to_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(slow);
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::ArrSet {
                    arr,
                    index,
                    src,
                    pc,
                } => {
                    if !ok(arr) || !ok(index) || !ok(src) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    a.load_rax(disp(src));
                    a.movabs_r11(HANDLE_TAG as i64);
                    a.and_rax_r11();
                    a.cmp_rax_r11();
                    a.je(slow); // a handle value needs the write barrier
                    recv_slot!(op_idx, arr, slow, recv_cache::Family::Arr);
                    a.p_array_plain(&layout, helpers.call_layout.realm, true, slow);
                    int_index!(index, slow);
                    a.p_arr_len_ptr(&layout);
                    a.cmp_rax_r8();
                    a.jae(slow);
                    // A hole is an absent index: storing to it is `[[Set]]` on a
                    // missing own property (prototype setters / proxy traps).
                    a.movabs_argreg(2, NanBox::hole().to_bits() as i64);
                    a.cmp_at_r11_rax8_rdx();
                    a.je(slow);
                    a.load_rdx(disp(src));
                    a.mov_at_r11_rax8_rdx();
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::CellGet { dst, cell, pc } => {
                    if !ok(dst) || !ok(cell) {
                        return None;
                    }
                    let other = a.new_label();
                    let slow = a.new_label();
                    let done = a.new_label();
                    // The array cell's checks, with `rcx` its live slot.
                    let walked = a.new_label();
                    match rc.sites[op_idx] {
                        // No way to compare first: one walk serves both
                        // kinds of cell (`recv_slot!`'s own, its fill too).
                        None
                        | Some(recv_cache::SitePlan {
                            check: false,
                            known: false,
                            ..
                        }) => {
                            if rc.sites[op_idx]
                                .is_some_and(|p| p.family != recv_cache::Family::Cell)
                            {
                                return None;
                            }
                            a.p_load_arena(helpers.call_layout.realm, &layout.props);
                            walk_to_slot!(cell, slow);
                            a.cmp_byte_at_rcx_imm(layout.off_cell_tag, layout.props.box_disc);
                            a.jne(walked);
                            if let Some(recv_cache::SitePlan {
                                way, fill: true, ..
                            }) = rc.sites[op_idx]
                            {
                                a.g_store(call_seq::r::RBP, rc_key(way), call_seq::r::RAX);
                                a.g_store(call_seq::r::RBP, rc_cell(way), call_seq::r::RCX);
                            }
                        }
                        _ => recv_slot!(op_idx, cell, other, recv_cache::Family::Cell),
                    }
                    a.g_load(call_seq::r::RAX, call_seq::r::RCX, layout.props.off_box_val);
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    // A one-element array cell (a parameter's): its present
                    // element 0, as `GetElem` reads it.
                    a.bind(other);
                    a.p_load_arena(helpers.call_layout.realm, &layout.props);
                    walk_to_slot!(cell, slow);
                    a.bind(walked);
                    family_checks!(recv_cache::Family::Arr, slow);
                    a.p_arr_len_ptr(&layout);
                    a.g_test(call_seq::r::R8);
                    a.g_jcc(call_seq::cc::E, slow);
                    a.g_load(call_seq::r::RAX, call_seq::r::R11, 0);
                    a.movabs_r11(NanBox::hole().to_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(slow);
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::CellSet { cell, src, pc } => {
                    if !ok(cell) || !ok(src) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    recv_slot!(op_idx, cell, slow, recv_cache::Family::Cell);
                    a.load_rax(disp(src));
                    a.p_barrier_check(&layout, helpers.call_layout.realm, slow);
                    a.load_rdx(disp(src));
                    a.g_store(call_seq::r::RCX, layout.props.off_box_val, call_seq::r::RDX);
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::Interp { pc } => {
                    if pc as usize != op_idx {
                        return None;
                    }
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    // A slow path through a site-filling helper with the
                    // exec-op ABI plus the site (`JitCallMethodHelper`).
                    macro_rules! site_slow {
                        ($helper:expr, $site:expr) => {{
                            a.mov_rdi_r15();
                            a.mov_rsi_rbx();
                            a.movabs_argreg(2, n_regs as i64);
                            a.movabs_argreg(3, op_ptr as i64);
                            a.movabs_argreg(4, $site as i64);
                            a.movabs_argreg(5, 0);
                            a.movabs_rax($helper as usize as i64);
                            a.call_rax();
                            throw_check!();
                        }};
                    }
                    let own_site = interp_sites
                        .get(pc as usize)
                        .copied()
                        .flatten()
                        .and_then(|(args, site)| Some((args, *call_sites.get(site as usize)?)));
                    match bytecode.get(pc as usize)? {
                        // `super(…)`: the native call sequence, `new.target`
                        // passed on and `this` bound from the result.
                        crate::nbvm::Op::SuperCallArgs {
                            dst,
                            ctor,
                            new_target,
                            args,
                            this,
                        } if args.len() <= 32
                            && ok(*dst)
                            && ok(*ctor)
                            && ok(*new_target)
                            && ok(*this)
                            && args.iter().all(|&r| ok(r))
                            && own_site.is_some() =>
                        {
                            let (args_site, site_ptr) = own_site?;
                            let argregs = call_args.get(args_site as usize)?;
                            if argregs != args {
                                return None;
                            }
                            let slow = a.new_label();
                            let done = a.new_label();
                            call_seq::emit_call(
                                &mut a,
                                &call_seq::CallEmit {
                                    dst: *dst,
                                    callee: call_seq::CalleeSrc::Construct {
                                        ctor: *ctor,
                                        new_target: *new_target,
                                        scratch: method_scratch,
                                        helper: construct_helper,
                                        result: construct_result_helper,
                                        alloc: construct_alloc_helper,
                                        bind_this: Some((*this, super_twice_helper)),
                                    },
                                    this: None,
                                    args: argregs,
                                    site: site_ptr,
                                    layout: &helpers.call_layout,
                                    throw_exit,
                                    slow,
                                    done,
                                },
                            )?;
                            a.bind(slow);
                            site_slow!(helpers.super_call, site_ptr);
                            a.bind(done);
                        }
                        // The super constructor / "no instance fields" of a
                        // constructor value seen before at this epoch (the
                        // site's `IdentMemo`), else the interpreter's (which
                        // refills it).
                        crate::nbvm::Op::SuperCtor { dst, f }
                            if ok(*dst) && ok(*f) && own_site.is_some() =>
                        {
                            let (_, site_ptr) = own_site?;
                            let slow = a.new_label();
                            let done = a.new_label();
                            a.p_ident_hit(site_ptr, disp(*f), slow);
                            a.g_load(call_seq::r::RAX, call_seq::r::R11, call_seq::IDENT_VALUE);
                            a.store_rax(disp(*dst));
                            a.jmp(done);
                            a.bind(slow);
                            site_slow!(helpers.super_ctor, site_ptr);
                            a.bind(done);
                        }
                        crate::nbvm::Op::InitFields { ctor, this }
                            if ok(*ctor) && ok(*this) && own_site.is_some() =>
                        {
                            let (_, site_ptr) = own_site?;
                            let slow = a.new_label();
                            let done = a.new_label();
                            a.p_ident_hit(site_ptr, disp(*ctor), slow);
                            a.jmp(done);
                            a.bind(slow);
                            site_slow!(helpers.init_fields, site_ptr);
                            a.bind(done);
                        }
                        // A step of an array iteration in by-index mode
                        // (`nbvm::iter_fast`: a Number cursor in `iter`, the
                        // array in `iter + 1`): a present element of a plain
                        // dense array below its length is the step's value
                        // and the cursor moves on, which is all
                        // `iter_fast::step` does for it. Anything else (the
                        // protocol mode, a latched `done`, the end, a hole,
                        // an array with side-table state) is the
                        // interpreter's op.
                        crate::nbvm::Op::IterNext {
                            dst,
                            done,
                            iter,
                            next,
                        } if ok(*dst)
                            && ok(*done)
                            && ok(*iter)
                            && usize::from(*next) == usize::from(*iter) + 1
                            && ok(*next) =>
                        {
                            let (dst, done, iter, next) = (*dst, *done, *iter, *next);
                            let slow = a.new_label();
                            let fin = a.new_label();
                            a.movabs_r11(QNAN as i64);
                            a.load_rax(disp(iter));
                            a.and_rax_r11();
                            a.cmp_rax_r11();
                            a.je(slow); // an iterator object: the protocol
                            a.load_rax(disp(done));
                            a.movabs_r11(tag_true as i64);
                            a.cmp_rax_r11();
                            a.je(slow);
                            recv_slot!(op_idx, next, slow, recv_cache::Family::Arr);
                            int_index!(iter, slow, true);
                            a.p_arr_len_ptr(&layout);
                            a.cmp_rax_r8();
                            let end = a.new_label();
                            a.jae(end);
                            a.mov_r8_rax();
                            a.mov_rax_at_r11_rax8();
                            a.movabs_r11(NanBox::hole().to_bits() as i64);
                            a.cmp_rax_r11();
                            a.je(slow);
                            a.store_rax(disp(dst));
                            // cursor = index + 1 (an exact Number: below the
                            // array's length).
                            numeric::NumGen::inc_r8_to_xmm0(&mut a);
                            a.movsd_mem_xmm0(disp(iter));
                            a.jmp(fin);
                            // The end (`index >= length`, the length the
                            // plain array's element count): latch `done`, and
                            // the step's value is `undefined`, as
                            // `iter_fast::step` does.
                            a.bind(end);
                            a.test_rax_rax();
                            a.g_jcc(0x8, slow); // js: a negative cursor is the interpreter's
                            a.movabs_rax(tag_true as i64);
                            a.store_rax(disp(done));
                            a.movabs_rax(NanBox::undefined().to_bits() as i64);
                            a.store_rax(disp(dst));
                            a.jmp(fin);
                            a.bind(slow);
                            // (A speculative loop's fast version leaves for
                            // its plain copy instead.)
                            match num.spec_bail(&mut a, op_idx) {
                                Some(bail) => a.jmp(bail),
                                None => exec_op!(pc),
                            }
                            a.bind(fin);
                        }
                        // The open of an iteration over a plain array (no
                        // named property, no prototype ever set: its
                        // prototype is the first `%Array.prototype%`) with
                        // the built-in iterator, as `iter_fast::can_open`'s
                        // memo last found it valid (`p_array_iter_open`): by
                        // index (`next` the array, `iter` the cursor `0`).
                        // Anything else is the interpreter's op.
                        crate::nbvm::Op::IterOpen { iter, next, src }
                            if usize::from(*next) == usize::from(*iter) + 1
                                && ok(*iter)
                                && ok(*next)
                                && ok(*src) =>
                        {
                            let (iter, next, src) = (*iter, *next, *src);
                            let slow = a.new_label();
                            let fin = a.new_label();
                            a.p_load_arena(helpers.call_layout.realm, &layout.props);
                            walk_to_slot!(src, slow);
                            family_checks!(recv_cache::Family::Arr, slow);
                            a.cmp_byte_at_rcx_imm(layout.off_arr_proto_set, 0);
                            a.jne(slow);
                            a.p_array_iter_open(&layout, helpers.call_layout.realm, slow);
                            a.load_rax(disp(src));
                            a.store_rax(disp(next));
                            a.movabs_rax(NanBox::number(0.0).to_bits() as i64);
                            a.store_rax(disp(iter));
                            a.jmp(fin);
                            a.bind(slow);
                            exec_op!(pc);
                            a.bind(fin);
                        }
                        // A literal's allocation: its own helper, not the
                        // one-op interpreter.
                        crate::nbvm::Op::NewObject { dst } if ok(*dst) => {
                            // Inline from the heap's free list, else the helper.
                            let slow = a.new_label();
                            let done = a.new_label();
                            let realm_off = helpers.call_layout.realm;
                            a.g_load(call_seq::r::R9, call_seq::r::R15, realm_off);
                            a.g_load(
                                call_seq::r::RSI,
                                call_seq::r::R9,
                                layout.props.default_proto,
                            );
                            a.p_alloc_object(&layout, realm_off, slow);
                            a.jmp(done);
                            a.bind(slow);
                            a.mov_rdi_r15();
                            a.movabs_rax(helpers.new_object as usize as i64);
                            a.call_rax();
                            a.bind(done);
                            a.store_rax(disp(*dst));
                        }
                        // A closure that needs nothing but its cell (see
                        // `compile_generic_closures`), in the main realm of
                        // a single-realm run whose closures' prototype is
                        // the recorded default: inline from the heap's free
                        // list, else the interpreter's op.
                        crate::nbvm::Op::MakeClosure {
                            dst,
                            func,
                            captures,
                        } if inline_closures.get(pc as usize).copied().unwrap_or(false)
                            && ok(*dst)
                            && captures.iter().all(|r| ok(*r)) =>
                        {
                            let slow = a.new_label();
                            let done = a.new_label();
                            let cl = &helpers.call_layout;
                            a.g_cmp_byte_imm(call_seq::r::R15, cl.main_realm, 0);
                            a.g_jcc(call_seq::cc::E, slow);
                            a.g_load(call_seq::r::R9, call_seq::r::R15, cl.realm);
                            a.g_cmp_byte_imm(call_seq::r::R9, cl.multi_realm, 0);
                            a.g_jcc(call_seq::cc::NE, slow);
                            a.g_cmp_byte_imm(call_seq::r::R9, cl.vmfn_plain, 0);
                            a.g_jcc(call_seq::cc::E, slow);
                            let caps: Vec<i32> = captures.iter().map(|r| disp(*r)).collect();
                            a.p_alloc_closure(&layout, cl.realm, *func, &caps, slow);
                            a.store_rax(disp(*dst));
                            a.jmp(done);
                            a.bind(slow);
                            exec_op!(pc);
                            a.bind(done);
                        }
                        // A captured variable's binding cell: inline from
                        // the heap's free list, else the interpreter's op.
                        crate::nbvm::Op::NewBox { dst } if ok(*dst) => {
                            let slow = a.new_label();
                            let done = a.new_label();
                            a.p_alloc_box(&layout, helpers.call_layout.realm, slow);
                            a.store_rax(disp(*dst));
                            a.jmp(done);
                            a.bind(slow);
                            exec_op!(pc);
                            a.bind(done);
                        }
                        // An object literal with a constant key set: inline
                        // in the site's memoized shape, else the helper
                        // (which builds it member by member and arms the
                        // memo).
                        crate::nbvm::Op::NewObjectLit { dst, lit }
                            if ok(*dst)
                                && lit.srcs.len() <= crate::object::INLINE_SLOTS
                                && lit.srcs.iter().all(|r| ok(*r)) =>
                        {
                            let slow = a.new_label();
                            let done = a.new_label();
                            let vals: Vec<i32> = lit.srcs.iter().map(|r| disp(*r)).collect();
                            a.p_alloc_object_lit(
                                &layout,
                                helpers.call_layout.realm,
                                core::ptr::from_ref(&lit.cache) as u64,
                                &vals,
                                slow,
                            );
                            a.store_rax(disp(*dst));
                            a.jmp(done);
                            a.bind(slow);
                            exec_op!(pc);
                            a.bind(done);
                        }
                        array_op @ (crate::nbvm::Op::NewArrayFrom { .. }
                        | crate::nbvm::Op::NewArray { .. }) => {
                            let op_ptr: *const crate::nbvm::Op = array_op;
                            // A small literal: inline from the heap's free
                            // list (its elements in the cell), else the helper.
                            let small = match array_op {
                                crate::nbvm::Op::NewArrayFrom { dst, elems }
                                    if ok(*dst)
                                        && elems.len() <= crate::cell::ARRAY_INLINE_CAP
                                        && elems.iter().all(|r| ok(*r)) =>
                                {
                                    Some((*dst, elems.iter().map(|r| disp(*r)).collect::<Vec<_>>()))
                                }
                                _ => None,
                            };
                            let done = a.new_label();
                            if let Some((dst, elems)) = &small {
                                let slow = a.new_label();
                                a.p_alloc_array(&layout, helpers.call_layout.realm, elems, slow);
                                a.store_rax(disp(*dst));
                                a.jmp(done);
                                a.bind(slow);
                            }
                            a.mov_rdi_r15();
                            a.mov_rsi_rbx();
                            a.movabs_argreg(2, n_regs as i64);
                            a.movabs_argreg(3, op_ptr as i64);
                            a.movabs_rax(helpers.new_array as usize as i64);
                            a.call_rax();
                            throw_check!();
                            a.bind(done);
                        }
                        _ => exec_op!(pc),
                    }
                }
                GenericOp::Add { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    both_numbers!(ra, rb, slow);
                    fbin_store!(FBinOp::Add, ra, rb, dst, done);
                    // Slow path: jit_helper_add(ctx, a, b).
                    a.bind(slow);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(ra));
                    a.load_argreg(2, disp(rb));
                    a.movabs_rax(add_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::NumAdd {
                    dst,
                    a: ra,
                    b: rb,
                    pc,
                } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    both_numbers!(ra, rb, slow);
                    fbin_store!(FBinOp::Add, ra, rb, dst, done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::GetProp {
                    dst,
                    obj,
                    site,
                    call,
                } => {
                    if !ok(dst) || !ok(obj) {
                        return None;
                    }
                    let call_site = *call_sites.get(call as usize)?;
                    let s = sites.get(site as usize)?;
                    let helper = a.new_label();
                    let done = a.new_label();
                    // ===== inline monomorphic property-get fast path =====
                    // A pure sequence of reads at runtime-verified offsets: walk
                    // arena → slot → cell → object → shape, compare the object's
                    // shape-Rc word to the site's cached shape, and load the cached
                    // slot. Any guard miss falls to the (correct) helper, which
                    // arms/updates the IC and handles accessors, prototype walks,
                    // misses and throws. No allocation and no call: the arena
                    // base is read from the realm, so it is GC- and realloc-safe.
                    // (A host-exotic object's reads are the host's.)
                    // A way hit that proves the compiled-in layout skips the
                    // shape compare; the way's full walk makes it, out of
                    // line (`cold_verifies`).
                    let known = s.baked.filter(|b| shape_known(op_idx, b.shape));
                    let verify = known.map(|_| a.new_label());
                    // In a speculative loop's fast version, a promoted
                    // result of a compiled-in layout is loaded straight into
                    // its SSE register: every other path leaves for the
                    // plain copy. (`p.have`: the frame holds the receiver's
                    // `extra` pointer, as its predecessor of the iteration
                    // left it.)
                    let direct = s.baked.zip(num.result_xmm(op_idx, dst)).and_then(|(b, x)| {
                        let p = rc.sites[op_idx].filter(|_| num.in_spec(op_idx))?;
                        let have = p.known && p.pred.is_some_and(|q| extra_in_frame[q]);
                        Some((b, x, p, have))
                    });
                    let direct =
                        direct.and_then(|d| num.spec_bail(&mut a, op_idx).map(|bl| (d, bl)));
                    let cell = !matches!(direct, Some(((b, _, p, have), _))
                        if p.known && known.is_some()
                            && !X64Assembler::p_fixed_slot_needs_rcx(&layout, b.slot, have));
                    recv_slot!(op_idx, obj, helper, recv_cache::Family::Obj, verify, cell);
                    if known.is_none() {
                        a.mov_rax_at_rcx(layout.off_shape);
                    }
                    if let Some(((b, x, p, have), bail)) = direct {
                        let cached = a.new_label();
                        if let Some(verify) = verify {
                            let resume = a.new_label();
                            a.bind(resume);
                            cold_verifies.push((verify, None, b.shape, cached, resume));
                        } else {
                            a.g_movabs(call_seq::r::R11, b.shape);
                            a.cmp_rax_r11();
                            a.jne(cached);
                        }
                        extra_in_frame[op_idx] = a.p_obj_fixed_slot_ex_xmm(
                            &layout,
                            b.slot,
                            Some(rc_extra(p.way)),
                            have,
                            x,
                        );
                        cold_bails.push((cached, bail));
                        // (`done`: no path of the fast version reaches it.)
                        let past = a.new_label();
                        let back = a.new_label();
                        let spill = a.new_label();
                        cold_reloads.push((spill, back, done, past, op_idx));
                        let mark = a.cold_begin();
                        a.bind(back);
                        numeric::NumGen::set_xmm_from_rax(&mut a, x);
                        a.jmp(past);
                        a.cold_end(mark);
                        ool_past = Some(past);
                    } else if let Some(b) = s.baked {
                        // The layout the site settled on, compiled in: one
                        // compare with the shape, one load at a fixed
                        // offset; any other shape takes the cache path
                        // (out of line, `cold_caches`).
                        let cached = a.new_label();
                        let back = a.new_label();
                        if let Some(verify) = verify {
                            let resume = a.new_label();
                            a.bind(resume);
                            cold_verifies.push((verify, None, b.shape, cached, resume));
                        } else {
                            a.g_movabs(call_seq::r::R11, b.shape);
                            a.cmp_rax_r11();
                            a.jne(cached);
                        }
                        match rc.sites[op_idx].filter(|_| num.in_spec(op_idx)) {
                            Some(p) => {
                                let have = p.known && p.pred.is_some_and(|q| extra_in_frame[q]);
                                extra_in_frame[op_idx] = a.p_obj_fixed_slot_ex(
                                    &layout,
                                    b.slot,
                                    false,
                                    rc_extra(p.way),
                                    have,
                                );
                            }
                            None => a.p_obj_fixed_slot_load(&layout, b.slot),
                        }
                        let x = num.result_xmm(op_idx, dst);
                        let spill = x.map(|_| a.new_label());
                        // (In a speculative loop's fast version another
                        // layout leaves for the copy: the way must keep
                        // proving this one to the site's known successors.
                        // A layout that keeps arriving gets the loop
                        // recompiled, `tierup`.)
                        match num.spec_bail(&mut a, op_idx) {
                            Some(bail) => cold_bails.push((cached, bail)),
                            None => cold_caches.push((cached, back, helper, spill, s.cache_ptr)),
                        }
                        a.bind(back);
                        if let (Some(x), Some(spill)) = (x, spill) {
                            let past = a.new_label();
                            numeric::NumGen::set_xmm_from_rax(&mut a, x);
                            cold_reloads.push((spill, back, done, past, op_idx));
                            ool_past = Some(past);
                        } else {
                            a.store_rax(disp(dst));
                            a.jmp(done);
                        }
                    } else {
                        a.movabs_r11(s.cache_ptr as i64); // r11 = &PropertyCache
                        a.cmp_rax_at_r11(layout.cache_shape_off);
                        a.jne(helper);
                        a.mov_eax_at_r11(layout.cache_slot_off);
                        a.add_dword_at_r11_imm(layout.cache_hits_off, 1);
                    }
                    if s.baked.is_some() {
                        // Emitted above.
                    } else if let Some(x) = num.result_xmm(op_idx, dst) {
                        // In a promoted loop: the value straight into its SSE
                        // register; the spill-array read and the slow paths'
                        // reload are out of line (`cold_reloads`).
                        let spill = a.new_label();
                        let back = a.new_label();
                        let past = a.new_label();
                        a.p_obj_slot_load_inline(&layout, helper, spill);
                        a.bind(back);
                        numeric::NumGen::set_xmm_from_rax(&mut a, x);
                        cold_reloads.push((spill, back, done, past, op_idx));
                        ool_past = Some(past);
                    } else {
                        a.p_obj_slot_addr(&layout, call_seq::r::R11, helper);
                        a.g_load(call_seq::r::RAX, call_seq::r::R11, 0); // rax = NanBox at slot
                        a.store_rax(disp(dst));
                        a.jmp(done);
                    }
                    // The miss path is out of line (see `cold_gets`), or in
                    // a speculative loop's fast version leaves for its copy.
                    if let Some(bail) = num.spec_bail(&mut a, op_idx) {
                        cold_bails.push((helper, bail));
                    } else {
                        cold_gets.push((
                            helper,
                            done,
                            dst,
                            obj,
                            s.method_ptr,
                            call_site,
                            a.call_spill_now(),
                        ));
                    }
                    if ool_past.is_none() {
                        a.bind(done);
                    }
                }
                GenericOp::GetMethod {
                    dst,
                    obj,
                    site,
                    call,
                } => {
                    if !ok(dst) || !ok(obj) {
                        return None;
                    }
                    let call_site = *call_sites.get(call as usize)?;
                    let s = sites.get(site as usize)?;
                    if s.method_ptr.is_null() {
                        return None;
                    }
                    let helper = a.new_label();
                    let done = a.new_label();
                    // An `Op::GetMethod`'s read, as `CallMethod` looks its
                    // method up: the site's chain cache inline (own or
                    // inherited data), else the helper (which serves a
                    // built-in through the site's memo).
                    a.p_method_lookup(
                        &layout,
                        helpers.call_layout.realm,
                        (call_seq::r::RBX, disp(obj)),
                        s.method_ptr,
                        helper,
                        None,
                    );
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(helper);
                    // (In a speculative loop's fast version, which makes
                    // no call, the slow path leaves for the plain copy.)
                    if let Some(b) = num.spec_bail(&mut a, op_idx) {
                        a.jmp(b);
                    }
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(obj));
                    a.movabs_argreg(2, s.method_ptr as i64);
                    a.movabs_argreg(3, call_site as i64);
                    a.movabs_rax(get_method_site_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::SetProp {
                    obj,
                    val,
                    site,
                    strict,
                } => {
                    if !ok(obj) || !ok(val) {
                        return None;
                    }
                    let s = sites.get(site as usize)?;
                    let helper = a.new_label();
                    let done = a.new_label();
                    // The in-place store's shape miss, its object walked and
                    // its store guards passed: maybe an add.
                    let shape_miss = a.new_label();
                    // ===== inline monomorphic property-SET fast path =====
                    // On a shape-identity hit, an in-place `slots[slot] = val`, but
                    // ONLY when provably sound: the store needs nothing from the
                    // write barrier (see `p_barrier_check`) and the object is not
                    // frozen with an empty `readonly` list (`Object::cached_set`'s
                    // success precondition). Everything else → the helper.
                    // As for `GetProp`: a layout-proving way hit skips the
                    // shape compare (the walk's resumption makes it, with the
                    // flags test, out of line).
                    let known = s.baked.filter(|b| shape_known(op_idx, b.shape));
                    let verify = known.map(|_| a.new_label());
                    // In a speculative loop's fast version every path but
                    // the in-place stores leaves for its plain copy: a slow
                    // path, an add, another layout than the compiled-in one
                    // (a known site's way must keep proving it).
                    let bail = num.spec_bail(&mut a, op_idx);
                    // A known site of the fast version whose flags test is
                    // proven, storing a Number through the frame's `extra`
                    // pointer, never reads the heap slot (`rcx`).
                    let cell = !(num.in_region(op_idx)
                        && bail.is_some()
                        && num.is_number_at(op_idx, val)
                        && known.is_some_and(|b| {
                            rc.sites[op_idx].is_some_and(|p| {
                                p.known
                                    && p.flags_from.is_some_and(|q| {
                                        set_proven.get(q).copied().unwrap_or(false)
                                    })
                                    && !X64Assembler::p_fixed_slot_needs_rcx(
                                        &layout,
                                        b.slot,
                                        p.pred.is_some_and(|q| extra_in_frame[q]),
                                    )
                            })
                        }));
                    recv_slot!(op_idx, obj, helper, recv_cache::Family::Obj, verify, cell);
                    // A site the interpreter saw add one key to one layout
                    // tries that add first, compiled in, and has everything
                    // else out of line (in a speculative loop's fast version
                    // an add leaves for the plain copy instead).
                    let mut cold_add = None;
                    if let Some(b) = s.add
                        && bail.is_none()
                        && s.baked.is_none()
                        && !s.method_ptr.is_null()
                    {
                        let try_store = a.new_label();
                        a.p_add_baked(
                            &layout,
                            helpers.call_layout.realm,
                            s.method_ptr,
                            b,
                            disp(val),
                            try_store,
                            helper,
                        );
                        a.jmp(done);
                        cold_add = Some(a.cold_begin());
                        a.bind(try_store);
                    }
                    // A frozen object's writes need the helper, as do those of
                    // a host-exotic, global or tracked-prototype object, and of
                    // an object with a non-writable property
                    // (`JIT_FLAG_READONLY`). (A watched object's existing data
                    // properties are plain stores: a value is no part of the
                    // layout its watchers validate.) Only Rust code changes
                    // the flags byte, so a receiver an earlier store of the
                    // iteration tested with none run since passes again.
                    let in_spec_region = num.in_region(op_idx) && bail.is_some();
                    let flags_proven = in_spec_region
                        && rc.sites[op_idx]
                            .filter(|p| p.known)
                            .and_then(|p| p.flags_from)
                            .is_some_and(|p| set_proven.get(p).copied().unwrap_or(false));
                    if !flags_proven {
                        a.p_test_byte_imm(
                            call_seq::r::RCX,
                            layout.props.off_jit_flags,
                            props::FLAGS_NO_INLINE_STORE,
                        );
                        a.jne(helper);
                    }
                    if in_spec_region {
                        set_proven[op_idx] = true;
                    }
                    if known.is_none() {
                        a.mov_rax_at_rcx(layout.off_shape);
                    }
                    // Where the in-place store of a compiled-in layout ends
                    // (every other path may have added a property, and ends
                    // at `done`, which empties the layout-proving ways).
                    let stored = a.new_label();
                    // With a compiled-in layout, everything but its in-place
                    // store is out of line (a cold region), so that store
                    // runs straight on into the next op.
                    let mut cold = None;
                    if let Some(b) = s.baked {
                        // The layout the site settled on, compiled in (never
                        // a primitive wrapper's): a compare with the shape and
                        // a store at a fixed offset; any other shape takes
                        // the cache path just below.
                        let cached = a.new_label();
                        if let Some(verify) = verify {
                            let resume = a.new_label();
                            a.bind(resume);
                            cold_verifies.push((verify, Some(helper), b.shape, cached, resume));
                        } else {
                            a.g_movabs(call_seq::r::R11, b.shape);
                            a.cmp_rax_r11();
                            a.jne(cached);
                        }
                        if !num.is_number_at(op_idx, val) {
                            a.load_rax(disp(val));
                            a.p_barrier_check(&layout, helpers.call_layout.realm, helper);
                        }
                        a.load_rdx(disp(val));
                        match rc.sites[op_idx].filter(|_| in_spec_region) {
                            Some(p) => {
                                let have = p.known && p.pred.is_some_and(|q| extra_in_frame[q]);
                                extra_in_frame[op_idx] = a.p_obj_fixed_slot_ex(
                                    &layout,
                                    b.slot,
                                    true,
                                    rc_extra(p.way),
                                    have,
                                );
                            }
                            None => a.p_obj_fixed_slot_store(&layout, b.slot),
                        }
                        a.jmp(stored);
                        cold = Some(a.cold_begin());
                        a.bind(cached);
                        // Another layout in a way that proves this one: empty
                        // it (and the others: it may be an add) — also before
                        // leaving for the plain copy, whose hit on the way
                        // would skip the compare.
                        for &w in &shape_ways {
                            a.store_empty(rc_key(w));
                        }
                        if let Some(b) = bail {
                            a.jmp(b);
                        }
                    }
                    a.movabs_r11(s.cache_ptr as i64);
                    a.cmp_rax_at_r11(layout.cache_shape_off);
                    a.jne(shape_miss);
                    // A primitive wrapper (its shape's brand bit 0) — the helper.
                    a.p_test_byte_imm(call_seq::r::RAX, layout.props.shape_brands, 1);
                    a.jne(helper);
                    if num.is_number_at(op_idx, val) {
                        // A Number: nothing for the write barrier to do, so
                        // straight to the slot (`r11` still the cache).
                        let spill = a.new_label();
                        a.mov_eax_at_r11(layout.cache_slot_off);
                        a.add_dword_at_r11_imm(layout.cache_hits_off, 1);
                        a.load_rdx(disp(val));
                        a.p_obj_slot_store_inline(&layout, helper, spill);
                        a.jmp(done);
                        a.bind(spill);
                        a.p_obj_slot_store_spill(&layout, done);
                    } else {
                        a.load_rax(disp(val));
                        a.p_barrier_check(&layout, helpers.call_layout.realm, helper);
                        a.movabs_r11(s.cache_ptr as i64);
                        a.mov_eax_at_r11(layout.cache_slot_off);
                        a.add_dword_at_r11_imm(layout.cache_hits_off, 1);
                        a.p_obj_slot_addr(&layout, call_seq::r::R11, helper);
                        a.load_rdx(disp(val));
                        a.g_store(call_seq::r::R11, 0, call_seq::r::RDX); // slots[slot] = val
                        a.jmp(done);
                    }
                    // ===== inline add-cache hit (`p_add_prop_inline`): the
                    // write adds the key `this.x = x` adds, by the site's
                    // recorded transition =====
                    let call_helper = a.new_label();
                    // After the store's shape miss the object is walked
                    // (`rcx`) and its store guards hold.
                    a.bind(shape_miss);
                    if let Some(b) = bail {
                        a.jmp(b);
                    }
                    if !s.method_ptr.is_null() {
                        // (No walk follows, so no arming test or arena: the
                        // add's own guards miss on an empty cache.)
                        a.p_add_prop_inline(
                            &layout,
                            helpers.call_layout.realm,
                            s.method_ptr,
                            disp(val),
                            call_helper,
                            true,
                        );
                        a.jmp(done);
                    }
                    a.jmp(call_helper);
                    a.bind(helper);
                    if let Some(b) = bail {
                        a.jmp(b);
                    }
                    if !s.method_ptr.is_null() {
                        a.p_add_armed(s.method_ptr, call_helper);
                        a.p_load_arena(helpers.call_layout.realm, &layout.props);
                        walk_to_slot!(obj, call_helper);
                        a.p_add_prop_inline(
                            &layout,
                            helpers.call_layout.realm,
                            s.method_ptr,
                            disp(val),
                            call_helper,
                            false,
                        );
                        a.jmp(done);
                    }
                    // jit_helper_set_prop_site(ctx, obj, site, val, strict).
                    a.bind(call_helper);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(obj));
                    a.movabs_argreg(2, s.method_ptr as i64);
                    a.load_argreg(3, disp(val));
                    a.movabs_argreg(4, i64::from(strict));
                    a.movabs_rax(set_site_helper as i64);
                    a.call_rax();
                    throw_check!();
                    if let Some(mark) = cold_add {
                        a.jmp(done);
                        a.cold_end(mark);
                    }
                    a.bind(done);
                    // Maybe an add: to an object a layout-proving way holds.
                    for &w in &shape_ways {
                        a.store_empty(rc_key(w));
                    }
                    if let Some(mark) = cold {
                        a.jmp(stored);
                        a.cold_end(mark);
                    }
                    a.bind(stored);
                }
                GenericOp::GetElem { dst, obj, index } => {
                    if !ok(dst) || !ok(obj) || !ok(index) {
                        return None;
                    }
                    let helper = a.new_label();
                    let done = a.new_label();
                    // ===== inline dense-array element GET fast path =====
                    // Receiver cell tag == Array, index an exact non-negative
                    // integer in `[0, dense_len)`, and — matching the interpreter's
                    // `vm_array_index_get` — bail on a hole (the helper walks the
                    // prototype chain). Everything else → `jit_helper_get_elem`.
                    recv_slot!(op_idx, obj, helper, recv_cache::Family::Arr);
                    // (A promoted loop's index may be in its SSE register.)
                    let index_x = num.result_xmm(op_idx, index);
                    match index_x {
                        Some(x) => numeric::NumGen::index_from_xmm(
                            &mut a,
                            x,
                            !num.is_number_at(op_idx, index),
                            helper,
                        ),
                        None => int_index!(index, helper, num.is_number_at(op_idx, index)),
                    }
                    a.p_arr_len_ptr(&layout);
                    a.cmp_rax_r8();
                    a.jae(helper); // OOB / negative
                    a.mov_rax_at_r11_rax8(); // rax = elems[idx]
                    a.movabs_r11(NanBox::hole().to_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(helper);
                    if let Some(x) = num.result_xmm(op_idx, dst) {
                        let past = a.new_label();
                        numeric::NumGen::set_xmm_from_rax(&mut a, x);
                        a.jmp(past);
                        past_reload = Some(past);
                    } else {
                        a.store_rax(disp(dst));
                        a.jmp(done);
                    }
                    // jit_helper_get_elem(ctx, recv, key) (or, in a
                    // speculative loop's fast version, its plain copy).
                    let bail = num.spec_bail(&mut a, op_idx);
                    a.bind(helper);
                    if let Some(b) = bail {
                        a.jmp(b);
                    }
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(obj));
                    match index_x {
                        Some(x) => numeric::NumGen::gpr_from_xmm(&mut a, 2, x), // rdx
                        None => a.load_argreg(2, disp(index)),
                    }
                    a.movabs_rax(get_elem_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::SetElem {
                    obj,
                    index,
                    val,
                    strict,
                } => {
                    if !ok(obj) || !ok(index) || !ok(val) {
                        return None;
                    }
                    let helper = a.new_label();
                    let done = a.new_label();
                    // ===== inline dense-array element SET fast path =====
                    // In place only when identical to `vm_set_elem`: a store the
                    // write barrier has nothing to do for (`p_barrier_check`),
                    // an array without side-table state (`p_array_plain`: not
                    // frozen/sealed, no per-index override, within
                    // `max_array_len`), and an in-bounds exact integer index.
                    recv_slot!(op_idx, obj, helper, recv_cache::Family::Arr);
                    a.p_array_plain(&layout, helpers.call_layout.realm, true, helper);
                    a.load_rax(disp(val));
                    a.p_barrier_check(&layout, helpers.call_layout.realm, helper);
                    int_index!(index, helper);
                    a.p_arr_len_ptr(&layout);
                    a.cmp_rax_r8();
                    a.jae(helper);
                    // A hole is an absent index: the helper's `[[Set]]` walks the
                    // prototype chain (setters, proxy traps).
                    a.movabs_argreg(2, NanBox::hole().to_bits() as i64);
                    a.cmp_at_r11_rax8_rdx();
                    a.je(helper);
                    a.load_rdx(disp(val));
                    a.mov_at_r11_rax8_rdx(); // elems[idx] = val
                    a.jmp(done);
                    // jit_helper_set_elem(ctx, recv, key, val).
                    a.bind(helper);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(obj));
                    a.load_argreg(2, disp(index));
                    a.load_argreg(3, disp(val));
                    a.movabs_argreg(4, i64::from(strict));
                    a.movabs_rax(set_elem_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
                GenericOp::ArrayLen { dst, arr } => {
                    if !ok(dst) || !ok(arr) {
                        return None;
                    }
                    let helper = a.new_label();
                    let done = a.new_label();
                    // ===== inline dense-array `.length` fast path =====
                    // Only for a receiver whose spec `.length` provably equals the
                    // dense element count (`p_array_plain`); else the helper.
                    recv_slot!(op_idx, arr, helper, recv_cache::Family::Arr);
                    a.p_arr_len(&layout, call_seq::r::RAX);
                    if let Some(x) = num.result_xmm(op_idx, dst) {
                        let past = a.new_label();
                        numeric::NumGen::set_xmm_from_int_rax(&mut a, x);
                        a.jmp(past);
                        past_reload = Some(past);
                    } else {
                        a.cvtsi2sd_xmm0_rax();
                        a.movsd_mem_xmm0(disp(dst));
                        a.jmp(done);
                    }
                    let bail = num.spec_bail(&mut a, op_idx);
                    a.bind(helper);
                    if let Some(b) = bail {
                        a.jmp(b);
                    }
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(arr));
                    a.movabs_rax(array_len_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::Arith {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let fbin = match op {
                        crate::nbvm::GA_SUB => Some(FBinOp::Sub),
                        crate::nbvm::GA_MUL => Some(FBinOp::Mul),
                        crate::nbvm::GA_DIV => Some(FBinOp::Div),
                        _ => None, // GA_MOD
                    };
                    let done = a.new_label();
                    let slow = a.new_label();
                    if let Some(fb) = fbin {
                        both_numbers!(ra, rb, slow);
                        fbin_store!(fb, ra, rb, dst, done);
                    } else {
                        // `%` of two integral Numbers that convert exactly to
                        // i64: the integer remainder, which is fmod's exact
                        // result (sign of the dividend; a zero takes the
                        // dividend's sign bit, -0 included). A zero or `-1`
                        // divisor (no `idiv` overflow), a fraction, NaN or
                        // an infinity takes the helper.
                        both_numbers!(ra, rb, slow);
                        a.movsd_xmm0_mem(disp(rb));
                        a.cvttsd2si_rax_xmm0();
                        a.cvtsi2sd_xmm1_rax();
                        a.ucomisd_xmm0_xmm1();
                        a.jne(slow);
                        a.jp(slow);
                        a.mov_rcx_rax();
                        a.test_rcx_rcx();
                        a.je(slow);
                        a.cmp_rcx_imm8(-1);
                        a.je(slow);
                        a.movsd_xmm0_mem(disp(ra));
                        a.cvttsd2si_rax_xmm0();
                        a.cvtsi2sd_xmm1_rax();
                        a.ucomisd_xmm0_xmm1();
                        a.jne(slow);
                        a.jp(slow);
                        a.cqo();
                        a.idiv_rcx();
                        let nonzero = a.new_label();
                        a.test_rdx_rdx();
                        a.jne(nonzero);
                        a.load_rax(disp(ra));
                        a.movabs_r11(i64::MIN); // the sign bit
                        a.and_rax_r11();
                        a.store_rax(disp(dst));
                        a.jmp(done);
                        a.bind(nonzero);
                        a.mov_rax_rdx();
                        a.cvtsi2sd_xmm0_rax();
                        a.movsd_mem_xmm0(disp(dst));
                        a.jmp(done);
                    }
                    a.bind(slow);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(ra));
                    a.load_argreg(2, disp(rb));
                    a.movabs_argreg(3, i64::from(op));
                    a.movabs_rax(arith_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::Lt { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    both_numbers!(ra, rb, slow);
                    // `a < b` == `b > a` (ordered); false for a NaN operand.
                    a.movsd_xmm0_mem(disp(rb));
                    a.ucomisd_xmm0_mem(disp(ra));
                    a.seta_rax();
                    a.movabs_r11(tag_false as i64);
                    a.add_rax_r11();
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(ra));
                    a.load_argreg(2, disp(rb));
                    a.movabs_rax(lt_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::Compare {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                    pc,
                } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    both_numbers!(ra, rb, slow);
                    // `ucomisd x, y` then `seta` = x > y, `setae` = x >= y, both
                    // false when unordered (NaN) — JS relational semantics.
                    let (x, y) = match op {
                        0 | 2 => (rb, ra), // a < b ≡ b > a;  a <= b ≡ b >= a
                        _ => (ra, rb),     // a > b;          a >= b
                    };
                    a.movsd_xmm0_mem(disp(x));
                    a.ucomisd_xmm0_mem(disp(y));
                    if op <= 1 {
                        a.seta_rax();
                    } else {
                        a.setae_rax();
                    }
                    a.movabs_r11(tag_false as i64);
                    a.add_rax_r11();
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::StrictEq { dst, a: ra, b: rb } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    both_numbers!(ra, rb, slow);
                    // Ordered equality (+0 === -0 true, NaN === NaN false).
                    a.movsd_xmm0_mem(disp(ra));
                    a.ucomisd_xmm0_mem(disp(rb));
                    a.ordered_equal_rax();
                    a.movabs_r11(tag_false as i64);
                    a.add_rax_r11();
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    // `jit_helper_strict_eq` never throws → no sentinel check.
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(ra));
                    a.load_argreg(2, disp(rb));
                    a.movabs_rax(strict_eq_helper as i64);
                    a.call_rax();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::ValueBin {
                    dst,
                    a: ra,
                    b: rb,
                    op,
                } => {
                    if !ok(dst) || !ok(ra) || !ok(rb) {
                        return None;
                    }
                    let is_loose =
                        op == crate::nbvm::VB_LOOSE_EQ || op == crate::nbvm::VB_LOOSE_NEQ;
                    let done = a.new_label();
                    let slow = a.new_label();
                    if is_loose {
                        both_numbers!(ra, rb, slow);
                        a.movsd_xmm0_mem(disp(ra));
                        a.ucomisd_xmm0_mem(disp(rb));
                        a.ordered_equal_rax();
                        if op == crate::nbvm::VB_LOOSE_NEQ {
                            a.xor_rax_imm(1);
                        }
                        a.movabs_r11(tag_false as i64);
                        a.add_rax_r11();
                        a.store_rax(disp(dst));
                        a.jmp(done);
                    }
                    a.bind(slow);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(ra));
                    a.load_argreg(2, disp(rb));
                    a.movabs_argreg(3, i64::from(op));
                    a.movabs_rax(value_bin_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                    a.bind(done);
                }
                GenericOp::Not { dst, a: ra } => {
                    if !ok(dst) || !ok(ra) {
                        return None;
                    }
                    if num.is_bool_at(op_idx, ra) {
                        // `!b` of a boolean: `b != true`.
                        a.load_rax(disp(ra));
                        a.movabs_r11(tag_true as i64);
                        a.cmp_rax_r11();
                        a.setne_rax();
                    } else {
                        // (In a speculative loop's fast version, the helper's
                        // values leave for the plain copy.)
                        let bail = num.spec_bail(&mut a, op_idx);
                        truthy_rax!(op_idx, ra, bail);
                        a.xor_rax_imm(1);
                    }
                    a.movabs_r11(tag_false as i64);
                    a.add_rax_r11();
                    a.store_rax(disp(dst));
                }
                GenericOp::Jump { target } => {
                    if target > ops.len() {
                        return None;
                    }
                    if target <= op_idx {
                        backedge!(target);
                    }
                    a.jmp(labels[target]);
                }
                GenericOp::JumpIfFalse { cond, target } => {
                    if !ok(cond) || target > ops.len() {
                        return None;
                    }
                    truthy_rax!(op_idx, cond, None);
                    a.test_rax_rax();
                    if target <= op_idx {
                        let stay = a.new_label();
                        a.jne(stay);
                        backedge!(target);
                        a.jmp(labels[target]);
                        a.bind(stay);
                    } else {
                        a.je(labels[target]); // falsy (truthy == 0) → branch
                    }
                }
                GenericOp::CallNative { dst, native, site } => {
                    if !ok(dst) {
                        return None;
                    }
                    let args = call_args.get(site as usize)?;
                    for (i, &ar) in args.iter().enumerate() {
                        if !ok(ar) {
                            return None;
                        }
                        a.load_rax(disp(ar));
                        a.store_rax(argbuf_disp(i));
                    }
                    // jit_helper_call_native(ctx, native_id, argv_ptr, argc).
                    a.mov_rdi_r15();
                    a.movabs_argreg(1, i64::from(native));
                    a.lea_argreg(2, argbuf_disp(0));
                    a.movabs_argreg(3, args.len() as i64);
                    a.movabs_rax(call_native_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(disp(dst));
                }
                GenericOp::Ret { src } => {
                    if !ok(src) {
                        return None;
                    }
                    a.load_rax(disp(src));
                    a.frame_epilogue();
                }
                GenericOp::CallKey {
                    dst,
                    recv,
                    key,
                    site,
                    cache,
                    pc,
                } => {
                    let args = call_args.get(site as usize)?;
                    if !ok(dst) || !ok(recv) || !ok(key) || args.iter().any(|&r| !ok(r)) {
                        return None;
                    }
                    if pc as usize != op_idx {
                        return None;
                    }
                    let site_ptr = *call_sites.get(cache as usize)?;
                    // method = recv[key], once: inline for an in-bounds,
                    // present element of a plain dense array (exactly
                    // `GetElem`'s fast path), else the interpreter's read.
                    let read = a.new_label();
                    let have = a.new_label();
                    recv_slot!(op_idx, recv, read, recv_cache::Family::Arr);
                    int_index!(key, read);
                    a.p_arr_len_ptr(&layout);
                    a.cmp_rax_r8();
                    a.jae(read);
                    a.mov_rax_at_r11_rax8();
                    a.movabs_r11(NanBox::hole().to_bits() as i64);
                    a.cmp_rax_r11();
                    a.je(read);
                    a.store_rax(method_scratch);
                    a.jmp(have);
                    a.bind(read);
                    a.mov_rdi_r15();
                    a.load_argreg(1, disp(recv));
                    a.load_argreg(2, disp(key));
                    a.movabs_rax(get_elem_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.store_rax(method_scratch);
                    a.bind(have);
                    let slow = a.new_label();
                    let done = a.new_label();
                    call_seq::emit_call(
                        &mut a,
                        &call_seq::CallEmit {
                            dst,
                            callee: call_seq::CalleeSrc::Frame {
                                scratch: method_scratch,
                            },
                            this: Some(recv),
                            args,
                            site: site_ptr,
                            layout: &helpers.call_layout,
                            throw_exit,
                            slow,
                            done,
                        },
                    )?;
                    a.bind(slow);
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_argreg(4, site_ptr as i64);
                    a.load_argreg(5, method_scratch);
                    a.movabs_rax(call_this_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
                GenericOp::Construct {
                    dst,
                    ctor,
                    site,
                    cache,
                    pc,
                } => {
                    let args = call_args.get(site as usize)?;
                    if !ok(dst) || !ok(ctor) || args.iter().any(|&r| !ok(r)) {
                        return None;
                    }
                    if pc as usize != op_idx {
                        return None;
                    }
                    let site_ptr = *call_sites.get(cache as usize)?;
                    let slow = a.new_label();
                    let done = a.new_label();
                    call_seq::emit_call(
                        &mut a,
                        &call_seq::CallEmit {
                            dst,
                            callee: call_seq::CalleeSrc::Construct {
                                ctor,
                                new_target: ctor,
                                scratch: method_scratch,
                                helper: construct_helper,
                                result: construct_result_helper,
                                alloc: construct_alloc_helper,
                                bind_this: None,
                            },
                            this: None,
                            args,
                            site: site_ptr,
                            layout: &helpers.call_layout,
                            throw_exit,
                            slow,
                            done,
                        },
                    )?;
                    a.bind(slow);
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_argreg(4, site_ptr as i64);
                    a.movabs_argreg(5, 0);
                    a.movabs_rax(construct_slow_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
                GenericOp::PlainPrims { dst, site, pc } => {
                    let rs = call_args.get(site as usize)?;
                    if !ok(dst) || rs.iter().any(|&r| !ok(r)) || pc as usize != op_idx {
                        return None;
                    }
                    let slow = a.new_label();
                    let done = a.new_label();
                    for &r in rs {
                        if num.is_number_at(op_idx, r) {
                            continue;
                        }
                        // A heap value (a String among them): the interpreter's.
                        a.load_rax(disp(r));
                        a.movabs_r11(HANDLE_TAG as i64);
                        a.and_rax_r11();
                        a.cmp_rax_r11();
                        a.je(slow);
                    }
                    a.movabs_rax(tag_true as i64);
                    a.store_rax(disp(dst));
                    a.jmp(done);
                    a.bind(slow);
                    exec_op!(pc);
                    a.bind(done);
                }
                GenericOp::CallWithMethod {
                    dst,
                    callee,
                    recv,
                    site,
                    cache,
                    ic,
                    pc,
                } => {
                    let args = call_args.get(site as usize)?;
                    if !ok(dst) || !ok(callee) || !ok(recv) || args.iter().any(|&r| !ok(r)) {
                        return None;
                    }
                    if pc as usize != op_idx {
                        return None;
                    }
                    let site_ptr = *call_sites.get(cache as usize)?;
                    let si = sites.get(ic as usize)?;
                    let slow = a.new_label();
                    let done = a.new_label();
                    call_seq::emit_call(
                        &mut a,
                        &call_seq::CallEmit {
                            dst,
                            callee: call_seq::CalleeSrc::Reg(callee),
                            this: Some(recv),
                            args,
                            site: site_ptr,
                            layout: &helpers.call_layout,
                            throw_exit,
                            slow,
                            done,
                        },
                    )?;
                    a.bind(slow);
                    let native = a.new_label();
                    // The built-in the site's helper last served (its bits
                    // in the site's identity memo): straight to the helper
                    // that serves it.
                    a.load_rax(disp(callee));
                    a.g_movabs(call_seq::r::R11, site_ptr as u64);
                    a.g_cmp_mem(call_seq::r::RAX, call_seq::r::R11, call_seq::IDENT_KEY);
                    a.g_jcc(call_seq::cc::E, native);
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_argreg(4, si.cache_ptr as i64);
                    a.movabs_argreg(5, site_ptr as i64);
                    a.movabs_rax(call_with_method_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.jmp(done);
                    a.bind(native);
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_argreg(4, si.cache_ptr as i64);
                    a.movabs_argreg(5, site_ptr as i64);
                    a.movabs_rax(call_native_got_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
                GenericOp::CallValue {
                    dst,
                    callee,
                    site,
                    cache,
                    pc,
                } => {
                    let args = call_args.get(site as usize)?;
                    if !ok(dst) || !ok(callee) || args.iter().any(|&r| !ok(r)) {
                        return None;
                    }
                    if pc as usize != op_idx {
                        return None;
                    }
                    let site_ptr = *call_sites.get(cache as usize)?;
                    let slow = a.new_label();
                    let done = a.new_label();
                    call_seq::emit_call(
                        &mut a,
                        &call_seq::CallEmit {
                            dst,
                            callee: call_seq::CalleeSrc::Reg(callee),
                            this: None,
                            args,
                            site: site_ptr,
                            layout: &helpers.call_layout,
                            throw_exit,
                            slow,
                            done,
                        },
                    )?;
                    a.bind(slow);
                    let op_ptr: *const crate::nbvm::Op = bytecode.get(pc as usize)?;
                    a.mov_rdi_r15();
                    a.mov_rsi_rbx();
                    a.movabs_argreg(2, n_regs as i64);
                    a.movabs_argreg(3, op_ptr as i64);
                    a.movabs_argreg(4, site_ptr as i64);
                    a.movabs_argreg(5, 0);
                    a.movabs_rax(call_value_helper as i64);
                    a.call_rax();
                    throw_check!();
                    a.bind(done);
                }
            }
            if in_region {
                a.set_call_spill(Vec::new());
                if ool_past.is_none() {
                    num.after_generic(&mut a, op_idx);
                }
            }
            if let Some(l) = past_reload.or(ool_past) {
                a.bind(l);
            }
        }
        num.set_copy(false);
        // Falling off the end of the stream (or branching to `len`).
        a.bind(labels[ops.len()]);
        a.movabs_rax(NanBox::jit_falloff_bits() as i64);
        a.frame_epilogue();
        // Leaving at a loop header for the interpreter (the poll recorded
        // which one).
        a.bind(leave_exit);
        a.movabs_rax(NanBox::jit_exit_bits() as i64);
        a.frame_epilogue();
        // The numeric emitter's out-of-line slow paths (after a `ret`, so no op
        // falls through into them).
        num.finish(&mut a, &labels, throw_exit);
        // The global binding cache's resets: the epoch changed since the
        // slots were validated, so empty them all, note the current epoch
        // (in `rax`) and refill this one.
        // (When the persistent cache holds the read slots validated under
        // the current epoch, the frame takes them and the op looks again.)
        let gc_persisted: Vec<(usize, i32)> =
            gc_reads.iter().map(|&k| (1 + k, gc_slot(k))).collect();
        for (reset, fetch, recheck) in core::mem::take(&mut cold_epochs) {
            a.bind(reset);
            a.clear_entry_slots();
            a.g_store(call_seq::r::RBP, gc_epoch, call_seq::r::RAX);
            if !gc_persisted.is_empty() {
                a.gp_restore(persist, 0, &gc_persisted, recheck);
            }
            a.jmp(fetch);
        }
        // The global binding cache's refills: a validated address is cached
        // and used, else the interpreter runs the op.
        let gc_group: Vec<usize> = gc_reads.iter().map(|&k| 1 + k).collect();
        for (fetch, back, done, pc, slot, k) in core::mem::take(&mut cold_globals) {
            let op_ptr: *const crate::nbvm::Op = bytecode.get(pc)?;
            let slow = a.new_label();
            a.bind(fetch);
            a.mov_rdi_r15();
            a.movabs_argreg(1, op_ptr as i64);
            a.movabs_rax(helpers.global_ref as usize as i64);
            a.call_rax();
            a.test_rax_rax();
            a.je(slow);
            a.g_store(call_seq::r::RBP, slot, call_seq::r::RAX);
            if gc_reads.contains(&k) {
                a.gp_save_one(persist, 0, &gc_group, 1 + k, gc_epoch);
            }
            a.jmp(back);
            a.bind(slow);
            exec_op!(pc);
            a.jmp(done);
        }
        // The out-of-line halves of promoted-loop `GetProp`s: the spill-array
        // read, and the slow paths' reload of the promoted result.
        for (spill, back, done, past, pc) in core::mem::take(&mut cold_reloads) {
            a.bind(spill);
            a.p_obj_slot_load_spill(&layout, back);
            a.bind(done);
            num.after_generic(&mut a, pc);
            a.jmp(past);
        }
        // The cache paths of sites with a compiled-in layout, for a receiver
        // of another shape (in `rax`; its heap slot in `rcx`): the site's
        // cache, else the miss path. The value goes back in `rax`.
        for (cached, back, miss, spill, cache) in core::mem::take(&mut cold_caches) {
            a.bind(cached);
            // Not the layout: a way that proves it may not keep this object.
            for &w in &shape_ways {
                a.store_empty(rc_key(w));
            }
            a.movabs_r11(cache as i64);
            a.cmp_rax_at_r11(layout.cache_shape_off);
            a.jne(miss);
            a.mov_eax_at_r11(layout.cache_slot_off);
            a.add_dword_at_r11_imm(layout.cache_hits_off, 1);
            match spill {
                // The promoted result's spill read is `cold_reloads`'.
                Some(spill) => {
                    a.p_obj_slot_load_inline(&layout, miss, spill);
                    a.jmp(back);
                }
                None => {
                    let spill = a.new_label();
                    a.p_obj_slot_load_inline(&layout, miss, spill);
                    a.jmp(back);
                    a.bind(spill);
                    a.p_obj_slot_load_spill(&layout, back);
                }
            }
        }
        // The shape compares that a layout-proving way's hit skips, after the
        // way's full walk: the store guards first, for a write (to `miss`),
        // then the layout (else to the site's cache path).
        for (verify, store_miss, shape, cached, resume) in core::mem::take(&mut cold_verifies) {
            a.bind(verify);
            if let Some(miss) = store_miss {
                a.p_test_byte_imm(
                    call_seq::r::RCX,
                    layout.props.off_jit_flags,
                    props::FLAGS_NO_INLINE_STORE,
                );
                a.jne(miss);
            }
            a.mov_rax_at_rcx(layout.off_shape);
            a.g_movabs(call_seq::r::R11, shape);
            a.cmp_rax_r11();
            a.jne(cached);
            a.jmp(resume);
        }
        // The receiver cache's full walks: each refills its way and resumes.
        for (walk, back, recv, miss, w, fam) in core::mem::take(&mut cold_walks) {
            a.bind(walk);
            a.p_load_arena(helpers.call_layout.realm, &layout.props);
            walk_to_slot!(recv, miss);
            family_checks!(fam, miss);
            a.g_store(call_seq::r::RBP, rc_key(w), call_seq::r::RAX);
            a.g_store(call_seq::r::RBP, rc_cell(w), call_seq::r::RCX);
            a.jmp(back);
        }
        for (entry, bail) in core::mem::take(&mut cold_bails) {
            a.bind(entry);
            // A layout-proving way may hold the receiver whose other layout
            // sent it here (a full walk fills the way before the shape
            // compare): empty them all, or the plain copy's hit on it would
            // skip that compare (`recv_cache::shape_ways`).
            for &w in &shape_ways {
                a.store_empty(rc_key(w));
            }
            a.jmp(bail);
        }
        // The `GetProp` miss paths: the inherited data property or getter,
        // else the helper; each returns to its op's `done`.
        for (helper, done, dst, obj, method_ptr, call_site, spill) in
            core::mem::take(&mut cold_gets)
        {
            // An inherited data property or getter, through the site's
            // chain cache: the value, or the getter's native call. (In a
            // promoted loop, its calls bracket the promoted registers.)
            a.set_call_spill(spill);
            a.bind(helper);
            let slow = a.new_label();
            let getter = a.new_label();
            a.p_method_lookup(
                &layout,
                helpers.call_layout.realm,
                (call_seq::r::RBX, disp(obj)),
                method_ptr,
                slow,
                Some(getter),
            );
            a.store_rax(disp(dst));
            a.jmp(done);
            a.bind(getter);
            a.store_rax(method_scratch);
            call_seq::emit_call(
                &mut a,
                &call_seq::CallEmit {
                    dst,
                    callee: call_seq::CalleeSrc::Frame {
                        scratch: method_scratch,
                    },
                    this: Some(obj),
                    args: &[],
                    site: call_site,
                    layout: &helpers.call_layout,
                    throw_exit,
                    slow,
                    done,
                },
            )?;
            // jit_helper_get_prop_site(ctx, obj, site, call site).
            a.bind(slow);
            a.mov_rdi_r15();
            a.load_argreg(1, disp(obj));
            a.movabs_argreg(2, method_ptr as i64);
            a.movabs_argreg(3, call_site as i64);
            a.movabs_rax(get_site_helper as i64);
            a.call_rax();
            throw_check!();
            a.store_rax(disp(dst));
            a.jmp(done);
            a.set_call_spill(Vec::new());
        }
        // Throw exit. In a body with `try` regions, first offer the pending fault
        // to the activation's innermost handler (exactly the interpreter's
        // `handle_throw!`): a caught throw resumes at its catch pc. Otherwise
        // return the sentinel so the caller reads the pending fault.
        a.bind(throw_exit);
        if has_inline {
            // A throw out of an inlined body: its activation ends here.
            call_seq::emit_inline_leave(&mut a, &helpers.call_layout, &inl_slots);
        }
        if !catch_targets.is_empty() {
            a.mov_rdi_r15();
            a.mov_rsi_rbx();
            a.movabs_argreg(2, n_regs as i64);
            a.movabs_rax(unwind_helper as i64);
            a.call_rax();
            // A throw may have left the host's running scope elsewhere: the
            // global binding cache starts over in the handler.
            if gc_plan.slots > 0 {
                a.store_empty(gc_epoch);
            }
            for &t in &catch_targets {
                a.cmp_rax_imm(i32::try_from(orig_pc(t)).ok()?);
                a.je(labels[t]);
            }
        }
        a.bind(refuse_exit);
        a.movabs_rax(NanBox::jit_throw_bits() as i64);
        a.frame_epilogue();
        // (Out of line, called: see `call_seq::emit_inline_stub`.)
        for (stub, argv) in core::mem::take(&mut inline_stubs) {
            a.bind(stub);
            call_seq::emit_inline_stub(&mut a, &helpers.call_layout, &inl_slots, &argv)?;
        }
        // The native-call entry (`call_seq::emit_native_entry`), behind one
        // that gives it a handler-stack segment of its own when the body has
        // `try` regions.
        let native = a.new_label();
        a.bind(native);
        call_seq::emit_native_entry(
            &mut a, frame, COUNTDOWN, n_regs, n_params, n_caps, new_target, labels[0],
        )?;
        let uses_handlers = ops
            .iter()
            .any(|o| matches!(o, GenericOp::PushHandler { .. } | GenericOp::PopHandler));
        let entry = if uses_handlers {
            let outer = a.new_label();
            a.bind(outer);
            call_seq::emit_handler_entry(&mut a, helpers.handler_frame as usize as u64, native);
            outer
        } else {
            native
        };
        let native_off = a.label_offset(entry);
        Self::from_code(&a.finish()).map(|f| (f, native_off))
    }

    /// Wraps already-emitted machine `code` into a callable region (the entry
    /// point for hand-assembled functions, e.g. one that calls another).
    #[must_use]
    pub fn from_machine_code(code: &[u8]) -> Option<Self> {
        Self::from_code(code)
    }

    /// Invokes a generic-tier (frame ABI) body:
    /// `extern "C" fn(ctx, regs, entry) -> u64`.
    #[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub fn call_generic(&self, ctx: *mut core::ffi::c_void, regs: *mut u64, entry: u64) -> u64 {
        // SAFETY: `buf` holds verified, self-emitted machine code following the
        // System V ABI for this signature; `ctx`/`regs` are valid for the call
        // (the caller guarantees a live `Ctx` and a register file at least as long
        // as the body's `n_regs`, which the emitted code indexes up to), and
        // `entry` is one of the body's entry pcs (any other value makes the body
        // return the throw sentinel without touching `regs`).
        #[allow(unsafe_code)]
        let f: extern "C" fn(*mut core::ffi::c_void, *mut u64, u64) -> u64 =
            unsafe { core::mem::transmute(self.buf.ptr()) };
        f(ctx, regs, entry)
    }

    /// The executable entry address of this function — the target a *caller's*
    /// native code emits a `call` to (intra-JIT calls). `0` if unavailable.
    #[must_use]
    pub fn code_ptr(&self) -> usize {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            self.buf.ptr() as usize
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            0
        }
    }

    /// Calls a compiled float function with up to 4 `f64` arguments.
    #[must_use]
    pub fn call_args_f64(&self, args: &[f64]) -> f64 {
        let mut a = [0.0f64; 4];
        for (slot, v) in a.iter_mut().zip(args.iter()) {
            *slot = *v;
        }
        self.call_f64x4(a)
    }

    /// [`Self::call_args_f64`] with the four argument registers given (the
    /// unused ones are ignored by the callee).
    #[must_use]
    #[inline]
    pub fn call_f64x4(&self, a: [f64; 4]) -> f64 {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: the compiled code follows the System V ABI for
            // `extern "C" fn(f64 x4) -> f64` (args in xmm0..xmm3, result in xmm0);
            // unused register args are ignored by the callee.
            #[allow(unsafe_code)]
            let f: extern "C" fn(f64, f64, f64, f64) -> f64 =
                unsafe { core::mem::transmute(self.buf.ptr()) };
            f(a[0], a[1], a[2], a[3])
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = a;
            unreachable!("JitFunction cannot be constructed on this target")
        }
    }

    /// Calls a compiled register-machine function with up to 6 `i64` arguments.
    #[must_use]
    pub fn call_args(&self, args: &[i64]) -> i64 {
        // The compiled code reads only the args it declared; pad to 6 so the call
        // matches a fixed 6-arg System V signature.
        let mut a = [0i64; 6];
        for (slot, v) in a.iter_mut().zip(args.iter()) {
            *slot = *v;
        }
        self.call_int6(a)
    }

    /// [`Self::call_args`] with the six argument registers given (the unused
    /// ones are ignored by the callee).
    #[must_use]
    #[inline]
    pub fn call_int6(&self, a: [i64; 6]) -> i64 {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: the compiled code follows the System V ABI for
            // `extern "C" fn(i64 x6) -> i64`; reading fewer than 6 args is sound
            // (extra register args are simply ignored by the callee).
            #[allow(unsafe_code)]
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                unsafe { core::mem::transmute(self.buf.ptr()) };
            f(a[0], a[1], a[2], a[3], a[4], a[5])
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = a;
            unreachable!("JitFunction cannot be constructed on this target")
        }
    }

    /// Calls the compiled function with one argument.
    #[must_use]
    pub fn call1(&self, a: i64) -> i64 {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: `buf` holds verified, self-emitted machine code following the
            // System V ABI for `extern "C" fn(i64) -> i64`; the memory is mapped
            // executable and outlives the call (it is owned by `self`).
            #[allow(unsafe_code)]
            let f: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(self.buf.ptr()) };
            f(a)
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = a;
            unreachable!("JitFunction cannot be constructed on this target")
        }
    }

    /// Calls the compiled function with two arguments.
    #[must_use]
    pub fn call2(&self, a: i64, b: i64) -> i64 {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            // SAFETY: as `call1`, for `extern "C" fn(i64, i64) -> i64`.
            #[allow(unsafe_code)]
            let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(self.buf.ptr()) };
            f(a, b)
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            let _ = (a, b);
            unreachable!("JitFunction cannot be constructed on this target")
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn from_code(code: &[u8]) -> Option<Self> {
        exec::ExecBuffer::new(code).map(|buf| Self { buf })
    }

    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    fn from_code(_code: &[u8]) -> Option<Self> {
        None
    }
}

/// A native binary operation for [`JitFunction::compile_binary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
}

#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
mod numeric;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod exec {
    //! W^X executable memory, mapped via direct Linux x86-64 syscalls (no libc).

    /// `PROT_READ | PROT_WRITE`
    const PROT_RW: usize = 0x1 | 0x2;
    /// `PROT_READ | PROT_EXEC`
    const PROT_RX: usize = 0x1 | 0x4;
    /// `MAP_PRIVATE | MAP_ANONYMOUS`
    const MAP_PRIVATE_ANON: usize = 0x02 | 0x20;

    const SYS_MMAP: usize = 9;
    const SYS_MPROTECT: usize = 10;
    const SYS_MUNMAP: usize = 11;

    /// Issues a Linux x86-64 syscall with up to six arguments.
    ///
    /// SAFETY: caller must pass a valid syscall number and arguments; this just
    /// executes the `syscall` instruction with the System V syscall ABI register
    /// assignment and clobbers `rcx`/`r11` as the kernel does.
    #[allow(unsafe_code)]
    unsafe fn syscall6(
        n: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        a5: usize,
        a6: usize,
    ) -> isize {
        let ret: isize;
        // SAFETY: a single `syscall` with the documented register inputs/clobbers.
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                in("r8") a5,
                in("r9") a6,
                out("rcx") _,
                out("r11") _,
                options(nostack, preserves_flags),
            );
        }
        ret
    }

    /// An executable code region of its own pages. Maps RW, copies the code,
    /// then flips to RX (W^X); unmaps on drop.
    ///
    /// The code does not start at the page boundary but at a per-buffer
    /// offset that rotates through the page in cache-line steps ("cache
    /// colouring"): every body starting at offset 0 would put all their entry
    /// paths (and their hot loops, at similar offsets) into the same few
    /// instruction-cache sets, which evict each other. The code is position
    /// independent (it reaches anything outside itself by absolute address).
    pub(super) struct ExecBuffer {
        base: *mut u8,
        ptr: *mut u8,
        len: usize,
    }

    /// The colour of the next buffer (see [`ExecBuffer`]).
    static NEXT_COLOUR: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

    /// Cache lines between successive buffers' start offsets (odd, so the
    /// offsets cycle through every line of a page).
    const COLOUR_STEP_LINES: usize = 13;

    impl ExecBuffer {
        /// Maps `code` into a fresh executable region, or `None` on failure.
        pub(super) fn new(code: &[u8]) -> Option<Self> {
            if code.is_empty() {
                return None;
            }
            let page = 4096;
            let colour = NEXT_COLOUR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            let offset = (colour * COLOUR_STEP_LINES * 64) % page;
            let len = (offset + code.len()).div_ceil(page) * page;
            // mmap(NULL, len, PROT_RW, MAP_PRIVATE|ANON, -1, 0)
            // SAFETY: a standard anonymous mmap; the kernel returns a new mapping
            // or a small negative errno.
            #[allow(unsafe_code)]
            let raw = unsafe {
                syscall6(
                    SYS_MMAP,
                    0,
                    len,
                    PROT_RW,
                    MAP_PRIVATE_ANON,
                    usize::MAX, // fd = -1
                    0,
                )
            };
            // mmap returns the address, or -errno in [-4095, -1].
            if (-4095..0).contains(&raw) {
                return None;
            }
            let base = raw as *mut u8;
            // SAFETY: `offset < page <= len`, inside the mapping.
            #[allow(unsafe_code)]
            let ptr = unsafe { base.add(offset) };
            // SAFETY: `ptr` points to `len - offset` freshly-mapped writable
            // bytes and `code.len() <= len - offset`; the regions do not overlap.
            #[allow(unsafe_code)]
            unsafe {
                core::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
            }
            // mprotect(base, len, PROT_RX): drop write, gain execute (W^X).
            // SAFETY: `base`/`len` describe the mapping just created.
            #[allow(unsafe_code)]
            let prot = unsafe { syscall6(SYS_MPROTECT, base as usize, len, PROT_RX, 0, 0, 0) };
            if prot < 0 {
                // Best-effort unmap before bailing.
                // SAFETY: unmapping the mapping we own.
                #[allow(unsafe_code)]
                unsafe {
                    syscall6(SYS_MUNMAP, base as usize, len, 0, 0, 0, 0);
                }
                return None;
            }
            Some(Self { base, ptr, len })
        }

        /// The executable code pointer.
        #[must_use]
        pub(super) fn ptr(&self) -> *const u8 {
            self.ptr
        }
    }

    impl Drop for ExecBuffer {
        fn drop(&mut self) {
            // SAFETY: unmapping exactly the region this buffer owns.
            #[allow(unsafe_code)]
            unsafe {
                syscall6(SYS_MUNMAP, self.base as usize, self.len, 0, 0, 0, 0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn arith_op_eval_matches_native_when_available() {
        let ops = vec![
            ArithOp::AddImm(5),
            ArithOp::MulImm(3),
            ArithOp::SubImm(2),
            ArithOp::XorImm(0x0f),
            ArithOp::Neg,
        ];
        let interp = eval_arith(&ops, 7);
        if let Some(f) = JitFunction::compile_arith(&ops) {
            assert_eq!(f.call1(7), interp, "JIT must match the interpreter");
        } else {
            assert!(!available(), "compile only fails when JIT is unavailable");
        }
    }

    /// A cold region moves after the rest of the code, its labels and jumps
    /// relocated, and the hot jump over it is dropped.
    #[test]
    fn assembler_cold_region_moves_out_of_line() {
        let mut a = X64Assembler::new();
        let cold = a.new_label();
        let after = a.new_label();
        a.je(cold); // 0..6
        a.jmp(after); // dropped by `bind(after)` once the region is out
        let m = a.cold_begin();
        a.bind(cold);
        a.zero_rax();
        a.jmp(after);
        a.cold_end(m);
        a.bind(after);
        a.ret(); // 6
        let code = a.finish();
        // [je cold][ret] then the region [xor rax, rax][jmp after].
        assert_eq!(code.len(), 6 + 1 + 3 + 5);
        assert_eq!(&code[0..2], &[0x0f, 0x84]);
        assert_eq!(i32::from_le_bytes(code[2..6].try_into().unwrap()), 1);
        assert_eq!(code[6], 0xc3);
        assert_eq!(&code[7..10], &[0x48, 0x31, 0xc0]);
        assert_eq!(code[10], 0xe9);
        assert_eq!(i32::from_le_bytes(code[11..15].try_into().unwrap()), 6 - 15);
    }

    #[test]
    fn assembler_emits_expected_bytes() {
        let mut a = X64Assembler::new();
        a.mov_rax_rdi();
        a.add_rax_rsi();
        a.ret();
        assert_eq!(a.code(), &[0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3]);
    }

    #[test]
    fn jit_arithmetic_runs_natively() {
        if !available() {
            return;
        }
        // f(x) = ((x + 10) * 2) - 3
        let ops = [ArithOp::AddImm(10), ArithOp::MulImm(2), ArithOp::SubImm(3)];
        let f = JitFunction::compile_arith(&ops).expect("jit available");
        for x in [-5i64, 0, 1, 100, 1_000_000] {
            assert_eq!(f.call1(x), (x + 10) * 2 - 3);
            assert_eq!(f.call1(x), eval_arith(&ops, x));
        }
    }

    #[test]
    fn jit_binary_ops_run_natively() {
        if !available() {
            return;
        }
        let add = JitFunction::compile_binary(BinOp::Add).unwrap();
        let sub = JitFunction::compile_binary(BinOp::Sub).unwrap();
        let mul = JitFunction::compile_binary(BinOp::Mul).unwrap();
        assert_eq!(add.call2(20, 22), 42);
        assert_eq!(sub.call2(50, 8), 42);
        assert_eq!(mul.call2(6, 7), 42);
        assert_eq!(mul.call2(-3, 4), -12);
    }

    #[test]
    fn stack_machine_compiles_and_runs() {
        // (a + b) * (a - 3)
        let prog = [
            StackOp::Arg(0),
            StackOp::Arg(1),
            StackOp::Add,
            StackOp::Arg(0),
            StackOp::Const(3),
            StackOp::Sub,
            StackOp::Mul,
        ];
        let oracle = |a: i64, b: i64| (a + b) * (a - 3);
        for (a, b) in [(7, 2), (10, -4), (0, 0), (-5, 5), (1000, 1)] {
            assert_eq!(eval_stack(&prog, [a, b]), oracle(a, b));
            if let Some(f) = JitFunction::compile_stack(&prog) {
                assert_eq!(f.call2(a, b), oracle(a, b), "jit stack ({a},{b})");
            }
        }
    }

    #[test]
    fn stack_machine_rejects_malformed() {
        assert!(
            JitFunction::compile_stack(&[StackOp::Add]).is_none(),
            "underflow"
        );
        assert!(
            JitFunction::compile_stack(&[StackOp::Arg(0), StackOp::Arg(1)]).is_none(),
            "two results"
        );
        assert!(JitFunction::compile_stack(&[]).is_none(), "empty");
    }

    #[test]
    fn register_machine_compiles_and_runs() {
        // r0=arg0, r1=arg1, r2=arg2; r3 = (r0 + r1) * r2 - r0 ; ret r3
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Arg { dst: 1, index: 1 },
            RegOp::Arg { dst: 2, index: 2 },
            RegOp::Bin {
                dst: 3,
                a: 0,
                b: 1,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 3,
                a: 3,
                b: 2,
                op: BinOp2::Mul,
            },
            RegOp::Bin {
                dst: 3,
                a: 3,
                b: 0,
                op: BinOp2::Sub,
            },
            RegOp::Ret { src: 3 },
        ];
        let oracle = |a: i64, b: i64, c: i64| (a + b) * c - a;
        for (a, b, c) in [
            (2, 3, 4),
            (10, -5, 2),
            (0, 0, 9),
            (-7, 7, -1),
            (100, 1, 1000),
        ] {
            assert_eq!(eval_reg(&ops, 4, &[a, b, c]), oracle(a, b, c));
            if let Some(f) = JitFunction::compile_reg(4, 3, &ops) {
                assert_eq!(
                    f.call_args(&[a, b, c]),
                    oracle(a, b, c),
                    "jit reg ({a},{b},{c})"
                );
            }
        }
    }

    #[test]
    fn register_machine_uses_constants_and_many_regs() {
        // A wider program exercising the spill-everything frame (>2 live regs).
        // r0=arg0; r1=100; r2=r0*r1; r3=7; r4=r2|r3; r5=r4^r0; ret r5
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Const { dst: 1, imm: 100 },
            RegOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::Mul,
            },
            RegOp::Const { dst: 3, imm: 7 },
            RegOp::Bin {
                dst: 4,
                a: 2,
                b: 3,
                op: BinOp2::Or,
            },
            RegOp::Bin {
                dst: 5,
                a: 4,
                b: 0,
                op: BinOp2::Xor,
            },
            RegOp::Ret { src: 5 },
        ];
        let oracle = |a: i64| ((a * 100) | 7) ^ a;
        for a in [0i64, 1, 5, 42, -3, 12345] {
            assert_eq!(eval_reg(&ops, 6, &[a]), oracle(a));
            if let Some(f) = JitFunction::compile_reg(6, 1, &ops) {
                assert_eq!(f.call_args(&[a]), oracle(a), "jit reg const ({a})");
            }
        }
    }

    #[test]
    fn lowers_real_nbvm_constant_function() {
        // A real program compiled by the VM's compiler; `1 + 2*3` folds to a
        // single integer `LoadConst` + `Return` (n_params = 0).
        let program = crate::parser::Parser::parse_program("1 + 2 * 3").expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let lowered = lower_nbvm(&protos[0]).expect("top-level should lower");
        assert_eq!(eval_reg(&lowered, protos[0].n_regs, &[]), 7);
        if let Some(f) = JitFunction::compile_reg(protos[0].n_regs, 0, &lowered) {
            assert_eq!(f.call_args(&[]), 7, "JIT runs real compiled bytecode");
        }
    }

    #[test]
    fn lowers_and_jits_a_real_arithmetic_function() {
        // Compile a real integer arithmetic function and JIT one of its protos.
        // (A product needs a positive constant factor to lower: see `pos_const`.)
        let src = "function f(a, b) { return a * 3 + a - b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        // Find the proto that lowers and takes two params (the body of `f`).
        let mut tested = false;
        for p in &protos {
            if p.n_params != 2 {
                continue;
            }
            if let Some(lowered) = lower_nbvm(p) {
                let oracle = |a: i64, b: i64| a * 3 + a - b;
                for (a, b) in [(2, 3), (10, -4), (0, 7), (-5, -5), (123, 2)] {
                    let via_ir = eval_reg(&lowered, p.n_regs, &[a, b]);
                    assert_eq!(via_ir, oracle(a, b), "lowered IR matches semantics");
                    if let Some(f) = JitFunction::compile_reg(p.n_regs, 2, &lowered) {
                        assert_eq!(f.call_args(&[a, b]), oracle(a, b), "JIT matches ({a},{b})");
                    }
                }
                tested = true;
            }
        }
        assert!(tested, "expected an integer arithmetic proto to lower");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_deopts_on_overflow_and_range() {
        use crate::nanbox::{NanBox, Unpacked};
        // f(a,b) = a * 2^30 + b. Within ±2^53 it runs natively; beyond it must
        // deopt (i64 and f64 would diverge), returning None so the interpreter
        // takes over.
        let src = "function f(a, b) { return a * 1073741824 + b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let p = protos.iter().find(|p| p.n_params == 2).unwrap();
        let jit = JitProto::compile(p).expect("f compiles");

        // In range → native result.
        let r = jit
            .call_guarded(&[NanBox::number(1000.0), NanBox::number(5.0)])
            .unwrap();
        assert_eq!(r.unpack(), Unpacked::Number(1_073_741_824_005.0));

        // 2^30 * 2^30 = 2^60 > 2^53 → deopt (None), NOT a wrong wrapped answer.
        let big = (1i64 << 30) as f64;
        assert!(
            jit.call_guarded(&[NanBox::number(big), NanBox::number(0.0)])
                .is_none(),
            "a product beyond 2^53 must deopt, not return a wrapped i64"
        );
        // A result that overflows i64 entirely also deopts.
        let huge = 9_000_000_000_000.0; // 9e12 * 2^30 ≈ 9.7e21 > i64::MAX
        assert!(
            jit.call_guarded(&[NanBox::number(huge), NanBox::number(0.0)])
                .is_none(),
            "i64-overflowing product must deopt"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_compiles_a_real_loop() {
        use crate::nanbox::{NanBox, Unpacked};
        // A real counted loop with a comparison and a backward branch.
        let src = "function f(n){ let s = 0; for (let i = 0; i < n; i = i + 1) { s = s + i; } return s; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let p = protos.iter().find(|p| p.n_params == 1).expect("f's proto");
        let lowered = lower_nbvm(p).expect("loop should lower (Lt/JumpIfFalse/Jump)");
        // The IR oracle computes sum(0..n).
        for n in [0i64, 1, 5, 10, 50] {
            assert_eq!(
                eval_reg(&lowered, p.n_regs, &[n]),
                n * (n - 1) / 2,
                "sum 0..{n}"
            );
        }
        // And the JIT runs it natively, end-to-end with the NanBox guard.
        let jit = JitProto::compile(p).expect("loop JIT-compiles");
        for n in [0i64, 1, 5, 10, 50, 100] {
            let r = jit
                .call_guarded(&[NanBox::number(n as f64)])
                .expect("native loop");
            assert_eq!(
                r.unpack(),
                Unpacked::Number((n * (n - 1) / 2) as f64),
                "jit sum 0..{n}"
            );
        }
    }

    #[test]
    fn float_jit_compiles_a_loop() {
        // A float loop with a fractional step and an f64 comparison:
        // sum of x for x in 0, 0.5, 1.0, ... while x < n.
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },           // 0: n
            FloatOp::Const { dst: 1, imm: 0.0 },         // 1: s = 0
            FloatOp::Const { dst: 2, imm: 0.0 },         // 2: x = 0
            FloatOp::Const { dst: 3, imm: 0.5 },         // 3: step = 0.5
            FloatOp::Lt { dst: 4, a: 2, b: 0 },          // 4: cond = x < n
            FloatOp::JumpIfFalse { cond: 4, target: 9 }, // 5: if !cond goto 9 (ret)
            FloatOp::Bin {
                dst: 1,
                a: 1,
                b: 2,
                op: FBinOp::Add,
            }, // 6: s += x
            FloatOp::Bin {
                dst: 2,
                a: 2,
                b: 3,
                op: FBinOp::Add,
            }, // 7: x += 0.5
            FloatOp::Jump { target: 4 },                 // 8: goto 4
            FloatOp::Ret { src: 1 },                     // 9: ret s
        ];
        let oracle = |n: f64| {
            let (mut s, mut x) = (0.0, 0.0);
            while x < n {
                s += x;
                x += 0.5;
            }
            s
        };
        for n in [0.0, 1.0, 3.0, 5.0, 10.0] {
            assert_eq!(eval_float(&ops, 5, &[n]), oracle(n));
            if let Some(f) = JitFunction::compile_float(5, 1, &ops) {
                assert_eq!(f.call_args_f64(&[n]), oracle(n), "float loop n={n}");
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_jitproto_runs_division_function() {
        use crate::nanbox::{NanBox, Unpacked};
        // A function using `/` and non-integer values takes the float path.
        let src = "function f(a, b) { return (a + b) * a / b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let p = protos.iter().find(|p| p.n_params == 2).unwrap();
        // It does NOT lower to the integer path (Div), but DOES to the float path.
        assert!(
            lower_nbvm(p).is_none(),
            "division shouldn't take the integer path"
        );
        assert!(lower_nbvm_float(p).is_some(), "should take the float path");
        let jit = JitProto::compile(p).expect("float JIT");

        let oracle = |a: f64, b: f64| (a + b) * a / b;
        for (a, b) in [(1.5, 2.5), (10.0, 4.0), (-3.0, 0.5), (7.0, 7.0)] {
            let r = jit
                .call_guarded(&[NanBox::number(a), NanBox::number(b)])
                .expect("number args run natively");
            match r.unpack() {
                Unpacked::Number(v) => assert!((v - oracle(a, b)).abs() < 1e-12, "{v}"),
                _ => panic!("expected number"),
            }
        }
        // A non-number argument deopts.
        assert!(
            jit.call_guarded(&[NanBox::null(), NanBox::number(1.0)])
                .is_none()
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_proto_end_to_end_with_deopt_guard() {
        use crate::nanbox::{NanBox, Unpacked};
        let src = "function f(a, b) { return a * 7 + a - b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let p = protos.iter().find(|p| p.n_params == 2).expect("f's proto");
        let jit = JitProto::compile(p).expect("f should JIT-compile");

        // Integer args → native execution, reboxed: 6*7 + 6 - 7 = 41.
        let r = jit
            .call_guarded(&[NanBox::number(6.0), NanBox::number(7.0)])
            .expect("integer args run natively");
        assert_eq!(r.unpack(), Unpacked::Number(41.0));
        let r = jit
            .call_guarded(&[NanBox::number(-3.0), NanBox::number(4.0)])
            .unwrap();
        assert_eq!(r.unpack(), Unpacked::Number(-3.0 * 7.0 + -3.0 - 4.0));

        // A non-integer argument deopts (the guard fails) → None.
        assert!(
            jit.call_guarded(&[NanBox::number(1.5), NanBox::number(2.0)])
                .is_none(),
            "non-integer arg must deopt"
        );
        // A non-number argument deopts too.
        assert!(
            jit.call_guarded(&[NanBox::boolean(true), NanBox::number(2.0)])
                .is_none(),
            "boolean arg must deopt"
        );
        // Wrong arity deopts.
        assert!(
            jit.call_guarded(&[NanBox::number(1.0)]).is_none(),
            "arity mismatch deopts"
        );
    }

    #[test]
    fn lower_rejects_read_before_write() {
        use crate::nbvm::{FnProto, Op};
        // reg 2 is read by Mul but never written and is not a parameter — this
        // would read an uninitialized slot (e.g. `this`/a capture), so it must
        // not JIT-lower.
        let proto = FnProto {
            ops: alloc::vec![Op::Mul { dst: 1, a: 0, b: 2 }, Op::Return { src: 1 },],
            n_regs: 3,
            n_params: 1,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: 0,
            name: alloc::string::String::new(),
            legacy: false,
            class_ctor: false,
            derived: false,
            is_generator: false,
            source_span: None,
            source: None,
            tier: Default::default(),
        };
        assert!(
            lower_nbvm(&proto).is_none(),
            "read-before-write must not lower"
        );

        // The same shape but with reg 2 written first *does* lower.
        let ok = FnProto {
            ops: alloc::vec![
                Op::LoadConst {
                    dst: 2,
                    value: crate::nanbox::NanBox::number(3.0)
                },
                Op::Mul { dst: 1, a: 0, b: 2 },
                Op::Return { src: 1 },
            ],
            n_regs: 3,
            n_params: 1,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: 0,
            name: alloc::string::String::new(),
            legacy: false,
            class_ctor: false,
            derived: false,
            is_generator: false,
            source_span: None,
            source: None,
            tier: Default::default(),
        };
        assert!(lower_nbvm(&ok).is_some(), "written-then-read should lower");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn lower_nbvm_wires_a_static_call() {
        use crate::nbvm::{FnProto, Op};
        // Callee B(x) = x * 3, compiled to native code.
        let b = FnProto {
            ops: alloc::vec![
                Op::LoadConst {
                    dst: 1,
                    value: crate::nanbox::NanBox::number(3.0)
                },
                Op::Mul { dst: 2, a: 0, b: 1 },
                Op::Return { src: 2 },
            ],
            n_regs: 3,
            n_params: 1,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: 0,
            name: alloc::string::String::new(),
            legacy: false,
            class_ctor: false,
            derived: false,
            is_generator: false,
            source_span: None,
            source: None,
            tier: Default::default(),
        };
        let bjit = JitProto::compile(&b).expect("compile B");
        let b_ptr = bjit.code_ptr();
        assert_ne!(b_ptr, 0);
        let mut registry = alloc::collections::BTreeMap::new();
        registry.insert(7u32, b_ptr as u64); // B is function-table index 7

        // Caller A(x) = B(x) + x, with an Op::Call to func 7.
        let a = FnProto {
            ops: alloc::vec![
                Op::Call {
                    dst: 1,
                    func: 7,
                    args: alloc::vec![0]
                }, // r1 = B(r0)
                Op::Add { dst: 2, a: 1, b: 0 }, // r2 = r1 + r0
                Op::Return { src: 2 },
            ],
            n_regs: 3,
            n_params: 1,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: 0,
            name: alloc::string::String::new(),
            legacy: false,
            class_ctor: false,
            derived: false,
            is_generator: false,
            source_span: None,
            source: None,
            tier: Default::default(),
        };
        // Without the registry the call can't lower; with it, it does.
        assert!(lower_nbvm(&a).is_none(), "unregistered call must bail");
        let lowered = lower_nbvm_with(&a, &registry).expect("registered call lowers");
        assert!(
            lowered.iter().any(|o| matches!(o, RegOp::Call { .. })),
            "emits a Call op"
        );

        // Compile A with the registry and run it natively: A(x) = 3x + x = 4x.
        let ajit = JitProto::compile_with_registry(&a, &registry).expect("compile A");
        for x in [0i64, 1, 5, 12] {
            let r = ajit
                .call_guarded(&[crate::nanbox::NanBox::number(x as f64)])
                .expect("no deopt");
            assert_eq!(
                r.unpack(),
                crate::nanbox::Unpacked::Number((4 * x) as f64),
                "A({x})"
            );
        }
    }

    #[test]
    fn does_not_lower_non_integer_functions() {
        // A function with a call / property access must not be JIT-lowered.
        let src = "function g(a){ return Math.max(a, 1); } g;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        for p in &protos {
            if p.n_params == 1 {
                assert!(lower_nbvm(p).is_none(), "g uses a call, must not lower");
            }
        }
    }

    #[test]
    fn optimize_reg_folds_constants() {
        // r0=2, r1=3, r2=r0+r1 (=5), r3=r2*r0 (=10), ret r3 — all foldable.
        let ops = [
            RegOp::Const { dst: 0, imm: 2 },
            RegOp::Const { dst: 1, imm: 3 },
            RegOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 3,
                a: 2,
                b: 0,
                op: BinOp2::Mul,
            },
            RegOp::Ret { src: 3 },
        ];
        // Folding collapses everything to `r3 = 10`; DCE then drops the
        // now-unread r0/r1/r2 — five ops become `Const r3=10; Ret r3`.
        let opt = optimize_reg(&ops, 4);
        assert_eq!(opt.len(), 2, "folded + DCE'd to two ops: {opt:?}");
        assert!(matches!(opt[0], RegOp::Const { imm: 10, .. }));
        assert!(matches!(opt[1], RegOp::Ret { .. }));
        // Observationally identical.
        assert_eq!(eval_reg(&opt, 4, &[]), eval_reg(&ops, 4, &[]));
        assert_eq!(eval_reg(&opt, 4, &[]), 10);
    }

    #[test]
    fn register_allocator_reuses_slots() {
        // Each temporary is used once then dead, so all collapse to few slots:
        // t0=arg0; t1=t0+t0; t2=t1+t1; t3=t2+t2; ret t3  (uses regs 0,1,2,3).
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Bin {
                dst: 1,
                a: 0,
                b: 0,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 2,
                a: 1,
                b: 1,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 3,
                a: 2,
                b: 2,
                op: BinOp2::Add,
            },
            RegOp::Ret { src: 3 },
        ];
        let (alloc, n) = allocate_reg(&ops, 4);
        // The chain's intervals overlap pairwise (each value lives across the next
        // op that reads it), so allocation needs only 2 slots, not 4.
        assert!(n <= 2, "allocated to {n} slots: {alloc:?}");
        // Observationally identical: f(x) = ((x+x)+(x+x))+... = 8x.
        for x in [0i64, 1, 3, -5, 100] {
            assert_eq!(eval_reg(&alloc, n, &[x]), eval_reg(&ops, 4, &[x]));
            assert_eq!(eval_reg(&alloc, n, &[x]), 8 * x);
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn allocated_program_runs_natively() {
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Arg { dst: 1, index: 1 },
            RegOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 3,
                a: 2,
                b: 0,
                op: BinOp2::Mul,
            },
            RegOp::Ret { src: 3 },
        ];
        let (alloc, n) = allocate_reg(&ops, 4);
        let f = JitFunction::compile_reg(n, 2, &alloc).unwrap();
        let oracle = |a: i64, b: i64| (a + b) * a;
        for (a, b) in [(3, 4), (10, -2), (0, 5)] {
            assert_eq!(f.call_args(&[a, b]), oracle(a, b), "alloc native ({a},{b})");
        }
    }

    #[test]
    fn allocator_preserves_a_loop() {
        // The sum(0..n) loop must survive allocation (branch targets unchanged,
        // overlapping live ranges kept distinct).
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Const { dst: 1, imm: 0 },
            RegOp::Const { dst: 2, imm: 0 },
            RegOp::Const { dst: 3, imm: 1 },
            RegOp::Lt { dst: 4, a: 2, b: 0 },
            RegOp::JumpIfFalse { cond: 4, target: 9 },
            RegOp::Bin {
                dst: 1,
                a: 1,
                b: 2,
                op: BinOp2::Add,
            },
            RegOp::Bin {
                dst: 2,
                a: 2,
                b: 3,
                op: BinOp2::Add,
            },
            RegOp::Jump { target: 4 },
            RegOp::Ret { src: 1 },
        ];
        let (alloc, n) = allocate_reg(&ops, 5);
        for x in [0i64, 1, 5, 10, 50] {
            assert_eq!(
                eval_reg(&alloc, n, &[x]),
                x * (x - 1) / 2,
                "loop sum 0..{x}"
            );
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn bit32_matches_js_toint32_semantics() {
        // f(a,b) = a & b, with JS ToInt32 truncation.
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Arg { dst: 1, index: 1 },
            RegOp::Bit32 {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::And,
            },
            RegOp::Ret { src: 2 },
        ];
        let (alloc, n) = allocate_reg(&optimize_reg(&ops, 3), 3);
        let f = JitFunction::compile_reg(n, 2, &alloc).unwrap();
        // Including a value above 2^32 to exercise ToInt32 truncation:
        // (2^32 + 0xF0) & 0xFF  ==  0xF0  (the high bits are dropped).
        for (a, b) in [
            (0xFF, 0x0F),
            (0xF0F0, 0x0FF0),
            (1 << 32 | 0xF0, 0xFF),
            (-1, 0x1234),
        ] {
            let expect = i64::from((a as i32) & (b as i32));
            assert_eq!(f.call_args(&[a, b]), expect, "({a:#x} & {b:#x})");
            assert_eq!(eval_reg(&ops, 3, &[a, b]), expect);
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn mod_compiles_and_deopts_on_zero() {
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Arg { dst: 1, index: 1 },
            RegOp::Mod { dst: 2, a: 0, b: 1 },
            RegOp::Ret { src: 2 },
        ];
        let (alloc, n) = allocate_reg(&optimize_reg(&ops, 3), 3);
        let f = JitFunction::compile_reg(n, 2, &alloc).unwrap();
        // JS `%`: sign follows the dividend.
        for (a, b) in [(17, 5), (17, -5), (100, 10), (3, 7), (0, 4)] {
            let expect = a % b;
            assert_eq!(f.call_args(&[a, b]), expect, "{a} % {b}");
            assert_eq!(eval_reg(&ops, 3, &[a, b]), expect);
        }
        // A negative dividend deopts: its zero remainder would be JS `-0`.
        for (a, b) in [(-17, 5), (-4, 2), (-1, 1)] {
            assert_eq!(f.call_args(&[a, b]), i64::MAX, "{a} % {b} must deopt");
        }
        // Divide-by-zero deopts: the sentinel (out of ±2^53) signals "bail".
        assert_eq!(f.call_args(&[5, 0]), i64::MAX, "% 0 must deopt");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn shift32_matches_js_semantics() {
        let run = |op: ShiftOp, a: i64, b: i64| -> i64 {
            let ops = [
                RegOp::Arg { dst: 0, index: 0 },
                RegOp::Arg { dst: 1, index: 1 },
                RegOp::Shift32 {
                    dst: 2,
                    a: 0,
                    b: 1,
                    op,
                },
                RegOp::Ret { src: 2 },
            ];
            let (alloc, n) = allocate_reg(&optimize_reg(&ops, 3), 3);
            let f = JitFunction::compile_reg(n, 2, &alloc).unwrap();
            let got = f.call_args(&[a, b]);
            assert_eq!(got, eval_reg(&ops, 3, &[a, b]), "jit vs oracle");
            got
        };
        // << : 1 << 4 == 16; count masked to 5 bits: 1 << 33 == 1 << 1 == 2.
        assert_eq!(run(ShiftOp::Shl, 1, 4), 16);
        assert_eq!(run(ShiftOp::Shl, 1, 33), 2);
        // >> : arithmetic — -8 >> 1 == -4 (sign-propagating).
        assert_eq!(run(ShiftOp::Sar, -8, 1), -4);
        // >>> : logical — -1 >>> 0 == 0xFFFFFFFF (4294967295, an unsigned u32).
        assert_eq!(run(ShiftOp::Shr, -1, 0), 0xFFFF_FFFF);
        assert_eq!(run(ShiftOp::Shr, -8, 1), 0x7FFF_FFFC);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn eqz_compiles_and_runs() {
        // f(x) = !(x < 5)  ≡  (x < 5) then Eqz → (x >= 5).
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Const { dst: 1, imm: 5 },
            RegOp::Lt { dst: 2, a: 0, b: 1 },
            RegOp::Eqz { dst: 2, a: 2 },
            RegOp::Ret { src: 2 },
        ];
        let opt = optimize_reg(&ops, 3);
        let (alloc, n) = allocate_reg(&opt, 3);
        let f = JitFunction::compile_reg(n, 1, &alloc).unwrap();
        for x in [0i64, 4, 5, 6, 100, -3] {
            let expect = i64::from(x >= 5); // !(x < 5)
            assert_eq!(f.call1(x), expect, "!(x<5) at x={x}");
            assert_eq!(eval_reg(&ops, 3, &[x]), expect);
        }
    }

    #[test]
    fn strength_reduction_identities() {
        // For each identity, build  r1 = arg0;  r2 = <const or arg>;  r3 = r1 <op> r2;  ret r3
        // and check the optimized program matches the algebraic identity.
        type Case = (&'static [RegOp], fn(i64) -> i64);
        let cases: &[Case] = &[
            // x + 0 = x
            (
                &[
                    RegOp::Arg { dst: 0, index: 0 },
                    RegOp::Const { dst: 1, imm: 0 },
                    RegOp::Bin {
                        dst: 2,
                        a: 0,
                        b: 1,
                        op: BinOp2::Add,
                    },
                    RegOp::Ret { src: 2 },
                ],
                |x| x,
            ),
            // x * 1 = x
            (
                &[
                    RegOp::Arg { dst: 0, index: 0 },
                    RegOp::Const { dst: 1, imm: 1 },
                    RegOp::Bin {
                        dst: 2,
                        a: 0,
                        b: 1,
                        op: BinOp2::Mul,
                    },
                    RegOp::Ret { src: 2 },
                ],
                |x| x,
            ),
            // x * 0 = 0
            (
                &[
                    RegOp::Arg { dst: 0, index: 0 },
                    RegOp::Const { dst: 1, imm: 0 },
                    RegOp::Bin {
                        dst: 2,
                        a: 0,
                        b: 1,
                        op: BinOp2::Mul,
                    },
                    RegOp::Ret { src: 2 },
                ],
                |_x| 0,
            ),
            // x - x = 0
            (
                &[
                    RegOp::Arg { dst: 0, index: 0 },
                    RegOp::Bin {
                        dst: 2,
                        a: 0,
                        b: 0,
                        op: BinOp2::Sub,
                    },
                    RegOp::Ret { src: 2 },
                ],
                |_x| 0,
            ),
            // x ^ x = 0
            (
                &[
                    RegOp::Arg { dst: 0, index: 0 },
                    RegOp::Bin {
                        dst: 2,
                        a: 0,
                        b: 0,
                        op: BinOp2::Xor,
                    },
                    RegOp::Ret { src: 2 },
                ],
                |_x| 0,
            ),
        ];
        for (ops, oracle) in cases {
            let opt = optimize_reg(ops, 3);
            for x in [0i64, 5, -7, 1234] {
                assert_eq!(eval_reg(&opt, 3, &[x]), oracle(x));
                assert_eq!(eval_reg(&opt, 3, &[x]), eval_reg(ops, 3, &[x]));
            }
        }
    }

    #[test]
    fn copy_propagation_forwards_moves() {
        // r0=arg0; r1=Move r0; r2=Move r1; r3=r2+r2; ret r3.
        // After copy-prop, r3 = r0 + r0 and the two Moves are dead → DCE drops them.
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Move { dst: 1, src: 0 },
            RegOp::Move { dst: 2, src: 1 },
            RegOp::Bin {
                dst: 3,
                a: 2,
                b: 2,
                op: BinOp2::Add,
            },
            RegOp::Ret { src: 3 },
        ];
        let prop = copy_propagate(&ops, 4);
        // The Bin now reads r0 directly (the chain collapsed in one step).
        assert!(
            matches!(prop[3], RegOp::Bin { a: 0, b: 0, .. }),
            "operands forwarded to the root: {prop:?}"
        );
        let opt = optimize_reg(&ops, 4);
        // No Move survives (both became dead and were eliminated).
        assert!(
            !opt.iter().any(|o| matches!(o, RegOp::Move { .. })),
            "moves eliminated: {opt:?}"
        );
        for x in [0i64, 3, -7, 1000] {
            assert_eq!(eval_reg(&opt, 4, &[x]), eval_reg(&ops, 4, &[x]));
            assert_eq!(eval_reg(&opt, 4, &[x]), x + x);
        }
    }

    #[test]
    fn copy_propagation_invalidates_on_overwrite() {
        // r1=Move r0; r0=Const 99; ret r1  — r1 must still be the OLD r0 (arg),
        // not 99, because r0 was overwritten after the copy.
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Move { dst: 1, src: 0 },
            RegOp::Const { dst: 0, imm: 99 },
            RegOp::Ret { src: 1 },
        ];
        let opt = optimize_reg(&ops, 2);
        for x in [5i64, -3, 42] {
            assert_eq!(
                eval_reg(&opt, 2, &[x]),
                x,
                "r1 keeps the pre-overwrite value"
            );
        }
    }

    #[test]
    fn dce_removes_dead_ops_and_remaps_branches() {
        // r0=arg0; r9=999 (dead, never read); loop: cond=r0<r0(false immediately
        // here we just test target remap) ... keep it simple: a dead const before
        // a branch, ensure the jump target still lands correctly after removal.
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },           // 0
            RegOp::Const { dst: 9, imm: 999 },         // 1 dead (r9 never read)
            RegOp::Const { dst: 1, imm: 0 },           // 2
            RegOp::Lt { dst: 2, a: 1, b: 0 },          // 3
            RegOp::JumpIfFalse { cond: 2, target: 6 }, // 4 -> op 6 (Ret)
            RegOp::Jump { target: 6 },                 // 5
            RegOp::Ret { src: 1 },                     // 6
        ];
        let opt = dce_reg(&ops);
        // The dead Const(r9) is gone; nothing else (all other dsts are read).
        assert!(
            !opt.iter().any(|o| matches!(o, RegOp::Const { dst: 9, .. })),
            "dead const removed: {opt:?}"
        );
        // Branch targets still resolve to the Ret, and the program is equivalent.
        for x in [0i64, 5, -3] {
            assert_eq!(eval_reg(&opt, 10, &[x]), eval_reg(&ops, 10, &[x]));
        }
    }

    #[test]
    fn optimize_reg_preserves_arg_dependent_and_branches() {
        // r0=arg0; r1=10; r2=r0*r1 (NOT foldable — depends on arg); ret r2.
        let ops = [
            RegOp::Arg { dst: 0, index: 0 },
            RegOp::Const { dst: 1, imm: 10 },
            RegOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::Mul,
            },
            RegOp::Ret { src: 2 },
        ];
        let opt = optimize_reg(&ops, 3);
        // The arg-dependent Bin stays a Bin.
        assert!(
            matches!(opt[2], RegOp::Bin { .. }),
            "arg-dependent op not folded"
        );
        for x in [0i64, 3, -7, 1000] {
            assert_eq!(eval_reg(&opt, 3, &[x]), eval_reg(&ops, 3, &[x]));
            assert_eq!(eval_reg(&opt, 3, &[x]), x * 10);
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn optimized_program_runs_natively() {
        let ops = [
            RegOp::Const { dst: 0, imm: 6 },
            RegOp::Const { dst: 1, imm: 7 },
            RegOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: BinOp2::Mul,
            },
            RegOp::Ret { src: 2 },
        ];
        let opt = optimize_reg(&ops, 3);
        let f = JitFunction::compile_reg(3, 0, &opt).unwrap();
        assert_eq!(f.call_args(&[]), 42, "folded 6*7 runs natively");
    }

    #[test]
    fn register_machine_rejects_malformed() {
        // Register index out of range.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Ret { src: 5 }]).is_none());
        // No Ret.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Const { dst: 0, imm: 1 }]).is_none());
        // Arg index beyond declared args.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Arg { dst: 0, index: 3 }]).is_none());
    }

    #[test]
    fn float_jit_compiles_and_runs_with_division() {
        // f(a, b) = (a + b) * a / b  — exercises SSE add/mul/div with non-integer
        // values, which the integer path can't handle.
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Arg { dst: 1, index: 1 },
            FloatOp::Bin {
                dst: 2,
                a: 0,
                b: 1,
                op: FBinOp::Add,
            },
            FloatOp::Bin {
                dst: 2,
                a: 2,
                b: 0,
                op: FBinOp::Mul,
            },
            FloatOp::Bin {
                dst: 2,
                a: 2,
                b: 1,
                op: FBinOp::Div,
            },
            FloatOp::Ret { src: 2 },
        ];
        let oracle = |a: f64, b: f64| (a + b) * a / b;
        for (a, b) in [(1.5, 2.5), (10.0, 4.0), (-3.5, 0.5), (7.0, 7.0), (0.1, 0.3)] {
            assert!((eval_float(&ops, 3, &[a, b]) - oracle(a, b)).abs() < 1e-12);
            if let Some(f) = JitFunction::compile_float(3, 2, &ops) {
                let got = f.call_args_f64(&[a, b]);
                assert!(
                    (got - oracle(a, b)).abs() < 1e-12,
                    "jit f64 ({a},{b}): {got}"
                );
            }
        }
    }

    #[test]
    fn float_jit_modulo() {
        // f(a, b) = a % b — float remainder via `a - trunc(a/b)*b`.
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Arg { dst: 1, index: 1 },
            FloatOp::Mod { dst: 0, a: 0, b: 1 },
            FloatOp::Ret { src: 0 },
        ];
        let oracle = |a: f64, b: f64| a % b; // Rust f64 `%` is the same IEEE remainder
        for (a, b) in [
            (10.0, 3.0),
            (10.5, 3.0),
            (-10.5, 3.0),
            (10.5, -3.0),
            (7.0, 7.0),
            (1.0, 0.25),
            (5.5, 2.2),
        ] {
            let want = oracle(a, b);
            assert!(
                (eval_float(&ops, 2, &[a, b]) - want).abs() < 1e-12,
                "oracle ({a} % {b})"
            );
            // The compiled path only exists on SSE4.1 hardware (roundsd); when it
            // does, it must agree with the oracle.
            if let Some(f) = JitFunction::compile_float(2, 2, &ops) {
                let got = f.call_args_f64(&[a, b]);
                assert!(
                    (got - want).abs() < 1e-12,
                    "jit ({a} % {b}): {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn float_jit_constants() {
        // f(x) = x * 0.5 + 1.25
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Const { dst: 1, imm: 0.5 },
            FloatOp::Const { dst: 2, imm: 1.25 },
            FloatOp::Bin {
                dst: 0,
                a: 0,
                b: 1,
                op: FBinOp::Mul,
            },
            FloatOp::Bin {
                dst: 0,
                a: 0,
                b: 2,
                op: FBinOp::Add,
            },
            FloatOp::Ret { src: 0 },
        ];
        for x in [4.0, -2.0, 0.0, 100.5] {
            let expect = x * 0.5 + 1.25;
            assert_eq!(eval_float(&ops, 3, &[x]), expect);
            if let Some(f) = JitFunction::compile_float(3, 1, &ops) {
                assert_eq!(f.call_args_f64(&[x]), expect, "const f64 ({x})");
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_code_calls_another_jit_function() {
        // Callee B(x) = x * 2 (a compiled arithmetic function).
        let b = JitFunction::compile_arith(&[ArithOp::MulImm(2)]).expect("compile B");
        let b_ptr = b.code_ptr();
        assert_ne!(b_ptr, 0);

        // Caller A(x) = B(x) + 1, hand-assembled to *call B's native code*:
        //   sub rsp, 8        ; align rsp to 16 before the call (System V)
        //   movabs rax, B     ; B's entry address
        //   call rax          ; B(rdi) -> rax   (rdi = A's arg, untouched)
        //   add rsp, 8
        //   add rax, 1
        //   ret
        let mut a = X64Assembler::new();
        a.sub_rsp_imm8(8);
        a.movabs_rax(b_ptr as i64);
        a.call_rax();
        a.add_rsp_imm8(8);
        a.add_rax_imm(1);
        a.ret();
        let af = JitFunction::from_machine_code(&a.finish()).expect("compile A");

        // A(x) = 2*x + 1, computed by A natively calling B natively.
        for x in [0i64, 1, 5, -3, 1000] {
            assert_eq!(af.call1(x), 2 * x + 1, "A({x}) via native call to B");
        }
        // B itself still works (its code wasn't disturbed).
        assert_eq!(b.call1(21), 42);
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_sqrt_and_abs_native() {
        // f(x) = sqrt(|x|).
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Abs { dst: 1, a: 0 },
            FloatOp::Sqrt { dst: 2, a: 1 },
            FloatOp::Ret { src: 2 },
        ];
        let f = JitFunction::compile_float(3, 1, &ops).unwrap();
        for x in [4.0f64, -9.0, 2.0, 0.0, -0.0, 1e6, 0.25] {
            let expect = x.abs().sqrt();
            let got = f.call_args_f64(&[x]);
            assert!(
                (got - expect).abs() < 1e-12 || (got.is_nan() && expect.is_nan()),
                "sqrt(|{x}|): got {got}, want {expect}"
            );
            assert_eq!(eval_float(&ops, 3, &[x]), expect);
        }
        // sqrt of a negative is NaN (no abs).
        let neg = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Sqrt { dst: 1, a: 0 },
            FloatOp::Ret { src: 1 },
        ];
        let g = JitFunction::compile_float(2, 1, &neg).unwrap();
        assert!(g.call_args_f64(&[-1.0]).is_nan(), "sqrt(-1) is NaN");
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_floor_ceil_native() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return; // roundsd unavailable; the lowering bails and this is moot
        }
        let floor = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Floor { dst: 1, a: 0 },
            FloatOp::Ret { src: 1 },
        ];
        let ceil = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Ceil { dst: 1, a: 0 },
            FloatOp::Ret { src: 1 },
        ];
        let ff = JitFunction::compile_float(2, 1, &floor).unwrap();
        let fc = JitFunction::compile_float(2, 1, &ceil).unwrap();
        for x in [3.7f64, -3.2, 2.0, -0.4, 1e6, -7.999] {
            assert_eq!(
                ff.call_args_f64(&[x]).to_bits(),
                x.floor().to_bits(),
                "floor({x})"
            );
            assert_eq!(
                fc.call_args_f64(&[x]).to_bits(),
                x.ceil().to_bits(),
                "ceil({x})"
            );
            assert_eq!(eval_float(&floor, 2, &[x]), x.floor());
            assert_eq!(eval_float(&ceil, 2, &[x]), x.ceil());
        }
        // `Math.trunc` — roundsd toward zero (0x0b). Preserves the sign of zero.
        let trunc = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Trunc { dst: 1, a: 0 },
            FloatOp::Ret { src: 1 },
        ];
        let ft = JitFunction::compile_float(2, 1, &trunc).unwrap();
        for x in [3.7f64, -3.7, 3.2, -0.5, 2.0, -0.0, 1e6, -7.999] {
            assert_eq!(
                ft.call_args_f64(&[x]).to_bits(),
                x.trunc().to_bits(),
                "trunc({x})"
            );
            assert_eq!(eval_float(&trunc, 2, &[x]).to_bits(), x.trunc().to_bits());
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_min_max_js_semantics() {
        let max = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Arg { dst: 1, index: 1 },
            FloatOp::Max { dst: 2, a: 0, b: 1 },
            FloatOp::Ret { src: 2 },
        ];
        let min = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Arg { dst: 1, index: 1 },
            FloatOp::Min { dst: 2, a: 0, b: 1 },
            FloatOp::Ret { src: 2 },
        ];
        let fmax = JitFunction::compile_float(3, 2, &max).unwrap();
        let fmin = JitFunction::compile_float(3, 2, &min).unwrap();
        let cases = [
            (3.0f64, 5.0),
            (5.0, 3.0),
            (-2.0, -7.0),
            (1.5, 1.5),
            (0.0, -0.0),
            (-0.0, 0.0),
            (f64::INFINITY, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
        ];
        for (x, y) in cases {
            // Native results match the oracle bit-for-bit (so ±0 and NaN agree).
            assert_eq!(
                fmax.call_args_f64(&[x, y]).to_bits(),
                eval_float(&max, 3, &[x, y]).to_bits(),
                "max({x}, {y})"
            );
            assert_eq!(
                fmin.call_args_f64(&[x, y]).to_bits(),
                eval_float(&min, 3, &[x, y]).to_bits(),
                "min({x}, {y})"
            );
        }
        // Spot-check the ±0 and NaN corners explicitly.
        assert_eq!(
            fmax.call_args_f64(&[0.0, -0.0]).to_bits(),
            0.0f64.to_bits(),
            "max(+0,-0)=+0"
        );
        assert_eq!(
            fmin.call_args_f64(&[0.0, -0.0]).to_bits(),
            (-0.0f64).to_bits(),
            "min(+0,-0)=-0"
        );
        assert!(
            fmax.call_args_f64(&[f64::NAN, 1.0]).is_nan(),
            "max(NaN,1)=NaN"
        );
        assert!(
            fmin.call_args_f64(&[1.0, f64::NAN]).is_nan(),
            "min(1,NaN)=NaN"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_eq_is_nan_aware() {
        // f(a,b) = (a === b) ? 1.0 : 0.0.
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Arg { dst: 1, index: 1 },
            FloatOp::Eq { dst: 2, a: 0, b: 1 },
            FloatOp::Ret { src: 2 },
        ];
        let f = JitFunction::compile_float(3, 2, &ops).unwrap();
        let cases = [
            (1.5, 1.5, 1.0),
            (1.5, 2.5, 0.0),
            (0.0, -0.0, 1.0),          // +0 === -0
            (f64::NAN, f64::NAN, 0.0), // NaN !== NaN
            (f64::NAN, 1.0, 0.0),
            (f64::INFINITY, f64::INFINITY, 1.0),
        ];
        for (a, b, expect) in cases {
            assert_eq!(f.call_args_f64(&[a, b]), expect, "{a} === {b}");
            assert_eq!(eval_float(&ops, 3, &[a, b]), expect);
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_eqz_handles_zero_and_nan() {
        // f(x) = !x  (1.0 for falsy x: ±0.0 or NaN; else 0.0).
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Eqz { dst: 1, a: 0 },
            FloatOp::Ret { src: 1 },
        ];
        let f = JitFunction::compile_float(2, 1, &ops).unwrap();
        for x in [0.0f64, -0.0, 1.5, -3.0, f64::NAN, f64::INFINITY] {
            let expect = if x == 0.0 || x.is_nan() { 1.0 } else { 0.0 };
            assert_eq!(f.call_args_f64(&[x]), expect, "!{x}");
            assert_eq!(eval_float(&ops, 2, &[x]), expect);
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_neg_compiles_and_runs() {
        // f(x) = -x + 1.5  (negation on the float path).
        let ops = [
            FloatOp::Arg { dst: 0, index: 0 },
            FloatOp::Neg { dst: 1, a: 0 },
            FloatOp::Const { dst: 2, imm: 1.5 },
            FloatOp::Bin {
                dst: 3,
                a: 1,
                b: 2,
                op: FBinOp::Add,
            },
            FloatOp::Ret { src: 3 },
        ];
        let f = JitFunction::compile_float(4, 1, &ops).unwrap();
        for x in [0.0f64, 2.5, -3.25, 100.0] {
            let expect = -x + 1.5;
            assert_eq!(f.call_args_f64(&[x]), expect, "-{x} + 1.5");
            assert_eq!(eval_float(&ops, 4, &[x]), expect);
        }
    }

    #[test]
    fn native_loop_sum() {
        if !available() {
            return;
        }
        let f = JitFunction::compile_sum_1_to_n().expect("jit available");
        for n in [0i64, 1, 2, 5, 10, 100, 1000] {
            assert_eq!(f.call1(n), n * (n + 1) / 2, "sum 1..={n}");
            assert_eq!(f.call1(n), eval_sum_1_to_n(n));
        }
        // A negative argument yields 0 (the loop never runs).
        assert_eq!(f.call1(-5), 0);
    }

    #[test]
    fn label_backpatch_forward_and_backward() {
        // A forward jump (skip) and a backward jump (loop) resolve to correct
        // rel32 offsets.
        let mut a = X64Assembler::new();
        let back = a.new_label();
        let fwd = a.new_label();
        a.bind(back);
        a.zero_rax();
        a.jmp(fwd); // forward
        a.add_rax_imm(99); // skipped
        a.bind(fwd);
        a.je(back); // backward operand is negative
        a.ret();
        let code = a.finish();
        // The forward jmp at offset 4 (E9 at 3) targets the `je` site; its rel32
        // must be non-negative; the backward `je` rel32 must be negative.
        // jmp E9 is at index 3, operand at 4..8.
        let jmp_rel = i32::from_le_bytes([code[4], code[5], code[6], code[7]]);
        assert!(jmp_rel >= 0, "forward jump is non-negative");
    }

    #[test]
    fn shifts_and_bitwise() {
        let ops = [ArithOp::ShlImm(4), ArithOp::OrImm(1), ArithOp::SarImm(1)];
        let interp = eval_arith(&ops, 3);
        assert_eq!(interp, ((3i64 << 4) | 1) >> 1);
        if let Some(f) = JitFunction::compile_arith(&ops) {
            assert_eq!(f.call1(3), interp);
        }
    }
}
