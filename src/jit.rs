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
                    (Some(x), Some(y)) if y != 0 => {
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
/// value — the JIT integer fast path only applies to such constants.
#[cfg(feature = "alloc")]
fn nanbox_int(v: crate::nanbox::NanBox) -> Option<i64> {
    match v.unpack() {
        crate::nanbox::Unpacked::Number(n) if n.is_finite() => {
            let i = n as i64;
            // Lossless round-trip and within the exact-integer range (±2^53).
            if (i as f64) == n
                && (-9.007_199_254_740_992e15..=9.007_199_254_740_992e15).contains(&n)
            {
                Some(i)
            } else {
                None
            }
        }
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
    if proto.n_regs > 64 || proto.n_params > 6 || proto.n_captures != 0 {
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
            Op::Mul { dst, a, b } => {
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
            // JS unary minus of an integer: `-x`. The native code guards i64::MIN
            // overflow and the exact-integer range, so it never diverges.
            Op::Neg { dst, a } => {
                let a = read(&written, *a)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                RegOp::Neg { dst: d, a }
            }
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
/// straight-line). Returns `None` otherwise, with the same def-use safety check.
#[cfg(feature = "alloc")]
#[must_use]
pub fn lower_nbvm_float(proto: &crate::nbvm::FnProto) -> Option<Vec<FloatOp>> {
    use crate::nbvm::Op;
    if proto.n_regs > 64 || proto.n_params > 4 || proto.n_captures != 0 {
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
            // Float unary minus `-x` (`0.0 - x`).
            Op::Neg { dst, a } => {
                let a = read(&written, *a)?;
                let d = reg8(*dst)?;
                written[*dst as usize] = true;
                FloatOp::Neg { dst: d, a }
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
}

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
            });
        }
        let ops = lower_nbvm_float(proto)?;
        let func = JitFunction::compile_float(proto.n_regs, proto.n_params, &ops)?;
        Some(Self {
            func,
            n_params: proto.n_params,
            kind: JitKind::Float,
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
                // Guard: every argument must be an exact integer.
                let mut ints = [0i64; 6];
                for (slot, a) in ints.iter_mut().zip(args.iter()) {
                    *slot = nanbox_int(*a)?;
                }
                let r = self.func.call_args(&ints[..self.n_params]);
                // A result outside ±2^53 is the overflow/range deopt sentinel.
                if (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&r) {
                    Some(crate::nanbox::NanBox::number(r as f64))
                } else {
                    None
                }
            }
            JitKind::Float => {
                // Guard: every argument must be a (finite) JS number.
                let mut fs = [0.0f64; 4];
                for (slot, a) in fs.iter_mut().zip(args.iter()) {
                    *slot = match a.unpack() {
                        crate::nanbox::Unpacked::Number(n) => n,
                        _ => return None,
                    };
                }
                let r = self.func.call_args_f64(&fs[..self.n_params]);
                Some(crate::nanbox::NanBox::number(r))
            }
        }
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
}

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

    /// Allocates a fresh, unbound label.
    pub fn new_label(&mut self) -> Label {
        self.labels.push(usize::MAX);
        Label(self.labels.len() - 1)
    }

    /// Binds `label` to the current emission point.
    pub fn bind(&mut self, label: Label) {
        self.labels[label.0] = self.code.len();
    }

    /// Resolves every recorded `rel32` jump fixup; call once after emission.
    /// Returns the finished machine code.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
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
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov rax, [rbp+disp]`.
    pub fn load_rax(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov [rbp+disp], rax`.
    pub fn store_rax(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x89, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `mov rcx, [rbp+disp]`.
    pub fn load_rcx(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x48, 0x8b, 0x8d]);
        self.code.extend_from_slice(&disp.to_le_bytes());
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
        self.code.extend_from_slice(&disp.to_le_bytes());
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
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    // --- SSE2 scalar-double (f64) addressing, for the float compiler. `xmm0` is
    // the accumulator; f64 args arrive in `xmm0..xmm3`; the result returns in
    // `xmm0`. ModRM `0x85` selects `[rbp + disp32]` with reg field `xmm0`. ---

    /// `movsd xmm0, [rbp+disp]`.
    pub fn movsd_xmm0_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `movsd [rbp+disp], xmm0`.
    pub fn movsd_mem_xmm0(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
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
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `ucomisd xmm0, [rbp+disp]` — unordered compare, set EFLAGS like `cmp`.
    pub fn ucomisd_xmm0_mem(&mut self, disp: i32) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x2e, 0x85]);
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `xorpd xmm1, xmm1` — set `xmm1` to `+0.0`.
    pub fn zero_xmm1(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x57, 0xc9]);
    }
    /// `xorpd xmm0, xmm0` — set `xmm0` to `+0.0`.
    pub fn zero_xmm0(&mut self) {
        self.code.extend_from_slice(&[0x66, 0x0f, 0x57, 0xc0]);
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
        self.code.extend_from_slice(&disp.to_le_bytes());
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
        self.code.extend_from_slice(&[0xff, 0xd0]);
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
        self.code.extend_from_slice(&disp.to_le_bytes());
    }
    /// `sub rsp, imm8`.
    pub fn sub_rsp_imm8(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, imm]);
    }
    /// `add rsp, imm8`.
    pub fn add_rsp_imm8(&mut self, imm: u8) {
        self.code.extend_from_slice(&[0x48, 0x83, 0xc4, imm]);
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

    /// Wraps already-emitted machine `code` into a callable region (the entry
    /// point for hand-assembled functions, e.g. one that calls another).
    #[must_use]
    pub fn from_machine_code(code: &[u8]) -> Option<Self> {
        Self::from_code(code)
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

    /// A page-aligned, executable code region. Maps RW, copies the code, then
    /// flips to RX (W^X); unmaps on drop.
    pub(super) struct ExecBuffer {
        ptr: *mut u8,
        len: usize,
    }

    impl ExecBuffer {
        /// Maps `code` into a fresh executable region, or `None` on failure.
        pub(super) fn new(code: &[u8]) -> Option<Self> {
            if code.is_empty() {
                return None;
            }
            let page = 4096;
            let len = code.len().div_ceil(page) * page;
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
            let ptr = raw as *mut u8;
            // SAFETY: `ptr` points to `len` freshly-mapped writable bytes and
            // `code.len() <= len`; the regions do not overlap.
            #[allow(unsafe_code)]
            unsafe {
                core::ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
            }
            // mprotect(ptr, len, PROT_RX): drop write, gain execute (W^X).
            // SAFETY: `ptr`/`len` describe the mapping just created.
            #[allow(unsafe_code)]
            let prot = unsafe { syscall6(SYS_MPROTECT, ptr as usize, len, PROT_RX, 0, 0, 0) };
            if prot < 0 {
                // Best-effort unmap before bailing.
                // SAFETY: unmapping the mapping we own.
                #[allow(unsafe_code)]
                unsafe {
                    syscall6(SYS_MUNMAP, ptr as usize, len, 0, 0, 0, 0);
                }
                return None;
            }
            Some(Self { ptr, len })
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
                syscall6(SYS_MUNMAP, self.ptr as usize, self.len, 0, 0, 0, 0);
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
        let src = "function f(a, b) { return a * b + a - b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        // Find the proto that lowers and takes two params (the body of `f`).
        let mut tested = false;
        for p in &protos {
            if p.n_params != 2 {
                continue;
            }
            if let Some(lowered) = lower_nbvm(p) {
                let oracle = |a: i64, b: i64| a * b + a - b;
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
        // f(a,b) = a * b. Within ±2^53 it runs natively; beyond it must deopt
        // (i64 and f64 would diverge), returning None so the interpreter takes over.
        let src = "function f(a, b) { return a * b; } f;";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let protos = crate::nbvm::compile_program(&program).expect("compile");
        let p = protos.iter().find(|p| p.n_params == 2).unwrap();
        let jit = JitProto::compile(p).expect("f compiles");

        // In range → native result.
        let r = jit
            .call_guarded(&[NanBox::number(1000.0), NanBox::number(1000.0)])
            .unwrap();
        assert_eq!(r.unpack(), Unpacked::Number(1_000_000.0));

        // 2^30 * 2^30 = 2^60 > 2^53 → deopt (None), NOT a wrong wrapped answer.
        let big = (1i64 << 30) as f64;
        assert!(
            jit.call_guarded(&[NanBox::number(big), NanBox::number(big)])
                .is_none(),
            "a product beyond 2^53 must deopt, not return a wrapped i64"
        );
        // A result that overflows i64 entirely also deopts.
        let huge = 3_000_000_000.0; // 3e9; 3e9 * 3e9 = 9e18 ~ i64 overflow
        assert!(
            jit.call_guarded(&[NanBox::number(huge), NanBox::number(huge)])
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
        let src = "function f(a, b) { return a * b + a - b; } f;";
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
        assert_eq!(r.unpack(), Unpacked::Number(-3.0 * 4.0 + -3.0 - 4.0));

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
        for (a, b) in [(17, 5), (-17, 5), (17, -5), (100, 10), (3, 7)] {
            let expect = a % b;
            assert_eq!(f.call_args(&[a, b]), expect, "{a} % {b}");
            assert_eq!(eval_reg(&ops, 3, &[a, b]), expect);
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
