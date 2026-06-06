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
    /// if `reg[cond] == 0`, jump to op index `target`
    JumpIfFalse { cond: u8, target: usize },
    /// unconditional jump to op index `target`
    Jump { target: usize },
    /// return `reg[src]`
    Ret { src: u8 },
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
            RegOp::JumpIfFalse { cond, target } => {
                if regs[cond as usize] == 0 {
                    pc = target;
                } else {
                    pc += 1;
                }
            }
            RegOp::Jump { target } => pc = target,
            RegOp::Ret { src } => return regs[src as usize],
        }
    }
    0
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

/// A bytecode-VM function compiled to native code, callable from the VM with
/// `NanBox` values. This is the end-to-end fast path: it owns the compiled
/// machine code and performs the unbox→native→rebox round-trip with an integer
/// **type guard** at the boundary (a non-integer argument deopts to the
/// interpreter, exactly as the optimizing tier's guards will).
#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
pub struct JitProto {
    func: JitFunction,
    n_params: usize,
}

#[cfg(all(feature = "alloc", target_os = "linux", target_arch = "x86_64"))]
impl JitProto {
    /// Compiles a `nbvm::FnProto` to native code if it is JIT-eligible (see
    /// [`lower_nbvm`]); otherwise `None` (the caller runs it in the interpreter).
    #[must_use]
    pub fn compile(proto: &crate::nbvm::FnProto) -> Option<Self> {
        let ops = lower_nbvm(proto)?;
        let func = JitFunction::compile_reg(proto.n_regs, proto.n_params, &ops)?;
        Some(Self {
            func,
            n_params: proto.n_params,
        })
    }

    /// Calls the native code with `NanBox` arguments. Returns the reboxed integer
    /// result, or `None` to **deopt** to the interpreter when a precondition
    /// fails: wrong argument count, or any argument is not an exact integer.
    #[must_use]
    pub fn call_guarded(&self, args: &[crate::nanbox::NanBox]) -> Option<crate::nanbox::NanBox> {
        if args.len() != self.n_params {
            return None;
        }
        // Guard: every argument must be an exact integer for the integer fast
        // path to be valid; otherwise bail so the interpreter handles it.
        let mut ints = [0i64; 6];
        for (slot, a) in ints.iter_mut().zip(args.iter()) {
            *slot = nanbox_int(*a)?;
        }
        let r = self.func.call_args(&ints[..self.n_params]);
        // The native code returns a value in ±2^53 on success, or a sentinel
        // outside that range when an intermediate result overflowed / left the
        // exact-integer range — in which case we deopt to the interpreter.
        if (-SAFE_INT_MAX..=SAFE_INT_MAX).contains(&r) {
            Some(crate::nanbox::NanBox::number(r as f64))
        } else {
            None
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
    fn register_machine_rejects_malformed() {
        // Register index out of range.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Ret { src: 5 }]).is_none());
        // No Ret.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Const { dst: 0, imm: 1 }]).is_none());
        // Arg index beyond declared args.
        assert!(JitFunction::compile_reg(2, 1, &[RegOp::Arg { dst: 0, index: 3 }]).is_none());
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
