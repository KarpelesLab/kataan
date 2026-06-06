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
