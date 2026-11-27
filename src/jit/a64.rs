//! An AArch64 (A64) machine-code emitter: the JIT's second backend.
//!
//! # Porting strategy
//!
//! The JIT's tiers lower bytecode to small IRs ([`RegOp`](super::RegOp) for
//! the Int tier, [`FloatOp`](super::FloatOp) for the Float tier,
//! `GenericOp` for the generic NanBox tier) and then *emit* machine code per
//! IR op. The x86-64 emitters drive [`X64Assembler`](super::X64Assembler)
//! with register-explicit, x86-shaped methods (`mov_rax_rdi`, `cmp_rax_r11`,
//! raw ModRM bytes in many places), so they cannot be retargeted by swapping
//! the assembler underneath. The AArch64 port is therefore a **per-arch
//! lowering of the same IRs**: the analyses, lowerings and runtime helpers
//! (everything above "emit an op") are shared, and each tier gets an A64
//! emitter next to its x86 one, written from the A64 instruction set (Arm
//! Architecture Reference Manual, A64 base + FP instructions), not translated
//! from the x86 code. This file is only the instruction encoder those
//! emitters use: plain register-numbered methods, labels with late-bound
//! branch fixups, and nothing tier-specific.
//!
//! The tiers come up one at a time, and an op an A64 emitter does not lower
//! yet makes the *whole function* ineligible for that tier, so it simply
//! stays in the interpreter (or a lower tier): a partial port is always
//! correct, only slower.
//!
//! State of the port: the Int and Float tiers are complete
//! (`a64_numeric`). The generic tier starts as a *baseline* body compiled
//! straight from the bytecode (`a64_generic`): the same frame ABI, entries,
//! polls, handlers and suspensions as the x86-64 body, with branches,
//! moves, constants and the Number fast paths of arithmetic and compares
//! inline, and every other op handed to the interpreter on the shared
//! register file (`jit_helper_exec_op`). Its hot ops have inline fast paths
//! (property reads through the inline caches, globals, dense array reads,
//! `.length`, captured variables, unary/bitwise/`%` on Numbers) and the
//! rest go to the x86-64 tier's own runtime helpers; calls between compiled
//! bodies enter through a native-call entry. It is on by default
//! (`KATAAN_JIT_A64GEN=0` turns the aarch64 JIT off).
//!
//! # Encoding notes
//!
//! Every instruction is one little-endian 32-bit word. Register number 31 is
//! `sp` or `xzr`/`wzr` depending on the instruction ([`SP`], [`ZR`]). A
//! branch to a label is emitted with a zero offset and patched by
//! [`A64Assembler::finish`]; a target out of the branch's reach (±128 MiB
//! for `b`/`bl`, ±1 MiB for `b.cond`/`cbz`, ±32 KiB for `tbz`) makes
//! `finish` return `None`, and the caller falls back to the interpreter
//! rather than emit wrong code. The encodings are checked against the GNU
//! assembler by `a64_encoding_tests` (a table generated offline).

// The encoder's surface is consumed tier by tier; parts only a later tier
// uses are dead on the way there (and on hosts whose JIT is x86-64, where
// only the encoding tests drive it).
#![allow(dead_code)]

use super::Label;
use alloc::vec::Vec;

/// A general-purpose register number (`x0`..`x30`; 31 is [`SP`] or [`ZR`]).
pub(crate) type Reg = u8;
/// A floating-point/SIMD register number (`d0`..`d31`).
pub(crate) type FReg = u8;

/// Register 31 read as the stack pointer (address bases, `add`/`sub` imm).
pub(crate) const SP: Reg = 31;
/// Register 31 read as the zero register (data-processing operands).
pub(crate) const ZR: Reg = 31;
/// The frame pointer (`x29`).
pub(crate) const FP: Reg = 29;
/// The link register (`x30`).
pub(crate) const LR: Reg = 30;
/// The first intra-procedure-call scratch register (`x16`, `ip0`): free to
/// clobber anywhere outside a call sequence; the encoder itself uses it to
/// materialize an out-of-range memory displacement.
pub(crate) const IP0: Reg = 16;
/// The second intra-procedure-call scratch register (`x17`, `ip1`).
pub(crate) const IP1: Reg = 17;

/// An A64 condition code (the `cond` field of `b.cond`/`csel`/...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Cond {
    /// Equal (`Z`).
    Eq = 0,
    /// Not equal (`!Z`).
    Ne = 1,
    /// Unsigned higher or same / carry set.
    Hs = 2,
    /// Unsigned lower / carry clear.
    Lo = 3,
    /// Negative (`N`); after `fcmp`: ordered less than.
    Mi = 4,
    /// Positive or zero.
    Pl = 5,
    /// Overflow (`V`); after `fcmp`: unordered.
    Vs = 6,
    /// No overflow.
    Vc = 7,
    /// Unsigned higher.
    Hi = 8,
    /// Unsigned lower or same.
    Ls = 9,
    /// Signed greater or equal.
    Ge = 10,
    /// Signed less than.
    Lt = 11,
    /// Signed greater than.
    Gt = 12,
    /// Signed less or equal.
    Le = 13,
    /// Always.
    Al = 14,
}

impl Cond {
    /// The opposite condition (`Al` has none and stays `Al`).
    #[must_use]
    pub(crate) fn invert(self) -> Cond {
        use Cond::*;
        match self {
            Eq => Ne,
            Ne => Eq,
            Hs => Lo,
            Lo => Hs,
            Mi => Pl,
            Pl => Mi,
            Vs => Vc,
            Vc => Vs,
            Hi => Ls,
            Ls => Hi,
            Ge => Lt,
            Lt => Ge,
            Gt => Le,
            Le => Gt,
            Al => Al,
        }
    }
}

/// A shift applied to the second register operand of a data-processing
/// instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shift {
    /// Logical shift left.
    Lsl = 0,
    /// Logical shift right.
    Lsr = 1,
    /// Arithmetic shift right.
    Asr = 2,
}

/// How a branch's offset field is laid out (see [`A64Assembler::finish`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fix {
    /// `b`/`bl`: 26-bit word offset at bit 0.
    Imm26,
    /// `b.cond`/`cbz`/`cbnz`/`ldr` (literal): 19-bit word offset at bit 5.
    Imm19,
    /// `tbz`/`tbnz`: 14-bit word offset at bit 5.
    Imm14,
}

/// The AArch64 machine-code emitter (AAPCS64 calling convention for the
/// code that calls or is called by Rust).
#[derive(Default)]
pub(crate) struct A64Assembler {
    code: Vec<u8>,
    /// Per-label byte offset (`usize::MAX` until [`bind`](Self::bind)).
    labels: Vec<usize>,
    /// Pending branch fixups: `(instruction offset, target label, layout)`.
    fixups: Vec<(usize, usize, Fix)>,
    /// Set when an operand could not be encoded (an immediate out of
    /// range): [`finish`](Self::finish) then refuses the code.
    failed: bool,
}

impl A64Assembler {
    /// A new, empty assembler.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The bytes emitted so far.
    #[must_use]
    pub(crate) fn code(&self) -> &[u8] {
        &self.code
    }

    /// The current offset (where the next instruction goes).
    #[must_use]
    pub(crate) fn pos(&self) -> usize {
        self.code.len()
    }

    /// Marks the code unusable (an operand this encoder cannot express):
    /// [`finish`](Self::finish) returns `None`.
    pub(crate) fn fail(&mut self) {
        self.failed = true;
    }

    /// Emits one instruction word.
    pub(crate) fn emit(&mut self, word: u32) {
        self.code.extend_from_slice(&word.to_le_bytes());
    }

    /// A fresh, unbound label.
    pub(crate) fn new_label(&mut self) -> Label {
        self.labels.push(usize::MAX);
        Label(self.labels.len() - 1)
    }

    /// Binds `label` to the current offset.
    pub(crate) fn bind(&mut self, label: Label) {
        self.labels[label.0] = self.code.len();
    }

    /// Whether `label` has been bound.
    #[must_use]
    pub(crate) fn is_bound(&self, label: Label) -> bool {
        self.labels[label.0] != usize::MAX
    }

    /// Resolves every branch and returns the code, or `None` when an operand
    /// did not encode, a branch target was never bound, or one is out of its
    /// branch's reach.
    #[must_use]
    pub(crate) fn finish(mut self) -> Option<Vec<u8>> {
        if self.failed {
            return None;
        }
        for &(at, label, fix) in &self.fixups {
            let target = *self.labels.get(label)?;
            if target == usize::MAX {
                return None;
            }
            let delta = (target as i64 - at as i64) >> 2;
            let (bits, shift) = match fix {
                Fix::Imm26 => (26, 0),
                Fix::Imm19 => (19, 5),
                Fix::Imm14 => (14, 5),
            };
            let lim = 1i64 << (bits - 1);
            if !(-lim..lim).contains(&delta) {
                return None;
            }
            let mask = (1u32 << bits) - 1;
            let mut w = u32::from_le_bytes(self.code[at..at + 4].try_into().ok()?);
            w |= ((delta as u32) & mask) << shift;
            self.code[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        Some(self.code)
    }

    fn branch_to(&mut self, word: u32, label: Label, fix: Fix) {
        self.fixups.push((self.code.len(), label.0, fix));
        self.emit(word);
    }

    // ---------------------------------------------------------------- moves

    /// `mov xd, xm` (`orr xd, xzr, xm`).
    pub(crate) fn mov(&mut self, d: Reg, m: Reg) {
        self.emit(0xAA00_03E0 | rm(m) | rd(d));
    }

    /// `mov wd, wm`: the low 32 bits, zero-extended.
    pub(crate) fn mov_w(&mut self, d: Reg, m: Reg) {
        self.emit(0x2A00_03E0 | rm(m) | rd(d));
    }

    /// `mov xd, sp` / `mov sp, xn` (`add xd, xn, #0`).
    pub(crate) fn mov_sp(&mut self, d: Reg, n: Reg) {
        self.emit(0x9100_0000 | rn(n) | rd(d));
    }

    /// `movz xd, #imm16, lsl #(16*hw)`.
    pub(crate) fn movz(&mut self, d: Reg, imm16: u16, hw: u8) {
        self.emit(0xD280_0000 | (u32::from(hw & 3) << 21) | (u32::from(imm16) << 5) | rd(d));
    }

    /// `movk xd, #imm16, lsl #(16*hw)`.
    pub(crate) fn movk(&mut self, d: Reg, imm16: u16, hw: u8) {
        self.emit(0xF280_0000 | (u32::from(hw & 3) << 21) | (u32::from(imm16) << 5) | rd(d));
    }

    /// `movn xd, #imm16, lsl #(16*hw)`.
    pub(crate) fn movn(&mut self, d: Reg, imm16: u16, hw: u8) {
        self.emit(0x9280_0000 | (u32::from(hw & 3) << 21) | (u32::from(imm16) << 5) | rd(d));
    }

    /// Loads the 64-bit constant `v` into `xd` in the fewest `movz`/`movn` +
    /// `movk` instructions (one to four).
    pub(crate) fn mov_imm(&mut self, d: Reg, v: u64) {
        let half = |i: u8| ((v >> (16 * u32::from(i))) & 0xFFFF) as u16;
        let zeros = (0..4).filter(|&i| half(i) == 0).count();
        let ones = (0..4).filter(|&i| half(i) == 0xFFFF).count();
        if ones > zeros {
            // Start from all-ones: `movn` the first half that isn't 0xFFFF.
            let first = (0..4).find(|&i| half(i) != 0xFFFF).unwrap_or(0);
            self.movn(d, !half(first), first);
            for i in (first + 1)..4 {
                if half(i) != 0xFFFF {
                    self.movk(d, half(i), i);
                }
            }
        } else {
            let first = (0..4).find(|&i| half(i) != 0).unwrap_or(0);
            self.movz(d, half(first), first);
            for i in (first + 1)..4 {
                if half(i) != 0 {
                    self.movk(d, half(i), i);
                }
            }
        }
    }

    // ------------------------------------------------------ add/sub/compare

    /// `add xd, xn, xm`.
    pub(crate) fn add(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x8B00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `adds xd, xn, xm` (sets the flags; `V` on signed overflow).
    pub(crate) fn adds(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0xAB00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `sub xd, xn, xm`.
    pub(crate) fn sub(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0xCB00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `subs xd, xn, xm`.
    pub(crate) fn subs(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0xEB00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `add xd, xn, xm, uxtx` (the extended-register form, whose `xn` may be
    /// `sp`).
    pub(crate) fn add_ext(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x8B20_6000 | rm(m) | rn(n) | rd(d));
    }

    /// `add xd, xn, xm, <shift> #amount`.
    pub(crate) fn add_shifted(&mut self, d: Reg, n: Reg, m: Reg, shift: Shift, amount: u8) {
        self.emit(
            0x8B00_0000
                | ((shift as u32) << 22)
                | rm(m)
                | (u32::from(amount & 63) << 10)
                | rn(n)
                | rd(d),
        );
    }

    /// `cmp xn, xm`.
    pub(crate) fn cmp(&mut self, n: Reg, m: Reg) {
        self.subs(ZR, n, m);
    }

    /// `cmp xn, xm, <shift> #amount`.
    pub(crate) fn cmp_shifted(&mut self, n: Reg, m: Reg, shift: Shift, amount: u8) {
        self.emit(
            0xEB00_0000
                | ((shift as u32) << 22)
                | rm(m)
                | (u32::from(amount & 63) << 10)
                | rn(n)
                | rd(ZR),
        );
    }

    /// `cmp wn, wm` (32-bit compare).
    pub(crate) fn cmp_w(&mut self, n: Reg, m: Reg) {
        self.emit(0x6B00_0000 | rm(m) | rn(n) | rd(ZR));
    }

    /// `neg xd, xm`.
    pub(crate) fn neg(&mut self, d: Reg, m: Reg) {
        self.sub(d, ZR, m);
    }

    /// `negs xd, xm` (`V` set for `i64::MIN`).
    pub(crate) fn negs(&mut self, d: Reg, m: Reg) {
        self.subs(d, ZR, m);
    }

    /// Encodes an add/sub immediate (`imm12`, optionally `lsl #12`) or `None`.
    fn arith_imm(imm: u64) -> Option<u32> {
        if imm < 0x1000 {
            Some((imm as u32) << 10)
        } else if imm & 0xFFF == 0 && imm < 0x100_0000 {
            Some((1 << 22) | (((imm >> 12) as u32) << 10))
        } else {
            None
        }
    }

    /// The add/sub-immediate family; a negative `imm` flips to the other
    /// operation (`add` and `sub` differ in bit 30 of `base`).
    fn addsub_imm(&mut self, base: u32, d: Reg, n: Reg, imm: i64) {
        let (op, mag) = if imm < 0 {
            (base ^ (1 << 30), imm.unsigned_abs())
        } else {
            (base, imm as u64)
        };
        match Self::arith_imm(mag) {
            Some(f) => self.emit(op | f | rn(n) | rd(d)),
            None => self.fail(),
        }
    }

    /// `add xd, xn, #imm` (`sub` for a negative `imm`); `xn`/`xd` may be `sp`.
    /// Fails the code for an immediate that does not encode.
    pub(crate) fn add_imm(&mut self, d: Reg, n: Reg, imm: i64) {
        self.addsub_imm(0x9100_0000, d, n, imm);
    }

    /// `sub xd, xn, #imm` (`add` for a negative `imm`).
    pub(crate) fn sub_imm(&mut self, d: Reg, n: Reg, imm: i64) {
        self.addsub_imm(0xD100_0000, d, n, imm);
    }

    /// `adds xd, xn, #imm` (`subs` for a negative `imm`).
    pub(crate) fn adds_imm(&mut self, d: Reg, n: Reg, imm: i64) {
        self.addsub_imm(0xB100_0000, d, n, imm);
    }

    /// `cmp xn, #imm` (`cmn` for a negative `imm`; `xn` is not `sp`).
    pub(crate) fn cmp_imm(&mut self, n: Reg, imm: i64) {
        self.addsub_imm(0xF100_0000, ZR, n, imm);
    }

    // ------------------------------------------------- multiply and divide

    /// `mul xd, xn, xm` (`madd xd, xn, xm, xzr`).
    pub(crate) fn mul(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x9B00_7C00 | rm(m) | rn(n) | rd(d));
    }

    /// `smulh xd, xn, xm`: the high 64 bits of the signed 128-bit product.
    pub(crate) fn smulh(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x9B40_7C00 | rm(m) | rn(n) | rd(d));
    }

    /// `msub xd, xn, xm, xa`: `xa - xn * xm`.
    pub(crate) fn msub(&mut self, d: Reg, n: Reg, m: Reg, a: Reg) {
        self.emit(0x9B00_8000 | rm(m) | (u32::from(a) << 10) | rn(n) | rd(d));
    }

    /// `sdiv xd, xn, xm` (truncating; `x / 0` is 0, no trap).
    pub(crate) fn sdiv(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x9AC0_0C00 | rm(m) | rn(n) | rd(d));
    }

    // -------------------------------------------------------------- logical

    /// `and xd, xn, xm`.
    pub(crate) fn and(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x8A00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `orr xd, xn, xm`.
    pub(crate) fn orr(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0xAA00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `eor xd, xn, xm`.
    pub(crate) fn eor(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0xCA00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `tst xn, xm` (`ands xzr, xn, xm`).
    pub(crate) fn tst(&mut self, n: Reg, m: Reg) {
        self.emit(0xEA00_0000 | rm(m) | rn(n) | rd(ZR));
    }

    /// `and wd, wn, wm`.
    pub(crate) fn and_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x0A00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `orr wd, wn, wm`.
    pub(crate) fn orr_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x2A00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `eor wd, wn, wm`.
    pub(crate) fn eor_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x4A00_0000 | rm(m) | rn(n) | rd(d));
    }

    /// `mvn wd, wm` (`orn wd, wzr, wm`).
    pub(crate) fn mvn_w(&mut self, d: Reg, m: Reg) {
        self.emit(0x2A20_03E0 | rm(m) | rd(d));
    }

    /// `mvn xd, xm`.
    pub(crate) fn mvn(&mut self, d: Reg, m: Reg) {
        self.emit(0xAA20_03E0 | rm(m) | rd(d));
    }

    // --------------------------------------------------- shifts and extends

    /// `lslv wd, wn, wm` (count taken mod 32).
    pub(crate) fn lsl_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x1AC0_2000 | rm(m) | rn(n) | rd(d));
    }

    /// `lsrv wd, wn, wm` (count taken mod 32).
    pub(crate) fn lsr_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x1AC0_2400 | rm(m) | rn(n) | rd(d));
    }

    /// `asrv wd, wn, wm` (count taken mod 32).
    pub(crate) fn asr_w(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x1AC0_2800 | rm(m) | rn(n) | rd(d));
    }

    /// `lslv xd, xn, xm` (count taken mod 64).
    pub(crate) fn lsl(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x9AC0_2000 | rm(m) | rn(n) | rd(d));
    }

    /// `asrv xd, xn, xm` (count taken mod 64).
    pub(crate) fn asr(&mut self, d: Reg, n: Reg, m: Reg) {
        self.emit(0x9AC0_2800 | rm(m) | rn(n) | rd(d));
    }

    /// `lsl xd, xn, #s` (`ubfm xd, xn, #(-s mod 64), #(63-s)`).
    pub(crate) fn lsl_imm(&mut self, d: Reg, n: Reg, s: u8) {
        let s = u32::from(s & 63);
        let immr = (64 - s) & 63;
        self.emit(0xD340_0000 | (immr << 16) | ((63 - s) << 10) | rn(n) | rd(d));
    }

    /// `lsr xd, xn, #s` (`ubfm xd, xn, #s, #63`).
    pub(crate) fn lsr_imm(&mut self, d: Reg, n: Reg, s: u8) {
        self.emit(0xD340_FC00 | (u32::from(s & 63) << 16) | rn(n) | rd(d));
    }

    /// `asr xd, xn, #s` (`sbfm xd, xn, #s, #63`).
    pub(crate) fn asr_imm(&mut self, d: Reg, n: Reg, s: u8) {
        self.emit(0x9340_FC00 | (u32::from(s & 63) << 16) | rn(n) | rd(d));
    }

    /// `sxtw xd, wn`: sign-extends the low 32 bits.
    pub(crate) fn sxtw(&mut self, d: Reg, n: Reg) {
        self.emit(0x9340_7C00 | rn(n) | rd(d));
    }

    // -------------------------------------------------- conditional select

    /// `cset xd, cond` (`csinc xd, xzr, xzr, !cond`): 1 when `cond`, else 0.
    pub(crate) fn cset(&mut self, d: Reg, cond: Cond) {
        self.emit(0x9A9F_07E0 | ((cond.invert() as u32) << 12) | rd(d));
    }

    /// `csel xd, xn, xm, cond`.
    pub(crate) fn csel(&mut self, d: Reg, n: Reg, m: Reg, cond: Cond) {
        self.emit(0x9A80_0000 | rm(m) | ((cond as u32) << 12) | rn(n) | rd(d));
    }

    // ------------------------------------------------------ loads and stores

    /// A load/store of a `size`-byte unit at `[base + disp]`: the scaled
    /// unsigned-offset form (`scaled_op`) when `disp` is a non-negative
    /// multiple of `size` in reach, else the unscaled 9-bit form
    /// (`unscaled_op`), else the address computed into [`IP0`] first (then
    /// `base` must not be `IP0`).
    fn ldst(&mut self, scaled_op: u32, unscaled_op: u32, size: i64, t: u8, base: Reg, disp: i64) {
        if disp >= 0 && disp % size == 0 && disp / size < 4096 {
            self.emit(scaled_op | (((disp / size) as u32) << 10) | rn(base) | rd(t));
        } else if (-256..256).contains(&disp) {
            self.emit(unscaled_op | (((disp as u32) & 0x1FF) << 12) | rn(base) | rd(t));
        } else {
            if base == IP0 {
                self.fail();
                return;
            }
            self.mov_imm(IP0, disp as u64);
            self.add_ext(IP0, base, IP0);
            self.emit(scaled_op | rn(IP0) | rd(t));
        }
    }

    /// `ldr xt, [xn, #disp]` (any displacement; see `ldst`).
    pub(crate) fn ldr(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0xF940_0000, 0xF840_0000, 8, t, base, disp);
    }

    /// `str xt, [xn, #disp]`.
    pub(crate) fn str(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0xF900_0000, 0xF800_0000, 8, t, base, disp);
    }

    /// `ldr wt, [xn, #disp]` (zero-extending 32-bit load).
    pub(crate) fn ldr_w(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0xB940_0000, 0xB840_0000, 4, t, base, disp);
    }

    /// `str wt, [xn, #disp]`.
    pub(crate) fn str_w(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0xB900_0000, 0xB800_0000, 4, t, base, disp);
    }

    /// `ldrh wt, [xn, #disp]`.
    pub(crate) fn ldrh(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0x7940_0000, 0x7840_0000, 2, t, base, disp);
    }

    /// `ldrb wt, [xn, #disp]`.
    pub(crate) fn ldrb(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0x3940_0000, 0x3840_0000, 1, t, base, disp);
    }

    /// `strb wt, [xn, #disp]`.
    pub(crate) fn strb(&mut self, t: Reg, base: Reg, disp: i64) {
        self.ldst(0x3900_0000, 0x3800_0000, 1, t, base, disp);
    }

    /// `ldr dt, [xn, #disp]`.
    pub(crate) fn ldr_d(&mut self, t: FReg, base: Reg, disp: i64) {
        self.ldst(0xFD40_0000, 0xFC40_0000, 8, t, base, disp);
    }

    /// `str dt, [xn, #disp]`.
    pub(crate) fn str_d(&mut self, t: FReg, base: Reg, disp: i64) {
        self.ldst(0xFD00_0000, 0xFC00_0000, 8, t, base, disp);
    }

    /// `ldr xt, [xn, xm, lsl #3]`.
    pub(crate) fn ldr_idx(&mut self, t: Reg, base: Reg, index: Reg) {
        self.emit(0xF860_7800 | rm(index) | rn(base) | rd(t));
    }

    /// `str xt, [xn, xm, lsl #3]`.
    pub(crate) fn str_idx(&mut self, t: Reg, base: Reg, index: Reg) {
        self.emit(0xF820_7800 | rm(index) | rn(base) | rd(t));
    }

    /// `str xt, [sp, #-16]!`: pushes one register, keeping `sp` 16-aligned.
    pub(crate) fn push(&mut self, t: Reg) {
        self.emit(0xF800_0C00 | ((0x1F0u32) << 12) | rn(SP) | rd(t));
    }

    /// `ldr xt, [sp], #16`: pops what [`push`](Self::push) pushed.
    pub(crate) fn pop(&mut self, t: Reg) {
        self.emit(0xF840_0400 | (16u32 << 12) | rn(SP) | rd(t));
    }

    /// `stp xt1, xt2, [xn, #imm]!` (pre-index; `imm` a multiple of 8 in
    /// `-512..=504`).
    pub(crate) fn stp_pre(&mut self, t1: Reg, t2: Reg, base: Reg, imm: i32) {
        self.pair(0xA980_0000, t1, t2, base, imm);
    }

    /// `ldp xt1, xt2, [xn], #imm` (post-index).
    pub(crate) fn ldp_post(&mut self, t1: Reg, t2: Reg, base: Reg, imm: i32) {
        self.pair(0xA8C0_0000, t1, t2, base, imm);
    }

    /// `stp xt1, xt2, [xn, #imm]`.
    pub(crate) fn stp(&mut self, t1: Reg, t2: Reg, base: Reg, imm: i32) {
        self.pair(0xA900_0000, t1, t2, base, imm);
    }

    /// `ldp xt1, xt2, [xn, #imm]`.
    pub(crate) fn ldp(&mut self, t1: Reg, t2: Reg, base: Reg, imm: i32) {
        self.pair(0xA940_0000, t1, t2, base, imm);
    }

    fn pair(&mut self, op: u32, t1: Reg, t2: Reg, base: Reg, imm: i32) {
        if imm % 8 != 0 || !(-512..=504).contains(&imm) {
            self.fail();
            return;
        }
        let imm7 = ((imm / 8) as u32) & 0x7F;
        self.emit(op | (imm7 << 15) | (u32::from(t2) << 10) | rn(base) | rd(t1));
    }

    // ------------------------------------------------------------- branches

    /// `b label`.
    pub(crate) fn b(&mut self, label: Label) {
        self.branch_to(0x1400_0000, label, Fix::Imm26);
    }

    /// `bl label`.
    pub(crate) fn bl(&mut self, label: Label) {
        self.branch_to(0x9400_0000, label, Fix::Imm26);
    }

    /// `b.cond label`.
    pub(crate) fn b_cond(&mut self, cond: Cond, label: Label) {
        self.branch_to(0x5400_0000 | cond as u32, label, Fix::Imm19);
    }

    /// `cbz xt, label`.
    pub(crate) fn cbz(&mut self, t: Reg, label: Label) {
        self.branch_to(0xB400_0000 | rd(t), label, Fix::Imm19);
    }

    /// `cbnz xt, label`.
    pub(crate) fn cbnz(&mut self, t: Reg, label: Label) {
        self.branch_to(0xB500_0000 | rd(t), label, Fix::Imm19);
    }

    /// `tbz xt, #bit, label`.
    pub(crate) fn tbz(&mut self, t: Reg, bit: u8, label: Label) {
        self.branch_to(0x3600_0000 | tb_bit(bit) | rd(t), label, Fix::Imm14);
    }

    /// `tbnz xt, #bit, label`.
    pub(crate) fn tbnz(&mut self, t: Reg, bit: u8, label: Label) {
        self.branch_to(0x3700_0000 | tb_bit(bit) | rd(t), label, Fix::Imm14);
    }

    /// `br xn`.
    pub(crate) fn br(&mut self, n: Reg) {
        self.emit(0xD61F_0000 | rn(n));
    }

    /// `blr xn`.
    pub(crate) fn blr(&mut self, n: Reg) {
        self.emit(0xD63F_0000 | rn(n));
    }

    /// `ret` (to `x30`).
    pub(crate) fn ret(&mut self) {
        self.emit(0xD65F_03C0);
    }

    /// Calls the absolute address `target` (`mov x16, #target; blr x16`).
    pub(crate) fn call_abs(&mut self, target: u64) {
        self.mov_imm(IP0, target);
        self.blr(IP0);
    }

    /// `nop`.
    pub(crate) fn nop(&mut self) {
        self.emit(0xD503_201F);
    }

    /// `brk #imm` (a trap for code that must be unreachable).
    pub(crate) fn brk(&mut self, imm: u16) {
        self.emit(0xD420_0000 | (u32::from(imm) << 5));
    }

    // -------------------------------------------------------- floating point

    /// `fadd dd, dn, dm`.
    pub(crate) fn fadd(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_2800 | rm(m) | rn(n) | rd(d));
    }

    /// `fsub dd, dn, dm`.
    pub(crate) fn fsub(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_3800 | rm(m) | rn(n) | rd(d));
    }

    /// `fmul dd, dn, dm`.
    pub(crate) fn fmul(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_0800 | rm(m) | rn(n) | rd(d));
    }

    /// `fdiv dd, dn, dm`.
    pub(crate) fn fdiv(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_1800 | rm(m) | rn(n) | rd(d));
    }

    /// `fmax dd, dn, dm` (NaN-propagating; `+0` above `-0`).
    pub(crate) fn fmax(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_4800 | rm(m) | rn(n) | rd(d));
    }

    /// `fmin dd, dn, dm` (NaN-propagating; `-0` below `+0`).
    pub(crate) fn fmin(&mut self, d: FReg, n: FReg, m: FReg) {
        self.emit(0x1E60_5800 | rm(m) | rn(n) | rd(d));
    }

    /// `fsqrt dd, dn`.
    pub(crate) fn fsqrt(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E61_C000 | rn(n) | rd(d));
    }

    /// `fabs dd, dn`.
    pub(crate) fn fabs(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E60_C000 | rn(n) | rd(d));
    }

    /// `fneg dd, dn`.
    pub(crate) fn fneg(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E61_4000 | rn(n) | rd(d));
    }

    /// `frintm dd, dn` (round toward -infinity: floor).
    pub(crate) fn frintm(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E65_4000 | rn(n) | rd(d));
    }

    /// `frintp dd, dn` (round toward +infinity: ceil).
    pub(crate) fn frintp(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E64_C000 | rn(n) | rd(d));
    }

    /// `frintz dd, dn` (round toward zero: trunc).
    pub(crate) fn frintz(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E65_C000 | rn(n) | rd(d));
    }

    /// `fcmp dn, dm` (unordered sets `C` and `V`).
    pub(crate) fn fcmp(&mut self, n: FReg, m: FReg) {
        self.emit(0x1E60_2000 | rm(m) | rn(n));
    }

    /// `fcmp dn, #0.0`.
    pub(crate) fn fcmp_zero(&mut self, n: FReg) {
        self.emit(0x1E60_2008 | rn(n));
    }

    /// `fmov dd, dn`.
    pub(crate) fn fmov(&mut self, d: FReg, n: FReg) {
        self.emit(0x1E60_4000 | rn(n) | rd(d));
    }

    /// `fmov dd, xn` (the bits, unconverted).
    pub(crate) fn fmov_from_x(&mut self, d: FReg, n: Reg) {
        self.emit(0x9E67_0000 | rn(n) | rd(d));
    }

    /// `fmov xd, dn` (the bits, unconverted).
    pub(crate) fn fmov_to_x(&mut self, d: Reg, n: FReg) {
        self.emit(0x9E66_0000 | rn(n) | rd(d));
    }

    /// `scvtf dd, xn` (signed integer to double).
    pub(crate) fn scvtf(&mut self, d: FReg, n: Reg) {
        self.emit(0x9E62_0000 | rn(n) | rd(d));
    }

    /// `fcvtzs xd, dn` (double to signed integer, toward zero, saturating).
    pub(crate) fn fcvtzs(&mut self, d: Reg, n: FReg) {
        self.emit(0x9E78_0000 | rn(n) | rd(d));
    }
}

/// The `Rd`/`Rt` field (bits 0..5).
#[inline]
fn rd(r: u8) -> u32 {
    u32::from(r & 31)
}

/// The `Rn` field (bits 5..10).
#[inline]
fn rn(r: u8) -> u32 {
    u32::from(r & 31) << 5
}

/// The `Rm` field (bits 16..21).
#[inline]
fn rm(r: u8) -> u32 {
    u32::from(r & 31) << 16
}

/// `tbz`/`tbnz`'s bit number: `b5` at bit 31, `b40` at bits 19..24.
#[inline]
fn tb_bit(bit: u8) -> u32 {
    let bit = u32::from(bit & 63);
    ((bit >> 5) << 31) | ((bit & 31) << 19)
}

#[cfg(test)]
#[path = "a64_enc_tests.rs"]
mod a64_encoding_tests;
