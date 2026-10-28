//! The generic tier's machine-code call sequence (`ROADMAP.md` §2.1): a call
//! from one compiled body to another that stays in native code.
//!
//! A call site keeps a one-entry cache ([`CallSite`]) naming the function it
//! last called and that function's native entry. The emitted sequence asks a
//! leaf helper which VM function the callee value is ([`CalleeInfo`]: its id
//! and captures, read straight out of the heap cell), compares that with the
//! cache, checks the interpreter's limits, lays the callee's register window
//! out on the per-thread register stack ([`JitStack`]) and calls the callee's
//! native entry (see `emit_native_entry`), which fills its own parameter,
//! capture and `this` registers. Anything the sequence does not handle — a
//! different callee, a full stack, the depth limit, several realms — takes the
//! site's helper, which is the interpreter's call (and refills the cache).
//!
//! The interpreter state the ordinary call keeps is kept here too, more
//! cheaply:
//!
//! * the call depth (`call_depth`, the realm's `vm_total_depth`) is bumped and
//!   restored in place, and the limits are checked before the call (the limit
//!   itself is the helper's: it throws the catchable `RangeError`);
//! * the GC publication: the caller's window is on the register stack, which
//!   the safepoint scans from its base to its top, so publishing it is one
//!   increment of `frames_published`;
//! * the `f.caller` / `f.arguments` activation stack: instead of pushing onto
//!   the realm's `vm_callee_stack`, each native activation leaves a header on
//!   the register stack — the previous header, the callee, where the realm's
//!   stack stood, and the arguments — and a reader merges the two stacks only
//!   when it needs them (`jit_tier::callee_view`).

use super::{Label, X64Assembler};

/// x86-64 register numbers (the ModRM/REX encoding).
pub(crate) mod r {
    pub(crate) const RAX: u8 = 0;
    pub(crate) const RCX: u8 = 1;
    pub(crate) const RDX: u8 = 2;
    pub(crate) const RBX: u8 = 3;
    pub(crate) const RSI: u8 = 6;
    pub(crate) const RDI: u8 = 7;
    pub(crate) const R8: u8 = 8;
    pub(crate) const R9: u8 = 9;
    pub(crate) const R10: u8 = 10;
    pub(crate) const R11: u8 = 11;
    pub(crate) const R15: u8 = 15;
}

/// Condition codes for [`X64Assembler::g_jcc`] (the low nibble of `0F 8x`).
pub(crate) mod cc {
    /// Unsigned `<` (CF=1).
    pub(crate) const B: u8 = 0x2;
    /// Unsigned `>=` (CF=0).
    pub(crate) const AE: u8 = 0x3;
    /// Equal (ZF=1).
    pub(crate) const E: u8 = 0x4;
    /// Not equal (ZF=0).
    pub(crate) const NE: u8 = 0x5;
    /// Unsigned `<=`.
    pub(crate) const BE: u8 = 0x6;
    /// Unsigned `>`.
    pub(crate) const A: u8 = 0x7;
}

/// The per-thread native register stack. Every compiled activation's register
/// window lives here (contiguously, innermost last), so the GC can root all of
/// them by scanning `base..top`, and a native call can lay out its callee's
/// window by bumping `top`. `#[repr(C)]`: emitted code addresses the fields at
/// the `STACK_*` offsets.
#[repr(C)]
#[derive(Debug)]
pub struct JitStack {
    /// One past the innermost window (the next free word).
    pub top: *mut u64,
    /// One past the last usable word.
    pub limit: *mut u64,
    /// The innermost native-call header (null when there is none). A header is
    /// [`FRAME_HEADER`] words — the previous header, the callee value, the
    /// realm's `vm_callee_stack` length at the call, the argument count and
    /// `new.target` — followed by the arguments, then the callee's window.
    pub frames: *mut u64,
    /// The first word.
    pub base: *mut u64,
}

/// [`JitStack::top`]'s offset.
pub(crate) const STACK_TOP: i32 = 0;
/// [`JitStack::limit`]'s offset.
pub(crate) const STACK_LIMIT: i32 = 8;
/// [`JitStack::frames`]'s offset.
pub(crate) const STACK_FRAMES: i32 = 16;
/// [`JitStack::base`]'s offset.
pub(crate) const STACK_BASE: i32 = 24;
/// Words in a native-call header (see [`JitStack::frames`]).
pub const FRAME_HEADER: usize = 5;

/// One callee a call site has seen. `#[repr(C)]`: emitted code reads it at
/// the `SITE_*` offsets.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CallTarget {
    /// The callee's [`CalleeInfo::key`]: function id and capture count, or
    /// [`CallSite::EMPTY`] (which no key equals — not even the
    /// not-a-closure `u64::MAX`).
    pub key: u64,
    /// The callee's native entry address (see `emit_native_entry`).
    pub code: u64,
    /// The callee's register-window size in bytes.
    pub frame_bytes: u64,
}

/// A call site's cache: up to [`CallSite::WAYS`] callees (a polymorphic site —
/// `task.run()` over several task classes — keeps each of them), replaced
/// round-robin. Empty until the site's helper fills it.
#[repr(C)]
#[derive(Debug)]
pub struct CallSite {
    /// The cached callees.
    pub targets: [CallTarget; CallSite::WAYS],
    /// The way the next new callee replaces.
    pub next: usize,
}

impl CallSite {
    /// The key of an empty way.
    pub const EMPTY: u64 = u64::MAX - 1;
    /// Callees a site remembers.
    pub const WAYS: usize = 4;

    /// An empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            targets: [CallTarget {
                key: Self::EMPTY,
                code: 0,
                frame_bytes: 0,
            }; Self::WAYS],
            next: 0,
        }
    }

    /// Whether the site already holds `key`.
    #[must_use]
    pub fn holds(&self, key: u64) -> bool {
        self.targets.iter().any(|t| t.key == key)
    }

    /// Remembers `target` (a no-op when its key is already held).
    pub fn fill(&mut self, target: CallTarget) {
        if self.holds(target.key) {
            return;
        }
        let way = self.next % Self::WAYS;
        self.targets[way] = target;
        self.next = (way + 1) % Self::WAYS;
    }
}

impl Default for CallSite {
    fn default() -> Self {
        Self::new()
    }
}

/// [`CallTarget`]'s size.
const TARGET_BYTES: i32 = 24;
const SITE_KEY: i32 = 0;
const SITE_CODE: i32 = 8;
const SITE_FRAME: i32 = 16;

/// What [`CalleeHelper`] reports about a callee value: `key` is
/// `function id | capture count << 32` for a VM closure (`u64::MAX`
/// otherwise), and `caps` points at its captured values. `#[repr(C)]` with two
/// integer words, so it comes back in `rax:rdx`.
#[repr(C)]
pub struct CalleeInfo {
    /// `id | n_captures << 32`, or `u64::MAX` for anything but a VM closure.
    pub key: u64,
    /// The closure's captured values (valid until the next allocation).
    pub caps: *const u64,
}

/// The leaf helper the call sequence identifies its callee with: `(ctx,
/// callee) -> CalleeInfo`. Never allocates, never throws.
pub type CalleeHelper = extern "C" fn(*mut core::ffi::c_void, u64) -> CalleeInfo;

/// The leaf helper a method call finds its callee with: `(ctx, recv, key_ptr,
/// key_len, cache, out) -> CalleeInfo` — the method `recv.key` through the
/// site's prototype-chain cache `cache` when it is a VM closure the
/// interpreter's method call would enter directly (its value stored to
/// `*out`), else the not-a-closure key. Never allocates (beyond refilling the
/// cache), never throws, never runs user code.
pub type MethodHelper = extern "C" fn(
    *mut core::ffi::c_void,
    u64,
    *const u8,
    usize,
    *mut core::ffi::c_void,
    *mut u64,
) -> CalleeInfo;

/// The leaf helper a `new` finds its callee and `this` with: `(ctx, ctor,
/// out) -> CalleeInfo` — for a constructor whose `[[Construct]]` the sequence
/// may run (an ordinary, non-derived VM constructor with no instance fields),
/// the object `OrdinaryCreateFromConstructor` makes, stored to `*out`; else the
/// not-a-closure key, having done nothing. Never throws, never runs user code.
///
/// `site` is the call site's cache: a constructor other than the one it names
/// is reported without creating anything (the sequence then takes the slow
/// path, which constructs it itself).
pub type ConstructHelper =
    extern "C" fn(*mut core::ffi::c_void, u64, *mut u64, *const CallSite) -> CalleeInfo;

/// The helper that finishes a `new` whose body returned the heap value
/// `result`: `(ctx, result, this) -> result if it is an object, else this`.
pub type ConstructResultHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u64;

/// Where a call sequence's callee comes from.
pub(crate) enum CalleeSrc {
    /// A value already in the native-frame slot `[rbp + scratch]` (a computed
    /// method, `Op::CallKey`); identified by the [`CalleeHelper`].
    Frame {
        /// The `rbp`-relative slot.
        scratch: i32,
    },
    /// A constructor register (`Op::Construct`, `new.target` the constructor
    /// itself), prepared by the [`ConstructHelper`], which leaves the new
    /// object in `[rbp + scratch]` (the call's `this`).
    Construct {
        /// The constructor register.
        ctor: u16,
        /// The `rbp`-relative scratch slot.
        scratch: i32,
        /// The preparation helper.
        helper: u64,
        /// The [`ConstructResultHelper`].
        result: u64,
    },
    /// A register (`Op::CallValue`); identified by the [`CalleeHelper`].
    Reg(u16),
    /// A method lookup (`Op::CallMethod`) by the [`MethodHelper`], whose value
    /// lands in the native-frame slot at `[rbp + scratch]`.
    Method {
        /// The receiver register.
        recv: u16,
        /// The key's bytes.
        key_ptr: *const u8,
        /// The key's length.
        key_len: usize,
        /// The site's property cache.
        cache: *mut core::ffi::c_void,
        /// The `rbp`-relative scratch slot.
        scratch: i32,
        /// The lookup helper.
        helper: u64,
    },
}

/// Where the call sequence finds the interpreter state it maintains — byte
/// offsets computed by the VM with `offset_of!` (`ctx`-relative, or relative to
/// the `Realm` the context's `realm` box points at).
#[derive(Clone, Copy, Debug)]
pub struct CallLayout {
    /// `ctx`: the `*mut JitStack`.
    pub stack: i32,
    /// `ctx`: `call_depth: usize`.
    pub call_depth: i32,
    /// `ctx`: `frames_published: usize`.
    pub frames_published: i32,
    /// `ctx`: the `Box<Realm>` pointer.
    pub realm: i32,
    /// realm: `vm_total_depth: usize`.
    pub total_depth: i32,
    /// realm: `limits.max_call_depth: usize`.
    pub max_depth: i32,
    /// realm: `multi_realm: bool`.
    pub multi_realm: i32,
    /// realm: `vm_callee_stack`'s length word.
    pub callee_len: i32,
}

impl X64Assembler {
    /// `[REX] opcode ModRM(mod=10, reg, base) [SIB] disp32` — a register/memory
    /// operand form with a 32-bit displacement off any base register.
    fn g_mem(&mut self, w: bool, opcode: &[u8], reg: u8, base: u8, disp: i32) {
        let rex = 0x40 | (u8::from(w) << 3) | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        self.code.push(rex);
        self.code.extend_from_slice(opcode);
        self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
        if base & 7 == 4 {
            self.code.push(0x24); // SIB: base only (rsp / r12)
        }
        self.code.extend_from_slice(&disp.to_le_bytes());
    }

    /// `op rm, reg` in register-direct form (`mod=11`), 64-bit.
    fn g_rr(&mut self, opcode: u8, rm: u8, reg: u8) {
        self.code
            .push(0x48 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1));
        self.code.push(opcode);
        self.code.push(0xc0 | ((reg & 7) << 3) | (rm & 7));
    }

    /// `mov dst, [base+disp]`.
    pub(crate) fn g_load(&mut self, dst: u8, base: u8, disp: i32) {
        self.g_mem(true, &[0x8b], dst, base, disp);
    }

    /// `mov [base+disp], src`.
    pub(crate) fn g_store(&mut self, base: u8, disp: i32, src: u8) {
        self.g_mem(true, &[0x89], src, base, disp);
    }

    /// `cmp reg, [base+disp]`.
    pub(crate) fn g_cmp_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.g_mem(true, &[0x3b], reg, base, disp);
    }

    /// `add reg, [base+disp]`.
    pub(crate) fn g_add_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.g_mem(true, &[0x03], reg, base, disp);
    }

    /// `lea reg, [base+disp]`.
    pub(crate) fn g_lea(&mut self, reg: u8, base: u8, disp: i32) {
        self.g_mem(true, &[0x8d], reg, base, disp);
    }

    /// `inc qword [base+disp]`.
    pub(crate) fn g_inc_mem(&mut self, base: u8, disp: i32) {
        self.g_mem(true, &[0xff], 0, base, disp);
    }

    /// `dec qword [base+disp]`.
    pub(crate) fn g_dec_mem(&mut self, base: u8, disp: i32) {
        self.g_mem(true, &[0xff], 1, base, disp);
    }

    /// `cmp byte [base+disp], imm8`.
    pub(crate) fn g_cmp_byte_imm(&mut self, base: u8, disp: i32, imm: u8) {
        self.g_mem(false, &[0x80], 7, base, disp);
        self.code.push(imm);
    }

    /// `mov qword [base+disp], imm32` (sign-extended).
    pub(crate) fn g_store_imm32(&mut self, base: u8, disp: i32, imm: i32) {
        self.g_mem(true, &[0xc7], 0, base, disp);
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `call qword [base+disp]`.
    pub(crate) fn g_call_mem(&mut self, base: u8, disp: i32) {
        self.g_mem(false, &[0xff], 2, base, disp);
    }

    /// `mov dst, src`.
    pub(crate) fn g_mov(&mut self, dst: u8, src: u8) {
        self.g_rr(0x89, dst, src);
    }

    /// `cmp a, b`.
    pub(crate) fn g_cmp(&mut self, a: u8, b: u8) {
        self.g_rr(0x39, a, b);
    }

    /// `and dst, src`.
    pub(crate) fn g_and(&mut self, dst: u8, src: u8) {
        self.g_rr(0x21, dst, src);
    }

    /// `test a, a`.
    pub(crate) fn g_test(&mut self, a: u8) {
        self.g_rr(0x85, a, a);
    }

    /// `cmp reg, imm32` (sign-extended).
    pub(crate) fn g_cmp_imm(&mut self, reg: u8, imm: i32) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0x81);
        self.code.push(0xc0 | (7 << 3) | (reg & 7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `movabs reg, imm64`.
    pub(crate) fn g_movabs(&mut self, reg: u8, imm: u64) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0xb8 + (reg & 7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }

    /// `jcc label` (`cond` from [`cc`]).
    pub(crate) fn g_jcc(&mut self, cond: u8, label: Label) {
        self.code.extend_from_slice(&[0x0f, 0x80 | cond]);
        self.emit_rel32(label);
    }

    /// `rep stosq` — store `rax` to `rcx` qwords from `rdi` up.
    pub(crate) fn g_rep_stosq(&mut self) {
        self.code.extend_from_slice(&[0xf3, 0x48, 0xab]);
    }

    /// The byte offset `label` was bound at (after emission).
    pub(crate) fn label_offset(&self, label: Label) -> usize {
        self.labels[label.0]
    }
}

/// Emits a generic body's **native-call entry**: `extern "C" fn(ctx, window,
/// argc, argv, caps, this) -> u64`, the target of the call sequence. It is the
/// body's own frame prologue followed by exactly the register-window layout
/// `call_with_inner` builds — arguments `0..n_params` (missing ones
/// `undefined`), captures `n_params..`, `this` after them, `new.target` (the
/// header word just below `argv`) in register `new_target` when the body
/// reads it, every other register `undefined` — and then the body itself from
/// pc 0 (`body`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_native_entry(
    a: &mut X64Assembler,
    frame: u32,
    countdown: i8,
    n_regs: usize,
    n_params: usize,
    n_caps: usize,
    new_target: Option<usize>,
    body: Label,
) -> Option<()> {
    let undefined = crate::nanbox::NanBox::undefined().to_bits();
    let slot = |i: usize| i32::try_from(i * 8).ok();
    a.frame_prologue(frame);
    a.mov_r15_rdi(); // pin ctx
    a.mov_rbx_rsi(); // pin the window
    let n_params = n_params.min(n_regs);
    if n_params > 0 {
        // rdx = argc, rcx = argv.
        let short = a.new_label();
        let done = a.new_label();
        a.g_cmp_imm(r::RDX, i32::try_from(n_params).ok()?);
        a.g_jcc(cc::B, short);
        for i in 0..n_params {
            a.g_load(r::RAX, r::RCX, slot(i)?);
            a.g_store(r::RBX, slot(i)?, r::RAX);
        }
        a.jmp(done);
        a.bind(short);
        a.g_movabs(r::R11, undefined);
        for i in 0..n_params {
            let missing = a.new_label();
            let store = a.new_label();
            a.g_cmp_imm(r::RDX, i32::try_from(i).ok()?);
            a.g_jcc(cc::BE, missing); // argc <= i
            a.g_load(r::RAX, r::RCX, slot(i)?);
            a.jmp(store);
            a.bind(missing);
            a.g_mov(r::RAX, r::R11);
            a.bind(store);
            a.g_store(r::RBX, slot(i)?, r::RAX);
        }
        a.bind(done);
    }
    let caps_end = (n_params + n_caps).min(n_regs);
    for (j, i) in (n_params..caps_end).enumerate() {
        a.g_load(r::RAX, r::R8, slot(j)?);
        a.g_store(r::RBX, slot(i)?, r::RAX);
    }
    let this_slot = n_params + n_caps;
    if this_slot < n_regs {
        a.g_store(r::RBX, slot(this_slot)?, r::R9);
    }
    // `new.target` (before the fill below reuses `rcx`; the fill skips it).
    let nt = new_target.filter(|&nt| nt > this_slot && nt < n_regs);
    if nt.is_some() {
        a.g_load(r::R10, r::RCX, -8);
    }
    // Every other register starts `undefined`.
    let rest = this_slot + 1;
    if rest < n_regs {
        let count = n_regs - rest;
        a.g_movabs(r::RAX, undefined);
        if count <= 16 {
            for i in rest..n_regs {
                a.g_store(r::RBX, slot(i)?, r::RAX);
            }
        } else {
            a.g_lea(r::RDI, r::RBX, slot(rest)?);
            a.g_movabs(r::RCX, count as u64);
            a.g_rep_stosq();
        }
    }
    if let Some(nt) = nt {
        a.g_store(r::RBX, slot(nt)?, r::R10);
    }
    a.mov_frame8_imm32(countdown, 1); // poll at the first back-edge
    a.jmp(body);
    Some(())
}

/// `rbp`'s register number.
const RBP: u8 = 5;

/// `(bits & HANDLE_TAG) == HANDLE_TAG` iff `bits` is a heap handle (the sign
/// and quiet-NaN bits both set).
const HANDLE_TAG: u64 = 0xfffc_0000_0000_0000;

/// One call site's emission inputs (see `emit_call`).
pub(crate) struct CallEmit<'a> {
    /// The register the result goes to.
    pub dst: u16,
    /// Where the callee comes from.
    pub callee: CalleeSrc,
    /// The `this` register, or `None` for `undefined` (`Op::CallValue`);
    /// ignored for [`CalleeSrc::Construct`] (`this` is the new object).
    pub this: Option<u16>,
    /// The argument registers.
    pub args: &'a [u16],
    /// The site's cache.
    pub site: *const CallSite,
    /// The VM state layout.
    pub layout: &'a CallLayout,
    /// The callee identification helper (for [`CalleeSrc::Reg`]).
    pub callee_helper: u64,
    /// Where a throw goes.
    pub throw_exit: Label,
    /// Where anything the sequence does not handle goes (the site's helper).
    pub slow: Label,
    /// Where execution continues after the call.
    pub done: Label,
}

/// Emits the native call sequence of one call site (see the module docs). On
/// any guard failure it jumps to `e.slow`, having changed nothing.
pub(crate) fn emit_call(a: &mut X64Assembler, e: &CallEmit<'_>) -> Option<()> {
    use crate::nanbox::NanBox;
    let l = e.layout;
    let reg = |x: u16| i32::from(x) * 8;
    let argc = e.args.len();
    let hdr = |i: usize| i32::try_from(i * 8).ok();
    // 1. Which function is the callee? (rax = key, rdx = captures)
    // `load_callee(reg)` reloads its value into `reg` afterwards.
    let (callee_base, callee_disp) = match e.callee {
        CalleeSrc::Frame { scratch } => {
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, RBP, scratch);
            a.g_movabs(r::RAX, e.callee_helper);
            a.call_rax();
            (RBP, scratch)
        }
        CalleeSrc::Reg(c) => {
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, r::RBX, reg(c));
            a.g_movabs(r::RAX, e.callee_helper);
            a.call_rax();
            (r::RBX, reg(c))
        }
        CalleeSrc::Method {
            recv,
            key_ptr,
            key_len,
            cache,
            scratch,
            helper,
        } => {
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, r::RBX, reg(recv));
            a.g_movabs(r::RDX, key_ptr as u64);
            a.g_movabs(r::RCX, key_len as u64);
            a.g_movabs(r::R8, cache as u64);
            a.g_lea(r::R9, RBP, scratch);
            a.g_movabs(r::RAX, helper);
            a.call_rax();
            (RBP, scratch)
        }
        CalleeSrc::Construct {
            ctor,
            scratch,
            helper,
            ..
        } => {
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, r::RBX, reg(ctor));
            a.g_lea(r::RDX, RBP, scratch);
            a.g_movabs(r::RCX, e.site as u64);
            a.g_movabs(r::RAX, helper);
            a.call_rax();
            (r::RBX, reg(ctor))
        }
    };
    // r11 = the matching way.
    a.g_movabs(r::R11, e.site as u64);
    let hit = a.new_label();
    for way in 0..CallSite::WAYS {
        if way > 0 {
            a.g_lea(r::R11, r::R11, TARGET_BYTES);
        }
        a.g_cmp_mem(r::RAX, r::R11, SITE_KEY);
        a.g_jcc(cc::E, hit);
    }
    a.jmp(e.slow);
    a.bind(hit);
    a.g_mov(r::R8, r::RDX); // captures, for the callee's entry
    // 2. The caller's window is on the register stack (so it is published),
    //    one realm, and room under both depth limits.
    a.g_load(r::R10, r::R15, l.stack);
    a.g_cmp_mem(r::RBX, r::R10, STACK_TOP);
    a.g_jcc(cc::AE, e.slow);
    a.g_cmp_mem(r::RBX, r::R10, STACK_BASE);
    a.g_jcc(cc::B, e.slow);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_cmp_byte_imm(r::R9, l.multi_realm, 0);
    a.g_jcc(cc::NE, e.slow);
    a.g_load(r::RAX, r::R15, l.call_depth);
    a.g_test(r::RAX);
    a.g_jcc(cc::E, e.slow);
    a.g_cmp_mem(r::RAX, r::R9, l.max_depth);
    a.g_jcc(cc::AE, e.slow);
    a.g_load(r::RAX, r::R9, l.total_depth);
    a.g_cmp_mem(r::RAX, r::R9, l.max_depth);
    a.g_jcc(cc::AE, e.slow);
    // 3. Room for the header, the arguments and the callee's window:
    //    rcx = header, rsi = window, rax = new top.
    a.g_load(r::RCX, r::R10, STACK_TOP);
    a.g_lea(r::RSI, r::RCX, hdr(FRAME_HEADER + argc)?);
    a.g_mov(r::RAX, r::RSI);
    a.g_add_mem(r::RAX, r::R11, SITE_FRAME);
    a.g_cmp_mem(r::RAX, r::R10, STACK_LIMIT);
    a.g_jcc(cc::A, e.slow);
    // 4. The header and the arguments.
    a.g_load(r::RDX, r::R10, STACK_FRAMES);
    a.g_store(r::RCX, 0, r::RDX);
    a.g_load(r::RDX, callee_base, callee_disp);
    a.g_store(r::RCX, 8, r::RDX);
    a.g_load(r::RDX, r::R9, l.callee_len);
    a.g_store(r::RCX, 16, r::RDX);
    a.g_store_imm32(r::RCX, 24, i32::try_from(argc).ok()?);
    match e.callee {
        // `new.target` is the constructor itself.
        CalleeSrc::Construct { ctor, .. } => a.g_load(r::RDX, r::RBX, reg(ctor)),
        _ => a.g_movabs(r::RDX, NanBox::undefined().to_bits()),
    }
    a.g_store(r::RCX, 32, r::RDX);
    for (i, &ar) in e.args.iter().enumerate() {
        a.g_load(r::RDX, r::RBX, reg(ar));
        a.g_store(r::RCX, hdr(FRAME_HEADER + i)?, r::RDX);
    }
    // 5. Commit: the activation is on the stack and counted.
    a.g_store(r::R10, STACK_FRAMES, r::RCX);
    a.g_store(r::R10, STACK_TOP, r::RAX);
    a.g_inc_mem(r::R15, l.call_depth);
    a.g_inc_mem(r::R9, l.total_depth);
    a.g_inc_mem(r::R15, l.frames_published);
    // 6. entry(ctx, window, argc, argv, caps, this).
    a.g_mov(r::RDI, r::R15);
    a.g_movabs(r::RDX, argc as u64);
    a.g_lea(r::RCX, r::RCX, hdr(FRAME_HEADER)?);
    match (&e.callee, e.this) {
        (CalleeSrc::Construct { scratch, .. }, _) => a.g_load(r::R9, RBP, *scratch),
        (_, Some(t)) => a.g_load(r::R9, r::RBX, reg(t)),
        (_, None) => a.g_movabs(r::R9, NanBox::undefined().to_bits()),
    }
    a.g_call_mem(r::R11, SITE_CODE);
    // 7. Pop the activation (the callee restored everything it changed).
    a.g_load(r::R10, r::R15, l.stack);
    a.g_load(r::RCX, r::R10, STACK_FRAMES);
    a.g_store(r::R10, STACK_TOP, r::RCX);
    a.g_load(r::RDX, r::RCX, 0);
    a.g_store(r::R10, STACK_FRAMES, r::RDX);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_dec_mem(r::R15, l.call_depth);
    a.g_dec_mem(r::R9, l.total_depth);
    a.g_dec_mem(r::R15, l.frames_published);
    // 8. The outcome: a throw, falling off the end (`undefined`), or a value.
    let store = a.new_label();
    a.g_movabs(r::R11, NanBox::jit_throw_bits());
    a.g_cmp(r::RAX, r::R11);
    a.g_jcc(cc::E, e.throw_exit);
    if let CalleeSrc::Construct {
        scratch, result, ..
    } = e.callee
    {
        // `[[Construct]]`'s result: an object the body returned, else `this`.
        // Only a heap value can be an object (the sentinel for falling off the
        // end is not one).
        let this = a.new_label();
        a.g_movabs(r::R11, HANDLE_TAG);
        a.g_mov(r::RCX, r::RAX);
        a.g_and(r::RCX, r::R11);
        a.g_cmp(r::RCX, r::R11);
        a.g_jcc(cc::NE, this);
        a.g_mov(r::RDI, r::R15);
        a.g_mov(r::RSI, r::RAX);
        a.g_load(r::RDX, RBP, scratch);
        a.g_movabs(r::RAX, result);
        a.call_rax();
        a.jmp(store);
        a.bind(this);
        a.g_load(r::RAX, RBP, scratch);
    } else {
        a.g_movabs(r::R11, NanBox::jit_falloff_bits());
        a.g_cmp(r::RAX, r::R11);
        a.g_jcc(cc::NE, store);
        a.g_movabs(r::RAX, NanBox::undefined().to_bits());
    }
    a.bind(store);
    a.g_store(r::RBX, reg(e.dst), r::RAX);
    a.jmp(e.done);
    Some(())
}
