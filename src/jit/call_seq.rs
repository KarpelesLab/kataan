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
    pub(crate) const RBP: u8 = 5;
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
///
/// **No word of the stack ever names a collected object.** Words below `top`
/// are roots. Words in `top..limit` may still hold what a finished activation
/// left there, but the collector zeroes them (and pulls `limit` down) before
/// every collection, so whatever they name was alive when it was written and
/// has not been through a collection since. Words from `limit` to `end` hold
/// no handle at all. So a native-call entry need not fill a register the body
/// writes before it reads it: until then the GC sees in it, at worst, an
/// object it keeps one cycle longer (see `prologue::entry_fill`).
#[repr(C)]
#[derive(Debug)]
pub struct JitStack {
    /// One past the innermost window (the next free word).
    pub top: *mut u64,
    /// One past the last word an activation may claim without asking: the end
    /// of the part a finished activation may have left values in. The VM
    /// raises it (up to `end`) for an activation it lays out itself
    /// (`jit_tier::reserve`); emitted code that finds no room takes its slow
    /// path, which is such an activation.
    pub limit: *mut u64,
    /// The innermost native-call header (null when there is none). A header is
    /// [`FRAME_HEADER`] words — the previous header, the callee value, the
    /// realm's `vm_callee_stack` length at the call, the argument count and
    /// `new.target` — followed by the arguments, then the callee's window.
    pub frames: *mut u64,
    /// The first word.
    pub base: *mut u64,
    /// One past the last word of the allocation.
    pub end: *mut u64,
}

/// [`JitStack::top`]'s offset.
pub(crate) const STACK_TOP: i32 = 0;
/// [`JitStack::limit`]'s offset.
pub(crate) const STACK_LIMIT: i32 = 8;
/// [`JitStack::frames`]'s offset.
pub(crate) const STACK_FRAMES: i32 = 16;
/// [`JitStack::base`]'s offset.
pub(crate) const STACK_BASE: i32 = 24;
const _: () = {
    use core::mem::offset_of;
    assert!(offset_of!(JitStack, top) == STACK_TOP as usize);
    assert!(offset_of!(JitStack, limit) == STACK_LIMIT as usize);
    assert!(offset_of!(JitStack, frames) == STACK_FRAMES as usize);
    assert!(offset_of!(JitStack, base) == STACK_BASE as usize);
};
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
    /// Per way, the last callee value the sequence identified as it (see
    /// [`CallMemo`]): a polymorphic site's values each keep theirs.
    pub memo: [CallMemo; CallSite::WAYS],
    /// A `new` site's inline constructor cache (see [`ConstructFast`]).
    pub fast: ConstructFast,
    /// An identity cache for a non-call op that keeps one here (see
    /// [`IdentMemo`]).
    pub ident: IdentMemo,
    /// A `new` site's constructor cache (unused by other sites).
    pub construct: super::props::ConstructCache,
}

/// A call site's identity memo: the callee **value** (its handle bits) the
/// sequence last identified as the way `target`, with its captures at
/// `caps`, at prototype epoch `stamp`.
///
/// While the epoch is unchanged, equal handle bits name the same heap cell
/// with the same contents: a freed slot's generation is bumped (a slot whose
/// generation would wrap is retired instead), and the ways a cell's contents
/// can move or be replaced under a handle — a cell swap, a compaction, a
/// snapshot restore's capture fill — all bump the epoch. A VM function
/// cell's function id never changes and its captures vector is never
/// resized, so the vector's buffer (`caps`) stays put while the cell lives
/// (when the arena grows, a move of the cell moves only the vector's
/// header). So a value equal to `bits` is the function the way `target` was
/// filled for, its captures at `caps`, and the sequence skips identifying
/// it. Cleared whenever a way is replaced.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CallMemo {
    /// The callee's `NanBox` bits, or [`CallSite::EMPTY`] (no value has
    /// those bits).
    pub bits: u64,
    /// The prototype epoch at which `bits` was identified.
    pub stamp: u64,
    /// The way it was identified as.
    pub target: *const CallTarget,
    /// Its captured values.
    pub caps: *const u64,
}

impl CallMemo {
    const fn empty() -> Self {
        Self {
            bits: CallSite::EMPTY,
            stamp: 0,
            target: core::ptr::null(),
            caps: core::ptr::null(),
        }
    }
}

/// A `new` (or `super(…)`) site's inline constructor cache, filled by
/// `jit_construct_info` for a constructor whose `[[Construct]]` with
/// `new.target` `nt` the two identities fix:
///
/// * a base constructor (`derived == 0`) — a VM function constructor
///   without instance fields — with `new.target` a class constructor whose
///   `prototype` is non-writable and non-configurable (so always the object
///   `proto`): allocate an ordinary object whose `[[Prototype]]` is `proto`
///   as `this`;
/// * a derived class constructor (`derived == 1`): `this` starts unbound
///   (the TDZ sentinel) and `super(…)` binds it.
///
/// Both constructors' cells and auxiliary property objects are watched, so
/// any change to their own properties (or their `[[Prototype]]`) bumps the
/// prototype epoch. While the epoch is still `stamp`, a construct of the
/// value `ctor` with `new.target` `nt` (equal handle bits: see [`CallMemo`])
/// prepares `this` so and enters the way `target` with the captures at
/// `caps`. Cleared whenever a way is replaced.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConstructFast {
    /// The constructor's `NanBox` bits, or [`CallSite::EMPTY`].
    pub ctor: u64,
    /// The `new.target`'s `NanBox` bits.
    pub nt: u64,
    /// The prototype epoch at which it was recorded.
    pub stamp: u64,
    /// The raw handle of `nt.prototype` (base constructors).
    pub proto: u64,
    /// `1` for a derived constructor (no allocation), else `0`.
    pub derived: u64,
    /// The way that enters its body.
    pub target: *const CallTarget,
    /// Its captured values.
    pub caps: *const u64,
}

impl ConstructFast {
    /// An empty cache.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            ctor: CallSite::EMPTY,
            nt: CallSite::EMPTY,
            stamp: 0,
            proto: 0,
            derived: 0,
            target: core::ptr::null(),
            caps: core::ptr::null(),
        }
    }
}

/// A one-entry identity cache of a compiled op whose result one value fixes
/// while the prototype epoch stands (`Op::SuperCtor`: a constructor's
/// `[[Prototype]]`; `Op::InitFields`: a constructor without instance
/// fields): the op on the value `key` at epoch `stamp` gives `value`. The
/// key's cell and auxiliary object are watched when it is filled.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IdentMemo {
    /// The operand's `NanBox` bits, or [`CallSite::EMPTY`].
    pub key: u64,
    /// The prototype epoch at which it was recorded.
    pub stamp: u64,
    /// The result.
    pub value: u64,
}

impl IdentMemo {
    /// An empty cache.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            key: CallSite::EMPTY,
            stamp: 0,
            value: 0,
        }
    }
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
            memo: [CallMemo::empty(); Self::WAYS],
            fast: ConstructFast::empty(),
            ident: IdentMemo::empty(),
            construct: super::props::ConstructCache::new(),
        }
    }

    /// Whether the site already holds `key`.
    #[must_use]
    pub fn holds(&self, key: u64) -> bool {
        self.targets.iter().any(|t| t.key == key)
    }

    /// The way holding `key`, if any.
    #[must_use]
    pub fn way_of(&self, key: u64) -> Option<*const CallTarget> {
        self.targets
            .iter()
            .find(|t| t.key == key)
            .map(core::ptr::from_ref)
    }

    /// Remembers `target` (a no-op when its key is already held).
    pub fn fill(&mut self, target: CallTarget) {
        if self.holds(target.key) {
            return;
        }
        let way = self.next % Self::WAYS;
        self.targets[way] = target;
        self.memo = [CallMemo::empty(); Self::WAYS];
        self.fast = ConstructFast::empty();
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

/// The leaf helper a method call finds its callee with when the inline
/// lookup misses: `(ctx, recv, site, out) -> CalleeInfo` — the method
/// `recv.key` (the [`MethodSite`](super::props::MethodSite)'s key, through its
/// property cache's prototype-chain cache) when it is a VM closure the
/// interpreter's method call would enter directly (its value stored to
/// `*out`), else the not-a-closure key. Never allocates (beyond refilling the
/// caches), never throws, never runs user code.
pub type MethodHelper = extern "C" fn(
    *mut core::ffi::c_void,
    u64,
    *mut super::props::MethodSite,
    *mut u64,
) -> CalleeInfo;

/// The leaf helper a `new` (or `super(…)`) finds its callee and `this`
/// with: `(ctx, ctor, new_target, out, site) -> CalleeInfo` — for a
/// constructor whose `[[Construct]]` the sequence may run (a VM constructor
/// with no instance fields, or a derived class constructor), the `this` it
/// starts with (the object `OrdinaryCreateFromConstructor` makes, or the
/// unbound sentinel for a derived one), stored to `*out`; else the
/// not-a-closure key, having done nothing. Never throws, never runs user
/// code.
///
/// `site` is the call site's cache: a constructor other than the one it names
/// is reported without creating anything (the sequence then takes the slow
/// path, which constructs it itself).
pub type ConstructHelper =
    extern "C" fn(*mut core::ffi::c_void, u64, u64, *mut u64, *const CallSite) -> CalleeInfo;

/// The helper that throws `super(…)`'s "already bound" `ReferenceError`:
/// `(ctx) -> the throw sentinel`, the error pending.
pub type SuperTwiceHelper = extern "C" fn(*mut core::ffi::c_void) -> u64;

/// The helper that finishes a `new` whose body returned the heap value
/// `result`, or whose `this` started unbound (a derived constructor):
/// `(ctx, result, this) -> result if it is an object, else this` — for a
/// derived constructor, else the throw sentinel with the `ReferenceError` or
/// `TypeError` pending.
pub type ConstructResultHelper = extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u64;

/// The helper a `new` served by the site's [`ConstructFast`] allocates its
/// `this` with: `(ctx, proto) -> the handle bits of a fresh ordinary object
/// whose [[Prototype]] is the raw handle proto`. Never throws, never runs
/// user code.
pub type ConstructAllocHelper = extern "C" fn(*mut core::ffi::c_void, u64) -> u64;

/// `CallSite` field offsets for emitted code.
fn site_off(f: usize) -> i32 {
    i32::try_from(f).expect("site offset fits in i32")
}

/// `CallSite`-relative offset of its [`IdentMemo`]'s `key`.
pub(crate) const IDENT_KEY: i32 =
    (core::mem::offset_of!(CallSite, ident) + core::mem::offset_of!(IdentMemo, key)) as i32;

/// `CallSite`-relative offset of its [`IdentMemo`]'s `value`.
pub(crate) const IDENT_VALUE: i32 =
    (core::mem::offset_of!(CallSite, ident) + core::mem::offset_of!(IdentMemo, value)) as i32;

impl X64Assembler {
    /// To `miss` unless the value at `[rbx + disp]` hits `site`'s
    /// [`IdentMemo`] (the recorded key, at the current prototype epoch); on a
    /// hit `r11` = the site (its result at [`IDENT_VALUE`]). Clobbers `rax`,
    /// `r9`, `r11`.
    pub(crate) fn p_ident_hit(&mut self, site: *const CallSite, disp: i32, miss: Label) {
        use core::mem::offset_of;
        let ident = offset_of!(CallSite, ident);
        self.g_load(r::RAX, r::RBX, disp);
        self.g_movabs(r::R11, site as u64);
        self.g_cmp_mem(r::RAX, r::R11, site_off(ident + offset_of!(IdentMemo, key)));
        self.g_jcc(cc::NE, miss);
        load_proto_epoch(self, r::R9);
        self.g_cmp_mem(
            r::R9,
            r::R11,
            site_off(ident + offset_of!(IdentMemo, stamp)),
        );
        self.g_jcc(cc::NE, miss);
    }
}

/// `mov dst, [PROTO_EPOCH]`.
fn load_proto_epoch(a: &mut X64Assembler, dst: u8) {
    a.g_movabs(dst, crate::object::proto_epoch_addr() as u64);
    a.g_load(dst, dst, 0);
}

/// Where a call sequence's callee comes from.
pub(crate) enum CalleeSrc {
    /// A value already in the native-frame slot `[rbp + scratch]` (a computed
    /// method, `Op::CallKey`); identified by the [`CalleeHelper`].
    Frame {
        /// The `rbp`-relative slot.
        scratch: i32,
    },
    /// A constructor register (`Op::Construct`, `new.target` the constructor
    /// itself; `Op::SuperCallArgs`, `new.target` a register), prepared inline
    /// from the site's [`ConstructFast`] (with the [`ConstructAllocHelper`])
    /// or by the [`ConstructHelper`], either of which leaves the call's
    /// `this` in `[rbp + scratch]`.
    Construct {
        /// The constructor register.
        ctor: u16,
        /// The `new.target` register.
        new_target: u16,
        /// The `rbp`-relative scratch slot.
        scratch: i32,
        /// The preparation helper.
        helper: u64,
        /// The [`ConstructAllocHelper`].
        alloc: u64,
        /// The [`ConstructResultHelper`].
        result: u64,
        /// For `super(…)`: the `this` register the result binds (when still
        /// unbound, else the [`SuperTwiceHelper`] `twice` throws).
        bind_this: Option<(u16, u64)>,
    },
    /// A register (`Op::CallValue`); identified by the [`CalleeHelper`].
    Reg(u16),
    /// A method lookup (`Op::CallMethod`) — inline through the site's
    /// [`MethodSite`](super::props::MethodSite), else by the [`MethodHelper`] —
    /// whose value lands in the native-frame slot at `[rbp + scratch]`.
    Method {
        /// The receiver register.
        recv: u16,
        /// The site's chain cache.
        site: *mut super::props::MethodSite,
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
    /// `ctx`: whether the run is hosted (`bool`).
    pub hosted: i32,
    /// `ctx`: whether the context runs in the main realm (`bool`).
    pub main_realm: i32,
    /// realm: whether a new VM closure needs no `[[Prototype]]` record
    /// (`bool`, `Realm::jit_vmfn_plain`).
    pub vmfn_plain: i32,
    /// What the back-edge poll's inline do-nothing test reads (see
    /// `emit_quiet_poll`).
    pub poll: PollLayout,
}

/// Where a back-edge poll finds what decides that it has nothing to do (the
/// early return of `jit_tier::jit_helper_backedge`): `realm`-relative
/// `agents_active: bool`, `interrupt: Option<Interrupt>` (or `None` when
/// `None` is not the null word on this build: then every poll calls), the
/// heap's allocation count and the collection threshold; `ctx`-relative the
/// leaving activation's window address and the poll interval.
#[derive(Clone, Copy, Debug)]
pub struct PollLayout {
    pub agents: i32,
    pub interrupt: Option<i32>,
    pub allocs: i32,
    pub threshold: i32,
    pub exit_regs: i32,
    pub interval: i32,
}

/// The back-edge poll's do-nothing test, inline: no agent scheduling, no
/// interrupt handle, no collection due by the allocation count, and not
/// the activation that may leave — `jit_helper_backedge`'s early return,
/// which it repeats (with the rest of the safepoint test) when called. Then
/// the frame's countdown (`[rbp+countdown]`) is reloaded with the poll
/// interval and control goes to `quiet`, with no call (so no receiver-cache
/// reset either); otherwise it falls through to the call. Clobbers `rax`
/// and `r11` only.
pub(crate) fn emit_quiet_poll(a: &mut X64Assembler, l: &CallLayout, countdown: i32, quiet: Label) {
    let p = &l.poll;
    let Some(interrupt) = p.interrupt else { return };
    let call = a.new_label();
    a.g_load(r::R11, r::R15, l.realm);
    a.g_cmp_byte_imm(r::R11, p.agents, 0);
    a.g_jcc(cc::NE, call);
    a.g_load(r::RAX, r::R11, interrupt);
    a.g_test(r::RAX);
    a.g_jcc(cc::NE, call);
    a.g_load(r::RAX, r::R11, p.allocs);
    a.g_cmp_mem(r::RAX, r::R11, p.threshold);
    a.g_jcc(cc::AE, call);
    a.g_cmp_mem(r::RBX, r::R15, p.exit_regs);
    a.g_jcc(cc::E, call);
    a.g_load(r::RAX, r::R15, p.interval);
    a.g_store(RBP, countdown, r::RAX);
    a.jmp(quiet);
    a.bind(call);
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

    /// `or reg, [base+disp]` (64-bit).
    pub(crate) fn g_or_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.g_mem(true, &[0x0b], reg, base, disp);
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
        self.inline_frame_hook();
        self.spill_before_call();
        self.g_mem(false, &[0xff], 2, base, disp);
        self.clear_recv_cache();
        self.reload_after_call();
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

    /// `call rax`.
    pub(crate) fn g_call_rax(&mut self) {
        self.code.extend_from_slice(&[0xff, 0xd0]);
    }

    /// `call label` (rel32).
    pub(crate) fn g_call_label(&mut self, label: Label) {
        self.code.push(0xe8);
        self.emit_rel32(label);
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
    fill: &[bool],
    n_params: usize,
    n_caps: usize,
    new_target: Option<usize>,
    body: Label,
) -> Option<()> {
    let undefined = crate::nanbox::NanBox::undefined().to_bits();
    let slot = |i: usize| i32::try_from(i * 8).ok();
    let n_regs = fill.len();
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
    // Every other register starts `undefined` — but one the body writes
    // before any read of it needs no store (`prologue::entry_fill`).
    let rest = this_slot + 1;
    let needed: alloc::vec::Vec<usize> = (rest..n_regs)
        .filter(|&i| fill[i] && Some(i) != nt)
        .collect();
    if !needed.is_empty() {
        let count = n_regs - rest;
        a.g_movabs(r::RAX, undefined);
        // Straight-line stores, or a four-store loop for a large window: a
        // `rep stosq` costs a few dozen cycles to start, more than the whole
        // fill of a typical window.
        if needed.len() <= 64 {
            for &i in &needed {
                a.g_store(r::RBX, slot(i)?, r::RAX);
            }
        } else {
            let quads = count / 4;
            let lp = a.new_label();
            a.g_lea(r::RDI, r::RBX, slot(rest)?);
            a.g_lea(r::RCX, r::RBX, slot(rest + quads * 4)?);
            a.bind(lp);
            for k in 0..4 {
                a.g_store(r::RDI, slot(k)?, r::RAX);
            }
            a.g_lea(r::RDI, r::RDI, 32);
            a.g_cmp(r::RDI, r::RCX);
            a.g_jcc(cc::B, lp);
            for i in rest + quads * 4..n_regs {
                a.g_store(r::RBX, slot(i)?, r::RAX);
            }
        }
    }
    if let Some(nt) = nt {
        a.g_store(r::RBX, slot(nt)?, r::R10);
    }
    a.mov_frame8_imm32(countdown, 1); // poll at the first back-edge
    a.jmp(body);
    Some(())
}

/// Emits the native-call entry of a body with `try` regions: the same
/// signature as `emit_native_entry`'s, calling that entry (`inner`) inside a
/// segment of the context's handler stack of its own. `helper` is the
/// `GenericHelpers::handler_frame` helper: `helper(ctx, 0, _)` opens the
/// segment and returns the caller's, `helper(ctx, 1, saved)` drops whatever
/// the body left in its segment (a throw that escaped it leaves its handlers
/// behind) and restores the caller's. The interpreter-entered activation gets
/// this from its Rust caller (`jit_tier::run_compiled_at`); a body entered
/// natively makes it itself.
pub(crate) fn emit_handler_entry(a: &mut X64Assembler, helper: u64, inner: Label) {
    // Slots below the prologue's saved `r15`/`rbx`: the six arguments, the
    // caller's segment and the result.
    const ARGS: [u8; 6] = [r::RDI, r::RSI, r::RDX, r::RCX, r::R8, r::R9];
    let slot = |i: i32| -24 - 8 * i;
    a.frame_prologue(64);
    for (i, &reg) in ARGS.iter().enumerate() {
        a.g_store(RBP, slot(i as i32), reg);
    }
    // rdi = ctx already.
    a.g_movabs(r::RSI, 0);
    a.g_movabs(r::RAX, helper);
    a.g_call_rax();
    a.g_store(RBP, slot(6), r::RAX);
    for (i, &reg) in ARGS.iter().enumerate() {
        a.g_load(reg, RBP, slot(i as i32));
    }
    a.g_call_label(inner);
    a.g_store(RBP, slot(7), r::RAX);
    a.g_load(r::RDI, RBP, slot(0));
    a.g_movabs(r::RSI, 1);
    a.g_load(r::RDX, RBP, slot(6));
    a.g_movabs(r::RAX, helper);
    a.g_call_rax();
    a.g_load(r::RAX, RBP, slot(7));
    a.frame_epilogue();
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
    let jl = super::compute_jit_layout();
    use core::mem::offset_of;
    // Way `w`'s memo fields (see `CallMemo`).
    let memo = |w: usize| offset_of!(CallSite, memo) + w * core::mem::size_of::<CallMemo>();
    let memo_bits = |w: usize| site_off(memo(w) + offset_of!(CallMemo, bits));
    let memo_stamp = |w: usize| site_off(memo(w) + offset_of!(CallMemo, stamp));
    let memo_target = |w: usize| site_off(memo(w) + offset_of!(CallMemo, target));
    let memo_caps = |w: usize| site_off(memo(w) + offset_of!(CallMemo, caps));
    // `identified`: the way is in r11 and the captures in r8 already (a memo
    // hit, or a constructor the site's `fast` cache prepared).
    let identified = a.new_label();
    // The memo check for the callee value in `rax` (see `CallMemo`): the
    // value one of the ways' memos names, at the current prototype epoch —
    // then r11 = the way, r8 = the captures and on to `identified`; else on,
    // `rax` unchanged.
    let memo_check = |a: &mut X64Assembler| {
        let decode = a.new_label();
        let found = a.new_label();
        let others = a.new_label();
        // Way 0's memo falls through (the monomorphic site's path).
        a.g_movabs(r::R11, e.site as u64);
        a.g_cmp_mem(r::RAX, r::R11, memo_bits(0));
        a.g_jcc(cc::NE, others);
        a.bind(found);
        load_proto_epoch(a, r::R9);
        a.g_cmp_mem(r::R9, r::R11, memo_stamp(0));
        a.g_jcc(cc::NE, decode);
        a.g_load(r::R8, r::R11, memo_caps(0));
        a.g_load(r::R11, r::R11, memo_target(0));
        a.jmp(identified);
        a.bind(others);
        for w in 1..CallSite::WAYS {
            let next = a.new_label();
            a.g_cmp_mem(r::RAX, r::R11, memo_bits(w));
            a.g_jcc(cc::NE, next);
            // r11 = way `w`'s memo, addressed as way 0's.
            a.g_lea(r::R11, r::R11, memo_bits(w) - memo_bits(0));
            a.jmp(found);
            a.bind(next);
        }
        a.bind(decode);
    };
    // 1. Which function is the callee? (rax = key, rdx = captures)
    // `load_callee(reg)` reloads its value into `reg` afterwards.
    let (callee_base, callee_disp) = match e.callee {
        CalleeSrc::Frame { scratch } => {
            a.g_load(r::RAX, RBP, scratch);
            memo_check(a);
            a.p_callee_info(jl, l.realm);
            (RBP, scratch)
        }
        CalleeSrc::Reg(c) => {
            a.g_load(r::RAX, r::RBX, reg(c));
            memo_check(a);
            a.p_callee_info(jl, l.realm);
            (r::RBX, reg(c))
        }
        CalleeSrc::Method {
            recv,
            site,
            scratch,
            helper,
        } => {
            // The site's chain cache, inline (hosted runs only, as the helper);
            // a miss asks the helper, which refills it.
            let lookup = a.new_label();
            let have = a.new_label();
            a.g_cmp_byte_imm(r::R15, l.hosted, 0);
            a.g_jcc(cc::E, lookup);
            a.p_method_lookup(jl, l.realm, (r::RBX, reg(recv)), site, lookup, None);
            a.g_store(RBP, scratch, r::RAX);
            memo_check(a);
            a.p_callee_info(jl, l.realm);
            a.jmp(have);
            a.bind(lookup);
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, r::RBX, reg(recv));
            a.g_movabs(r::RDX, site as u64);
            a.g_lea(r::RCX, RBP, scratch);
            a.g_movabs(r::RAX, helper);
            a.call_rax();
            a.bind(have);
            (RBP, scratch)
        }
        CalleeSrc::Construct {
            ctor,
            new_target,
            scratch,
            helper,
            alloc,
            ..
        } => {
            // The site's `fast` cache (see `ConstructFast`): the same
            // constructor and `new.target` values at the same prototype epoch.
            let fast = offset_of!(CallSite, fast);
            let prep = a.new_label();
            a.g_load(r::RAX, r::RBX, reg(ctor));
            a.g_movabs(r::R11, e.site as u64);
            a.g_cmp_mem(
                r::RAX,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, ctor)),
            );
            a.g_jcc(cc::NE, prep);
            if new_target != ctor {
                a.g_load(r::RAX, r::RBX, reg(new_target));
                a.g_cmp_mem(
                    r::RAX,
                    r::R11,
                    site_off(fast + offset_of!(ConstructFast, nt)),
                );
                a.g_jcc(cc::NE, prep);
            }
            load_proto_epoch(a, r::R9);
            a.g_cmp_mem(
                r::R9,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, stamp)),
            );
            a.g_jcc(cc::NE, prep);
            let by_helper = a.new_label();
            let made = a.new_label();
            let base = a.new_label();
            // A derived constructor starts with `this` unbound.
            a.g_cmp_byte_imm(
                r::R11,
                site_off(fast + offset_of!(ConstructFast, derived)),
                0,
            );
            a.g_jcc(cc::E, base);
            a.g_movabs(r::RAX, NanBox::tdz().to_bits());
            a.jmp(made);
            a.bind(base);
            a.g_load(
                r::RSI,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, proto)),
            );
            a.p_alloc_object(jl, l.realm, by_helper);
            a.jmp(made);
            a.bind(by_helper);
            a.g_mov(r::RDI, r::R15);
            a.g_movabs(r::RAX, alloc);
            a.call_rax();
            // That allocation may have grown the heap (or collected), moving
            // every cell — the constructor's own, whose inline captures
            // `caps` points into, among them — and bumping the epoch: then
            // the record's `caps` is stale, so the helper identifies the
            // callee afresh (making its own `this`; this one is garbage).
            a.g_movabs(r::R11, e.site as u64);
            load_proto_epoch(a, r::R9);
            a.g_cmp_mem(
                r::R9,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, stamp)),
            );
            a.g_jcc(cc::NE, prep);
            a.bind(made);
            a.g_store(RBP, scratch, r::RAX);
            a.g_movabs(r::R11, e.site as u64);
            a.g_load(
                r::R8,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, caps)),
            );
            a.g_load(
                r::R11,
                r::R11,
                site_off(fast + offset_of!(ConstructFast, target)),
            );
            a.jmp(identified);
            a.bind(prep);
            a.g_mov(r::RDI, r::R15);
            a.g_load(r::RSI, r::RBX, reg(ctor));
            a.g_load(r::RDX, r::RBX, reg(new_target));
            a.g_lea(r::RCX, RBP, scratch);
            a.g_movabs(r::R8, e.site as u64);
            a.g_movabs(r::RAX, helper);
            a.call_rax();
            (r::RBX, reg(ctor))
        }
    };
    // r11 = the matching way (and, to memoize the callee in, r10 = its
    // memo, addressed as way 0's).
    a.g_movabs(r::R11, e.site as u64);
    let hit = a.new_label();
    let memoize = !matches!(e.callee, CalleeSrc::Construct { .. });
    let mut way_hits = alloc::vec::Vec::with_capacity(CallSite::WAYS);
    for way in 0..CallSite::WAYS {
        if way > 0 {
            a.g_lea(r::R11, r::R11, TARGET_BYTES);
        }
        a.g_cmp_mem(r::RAX, r::R11, SITE_KEY);
        if memoize {
            let h = a.new_label();
            a.g_jcc(cc::E, h);
            way_hits.push(h);
        } else {
            a.g_jcc(cc::E, hit);
        }
    }
    a.jmp(e.slow);
    for (way, h) in way_hits.into_iter().enumerate() {
        a.bind(h);
        a.g_movabs(
            r::R10,
            e.site as u64 + (memo_bits(way) - memo_bits(0)) as u64,
        );
        a.jmp(hit);
    }
    a.bind(hit);
    a.g_mov(r::R8, r::RDX); // captures, for the callee's entry
    if memoize {
        // Memoize the callee in its way's memo (see `CallMemo`).
        a.g_load(r::R9, callee_base, callee_disp);
        a.g_store(r::R10, memo_bits(0), r::R9);
        load_proto_epoch(a, r::R9);
        a.g_store(r::R10, memo_stamp(0), r::R9);
        a.g_store(r::R10, memo_target(0), r::R11);
        a.g_store(r::R10, memo_caps(0), r::R8);
    }
    a.bind(identified);
    // (In an inlined body: its activation, if still pending, is pushed
    // first — the callee's header goes above it. Preserves every register.)
    a.inline_frame_hook();
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
        CalleeSrc::Construct { new_target, .. } => a.g_load(r::RDX, r::RBX, reg(new_target)),
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
        scratch,
        result,
        bind_this,
        ..
    } = e.callee
    {
        // `[[Construct]]`'s result: an object the body returned, else `this`.
        // Only a heap value can be an object (the sentinel for falling off the
        // end is not one). A derived constructor's (`this` started unbound)
        // is validated by the helper too.
        let this = a.new_label();
        let check = a.new_label();
        let have = a.new_label();
        a.g_movabs(r::R11, HANDLE_TAG);
        a.g_mov(r::RCX, r::RAX);
        a.g_and(r::RCX, r::R11);
        a.g_cmp(r::RCX, r::R11);
        a.g_jcc(cc::NE, this);
        a.bind(check);
        a.g_mov(r::RDI, r::R15);
        a.g_mov(r::RSI, r::RAX);
        a.g_load(r::RDX, RBP, scratch);
        a.g_movabs(r::RAX, result);
        a.call_rax();
        a.g_movabs(r::R11, NanBox::jit_throw_bits());
        a.g_cmp(r::RAX, r::R11);
        a.g_jcc(cc::E, e.throw_exit);
        a.jmp(have);
        a.bind(this);
        a.g_load(r::RCX, RBP, scratch);
        a.g_movabs(r::R11, NanBox::tdz().to_bits());
        a.g_cmp(r::RCX, r::R11);
        a.g_jcc(cc::E, check);
        a.g_mov(r::RAX, r::RCX);
        a.bind(have);
        if let Some((this_reg, twice)) = bind_this {
            // `super(…)` binds `this` once.
            let bind = a.new_label();
            a.g_load(r::RCX, r::RBX, reg(this_reg));
            a.g_movabs(r::R11, NanBox::tdz().to_bits());
            a.g_cmp(r::RCX, r::R11);
            a.g_jcc(cc::E, bind);
            a.g_mov(r::RDI, r::R15);
            a.g_movabs(r::RAX, twice);
            a.call_rax();
            a.jmp(e.throw_exit);
            a.bind(bind);
            a.g_store(r::RBX, reg(this_reg), r::RAX);
        }
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

/// The native-frame slots of a body's inlined calls (`super::inline`), all
/// `rbp`-relative. An inlined activation's native-call header is pushed only
/// when Rust code is about to run inside it (lazily: see
/// [`emit_inline_guard`]), so `pend` and `hdr` describe one of three states:
/// none active (both `RECV_CACHE_EMPTY`), one pending (`pend` = the address
/// of its materialization stub), one materialized (`hdr` = its header).
#[derive(Clone, Copy)]
pub(crate) struct InlineSlots {
    /// The pending activation's materialization stub, or empty.
    pub pend: i32,
    /// The materialized activation's header, or empty.
    pub hdr: i32,
    /// The active activation's callee value (the header's second word).
    pub callee: i32,
    /// Whether the guards' limit checks hold (anything but
    /// `RECV_CACHE_EMPTY`): set by a guard that made them, and emptied with
    /// the receiver cache's keys — by the prologue and after every call,
    /// the only places the facts they check (the caller's window on the
    /// register stack, one realm, the depths, the stack top) can change.
    pub ok: i32,
    /// The most arguments any of the body's inlined calls passes (the room
    /// the checks ask for).
    pub max_argc: usize,
}

/// One inlined call's guard (`Op::InlineGuard`, see `super::inline`).
pub(crate) struct InlineEmit<'a> {
    /// The boolean result register.
    pub dst: u16,
    /// The callee key the inlined body was compiled for.
    pub key: u64,
    /// Where the callee comes from: [`CalleeSrc::Reg`] or
    /// [`CalleeSrc::Method`] (whose `helper` is unused: a lookup the site's
    /// chain cache cannot answer is the call's to make).
    pub callee: CalleeSrc,
    /// The call's argument registers.
    pub args: &'a [u16],
    /// The first capture register of the inlined body.
    pub caps: u16,
    /// How many captures.
    pub n_caps: u16,
    /// The VM state layout.
    pub layout: &'a CallLayout,
    /// The body's inline slots.
    pub slots: InlineSlots,
    /// This site's materialization stub ([`emit_inline_stub`]).
    pub stub: Label,
    /// The guarded call's cache, whose [`IdentMemo`] (unused by a call op)
    /// memoizes the callee value last identified as `key` with its
    /// captures (as [`CallMemo`] does for the call sequence).
    pub memo: *const CallSite,
    /// The call is a property read's inherited getter (`callee` the
    /// [`CalleeSrc::Method`] lookup of the read): a data property is a miss.
    pub getter: bool,
    /// The receiver cache's way for a method's receiver
    /// (`super::recv_cache`), if the guard is one of its sites.
    pub way: Option<RecvWay>,
    /// A `new` site's constructor register (also its `new.target`): the
    /// guard then requires the call's [`ConstructFast`] record for a base
    /// constructor and allocates `this` as it says, into the register after
    /// the captures (`n_caps` counts it).
    pub new_target: Option<u16>,
    /// For a `new`: the helper that allocates `this` when the heap's free
    /// list is empty ([`ConstructAllocHelper`]; `None`: that is a miss), and
    /// the receiver-cache way the fresh object fills.
    pub new_alloc: Option<u64>,
    pub new_way: Option<RecvWay>,
}

/// A receiver-cache way as one site uses it (see `super::recv_cache::SitePlan`).
#[derive(Clone, Copy)]
pub(crate) struct RecvWay {
    /// The `rbp`-relative slot of the way's key (the receiver's bits).
    pub key: i32,
    /// The `rbp`-relative slot of the way's heap slot address.
    pub cell: i32,
    /// The site compares the receiver with the way first.
    pub check: bool,
    /// The site fills the way after a walk.
    pub fill: bool,
    /// The way still holds the receiver (no compare).
    pub known: bool,
    /// The layout every other site of the way compiles in, if they all do
    /// (`super::recv_cache::shape_ways`): the walk fills the way only with
    /// an object of that layout, so the way keeps proving it.
    pub shape: Option<u64>,
}

/// Emits an inlined call's guard: identifies the callee exactly as
/// [`emit_call`] does (the method through the site's chain cache, then the
/// closure cell, memoized), and when it is `e.key`'s function and the call
/// sequence's limits allow the activation, makes it pending — the callee's
/// captures loaded into its registers, `e.dst` set to `true`. Otherwise sets
/// `e.dst` to `false`, having changed nothing else. Makes no call.
///
/// A pending activation is what the call sequence's would be but for its
/// native-call header and depth counts, which only Rust code can observe:
/// every `call` the body makes first runs the pending activation's stub
/// ([`X64Assembler::inline_frame_hook`]), which pushes them exactly as the
/// call sequence would have at the call. Nothing in between can change what
/// they would be (the arguments are the caller's registers, which the
/// inlined body never writes; the stack top and depths move only in Rust
/// code), and the limits are checked here, so the activation is the call's.
pub(crate) fn emit_inline_guard(a: &mut X64Assembler, e: &InlineEmit<'_>) -> Option<()> {
    use crate::nanbox::NanBox;
    use core::mem::offset_of;
    let l = e.layout;
    let reg = |x: u16| i32::from(x) * 8;
    let hdr = |i: usize| i32::try_from(i * 8).ok();
    let argc = e.args.len();
    let jl = super::compute_jit_layout();
    let ident = offset_of!(CallSite, ident);
    let memo_key = site_off(ident + offset_of!(IdentMemo, key));
    let memo_stamp = site_off(ident + offset_of!(IdentMemo, stamp));
    let memo_caps = site_off(ident + offset_of!(IdentMemo, value));
    let miss = a.new_label();
    let done = a.new_label();
    let identified = a.new_label();
    // 1. The callee value, into the `callee` slot (and rax).
    match e.callee {
        CalleeSrc::Reg(c) => a.g_load(r::RAX, r::RBX, reg(c)),
        CalleeSrc::Method { recv, site, .. } => {
            a.g_cmp_byte_imm(r::R15, l.hosted, 0);
            a.g_jcc(cc::E, miss);
            let got = a.new_label();
            let getter = e.getter.then_some(got);
            match e.way {
                // The receiver's heap slot from its way, else walked (and
                // the way filled).
                Some(w) => {
                    let walk = a.new_label();
                    let walked = a.new_label();
                    if w.known {
                        a.g_load(r::RCX, RBP, w.cell);
                    } else {
                        if w.check {
                            a.g_load(r::RAX, r::RBX, reg(recv));
                            a.g_cmp_mem(r::RAX, RBP, w.key);
                            a.g_jcc(cc::NE, walk);
                            a.g_load(r::RCX, RBP, w.cell);
                            a.jmp(walked);
                        }
                        a.bind(walk);
                        a.p_walk_plain(jl, l.realm, (r::RBX, reg(recv)), miss);
                        if w.fill {
                            let skip = a.new_label();
                            if let Some(shape) = w.shape {
                                a.g_load(r::RAX, r::RCX, jl.off_shape);
                                a.g_movabs(r::R11, shape);
                                a.g_cmp(r::RAX, r::R11);
                                a.g_jcc(cc::NE, skip);
                            }
                            a.g_load(r::RAX, r::RBX, reg(recv));
                            a.g_store(RBP, w.key, r::RAX);
                            a.g_store(RBP, w.cell, r::RCX);
                            a.bind(skip);
                            if w.shape.is_some() {
                                a.g_load(r::RAX, r::RBX, reg(recv));
                            }
                        }
                    }
                    a.bind(walked);
                    a.p_method_lookup_walked(jl, l.realm, site, miss, getter);
                }
                None => a.p_method_lookup(jl, l.realm, (r::RBX, reg(recv)), site, miss, getter),
            }
            if e.getter {
                a.jmp(miss); // a data property
                a.bind(got);
            }
        }
        _ => return None,
    }
    a.g_store(RBP, e.slots.callee, r::RAX);
    // 2. Its function: the memo (the same value at the same prototype
    //    epoch, see `CallMemo`), else the cell, memoized when it is `key`'s.
    let decode = a.new_label();
    a.g_movabs(r::R11, e.memo as u64);
    a.g_cmp_mem(r::RAX, r::R11, memo_key);
    a.g_jcc(cc::NE, decode);
    load_proto_epoch(a, r::R9);
    a.g_cmp_mem(r::R9, r::R11, memo_stamp);
    a.g_jcc(cc::NE, decode);
    a.g_load(r::R8, r::R11, memo_caps);
    a.jmp(identified);
    a.bind(decode);
    a.p_callee_info(jl, l.realm);
    a.g_movabs(r::R11, e.key);
    a.g_cmp(r::RAX, r::R11);
    a.g_jcc(cc::NE, miss);
    a.g_mov(r::R8, r::RDX);
    a.g_movabs(r::R11, e.memo as u64);
    a.g_load(r::RAX, RBP, e.slots.callee);
    a.g_store(r::R11, memo_key, r::RAX);
    load_proto_epoch(a, r::R9);
    a.g_store(r::R11, memo_stamp, r::R9);
    a.g_store(r::R11, memo_caps, r::R8);
    a.bind(identified);
    // 3. The caller's window on the register stack, one realm, room under
    //    both depth limits and for the header and the arguments (as
    //    `emit_call`) — unless no call since a guard checked them.
    let checked = a.new_label();
    a.g_load(r::RAX, RBP, e.slots.ok);
    a.g_cmp_imm(r::RAX, super::RECV_CACHE_EMPTY);
    a.g_jcc(cc::NE, checked);
    a.g_load(r::R10, r::R15, l.stack);
    a.g_cmp_mem(r::RBX, r::R10, STACK_TOP);
    a.g_jcc(cc::AE, miss);
    a.g_cmp_mem(r::RBX, r::R10, STACK_BASE);
    a.g_jcc(cc::B, miss);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_cmp_byte_imm(r::R9, l.multi_realm, 0);
    a.g_jcc(cc::NE, miss);
    a.g_load(r::RAX, r::R15, l.call_depth);
    a.g_test(r::RAX);
    a.g_jcc(cc::E, miss);
    a.g_cmp_mem(r::RAX, r::R9, l.max_depth);
    a.g_jcc(cc::AE, miss);
    a.g_load(r::RAX, r::R9, l.total_depth);
    a.g_cmp_mem(r::RAX, r::R9, l.max_depth);
    a.g_jcc(cc::AE, miss);
    a.g_load(r::RAX, r::R10, STACK_TOP);
    a.g_lea(
        r::RAX,
        r::RAX,
        hdr(FRAME_HEADER + argc.max(e.slots.max_argc))?,
    );
    a.g_cmp_mem(r::RAX, r::R10, STACK_LIMIT);
    a.g_jcc(cc::A, miss);
    a.g_store_imm32(RBP, e.slots.ok, 0);
    a.bind(checked);
    // 3b. A `new`: `this`, as the call's `[[Construct]]` would make it.
    let n_caps = e.n_caps - u16::from(e.new_target.is_some());
    if let Some(ctor) = e.new_target {
        emit_inline_new_this(a, e, ctor, e.caps.checked_add(n_caps)?, miss)?;
    }
    // 4. Pending, its captures in the inlined body's registers.
    a.g_lea_label(r::RAX, e.stub);
    a.g_store(RBP, e.slots.pend, r::RAX);
    for i in 0..n_caps {
        a.g_load(r::RAX, r::R8, hdr(usize::from(i))?);
        a.g_store(r::RBX, reg(e.caps.checked_add(i)?), r::RAX);
    }
    a.g_movabs(r::RAX, NanBox::boolean(true).to_bits());
    a.g_store(r::RBX, reg(e.dst), r::RAX);
    a.jmp(done);
    a.bind(miss);
    a.g_movabs(r::RAX, NanBox::boolean(false).to_bits());
    a.g_store(r::RBX, reg(e.dst), r::RAX);
    a.bind(done);
    Some(())
}

/// An inlined `new`'s `this` (`emit_inline_guard`, `e.new_target` the
/// constructor register `ctor`): the call's [`ConstructFast`] record must
/// hold for this constructor as its own `new.target`, at the current
/// prototype epoch, for a base constructor whose way is `e.key`'s — so the
/// construct is exactly "allocate an ordinary object whose `[[Prototype]]`
/// is `proto` (`new.target.prototype`, which the record proves fixed), then
/// run the body" (`OrdinaryCreateFromConstructor` observes nothing else) —
/// and the object is allocated (`p_alloc_object`, else `e.new_alloc`'s
/// helper, else a miss), into register `this` (and `e.new_way`). Otherwise
/// to `miss`, having changed nothing observable (the call op makes the
/// construct). Keeps `r8` (the captures' pointer).
fn emit_inline_new_this(
    a: &mut X64Assembler,
    e: &InlineEmit<'_>,
    ctor: u16,
    this: u16,
    miss: Label,
) -> Option<()> {
    use core::mem::offset_of;
    let reg = |x: u16| i32::from(x) * 8;
    let fast = offset_of!(CallSite, fast);
    let f = |o: usize| site_off(fast + o);
    a.g_movabs(r::R11, e.memo as u64);
    a.g_load(r::RAX, r::RBX, reg(ctor));
    a.g_cmp_mem(r::RAX, r::R11, f(offset_of!(ConstructFast, ctor)));
    a.g_jcc(cc::NE, miss);
    a.g_cmp_mem(r::RAX, r::R11, f(offset_of!(ConstructFast, nt)));
    a.g_jcc(cc::NE, miss);
    load_proto_epoch(a, r::R9);
    a.g_cmp_mem(r::R9, r::R11, f(offset_of!(ConstructFast, stamp)));
    a.g_jcc(cc::NE, miss);
    a.g_load(r::RAX, r::R11, f(offset_of!(ConstructFast, derived)));
    a.g_test(r::RAX);
    a.g_jcc(cc::NE, miss);
    a.g_load(r::R9, r::R11, f(offset_of!(ConstructFast, target)));
    a.g_test(r::R9);
    a.g_jcc(cc::E, miss);
    a.g_movabs(r::RAX, e.key);
    a.g_cmp_mem(
        r::RAX,
        r::R9,
        i32::try_from(offset_of!(CallTarget, key)).ok()?,
    );
    a.g_jcc(cc::NE, miss);
    a.g_load(r::RSI, r::R11, f(offset_of!(ConstructFast, proto)));
    let made = a.new_label();
    let by_helper = a.new_label();
    a.p_alloc_object(
        super::compute_jit_layout(),
        e.layout.realm,
        if e.new_alloc.is_some() {
            by_helper
        } else {
            miss
        },
    );
    // (`rcx`: the fresh object's heap slot.)
    if let Some(w) = e.new_way {
        a.g_store(RBP, w.key, r::RAX);
        a.g_store(RBP, w.cell, r::RCX);
    }
    if let Some(alloc) = e.new_alloc {
        a.jmp(made);
        // The free list is empty: the allocation the call's sequence would
        // make (`rsi` kept), which may collect — before the activation,
        // as there. The captures' pointer is reloaded from the memo.
        a.bind(by_helper);
        a.g_mov(r::RDI, r::R15);
        a.g_movabs(r::RAX, alloc);
        a.call_rax();
        // Growing the heap (or collecting) moves every cell and bumps the
        // epoch, leaving the memo's captures' pointer stale: then miss (the
        // object just made is garbage, unobservable).
        a.g_movabs(r::R11, e.memo as u64);
        load_proto_epoch(a, r::R9);
        a.g_cmp_mem(r::R9, r::R11, f(offset_of!(ConstructFast, stamp)));
        a.g_jcc(cc::NE, miss);
        a.bind(made);
        a.g_movabs(r::R11, e.memo as u64);
        a.g_load(
            r::R8,
            r::R11,
            site_off(offset_of!(CallSite, ident) + offset_of!(IdentMemo, value)),
        );
    }
    a.g_store(r::RBX, reg(this), r::RAX);
    Some(())
}

/// Emits an inlined call's materialization stub: called (by
/// [`X64Assembler::inline_frame_hook`]) with the call's activation pending,
/// it pushes the native-call header and the arguments (the caller's
/// registers `args`; `new_target` the register a `new`'s header records)
/// and bumps the depth counts exactly as [`emit_call`] does, records the
/// header and clears `pend`. Preserves every register.
pub(crate) fn emit_inline_stub(
    a: &mut X64Assembler,
    l: &CallLayout,
    s: &InlineSlots,
    args: &[u16],
    new_target: Option<u16>,
) -> Option<()> {
    use crate::nanbox::NanBox;
    let reg = |x: u16| i32::from(x) * 8;
    let hdr = |i: usize| i32::try_from(i * 8).ok();
    // push rax, rcx, rdx, r9, r10.
    a.raw(&[0x50, 0x51, 0x52, 0x41, 0x51, 0x41, 0x52]);
    a.g_load(r::R10, r::R15, l.stack);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_load(r::RCX, r::R10, STACK_TOP);
    a.g_lea(r::RAX, r::RCX, hdr(FRAME_HEADER + args.len())?);
    a.g_load(r::RDX, r::R10, STACK_FRAMES);
    a.g_store(r::RCX, 0, r::RDX);
    a.g_load(r::RDX, RBP, s.callee);
    a.g_store(r::RCX, 8, r::RDX);
    a.g_load(r::RDX, r::R9, l.callee_len);
    a.g_store(r::RCX, 16, r::RDX);
    a.g_store_imm32(r::RCX, 24, i32::try_from(args.len()).ok()?);
    match new_target {
        Some(nt) => a.g_load(r::RDX, r::RBX, reg(nt)),
        None => a.g_movabs(r::RDX, NanBox::undefined().to_bits()),
    }
    a.g_store(r::RCX, 32, r::RDX);
    for (i, &ar) in args.iter().enumerate() {
        a.g_load(r::RDX, r::RBX, reg(ar));
        a.g_store(r::RCX, hdr(FRAME_HEADER + i)?, r::RDX);
    }
    a.g_store(r::R10, STACK_FRAMES, r::RCX);
    a.g_store(r::R10, STACK_TOP, r::RAX);
    a.g_inc_mem(r::R15, l.call_depth);
    a.g_inc_mem(r::R9, l.total_depth);
    a.g_inc_mem(r::R15, l.frames_published);
    a.g_store(RBP, s.hdr, r::RCX);
    a.g_store_imm32(RBP, s.pend, super::RECV_CACHE_EMPTY);
    // pop r10, r9, rdx, rcx, rax; ret.
    a.raw(&[0x41, 0x5a, 0x41, 0x59, 0x5a, 0x59, 0x58, 0xc3]);
    Some(())
}

/// Emits an inlined activation's end (`Op::InlineLeave`), or the end of one
/// a throw left (at the body's throw exit, where none may be active): a
/// pending one is dropped; a materialized one's header is popped and the
/// depth counts restored, as the call sequence does after its callee
/// returns. Preserves `rax`.
pub(crate) fn emit_inline_leave(a: &mut X64Assembler, l: &CallLayout, s: &InlineSlots) {
    let materialized = a.new_label();
    let done = a.new_label();
    a.g_load(r::RCX, RBP, s.pend);
    a.g_cmp_imm(r::RCX, super::RECV_CACHE_EMPTY);
    a.g_jcc(cc::E, materialized);
    a.g_store_imm32(RBP, s.pend, super::RECV_CACHE_EMPTY);
    a.jmp(done);
    a.bind(materialized);
    a.g_load(r::RCX, RBP, s.hdr);
    a.g_cmp_imm(r::RCX, super::RECV_CACHE_EMPTY);
    a.g_jcc(cc::E, done);
    a.g_load(r::R10, r::R15, l.stack);
    a.g_store(r::R10, STACK_TOP, r::RCX);
    a.g_load(r::RDX, r::RCX, 0);
    a.g_store(r::R10, STACK_FRAMES, r::RDX);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_dec_mem(r::R15, l.call_depth);
    a.g_dec_mem(r::R9, l.total_depth);
    a.g_dec_mem(r::R15, l.frames_published);
    a.g_store_imm32(RBP, s.hdr, super::RECV_CACHE_EMPTY);
    a.bind(done);
}

impl X64Assembler {
    /// Makes every later `call` first run a pending inlined activation's
    /// materialization stub, whose address is in `[rbp + slot]` (`None`: no
    /// hook), and the frame prologue empty the slots `clear`.
    pub(crate) fn set_inline_pend(&mut self, slot: Option<i32>, clear: alloc::vec::Vec<i32>) {
        self.inline_pend = slot;
        self.inline_clear = clear;
    }

    /// Before a `call` in a body that inlines calls: `if [rbp + pend] !=
    /// empty { call [rbp + pend] }` (see [`emit_inline_guard`]). Changes no
    /// register.
    pub(crate) fn inline_frame_hook(&mut self) {
        if let Some(d) = self.inline_pend {
            // cmp qword [rbp+d], -1 ; je +7 ; call qword [rbp+d] (7 bytes)
            self.g_mem(true, &[0x83], 7, RBP, d);
            self.code.push(0xff);
            self.code.extend_from_slice(&[0x74, 0x07]);
            self.g_mem(false, &[0xff], 2, RBP, d);
        }
    }

    /// `lea reg, [rip + label]`.
    pub(crate) fn g_lea_label(&mut self, reg: u8, label: Label) {
        self.code.push(0x48 | ((reg >> 3) & 1) << 2);
        self.code.push(0x8d);
        self.code.push(0x05 | ((reg & 7) << 3));
        self.emit_rel32(label);
    }
}
