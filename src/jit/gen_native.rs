//! A compiled `g.next(v)` that resumes a compiled sync generator body in
//! native code (`KATAAN_JIT_GENNATIVE`): the call site jumps straight into the
//! body's resumption point on the register file kept in the generator's frame
//! cell, with none of the Rust resumption path (`nbvm::gen_fast::lean_next`)
//! in between.
//!
//! The site's [`GenMemo`] is filled by that Rust path once it resumed a hot
//! generator there (`gen_fast::native_fill`). A way records what the path
//! proved for the generators of one function:
//!
//! * the receiver: an ordinary object of the generator layout (`shape`, one
//!   own property — the hidden frame slot) whose `[[Prototype]]` is `proto`,
//!   at prototype epoch `stamp` the `[[Get]]` of `next` from it finding the
//!   data property whose value is at `value`, which still holds the
//!   intrinsic `%GeneratorPrototype%.next` (`method`) — exactly the
//!   `site_frame` proof (while the epoch stands no watched object of the
//!   chain changed its keys or layout and no cell moved);
//! * the frame: function `func`, suspended at a plain `yield` (the table
//!   `tab` of the compiled body `code` has an entry for its resumption pc —
//!   one whose op before is the `Op::YieldResume`), its register file kept
//!   at the body's size `n_regs`, no handler on its stack.
//!
//! The sequence then does what the Rust path does for such a resumption: the
//! frame marked executing and listed among the running ones
//! (`Realm::vm_gen_native`, which every collection roots with its register
//! file, compiled code storing into it under no barrier), the realm's depth
//! counted, a handler segment of the body's own opened and the register file
//! declared rooted (`TierRuntime::rooted_regs`, so the body may make native
//! calls), and the body entered at its resumption stub with the sent value
//! (in `Suspension::value`), which the stub lands in the `yield`'s
//! destination register (and `0`, a `next`, in its mode register) before it
//! continues at the resumption pc's entry. The frame cell must need no write
//! barrier for that (young, or due a whole rescan by its next minor
//! collection; no incremental marking), as the Rust path's mutable borrow of
//! it would have made it.
//!
//! When the body suspends again at a plain `yield` (an entry in the table
//! for the new pc), with its handler segment empty, no sweep since the entry
//! (the heap's handle epoch: the cell still young or remembered, as the Rust
//! path's reuse of its first borrow requires) and no incremental marking,
//! the frame records the suspension in place and the iterator result is
//! made — the site's one reused result object (`CallSite::iter_reuse`)
//! when its results never escape, written in line when that needs no
//! barrier. Every other outcome — a `yield*` step, a return, a throw, a
//! deoptimization exit — goes to `jit_helper_gen_native_finish`, which runs
//! the Rust path's ending of the resumption from the same state.
//!
//! A way also counts down: every [`RUNS`]th resumption through it takes the
//! Rust path, which counts the function's resumptions (its recompilation
//! cadence) and refills the way.

use super::call_seq::{CallLayout, CallSite, cc, r};
use super::{Label, X64Assembler};
use core::mem::offset_of;

/// The resumptions a way serves before the Rust path takes one (and
/// refills it).
pub(crate) const RUNS: i64 = 64;

/// What one way of a [`GenMemo`] proved (see the module docs). `#[repr(C)]`:
/// emitted code reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GenWay {
    /// The generator object's shape word (`0`: an empty way).
    pub shape: u64,
    /// Its `[[Prototype]]` (a raw handle).
    pub proto: u64,
    /// The prototype epoch of the proof.
    pub stamp: u64,
    /// The address of `next`'s value on the prototype chain.
    pub value: u64,
    /// `%GeneratorPrototype%.next`'s bits.
    pub method: u64,
    /// The generator function's id.
    pub func: u64,
    /// Its compiled body's code address.
    pub code: u64,
    /// The register-file size the body runs on.
    pub n_regs: u64,
    /// The body's resumption stub table (`u32` code offsets by the
    /// function's own pc, `0` for none) and its length.
    pub tab: u64,
    pub tab_len: u64,
    /// Resumptions left before the Rust path takes one.
    pub countdown: i64,
}

impl GenWay {
    /// The stamp of a way recorded for generators the sequence declines (no
    /// epoch is ever that).
    pub const NO_STAMP: u64 = u64::MAX;

    /// An empty way.
    pub const EMPTY: Self = Self {
        shape: 0,
        proto: 0,
        stamp: 0,
        value: 0,
        method: 0,
        func: 0,
        code: 0,
        n_regs: 0,
        tab: 0,
        tab_len: 0,
        countdown: 0,
    };
}

/// Ways a site keeps (a scheduler's `gen.next(v)` resumes the generators of
/// a few functions).
pub(crate) const WAYS: usize = 2;

/// A `g.next(v)` call site's native resumption memo (see the module docs).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GenMemo {
    /// The ways.
    pub ways: [GenWay; WAYS],
    /// The way the next new function replaces.
    pub next: u64,
    /// The realm's iterator-result layout's shape word (`0` before one is
    /// known): what a reused result must still have.
    pub iter_shape: u64,
}

impl GenMemo {
    /// An empty memo.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ways: [GenWay::EMPTY; WAYS],
            next: 0,
            iter_shape: 0,
        }
    }

    /// Whether the way serving generator objects of `shape` and prototype
    /// `proto` wants no refill at prototype epoch `epoch`: it proved its
    /// facts at that epoch and is still counting down, or it is a declined
    /// one ([`GenWay::NO_STAMP`]) still counting down — which this counts.
    pub fn serving(&mut self, shape: u64, proto: u64, epoch: u64) -> bool {
        self.ways.iter_mut().any(|w| {
            if w.shape != shape || w.proto != proto || w.countdown <= 0 {
                return false;
            }
            if w.stamp == GenWay::NO_STAMP {
                w.countdown -= 1;
                return true;
            }
            w.stamp == epoch
        })
    }

    /// Records `way`: over the way of the same generator layout and
    /// prototype, else over the next one round-robin.
    pub fn fill(&mut self, way: GenWay) {
        let i = match self
            .ways
            .iter()
            .position(|w| w.shape == way.shape && w.proto == way.proto)
        {
            Some(i) => i,
            None => {
                let i = (self.next as usize) % WAYS;
                self.next = ((i + 1) % WAYS) as u64;
                i
            }
        };
        self.ways[i] = way;
    }
}

impl Default for GenMemo {
    fn default() -> Self {
        Self::new()
    }
}

/// Where the sequence finds the state it keeps (offsets; see the module
/// docs), probed by the VM (`jit_tier::call_layout`).
#[derive(Clone, Copy, Debug)]
pub struct GenLayout {
    /// The `Cell::Frame` discriminant byte.
    pub frame_disc: u8,
    /// `Cell`-relative offset of the frame box's pointer word.
    pub frame_box: i32,
    /// `VmFrame`-relative offsets: `func: u32`, `pc: u32`, `dst: u16`,
    /// `state: u8`, the register file's buffer pointer and length words and
    /// the handler stack's length word.
    pub f_func: i32,
    pub f_pc: i32,
    pub f_dst: i32,
    pub f_state: i32,
    pub f_regs_ptr: i32,
    pub f_regs_len: i32,
    pub f_handlers_len: i32,
    /// The generator states (`nbvm::GEN_*`): suspended at a `yield`,
    /// executing.
    pub s_yield: u8,
    pub s_start: u8,
    pub s_executing: u8,
    /// realm: `vm_gen_native` (`[u64; live_max]`) and its count (`u64`).
    pub live: i32,
    pub live_n: i32,
    pub live_max: i32,
    /// realm: the heap's handle epoch (`u32`) and its GC-verification
    /// state word (null while off).
    pub heap_epoch: i32,
    pub poison: i32,
    /// ctx: three words `jit_helper_gen_native_finish` reads — the body's
    /// raw outcome, the frame cell's handle bits and the caller's handler
    /// segment base.
    pub scratch: i32,
}

/// Whether compiled `g.next(v)` sites resume compiled generator bodies
/// natively (`KATAAN_JIT_GENNATIVE=0` turns it off, read once per process).
#[must_use]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_GENNATIVE").is_none_or(|v| v != "0"))
}

/// The bit a native resumption's entry word sets (an address otherwise; the
/// body's entry dispatch takes any other entry as a pc).
const ENTRY_BIT: u8 = 63;

/// `CallSite`-relative offset of the memo's field `f`.
fn memo_off(f: usize) -> i32 {
    i32::try_from(offset_of!(CallSite, gnat) + f).expect("site offset fits in i32")
}

/// `GenWay`-relative offset of a field.
fn way_off(f: usize) -> i32 {
    i32::try_from(f).expect("way offset fits in i32")
}

/// In a sync generator body's entry dispatch (the entry word in `rdx`): a
/// native resumption's entry (its top bit set) goes to `native`.
pub(crate) fn emit_dispatch_check(a: &mut X64Assembler, native: Label) {
    a.g_test(r::RDX);
    a.g_jcc(0x8, native); // js
}

/// The body's native dispatch (`native`, out of line, before the entry's
/// own `call_seq::emit_act_ok`): the activation may make native calls —
/// its register file is the one its caller declared rooted, and the
/// caller, compiled code that may itself make native calls, runs at a
/// nonzero depth, which no native call changes — and the entry word with
/// its top bit cleared is the resumption stub's address.
pub(crate) fn emit_native_dispatch(a: &mut X64Assembler, native: Label) {
    a.bind(native);
    super::call_seq::emit_act_ok_known(a);
    // btr rdx, 63; jmp rdx
    a.code
        .extend_from_slice(&[0x48, 0x0f, 0xba, 0xf2, ENTRY_BIT]);
    a.code.extend_from_slice(&[0xff, 0xe2]);
}

/// The bit of a stub table entry (`JitProto::gen_resume`) that marks a
/// start stub: the resumption pc of the body's `Op::GeneratorStart`, which
/// only a generator suspended at its start (`GEN_SUSPENDED_START`) is
/// resumed at, and where no sent value lands.
pub(crate) const START_STUB: u32 = 1 << 31;

/// A resumption stub (bound at `stub`): for a `yield` (`land` its
/// destination and mode registers), the sent value (`Suspension::value` at
/// `[r15 + susp + 8]`) into the destination, a `next` (`0`) into the mode
/// register; then on to the resumption pc's entry `entry` — what the Rust
/// path writes into the register file before it enters there (nothing, at
/// the generator's start).
pub(crate) fn emit_land_stub(
    a: &mut X64Assembler,
    stub: Label,
    susp: i32,
    land: Option<(u16, u16)>,
    entry: Label,
) {
    a.bind(stub);
    if let Some((dst, mode)) = land {
        a.g_load(r::RAX, r::R15, susp + 8);
        a.g_store(r::RBX, i32::from(dst) * 8, r::RAX);
        debug_assert_eq!(crate::nanbox::NanBox::number(0.0).to_bits(), 0);
        a.g_store_imm32(r::RBX, i32::from(mode) * 8, 0);
    }
    a.jmp(entry);
}

/// What [`emit_resume`] needs of its site.
pub(crate) struct ResumeEmit {
    /// The op's call site.
    pub site: *const CallSite,
    /// The receiver, argument and destination registers.
    pub recv: u16,
    pub arg: Option<u16>,
    pub dst: u16,
    /// Whether the site's iterator results never escape (one reused).
    pub reuse: bool,
    /// `jit_helper_gen_native_finish` and `jit_helper_gen_native_result`.
    pub finish: u64,
    pub result: u64,
    /// The helper call's arguments: the register count and the op.
    pub n_regs: usize,
    pub op: u64,
    /// Where anything the sequence does not take goes (the site's lean
    /// helper), with nothing done; where it ends with the result stored;
    /// the body's throw exit.
    pub slow: Label,
    pub done: Label,
    pub throw_exit: Label,
}

/// The native resumption of a `g.next(v)` site (see the module docs). Falls
/// to `e.slow` having changed nothing, else ends at `e.done` with the
/// iterator result in the destination register (or at `e.throw_exit`).
pub(crate) fn emit_resume(a: &mut X64Assembler, l: &CallLayout, e: &ResumeEmit) {
    let jl = super::compute_jit_layout();
    let g = &l.gens;
    let reg = |x: u16| i32::from(x) * 8;
    let slow = e.slow;
    let wsz = way_off(core::mem::size_of::<GenWay>());
    let w = |f: usize| way_off(f);
    a.dbg_mark("g-native");
    // 1. The limits a native call checks (`r9` = the realm), and a hosted
    //    run (the Rust path's condition).
    super::call_seq::emit_limit_checks(a, l, slow);
    a.g_cmp_byte_imm(r::R15, l.hosted, 0);
    a.g_jcc(cc::E, slow);
    // 2. The receiver (`rcx` its slot) and the way of its layout and
    //    prototype (`rsi`).
    a.p_walk_plain(jl, l.realm, (r::RBX, reg(e.recv)), slow);
    a.g_load(r::RAX, r::RCX, jl.off_shape);
    a.g_load(r::RDX, r::RCX, jl.props.off_proto);
    a.g_movabs(
        r::RSI,
        e.site as u64 + memo_off(offset_of!(GenMemo, ways)) as u64,
    );
    let found = a.new_label();
    for k in 0..WAYS {
        let next = a.new_label();
        let at = wsz * i32::try_from(k).unwrap_or(0);
        a.g_cmp_mem(r::RAX, r::RSI, at + w(offset_of!(GenWay, shape)));
        a.g_jcc(cc::NE, next);
        a.g_cmp_mem(r::RDX, r::RSI, at + w(offset_of!(GenWay, proto)));
        a.g_jcc(cc::NE, next);
        if at != 0 {
            a.g_lea(r::RSI, r::RSI, at);
        }
        a.jmp(found);
        a.bind(next);
    }
    a.jmp(slow);
    a.bind(found);
    // The proof stands: same epoch, no own accessor, `next` still the
    // intrinsic.
    a.g_load_abs(r::R9, crate::object::proto_epoch_addr() as u64);
    a.g_cmp_mem(r::R9, r::RSI, w(offset_of!(GenWay, stamp)));
    a.g_jcc(cc::NE, slow);
    super::call_seq::emit_no_accessors(a, slow);
    a.g_load(r::R9, r::RSI, w(offset_of!(GenWay, value)));
    a.g_load(r::RAX, r::R9, 0);
    a.g_cmp_mem(r::RAX, r::RSI, w(offset_of!(GenWay, method)));
    a.g_jcc(cc::NE, slow);
    // 3. The frame cell (its one own slot: `r8` its handle bits, `rcx` its
    //    heap slot, `rdi` the boxed frame).
    a.g_load(r::R8, r::RCX, jl.off_inline);
    a.p_arena(l.realm, &jl.props);
    a.g_mov(r::RAX, r::R8);
    a.p_cmp_handle_floor(r::RAX);
    a.g_jcc(cc::B, slow);
    a.p_slot_rooted(jl, r::RCX, slow);
    a.g_cmp_byte_imm(r::RCX, jl.off_cell_tag, g.frame_disc);
    a.g_jcc(cc::NE, slow);
    a.g_load(r::RDI, r::RCX, jl.off_cell_tag + g.frame_box);
    // mov eax, [rdi + func] (zero-extended)
    a.g_mem(false, &[0x8b], r::RAX, r::RDI, g.f_func);
    a.g_cmp_mem(r::RAX, r::RSI, w(offset_of!(GenWay, func)));
    a.g_jcc(cc::NE, slow);
    a.g_load(r::RAX, r::RDI, g.f_regs_len);
    a.g_cmp_mem(r::RAX, r::RSI, w(offset_of!(GenWay, n_regs)));
    a.g_jcc(cc::NE, slow);
    a.g_mem(true, &[0x83], 7, r::RDI, g.f_handlers_len); // cmp qword [rdi+len], 0
    a.code.push(0);
    a.g_jcc(cc::NE, slow);
    // The resumption stub of its pc (`rdx`, a code offset).
    a.g_mem(false, &[0x8b], r::RAX, r::RDI, g.f_pc); // mov eax, [rdi + pc]
    a.g_cmp_mem(r::RAX, r::RSI, w(offset_of!(GenWay, tab_len)));
    a.g_jcc(cc::AE, slow);
    a.g_load(r::RDX, r::RSI, w(offset_of!(GenWay, tab)));
    a.code.extend_from_slice(&[0x8b, 0x14, 0x82]); // mov edx, [rdx + rax*4]
    // Suspended at a `yield` with a landing stub there, or at its start
    // with a start stub there.
    let at_start = a.new_label();
    let stub_ok = a.new_label();
    a.g_cmp_byte_imm(r::RDI, g.f_state, g.s_yield);
    a.g_jcc(cc::NE, at_start);
    a.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
    a.g_jcc(cc::E, slow);
    a.g_jcc(0x8, slow); // js
    a.jmp(stub_ok);
    a.bind(at_start);
    a.g_cmp_byte_imm(r::RDI, g.f_state, g.s_start);
    a.g_jcc(cc::NE, slow);
    a.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
    a.g_jcc(0x9, slow); // jns
    a.code.extend_from_slice(&[0x81, 0xe2]); // and edx, !START_STUB
    a.code.extend_from_slice(&(!START_STUB).to_le_bytes());
    a.bind(stub_ok);
    // The cell needs no barrier: no incremental marking, and young or due a
    // whole rescan.
    a.g_load(r::R9, r::R15, l.realm);
    a.g_cmp_byte_imm(r::R9, jl.props.marking, 0);
    a.g_jcc(cc::NE, slow);
    let clean = a.new_label();
    a.g_cmp_byte_imm(r::RCX, jl.props.off_slot_age, crate::gc::OLD_AGE);
    a.g_jcc(cc::B, clean);
    a.p_test_byte_imm(
        r::RCX,
        jl.props.off_slot_dirty,
        crate::heap::JIT_DIRTY_WHOLE,
    );
    a.g_jcc(cc::E, slow);
    a.bind(clean);
    // Room on the running list; the way's countdown.
    a.g_load(r::RAX, r::R9, g.live_n);
    a.g_cmp_imm(r::RAX, g.live_max);
    a.g_jcc(cc::AE, slow);
    a.g_dec_mem(r::RSI, w(offset_of!(GenWay, countdown)));
    a.g_jcc(0xe, slow); // jle
    // 4. The resumption: the sent value, the state, the running list, the
    //    depth.
    a.g_mov(r::R11, r::RSI);
    match e.arg {
        Some(v) => a.g_load(r::RSI, r::RBX, reg(v)),
        None => a.g_movabs(r::RSI, crate::nanbox::NanBox::undefined().to_bits()),
    }
    a.g_store(r::R15, l.susp + 8, r::RSI);
    a.g_store_byte_imm(r::RDI, g.f_state, g.s_executing);
    a.g_lea(r::R10, r::R9, g.live);
    a.code.extend_from_slice(&[0x4d, 0x89, 0x04, 0xc2]); // mov [r10 + rax*8], r8
    a.g_inc_mem(r::R9, g.live_n);
    a.g_inc_mem(r::R9, l.total_depth);
    // What the end needs, on the machine stack (six words: the alignment
    // stays): the frame, its handle, the caller's handler segment base and
    // rooted file, the heap epoch, the way.
    let push = |a: &mut X64Assembler, reg: u8| {
        if reg >= 8 {
            a.code.push(0x41);
        }
        a.code.push(0x50 | (reg & 7));
    };
    push(a, r::RDI);
    push(a, r::R8);
    a.g_mem(false, &[0xff], 6, r::R15, l.handlers_base); // push qword [..]
    a.g_mem(false, &[0xff], 6, r::R15, l.rooted);
    a.g_mem(false, &[0x8b], r::RAX, r::R9, g.heap_epoch); // mov eax, [epoch]
    push(a, r::RAX);
    push(a, r::R11);
    // The body's own handler segment; its register file rooted.
    a.g_load(r::RAX, r::R15, l.handlers_len);
    a.g_store(r::R15, l.handlers_base, r::RAX);
    a.g_load(r::RSI, r::RDI, g.f_regs_ptr);
    a.g_store(r::R15, l.rooted, r::RSI);
    // Enter at the stub: (ctx, regs, the entry word).
    a.g_add_mem(r::RDX, r::R11, w(offset_of!(GenWay, code)));
    a.code
        .extend_from_slice(&[0x48, 0x0f, 0xba, 0xea, ENTRY_BIT]); // bts rdx, 63
    a.g_mov(r::RDI, r::R15);
    a.g_load(r::RAX, r::R11, w(offset_of!(GenWay, code)));
    a.call_rax();
    // 5. The end (`rax` the body's outcome).
    let pop = |a: &mut X64Assembler, reg: u8| {
        if reg >= 8 {
            a.code.push(0x41);
        }
        a.code.push(0x58 | (reg & 7));
    };
    pop(a, r::R11); // the way
    pop(a, r::R10); // the epoch
    pop(a, r::R9); // the caller's rooted file
    pop(a, r::R8); // the caller's handler base
    pop(a, r::RSI); // the frame's handle
    pop(a, r::RDI); // the frame
    a.g_store(r::R15, l.rooted, r::R9);
    a.g_load(r::R9, r::R15, l.realm);
    a.g_dec_mem(r::R9, g.live_n);
    a.g_dec_mem(r::R9, l.total_depth);
    let fin = a.new_label();
    // A plain `yield` (the exit sentinel, a suspension pending at a pc with
    // a resumption stub), the handler segment empty, no sweep since the
    // entry, no incremental marking.
    a.g_movabs(r::RCX, crate::nanbox::NanBox::jit_exit_bits());
    a.g_cmp(r::RAX, r::RCX);
    a.g_jcc(cc::NE, fin);
    a.g_load(r::RCX, r::R15, l.susp);
    a.g_cmp_imm(r::RCX, -1);
    a.g_jcc(cc::E, fin);
    a.g_load(r::RDX, r::R15, l.handlers_len);
    a.g_cmp_mem(r::RDX, r::R15, l.handlers_base);
    a.g_jcc(cc::NE, fin);
    a.g_cmp_mem(r::RCX, r::R11, w(offset_of!(GenWay, tab_len)));
    a.g_jcc(cc::AE, fin);
    a.g_load(r::RDX, r::R11, w(offset_of!(GenWay, tab)));
    a.code.extend_from_slice(&[0x8b, 0x14, 0x8a]); // mov edx, [rdx + rcx*4]
    a.code.extend_from_slice(&[0x85, 0xd2]); // test edx, edx
    a.g_jcc(cc::E, fin);
    a.g_jcc(0x8, fin); // js (a start stub's pc)
    a.g_mem(false, &[0x8b], r::RDX, r::R9, g.heap_epoch); // mov edx, [epoch]
    a.g_cmp(r::RDX, r::R10);
    a.g_jcc(cc::NE, fin);
    a.g_cmp_byte_imm(r::R9, jl.props.marking, 0);
    a.g_jcc(cc::NE, fin);
    // The suspension, recorded in the frame.
    a.g_mem(false, &[0x89], r::RCX, r::RDI, g.f_pc); // mov [rdi + pc], ecx
    a.g_load(r::RDX, r::R15, l.susp + 16);
    a.code.push(0x66); // mov [rdi + dst], dx
    a.g_mem(false, &[0x89], r::RDX, r::RDI, g.f_dst);
    a.g_store_byte_imm(r::RDI, g.f_state, g.s_yield);
    a.g_store_imm32(r::R15, l.susp, -1);
    a.g_store(r::R15, l.handlers_base, r::R8);
    // 6. The iterator result (the yielded value in `rsi`).
    a.g_load(r::RSI, r::R15, l.susp + 8);
    let by_helper = a.new_label();
    if e.reuse {
        // The site's result object, while it lives with the result layout,
        // the store needing no barrier (not in GC-verification mode, where
        // a freed object stays in its slot).
        a.g_mem(true, &[0x83], 7, r::R9, g.poison); // cmp qword [r9+poison], 0
        a.code.push(0);
        a.g_jcc(cc::NE, by_helper);
        a.p_arena(l.realm, &jl.props);
        a.g_movabs(r::R11, e.site as u64);
        a.g_load(
            r::RAX,
            r::R11,
            i32::try_from(offset_of!(CallSite, iter_reuse)).unwrap_or(0),
        );
        a.p_cmp_handle_floor(r::RAX);
        a.g_jcc(cc::B, by_helper);
        a.p_slot_of(jl, by_helper);
        a.p_object_not_exotic(jl, by_helper);
        a.g_load(r::RAX, r::RCX, jl.off_shape);
        a.g_movabs(r::R11, e.site as u64);
        a.g_cmp_mem(r::RAX, r::R11, memo_off(offset_of!(GenMemo, iter_shape)));
        a.g_jcc(cc::NE, by_helper);
        a.g_mov(r::RAX, r::RSI);
        a.p_barrier_check(jl, l.realm, by_helper);
        a.g_store(r::RCX, jl.off_inline, r::RSI);
        a.g_movabs(r::RAX, crate::nanbox::NanBox::boolean(false).to_bits());
        a.g_store(r::RCX, jl.off_inline + 8, r::RAX);
        a.g_movabs(r::R11, e.site as u64);
        a.g_load(
            r::RAX,
            r::R11,
            i32::try_from(offset_of!(CallSite, iter_reuse)).unwrap_or(0),
        );
        a.g_store(r::RBX, reg(e.dst), r::RAX);
        a.jmp(e.done);
    }
    // (Else, or when the reused one will not do: the helper's result.)
    let cold = e.reuse.then(|| a.cold_begin());
    a.bind(by_helper);
    a.g_mov(r::RDI, r::R15);
    a.g_movabs(r::RDX, e.site as u64);
    a.g_movabs(r::RCX, u64::from(e.reuse));
    a.g_movabs(r::RAX, e.result);
    a.call_rax();
    a.g_store(r::RBX, reg(e.dst), r::RAX);
    a.jmp(e.done);
    if let Some(c) = cold {
        a.cold_end(c);
    }
    // Any other end: the Rust path's (it stores the result).
    // (`rax`, `rsi` and `r8` as the end left them.)
    let cold = a.cold_begin();
    a.bind(fin);
    a.g_store(r::R15, g.scratch, r::RAX);
    a.g_store(r::R15, g.scratch + 8, r::RSI);
    a.g_store(r::R15, g.scratch + 16, r::R8);
    a.g_mov(r::RDI, r::R15);
    a.g_mov(r::RSI, r::RBX);
    a.g_movabs(r::RDX, e.n_regs as u64);
    a.g_movabs(r::RCX, e.op);
    a.g_movabs(r::R8, u64::from(e.reuse));
    a.g_movabs(r::R9, e.site as u64);
    a.g_movabs(r::RAX, e.finish);
    a.call_rax();
    a.p_cmp_throw_rax();
    a.g_jcc(cc::E, e.throw_exit);
    a.jmp(e.done);
    a.cold_end(cold);
}
