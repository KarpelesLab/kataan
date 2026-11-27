//! The generic (NanBox) tier on AArch64: a **baseline** body over the frame
//! ABI, `extern "C" fn(ctx, regs, entry) -> u64` (see
//! [`JitFunction::compile_generic`](super::JitFunction::compile_generic) for
//! the contract the interpreter relies on), compiled straight from the
//! function's bytecode.
//!
//! What it lowers natively: constants and moves, the branches (with the
//! back-edge poll on every backward one), `return`, the Number fast paths of
//! `+ - * /`, `%`, the comparisons, `===`, `!`, unary `-`, `++`/`--` and the
//! bitwise operators and shifts (the NaN results made the canonical NaN);
//! and inline over the heap layout the shared probe (`compute_jit_layout`)
//! describes: a property read's monomorphic hit through its site's inline
//! cache, a global binding through a cached address checked against the
//! global binding epoch, a plain array's present element and `.length`, a
//! captured variable's box. Their misses, property writes and method calls
//! call the x86-64 tier's runtime helpers (each site with its own caches,
//! [`Sites`]); every other op is the interpreter's own, run on the same
//! register file by `jit_helper_exec_op`. So the body has the interpreter's
//! semantics by construction. A body that no `try` region, suspension or
//! early `new.target` keeps from it has a native-call entry, which
//! `jit_tier::call_compiled_with` enters a compiled callee through. `try` regions keep their
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
//! survive the helpers); `x9..x15`, `d0..d2` are scratch. The native frame
//! holds no values (everything lives in the register file the collector
//! roots), so a collection in any helper is safe at any point.

// Only an aarch64 build runs this tier (the emitter is built and tested
// everywhere).
#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use super::Label;
use super::a64::{A64Assembler, Cond, FP, LR, SP, Shift};
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
    /// `jit_helper_get_prop(ctx, obj, key, key_len, cache) -> value | throw`.
    pub get: u64,
    /// `jit_helper_set_prop(ctx, obj, key, key_len, cache, value)` (sloppy).
    pub set: u64,
    /// [`set`](Self::set) for strict code.
    pub set_strict: u64,
    /// `jit_helper_init_prop`: a literal's property definition (as `set`).
    pub init_prop: u64,
    /// `jit_helper_call_method(ctx, regs, n_regs, op, cache, null) -> 0 | throw`.
    pub call_method: u64,
    /// `jit_helper_global_ref(ctx, op) -> binding address | 0`.
    pub global_ref: u64,
    /// `jit_helper_get_elem(ctx, recv, key) -> value | throw`.
    pub get_elem: u64,
    /// `jit_helper_set_elem(ctx, recv, key, value, strict) -> _ | throw`.
    pub set_elem: u64,
    /// `jit_helper_array_len(ctx, recv) -> value | throw`.
    pub array_len: u64,
    /// `jit_helper_get_method_site(ctx, obj, site, call) -> value | throw`.
    pub get_method_site: u64,
    /// `jit_helper_call_with_method(ctx, regs, n_regs, op, cache, null)`.
    pub call_with_method: u64,
    /// `jit_helper_strict_eq(ctx, a, b) -> boolean` (never throws).
    pub strict_eq: u64,
    /// `jit_helper_add(ctx, a, b) -> value | throw` (`+`).
    pub add: u64,
    /// `jit_helper_arith(ctx, a, b, GA_*) -> value | throw` (`- * / %`).
    pub arith: u64,
    /// `jit_helper_value_bin(ctx, a, b, VB_*) -> value | throw`.
    pub value_bin: u64,
    /// `jit_helper_call_value(ctx, regs, n_regs, op, site, _) -> 0 | throw`:
    /// a `CallValue`'s slow path, which also points the site at the callee.
    pub call_value: u64,
    /// `jit_helper_construct(ctx, regs, n_regs, op, site, _) -> 0 | throw`.
    pub construct: u64,
    /// `jit_construct_info(ctx, ctor, new_target, out, site) -> CalleeInfo`.
    pub construct_info: u64,
    /// `jit_helper_construct_result(ctx, result, this) -> value | throw`.
    pub construct_result: u64,
}

/// The heap layout the inline property read walks (every offset from the
/// shared probe, `compute_jit_layout`; never written by hand).
#[derive(Clone, Copy)]
pub(crate) struct Layout {
    /// `Ctx`-relative offset of the `Box<Realm>` pointer.
    pub realm: i32,
    /// `Realm`-relative offsets of the heap arena's slot pointer and length.
    pub heap_ptr: i32,
    pub heap_len: i32,
    /// The arena's slot stride.
    pub stride: i32,
    /// Slot-relative offsets and tags (see `JitLayout`).
    pub off_cell_tag: i32,
    pub cell_object_disc: u8,
    pub off_od_tag: i32,
    pub obj_shaped_disc: u8,
    pub off_jit_flags: i32,
    pub off_shape: i32,
    pub off_slots_len: i32,
    pub off_inline: i32,
    pub off_extra: i32,
    /// `ObjectExtra`-relative offsets of the near spill and the far spill's
    /// data pointer.
    pub extra_near: i32,
    pub extra_far_ptr: i32,
    /// `PropertyCache`-relative offsets of the cached shape and slot.
    pub cache_shape_off: i32,
    pub cache_slot_off: i32,
    /// A `Cell::Array`'s tag, side-state flag byte, inline element count
    /// byte (`ARRAY_ON_HEAP`: the `Vec` holds them), first inline element,
    /// and its `Vec`'s data pointer and length.
    pub cell_array_disc: u8,
    pub off_arr_side: i32,
    pub off_arr_inline_len: i32,
    pub off_arr_inline: i32,
    pub off_arr_ptr: i32,
    pub off_arr_len: i32,
    /// A `Cell::Box`'s tag and value.
    pub box_disc: u8,
    pub off_box_val: i32,
    /// The global binding epoch's address (`env`).
    pub global_epoch: u64,
    /// The flags-byte bits that keep a store out of line
    /// (`props::FLAGS_NO_INLINE_STORE`), and the shape's brand byte (bit 0:
    /// a primitive wrapper), shape-word-relative.
    pub no_inline_store: u8,
    pub shape_brands: i32,
    /// `Realm`-relative: the incremental marker's running byte; slot-
    /// relative: the GC age byte and the dirty-state byte.
    pub marking: i32,
    pub off_slot_age: i32,
    pub off_slot_dirty: i32,
    /// The native call sequence's: `ctx`-relative the `*mut JitStack` and
    /// `call_depth`; `Realm`-relative `vm_total_depth`, the call depth
    /// limit, `multi_realm` and the header's realm mark word (`vm_mark`).
    pub stack: i32,
    pub call_depth: i32,
    pub total_depth: i32,
    pub max_depth: i32,
    pub multi_realm: i32,
    pub vm_mark: i32,
    /// A `Cell::VmFunction`'s tag, `func: u32`, and its captures: inline
    /// count byte (`ARRAY_ON_HEAP`: in the `Vec`), first inline capture,
    /// and the `Vec`'s data pointer and length.
    pub vmfn_disc: u8,
    pub off_vmfn_func: i32,
    pub off_vmfn_caps_inline_len: i32,
    pub off_vmfn_caps_inline: i32,
    pub off_vmfn_caps_ptr: i32,
    pub off_vmfn_caps_len: i32,
}

/// The per-site state a body's code points into: one `PropertyCache` per
/// property op ([`cache_sites`]), owned by the `JitProto` alongside the
/// code (the vector is never resized, so the addresses stay put).
#[derive(Default)]
pub(crate) struct Sites {
    pub caches: Vec<core::cell::UnsafeCell<crate::ic::PropertyCache>>,
    /// One `[epoch, binding address]` pair per global binding op: the
    /// address `jit_helper_global_ref` gave, good while the global binding
    /// epoch is the one noted.
    pub globals: Vec<core::cell::UnsafeCell<[u64; 2]>>,
    /// Per `GetMethod`: its method site (over its `caches` entry) and the
    /// call site its getter fills.
    pub methods: Vec<(
        core::cell::UnsafeCell<super::MethodSite>,
        core::cell::UnsafeCell<super::CallSite>,
    )>,
    /// Per `CallValue`: its call site (the callees the native call
    /// sequence enters, which its slow path fills).
    pub calls: Vec<core::cell::UnsafeCell<super::CallSite>>,
}

impl Sites {
    /// One fresh cache per property op and global op of `ops`.
    pub(crate) fn for_ops(ops: &[Op]) -> Self {
        let n = ops.iter().filter(|op| cache_op(op)).count();
        let g = ops.iter().filter(|op| global_op(op)).count();
        let caches: Vec<core::cell::UnsafeCell<crate::ic::PropertyCache>> =
            (0..n).map(|_| core::cell::UnsafeCell::default()).collect();
        // (The caches are in place for good: the method sites point at them.)
        let methods = ops
            .iter()
            .filter(|op| cache_op(op))
            .zip(caches.iter())
            .filter_map(|(op, c)| match op {
                Op::GetMethod { key, .. } => Some((
                    core::cell::UnsafeCell::new(super::MethodSite::new(key, c.get().cast())),
                    core::cell::UnsafeCell::new(super::CallSite::new()),
                )),
                _ => None,
            })
            .collect();
        let calls = ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Op::CallValue { .. } | Op::CallWithMethod { .. } | Op::Construct { .. }
                )
            })
            .map(|_| core::cell::UnsafeCell::new(super::CallSite::new()))
            .collect();
        Self {
            caches,
            calls,
            globals: (0..g)
                .map(|_| core::cell::UnsafeCell::new([0; 2]))
                .collect(),
            methods,
        }
    }
}

/// Whether `op` reads or writes a global binding through a [`Sites`] pair.
fn global_op(op: &Op) -> bool {
    matches!(op, Op::LoadGlobal { .. } | Op::StoreGlobal { .. })
}

/// Whether `op` gets an inline cache of its own ([`Sites`]).
fn cache_op(op: &Op) -> bool {
    matches!(
        op,
        Op::GetProp { .. }
            | Op::GetMethod { .. }
            | Op::SetProp { .. }
            | Op::SetPropStrict { .. }
            | Op::InitProp { .. }
            | Op::CallMethod { .. }
            | Op::CallWithMethod { .. }
    )
}

/// `crate::object::INLINE_SLOTS`, and where the far spill begins.
const INLINE: i64 = crate::object::INLINE_SLOTS as i64;
const FAR_FROM: i64 = (crate::object::INLINE_SLOTS + crate::object::SPILL_NEAR) as i64;

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
/// The register-window layout a body's native-call entry builds (see
/// [`compile`]): its parameter and capture counts, and the register it
/// reads `new.target` into, if any.
#[derive(Clone, Copy)]
pub(crate) struct NativeShape {
    pub n_params: usize,
    pub n_caps: usize,
    pub new_target: Option<usize>,
}

/// With `native`, the code also gets a **native-call entry** (its offset
/// is returned): `extern "C" fn(ctx, window, argc, argv, caps, this) -> u64`,
/// which `jit_tier::enter_native` calls on a window it reserved on the
/// register stack below the call's header and arguments. It lays the window
/// out as `call_with_inner` does — arguments `0..n_params` (missing ones
/// `undefined`), captures after them, `this` after those, `new.target`
/// (the header word just below `argv`) in its register, every other
/// register `undefined` — and runs the body from pc 0.
pub(crate) fn compile(
    ops: &[Op],
    n_regs: usize,
    (is_async, is_generator): (bool, bool),
    entries: &[usize],
    (h, l, sites): (&Helpers, &Layout, &Sites),
    native: Option<NativeShape>,
) -> Option<(Vec<u8>, Option<usize>)> {
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
    let prologue = |a: &mut A64Assembler| {
        a.stp_pre(FP, LR, SP, -48);
        a.mov_sp(FP, SP);
        a.stp(CTX, REGS, SP, 16);
        a.stp(COUNT, QNAN_R, SP, 32);
        a.mov(CTX, 0);
        a.mov(REGS, 1);
        a.mov_imm(COUNT, 1); // poll at the first back-edge
        a.mov_imm(QNAN_R, QNAN);
    };
    prologue(&mut a);
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

    // Each property op's cache, in op order (`Sites::for_ops`).
    let mut next_cache = 0usize;
    let mut next_global = 0usize;
    let mut next_method = 0usize;
    let mut next_call = 0usize;
    let hole_bits = NanBox::hole().to_bits();
    // `x14` = the arena slot of the heap handle in `x9`, else to `miss`.
    // Clobbers `x10..x13`, `x15`.
    let walk = |a: &mut A64Assembler, miss: Label| {
        a.mov_imm(10, NanBox::handle(0).to_bits());
        a.cmp(9, 10);
        a.b_cond(Cond::Lo, miss);
        a.ldr(11, CTX, i64::from(l.realm));
        a.ldr(12, 11, i64::from(l.heap_ptr));
        a.ldr(13, 11, i64::from(l.heap_len));
        a.mov_w(14, 9);
        a.cmp(14, 13);
        a.b_cond(Cond::Hs, miss);
        a.mov_imm(15, l.stride as u64);
        a.mul(14, 14, 15);
        a.add(14, 14, 12);
    };
    // `x12` = the element count and `x13` = the first element's address of
    // the plain array whose slot is `x14`, else (not an array, or one with
    // side state) to `miss`. Clobbers `x10`.
    let array = |a: &mut A64Assembler, miss: Label| {
        let on_heap = a.new_label();
        let have = a.new_label();
        a.ldrb(10, 14, i64::from(l.off_cell_tag));
        a.cmp_imm(10, i64::from(l.cell_array_disc));
        a.b_cond(Cond::Ne, miss);
        a.ldrb(10, 14, i64::from(l.off_arr_side));
        a.cbnz(10, miss);
        a.ldrb(12, 14, i64::from(l.off_arr_inline_len));
        a.cmp_imm(12, i64::from(crate::cell::ARRAY_ON_HEAP));
        a.b_cond(Cond::Eq, on_heap);
        a.add_imm(13, 14, i64::from(l.off_arr_inline));
        a.b(have);
        a.bind(on_heap);
        a.ldr(12, 14, i64::from(l.off_arr_len));
        a.ldr(13, 14, i64::from(l.off_arr_ptr));
        a.bind(have);
    };
    // `x15` = the address of the value slot the property cache at `cache`
    // names on the object whose handle is in `x9` (an ordinary, shaped, not
    // host-exotic object of the cached shape; for a `store`, also none of
    // the flags that keep stores out of line and no primitive wrapper),
    // else to `miss`. `x14` = its arena slot. Clobbers `x10..x13`.
    let prop_slot = |a: &mut A64Assembler, cache: u64, store: bool, miss: Label| {
        let spill = a.new_label();
        let far = a.new_label();
        let have = a.new_label();
        walk(a, miss);
        a.ldrb(10, 14, i64::from(l.off_cell_tag));
        a.cmp_imm(10, i64::from(l.cell_object_disc));
        a.b_cond(Cond::Ne, miss);
        if (l.off_od_tag, l.obj_shaped_disc) != (l.off_cell_tag, l.cell_object_disc) {
            a.ldrb(10, 14, i64::from(l.off_od_tag));
            a.cmp_imm(10, i64::from(l.obj_shaped_disc));
            a.b_cond(Cond::Ne, miss);
        }
        a.ldrb(10, 14, i64::from(l.off_jit_flags));
        if store {
            a.mov_imm(11, u64::from(l.no_inline_store));
            a.tst(10, 11);
            a.b_cond(Cond::Ne, miss);
        } else {
            a.tbnz(
                10,
                crate::object::JIT_FLAG_EXOTIC.trailing_zeros() as u8,
                miss,
            );
        }
        a.ldr(10, 14, i64::from(l.off_shape));
        a.mov_imm(11, cache);
        a.ldr(12, 11, i64::from(l.cache_shape_off));
        a.cmp(10, 12);
        a.b_cond(Cond::Ne, miss);
        if store {
            a.ldrb(13, 10, i64::from(l.shape_brands));
            a.tbnz(13, 0, miss);
        }
        a.ldr_w(12, 11, i64::from(l.cache_slot_off));
        a.ldrh(13, 14, i64::from(l.off_slots_len));
        a.cmp(12, 13);
        a.b_cond(Cond::Hs, miss);
        a.cmp_imm(12, INLINE);
        a.b_cond(Cond::Hs, spill);
        a.add_shifted(15, 14, 12, Shift::Lsl, 3);
        a.add_imm(15, 15, i64::from(l.off_inline));
        a.b(have);
        a.bind(spill);
        a.ldr(15, 14, i64::from(l.off_extra));
        a.cmp_imm(12, FAR_FROM);
        a.b_cond(Cond::Hs, far);
        a.add_shifted(15, 15, 12, Shift::Lsl, 3);
        a.add_imm(15, 15, i64::from(l.extra_near) - 8 * INLINE);
        a.b(have);
        a.bind(far);
        a.ldr(15, 15, i64::from(l.extra_far_ptr));
        a.add_shifted(15, 15, 12, Shift::Lsl, 3);
        a.sub_imm(15, 15, 8 * FAR_FROM);
        a.bind(have);
    };
    for (pc, op) in ops.iter().enumerate() {
        a.bind(labels[pc]);
        let cache = if cache_op(op) {
            next_cache += 1;
            sites.caches.get(next_cache - 1)?.get() as u64
        } else if global_op(op) {
            next_global += 1;
            sites.globals.get(next_global - 1)?.get() as u64
        } else {
            0
        };
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
                // Not two Numbers: the interpreter's operation, by its helper.
                a.mov(0, CTX);
                a.mov(1, 9);
                a.mov(2, 10);
                match op {
                    Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. } => {
                        let ga = match op {
                            Op::Sub { .. } => crate::nbvm::GA_SUB,
                            Op::Mul { .. } => crate::nbvm::GA_MUL,
                            _ => crate::nbvm::GA_DIV,
                        };
                        a.mov_imm(3, u64::from(ga));
                        a.call_abs(h.arith);
                    }
                    _ => a.call_abs(h.add),
                }
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.str(0, REGS, slot(dst));
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
                if matches!(op, Op::StrictEq { .. }) {
                    // (Not both Numbers: the helper, which never throws.)
                    a.mov(0, CTX);
                    a.mov(1, 9);
                    a.mov(2, 10);
                    a.call_abs(h.strict_eq);
                    a.str(0, REGS, slot(dst));
                } else {
                    exec_op(&mut a, pc);
                }
                a.bind(done);
            }
            Op::GetProp { dst, obj, ref key } | Op::GetMethod { dst, obj, ref key } => {
                // The monomorphic hit inline: an ordinary shaped object whose
                // shape is the cache's, its cached slot loaded. Anything else
                // is the helper, which also arms the cache.
                let miss = a.new_label();
                let done = a.new_label();
                let got = a.new_label();
                a.ldr(9, REGS, slot(obj));
                prop_slot(&mut a, cache, false, miss);
                a.ldr(0, 15, 0);
                a.b(got);
                a.bind(miss);
                a.mov(0, CTX);
                a.mov(1, 9);
                if matches!(op, Op::GetMethod { .. }) {
                    // (A method read first asks the site's built-in memo.)
                    let (m, c) = sites.methods.get(next_method)?;
                    next_method += 1;
                    a.mov_imm(2, m.get() as u64);
                    a.mov_imm(3, c.get() as u64);
                    a.call_abs(h.get_method_site);
                } else {
                    a.mov_imm(2, key.as_ptr() as u64);
                    a.mov_imm(3, key.len() as u64);
                    a.mov_imm(4, cache);
                    a.call_abs(h.get);
                }
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.bind(got);
                a.str(0, REGS, slot(dst));
                a.bind(done);
            }
            Op::SetProp { obj, ref key, src }
            | Op::SetPropStrict { obj, ref key, src }
            | Op::InitProp { obj, ref key, src } => {
                let helper = match op {
                    Op::SetProp { .. } => h.set,
                    Op::SetPropStrict { .. } => h.set_strict,
                    _ => h.init_prop,
                };
                let miss = a.new_label();
                let done = a.new_label();
                if !matches!(op, Op::InitProp { .. }) {
                    // A Number into an existing writable data slot of the
                    // cache's shape, in place (as the x86-64 tier's Number
                    // store: nothing for the write barrier, and the layout's
                    // record of Number-only slots holds either way). Any
                    // other value or object is the helper's.
                    a.ldr(9, REGS, slot(src));
                    a.and(11, 9, QNAN_R);
                    a.cmp(11, QNAN_R);
                    a.b_cond(Cond::Eq, miss);
                    a.mov(8, 9);
                    a.ldr(9, REGS, slot(obj));
                    prop_slot(&mut a, cache, true, miss);
                    a.str(8, 15, 0);
                    a.b(done);
                }
                a.bind(miss);
                a.mov(0, CTX);
                a.ldr(1, REGS, slot(obj));
                a.mov_imm(2, key.as_ptr() as u64);
                a.mov_imm(3, key.len() as u64);
                a.mov_imm(4, cache);
                a.ldr(5, REGS, slot(src));
                a.call_abs(helper);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.bind(done);
            }
            Op::CallValue {
                dst,
                callee,
                ref args,
                ..
            } => {
                let site = sites.calls.get(next_call)?.get() as u64;
                next_call += 1;
                // (`jit_helper_call_value(.., op, site, _)`.)
                emit_call_value(
                    &mut a,
                    l,
                    (op, n_regs),
                    (dst, callee, None, args),
                    (site, h.call_value, site),
                    thrown,
                )?;
            }
            Op::CallWithMethod {
                dst,
                callee,
                recv,
                ref args,
                ..
            } => {
                // (`jit_helper_call_with_method(.., op, cache, site)`.)
                let site = sites.calls.get(next_call)?.get() as u64;
                next_call += 1;
                emit_call_value(
                    &mut a,
                    l,
                    (op, n_regs),
                    (dst, callee, Some(recv), args),
                    (site, h.call_with_method, cache),
                    thrown,
                )?;
            }
            Op::Construct {
                dst,
                ctor,
                ref args,
            } => {
                let site = sites.calls.get(next_call)?.get() as u64;
                next_call += 1;
                emit_construct(
                    &mut a,
                    l,
                    (op, n_regs),
                    (dst, ctor, args),
                    (site, h),
                    thrown,
                )?;
            }
            Op::CallMethod { .. } => {
                a.mov(0, CTX);
                a.mov(1, REGS);
                a.mov_imm(2, n_regs as u64);
                a.mov_imm(3, core::ptr::from_ref(op) as u64);
                a.mov_imm(4, cache);
                a.mov_imm(5, 0);
                let helper = if matches!(op, Op::CallMethod { .. }) {
                    h.call_method
                } else {
                    h.call_with_method
                };
                a.call_abs(helper);
                a.cbnz(0, thrown);
            }
            Op::CheckTdz { src } => {
                // Nothing unless the binding is still uninitialized (then the
                // interpreter's op throws).
                let fine = a.new_label();
                a.ldr(9, REGS, slot(src));
                a.mov_imm(10, NanBox::tdz().to_bits());
                a.cmp(9, 10);
                a.b_cond(Cond::Ne, fine);
                exec_op(&mut a, pc);
                a.bind(fine);
            }
            Op::RequireObjectCoercible { src, .. } => {
                // Nothing unless the value reads as `undefined` or `null`
                // (`QNAN | 1, 2, 5, 6, 7`: not `false`/`true`, `| 3, 4`).
                let fine = a.new_label();
                a.ldr(9, REGS, slot(src));
                a.mov_imm(10, QNAN);
                a.sub(10, 9, 10);
                a.cmp_imm(10, 7);
                a.b_cond(Cond::Hi, fine);
                a.cmp_imm(10, 3);
                a.b_cond(Cond::Eq, fine);
                a.cmp_imm(10, 4);
                a.b_cond(Cond::Eq, fine);
                exec_op(&mut a, pc);
                a.bind(fine);
            }
            Op::LoadGlobal { dst, .. } | Op::StoreGlobal { src: dst, .. } => {
                // Through the site's `[epoch, address]` pair while the global
                // binding epoch is the noted one; else a fresh address from
                // the helper, else the interpreter's op.
                let fetch = a.new_label();
                let hit = a.new_label();
                let slow = a.new_label();
                let done = a.new_label();
                a.mov_imm(11, cache);
                a.ldr(12, 11, 0);
                a.ldr(13, 11, 8);
                a.mov_imm(10, l.global_epoch);
                a.ldr(10, 10, 0);
                a.cmp(10, 12);
                a.b_cond(Cond::Ne, fetch);
                a.cbz(13, fetch);
                a.bind(hit);
                if matches!(op, Op::LoadGlobal { .. }) {
                    a.ldr(9, 13, 0);
                    a.str(9, REGS, slot(dst));
                } else {
                    a.ldr(9, REGS, slot(dst));
                    a.str(9, 13, 0);
                }
                a.b(done);
                a.bind(fetch);
                a.mov(0, CTX);
                a.mov_imm(1, core::ptr::from_ref(op) as u64);
                a.call_abs(h.global_ref);
                a.cbz(0, slow);
                a.mov(13, 0);
                a.mov_imm(10, l.global_epoch);
                a.ldr(10, 10, 0);
                a.mov_imm(11, cache);
                a.str(10, 11, 0);
                a.str(13, 11, 8);
                a.b(hit);
                a.bind(slow);
                exec_op(&mut a, pc);
                a.bind(done);
            }
            Op::GetKey { dst, obj, key } => {
                // A plain array's present element at an integer index inline;
                // anything else is the helper.
                let miss = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(obj));
                walk(&mut a, miss);
                array(&mut a, miss);
                a.ldr(10, REGS, slot(key));
                a.and(11, 10, QNAN_R);
                a.cmp(11, QNAN_R);
                a.b_cond(Cond::Eq, miss);
                a.fmov_from_x(0, 10);
                a.fcvtzs(11, 0);
                a.scvtf(1, 11);
                a.fcmp(0, 1);
                a.b_cond(Cond::Ne, miss);
                a.cmp(11, 12);
                a.b_cond(Cond::Hs, miss); // (also a negative index)
                a.ldr_idx(0, 13, 11);
                a.mov_imm(10, hole_bits);
                a.cmp(0, 10);
                a.b_cond(Cond::Eq, miss);
                a.str(0, REGS, slot(dst));
                a.b(done);
                a.bind(miss);
                a.mov(0, CTX);
                a.ldr(1, REGS, slot(obj));
                a.ldr(2, REGS, slot(key));
                a.call_abs(h.get_elem);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.str(0, REGS, slot(dst));
                a.bind(done);
            }
            Op::SetKey { obj, key, src } | Op::SetKeyStrict { obj, key, src } => {
                // A plain array's present element at an integral in-bounds
                // index, overwritten in place when the write barrier has
                // nothing to do (as the x86-64 tier's: a non-handle value; or
                // no incremental marking, and a young container, one its next
                // minor collection rescans whole, or an old target). Anything
                // else is the helper's.
                let miss = a.new_label();
                let done = a.new_label();
                let no_barrier = a.new_label();
                let old = crate::gc::OLD_AGE;
                a.ldr(9, REGS, slot(obj));
                walk(&mut a, miss);
                array(&mut a, miss);
                // The barrier (x0..x3 scratch; x12/x13/x14 kept).
                a.ldr(8, REGS, slot(src));
                a.mov_imm(0, NanBox::handle(0).to_bits());
                a.cmp(8, 0);
                a.b_cond(Cond::Lo, no_barrier);
                a.ldr(1, CTX, i64::from(l.realm));
                a.ldrb(2, 1, i64::from(l.marking));
                a.cbnz(2, miss);
                a.ldrb(2, 14, i64::from(l.off_slot_age));
                a.cmp_imm(2, i64::from(old));
                a.b_cond(Cond::Lo, no_barrier);
                a.ldrb(2, 14, i64::from(l.off_slot_dirty));
                a.mov_imm(3, u64::from(crate::heap::JIT_DIRTY_WHOLE));
                a.tst(2, 3);
                a.b_cond(Cond::Ne, no_barrier);
                // The target (rooted: the value being stored): its age.
                a.ldr(2, 1, i64::from(l.heap_ptr));
                a.ldr(3, 1, i64::from(l.heap_len));
                a.mov_w(0, 8);
                a.cmp(0, 3);
                a.b_cond(Cond::Hs, miss);
                a.mov_imm(3, l.stride as u64);
                a.mul(0, 0, 3);
                a.add(0, 0, 2);
                a.ldrb(2, 0, i64::from(l.off_slot_age));
                a.cmp_imm(2, i64::from(old));
                a.b_cond(Cond::Lo, miss);
                a.bind(no_barrier);
                // The index: an integral Number below the length.
                a.ldr(10, REGS, slot(key));
                a.and(11, 10, QNAN_R);
                a.cmp(11, QNAN_R);
                a.b_cond(Cond::Eq, miss);
                a.fmov_from_x(0, 10);
                a.fcvtzs(11, 0);
                a.scvtf(1, 11);
                a.fcmp(0, 1);
                a.b_cond(Cond::Ne, miss);
                a.cmp(11, 12);
                a.b_cond(Cond::Hs, miss); // (also a negative index)
                // Over no hole (an own data element, so no setter can run).
                a.ldr_idx(10, 13, 11);
                a.mov_imm(15, hole_bits);
                a.cmp(10, 15);
                a.b_cond(Cond::Eq, miss);
                a.str_idx(8, 13, 11);
                a.b(done);
                a.bind(miss);
                a.mov(0, CTX);
                a.ldr(1, REGS, slot(obj));
                a.ldr(2, REGS, slot(key));
                a.ldr(3, REGS, slot(src));
                a.mov_imm(4, u64::from(matches!(op, Op::SetKeyStrict { .. })));
                a.call_abs(h.set_elem);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.bind(done);
            }
            Op::ArrayLen { dst, arr } => {
                let miss = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(arr));
                walk(&mut a, miss);
                array(&mut a, miss);
                a.scvtf(0, 12);
                a.str_d(0, REGS, slot(dst));
                a.b(done);
                a.bind(miss);
                a.mov(0, CTX);
                a.ldr(1, REGS, slot(arr));
                a.call_abs(h.array_len);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.str(0, REGS, slot(dst));
                a.bind(done);
            }
            Op::Not { dst, a: src } => {
                // `!b` of a boolean inline, else ToBoolean by the helper.
                let to_true = a.new_label();
                let to_false = a.new_label();
                let store = a.new_label();
                a.ldr(1, REGS, slot(src));
                a.mov_imm(9, false_bits);
                a.cmp(1, 9);
                a.b_cond(Cond::Eq, to_true);
                a.mov_imm(10, true_bits);
                a.cmp(1, 10);
                a.b_cond(Cond::Eq, to_false);
                a.mov(0, CTX);
                a.call_abs(h.truthy);
                a.cbz(0, to_true);
                a.bind(to_false);
                a.mov_imm(9, false_bits);
                a.b(store);
                a.bind(to_true);
                a.mov_imm(9, true_bits);
                a.bind(store);
                a.str(9, REGS, slot(dst));
            }
            Op::Neg { dst, a: src } | Op::Step { dst, src, .. } | Op::ToNumeric { dst, src } => {
                let slow = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(src));
                a.and(11, 9, QNAN_R);
                a.cmp(11, QNAN_R);
                a.b_cond(Cond::Eq, slow);
                match *op {
                    Op::Neg { .. } | Op::Step { .. } => {
                        a.fmov_from_x(0, 9);
                        match *op {
                            Op::Neg { .. } => a.fneg(0, 0),
                            Op::Step { dec, .. } => {
                                a.mov_imm(10, 1.0f64.to_bits());
                                a.fmov_from_x(1, 10);
                                if dec {
                                    a.fsub(0, 0, 1);
                                } else {
                                    a.fadd(0, 0, 1);
                                }
                            }
                            _ => {}
                        }
                        let nan = a.new_label();
                        a.fcmp(0, 0);
                        a.b_cond(Cond::Vs, nan);
                        a.str_d(0, REGS, slot(dst));
                        a.b(done);
                        a.bind(nan);
                        a.mov_imm(9, CANONICAL_NAN);
                        a.str(9, REGS, slot(dst));
                    }
                    _ => a.str(9, REGS, slot(dst)),
                }
                a.b(done);
                a.bind(slow);
                exec_op(&mut a, pc);
                a.bind(done);
            }
            Op::ValueBin {
                dst,
                op: vb @ 1..=6,
                a: ra,
                b: rb,
            } => {
                // Two integral Numbers: the 32-bit operation on their ToInt32
                // (exact below 2^63, where an integral double's low 32 bits
                // are its ToInt32).
                let slow = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(ra));
                a.ldr(10, REGS, slot(rb));
                both_numbers(&mut a, slow);
                a.fcvtzs(11, 0);
                a.scvtf(2, 11);
                a.fcmp(0, 2);
                a.b_cond(Cond::Ne, slow);
                // (Within ±2^53: `fcvtzs` saturates, and a saturated
                // result can convert back equal, as 2^63 does.)
                a.mov_imm(15, 1 << 53);
                a.add(14, 11, 15);
                a.mov_imm(15, 1 << 54);
                a.cmp(14, 15);
                a.b_cond(Cond::Hi, slow);
                a.fcvtzs(12, 1);
                a.scvtf(2, 12);
                a.fcmp(1, 2);
                a.b_cond(Cond::Ne, slow);
                // (Within ±2^53: `fcvtzs` saturates, and a saturated
                // result can convert back equal, as 2^63 does.)
                a.mov_imm(15, 1 << 53);
                a.add(14, 12, 15);
                a.mov_imm(15, 1 << 54);
                a.cmp(14, 15);
                a.b_cond(Cond::Hi, slow);
                match vb {
                    1 => a.and_w(11, 11, 12),
                    2 => a.orr_w(11, 11, 12),
                    3 => a.eor_w(11, 11, 12),
                    4 => a.lsl_w(11, 11, 12),
                    5 => a.asr_w(11, 11, 12),
                    _ => a.lsr_w(11, 11, 12),
                }
                if vb != 6 {
                    a.sxtw(11, 11);
                }
                a.scvtf(0, 11);
                a.str_d(0, REGS, slot(dst));
                a.b(done);
                a.bind(slow);
                a.mov(0, CTX);
                a.mov(1, 9);
                a.mov(2, 10);
                a.mov_imm(3, u64::from(vb));
                a.call_abs(h.value_bin);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.str(0, REGS, slot(dst));
                a.bind(done);
            }
            Op::Mod { dst, a: ra, b: rb } => {
                // Integral `a >= 0`, `b > 0` below 2^53: the integer
                // remainder (exact, and never `-0`); else the interpreter.
                let slow = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(ra));
                a.ldr(10, REGS, slot(rb));
                both_numbers(&mut a, slow);
                a.fcvtzs(11, 0);
                a.scvtf(2, 11);
                a.fcmp(0, 2);
                a.b_cond(Cond::Ne, slow);
                // (Within ±2^53: `fcvtzs` saturates, and a saturated
                // result can convert back equal, as 2^63 does.)
                a.mov_imm(15, 1 << 53);
                a.add(14, 11, 15);
                a.mov_imm(15, 1 << 54);
                a.cmp(14, 15);
                a.b_cond(Cond::Hi, slow);
                a.fcvtzs(12, 1);
                a.scvtf(2, 12);
                a.fcmp(1, 2);
                a.b_cond(Cond::Ne, slow);
                // (Within ±2^53: `fcvtzs` saturates, and a saturated
                // result can convert back equal, as 2^63 does.)
                a.mov_imm(15, 1 << 53);
                a.add(14, 12, 15);
                a.mov_imm(15, 1 << 54);
                a.cmp(14, 15);
                a.b_cond(Cond::Hi, slow);
                a.tbnz(11, 63, slow);
                a.cmp_imm(12, 0);
                a.b_cond(Cond::Le, slow);
                // (A -0 dividend converts to 0 but its result is -0.)
                a.tbnz(9, 63, slow);
                a.sdiv(13, 11, 12);
                a.msub(11, 13, 12, 11);
                a.scvtf(0, 11);
                a.str_d(0, REGS, slot(dst));
                a.b(done);
                a.bind(slow);
                a.mov(0, CTX);
                a.mov(1, 9);
                a.mov(2, 10);
                a.mov_imm(3, u64::from(crate::nbvm::GA_MOD));
                a.call_abs(h.arith);
                a.mov_imm(9, throw_bits);
                a.cmp(0, 9);
                a.b_cond(Cond::Eq, thrown);
                a.str(0, REGS, slot(dst));
                a.bind(done);
            }
            Op::CellGet { dst, cell } => {
                // A `Cell::Box`'s value, or a one-element array cell's present
                // element, inline; anything else is the interpreter's op.
                let miss = a.new_label();
                let done = a.new_label();
                let not_box = a.new_label();
                let got = a.new_label();
                a.ldr(9, REGS, slot(cell));
                walk(&mut a, miss);
                a.ldrb(10, 14, i64::from(l.off_cell_tag));
                a.cmp_imm(10, i64::from(l.box_disc));
                a.b_cond(Cond::Ne, not_box);
                a.ldr(0, 14, i64::from(l.off_box_val));
                a.b(got);
                a.bind(not_box);
                array(&mut a, miss);
                a.cbz(12, miss);
                a.ldr(0, 13, 0);
                a.mov_imm(10, hole_bits);
                a.cmp(0, 10);
                a.b_cond(Cond::Eq, miss);
                a.bind(got);
                a.str(0, REGS, slot(dst));
                a.b(done);
                a.bind(miss);
                exec_op(&mut a, pc);
                a.bind(done);
            }
            Op::BindThis { this } => {
                // An ordinary object is its own `this`: nothing to do. Anything
                // else is the interpreter's op.
                let slow = a.new_label();
                let done = a.new_label();
                a.ldr(9, REGS, slot(this));
                walk(&mut a, slow);
                a.ldrb(10, 14, i64::from(l.off_cell_tag));
                a.cmp_imm(10, i64::from(l.cell_object_disc));
                a.b_cond(Cond::Eq, done);
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
    // The native-call entry (see the function docs): x2 = argc, x3 = argv,
    // x4 = the captures, x5 = `this`.
    let native_at = native.map(|ns| {
        let at = a.pos();
        prologue(&mut a);
        let undef = NanBox::undefined().to_bits();
        a.mov_imm(9, undef);
        for i in 0..ns.n_params.min(n_regs) {
            let missing = a.new_label();
            let next = a.new_label();
            a.cmp_imm(2, i as i64);
            a.b_cond(Cond::Ls, missing);
            a.ldr(10, 3, 8 * i as i64);
            a.str(10, REGS, 8 * i as i64);
            a.b(next);
            a.bind(missing);
            a.str(9, REGS, 8 * i as i64);
            a.bind(next);
        }
        for j in 0..ns.n_caps {
            let r = ns.n_params + j;
            if r < n_regs {
                a.ldr(10, 4, 8 * j as i64);
                a.str(10, REGS, 8 * r as i64);
            }
        }
        let this_at = ns.n_params + ns.n_caps;
        if this_at < n_regs {
            a.str(5, REGS, 8 * this_at as i64);
        }
        for r in (this_at + 1)..n_regs {
            a.str(9, REGS, 8 * r as i64);
        }
        if let Some(nt) = ns.new_target.filter(|&nt| nt < n_regs) {
            a.ldr(10, 3, -8);
            a.str(10, REGS, 8 * nt as i64);
        }
        a.b(labels[0]);
        at
    });
    Some((a.finish()?, native_at))
}

/// The ways of a call site the native call sequence tries (of
/// `CallSite::WAYS`).
const CALL_WAYS: usize = 4;

/// `CallValue` (`dst = callee(args…)`, `this` undefined) as a **native call
/// sequence**, as the x86-64 tier makes it: the callee identified inline
/// (a VM closure: function id and capture count, the key `CallTarget`s
/// are filled under, and its captures); one of the site's first ways
/// holding that key gives the callee's native-call entry and window size;
/// then, with the depth limits not reached, a single realm, the native
/// stack above `JitStack::sp_limit`, this activation's window on the
/// register stack and room there, the call lays out a native-call header
/// (the previous header, the callee, the realm's mark word, the count,
/// `undefined`) and the arguments at the stack's top, as
/// `jit_tier::enter_native` does, counts the call depth, and enters the
/// callee, which builds its window above them. Anything else — and the
/// first call, which fills the way — is `jit_helper_call_value`, the
/// interpreter's call.
fn emit_call_value(
    a: &mut A64Assembler,
    l: &Layout,
    (op, n_regs): (&Op, usize),
    (dst, callee, this, args): (u16, u16, Option<u16>, &[u16]),
    (site, slow_helper, slow_arg4): (u64, u64, u64),
    thrown: Label,
) -> Option<()> {
    emit_native_call(
        a,
        l,
        (op, n_regs),
        (dst, callee, this, args),
        (site, slow_helper, slow_arg4),
        thrown,
        None,
    )
}

/// `new C(args…)` (`Construct`) as a native call sequence: the x86-64
/// tier's leaf `jit_construct_info` checks the constructor against the site
/// and makes the `this` `[[Construct]]` starts with (only a plain one is
/// taken here: a derived constructor's unbound `this`, or a constructor
/// whose body is another function's, a forwarding default constructor's,
/// is the slow path); the header then marks a construct and carries
/// `new.target` (the constructor); a returned heap value goes through
/// `jit_helper_construct_result` (an object is the result, else `this`).
/// The slow path is `jit_helper_construct`, which fills the site.
fn emit_construct(
    a: &mut A64Assembler,
    l: &Layout,
    (op, n_regs): (&Op, usize),
    (dst, ctor, args): (u16, u16, &[u16]),
    (site, h): (u64, &Helpers),
    thrown: Label,
) -> Option<()> {
    emit_native_call(
        a,
        l,
        (op, n_regs),
        (dst, ctor, None, args),
        (site, h.construct, site),
        thrown,
        Some((h.construct_info, h.construct_result)),
    )
}

/// The native call sequence of [`emit_call_value`] and [`emit_construct`]
/// (`construct`: the latter's `(construct_info, construct_result)`).
#[allow(clippy::too_many_arguments)]
fn emit_native_call(
    a: &mut A64Assembler,
    l: &Layout,
    (op, n_regs): (&Op, usize),
    (dst, callee, this, args): (u16, u16, Option<u16>, &[u16]),
    (site, slow_helper, slow_arg4): (u64, u64, u64),
    thrown: Label,
    construct: Option<(u64, u64)>,
) -> Option<()> {
    use super::call_seq::{CallSite, CallTarget, FRAME_HEADER, STACK_BASE, STACK_FRAMES};
    use super::call_seq::{STACK_LIMIT, STACK_SP_LIMIT, STACK_TOP};
    let off = |v: usize| i64::try_from(v).ok();
    let t0 = off(core::mem::offset_of!(CallSite, targets))?;
    let tsz = off(core::mem::size_of::<CallTarget>())?;
    let t_key = off(core::mem::offset_of!(CallTarget, key))?;
    let t_code = off(core::mem::offset_of!(CallTarget, code))?;
    let t_frame = off(core::mem::offset_of!(CallTarget, frame_bytes))?;
    let argc = args.len() as i64;
    let hdr_bytes = 8 * (FRAME_HEADER as i64 + argc);
    let slow = a.new_label();
    let done = a.new_label();
    let hit = a.new_label();
    let have_caps = a.new_label();
    let caps_heap = a.new_label();
    // A construct's `this` slot on the native stack (`[sp]`), released on
    // every way out; `x8` = the key `jit_construct_info` reported.
    let slow_free = a.new_label();
    let slow_at = if construct.is_some() { slow_free } else { slow };
    if let Some((info, _)) = construct {
        a.sub_imm(SP, SP, 16);
        a.mov(0, CTX);
        a.ldr(1, REGS, slot(callee));
        a.mov(2, 1);
        a.mov_sp(3, SP);
        a.mov_imm(4, site);
        a.call_abs(info);
        a.mov_imm(9, u64::MAX);
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, slow_free);
        a.mov(8, 0);
        // (A plain `this`: a heap object, not a derived constructor's
        // unbound sentinel.)
        a.ldr(9, SP, 0);
        a.mov_imm(10, NanBox::handle(0).to_bits());
        a.cmp(9, 10);
        a.b_cond(Cond::Lo, slow_free);
    }
    let slow = slow_at;
    // The callee: a VM closure's key (x10) and captures (x12).
    a.ldr(9, REGS, slot(callee));
    a.mov_imm(10, NanBox::handle(0).to_bits());
    a.cmp(9, 10);
    a.b_cond(Cond::Lo, slow);
    a.ldr(11, CTX, i64::from(l.realm));
    a.ldr(12, 11, i64::from(l.heap_ptr));
    a.ldr(13, 11, i64::from(l.heap_len));
    a.mov_w(14, 9);
    a.cmp(14, 13);
    a.b_cond(Cond::Hs, slow);
    a.mov_imm(15, l.stride as u64);
    a.mul(14, 14, 15);
    a.add(14, 14, 12);
    a.ldrb(10, 14, i64::from(l.off_cell_tag));
    a.cmp_imm(10, i64::from(l.vmfn_disc));
    a.b_cond(Cond::Ne, slow);
    a.ldrb(11, 14, i64::from(l.off_vmfn_caps_inline_len));
    a.cmp_imm(11, i64::from(crate::cell::ARRAY_ON_HEAP));
    a.b_cond(Cond::Eq, caps_heap);
    a.add_imm(12, 14, i64::from(l.off_vmfn_caps_inline));
    a.b(have_caps);
    a.bind(caps_heap);
    a.ldr(12, 14, i64::from(l.off_vmfn_caps_ptr));
    a.ldr(11, 14, i64::from(l.off_vmfn_caps_len));
    a.bind(have_caps);
    a.ldr_w(10, 14, i64::from(l.off_vmfn_func));
    a.add_shifted(10, 10, 11, Shift::Lsl, 32);
    if construct.is_some() {
        // (The constructor runs its own body: not a forwarding one's.)
        a.cmp(10, 8);
        a.b_cond(Cond::Ne, slow);
    }
    // A way of the site holding it: x15 = the entry, x16 = the window size.
    a.mov_imm(13, site);
    for w in 0..CALL_WAYS as i64 {
        let next = a.new_label();
        a.ldr(15, 13, t0 + w * tsz + t_key);
        a.cmp(15, 10);
        a.b_cond(Cond::Ne, next);
        a.ldr(15, 13, t0 + w * tsz + t_code);
        a.ldr(16, 13, t0 + w * tsz + t_frame);
        a.b(hit);
        a.bind(next);
    }
    a.b(slow);
    a.bind(hit);
    // The depth limits (`call_with`'s recursion guard), one realm, a nested
    // call (`ctx.call_depth != 0`, as `call_compiled_with` requires).
    a.ldr(11, CTX, i64::from(l.realm));
    a.ldr(0, CTX, i64::from(l.call_depth));
    a.cbz(0, slow);
    a.ldr(1, 11, i64::from(l.max_depth));
    a.cmp(0, 1);
    a.b_cond(Cond::Hs, slow);
    a.ldr(2, 11, i64::from(l.total_depth));
    a.cmp(2, 1);
    a.b_cond(Cond::Hs, slow);
    a.ldrb(3, 11, i64::from(l.multi_realm));
    a.cbnz(3, slow);
    // The native stack, and this window on the register stack.
    a.ldr(4, CTX, i64::from(l.stack));
    a.ldr(5, 4, i64::from(STACK_SP_LIMIT));
    a.mov_sp(6, SP);
    a.cmp(6, 5);
    a.b_cond(Cond::Lo, slow);
    a.ldr(5, 4, i64::from(STACK_BASE));
    a.cmp(REGS, 5);
    a.b_cond(Cond::Lo, slow);
    a.ldr(6, 4, i64::from(STACK_TOP)); // x6 = the header
    a.cmp(REGS, 6);
    a.b_cond(Cond::Hs, slow);
    // Room: header, arguments and the callee's window under the limit.
    a.ldr(5, 4, i64::from(STACK_LIMIT));
    a.add_imm(7, 6, hdr_bytes);
    a.add(7, 7, 16); // x7 = one past the callee's window
    a.cmp(7, 5);
    a.b_cond(Cond::Hi, slow);
    // The header and the arguments.
    a.ldr(5, 4, i64::from(STACK_FRAMES));
    a.str(5, 6, 0);
    a.str(9, 6, 8);
    a.ldr(5, 11, i64::from(l.vm_mark));
    a.str(5, 6, 16);
    let count = if construct.is_some() {
        argc as u64 | super::call_seq::HDR_CONSTRUCT
    } else {
        argc as u64
    };
    a.mov_imm(5, count);
    a.str(5, 6, 24);
    if construct.is_some() {
        a.str(9, 6, 32); // `new.target`: the constructor
    } else {
        a.mov_imm(5, NanBox::undefined().to_bits());
        a.str(5, 6, 32);
    }
    for (i, &r) in args.iter().enumerate() {
        a.ldr(5, REGS, slot(r));
        a.str(5, 6, 8 * (FRAME_HEADER as i64 + i as i64));
    }
    a.str(6, 4, i64::from(STACK_FRAMES));
    a.str(7, 4, i64::from(STACK_TOP));
    // The call depth (x0, x2: the depths read above).
    a.add_imm(0, 0, 1);
    a.str(0, CTX, i64::from(l.call_depth));
    a.add_imm(2, 2, 1);
    a.str(2, 11, i64::from(l.total_depth));
    // Enter: (ctx, window, argc, argv, caps, this), the header kept on the
    // native stack across the call.
    a.push(6);
    a.mov(0, CTX);
    a.add_imm(1, 6, hdr_bytes);
    a.mov_imm(2, argc as u64);
    a.add_imm(3, 6, 8 * FRAME_HEADER as i64);
    a.mov(4, 12);
    match this {
        Some(r) => a.ldr(5, REGS, slot(r)),
        None if construct.is_some() => a.ldr(5, SP, 16),
        None => a.mov_imm(5, NanBox::undefined().to_bits()),
    }
    a.blr(15);
    // Back: the header and window released, the depth restored.
    a.pop(6);
    a.ldr(4, CTX, i64::from(l.stack));
    a.str(6, 4, i64::from(STACK_TOP));
    a.ldr(5, 6, 0);
    a.str(5, 4, i64::from(STACK_FRAMES));
    a.ldr(5, CTX, i64::from(l.call_depth));
    a.sub_imm(5, 5, 1);
    a.str(5, CTX, i64::from(l.call_depth));
    a.ldr(11, CTX, i64::from(l.realm));
    a.ldr(5, 11, i64::from(l.total_depth));
    a.sub_imm(5, 5, 1);
    a.str(5, 11, i64::from(l.total_depth));
    // The outcome: a value, the fall-off's `undefined`, or a throw.
    if let Some((_, result)) = construct {
        // `this` unless the body returned an object (a heap value: the
        // helper decides); the slot released first.
        let use_this = a.new_label();
        let have = a.new_label();
        a.ldr(1, SP, 0);
        a.add_imm(SP, SP, 16);
        a.mov_imm(9, NanBox::jit_throw_bits());
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, thrown);
        a.mov_imm(9, NanBox::handle(0).to_bits());
        a.cmp(0, 9);
        a.b_cond(Cond::Lo, use_this); // (also the fall-off sentinel)
        a.mov(2, 1);
        a.mov(1, 0);
        a.mov(0, CTX);
        a.call_abs(result);
        a.mov_imm(9, NanBox::jit_throw_bits());
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, thrown);
        a.b(have);
        a.bind(use_this);
        a.mov(0, 1);
        a.bind(have);
        a.str(0, REGS, slot(dst));
        a.b(done);
        a.bind(slow_free);
        a.add_imm(SP, SP, 16);
    } else {
        let value = a.new_label();
        a.mov_imm(9, NanBox::jit_falloff_bits());
        a.cmp(0, 9);
        a.b_cond(Cond::Ne, value);
        a.mov_imm(0, NanBox::undefined().to_bits());
        a.bind(value);
        a.mov_imm(9, NanBox::jit_throw_bits());
        a.cmp(0, 9);
        a.b_cond(Cond::Eq, thrown);
        a.str(0, REGS, slot(dst));
        a.b(done);
    }
    // The interpreter's call, which also fills the site.
    if construct.is_none() {
        a.bind(slow);
    }
    a.mov(0, CTX);
    a.mov(1, REGS);
    a.mov_imm(2, n_regs as u64);
    a.mov_imm(3, core::ptr::from_ref(op) as u64);
    a.mov_imm(4, slow_arg4);
    a.mov_imm(5, site);
    a.call_abs(slow_helper);
    a.cbnz(0, thrown);
    a.bind(done);
    Some(())
}

/// Whether the baseline generic tier runs on this target: aarch64, unless
/// `KATAAN_JIT_A64GEN=0` turns it (and with it the aarch64 JIT: see
/// `jit::backend_enabled`) off (read once per process; always on in the
/// crate's own tests).
#[must_use]
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(target_arch = "aarch64")
        && (cfg!(test)
            || *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_A64GEN").is_none_or(|v| v != "0")))
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
            get: helpers.get as usize as u64,
            set: helpers.set as usize as u64,
            set_strict: helpers.set_strict as usize as u64,
            init_prop: helpers.init_prop as usize as u64,
            call_method: helpers.call_method as usize as u64,
            call_value: helpers.call_value as usize as u64,
            construct: helpers.construct as usize as u64,
            construct_info: helpers.construct_info as usize as u64,
            construct_result: helpers.construct_result as usize as u64,
            global_ref: helpers.global_ref as usize as u64,
            get_elem: helpers.get_elem as usize as u64,
            set_elem: helpers.set_elem as usize as u64,
            array_len: helpers.array_len as usize as u64,
            get_method_site: helpers.get_method_site as usize as u64,
            call_with_method: helpers.call_with_method as usize as u64,
            strict_eq: helpers.strict_eq as usize as u64,
            add: helpers.add as usize as u64,
            arith: helpers.arith as usize as u64,
            value_bin: helpers.value_bin as usize as u64,
        };
        let jl = super::compute_jit_layout();
        let (heap_ptr, heap_len) = (jl.props.heap_ptr, jl.props.heap_len);
        let l = Layout {
            realm: helpers.call_layout.realm,
            heap_ptr,
            heap_len,
            stride: jl.slot_stride,
            off_cell_tag: jl.off_cell_tag,
            cell_object_disc: jl.cell_object_disc,
            off_od_tag: jl.off_od_tag,
            obj_shaped_disc: jl.obj_shaped_disc,
            off_jit_flags: jl.props.off_jit_flags,
            off_shape: jl.off_shape,
            off_slots_len: jl.off_slots_len,
            off_inline: jl.off_inline,
            off_extra: jl.props.off_extra,
            extra_near: jl.extra_near,
            extra_far_ptr: jl.extra_far_ptr,
            cache_shape_off: jl.cache_shape_off,
            cache_slot_off: jl.cache_slot_off,
            cell_array_disc: jl.cell_array_disc,
            off_arr_side: jl.off_arr_side,
            off_arr_inline_len: jl.off_arr_inline_len,
            off_arr_inline: jl.off_arr_inline,
            off_arr_ptr: jl.off_arr_ptr,
            off_arr_len: jl.off_arr_len,
            box_disc: jl.props.box_disc,
            off_box_val: jl.props.off_box_val,
            global_epoch: crate::env::global_binding_epoch_addr() as u64,
            no_inline_store: super::props::FLAGS_NO_INLINE_STORE,
            shape_brands: jl.props.shape_brands,
            marking: jl.props.marking,
            off_slot_age: jl.props.off_slot_age,
            off_slot_dirty: jl.props.off_slot_dirty,
            stack: helpers.call_layout.stack,
            call_depth: helpers.call_layout.call_depth,
            total_depth: helpers.call_layout.total_depth,
            max_depth: helpers.call_layout.max_depth,
            multi_realm: helpers.call_layout.multi_realm,
            vm_mark: helpers.call_layout.callee_len,
            vmfn_disc: jl.props.vmfn_disc,
            off_vmfn_func: jl.props.off_vmfn_func,
            off_vmfn_caps_inline_len: jl.props.off_vmfn_caps_inline_len,
            off_vmfn_caps_inline: jl.props.off_vmfn_caps_inline,
            off_vmfn_caps_ptr: jl.props.off_vmfn_caps_ptr,
            off_vmfn_caps_len: jl.props.off_vmfn_caps_len,
        };
        let sites = alloc::boxed::Box::new(Sites::for_ops(&bytecode));
        // A native-call entry (what `jit_tier::call_compiled_with` enters a
        // compiled callee through) for a body a call can enter that way:
        // not a generator or async function, no `try` regions (whose handler
        // stack segment the entry would have to open), and a `new.target`
        // register, if any, past the inputs (as the x86-64 tier's rule).
        let new_target = proto.ops.iter().take(2).find_map(|o| match o {
            Op::LoadNewTarget { dst } => Some(*dst as usize),
            _ => None,
        });
        let native = (!proto.is_async
            && !proto.is_generator
            && !bytecode.iter().any(|o| matches!(o, Op::PushHandler { .. }))
            && new_target.is_none_or(|nt| nt > proto.n_params + proto.n_captures))
        .then_some(NativeShape {
            n_params: proto.n_params,
            n_caps: proto.n_captures,
            new_target,
        });
        let (code, native_at) = compile(
            &bytecode,
            proto.n_regs,
            kind,
            &entries,
            (&h, &l, &sites),
            native,
        )?;
        let func = super::JitFunction::from_code(&code)?;
        Some(Self {
            func,
            n_params: proto.n_params,
            kind: super::JitKind::Generic,
            native_entry: native_at.map(|at| super::NativeEntries {
                // (Only Rust enters an aarch64 body natively: no compiled
                // call sequence here takes the other entries.)
                general: at,
                rust: at,
                exact: at,
                exact_pinned: None,
                direct: None,
                direct_pinned: None,
                room: 0,
                legacy: proto.legacy,
            }),
            n_regs: proto.n_regs,
            osr_entries: entries,
            bytecode: Some(bytecode),
            sites: None,
            pc_map: None,
            returns_bool: false,
            gen_resume: Vec::new(),
            a64: Some(sites),
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
        get: 0x8000,
        set: 0x9000,
        set_strict: 0xa000,
        init_prop: 0xb000,
        call_method: 0xc000,
        call_value: 0x1_7000,
        construct: 0x1_8000,
        construct_info: 0x1_9000,
        construct_result: 0x1_a000,
        global_ref: 0xd000,
        get_elem: 0xe000,
        set_elem: 0xf000,
        array_len: 0x1_0000,
        get_method_site: 0x1_1000,
        call_with_method: 0x1_2000,
        strict_eq: 0x1_3000,
        add: 0x1_4000,
        arith: 0x1_5000,
        value_bin: 0x1_6000,
    };

    const L: Layout = Layout {
        realm: 8,
        heap_ptr: 16,
        heap_len: 24,
        stride: 72,
        off_cell_tag: 0,
        cell_object_disc: 0,
        off_od_tag: 0,
        obj_shaped_disc: 0,
        off_jit_flags: 1,
        off_shape: 8,
        off_slots_len: 16,
        off_inline: 24,
        off_extra: 56,
        extra_near: 0,
        extra_far_ptr: 64,
        cache_shape_off: 0,
        cache_slot_off: 8,
        cell_array_disc: 3,
        off_arr_side: 2,
        off_arr_inline_len: 3,
        off_arr_inline: 8,
        off_arr_ptr: 40,
        off_arr_len: 48,
        box_disc: 4,
        off_box_val: 8,
        global_epoch: 0x2_0000,
        no_inline_store: 0x3e,
        shape_brands: 16,
        marking: 32,
        off_slot_age: 60,
        off_slot_dirty: 61,
        stack: 40,
        call_depth: 48,
        total_depth: 56,
        max_depth: 64,
        multi_realm: 72,
        vm_mark: 80,
        vmfn_disc: 5,
        off_vmfn_func: 8,
        off_vmfn_caps_inline_len: 12,
        off_vmfn_caps_inline: 16,
        off_vmfn_caps_ptr: 40,
        off_vmfn_caps_len: 48,
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
                let sites = Sites::for_ops(&p.ops);
                let native = Some(NativeShape {
                    n_params: p.n_params,
                    n_caps: p.n_captures,
                    new_target: None,
                });
                let code = compile(
                    &p.ops,
                    p.n_regs,
                    (false, false),
                    &entries,
                    (&H, &L, &sites),
                    native,
                );
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
