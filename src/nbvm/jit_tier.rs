//! The VM side of the generic machine-code tier (`ROADMAP.md` §2.1): entering
//! compiled bodies on a call or mid-loop (on-stack replacement), the runtime
//! helpers that let a body hand an op back to the interpreter or poll at a loop
//! back-edge, and the `KATAAN_JIT_LOG` diagnostics.
//!
//! A generic body runs on the **interpreter's own register file** and numbers
//! its ops exactly as the bytecode does (see `crate::jit::JitFunction::
//! compile_generic`). That shared state is what makes the tier simple to keep
//! correct: there is no value translation at the boundary, any op the native
//! code does not lower is executed by [`run_frame`] itself on the same registers,
//! and a loop that got hot in the interpreter can continue in native code from
//! its header with nothing to reconstruct.

use super::{
    Ctx, FnProto, FrameExit, Handle, NanBox, Op, Reg, VmError, jit_for, run_frame, vm_agent_tick,
    vm_error, vm_load_global, vm_safepoint, vm_store_global_fast,
};
use crate::jit::{CallSite, CalleeInfo, FRAME_HEADER, JitStack};
use alloc::rc::Rc;

/// Back-edges an interpreted activation runs before it tries to continue in
/// native code (on-stack replacement).
const OSR_THRESHOLD: u32 = 500;

/// How the tier is engaged, from `KATAAN_JIT` (read once per process): `0` turns
/// it off, `eager` compiles every eligible function on its first call and enters
/// every eligible loop on its first back-edge (to exercise the native code as
/// widely as possible, e.g. under the conformance corpus); anything else — or
/// unset — is the normal hotness-driven tiering.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Normal,
    Eager,
}

fn mode() -> Mode {
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("KATAAN_JIT").as_deref() {
        Ok("0" | "off") => Mode::Off,
        Ok("eager") => Mode::Eager,
        _ => Mode::Normal,
    })
}

/// Whether a call should try the native tier: the function has tiered up
/// (`tiered`), or the tier is eager.
pub(super) fn call_is_hot(tiered: bool) -> bool {
    match mode() {
        Mode::Off => false,
        Mode::Normal => tiered,
        Mode::Eager => true,
    }
}

/// Back-edges before an interpreted activation tries on-stack replacement
/// (`u32::MAX`, never, when the tier is off).
pub(super) fn osr_threshold() -> u32 {
    match mode() {
        Mode::Off => u32::MAX,
        Mode::Normal => OSR_THRESHOLD,
        Mode::Eager => 1,
    }
}

/// Back-edges a compiled loop runs between polls of [`jit_helper_backedge`]
/// while no agent is scheduling (with agents, it polls every iteration, as the
/// interpreter does).
const POLL_INTERVAL: u64 = 1024;

/// The generic tier's per-context runtime state.
pub(super) struct TierRuntime {
    /// The exception-handler stacks of the running compiled activations: one
    /// shared `Vec` of `(catch_pc, catch_reg)` (the interpreter's `handlers`
    /// layout, innermost last), where the innermost compiled activation owns the
    /// entries from `base` up. [`run_compiled`] opens a segment on entry and drops
    /// it on exit, so an activation never sees (or pops) its callers' handlers,
    /// and the common case reuses the allocation.
    pub(super) handlers: alloc::vec::Vec<(usize, Reg)>,
    base: usize,
    /// By function id: the compiled body [`call_compiled`] may enter directly —
    /// set once the function passed its (static) eligibility test, compiled and
    /// got hot, none of which is ever undone.
    direct: alloc::vec::Vec<Option<Rc<crate::jit::JitProto>>>,
    /// This thread's register stack (see [`JitStack`]), set before any
    /// compiled body runs. The emitted call sequence reads it at a fixed offset.
    pub(super) stack: StackPtr,
    /// Whether the context has a host (`Ctx::host` is `Some`; it never
    /// changes). Compiled method calls read it inline.
    pub(super) hosted: bool,
    /// Whether the context runs in the main realm (`Ctx::in_realm` is
    /// `None`; it never changes): a closure it makes needs no realm tag.
    /// Compiled code's inline closure allocation reads it.
    pub(super) main_realm: bool,
    /// The innermost activation that may leave at a loop header to be
    /// recompiled (one [`osr`] entered): its register window's address and
    /// its body (`0, null` for none). Only the back-edge polls of that very
    /// activation — whose caller can continue it — may ask it to leave.
    exit_regs: usize,
    exit_jit: *const crate::jit::JitProto,
    /// The loop header such an activation left at.
    resume: Option<usize>,
    /// Where the activation that just left continues: that header and its
    /// handler stack there (see [`run_compiled_at`]).
    left: Option<(usize, alloc::vec::Vec<(usize, Reg)>)>,
    /// What a back-edge poll returns to its loop: polls until the next one.
    poll_interval: u64,
}

impl Default for TierRuntime {
    fn default() -> Self {
        Self {
            handlers: alloc::vec::Vec::new(),
            base: 0,
            direct: alloc::vec::Vec::new(),
            stack: StackPtr::default(),
            hosted: false,
            main_realm: true,
            exit_regs: 0,
            exit_jit: core::ptr::null(),
            resume: None,
            left: None,
            poll_interval: if exit_stress() { 1 } else { POLL_INTERVAL },
        }
    }
}

impl TierRuntime {
    /// The state of a hosted context.
    pub(super) fn hosted() -> Self {
        Self {
            hosted: true,
            ..Self::default()
        }
    }
}

/// A pointer to the running thread's [`JitStack`] (null until first use).
#[repr(transparent)]
pub(super) struct StackPtr(pub(super) *mut JitStack);

impl Default for StackPtr {
    fn default() -> Self {
        Self(core::ptr::null_mut())
    }
}

/// Words in a thread's register stack: far more than the deepest legal call
/// chain needs (`max_call_depth` activations of typical size). Committed by the
/// OS only as it is touched.
const STACK_WORDS: usize = 1 << 20;

/// Owns one thread's register stack memory.
struct StackOwner {
    stack: *mut JitStack,
}

impl StackOwner {
    fn new() -> Self {
        let layout = std::alloc::Layout::array::<u64>(STACK_WORDS).expect("stack layout");
        // SAFETY: a non-zero-sized layout; the result is checked for null.
        #[allow(unsafe_code)]
        let mem = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<u64>();
        if mem.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let stack = alloc::boxed::Box::into_raw(alloc::boxed::Box::new(JitStack {
            top: mem,
            // SAFETY: one past the end of the allocation.
            #[allow(unsafe_code)]
            limit: unsafe { mem.add(STACK_WORDS) },
            frames: core::ptr::null_mut(),
            base: mem,
        }));
        Self { stack }
    }
}

impl Drop for StackOwner {
    fn drop(&mut self) {
        // SAFETY: `stack` came from `Box::into_raw` and its memory from
        // `alloc_zeroed` with this layout; the thread is exiting, so no context
        // (and no compiled code) uses either any more.
        #[allow(unsafe_code)]
        unsafe {
            let st = alloc::boxed::Box::from_raw(self.stack);
            let layout = std::alloc::Layout::array::<u64>(STACK_WORDS).expect("stack layout");
            std::alloc::dealloc(st.base.cast::<u8>(), layout);
        }
    }
}

std::thread_local! {
    static STACK: StackOwner = StackOwner::new();
}

/// The running thread's register stack, recorded in `ctx`.
fn stack_of(ctx: &mut Ctx) -> *mut JitStack {
    if ctx.jit_rt.stack.0.is_null() {
        ctx.jit_rt.stack.0 = STACK.with(|o| o.stack);
    }
    ctx.jit_rt.stack.0
}

/// Whether `regs` lies in the running thread's register stack (below its
/// top): a compiled activation's window, which the GC roots already.
fn on_register_stack(ctx: &Ctx, regs: &[NanBox]) -> bool {
    let p = ctx.jit_rt.stack.0;
    if p.is_null() {
        return false;
    }
    // SAFETY: `p` is this thread's live stack; only its bounds are read.
    #[allow(unsafe_code)]
    let (base, top) = unsafe { ((*p).base as usize, (*p).top as usize) };
    let a = regs.as_ptr() as usize;
    a >= base && a + regs.len() * 8 <= top
}

/// The live part of the register stack — every compiled activation's window
/// and native-call header — as GC roots (empty before first use).
pub(super) fn stack_roots<'c>(ctx: &'c Ctx<'_>) -> &'c [NanBox] {
    let p = ctx.jit_rt.stack.0;
    if p.is_null() {
        return &[];
    }
    // SAFETY: `p` is this thread's live stack; `base..top` is initialized (every
    // word below `top` was written when its activation was laid out) and no
    // compiled code runs while the caller (a safepoint) reads it.
    #[allow(unsafe_code)]
    unsafe {
        let st = &*p;
        let n = st.top.offset_from(st.base) as usize;
        core::slice::from_raw_parts(st.base.cast::<NanBox>(), n)
    }
}

/// Runs compiled `jit` on `regs` from bytecode pc `entry` with the activation's
/// handler stack `handlers` (non-empty only for an on-stack replacement inside a
/// `try`), translating its raw result word into the interpreter's frame outcome.
/// `None` if nothing ran (not a generic body, short register file, or not an
/// entry point).
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn run_compiled(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
    entry: usize,
    handlers: &[(usize, Reg)],
) -> Option<Result<FrameExit, VmError>> {
    run_compiled_at(ctx, funcs, jit, regs, entry, handlers, false, true)
}

/// [`run_compiled`] for a caller that drops `regs` afterwards: the
/// activation's final register values are not copied back.
fn run_compiled_fresh(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
) -> Option<Result<FrameExit, VmError>> {
    run_compiled_at(ctx, funcs, jit, regs, 0, &[], false, false)
}

/// [`run_compiled`]; when `exitable`, the activation's back-edge polls may
/// make it leave at a loop header: then it returns `Ok(Return(None))` and
/// [`TierRuntime::left`] holds the header and the handler stack there (the
/// register file is up to date). Otherwise `regs` holds the activation's
/// final registers only when `keep` (a caller that drops them need not have
/// them copied back).
#[inline]
#[allow(clippy::too_many_arguments)]
fn run_compiled_at(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
    entry: usize,
    handlers: &[(usize, Reg)],
    exitable: bool,
    keep: bool,
) -> Option<Result<FrameExit, VmError>> {
    let saved = ctx.jit_funcs.replace(funcs as *const [FnProto]);
    let outer_base = ctx.jit_rt.base;
    let base = ctx.jit_rt.handlers.len();
    ctx.jit_rt.base = base;
    ctx.jit_rt.handlers.extend_from_slice(handlers);
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    // The activation runs on the register stack (where the GC sees it and its
    // calls can stay native), on a copy of `regs` written back afterwards; on
    // the rare full stack, on `regs` itself.
    let st = stack_of(ctx);
    // SAFETY: `st` is this thread's live stack; nothing else borrows it.
    #[allow(unsafe_code)]
    let (top, room) = unsafe { ((*st).top, (*st).limit.offset_from((*st).top) as usize) };
    // A body that inlined calls has a larger window than the function's own
    // registers (its inlined bodies' sub-window, which starts `undefined`).
    let n = regs.len().max(jit.n_regs());
    let raw = if n <= room {
        // SAFETY: `top..top + n` is free, in-bounds stack memory; it is
        // claimed (by moving `top`) for exactly this activation.
        #[allow(unsafe_code)]
        let window = unsafe {
            core::ptr::copy_nonoverlapping(regs.as_ptr().cast::<u64>(), top, regs.len());
            for i in regs.len()..n {
                top.add(i).write(NanBox::undefined().to_bits());
            }
            (*st).top = top.add(n);
            core::slice::from_raw_parts_mut(top.cast::<NanBox>(), n)
        };
        let outer = (ctx.jit_rt.exit_regs, ctx.jit_rt.exit_jit);
        if exitable {
            ctx.jit_rt.exit_regs = window.as_ptr() as usize;
            ctx.jit_rt.exit_jit = jit as *const crate::jit::JitProto;
        }
        // SAFETY: as above.
        #[allow(unsafe_code)]
        let frames = unsafe { (*st).frames };
        let raw = jit.call_frame_raw(ctx_ptr, window, entry);
        // Every activation the body pushed (calls, inlined calls) is gone.
        #[allow(unsafe_code)]
        let after = unsafe { (*st).frames };
        debug_assert_eq!(frames, after, "a compiled activation left a call header");
        (ctx.jit_rt.exit_regs, ctx.jit_rt.exit_jit) = outer;
        if keep || raw == Some(NanBox::jit_exit_bits()) {
            let k = regs.len();
            regs.copy_from_slice(&window[..k]);
        }
        // SAFETY: as above; the activation is over.
        #[allow(unsafe_code)]
        unsafe {
            (*st).top = top;
        }
        raw
    } else {
        // (The rare full stack: such an activation never leaves. A body
        // whose window is larger than `regs` does not run: `call_frame_raw`
        // refuses it.)
        jit.call_frame_raw(ctx_ptr, regs, entry)
    };
    if raw == Some(NanBox::jit_exit_bits()) {
        return Some(left_at(ctx, base, outer_base, saved));
    }
    ctx.jit_rt.handlers.truncate(base);
    ctx.jit_rt.base = outer_base;
    ctx.jit_funcs = saved;
    Some(finish_raw(ctx, raw?))
}

/// The end of [`run_compiled_at`] for an activation that left at a loop
/// header (its handlers are `handlers[base..]`).
#[cold]
fn left_at(
    ctx: &mut Ctx,
    base: usize,
    outer_base: usize,
    saved: Option<*const [FnProto]>,
) -> Result<FrameExit, VmError> {
    let handlers = ctx.jit_rt.handlers.split_off(base);
    ctx.jit_rt.base = outer_base;
    ctx.jit_funcs = saved;
    match ctx.jit_rt.resume.take() {
        Some(pc) => {
            ctx.jit_rt.left = Some((pc, handlers));
            Ok(FrameExit::Return(None))
        }
        None => Err(VmError::Unsupported),
    }
}

/// A compiled activation's raw result word as the interpreter's frame
/// outcome.
fn finish_raw(ctx: &mut Ctx, raw: u64) -> Result<FrameExit, VmError> {
    if raw == NanBox::jit_falloff_bits() {
        return Ok(FrameExit::Return(None));
    }
    let nb = NanBox::from_bits(raw);
    if nb.is_jit_throw_sentinel() {
        let err = ctx
            .jit_pending_fault
            .take()
            .or_else(|| ctx.jit_pending.take().map(VmError::Thrown))
            .unwrap_or_else(|| {
                VmError::Thrown(vm_error(ctx, "Error", "JIT throw without pending value"))
            });
        return Err(err);
    }
    Ok(FrameExit::Return(Some(nb)))
}

/// The call-entry hook (`call_with_inner`, once function `id` is hot): compile
/// it if it is eligible and run it on the activation's register file `regs`
/// (already bound by the caller: parameters, captures, `this`). `None` means
/// "not compiled — interpret it".
pub(super) fn enter_on_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
    regs: &mut [NanBox],
    ic: &[crate::ic::PropertyCache],
) -> Option<Result<NanBox, VmError>> {
    let proto = funcs.get(id)?;
    if proto.is_async || proto.is_generator {
        return None;
    }
    let jit = jit_for(funcs, id, ic)?;
    if jit.is_generic() {
        return match run_tiering(ctx, funcs, id, jit, regs, 0, &[], ic, false)? {
            Ok(FrameExit::Return(v)) => Some(Ok(v.unwrap_or(NanBox::undefined()))),
            Ok(_) => Some(Err(VmError::Unsupported)),
            Err(e) => Some(Err(e)),
        };
    }
    // The Int/Float tiers model only the parameters (no captures, no `this`, no
    // rest array), and deopt (`None`) on any argument they cannot represent.
    if proto.rest_from.is_some() || proto.n_captures != 0 {
        return None;
    }
    // Arguments past the parameters are unobservable to such a body (it has
    // no `arguments` object), so a callback a built-in passes `(value, index,
    // array)` still runs compiled.
    jit.call_guarded(args.get(..proto.n_params)?).map(Ok)
}

/// On-stack replacement from a hot interpreted loop: `program` is the running
/// body, `regs` its register file, and `target` the loop header the back-edge is
/// about to jump to. If `program` is a function-table body (or its optimized
/// tier, which keeps every pc) that compiles, the rest of the activation runs in
/// native code from `target` — inheriting the frame's `handlers` — and its
/// outcome is returned; `None` means "keep interpreting" (and the caller stops
/// asking for this activation).
pub(super) fn osr(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    program: &[Op],
    regs: &mut [NanBox],
    target: usize,
    handlers: &[(usize, Reg)],
    ic: &[crate::ic::PropertyCache],
) -> Option<Result<FrameExit, VmError>> {
    if mode() == Mode::Off {
        return None;
    }
    let ptr = program.as_ptr();
    let id = funcs
        .iter()
        .position(|f| core::ptr::eq(f.ops.as_ptr(), ptr) && f.ops.len() == program.len())
        .or_else(|| {
            funcs.iter().position(|f| {
                f.tier
                    .body()
                    .is_some_and(|b| core::ptr::eq(b.as_ptr(), ptr) && b.len() == program.len())
            })
        })?;
    let proto = funcs.get(id)?;
    if proto.is_generator || proto.ops.len() != program.len() {
        return None;
    }
    let jit: Rc<crate::jit::JitProto> = jit_for(funcs, id, ic)?;
    if !jit.has_entry(target) {
        return None;
    }
    if jit_log_enabled() {
        jit_log(
            id,
            proto,
            &alloc::format!("on-stack replacement at pc {target}"),
        );
    }
    run_tiering(ctx, funcs, id, jit, regs, target, handlers, ic, true)
}

/// Runs function `id`'s compiled body `jit` on `regs` from `entry` with
/// `handlers` (as [`run_compiled`]) as an activation that may leave at a loop
/// header to be recompiled, continuing there in the new code; `ic` is the
/// activation's interpreter caches, if it holds them. `keep`: as
/// [`run_compiled_at`]'s.
#[allow(clippy::too_many_arguments)]
fn run_tiering(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    mut jit: Rc<crate::jit::JitProto>,
    regs: &mut [NanBox],
    entry: usize,
    handlers: &[(usize, Reg)],
    ic: &[crate::ic::PropertyCache],
    keep: bool,
) -> Option<Result<FrameExit, VmError>> {
    let mut r = run_compiled_at(ctx, funcs, &jit, regs, entry, handlers, true, keep)?;
    // The body left at loop header `pc` to be recompiled: continue there in
    // the function's newest code (the old body, which can always be entered
    // there, if the recompilation failed).
    while let Some((pc, h)) = ctx.jit_rt.left.take() {
        let new = recompile(ctx, funcs, id, &jit, ic).unwrap_or_else(|| Rc::clone(&jit));
        // (A recompilation with a larger window may not fit a full stack;
        // the old body, which ran here, does.)
        r = match run_compiled_at(ctx, funcs, &new, regs, pc, &h, true, keep) {
            Some(x) => {
                jit = new;
                x
            }
            None => run_compiled_at(ctx, funcs, &jit, regs, pc, &h, true, keep)?,
        };
    }
    Some(r)
}

/// Function `id`'s generic body `old`, compiled again with what its property
/// sites' caches learned (and, for sites it never ran, the interpreter's
/// caches `ic`): the function's code from now on. If another activation
/// already replaced `old`, that newer code. `None` if neither has an entry
/// wherever `old` has (it always does: both are lowered from the same
/// bytecode), or nothing compiled.
fn recompile(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    old: &Rc<crate::jit::JitProto>,
    ic: &[crate::ic::PropertyCache],
) -> Option<Rc<crate::jit::JitProto>> {
    let proto = funcs.get(id)?;
    if let Some(Some(cur)) = proto.tier.jit()
        && !Rc::ptr_eq(&cur, old)
    {
        return cur.is_generic().then_some(cur);
    }
    if old.generation() >= crate::jit::MAX_GENERATION {
        return None; // (only under `KATAAN_JIT_EXIT_STRESS`)
    }
    let mut warm = old.runtime_warm();
    if warm.len() < ic.len() {
        warm.resize(ic.len(), None);
    }
    for (w, c) in warm.iter_mut().zip(ic) {
        if w.is_none() {
            *w = c.warm_entry();
        }
    }
    // The callees its call sites settled on, else the interpreter's.
    let mut hints = old.call_hints();
    if hints.len() < ic.len() {
        hints.resize(ic.len(), crate::ic::CalleeHint::None);
    }
    for (h, c) in hints.iter_mut().zip(ic) {
        if *h == crate::ic::CalleeHint::None {
            *h = c.callee_hint();
        }
    }
    let mut new = crate::jit::JitProto::compile_generic_inlining(
        proto,
        &super::jit_generic_helpers(),
        &warm,
        &|f| {
            funcs
                .get(f as usize)
                .is_some_and(|p| !(p.is_async && !p.is_generator))
        },
        &|pc, _| inline_candidate(funcs, id, &hints, pc),
    )?;
    new.set_generation(old.generation() + 1);
    let new = Rc::new(new);
    if jit_log_enabled() {
        jit_log(
            id,
            proto,
            &alloc::format!(
                "recompiled (generation {}, {} calls inlined)",
                new.generation(),
                new.inlined()
            ),
        );
    }
    // The old code stays alive with the function: compiled callers may hold
    // its address.
    proto.tier.replace_jit(Rc::clone(&new));
    if let Some(d) = ctx.jit_rt.direct.get_mut(id)
        && d.is_some()
    {
        *d = Some(Rc::clone(&new));
    }
    Some(new)
}

/// `Op::PushHandler` in compiled code: push `(target, reg)` onto the running
/// activation's handler stack, bounded as the interpreter bounds its own (a
/// catchable `RangeError`, returned as the throw sentinel, on overflow).
///
/// # Safety
/// `ctx` must be the live, otherwise-unborrowed `Ctx` the body was entered with.
pub(crate) extern "C" fn jit_helper_push_handler(
    ctx: *mut core::ffi::c_void,
    target: u64,
    reg: u64,
) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let h = &mut ctx.jit_rt;
    if h.handlers.len() - h.base >= ctx.realm.limits.max_handler_depth {
        let e = vm_error(ctx, "RangeError", "Handler stack overflow");
        ctx.jit_pending_fault = Some(VmError::Thrown(e));
        return NanBox::jit_throw_bits();
    }
    // Lossless: the emitter passes a bytecode pc and a register index.
    h.handlers.push((target as usize, reg as Reg));
    0
}

/// A natively entered body's handler-stack segment
/// (`crate::jit::GenericHelpers::handler_frame`): with `leave == 0`, open a
/// segment of its own (from the stack's current top) and return the caller's
/// base; otherwise drop what the body left in its segment and restore the
/// caller's base `saved` — what [`run_compiled_at`] does around an
/// interpreter-entered activation.
///
/// # Safety
/// As [`jit_helper_push_handler`].
pub(crate) extern "C" fn jit_helper_handler_frame(
    ctx: *mut core::ffi::c_void,
    leave: u64,
    saved: u64,
) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let h = &mut ctx.jit_rt;
    if leave == 0 {
        let outer = h.base;
        h.base = h.handlers.len();
        return outer as u64;
    }
    h.handlers.truncate(h.base);
    h.base = saved as usize;
    0
}

/// `Op::PopHandler` in compiled code: drop the innermost handler of the running
/// activation (a no-op on an empty stack, as in the interpreter).
///
/// # Safety
/// As [`jit_helper_push_handler`].
pub(crate) extern "C" fn jit_helper_pop_handler(ctx: *mut core::ffi::c_void, _: u64) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let h = &mut ctx.jit_rt;
    if h.handlers.len() > h.base {
        h.handlers.pop();
    }
    0
}

/// The throw exit of a compiled body with `try` regions — the interpreter's
/// `handle_throw!`: if the pending fault is a thrown value and the activation
/// has a handler, pop it, bind the value in its register and return its catch pc
/// (the body resumes there); otherwise leave the fault pending and return
/// [`crate::jit::NO_HANDLER`] (the body returns the throw sentinel).
///
/// # Safety
/// As [`jit_helper_exec_op`]: `regs` is the activation's `n_regs`-long register
/// file.
pub(crate) extern "C" fn jit_helper_unwind(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: the activation's register file, live and unaliased for this call.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs) };
    if ctx.jit_rt.handlers.len() <= ctx.jit_rt.base {
        return crate::jit::NO_HANDLER;
    }
    // The same precedence `run_compiled` reads the pending fault with.
    let thrown = match ctx.jit_pending_fault.take() {
        Some(VmError::Thrown(v)) => v,
        Some(other) => {
            ctx.jit_pending_fault = Some(other);
            return crate::jit::NO_HANDLER;
        }
        None => match ctx.jit_pending.take() {
            Some(v) => v,
            None => return crate::jit::NO_HANDLER,
        },
    };
    let Some((target, reg)) = ctx.jit_rt.handlers.pop() else {
        return crate::jit::NO_HANDLER; // unreachable: checked above
    };
    match regs.get_mut(reg as usize) {
        Some(slot) => *slot = thrown,
        None => {
            ctx.jit_pending = Some(thrown);
            return crate::jit::NO_HANDLER;
        }
    }
    target as u64
}

/// The generic-tier helper that validates a global binding for direct access
/// from a compiled loop (`crate::jit::numeric`): for `Op::LoadGlobal`, the
/// address of the global scope's own initialized binding of the name — exactly
/// where [`vm_load_global`]'s first step reads; for `Op::StoreGlobal`, the
/// address of that binding when [`vm_store_global_fast`] would assign it without
/// mirroring the write onto the global object (the running scope is the global
/// scope, the binding is plain and initialized, the global object has no
/// like-named own property — and is a global object, `JIT_FLAG_GLOBAL`, whose
/// key changes the global binding epoch tracks). `0` otherwise. The address
/// stays valid, and the facts true, until Rust code next runs (which alone can
/// add or free a binding, change the running scope or realm, or give the
/// global object a property) — and past that for as long as the global binding
/// epoch (`crate::env`) is unchanged, since every such change but the running
/// scope's bumps it, and the running scope a compiled activation sees is the
/// same after each call returns (the generic tier's cache, which also starts
/// over in a `catch` handler, relies on this; promoted loops drop the address
/// after a call).
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with and `op` a
/// pointer into the bytecode the calling `JitProto` owns.
pub(crate) extern "C" fn jit_helper_global_ref(ctx: *mut core::ffi::c_void, op: u64) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op = unsafe { &*(op as usize as *const Op) };
    let Some(host) = ctx.host.as_deref() else {
        return 0;
    };
    let p = match op {
        Op::LoadGlobal { name, cache, .. } => {
            host.global_scope().jit_binding_ptr(name, cache, false)
        }
        Op::StoreGlobal {
            name,
            strict,
            resolved,
            cache,
            ..
        } if !(*strict && resolved.is_some()) && !name.starts_with('\u{1}') => {
            host.global_scope_if_current().and_then(|(scope, global)| {
                let p = scope.jit_binding_ptr(name, &cache.slot, true)?;
                let g = Handle::from_raw(global.as_handle()?);
                // Only a global object whose key changes bump the global
                // binding epoch (which the compiled cache holds the answer
                // across calls under).
                let tracked = ctx
                    .realm
                    .jit_object(g)
                    .is_some_and(|o| o.jit_flags() & crate::object::JIT_FLAG_GLOBAL != 0);
                let owned = super::global_owns_cached(&ctx.realm, g, name, cache);
                (tracked && !owned).then_some(p)
            })
        }
        _ => None,
    };
    p.map_or(0, |p| p as usize as u64)
}

/// `Math.f(x)` for an `Op::CallMethod` of one argument at `op`: the result
/// bits when `recv`'s own `key` is still the built-in `Math` function and `x` a
/// Number (`math_fast::call`, the interpreter's own shortcut: then the call has
/// no other effect), else the throw sentinel, on which the compiled code runs
/// the op through the interpreter instead (nothing has happened yet).
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with and `op` a
/// pointer into the bytecode the calling `JitProto` owns.
pub(crate) extern "C" fn jit_helper_math1(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    x: u64,
    op: u64,
) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op = unsafe { &*(op as usize as *const Op) };
    let Op::CallMethod { key, .. } = op else {
        return NanBox::jit_throw_bits();
    };
    super::math_fast::call(
        &ctx.realm,
        NanBox::from_bits(recv),
        key,
        &[0],
        &[NanBox::from_bits(x)],
    )
    .map_or(NanBox::jit_throw_bits(), NanBox::to_bits)
}

/// Records where a method site's key is found on `recv`
/// (`crate::jit::MethodSite::fill`, a pure read of the receiver's chain) so
/// the site's inline chain cache hits next time — for compiled code that
/// computes a built-in `Math.f(x)` inline after finding the method there.
/// Returns 0.
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with and `site`
/// the calling `JitProto`'s method site for the call.
pub(crate) extern "C" fn jit_helper_site_fill(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    site: u64,
    _unused: u64,
) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`,
    // and the site is the calling `JitProto`'s (touched only by this helper
    // while it runs).
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe {
        (
            &mut *(ctx as *mut Ctx),
            &mut *(site as usize as *mut crate::jit::MethodSite),
        )
    };
    if ctx.host.is_some()
        && let Some(h) = NanBox::from_bits(recv).as_handle()
    {
        site.fill(&mut ctx.realm, Handle::from_raw(h));
    }
    0
}

/// `Op::NewObject` from compiled code: a fresh ordinary object's handle bits
/// (exactly the interpreter arm, which cannot fail). It allocates and runs no
/// JavaScript.
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with.
pub(crate) extern "C" fn jit_helper_new_object(ctx: *mut core::ffi::c_void, _unused: u64) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    NanBox::handle(ctx.realm.new_object().to_raw()).to_bits()
}

/// `Op::NewArrayFrom` (and a small `Op::NewArray`) from compiled code: the
/// array literal, written to the op's destination register — exactly the
/// interpreter arm, without the one-op dispatch. Any other op (an `Op::NewArray`
/// past the length cap) is [`jit_helper_exec_op`]'s.
///
/// # Safety
/// As [`jit_helper_exec_op`].
pub(crate) extern "C" fn jit_helper_new_array(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    op: *const Op,
) -> u64 {
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op_ref = unsafe { &*op };
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`,
    // and `regs` the activation's `n_regs`-long register file (`NanBox` is
    // `repr(transparent)` over `u64`), live and unaliased for this call.
    #[allow(unsafe_code)]
    let (cx, file) = unsafe {
        (
            &mut *(ctx as *mut Ctx),
            core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs),
        )
    };
    let (dst, vals) = match op_ref {
        Op::NewArrayFrom { dst, elems } => {
            let h = cx
                .realm
                .new_array_from_iter(elems.iter().map(|r| file[*r as usize]));
            file[*dst as usize] = NanBox::handle(h.to_raw());
            return 0;
        }
        Op::NewArray { dst, len } if *len <= cx.realm.limits.max_array_len => {
            (*dst, alloc::vec![NanBox::undefined(); *len])
        }
        _ => return jit_helper_exec_op(ctx, regs, n_regs, op),
    };
    file[dst as usize] = NanBox::handle(cx.realm.new_array(vals).to_raw());
    0
}

/// The generic-tier helper that runs one bytecode op — any op without a native
/// lowering — through the interpreter itself, on the body's register file. The
/// op is run as a one-op program, which is exact for every op the tier accepts:
/// those touch only the register file and the context, never the frame's pc or
/// handler stack (`crate::jit::lower_nbvm_generic` refuses the ones that do). A
/// fault is stashed in `ctx.jit_pending_fault` and the throw sentinel returned.
///
/// # Safety
/// `ctx` must be the live `Ctx` (with `jit_funcs` set) the compiled body was
/// entered with, `regs` its `n_regs`-long register file, and `op` a pointer into
/// the bytecode the calling `JitProto` owns. The reentrancy is single-threaded:
/// the native caller holds no Rust borrow of either while this runs.
pub(crate) extern "C" fn jit_helper_exec_op(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    op: *const Op,
) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: `regs` is the activation's register file (`NanBox` is
    // `repr(transparent)` over `u64`), live and unaliased for this call.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op = unsafe { &*op };
    // Direct forms of the hottest handed-back ops — each exactly its
    // interpreter arm, without setting up a one-op frame.
    let direct = match op {
        Op::LoadGlobal { dst, name, cache } => {
            Some(vm_load_global(ctx, name, cache).map(|v| regs[*dst as usize] = v))
        }
        // `StoreGlobal`'s plain-binding fast path (the interpreter arm's first
        // step; its strict unresolvable-reference check precedes it).
        Op::StoreGlobal {
            name,
            src,
            strict,
            resolved,
            cache,
        } if !(*strict && resolved.is_some()) => {
            vm_store_global_fast(ctx, name, cache, regs[*src as usize]).then_some(Ok(()))
        }
        Op::CallValue { dst, callee, args } => {
            let callee = regs[*callee as usize];
            match call_compiled(ctx, funcs, regs, callee, args, NanBox::undefined()) {
                Some(r) => Some(r.map(|v| regs[*dst as usize] = v)),
                // `String(n)`, `parseInt(s)`, …: `call_closure`'s own first step.
                None if ctx.host.is_some() => {
                    let argv = super::ArgBuf::from_regs(regs, args);
                    super::native_call::try_call_value(&mut ctx.realm, callee, &argv).map(|v| {
                        regs[*dst as usize] = v;
                        Ok(())
                    })
                }
                None => None,
            }
        }
        // Literal allocations (an object or array literal per loop iteration):
        // exactly their interpreter arms, which cannot fail.
        Op::NewObject { dst } => {
            regs[*dst as usize] = NanBox::handle(ctx.realm.new_object().to_raw());
            Some(Ok(()))
        }
        // An object literal with a constant key set (its interpreter arm).
        Op::NewObjectLit { dst, lit } => {
            Some(super::objlit::vm_new_object_lit(ctx, lit, regs).map(|o| regs[*dst as usize] = o))
        }
        // A closure (its interpreter arm, which cannot fail).
        Op::MakeClosure {
            dst,
            func,
            captures,
        } => {
            let cells = super::ArgBuf::from_regs(regs, captures);
            let handle = ctx.realm.new_vm_function_from(*func, &cells);
            super::install_fn_name_length(&mut ctx.realm, handle, funcs.get(*func as usize));
            super::link_function_proto(ctx, handle);
            super::link_async_proto(ctx, funcs.get(*func as usize), handle);
            super::tag_new_fn_realm(ctx, handle);
            regs[*dst as usize] = NanBox::handle(handle.to_raw());
            Some(Ok(()))
        }
        // (A capture cell — a one-element array — per loop iteration.) The
        // interpreter's length cap stays its own (an error path).
        Op::NewArray { dst, len } if *len <= ctx.realm.limits.max_array_len => {
            let vals = alloc::vec![NanBox::undefined(); *len];
            regs[*dst as usize] = NanBox::handle(ctx.realm.new_array(vals).to_raw());
            Some(Ok(()))
        }
        Op::NewBox { dst } => {
            regs[*dst as usize] = NanBox::handle(ctx.realm.new_box().to_raw());
            Some(Ok(()))
        }
        Op::CellGet { dst, cell } => regs[*cell as usize].as_handle().map(|h| {
            regs[*dst as usize] = super::vm_cell_get(ctx, Handle::from_raw(h));
            Ok(())
        }),
        Op::CellSet { cell, src } => regs[*cell as usize].as_handle().map(|h| {
            super::vm_cell_set(ctx, Handle::from_raw(h), regs[*src as usize]);
            Ok(())
        }),
        Op::NewArrayFrom { dst, elems } => {
            let h = ctx
                .realm
                .new_array_from_iter(elems.iter().map(|r| regs[*r as usize]));
            regs[*dst as usize] = NanBox::handle(h.to_raw());
            Some(Ok(()))
        }
        // The array-iteration fast path (`iter_fast`) of a destructuring or a
        // spread: exactly the interpreter arms (a throw goes to the caller's
        // handlers either way).
        Op::IterOpen { iter, next, src }
            if *next == iter.wrapping_add(1)
                && super::iter_fast::can_open(&mut ctx.realm, regs[*src as usize]) =>
        {
            regs[*next as usize] = regs[*src as usize];
            regs[*iter as usize] = NanBox::number(0.0);
            Some(Ok(()))
        }
        Op::IterNext {
            dst,
            done,
            iter,
            next,
        } => Some(
            super::vm_iter_step(ctx, regs, *done, *iter, *next)
                .map(|v| regs[*dst as usize] = v.unwrap_or(NanBox::undefined())),
        ),
        Op::IterClose { iter, done, quiet } if regs[*iter as usize].is_number() => {
            if regs[*done as usize].as_boolean() == Some(true) {
                Some(Ok(()))
            } else {
                regs[*done as usize] = NanBox::boolean(true);
                match super::iter_fast::close(ctx, regs, *iter) {
                    Err(super::HostError::Fault) => Some(Err(VmError::Unsupported)),
                    Err(super::HostError::Thrown(e)) if !*quiet => Some(Err(VmError::Thrown(e))),
                    _ => Some(Ok(())),
                }
            }
        }
        _ => None,
    };
    if let Some(r) = direct {
        return match r {
            Ok(()) => 0,
            Err(e) => {
                ctx.jit_pending_fault = Some(e);
                NanBox::jit_throw_bits()
            }
        };
    }
    // A property op keeps its inline cache across executions, in a table
    // keyed by the op's address: the op lives in bytecode the `JitProto` owns,
    // which the function's tier slot keeps alive for longer than the `Ctx`, so the address
    // names this one site for as long as the table exists. (A fresh cache per
    // execution would miss, and re-arm, every single time.)
    let cached = matches!(
        op,
        Op::GetProp { .. }
            | Op::SetProp { .. }
            | Op::SetPropStrict { .. }
            | Op::InitProp { .. }
            | Op::CallMethod { .. }
    );
    let r = if cached {
        let key = core::ptr::from_ref(op) as usize;
        let mut ic = core::mem::take(&mut ctx.exec_op_ic);
        ic.clear();
        // Taken out of its entry, not removed (no tree rebalancing per run);
        // a reentrant run of the same op meanwhile sees an empty cache, as
        // a removed entry would show it.
        ic.push(
            ctx.exec_op_ics
                .get_mut(&key)
                .map(core::mem::take)
                .unwrap_or_default(),
        );
        let r = super::run_frame_at_ic(
            ctx,
            funcs,
            core::slice::from_ref(op),
            regs,
            0,
            alloc::vec::Vec::new(),
            None,
            &mut ic,
        );
        if let Some(c) = ic.pop() {
            match ctx.exec_op_ics.get_mut(&key) {
                Some(slot) => *slot = c,
                None => {
                    ctx.exec_op_ics.insert(key, c);
                }
            }
        }
        ctx.exec_op_ic = ic;
        r
    } else {
        run_frame(ctx, funcs, core::slice::from_ref(op), regs)
    };
    match r {
        Ok(FrameExit::Return(None)) => 0,
        Ok(_) => {
            ctx.jit_pending_fault = Some(VmError::Unsupported);
            NanBox::jit_throw_bits()
        }
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// `Op::CallValue` from compiled code to a callee that is itself compiled: the
/// interpreter's call (`Op::CallValue` → `call_with` → `call_with_inner`) with
/// everything that cannot apply to such a callee left out, entering the
/// callee's native body directly. `None` — nothing done, run the ordinary call —
/// unless the callee is a plain VM closure (not a class constructor, async,
/// generator, rest-parameter, `arguments` or `new.target` body) whose generic
/// body is already compiled and hot, in a single-realm run.
///
/// It keeps every piece of interpreter state the ordinary call keeps: the call
/// depth limits (a catchable `RangeError`), the callee/arguments stacks the
/// `arguments`/`caller` machinery reads, the GC publication of the caller's
/// registers and the callee's inputs, and the tier-up counter.
fn call_compiled(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    callee_val: NanBox,
    args: &[Reg],
    this: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let argv = super::ArgBuf::from_regs(regs, args);
    call_compiled_with(ctx, funcs, Some(regs), callee_val, this, &argv)
}

/// [`call_compiled`] with the arguments `args`, and the caller's registers
/// `publish` published for the GC — or, for a call from a Rust frame that
/// holds values the collector cannot see (`None`: a getter call from a
/// property helper), not, which declines collection inside the callee exactly
/// as the ordinary call from there does.
///
/// A body with a native-call entry is entered there ([`enter_native`]), as a
/// compiled caller's call sequence enters it: its window is laid out on the
/// register stack by the entry itself, with nothing copied in or out.
pub(super) fn call_compiled_with(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    publish: Option<&[NanBox]>,
    callee_val: NanBox,
    this: NanBox,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let handle = Handle::from_raw(callee_val.as_handle()?);
    let (fid, _) = ctx.realm.vm_function(handle)?;
    let id = fid as usize;
    let proto = funcs.get(id)?;
    if ctx.realm.multi_realm || ctx.call_depth == 0 {
        return None;
    }
    let jit = direct_jit(ctx, proto, id)?;
    // `call_with`'s recursion guard.
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
    {
        let e = vm_error(ctx, "RangeError", "Maximum call stack size exceeded");
        return Some(Err(VmError::Thrown(e)));
    }
    // The bookkeeping of `Op::CallValue` + `call_with` (+ `call_with_inner`'s
    // input publication), restored below on every outcome. A caller window on
    // the register stack is a GC root already.
    let mark = ctx.frame_shadow.len();
    let published = ctx.frames_published;
    if let Some(regs) = publish {
        if !on_register_stack(ctx, regs) {
            ctx.frame_shadow.extend_from_slice(regs);
        }
        ctx.frames_published += 1;
    }
    if jit.native_entry().is_some()
        && jit.n_regs() >= proto.n_regs
        && let Some(caps) = ctx
            .realm
            .vm_function(handle)
            .and_then(|(_, c)| (c.len() >= proto.n_captures).then_some(c.as_ptr()))
    {
        // SAFETY: `jit` is the callee's compiled body with a native-call entry
        // over its own register count; `caps` points at the closure's
        // captures, which the callee value (held by the caller) keeps alive.
        #[allow(unsafe_code)]
        let r = unsafe {
            enter_native(
                ctx,
                funcs,
                &jit,
                jit.n_regs(),
                callee_val,
                caps.cast::<u64>(),
                this,
                args,
            )
        };
        if let Some(r) = r {
            ctx.frames_published = published;
            ctx.frame_shadow.truncate(mark);
            return Some(r);
        }
    }
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let callee_mark = ctx.realm.vm_callee_stack.len();
    let args_mark = ctx.realm.vm_args_buf.len();
    ctx.realm.vm_callee_stack.push(callee_val);
    ctx.realm.vm_callee_args.push(args_mark);
    ctx.realm.vm_args_buf.extend_from_slice(args);
    // The callee's register file: parameters, captures, `this` (undefined).
    let mut cregs: alloc::vec::Vec<NanBox> = ctx.realm.vm_reg_pool.pop().unwrap_or_default();
    cregs.resize(proto.n_regs, NanBox::undefined());
    for (slot, a) in cregs.iter_mut().zip(
        ctx.realm.vm_args_buf[args_mark..]
            .iter()
            .take(proto.n_params),
    ) {
        *slot = *a;
    }
    if let Some((_, caps)) = ctx.realm.vm_function(handle) {
        for (i, c) in caps.iter().enumerate().take(proto.n_captures) {
            cregs[proto.n_params + i] = *c;
        }
    }
    if let Some(slot) = cregs.get_mut(proto.n_params + proto.n_captures) {
        *slot = this;
    }
    if ctx.gc_enabled {
        let n = ctx.realm.vm_args_buf.len();
        ctx.frame_shadow
            .extend_from_slice(&ctx.realm.vm_args_buf[args_mark..n]);
        if let Some((_, caps)) = ctx.realm.vm_function(handle) {
            ctx.frame_shadow.extend_from_slice(caps);
        }
        ctx.frame_shadow.push(this);
    }
    proto.tier.bump();
    let outcome = run_compiled_fresh(ctx, funcs, &jit, &mut cregs);
    ctx.realm.vm_callee_stack.truncate(callee_mark);
    ctx.realm.vm_callee_args.truncate(callee_mark);
    ctx.realm.vm_args_buf.truncate(args_mark);
    ctx.frames_published = published;
    ctx.frame_shadow.truncate(mark);
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    if ctx.realm.vm_reg_pool.len() < 64 {
        cregs.clear();
        ctx.realm.vm_reg_pool.push(cregs);
    }
    Some(match outcome? {
        Ok(FrameExit::Return(v)) => Ok(v.unwrap_or(NanBox::undefined())),
        Ok(_) => Err(VmError::Unsupported),
        Err(e) => Err(e),
    })
}

/// `Call(callee, this, args)` of the VM closure `callee` through `jit`'s
/// native-call entry, from Rust — the compiled call sequence's
/// (`crate::jit::call_seq`) layout: the native-call header, the arguments and
/// the callee's `n_regs` register window go on the register stack (so they
/// are GC roots, and the header stands in for the `vm_callee_stack` entry),
/// and the entry fills the window itself. `None` (nothing done) when the
/// register stack has no room. The caller has checked the depth limits and
/// published its own frame (or declined collection).
///
/// # Safety
/// `jit` is `callee`'s function's compiled body, with a native-call entry
/// over `n_regs` registers; `caps` points at the closure's `n_captures`
/// captured values, valid until the entry has run (it copies them first).
#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline(always)]
unsafe fn enter_native(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    n_regs: usize,
    callee: NanBox,
    caps: *const u64,
    this: NanBox,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    use crate::jit::FRAME_HEADER;
    let st = stack_of(ctx);
    let argc = args.len();
    // SAFETY: `st` is this thread's live register stack, which nothing else
    // borrows while this runs; every word written is in `top..limit`
    // (checked first), claimed by moving `top` and released before
    // returning, exactly as the compiled call sequence claims it.
    #[allow(unsafe_code)]
    let (hdr, window) = unsafe {
        let hdr = (*st).top;
        let room = (*st).limit.offset_from(hdr) as usize;
        if room < FRAME_HEADER + argc + n_regs {
            return None;
        }
        hdr.write((*st).frames as u64);
        hdr.add(1).write(callee.to_bits());
        hdr.add(2).write(ctx.realm.vm_callee_stack.len() as u64);
        hdr.add(3).write(argc as u64);
        hdr.add(4).write(NanBox::undefined().to_bits());
        let argv = hdr.add(FRAME_HEADER);
        // Unrolled for the built-ins' fixed arities (a loop over the
        // slice is a `memcpy` call).
        match *args {
            [a, b, c] => {
                argv.write(a.to_bits());
                argv.add(1).write(b.to_bits());
                argv.add(2).write(c.to_bits());
            }
            [a, b] => {
                argv.write(a.to_bits());
                argv.add(1).write(b.to_bits());
            }
            _ => {
                for (i, a) in args.iter().enumerate() {
                    argv.add(i).write(a.to_bits());
                }
            }
        }
        let window = hdr.add(FRAME_HEADER + argc);
        (*st).frames = hdr;
        (*st).top = window.add(n_regs);
        (hdr, window)
    };
    let published = ctx.frames_published;
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let saved = ctx.jit_funcs.replace(funcs as *const [FnProto]);
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    // SAFETY: the window, header and arguments were laid out above as the
    // entry's contract requires; `caps` is the caller's (see the function
    // contract).
    #[allow(unsafe_code)]
    let raw = unsafe {
        jit.call_native_entry(
            ctx_ptr,
            window,
            argc as u64,
            hdr.add(FRAME_HEADER),
            caps,
            this.to_bits(),
        )
    };
    ctx.jit_funcs = saved;
    // SAFETY: as above; the activation is over (the callee restored the
    // stack it used), so its header and window are released.
    #[allow(unsafe_code)]
    unsafe {
        (*st).top = hdr;
        (*st).frames = hdr.read() as *mut u64;
    }
    ctx.frames_published = published;
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    let raw = raw?;
    if raw == NanBox::jit_falloff_bits() {
        return Some(Ok(NanBox::undefined()));
    }
    let nb = NanBox::from_bits(raw);
    if nb.is_jit_throw_sentinel() {
        let err = ctx
            .jit_pending_fault
            .take()
            .or_else(|| ctx.jit_pending.take().map(VmError::Thrown))
            .unwrap_or_else(|| {
                VmError::Thrown(vm_error(ctx, "Error", "JIT throw without pending value"))
            });
        return Some(Err(err));
    }
    Some(Ok(nb))
}

/// One callee, called many times from a Rust loop (the array higher-order
/// built-ins, `super::array_hof`), entered as [`call_compiled_with`] enters
/// it — with the lookups that cannot change between calls done once: the
/// closure's function, its compiled body, its captures (a closure's capture
/// list is fixed at creation), and a register file reused across calls.
///
/// When the body has a native-call entry, a call is the compiled call
/// sequence's (`crate::jit::call_seq`), done from Rust: the native-call
/// header, the arguments and the callee's register window are laid out on
/// the register stack (so they are GC roots, and the header stands in for
/// the `vm_callee_stack` entry, as for a compiled caller), and the entry fills
/// the window itself. No register file is copied in or out.
pub(super) struct CompiledCallback {
    callee: NanBox,
    id: usize,
    jit: Rc<crate::jit::JitProto>,
    captures: alloc::vec::Vec<NanBox>,
    n_params: usize,
    /// The window size, when the native-call entry may be used.
    native_regs: Option<usize>,
    regs: alloc::vec::Vec<NanBox>,
}

impl CompiledCallback {
    /// The prepared call of `callee`, when [`call_compiled_with`] would enter
    /// its compiled body directly right now; `None` otherwise.
    pub(super) fn new(ctx: &mut Ctx, funcs: &[FnProto], callee: NanBox) -> Option<Self> {
        let handle = Handle::from_raw(callee.as_handle()?);
        let (fid, _) = ctx.realm.vm_function(handle)?;
        let id = fid as usize;
        let proto = funcs.get(id)?;
        if ctx.realm.multi_realm || ctx.call_depth == 0 {
            return None;
        }
        let jit = direct_jit(ctx, proto, id)?;
        let (_, caps) = ctx.realm.vm_function(handle)?;
        if caps.len() < proto.n_captures {
            return None;
        }
        let captures = caps.iter().copied().take(proto.n_captures).collect();
        // What `fill_call_site` requires of a native-call target.
        let native_regs =
            (jit.native_entry().is_some() && jit.n_regs() >= proto.n_regs).then_some(jit.n_regs());
        Some(Self {
            callee,
            id,
            jit,
            captures,
            n_params: proto.n_params,
            native_regs,
            regs: alloc::vec::Vec::new(),
        })
    }

    /// `Call(callee, this, args)` from a caller that published its own frame
    /// (so the callee may collect) — exactly [`call_compiled_with`]'s
    /// activation.
    pub(super) fn call(
        &mut self,
        ctx: &mut Ctx,
        funcs: &[FnProto],
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, VmError> {
        if ctx.call_depth >= ctx.realm.limits.max_call_depth
            || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        {
            let e = vm_error(ctx, "RangeError", "Maximum call stack size exceeded");
            return Err(VmError::Thrown(e));
        }
        if let Some(n_regs) = self.native_regs
            && let Some(r) = self.call_native(ctx, funcs, n_regs, this, args)
        {
            return r;
        }
        self.call_framed(ctx, funcs, this, args)
    }

    /// [`Self::call`] through the body's native-call entry; `None` (nothing
    /// done) when the register stack has no room.
    #[inline]
    fn call_native(
        &mut self,
        ctx: &mut Ctx,
        funcs: &[FnProto],
        n_regs: usize,
        this: NanBox,
        args: &[NanBox],
    ) -> Option<Result<NanBox, VmError>> {
        // SAFETY: `self.captures` holds the closure's `n_captures` captured
        // values (kept alive by the closure, which the caller roots).
        #[allow(unsafe_code)]
        unsafe {
            enter_native(
                ctx,
                funcs,
                &self.jit,
                n_regs,
                self.callee,
                self.captures.as_ptr().cast::<u64>(),
                this,
                args,
            )
        }
    }

    /// [`Self::call`] on a register file of its own, copied onto the register
    /// stack by [`run_compiled`].
    fn call_framed(
        &mut self,
        ctx: &mut Ctx,
        funcs: &[FnProto],
        this: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, VmError> {
        let proto = &funcs[self.id];
        let mark = ctx.frame_shadow.len();
        let published = ctx.frames_published;
        ctx.call_depth += 1;
        ctx.realm.vm_total_depth += 1;
        let callee_mark = ctx.realm.vm_callee_stack.len();
        let args_mark = ctx.realm.vm_args_buf.len();
        ctx.realm.vm_callee_stack.push(self.callee);
        ctx.realm.vm_callee_args.push(args_mark);
        ctx.realm.vm_args_buf.extend_from_slice(args);
        // The register file: parameters (missing ones `undefined`), captures,
        // `this`, the rest `undefined`.
        let regs = &mut self.regs;
        regs.clear();
        regs.resize(proto.n_regs, NanBox::undefined());
        let n = self.n_params.min(args.len()).min(regs.len());
        regs[..n].copy_from_slice(&args[..n]);
        for (slot, c) in regs.iter_mut().skip(self.n_params).zip(&self.captures) {
            *slot = *c;
        }
        if let Some(slot) = regs.get_mut(self.n_params + self.captures.len()) {
            *slot = this;
        }
        if ctx.gc_enabled {
            ctx.frame_shadow.extend_from_slice(args);
            ctx.frame_shadow.extend_from_slice(&self.captures);
            ctx.frame_shadow.push(this);
        }
        proto.tier.bump();
        let outcome = run_compiled_fresh(ctx, funcs, &self.jit, regs);
        ctx.realm.vm_callee_stack.truncate(callee_mark);
        ctx.realm.vm_callee_args.truncate(callee_mark);
        ctx.realm.vm_args_buf.truncate(args_mark);
        ctx.frames_published = published;
        ctx.frame_shadow.truncate(mark);
        ctx.call_depth -= 1;
        ctx.realm.vm_total_depth -= 1;
        match outcome {
            Some(Ok(FrameExit::Return(v))) => Ok(v.unwrap_or(NanBox::undefined())),
            Some(Ok(_)) | None => Err(VmError::Unsupported),
            Some(Err(e)) => Err(e),
        }
    }
}

/// The Int/Float-tier native code of `proto`, when [`super::unboxed_fast_call`]
/// would call it right now. Once true it stays true (a function's tier-up and
/// its compiled code are both permanent), so a loop calling `proto` many times
/// can look it up once and call [`crate::jit::JitProto::call_guarded`] itself.
pub(super) fn unboxed_target(proto: &FnProto) -> Option<Rc<crate::jit::JitProto>> {
    if proto.rest_from.is_some() || proto.n_captures != 0 || proto.is_async || proto.is_generator {
        return None;
    }
    let t = proto.tier.0.try_borrow().ok()?;
    if !call_is_hot(t.body.is_some()) {
        return None;
    }
    let j = t.jit.as_ref()?.as_ref()?;
    (!j.is_generic()).then(|| Rc::clone(j))
}

/// The getter half of a compiled body's static-key property read
/// (`jit_helper_get_prop`): when `vm_get_prop` would begin by calling a VM
/// getter found through the site's prototype-chain cache — the receiver an
/// ordinary object (not host-exotic, not an array `length`) without the key
/// as a cached own data slot — and that getter is a [`call_compiled`] target,
/// call it directly, `this` the receiver, no arguments. `None`: do the
/// ordinary read.
pub(crate) fn compiled_getter(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    cache: &mut crate::ic::PropertyCache,
) -> Option<Result<NanBox, VmError>> {
    // Only a site whose chain cache is armed on an accessor can hit one (and
    // only there is the repeated lookup below worth it).
    if !cache.chain.as_ref().is_some_and(|c| c.accessor) {
        return None;
    }
    let h = recv.as_handle().map(Handle::from_raw)?;
    if ctx.realm.is_host_exotic(h)
        || (key == "length" && ctx.realm.plain_array_length(h).is_some())
        || ctx.realm.object_cached_get(h, key, cache).is_some()
    {
        return None;
    }
    match ctx.realm.chain_cached_get(h, key, cache)? {
        crate::realm::ChainHit::Accessor(getter, _, _)
            if getter
                .as_handle()
                .is_some_and(|g| ctx.realm.is_vm_function(Handle::from_raw(g))) =>
        {
            call_compiled_with(ctx, funcs, None, getter, recv, &[])
        }
        _ => None,
    }
}

/// `Op::CallMethod` in compiled code. The one case it runs itself is the hot
/// one: a hosted call whose method (looked up through the site's persistent
/// cache `cache`) is a VM closure with a compiled body, which it enters
/// directly ([`call_compiled`]) — exactly what `run_frame_at`'s `Op::CallMethod`
/// arm does for such a method, minus the generic call path. Everything else
/// (built-ins, `Math` fast paths, accessors, primitives, the legacy `caller`,
/// an unhosted run) is that arm itself, via [`jit_helper_exec_op`].
///
/// # Safety
/// As [`jit_helper_exec_op`]; `cache` is the site's `PropertyCache`, owned by
/// the calling `JitProto` and not otherwise borrowed while this runs.
pub(crate) extern "C" fn jit_helper_call_method(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    cache: *mut core::ffi::c_void,
    call_site: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    if let Op::CallMethod {
        dst,
        recv,
        key,
        args,
    } = op
    {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
        // SAFETY: the activation's register file, live and unaliased here.
        #[allow(unsafe_code)]
        let regs = unsafe { core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs) };
        // SAFETY: the site's own cache (see the function contract).
        #[allow(unsafe_code)]
        let cache = unsafe { &mut *cache.cast::<crate::ic::PropertyCache>() };
        let recv_val = regs[*recv as usize];
        // A built-in method the site has resolved before: called directly
        // (the interpreter arm's first step too).
        {
            // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
            #[allow(unsafe_code)]
            let funcs = unsafe {
                &*ctx
                    .jit_funcs
                    .expect("jit_funcs set before generic dispatch")
            };
            if let Some(r) =
                super::native_call::try_call(ctx, funcs, recv_val, key, regs, args, cache)
            {
                return match r {
                    Ok(v) => {
                        regs[*dst as usize] = v;
                        0
                    }
                    Err(e) => {
                        ctx.jit_pending_fault = Some(e);
                        NanBox::jit_throw_bits()
                    }
                };
            }
        }
        if ctx.host.is_some()
            && key != "caller"
            && let Some(closure) = super::vm_method_of_cached(ctx, recv_val, key, cache)
        {
            // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
            #[allow(unsafe_code)]
            let funcs = unsafe {
                &*ctx
                    .jit_funcs
                    .expect("jit_funcs set before generic dispatch")
            };
            if !call_site.is_null() {
                // SAFETY: the op's own `CallSite` (see the function contract).
                #[allow(unsafe_code)]
                let site = unsafe { &mut *call_site.cast::<CallSite>() };
                fill_call_site(ctx, closure, site);
            }
            if let Some(r) = call_compiled(ctx, funcs, regs, closure, args, recv_val) {
                return match r {
                    Ok(v) => {
                        regs[*dst as usize] = v;
                        0
                    }
                    Err(e) => {
                        ctx.jit_pending_fault = Some(e);
                        NanBox::jit_throw_bits()
                    }
                };
            }
        }
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

/// [`direct_target`], memoized per function id.
fn direct_jit(ctx: &mut Ctx, proto: &FnProto, id: usize) -> Option<Rc<crate::jit::JitProto>> {
    match ctx.jit_rt.direct.get(id) {
        Some(Some(j)) => Some(Rc::clone(j)),
        _ => {
            let j = direct_target(proto)?;
            if ctx.jit_rt.direct.len() <= id {
                ctx.jit_rt.direct.resize(id + 1, None);
            }
            ctx.jit_rt.direct[id] = Some(Rc::clone(&j));
            Some(j)
        }
    }
}

/// `Op::CallValue`'s slow path in compiled code (the native call sequence's
/// fallback, see `crate::jit::call_seq`): point the site's cache `site` at the
/// callee when the sequence can enter it (a [`call_compiled`] target with a
/// native-call entry), then run the call as the interpreter would
/// ([`jit_helper_exec_op`], which enters a compiled callee directly).
///
/// # Safety
/// As [`jit_helper_exec_op`]; `site` is the op's `CallSite`, owned by the
/// calling `JitProto` and not otherwise borrowed while this runs.
pub(crate) extern "C" fn jit_helper_call_value(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    if let Op::CallValue { callee, .. } = op {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
        // SAFETY: the activation's register file, live and unaliased here.
        #[allow(unsafe_code)]
        let callee_val = unsafe { *regs_raw.add(*callee as usize) };
        // SAFETY: the site's own cache (see the function contract).
        #[allow(unsafe_code)]
        let site = unsafe { &mut *site.cast::<CallSite>() };
        fill_call_site(ctx, NanBox::from_bits(callee_val), site);
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

/// Points `site` at `callee` if the native call sequence may enter it: a VM
/// closure whose function [`direct_jit`] accepts and whose compiled body has
/// a native-call entry over the function's own register count.
fn fill_call_site(ctx: &mut Ctx, callee: NanBox, site: &mut CallSite) {
    let Some(raw) = callee.as_handle() else {
        return;
    };
    let Some((fid, n_caps)) = ctx
        .realm
        .vm_function(Handle::from_raw(raw))
        .map(|(f, caps)| (f, caps.len()))
    else {
        return;
    };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let id = fid as usize;
    let Some(proto) = funcs.get(id) else {
        return;
    };
    if n_caps < proto.n_captures {
        return;
    }
    let Some(jit) = direct_jit(ctx, proto, id) else {
        return;
    };
    let Some(code) = jit.native_entry() else {
        return;
    };
    // (An inlining body's window is larger than the function's own.)
    if jit.n_regs() < proto.n_regs {
        return;
    }
    // The compiled body stays alive for the context's life (`direct` and the
    // JIT cache never drop one), so the raw entry address does too.
    site.fill(crate::jit::CallTarget {
        key: u64::from(fid) | ((n_caps as u64) << 32),
        code,
        frame_bytes: (jit.n_regs() * 8) as u64,
    });
}

/// The call sequence's callee identification (`crate::jit::CalleeHelper`): a VM
/// closure's function id and capture count (`CalleeInfo::key`) and its
/// captures. A leaf read: no allocation, no throw.
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with.
pub(crate) extern "C" fn jit_callee_info(ctx: *mut core::ffi::c_void, callee: u64) -> CalleeInfo {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    match NanBox::from_bits(callee)
        .as_handle()
        .and_then(|h| ctx.realm.vm_function(Handle::from_raw(h)))
    {
        Some((fid, caps)) => CalleeInfo {
            key: u64::from(fid) | ((caps.len() as u64) << 32),
            caps: caps.as_ptr().cast::<u64>(),
        },
        None => CalleeInfo {
            key: u64::MAX,
            caps: core::ptr::null(),
        },
    }
}

/// `Op::Construct`'s slow path in compiled code: point the site's cache at
/// the constructor when the native sequence can run its `[[Construct]]`
/// ([`construct_target`]), then construct as the interpreter does.
///
/// # Safety
/// As [`jit_helper_call_value`].
pub(crate) extern "C" fn jit_helper_construct(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    if let Op::Construct { ctor, .. } = op {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let (ctx, ctor_val, site) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                NanBox::from_bits(*regs_raw.add(*ctor as usize)),
                &mut *site.cast::<CallSite>(),
            )
        };
        fill_construct_site(ctx, ctor_val, site);
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

/// `Op::SuperCallArgs`'s slow path in compiled code: point the site's cache
/// at the super constructor when the native sequence can run its
/// `[[Construct]]` ([`construct_target`]), then run `super(…)` as the
/// interpreter does.
///
/// # Safety
/// As [`jit_helper_call_value`].
pub(crate) extern "C" fn jit_helper_super_call(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    if let Op::SuperCallArgs { ctor, .. } = op {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let (ctx, ctor_val, site) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                NanBox::from_bits(*regs_raw.add(*ctor as usize)),
                &mut *site.cast::<CallSite>(),
            )
        };
        fill_construct_site(ctx, ctor_val, site);
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

/// Whether `fh` is a **forwarding** constructor: a derived class's default
/// constructor (`constructor(...args) { super(...args); }`, which forwards
/// its arguments list without iterating it) with no instance fields of its
/// own. Its `[[Construct]]` with `new.target` `nt` is its super
/// constructor's with the same arguments and `nt`: the binding of `this`
/// and its (absent) fields add nothing. Recognized by its shape — a derived
/// class constructor whose one parameter is the rest array in register 0,
/// passed as is to `Op::SuperCall` (the compiler builds a fresh array for
/// any spread a program writes, so only the synthesized constructor
/// forwards register 0).
fn is_forwarding_ctor(ctx: &Ctx, funcs: &[FnProto], fh: Handle) -> bool {
    let Some(p) = ctx
        .realm
        .vm_function(fh)
        .and_then(|(id, _)| funcs.get(id as usize))
    else {
        return false;
    };
    p.derived
        && p.class_ctor
        && p.n_params == 1
        && p.rest_from == Some(0)
        && p.ops
            .iter()
            .any(|o| matches!(o, Op::SuperCall { args: 0, .. }))
        && ctx.realm.jit_aux_of(fh).is_some()
        && ctx.realm.get_property(fh, super::FIELDS_KEY).is_none()
}

/// The constructor a construct of `fh` runs the body of: `fh` itself, or —
/// through a chain of forwarding constructors ([`is_forwarding_ctor`]) —
/// the first super constructor (a VM class constructor) that is not one,
/// with the forwarding ones passed (whose cells a cache relying on the chain
/// watches).
fn construct_body_of(
    ctx: &Ctx,
    funcs: &[FnProto],
    fh: Handle,
) -> Option<(Handle, alloc::vec::Vec<Handle>)> {
    let mut cur = fh;
    let mut passed = alloc::vec::Vec::new();
    while is_forwarding_ctor(ctx, funcs, cur) {
        if passed.len() >= 8 {
            return None;
        }
        passed.push(cur);
        let parent = ctx.realm.object_proto(cur)?;
        let class = ctx
            .realm
            .vm_function(parent)
            .and_then(|(id, _)| funcs.get(id as usize))
            .is_some_and(|p| p.class_ctor);
        if !class {
            return None;
        }
        cur = parent;
    }
    Some((cur, passed))
}

/// Points `site` at the constructor `ctor` if the native call sequence may
/// run its `[[Construct]]` ([`construct_target`]) — for a forwarding
/// constructor, the one whose body it runs ([`construct_body_of`]).
fn fill_construct_site(ctx: &mut Ctx, ctor: NanBox, site: &mut CallSite) {
    let Some(raw) = ctor.as_handle() else {
        return;
    };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let Some((body, _)) = construct_body_of(ctx, funcs, Handle::from_raw(raw)) else {
        return;
    };
    let Some((fid, n_caps)) = ctx.realm.vm_function(body).map(|(f, caps)| (f, caps.len())) else {
        return;
    };
    if let Some(proto) = funcs.get(fid as usize)
        && n_caps >= proto.n_captures
        && let Some(jit) = construct_target(proto)
        && let Some(code) = jit.native_entry()
        && jit.n_regs() >= proto.n_regs
    {
        site.fill(crate::jit::CallTarget {
            key: u64::from(fid) | ((n_caps as u64) << 32),
            code,
            frame_bytes: (jit.n_regs() * 8) as u64,
        });
    }
}

/// `Op::SuperCtor` / `Op::InitFields` in compiled code when the site's
/// identity cache (`crate::jit::IdentMemo`) missed: the interpreter's op,
/// then the cache refilled for a VM class constructor operand (`f` /
/// `ctor`, with an auxiliary object), its cell and auxiliary object watched
/// first: `SuperCtor`'s result is the constructor's `[[Prototype]]`, which
/// only a change to the watched cell (or a default intrinsic prototype)
/// changes; an `InitFields` of a constructor without an own instance-field
/// initializer does nothing, which only an addition to its watched
/// auxiliary object changes.
///
/// # Safety
/// As [`jit_helper_call_value`].
pub(crate) extern "C" fn jit_helper_ident_op(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
) -> u64 {
    let r = jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw);
    if r == NanBox::jit_throw_bits() {
        return r;
    }
    // SAFETY: see the function contract (as `jit_helper_exec_op`): the op is
    // done, the register file is live and unaliased again.
    #[allow(unsafe_code)]
    let (op, ctx, regs, site) = unsafe {
        (
            &*op_raw,
            &mut *(ctx_raw as *mut Ctx),
            core::slice::from_raw_parts(regs_raw.cast_const().cast::<NanBox>(), n_regs),
            &mut *site.cast::<CallSite>(),
        )
    };
    let (key, value) = match op {
        Op::SuperCtor { dst, f } => (regs[*f as usize], regs[*dst as usize]),
        Op::InitFields { ctor, .. } => (regs[*ctor as usize], NanBox::undefined()),
        _ => return r,
    };
    let Some(fh) = key
        .as_handle()
        .map(Handle::from_raw)
        .filter(|&h| ctx.realm.is_vm_function(h))
    else {
        return r;
    };
    let Some(aux) = ctx.realm.jit_aux_of(fh) else {
        return r;
    };
    if matches!(op, Op::InitFields { .. })
        && ctx.realm.get_property(fh, super::FIELDS_KEY).is_some()
    {
        return r;
    }
    ctx.realm.watch_link(fh.to_raw(), Some(aux.to_raw()));
    site.ident = crate::jit::IdentMemo {
        key: key.to_bits(),
        stamp: crate::object::proto_epoch(),
        value: value.to_bits(),
    };
    r
}

/// `Op::CallKey`'s call half in compiled code (the native sequence's
/// fallback): the method `recv[key]` was already read (once, by the body);
/// point the site's cache at it when the sequence can enter it, then call it
/// with `this` the receiver exactly as `run_frame_at`'s `Op::CallKey` arm does
/// after its read.
///
/// # Safety
/// As [`jit_helper_call_value`].
pub(crate) extern "C" fn jit_helper_call_this(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    method: u64,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let (op, ctx, regs, site) = unsafe {
        (
            &*op_raw,
            &mut *(ctx_raw as *mut Ctx),
            core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs),
            &mut *site.cast::<CallSite>(),
        )
    };
    let Op::CallKey {
        dst, recv, args, ..
    } = op
    else {
        ctx.jit_pending_fault = Some(VmError::Unsupported);
        return NanBox::jit_throw_bits();
    };
    let method = NanBox::from_bits(method);
    fill_call_site(ctx, method, site);
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let recv_val = regs[*recv as usize];
    let argv: alloc::vec::Vec<NanBox> = args.iter().map(|a| regs[*a as usize]).collect();
    let pub_mark = ctx.frame_shadow.len();
    ctx.frame_shadow.extend_from_slice(regs);
    ctx.frames_published += 1;
    let r = super::call_closure(ctx, funcs, method, &argv, recv_val);
    ctx.frames_published -= 1;
    ctx.frame_shadow.truncate(pub_mark);
    match r {
        Ok(v) => {
            regs[*dst as usize] = v;
            0
        }
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// Whether the native sequence may run `new f()` for function `id`
/// (`proto`): what [`direct_target`] accepts for a call, except that a class
/// constructor is fine (this *is* its `[[Construct]]`), derived or not.
fn construct_target(proto: &FnProto) -> Option<Rc<crate::jit::JitProto>> {
    if proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto
            .ops
            .iter()
            .take(2)
            .any(|o| matches!(o, Op::MakeArguments { .. }))
        || !call_is_hot(proto.tier.has_body())
    {
        return None;
    }
    match proto.tier.jit() {
        Some(Some(j)) if j.is_generic() => Some(j),
        _ => None,
    }
}

/// A construct's preparation for the native call sequence
/// (`crate::jit::ConstructHelper`) — `new ctor()` (`nt` is `ctor`) or
/// `super()` (`nt` the `new.target`): when `ctor` is the VM function `site`
/// names and `vm_construct` would run it as an ordinary `[[Construct]]` —
/// a constructor (`VM_CTOR`) that is either a derived class constructor
/// (`this` starts unbound) or has no instance-field initializer to run first
/// and a VM function `nt` whose `prototype` is an object — prepare `this`
/// exactly as it does (`OrdinaryCreateFromConstructor` from `nt`, or the
/// unbound sentinel), store it to `out` and report the callee as
/// [`jit_callee_info`]. Anything else changes nothing and reports the
/// not-a-closure key.
///
/// # Safety
/// `ctx` the live `Ctx` (with `jit_funcs` set), `out` writable, `site` the
/// calling sequence's cache.
pub(crate) extern "C" fn jit_construct_info(
    ctx_raw: *mut core::ffi::c_void,
    ctor: u64,
    nt: u64,
    out: *mut u64,
    site: *const CallSite,
) -> CalleeInfo {
    const NONE: CalleeInfo = CalleeInfo {
        key: u64::MAX,
        caps: core::ptr::null(),
    };
    // SAFETY: see the function contract (the site is the calling `JitProto`'s,
    // interior-mutable: only this helper touches its construct cache).
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe { (&mut *(ctx_raw as *mut Ctx), &mut *site.cast_mut()) };
    let Some(called) = NanBox::from_bits(ctor).as_handle().map(Handle::from_raw) else {
        return NONE;
    };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // The constructor whose body runs (past forwarding default constructors).
    let Some((fh, passed)) = construct_body_of(ctx, funcs, called) else {
        return NONE;
    };
    let body_val = NanBox::handle(fh.to_raw()).to_bits();
    let Some((fid, n_caps)) = ctx.realm.vm_function(fh).map(|(f, c)| (f, c.len())) else {
        return NONE;
    };
    if !site.holds(u64::from(fid) | ((n_caps as u64) << 32)) {
        return NONE;
    }
    let store = |out: *mut u64, v: NanBox| {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        unsafe {
            *out = v.to_bits();
        }
    };
    // The site's last constructor, still as it was: no lookups.
    if nt == ctor
        && passed.is_empty()
        && let Some(p) = site.construct.hit(&ctx.realm, ctor)
    {
        fill_construct_fast(ctx, site, (fh, fid), (ctor, &passed), nt, Some(p));
        let obj = ctx.realm.jit_new_object_with_proto(p);
        store(out, NanBox::handle(obj.to_raw()));
        return jit_callee_info(ctx_raw, ctor);
    }
    let Some(proto) = funcs.get(fid as usize) else {
        return NONE;
    };
    if ctx.realm.get_property(fh, super::VM_CTOR).is_none() {
        return NONE;
    }
    if proto.derived {
        // `this` is bound by `super(...)`.
        if proto.class_ctor {
            fill_construct_fast(ctx, site, (fh, fid), (ctor, &passed), nt, None);
        }
        store(out, NanBox::tdz());
        return jit_callee_info(ctx_raw, body_val);
    }
    if proto.class_ctor && ctx.realm.get_property(fh, super::FIELDS_KEY).is_some() {
        return NONE;
    }
    // `new.target.prototype`, read as `vm_construct` reads it for a VM
    // function `new.target`.
    let Some(nth) = NanBox::from_bits(nt)
        .as_handle()
        .map(Handle::from_raw)
        .filter(|&h| ctx.realm.is_vm_function(h))
    else {
        return NONE;
    };
    let Some(p) = ctx
        .realm
        .get_property(nth, "prototype")
        .filter(|p| super::is_object_value(&ctx.realm, *p))
        .and_then(|p| p.as_handle())
    else {
        return NONE;
    };
    if nt == ctor && passed.is_empty() {
        site.construct.fill(&ctx.realm, ctor, NanBox::handle(p));
    }
    fill_construct_fast(
        ctx,
        site,
        (fh, fid),
        (ctor, &passed),
        nt,
        Some(Handle::from_raw(p)),
    );
    let obj = ctx.realm.jit_new_object_with_proto(Handle::from_raw(p));
    store(out, NanBox::handle(obj.to_raw()));
    jit_callee_info(ctx_raw, body_val)
}

/// Records a construct of the value `ctor` with `new.target` `nt`, which
/// runs the body of the constructor `fh` (function `fid`: `ctor` itself, or
/// past the forwarding constructors `passed`), in `site`'s inline
/// constructor cache (`crate::jit::ConstructFast`) when the identities fix
/// its `[[Construct]]` (`jit_construct_info` having checked the rest): a
/// derived class constructor (`proto` `None`: `this` starts unbound), or a
/// base constructor without an instance-field initializer whose `new.target`
/// is a class constructor, whose `prototype` (`proto`) is then non-writable
/// and non-configurable; the site holding a way for it. Every cell involved
/// and its auxiliary object are watched, so a later change to their own
/// properties or `[[Prototype]]` (which picks a forwarding constructor's
/// super constructor) retires the record with the prototype epoch.
fn fill_construct_fast(
    ctx: &mut Ctx,
    site: &mut CallSite,
    (fh, fid): (Handle, u32),
    (ctor, passed): (u64, &[Handle]),
    nt: u64,
    proto: Option<Handle>,
) {
    if site.fast.ctor == ctor
        && site.fast.nt == nt
        && site.fast.stamp == crate::object::proto_epoch()
    {
        return;
    }
    let Some((n_caps, caps)) = ctx
        .realm
        .vm_function(fh)
        .map(|(_, c)| (c.len(), c.as_ptr().cast::<u64>()))
    else {
        return;
    };
    let Some(target) = site.way_of(u64::from(fid) | ((n_caps as u64) << 32)) else {
        return;
    };
    let Some(aux) = ctx.realm.jit_aux_of(fh) else {
        return;
    };
    if ctx.realm.get_property(fh, super::VM_CTOR).is_none()
        || (proto.is_some() && ctx.realm.get_property(fh, super::FIELDS_KEY).is_some())
    {
        return;
    }
    if let Some(p) = proto {
        // `new.target`: a class constructor, its `prototype` fixed.
        let Some(nth) = NanBox::from_bits(nt).as_handle().map(Handle::from_raw) else {
            return;
        };
        // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
        #[allow(unsafe_code)]
        let funcs = unsafe {
            &*ctx
                .jit_funcs
                .expect("jit_funcs set before generic dispatch")
        };
        let class = ctx
            .realm
            .vm_function(nth)
            .and_then(|(id, _)| funcs.get(id as usize))
            .is_some_and(|f| f.class_ctor);
        let Some(nt_aux) = ctx.realm.jit_aux_of(nth) else {
            return;
        };
        let fixed = class
            && ctx.realm.jit_object(nt_aux).is_some_and(|o| {
                o.is_readonly("prototype")
                    && o.is_non_configurable("prototype")
                    && o.get("prototype").and_then(|v| v.as_handle()) == Some(p.to_raw())
            });
        if !fixed {
            return;
        }
        ctx.realm.watch_link(nth.to_raw(), Some(nt_aux.to_raw()));
    }
    ctx.realm.watch_link(fh.to_raw(), Some(aux.to_raw()));
    for &f in passed {
        let a = ctx.realm.jit_aux_of(f).map(Handle::to_raw);
        ctx.realm.watch_link(f.to_raw(), a);
    }
    site.fast = crate::jit::ConstructFast {
        ctor,
        nt,
        stamp: crate::object::proto_epoch(),
        proto: proto.map_or(0, Handle::to_raw),
        derived: u64::from(proto.is_none()),
        target,
        caps,
    };
}

/// The `this` of a `new` its site's inline constructor cache serves
/// (`crate::jit::ConstructAllocHelper`): a fresh ordinary object whose
/// `[[Prototype]]` is `proto`, as `jit_construct_info` makes it.
///
/// # Safety
/// `ctx` the live `Ctx`; `proto` the raw handle of a live object.
pub(crate) extern "C" fn jit_construct_alloc(ctx: *mut core::ffi::c_void, proto: u64) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let obj = ctx.realm.jit_new_object_with_proto(Handle::from_raw(proto));
    NanBox::handle(obj.to_raw()).to_bits()
}

/// A construct's result when the constructor body returned the heap value
/// `result`, or `this` started unbound (`crate::jit::ConstructResultHelper`):
/// `result` if it is an object, else `this` — `vm_construct`'s last step;
/// for a derived constructor, the `ReferenceError` (unbound `this`) or
/// `TypeError` (anything else) it throws, pending.
///
/// # Safety
/// `ctx` the live `Ctx`.
pub(crate) extern "C" fn jit_construct_result(
    ctx: *mut core::ffi::c_void,
    result: u64,
    this: u64,
) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    if super::is_object_value(&ctx.realm, NanBox::from_bits(result)) {
        return result;
    }
    if !NanBox::from_bits(this).is_tdz() {
        return this;
    }
    // A derived constructor's body result (`Op::DerivedResult`), checked as
    // `vm_construct` checks it.
    let e = if NanBox::from_bits(result).is_tdz() {
        super::vm_error(
            ctx,
            "ReferenceError",
            "Must call super constructor in derived class before returning",
        )
    } else {
        super::vm_error(
            ctx,
            "TypeError",
            "Derived constructors may only return object or undefined",
        )
    };
    ctx.jit_pending_fault = Some(VmError::Thrown(e));
    NanBox::jit_throw_bits()
}

/// `super(…)` when `this` is already bound (`crate::jit::SuperTwiceHelper`):
/// the `ReferenceError` the interpreter's `super(…)` throws, pending.
///
/// # Safety
/// `ctx` the live `Ctx`.
pub(crate) extern "C" fn jit_super_twice(ctx: *mut core::ffi::c_void) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let e = super::vm_error(
        ctx,
        "ReferenceError",
        "Super constructor may only be called once",
    );
    ctx.jit_pending_fault = Some(VmError::Thrown(e));
    NanBox::jit_throw_bits()
}

/// A method call's callee lookup for the native call sequence
/// (`crate::jit::MethodHelper`), when the site's inline chain cache missed:
/// exactly the case `jit_helper_call_method` enters directly — a hosted call,
/// not the legacy `caller`, whose method (through the site's property cache's
/// chain cache) is a VM closure — reported as [`jit_callee_info`] does, with
/// the closure stored to `out`; and the site's inline cache refilled from the
/// receiver. Pure: it only reads (and refills the caches), so the slow path
/// may repeat it.
///
/// # Safety
/// `ctx` the live `Ctx`; `site` the calling `JitProto`'s method site; `out`
/// writable.
pub(crate) extern "C" fn jit_method_info(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    site: *mut crate::jit::MethodSite,
    out: *mut u64,
) -> CalleeInfo {
    const NONE: CalleeInfo = CalleeInfo {
        key: u64::MAX,
        caps: core::ptr::null(),
    };
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let (ctx_ref, site) = unsafe { (&mut *(ctx as *mut Ctx), &mut *site) };
    // SAFETY: the site's property cache, owned by the same `JitProto`.
    #[allow(unsafe_code)]
    let cache = unsafe { &mut *site.cache.cast::<crate::ic::PropertyCache>() };
    let key = site.key();
    debug_assert_eq!(ctx_ref.host.is_some(), ctx_ref.jit_rt.hosted);
    // A site that last called a built-in (`s.charCodeAt`, `m.get`, …) goes
    // straight to its helper, which serves it (and, if the method has become
    // a VM closure, enters that compiled too).
    if ctx_ref.host.is_none() || key == "caller" || super::native_call::is_native_site(cache) {
        return NONE;
    }
    match super::vm_method_of_cached(ctx_ref, NanBox::from_bits(recv), key, cache) {
        Some(f) => {
            if let Some(h) = NanBox::from_bits(recv).as_handle() {
                site.fill(&mut ctx_ref.realm, Handle::from_raw(h));
            }
            // SAFETY: see the function contract.
            #[allow(unsafe_code)]
            unsafe {
                *out = f.to_bits();
            }
            jit_callee_info(ctx, f.to_bits())
        }
        None => NONE,
    }
}

/// Where the call sequence finds the state it maintains (see
/// `crate::jit::CallLayout`).
pub(super) fn call_layout() -> crate::jit::CallLayout {
    use core::mem::offset_of;
    let i = |v: usize| i32::try_from(v).expect("layout offset fits in i32");
    // A `Vec`'s length word sits at the same offset whatever its element type.
    let probe: alloc::vec::Vec<NanBox> = alloc::vec::Vec::with_capacity(5);
    let len_off = vec_len_offset(&probe);
    crate::jit::CallLayout {
        stack: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, stack)),
        call_depth: i(offset_of!(Ctx<'static>, call_depth)),
        frames_published: i(offset_of!(Ctx<'static>, frames_published)),
        realm: i(offset_of!(Ctx<'static>, realm)),
        total_depth: i(offset_of!(crate::realm::Realm, vm_total_depth)),
        max_depth: i(offset_of!(crate::realm::Realm, limits)
            + offset_of!(crate::limits::Limits, max_call_depth)),
        multi_realm: i(offset_of!(crate::realm::Realm, multi_realm)),
        callee_len: i(offset_of!(crate::realm::Realm, vm_callee_stack) + len_off),
        hosted: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, hosted)),
        main_realm: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, main_realm)),
        vmfn_plain: i(crate::realm::Realm::jit_vmfn_plain_offset()),
    }
}

/// The byte offset of the length word inside `v` (a `Vec` is three words —
/// pointer, capacity, length — in an unspecified order), found by pushing an
/// element and seeing which word changed by one.
fn vec_len_offset(v: &[NanBox]) -> usize {
    let words = |v: &alloc::vec::Vec<NanBox>| -> [usize; 3] {
        // SAFETY: a `Vec` is exactly three `usize`-sized words (checked below).
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::read((v as *const alloc::vec::Vec<NanBox>).cast::<[usize; 3]>())
        }
    };
    assert_eq!(core::mem::size_of::<alloc::vec::Vec<NanBox>>(), 3 * 8);
    let mut w = v.to_vec();
    w.reserve(8);
    let before = words(&w);
    w.push(NanBox::undefined());
    let after = words(&w);
    let at = (0..3)
        .find(|&k| after[k] == before[k] + 1 && after[k] == w.len())
        .expect("Vec length word");
    at * 8
}

/// The realm's activation stack (`vm_callee_stack`, `vm_callee_args`,
/// `vm_args_buf`) merged with the native-call headers on the register stack —
/// the full innermost-last list of VM activations the legacy `f.caller` /
/// `f.arguments` read. `None` when no native call is in flight (the realm's
/// stacks are already the whole list).
pub(super) fn callee_view(
    ctx: &Ctx,
) -> Option<(
    alloc::vec::Vec<NanBox>,
    alloc::vec::Vec<usize>,
    alloc::vec::Vec<NanBox>,
)> {
    let st = STACK.with(|o| o.stack);
    // Native activations, innermost first: (stack mark, callee, arguments).
    let mut natives: alloc::vec::Vec<(usize, NanBox, &[NanBox])> = alloc::vec::Vec::new();
    // SAFETY: this thread's stack; every header below `top` was written by a
    // call sequence and is linked from `frames`. No compiled code runs now.
    #[allow(unsafe_code)]
    unsafe {
        let mut f = (*st).frames;
        while !f.is_null() {
            let argc = *f.add(3) as usize;
            natives.push((
                *f.add(2) as usize,
                NanBox::from_bits(*f.add(1)),
                core::slice::from_raw_parts(f.add(FRAME_HEADER).cast::<NanBox>(), argc),
            ));
            f = (*f) as *mut u64;
        }
    }
    if natives.is_empty() {
        return None;
    }
    natives.reverse();
    let r = &ctx.realm;
    let n = r.vm_callee_stack.len();
    let mut callees = alloc::vec::Vec::new();
    let mut starts = alloc::vec::Vec::new();
    let mut buf = alloc::vec::Vec::new();
    let mut push = |c: NanBox, a: &[NanBox]| {
        callees.push(c);
        starts.push(buf.len());
        buf.extend_from_slice(a);
    };
    let mut k = 0;
    for i in 0..=n {
        // A native activation whose call saw `i` realm entries sits above them.
        while k < natives.len() && (natives[k].0 <= i || i == n) {
            push(natives[k].1, natives[k].2);
            k += 1;
        }
        if i < n {
            let start = r.vm_callee_args[i];
            let end = r
                .vm_callee_args
                .get(i + 1)
                .copied()
                .unwrap_or(r.vm_args_buf.len());
            push(r.vm_callee_stack[i], &r.vm_args_buf[start..end]);
        }
    }
    Some((callees, starts, buf))
}

/// Whether the function `proto` may be entered by [`call_compiled`]: a plain
/// VM closure body (not a class constructor, async, generator, rest-parameter
/// or `arguments` one) that is hot and already compiled to the generic tier —
/// its compiled body, if so.
fn direct_target(proto: &FnProto) -> Option<Rc<crate::jit::JitProto>> {
    if proto.class_ctor
        || proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        // (`new.target` is fine: the callee's register starts `undefined`,
        // a call's `new.target`.)
        || proto
            .ops
            .iter()
            .take(2)
            .any(|o| matches!(o, Op::MakeArguments { .. }))
        || !call_is_hot(proto.tier.has_body())
    {
        return None;
    }
    match proto.tier.jit() {
        Some(Some(j)) if j.is_generic() => Some(j),
        Some(Some(_)) => generic_twin(proto),
        _ => None,
    }
}

/// The generic-tier body compiled callers enter for a function whose own
/// native code is an Int/Float-tier body: those have no native-call entry,
/// so without it every call from compiled code took the interpreter's call
/// (several times the cost of the call sequence). Only for a body without a
/// loop or a static call — straight-line code, where the unboxed tier gains
/// nothing over the generic one; a loop or a recursion (`fib`) keeps its
/// Int-tier body. Both are compilations of the same bytecode, so either one
/// runs any call exactly.
fn generic_twin(proto: &FnProto) -> Option<Rc<crate::jit::JitProto>> {
    if let Some(t) = proto.tier.twin() {
        return t;
    }
    let straight = !proto.ops.iter().enumerate().any(|(pc, op)| match op {
        Op::Jump { target } | Op::JumpIfFalse { target, .. } => *target <= pc,
        Op::Call { .. } => true,
        _ => false,
    });
    let twin = straight
        .then(|| {
            crate::jit::JitProto::compile_generic_warm(
                proto,
                &super::jit_generic_helpers(),
                &super::tier_warm(proto),
            )
        })
        .flatten()
        .filter(|j| j.is_generic())
        .map(Rc::new);
    proto.tier.set_twin(twin.clone());
    twin
}

/// The generic-tier loop back-edge poll — what the interpreter does on a
/// backward `Jump`: observe a host interrupt, give a scheduled agent its turn,
/// and offer the GC a safepoint over this frame's registers. Returns the number
/// of back-edges until the next poll, or the throw sentinel (with
/// `VmError::Interrupted` pending).
///
/// # Safety
/// As [`jit_helper_exec_op`]; `ops`/`n_ops` is the running bytecode.
pub(crate) extern "C" fn jit_helper_backedge(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    ops: *const Op,
    n_ops: usize,
    header: usize,
) -> u64 {
    let regs_addr = regs as usize;
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts(regs.cast::<NanBox>(), n_regs) };
    // SAFETY: the `JitProto`'s owned bytecode, alive while its code runs.
    #[allow(unsafe_code)]
    let program = unsafe { core::slice::from_raw_parts(ops, n_ops) };
    if ctx
        .realm
        .interrupt
        .as_ref()
        .is_some_and(crate::interrupt::Interrupt::is_tripped)
    {
        ctx.jit_pending_fault = Some(VmError::Interrupted);
        return NanBox::jit_throw_bits();
    }
    if ctx.realm.agents_active {
        if vm_agent_tick(ctx) {
            ctx.jit_pending_fault = Some(VmError::Interrupted);
            return NanBox::jit_throw_bits();
        }
        vm_safepoint(ctx, funcs, program, regs);
        return if regs_addr == ctx.jit_rt.exit_regs && wants_to_leave(ctx, header) {
            NanBox::jit_exit_bits()
        } else {
            1
        };
    }
    vm_safepoint(ctx, funcs, program, regs);
    if regs_addr == ctx.jit_rt.exit_regs && wants_to_leave(ctx, header) {
        return NanBox::jit_exit_bits();
    }
    ctx.jit_rt.poll_interval
}

/// `KATAAN_JIT_EXIT_STRESS` (a test knob, read once per process): every
/// back-edge of an activation that may leave at a loop header polls and
/// leaves there, to be recompiled while the budget lasts and re-entered
/// unchanged after.
fn exit_stress() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_EXIT_STRESS").is_some_and(|v| v != "0"))
}

/// Whether the polling activation, the one that may leave
/// ([`TierRuntime`]'s exit fields: its window is the one polling), should
/// leave at loop header `header` to be recompiled: its body wants it
/// (`JitProto::wants_recompile`). Records the header if so.
fn wants_to_leave(ctx: &mut Ctx, header: usize) -> bool {
    if ctx.jit_rt.exit_jit.is_null() {
        return false;
    }
    // SAFETY: `exit_jit` is the body of the activation whose window is
    // `regs`, alive (its caller holds it) while that activation runs.
    #[allow(unsafe_code)]
    let jit = unsafe { &*ctx.jit_rt.exit_jit };
    if !exit_stress() && !jit.wants_recompile(header) {
        return false;
    }
    // (The function's own pc: an inlining body numbers its ops otherwise.)
    ctx.jit_rt.resume = Some(jit.orig_pc(header));
    true
}

/// Whether the generic tier inlines calls (`KATAAN_JIT_INLINE=0` turns it
/// off; read once per process).
fn inline_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_INLINE").is_none_or(|v| v != "0"))
}

/// The callee function `caller`'s call site at pc `pc` inlines, if any: the
/// one VM function its feedback `hints` names, with that function's warm
/// inline caches (its compiled body's, else its interpreter's). Whether it
/// can be inlined at all is `crate::jit::inline`'s to decide.
pub(super) fn inline_candidate<'a>(
    funcs: &'a [FnProto],
    caller: usize,
    hints: &[crate::ic::CalleeHint],
    pc: usize,
) -> Option<crate::jit::inline::Candidate<'a>> {
    let crate::ic::CalleeHint::One(key) = *hints.get(pc)? else {
        return None;
    };
    let fid = (key & 0xffff_ffff) as usize;
    if fid == caller || !inline_enabled() {
        return None;
    }
    let proto = funcs.get(fid)?;
    if !crate::jit::inline::callee_ok(proto) {
        return None;
    }
    let mut warm = match proto.tier.jit() {
        Some(Some(j)) if j.is_generic() => j.runtime_warm(),
        _ => alloc::vec::Vec::new(),
    };
    let interp = super::tier_warm(proto);
    if warm.len() < interp.len() {
        warm.resize(interp.len(), None);
    }
    for (w, i) in warm.iter_mut().zip(interp) {
        if w.is_none() {
            *w = i;
        }
    }
    Some(crate::jit::inline::Candidate { key, proto, warm })
}

/// Whether `KATAAN_JIT_LOG` is set: report, per function, whether the JIT
/// compiled it and why not (to stderr; `=2` also dumps the bytecode). Read once
/// per process.
pub(super) fn jit_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_LOG").is_some_and(|v| v != "0"))
}

/// One `KATAAN_JIT_LOG` line for function `id`.
pub(super) fn jit_log(id: usize, proto: &FnProto, what: &str) {
    let name = if proto.name.is_empty() {
        "<anonymous>"
    } else {
        proto.name.as_str()
    };
    std::eprintln!(
        "[jit] fn #{id} {name} ({} ops, {} regs, {} params): {what}",
        proto.ops.len(),
        proto.n_regs,
        proto.n_params
    );
    if std::env::var_os("KATAAN_JIT_LOG").is_some_and(|v| v == "2") {
        for (pc, op) in proto.ops.iter().enumerate() {
            std::eprintln!("[jit]   {pc:4} {op:?}");
        }
    }
}

/// Test entry: runs compiled `jit` as a plain call of `args` (no captures, `this`
/// undefined), laying the register file out as `call_with_inner` does.
#[cfg(test)]
pub(super) fn call_generic(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let mut regs = alloc::vec![NanBox::undefined(); jit.n_regs()];
    for (slot, a) in regs.iter_mut().zip(args.iter().take(jit.n_params())) {
        *slot = *a;
    }
    match run_compiled(ctx, funcs, jit, &mut regs, 0, &[])? {
        Ok(FrameExit::Return(v)) => Some(Ok(v.unwrap_or(NanBox::undefined()))),
        Ok(_) => Some(Err(VmError::Unsupported)),
        Err(e) => Some(Err(e)),
    }
}

/// `Op::SetProp`'s slow path in compiled code (`crate::jit::JitSetSiteHelper`):
/// a write the site's add cache recorded is made directly (see
/// `crate::jit::MethodSite::try_add`); anything else is the interpreter's
/// write ([`jit_set_prop_impl`](super::jit_set_prop_impl)), after which an add
/// it made is recorded.
///
/// # Safety
/// `ctx` the live `Ctx`; `site` the calling `JitProto`'s site for this write.
pub(crate) extern "C" fn jit_helper_set_prop_site(
    ctx_raw: *mut core::ffi::c_void,
    obj: u64,
    site: *mut crate::jit::MethodSite,
    val: u64,
    strict: u64,
) -> u64 {
    // SAFETY: see the function contract; the site is the calling `JitProto`'s
    // (interior-mutable, touched only by this helper while it runs).
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe { (&mut *(ctx_raw as *mut Ctx), &mut *site) };
    if site.try_add(&mut ctx.realm, obj, NanBox::from_bits(val)) {
        return NanBox::undefined().to_bits();
    }
    let recv = NanBox::from_bits(obj).as_handle().map(Handle::from_raw);
    let before = recv.and_then(|h| {
        ctx.realm
            .jit_object(h)
            .and_then(|o| o.shaped_shape().cloned())
    });
    let key = site.key();
    let r = super::jit_set_prop_impl(
        ctx_raw,
        obj,
        key.as_ptr(),
        key.len(),
        site.cache,
        val,
        strict != 0,
    );
    if let (Some(h), Some(before)) = (recv, before)
        && r != NanBox::jit_throw_bits()
    {
        // SAFETY: as above; the write's own borrow of the context has ended.
        #[allow(unsafe_code)]
        let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
        site.fill_add(&mut ctx.realm, h, &before);
    }
    r
}

/// `Op::GetProp`'s slow path in compiled code (`crate::jit::JitGetSiteHelper`):
/// the interpreter's read ([`jit_helper_get_prop`](super::jit_helper_get_prop)),
/// after which the site's chain cache is refilled from the receiver and, when
/// the read is an inherited VM getter's, the site's call cache from it.
///
/// # Safety
/// `ctx` the live `Ctx`; `site`/`call` the calling `JitProto`'s caches for
/// this read.
pub(crate) extern "C" fn jit_helper_get_prop_site(
    ctx_raw: *mut core::ffi::c_void,
    obj: u64,
    site: *mut crate::jit::MethodSite,
    call: *const CallSite,
) -> u64 {
    // SAFETY: the calling `JitProto`'s site, touched only by this helper.
    #[allow(unsafe_code)]
    let site = unsafe { &mut *site };
    let key = site.key();
    let r = super::jit_helper_get_prop(ctx_raw, obj, key.as_ptr(), key.len(), site.cache);
    if r != NanBox::jit_throw_bits()
        && let Some(h) = NanBox::from_bits(obj).as_handle().map(Handle::from_raw)
    {
        // SAFETY: the read's own borrow of the context has ended; the call
        // cache is the site's (interior-mutable).
        #[allow(unsafe_code)]
        let (ctx, call) = unsafe { (&mut *(ctx_raw as *mut Ctx), &mut *call.cast_mut()) };
        if let Some(getter) = site.fill_get(&mut ctx.realm, h) {
            fill_call_site(ctx, getter, call);
        }
    }
    r
}
