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

mod deopt;
pub(crate) use deopt::{jit_helper_deopt, jit_helper_get_virtual};

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
    /// The bodies a recompilation replaced in `direct`, kept for the
    /// context's life as `direct` keeps its own: compiled call sites filled
    /// from `direct` hold their raw entry addresses. The function need not
    /// keep a replaced body itself: `direct` is by function id, and a table
    /// eval code extended holds a copy of each function with a tier slot of
    /// its own, so the body can be an older copy's (whose table may be gone),
    /// owned by nothing else.
    direct_retired: alloc::vec::Vec<Rc<crate::jit::JitProto>>,
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
    /// Set by a compiled integer version's overflow bail (or entry
    /// failure) that asked for a register's demotion: the next back-edge
    /// poll calls [`jit_helper_backedge`], which has the polling body
    /// recompiled ([`demote_poll`]).
    demote: bool,
    /// A sync generator's compiled body that just suspended
    /// ([`jit_helper_suspend`]): the pc its resumption continues at (the
    /// function's own), the value it yields and the register a `next(v)`
    /// lands `v` in.
    suspended: Option<(usize, NanBox, Reg)>,
    /// An activation that just left its compiled body at a deoptimization
    /// exit (`crate::jit::deopt`, [`jit_helper_deopt`]): the function's own
    /// pc the interpreter continues it at and its handler stack there.
    deopt: Option<(usize, alloc::vec::Vec<(usize, Reg)>)>,
    /// A proper tail call the activation that just left its compiled body
    /// handed to the trampoline ([`jit_helper_tail_call`]).
    tail: Option<TailCall>,
}

/// A proper tail call handed back to `call_with_inner`'s trampoline: what
/// `FrameExit::Tail` carries.
pub(super) struct TailCall {
    id: usize,
    args: alloc::vec::Vec<NanBox>,
    captures: alloc::vec::Vec<NanBox>,
    this: NanBox,
}

impl TailCall {
    fn into_exit(self) -> FrameExit {
        FrameExit::Tail {
            id: self.id,
            args: self.args,
            captures: self.captures,
            this: self.this,
        }
    }
}

impl Default for TierRuntime {
    fn default() -> Self {
        Self {
            handlers: alloc::vec::Vec::new(),
            base: 0,
            direct: alloc::vec::Vec::new(),
            direct_retired: alloc::vec::Vec::new(),
            stack: StackPtr::default(),
            hosted: false,
            main_realm: true,
            exit_regs: 0,
            exit_jit: core::ptr::null(),
            resume: None,
            left: None,
            poll_interval: if exit_stress() { 1 } else { POLL_INTERVAL },
            demote: false,
            suspended: None,
            deopt: None,
            tail: None,
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

    /// Readies the state of a pooled hosted context (`hosted_pool`) whose
    /// call has returned for its next one, as if it were [`Self::hosted`]
    /// (keeping only the thread's register stack and the allocations):
    /// `false` when anything is in a state a finished call does not leave,
    /// and the context should be dropped instead. The compiled-callee memo
    /// goes, with the bodies it kept alive, as when the context is dropped.
    pub(super) fn reset_hosted(&mut self) -> bool {
        if self.base != 0
            || self.exit_regs != 0
            || !self.exit_jit.is_null()
            || self.resume.is_some()
            || self.left.is_some()
            || self.demote
            || self.suspended.is_some()
            || self.deopt.is_some()
            || self.tail.is_some()
            || !self.hosted
        {
            return false;
        }
        self.handlers.clear();
        if !self.direct.is_empty() {
            self.direct.clear();
        }
        if !self.direct_retired.is_empty() {
            self.direct_retired.clear();
        }
        true
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

/// The granule (in words) by which [`reserve`] raises a register stack's
/// `limit`, and the most of it a collection leaves above `top` (see
/// [`scrub_stack`]): small enough that zeroing it at a collection is cheap,
/// large enough that a call chain rarely outgrows it between collections.
const STACK_CHUNK: usize = 1 << 11;

/// Makes room for `words` more words at the top of `st` if the stack has
/// them: raises its `limit` (see [`JitStack`]) when `top + words` lies past
/// it. Returns the room under `limit` afterwards.
///
/// # Safety
/// `st` is this thread's live register stack, which nothing else borrows.
#[allow(unsafe_code)]
unsafe fn reserve(st: *mut JitStack, words: usize) -> usize {
    // SAFETY: see the function contract; every pointer is inside (or one past)
    // the allocation.
    unsafe {
        let top = (*st).top;
        let room = (*st).limit.offset_from(top) as usize;
        if room >= words {
            return room;
        }
        let left = (*st).end.offset_from(top) as usize;
        let want = words.saturating_add(STACK_CHUNK).min(left);
        if want > room {
            // `top + room..top + want` held no handle (it was at or past
            // `limit`), so admitting it keeps the invariant.
            (*st).limit = top.add(want);
        }
        (*st).limit.offset_from(top) as usize
    }
}

/// Before a collection: zeroes what finished activations left in the
/// register stack above `top`, so no word of it names an object this
/// collection frees, and pulls `limit` down to at most a chunk past `top`
/// (see [`JitStack`]).
pub(super) fn scrub_stack(ctx: &Ctx) {
    let p = ctx.jit_rt.stack.0;
    if p.is_null() {
        return;
    }
    // SAFETY: `p` is this thread's live stack; no compiled code runs while a
    // safepoint does, and `top..limit` belongs to no activation.
    #[allow(unsafe_code)]
    unsafe {
        let top = (*p).top;
        let limit = (*p).limit;
        let dirty = limit.offset_from(top);
        if dirty > 0 {
            core::ptr::write_bytes(top, 0, dirty as usize);
        }
        let keep = ((*p).end.offset_from(top) as usize).min(STACK_CHUNK);
        if dirty > keep as isize {
            (*p).limit = top.add(keep);
        }
    }
}

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
            limit: unsafe { mem.add(STACK_CHUNK.min(STACK_WORDS)) },
            frames: core::ptr::null_mut(),
            base: mem,
            // SAFETY: one past the end of the allocation.
            #[allow(unsafe_code)]
            end: unsafe { mem.add(STACK_WORDS) },
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
    // A body that inlined calls has a larger window than the function's own
    // registers (its inlined bodies' sub-window, which starts `undefined`).
    let n = regs.len().max(jit.n_regs());
    // SAFETY: `st` is this thread's live stack; nothing else borrows it.
    #[allow(unsafe_code)]
    let (top, room) = unsafe { ((*st).top, reserve(st, n)) };
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
    // A proper tail call handed to the trampoline: the activation is over.
    if let Some(t) = ctx.jit_rt.tail.take() {
        return Ok(t.into_exit());
    }
    // A sync generator's body suspended (the register file is up to date):
    // the interpreter's own suspension outcome.
    if let Some((pc, value, dst)) = ctx.jit_rt.suspended.take() {
        return Ok(FrameExit::Yield {
            value,
            pc,
            handlers,
            dst,
        });
    }
    // A deoptimization exit: the interpreter continues the frame (see
    // `run_tiering`; the exit took the handler stack).
    if ctx.jit_rt.deopt.is_some() {
        return Ok(FrameExit::Return(None));
    }
    match ctx.jit_rt.resume.take() {
        Some(pc) => {
            ctx.jit_rt.left = Some((pc, handlers));
            Ok(FrameExit::Return(None))
        }
        None => Err(VmError::Unsupported),
    }
}

/// The frame an on-stack replacement ([`osr`]) entered, which left its
/// compiled body at a deoptimization exit: the pc its interpreter loop
/// continues at and the handler stack there.
pub(super) fn take_deopt(ctx: &mut Ctx) -> Option<(usize, alloc::vec::Vec<(usize, Reg)>)> {
    ctx.jit_rt.deopt.take()
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
/// "not compiled — interpret it"; the outcome is a return or a proper tail
/// call (`FrameExit::Tail`, which the caller's trampoline makes).
pub(super) fn enter_on_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
    regs: &mut [NanBox],
    ic: &[crate::ic::PropertyCache],
) -> Option<Result<FrameExit, VmError>> {
    let proto = funcs.get(id)?;
    if proto.is_async || proto.is_generator {
        return None;
    }
    if proto.tier.jit().is_none() {
        super::global_mirror::prefill(ctx, proto);
    }
    let mut jit = jit_for(funcs, id, ic)?;
    if jit.is_generic() {
        // A call site it could not inline has settled since: recompile
        // first (`JitProto::wants_recompile_on_entry`).
        if jit.wants_recompile_on_entry(u32::try_from(id).unwrap_or(u32::MAX))
            && let Some(new) = recompile(ctx, funcs, id, &jit, ic)
        {
            jit = new;
        }
        return match run_tiering(ctx, funcs, id, jit, regs, 0, &[], ic, false, false)? {
            Ok(FrameExit::Yield { .. }) => Some(Err(VmError::Unsupported)),
            // A return, or a proper tail call the caller's trampoline makes.
            r => Some(r),
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
    jit.call_guarded(args.get(..proto.n_params)?)
        .map(|v| Ok(FrameExit::Return(Some(v))))
}

/// [`enter_on_call`] for an async function (hosted): its body compiled (on
/// first use) and run from the start on the activation's register file, to
/// its first `await` (a suspension, as its resumption runs: see
/// [`resume_generator`]) or its end. `None` — not compiled — interprets it.
pub(super) fn enter_async_on_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    regs: &mut [NanBox],
    ic: &[crate::ic::PropertyCache],
) -> Option<Result<FrameExit, VmError>> {
    let proto = funcs.get(id)?;
    if !proto.is_async || proto.is_generator {
        return None;
    }
    if proto.tier.jit().is_none() {
        super::global_mirror::prefill(ctx, proto);
    }
    let jit = jit_for(funcs, id, ic)?;
    if !jit.is_generic() {
        return None;
    }
    run_tiering(ctx, funcs, id, jit, regs, 0, &[], ic, false, false)
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
    if proto.ops.len() != program.len() {
        return None;
    }
    if proto.tier.jit().is_none() {
        super::global_mirror::prefill(ctx, proto);
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
    run_tiering(ctx, funcs, id, jit, regs, target, handlers, ic, true, true)
}

/// Resumes the suspended sync generator of function `id` in native code: its
/// register file `regs` (with the resumption's value and kind already bound),
/// from `pc` with its handler stack `handlers` — exactly what the
/// interpreter's run of the frame from there does. `ic` is the body's
/// inline caches; `resumes` the function's resumptions so far. `None`
/// (nothing ran) when the body has no compiled code entered there: interpret
/// it.
#[allow(clippy::too_many_arguments)]
pub(super) fn resume_generator(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    regs: &mut [NanBox],
    pc: usize,
    handlers: &[(usize, Reg)],
    ic: &[crate::ic::PropertyCache],
    resumes: u32,
) -> Option<Result<FrameExit, VmError>> {
    let proto = funcs.get(id)?;
    if proto.tier.jit().is_none() {
        super::global_mirror::prefill(ctx, proto);
    }
    let jit = jit_for(funcs, id, ic)?;
    if !jit.is_generic() || !jit.has_entry(pc) {
        return None;
    }
    // A resumption typically runs a loop's single iteration, whose back-edge
    // poll would be the slow one of an activation that may leave to be
    // recompiled: only one resumption in `RECOMPILE_EVERY` may (each one
    // under `KATAAN_JIT_EXIT_STRESS`), which still recompiles a body whose
    // sites learned more.
    if resumes.is_multiple_of(RECOMPILE_EVERY) || exit_stress() {
        run_tiering(ctx, funcs, id, jit, regs, pc, handlers, ic, true, false)
    } else {
        run_compiled_at(ctx, funcs, &jit, regs, pc, handlers, false, true)
    }
}

/// Whether the `n`th resumption of a generator function's generators is
/// one that may leave its compiled body to be recompiled (see
/// [`resume_generator`]).
pub(super) fn resume_may_recompile(n: u32) -> bool {
    n.is_multiple_of(RECOMPILE_EVERY) || exit_stress()
}

/// The outcome of [`run_generator_body`].
pub(super) enum GenRun {
    /// The body suspended: it yields `value`, and its resumption continues
    /// at `pc` with the sent value in `dst` (its handlers are back in the
    /// caller's vector).
    Yield { value: NanBox, pc: usize, dst: Reg },
    /// Any other end, as the interpreter's frame outcome.
    Exit(Result<FrameExit, VmError>),
    /// The body refused to run (nothing ran).
    Refused,
}

/// A hot generator's resumption that may not leave its compiled body (see
/// [`resume_generator`]): `jit` (a generic body) run from `pc` on the
/// generator's register file and its handler stack `handlers` — at a
/// suspension the body's handlers are moved back into `handlers` (no new
/// vector), and what the body yields is returned without making a frame
/// outcome. Exactly [`run_compiled_at`]'s run otherwise.
pub(super) fn run_generator_body(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
    pc: usize,
    handlers: &mut alloc::vec::Vec<(usize, Reg)>,
) -> GenRun {
    let saved = ctx.jit_funcs.replace(funcs as *const [FnProto]);
    let outer_base = ctx.jit_rt.base;
    let base = ctx.jit_rt.handlers.len();
    ctx.jit_rt.base = base;
    if !handlers.is_empty() {
        ctx.jit_rt.handlers.extend_from_slice(handlers);
    }
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    let st = stack_of(ctx);
    let n = regs.len().max(jit.n_regs());
    // SAFETY: as `run_compiled_at`.
    #[allow(unsafe_code)]
    let (top, room) = unsafe { ((*st).top, reserve(st, n)) };
    let raw = if n <= room {
        // SAFETY: as `run_compiled_at`: `top..top + n` is free stack memory,
        // claimed for exactly this activation.
        #[allow(unsafe_code)]
        let window = unsafe {
            core::ptr::copy_nonoverlapping(regs.as_ptr().cast::<u64>(), top, regs.len());
            for i in regs.len()..n {
                top.add(i).write(NanBox::undefined().to_bits());
            }
            (*st).top = top.add(n);
            core::slice::from_raw_parts_mut(top.cast::<NanBox>(), n)
        };
        let raw = jit.call_frame_raw(ctx_ptr, window, pc);
        if raw.is_some() {
            regs.copy_from_slice(&window[..regs.len()]);
        }
        // SAFETY: as above; the activation is over.
        #[allow(unsafe_code)]
        unsafe {
            (*st).top = top;
        }
        raw
    } else {
        jit.call_frame_raw(ctx_ptr, regs, pc)
    };
    let Some(raw) = raw else {
        ctx.jit_rt.handlers.truncate(base);
        ctx.jit_rt.base = outer_base;
        ctx.jit_funcs = saved;
        return GenRun::Refused;
    };
    if raw == NanBox::jit_exit_bits() {
        if let Some((pc, value, dst)) = ctx.jit_rt.suspended.take() {
            handlers.clear();
            handlers.extend(ctx.jit_rt.handlers.drain(base..));
            ctx.jit_rt.base = outer_base;
            ctx.jit_funcs = saved;
            return GenRun::Yield { value, pc, dst };
        }
        return GenRun::Exit(left_at(ctx, base, outer_base, saved));
    }
    ctx.jit_rt.handlers.truncate(base);
    ctx.jit_rt.base = outer_base;
    ctx.jit_funcs = saved;
    GenRun::Exit(finish_raw(ctx, raw))
}

/// [`resume_generator`] for a frame that owns its handler stack: a
/// resumption that may not leave the body runs [`run_generator_body`], whose
/// suspension moves the body's handlers back into `handlers` (handed out in
/// the outcome) instead of making a new vector.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn resume_frame(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    regs: &mut [NanBox],
    pc: usize,
    handlers: &mut alloc::vec::Vec<(usize, Reg)>,
    ic: &[crate::ic::PropertyCache],
    resumes: u32,
) -> Option<Result<FrameExit, VmError>> {
    let jit = match funcs.get(id)?.tier.jit() {
        Some(Some(j)) if j.is_generic() && !resume_may_recompile(resumes) => j,
        _ => return resume_generator(ctx, funcs, id, regs, pc, handlers, ic, resumes),
    };
    match run_generator_body(ctx, funcs, &jit, regs, pc, handlers) {
        GenRun::Yield { value, pc, dst } => Some(Ok(FrameExit::Yield {
            value,
            pc,
            handlers: core::mem::take(handlers),
            dst,
        })),
        GenRun::Exit(r) => Some(r),
        GenRun::Refused => None,
    }
}

/// How often a generator's resumption may leave its compiled body to be
/// recompiled (see [`resume_generator`]).
const RECOMPILE_EVERY: u32 = 64;

/// Whether a resumption of a generator whose function has been resumed
/// `resumes` times should run compiled code.
pub(super) fn resume_is_hot(resumes: u32) -> bool {
    match mode() {
        Mode::Off => false,
        Mode::Normal => resumes >= RESUME_THRESHOLD,
        Mode::Eager => true,
    }
}

/// Resumptions of a generator function's generators before they run in
/// native code.
const RESUME_THRESHOLD: u32 = 32;

/// A sync generator's suspension op in compiled code
/// (`crate::jit::JitSuspendHelper`): `Op::YieldResume` and
/// `Op::GeneratorStart` suspend (recording what is yielded and where the
/// resumption continues, `pc + 1`), and an `Op::YieldDelegate` step runs as
/// its interpreter arm does — a not-done inner result suspends at the op
/// itself (it runs again on resumption), a done one falls through.
///
/// # Safety
/// As [`jit_helper_exec_op`]; `pc` is the op's pc in the function's own
/// numbering.
pub(crate) extern "C" fn jit_helper_suspend(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    op: *const Op,
    pc: usize,
) -> u64 {
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: the activation's register file, live and unaliased here.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs) };
    // SAFETY: the calling `JitProto`'s owned bytecode.
    #[allow(unsafe_code)]
    let op = unsafe { &*op };
    match op {
        // (An async function's `await` suspends as a `yield` does: its
        // resumption binds the awaited value to `dst`, or throws there.)
        Op::YieldResume { dst, src, .. } | Op::Await { dst, src } => {
            ctx.jit_rt.suspended = Some((pc + 1, regs[*src as usize], *dst));
            NanBox::jit_exit_bits()
        }
        Op::GeneratorStart => {
            ctx.jit_rt.suspended = Some((pc + 1, NanBox::undefined(), 0));
            NanBox::jit_exit_bits()
        }
        Op::YieldDelegate {
            dst,
            iter,
            next,
            mode,
        } => {
            // SAFETY: `jit_funcs` outlives this call.
            #[allow(unsafe_code)]
            let funcs = unsafe {
                &*ctx
                    .jit_funcs
                    .expect("jit_funcs set before generic dispatch")
            };
            match super::yield_delegate_op(ctx, funcs, regs, *dst, *iter, *next, *mode) {
                Ok(Some(res)) => {
                    ctx.jit_rt.suspended = Some((pc, res, *dst));
                    NanBox::jit_exit_bits()
                }
                Ok(None) => 0,
                Err(e) => {
                    ctx.jit_pending_fault = Some(e);
                    NanBox::jit_throw_bits()
                }
            }
        }
        _ => {
            ctx.jit_pending_fault = Some(VmError::Unsupported);
            NanBox::jit_throw_bits()
        }
    }
}

/// Runs function `id`'s compiled body `jit` on `regs` from `entry` with
/// `handlers` (as [`run_compiled`]) as an activation that may leave at a loop
/// header to be recompiled, continuing there in the new code; `ic` is the
/// activation's interpreter caches, if it holds them. `keep`: as
/// [`run_compiled_at`]'s.
///
/// An activation that leaves at a deoptimization exit (`crate::jit::deopt`)
/// continues in the interpreter: here, or — `own_loop`, an on-stack
/// replacement — in the interpreter loop that entered it, which
/// [`take_deopt`] tells where.
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
    own_loop: bool,
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
    if !own_loop && let Some((pc, h)) = ctx.jit_rt.deopt.take() {
        // (`regs` holds the frame: the exit's sentinel had it copied back.)
        let proto = funcs.get(id)?;
        return Some(super::run_frame_at(
            ctx, funcs, &proto.ops, regs, pc, h, None,
        ));
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
    super::global_mirror::prefill(ctx, proto);
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
    // (And no speculative version for a loop whose fast version kept
    // leaving it.)
    let cold = cold_call_pcs(proto, &hints);
    let mut new = crate::jit::with_entry_poll(!cold.is_empty(), || {
        crate::jit::JitProto::compile_generic_respec(
            proto,
            &super::jit_generic_helpers(),
            &warm,
            &|f| {
                funcs
                    .get(f as usize)
                    .is_some_and(|p| !(p.is_async && !p.is_generator))
            },
            &|pc, _| inline_candidate(funcs, id, &hints, pc),
            &old.spec_off(),
            &old.int_demote(),
        )
    })?;
    new.note_cold_calls(&cold);
    new.note_function(id);
    new.set_generation(old.generation() + 1);
    let new = Rc::new(new);
    if jit_log_enabled() {
        jit_log(
            id,
            proto,
            &alloc::format!(
                "recompiled (generation {}, {} bytes, {} calls inlined: {:?})",
                new.generation(),
                new.code_len(),
                new.inlined(),
                new.inlined_sites()
            ),
        );
    }
    // The old code stays alive with the function: compiled callers may hold
    // its address.
    proto.tier.replace_jit(Rc::clone(&new));
    if let Some(d) = ctx.jit_rt.direct.get_mut(id)
        && let Some(prev) = d.as_mut()
    {
        // (Compiled call sites may still enter the body it replaces: see
        // `direct_retired`.)
        let prev = core::mem::replace(prev, Rc::clone(&new));
        ctx.jit_rt.direct_retired.push(prev);
    }
    retarget(ctx, funcs, id, old, &new);
    Some(new)
}

/// Points the compiled call sites that enter function `id`'s body `old`
/// at its recompilation `new` (`JitProto::retarget_calls`), when the call
/// sequence may enter `new` (as [`fill_call_site`] tests): every body of
/// the function table's functions, and the bodies [`direct_jit`] handed
/// out.
fn retarget(
    ctx: &Ctx,
    funcs: &[FnProto],
    id: usize,
    old: &crate::jit::JitProto,
    new: &crate::jit::JitProto,
) {
    let (Some(proto), Ok(fid)) = (funcs.get(id), u32::try_from(id)) else {
        return;
    };
    if new.native_entry().is_none() || new.n_regs() < proto.n_regs {
        return;
    }
    for f in funcs {
        for b in f.tier.bodies() {
            b.retarget_calls(fid, old, new);
        }
    }
    for b in ctx.jit_rt.direct.iter().flatten() {
        b.retarget_calls(fid, old, new);
    }
    for b in &ctx.jit_rt.direct_retired {
        b.retarget_calls(fid, old, new);
    }
}

/// The native-call entry's recompilation poll (`GenericHelpers::entry_poll`,
/// `crate::jit::tierup`'s `ENTRY_POLL`): its countdown, the word at `word`,
/// ran out. The function whose code polls is recompiled now if it asks
/// for it (`JitProto::entry_poll`, which sets the next countdown); the
/// activation that polled continues in the old code. Code that is no
/// longer any function's stops polling. Runs no JavaScript, allocates
/// nothing on the heap, never throws: returns `0`.
///
/// # Safety
/// `ctx` the live `Ctx`; `word` the polling body's countdown word.
pub(crate) extern "C" fn jit_helper_entry_poll(ctx: *mut core::ffi::c_void, word: u64) -> u64 {
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let found = funcs
        .iter()
        .enumerate()
        .find_map(|(id, p)| match p.tier.jit() {
            Some(Some(j)) if j.entry_poll_addr() == word => Some((id, j)),
            _ => None,
        });
    let Some((id, jit)) = found else {
        // SAFETY: the polling body's own countdown word, which only its
        // code (suspended in this call) writes.
        #[allow(unsafe_code)]
        unsafe {
            *(word as *mut u64) = u64::MAX;
        }
        return 0;
    };
    let yes = jit.entry_poll(u32::try_from(id).unwrap_or(u32::MAX));
    if jit_log_enabled() {
        jit_log(
            id,
            &funcs[id],
            &alloc::format!("entry poll: {}", if yes { "recompile" } else { "not yet" }),
        );
    }
    if yes {
        let _ = recompile(ctx, funcs, id, &jit, &[]);
    }
    0
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
/// key changes the global binding epoch tracks) — or, for an op whose compiled
/// code mirrors the write (`super::global_mirror`), when it would mirror it
/// onto a writable data property it can store into directly, whose address it
/// leaves in the op's mirror cache. `0` otherwise. The address
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
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode, alive while its
    // code runs.
    #[allow(unsafe_code)]
    let op = unsafe { &*(op as usize as *const Op) };
    let p = match op {
        Op::LoadGlobal { name, cache, .. } => ctx
            .host
            .as_deref()
            .and_then(|host| host.global_scope().jit_binding_ptr(name, cache, false))
            .or_else(|| super::global_obj::jit_load_ref(ctx, name, cache)),
        Op::StoreGlobal {
            name,
            strict,
            resolved,
            cache,
            ..
        } if !(*strict && resolved.is_some()) && !name.starts_with('\u{1}') => {
            global_store_ref(ctx, name, cache)
        }
        _ => None,
    };
    p.map_or(0, |p| p as usize as u64)
}

/// [`jit_helper_global_ref`] of a `StoreGlobal` of `name`.
fn global_store_ref(
    ctx: &mut Ctx,
    name: &str,
    cache: &super::StoreGlobalCache,
) -> Option<*mut NanBox> {
    // The address this op validated last, under the same global binding
    // epoch (no binding added, freed or moved, the global scope not
    // swapped), while the running scope is still the global scope and the
    // global object it was validated against still has no like-named own
    // property (the same object at the same key epoch): what the steps
    // below would find again. (A body's frame cache holds a write address
    // only for one activation, so a small function writing a global counter
    // validated it on every call.)
    let epoch = crate::env::global_binding_epoch();
    let (memo_epoch, memo) = cache.wmemo.get();
    if memo != 0
        && memo_epoch == epoch
        && cache.owned.get() == 1
        && let Some((_, g)) = ctx.host.as_deref()?.global_scope_if_current()
        && g.as_handle() == Some(cache.global.get())
        && ctx
            .realm
            .jit_object(Handle::from_raw(cache.global.get()))
            .is_some_and(|o| o.key_epoch() == cache.epoch.get())
    {
        cache.jit.set(0);
        return Some(memo as usize as *mut NanBox);
    }
    cache.wmemo.set((0, 0));
    let Some((p, g)) =
        ctx.host
            .as_deref()?
            .global_scope_if_current()
            .and_then(|(scope, global)| {
                let p = scope.jit_binding_ptr(name, &cache.slot, true)?;
                Some((p, Handle::from_raw(global.as_handle()?)))
            })
    else {
        // No plain global-scope binding: an existing writable data property
        // of the global object, written as a binding-less mirror write.
        return super::global_obj::jit_store_ref(ctx, name, cache);
    };
    // Only a global object whose key changes bump the global binding epoch
    // (which the compiled cache holds the answer across calls under).
    let tracked = ctx
        .realm
        .jit_object(g)
        .is_some_and(|o| o.jit_flags() & crate::object::JIT_FLAG_GLOBAL != 0);
    if !tracked {
        return None;
    }
    // A global `let`/`const`/`class` is not the like-named property's binding.
    // (The lexical test, a search of the scope's lexical names, only matters
    // for a name the global object owns: asked second, as most such writes
    // are a script's `let` counter's, run once per activation.)
    if !super::global_owns_cached(&ctx.realm, g, name, cache) {
        cache.jit.set(0);
        // (Memoized only once the owned-property answer is cached against
        // this global object's key epoch, which the memo's check reads.)
        if cache.owned.get() == 1 && cache.global.get() == g.to_raw() {
            cache.wmemo.set((epoch, p as usize as u64));
        }
        return Some(p);
    }
    if ctx
        .host
        .as_deref()
        .and_then(|h| h.global_scope_if_current())
        .is_some_and(|(scope, _)| scope.has_lexical(name))
    {
        cache.jit.set(0);
        return Some(p);
    }
    // A global `var`: the write also goes to the global object's property,
    // when the op's code mirrors it (`super::global_mirror`).
    if !cache.jit.on() {
        return None;
    }
    let m = ctx.realm.jit_data_ptr(g, name)?;
    cache.jit.set(m as usize as u64);
    Some(p)
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

/// Records an inline `Math.f(x)` site's proof that `recv`'s method is the
/// built-in `native` (`crate::jit::MethodSite::fill_math`, a pure read of
/// the receiver's chain that watches its links), which later activations
/// then take with no walk. Returns 0.
///
/// # Safety
/// As [`jit_helper_site_fill`].
pub(crate) extern "C" fn jit_helper_math_memo(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    site: u64,
    native: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe {
        (
            &mut *(ctx as *mut Ctx),
            &mut *(site as usize as *mut crate::jit::MethodSite),
        )
    };
    if ctx.host.is_some()
        && let Some(h) = NanBox::from_bits(recv).as_handle()
        && let Ok(native) = u16::try_from(native)
    {
        site.fill_math(&mut ctx.realm, Handle::from_raw(h), native);
    }
    0
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

/// A direct built-in call site's miss (`crate::jit::direct_call`): refills
/// the site's memo of the receiver `recv`'s own method and proves it the
/// built-in of kind `kind` (`MethodSite::fill_direct`). Returns `0`; runs no
/// code and allocates nothing.
///
/// # Safety
/// As [`jit_helper_site_fill`].
pub(crate) extern "C" fn jit_helper_direct_fill(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    site: u64,
    kind: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
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
        let h = Handle::from_raw(h);
        if kind as u8 == crate::jit::direct_call::K_PROTO_HAS_OWN {
            site.fill_direct_chain(&mut ctx.realm, h);
        } else {
            site.fill_direct(&mut ctx.realm, h, |realm, v| {
                super::native_call::direct_proves(realm, v, kind as u8)
            });
        }
    }
    0
}

/// A direct built-in call (`crate::jit::direct_call`) of kind `kind` on the
/// argument values `a0`, `a1`, at a site that proved its method that
/// built-in: the result's bits, or the throw sentinel (which it never
/// returns otherwise) when the general call must make it — having done
/// nothing. Runs no code, allocates nothing and never throws.
///
/// # Safety
/// As [`jit_helper_site_fill`].
pub(crate) extern "C" fn jit_helper_direct_call(
    ctx: *mut core::ffi::c_void,
    kind: u64,
    a0: u64,
    a1: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    super::native_call::direct_run(
        &mut ctx.realm,
        kind as u8,
        NanBox::from_bits(a0),
        NanBox::from_bits(a1),
    )
    .map_or(NanBox::jit_throw_bits(), NanBox::to_bits)
}

/// [`jit_helper_direct_call`] for a direct `Object.hasOwn(o, k)` at the
/// method site `site`, which also notes an answer the site's compiled code
/// can then give itself (`MethodSite::note_own_key`).
///
/// # Safety
/// As [`jit_helper_site_fill`].
pub(crate) extern "C" fn jit_helper_direct_has_own(
    ctx: *mut core::ffi::c_void,
    o: u64,
    k: u64,
    site: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe {
        (
            &mut *(ctx as *mut Ctx),
            &mut *(site as usize as *mut crate::jit::MethodSite),
        )
    };
    let (o, k) = (NanBox::from_bits(o), NanBox::from_bits(k));
    match super::native_call::direct_run(&mut ctx.realm, crate::jit::direct_call::K_HAS_OWN, o, k) {
        Some(r) => {
            site.note_own_key(&ctx.realm, o, k, r);
            r.to_bits()
        }
        None => NanBox::jit_throw_bits(),
    }
}

/// `Op::ToStr` (`lazy` zero) or `Op::ToStrLazy` of `v` when that runs no code:
/// a string is itself, and a primitive is its text (`ToStr`) or itself
/// (`ToStrLazy`, whose `Concat` makes the text). `0`, having done nothing, for
/// any other value (and for a `+0` a lazy op keeps: its op does that too).
pub(crate) extern "C" fn jit_helper_to_str(
    ctx: *mut core::ffi::c_void,
    v: u64,
    lazy: u64,
    _unused: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    let value = NanBox::from_bits(v);
    match value.as_handle() {
        Some(h) if ctx.realm.is_string_handle(Handle::from_raw(h)) => v,
        Some(_) => 0,
        None if lazy != 0 => v,
        None => NanBox::handle(ctx.realm.new_primitive_string(value).to_raw()).to_bits(),
    }
}

/// `Op::SwitchStr` at `op` of the discriminant `v`: the index of the first
/// case whose literal is `v`'s contents, `-1` for none, as a Number's bits
/// (what its interpreter arm writes). Reads the heap only: no allocation, no
/// throw, no JavaScript.
///
/// # Safety
/// `ctx` must be the live `Ctx` the compiled body was entered with and `op` a
/// pointer into the bytecode the calling `JitProto` owns.
pub(crate) extern "C" fn jit_helper_switch_str(
    ctx: *mut core::ffi::c_void,
    v: u64,
    op: u64,
    _unused: u64,
) -> u64 {
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let (ctx, op) = unsafe { (&*(ctx as *const Ctx), &*(op as usize as *const Op)) };
    let i = match op {
        Op::SwitchStr { table, memo, .. } => {
            ctx.realm
                .string_switch_index_memo(NanBox::from_bits(v), table, memo)
        }
        _ => None,
    };
    NanBox::number(i.map_or(-1.0, f64::from)).to_bits()
}

/// Proves (or finds it cannot) what an inline `s.charCodeAt(i)` relies on:
/// see [`MethodSite::set_str_proof`](crate::jit::MethodSite::set_str_proof).
pub(crate) extern "C" fn jit_helper_str_proof(
    ctx: *mut core::ffi::c_void,
    _recv: u64,
    site: u64,
    _unused: u64,
) -> u64 {
    // SAFETY: as `jit_helper_site_fill`.
    #[allow(unsafe_code)]
    let (ctx, site) = unsafe {
        (
            &mut *(ctx as *mut Ctx),
            &mut *(site as usize as *mut crate::jit::MethodSite),
        )
    };
    if site.key() == "push" {
        super::native_call::prove_array_push(&mut ctx.realm, site);
    } else if site.key() == "pop" {
        super::native_call::prove_array_pop(&mut ctx.realm, site);
    } else {
        super::native_call::prove_char_code_at(&mut ctx.realm, site);
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

/// `Op::InitFnPrototype` of the closure `f` compiled code has just made
/// inline (in the main realm of a single-realm run, so with no auxiliary
/// object): what the interpreter arm does, without the one-op dispatch.
///
/// # Safety
/// As [`jit_helper_new_object`].
pub(crate) extern "C" fn jit_helper_owe_fn_proto(ctx: *mut core::ffi::c_void, f: u64) -> u64 {
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    if let Some(fh) = NanBox::from_bits(f).as_handle().map(Handle::from_raw)
        && !ctx.realm.defer_fresh_vm_fn_prototype(fh)
    {
        super::init_fn_prototype(&mut ctx.realm, fh);
    }
    0
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
/// `Op::IterNext` in compiled code (the exec-op ABI): a `for-of` step over
/// a hot VM generator resumes it directly (`gen_fast::jit_for_of_step`);
/// any other step is exactly the interpreter arm's.
///
/// # Safety
/// As [`jit_helper_exec_op`].
pub(crate) extern "C" fn jit_helper_iter_next(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    op: *const Op,
) -> u64 {
    // SAFETY: the calling `JitProto`'s owned bytecode.
    #[allow(unsafe_code)]
    let Op::IterNext {
        dst,
        done,
        iter,
        next,
    } = (unsafe { &*op })
    else {
        return jit_helper_exec_op(ctx, regs, n_regs, op);
    };
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: as `jit_helper_exec_op`.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs) };
    let (dst, done, iter, next) = (*dst, *done, *iter, *next);
    let r = match super::gen_fast::jit_for_of_step(ctx, funcs, regs, done, iter, next) {
        Some(r) => r,
        None => super::vm_iter_step(ctx, funcs, regs, done, iter, next),
    };
    match r {
        Ok(v) => {
            regs[dst as usize] = v.unwrap_or(NanBox::undefined());
            0
        }
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// A compiled string literal's first evaluation (`Op::NewString`, see
/// `crate::realm::LitCells`), with the exec-op ABI plus the body's literal
/// table (`cells`) and the site's word in it (`index`): the literal's String,
/// written to its register and kept in the word, which the code loads from
/// then on. Never throws.
///
/// # Safety
/// As [`jit_helper_exec_op`]; `cells` must be `Rc::as_ptr` of the body's
/// live `LitCells` (the body's `JitProto` holds it while its code runs).
pub(crate) extern "C" fn jit_helper_lit_string(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    cells: *mut core::ffi::c_void,
    index: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: as `jit_helper_exec_op`; `cells` came from `Rc::as_ptr` of a
    // table whose owner is alive, so taking one more strong count for the
    // duration of this call is sound.
    #[allow(unsafe_code)]
    let (ctx, regs, op, cells) = unsafe {
        let p = cells.cast_const().cast::<crate::realm::LitCells>();
        Rc::increment_strong_count(p);
        (
            &mut *(ctx_raw as *mut Ctx),
            core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs),
            &*op_raw,
            Rc::from_raw(p),
        )
    };
    if let Op::NewString { dst, value } = op {
        regs[*dst as usize] = ctx.realm.fill_lit_cell(&cells, index as usize, value);
    }
    0
}

/// A compiled `for-in` loop's `Op::EnumKeys` (`crate::jit::forin`), with the
/// exec-op ABI plus its `ForInSite` (`site`): a memoized key list's array
/// (its interpreter arm's first step), the site then holding the list's key
/// table, else (the site holding none) the interpreter's op.
///
/// # Safety
/// As [`jit_helper_exec_op`]; `site` must be the body's live `ForInSite`.
pub(crate) extern "C" fn jit_helper_enum_keys(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
) -> u64 {
    {
        // SAFETY: as `jit_helper_exec_op`; these borrows end before it runs.
        #[allow(unsafe_code)]
        let (ctx, regs, op, site) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs),
                &*op_raw,
                &*site.cast_const().cast::<crate::realm::ForInSite>(),
            )
        };
        // The table the site holds, when it still is the object's.
        if let Op::EnumKeys { dst, obj } = op
            && let Some(h) = regs[*obj as usize].as_handle()
            && ctx.host.is_some()
            && let Some(keys) = site
                .held()
                .as_deref()
                .and_then(|t| ctx.realm.for_in_table_hit(Handle::from_raw(h), t))
        {
            regs[*dst as usize] = keys;
            return 0;
        }
        if let Op::EnumKeys { dst, obj } = op
            && ctx.host.is_some()
            && let Some(h) = regs[*obj as usize].as_handle()
            && let Some((keys, table)) = ctx.realm.for_in_cached_table(Handle::from_raw(h))
        {
            if let Some(t) = &table
                && t.shape_word.get() == 0
            {
                t.shape_word.set(crate::jit::shape_word(t.shape()));
            }
            site.hold(table);
            regs[*dst as usize] = keys;
            return 0;
        }
        site.hold(None);
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

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
        Op::CallValue {
            dst, callee, args, ..
        } => {
            let callee = regs[*callee as usize];
            match call_compiled(ctx, funcs, regs, callee, args, NanBox::undefined()) {
                Some(r) => Some(r.map(|v| regs[*dst as usize] = v)),
                // Any other VM closure (an interpreted, async or generator
                // function): the interpreter arm's call, without a one-op
                // frame around it.
                None if let Some((id, caps)) = callee.as_handle().and_then(|h| {
                    ctx.realm
                        .vm_function(Handle::from_raw(h))
                        .map(|(f, c)| (f as usize, super::ArgBuf::from_slice(c)))
                }) =>
                {
                    let argv = super::ArgBuf::from_regs(regs, args);
                    ctx.pending_callee = Some(callee);
                    // (A window on the register stack is a GC root already.)
                    let pub_mark = ctx.frame_shadow.len();
                    if !on_register_stack(ctx, regs) {
                        ctx.frame_shadow.extend_from_slice(regs);
                    }
                    ctx.frames_published += 1;
                    let r = super::call_with(ctx, funcs, id, &argv, &caps, NanBox::undefined());
                    ctx.frames_published -= 1;
                    ctx.frame_shadow.truncate(pub_mark);
                    Some(r.map(|v| regs[*dst as usize] = v))
                }
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
        // `f(...xs)` over a plain array (its interpreter arm), entering a
        // compiled callee directly; when the spread is not plain, `ok` says
        // so and nothing else happens.
        Op::CallSpreadPlain {
            dst,
            callee,
            this,
            items,
            src,
            ok,
        } => match super::plain_spread_argv(ctx, regs, items, regs[*src as usize]) {
            None => {
                regs[*ok as usize] = NanBox::boolean(false);
                Some(Ok(()))
            }
            Some(argv) => {
                let (f, t) = (regs[*callee as usize], regs[*this as usize]);
                call_compiled_with(ctx, funcs, Some(regs), f, t, &argv).map(|r| {
                    regs[*ok as usize] = NanBox::boolean(true);
                    r.map(|v| regs[*dst as usize] = v)
                })
            }
        },
        // An inlined spread's elements (its interpreter arm, which cannot
        // fail).
        Op::SpreadInto { ok, src, dsts } => {
            super::spread_into(ctx, regs, *ok, *src, dsts);
            Some(Ok(()))
        }
        // Whether a fresh array destructures without running code (its
        // interpreter arm, which cannot fail).
        Op::FreshArrayIterPlain { dst } => {
            regs[*dst as usize] =
                NanBox::boolean(super::iter_fast::fresh_array_is_plain(&mut ctx.realm));
            Some(Ok(()))
        }
        // A fresh function's own `prototype` (its interpreter arm, which
        // cannot fail).
        Op::InitFnPrototype { f } => {
            if let Some(fh) = regs[*f as usize].as_handle().map(Handle::from_raw) {
                super::init_fn_prototype(&mut ctx.realm, fh);
            }
            Some(Ok(()))
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
        // Object spread / rest of a plain object into an empty one (the
        // interpreter arm's first step); anything else takes the one-op frame.
        Op::ObjectSpread { dst, src } => {
            super::spread_plain_vm(ctx, regs[*dst as usize], regs[*src as usize], &[])
                .then_some(Ok(()))
        }
        Op::ObjectRest { dst, src, exclude } => {
            let t = NanBox::handle(ctx.realm.new_object().to_raw());
            super::spread_plain_vm(ctx, t, regs[*src as usize], exclude).then(|| {
                regs[*dst as usize] = t;
                Ok(())
            })
        }
        // A class field's define: its interpreter arm.
        Op::DefineData {
            obj,
            key,
            src,
            attrs,
        } => Some(super::vm_define_data(
            ctx,
            regs[*obj as usize],
            regs[*key as usize],
            regs[*src as usize],
            *attrs,
        )),
        // `this.#x`, `this.#x = v`, `#x in o`, a private field's add: exactly
        // their interpreter arm.
        Op::PrivateGet { .. }
        | Op::PrivateSet { .. }
        | Op::PrivateIn { .. }
        | Op::PrivateDefine { .. } => Some(super::private_ops::exec(ctx, funcs, op, regs)),
        // `x instanceof F` decided from the heap alone (the interpreter arm's
        // own first step); anything else takes the one-op frame.
        Op::Relational {
            dst,
            instanceof: true,
            a,
            b,
        } => {
            super::instance_of_fast(&mut ctx.realm, regs[*a as usize], regs[*b as usize]).map(|r| {
                regs[*dst as usize] = NanBox::boolean(r);
                Ok(())
            })
        }
        // `key in obj` over an ordinary chain (the interpreter arm's first
        // step, also a `for-in`'s per-key liveness check).
        Op::Relational {
            dst,
            instanceof: false,
            a,
            b,
        } => ctx
            .realm
            .has_property_by_string_noting(regs[*a as usize], regs[*b as usize])
            .map(|r| {
                regs[*dst as usize] = NanBox::boolean(r);
                Ok(())
            }),
        // A `for-in`'s memoized key list (the hosted interpreter arm's first
        // step).
        Op::EnumKeys { dst, obj } if ctx.host.is_some() => regs[*obj as usize]
            .as_handle()
            .and_then(|h| ctx.realm.for_in_cached_array(Handle::from_raw(h)))
            .map(|keys| {
                regs[*dst as usize] = keys;
                Ok(())
            }),
        // `typeof v` (its interpreter arm, which cannot fail).
        Op::TypeOf { dst, a } => {
            let t = ctx.realm.type_of_value(regs[*a as usize]);
            regs[*dst as usize] = NanBox::handle(ctx.realm.type_name_string(t).to_raw());
            Some(Ok(()))
        }
        // `typeof name` for a plain global (the interpreter arm's first step).
        Op::TypeofGlobal { dst, name, cache } => {
            super::typeof_fast::type_of(ctx, name, cache).map(|t| {
                regs[*dst as usize] = NanBox::handle(ctx.realm.type_name_string(t).to_raw());
                Ok(())
            })
        }
        // A sloppy function's `this`: the interpreter arm's fast path (one
        // realm: `undefined`/`null` binds the global `this`; an object is
        // its own ToObject).
        Op::BindThis { this } => {
            let t = regs[*this as usize];
            match ctx.host.as_deref() {
                Some(h) if (t.is_undefined() || t.is_null()) && !ctx.realm.multi_realm => {
                    regs[*this as usize] = h.global_this();
                    Some(Ok(()))
                }
                Some(_) if super::is_object_value(&ctx.realm, t) => Some(Ok(())),
                _ => None,
            }
        }
        Op::StrEqConst {
            dst,
            src,
            lit,
            kind,
        } => Some(
            super::vm_str_eq_const(ctx, funcs, regs[*src as usize], lit, *kind)
                .map(|v| regs[*dst as usize] = v),
        ),
        Op::SwitchStr {
            dst,
            src,
            table,
            memo,
        } => {
            let i = ctx
                .realm
                .string_switch_index_memo(regs[*src as usize], table, memo);
            regs[*dst as usize] = NanBox::number(i.map_or(-1.0, f64::from));
            Some(Ok(()))
        }
        // `${v}` / `String(v)`'s ToString of a primitive (the interpreter
        // arm's non-object cases: a string is itself, anything else its
        // digits or name, a small integer's interned); an object runs code
        // and takes the one-op frame.
        Op::ToStr { dst, src } | Op::ToStrLazy { dst, src } => {
            let v = regs[*src as usize];
            match v.as_handle() {
                Some(h) if ctx.realm.is_string_handle(Handle::from_raw(h)) => {
                    regs[*dst as usize] = v;
                    Some(Ok(()))
                }
                Some(_) => None,
                None if matches!(op, Op::ToStrLazy { .. }) => {
                    regs[*dst as usize] = v;
                    Some(Ok(()))
                }
                None => {
                    regs[*dst as usize] =
                        NanBox::handle(ctx.realm.new_primitive_string(v).to_raw());
                    Some(Ok(()))
                }
            }
        }
        // A template literal's concatenation when every part is a string or a
        // primitive (its interpreter arm's first step, which runs no code);
        // anything else takes the one-op frame.
        Op::Concat { dst, parts } if parts.len() <= 16 => {
            let mut vals = [NanBox::undefined(); 16];
            for (slot, r) in vals.iter_mut().zip(parts.iter()) {
                *slot = regs[*r as usize];
            }
            match ctx.realm.concat_parts(&vals[..parts.len()]) {
                Some(Some(h)) => {
                    regs[*dst as usize] = NanBox::handle(h.to_raw());
                    Some(Ok(()))
                }
                _ => None,
            }
        }
        // A relational comparison of two strings (a sort comparator's
        // `a[0] < b[0]`): the interpreter arm's string case, which runs no
        // code. Anything else takes the one-op frame.
        Op::Compare { dst, op, a, b } => {
            super::string_compare(&mut ctx.realm, regs[*a as usize], regs[*b as usize], *op).map(
                |v| {
                    regs[*dst as usize] = NanBox::boolean(v);
                    Ok(())
                },
            )
        }
        // A string literal (`'k' + i` evaluates its `'k'` every iteration).
        Op::NewString { dst, value } => {
            regs[*dst as usize] = NanBox::handle(ctx.realm.new_literal_string(value).to_raw());
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
        // A generator's open (the interpreter arm's first step).
        Op::IterOpen { iter, next, src } => {
            let v = regs[*src as usize];
            super::gen_fast::open(ctx, funcs, v).map(|n| {
                regs[*iter as usize] = v;
                regs[*next as usize] = n;
                Ok(())
            })
        }
        Op::IterNext {
            dst,
            done,
            iter,
            next,
        } => Some(
            super::vm_iter_step(ctx, funcs, regs, *done, *iter, *next)
                .map(|v| regs[*dst as usize] = v.unwrap_or(NanBox::undefined())),
        ),
        // A `for await` step's `next()` call, and the check of what it
        // resolved to: exactly their interpreter arms.
        Op::AsyncIterStep {
            dst,
            iter,
            next,
            sync,
        } => Some(super::async_iter_step(
            ctx, funcs, regs, *dst, *iter, *next, *sync,
        )),
        Op::CheckObject { src } if super::is_object_value(&ctx.realm, regs[*src as usize]) => {
            Some(Ok(()))
        }
        // `for (const [k, v] of map)`: exactly its interpreter arm.
        Op::IterNextEntry {
            dst,
            fused,
            key,
            val,
            done,
            iter,
            next,
        } => Some(
            if let Some(step) = super::iter_fast::entry_step(ctx, regs, *done, *iter, *next) {
                regs[*fused as usize] = NanBox::boolean(true);
                match step {
                    Some((k, v)) => {
                        regs[*key as usize] = k;
                        regs[*val as usize] = v;
                    }
                    None => regs[*done as usize] = NanBox::boolean(true),
                }
                Ok(())
            } else {
                regs[*fused as usize] = NanBox::boolean(false);
                super::vm_iter_step(ctx, funcs, regs, *done, *iter, *next)
                    .map(|v| regs[*dst as usize] = v.unwrap_or(NanBox::undefined()))
            },
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
            | Op::CallWithMethod { .. }
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

/// `Call(callee, this, args)` of a VM closure from a host built-in (a
/// callback: a `JSON.parse` reviver, a getter the host reads, a
/// comparator…), in the fresh hosted context `ctx` that
/// `super::call_vm_function` made for it (call depth 0, no collection): when
/// the closure is a [`direct_target`] with a native-call entry, its compiled
/// body is entered there ([`enter_native`]), as a compiled caller enters it
/// — none of `call_closure` → `call_with` → `call_with_inner`'s generic
/// activation. `None` (nothing done) for anything else, which takes that
/// ordinary call.
///
/// What the ordinary call keeps is kept: the call-depth limits and the
/// native-stack check (a catchable `RangeError`), the callee's
/// `vm_callee_stack` entry (the native-call header stands in for it), and a
/// throw's propagation (the pending value or fault).
pub(super) fn call_hosted(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    callee: NanBox,
    this: NanBox,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let handle = Handle::from_raw(callee.as_handle()?);
    let (fid, caps) = ctx.realm.vm_function(handle)?;
    let proto = funcs.get(fid as usize)?;
    if ctx.realm.multi_realm || caps.len() < proto.n_captures {
        return None;
    }
    let caps = caps.as_ptr();
    let jit = direct_target(proto)?;
    let n_regs = jit.n_regs();
    if jit.native_entry().is_none() || n_regs < proto.n_regs {
        return None;
    }
    // `call_with`'s recursion guard.
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        || crate::stack::exhausted()
    {
        let e = vm_error(ctx, "RangeError", super::STACK_EXHAUSTED);
        return Some(Err(VmError::Thrown(e)));
    }
    // SAFETY: `jit` is the callee's compiled body with a native-call entry
    // over `n_regs` registers; `caps` points at the closure's captures, which
    // the callee value (held by the host caller) keeps alive, and nothing
    // runs between taking it and the entry copying them.
    #[allow(unsafe_code)]
    unsafe {
        enter_native(
            ctx,
            funcs,
            &jit,
            n_regs,
            callee,
            caps.cast::<u64>(),
            this,
            args,
        )
    }
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
        let room = reserve(st, FRAME_HEADER + argc + n_regs);
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
    let hit = if ctx.realm.is_vm_function(h) {
        // A static getter (`C.kind`): through the closure-chain cache, as
        // `vm_get_prop` reads it.
        ctx.realm.fn_chain_get(h, key, cache)?
    } else {
        ctx.realm.chain_cached_get(h, key, cache)?
    };
    match hit {
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

/// [`compiled_getter`] for a static-key property write: when the site's
/// prototype-chain cache finds an accessor (own or inherited, which takes
/// precedence over everything `vm_set_prop_mode` would do) whose setter is a
/// [`call_compiled`] target, on an ordinary object that is not host-exotic and
/// not the global object, call it directly, `this` the receiver, the value its
/// argument. `None`: do the ordinary write.
pub(crate) fn compiled_setter(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    value: NanBox,
    cache: &crate::ic::PropertyCache,
) -> Option<Result<NanBox, VmError>> {
    if !cache.chain.as_ref().is_some_and(|c| c.accessor) {
        return None;
    }
    let h = recv.as_handle().map(Handle::from_raw)?;
    if ctx.realm.is_host_exotic(h) || ctx.host.as_deref().is_none_or(|x| x.is_global_object(recv)) {
        return None;
    }
    match ctx.realm.chain_cached_get(h, key, cache)? {
        crate::realm::ChainHit::Accessor(_, setter, _)
            if setter
                .as_handle()
                .is_some_and(|s| ctx.realm.is_vm_function(Handle::from_raw(s))) =>
        {
            call_compiled_with(ctx, funcs, None, setter, recv, &[value])
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
                if key == "call"
                    && r.is_ok()
                    && !call_site.is_null()
                    && let Some(method) = super::native_call::fn_call_method(cache)
                {
                    // SAFETY: the op's own `CallSite` (see the function contract).
                    #[allow(unsafe_code)]
                    let site = unsafe { &mut *call_site.cast::<CallSite>() };
                    fill_fncall_memo(ctx, site, recv_val, method);
                }
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

/// `Op::CallMethod` of a `get`/`set`/`has`/`add`/`delete` key in compiled
/// code, tried before the call sequence: when the site last called that
/// `Map.prototype`/`Set.prototype` method on this very receiver and still
/// proves it ([`super::native_call::try_coll`]), the result is stored to the
/// destination register and `0` returned; anything else returns `1`, having
/// done nothing, and the compiled code takes its ordinary call. Never throws.
///
/// # Safety
/// As [`jit_helper_call_method`] (`call_site` is unused).
pub(crate) extern "C" fn jit_helper_coll_call(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    cache: *mut core::ffi::c_void,
    _call_site: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: see the function contract (as `jit_helper_exec_op`).
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    let Op::CallMethod {
        dst, recv, args, ..
    } = op
    else {
        return 1;
    };
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
    // SAFETY: the activation's register file, live and unaliased here.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs) };
    // SAFETY: the site's own cache (see the function contract).
    #[allow(unsafe_code)]
    let cache = unsafe { &mut *cache.cast::<crate::ic::PropertyCache>() };
    match super::native_call::try_coll_inline(ctx, regs[*recv as usize], regs, args, cache) {
        Some(v) => {
            regs[*dst as usize] = v;
            0
        }
        None => 1,
    }
}

/// `Op::CallWithMethod` in compiled code (the native call sequence's
/// fallback): point the site's call cache `call_site` at the method when the
/// sequence can enter it, then make the call as the interpreter's arm does
/// (`call_got_method`, with the site's persistent `cache` for its built-in
/// memo), entering a compiled VM callee directly.
///
/// # Safety
/// As [`jit_helper_call_method`].
pub(crate) extern "C" fn jit_helper_call_with_method(
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
    let Op::CallWithMethod {
        dst,
        callee,
        recv,
        key,
        args,
    } = op
    else {
        return jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw);
    };
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
    // SAFETY: the activation's register file, live and unaliased here.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs) };
    // SAFETY: the site's own cache (see the function contract).
    #[allow(unsafe_code)]
    let cache = unsafe { &mut *cache.cast::<crate::ic::PropertyCache>() };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let method = regs[*callee as usize];
    let recv_val = regs[*recv as usize];
    let r = if method
        .as_handle()
        .is_some_and(|h| ctx.realm.is_vm_function(Handle::from_raw(h)))
    {
        if !call_site.is_null() {
            // SAFETY: the op's own `CallSite` (see the function contract).
            #[allow(unsafe_code)]
            let site = unsafe { &mut *call_site.cast::<CallSite>() };
            fill_call_site(ctx, method, site);
        }
        match call_compiled(ctx, funcs, regs, method, args, recv_val) {
            Some(r) => r,
            None => super::call_got_method(ctx, funcs, regs, method, recv_val, key, args, cache),
        }
    } else {
        let served =
            super::native_call::try_call_got(ctx, funcs, method, recv_val, regs, args, cache);
        if !call_site.is_null() {
            // SAFETY: the op's own `CallSite` (see the function contract).
            #[allow(unsafe_code)]
            let site = unsafe { &mut *call_site.cast::<CallSite>() };
            // A built-in served here: the compiled code's shortcut to
            // [`jit_helper_call_native_got`] (see `IDENT_KEY`).
            site.ident.key = if served.is_some() {
                method.to_bits()
            } else {
                CallSite::EMPTY
            };
            // Which built-in it is (its id + 1, `0` for none): compiled code
            // computes `Math.imul` / `Math.clz32` of that very function
            // itself (`JitProto::compile_generic`'s `CallWithMethod`).
            // (One of the built-ins compiled code calls directly is that
            // instead: `direct_call::got_value`.)
            let direct = super::native_call::direct_kind(&ctx.realm, method);
            site.ident.value = if served.is_none() {
                0
            } else if direct != 0 {
                crate::jit::direct_call::got_value(direct)
            } else {
                method
                    .as_handle()
                    .and_then(|h| ctx.realm.native_at(Handle::from_raw(h)))
                    .map_or(0, |id| u64::from(id) + 1)
            };
            // The same memo is an inlined guard's of this call
            // (`call_seq::emit_inline_guard`), which takes a key at the
            // current epoch as its VM function: no epoch is `u64::MAX`, so
            // the guard never takes a built-in for its inlined callee.
            site.ident.stamp = u64::MAX;
        }
        match served {
            Some(r) => r,
            None => super::call_got_method(ctx, funcs, regs, method, recv_val, key, args, cache),
        }
    };
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

/// `Op::CallWithMethod` in compiled code for the built-in its site's
/// identity memo names (the one [`jit_helper_call_with_method`] last served):
/// `native_call::try_call_got` directly, else that helper.
///
/// # Safety
/// As [`jit_helper_call_with_method`].
pub(crate) extern "C" fn jit_helper_call_native_got(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    cache: *mut core::ffi::c_void,
    call_site: *mut core::ffi::c_void,
) -> u64 {
    {
        // SAFETY: see the function contract (as `jit_helper_exec_op`).
        #[allow(unsafe_code)]
        let op = unsafe { &*op_raw };
        let Op::CallWithMethod {
            dst,
            callee,
            recv,
            args,
            ..
        } = op
        else {
            return jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw);
        };
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let (ctx, regs, cache) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs),
                &mut *cache.cast::<crate::ic::PropertyCache>(),
            )
        };
        // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
        #[allow(unsafe_code)]
        let funcs = unsafe {
            &*ctx
                .jit_funcs
                .expect("jit_funcs set before generic dispatch")
        };
        let (method, recv_val) = (regs[*callee as usize], regs[*recv as usize]);
        if let Some(r) =
            super::native_call::try_call_got(ctx, funcs, method, recv_val, regs, args, cache)
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
    jit_helper_call_with_method(ctx_raw, regs_raw, n_regs, op_raw, cache, call_site)
}

/// `Op::GetMethod`'s read in compiled code (the bytecode the tier compiles
/// holds it as a `GetProp`) when the site's inline caches missed: a built-in
/// the site's memo proves (`native_call::try_get`), else
/// [`jit_helper_get_prop_site`].
///
/// # Safety
/// As [`jit_helper_get_prop_site`].
pub(crate) extern "C" fn jit_helper_get_method_site(
    ctx_raw: *mut core::ffi::c_void,
    obj: u64,
    site: *mut crate::jit::MethodSite,
    call: *const CallSite,
) -> u64 {
    {
        // SAFETY: the calling `JitProto`'s site and its property cache,
        // touched only by this helper while it runs; `ctx` the live `Ctx`.
        #[allow(unsafe_code)]
        let (ctx, site_ref) = unsafe { (&mut *(ctx_raw as *mut Ctx), &*site) };
        // SAFETY: as above.
        #[allow(unsafe_code)]
        let cache = unsafe { &mut *site_ref.cache.cast::<crate::ic::PropertyCache>() };
        // A Map/Set method on a default collection: no inline chain cache
        // can serve a collection receiver, so there is nothing to fill.
        if let Some(v) = super::native_call::try_get_coll(ctx, NanBox::from_bits(obj), cache) {
            return v.to_bits();
        }
        if let Some(v) =
            super::native_call::try_get_general(ctx, NanBox::from_bits(obj), site_ref.key(), cache)
        {
            // (And where the site's inline chain cache finds it, so the
            // compiled read hits inline next time: `Math.min` read for a
            // `CallWithMethod` the tier computes inline.)
            if let Some(h) = NanBox::from_bits(obj).as_handle() {
                // SAFETY: as above (the site is touched only here now).
                #[allow(unsafe_code)]
                let site = unsafe { &mut *site };
                site.fill(&mut ctx.realm, Handle::from_raw(h));
                // (A built-in function's own method: `Array.isArray`. An
                // array's are inherited.)
                if ctx.realm.native_at(Handle::from_raw(h)).is_some() {
                    site.fill_native_own(&mut ctx.realm, Handle::from_raw(h));
                }
            }
            return v.to_bits();
        }
    }
    jit_helper_get_prop_site(ctx_raw, obj, site, call)
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

/// A proper tail call (`Op::TailCall` / `Op::TailCallValue`) in compiled code
/// (`crate::jit::GenericOp::CallValue::tail`): its slow path (`deep` null:
/// the native call sequence could not make the ordinary call; `site`, when
/// not null, is the op's `CallSite`, refilled as `Op::CallValue`'s helper
/// does) and its deep path (`deep` non-null: the realm's call depth is half
/// its limit or more). Returns what the activation returns.
///
/// On the deep path, an activation [`run_tiering`] entered (its window is
/// `exit_regs`, and its caller is `call_with_inner`'s trampoline or an
/// interpreter loop that returns the frame's outcome to one) hands the call
/// to the trampoline exactly as the interpreter's arm of the op does — the
/// exit sentinel, with the call in [`TierRuntime::tail`] — so a chain of tail
/// calls runs in constant space. Anything else makes the ordinary call
/// (nested once: its callee, which the trampoline enters, hands its own tail
/// calls back).
///
/// # Safety
/// As [`jit_helper_exec_op`]; `site` is null or the op's `CallSite`, owned by
/// the calling `JitProto` and not otherwise borrowed while this runs.
pub(crate) extern "C" fn jit_helper_tail_call(
    ctx_raw: *mut core::ffi::c_void,
    regs_raw: *mut u64,
    n_regs: usize,
    op_raw: *const Op,
    site: *mut core::ffi::c_void,
    deep: *mut core::ffi::c_void,
) -> u64 {
    let regs_addr = regs_raw as usize;
    // SAFETY: see the function contract — a live, otherwise-unborrowed `Ctx`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: the activation's register file, live and unaliased for this call.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs) };
    // SAFETY: `op` points into the `JitProto`'s owned bytecode.
    #[allow(unsafe_code)]
    let op = unsafe { &*op_raw };
    let hand_back = !deep.is_null() && regs_addr == ctx.jit_rt.exit_regs;
    let (callee, args, static_id) = match op {
        Op::TailCall { func, args } => (NanBox::undefined(), args, Some(*func as usize)),
        Op::TailCallValue { callee, args } => (regs[*callee as usize], args, None),
        _ => {
            ctx.jit_pending_fault = Some(VmError::Unsupported);
            return NanBox::jit_throw_bits();
        }
    };
    if static_id.is_none() && !site.is_null() {
        // SAFETY: the site's own cache (see the function contract).
        #[allow(unsafe_code)]
        let site = unsafe { &mut *site.cast::<CallSite>() };
        fill_call_site(ctx, callee, site);
    }
    let argv = super::ArgBuf::from_regs(regs, args);
    if hand_back {
        // The interpreter arm's tail call (`Op::TailCall` / `TailCallValue`).
        let target = match static_id {
            Some(id) => Some((id, alloc::vec::Vec::new())),
            None => callee
                .as_handle()
                .map(Handle::from_raw)
                .filter(|&h| ctx.realm.is_vm_function(h))
                .and_then(|h| ctx.realm.vm_function(h))
                .filter(|(f, _)| (*f as usize) < funcs.len())
                .map(|(f, c)| (f as usize, c.to_vec()))
                .filter(|_| !super::tail_call_changes_realm(ctx, callee)),
        };
        if let Some((id, captures)) = target {
            ctx.jit_rt.tail = Some(TailCall {
                id,
                args: argv.to_vec(),
                captures,
                this: NanBox::undefined(),
            });
            return NanBox::jit_exit_bits();
        }
    }
    // The ordinary call (the interpreter arm's fallback), the caller's
    // registers published for the GC. A deep one goes through `call_with`,
    // whose trampoline then takes the callee's own tail calls (a compiled
    // callee entered natively would only nest again).
    let compiled = match static_id {
        None if deep.is_null() => {
            call_compiled(ctx, funcs, regs, callee, args, NanBox::undefined())
        }
        _ => None,
    };
    let r = match compiled {
        Some(r) => r,
        None => {
            let pub_mark = ctx.frame_shadow.len();
            if !on_register_stack(ctx, regs) {
                ctx.frame_shadow.extend_from_slice(regs);
            }
            ctx.frames_published += 1;
            let r = match static_id {
                Some(id) => super::vm_call(ctx, funcs, id, NanBox::undefined(), &argv),
                None => super::call_closure(ctx, funcs, callee, &argv, NanBox::undefined()),
            };
            ctx.frames_published -= 1;
            ctx.frame_shadow.truncate(pub_mark);
            r
        }
    };
    match r {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
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
    let Some(code) = jit.native_entry_for(site) else {
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
        frame_bytes: jit.native_frame_bytes(),
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
    if let Op::Construct { dst, ctor, args } = op {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let (ctx, regs, site) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                core::slice::from_raw_parts_mut(regs_raw.cast::<NanBox>(), n_regs),
                &mut *site.cast::<CallSite>(),
            )
        };
        let ctor_val = regs[*ctor as usize];
        // `new Map()` / `new Set()` of the default constructors (the
        // interpreter arm's step for a built-in constructor).
        if args.len() <= 1 {
            let argv = super::ArgBuf::from_regs(regs, args);
            if let Some(v) = ctx.realm.new_default_collection(ctor_val, &argv) {
                regs[*dst as usize] = v;
                return 0;
            }
        }
        fill_construct_site(ctx, ctor_val, site);
    }
    jit_helper_exec_op(ctx_raw, regs_raw, n_regs, op_raw)
}

/// `Op::CallSpreadPlain`'s slow path in compiled code: point the site's
/// cache at the callee when the native sequence can enter it, and — for a
/// plain array whose iteration is the built-in one — publish that open to
/// the sequence's inline test (`iter_fast::can_open`, whose answer is
/// side-effect free), then run the op as the interpreter does (which says
/// in `ok` whether it made the call).
///
/// # Safety
/// As [`jit_helper_call_value`].
pub(crate) extern "C" fn jit_helper_call_spread(
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
    if let Op::CallSpreadPlain { callee, src, .. } = op {
        // SAFETY: see the function contract.
        #[allow(unsafe_code)]
        let (ctx, callee_val, src_val, site) = unsafe {
            (
                &mut *(ctx_raw as *mut Ctx),
                NanBox::from_bits(*regs_raw.add(*callee as usize)),
                NanBox::from_bits(*regs_raw.add(*src as usize)),
                &mut *site.cast::<CallSite>(),
            )
        };
        let _ = super::iter_fast::can_open(&mut ctx.realm, src_val);
        fill_call_site(ctx, callee_val, site);
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
        && let Some(code) = jit.native_entry_for(site)
        && jit.n_regs() >= proto.n_regs
    {
        site.fill(crate::jit::CallTarget {
            key: u64::from(fid) | ((n_caps as u64) << 32),
            code,
            frame_bytes: jit.native_frame_bytes(),
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
    ctx.realm.materialize_fn_proto(nth);
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

/// Fills `site`'s [`crate::jit::FnCallMemo`] for the method call
/// `recv.call(…)` that just reached `%Function.prototype.call%` (the value
/// `method`), when its terms hold now: `recv` a VM function whose auxiliary
/// object owns no `call`, whose `[[Prototype]]` is an ordinary shaped
/// object holding `method` as its `call` data property; all three objects
/// watched. Also points the site at `recv` itself, the callee the compiled
/// sequence then calls.
fn fill_fncall_memo(ctx: &mut Ctx, site: &mut CallSite, recv: NanBox, method: u64) {
    let epoch = crate::object::proto_epoch();
    if site.fncall.recv == recv.to_bits() && site.fncall.stamp == epoch {
        return;
    }
    let Some(fh) = recv.as_handle().map(Handle::from_raw) else {
        return;
    };
    // A built-in function (`hasOwnProperty.call(o, k)`): the memo alone,
    // which compiled code may use to identify `f` (`compile_generic`); the
    // call sequence knows no native callee (its slow path calls it).
    let native = ctx.realm.native_at(fh).is_some();
    if !native && !ctx.realm.is_vm_function(fh) {
        return;
    }
    // (The memo watches the closure's own properties where they live.)
    ctx.realm.materialize_fn_proto(fh);
    let Some(aux) = ctx.realm.jit_aux_of(fh) else {
        return;
    };
    if ctx
        .realm
        .jit_object(aux)
        .is_none_or(|o| o.has_own_key("call"))
    {
        return;
    }
    let Some(holder) = ctx.realm.object_proto(fh) else {
        return;
    };
    let Some(slot) = ctx.realm.jit_object(holder).and_then(|o| {
        let k = o.shaped_shape()?.lookup("call")?;
        (o.slot(k).map(NanBox::to_bits) == Some(method))
            .then(|| o.jit_slot_ptr(k))
            .flatten()
    }) else {
        return;
    };
    ctx.realm.watch_link(fh.to_raw(), Some(aux.to_raw()));
    ctx.realm.watch_link(holder.to_raw(), None);
    if !native {
        fill_call_site(ctx, recv, site);
    }
    site.fncall = crate::jit::FnCallMemo {
        recv: recv.to_bits(),
        stamp: crate::object::proto_epoch(),
        slot: slot.cast::<u64>(),
        method,
    };
}

/// Records a construct of the value `ctor` with `new.target` `nt`, which
/// runs the body of the constructor `fh` (function `fid`: `ctor` itself, or
/// past the forwarding constructors `passed`), in `site`'s inline
/// constructor cache (`crate::jit::ConstructFast`) when the identities fix
/// its `[[Construct]]` (`jit_construct_info` having checked the rest): a
/// derived class constructor (`proto` `None`: `this` starts unbound), or a
/// base constructor without an instance-field initializer whose `new.target`
/// is a class constructor, whose `prototype` (`proto`) is then non-writable
/// and non-configurable, or an ordinary function, whose `prototype` data
/// property's slot the record then checks for `proto` on every use; the site
/// holding a way for it. Every cell involved
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
        && proto.map_or(0, Handle::to_raw) == site.fast.proto
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
    let mut pslot: *const u64 = core::ptr::null();
    if let Some(p) = proto {
        // `new.target`: a class constructor, its `prototype` fixed, or an
        // ordinary function, its `prototype` checked on every use.
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
        let pv = NanBox::handle(p.to_raw());
        let Some(o) = ctx.realm.jit_object(nt_aux) else {
            return;
        };
        if class
            && o.is_readonly("prototype")
            && o.is_non_configurable("prototype")
            && o.get("prototype").and_then(|v| v.as_handle()) == Some(p.to_raw())
        {
            // Fixed: no per-use check.
        } else {
            // An ordinary function's `prototype`, a writable data property:
            // the record checks its value where it lives (the auxiliary
            // object's slot, which stays put while the watched object keeps
            // its layout).
            let Some(ptr) = o
                .shaped_shape()
                .and_then(|sh| sh.lookup("prototype"))
                .filter(|&k| o.slot(k).map(NanBox::to_bits) == Some(pv.to_bits()))
                .and_then(|k| o.jit_slot_ptr(k))
            else {
                return;
            };
            pslot = ptr.cast::<u64>();
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
        pslot,
        pbits: proto.map_or(0, |p| NanBox::handle(p.to_raw()).to_bits()),
        target,
        caps,
        // (The constructor's own function: no forwarding constructor
        // between it and the body the way enters.)
        ident: if passed.is_empty() {
            u64::from(fid) | ((n_caps as u64) << 32)
        } else {
            crate::jit::CallSite::EMPTY
        },
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
        realm: i(offset_of!(Ctx<'static>, realm)),
        total_depth: i(offset_of!(crate::realm::Realm, vm_total_depth)),
        max_depth: i(offset_of!(crate::realm::Realm, limits)
            + offset_of!(crate::limits::Limits, max_call_depth)),
        multi_realm: i(offset_of!(crate::realm::Realm, multi_realm)),
        str_special: i(offset_of!(crate::realm::Realm, str_special)),
        callee_len: i(offset_of!(crate::realm::Realm, vm_callee_stack) + len_off),
        hosted: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, hosted)),
        main_realm: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, main_realm)),
        vmfn_plain: i(crate::realm::Realm::jit_vmfn_plain_offset()),
        poll: poll_layout(),
    }
}

/// [`crate::jit::PollLayout`]: where the inline back-edge poll reads the
/// conditions of [`jit_helper_backedge`]'s early return.
fn poll_layout() -> crate::jit::PollLayout {
    use core::mem::offset_of;
    let i = |v: usize| i32::try_from(v).expect("layout offset fits in i32");
    let (allocs, threshold) = crate::realm::Realm::jit_gc_trigger_offsets();
    crate::jit::PollLayout {
        agents: i(offset_of!(crate::realm::Realm, agents_active)),
        interrupt: interrupt_none_is_null().then(|| i(offset_of!(crate::realm::Realm, interrupt))),
        allocs: i(allocs),
        threshold: i(threshold),
        exit_regs: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, exit_regs)),
        interval: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, poll_interval)),
        demote: i(offset_of!(Ctx<'static>, jit_rt) + offset_of!(TierRuntime, demote)),
    }
}

/// Whether an `Option<Interrupt>` is one word that is zero exactly when it is
/// `None` (the null-pointer niche of the `Arc` inside), so compiled code can
/// test "no interrupt handle" with one compare.
fn interrupt_none_is_null() -> bool {
    type Opt = Option<crate::interrupt::Interrupt>;
    if core::mem::size_of::<Opt>() != 8 {
        return false;
    }
    let word = |o: &Opt| {
        // SAFETY: `Opt` is exactly eight bytes (checked above), all of them
        // initialized (a pointer-sized niche-optimized enum has no padding).
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::read((o as *const Opt).cast::<u64>())
        }
    };
    word(&None) == 0 && word(&Some(crate::interrupt::Interrupt::new())) != 0
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
            let argc = crate::jit::header_argc(*f.add(3));
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
    // The usual poll has nothing to do: no interrupt, no agents, no
    // collection due, and not the activation that may leave (the body of a
    // short loop polls at its first back-edge on every call).
    if !ctx.realm.agents_active
        && regs_addr != ctx.jit_rt.exit_regs
        && !ctx.jit_rt.demote
        && !super::vm_safepoint_due(ctx)
        && !ctx
            .realm
            .interrupt
            .as_ref()
            .is_some_and(crate::interrupt::Interrupt::is_tripped)
    {
        return ctx.jit_rt.poll_interval;
    }
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
    if core::mem::take(&mut ctx.jit_rt.demote) {
        demote_poll(ctx, funcs, ops);
    }
    ctx.jit_rt.poll_interval
}

/// A back-edge poll after an integer version asked for a register's
/// demotion (`TierRuntime::demote`), in an activation that cannot leave to
/// be recompiled (one a compiled caller called): the polling body's
/// function, if its body `wants_demote`, is recompiled now — its later
/// calls run the new code; this activation finishes in the old one.
fn demote_poll(ctx: &mut Ctx, funcs: &[FnProto], ops: *const Op) {
    let Some((id, jit)) = funcs
        .iter()
        .enumerate()
        .find_map(|(id, p)| match p.tier.jit() {
            Some(Some(j)) if j.runs(ops) => Some((id, j)),
            _ => None,
        })
    else {
        return;
    };
    if jit.wants_demote() {
        let _ = recompile(ctx, funcs, id, &jit, &[]);
    }
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
    // (The function's own pc: an inlining body numbers its ops otherwise.
    // An inlined callee's loop has none — its activation is pending, the
    // interpreter has no frame to resume — so it never leaves.)
    let own = jit.orig_pc(header);
    if own == usize::MAX || (!exit_stress() && !jit.wants_recompile(header)) {
        return false;
    }
    ctx.jit_rt.resume = Some(own);
    true
}

/// Whether the generic tier inlines calls (`KATAAN_JIT_INLINE=0` turns it
/// off; read once per process).
fn inline_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KATAAN_JIT_INLINE").is_none_or(|v| v != "0"))
}

/// The pcs of `proto`'s call sites with no callee feedback in `hints` (none
/// to inline when it is compiled): a body recompiles when one of them settles
/// on a single callee (`JitProto::note_cold_calls`).
pub(super) fn cold_call_pcs(
    proto: &FnProto,
    hints: &[crate::ic::CalleeHint],
) -> alloc::vec::Vec<usize> {
    if !inline_enabled() {
        return alloc::vec::Vec::new();
    }
    proto
        .ops
        .iter()
        .enumerate()
        .filter(|&(pc, op)| {
            matches!(
                op,
                Op::CallMethod { .. }
                    | Op::CallValue { .. }
                    | Op::CallWithMethod { .. }
                    | Op::CallKey { .. }
                    | Op::Construct { .. }
                    | Op::SuperCallArgs { .. }
                    | Op::CallSpreadPlain { .. }
            ) && hints
                .get(pc)
                .is_none_or(|h| *h == crate::ic::CalleeHint::None)
        })
        .map(|(pc, _)| pc)
        .collect()
}

/// The callee functions `caller`'s call site at pc `pc` inlines: the VM
/// functions its feedback `hints` names (one, or each of a polymorphic
/// site's), each with its warm inline caches (its compiled body's, else its
/// interpreter's) and the callees of its own call sites (by its own
/// feedback: its compiled body's, else its interpreter's), which its spliced
/// body inlines in turn. Whether they can be inlined at all is
/// `crate::jit::inline`'s to decide.
pub(super) fn inline_candidate<'a>(
    funcs: &'a [FnProto],
    caller: usize,
    hints: &[crate::ic::CalleeHint],
    pc: usize,
) -> alloc::vec::Vec<crate::jit::inline::Candidate<'a>> {
    let keys: alloc::vec::Vec<u64> = match hints.get(pc) {
        Some(&crate::ic::CalleeHint::One(key)) => alloc::vec![key],
        Some(h) => h
            .poly_fns()
            .into_iter()
            .filter_map(|f| {
                let p = funcs.get(f as usize)?;
                Some(u64::from(f) | ((p.n_captures as u64) << 32))
            })
            .collect(),
        None => alloc::vec::Vec::new(),
    };
    keys.into_iter()
        .filter_map(|k| inline_candidate_one(funcs, caller, k))
        .collect()
}

/// [`inline_candidate`] for the callee key `key`.
fn inline_candidate_one(
    funcs: &[FnProto],
    caller: usize,
    key: u64,
) -> Option<crate::jit::inline::Candidate<'_>> {
    let mut c = inline_candidate_key(funcs, caller, key)?;
    c.nested = nested_candidates(funcs, caller, &c, false);
    // (A derived constructor's `super(…)` callee, itself derived, brings its
    // own `super(…)` callee: `inline::super_nested`.)
    // (Any other callee brings the callees of its own call sites: the third
    // level, `inline::third_level`.)
    for (p, n) in &mut c.nested {
        if matches!(c.proto.ops.get(*p), Some(Op::SuperCallArgs { .. })) {
            if n.proto.derived {
                n.nested = nested_candidates(funcs, caller, n, true);
            }
        } else {
            n.nested = nested_candidates(funcs, caller, n, false);
        }
    }
    Some(c)
}

/// The callees of `c`'s own call sites, by its feedback (its compiled
/// body's, else its interpreter's) — of its `super(…)` sites only when
/// `supers` — never `caller` itself, whose body is being compiled.
fn nested_candidates<'a>(
    funcs: &'a [FnProto],
    caller: usize,
    c: &crate::jit::inline::Candidate<'a>,
    supers: bool,
) -> alloc::vec::Vec<(usize, crate::jit::inline::Candidate<'a>)> {
    let fid = (c.key & 0xffff_ffff) as usize;
    let proto = c.proto;
    let mut own = match proto.tier.jit() {
        Some(Some(j)) if j.is_generic() => j.call_hints(),
        _ => alloc::vec::Vec::new(),
    };
    let interp = super::tier_hints(proto);
    if own.len() < interp.len() {
        own.resize(interp.len(), crate::ic::CalleeHint::None);
    }
    for (h, i) in own.iter_mut().zip(interp) {
        if *h == crate::ic::CalleeHint::None {
            *h = i;
        }
    }
    (0..proto.ops.len())
        .filter(|&p| !supers || matches!(proto.ops[p], Op::SuperCallArgs { .. }))
        .filter_map(|p| {
            let crate::ic::CalleeHint::One(k) = *own.get(p)? else {
                return None;
            };
            let n = inline_candidate_key(funcs, fid, k)?;
            // (Not the outer caller again: its body is being compiled.)
            ((n.key & 0xffff_ffff) as usize != caller).then_some((p, n))
        })
        .collect()
}

/// [`inline_candidate_one`] without the callee's own inlined calls.
fn inline_candidate_key(
    funcs: &[FnProto],
    caller: usize,
    key: u64,
) -> Option<crate::jit::inline::Candidate<'_>> {
    let fid = (key & 0xffff_ffff) as usize;
    if fid == caller || !inline_enabled() {
        return None;
    }
    let proto = funcs.get(fid)?;
    if !crate::jit::inline::callee_ok(proto) && !crate::jit::inline::construct_ok(proto) {
        return None;
    }
    let (mut warm, demote) = match proto.tier.jit() {
        Some(Some(j)) if j.is_generic() => (j.runtime_warm(), j.own_int_demote()),
        _ => (alloc::vec::Vec::new(), alloc::vec::Vec::new()),
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
    Some(crate::jit::inline::Candidate {
        key,
        proto,
        warm,
        nested: alloc::vec::Vec::new(),
        demote,
    })
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
        // A write whose chain cache found an accessor: the site's setter
        // entry, for an inlined setter's guard (`MethodSite::fill_set`).
        // SAFETY: the site's own `PropertyCache`, not borrowed elsewhere
        // while this helper runs.
        #[allow(unsafe_code)]
        let accessor = unsafe { &*(site.cache as *const crate::ic::PropertyCache) }
            .chain
            .as_ref()
            .is_some_and(|c| c.accessor);
        if accessor {
            site.fill_set(&mut ctx.realm, h);
        }
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
        site.fill_native_own(&mut ctx.realm, h);
    }
    r
}

#[cfg(test)]
mod direct_memo_tests {
    use super::super::{Ctx, FnProto, PropertyCache, Realm, compile_program, jit_generic_helpers};
    use super::{Rc, TierRuntime, recompile};
    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn ctx() -> Ctx<'static> {
        Ctx {
            realm: Box::new(Realm::new()),
            host: None,
            output: String::new(),
            microtasks: alloc::collections::VecDeque::new(),
            gen_ic: PropertyCache::new(),
            gen_open_memo: None,
            agen_ic: PropertyCache::new(),
            jit_pending: None,
            jit_pending_fault: None,
            jit_funcs: None,
            jit_rt: TierRuntime::default(),
            jit_shadow: Vec::new(),
            call_depth: 0,
            gc_enabled: false,
            gc_lock: 0,
            frame_shadow: Vec::new(),
            frames_published: 0,
            top_frame_roots: Vec::new(),
            pending_callee: None,
            pending_new_target: None,
            exec_op_ic: Vec::new(),
            exec_op_ics: alloc::collections::BTreeMap::new(),
            in_realm: None,
        }
    }

    /// A recompilation that replaces the direct-entry memo keeps the body it
    /// replaces alive for the context's life, even when the function does
    /// not own that body (the memo is by id, and the body can be an earlier
    /// table copy's: compiled call sites still enter it), and it gives no
    /// memo to a function that had none (a generator's body has no native
    /// entry a call may take).
    #[test]
    fn recompile_keeps_replaced_direct_body() {
        let program = crate::parser::Parser::parse_program(
            "function f(o, k) { return o.a + k; }\nfunction g(o) { return o.b; }",
        )
        .expect("parse");
        let funcs: Vec<FnProto> = compile_program(&program).expect("compile");
        let f = funcs.iter().position(|p| p.name == "f").expect("f");
        let g = funcs.iter().position(|p| p.name == "g").expect("g");
        let h = jit_generic_helpers();
        let compile = |id: usize| {
            Rc::new(crate::jit::JitProto::compile_generic(&funcs[id], &h).expect("generic body"))
        };
        // The memo holds a body the function's own slot does not (an
        // older copy's): the only owner of that body.
        let foreign = compile(f);
        let weak = Rc::downgrade(&foreign);
        let own = compile(f);
        funcs[f].tier.set_jit(Some(Rc::clone(&own)));
        let mut cx = ctx();
        cx.jit_rt.direct.resize(funcs.len(), None);
        cx.jit_rt.direct[f] = Some(foreign);
        let new = recompile(&mut cx, &funcs, f, &own, &[]).expect("recompiled");
        assert!(
            cx.jit_rt.direct[f]
                .as_ref()
                .is_some_and(|d| Rc::ptr_eq(d, &new))
        );
        assert!(
            weak.upgrade().is_some(),
            "the replaced direct body was freed"
        );
        // No memo: a recompilation does not make one.
        let own_g = compile(g);
        funcs[g].tier.set_jit(Some(Rc::clone(&own_g)));
        let _ = recompile(&mut cx, &funcs, g, &own_g, &[]).expect("recompiled");
        assert!(cx.jit_rt.direct[g].is_none());
    }
}

#[cfg(test)]
mod poll_layout_tests {
    use super::poll_layout;

    /// The inline back-edge poll's realm offsets read exactly what the safe
    /// accessors do, on a live realm, before and after the fields change.
    #[test]
    fn poll_layout_matches_safe_reads() {
        let p = poll_layout();
        let interrupt = p
            .interrupt
            .expect("Option<Interrupt> has the null niche here");
        let mut realm = crate::realm::Realm::new();
        #[allow(unsafe_code)]
        let read = |realm: &crate::realm::Realm, off: i32| -> u64 {
            let base = (realm as *const crate::realm::Realm).cast::<u8>();
            // SAFETY: every offset is a field's, from `offset_of!`, inside the
            // realm (the 8-byte fields read whole; of the `bool`, the low
            // byte is compared).
            unsafe { core::ptr::read_unaligned(base.add(off as usize).cast::<u64>()) }
        };
        let check = |realm: &crate::realm::Realm| {
            assert_eq!(read(realm, p.agents) & 0xff, u64::from(realm.agents_active));
            assert_eq!(read(realm, interrupt) == 0, realm.interrupt.is_none());
            assert_eq!(read(realm, p.allocs), realm.gc_pressure() as u64);
            assert_eq!(read(realm, p.threshold), realm.gc_next_threshold() as u64);
        };
        check(&realm);
        for _ in 0..5 {
            realm.new_object();
        }
        realm.agents_active = true;
        realm.interrupt = Some(crate::interrupt::Interrupt::new());
        check(&realm);
        assert!(realm.gc_pressure() >= 5);
    }
}
