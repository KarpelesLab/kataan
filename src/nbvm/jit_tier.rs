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
    Ctx, FnProto, FrameExit, Handle, NanBox, Op, Reg, VmError, ensure_jit, run_frame,
    vm_agent_tick, vm_error, vm_load_global, vm_safepoint,
};
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
#[derive(Default)]
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
}

/// Runs compiled `jit` on `regs` from bytecode pc `entry` with the activation's
/// handler stack `handlers` (non-empty only for an on-stack replacement inside a
/// `try`), translating its raw result word into the interpreter's frame outcome.
/// `None` if nothing ran (not a generic body, short register file, or not an
/// entry point).
pub(super) fn run_compiled(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
    entry: usize,
    handlers: &[(usize, Reg)],
) -> Option<Result<FrameExit, VmError>> {
    let saved = ctx.jit_funcs.replace(funcs as *const [FnProto]);
    let outer_base = ctx.jit_rt.base;
    let base = ctx.jit_rt.handlers.len();
    ctx.jit_rt.base = base;
    ctx.jit_rt.handlers.extend_from_slice(handlers);
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    let raw = jit.call_frame_raw(ctx_ptr, regs, entry);
    ctx.jit_rt.handlers.truncate(base);
    ctx.jit_rt.base = outer_base;
    ctx.jit_funcs = saved;
    let raw = raw?;
    if raw == NanBox::jit_falloff_bits() {
        return Some(Ok(FrameExit::Return(None)));
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
    Some(Ok(FrameExit::Return(Some(nb))))
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
) -> Option<Result<NanBox, VmError>> {
    let proto = funcs.get(id)?;
    if proto.is_async || proto.is_generator {
        return None;
    }
    let mut stack = alloc::collections::BTreeSet::new();
    let jit = ensure_jit(&mut ctx.jit_cache, funcs, id, &mut stack)?;
    if jit.is_generic() {
        return match run_compiled(ctx, funcs, &jit, regs, 0, &[])? {
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
    jit.call_guarded(args).map(Ok)
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
) -> Option<Result<FrameExit, VmError>> {
    if mode() == Mode::Off {
        return None;
    }
    let ptr = program.as_ptr();
    let id = funcs
        .iter()
        .position(|f| core::ptr::eq(f.ops.as_ptr(), ptr) && f.ops.len() == program.len())
        .or_else(|| {
            ctx.tiers.iter().enumerate().find_map(|(id, t)| {
                t.body
                    .as_ref()
                    .filter(|b| core::ptr::eq(b.as_ptr(), ptr) && b.len() == program.len())
                    .map(|_| id)
            })
        })?;
    let proto = funcs.get(id)?;
    if proto.is_generator || proto.ops.len() != program.len() {
        return None;
    }
    let mut stack = alloc::collections::BTreeSet::new();
    let jit: Rc<crate::jit::JitProto> = ensure_jit(&mut ctx.jit_cache, funcs, id, &mut stack)?;
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
    run_compiled(ctx, funcs, &jit, regs, target, handlers)
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
        Op::CallValue { dst, callee, args } => {
            let callee = regs[*callee as usize];
            call_compiled(ctx, funcs, regs, callee, args, NanBox::undefined())
                .map(|r| r.map(|v| regs[*dst as usize] = v))
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
    // which `ctx.jit_cache` keeps alive for the whole `Ctx`, so the address
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
        ic.push(ctx.exec_op_ics.remove(&key).unwrap_or_default());
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
            ctx.exec_op_ics.insert(key, c);
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
    let handle = Handle::from_raw(callee_val.as_handle()?);
    let (fid, _) = ctx.realm.vm_function(handle)?;
    let id = fid as usize;
    let proto = funcs.get(id)?;
    if ctx.realm.multi_realm || ctx.call_depth == 0 {
        return None;
    }
    let jit = match ctx.jit_rt.direct.get(id) {
        Some(Some(j)) => Rc::clone(j),
        _ => {
            let j = direct_target(ctx, proto, id)?;
            if ctx.jit_rt.direct.len() <= id {
                ctx.jit_rt.direct.resize(id + 1, None);
            }
            ctx.jit_rt.direct[id] = Some(Rc::clone(&j));
            j
        }
    };
    // `call_with`'s recursion guard.
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
    {
        let e = vm_error(ctx, "RangeError", "Maximum call stack size exceeded");
        return Some(Err(VmError::Thrown(e)));
    }
    // The callee's register file: parameters, captures, `this` (undefined).
    let mut cregs: alloc::vec::Vec<NanBox> = ctx.reg_pool.pop().unwrap_or_default();
    cregs.resize(proto.n_regs, NanBox::undefined());
    for (slot, a) in cregs.iter_mut().zip(args.iter().take(proto.n_params)) {
        *slot = regs[*a as usize];
    }
    if let Some((_, caps)) = ctx.realm.vm_function(handle) {
        for (i, c) in caps.iter().enumerate().take(proto.n_captures) {
            cregs[proto.n_params + i] = *c;
        }
    }
    if let Some(slot) = cregs.get_mut(proto.n_params + proto.n_captures) {
        *slot = this;
    }
    // The bookkeeping of `Op::CallValue` + `call_with` (+ `call_with_inner`'s
    // input publication), restored below on every outcome.
    let mark = ctx.frame_shadow.len();
    let published = ctx.frames_published;
    ctx.frame_shadow.extend_from_slice(regs);
    ctx.frames_published += 1;
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let callee_mark = ctx.realm.vm_callee_stack.len();
    let args_mark = ctx.realm.vm_args_buf.len();
    ctx.realm.vm_callee_stack.push(callee_val);
    ctx.realm.vm_callee_args.push(args_mark);
    ctx.realm
        .vm_args_buf
        .extend(args.iter().map(|a| regs[*a as usize]));
    if ctx.gc_enabled {
        let n = ctx.realm.vm_args_buf.len();
        ctx.frame_shadow
            .extend_from_slice(&ctx.realm.vm_args_buf[args_mark..n]);
        if let Some((_, caps)) = ctx.realm.vm_function(handle) {
            ctx.frame_shadow.extend_from_slice(caps);
        }
        ctx.frame_shadow.push(this);
    }
    if let Some(t) = ctx.tiers.get_mut(id) {
        t.calls = t.calls.saturating_add(1);
    }
    let outcome = run_compiled(ctx, funcs, &jit, &mut cregs, 0, &[]);
    ctx.realm.vm_callee_stack.truncate(callee_mark);
    ctx.realm.vm_callee_args.truncate(callee_mark);
    ctx.realm.vm_args_buf.truncate(args_mark);
    ctx.frames_published = published;
    ctx.frame_shadow.truncate(mark);
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    if ctx.reg_pool.len() < 64 {
        cregs.clear();
        ctx.reg_pool.push(cregs);
    }
    Some(match outcome? {
        Ok(FrameExit::Return(v)) => Ok(v.unwrap_or(NanBox::undefined())),
        Ok(_) => Err(VmError::Unsupported),
        Err(e) => Err(e),
    })
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

/// Whether function `id` (`proto`) may be entered by [`call_compiled`]: a plain
/// VM closure body (not a class constructor, async, generator, rest-parameter,
/// `arguments` or `new.target` one) that is hot and already compiled to the
/// generic tier — its compiled body, if so.
fn direct_target(ctx: &Ctx, proto: &FnProto, id: usize) -> Option<Rc<crate::jit::JitProto>> {
    if proto.class_ctor
        || proto.is_async
        || proto.is_generator
        || proto.rest_from.is_some()
        || proto
            .ops
            .iter()
            .take(2)
            .any(|o| matches!(o, Op::MakeArguments { .. } | Op::LoadNewTarget { .. }))
        || !call_is_hot(ctx.tiers.get(id).is_some_and(|t| t.body.is_some()))
    {
        return None;
    }
    match ctx.jit_cache.get(&id) {
        Some(Some(j)) if j.is_generic() && j.n_regs() <= proto.n_regs => Some(Rc::clone(j)),
        _ => None,
    }
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
) -> u64 {
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
        return 1;
    }
    vm_safepoint(ctx, funcs, program, regs);
    POLL_INTERVAL
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
