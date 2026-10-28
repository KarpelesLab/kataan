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
    Ctx, FnProto, FrameExit, NanBox, Op, VmError, ensure_jit, run_frame, vm_agent_tick, vm_error,
    vm_safepoint,
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

/// Runs compiled `jit` on `regs` from bytecode pc `entry`, translating its raw
/// result word into the interpreter's frame outcome. `None` if nothing ran (not a
/// generic body, short register file, or not an entry point).
pub(super) fn run_compiled(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    regs: &mut [NanBox],
    entry: usize,
) -> Option<Result<FrameExit, VmError>> {
    let saved = ctx.jit_funcs.replace(funcs as *const [FnProto]);
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    let raw = jit.call_frame_raw(ctx_ptr, regs, entry);
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
        return match run_compiled(ctx, funcs, &jit, regs, 0)? {
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
/// native code from `target` and its outcome is returned; `None` means "keep
/// interpreting" (and the caller stops asking for this activation).
pub(super) fn osr(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    program: &[Op],
    regs: &mut [NanBox],
    target: usize,
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
    run_compiled(ctx, funcs, &jit, regs, target)
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
    match run_frame(ctx, funcs, core::slice::from_ref(op), regs) {
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
    match run_compiled(ctx, funcs, jit, &mut regs, 0)? {
        Ok(FrameExit::Return(v)) => Some(Ok(v.unwrap_or(NanBox::undefined()))),
        Ok(_) => Some(Err(VmError::Unsupported)),
        Err(e) => Some(Err(e)),
    }
}
