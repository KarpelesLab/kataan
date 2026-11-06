//! The VM side of the generic tier's deoptimization exits (see
//! `crate::jit::deopt` for the design): the helper an exit calls, which
//! runs what is left of the activation in the interpreter.

use super::super::{Ctx, FnProto, FrameExit, NanBox, Reg, VmError, run_frame_at, with_host};
use super::stack_of;
use crate::jit::DeoptInfo;

/// A deoptimization exit (`crate::jit::GenericHelpers::deopt`): the
/// activation whose `n_regs`-word window is `regs` leaves its compiled body
/// at body pc `at` (low 32 bits) with `at >> 32` inlined activations
/// active there (their native-call headers pushed by the exit's call), and
/// the interpreter takes over from that op.
///
/// The inlined levels finish first, each in the interpreter on the spliced
/// bytecode up to its `InlineLeave` (whose header is then popped), then the
/// outermost site's tail; a throw out of one ends them all and goes to the
/// activation's own handlers. The rest of the activation, from the
/// function's own pc, is handed back when an interpreter loop below can
/// continue it (`TierRuntime::deopt`; the result is then the exit
/// sentinel), else run here. The result is the body's: a value, the
/// fall-off sentinel, or the throw sentinel with the fault pending.
///
/// # Safety
/// `ctx` must be the live, otherwise-unborrowed `Ctx` the body was entered
/// with, `regs` the activation's window and `info` the body's record (alive
/// while its code runs).
pub(crate) extern "C" fn jit_helper_deopt(
    ctx: *mut core::ffi::c_void,
    regs: *mut u64,
    n_regs: usize,
    info: *const DeoptInfo,
    at: u64,
) -> u64 {
    let regs_addr = regs as usize;
    // SAFETY: see the function contract.
    #[allow(unsafe_code)]
    let (ctx, info) = unsafe { (&mut *(ctx as *mut Ctx), &*info) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs: &[FnProto] = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: the activation's window, live and unaliased for this call.
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(regs.cast::<NanBox>(), n_regs) };
    // SAFETY: the body is owned with the code that called here.
    #[allow(unsafe_code)]
    let body = unsafe { info.body() };
    info.note_exit();
    let pc = (at & 0xffff_ffff) as usize;
    let level = (at >> 32) as usize;
    // (`KATAAN_JIT_LOG`: a body's first exits, and then every thousandth.)
    if super::jit_log_enabled() && (info.taken.get() <= 4 || info.taken.get() % 1000 == 0) {
        let n = info.taken.get();
        let _ = with_function(ctx, funcs, info, |_, _, proto| {
            std::eprintln!(
                "[jit] fn {} ({} ops): deopt #{n} at body pc {pc}, {level} inlined levels",
                proto.name,
                proto.ops.len()
            );
        });
    }
    // The activation's handlers (its own pcs): its own segment, when the
    // interpreter entered it (`run_compiled_at` opens one) or the body has
    // `try` regions (a native entry then opens one); otherwise the segment
    // in force is a caller's.
    let mut handlers: alloc::vec::Vec<(usize, Reg)> =
        if info.handlers || regs_addr == ctx.jit_rt.exit_regs {
            let base = ctx.jit_rt.base;
            ctx.jit_rt.handlers.split_off(base)
        } else {
            alloc::vec::Vec::new()
        };
    // The inlined levels, innermost first, each to its leave.
    let mut at_pc = pc;
    let mut fault: Option<VmError> = None;
    for _ in 0..level {
        match crate::jit::deopt_leave_of(body, at_pc) {
            Some(leave) => {
                if fault.is_none() {
                    fault = segment(ctx, funcs, &body[..leave], regs, at_pc);
                }
                at_pc = leave + 1;
            }
            None => fault = fault.or(Some(VmError::Unsupported)),
        }
        pop_inline_header(ctx);
    }
    // The outermost site's tail, to the op after the site.
    if level > 0 && fault.is_none() {
        match crate::jit::deopt_site_end(body, at_pc) {
            Some(end) => {
                fault = segment(ctx, funcs, body.get(..end).unwrap_or(body), regs, at_pc);
                at_pc = end;
            }
            None => fault = Some(VmError::Unsupported),
        }
    }
    let mut own = info.own_pc(at_pc);
    match fault {
        None => {}
        // A throw out of an inlined body: the activation's own handler.
        Some(VmError::Thrown(v)) => match handlers.pop() {
            Some((target, reg)) if (reg as usize) < regs.len() => {
                regs[reg as usize] = v;
                own = Some(target);
            }
            _ => return throw(ctx, VmError::Thrown(v)),
        },
        Some(e) => return throw(ctx, e),
    }
    let Some(own) = own else {
        return throw(ctx, VmError::Unsupported);
    };
    // An activation the interpreter entered: its loop continues the frame.
    if regs_addr == ctx.jit_rt.exit_regs {
        ctx.jit_rt.deopt = Some((own, handlers));
        return NanBox::jit_exit_bits();
    }
    let r = with_function(ctx, funcs, info, |ctx, funcs, proto| {
        run_frame_at(ctx, funcs, &proto.ops, regs, own, handlers, None)
    });
    match r {
        Some(Ok(FrameExit::Return(Some(v)))) => v.to_bits(),
        Some(Ok(FrameExit::Return(None))) => NanBox::jit_falloff_bits(),
        Some(Ok(_)) | None => throw(ctx, VmError::Unsupported),
        Some(Err(e)) => throw(ctx, e),
    }
}

/// Runs `program` (a prefix of a spliced body, ending at an inlined body's
/// leave or past an inlined site) from `start` on `regs` with no handlers
/// of its own: `None` when it reached its end, else the fault that ended it.
fn segment(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    program: &[crate::nbvm::Op],
    regs: &mut [NanBox],
    start: usize,
) -> Option<VmError> {
    match run_frame_at(
        ctx,
        funcs,
        program,
        regs,
        start,
        alloc::vec::Vec::new(),
        None,
    ) {
        Ok(FrameExit::Return(None)) => None,
        Ok(_) => Some(VmError::Unsupported),
        Err(e) => Some(e),
    }
}

/// Ends the innermost inlined activation, whose native-call header the
/// exit's call pushed: pops the header and restores the depth count, as
/// its `InlineLeave` would have (`call_seq::emit_inline_leave`).
fn pop_inline_header(ctx: &mut Ctx) {
    let st = stack_of(ctx);
    // SAFETY: `st` is this thread's live register stack; its innermost
    // header is the inlined activation's (every call made since returned).
    #[allow(unsafe_code)]
    unsafe {
        let hdr = (*st).frames;
        if hdr.is_null() {
            return;
        }
        (*st).top = hdr;
        (*st).frames = hdr.read() as *mut u64;
    }
    ctx.realm.vm_total_depth = ctx.realm.vm_total_depth.saturating_sub(1);
}

/// The pending fault `e` and the throw sentinel.
fn throw(ctx: &mut Ctx, e: VmError) -> u64 {
    ctx.jit_pending_fault = Some(e);
    NanBox::jit_throw_bits()
}

/// Runs `f` with the function `info` is a body of and the function table
/// it is in: by the id the body recorded, else found by its code (a body
/// compiled where the id was not at hand), which is then recorded.
fn with_function<R>(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    info: &DeoptInfo,
    f: impl FnOnce(&mut Ctx, &[FnProto], &FnProto) -> R,
) -> Option<R> {
    let owns = |p: &FnProto| {
        let ptr = core::ptr::from_ref(info);
        p.tier
            .jit()
            .flatten()
            .is_some_and(|j| j.owns_deopt_info(ptr))
            || p.tier
                .twin()
                .flatten()
                .is_some_and(|j| j.owns_deopt_info(ptr))
    };
    let id = info.func.get() as usize;
    if id < funcs.len() {
        return Some(f(ctx, funcs, &funcs[id]));
    }
    if info.func.get() == u32::MAX
        && let Some(i) = funcs.iter().position(owns)
    {
        info.func.set(u32::try_from(i).unwrap_or(u32::MAX));
        return Some(f(ctx, funcs, &funcs[i]));
    }
    // A function eval code defined after this table was taken lives in the
    // host's newer (extended) table (as `call_with` finds it).
    let newer = with_host(ctx, |h| h.vm_table_now()).flatten()?;
    let i = if id < newer.len() {
        id
    } else {
        let i = newer.iter().position(owns)?;
        info.func.set(u32::try_from(i).unwrap_or(u32::MAX));
        i
    };
    Some(f(ctx, &newer, &newer[i]))
}
