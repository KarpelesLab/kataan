//! The lean resumption of a hot VM async function parked on an `await`
//! (the job an await's reaction runs: see `super::resume_async_frame`).
//!
//! The general resumption moves the register file and handler stack out of
//! the frame cell, copies the registers onto a window of the register stack
//! for the compiled body, copies them back, builds the frame outcome, and
//! moves everything back into the cell at the next `await`. Here, as for a
//! hot generator's `next(v)` (`super::gen_fast`), the body runs in place on
//! the cell's register file — kept at the compiled body's size between
//! resumptions, and rooted while it runs by `Realm::vm_gen_running` — with
//! one borrow of the cell for every read and write before the run, the
//! frame written back through that borrow at the next `await` when no sweep
//! ran meanwhile, and the next `await` registered straight from there.

use super::jit_tier::{self, GenRun};
use super::{
    Ctx, FnProto, FnTable, HostError, Op, VmError, await_settled_now, drive_async, run_frame_at,
    with_host,
};
use crate::heap::Handle;
use crate::nanbox::NanBox;

/// What [`lean_resume`] did.
pub(super) enum Lean {
    /// Nothing: the frame is not one it resumes (the general path does).
    Untaken,
    /// The resumption ran to its next suspension or its end (the frame's
    /// outcome handled, as `drive_async` handles it).
    Done(Result<(), VmError>),
}

/// The fulfilled resumption of the hot async function parked in its frame
/// cell `fh` (its own `controller`) with the awaited value `v`, run in
/// place: to its next `await` (registered as `drive_async` registers it, or,
/// for a value [`await_settled_now`] proves would resume it next, continued
/// with in place), or to its end (its promise settled by `drive_async`).
/// `Untaken`, with nothing done, for a frame whose body is not compiled or
/// not hot, a resumption that may recompile it, or a register file of
/// another size.
#[inline(never)]
pub(super) fn lean_resume(
    ctx: &mut Ctx,
    table: &FnTable,
    controller: NanBox,
    fh: Handle,
    v: NanBox,
) -> Lean {
    let epoch = ctx.realm.heap_epoch();
    let Some(f) = ctx.realm.vm_frame_mut(fh) else {
        return Lean::Untaken;
    };
    let fp: *mut crate::cell::VmFrame = f;
    let (id, p, mut pc, dst, len) = (f.func as usize, f.aux, f.pc as usize, f.dst, f.regs.len());
    let Some(proto) = table.get(id) else {
        return Lean::Untaken;
    };
    let own = proto.n_regs;
    let Some(mut jit) = hot_body(proto, Some(len)) else {
        return Lean::Untaken;
    };
    let n = jit.n_regs().max(own);
    // SAFETY: `fp` is the frame cell's frame, borrowed just above; nothing
    // else borrowed the realm since.
    #[allow(unsafe_code)]
    let f = unsafe { &mut *fp };
    if len != n {
        f.regs.resize(n, NanBox::undefined());
    }
    if let Some(slot) = f.regs.get_mut(dst as usize) {
        *slot = v;
    }
    let (ptr, rlen) = (f.regs.as_mut_ptr(), f.regs.len());
    let mut handlers = core::mem::take(&mut f.handlers);
    // While the frame runs, nothing else need reference its controller (the
    // reaction that resumed it has been dequeued) or its result promise:
    // both are rooted for the body's safepoints, with the settled value (as
    // the general resumption roots them).
    let mark = ctx.frame_shadow.len();
    ctx.frame_shadow.extend_from_slice(&[controller, p, v]);
    // SAFETY: the buffer belongs to the frame cell's register file, which
    // nothing resizes, moves or frees while the frame runs (the cell is its
    // own controller, rooted above; a running frame has no reaction that
    // could resume it; a collection only reads it).
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(ptr, rlen) };
    let r = loop {
        ctx.realm.vm_gen_running.push(fh);
        let run = jit_tier::run_generator_body_in_place(ctx, table, &jit, regs, pc, &mut handlers);
        ctx.realm.vm_gen_running.pop();
        match run {
            GenRun::Yield {
                value,
                pc: at,
                dst: d,
            } => {
                // An `await` whose resumption would be the very next job, of
                // a value already settled as fulfilled: that job's run, here
                // and now, while the body stays hot (the registers are still
                // the cell's, rooted while the body runs).
                let settled = await_settled_now(ctx, table, value);
                if let Some(w) = settled
                    && let Some(j) = hot_body(proto, Some(rlen))
                    && j.n_regs().max(own) == rlen
                {
                    jit = j;
                    regs[d as usize] = w;
                    pc = at;
                    continue;
                }
                // With no sweep since the borrow above, the cell is where it
                // was and still remembered (or young): its frame is written
                // through the pointer that borrow gave.
                let f = if ctx.realm.heap_epoch() == epoch {
                    // SAFETY: as above, and no sweep moved or freed the box.
                    #[allow(unsafe_code)]
                    Some(unsafe { &mut *fp })
                } else {
                    ctx.realm.vm_frame_mut(fh)
                };
                let Some(f) = f else {
                    break Err(VmError::NotAnObject);
                };
                f.handlers = handlers;
                f.pc = at as u32;
                f.dst = d;
                let fns = f.resume_fns;
                ctx.realm.vm_frame_stored(fh);
                // (A settled value with the body no longer hot: its run, by
                // the general resumption.)
                if let Some(w) = settled {
                    break super::resume_parked(ctx, table, controller, fh, w);
                }
                break match with_host(ctx, |h| h.await_resume(value, fns)) {
                    Some(Ok(())) => Ok(()),
                    // PromiseResolve threw: that throw happens at the
                    // `await`, in the interpreter.
                    Some(Err(HostError::Thrown(e))) => {
                        let (mut rg, hs) = take_frame(ctx, fh, own);
                        let r = run_frame_at(ctx, table, &proto.ops, &mut rg, at, hs, Some(e));
                        drive_async(ctx, table, id, Some(fh), p, rg, r)
                    }
                    _ => Err(VmError::Unsupported),
                };
            }
            GenRun::Exit(r) => {
                let (rg, _) = take_frame(ctx, fh, own);
                break drive_async(ctx, table, id, Some(fh), p, rg, r);
            }
            // (A body that refuses to run — no entry at `pc` — is
            // interpreted, cold.)
            GenRun::Refused => {
                let (mut rg, _) = take_frame(ctx, fh, own);
                let r = run_frame_at(ctx, table, &proto.ops, &mut rg, pc, handlers, None);
                break drive_async(ctx, table, id, Some(fh), p, rg, r);
            }
        }
    };
    ctx.frame_shadow.truncate(mark);
    Lean::Done(r)
}

/// The compiled generic body of `proto` when its next resumption runs it
/// without leaving it (hot, and not one that may recompile it), counting
/// that resumption; `len`: the register file's size, which must be the
/// function's own or the body's.
fn hot_body(proto: &FnProto, len: Option<usize>) -> Option<alloc::rc::Rc<crate::jit::JitProto>> {
    let mut t = proto.tier.0.try_borrow_mut().ok()?;
    let n = t.resumes.saturating_add(1);
    let j = match &t.jit {
        Some(Some(j))
            if j.is_generic()
                && jit_tier::resume_is_hot(n)
                && !jit_tier::resume_may_recompile(n)
                && len.is_none_or(|l| l == proto.n_regs || l == j.n_regs().max(proto.n_regs)) =>
        {
            alloc::rc::Rc::clone(j)
        }
        _ => return None,
    };
    t.resumes = n;
    Some(j)
}

/// The call of the hot async function `id` (`proto`, a plain one: no rest
/// parameter, `arguments` or `new.target`, called without `new`) with
/// `args`, `captures` and `this_val`, from `call_with_inner` (whose caller
/// did the activation's bookkeeping): its compiled body run from its start
/// on a window of the register stack, as a hot generator's is, without the
/// general path's tier-up checks and interpreter caches — to its first
/// `await` or its end, handled by `drive_async` — and its promise
/// returned. `None`, with nothing done, for a body not compiled or not hot,
/// or a call that may recompile it (one in `RECOMPILE_EVERY`, which the
/// general path makes).
#[allow(clippy::too_many_arguments)]
pub(super) fn call_lean(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    proto: &FnProto,
    args: &[NanBox],
    captures: &[NanBox],
    this_val: NanBox,
    new_target: Option<NanBox>,
) -> Option<Result<NanBox, VmError>> {
    if new_target.is_some()
        || proto.rest_from.is_some()
        || ctx.host.is_none()
        || matches!(proto.ops.first(), Some(Op::MakeArguments { .. }))
        || proto
            .ops
            .iter()
            .take(2)
            .any(|op| matches!(op, Op::LoadNewTarget { .. }))
    {
        return None;
    }
    let jit = {
        let mut t = proto.tier.0.try_borrow_mut().ok()?;
        let n = t.calls.saturating_add(1);
        let j = match &t.jit {
            Some(Some(j))
                if j.is_generic()
                    && jit_tier::call_is_hot(t.body.is_some())
                    && (t.body.is_some() || n != super::TIER_UP_THRESHOLD)
                    && !jit_tier::resume_may_recompile(n) =>
            {
                alloc::rc::Rc::clone(j)
            }
            _ => return None,
        };
        t.calls = n;
        j
    };
    let mut regs = ctx.realm.vm_reg_pool.pop().unwrap_or_default();
    regs.resize(proto.n_regs, NanBox::undefined());
    for (slot, a) in regs.iter_mut().zip(args.iter().take(proto.n_params)) {
        *slot = *a;
    }
    for (i, c) in captures.iter().enumerate().take(proto.n_captures) {
        regs[proto.n_params + i] = *c;
    }
    if let Some(slot) = regs.get_mut(proto.n_params + proto.n_captures) {
        *slot = this_val;
    }
    // The activation's inputs, published as `call_with_inner` publishes them
    // (its caller truncates the shadow stack).
    if ctx.gc_enabled {
        if ctx.call_depth == 1 {
            ctx.top_frame_roots.clear();
            ctx.top_frame_roots.extend_from_slice(args);
            ctx.top_frame_roots.extend_from_slice(captures);
            ctx.top_frame_roots.push(this_val);
        } else {
            ctx.frame_shadow.extend_from_slice(args);
            ctx.frame_shadow.extend_from_slice(captures);
            ctx.frame_shadow.push(this_val);
        }
    }
    let p = super::vm_new_promise(ctx)?;
    // (The result promise lives only in this Rust local until the body
    // returns or parks on a controller.)
    ctx.frame_shadow.push(p);
    let mut handlers = alloc::vec::Vec::new();
    let r = match jit_tier::run_generator_body(ctx, funcs, &jit, &mut regs, 0, &mut handlers) {
        GenRun::Yield { value, pc, dst } => Ok(super::FrameExit::Yield {
            value,
            pc,
            handlers,
            dst,
        }),
        GenRun::Exit(r) => r,
        // (Nothing ran: the interpreter runs it.)
        GenRun::Refused => run_frame_at(ctx, funcs, &proto.ops, &mut regs, 0, handlers, None),
    };
    Some(drive_async(ctx, funcs, id, None, p, regs, r).map(|()| p))
}

/// Whether [`lean_resume`] would take the next resumption of a frame of
/// `proto` whose register file has `len` registers (nothing counted).
pub(super) fn lean_ready(proto: &FnProto, len: usize) -> bool {
    let Ok(t) = proto.tier.0.try_borrow() else {
        return false;
    };
    let n = t.resumes.saturating_add(1);
    matches!(&t.jit, Some(Some(j))
        if j.is_generic()
            && jit_tier::resume_is_hot(n)
            && !jit_tier::resume_may_recompile(n)
            && (len == proto.n_regs || len == j.n_regs().max(proto.n_regs)))
}

/// The register file (cut back to the function's own `own` registers) and
/// handler stack moved out of the frame cell `fh`.
fn take_frame(
    ctx: &mut Ctx,
    fh: Handle,
    own: usize,
) -> (
    alloc::vec::Vec<NanBox>,
    alloc::vec::Vec<(usize, super::Reg)>,
) {
    match ctx.realm.vm_frame_mut(fh) {
        Some(f) => {
            let mut regs = core::mem::take(&mut f.regs);
            regs.truncate(own);
            (regs, core::mem::take(&mut f.handlers))
        }
        None => (alloc::vec::Vec::new(), alloc::vec::Vec::new()),
    }
}
