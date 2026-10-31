//! The resumption of a suspended sync generator (`%GeneratorPrototype%`'s
//! `next` / `throw` / `return`), and its direct forms for the VM's own
//! callers.
//!
//! A generator object keeps its suspended frame in a hidden slot ([`VM_GEN`])
//! naming its frame cell (`crate::cell::VmFrame`: function, state, resume
//! point, register file, handler stack). A resume moves the register file
//! and handler stack out of the cell and runs the frame; a `yield` moves
//! them back and updates the rest in place, so a `yield` costs no
//! allocation beyond its iterator result.
//!
//! Called from the VM itself — `gen.next(v)` at a method-call site that
//! proved the method is the intrinsic (`native_call`), or a `for-of` step
//! whose `next` is the intrinsic — the resume runs in the caller's VM
//! context, without the host round trip (which would swap the whole realm
//! out and back), and a `for-of` step reads the yielded value without making
//! the result object at all: CreateIterResultObject's fresh object is not
//! observable there, its `done` and `value` being ordinary own data
//! properties. Either is what the host's call of the intrinsic does: the
//! same brand check, the same resume, and (without a host, or with several
//! realms) the ordinary path instead.

use super::{
    Ctx, FnProto, FrameExit, GEN_COMPLETED, GEN_EXECUTING, GEN_SUSPENDED_START,
    GEN_SUSPENDED_YIELD, Op, STACK_EXHAUSTED, VM_GEN, VmError, host_get_str, run_frame_at_ic,
    vm_error,
};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::vec::Vec;

/// What one resume of a generator ended in.
pub(super) enum GenOut {
    /// A `yield value`.
    Yield(NanBox),
    /// A `yield*` step handing the inner iterator's result object through.
    Delegated(NanBox),
    /// The generator completed (a `return`, or a `return(v)` resumption)
    /// with this value.
    Done(NanBox),
}

/// The generator states, as a frame cell holds them.
const S_START: u8 = GEN_SUSPENDED_START as u8;
const S_YIELD: u8 = GEN_SUSPENDED_YIELD as u8;
const S_EXECUTING: u8 = GEN_EXECUTING as u8;
const S_COMPLETED: u8 = GEN_COMPLETED as u8;

/// The generator object `gh`'s frame cell, when it is a sync VM generator.
fn state_of(ctx: &mut Ctx, gh: Handle) -> Option<Handle> {
    ctx.realm
        .object_get_cached(gh, VM_GEN, &mut ctx.gen_ic)
        .or_else(|| ctx.realm.get_property(gh, VM_GEN))
        .and_then(|s| s.as_handle())
        .map(Handle::from_raw)
}

/// GeneratorResume / GeneratorResumeAbrupt of the VM generator `gh`: `mode`
/// 0 = `next(v)`, 1 = `throw(v)`, 2 = `return(v)`.
pub(super) fn resume(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    gh: Handle,
    mode: u8,
    v: NanBox,
) -> Result<GenOut, VmError> {
    let fh = state_of(ctx, gh).ok_or(VmError::NotAnObject)?;
    resume_state(ctx, funcs, fh, mode, v)
}

/// Marks the frame `fh` completed.
fn complete(ctx: &mut Ctx, fh: Handle) {
    if let Some(f) = ctx.realm.vm_frame_mut(fh) {
        f.state = S_COMPLETED;
    }
}

/// [`resume`] of the generator whose frame cell is `fh`.
fn resume_state(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    fh: Handle,
    mode: u8,
    v: NanBox,
) -> Result<GenOut, VmError> {
    let (id, state, pc, dst, has_handlers) = match ctx.realm.vm_frame(fh) {
        Some(f) => (
            f.func as usize,
            f.state,
            f.pc as usize,
            f.dst,
            !f.handlers.is_empty(),
        ),
        None => return Err(VmError::NotAnObject),
    };
    if state == S_EXECUTING {
        let e = vm_error(ctx, "TypeError", "Generator is already running");
        return Err(VmError::Thrown(e));
    }
    let proto = funcs.get(id);
    // Suspended inside a `yield*` (the delegation step) or at a capturing
    // `YieldResume`: every resumption kind lands in registers.
    let delegate = (state == S_YIELD).then_some(proto).flatten().and_then(|p| {
        if let Some(Op::YieldDelegate { dst, mode, .. }) = p.ops.get(pc) {
            return Some((*dst, *mode));
        }
        match pc.checked_sub(1).and_then(|i| p.ops.get(i)) {
            Some(Op::YieldResume { dst, mode, .. }) => Some((*dst, *mode)),
            _ => None,
        }
    });
    // A `return(v)` at a `yield` inside a try / for-of / destructuring region
    // must run that region's cleanup (iterator closes) — the interpreter's.
    if delegate.is_none() && mode == 2 && state == S_YIELD && has_handlers {
        return Err(VmError::Unsupported);
    }
    if delegate.is_none() && (state == S_COMPLETED || (state == S_START && mode != 0) || mode == 2)
    {
        complete(ctx, fh);
        return match mode {
            1 => Err(VmError::Thrown(v)),
            2 => Ok(GenOut::Done(v)),
            _ => Ok(GenOut::Done(NanBox::undefined())),
        };
    }
    let Some(proto) = proto else {
        return Err(VmError::Unsupported);
    };
    // The register file and handler stack move out of the frame for the run
    // (and back in at the next `yield`); a completed generator's stay empty.
    let (mut regs, handlers) = match ctx.realm.vm_frame_mut(fh) {
        Some(f) => {
            f.state = S_EXECUTING;
            (
                core::mem::take(&mut f.regs),
                core::mem::take(&mut f.handlers),
            )
        }
        None => return Err(VmError::NotAnObject),
    };
    let inject = if let Some((ddst, dmode)) = delegate {
        regs[ddst as usize] = v;
        regs[dmode as usize] = NanBox::number(f64::from(mode));
        None
    } else {
        if mode == 0
            && state == S_YIELD
            && let Some(slot) = regs.get_mut(dst as usize)
        {
            *slot = v;
        }
        (mode == 1).then_some(v)
    };
    // The body's inline caches persist across resumptions, as they do across
    // calls (see `TierState`): taken out for the run (a nested resume of
    // the same function runs cold) and put back after it.
    let body = proto.ops.as_ptr() as usize;
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    let (mut hot, mut resumes) = (false, 0);
    let mut ic = match proto.tier.0.try_borrow_mut() {
        Ok(mut t) => {
            #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
            {
                t.resumes = t.resumes.saturating_add(1);
                resumes = t.resumes;
                hot = inject.is_none()
                    && !matches!(t.jit, Some(None))
                    && super::jit_tier::resume_is_hot(t.resumes);
            }
            if t.ic_body == body {
                core::mem::take(&mut t.ic)
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    };
    // A hot generator's body resumes in native code (see
    // `jit_tier::resume_generator`), on the same register file.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    let compiled = if hot {
        super::jit_tier::resume_generator(ctx, funcs, id, &mut regs, pc, &handlers, &ic, resumes)
    } else {
        None
    };
    #[cfg(not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")))]
    let compiled = None;
    let r = match compiled {
        Some(r) => r,
        None => run_frame_at_ic(
            ctx, funcs, &proto.ops, &mut regs, pc, handlers, inject, &mut ic,
        ),
    };
    if let Ok(mut t) = proto.tier.0.try_borrow_mut() {
        t.ic = ic;
        t.ic_body = body;
    }
    match r {
        Ok(FrameExit::Yield {
            value,
            pc,
            handlers,
            dst,
        }) => {
            if let Some(f) = ctx.realm.vm_frame_mut(fh) {
                f.regs = regs;
                f.handlers = handlers;
                f.pc = pc as u32;
                f.dst = dst;
                f.state = S_YIELD;
            }
            ctx.realm.vm_frame_stored(fh);
            // A `yield*` step hands the inner result object through as is.
            if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                Ok(GenOut::Delegated(value))
            } else {
                Ok(GenOut::Yield(value))
            }
        }
        Ok(FrameExit::Return(ret)) => {
            complete(ctx, fh);
            recycle(ctx, regs);
            Ok(GenOut::Done(ret.unwrap_or(NanBox::undefined())))
        }
        Ok(FrameExit::Tail { .. }) => {
            complete(ctx, fh);
            Err(VmError::Unsupported)
        }
        Err(err) => {
            complete(ctx, fh);
            recycle(ctx, regs);
            Err(err)
        }
    }
}

/// A finished frame's register file goes back to the pool, as an ordinary
/// call's does.
fn recycle(ctx: &mut Ctx, mut regs: Vec<NanBox>) {
    if ctx.realm.vm_reg_pool.len() < 64 {
        regs.clear();
        ctx.realm.vm_reg_pool.push(regs);
    }
}

/// The iterator result of a resume's outcome.
pub(super) fn result_object(ctx: &mut Ctx, out: GenOut) -> NanBox {
    let (v, done) = match out {
        GenOut::Delegated(r) => return r,
        GenOut::Yield(v) => (v, false),
        GenOut::Done(v) => (v, true),
    };
    NanBox::handle(ctx.realm.new_iter_result(v, done).to_raw())
}

/// Runs `f` — a resume in the caller's own context — as a nested activation:
/// counted against the recursion limits (a catchable `RangeError` past
/// them), and with the caller's registers unpublished, so no collection runs
/// inside it (as none does on the host's path, whose context forbids it).
fn nested<T>(ctx: &mut Ctx, f: impl FnOnce(&mut Ctx) -> Result<T, VmError>) -> Result<T, VmError> {
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        || crate::stack::exhausted()
    {
        let e = vm_error(ctx, "RangeError", STACK_EXHAUSTED);
        return Err(VmError::Thrown(e));
    }
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let r = f(ctx);
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    r
}

/// The frame cell of the generator object `g` when the direct path may
/// resume it: a hosted, single-realm run, and `g` a sync VM generator of a
/// function in `funcs`.
fn direct_target(ctx: &mut Ctx, funcs: &[FnProto], g: NanBox) -> Option<Handle> {
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let fh = state_of(ctx, Handle::from_raw(g.as_handle()?))?;
    let id = ctx.realm.vm_frame(fh)?.func as usize;
    (id < funcs.len()).then_some(fh)
}

/// `g.next(v)` / `g.throw(v)` / `g.return(v)` (`mode` 0 / 1 / 2) for a call
/// site that proved the method is the intrinsic: the iterator result, or
/// `None` (nothing done) to take the host's call.
pub(super) fn call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    g: NanBox,
    mode: u8,
    v: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let sh = direct_target(ctx, funcs, g)?;
    Some(nested(ctx, |ctx| resume_state(ctx, funcs, sh, mode, v)).map(|o| result_object(ctx, o)))
}

/// A `for-of` step over the generator `it` whose `next` is the intrinsic
/// `%GeneratorPrototype%.next`: the yielded value (`Some`), or `None` once
/// done; `None` from the outer `Option` (nothing done) to take the general
/// step. A `yield*` step's inner result object is read as IteratorStep
/// reads any result: `done`, then `value`, through `[[Get]]`.
pub(super) fn for_of_step(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    it: NanBox,
) -> Option<Result<Option<NanBox>, VmError>> {
    let sh = direct_target(ctx, funcs, it)?;
    let out = match nested(ctx, |ctx| {
        resume_state(ctx, funcs, sh, 0, NanBox::undefined())
    }) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    Some(match out {
        GenOut::Yield(v) => Ok(Some(v)),
        GenOut::Done(_) => Ok(None),
        GenOut::Delegated(r) => (|| {
            let done = host_get_str(ctx, r, "done")?;
            if ctx.realm.truthy(done) {
                return Ok(None);
            }
            host_get_str(ctx, r, "value").map(Some)
        })(),
    })
}
