//! The resumption of a suspended sync generator (`%GeneratorPrototype%`'s
//! `next` / `throw` / `return`), and its direct forms for the VM's own
//! callers.
//!
//! A generator object keeps its suspended frame in a hidden state array
//! ([`VM_GEN`]): `[func id, state, pc, dst, registers, handlers]`. A resume
//! moves the register file out of its array and runs the frame; a `yield`
//! moves it back and updates the other entries in place, so a `yield` costs
//! no allocation beyond its iterator result.
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
    GEN_SUSPENDED_YIELD, Op, Reg, STACK_EXHAUSTED, VM_GEN, VmError, host_get_str, run_frame_at_ic,
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

/// The generator object `gh`'s state array, when it is a sync VM generator.
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
    let sh = state_of(ctx, gh).ok_or(VmError::NotAnObject)?;
    resume_state(ctx, funcs, sh, mode, v)
}

/// [`resume`] of the generator whose state array is `sh`.
fn resume_state(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    sh: Handle,
    mode: u8,
    v: NanBox,
) -> Result<GenOut, VmError> {
    let (id, state, pc, dst, regs_h, hs_h) = {
        let e = ctx.realm.array_elements(sh).unwrap_or(&[]);
        let num = |i: usize| e.get(i).and_then(|x| x.as_number()).unwrap_or(0.0);
        let h = |i: usize| e.get(i).and_then(|x| x.as_handle()).map(Handle::from_raw);
        (
            num(0) as usize,
            num(1),
            num(2) as usize,
            num(3) as Reg,
            h(4),
            h(5),
        )
    };
    if state == GEN_EXECUTING {
        let e = vm_error(ctx, "TypeError", "Generator is already running");
        return Err(VmError::Thrown(e));
    }
    // Suspended inside a `yield*` (the delegation step) or at a capturing
    // `YieldResume`: every resumption kind lands in registers.
    let delegate = (state == GEN_SUSPENDED_YIELD)
        .then(|| funcs.get(id))
        .flatten()
        .and_then(|p| {
            if let Some(Op::YieldDelegate { dst, mode, .. }) = p.ops.get(pc) {
                return Some((*dst, *mode));
            }
            match pc.checked_sub(1).and_then(|i| p.ops.get(i)) {
                Some(Op::YieldResume { dst, mode, .. }) => Some((*dst, *mode)),
                _ => None,
            }
        });
    let handlers: Vec<(usize, Reg)> = match hs_h.and_then(|h| ctx.realm.array_elements(h)) {
        Some(hs) if !hs.is_empty() => hs
            .chunks(2)
            .map(|c| {
                (
                    c[0].as_number().unwrap_or(0.0) as usize,
                    c.get(1).and_then(|x| x.as_number()).unwrap_or(0.0) as Reg,
                )
            })
            .collect(),
        _ => Vec::new(),
    };
    let had_handlers = handlers.len();
    // A `return(v)` at a `yield` inside a try / for-of / destructuring region
    // must run that region's cleanup (iterator closes) — the interpreter's.
    if delegate.is_none() && mode == 2 && state == GEN_SUSPENDED_YIELD && !handlers.is_empty() {
        return Err(VmError::Unsupported);
    }
    if delegate.is_none()
        && (state == GEN_COMPLETED || (state == GEN_SUSPENDED_START && mode != 0) || mode == 2)
    {
        ctx.realm.internal_array_set_number(sh, 1, GEN_COMPLETED);
        return match mode {
            1 => Err(VmError::Thrown(v)),
            2 => Ok(GenOut::Done(v)),
            _ => Ok(GenOut::Done(NanBox::undefined())),
        };
    }
    let Some(proto) = funcs.get(id) else {
        return Err(VmError::Unsupported);
    };
    // The register file moves out of its array for the run (and back in at
    // the next `yield`); a completed generator's stays empty.
    let mut regs = regs_h
        .and_then(|h| ctx.realm.internal_array_take(h))
        .unwrap_or_default();
    let inject = if let Some((ddst, dmode)) = delegate {
        regs[ddst as usize] = v;
        regs[dmode as usize] = NanBox::number(f64::from(mode));
        None
    } else {
        if mode == 0
            && state == GEN_SUSPENDED_YIELD
            && let Some(slot) = regs.get_mut(dst as usize)
        {
            *slot = v;
        }
        (mode == 1).then_some(v)
    };
    ctx.realm.internal_array_set_number(sh, 1, GEN_EXECUTING);
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
            if let Some(rh) = regs_h {
                ctx.realm.internal_array_put(rh, regs);
            }
            // (Most bodies suspend outside any handler region, as they
            // resumed: then the handler array stays as it is, empty.)
            if let Some(hh) = hs_h
                && !(handlers.is_empty() && had_handlers == 0)
            {
                ctx.realm.internal_array_set_numbers(
                    hh,
                    handlers
                        .iter()
                        .flat_map(|(t, reg)| [*t as f64, f64::from(*reg)]),
                );
            }
            ctx.realm.internal_array_set_numbers_from(
                sh,
                1,
                &[GEN_SUSPENDED_YIELD, pc as f64, f64::from(dst)],
            );
            // A `yield*` step hands the inner result object through as is.
            if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                Ok(GenOut::Delegated(value))
            } else {
                Ok(GenOut::Yield(value))
            }
        }
        Ok(FrameExit::Return(ret)) => {
            ctx.realm.internal_array_set_number(sh, 1, GEN_COMPLETED);
            Ok(GenOut::Done(ret.unwrap_or(NanBox::undefined())))
        }
        Ok(FrameExit::Tail { .. }) => {
            ctx.realm.internal_array_set_number(sh, 1, GEN_COMPLETED);
            Err(VmError::Unsupported)
        }
        Err(err) => {
            ctx.realm.internal_array_set_number(sh, 1, GEN_COMPLETED);
            Err(err)
        }
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

/// The state array of the generator object `g` when the direct path may
/// resume it: a hosted, single-realm run, and `g` a sync VM generator of a
/// function in `funcs`.
fn direct_target(ctx: &mut Ctx, funcs: &[FnProto], g: NanBox) -> Option<Handle> {
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let sh = state_of(ctx, Handle::from_raw(g.as_handle()?))?;
    let id = ctx.realm.array_elements(sh)?.first()?.as_number()? as usize;
    (id < funcs.len()).then_some(sh)
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
