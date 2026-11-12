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
#[inline]
fn state_of(ctx: &mut Ctx, gh: Handle) -> Option<Handle> {
    ctx.realm.generator_frame(gh, VM_GEN, &mut ctx.gen_ic)
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
    let at = FrameAt::read(ctx, fh).ok_or(VmError::NotAnObject)?;
    resume_state_at(ctx, funcs, fh, at, mode, v)
}

/// What a resumption reads of a generator's frame cell before it runs.
#[derive(Clone, Copy)]
struct FrameAt {
    id: usize,
    state: u8,
    pc: usize,
    dst: u16,
    has_handlers: bool,
}

impl FrameAt {
    #[inline]
    fn read(ctx: &Ctx, fh: Handle) -> Option<Self> {
        let f = ctx.realm.vm_frame(fh)?;
        Some(FrameAt {
            id: f.func as usize,
            state: f.state,
            pc: f.pc as usize,
            dst: f.dst,
            has_handlers: !f.handlers.is_empty(),
        })
    }
}

/// [`resume_state`] with the frame cell `fh`'s fields already read (`at`).
fn resume_state_at(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    fh: Handle,
    at: FrameAt,
    mode: u8,
    v: NanBox,
) -> Result<GenOut, VmError> {
    let FrameAt {
        id,
        state,
        pc,
        dst,
        has_handlers,
    } = at;
    if state == S_EXECUTING {
        let e = vm_error(ctx, "TypeError", "Generator is already running");
        return Err(VmError::Thrown(e));
    }
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    if mode == 0
        && let Some(r) = hot_next(ctx, funcs, fh, id, state, pc, dst, v)
    {
        return r;
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
    finish(ctx, proto, fh, regs, r)
}

/// The end of a resumption that ran: the frame's new state (and, at a
/// `yield`, its register file and handlers back in it) and the outcome.
fn finish(
    ctx: &mut Ctx,
    proto: &FnProto,
    fh: Handle,
    regs: Vec<NanBox>,
    r: Result<FrameExit, VmError>,
) -> Result<GenOut, VmError> {
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

/// A `next(v)` resumption of a generator whose function's compiled body is
/// hot, straight into that body: the general path's decisions (see
/// [`resume_state`]) for the common states — suspended at its start, or at a
/// plain or capturing `yield` (not inside a `yield*` step) — without its
/// per-resumption bookkeeping (the inline caches stay with the function:
/// compiled code does not use them). `None` (nothing done) for anything
/// else, including a resumption due to recompile or exit-stress the body,
/// which the general path makes.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn hot_next(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    fh: Handle,
    id: usize,
    state: u8,
    pc: usize,
    dst: u16,
    v: NanBox,
) -> Option<Result<GenOut, VmError>> {
    let proto = funcs.get(id)?;
    // Where the sent value lands: a capturing `YieldResume`'s `dst` (and
    // its `mode` register, 0 = next), or the frame's `dst`.
    let land = match state {
        S_START => None,
        S_YIELD => {
            if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                return None;
            }
            match pc.checked_sub(1).and_then(|i| proto.ops.get(i)) {
                Some(Op::YieldResume { dst, mode, .. }) => Some((*dst, Some(*mode))),
                _ => Some((dst, None)),
            }
        }
        _ => return None,
    };
    let jit = {
        let mut t = proto.tier.0.try_borrow_mut().ok()?;
        let n = t.resumes.saturating_add(1);
        if !super::jit_tier::resume_is_hot(n) || super::jit_tier::resume_may_recompile(n) {
            return None;
        }
        // (`run_generator_body` refuses a body without an entry at `pc`.)
        let j = match &t.jit {
            Some(Some(j)) if j.is_generic() => alloc::rc::Rc::clone(j),
            _ => return None,
        };
        t.resumes = n;
        j
    };
    Some(run_in_cell(ctx, funcs, proto, &jit, fh, pc, land, v))
}

/// Where a `next(v)` resumption's sent value lands: a capturing
/// `YieldResume`'s `dst` and its `mode` register (set to 0 = next), or the
/// frame's `dst`; `None` at the generator's start.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
type Land = Option<(u16, Option<u16>)>;

/// The run of a hot generator's compiled body for a `next(v)` resumption
/// ([`hot_next`], [`site_next_hot`]), its register file left in the frame
/// cell `fh`: sized there to the body's register count (the registers past
/// the function's own, its inlined callees', start undefined, as on a
/// window), the sent value landed (`land`), and run in place. While it runs
/// the cell is on `Realm::vm_gen_running`, which roots the registers and
/// keeps the cell remembered; a resumption of the running generator throws
/// before touching them. At a `yield` they stay there (truncated back to
/// the function's own); any other end moves them out for [`finish`] (or,
/// for a body that refuses to run, the interpreter).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_in_cell(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    proto: &FnProto,
    jit: &crate::jit::JitProto,
    fh: Handle,
    pc: usize,
    land: Land,
    v: NanBox,
) -> Result<GenOut, VmError> {
    use super::jit_tier::GenRun;
    let own = proto.n_regs;
    let n = jit.n_regs().max(own);
    let (ptr, len, mut handlers) = match ctx.realm.vm_frame_mut(fh) {
        Some(f) if f.regs.len() == own => {
            f.state = S_EXECUTING;
            f.regs.resize(n, NanBox::undefined());
            match land {
                Some((d, Some(m))) => {
                    f.regs[d as usize] = v;
                    f.regs[m as usize] = NanBox::number(0.0);
                }
                Some((d, None)) => {
                    if let Some(slot) = f.regs.get_mut(d as usize) {
                        *slot = v;
                    }
                }
                None => {}
            }
            (
                f.regs.as_mut_ptr(),
                f.regs.len(),
                core::mem::take(&mut f.handlers),
            )
        }
        Some(_) => return run_on_window(ctx, funcs, proto, jit, fh, pc, land, v),
        None => return Err(VmError::NotAnObject),
    };
    ctx.realm.vm_gen_running.push(fh);
    // SAFETY: the buffer belongs to the frame cell's register file, which
    // nothing resizes, moves or frees while the generator runs (the cell is
    // reachable from the generator its resumer published; a resumption of a
    // running generator throws before touching it; a collection only reads
    // it).
    #[allow(unsafe_code)]
    let regs = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    let run =
        super::jit_tier::run_generator_body_in_place(ctx, funcs, jit, regs, pc, &mut handlers);
    ctx.realm.vm_gen_running.pop();
    if let GenRun::Yield { value, pc, dst } = run {
        if let Some(f) = ctx.realm.vm_frame_mut(fh) {
            f.regs.truncate(own);
            f.handlers = handlers;
            f.pc = pc as u32;
            f.dst = dst;
            f.state = S_YIELD;
        }
        ctx.realm.vm_frame_stored(fh);
        // A `yield*` step hands the inner result object through as is.
        return Ok(
            if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                GenOut::Delegated(value)
            } else {
                GenOut::Yield(value)
            },
        );
    }
    let mut regs = match ctx.realm.vm_frame_mut(fh) {
        Some(f) => core::mem::take(&mut f.regs),
        None => return Err(VmError::NotAnObject),
    };
    regs.truncate(own);
    let r = match run {
        GenRun::Exit(r) => r,
        // (A body that refuses to run — no entry at `pc` — is interpreted,
        // cold.)
        _ => run_frame_at_ic(
            ctx,
            funcs,
            &proto.ops,
            &mut regs,
            pc,
            handlers,
            None,
            &mut Vec::new(),
        ),
    };
    finish(ctx, proto, fh, regs, r)
}

/// [`run_in_cell`] for a register file not of the function's own size:
/// moved out of the cell and run on a window of the register stack.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[cold]
fn run_on_window(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    proto: &FnProto,
    jit: &crate::jit::JitProto,
    fh: Handle,
    pc: usize,
    land: Land,
    v: NanBox,
) -> Result<GenOut, VmError> {
    use super::jit_tier::GenRun;
    let (mut regs, mut handlers) = match ctx.realm.vm_frame_mut(fh) {
        Some(f) => {
            f.state = S_EXECUTING;
            (
                core::mem::take(&mut f.regs),
                core::mem::take(&mut f.handlers),
            )
        }
        None => return Err(VmError::NotAnObject),
    };
    match land {
        Some((d, Some(m))) => {
            regs[d as usize] = v;
            regs[m as usize] = NanBox::number(0.0);
        }
        Some((d, None)) => {
            if let Some(slot) = regs.get_mut(d as usize) {
                *slot = v;
            }
        }
        None => {}
    }
    let r = match super::jit_tier::run_generator_body(ctx, funcs, jit, &mut regs, pc, &mut handlers)
    {
        GenRun::Yield { value, pc, dst } => {
            if let Some(f) = ctx.realm.vm_frame_mut(fh) {
                f.regs = regs;
                f.handlers = handlers;
                f.pc = pc as u32;
                f.dst = dst;
                f.state = S_YIELD;
            }
            ctx.realm.vm_frame_stored(fh);
            return Ok(
                if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                    GenOut::Delegated(value)
                } else {
                    GenOut::Yield(value)
                },
            );
        }
        GenRun::Exit(r) => r,
        GenRun::Refused => run_frame_at_ic(
            ctx,
            funcs,
            &proto.ops,
            &mut regs,
            pc,
            handlers,
            None,
            &mut Vec::new(),
        ),
    };
    finish(ctx, proto, fh, regs, r)
}

/// The compiled tier's `for-of` step (`jit_tier::jit_helper_iter_next`)
/// over a sync VM generator whose `next` is the intrinsic and whose body
/// runs compiled: [`for_of_step`]'s outcome, written to the loop's `done`
/// register as `vm_iter_step` writes it, with the checks and the resume
/// folded into one pass. `None` (nothing done) for anything else, which
/// takes the general step.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(super) fn jit_for_of_step(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &mut [NanBox],
    done: super::Reg,
    iter: super::Reg,
    next: super::Reg,
) -> Option<Result<Option<NanBox>, VmError>> {
    let gh = Handle::from_raw(regs[iter as usize].as_handle()?);
    let nx = Handle::from_raw(regs[next as usize].as_handle()?);
    if regs[done as usize].as_boolean() == Some(true)
        || ctx.host.is_none()
        || ctx.realm.multi_realm
        || ctx.realm.native_at(nx) != Some(crate::nbexec::N_GEN_NEXT)
    {
        return None;
    }
    let fh = state_of(ctx, gh)?;
    let (id, state, pc, dst) = {
        let f = ctx.realm.vm_frame(fh)?;
        (f.func as usize, f.state, f.pc as usize, f.dst)
    };
    // (Past the recursion limits, the general step throws.)
    if id >= funcs.len()
        || ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        || crate::stack::exhausted()
    {
        return None;
    }
    let mark = publish(ctx, regs, regs[iter as usize], NanBox::undefined());
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let r = hot_next(ctx, funcs, fh, id, state, pc, dst, NanBox::undefined());
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    unpublish(ctx, mark);
    let out = match r? {
        Ok(o) => o,
        Err(e) => {
            regs[done as usize] = NanBox::boolean(true);
            return Some(Err(e));
        }
    };
    let step = match out {
        GenOut::Yield(v) => Ok(Some(v)),
        GenOut::Done(_) => Ok(None),
        GenOut::Delegated(r) => (|| {
            let d = host_get_str(ctx, r, "done", None)?;
            if ctx.realm.truthy(d) {
                return Ok(None);
            }
            host_get_str(ctx, r, "value", None).map(Some)
        })(),
    };
    regs[done as usize] = NanBox::boolean(!matches!(step, Ok(Some(_))));
    Some(step)
}

/// `GetIterator(g)` for a sync VM generator `g` of `funcs` when it runs no
/// code: `g[@@iterator]` is a data property (own or inherited, no accessor
/// or proxy on the way) holding `%IteratorPrototype%[@@iterator]`, which
/// returns its receiver, and `g.next` likewise is the intrinsic
/// `%GeneratorPrototype%.next`. `Some(next)` — the iterator is `g` itself —
/// or `None` (nothing done) for the general protocol.
///
/// A generator owning nothing but its state slot defers both lookups to its
/// prototype, whose chain's outcome is memoized ([`OpenMemo`]).
pub(super) fn open(ctx: &mut Ctx, funcs: &[FnProto], g: NanBox) -> Option<NanBox> {
    direct_target(ctx, funcs, g)?;
    let gh = Handle::from_raw(g.as_handle()?);
    let Some(proto) = ctx.realm.generator_owns_only_state(gh) else {
        let realm = &ctx.realm;
        let key = realm.vm_array_iter().iter_key.as_deref()?;
        let f = super::iter_fast::plain_lookup(realm, gh, key)??;
        let n = super::iter_fast::plain_lookup(realm, gh, "next")??;
        return open_fns_ok(ctx, f, n);
    };
    let epoch = crate::object::proto_epoch();
    if let Some(m) = &ctx.gen_open_memo
        && m.proto == proto.to_raw()
        && m.stamp == epoch
    {
        let f = ctx.realm.stamped_slot(&m.iter_at)?;
        let n = ctx.realm.stamped_slot(&m.next_at)?;
        return open_fns_ok(ctx, f, n);
    }
    ctx.gen_open_memo = None;
    let key: alloc::boxed::Box<str> = ctx.realm.vm_array_iter().iter_key.clone()?;
    let iter_at = ctx.realm.watch_get_chain(proto, &key)??;
    let next_at = ctx.realm.watch_get_chain(proto, "next")??;
    let f = ctx.realm.stamped_slot(&iter_at)?;
    let n = ctx.realm.stamped_slot(&next_at)?;
    ctx.gen_open_memo = Some(OpenMemo {
        proto: proto.to_raw(),
        stamp: crate::object::proto_epoch(),
        iter_at,
        next_at,
    });
    open_fns_ok(ctx, f, n)
}

/// [`open`]'s test of what the lookups found: `f` (`@@iterator`) returns
/// its receiver and `n` is the intrinsic `next`.
fn open_fns_ok(ctx: &Ctx, f: NanBox, n: NanBox) -> Option<NanBox> {
    let realm = &ctx.realm;
    (realm.native_at(Handle::from_raw(f.as_handle()?))
        == Some(crate::nbexec::N_ITERATOR_PROTO_SELF)
        && realm.native_at(Handle::from_raw(n.as_handle()?)) == Some(crate::nbexec::N_GEN_NEXT))
    .then_some(n)
}

/// [`open`]'s memo: where the chain from the generators' prototype `proto`
/// holds `@@iterator` and `next`, found at prototype epoch `stamp`. While
/// the epoch stands, no object on either (watched) chain changed its keys,
/// layout or `[[Prototype]]`, so both are still data properties at those
/// slots: only their values are re-read.
pub(super) struct OpenMemo {
    proto: u64,
    stamp: u64,
    iter_at: crate::realm::builtins_fast::StampedSlot,
    next_at: crate::realm::builtins_fast::StampedSlot,
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
/// them), and with the calling frame published for a collection inside the
/// resumed body (see [`publish`]). (The host's path never collects: its
/// context forbids it.)
fn nested<T>(
    ctx: &mut Ctx,
    regs: &[NanBox],
    g: NanBox,
    v: NanBox,
    f: impl FnOnce(&mut Ctx) -> Result<T, VmError>,
) -> Result<T, VmError> {
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        || crate::stack::exhausted()
    {
        let e = vm_error(ctx, "RangeError", STACK_EXHAUSTED);
        return Err(VmError::Thrown(e));
    }
    let mark = publish(ctx, regs, g, v);
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let r = f(ctx);
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    unpublish(ctx, mark);
    r
}

/// Publishes the frame that resumes the generator `g` — its registers
/// `regs`, the generator and the sent value `v` — as an audited call does
/// (`Ctx::frame_shadow`), so the safepoints of the resumed body may collect:
/// that body runs on a register file moved out of the generator's frame
/// cell, which is then this activation's own (rooted by the safepoint that
/// runs on it), and nothing else of the resume lives in a Rust local the
/// collector cannot see. A window on the JIT register stack is rooted
/// already. Returns the mark [`unpublish`] restores.
#[inline]
pub(super) fn publish(ctx: &mut Ctx, regs: &[NanBox], g: NanBox, v: NanBox) -> usize {
    let mark = ctx.frame_shadow.len();
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    let rooted = super::jit_tier::on_register_stack(ctx, regs);
    #[cfg(not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")))]
    let rooted = false;
    if !rooted {
        ctx.frame_shadow.extend_from_slice(regs);
    }
    ctx.frame_shadow.extend_from_slice(&[g, v]);
    ctx.frames_published += 1;
    mark
}

/// Undoes [`publish`].
#[inline]
pub(super) fn unpublish(ctx: &mut Ctx, mark: usize) {
    ctx.frames_published -= 1;
    ctx.frame_shadow.truncate(mark);
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

/// The method names of the resumption modes.
const MODE_KEYS: [&str; 3] = ["next", "throw", "return"];

/// [`site_call`]'s memo: for a few generator prototypes (a call site such as
/// a scheduler's `gen.next(v)` resumes generators of several functions, each
/// with its own prototype), where the `[[Get]]` of a resumption method from
/// that prototype finds its data property, validated at prototype epoch
/// `stamp`. While the epoch stands, no object on the (watched) chain changed
/// its keys, layout or `[[Prototype]]`, so the property is still at that
/// slot: only its value is re-read.
#[derive(Default)]
pub(super) struct SiteMemo {
    entries: [Option<SiteEntry>; 4],
    /// The entry the next fill replaces.
    next: usize,
}

struct SiteEntry {
    proto: u64,
    mode: u8,
    stamp: u64,
    at: crate::realm::builtins_fast::StampedSlot,
}

/// `g.next(v)` / `g.throw(v)` / `g.return(v)` (`mode` 0 / 1 / 2) at a method
/// call site `g[key](v)` whose memo says it called the intrinsic `method`
/// before, proved here without the site's general receiver walk: `g` is a
/// generator object owning nothing but its state slot, so `g[key]` is the
/// `[[Get]]` of `key` from its prototype, which [`SiteMemo`] answers. The
/// iterator result, or `None` (nothing done) for the site's general path.
#[allow(clippy::too_many_arguments)]
pub(super) fn site_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    key: &str,
    mode: u8,
    method: u64,
    v: NanBox,
) -> Option<Result<NanBox, VmError>> {
    if MODE_KEYS.get(usize::from(mode)) != Some(&key) || ctx.host.is_none() || ctx.realm.multi_realm
    {
        return None;
    }
    let gh = Handle::from_raw(recv.as_handle()?);
    // (A generator owning nothing but its state slot: `g[key]` is the
    // `[[Get]]` from its prototype, and the slot names its frame cell.)
    let (proto, fh) = ctx
        .realm
        .generator_proto_and_frame(gh, VM_GEN, &mut ctx.gen_ic)?;
    let raw = proto.to_raw();
    let epoch = crate::object::proto_epoch();
    let m = &ctx.gen_site_memo;
    let mut hit = None;
    for e in m.entries.iter().flatten() {
        if e.proto == raw && e.mode == mode && e.stamp == epoch {
            hit = Some(&e.at);
            break;
        }
    }
    let found = match hit {
        Some(at) => ctx.realm.stamped_slot(at)?,
        None => {
            let at = ctx.realm.watch_get_chain(proto, key)??;
            let found = ctx.realm.stamped_slot(&at)?;
            let m = &mut ctx.gen_site_memo;
            let i = m.next;
            m.next = (i + 1) % m.entries.len();
            m.entries[i] = Some(SiteEntry {
                proto: raw,
                mode,
                stamp: crate::object::proto_epoch(),
                at,
            });
            found
        }
    };
    if found.to_bits() != method {
        return None;
    }
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    if mode == 0
        && let Some(r) = site_next_hot(ctx, funcs, regs, recv, fh, v)
    {
        return Some(r);
    }
    let at = FrameAt::read(ctx, fh)?;
    if at.id >= funcs.len() {
        return None;
    }
    Some(
        nested(ctx, regs, recv, v, |ctx| {
            resume_state_at(ctx, funcs, fh, at, mode, v)
        })
        .map(|o| result_object(ctx, o)),
    )
}

/// [`site_call`]'s `g.next(v)` when `g`'s function's compiled body is hot
/// and `g` is suspended at its start or at a plain or capturing `yield`:
/// [`nested`] + [`resume_state_at`] + [`hot_next`] + [`result_object`] in one
/// pass over the frame cell, the iterator result made straight from the
/// yielded value. `None` (nothing done) for anything else, including a
/// resumption past the recursion limits (the general path throws).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[inline]
fn site_next_hot(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    caller: &[NanBox],
    g: NanBox,
    fh: Handle,
    v: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let (id, state, pc, dst) = {
        let f = ctx.realm.vm_frame(fh)?;
        (f.func as usize, f.state, f.pc as usize, f.dst)
    };
    let proto = funcs.get(id)?;
    // Where the sent value lands (as `hot_next`).
    let land = match state {
        S_START => None,
        S_YIELD => {
            if matches!(proto.ops.get(pc), Some(Op::YieldDelegate { .. })) {
                return None;
            }
            match pc.checked_sub(1).and_then(|i| proto.ops.get(i)) {
                Some(Op::YieldResume { dst, mode, .. }) => Some((*dst, Some(*mode))),
                _ => Some((dst, None)),
            }
        }
        _ => return None,
    };
    if ctx.call_depth >= ctx.realm.limits.max_call_depth
        || ctx.realm.vm_total_depth >= ctx.realm.limits.max_call_depth
        || crate::stack::exhausted()
    {
        return None;
    }
    let jit = {
        let mut t = proto.tier.0.try_borrow_mut().ok()?;
        let n = t.resumes.saturating_add(1);
        if !super::jit_tier::resume_is_hot(n) || super::jit_tier::resume_may_recompile(n) {
            return None;
        }
        let j = match &t.jit {
            Some(Some(j)) if j.is_generic() => alloc::rc::Rc::clone(j),
            _ => return None,
        };
        t.resumes = n;
        j
    };
    let mark = publish(ctx, caller, g, v);
    ctx.call_depth += 1;
    ctx.realm.vm_total_depth += 1;
    let out = run_in_cell(ctx, funcs, proto, &jit, fh, pc, land, v);
    ctx.call_depth -= 1;
    ctx.realm.vm_total_depth -= 1;
    unpublish(ctx, mark);
    Some(out.map(|o| result_object(ctx, o)))
}

/// `g.next(v)` / `g.throw(v)` / `g.return(v)` (`mode` 0 / 1 / 2) for a call
/// site that proved the method is the intrinsic: the iterator result, or
/// `None` (nothing done) to take the host's call.
pub(super) fn call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    g: NanBox,
    mode: u8,
    v: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let sh = direct_target(ctx, funcs, g)?;
    Some(
        nested(ctx, regs, g, v, |ctx| resume_state(ctx, funcs, sh, mode, v))
            .map(|o| result_object(ctx, o)),
    )
}

/// A `for-of` step over the generator `it` whose `next` is the intrinsic
/// `%GeneratorPrototype%.next`: the yielded value (`Some`), or `None` once
/// done; `None` from the outer `Option` (nothing done) to take the general
/// step. A `yield*` step's inner result object is read as IteratorStep
/// reads any result: `done`, then `value`, through `[[Get]]`.
pub(super) fn for_of_step(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    it: NanBox,
) -> Option<Result<Option<NanBox>, VmError>> {
    let sh = direct_target(ctx, funcs, it)?;
    let u = NanBox::undefined();
    let out = match nested(ctx, regs, it, u, |ctx| resume_state(ctx, funcs, sh, 0, u)) {
        Ok(o) => o,
        Err(e) => return Some(Err(e)),
    };
    Some(match out {
        GenOut::Yield(v) => Ok(Some(v)),
        GenOut::Done(_) => Ok(None),
        GenOut::Delegated(r) => (|| {
            let done = host_get_str(ctx, r, "done", None)?;
            if ctx.realm.truthy(done) {
                return Ok(None);
            }
            host_get_str(ctx, r, "value", None).map(Some)
        })(),
    })
}
