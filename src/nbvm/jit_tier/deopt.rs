//! The VM side of the generic tier's deoptimization exits (see
//! `crate::jit::deopt` for the design): the helper an exit calls, which
//! runs what is left of the activation in the interpreter.

use super::super::{Ctx, FnProto, FrameExit, NanBox, Reg, VmError, run_frame_at, with_host};
use super::stack_of;
use crate::heap::Handle;
use crate::jit::DeoptInfo;
use crate::jit::sink::{MatObj, SinkExits};

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
    let level = ((at >> 32) & 0xff) as usize;
    let after = (at >> crate::jit::DEOPT_AFTER_SHIFT) & 1 != 0;
    // The body's sunk objects (`crate::jit::sink`): made first, as the
    // interpreter's frame would have them here.
    if let Some(sx) = info.sink.as_deref() {
        let forced = after
            || sx.plan.exit_before.get(pc) == Some(&true)
            || sx.plan.vget.get(pc) == Some(&true);
        refresh_memos(ctx, sx, regs, pc, after);
        if !after {
            for &(f, recv, site) in &sx.fills {
                if f == pc
                    && let Some(h) = regs.get(recv as usize).and_then(|v| v.as_handle())
                {
                    // SAFETY: the site is owned with the code that called
                    // here; only Rust code (this thread's) writes it.
                    #[allow(unsafe_code)]
                    unsafe { &mut *site }.fill(&mut ctx.realm, Handle::from_raw(h));
                }
            }
        }
        materialize(ctx, sx, regs, sx.plan.mats_at(pc, after));
        if forced {
            let n = sx.forced.get() + 1;
            sx.forced.set(n);
            if super::jit_log_enabled() && n.is_power_of_two() {
                std::eprintln!("[jit] sink forced exit #{n} at body pc {pc} (after: {after})");
            }
            if n == crate::jit::sink::exit_limit() {
                retire_sinking(ctx, funcs, info);
            }
        }
    }
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
    let mut own = info.own_pc(at_pc).or_else(|| {
        // (An own-level fallback call op that left before it ran: its own
        // call op.)
        (level == 0)
            .then(|| info.sink.as_deref().and_then(|x| x.own_call(at_pc)))
            .flatten()
    });
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
        // A proper tail call, with no trampoline below this native
        // activation: the ordinary call (its callee's own tail calls go to
        // the trampoline `call_with` enters it through).
        Some(Ok(FrameExit::Tail {
            id,
            args,
            captures,
            this,
        })) => match super::super::call_with(ctx, funcs, id, &args, &captures, this) {
            Ok(v) => v.to_bits(),
            Err(e) => throw(ctx, e),
        },
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

/// Writes every object of `objs` (innermost first) into the registers that
/// hold its stand-in: an ordinary object whose `[[Prototype]]` is the
/// stand-in (the site's prototype, in its `preg`) with its stored fields as
/// own data properties, in order — what the constructor's plain adds made.
fn materialize(ctx: &mut Ctx, sx: &SinkExits, regs: &mut [NanBox], objs: &[MatObj]) {
    for m in objs {
        let Some(site) = sx.plan.sites.get(m.site as usize) else {
            continue;
        };
        // (The stand-in's own `[[Prototype]]`: the site's prototype when
        // the object was made.)
        let Some(proto) = regs
            .get(site.preg as usize)
            .and_then(|v| v.as_handle())
            .and_then(|h| ctx.realm.object_proto(Handle::from_raw(h)))
        else {
            continue;
        };
        let h = ctx.realm.new_object_with_proto(Some(proto));
        for (key, &f) in site.keys.iter().zip(&site.fregs).take(m.n as usize) {
            let v = regs.get(f as usize).copied().unwrap_or(NanBox::undefined());
            ctx.realm.set_property(h, key, v);
        }
        let v = NanBox::handle(h.to_raw());
        for &r in &m.regs {
            if let Some(x) = regs.get_mut(r as usize) {
                *x = v;
            }
        }
    }
}

/// Fills the guard memos of `sx`'s sites that are stale, from their
/// construct records (`crate::jit::ConstructFast`): when a record holds at
/// the current epoch, its prototype's chain is checked (see
/// [`chain_plain`]). An exit at (or just after) a site's own fallback
/// `Construct` first fills that record, as the compiled call sequence
/// would on its first runs (which never come while the site is sunk).
fn refresh_memos(ctx: &mut Ctx, sx: &SinkExits, regs: &[NanBox], pc: usize, after: bool) {
    let epoch = crate::object::proto_epoch();
    for (k, site) in sx.plan.sites.iter().enumerate() {
        let (Some(memo), Some(&call)) = (sx.memos.get(k), sx.calls.get(k)) else {
            continue;
        };
        if memo.stamp.get() == epoch || call.is_null() {
            continue;
        }
        let at_fallback = if after {
            pc == site.fallback + 1
        } else {
            pc == site.fallback
        };
        // SAFETY: the call site is owned with the code that called here, and
        // only Rust code (this thread's, now) touches it.
        #[allow(unsafe_code)]
        let cs = unsafe { &mut *call.cast_mut() };
        if at_fallback && let Some(ctor) = regs.get(site.ctor as usize).copied() {
            let ctx_raw = core::ptr::from_mut(ctx).cast::<core::ffi::c_void>();
            super::fill_construct_site(ctx, ctor, cs);
            let mut out = 0u64;
            let _ = super::jit_construct_info(
                ctx_raw,
                ctor.to_bits(),
                ctor.to_bits(),
                &raw mut out,
                core::ptr::from_ref(cs),
            );
        }
        let f = &cs.fast;
        if f.ctor == crate::jit::CallSite::EMPTY
            || f.derived != 0
            || f.stamp != crate::object::proto_epoch()
            || f.proto == 0
        {
            continue;
        }
        let proto = Handle::from_raw(f.proto);
        if chain_plain(ctx, proto, &site.keys) {
            // The stand-in: the layout the site's objects have.
            let t = ctx.realm.new_object_with_proto(Some(proto));
            for key in &site.keys {
                ctx.realm.set_property(t, key, NanBox::number(0.0));
            }
            ctx.realm
                .fill_lit_word(&sx.lits, sx.lit_base + k, NanBox::handle(t.to_raw()));
            memo.proto.set(f.proto);
            memo.stamp.set(crate::object::proto_epoch());
        }
    }
}

/// Whether adding each of `keys` to a fresh ordinary object whose
/// `[[Prototype]]` is `proto` is a plain add: every object on the chain an
/// ordinary, non-exotic one with no accessor and no non-writable property
/// of those keys. The chain's objects are watched, so a change to any of
/// them retires the answer with the prototype epoch.
fn chain_plain(ctx: &mut Ctx, proto: Handle, keys: &[alloc::string::String]) -> bool {
    let mut cur = Some(proto);
    let mut depth = 0;
    let mut seen: alloc::vec::Vec<Handle> = alloc::vec::Vec::new();
    while let Some(h) = cur {
        depth += 1;
        if depth > 64 {
            return false;
        }
        let Some(o) = ctx.realm.jit_object(h) else {
            return false;
        };
        if o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0 {
            return false;
        }
        for k in keys {
            if o.accessor(k).is_some() || (o.has_own_key(k) && (o.is_readonly(k) || o.is_frozen()))
            {
                return false;
            }
        }
        seen.push(h);
        cur = o.proto();
    }
    for h in seen {
        ctx.realm.watch_link(h.to_raw(), None);
    }
    true
}

/// A body whose sunk objects' exits keep happening: recompiled now, without
/// sinking (`crate::jit::JitProto::spec_off` says so once the count is
/// reached), so the activations to come run compiled code that leaves no
/// more. (This activation goes on in the interpreter; the old code stays
/// alive with the function.)
fn retire_sinking(ctx: &mut Ctx, funcs: &[FnProto], info: &DeoptInfo) {
    let _ = with_function(ctx, funcs, info, |_, _, _| ());
    let id = info.func.get() as usize;
    let Some(proto) = funcs.get(id) else {
        return;
    };
    let Some(Some(old)) = proto.tier.jit() else {
        return;
    };
    if !old.owns_deopt_info(core::ptr::from_ref(info)) {
        return;
    }
    let done = super::recompile(ctx, funcs, id, &old, &[]).is_some();
    if super::jit_log_enabled() {
        std::eprintln!(
            "[jit] fn {}: its sunk objects' exits keep happening: {}",
            proto.name,
            if done {
                "recompiled without sinking"
            } else {
                "not recompiled"
            }
        );
    }
}

/// `Op::GetMethod`'s slow path on a sunk object's stand-in
/// (`crate::jit::sink`: the prototype the object would have, whose own
/// properties are the object's inherited ones): the property, through the
/// ordinary slow path (which fills the site's caches), when the chain from
/// the stand-in holds it as a plain data property (or not at all), with
/// nothing but ordinary objects on the way; else the exit sentinel (a
/// getter would see the stand-in as its receiver, a proxy its trap), and
/// the op leaves for the interpreter.
///
/// # Safety
/// As `jit_helper_get_method_site`.
pub(crate) extern "C" fn jit_helper_get_virtual(
    ctx_raw: *mut core::ffi::c_void,
    obj: u64,
    site: *mut crate::jit::MethodSite,
    call: *const crate::jit::CallSite,
) -> u64 {
    {
        // SAFETY: as `jit_helper_get_method_site`.
        #[allow(unsafe_code)]
        let (ctx, site_ref) = unsafe { (&*(ctx_raw as *mut Ctx), &*site) };
        let key = site_ref.key();
        let mut cur = NanBox::from_bits(obj).as_handle().map(Handle::from_raw);
        let mut depth = 0;
        while let Some(h) = cur {
            depth += 1;
            let Some(o) = ctx.realm.jit_object(h) else {
                return NanBox::jit_exit_bits();
            };
            if depth > 64 || o.jit_flags() & crate::object::JIT_FLAG_EXOTIC != 0 {
                return NanBox::jit_exit_bits();
            }
            if o.accessor(key).is_some() {
                return NanBox::jit_exit_bits();
            }
            if o.has_own_key(key) {
                break;
            }
            cur = o.proto();
        }
        if cur.is_none() && depth == 0 {
            return NanBox::jit_exit_bits();
        }
    }
    super::jit_helper_get_method_site(ctx_raw, obj, site, call)
}
