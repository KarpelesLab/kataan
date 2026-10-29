//! `String.prototype.replace` / `replaceAll` with a RegExp and a VM function,
//! its replacer calls made inside the VM.
//!
//! `s.replace(re, fn)` used to be one host call whose `RegExp.prototype
//! [@@replace]` called `fn` once per match through the host: each call built
//! and dropped a fresh VM context over a swapped-in realm and took the
//! general call path. Here the host runs the algorithm only up to the end of
//! its match collection ([`VmHost::replace_plan`](super::VmHost::replace_plan)) — every observable step
//! before the first replacer call: the `@@replace` lookup, `flags`,
//! `lastIndex`, the `exec` calls, the legacy `RegExp` statics — and hands
//! back the matches when what remains is exactly "call `fn` per match,
//! `ToString` its result, splice". This module does that remainder (ECMA-262
//! 22.2.6.11 steps 14–15) in the calling VM context, so each replacer call is
//! an ordinary VM call entered as the array higher-order built-ins enter
//! theirs ([`super::array_hof::Callee`]).
//!
//! GC: the caller's registers, the replacer and the subject are published in
//! [`Ctx::frame_shadow`] while the loop runs; the accumulated text is Rust
//! bytes, and each call's argument strings are made just before the call
//! (allocation never collects; only a safepoint inside the callee does, and
//! by then the arguments are the callee's inputs).

use super::{Ctx, FnProto, HostError, Reg, VmError, vm_error, with_host};
use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use alloc::rc::Rc;
use alloc::vec::Vec;

/// What [`VmHost::replace_plan`](super::VmHost::replace_plan) did.
pub enum ReplaceStep {
    /// The whole call ran (a user `@@replace`, a user `exec`, named groups,
    /// the string algorithm…): its result.
    Done(NanBox),
    /// The matches of the intrinsic `@@replace`, whose replacer calls are
    /// left to the caller.
    Plan(ReplacePlan),
}

/// The built-in matches of a `RegExp.prototype[@@replace]` with a function
/// replacer, collected (in order) and not yet replaced.
pub struct ReplacePlan {
    /// `S`, the subject string (each call's last argument).
    pub(crate) subject: NanBox,
    /// `S`'s code units.
    pub(crate) units: Rc<Vec<u16>>,
    /// Spans per match: the whole match, then each capture group.
    pub(crate) stride: usize,
    /// Every match's spans, `stride` per match (code-unit offsets; `None` for
    /// a group that did not participate).
    pub(crate) spans: Vec<Option<(usize, usize)>>,
}

/// `recv.replace(regs[args[0]], regs[args[1]])` (`all`: `replaceAll`) once
/// the call site has proven the method is the intrinsic one: when `recv` is a
/// string, the search value a RegExp object and the replacement a VM
/// function, the call's result; `None` (nothing done) otherwise.
pub(super) fn try_run(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    args: &[Reg],
    all: bool,
) -> Option<Result<NanBox, VmError>> {
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    let (search, replace) = (arg(0), arg(1));
    if ctx.realm.multi_realm
        || !ctx
            .realm
            .is_string_handle(Handle::from_raw(recv.as_handle()?))
        || ctx
            .realm
            .regexp_flags(Handle::from_raw(search.as_handle()?))
            .is_none()
        || !ctx
            .realm
            .is_vm_function(Handle::from_raw(replace.as_handle()?))
    {
        return None;
    }
    let step = with_host(ctx, |h| h.replace_plan(recv, search, replace, all))??;
    Some(match step {
        Err(e) => Err(VmError::from(e)),
        Ok(ReplaceStep::Done(v)) => Ok(v),
        Ok(ReplaceStep::Plan(plan)) => {
            let mark = ctx.frame_shadow.len();
            ctx.frame_shadow.extend_from_slice(regs);
            ctx.frame_shadow
                .extend_from_slice(&[recv, search, replace, plan.subject]);
            ctx.frames_published += 1;
            let r = replace_matches(ctx, funcs, replace, &plan);
            ctx.frames_published -= 1;
            ctx.frame_shadow.truncate(mark);
            r
        }
    })
}

/// `RegExp.prototype[@@replace]` steps 14–15 over `plan` with the function
/// `replace`: per match, `Call(replace, undefined, «matched, ...captures,
/// position, S»)` and `ToString` of its result, spliced in when the match
/// starts at or past the text already consumed.
fn replace_matches(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    replace: NanBox,
    plan: &ReplacePlan,
) -> Result<NanBox, VmError> {
    let units: &[u16] = &plan.units;
    let length = units.len();
    let mut callee = super::array_hof::Callee::new(ctx, funcs, replace);
    let mut out: Vec<u8> = Vec::with_capacity(length);
    let mut next = 0usize;
    let mut argv: Vec<NanBox> = Vec::with_capacity(plan.stride + 2);
    for m in plan.spans.chunks(plan.stride.max(1)) {
        let Some(Some((start, end))) = m.first().copied() else {
            continue;
        };
        let pos = start.min(length);
        argv.clear();
        for g in m {
            argv.push(match *g {
                Some((a, b)) => {
                    let a = a.min(length);
                    let b = b.min(length).max(a);
                    NanBox::handle(ctx.realm.new_string_units(&units[a..b]).to_raw())
                }
                None => NanBox::undefined(),
            });
        }
        argv.push(NanBox::number(pos as f64));
        argv.push(plan.subject);
        let r = callee.invoke(ctx, &argv, NanBox::undefined())?;
        if pos >= next {
            crate::wtf8::append_utf16(&mut out, &units[next.min(pos)..pos]);
            append_to_string(ctx, &mut out, r)?;
            next = pos + (end - start);
            if out.len() > crate::rope::MAX_STRING_LEN {
                let e = vm_error(ctx, "RangeError", "Invalid string length");
                return Err(VmError::Thrown(e));
            }
        } else {
            // `ToString(replValue)` runs for every result.
            append_to_string(ctx, &mut Vec::new(), r)?;
        }
    }
    if next < length {
        crate::wtf8::append_utf16(&mut out, &units[next..]);
    }
    Ok(NanBox::handle(ctx.realm.new_string_wtf8(out).to_raw()))
}

/// Appends `ToString(v)`'s WTF-8 bytes to `out`: a string's own, a number's
/// or another primitive's directly, anything else through the host (an
/// object's `ToPrimitive` runs user code; a symbol throws).
fn append_to_string(ctx: &mut Ctx, out: &mut Vec<u8>, v: NanBox) -> Result<(), VmError> {
    match v.unpack() {
        Unpacked::Undefined => out.extend_from_slice(b"undefined"),
        Unpacked::Null => out.extend_from_slice(b"null"),
        Unpacked::Bool(b) => out.extend_from_slice(if b { b"true" } else { b"false" }),
        _ => {
            if let Some(n) = v.as_number() {
                out.extend_from_slice(crate::realm::js_number_string(n).as_bytes());
                return Ok(());
            }
            let s = match v.as_handle().map(Handle::from_raw) {
                Some(h) if ctx.realm.is_string_handle(h) => h,
                _ => {
                    let s = with_host(ctx, |h| h.to_string(v))
                        .unwrap_or(Err(HostError::Fault))
                        .map_err(VmError::from)?;
                    Handle::from_raw(s.as_handle().ok_or(VmError::Unsupported)?)
                }
            };
            ctx.realm.flatten_string(s);
            let bytes = ctx.realm.string_leaf_bytes(s).ok_or(VmError::Unsupported)?;
            crate::wtf8::append(out, bytes);
        }
    }
    Ok(())
}
