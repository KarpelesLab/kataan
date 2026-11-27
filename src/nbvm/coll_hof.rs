//! `Map.prototype.forEach` / `Set.prototype.forEach` run inside the VM.
//!
//! The host's `forEach` walked the entries and re-entered the VM for every
//! callback through a fresh context (a realm swap and the generic host call
//! path per entry). Here the walk runs in the calling VM context, each
//! callback an ordinary VM call ([`super::array_hof::Callee`], which takes the
//! compiled path when there is one).
//!
//! The walk is the live iterator's: a cursor state attached to the
//! collection (`Realm::collection_iter_state`, entries kind) stepped by
//! `Realm::collection_iter_advance_kv`, which skips deleted entries and
//! reaches entries added during the walk — exactly the `[[MapData]]` /
//! `[[SetData]]` list walk of 24.1.3.5 / 24.2.4.7, whose index-based steps
//! the iterator follows too.
//!
//! Only a VM-function callback on a (non-weak) collection of the method's
//! own kind in a single-realm run is taken; anything else (a native or bound
//! callback, a non-callable's TypeError, a wrong receiver) returns `None`
//! and the host's built-in runs as before.

use super::{Ctx, FnProto, VmError};
use crate::heap::Handle;
use crate::nanbox::NanBox;

/// `recv.forEach(callbackfn, thisArg)` for the Map (`set` false) or Set
/// (`set` true) `h`; `None` (nothing done) to take the host's call.
pub(super) fn for_each(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    h: Handle,
    set: bool,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let cb = args.first().copied()?;
    if ctx.realm.multi_realm
        || ctx.realm.collection_is_set(h) != Some(set)
        || ctx.realm.collection_is_weak(h)
        || !cb
            .as_handle()
            .is_some_and(|c| ctx.realm.is_vm_function(Handle::from_raw(c)))
    {
        return None;
    }
    let this_arg = args.get(1).copied().unwrap_or(NanBox::undefined());
    let state = ctx.realm.collection_iter_state(h, 2);
    // Publish the caller's registers (unless they are a GC root already)
    // and what the walk holds.
    let mark = ctx.frame_shadow.len();
    #[cfg(all(feature = "jit", jit_native))]
    let rooted = super::array_hof::lean_on() && super::jit_tier::on_register_stack(ctx, regs);
    #[cfg(not(all(feature = "jit", jit_native)))]
    let rooted = false;
    if !rooted {
        ctx.frame_shadow.extend_from_slice(regs);
    }
    ctx.frame_shadow
        .extend_from_slice(&[recv, cb, this_arg, NanBox::handle(state.to_raw())]);
    ctx.frames_published += 1;
    let mut callee = super::array_hof::Callee::new(ctx, funcs, cb);
    let r = loop {
        let Some((_, k, v, is_set)) = ctx.realm.collection_iter_advance_kv(state, h) else {
            // (The exhausted cursor has detached itself.)
            break Ok(NanBox::undefined());
        };
        // A Set's entries pass the element as both value and key.
        let val = if is_set { k } else { v };
        if let Err(e) = callee.invoke(ctx, &[val, k, recv], this_arg) {
            ctx.realm.collection_iter_detach(h);
            break Err(e);
        }
    };
    ctx.frames_published -= 1;
    ctx.frame_shadow.truncate(mark);
    Some(r)
}
