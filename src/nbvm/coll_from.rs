//! `new Set(array)` of the default `Set` constructor without the host.
//!
//! The Set constructor (with an iterable) is: OrdinaryCreateFromConstructor
//! (the default constructor's non-writable `prototype`), `adder =
//! Get(set, "add")`, then GetIterator(iterable) and a call of `adder` per
//! value. When the constructor is the single realm's default `Set` (with
//! `new.target` itself), `adder` resolves along ordinary data properties to
//! the intrinsic `Set.prototype.add`, and the iterable is a plain array
//! whose iteration is the built-in one with no hole
//! (`iter_fast::plain_array_values`: only own elements read, no code run),
//! nothing of that is observable but the result: a fresh Set holding the
//! values in order (each added as the intrinsic `add` adds it). Anything else
//! is the general path's.

use super::{Ctx, FnProto, VmError, with_host_regs};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::coll_fast::CollOp;

/// `new ctor(...args)` as described in the module docs, or `None` (nothing
/// observable done) when it does not apply.
pub(super) fn new_set_from_array(ctx: &mut Ctx, ctor: NanBox, args: &[NanBox]) -> Option<NanBox> {
    let [items] = *args else {
        return None;
    };
    if !ctx.realm.is_default_set_ctor(ctor) {
        return None;
    }
    let vals =
        super::ArgBuf::from_slice(super::iter_fast::plain_array_values(&mut ctx.realm, items)?);
    let set = ctx.realm.new_default_collection(ctor, &[])?;
    let sh = Handle::from_raw(set.as_handle()?);
    // (The fresh Set owns no property: the lookup starts at its prototype.)
    let proto = ctx.realm.object_proto(sh)?;
    let add = super::array_hof::chain_data(&ctx.realm, proto, "add").ok()??;
    if !super::native_call::is_set_add(&ctx.realm, add) {
        return None;
    }
    for &v in vals.iter() {
        ctx.realm
            .coll_fast(sh, CollOp::SetAdd, v, NanBox::undefined())?;
    }
    Some(set)
}

/// `new Set(gen)` of the default `Set` constructor over a VM generator whose
/// `GetIterator` runs no code (`gen_fast::open`: its `@@iterator` returns it
/// and its `next` is the intrinsic): the Set made as the host makes it, the
/// intrinsic `add` (looked up as in [`new_set_from_array`]) applied to each
/// value the generator yields, resumed here (`gen_fast::for_of_step`). A
/// throw from the generator propagates (IteratorStep's abrupt completion
/// closes nothing). `None`, having done nothing observable, when this does
/// not apply.
pub(super) fn new_set_from_gen(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    ctor: NanBox,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    let [items] = *args else {
        return None;
    };
    if !ctx.realm.is_default_set_ctor(ctor) {
        return None;
    }
    let next = super::gen_fast::open(ctx, funcs, items)?;
    let set = ctx.realm.new_default_collection(ctor, &[])?;
    let sh = Handle::from_raw(set.as_handle()?);
    let proto = ctx.realm.object_proto(sh)?;
    let add = super::array_hof::chain_data(&ctx.realm, proto, "add").ok()??;
    if !super::native_call::is_set_add(&ctx.realm, add) {
        return None;
    }
    // (The Set is held across the generator's resumptions.)
    let mark = ctx.realm.host_roots.vals.len();
    ctx.realm.host_roots.vals.push(set);
    let r = loop {
        let step = match super::gen_fast::for_of_step(ctx, funcs, regs, items) {
            Some(r) => r,
            None => match with_host_regs(ctx, regs, |h| h.iter_step(items, next)) {
                Some(r) => r.map_err(VmError::from),
                None => Err(VmError::Unsupported),
            },
        };
        match step {
            Ok(Some(v)) => {
                if ctx
                    .realm
                    .coll_fast(sh, CollOp::SetAdd, v, NanBox::undefined())
                    .is_none()
                {
                    break Err(VmError::Unsupported);
                }
            }
            Ok(None) => break Ok(set),
            Err(e) => break Err(e),
        }
    };
    ctx.realm.host_roots.vals.truncate(mark);
    Some(r)
}
