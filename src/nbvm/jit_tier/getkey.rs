//! Keyed reads `o[k]` that meet an accessor (`KATAAN_JIT_GETKEY`).
//!
//! The String-key memos answer only for receivers without accessors, so a
//! read of any key of an object that has one (`{ id, get price() {…} }`)
//! took the general element read, and a getter it found was called from
//! Rust. [`jit_helper_get_elem_acc`] first asks the text memo's
//! accessor-aware probe (`Realm::key_text_probe_acc`): an own data slot, a
//! key on no object of the chain, the receiver's own accessor of the key's
//! name, or an inherited accessor the memo recorded. A getter that is a VM
//! function goes back to compiled code (in its frame's method scratch slot), which calls it through the op's own
//! call site (`call_seq::emit_call`, `this` the receiver); when that
//! sequence cannot, [`jit_helper_get_elem_site`] fills the site and calls
//! the getter from here. Both probes are pure lookups, so asking twice runs
//! nothing twice.

use super::super::{Ctx, VmError, jit_get_elem_general, jit_lent};
use super::fill_call_site;
use crate::heap::Handle;
use crate::jit::CallSite;
use crate::nanbox::NanBox;
use crate::realm::AccHit;

/// What the accessor-aware probe decides for `recv[key]` alone: the value,
/// or a VM-function getter; `None` when the general read must run.
#[inline]
fn probe(ctx: &Ctx, recv: u64, key: u64) -> Option<Result<NanBox, NanBox>> {
    let (recv, key) = (NanBox::from_bits(recv), NanBox::from_bits(key));
    // A Symbol key naming an own data slot (`realm::sym_memo`).
    if let (Some(r), Some(k)) = (recv.as_handle(), key.as_handle())
        && let Some(id) = ctx.realm.symbol_id(Handle::from_raw(k))
        && let Some(v) = ctx.realm.sym_own_data(Handle::from_raw(r), key, id)
    {
        return Some(Ok(v));
    }
    if let Some((o, slot)) = ctx.realm.key_memo_probe(recv, key)
        && let Some(v) = o.slot(slot)
    {
        return Some(Ok(v));
    }
    match ctx.realm.key_text_probe_acc(recv, key)? {
        AccHit::Slot(o, slot) => o.slot(slot).map(Ok),
        AccHit::Absent => Some(Ok(NanBox::undefined())),
        AccHit::Getter(g) if g.is_undefined() => Some(Ok(g)),
        AccHit::Getter(g) => g
            .as_handle()
            .is_some_and(|h| ctx.realm.is_vm_function(Handle::from_raw(h)))
            .then_some(Err(g)),
    }
}

jit_extern! {
/// `recv[key]` for compiled code that can call a getter itself: `*out`
/// (the frame's method scratch slot) is set to the VM-function getter to
/// call with the receiver as `this`, or to `0` when the result is the
/// read's value (or the throw sentinel).
///
/// # Safety
/// As `jit_helper_get_elem`; `out` is a slot of the caller's native frame.
pub(crate) fn jit_helper_get_elem_acc(
    ctx_raw: *mut core::ffi::c_void,
    recv: u64,
    key: u64,
    out: *mut u64,
) -> u64 {
    let (v, g) = 'read: {
        // (A key that is no String, a Number most often: neither memo
        // knows it.)
        if NanBox::from_bits(key).as_handle().is_none() {
            break 'read (jit_get_elem_general(ctx_raw, recv, key), 0);
        }
        // SAFETY: as `jit_helper_get_elem`; this borrow ends before the
        // general read runs.
        #[allow(unsafe_code)]
        let ctx = unsafe { &*(ctx_raw as *const Ctx) };
        match probe(ctx, recv, key) {
            Some(Ok(v)) => break 'read (v.to_bits(), 0),
            Some(Err(g)) => break 'read (g.to_bits(), g.to_bits()),
            None => {}
        }
        (jit_get_elem_general(ctx_raw, recv, key), 0)
    };
    // SAFETY: the caller's frame slot (see the function contract).
    #[allow(unsafe_code)]
    unsafe {
        out.write(g);
    }
    v
}
}

jit_extern! {
/// `recv[key]` where compiled code's call of the getter
/// [`jit_helper_get_elem_acc`] found could not be made: the op's call site
/// is filled for that getter and the read made here, its getter called
/// with the receiver as `this`.
///
/// # Safety
/// As `jit_helper_get_elem`; `site` is the op's own call site, owned by
/// the calling `JitProto` and not otherwise borrowed while this runs.
pub(crate) fn jit_helper_get_elem_site(
    ctx_raw: *mut core::ffi::c_void,
    recv: u64,
    key: u64,
    site: *mut CallSite,
) -> u64 {
    // SAFETY: as `jit_helper_get_elem`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx_raw as *mut Ctx) };
    let g = match probe(ctx, recv, key) {
        Some(Ok(v)) => return v.to_bits(),
        Some(Err(g)) => g,
        None => return jit_get_elem_general(ctx_raw, recv, key),
    };
    // SAFETY: the site's own cache (see the function contract).
    #[allow(unsafe_code)]
    let site = unsafe { &mut *site };
    fill_call_site(ctx, g, site);
    // SAFETY: `jit_funcs` outlives this call (borrowed down the call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let lent = jit_lent(ctx);
    let r: Result<NanBox, VmError> =
        super::super::call_closure_pub(ctx, funcs, lent, g, &[], NanBox::from_bits(recv));
    match r {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}
}
