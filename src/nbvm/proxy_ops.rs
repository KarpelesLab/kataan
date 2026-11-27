//! A proxy's `[[Get]]`, `[[Set]]` and `[[HasProperty]]` run in the VM.
//!
//! Every property operation on a Proxy used to cross to the host, which
//! re-resolved the key by name, looked the trap up with a full `[[Get]]` on
//! the handler, and called the trap back into the VM through its audited
//! callback machinery: thousands of instructions before the trap body ran.
//!
//! Here the operation is answered in the VM when each step it takes before
//! calling code is one that runs no code (so declining at any of them, and
//! handing the whole operation to the host, is unobservable):
//!
//! - the proxy is live (a revoked one throws: the host's);
//! - the trap lookup `GetMethod(handler, name)` finds a data property (or
//!   nothing) over an ordinary handler chain with no accessor
//!   ([`Realm::ordinary_chain_get`]);
//! - a trap found is a VM function that is not a class constructor (it is
//!   called here, with the handler as `this`, exactly as the host calls it);
//!   no trap forwards to the target: another proxy repeats the steps, an
//!   ordinary target answers from its own chain walk;
//! - a trap is called only when the target is not itself a proxy, so the
//!   invariant checks after the call read the target's own property without
//!   running code. They are the host's checks, over the same realm queries,
//!   with the same messages.
//!
//! The values the operation holds across the trap call (target, handler,
//! key, receiver) are listed in the host root set for the call's duration.

use super::{Ctx, FnProto, VmError, with_host};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::{ChainHit, Realm};

/// What `GetMethod(handler, name)` finds, decided without running code.
enum Trap {
    /// `undefined` / `null` / no property: forward to the target.
    Absent,
    /// A VM function (not a class constructor) to call.
    Fn(NanBox),
}

/// Whether the VM proxy operations are on (`KATAAN_PROXY_VM=0`: off).
fn enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_PROXY_VM").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// The trap `name` of `handler`, when finding it runs no code; `None` when
/// unknown (an accessor, a proxy or an exotic object on the handler chain,
/// or a callable this module does not call itself).
fn trap_quiet(realm: &Realm, funcs: &[FnProto], handler: Handle, name: &str) -> Option<Trap> {
    match realm.ordinary_chain_get(handler, name)?.0 {
        ChainHit::Absent => Some(Trap::Absent),
        ChainHit::Data(v) if v.is_undefined() || v.is_null() => Some(Trap::Absent),
        ChainHit::Data(v) => {
            let (id, _) = realm.vm_function(Handle::from_raw(v.as_handle()?))?;
            if funcs.get(id as usize).is_none_or(|p| p.class_ctor) {
                return None;
            }
            Some(Trap::Fn(v))
        }
        ChainHit::Accessor(..) => None,
    }
}

/// The `(target, handler)` of the live proxy `h` (`None`: not a proxy, or
/// revoked).
fn live_proxy(realm: &Realm, h: Handle) -> Option<(Handle, Handle)> {
    let th = realm.proxy_at(h)?;
    (!realm.proxy_revoked(h)).then_some(th)
}

/// A property key for a proxy operation: its storage form and its value
/// (what a trap receives), for a primitive key whose `ToPropertyKey` runs no
/// code. `None` for an object key (or an internal one).
pub(super) fn elem_key(realm: &mut Realm, key: NanBox) -> Option<(alloc::string::String, NanBox)> {
    let ks = if let Some(raw) = key.as_handle() {
        let h = Handle::from_raw(raw);
        if let Some(id) = realm.symbol_id(h) {
            return Some((crate::shape::symbol_key(id), key));
        }
        realm.string_key(h)?
    } else {
        realm.display_key(key)
    };
    if crate::realm::is_internal_key(&ks) {
        return None;
    }
    let kv = if key.as_handle().is_some() {
        key
    } else {
        realm.key_value(&ks)
    };
    Some((ks, kv))
}

/// Calls trap `f` with `this` = the handler, rooting `held` for the call.
fn call_trap(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: Option<&[NanBox]>,
    held: &[NanBox],
    f: NanBox,
    this: NanBox,
    argv: &[NanBox],
) -> Result<NanBox, VmError> {
    let mark = ctx.realm.host_roots.vals.len();
    ctx.realm.host_roots.vals.extend_from_slice(held);
    let r = super::call_closure_compiled(ctx, funcs, regs, f, argv, this);
    ctx.realm.host_roots.vals.truncate(mark);
    r
}

/// A `TypeError` with `msg`, made by the host as its own errors are.
fn type_error(ctx: &mut Ctx, msg: &str) -> VmError {
    match with_host(ctx, |h| h.make_error("TypeError", msg)) {
        Some(e) => VmError::Thrown(e),
        None => VmError::Unsupported,
    }
}

fn hv(h: Handle) -> NanBox {
    NanBox::handle(h.to_raw())
}

/// The storage key `ks` is one these operations take (not internal: a
/// private name, a hidden slot).
fn key_ok(ks: &str) -> bool {
    !crate::realm::is_internal_key(ks) || ks.starts_with(crate::shape::SYMBOL_KEY_PREFIX)
}

/// The proxy `[[Get]]` success invariants (10.5.8 steps 9-10): the host's
/// `proxy_get_invariant_check`.
fn get_violation(
    realm: &Realm,
    target: Handle,
    name: &str,
    result: NanBox,
) -> Option<&'static str> {
    if let Some((getter, _)) = realm.accessor(target, name) {
        if realm.property_is_non_configurable(target, name)
            && getter.is_undefined()
            && !result.is_undefined()
        {
            return Some(
                "proxy 'get' returned a value for a non-configurable accessor with no getter",
            );
        }
    } else if realm.has_own(target, name)
        && realm.property_is_non_configurable(target, name)
        && realm.property_is_readonly(target, name)
    {
        let actual = realm
            .get_property(target, name)
            .unwrap_or(NanBox::undefined());
        if !realm.strict_equals(result, actual) {
            return Some(
                "proxy 'get' returned a different value for a non-configurable non-writable property",
            );
        }
    }
    None
}

/// The proxy `[[Set]]` success invariants (10.5.9 step 10): the host's
/// `proxy_set_invariant_check`.
fn set_violation(realm: &Realm, target: Handle, key: &str, value: NanBox) -> Option<&'static str> {
    if let Some((_, s)) = realm.accessor(target, key) {
        if realm.property_is_non_configurable(target, key) && s.is_undefined() {
            return Some(
                "proxy 'set' trap succeeded for a non-configurable accessor with no setter",
            );
        }
        return None;
    }
    if realm.has_own(target, key)
        && realm.property_is_non_configurable(target, key)
        && realm.property_is_readonly(target, key)
    {
        let cur = realm
            .get_property(target, key)
            .unwrap_or(NanBox::undefined());
        if !realm.strict_equals(cur, value) {
            return Some(
                "proxy 'set' trap succeeded for a non-configurable non-writable property with a different value",
            );
        }
    }
    None
}

/// The proxy `[[HasProperty]]` invariants for a `false` result (10.5.7
/// step 9): the host's checks in `has_property_proxied`.
fn has_false_violation(realm: &Realm, target: Handle, key: &str) -> Option<&'static str> {
    let target_has = realm.has_own(target, key)
        || realm.accessor(target, key).is_some()
        || (key == "length" && realm.is_array(target))
        || realm
            .array_length(target)
            .is_some_and(|len| key.parse::<usize>().is_ok_and(|i| i < len));
    if !target_has {
        return None;
    }
    let nonconfigurable = if key == "length" && realm.is_array(target) {
        true
    } else if realm.accessor(target, key).is_some() || realm.has_own(target, key) {
        realm.property_is_non_configurable(target, key)
    } else {
        false
    };
    if nonconfigurable {
        return Some(
            "proxy 'has' trap returned false for a non-configurable property of the target",
        );
    }
    if !realm.is_extensible(target) {
        return Some("proxy 'has' trap returned false for a property of a non-extensible target");
    }
    None
}

/// `proxy.[[Get]](key, proxy)` for the proxy `recv` (see the module docs).
/// `kv`: the key's value when the caller has it. `None`: the host's.
pub(super) fn get(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: Option<&[NanBox]>,
    recv: NanBox,
    ks: &str,
    kv: Option<NanBox>,
) -> Option<Result<NanBox, VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm || !key_ok(ks) || !enabled() {
        return None;
    }
    let mut obj = Handle::from_raw(recv.as_handle()?);
    for _ in 0..32 {
        let (target, handler) = live_proxy(&ctx.realm, obj)?;
        match trap_quiet(&ctx.realm, funcs, handler, "get")? {
            Trap::Absent => {
                if ctx.realm.proxy_at(target).is_some() {
                    obj = target;
                    continue;
                }
                // (A data property or nothing: the Receiver plays no part.)
                return match ctx.realm.ordinary_chain_get(target, ks)?.0 {
                    ChainHit::Data(v) => Some(Ok(v)),
                    ChainHit::Absent => Some(Ok(NanBox::undefined())),
                    ChainHit::Accessor(..) => None,
                };
            }
            Trap::Fn(f) => {
                if ctx.realm.proxy_at(target).is_some() {
                    return None;
                }
                let key = kv.unwrap_or_else(|| ctx.realm.key_value(ks));
                let (t, hd) = (hv(target), hv(handler));
                let r = call_trap(
                    ctx,
                    funcs,
                    regs,
                    &[t, hd, key, recv],
                    f,
                    hd,
                    &[t, key, recv],
                );
                let v = match r {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                if let Some(msg) = get_violation(&ctx.realm, target, ks, v) {
                    return Some(Err(type_error(ctx, msg)));
                }
                return Some(Ok(v));
            }
        }
    }
    None
}

/// `proxy.[[Set]](key, value, proxy)` for the proxy `recv` from an
/// assignment (`strict`: a failed set throws). `None`: the host's.
#[allow(clippy::too_many_arguments)]
pub(super) fn set(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: Option<&[NanBox]>,
    recv: NanBox,
    ks: &str,
    kv: Option<NanBox>,
    value: NanBox,
    strict: bool,
) -> Option<Result<(), VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm || !key_ok(ks) || !enabled() {
        return None;
    }
    let recv_h = Handle::from_raw(recv.as_handle()?);
    let mut obj = recv_h;
    for _ in 0..32 {
        let (target, handler) = live_proxy(&ctx.realm, obj)?;
        match trap_quiet(&ctx.realm, funcs, handler, "set")? {
            Trap::Absent => {
                if ctx.realm.proxy_at(target).is_some() {
                    obj = target;
                    continue;
                }
                // OrdinarySet(target, key, value, Receiver = the proxy): an
                // own data property of the target, or the key on no object
                // of its chain, makes the write the Receiver's define.
                return match ctx.realm.ordinary_chain_get(target, ks)?.0 {
                    ChainHit::Data(_) if ctx.realm.has_own(target, ks) => {
                        ctx.realm.proxy_receiver_define_quiet(recv_h, ks, value)?;
                        Some(Ok(()))
                    }
                    ChainHit::Absent => {
                        ctx.realm.proxy_receiver_define_quiet(recv_h, ks, value)?;
                        Some(Ok(()))
                    }
                    _ => None,
                };
            }
            Trap::Fn(f) => {
                if ctx.realm.proxy_at(target).is_some() {
                    return None;
                }
                let key = kv.unwrap_or_else(|| ctx.realm.key_value(ks));
                let (t, hd) = (hv(target), hv(handler));
                let r = call_trap(
                    ctx,
                    funcs,
                    regs,
                    &[t, hd, key, recv, value],
                    f,
                    hd,
                    &[t, key, value, recv],
                );
                let r = match r {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };
                if !ctx.realm.truthy(r) {
                    if strict {
                        let msg = alloc::format!(
                            "'set' on proxy: trap returned falsish for property '{ks}'"
                        );
                        return Some(Err(type_error(ctx, &msg)));
                    }
                    return Some(Ok(()));
                }
                if let Some(msg) = set_violation(&ctx.realm, target, ks, value) {
                    return Some(Err(type_error(ctx, msg)));
                }
                return Some(Ok(()));
            }
        }
    }
    None
}

/// `proxy.[[HasProperty]](key)` for the proxy `obj` (`key in obj`).
/// `None`: the host's.
pub(super) fn has(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: Option<&[NanBox]>,
    obj: NanBox,
    key: NanBox,
) -> Option<Result<bool, VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm || !enabled() {
        return None;
    }
    let mut cur = Handle::from_raw(obj.as_handle()?);
    ctx.realm.proxy_at(cur)?;
    let (ks, kv) = elem_key(&mut ctx.realm, key)?;
    for _ in 0..32 {
        let (target, handler) = live_proxy(&ctx.realm, cur)?;
        match trap_quiet(&ctx.realm, funcs, handler, "has")? {
            Trap::Absent => {
                if ctx.realm.proxy_at(target).is_some() {
                    cur = target;
                    continue;
                }
                return match ctx.realm.ordinary_chain_get(target, &ks)?.0 {
                    ChainHit::Absent => Some(Ok(false)),
                    _ => Some(Ok(true)),
                };
            }
            Trap::Fn(f) => {
                if ctx.realm.proxy_at(target).is_some() {
                    return None;
                }
                let (t, hd) = (hv(target), hv(handler));
                let r = call_trap(ctx, funcs, regs, &[t, hd, kv, obj], f, hd, &[t, kv]);
                let r = match r {
                    Ok(v) => ctx.realm.truthy(v),
                    Err(e) => return Some(Err(e)),
                };
                if !r && let Some(msg) = has_false_violation(&ctx.realm, target, &ks) {
                    return Some(Err(type_error(ctx, msg)));
                }
                return Some(Ok(r));
            }
        }
    }
    None
}
