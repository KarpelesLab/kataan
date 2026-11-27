//! `Op::SuperGet` / `Op::SuperSet` without the host, shared by the
//! interpreter loop and the compiled tier's helper (`jit_helper_exec_op`) so
//! the two cannot diverge.
//!
//! `super.k` is `home.[[Prototype]].[[Get]](k, this)`: over a chain of
//! ordinary shaped objects (the usual class prototypes) the site's chain
//! cache (`Realm::chain_cached_get`, validated against the live chain on
//! every hit) answers it exactly — a data property's value, `undefined` for
//! a key no object has, or an accessor whose getter is then called with
//! `this` as its receiver. `super.k = v` is served only when the chain's
//! first holder of `k` is an accessor with a setter (called with `this`):
//! every other case (a data property, whose write lands on `this` with its
//! own checks; no holder; a setter-less accessor) is the host's. Anything
//! else — a proxy or exotic object on the chain, a symbol or index key, a
//! static method's base (a function) — is a miss, for the caller's general
//! path, which keeps every error and ordering.

use super::{Ctx, FnProto, VmError, call_closure_compiled, private_ops::KeyText, super_base};
use crate::heap::Handle;
use crate::ic::PropertyCache;
use crate::nanbox::NanBox;
use crate::realm::ChainHit;
use alloc::boxed::Box;

/// One `super` property site's chain cache. Empty when cold, and when
/// copied (a fresh copy of the bytecode starts cold).
#[derive(Default)]
pub struct SuperCache(core::cell::Cell<Option<Box<PropertyCache>>>);

impl Clone for SuperCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl core::fmt::Debug for SuperCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SuperCache")
    }
}

/// The holder of `key` on `base`'s chain, through `cache`: `None` when the
/// site cannot serve it (see the module docs), having armed the cache for
/// the next time when the chain allows it.
fn lookup(ctx: &Ctx, base: Handle, key: &str, cache: &SuperCache) -> Option<ChainHit> {
    if key.is_empty()
        || key.starts_with('\0')
        || key.as_bytes()[0].is_ascii_digit()
        || ctx.realm.is_host_exotic(base)
        || ctx.realm.is_vm_function(base)
    {
        return None;
    }
    let mut c = cache
        .0
        .take()
        .unwrap_or_else(|| Box::new(PropertyCache::new()));
    let hit = ctx.realm.chain_cached_get(base, key, &c);
    if hit.is_none() {
        ctx.realm.chain_cache_fill(base, key, &mut c);
    }
    cache.0.set(Some(c));
    hit
}

/// The base and key text of a `super` reference (`None`: not a string key
/// on an object base).
fn parts(ctx: &mut Ctx, home: NanBox, key: NanBox) -> Option<Result<(Handle, KeyText), VmError>> {
    let base = match super_base(ctx, home) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let b = Handle::from_raw(base.as_handle()?);
    let kh = Handle::from_raw(key.as_handle()?);
    ctx.realm.string_leaf_bytes(kh)?;
    Some(Ok((b, KeyText::of(&ctx.realm, key))))
}

/// `super[key]` read with receiver `this` (`None`: take the general path,
/// nothing having run).
pub(super) fn get(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    (home, key, this): (NanBox, NanBox, NanBox),
    cache: &SuperCache,
) -> Option<Result<NanBox, VmError>> {
    let (base, k) = match parts(ctx, home, key)? {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };
    match lookup(ctx, base, k.as_str(), cache)? {
        ChainHit::Data(v) => Some(Ok(v)),
        ChainHit::Absent => Some(Ok(NanBox::undefined())),
        ChainHit::Accessor(getter, _, _) if getter.as_handle().is_some() => Some(
            call_closure_compiled(ctx, funcs, Some(regs), getter, &[], this),
        ),
        // An inherited getter-less accessor reads `undefined`.
        ChainHit::Accessor(_, _, false) => Some(Ok(NanBox::undefined())),
        ChainHit::Accessor(..) => None,
    }
}

/// `super[key] = v` with receiver `this` when its holder is an accessor
/// with a setter (`None`: take the general path, nothing having run).
pub(super) fn set(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    (home, key, v, this): (NanBox, NanBox, NanBox, NanBox),
    cache: &SuperCache,
) -> Option<Result<(), VmError>> {
    let (base, k) = match parts(ctx, home, key)? {
        Ok(p) => p,
        Err(e) => return Some(Err(e)),
    };
    match lookup(ctx, base, k.as_str(), cache)? {
        ChainHit::Accessor(_, setter, _) if setter.as_handle().is_some() => {
            Some(call_closure_compiled(ctx, funcs, Some(regs), setter, &[v], this).map(|_| ()))
        }
        _ => None,
    }
}
