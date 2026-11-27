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

use super::{Ctx, FnProto, VmError, call_closure_compiled, super_base};
use crate::heap::Handle;
use crate::ic::PropertyCache;
use crate::nanbox::NanBox;
use crate::realm::ChainHit;
use alloc::boxed::Box;

/// One `super` property site's chain cache. Empty when cold, and when
/// copied (a fresh copy of the bytecode starts cold).
#[derive(Default)]
pub struct SuperCache(core::cell::Cell<Option<Box<SiteMemo>>>);

/// A [`SuperCache`]'s contents: the chain cache, and the key the site last
/// looked up (its value's bits, which name one string cell for as long as
/// they are a live handle, a recycled slot carrying another generation),
/// with its text, so a static key's text is not copied and checked on
/// every access.
struct SiteMemo {
    chain: PropertyCache,
    key_bits: u64,
    key: Box<str>,
    /// The accessor the site last called: the base's bits, the prototype
    /// epoch it was found at with every link up to its holder watched
    /// (`Realm::watch_accessor_chain`; `0`: none), the key's bits, and the
    /// function. While the epoch stands, the same base and key find it
    /// again without a walk.
    acc_base: u64,
    acc_stamp: u64,
    acc_key: u64,
    acc_fn: NanBox,
}

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

/// The holder of `key`'s text on `base`'s chain, through `cache`: `None`
/// when the site cannot serve it (see the module docs), having armed the
/// cache for the next time when the chain allows it.
fn lookup(ctx: &Ctx, base: Handle, key: NanBox, cache: &SuperCache) -> Option<ChainHit> {
    let kh = Handle::from_raw(key.as_handle()?);
    if ctx.realm.is_host_exotic(base) || ctx.realm.is_vm_function(base) {
        return None;
    }
    let mut m = cache.0.take().unwrap_or_else(|| {
        Box::new(SiteMemo {
            chain: PropertyCache::new(),
            key_bits: 0,
            key: Box::from(""),
            acc_base: 0,
            acc_stamp: 0,
            acc_key: 0,
            acc_fn: NanBox::undefined(),
        })
    });
    if m.key_bits != key.to_bits() {
        let Some(text) = ctx
            .realm
            .string_leaf_bytes(kh)
            .and_then(|b| core::str::from_utf8(b).ok())
        else {
            cache.0.set(Some(m));
            return None;
        };
        m.key = Box::from(text);
        m.key_bits = key.to_bits();
    }
    let k: &str = &m.key;
    if k.is_empty() || k.starts_with('\0') || k.as_bytes()[0].is_ascii_digit() {
        cache.0.set(Some(m));
        return None;
    }
    let hit = ctx.realm.chain_cached_get(base, k, &m.chain);
    if hit.is_none() {
        let SiteMemo { chain, key, .. } = &mut *m;
        ctx.realm.chain_cache_fill(base, key, chain);
    }
    cache.0.set(Some(m));
    hit
}

/// The accessor function the site's memo holds for `base` and `key` at the
/// current prototype epoch, if any (see [`SiteMemo`]).
fn memo_accessor(base: Handle, key: NanBox, cache: &SuperCache) -> Option<NanBox> {
    let m = cache.0.take()?;
    let hit = (m.acc_stamp == crate::object::proto_epoch()
        && m.acc_base == base.to_raw()
        && m.acc_key == key.to_bits())
    .then_some(m.acc_fn);
    cache.0.set(Some(m));
    hit
}

/// Records `f`, the accessor function the `[[Get]]`/`[[Set]]` of `key` from
/// `base` just found, in the site's memo when the chain up to its holder can
/// be watched.
fn memo_record(ctx: &mut Ctx, base: Handle, key: NanBox, f: NanBox, cache: &SuperCache) {
    let Some(mut m) = cache.0.take() else {
        return;
    };
    if m.key_bits == key.to_bits() && ctx.realm.watch_accessor_chain(base, &m.key) {
        m.acc_base = base.to_raw();
        m.acc_key = key.to_bits();
        m.acc_fn = f;
        m.acc_stamp = crate::object::proto_epoch();
    } else {
        m.acc_stamp = 0;
    }
    cache.0.set(Some(m));
}

/// The base of a `super` reference (`None`: not an object base).
fn base_of(ctx: &mut Ctx, home: NanBox) -> Option<Result<Handle, VmError>> {
    let base = match super_base(ctx, home) {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(Handle::from_raw(base.as_handle()?)))
}

/// `super[key] = v` of a captured base (`Op::SuperSetBase`), as [`set`].
pub(super) fn set_base(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    (base, key, v, this): (NanBox, NanBox, NanBox, NanBox),
    cache: &SuperCache,
) -> Option<Result<(), VmError>> {
    let base = Handle::from_raw(base.as_handle()?);
    if let Some(setter) = memo_accessor(base, key, cache) {
        return Some(call_closure_compiled(ctx, funcs, Some(regs), setter, &[v], this).map(|_| ()));
    }
    match lookup(ctx, base, key, cache)? {
        ChainHit::Accessor(_, setter, _) if setter.as_handle().is_some() => {
            memo_record(ctx, base, key, setter, cache);
            Some(call_closure_compiled(ctx, funcs, Some(regs), setter, &[v], this).map(|_| ()))
        }
        _ => None,
    }
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
    let base = match base_of(ctx, home)? {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    if let Some(getter) = memo_accessor(base, key, cache) {
        return Some(call_closure_compiled(
            ctx,
            funcs,
            Some(regs),
            getter,
            &[],
            this,
        ));
    }
    match lookup(ctx, base, key, cache)? {
        ChainHit::Data(v) => Some(Ok(v)),
        ChainHit::Absent => Some(Ok(NanBox::undefined())),
        ChainHit::Accessor(getter, _, _) if getter.as_handle().is_some() => {
            memo_record(ctx, base, key, getter, cache);
            Some(call_closure_compiled(
                ctx,
                funcs,
                Some(regs),
                getter,
                &[],
                this,
            ))
        }
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
    let base = match base_of(ctx, home)? {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    match lookup(ctx, base, key, cache)? {
        ChainHit::Accessor(_, setter, _) if setter.as_handle().is_some() => {
            Some(call_closure_compiled(ctx, funcs, Some(regs), setter, &[v], this).map(|_| ()))
        }
        _ => None,
    }
}
