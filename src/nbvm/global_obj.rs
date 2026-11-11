//! Fast paths for the global environment's **object record**: a name the
//! global scope does not bind but the global object owns as a plain data
//! property — an implicit global (`n = 0` in sloppy code), a
//! `globalThis.x = …` / `this.x = …` property, or a host-installed one.
//!
//! `GetValue` of such a reference is `[[Get]]` on the global object and
//! `PutValue` (of an existing writable property) `[[Set]]` on it; for an own
//! data property of the ordinary global object neither runs any code, so the
//! interpreter reads or stores the property directly instead of going through
//! the host's full identifier resolution (a realm swap, a scope-chain walk,
//! `HasProperty`, then `[[Get]]`/`[[Set]]` by name: ~10k instructions).
//!
//! Compiled code goes further (`jit_helper_global_ref`): a read gets the
//! property value's address (`Realm::jit_data_ptr`), valid under the global
//! binding epoch exactly as a global `var`'s mirror target is
//! (`super::global_mirror`: a key added to or removed from the global object,
//! a new global-scope binding, or a change to whether a write to the property
//! stores all bump it). A write is the mirror write of a global `var` with no
//! binding: the validated "binding" address is the op's own write-only sink
//! (`MirrorCache::sink_ptr`), and the store reaches the property through the
//! mirror target — inline for a non-handle, through
//! `jit_helper_global_mirror` (the write barrier) for a heap handle.

use super::{Ctx, Handle, NanBox, StoreGlobalCache};
use crate::env::SlotCache;

/// The global object when a global reference of `name` that the global scope
/// does not bind resolves to the global object record (see
/// `VmHost::global_object_record`), and the global scope indeed lacks `name`.
#[inline]
fn record_for(ctx: &Ctx, name: &str, slot: &SlotCache) -> Option<Handle> {
    // A module-environment name (`split_module_name`) is never the global's.
    if name.starts_with('\u{1}') {
        return None;
    }
    let (scope, global) = ctx.host.as_deref()?.global_object_record()?;
    if scope.cached_has_local(name, slot) {
        return None;
    }
    global.as_handle().map(Handle::from_raw)
}

/// `GetValue` of the global reference `name` when the global scope does not
/// bind it and the global object owns it as a plain data property: its value.
/// `None` (nothing done) for anything else, which takes the host's general
/// path.
#[inline]
pub(super) fn load(ctx: &Ctx, name: &str, slot: &SlotCache) -> Option<NanBox> {
    let g = record_for(ctx, name, slot)?;
    ctx.realm.own_plain_data(g, name)
}

/// `PutValue` of the global reference `name` when the global scope does not
/// bind it and the global object owns it as a writable data property: stores
/// `v` there (with the write barrier). `false` (nothing done) for anything
/// else.
#[inline]
pub(super) fn store(ctx: &mut Ctx, name: &str, cache: &StoreGlobalCache, v: NanBox) -> bool {
    let Some(g) = record_for(ctx, name, &cache.slot) else {
        return false;
    };
    ctx.realm.set_own_plain_data(g, name, v)
}

/// `[[Set]]` of `recv[key] = v` when `recv` is the running realm's global
/// object and owns `key` as a writable data property: stores it there (with
/// the write barrier) and, as the host's `sync_global_object_write` does,
/// into a like-named global `var`/function binding (a global lexical is no
/// property of the global object, so it is left alone). `false` (nothing
/// done) for anything else.
pub(super) fn set_prop(ctx: &mut Ctx, recv: NanBox, key: &str, v: NanBox) -> bool {
    let Some((scope, global)) = ctx.host.as_deref().and_then(|h| h.global_env()) else {
        return false;
    };
    if recv != global {
        return false;
    }
    let Some(g) = global.as_handle().map(Handle::from_raw) else {
        return false;
    };
    if !ctx.realm.set_own_plain_data(g, key, v) {
        return false;
    }
    if scope.get(key).is_some() && !scope.has_lexical(key) {
        scope.set(key, v);
    }
    true
}

/// Whether `g` is a global object whose key changes the global binding epoch
/// tracks (`crate::object::JIT_FLAG_GLOBAL`).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
fn tracked(ctx: &Ctx, g: Handle) -> bool {
    ctx.realm
        .jit_object(g)
        .is_some_and(|o| o.jit_flags() & crate::object::JIT_FLAG_GLOBAL != 0)
}

/// `jit_helper_global_ref` of an `Op::LoadGlobal` of a name the global scope
/// does not bind: the address of the global object's own data property's
/// value (see the module docs).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(super) fn jit_load_ref(ctx: &mut Ctx, name: &str, slot: &SlotCache) -> Option<*mut NanBox> {
    let g = record_for(ctx, name, slot)?;
    if !tracked(ctx, g) {
        return None;
    }
    ctx.realm.jit_data_ptr(g, name)
}

/// `jit_helper_global_ref` of an `Op::StoreGlobal` of a name the global scope
/// does not bind, for an op whose compiled code mirrors the write: the op's
/// sink as its binding address, the property value's address left as its
/// mirror target (see the module docs).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(super) fn jit_store_ref(
    ctx: &mut Ctx,
    name: &str,
    cache: &StoreGlobalCache,
) -> Option<*mut NanBox> {
    if !cache.jit.on() {
        return None;
    }
    let g = record_for(ctx, name, &cache.slot)?;
    if !tracked(ctx, g) {
        return None;
    }
    let m = ctx.realm.jit_data_ptr(g, name)?;
    cache.jit.set(m as usize as u64);
    Some(cache.jit.sink_ptr())
}
