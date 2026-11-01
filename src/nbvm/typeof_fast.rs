//! `typeof name` for a global-environment name, without the host.
//!
//! `Op::TypeofGlobal` (a `typeof` of a name no enclosing function binds) used
//! to ask the host every time, which swaps the realm in and out and resolves
//! the name from scratch. When the running scope is the global scope, the
//! answer is one of three things the VM can read itself: the value of an
//! initialized global binding (found through a slot cache), the value of an
//! own **data** property of the global object, or `"undefined"` for a name
//! neither provides (an unresolvable reference, which `typeof` tolerates).
//! Anything else (a module name, the temporal dead zone, an accessor, an
//! inherited property, an exotic object on the global object's chain, an
//! inner scope running) is the host's.
//!
//! Absence is remembered per site: the global binding epoch
//! (`crate::env`) moves whenever the global scope gains, loses or
//! re-initializes a binding and whenever the global object's own keys
//! change, so "not bound in the global scope" (and "not own on the global
//! object") hold while it is unchanged; "not inherited either" is checked
//! against the recorded handles and key epochs of the global object's
//! prototype chain.

use super::Ctx;
use crate::env::SlotCache;
use crate::heap::Handle;
use core::cell::Cell;

/// How many prototype-chain links an absence proof records.
const LINKS: usize = 4;

/// `Op::TypeofGlobal`'s inline cache (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct TypeofGlobalCache {
    /// The global-scope slot of the name, when it binds it.
    slot: SlotCache,
    /// The global binding epoch at which the name was seen unbound in the
    /// global scope (`0`: never).
    unbound_at: Cell<u64>,
    /// The global object (raw handle) that proof was made against.
    global: Cell<u64>,
    /// With `unbound_at`: the name is not own on the global object either,
    /// nor inherited through the recorded chain `links[..n_links]`.
    absent: Cell<bool>,
    n_links: Cell<u8>,
    links: [Cell<(u64, u32)>; LINKS],
}

/// `typeof name` decided without the host (see the module docs); `None`
/// when the host must decide.
pub(super) fn type_of(ctx: &Ctx, name: &str, cache: &TypeofGlobalCache) -> Option<&'static str> {
    if name.starts_with('\u{1}') {
        return None;
    }
    let (scope, global) = ctx.host.as_deref()?.global_object_record()?;
    let g = Handle::from_raw(global.as_handle()?);
    let epoch = crate::env::global_binding_epoch();
    let proven = cache.unbound_at.get() == epoch && cache.global.get() == g.to_raw();
    if !proven {
        if let Some(v) = scope.cached_get_local(name, &cache.slot) {
            return (!v.is_tdz()).then(|| ctx.realm.type_of_value(v));
        }
        if scope.get(name).is_some() {
            return None;
        }
    }
    if proven && cache.absent.get() && chain_unchanged(ctx, g, cache) {
        return Some("undefined");
    }
    if let Some(v) = ctx.realm.own_data_named(g, name) {
        if !proven {
            cache.unbound_at.set(epoch);
            cache.global.set(g.to_raw());
            cache.absent.set(false);
        }
        return Some(ctx.realm.type_of_value(v));
    }
    if ctx.realm.has_property_named(g, name) != Some(false) {
        return None;
    }
    // Unresolvable: record the proof.
    let mut n = 0;
    let mut cur = ctx.realm.ordinary_link(g)?.1;
    while let Some(h) = cur {
        let (epoch_h, next) = ctx.realm.ordinary_link(h)?;
        if n == LINKS {
            return Some("undefined");
        }
        cache.links[n].set((h.to_raw(), epoch_h));
        n += 1;
        cur = next;
    }
    cache.unbound_at.set(epoch);
    cache.global.set(g.to_raw());
    cache.n_links.set(n as u8);
    cache.absent.set(true);
    Some("undefined")
}

/// Whether the global object's prototype chain is still the recorded one,
/// link by link (handle and key epoch), ending where it ended. (A recorded
/// link was an ordinary one; with the same key epoch it still has the same
/// keys, so only its handle, epoch and `[[Prototype]]` are compared.)
fn chain_unchanged(ctx: &Ctx, g: Handle, cache: &TypeofGlobalCache) -> bool {
    let Some((_, mut cur)) = ctx.realm.link_epoch(g) else {
        return false;
    };
    for link in &cache.links[..usize::from(cache.n_links.get())] {
        let (raw, epoch) = link.get();
        let Some(h) = cur.filter(|h| h.to_raw() == raw) else {
            return false;
        };
        match ctx.realm.link_epoch(h) {
            Some((e, next)) if e == epoch => cur = next,
            _ => return false,
        }
    }
    cur.is_none()
}
