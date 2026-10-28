//! Built-in methods called straight from a method-call site.
//!
//! `recv.key(…)` on a hosted run normally goes to the host, which re-resolves
//! `key` by name and walks a long by-name dispatch cascade before it reaches
//! the built-in — for `arr.push(x)` or `s.charCodeAt(i)` that round trip is
//! many times the cost of the method itself.
//!
//! A [`NativeSite`] (one per [`Op::CallMethod`](super::Op::CallMethod) site,
//! kept in the site's [`PropertyCache`]) remembers which intrinsic the call
//! resolved to: the function cell's identity, a [`FastOp`] naming its
//! implementation, and a [`ChainKeyCache`] proving the `[[Get]]` of `key`
//! still reaches that cell. A call is served here only when all of these hold
//! at the moment of the call:
//!
//! - the receiver is a kind whose own properties are known not to include
//!   `key` ([`Realm::method_lookup_start`]: a string primitive, an ordinary
//!   non-wrapper object, an array, a RegExp, a Map/Set; never a proxy or a
//!   host-exotic object), so the lookup continues at its `[[Prototype]]`;
//! - from there the chain is still exactly the recorded objects with the same
//!   own key sets (identity plus key epoch, so an added shadowing method, a
//!   getter, a deleted method or a `setPrototypeOf` all miss), and the holder's
//!   *current* value is the recorded cell (so `Array.prototype.push = f`
//!   misses);
//! - the operation's own preconditions hold (receiver brand, arguments that
//!   convert without running code, the array fast-path guards, …).
//!
//! Anything else returns `None` and the caller takes its ordinary path, so a
//! hit is always the very call the slow path would have made: the resolved
//! method is the intrinsic, `this` is the receiver, and the implementation is
//! the one the host's dispatch runs for that receiver and name. A
//! multi-realm run is left alone (an intrinsic from another realm would build
//! its results and errors in that realm).
//!
//! The entry point is [`try_call`], which receives the live VM context, so an
//! implementation that must call back into VM code (a callback-taking method)
//! can do so with `ctx`/`funcs` rather than through the host.

use super::{Ctx, FnProto, Reg, VmError, with_host};
use crate::heap::Handle;
use crate::ic::PropertyCache;
use crate::nanbox::{NanBox, Unpacked};
use crate::nbexec::{
    N_ARRAY_PROTO_FN, N_MAP_PROTO_FN, N_OBJ_PROTO_HASOWN, N_REGEXP_PROTO_FN, N_SET_PROTO_FN,
    N_STRING_PROTO_FN,
};
use crate::realm::Realm;
use crate::realm::builtins_fast::MethodSiteCache;
use alloc::boxed::Box;

/// How many times a site may (re)resolve its method before it stops trying.
const FILL_LIMIT: u8 = 8;

/// The built-in implementations a site can call directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FastOp {
    /// Nothing known (cold, or the method is not one served here).
    None,
    /// `Array.prototype.push`.
    ArrayPush,
    /// `String.prototype.charCodeAt`.
    StrCharCodeAt,
    /// `String.prototype.charAt`.
    StrCharAt,
    /// `Map.prototype.get`.
    MapGet,
    /// `Map.prototype.set`.
    MapSet,
    /// `Map.prototype.has`.
    MapHas,
    /// `Set.prototype.add`.
    SetAdd,
    /// `Set.prototype.has`.
    SetHas,
    /// `Object.prototype.hasOwnProperty`.
    ObjHasOwn,
    /// `RegExp.prototype.test` / `exec`: the resolved function called through
    /// the host directly (no by-name re-resolution).
    RegExpCall,
    /// An `Array.prototype` higher-order built-in (`map`, `forEach`, `sort`,
    /// …), whose loop runs in the VM (`array_hof`).
    ArrayHof(super::array_hof::Hof),
}

/// A method-call site's memo of the intrinsic it calls (see the module docs).
pub(crate) struct NativeSite {
    /// Proves the `[[Get]]` of the key from a receiver still finds `method`.
    chain: MethodSiteCache,
    /// The intrinsic function cell's bits.
    method: u64,
    op: FastOp,
    fills: u8,
}

/// The function cell `v` as one of the built-ins served here.
fn classify(realm: &Realm, v: NanBox) -> FastOp {
    let Some(f) = v.as_handle().map(Handle::from_raw) else {
        return FastOp::None;
    };
    if realm.native_at(f) == Some(N_OBJ_PROTO_HASOWN) {
        return FastOp::ObjHasOwn;
    }
    let Some((id, target)) = realm.bound_native_at(f) else {
        return FastOp::None;
    };
    let Some(name) = realm.string_leaf_bytes(target) else {
        return FastOp::None;
    };
    match (id, name) {
        (N_ARRAY_PROTO_FN, b"push") => FastOp::ArrayPush,
        (N_STRING_PROTO_FN, b"charCodeAt") => FastOp::StrCharCodeAt,
        (N_STRING_PROTO_FN, b"charAt") => FastOp::StrCharAt,
        (N_MAP_PROTO_FN, b"get") => FastOp::MapGet,
        (N_MAP_PROTO_FN, b"set") => FastOp::MapSet,
        (N_MAP_PROTO_FN, b"has") => FastOp::MapHas,
        (N_SET_PROTO_FN, b"add") => FastOp::SetAdd,
        (N_SET_PROTO_FN, b"has") => FastOp::SetHas,
        (N_REGEXP_PROTO_FN, b"test" | b"exec") => FastOp::RegExpCall,
        (N_ARRAY_PROTO_FN, n) => {
            super::array_hof::hof_for(n).map_or(FastOp::None, FastOp::ArrayHof)
        }
        _ => FastOp::None,
    }
}

/// Whether a lookup of `key` may be memoized by a [`ChainKeyCache`]: a named
/// key, not one a receiver can own exotically (an index, `length`, a
/// RegExp's `lastIndex`).
fn cacheable_key(key: &str) -> bool {
    !key.is_empty()
        && key != "length"
        && key != "lastIndex"
        && !key.as_bytes()[0].is_ascii_digit()
        && !key.starts_with('-')
}

/// Whether `site` still proves that `recv[key]` is its recorded intrinsic.
fn site_hit(realm: &Realm, site: &mut NativeSite, h: Handle, key: &str) -> bool {
    site.op != FastOp::None
        && realm
            .method_site_get(h, key, &mut site.chain)
            .is_some_and(|v| v.to_bits() == site.method)
}

/// Re-resolves the site for receiver `h`: records the intrinsic `recv[key]`
/// reaches, if it is one served here.
fn fill(realm: &Realm, site: &mut NativeSite, h: Handle, key: &str) {
    site.fills += 1;
    site.op = FastOp::None;
    if let Some(v) = realm.method_site_fill(h, key, &mut site.chain) {
        site.op = classify(realm, v);
        site.method = v.to_bits();
    }
}

/// Whether the site of `cache` currently calls a built-in served by
/// [`try_call`] — a hint that its method is not a VM function, so a caller
/// looking for one may skip its own lookup. It does not re-validate the
/// site: a stale answer only sends a VM-function call down the host's
/// general path, which is slower but exact (and the next [`try_call`] miss
/// re-resolves or retires the site).
pub(super) fn is_native_site(cache: &PropertyCache) -> bool {
    cache
        .native
        .as_deref()
        .is_some_and(|s| s.op != FastOp::None)
}

/// `recv[key](args…)` with `this = recv` when the site's method is a built-in
/// served here (see the module docs); `None` to take the ordinary path. The
/// arguments are registers `args` of `regs`.
pub(super) fn try_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
) -> Option<Result<NanBox, VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let h = Handle::from_raw(recv.as_handle()?);
    let site = cache.native.get_or_insert_with(|| {
        Box::new(NativeSite {
            chain: MethodSiteCache::default(),
            method: 0,
            op: FastOp::None,
            fills: 0,
        })
    });
    if !site_hit(&ctx.realm, site, h, key) {
        if site.fills >= FILL_LIMIT || !cacheable_key(key) {
            // Retired: a site that keeps changing its mind calls the general
            // path from now on.
            site.op = FastOp::None;
            return None;
        }
        fill(&ctx.realm, site, h, key);
        if site.op == FastOp::None {
            return None;
        }
    }
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    if let FastOp::ArrayHof(op) = site.op {
        let argv = super::ArgBuf::from_regs(regs, args);
        return super::array_hof::run(ctx, funcs, regs, recv, op, &argv);
    }
    let realm = &mut ctx.realm;
    let v = match site.op {
        FastOp::None | FastOp::ArrayHof(_) => return None,
        FastOp::ArrayPush => {
            // Exactly the host's dense-array `push` guards: no index side table,
            // no inherited index accessor, the length is the dense length and
            // still writable.
            let dense_len = realm.array_elements(h)?.len();
            if realm.array_has_index_overrides(h)
                || realm.proto_index_accessor_dirty()
                || realm.array_length(h) != Some(dense_len)
                || realm.array_length_is_readonly(h)
            {
                return None;
            }
            let mut len = dense_len;
            for r in args {
                len = realm.array_push(h, regs[*r as usize]).unwrap_or(len);
            }
            NanBox::number(len as f64)
        }
        FastOp::StrCharCodeAt | FastOp::StrCharAt => {
            if !realm.is_string_handle(h) {
                return None;
            }
            // ToIntegerOrInfinity of a Number (or `undefined`) runs no code.
            let a = arg(0);
            let n = match a.as_number() {
                Some(n) => n,
                None if a.is_undefined() => 0.0,
                None => return None,
            };
            let n = if n.is_nan() { 0.0 } else { n };
            // `n` truncates to a non-negative index exactly when it exceeds -1.
            let idx = (n > -1.0).then_some(n as usize);
            realm.flatten_string(h);
            let unit = realm.string_unit_at_index(h, idx)?;
            match (site.op, unit) {
                (FastOp::StrCharCodeAt, Some(u)) => NanBox::number(f64::from(u)),
                (FastOp::StrCharCodeAt, None) => NanBox::number(f64::NAN),
                (_, Some(u)) => NanBox::handle(
                    realm
                        .new_string_wtf8(crate::wtf8::from_utf16(&[u]))
                        .to_raw(),
                ),
                (_, None) => NanBox::handle(realm.new_string("").to_raw()),
            }
        }
        FastOp::MapGet | FastOp::MapSet | FastOp::MapHas | FastOp::SetAdd | FastOp::SetHas => {
            let want_set = matches!(site.op, FastOp::SetAdd | FastOp::SetHas);
            if realm.collection_is_set(h) != Some(want_set) || realm.collection_is_weak(h) {
                return None;
            }
            match site.op {
                FastOp::MapGet => realm
                    .collection_get(h, arg(0))
                    .unwrap_or(NanBox::undefined()),
                FastOp::MapSet => {
                    realm.collection_set(h, arg(0), arg(1));
                    recv
                }
                FastOp::SetAdd => {
                    realm.collection_set(h, arg(0), arg(0));
                    recv
                }
                _ => NanBox::boolean(realm.collection_has(h, arg(0))),
            }
        }
        FastOp::ObjHasOwn => {
            // An ordinary object receiver (ToObject is the identity), and a key
            // whose ToPropertyKey runs no code: a string or another primitive
            // that is not a Symbol.
            realm.object_key_epoch(h)?;
            let k = arg(0);
            let leaf;
            let owned;
            let key: &str = match k.as_handle().map(Handle::from_raw) {
                // A flat, well-formed string is its own key, borrowed.
                Some(kh)
                    if let Some(b) = realm.string_leaf_bytes(kh)
                        && let Ok(t) = core::str::from_utf8(b) =>
                {
                    leaf = t;
                    leaf
                }
                Some(kh) if realm.is_string_handle(kh) => {
                    owned = realm.to_display_string(k);
                    &owned
                }
                Some(_) => return None,
                None => match k.unpack() {
                    Unpacked::Number(_)
                    | Unpacked::Bool(_)
                    | Unpacked::Undefined
                    | Unpacked::Null => {
                        owned = realm.to_display_string(k);
                        &owned
                    }
                    _ => return None,
                },
            };
            NanBox::boolean(realm.has_own(h, key))
        }
        FastOp::RegExpCall => {
            realm.regexp_flags(h)?;
            let method = NanBox::from_bits(site.method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host(ctx, |host| host.call(method, recv, &argv))
                    .unwrap_or(Err(super::HostError::Fault))
                    .map_err(VmError::from),
            );
        }
    };
    Some(Ok(v))
}
