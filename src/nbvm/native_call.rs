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
    N_ARRAY_IS_ARRAY, N_ARRAY_PROTO_FN, N_JSON_PARSE, N_JSON_STRINGIFY, N_MAP_PROTO_FN,
    N_NUMBER_PROTO_FN, N_OBJ_PROTO_HASOWN, N_OBJECT_ASSIGN, N_OBJECT_ENTRIES, N_OBJECT_KEYS,
    N_REGEXP_PROTO_FN, N_SET_PROTO_FN, N_STATIC_METHOD, N_STRING_PROTO_FN,
};
use crate::realm::Realm;
use crate::realm::builtins_fast::MethodSiteCache;
use crate::realm::coll_fast::CollOp;
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
    /// `Array.prototype.pop`.
    ArrayPop,
    ArrayShift,
    /// `String.prototype.indexOf`.
    StrIndexOf,
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
    /// `Map.prototype.delete`.
    MapDelete,
    /// `Set.prototype.delete`.
    SetDelete,
    /// `Object.prototype.hasOwnProperty`.
    ObjHasOwn,
    /// `RegExp.prototype.test` / `exec`: the resolved function called through
    /// the host directly (no by-name re-resolution).
    RegExpCall,
    /// `String.prototype.slice`.
    StrSlice,
    /// `String.prototype.substring`.
    StrSubstring,
    /// `String.prototype.startsWith`.
    StrStartsWith,
    /// `String.prototype.endsWith`.
    StrEndsWith,
    /// `String.prototype.includes`.
    StrIncludes,
    /// `String.fromCharCode`.
    StrFromCharCode,
    /// `Array.isArray`.
    ArrayIsArray,
    /// `Function.prototype.call` on a VM function.
    FnCall,
    /// `Function.prototype.apply` on a VM function.
    FnApply,
    /// An `Array.prototype` higher-order built-in (`map`, `forEach`, `sort`,
    /// …), whose loop runs in the VM (`array_hof`).
    ArrayHof(super::array_hof::Hof),
    /// `Number.prototype.toFixed` on a Number primitive.
    NumToFixed,
    /// `Number.prototype.toString` on a Number primitive.
    NumToString,
    /// `Array.prototype.indexOf`.
    ArrIndexOf,
    /// `Array.prototype.includes`.
    ArrIncludes,
    /// `Array.prototype.join`.
    ArrJoin,
    /// `Array.prototype.slice`.
    ArrSlice,
    /// `String.prototype.toUpperCase` / `toLowerCase`.
    StrCase(bool),
    /// `String.prototype.trim` / `trimStart` / `trimEnd` (which ends).
    StrTrim(bool, bool),
    /// `String.prototype.padStart` / `padEnd` (whether at the start).
    StrPad(bool),
    /// `String.prototype.split`.
    StrSplit,
    /// `String.prototype.replace`.
    StrReplace,
    /// `String.prototype.replaceAll` (with a RegExp and a VM function, its
    /// replacer calls run in the VM; otherwise called through the host
    /// directly, as [`FastOp::HostDirect`]).
    StrReplaceAll,
    /// `Object.keys`.
    ObjKeys,
    /// `Object.entries`.
    ObjEntries,
    /// A built-in `Math` function (its native id), computed by
    /// `math_fast::compute` on Number arguments.
    #[cfg(feature = "std")]
    Math(u16),
    /// A built-in whose implementation is the host's (`Object.assign`,
    /// `JSON.stringify`, `JSON.parse`, and every built-in not served
    /// otherwise): the resolved function is called through the host
    /// directly, with no by-name re-resolution.
    HostDirect,
}

impl FastOp {
    /// The `Map`/`Set` method this is, for `Realm::coll_fast`.
    #[inline]
    fn coll_op(self) -> Option<CollOp> {
        Some(match self {
            FastOp::MapGet => CollOp::MapGet,
            FastOp::MapSet => CollOp::MapSet,
            FastOp::MapHas => CollOp::MapHas,
            FastOp::MapDelete => CollOp::MapDelete,
            FastOp::SetAdd => CollOp::SetAdd,
            FastOp::SetHas => CollOp::SetHas,
            FastOp::SetDelete => CollOp::SetDelete,
            _ => return None,
        })
    }
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
    match realm.native_at(f) {
        Some(N_OBJ_PROTO_HASOWN) => return FastOp::ObjHasOwn,
        Some(N_ARRAY_IS_ARRAY) => return FastOp::ArrayIsArray,
        Some(N_OBJECT_KEYS) => return FastOp::ObjKeys,
        Some(N_OBJECT_ENTRIES) => return FastOp::ObjEntries,
        Some(N_OBJECT_ASSIGN | N_JSON_STRINGIFY | N_JSON_PARSE) => return FastOp::HostDirect,
        #[cfg(feature = "std")]
        Some(id) if super::math_fast::is_math_native(id) => return FastOp::Math(id),
        _ => {}
    }
    let Some((id, target)) = realm.bound_native_at(f) else {
        return other_native(realm, f);
    };
    // A readable static method binds `[constructor, name]`; only `String`
    // has a `fromCharCode`.
    if id == N_STATIC_METHOD {
        return match realm
            .array_elements(target)
            .and_then(|p| p.get(1).copied())
            .and_then(NanBox::as_handle)
            .and_then(|n| realm.string_leaf_bytes(Handle::from_raw(n)))
        {
            Some(b"fromCharCode") => FastOp::StrFromCharCode,
            _ => FastOp::HostDirect,
        };
    }
    let Some(name) = realm.string_leaf_bytes(target) else {
        return FastOp::None;
    };
    match (id, name) {
        (N_ARRAY_PROTO_FN, b"push") => FastOp::ArrayPush,
        (N_ARRAY_PROTO_FN, b"pop") => FastOp::ArrayPop,
        (N_ARRAY_PROTO_FN, b"shift") => FastOp::ArrayShift,
        (N_STRING_PROTO_FN, b"indexOf") => FastOp::StrIndexOf,
        (N_STRING_PROTO_FN, b"charCodeAt") => FastOp::StrCharCodeAt,
        (N_STRING_PROTO_FN, b"charAt") => FastOp::StrCharAt,
        (N_MAP_PROTO_FN, b"get") => FastOp::MapGet,
        (N_MAP_PROTO_FN, b"set") => FastOp::MapSet,
        (N_MAP_PROTO_FN, b"has") => FastOp::MapHas,
        (N_SET_PROTO_FN, b"add") => FastOp::SetAdd,
        (N_SET_PROTO_FN, b"has") => FastOp::SetHas,
        (N_MAP_PROTO_FN, b"delete") => FastOp::MapDelete,
        (N_SET_PROTO_FN, b"delete") => FastOp::SetDelete,
        (N_REGEXP_PROTO_FN, b"test" | b"exec") => FastOp::RegExpCall,
        (N_STRING_PROTO_FN, b"slice") => FastOp::StrSlice,
        (N_STRING_PROTO_FN, b"substring") => FastOp::StrSubstring,
        (N_STRING_PROTO_FN, b"startsWith") => FastOp::StrStartsWith,
        (N_STRING_PROTO_FN, b"endsWith") => FastOp::StrEndsWith,
        (N_STRING_PROTO_FN, b"includes") => FastOp::StrIncludes,
        (N_STRING_PROTO_FN, b"toUpperCase") => FastOp::StrCase(true),
        (N_STRING_PROTO_FN, b"toLowerCase") => FastOp::StrCase(false),
        (N_STRING_PROTO_FN, b"trim") => FastOp::StrTrim(true, true),
        (N_STRING_PROTO_FN, b"trimStart") => FastOp::StrTrim(true, false),
        (N_STRING_PROTO_FN, b"trimEnd") => FastOp::StrTrim(false, true),
        (N_STRING_PROTO_FN, b"padStart") => FastOp::StrPad(true),
        (N_STRING_PROTO_FN, b"padEnd") => FastOp::StrPad(false),
        (N_STRING_PROTO_FN, b"split") => FastOp::StrSplit,
        (N_STRING_PROTO_FN, b"replace") => FastOp::StrReplace,
        (N_STRING_PROTO_FN, b"replaceAll") => FastOp::StrReplaceAll,
        #[cfg(feature = "std")]
        (N_NUMBER_PROTO_FN, b"toFixed") => FastOp::NumToFixed,
        (N_NUMBER_PROTO_FN, b"toString") => FastOp::NumToString,
        (N_ARRAY_PROTO_FN, b"indexOf") => FastOp::ArrIndexOf,
        (N_ARRAY_PROTO_FN, b"includes") => FastOp::ArrIncludes,
        (N_ARRAY_PROTO_FN, b"join") => FastOp::ArrJoin,
        (N_ARRAY_PROTO_FN, b"slice") => FastOp::ArrSlice,
        // `Function.prototype`'s methods share the Array methods' dispatch id.
        (N_ARRAY_PROTO_FN, b"call") => FastOp::FnCall,
        (N_ARRAY_PROTO_FN, b"apply") => FastOp::FnApply,
        (N_ARRAY_PROTO_FN, n) => {
            super::array_hof::hof_for(n).map_or(FastOp::HostDirect, FastOp::ArrayHof)
        }
        _ => FastOp::HostDirect,
    }
}

/// Any other built-in function: called through the host directly
/// ([`FastOp::HostDirect`]) — exactly the general path's `Call(method, recv,
/// args)` once its `[[Get]]` (which the site proves) is done.
fn other_native(realm: &Realm, f: Handle) -> FastOp {
    if realm.is_callable_cell(f) {
        FastOp::HostDirect
    } else {
        FastOp::None
    }
}

/// Whether a lookup of `key` may be memoized by a [`MethodSiteCache`]: a named
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
/// `prim`: `h` is a primitive receiver's prototype intrinsic, where the
/// lookup starts.
fn site_hit(realm: &Realm, site: &mut NativeSite, h: Handle, prim: bool, key: &str) -> bool {
    site.op != FastOp::None
        && if prim {
            realm.method_site_get_prim(h, key, &mut site.chain)
        } else {
            realm.method_site_get(h, key, &mut site.chain)
        }
        .is_some_and(|v| v.to_bits() == site.method)
}

/// Re-resolves the site for receiver `h` (see [`site_hit`] for `prim`):
/// records the intrinsic `recv[key]` reaches, if it is one served here.
fn fill(realm: &mut Realm, site: &mut NativeSite, h: Handle, prim: bool, key: &str) {
    site.fills += 1;
    site.op = FastOp::None;
    let found = if prim {
        realm.method_site_fill_prim(h, key, &mut site.chain)
    } else {
        realm.method_site_fill(h, key, &mut site.chain)
    };
    if let Some(v) = found {
        site.op = classify(realm, v);
        site.method = v.to_bits();
    }
}

/// The elements of the array `h` when reading them is plain data access: a
/// dense array with no holes (a hole would read through the prototype), no
/// index side table, no inherited index accessor, and its length the dense
/// length.
fn dense_elements(realm: &Realm, h: Handle) -> Option<&[NanBox]> {
    let e = realm.array_elements(h)?;
    if realm.array_has_index_overrides(h)
        || realm.proto_index_accessor_dirty()
        || realm.array_length(h) != Some(e.len())
        || e.iter().any(|v| v.is_hole())
    {
        return None;
    }
    Some(e)
}

/// A relative index argument (`ToIntegerOrInfinity`, then counted from the
/// end when negative) clamped to `0..=len`, when it converts without running
/// code; `undefined` is `default`.
fn rel_index(v: NanBox, len: usize, default: usize) -> Option<usize> {
    Some(match int_arg(v)? {
        None => default,
        Some(n) if n < 0.0 => (len as f64 + n).max(0.0) as usize,
        Some(n) => (n as usize).min(len),
    })
}

/// `ToLength(v)` of a Number or `undefined` (which runs no code).
fn length_arg(v: NanBox) -> Option<usize> {
    Some(match int_arg(v)? {
        None => 0,
        Some(n) => n.max(0.0) as usize,
    })
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

/// `ToIntegerOrInfinity(v)` when it runs no code: `Some(None)` for
/// `undefined` (left to the caller, whose default differs per method),
/// `Some(Some(n))` for a Number, `None` for anything else.
fn int_arg(v: NanBox) -> Option<Option<f64>> {
    if v.is_undefined() {
        return Some(None);
    }
    let n = v.as_number()?;
    Some(Some(if n.is_nan() {
        0.0
    } else if n.is_infinite() {
        n
    } else {
        // Truncation toward zero (saturating past the i64 range, far beyond
        // any string length these positions are clamped to); `-0` becomes 0.
        (n as i64) as f64 + 0.0
    }))
}

/// A string receiver, flattened, as its rope (one leaf).
fn str_parts(realm: &mut Realm, h: Handle) -> Option<crate::rope::Rope> {
    // A string cell with a function marker is a function, not a string.
    if !realm.typeof_is_string(h) {
        return None;
    }
    realm.flatten_string(h);
    let rope = realm.string_rope(h)?;
    rope.as_leaf_bytes()?;
    Some(rope)
}

/// A string argument, flattened (`ToString` is then the identity).
fn str_arg(realm: &mut Realm, v: NanBox) -> Option<crate::rope::Rope> {
    let h = v.as_handle().map(Handle::from_raw)?;
    str_parts(realm, h)
}

/// Calls the VM function `f` with `this` and `argv`, publishing the calling
/// frame's registers for a collection inside the callee (as the interpreter's
/// call arms do).
fn call_vm(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    f: NanBox,
    this: NanBox,
    argv: &[NanBox],
) -> Result<NanBox, VmError> {
    let pub_mark = ctx.frame_shadow.len();
    ctx.frame_shadow.extend_from_slice(regs);
    ctx.frames_published += 1;
    let r = super::call_closure(ctx, funcs, f, argv, this);
    ctx.frames_published -= 1;
    ctx.frame_shadow.truncate(pub_mark);
    r
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
    if cache
        .native
        .as_deref()
        .is_some_and(|s| s.op.coll_op().is_some())
        && let Some(v) = try_coll(ctx, recv, regs, args, cache)
    {
        return Some(Ok(v));
    }
    // A Number receiver owns no property: its lookup starts at
    // `%Number.prototype%`, which then stands in for it in the site's chain.
    let (h, prim) = match recv.as_handle() {
        Some(r) => (Handle::from_raw(r), false),
        None if recv.as_number().is_some() => (ctx.realm.primitive_proto(recv)?, true),
        None => return None,
    };
    let site = cache.native.get_or_insert_with(|| {
        Box::new(NativeSite {
            chain: MethodSiteCache::default(),
            method: 0,
            op: FastOp::None,
            fills: 0,
        })
    });
    if !site_hit(&ctx.realm, site, h, prim, key) {
        if site.fills >= FILL_LIMIT || !cacheable_key(key) {
            // Retired: a site that keeps changing its mind calls the general
            // path from now on.
            site.op = FastOp::None;
            return None;
        }
        fill(&mut ctx.realm, site, h, prim, key);
        if site.op == FastOp::None {
            return None;
        }
    } else if !prim {
        ctx.realm.method_site_note_recv(h, &mut site.chain);
    }
    let (op, method) = (site.op, site.method);
    run_op(ctx, funcs, op, method, recv, h, prim, regs, args)
}

/// `recv.key(args…)` when the site last called a `Map.prototype` /
/// `Set.prototype` method on this very Map/Set and still proves it is that
/// intrinsic (`Realm::method_site_memo_hit`): the method's lean form
/// (`Realm::coll_fast`). `None`, having done nothing, otherwise. Never runs
/// code and never throws (a receiver of the wrong kind is a miss).
#[inline(never)]
pub(super) fn try_coll(
    ctx: &mut Ctx,
    recv: NanBox,
    regs: &[NanBox],
    args: &[Reg],
    cache: &PropertyCache,
) -> Option<NanBox> {
    let site = cache.native.as_deref()?;
    let op = site.op.coll_op()?;
    let h = Handle::from_raw(recv.as_handle()?);
    if ctx.host.is_none()
        || ctx.realm.multi_realm
        || !ctx.realm.method_site_memo_hit(h, &site.chain, site.method)
    {
        return None;
    }
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    ctx.realm.coll_fast(h, op, arg(0), arg(1))
}

/// `Call(method, recv, args…)` for a method value already read (EvaluateCall
/// reads a callee before its arguments): served here when `method` is one of
/// the built-ins of [`classify`], memoized per site by the method cell.
/// `None` to take the ordinary path.
pub(super) fn try_call_got(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    method: NanBox,
    recv: NanBox,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
) -> Option<Result<NanBox, VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm || method.as_handle().is_none() {
        return None;
    }
    let (h, prim) = match recv.as_handle() {
        Some(r) => (Handle::from_raw(r), false),
        None if recv.as_number().is_some() => (ctx.realm.primitive_proto(recv)?, true),
        None => return None,
    };
    let site = cache.native.get_or_insert_with(|| {
        Box::new(NativeSite {
            chain: MethodSiteCache::default(),
            method: 0,
            op: FastOp::None,
            fills: 0,
        })
    });
    if site.method != method.to_bits() {
        if site.fills >= FILL_LIMIT {
            site.op = FastOp::None;
            return None;
        }
        site.fills += 1;
        site.method = method.to_bits();
        site.op = classify(&ctx.realm, method);
    }
    if site.op == FastOp::None {
        return None;
    }
    let (op, method) = (site.op, site.method);
    run_op(ctx, funcs, op, method, recv, h, prim, regs, args)
}

/// `recv[key]` when the site (an `Op::GetMethod`'s) proves it is the
/// built-in it recorded, as [`try_call`] proves its method (see the module
/// docs): that very function cell, read without running code. `None` to take
/// the ordinary `[[Get]]`.
pub(super) fn try_get(
    ctx: &mut Ctx,
    recv: NanBox,
    key: &str,
    cache: &mut PropertyCache,
) -> Option<NanBox> {
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let (h, prim) = match recv.as_handle() {
        Some(r) => (Handle::from_raw(r), false),
        None if recv.as_number().is_some() => (ctx.realm.primitive_proto(recv)?, true),
        None => return None,
    };
    let site = cache.native.get_or_insert_with(|| {
        Box::new(NativeSite {
            chain: MethodSiteCache::default(),
            method: 0,
            op: FastOp::None,
            fills: 0,
        })
    });
    // (The proof `site_hit` / `fill` make for `try_call`, written out here so
    // that those stay `try_call`'s alone: inlined into it as before.)
    let hit = site.op != FastOp::None
        && if prim {
            ctx.realm.method_site_get_prim(h, key, &mut site.chain)
        } else {
            ctx.realm.method_site_get(h, key, &mut site.chain)
        }
        .is_some_and(|v| v.to_bits() == site.method);
    if !hit {
        if site.fills >= FILL_LIMIT || !cacheable_key(key) {
            site.op = FastOp::None;
            return None;
        }
        site.fills += 1;
        site.op = FastOp::None;
        let found = if prim {
            ctx.realm.method_site_fill_prim(h, key, &mut site.chain)
        } else {
            ctx.realm.method_site_fill(h, key, &mut site.chain)
        };
        let v = found?;
        site.op = classify(&ctx.realm, v);
        site.method = v.to_bits();
        if site.op == FastOp::None {
            return None;
        }
    }
    Some(NanBox::from_bits(site.method))
}

/// The rest of [`try_call`] / [`try_call_got`] once the site knows its
/// built-in `op` (the function cell `method`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_op(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    op: FastOp,
    method: u64,
    recv: NanBox,
    h: Handle,
    prim: bool,
    regs: &[NanBox],
    args: &[Reg],
) -> Option<Result<NanBox, VmError>> {
    // Only the Number methods take a primitive receiver; anything else the
    // chain of `%Number.prototype%` reaches (`hasOwnProperty`, …) would get
    // the prototype as its `this`.
    if prim != matches!(op, FastOp::NumToFixed | FastOp::NumToString) {
        return None;
    }
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    if let FastOp::ArrayHof(op) = op {
        let argv = super::ArgBuf::from_regs(regs, args);
        return super::array_hof::run(ctx, funcs, regs, recv, op, &argv);
    }
    // A RegExp and a VM function: the replacer calls run here
    // (`str_replace`).
    if matches!(op, FastOp::StrReplace | FastOp::StrReplaceAll) {
        let all = op == FastOp::StrReplaceAll;
        if let Some(r) = super::str_replace::try_run(ctx, funcs, regs, recv, args, all) {
            return Some(r);
        }
        if all {
            let method = NanBox::from_bits(method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host(ctx, |host| host.call(method, recv, &argv))
                    .unwrap_or(Err(super::HostError::Fault))
                    .map_err(VmError::from),
            );
        }
    }
    let realm = &mut ctx.realm;
    let v = match op {
        FastOp::None | FastOp::ArrayHof(_) | FastOp::StrReplaceAll => return None,
        #[cfg(feature = "std")]
        FastOp::Math(id) => super::math_fast::compute(realm, id, args, regs)?,
        FastOp::ArrayPush => {
            // An array that never had side-table state: one cell access.
            if let [a] = *args
                && let Some(n) = realm.array_push_plain(h, &[regs[a as usize]])
            {
                return Some(Ok(NanBox::number(n as f64)));
            }
            // Exactly the host's dense-array `push` guards: no index side table,
            // no inherited index accessor, the length is the dense length and
            // still writable.
            let dense_len = realm.array_elements(h)?.len();
            if realm.array_has_index_overrides(h)
                || realm.proto_index_accessor_dirty()
                || realm.array_length(h) != Some(dense_len)
                || realm.array_length_is_readonly(h)
                || realm.array_is_non_extensible(h)
            {
                return None;
            }
            let mut len = dense_len;
            for r in args {
                len = realm.array_push(h, regs[*r as usize]).unwrap_or(len);
            }
            NanBox::number(len as f64)
        }
        FastOp::ArrayPop => {
            if let Some(v) = realm.array_pop_plain(h) {
                return Some(Ok(v));
            }
            // The host's dense-array `pop` guards; an inherited getter could fire
            // only for a hole at the index read.
            let dense_len = realm.array_elements(h)?.len();
            if realm.array_has_index_overrides(h)
                || realm.proto_index_accessor_dirty()
                || (dense_len > 0 && realm.array_hole_at(h, dense_len - 1))
                || realm.array_length(h) != Some(dense_len)
                || realm.array_length_is_readonly(h)
            {
                return None;
            }
            realm.array_pop(h)
        }
        FastOp::ArrayShift => realm.array_shift_fast(h)?,
        FastOp::StrIndexOf => {
            // A string needle (ToString is the identity) and a Number or absent
            // position (ToIntegerOrInfinity runs no code).
            let needle = arg(0).as_handle().map(Handle::from_raw)?;
            if !realm.is_string_handle(h) || !realm.is_string_handle(needle) {
                return None;
            }
            let pos = arg(1);
            let pos = match pos.as_number() {
                Some(n) => n,
                None if pos.is_undefined() => 0.0,
                None => return None,
            };
            realm.flatten_string(h);
            realm.flatten_string(needle);
            let rope = realm.string_rope(h)?;
            let hay = rope.as_leaf_bytes()?;
            let units = rope.utf16_len();
            // `max(0)` maps NaN to 0 as ToIntegerOrInfinity does; the cast
            // truncates the rest and saturates +Infinity.
            let from = (pos.max(0.0) as usize).min(units);
            let pat = realm.string_leaf_bytes(needle)?;
            NanBox::number(crate::nbexec::index_of_units(
                hay,
                pat,
                from,
                rope.is_ascii(),
            ))
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
            match (op, unit) {
                (FastOp::StrCharCodeAt, Some(u)) => NanBox::number(f64::from(u)),
                (FastOp::StrCharCodeAt, None) => NanBox::number(f64::NAN),
                (_, Some(u)) => NanBox::handle(realm.unit_string(u).to_raw()),
                (_, None) => NanBox::handle(realm.new_string("").to_raw()),
            }
        }
        FastOp::MapGet
        | FastOp::MapSet
        | FastOp::MapHas
        | FastOp::MapDelete
        | FastOp::SetAdd
        | FastOp::SetHas
        | FastOp::SetDelete => {
            let want_set = matches!(op, FastOp::SetAdd | FastOp::SetHas | FastOp::SetDelete);
            if realm.collection_is_set(h) != Some(want_set) || realm.collection_is_weak(h) {
                return None;
            }
            match op {
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
                FastOp::MapDelete | FastOp::SetDelete => {
                    NanBox::boolean(realm.collection_delete(h, arg(0)))
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
                // A flat, well-formed string is its own key, borrowed (one that
                // starts with NUL is escaped, see `realm::is_internal_key`).
                Some(kh)
                    if let Some(b) = realm.string_leaf_bytes(kh)
                        && b.first() != Some(&0)
                        && let Ok(t) = core::str::from_utf8(b) =>
                {
                    leaf = t;
                    leaf
                }
                Some(kh) if realm.is_string_handle(kh) => {
                    owned = crate::realm::user_key_owned(realm.to_display_string(k));
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
        FastOp::StrSlice | FastOp::StrSubstring => {
            let rope = str_parts(realm, h)?;
            let units = rope.utf16_len();
            let start = int_arg(arg(0))?.unwrap_or(0.0);
            let end = int_arg(arg(1))?;
            let (a, b) = if op == FastOp::StrSlice {
                let idx = |n: f64| -> usize {
                    if n < 0.0 {
                        (units as f64 + n).max(0.0) as usize
                    } else {
                        (n as usize).min(units)
                    }
                };
                let a = idx(start);
                let b = end.map_or(units, idx);
                if a < b { (a, b) } else { (a, a) }
            } else {
                let clamp = |n: f64| (n.max(0.0) as usize).min(units);
                let a = clamp(start);
                let b = end.map_or(units, clamp);
                if a > b { (b, a) } else { (a, b) }
            };
            NanBox::handle(realm.new_string_slice(&rope, a, b)?.to_raw())
        }
        FastOp::StrStartsWith | FastOp::StrEndsWith | FastOp::StrIncludes => {
            // A string search argument is not a RegExp, and converts as itself.
            let needle = str_arg(realm, arg(0))?;
            let rope = str_parts(realm, h)?;
            let (bytes, units) = (rope.as_leaf_bytes()?, rope.utf16_len());
            let pat = needle.as_leaf_bytes()?;
            let default = if op == FastOp::StrEndsWith { units } else { 0 };
            let pos = int_arg(arg(1))?.map_or(default, |n| (n.max(0.0) as usize).min(units));
            NanBox::boolean(match op {
                FastOp::StrStartsWith => {
                    crate::nbexec::starts_with_at_units(bytes, pat, pos, rope.is_ascii())
                }
                FastOp::StrEndsWith => {
                    crate::nbexec::ends_with_at_units(bytes, pat, pos, rope.is_ascii())
                }
                _ => crate::nbexec::index_of_units(bytes, pat, pos, rope.is_ascii()) >= 0.0,
            })
        }
        FastOp::StrFromCharCode => {
            // ToNumber of a Number runs no code; ToUint16 as the host does.
            let mut units = alloc::vec::Vec::with_capacity(args.len());
            for r in args {
                let n = regs[*r as usize].as_number()?;
                // Modular all the way up (an `as i64` cast saturates past 2^63).
                units.push(realm.to_uint32(NanBox::number(n)) as u16);
            }
            if let [u] = units[..] {
                NanBox::handle(realm.unit_string(u).to_raw())
            } else {
                let out = crate::wtf8::from_utf16(&units);
                NanBox::handle(realm.new_string_wtf8(out).to_raw())
            }
        }
        FastOp::ArrayIsArray => match arg(0).as_handle().map(Handle::from_raw) {
            // A proxy is unwrapped (and may be revoked): the host's.
            Some(a) if realm.proxy_at(a).is_some() => return None,
            Some(a) => NanBox::boolean(realm.is_array(a) && !realm.is_vm_function(a)),
            None => NanBox::boolean(false),
        },
        FastOp::FnCall | FastOp::FnApply => {
            if !realm.is_vm_function(h) {
                return None;
            }
            let this = arg(0);
            let argv: alloc::vec::Vec<NanBox> = if op == FastOp::FnCall {
                args.iter().skip(1).map(|r| regs[*r as usize]).collect()
            } else {
                // CreateListFromArrayLike of a plain dense array reads only its
                // own elements: no hole (which would read through the
                // prototype), no index side table, the length its dense length.
                let list = arg(1);
                match list.as_handle().map(Handle::from_raw) {
                    None if list.is_undefined() || list.is_null() => alloc::vec::Vec::new(),
                    Some(l) => {
                        let elems = realm.array_elements(l)?;
                        if realm.array_has_index_overrides(l)
                            || realm.array_length(l) != Some(elems.len())
                            || elems.iter().any(|e| e.is_hole())
                        {
                            return None;
                        }
                        elems.to_vec()
                    }
                    None => return None,
                }
            };
            // A callee with a compiled body is entered directly, as a
            // compiled caller's `f(...)` enters it.
            #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
            if let Some(r) =
                super::jit_tier::call_compiled_with(ctx, funcs, Some(regs), recv, this, &argv)
            {
                return Some(r);
            }
            return Some(call_vm(ctx, funcs, regs, recv, this, &argv));
        }
        FastOp::RegExpCall => {
            realm.regexp_flags(h)?;
            let method = NanBox::from_bits(method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host(ctx, |host| host.call_regexp_method(method, recv, &argv))
                    .unwrap_or(Err(super::HostError::Fault))
                    .map_err(VmError::from),
            );
        }
        FastOp::HostDirect => {
            let method = NanBox::from_bits(method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host(ctx, |host| host.call(method, recv, &argv))
                    .unwrap_or(Err(super::HostError::Fault))
                    .map_err(VmError::from),
            );
        }
        #[cfg(feature = "std")]
        FastOp::NumToFixed => {
            let n = recv.as_number()?;
            let d = int_arg(arg(0))?.unwrap_or(0.0);
            if !(0.0..=100.0).contains(&d) {
                return None;
            }
            let out = crate::nbexec::number_to_fixed(n, d as usize);
            NanBox::handle(realm.new_string(&out).to_raw())
        }
        #[cfg(not(feature = "std"))]
        FastOp::NumToFixed => return None,
        FastOp::NumToString => {
            let n = recv.as_number()?;
            let radix = int_arg(arg(0))?.unwrap_or(10.0);
            if !(2.0..=36.0).contains(&radix) {
                return None;
            }
            let out = if radix == 10.0 || !n.is_finite() {
                crate::realm::js_number_string(n)
            } else {
                crate::nbexec::int_to_radix(n, radix as u32)
            };
            NanBox::handle(realm.new_string(&out).to_raw())
        }
        FastOp::ArrIndexOf | FastOp::ArrIncludes => {
            let elems = dense_elements(realm, h)?;
            let len = elems.len();
            let target = arg(0);
            // `fromIndex` converts before the search; a zero length skips it.
            let from = if len == 0 {
                0
            } else {
                match int_arg(arg(1))? {
                    None => 0,
                    Some(n) if n < 0.0 => (len as f64 + n).max(0.0) as usize,
                    Some(n) => (n.min(len as f64)) as usize,
                }
            };
            let tail = elems.get(from..).unwrap_or(&[]);
            if op == FastOp::ArrIndexOf {
                let i = tail.iter().position(|e| realm.strict_equals(*e, target));
                NanBox::number(i.map_or(-1.0, |i| (from + i) as f64))
            } else {
                NanBox::boolean(tail.iter().any(|e| realm.same_value_zero(*e, target)))
            }
        }
        FastOp::ArrJoin => {
            let sep_v = arg(0);
            let sep: alloc::vec::Vec<u8> = if sep_v.is_undefined() {
                alloc::vec![b',']
            } else {
                str_arg(realm, sep_v)?.as_leaf_bytes()?.to_vec()
            };
            let elems: alloc::vec::Vec<NanBox> = dense_elements(realm, h)?.to_vec();
            // Sized for short elements, so a typical join grows at most once.
            let mut out = alloc::vec::Vec::with_capacity(elems.len() * (sep.len() + 3));
            for (i, e) in elems.into_iter().enumerate() {
                if i > 0 {
                    crate::wtf8::append(&mut out, &sep);
                }
                // Elements whose ToString runs no code: strings, numbers,
                // booleans, and `undefined`/`null` (which join as empty).
                match e.unpack() {
                    Unpacked::Undefined | Unpacked::Null => {}
                    Unpacked::Bool(b) => out.extend_from_slice(if b { b"true" } else { b"false" }),
                    Unpacked::Number(n) => {
                        crate::realm::builtins_fast::push_number_bytes(&mut out, n);
                    }
                    Unpacked::Handle(_) => {
                        let part = str_arg(realm, e)?;
                        crate::wtf8::append(&mut out, part.as_leaf_bytes()?);
                    }
                }
            }
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::ArrSlice => {
            // `ArraySpeciesCreate` must be the plain `ArrayCreate`.
            if !super::array_hof::default_species(realm, h) {
                return None;
            }
            let elems = dense_elements(realm, h)?;
            let len = elems.len();
            let a = rel_index(arg(0), len, 0)?;
            let b = rel_index(arg(1), len, len)?;
            let part = elems.get(a..b.max(a)).unwrap_or(&[]).to_vec();
            NanBox::handle(realm.new_array(part).to_raw())
        }
        FastOp::StrCase(upper) => {
            let rope = str_parts(realm, h)?;
            let out = crate::nbexec::case_map_wtf8(rope.as_leaf_bytes()?, upper);
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::StrTrim(start, end) => {
            let rope = str_parts(realm, h)?;
            let bytes = rope.as_leaf_bytes()?;
            let out = crate::nbexec::trim_wtf8(bytes, start, end);
            if out.len() == bytes.len() {
                recv
            } else {
                let out = out.to_vec();
                NanBox::handle(realm.new_string_wtf8(out).to_raw())
            }
        }
        FastOp::StrPad(at_start) => {
            let target = length_arg(arg(0))?;
            let fill_v = arg(1);
            let fill = if fill_v.is_undefined() {
                None
            } else {
                Some(str_arg(realm, fill_v)?)
            };
            let rope = str_parts(realm, h)?;
            // A huge target is the host's (its length check throws).
            if target > rope.utf16_len() + (1 << 20) {
                return None;
            }
            let pad: &[u8] = match &fill {
                None => b" ",
                Some(f) => f.as_leaf_bytes()?,
            };
            let out = crate::nbexec::pad_units(rope.as_leaf_bytes()?, target, pad, at_start);
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::StrSplit => {
            // A non-empty string separator that cannot split a surrogate pair,
            // and no limit.
            if !arg(1).is_undefined() {
                return None;
            }
            let sep = str_arg(realm, arg(0))?;
            let sep_b = sep.as_leaf_bytes()?;
            if sep_b.is_empty() || crate::nbexec::split_sensitive(sep_b) {
                return None;
            }
            let rope = str_parts(realm, h)?;
            // `split_units`' byte scan (exact for a separator that cannot
            // split a pair), with each piece copied once, straight into its
            // string (inline when short).
            let bytes = rope.as_leaf_bytes()?;
            let n = sep_b.len();
            let mut parts = alloc::vec::Vec::new();
            let (mut start, mut i) = (0usize, 0usize);
            while i + n <= bytes.len() {
                if bytes[i] == sep_b[0] && &bytes[i..i + n] == sep_b {
                    let piece = crate::rope::Rope::from_bytes(&bytes[start..i]);
                    parts.push(NanBox::handle(realm.new_string_from_rope(piece).to_raw()));
                    i += n;
                    start = i;
                } else {
                    i += 1;
                }
            }
            let piece = crate::rope::Rope::from_bytes(&bytes[start..]);
            parts.push(NanBox::handle(realm.new_string_from_rope(piece).to_raw()));
            NanBox::handle(realm.new_array(parts).to_raw())
        }
        FastOp::StrReplace => {
            // A string pattern (no `@@replace` is consulted for a primitive)
            // and a string replacement with no `$` substitution.
            let pat = str_arg(realm, arg(0))?;
            let rep = str_arg(realm, arg(1))?;
            let (pat_b, rep_b) = (pat.as_leaf_bytes()?, rep.as_leaf_bytes()?);
            if rep_b.contains(&b'$') || crate::nbexec::split_sensitive(pat_b) {
                return None;
            }
            let rope = str_parts(realm, h)?;
            let bytes = rope.as_leaf_bytes()?;
            let at = if pat_b.is_empty() {
                Some(0)
            } else {
                bytes.windows(pat_b.len()).position(|w| w == pat_b)
            };
            match at {
                None => recv,
                Some(i) => {
                    let mut out = alloc::vec::Vec::with_capacity(bytes.len() + rep_b.len());
                    out.extend_from_slice(&bytes[..i]);
                    crate::wtf8::append(&mut out, rep_b);
                    crate::wtf8::append(&mut out, &bytes[i + pat_b.len()..]);
                    NanBox::handle(realm.new_string_wtf8(out).to_raw())
                }
            }
        }
        FastOp::ObjKeys | FastOp::ObjEntries => {
            let o = arg(0).as_handle().map(Handle::from_raw)?;
            let keys = realm.plain_enumerable_key_ropes(o)?;
            let mut out = alloc::vec::Vec::with_capacity(keys.len());
            for k in keys.iter() {
                let kv = NanBox::handle(realm.new_string_from_rope(k.clone()).to_raw());
                out.push(if op == FastOp::ObjKeys {
                    kv
                } else {
                    let v = realm.get_property(o, k.as_leaf_str()?)?;
                    NanBox::handle(realm.new_array(alloc::vec![kv, v]).to_raw())
                });
            }
            NanBox::handle(realm.new_array(out).to_raw())
        }
    };
    Some(Ok(v))
}

/// Whether `v` is a primitive whose conversions run no code and cannot throw:
/// anything but an object, a Symbol or a BigInt (a string is a heap handle).
fn plain_primitive(realm: &Realm, v: NanBox) -> bool {
    v.as_handle()
        .is_none_or(|h| realm.is_string_handle(Handle::from_raw(h)))
}

/// `f(...args)` for a plain call of one of the global conversion built-ins
/// (`String`, `Number`, `Boolean`, `isNaN`, `isFinite`, `parseInt`,
/// `parseFloat`) whose arguments are [plain primitives](plain_primitive),
/// computed here instead of through the host: the very steps the host's
/// `call_native` takes for such arguments (no user code can run, nothing can
/// throw, and the results are realm-independent). `None` for anything else —
/// the caller takes the host's general path. A multi-realm run is left alone.
pub(super) fn try_call_value(realm: &mut Realm, f: NanBox, args: &[NanBox]) -> Option<NanBox> {
    use crate::nbexec::{
        N_BOOLEAN, N_IS_FINITE, N_IS_NAN, N_NUMBER, N_PARSE_FLOAT, N_PARSE_INT, N_STRING,
    };
    let id = realm.native_at(Handle::from_raw(f.as_handle()?))?;
    if realm.multi_realm {
        return None;
    }
    let a0 = args.first().copied().unwrap_or(NanBox::undefined());
    Some(match id {
        // `Boolean(v)` is ToBoolean, which never runs code.
        N_BOOLEAN => NanBox::boolean(realm.truthy(a0)),
        N_STRING if args.is_empty() => NanBox::handle(realm.new_string("").to_raw()),
        N_STRING if plain_primitive(realm, a0) => {
            if a0.as_handle().is_some() {
                a0
            } else {
                // A number shares its rope (or a digit's interned cell).
                NanBox::handle(realm.new_primitive_string(a0).to_raw())
            }
        }
        N_NUMBER if args.is_empty() => NanBox::number(0.0),
        N_NUMBER if plain_primitive(realm, a0) => NanBox::number(realm.to_number(a0)),
        N_IS_NAN if plain_primitive(realm, a0) => NanBox::boolean(realm.to_number(a0).is_nan()),
        N_IS_FINITE if plain_primitive(realm, a0) => {
            NanBox::boolean(realm.to_number(a0).is_finite())
        }
        N_PARSE_FLOAT if plain_primitive(realm, a0) => {
            let s = realm.to_display_string(a0);
            NanBox::number(crate::nbexec::parse_float_prefix(
                s.trim_matches(crate::realm::is_js_whitespace),
            ))
        }
        N_PARSE_INT if plain_primitive(realm, a0) => {
            let radix = match args.get(1).copied() {
                None => 0,
                Some(r) if matches!(r.unpack(), Unpacked::Undefined) => 0,
                Some(r) if plain_primitive(realm, r) => {
                    i64::from(realm.to_int32(NanBox::number(realm.to_number(r))))
                }
                Some(_) => return None,
            };
            let s = realm.to_display_string(a0);
            if radix != 0 && !(2..=36).contains(&radix) {
                NanBox::number(f64::NAN)
            } else {
                NanBox::number(crate::nbexec::parse_int(&s, radix as u32))
            }
        }
        _ => return None,
    })
}
