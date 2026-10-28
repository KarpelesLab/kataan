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
    N_ARRAY_IS_ARRAY, N_ARRAY_PROTO_FN, N_MAP_PROTO_FN, N_OBJ_PROTO_HASOWN, N_REGEXP_PROTO_FN,
    N_SET_PROTO_FN, N_STATIC_METHOD, N_STRING_PROTO_FN,
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
    /// `Array.prototype.pop`.
    ArrayPop,
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
        _ => {}
    }
    let Some((id, target)) = realm.bound_native_at(f) else {
        return FastOp::None;
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
            _ => FastOp::None,
        };
    }
    let Some(name) = realm.string_leaf_bytes(target) else {
        return FastOp::None;
    };
    match (id, name) {
        (N_ARRAY_PROTO_FN, b"push") => FastOp::ArrayPush,
        (N_ARRAY_PROTO_FN, b"pop") => FastOp::ArrayPop,
        (N_STRING_PROTO_FN, b"indexOf") => FastOp::StrIndexOf,
        (N_STRING_PROTO_FN, b"charCodeAt") => FastOp::StrCharCodeAt,
        (N_STRING_PROTO_FN, b"charAt") => FastOp::StrCharAt,
        (N_MAP_PROTO_FN, b"get") => FastOp::MapGet,
        (N_MAP_PROTO_FN, b"set") => FastOp::MapSet,
        (N_MAP_PROTO_FN, b"has") => FastOp::MapHas,
        (N_SET_PROTO_FN, b"add") => FastOp::SetAdd,
        (N_SET_PROTO_FN, b"has") => FastOp::SetHas,
        (N_REGEXP_PROTO_FN, b"test" | b"exec") => FastOp::RegExpCall,
        (N_STRING_PROTO_FN, b"slice") => FastOp::StrSlice,
        (N_STRING_PROTO_FN, b"substring") => FastOp::StrSubstring,
        (N_STRING_PROTO_FN, b"startsWith") => FastOp::StrStartsWith,
        (N_STRING_PROTO_FN, b"endsWith") => FastOp::StrEndsWith,
        (N_STRING_PROTO_FN, b"includes") => FastOp::StrIncludes,
        // `Function.prototype`'s methods share the Array methods' dispatch id.
        (N_ARRAY_PROTO_FN, b"call") => FastOp::FnCall,
        (N_ARRAY_PROTO_FN, b"apply") => FastOp::FnApply,
        (N_ARRAY_PROTO_FN, n) => {
            super::array_hof::hof_for(n).map_or(FastOp::None, FastOp::ArrayHof)
        }
        _ => FastOp::None,
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
    if !realm.is_string_handle(h) {
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
        FastOp::ArrayPop => {
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
        FastOp::StrSlice | FastOp::StrSubstring => {
            let rope = str_parts(realm, h)?;
            let (bytes, units) = (rope.as_leaf_bytes()?, rope.utf16_len());
            let start = int_arg(arg(0))?.unwrap_or(0.0);
            let end = int_arg(arg(1))?;
            let (a, b) = if site.op == FastOp::StrSlice {
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
            let out = crate::wtf8::slice_utf16_ascii(bytes, a, b, rope.is_ascii());
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::StrStartsWith | FastOp::StrEndsWith | FastOp::StrIncludes => {
            // A string search argument is not a RegExp, and converts as itself.
            let needle = str_arg(realm, arg(0))?;
            let rope = str_parts(realm, h)?;
            let (bytes, units) = (rope.as_leaf_bytes()?, rope.utf16_len());
            let pat = needle.as_leaf_bytes()?;
            let default = if site.op == FastOp::StrEndsWith {
                units
            } else {
                0
            };
            let pos = int_arg(arg(1))?.map_or(default, |n| (n.max(0.0) as usize).min(units));
            NanBox::boolean(match site.op {
                FastOp::StrStartsWith => {
                    let start = crate::nbexec::unit_to_byte(bytes, pos);
                    bytes.len() - start >= pat.len() && bytes[start..start + pat.len()] == *pat
                }
                FastOp::StrEndsWith => {
                    let end = crate::nbexec::unit_to_byte(bytes, pos);
                    end >= pat.len() && bytes[end - pat.len()..end] == *pat
                }
                _ => crate::nbexec::index_of_units(bytes, pat, pos, rope.is_ascii()) >= 0.0,
            })
        }
        FastOp::StrFromCharCode => {
            // ToNumber of a Number runs no code; ToUint16 as the host does.
            let mut units = alloc::vec::Vec::with_capacity(args.len());
            for r in args {
                let n = regs[*r as usize].as_number()?;
                units.push(if n.is_finite() {
                    (n as i64).rem_euclid(65536) as u16
                } else {
                    0
                });
            }
            let out = crate::wtf8::from_utf16(&units);
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
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
            let argv: alloc::vec::Vec<NanBox> = if site.op == FastOp::FnCall {
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
            return Some(call_vm(ctx, funcs, regs, recv, this, &argv));
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
