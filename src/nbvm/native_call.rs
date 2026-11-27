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

use super::{Ctx, FnProto, Reg, VmError, with_host, with_host_regs};
use crate::heap::Handle;
use crate::ic::PropertyCache;
use crate::nanbox::{NanBox, Unpacked};
use crate::nbexec::{
    N_ARRAY_IS_ARRAY, N_ARRAY_PROTO_FN, N_DATA_VIEW_PROTO_FN, N_DATE_PROTO_FN, N_GEN_NEXT,
    N_GEN_RETURN, N_GEN_THROW, N_JSON_PARSE, N_JSON_STRINGIFY, N_MAP_PROTO_FN, N_NUMBER,
    N_NUMBER_PROTO_FN, N_OBJ_PROTO_HASOWN, N_OBJECT_ASSIGN, N_OBJECT_CREATE, N_OBJECT_ENTRIES,
    N_OBJECT_GET_OWN_NAMES, N_OBJECT_GET_PROTO, N_OBJECT_HAS_OWN, N_OBJECT_KEYS, N_OBJECT_VALUES,
    N_REFLECT_GET_PROTO, N_REFLECT_HAS, N_REGEXP_PROTO_FN, N_SET_PROTO_FN, N_STATIC_METHOD,
    N_STRING_PROTO_FN, N_WEAKMAP_PROTO_FN, N_WEAKREF_DEREF, N_WEAKSET_PROTO_FN,
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
    /// `Array.prototype.unshift`.
    ArrayUnshift,
    /// `Array.prototype.splice`.
    ArraySplice,
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
    /// A `WeakMap.prototype` / `WeakSet.prototype` method (`get`, `set`,
    /// `has`, `delete` / `add`, `has`, `delete`), named by the `Map`/`Set`
    /// operation it shares ([`Realm::weak_coll_fast`]).
    WeakColl(CollOp),
    /// `WeakRef.prototype.deref`.
    WeakRefDeref,
    /// A `%TypedArray%.prototype` method with a lean form
    /// (`Realm::ta_method`).
    TypedArr(crate::realm::typed_methods::TaMethod),
    /// `Date.UTC` (served for Number arguments).
    DateUtc,
    /// `BigInt.asUintN` / (`true`) `BigInt.asIntN`, served for an integral
    /// Number width (`ToIndex` the identity) and a BigInt value.
    BigIntAsN(bool),
    /// `Object.fromEntries` over a plain array of plain `[key, value]`
    /// arrays (`obj_from_entries_fast`).
    ObjFromEntries,
    /// A `Date.prototype` getter.
    DateGet(crate::realm::date_fields::DateGet),
    /// A `Date.prototype` component setter (served for Number arguments).
    DateSet(crate::realm::date_fields::DateSet),
    /// `Object.prototype.hasOwnProperty`.
    ObjHasOwn,
    /// `RegExp.prototype.test` / `exec`: the resolved function called through
    /// the host directly (no by-name re-resolution).
    RegExpCall,
    /// `String.prototype.slice`.
    StrSlice,
    /// `String.prototype.substring`.
    StrSubstring,
    /// `String.prototype.substr` (Annex B.2.2.1).
    StrSubstr,
    /// `String.prototype.startsWith`.
    StrStartsWith,
    /// `String.prototype.endsWith`.
    StrEndsWith,
    /// `String.prototype.includes`.
    StrIncludes,
    /// `String.fromCharCode`.
    StrFromCharCode,
    /// `Number.isInteger` / `isSafeInteger` / `isFinite` / `isNaN`: a
    /// predicate of its argument that converts nothing (anything but a
    /// Number is `false`).
    NumPred(NumPred),
    /// `Array.isArray`.
    ArrayIsArray,
    /// `Promise.resolve` (served for a primitive argument, see
    /// `promise_resolve_fast`; anything else is the host's call).
    PromiseResolve,
    /// `Function.prototype.call` on a VM function.
    FnCall,
    /// `Function.prototype.apply` on a VM function.
    FnApply,
    /// `Function.prototype.bind` of a VM function (`fn_bind`).
    FnBind,
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
    /// `Array.prototype.flat`.
    ArrFlat,
    /// `Array.prototype.concat`.
    ArrConcat,
    /// `Array.prototype.fill`.
    ArrFill,
    /// `Array.prototype.at`.
    ArrAt,
    /// `Array.prototype.toString` (whose `join` is the built-in one).
    ArrToString,
    /// `Number.prototype.toPrecision` on a Number primitive.
    NumToPrecision,
    /// `Array.prototype.keys` / `values` / `entries` (the iterator's kind:
    /// 0 keys, 1 values, 2 entries).
    ArrIter(u8),
    /// `Map.prototype` / `Set.prototype` (whether a Set's) `keys` / `values`
    /// / `entries` (the iterator's kind, as for [`FastOp::ArrIter`]).
    CollIter(bool, u8),
    /// `Map.prototype.forEach` / `Set.prototype.forEach` (whether a Set's),
    /// whose walk runs in the VM (`coll_hof`).
    CollForEach(bool),
    /// `Array.prototype.lastIndexOf`.
    ArrLastIndexOf,
    /// `Array.prototype.reverse`.
    ArrReverse,
    /// `String.prototype.at`.
    StrAt,
    /// `String.prototype.codePointAt`.
    StrCodePointAt,
    /// `String.prototype.concat`.
    StrConcat,
    /// `String.prototype.repeat`.
    StrRepeat,
    /// `String.prototype.lastIndexOf`.
    StrLastIndexOf,
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
    /// `String.prototype.match` (with a RegExp the host can match in one
    /// step; otherwise called through the host directly, as
    /// [`FastOp::HostDirect`]).
    StrMatch,
    /// `String.prototype.replaceAll` (with a RegExp and a VM function, its
    /// replacer calls run in the VM; otherwise called through the host
    /// directly, as [`FastOp::HostDirect`]).
    StrReplaceAll,
    /// `Object.keys`.
    ObjKeys,
    /// `Object.getOwnPropertyNames`.
    ObjNames,
    /// `Object.getPrototypeOf` / `Reflect.getPrototypeOf`.
    ObjGetProto,
    /// `Reflect.has`.
    ReflectHas,
    /// `Reflect.get`.
    ReflectGet,
    /// `Reflect.set`.
    ReflectSet,
    /// `Reflect.apply(f, this, list)`: `FnApply` with `f` its receiver.
    ReflectApply,
    /// `%IteratorHelperPrototype%.next` (`iter_helper`).
    IterHelperNext,
    /// A consuming `Iterator.prototype` method (`iter_helper::consume`).
    IterConsume(super::iter_helper::Consume),
    /// `Object.entries`.
    ObjEntries,
    /// `Object.values`.
    ObjValues,
    /// `Object.hasOwn(o, k)`: `ObjHasOwn` with `o` the receiver.
    ObjStaticHasOwn,
    /// `Object.create(proto)` (no property descriptors).
    ObjCreate,
    /// `Array.from(items)` (no `mapFn`).
    ArrayFrom,
    /// `String.prototype.search`.
    StrSearch,
    /// `Object.assign(target, …sources)`: plain copies done in the VM
    /// ([`obj_assign_fast`]), anything else through the host directly (as
    /// [`FastOp::HostDirect`]).
    ObjAssign,
    /// A built-in `Math` function (its native id), computed by
    /// `math_fast::compute` on Number arguments.
    #[cfg(feature = "std")]
    Math(u16),
    /// A `DataView.prototype` `get*` / `set*` accessor (not BigInt or
    /// Float16), served by `nbexec::dataview_fast`.
    DataView(crate::nbexec::DvAccess),
    /// `%GeneratorPrototype%.next` / `throw` / `return` (the resumption
    /// mode, 0 / 1 / 2) on a VM generator: resumed in the VM (`gen_fast`).
    GenResume(u8),
    /// A built-in whose implementation is the host's (`JSON.stringify`,
    /// `JSON.parse`, and every built-in not served otherwise): the resolved
    /// function is called through the host directly, with no by-name
    /// re-resolution.
    HostDirect,
}

/// Which `Number` static predicate a [`FastOp::NumPred`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumPred {
    Integer,
    SafeInteger,
    Finite,
    NaN,
}

/// IsIntegralNumber(`n`): finite, and its own truncation — the host's
/// `n % 1 == 0` without the software `fmod`. A double of magnitude at least
/// 2^52 has no fraction bits; below that, the conversion to `i64` truncates
/// exactly.
fn integral(n: f64) -> bool {
    const BIG: f64 = 4_503_599_627_370_496.0;
    n.is_finite() && (!(-BIG..=BIG).contains(&n) || (n as i64) as f64 == n)
}

impl NumPred {
    /// The predicate on `v` (the host's `Number.*` static on it).
    fn test(self, v: NanBox) -> bool {
        v.as_number().is_some_and(|n| match self {
            NumPred::Integer => integral(n),
            NumPred::SafeInteger => {
                // (`|n| <= 2^53 - 1`, without `f64::abs`, a std method.)
                integral(n) && (-9_007_199_254_740_991.0..=9_007_199_254_740_991.0).contains(&n)
            }
            NumPred::Finite => n.is_finite(),
            NumPred::NaN => n.is_nan(),
        })
    }
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
    /// [`try_call`]'s second way: the resolution this one replaced, for a
    /// site whose receivers alternate between two prototype chains that both
    /// reach a built-in served here (`gen.next()` over the generators of two
    /// generator functions, each with its own `prototype`). A hit on it
    /// swaps the two, so the fields above are always the latest hit's.
    alt: Option<Box<AltWay>>,
}

/// A [`NativeSite`]'s other way.
pub(crate) struct AltWay {
    chain: MethodSiteCache,
    method: u64,
    op: FastOp,
}

impl NativeSite {
    fn new() -> Self {
        Self {
            chain: MethodSiteCache::default(),
            method: 0,
            op: FastOp::None,
            fills: 0,
            alt: None,
        }
    }

    /// Exchanges the site's resolution with its other way's.
    fn swap_alt(&mut self) {
        if let Some(alt) = self.alt.as_deref_mut() {
            core::mem::swap(&mut self.chain, &mut alt.chain);
            core::mem::swap(&mut self.method, &mut alt.method);
            core::mem::swap(&mut self.op, &mut alt.op);
        }
    }

    /// Keeps the current resolution (one that served a built-in) as the
    /// other way, before a refill overwrites it.
    fn keep_as_alt(&mut self) {
        if matches!(self.op, FastOp::None) {
            return;
        }
        let alt = self.alt.get_or_insert_with(|| {
            Box::new(AltWay {
                chain: MethodSiteCache::default(),
                method: 0,
                op: FastOp::None,
            })
        });
        core::mem::swap(&mut self.chain, &mut alt.chain);
        alt.method = self.method;
        alt.op = self.op;
    }
}

/// The bits of `%Function.prototype.call%` when the site `cache` resolved
/// its method to it (as [`try_call`] serves it).
#[cfg_attr(not(jit_native), allow(dead_code))]
pub(super) fn fn_call_method(cache: &PropertyCache) -> Option<u64> {
    cache
        .native
        .as_deref()
        .filter(|s| matches!(s.op, FastOp::FnCall))
        .map(|s| s.method)
}

/// Whether the function value `v` is the intrinsic `Set.prototype.add`.
pub(super) fn is_set_add(realm: &Realm, v: NanBox) -> bool {
    matches!(classify(realm, v), FastOp::SetAdd)
}

/// The function cell `v` as one of the built-ins served here.
fn classify(realm: &Realm, v: NanBox) -> FastOp {
    let Some(f) = v.as_handle().map(Handle::from_raw) else {
        return FastOp::None;
    };
    match realm.native_at(f) {
        Some(N_OBJ_PROTO_HASOWN) => return FastOp::ObjHasOwn,
        Some(N_ARRAY_IS_ARRAY) => return FastOp::ArrayIsArray,
        Some(crate::nbexec::N_ARRAY_FROM) => return FastOp::ArrayFrom,
        Some(N_OBJECT_KEYS) => return FastOp::ObjKeys,
        Some(N_OBJECT_GET_OWN_NAMES) => return FastOp::ObjNames,
        Some(N_OBJECT_GET_PROTO | N_REFLECT_GET_PROTO) => return FastOp::ObjGetProto,
        Some(N_REFLECT_HAS) => return FastOp::ReflectHas,
        Some(crate::nbexec::N_REFLECT_GET) => return FastOp::ReflectGet,
        Some(crate::nbexec::N_REFLECT_SET) => return FastOp::ReflectSet,
        Some(crate::nbexec::N_REFLECT_APPLY) => return FastOp::ReflectApply,
        Some(crate::nbexec::N_ITER_HELPER_NEXT) => return FastOp::IterHelperNext,
        Some(N_OBJECT_ENTRIES) => return FastOp::ObjEntries,
        Some(N_OBJECT_VALUES) => return FastOp::ObjValues,
        Some(N_OBJECT_HAS_OWN) => return FastOp::ObjStaticHasOwn,
        Some(N_OBJECT_CREATE) => return FastOp::ObjCreate,
        Some(N_OBJECT_ASSIGN) => return FastOp::ObjAssign,
        Some(N_JSON_STRINGIFY | N_JSON_PARSE) => return FastOp::HostDirect,
        Some(N_WEAKREF_DEREF) => return FastOp::WeakRefDeref,
        Some(crate::nbexec::N_OBJECT_FROM_ENTRIES) => return FastOp::ObjFromEntries,
        Some(N_GEN_NEXT) => return FastOp::GenResume(0),
        Some(N_GEN_THROW) => return FastOp::GenResume(1),
        Some(N_GEN_RETURN) => return FastOp::GenResume(2),
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
        let pair = realm.array_elements(target);
        let ctor = pair
            .and_then(|p| p.first().copied())
            .and_then(NanBox::as_handle)
            .and_then(|c| realm.native_at(Handle::from_raw(c)));
        return match pair
            .and_then(|p| p.get(1).copied())
            .and_then(NanBox::as_handle)
            .and_then(|n| realm.string_leaf_bytes(Handle::from_raw(n)))
        {
            Some(b"fromCharCode") => FastOp::StrFromCharCode,
            // (The host dispatches these by the bound constructor's native id.)
            Some(b"isInteger") if ctor == Some(N_NUMBER) => FastOp::NumPred(NumPred::Integer),
            Some(b"isSafeInteger") if ctor == Some(N_NUMBER) => {
                FastOp::NumPred(NumPred::SafeInteger)
            }
            Some(b"isFinite") if ctor == Some(N_NUMBER) => FastOp::NumPred(NumPred::Finite),
            Some(b"isNaN") if ctor == Some(N_NUMBER) => FastOp::NumPred(NumPred::NaN),
            Some(b"resolve") if ctor == Some(crate::nbexec::N_PROMISE) => FastOp::PromiseResolve,
            Some(b"UTC") if ctor == Some(crate::nbexec::N_DATE) => FastOp::DateUtc,
            Some(b"asUintN") if ctor == Some(crate::nbexec::N_BIGINT) => FastOp::BigIntAsN(false),
            Some(b"asIntN") if ctor == Some(crate::nbexec::N_BIGINT) => FastOp::BigIntAsN(true),
            _ => FastOp::HostDirect,
        };
    }
    let Some(name) = realm.string_leaf_bytes(target) else {
        return FastOp::None;
    };
    match (id, name) {
        (crate::nbexec::N_ITERATOR_PROTO_FN, n)
            if let Some(c) = super::iter_helper::Consume::of(n) =>
        {
            FastOp::IterConsume(c)
        }
        (N_DATA_VIEW_PROTO_FN, n) => {
            crate::nbexec::dv_access_any(n).map_or(FastOp::HostDirect, FastOp::DataView)
        }
        (N_ARRAY_PROTO_FN, b"push") => FastOp::ArrayPush,
        (N_ARRAY_PROTO_FN, b"pop") => FastOp::ArrayPop,
        (N_ARRAY_PROTO_FN, b"shift") => FastOp::ArrayShift,
        (N_ARRAY_PROTO_FN, b"unshift") => FastOp::ArrayUnshift,
        (N_ARRAY_PROTO_FN, b"splice") => FastOp::ArraySplice,
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
        (N_DATE_PROTO_FN, n) => match core::str::from_utf8(n) {
            Ok(n) if let Some(g) = crate::realm::date_fields::DateGet::of(n) => FastOp::DateGet(g),
            Ok(n) if let Some(st) = crate::realm::date_fields::DateSet::of(n) => {
                FastOp::DateSet(st)
            }
            _ => FastOp::HostDirect,
        },
        (crate::nbexec::N_TYPED_ARRAY_PROTO_FN, n) => crate::realm::typed_methods::TaMethod::of(n)
            .map_or(FastOp::HostDirect, FastOp::TypedArr),
        (N_WEAKMAP_PROTO_FN, b"get") => FastOp::WeakColl(CollOp::MapGet),
        (N_WEAKMAP_PROTO_FN, b"set") => FastOp::WeakColl(CollOp::MapSet),
        (N_WEAKMAP_PROTO_FN, b"has") => FastOp::WeakColl(CollOp::MapHas),
        (N_WEAKMAP_PROTO_FN, b"delete") => FastOp::WeakColl(CollOp::MapDelete),
        (N_WEAKSET_PROTO_FN, b"add") => FastOp::WeakColl(CollOp::SetAdd),
        (N_WEAKSET_PROTO_FN, b"has") => FastOp::WeakColl(CollOp::SetHas),
        (N_WEAKSET_PROTO_FN, b"delete") => FastOp::WeakColl(CollOp::SetDelete),
        (N_REGEXP_PROTO_FN, b"test" | b"exec") => FastOp::RegExpCall,
        (N_STRING_PROTO_FN, b"slice") => FastOp::StrSlice,
        (N_STRING_PROTO_FN, b"substring") => FastOp::StrSubstring,
        (N_STRING_PROTO_FN, b"substr") => FastOp::StrSubstr,
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
        (N_STRING_PROTO_FN, b"match") => FastOp::StrMatch,
        (N_STRING_PROTO_FN, b"search") => FastOp::StrSearch,
        (N_STRING_PROTO_FN, b"replaceAll") => FastOp::StrReplaceAll,
        #[cfg(feature = "std")]
        (N_NUMBER_PROTO_FN, b"toFixed") => FastOp::NumToFixed,
        (N_NUMBER_PROTO_FN, b"toString") => FastOp::NumToString,
        (N_ARRAY_PROTO_FN, b"indexOf") => FastOp::ArrIndexOf,
        (N_ARRAY_PROTO_FN, b"includes") => FastOp::ArrIncludes,
        (N_ARRAY_PROTO_FN, b"join") => FastOp::ArrJoin,
        (N_ARRAY_PROTO_FN, b"slice") => FastOp::ArrSlice,
        (N_ARRAY_PROTO_FN, b"flat") => FastOp::ArrFlat,
        (N_ARRAY_PROTO_FN, b"concat") => FastOp::ArrConcat,
        (N_ARRAY_PROTO_FN, b"fill") => FastOp::ArrFill,
        (N_ARRAY_PROTO_FN, b"at") => FastOp::ArrAt,
        (N_ARRAY_PROTO_FN, b"toString") => FastOp::ArrToString,
        (N_NUMBER_PROTO_FN, b"toPrecision") => FastOp::NumToPrecision,
        (N_MAP_PROTO_FN, b"forEach") => FastOp::CollForEach(false),
        (N_SET_PROTO_FN, b"forEach") => FastOp::CollForEach(true),
        (N_MAP_PROTO_FN, b"keys") => FastOp::CollIter(false, 0),
        (N_MAP_PROTO_FN, b"values") => FastOp::CollIter(false, 1),
        (N_MAP_PROTO_FN, b"entries") => FastOp::CollIter(false, 2),
        (N_SET_PROTO_FN, b"keys") => FastOp::CollIter(true, 0),
        (N_SET_PROTO_FN, b"values") => FastOp::CollIter(true, 1),
        (N_SET_PROTO_FN, b"entries") => FastOp::CollIter(true, 2),
        (N_ARRAY_PROTO_FN, b"keys") => FastOp::ArrIter(0),
        (N_ARRAY_PROTO_FN, b"values") => FastOp::ArrIter(1),
        (N_ARRAY_PROTO_FN, b"entries") => FastOp::ArrIter(2),
        (N_ARRAY_PROTO_FN, b"lastIndexOf") => FastOp::ArrLastIndexOf,
        (N_ARRAY_PROTO_FN, b"reverse") => FastOp::ArrReverse,
        (N_STRING_PROTO_FN, b"at") => FastOp::StrAt,
        (N_STRING_PROTO_FN, b"codePointAt") => FastOp::StrCodePointAt,
        (N_STRING_PROTO_FN, b"concat") => FastOp::StrConcat,
        (N_STRING_PROTO_FN, b"repeat") => FastOp::StrRepeat,
        (N_STRING_PROTO_FN, b"lastIndexOf") => FastOp::StrLastIndexOf,
        // `Function.prototype`'s methods share the Array methods' dispatch id.
        (N_ARRAY_PROTO_FN, b"call") => FastOp::FnCall,
        (N_ARRAY_PROTO_FN, b"apply") => FastOp::FnApply,
        (N_ARRAY_PROTO_FN, b"bind") => FastOp::FnBind,
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
    !matches!(site.op, FastOp::None)
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

/// `FlattenIntoArray(target, h, len, 0, depth)` for an array whose every
/// level is dense (each index an own data element, no hole, nothing
/// inherited to consult): `HasProperty` is then true for every index and
/// each `Get` reads the element, so the elements are copied, those that
/// are Arrays spliced in while `depth` lasts. `None`, having appended
/// what it had, when some level is not (the caller discards `out`) — or
/// holds a Proxy, which `IsArray` may see through to run traps.
fn flatten_dense(
    realm: &Realm,
    h: Handle,
    depth: usize,
    out: &mut alloc::vec::Vec<NanBox>,
) -> Option<()> {
    for &v in dense_elements(realm, h)? {
        if depth > 0
            && let Some(e) = v.as_handle().map(Handle::from_raw)
        {
            if realm.proxy_at(e).is_some() {
                return None;
            }
            if realm.is_array(e) {
                flatten_dense(realm, e, depth - 1, out)?;
                continue;
            }
        }
        out.push(v);
    }
    Some(())
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
/// The resumption mode and method of a site whose built-in is a generator's
/// `next` / `throw` / `return` (`FastOp::GenResume`).
#[cfg(all(feature = "jit", jit_native))]
#[inline]
pub(super) fn gen_resume_site(cache: &PropertyCache) -> Option<(u8, u64)> {
    match cache.native.as_deref() {
        Some(NativeSite {
            op: FastOp::GenResume(mode),
            method,
            ..
        }) => Some((*mode, *method)),
        _ => None,
    }
}

/// [`gen_resume_site`]'s mode.
#[cfg(all(feature = "jit", jit_native))]
#[inline]
pub(super) fn gen_resume_mode(cache: &PropertyCache) -> Option<u8> {
    gen_resume_site(cache).map(|(m, _)| m)
}

pub(super) fn is_native_site(cache: &PropertyCache) -> bool {
    cache
        .native
        .as_deref()
        .is_some_and(|s| !matches!(s.op, FastOp::None))
}

/// Whether function `proto`'s call op at `pc` was never seen calling a
/// JavaScript function: neither the interpreter's cache of the site (when
/// it has one) nor its compiled body's call site (when it has one) names a
/// callee (`CalleeHint::None`: a built-in it served, such as `m.has(k)` or
/// `a.push(x)`, or a call not made yet, such as a `throw new Error(...)`).
/// Such a call pushes no activation (`crate::jit::inline` inlines a
/// looping callee whose calls are all such). Only a guess at what later
/// calls do: any call op inlined is compiled with its full slow path.
/// `false` when neither knows anything (a copy of the function, whose tier
/// state starts empty).
#[cfg(all(feature = "jit", jit_native))]
pub(crate) fn no_script_callee(proto: &FnProto, pc: usize) -> bool {
    use crate::ic::CalleeHint;
    let mut known = false;
    if let Ok(t) = proto.tier.0.try_borrow()
        && let Some(c) = t.ic.get(pc)
    {
        if crate::jit::poly::hint_of(c) != CalleeHint::None {
            return false;
        }
        known = true;
    }
    if let Some(Some(j)) = proto.tier.jit()
        && j.is_generic()
        && let Some(h) = j.call_hints().get(pc)
    {
        if *h != CalleeHint::None {
            return false;
        }
        known = true;
    }
    known
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
    // A flat plain string (the common case): its rope, in one lookup.
    if str_parts_lean_on()
        && let Some(rope) = realm.flat_plain_string_rope(h)
    {
        return Some(rope);
    }
    // A string cell with a function marker is a function, not a string.
    if !realm.typeof_is_string(h) {
        return None;
    }
    realm.flatten_string(h);
    let rope = realm.string_rope(h)?;
    rope.as_leaf_bytes()?;
    Some(rope)
}

/// Whether [`str_parts`] serves a flat plain string in one cell lookup
/// (`KATAAN_STR_PARTS_LEAN=0` takes the general steps; read once per
/// process).
fn str_parts_lean_on() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_STR_PARTS_LEAN").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// A string argument, flattened (`ToString` is then the identity).
fn str_arg(realm: &mut Realm, v: NanBox) -> Option<crate::rope::Rope> {
    let h = v.as_handle().map(Handle::from_raw)?;
    str_parts(realm, h)
}

/// The storage key of a `Reflect` key argument whose ToPropertyKey runs no
/// code: a String (escaped into the user namespace) or a Symbol.
fn reflect_key(realm: &Realm, k: NanBox) -> Option<alloc::string::String> {
    let kh = Handle::from_raw(k.as_handle()?);
    if let Some(id) = realm.symbol_id(kh) {
        return Some(crate::shape::symbol_key(id));
    }
    realm.string_key(kh)
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
#[allow(clippy::too_many_arguments)]
pub(super) fn try_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
    from_jit: bool,
) -> Option<Result<NanBox, VmError>> {
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    // A generator's `next` / `throw` / `return`: proved by the generator's
    // own memo (`gen_fast::site_call`), whose receivers are ever new objects.
    if let Some(site) = cache.native.as_deref()
        && let FastOp::GenResume(mode) = site.op
    {
        let method = site.method;
        let arg = args
            .first()
            .map_or(NanBox::undefined(), |r| regs[*r as usize]);
        // (Compiled code's register window is a GC root already.)
        #[cfg(all(feature = "jit", jit_native))]
        let rooted =
            from_jit && super::gen_fast::jit_site_on() && super::jit_tier::window_rooted(ctx);
        #[cfg(not(all(feature = "jit", jit_native)))]
        let rooted = {
            let _ = from_jit;
            false
        };
        if let Some(r) =
            super::gen_fast::site_call(ctx, funcs, regs, recv, key, mode, method, arg, rooted)
        {
            return Some(r);
        }
    }
    // A static method (`Object.hasOwn`, `Number.isInteger`): the receiver the
    // site recorded, owning the method itself, unchanged — the general
    // path's `site_hit` on it, in a few compares.
    if let Some(site) = cache.native.as_deref()
        && !matches!(site.op, FastOp::None)
        && let Some(r) = recv.as_handle().map(Handle::from_raw)
        && ctx.realm.method_site_own_hit(r, &site.chain, site.method)
    {
        let (op, method) = (site.op, site.method);
        return run_op(ctx, funcs, op, method, recv, r, false, regs, args);
    }
    if cache
        .native
        .as_deref()
        .is_some_and(|s| s.op.coll_op().is_some() || matches!(s.op, FastOp::WeakColl(_)))
        && let Some(v) = try_coll(ctx, recv, regs, args, cache)
    {
        return Some(Ok(v));
    }
    if let Some(v) = try_str_unit(ctx, recv, key, regs, args, cache) {
        return Some(Ok(v));
    }
    // A Number receiver owns no property: its lookup starts at
    // `%Number.prototype%`, which then stands in for it in the site's chain.
    // So does `%String.prototype%` for a plain string cell, and
    // `%Array.prototype%` for an array without side-table state, which own
    // only their indices and `length` (keys `cacheable_key` turns away):
    // their lookup is then the same few checks, with the receiver itself the
    // method's `this` (`this_h`).
    let (h, prim, this_h) = match recv.as_handle() {
        Some(r) => {
            let r = Handle::from_raw(r);
            match ctx.realm.plain_recv_lookup_start(r) {
                Some(start) if cacheable_key(key) => (start, true, Some(r)),
                _ => (r, false, None),
            }
        }
        None if recv.as_number().is_some() => (ctx.realm.primitive_proto(recv)?, true, None),
        None => return None,
    };
    let site = cache
        .native
        .get_or_insert_with(|| Box::new(NativeSite::new()));
    let mut hit = site_hit(&ctx.realm, site, h, prim, key);
    if !hit && site.alt.is_some() && site.fills < FILL_LIMIT {
        // The other way (see `NativeSite::alt`).
        site.swap_alt();
        hit = site_hit(&ctx.realm, site, h, prim, key);
        if !hit {
            site.swap_alt();
        }
    }
    if !hit {
        if site.fills >= FILL_LIMIT || !cacheable_key(key) {
            // Retired: a site that keeps changing its mind calls the general
            // path from now on.
            site.op = FastOp::None;
            site.alt = None;
            return None;
        }
        site.keep_as_alt();
        fill(&mut ctx.realm, site, h, prim, key);
        if matches!(site.op, FastOp::None) {
            return None;
        }
    } else if !prim {
        let objects = matches!(site.op, FastOp::GenResume(_));
        ctx.realm.method_site_note_recv(h, &mut site.chain, objects);
    }
    let (op, method) = (site.op, site.method);
    match this_h {
        Some(s) => run_op(ctx, funcs, op, method, recv, s, false, regs, args),
        None => run_op(ctx, funcs, op, method, recv, h, prim, regs, args),
    }
}

/// What compiled code's inline `s.charCodeAt(i)` relies on, for `site`: the
/// method a plain string's lookup finds, an own data property of
/// `%String.prototype%`, is the built-in (`MethodSite::set_str_proof`).
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_char_code_at(realm: &mut Realm, site: &mut crate::jit::MethodSite) {
    // (`%String.prototype%` is a String wrapper, whose string only owns
    // indices and `length`.)
    let start = realm.string_proto_intrinsic;
    prove_proto_method(realm, site, start, "charCodeAt", FastOp::StrCharCodeAt);
}

/// What compiled code's inline `a.push(x)` relies on, for `site`: the method
/// the lookup of an array without side-table state or a set `[[Prototype]]`
/// finds (`plain_recv_lookup_start`: an own data property of the realm's
/// first `%Array.prototype%`) is the built-in `push` (as
/// [`prove_char_code_at`]).
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_array_push(realm: &mut Realm, site: &mut crate::jit::MethodSite) {
    // (`%Array.prototype%` is an array, whose named properties are its
    // auxiliary object's: the object watched and read here. An array keeps
    // its auxiliary object for life, except across a cell swap, which bumps
    // the prototype epoch.)
    let start = realm
        .root_array_proto()
        .and_then(|p| match realm.jit_object(p) {
            Some(_) => Some(p),
            None => realm.jit_aux_of(p),
        });
    prove_proto_method(realm, site, start, "push", FastOp::ArrayPush);
}

/// What compiled code's inline `m.get(k)` / `m.has(k)` / `s.has(k)` (and
/// `m.set(k, v)` / `s.add(k)` / `.delete(k)`) relies on, for `site` and the
/// collection `recv` (`crate::jit::coll`): the proof
/// that a collection of `recv`'s kind with the default `[[Prototype]]`
/// finds the built-in. Whether it was made.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_coll(
    realm: &mut Realm,
    site: &mut crate::jit::MethodSite,
    recv: Handle,
) -> bool {
    let Some((is_set, start)) = realm.jit_coll_default(recv) else {
        return false;
    };
    let want = match (site.key(), is_set) {
        ("get", false) => FastOp::MapGet,
        ("has", false) => FastOp::MapHas,
        ("has", true) => FastOp::SetHas,
        ("set", false) => FastOp::MapSet,
        ("add", true) => FastOp::SetAdd,
        ("delete", false) => FastOp::MapDelete,
        ("delete", true) => FastOp::SetDelete,
        _ => return false,
    };
    site.fill_coll(realm, start, is_set, |realm, v| classify(realm, v) == want)
}

/// What compiled code's inline `DataView` accessor relies on, for `site`
/// and the view `recv` (`crate::jit::dataview`): its memo of the view, whose
/// method the lookup proves the built-in accessor `acc`. Whether it was
/// filled.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_dataview(
    realm: &mut Realm,
    site: &mut crate::jit::MethodSite,
    recv: Handle,
    acc: crate::nbexec::DvAccess,
) -> bool {
    let facts = crate::nbexec::dv_view_facts(realm, recv, acc.set);
    site.fill_dv(realm, recv, facts, |realm, v| {
        classify(realm, v) == FastOp::DataView(acc)
    })
}

/// What compiled code's direct call of an array higher-order built-in
/// relies on (`jit_tier::jit_helper_hof_call`), for `site`: as
/// [`prove_array_push`], of the built-in the site's key names.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_array_hof(realm: &mut Realm, site: &mut crate::jit::MethodSite) {
    let start = realm
        .root_array_proto()
        .and_then(|p| match realm.jit_object(p) {
            Some(_) => Some(p),
            None => realm.jit_aux_of(p),
        });
    let key = alloc::string::String::from(site.key());
    match super::array_hof::hof_for(key.as_bytes()) {
        Some(op) => prove_proto_method(realm, site, start, &key, FastOp::ArrayHof(op)),
        None => site.set_str_proof(None),
    }
}

/// What compiled code's inline `a.pop()` relies on, for `site`: as
/// [`prove_array_push`], of the built-in `pop`.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_array_pop(realm: &mut Realm, site: &mut crate::jit::MethodSite) {
    let start = realm
        .root_array_proto()
        .and_then(|p| match realm.jit_object(p) {
            Some(_) => Some(p),
            None => realm.jit_aux_of(p),
        });
    prove_proto_method(realm, site, start, "pop", FastOp::ArrayPop);
}

/// Whether the case mapping of a one-character ASCII string answers its
/// interned cell (`KATAAN_UNIT_CASE=0` makes a new one).
fn unit_case_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_UNIT_CASE").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// Whether `toUpperCase` / `toLowerCase` of an ASCII string map it in place
/// into its new node (or answer the receiver when nothing changes), and a
/// `split` that finds no separator answers the receiver as its one piece
/// (`KATAAN_ASCII_CASE=0` turns them off; read once per process).
fn ascii_case_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_ASCII_CASE").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// The `String.prototype` methods compiled code calls straight through
/// [`str_call`] (`crate::jit`'s direct string call): each a [`run_op`]
/// string op that runs no code (it declines whatever input could). The
/// code of `key`, or `None`.
#[cfg(all(feature = "jit", jit_native))]
pub(crate) fn str_call_code(key: &str) -> Option<u8> {
    Some(match key {
        "indexOf" => 1,
        "slice" => 2,
        "substring" => 3,
        "startsWith" => 4,
        "endsWith" => 5,
        "includes" => 6,
        "toUpperCase" => 7,
        "toLowerCase" => 8,
        "trim" => 9,
        "trimStart" => 10,
        "trimEnd" => 11,
        "padStart" => 12,
        "padEnd" => 13,
        "charAt" => 14,
        "split" => 15,
        "replace" => 16,
        "match" => 17,
        "replaceAll" => 18,
        "substr" => 19,
        "charCodeAt" => 20,
        "codePointAt" => 21,
        "at" => 22,
        _ => return None,
    })
}

/// The op of a [`str_call_code`].
#[cfg(all(feature = "jit", jit_native))]
fn str_call_op(code: u8) -> Option<FastOp> {
    Some(match code {
        1 => FastOp::StrIndexOf,
        2 => FastOp::StrSlice,
        3 => FastOp::StrSubstring,
        4 => FastOp::StrStartsWith,
        5 => FastOp::StrEndsWith,
        6 => FastOp::StrIncludes,
        7 => FastOp::StrCase(true),
        8 => FastOp::StrCase(false),
        9 => FastOp::StrTrim(true, true),
        10 => FastOp::StrTrim(true, false),
        11 => FastOp::StrTrim(false, true),
        12 => FastOp::StrPad(true),
        13 => FastOp::StrPad(false),
        14 => FastOp::StrCharAt,
        15 => FastOp::StrSplit,
        16 => FastOp::StrReplace,
        17 => FastOp::StrMatch,
        18 => FastOp::StrReplaceAll,
        19 => FastOp::StrSubstr,
        20 => FastOp::StrCharCodeAt,
        21 => FastOp::StrCodePointAt,
        22 => FastOp::StrAt,
        _ => return None,
    })
}

/// What compiled code's direct string call relies on, for `site` (whose
/// key has a [`str_call_code`]): as [`prove_char_code_at`], of that
/// built-in. Whether it was made (the site's key one and the method
/// found).
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_str_method(realm: &mut Realm, site: &mut crate::jit::MethodSite) -> bool {
    let Some(want) = str_call_code(site.key()).and_then(str_call_op) else {
        return false;
    };
    let start = realm.string_proto_intrinsic;
    let key = alloc::string::String::from(site.key());
    prove_proto_method(realm, site, start, &key, want);
    site.str_proof_found()
}

/// A direct string call from compiled code: `recv.key(args…)` with `recv`
/// a plain string cell whose lookup of the key the site proved finds the
/// built-in `method` (`code`'s, [`str_call_code`]): that op's answer
/// ([`run_op`]), or `None`, having done nothing, for the general call.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn str_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    regs: &[NanBox],
    recv: NanBox,
    args: &[Reg],
    method: u64,
    code: u8,
) -> Option<Result<NanBox, VmError>> {
    let lent = code & STR_CALL_LENT != 0;
    let op = str_call_op(code & !STR_CALL_LENT)?;
    let h = Handle::from_raw(recv.as_handle()?);
    if ctx.host.is_none() || ctx.realm.multi_realm || args.len() > 2 {
        return None;
    }
    #[cfg(feature = "regex")]
    if matches!(op, FastOp::StrReplace)
        && super::str_replace::lean_on()
        && let Some(v) = super::str_replace::try_never(ctx, regs, recv, h, args)
    {
        if lent {
            ctx.realm.regexp_lend_return(regs[args[0] as usize]);
        }
        return Some(Ok(v));
    }
    if matches!(op, FastOp::StrReplace)
        && super::str_replace::lean_on()
        && let Some(r) = super::str_replace::try_lean(ctx, funcs, regs, recv, h, args)
    {
        // The one-step algorithm ran: a lending literal's object reached
        // no code (see `STR_CALL_LENT`).
        if lent && r.is_ok() {
            ctx.realm.regexp_lend_return(regs[args[0] as usize]);
        }
        return Some(r);
    }
    // `s.match(/re/)` of a lending literal: the one-step match first (as
    // `run_op`'s), so its object goes back to the pool when it ran.
    if lent
        && matches!(op, FastOp::StrMatch)
        && let [a] = *args
        && ctx.realm.is_string_handle(h)
        && ctx.realm.typeof_is_string(h)
    {
        let rx = regs[a as usize];
        if rx
            .as_handle()
            .is_some_and(|r| ctx.realm.regexp_flags(Handle::from_raw(r)).is_some())
            && let Some(Some(r)) = with_host(ctx, |host| host.string_match(recv, rx))
        {
            if r.is_ok() {
                ctx.realm.regexp_lend_return(rx);
            }
            return Some(r.map_err(VmError::from));
        }
    }
    run_op(ctx, funcs, op, method, recv, h, false, regs, args)
}

/// The flag of a [`str_call`] code for a call whose first argument the op
/// right before it made with a lending regular expression literal
/// (`Op::NewRegExp`'s `lend`) and which nothing else enters (compiled code
/// proves that): once the call ran as the one-step algorithm over a
/// pristine RegExp — which hands the object to no code and gives it no
/// property — nothing can reach it, and it goes back to the realm's lending
/// pool (`Realm::regexp_lend_return`).
#[cfg(all(feature = "jit", jit_native))]
pub(crate) const STR_CALL_LENT: u8 = 0x80;

/// What compiled code's inline `a.shift()` relies on, for `site`: as
/// [`prove_array_push`], of the built-in `shift`.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn prove_array_shift(realm: &mut Realm, site: &mut crate::jit::MethodSite) {
    let start = realm
        .root_array_proto()
        .and_then(|p| match realm.jit_object(p) {
            Some(_) => Some(p),
            None => realm.jit_aux_of(p),
        });
    prove_proto_method(realm, site, start, "shift", FastOp::ArrayShift);
}

/// The proof of [`prove_char_code_at`] / [`prove_array_push`]: `key` is a
/// data property of the ordinary object `start` (watched), the built-in
/// `want`.
#[cfg(all(feature = "jit", jit_native))]
fn prove_proto_method(
    realm: &mut Realm,
    site: &mut crate::jit::MethodSite,
    start: Option<Handle>,
    key: &str,
    want: FastOp,
) {
    let found = start.and_then(|start| {
        let o = realm.jit_object(start)?;
        let shape = o.shaped_shape()?;
        if o.accessor_index(key).is_some() {
            return None;
        }
        let slot = shape.lookup(key)?;
        let ptr = o.jit_slot_ptr(slot)?;
        let value = o.slot(slot)?;
        Some((start, ptr as u64, value))
    });
    match found {
        Some((start, ptr, value)) if classify(realm, value) == want => {
            realm.jit_watch(start);
            site.set_str_proof(Some((ptr, value.to_bits())));
        }
        _ => site.set_str_proof(None),
    }
}

/// `s.charCodeAt(i)` / `s.charAt(i)` with `s` a flat all-ASCII string and `i`
/// a Number or absent, at a site already resolved to that intrinsic for a
/// string receiver (see [`try_call`]): [`run_op`]'s answer read straight off
/// the string's bytes. `None`, having done nothing, otherwise.
#[inline]
fn try_str_unit(
    ctx: &mut Ctx,
    recv: NanBox,
    key: &str,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
) -> Option<NanBox> {
    let site = cache.native.as_deref_mut()?;
    let op = site.op;
    if !matches!(op, FastOp::StrCharCodeAt | FastOp::StrCharAt) {
        return None;
    }
    let h = Handle::from_raw(recv.as_handle()?);
    let n = match args.first().map(|r| regs[*r as usize]) {
        Some(a) => a.as_number()?,
        None => 0.0,
    };
    let unit = ctx.realm.ascii_unit_at(h, n)?;
    let start = ctx.realm.plain_recv_lookup_start(h)?;
    if ctx
        .realm
        .method_site_get_prim(start, key, &mut site.chain)?
        .to_bits()
        != site.method
    {
        return None;
    }
    Some(match (op, unit) {
        (FastOp::StrCharCodeAt, Some(u)) => NanBox::number(f64::from(u)),
        (FastOp::StrCharCodeAt, None) => NanBox::number(f64::NAN),
        (_, Some(u)) => NanBox::handle(ctx.realm.unit_string(u).to_raw()),
        (_, None) => NanBox::handle(ctx.realm.new_string("").to_raw()),
    })
}

/// `recv.key(args…)` when the site last called a `Map.prototype` /
/// `Set.prototype` method and still proves that `recv[key]` is that
/// intrinsic (`Realm::coll_site_call`): the method's lean form
/// (`Realm::coll_fast`). `None`, having done nothing, otherwise. Never runs
/// code and never throws (a receiver of the wrong kind is a miss).
#[inline(never)]
pub(super) fn try_coll(
    ctx: &mut Ctx,
    recv: NanBox,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
) -> Option<NanBox> {
    try_coll_inline(ctx, recv, regs, args, cache)
}

/// [`try_coll`] in its caller's line (compiled code's helper is nothing but
/// this call).
#[inline(always)]
pub(super) fn try_coll_inline(
    ctx: &mut Ctx,
    recv: NanBox,
    regs: &[NanBox],
    args: &[Reg],
    cache: &mut PropertyCache,
) -> Option<NanBox> {
    let site = cache.native.as_deref_mut()?;
    let (op, weak) = match site.op {
        FastOp::WeakColl(op) => (op, true),
        other => (other.coll_op()?, false),
    };
    let h = Handle::from_raw(recv.as_handle()?);
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    if weak {
        // (An add of a non-object key is the general path's TypeError, or
        // a symbol's registry check.)
        if matches!(op, CollOp::MapSet | CollOp::SetAdd)
            && !super::is_object_value(&ctx.realm, arg(0))
        {
            return None;
        }
        return ctx.realm.coll_site_call::<true>(
            h,
            &mut site.chain,
            site.method,
            op,
            arg(0),
            arg(1),
        );
    }
    ctx.realm
        .coll_site_call::<false>(h, &mut site.chain, site.method, op, arg(0), arg(1))
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
    let site = cache
        .native
        .get_or_insert_with(|| Box::new(NativeSite::new()));
    if site.method != method.to_bits() {
        if site.fills >= FILL_LIMIT {
            site.op = FastOp::None;
            return None;
        }
        site.fills += 1;
        site.method = method.to_bits();
        site.op = classify(&ctx.realm, method);
    }
    if matches!(site.op, FastOp::None) {
        return None;
    }
    let (op, method) = (site.op, site.method);
    // A Map/Set method (the method value is already read): its lean form
    // straight away, as `run_op` ends up doing (a receiver of the wrong
    // kind is a miss, the general path's `TypeError`).
    if let Some(cop) = op.coll_op() {
        if prim {
            return None;
        }
        let arg = |i: usize| {
            args.get(i)
                .map_or(NanBox::undefined(), |r| regs[*r as usize])
        };
        return ctx.realm.coll_fast(h, cop, arg(0), arg(1)).map(Ok);
    }
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
    if let Some(v) = try_get_coll(ctx, recv, cache) {
        return Some(v);
    }
    try_get_general(ctx, recv, key, cache)
}

/// [`try_get`] at a site that last read a `Map.prototype` /
/// `Set.prototype` method, for a Map/Set receiver with its default
/// `[[Prototype]]` (`Realm::coll_site_method_hit`): the method, proved in a
/// few compares whichever collection the receiver is. `None`, having done
/// nothing, otherwise.
#[inline]
pub(super) fn try_get_coll(ctx: &Ctx, recv: NanBox, cache: &PropertyCache) -> Option<NanBox> {
    let site = cache.native.as_deref()?;
    site.op.coll_op()?;
    let h = Handle::from_raw(recv.as_handle()?);
    if ctx.host.is_none() || ctx.realm.multi_realm {
        return None;
    }
    ctx.realm
        .coll_site_method_hit(h, &site.chain, site.method)
        .then(|| NanBox::from_bits(site.method))
}

/// [`try_get`] without its collection shortcut ([`try_get_coll`]).
pub(super) fn try_get_general(
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
    let site = cache
        .native
        .get_or_insert_with(|| Box::new(NativeSite::new()));
    // (The proof `site_hit` / `fill` make for `try_call`, written out here so
    // that those stay `try_call`'s alone: inlined into it as before.)
    let hit = !matches!(site.op, FastOp::None)
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
        if matches!(site.op, FastOp::None) {
            return None;
        }
    }
    Some(NanBox::from_bits(site.method))
}

/// `Object.assign(target, …sources)` (the arguments are registers `args` of
/// `regs`) when every source is `null`, `undefined` or a plain copy onto the
/// plain object `target` (`Realm::assign_plain`, which runs no code and
/// cannot throw): `target`. `None` otherwise, possibly after copying some
/// leading sources: the caller then makes the whole call the general way,
/// whose repeat of those copies stores the same values under keys the
/// target already owns as plain data properties, which nothing observes.
fn obj_assign_fast(realm: &mut Realm, regs: &[NanBox], args: &[Reg]) -> Option<NanBox> {
    let (first, sources) = args.split_first()?;
    let target = regs[*first as usize];
    let t = Handle::from_raw(target.as_handle()?);
    // An ordinary object cell (ToObject is the identity; a primitive String,
    // Symbol or BigInt target is boxed, the host's).
    realm.object_key_epoch(t)?;
    if realm.has_argument_map_marker(t) {
        return None;
    }
    for r in sources {
        let s = regs[*r as usize];
        if s.is_undefined() || s.is_null() {
            continue;
        }
        let s = Handle::from_raw(s.as_handle()?);
        if realm.has_argument_map_marker(s) || !realm.assign_plain(t, s) {
            return None;
        }
    }
    Some(target)
}

/// [`run_op`] for an object receiver, not inlined: the `F.call(t, …)` form
/// re-enters it from inside itself, which would otherwise keep it out of
/// line in its callers.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn run_op_out_of_line(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    op: FastOp,
    method: u64,
    recv: NanBox,
    h: Handle,
    regs: &[NanBox],
    args: &[Reg],
) -> Option<Result<NanBox, VmError>> {
    run_op(ctx, funcs, op, method, recv, h, false, regs, args)
}

/// Whether `Promise.resolve(x)` of a primitive is served without the host
/// ([`promise_resolve_fast`], `KATAAN_PROMISE_RESOLVE`, default on).
fn promise_resolve_lean_on() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_PROMISE_RESOLVE").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    {
        true
    }
}

/// `Promise.resolve(x)` ([`FastOp::PromiseResolve`]) with the receiver `h`
/// the intrinsic `%Promise%` (with one realm, the only cell of that native)
/// and `x` a primitive held in the value itself (no string, symbol or
/// BigInt): PromiseResolve's `IsPromise(x)` is false, and
/// NewPromiseCapability(%Promise%) then resolving with a non-object only
/// makes a promise of the intrinsic `%Promise.prototype%` fulfilled with
/// `x` — no code runs, nothing is queued (it has no reactions yet). That
/// promise, made as the host makes it; `None`, having done nothing
/// observable, otherwise.
fn promise_resolve_fast(ctx: &mut Ctx, h: Handle, x: NanBox) -> Option<NanBox> {
    if !promise_resolve_lean_on()
        || ctx.realm.multi_realm
        || x.as_handle().is_some()
        || ctx.realm.native_at(h) != Some(crate::nbexec::N_PROMISE)
    {
        return None;
    }
    let p = super::vm_new_promise(ctx)?;
    super::settle_unwatched(ctx, p, x).then_some(p)
}

/// `Object.fromEntries(src)` when iterating `src` and reading its entries
/// runs no code: `src` a plain array whose iteration is the built-in one
/// (`iter_fast::plain_array_values`), each entry a plain array of at least
/// two present elements (so `Get(entry, "0")` / `Get(entry, "1")` read them)
/// whose key is a String, a Symbol or another primitive held in the value
/// (ToPropertyKey runs no code). The new object, made as the host makes it;
/// `None`, having done nothing observable, for anything else.
fn obj_from_entries_fast(realm: &mut Realm, src: NanBox) -> Option<NanBox> {
    let entries: alloc::vec::Vec<NanBox> =
        super::iter_fast::plain_array_values(realm, src)?.to_vec();
    let mut pairs: alloc::vec::Vec<(NanBox, NanBox)> =
        alloc::vec::Vec::with_capacity(entries.len());
    for e in entries {
        let (_, el) = realm.plain_root_array_elems(Handle::from_raw(e.as_handle()?))?;
        match el {
            [k, v, ..] if !k.is_hole() && !v.is_hole() => pairs.push((*k, *v)),
            _ => return None,
        }
        let k = pairs.last()?.0;
        if let Some(kh) = k.as_handle().map(Handle::from_raw)
            && !realm.is_string_handle(kh)
            && realm.symbol_at(kh).is_none()
        {
            return None;
        }
    }
    let obj = realm.new_object();
    for (k, v) in pairs {
        let key = match k
            .as_handle()
            .and_then(|h| realm.symbol_at(Handle::from_raw(h)))
        {
            Some((_, id)) => alloc::format!("\u{0}sym:{id}"),
            None => realm.display_key(k),
        };
        realm.set_property(obj, &key, v);
    }
    Some(NanBox::handle(obj.to_raw()))
}

/// `Object.prototype.hasOwnProperty` on the receiver `h` for the key value
/// `k` ([`FastOp::ObjHasOwn`]) when that runs no code; `None`, having done
/// nothing observable, otherwise.
fn obj_has_own_fast(realm: &mut Realm, h: Handle, k: NanBox) -> Option<bool> {
    // An ordinary object receiver (ToObject is the identity), and a key
    // whose ToPropertyKey runs no code: a string or another primitive
    // that is not a Symbol. (A host-exotic receiver — a module
    // namespace, whose [[GetOwnProperty]] may throw for a binding
    // still in its TDZ, which `.call` and `Object.hasOwn` reach —
    // is the general path's.)
    realm.object_key_epoch(h)?;
    if realm.is_host_exotic(h) {
        return None;
    }
    // A key String whose own slot in the receiver's layout is
    // memoized (a `for-in` key of its list, a key the String-key slot
    // memo names): an own data property.
    let o = NanBox::handle(h.to_raw());
    if realm.for_in_key_slot(o, k).is_some() || realm.key_memo_probe(o, k).is_some() {
        return Some(true);
    }
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
            owned = realm.display_key(k);
            &owned
        }
        Some(_) => return None,
        None => match k.unpack() {
            Unpacked::Number(_) | Unpacked::Bool(_) | Unpacked::Undefined | Unpacked::Null => {
                owned = realm.to_display_string(k);
                &owned
            }
            _ => return None,
        },
    };
    // (A plain ordinary object's own keys are its own; `has_own`'s
    // String-wrapper and function-metadata checks are for others.)
    let own = realm
        .ordinary_has_own(h, key)
        .unwrap_or_else(|| realm.has_own(h, key));
    // An own data key String: noted in the String-key slot memo, so
    // the next ask is its probe above (`key_memo_note`'s rules).
    if own && k.as_handle().is_some() {
        let _ = realm.own_data_missed(o, k);
    }
    Some(own)
}

/// `recv.split(sep, limit)` for a string `recv` (the cell `h`) and a RegExp
/// `sep` the host can split by in one step (see `VmHost::string_split`):
/// the result; `None` for anything else.
#[inline(never)]
fn split_by_regexp(
    ctx: &mut Ctx,
    recv: NanBox,
    h: Handle,
    sep: NanBox,
    limit: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let r = sep.as_handle()?;
    if ctx.realm.multi_realm
        || ctx.realm.regexp_flags(Handle::from_raw(r)).is_none()
        || !ctx.realm.is_string_handle(h)
        || !ctx.realm.typeof_is_string(h)
    {
        return None;
    }
    let r = with_host(ctx, |host| host.string_split(recv, sep, limit))??;
    Some(r.map_err(VmError::from))
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
    if prim
        != matches!(
            op,
            FastOp::NumToFixed | FastOp::NumToString | FastOp::NumToPrecision
        )
    {
        return None;
    }
    let arg = |i: usize| {
        args.get(i)
            .map_or(NanBox::undefined(), |r| regs[*r as usize])
    };
    if let FastOp::GenResume(mode) = op {
        return super::gen_fast::call(ctx, funcs, regs, recv, mode, arg(0));
    }
    if matches!(op, FastOp::IterHelperNext) {
        return super::iter_helper::site_next(ctx, funcs, regs, recv, NanBox::from_bits(method));
    }
    if let FastOp::IterConsume(c) = op {
        let argv = super::ArgBuf::from_regs(regs, args);
        return super::iter_helper::consume(ctx, funcs, regs, recv, c, &argv);
    }
    // `Object.prototype.hasOwnProperty.call(o, k)` (the `for-in` idiom):
    // `hasOwnProperty`'s own fast op, with `o` its receiver — what the call
    // amounts to once `Function.prototype.call` (which the site proved) has
    // run. (Anything that op does not decide is the general path's.)
    // `Object.hasOwn(o, k)` likewise (`o` an object, `k` a key whose
    // ToPropertyKey runs no code: the order of the two conversions is then
    // unobservable).
    // `Reflect.apply(f, this, list)` with `list` an object (a primitive one
    // is the host's TypeError): `f.apply(this, list)`'s op.
    if matches!(op, FastOp::ReflectApply) {
        let f = arg(0).as_handle()?;
        // (`FnApply` takes only an array or an `arguments` object then.)
        arg(2).as_handle()?;
        let rest = args.get(1..).unwrap_or(&[]);
        return run_op(
            ctx,
            funcs,
            FastOp::FnApply,
            method,
            arg(0),
            Handle::from_raw(f),
            false,
            regs,
            rest,
        );
    }
    if matches!(op, FastOp::ObjStaticHasOwn) {
        let t = arg(0).as_handle()?;
        let rest = args.get(1..).unwrap_or(&[]);
        return run_op(
            ctx,
            funcs,
            FastOp::ObjHasOwn,
            method,
            arg(0),
            Handle::from_raw(t),
            false,
            regs,
            rest,
        );
    }
    // `F.call(t, …)` of a built-in `F` served here (`Array.prototype.slice
    // .call(a, 1)`, `Map.prototype.get.call(m, k)`) with an object `t`: `F`'s
    // own fast op, with `t` its receiver — what `Call(F, t, …)` amounts to
    // (each op checks its receiver's brand itself; anything it does not
    // decide is the host's call of `F`, below).
    // (A string cell with a function marker is a function, which no string
    // op may take for a string.)
    if matches!(op, FastOp::FnCall)
        && let Some(t) = arg(0).as_handle()
        && !ctx.realm.is_vm_function(h)
        && !(ctx.realm.is_string_handle(Handle::from_raw(t))
            && !ctx.realm.typeof_is_string(Handle::from_raw(t)))
    {
        let inner = classify(&ctx.realm, recv);
        if !matches!(
            inner,
            FastOp::None
                | FastOp::HostDirect
                | FastOp::FnCall
                | FastOp::FnApply
                | FastOp::FnBind
                | FastOp::ReflectApply
                | FastOp::IterHelperNext
                | FastOp::IterConsume(_)
                | FastOp::ObjStaticHasOwn
                | FastOp::NumToFixed
                | FastOp::NumToString
                | FastOp::NumToPrecision
                | FastOp::GenResume(_)
        ) {
            let rest = args.get(1..).unwrap_or(&[]);
            if let Some(r) = run_op_out_of_line(
                ctx,
                funcs,
                inner,
                recv.to_bits(),
                arg(0),
                Handle::from_raw(t),
                regs,
                rest,
            ) {
                return Some(r);
            }
        }
    }
    if let FastOp::ArrayHof(op) = op {
        let argv = super::ArgBuf::from_regs(regs, args);
        return super::array_hof::run(ctx, funcs, regs, recv, op, &argv);
    }
    if let FastOp::CollForEach(set) = op {
        let argv = super::ArgBuf::from_regs(regs, args);
        return super::coll_hof::for_each(ctx, funcs, regs, recv, h, set, &argv);
    }
    // A RegExp and a VM function: the replacer calls run here
    // (`str_replace`).
    if matches!(op, FastOp::StrReplace | FastOp::StrReplaceAll) {
        let all = matches!(op, FastOp::StrReplaceAll);
        if let Some(r) = super::str_replace::try_run(ctx, funcs, regs, recv, args, all) {
            return Some(r);
        }
        if all {
            let method = NanBox::from_bits(method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
                    .unwrap_or(Err(super::HostError::Fault))
                    .map_err(VmError::from),
            );
        }
    }
    // A string matched against a RegExp the host can match in one step.
    if matches!(op, FastOp::StrMatch) {
        let rx = arg(0);
        if !ctx.realm.multi_realm
            && ctx.realm.is_string_handle(h)
            && ctx.realm.typeof_is_string(h)
            && rx
                .as_handle()
                .is_some_and(|r| ctx.realm.regexp_flags(Handle::from_raw(r)).is_some())
            && let Some(Some(r)) = with_host(ctx, |host| host.string_match(recv, rx))
        {
            return Some(r.map_err(VmError::from));
        }
        let method = NanBox::from_bits(method);
        let argv = super::ArgBuf::from_regs(regs, args);
        return Some(
            with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
                .unwrap_or(Err(super::HostError::Fault))
                .map_err(VmError::from),
        );
    }
    // A string searched with a RegExp whose `@@search` and `exec` are the
    // intrinsics: that `@@search`, without the general call's GetMethod and
    // second call.
    if matches!(op, FastOp::StrSearch) {
        let rx = arg(0);
        if !ctx.realm.multi_realm
            && ctx.realm.is_string_handle(h)
            && ctx.realm.typeof_is_string(h)
            && rx
                .as_handle()
                .is_some_and(|r| ctx.realm.regexp_flags(Handle::from_raw(r)).is_some())
            && let Some(Some(r)) = with_host(ctx, |host| host.string_search(recv, rx))
        {
            return Some(r.map_err(VmError::from));
        }
        let method = NanBox::from_bits(method);
        let argv = super::ArgBuf::from_regs(regs, args);
        return Some(
            with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
                .unwrap_or(Err(super::HostError::Fault))
                .map_err(VmError::from),
        );
    }
    if matches!(op, FastOp::PromiseResolve) {
        if let Some(p) = promise_resolve_fast(ctx, h, arg(0)) {
            return Some(Ok(p));
        }
        let method = NanBox::from_bits(method);
        let argv = super::ArgBuf::from_regs(regs, args);
        return Some(
            with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
                .unwrap_or(Err(super::HostError::Fault))
                .map_err(VmError::from),
        );
    }
    if matches!(op, FastOp::ObjAssign) {
        if let Some(v) = obj_assign_fast(&mut ctx.realm, regs, args) {
            return Some(Ok(v));
        }
        let method = NanBox::from_bits(method);
        let argv = super::ArgBuf::from_regs(regs, args);
        return Some(
            with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
                .unwrap_or(Err(super::HostError::Fault))
                .map_err(VmError::from),
        );
    }
    let realm = &mut ctx.realm;
    let v = match op {
        FastOp::None
        | FastOp::PromiseResolve
        | FastOp::ObjAssign
        | FastOp::ArrayHof(_)
        | FastOp::CollForEach(_)
        | FastOp::StrReplaceAll
        | FastOp::StrMatch
        | FastOp::StrSearch
        | FastOp::GenResume(_)
        | FastOp::ReflectApply
        | FastOp::IterHelperNext
        | FastOp::IterConsume(_)
        | FastOp::ObjStaticHasOwn => {
            return None;
        }
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
        FastOp::ArrayUnshift => {
            let items = super::ArgBuf::from_regs(regs, args);
            NanBox::number(realm.array_unshift_fast(h, &items)? as f64)
        }
        FastOp::ArraySplice => {
            // Number (or absent) positions, whose conversions run no code, and
            // a plain `ArraySpeciesCreate`.
            let len = realm.array_elements(h)?.len();
            let start = rel_index(arg(0), len, 0)?;
            let delete = match args.len() {
                0 => 0,
                1 => len - start,
                _ => length_arg(arg(1))?.min(len - start),
            };
            if !super::array_hof::default_species_armed(realm, h) {
                return None;
            }
            let items = args.get(2..).unwrap_or(&[]);
            let items: alloc::vec::Vec<NanBox> = items.iter().map(|r| regs[*r as usize]).collect();
            let removed = realm.array_splice_fast(h, start, delete, &items)?;
            NanBox::handle(realm.new_array(removed).to_raw())
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
            // (A `Concat` is read by a short tree walk, or flattened once.)
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
            // The method's lean form, on one read of the receiver's cell (a
            // receiver of another kind, or a weak one, is a miss).
            realm.coll_fast(h, op.coll_op()?, arg(0), arg(1))?
        }
        FastOp::WeakColl(op) => {
            // A key that cannot be held weakly is the general path's (its
            // `TypeError` for an add, a symbol's registry check).
            let a = arg(0);
            if matches!(op, CollOp::MapSet | CollOp::SetAdd) && !super::is_object_value(realm, a) {
                return None;
            }
            realm.weak_coll_fast(h, op, a, arg(1))?
        }
        FastOp::WeakRefDeref => realm.weak_ref_target(h)?,
        FastOp::DateGet(g) => NanBox::number(g.get(realm.date_at(h)?)),
        FastOp::TypedArr(m) => {
            let n = args.len().min(3);
            let argv = [arg(0), arg(1), arg(2)];
            realm.ta_method(h, m, &argv[..n])?
        }
        FastOp::ObjFromEntries => obj_from_entries_fast(realm, arg(0))?,
        FastOp::DateUtc => {
            // Number arguments (ToNumber the identity), the seven read.
            let mut nums = [0.0; 7];
            let n = args.len().min(7);
            for (i, slot) in nums.iter_mut().enumerate().take(n) {
                *slot = arg(i).as_number()?;
            }
            NanBox::number(crate::realm::date_fields::date_utc(&nums[..n]))
        }
        FastOp::BigIntAsN(signed) => {
            let bits = arg(0).as_number()?;
            if !(0.0..9_007_199_254_740_992.0).contains(&bits) || (bits as u64) as f64 != bits {
                return None;
            }
            let x = arg(1);
            let max = realm.limits.max_bigint_bits;
            let r = realm
                .bigint_ref(Handle::from_raw(x.as_handle()?))?
                .as_n_bits(bits as u64, signed, max)?;
            NanBox::handle(realm.new_bigint(r).to_raw())
        }
        FastOp::DateSet(set) => {
            // Number arguments (ToNumber the identity; an absent primary one
            // is `undefined`'s NaN), the date value read before them.
            let ms = realm.date_at(h)?;
            let take = set.max_components().min(args.len().max(1));
            let mut comps = [f64::NAN; 4];
            for (i, c) in comps.iter_mut().enumerate().take(take.min(args.len())) {
                *c = arg(i).as_number()?;
            }
            let nms = set.apply(ms, &comps[..take]);
            if set.writes_on_nan_date() || !ms.is_nan() {
                realm.set_date_ms(h, nms);
            }
            NanBox::number(nms)
        }
        FastOp::ObjHasOwn => NanBox::boolean(obj_has_own_fast(realm, h, arg(0))?),
        FastOp::StrSlice | FastOp::StrSubstring | FastOp::StrSubstr => {
            // A `Concat` is sliced by walking its tree when that is cheap (see
            // `Rope::slice_units_tree`), and flattened only otherwise.
            let unflat = realm
                .typeof_is_string(h)
                .then(|| realm.string_rope(h))
                .flatten()
                .filter(|r| r.as_leaf_bytes().is_none());
            let rope = match &unflat {
                Some(r) => r.clone(),
                None => str_parts(realm, h)?,
            };
            let units = rope.utf16_len();
            let start = int_arg(arg(0))?.unwrap_or(0.0);
            let end = int_arg(arg(1))?;
            let (a, b) = if matches!(op, FastOp::StrSubstr) {
                // B.2.2.1 steps 4-11: a negative start counts from the end,
                // the length (all of it when `undefined`) is clamped to
                // `0..=size`, the end to `size`.
                let a = if start < 0.0 {
                    (units as f64 + start).max(0.0) as usize
                } else {
                    (start as usize).min(units)
                };
                let len = end.map_or(units, |n| (n.max(0.0) as usize).min(units));
                (a, a.saturating_add(len).min(units))
            } else if matches!(op, FastOp::StrSlice) {
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
            if unflat.is_some() {
                if let Some(out) = realm.new_string_slice_tree(&rope, a, b) {
                    return Some(Ok(NanBox::handle(out.to_raw())));
                }
                let rope = str_parts(realm, h)?;
                return Some(Ok(NanBox::handle(
                    realm.new_string_slice(&rope, a, b)?.to_raw(),
                )));
            }
            NanBox::handle(realm.new_string_slice(&rope, a, b)?.to_raw())
        }
        FastOp::StrStartsWith | FastOp::StrEndsWith | FastOp::StrIncludes => {
            // A string search argument is not a RegExp, and converts as itself.
            let needle = str_arg(realm, arg(0))?;
            let pat = needle.as_leaf_bytes()?;
            // An affix test on a `Concat` compares just the window it names
            // (`str_rope::window_is`), without flattening the receiver.
            if !matches!(op, FastOp::StrIncludes)
                && let Some(r) = realm
                    .typeof_is_string(h)
                    .then(|| realm.string_rope(h))
                    .flatten()
                    .filter(crate::rope::Rope::is_concat)
                && realm.rope_walk_ok(&r)
            {
                let units = r.utf16_len();
                let default = if matches!(op, FastOp::StrEndsWith) {
                    units
                } else {
                    0
                };
                let pos = int_arg(arg(1))?.map_or(default, |n| (n.max(0.0) as usize).min(units));
                let pat_units = needle.utf16_len();
                let at = if matches!(op, FastOp::StrEndsWith) {
                    pos.checked_sub(pat_units)
                } else {
                    Some(pos)
                };
                let hit = match at {
                    None => Some(false),
                    Some(at) => crate::nbexec::str_rope::window_is(&r, at, pat, pat_units),
                };
                if let Some(hit) = hit {
                    return Some(Ok(NanBox::boolean(hit)));
                }
            }
            let rope = str_parts(realm, h)?;
            let (bytes, units) = (rope.as_leaf_bytes()?, rope.utf16_len());
            let default = if matches!(op, FastOp::StrEndsWith) {
                units
            } else {
                0
            };
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
        FastOp::DataView(acc) => {
            crate::nbexec::dataview_fast(realm, recv, acc, arg(0), [arg(1), arg(2)])?
        }
        FastOp::StrFromCharCode => {
            // One code unit given as itself (ToUint16 of `0 <= n < 2^16` is
            // its truncation): the realm's interned one-unit string.
            if let [r] = *args
                && let Some(n) = regs[r as usize].as_number()
                && (0.0..65536.0).contains(&n)
            {
                return Some(Ok(NanBox::handle(realm.unit_string(n as u16).to_raw())));
            }
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
        FastOp::ObjCreate => {
            // An Object or `null` prototype and no Properties argument (an
            // explicit `undefined` is none): OrdinaryObjectCreate(proto). A
            // non-object prototype's TypeError, and the descriptors, are the
            // host's.
            if !arg(1).is_undefined() {
                return None;
            }
            let p = arg(0);
            let proto = match p.as_handle().map(Handle::from_raw) {
                None if p.is_null() => None,
                Some(o)
                    if !realm.is_string_handle(o)
                        && realm.symbol_at(o).is_none()
                        && realm.bigint_at(o).is_none() =>
                {
                    Some(o)
                }
                _ => return None,
            };
            NanBox::handle(realm.new_object_with_proto(proto).to_raw())
        }
        FastOp::NumPred(p) => NanBox::boolean(p.test(arg(0))),
        FastOp::ArrayFrom => {
            // `Array.from(items)` (`mapFn` absent or `undefined`) with `this`
            // the (single) realm's own `%Array%`, of a plain array whose
            // iteration is the built-in one and has no hole
            // (`iter_fast::plain_array_values`): `Construct(%Array%)` makes a
            // fresh array (its prototype read from a non-writable
            // `Array.prototype`), the iteration reads only own elements and
            // runs nothing, and each value's CreateDataPropertyOrThrow on the
            // fresh array is an append — so the result is a copy of the
            // elements. Anything else is the host's.
            if realm.multi_realm
                || !arg(1).is_undefined()
                || realm.native_at(h) != Some(crate::nbexec::N_BASE_ARRAY)
            {
                return None;
            }
            let vals = super::iter_fast::plain_array_values(realm, arg(0))?.to_vec();
            NanBox::handle(realm.new_array(vals).to_raw())
        }
        FastOp::ArrayIsArray => match arg(0).as_handle().map(Handle::from_raw) {
            // A proxy is unwrapped (and may be revoked): the host's.
            Some(a) if realm.proxy_at(a).is_some() => return None,
            Some(a) => NanBox::boolean(realm.is_array(a) && !realm.is_vm_function(a)),
            None => NanBox::boolean(false),
        },
        FastOp::FnBind => {
            let bound: alloc::vec::Vec<NanBox> =
                args.iter().skip(1).map(|r| regs[*r as usize]).collect();
            fn_bind(realm, h, arg(0), bound)?
        }
        FastOp::FnCall | FastOp::FnApply => {
            // (Another callable — a built-in — is called through the host
            // below, with the same `this` and arguments.)
            let vm = realm.is_vm_function(h);
            if !vm && (!realm.is_callable_cell(h) || realm.proxy_at(h).is_some()) {
                return None;
            }
            let this = arg(0);
            let argv: alloc::vec::Vec<NanBox> = if matches!(op, FastOp::FnCall) {
                args.iter().skip(1).map(|r| regs[*r as usize]).collect()
            } else {
                // CreateListFromArrayLike of a plain dense array reads only its
                // own elements: no hole (which would read through the
                // prototype), no index side table, the length its dense length.
                let list = arg(1);
                match list.as_handle().map(Handle::from_raw) {
                    None if list.is_undefined() || list.is_null() => alloc::vec::Vec::new(),
                    Some(l) => match realm.array_elements(l) {
                        Some(elems) => {
                            if realm.array_has_index_overrides(l)
                                || realm.array_length(l) != Some(elems.len())
                                || elems.iter().any(|e| e.is_hole())
                            {
                                return None;
                            }
                            elems.to_vec()
                        }
                        // An unchanged `arguments` object: its own indices.
                        None => realm.arguments_values(l)?,
                    },
                    None => return None,
                }
            };
            // `Call(F, this, argv)` of a built-in `F`: the host's call of the
            // function itself, as `FastOp::HostDirect` makes it.
            if !vm {
                return Some(
                    with_host_regs(ctx, regs, |host| host.call(recv, this, &argv))
                        .unwrap_or(Err(super::HostError::Fault))
                        .map_err(VmError::from),
                );
            }
            // A callee with a compiled body is entered directly, as a
            // compiled caller's `f(...)` enters it.
            #[cfg(all(feature = "jit", jit_native))]
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
                with_host_regs(ctx, regs, |host| {
                    host.call_regexp_method(method, recv, &argv)
                })
                .unwrap_or(Err(super::HostError::Fault))
                .map_err(VmError::from),
            );
        }
        FastOp::HostDirect => {
            let method = NanBox::from_bits(method);
            let argv = super::ArgBuf::from_regs(regs, args);
            return Some(
                with_host_regs(ctx, regs, |host| host.call(method, recv, &argv))
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
            // Holes are not scanned for up front (that would cost a pass over
            // the whole array when the match is early): the search stops at
            // the first hole before a match and leaves it to the host (whose
            // answer may come from the prototype chain).
            let elems = realm.array_elements(h)?;
            if realm.array_has_index_overrides(h)
                || realm.proto_index_accessor_dirty()
                || realm.array_length(h) != Some(elems.len())
            {
                return None;
            }
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
            // A primitive target is decided by the boxed values alone (no
            // string or BigInt content to compare): the tight loop.
            let prim = target.as_handle().is_none();
            let nan =
                matches!(op, FastOp::ArrIncludes) && target.as_number().is_some_and(f64::is_nan);
            let pos = if prim && !nan {
                tail.iter()
                    .position(|e| e.is_hole() || e.strict_equals(target))
            } else if prim {
                tail.iter()
                    .position(|e| e.is_hole() || e.as_number().is_some_and(f64::is_nan))
            } else {
                tail.iter()
                    .position(|e| e.is_hole() || realm.strict_equals(*e, target))
            };
            if pos.is_some_and(|i| tail[i].is_hole()) {
                return None;
            }
            if matches!(op, FastOp::ArrIndexOf) {
                NanBox::number(pos.map_or(-1.0, |i| (from + i) as f64))
            } else {
                NanBox::boolean(pos.is_some())
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
                        let bytes = part.as_leaf_bytes()?;
                        // Past the string length cap: the host's join throws
                        // the RangeError (before this buffer grows further).
                        if out.len() + bytes.len() + sep.len() > realm.limits.max_string_len {
                            return None;
                        }
                        crate::wtf8::append(&mut out, bytes);
                    }
                }
            }
            if out.len() > realm.limits.max_string_len {
                return None;
            }
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::ArrSlice => {
            // `ArraySpeciesCreate` must be the plain `ArrayCreate`.
            if !super::array_hof::default_species_armed(realm, h) {
                return None;
            }
            let elems = dense_elements(realm, h)?;
            let len = elems.len();
            let a = rel_index(arg(0), len, 0)?;
            let b = rel_index(arg(1), len, len)?;
            let part = elems.get(a..b.max(a)).unwrap_or(&[]).to_vec();
            NanBox::handle(realm.new_array(part).to_raw())
        }
        FastOp::ArrConcat => {
            // `O` an array with no own named property (so no own
            // `constructor` or `@@isConcatSpreadable`) and its default
            // prototype, `ArraySpeciesCreate(O, 0)` the plain `ArrayCreate`,
            // and `@@isConcatSpreadable` absent along that prototype's
            // chain of ordinary data properties: every array item of that
            // same kind is spread (`IsArray`), every other item that is not
            // an object appended. Each array is dense with no hole, so its
            // `HasProperty`/`Get` per index are its elements. Anything else
            // (another object, a hole, a proxy) is the host's.
            let p = realm.plain_array_root_proto(h)?;
            if realm.is_host_exotic(h) || !super::array_hof::default_species_armed(realm, h) {
                return None;
            }
            let epoch = crate::object::proto_epoch();
            if realm.vm_array_iter().concat_spread_absent != Some((p.to_raw(), epoch)) {
                let key: alloc::boxed::Box<str> =
                    realm.vm_array_iter().concat_spreadable_key.clone()?;
                if !matches!(super::array_hof::chain_data(realm, p, &key), Ok(None))
                    || !matches!(realm.watch_get_chain(p, &key), Some(None))
                {
                    return None;
                }
                // (Watching may itself have moved the epoch.)
                realm.vm_array_iter_mut().concat_spread_absent =
                    Some((p.to_raw(), crate::object::proto_epoch()));
            }
            let item = |i: usize| {
                if i == 0 {
                    recv
                } else {
                    regs[args[i - 1] as usize]
                }
            };
            let mut total = 0usize;
            for i in 0..=args.len() {
                let v = item(i);
                total += match v.as_handle().map(Handle::from_raw) {
                    Some(e) if realm.typeof_is_string(e) => 1,
                    Some(e) => {
                        if realm.plain_array_root_proto(e) != Some(p) || realm.is_host_exotic(e) {
                            return None;
                        }
                        dense_elements(realm, e)?.len()
                    }
                    None => 1,
                };
            }
            if total > realm.limits.max_array_len {
                return None;
            }
            let mut out = alloc::vec::Vec::with_capacity(total);
            for i in 0..=args.len() {
                let v = item(i);
                match v.as_handle().map(Handle::from_raw) {
                    Some(e) if !realm.typeof_is_string(e) => {
                        out.extend_from_slice(realm.array_elements(e)?);
                    }
                    _ => out.push(v),
                }
            }
            NanBox::handle(realm.new_array(out).to_raw())
        }
        FastOp::ArrFill => {
            // Positions that convert without running code, over an array
            // whose every `Set(O, k, value, true)` is its store
            // (`Realm::array_fill_plain`).
            let len = realm.array_length(h)?;
            let a = rel_index(arg(1), len, 0)?;
            let b = rel_index(arg(2), len, len)?;
            if !realm.array_fill_plain(h, a, b, arg(0)) {
                return None;
            }
            recv
        }
        FastOp::CollIter(set, kind) => {
            // `CreateMapIterator` / `CreateSetIterator` of a collection of the
            // method's own kind (anything else is the host's TypeError).
            if realm.collection_is_set(h) != Some(set) {
                return None;
            }
            NanBox::handle(realm.new_live_collection_iterator(h, kind)?.to_raw())
        }
        FastOp::ArrIter(kind) => {
            // `CreateArrayIterator(O, kind)` of a real array: the host's live
            // iterator, made here once `%ArrayIteratorPrototype%` exists.
            if !realm.is_plain_array_cell(h) {
                return None;
            }
            NanBox::handle(realm.new_live_array_iterator(h, kind)?.to_raw())
        }
        FastOp::ArrToString => {
            // `func = Get(array, "join")` along a chain of data properties is
            // the built-in `join`, called with no arguments: its own fast op.
            let join = super::array_hof::chain_data(realm, h, "join").ok()??;
            let (id, name) = realm.bound_native_at(Handle::from_raw(join.as_handle()?))?;
            if id != N_ARRAY_PROTO_FN || realm.string_leaf_bytes(name) != Some(b"join") {
                return None;
            }
            return run_op_out_of_line(
                ctx,
                funcs,
                FastOp::ArrJoin,
                join.to_bits(),
                recv,
                h,
                regs,
                &[],
            );
        }
        FastOp::NumToPrecision => {
            let n = recv.as_number()?;
            let p = arg(0);
            if p.is_undefined() || !n.is_finite() && p.is_number() {
                // No precision (or a non-finite `this` once a Number
                // precision is converted): ToString of the value.
                NanBox::handle(
                    realm
                        .new_string(&crate::realm::js_number_string(n))
                        .to_raw(),
                )
            } else {
                let p = int_arg(p)??;
                if !(1.0..=100.0).contains(&p) {
                    return None;
                }
                NanBox::handle(
                    realm
                        .new_string(&crate::nbexec::format_precision(n, p as usize))
                        .to_raw(),
                )
            }
        }
        FastOp::ArrAt => {
            // `k` outside `0..len` is `undefined` with no `Get`; inside, a
            // present slot with no per-index attribute is an own data read
            // (a hole is the host's: its `Get` walks the prototype chain).
            let e = realm.array_elements(h)?;
            if realm.array_has_index_overrides(h) || realm.array_length(h) != Some(e.len()) {
                return None;
            }
            let len = e.len();
            let rel = int_arg(arg(0))?.unwrap_or(0.0);
            let k = if rel < 0.0 { len as f64 + rel } else { rel };
            if !(0.0..len as f64).contains(&k) {
                NanBox::undefined()
            } else {
                let v = e[k as usize];
                if v.is_hole() {
                    return None;
                }
                v
            }
        }
        FastOp::ArrLastIndexOf => {
            // As `indexOf`, searching down from `fromIndex` (when given:
            // `undefined` converts to 0, unlike an absent one).
            let elems = realm.array_elements(h)?;
            if realm.array_has_index_overrides(h)
                || realm.proto_index_accessor_dirty()
                || realm.array_length(h) != Some(elems.len())
            {
                return None;
            }
            let len = elems.len();
            if len == 0 {
                return Some(Ok(NanBox::number(-1.0)));
            }
            let from = if args.len() >= 2 {
                let n = int_arg(arg(1))?.unwrap_or(0.0);
                if n >= 0.0 {
                    n.min(len as f64 - 1.0)
                } else {
                    len as f64 + n
                }
            } else {
                len as f64 - 1.0
            };
            if from < 0.0 {
                return Some(Ok(NanBox::number(-1.0)));
            }
            let target = arg(0);
            let head = &elems[..=from as usize];
            let pos = if target.as_handle().is_none() {
                head.iter()
                    .rposition(|e| e.is_hole() || e.strict_equals(target))
            } else {
                head.iter()
                    .rposition(|e| e.is_hole() || realm.strict_equals(*e, target))
            };
            if pos.is_some_and(|i| head[i].is_hole()) {
                return None;
            }
            NanBox::number(pos.map_or(-1.0, |i| i as f64))
        }
        FastOp::ArrReverse => {
            // Every `[[Get]]` / `[[Set]]` an own data element
            // (`Realm::array_reverse_plain`).
            if !realm.array_reverse_plain(h) {
                return None;
            }
            recv
        }
        FastOp::StrAt => {
            if !realm.typeof_is_string(h) {
                return None;
            }
            let units = realm.string_utf16_len(h)?;
            let rel = int_arg(arg(0))?.unwrap_or(0.0);
            let k = if rel < 0.0 { units as f64 + rel } else { rel };
            if !(0.0..units as f64).contains(&k) {
                NanBox::undefined()
            } else {
                match realm.string_unit_at_index(h, Some(k as usize))? {
                    Some(u) => NanBox::handle(realm.unit_string(u).to_raw()),
                    None => NanBox::undefined(),
                }
            }
        }
        FastOp::StrCodePointAt => {
            if !realm.typeof_is_string(h) {
                return None;
            }
            let n = int_arg(arg(0))?.unwrap_or(0.0);
            if n < 0.0 {
                return Some(Ok(NanBox::undefined()));
            }
            let i = n as usize;
            match realm.string_unit_at_index(h, Some(i))? {
                None => NanBox::undefined(),
                Some(u) if (0xD800..0xDC00).contains(&u) => {
                    let cp = match realm.string_unit_at_index(h, i.checked_add(1))? {
                        Some(low) if (0xDC00..0xE000).contains(&low) => {
                            0x1_0000 + ((u32::from(u) - 0xD800) << 10) + (u32::from(low) - 0xDC00)
                        }
                        _ => u32::from(u),
                    };
                    NanBox::number(f64::from(cp))
                }
                Some(u) => NanBox::number(f64::from(u)),
            }
        }
        FastOp::StrConcat => {
            // String and Number arguments (whose ToString runs no code),
            // appended in order; past the length cap the host throws.
            if !realm.typeof_is_string(h) {
                return None;
            }
            let mut acc = recv;
            for i in 0..args.len() {
                let a = arg(i);
                let ok = a.is_number()
                    || a.as_handle()
                        .is_some_and(|x| realm.typeof_is_string(Handle::from_raw(x)));
                if !ok {
                    return None;
                }
                acc = realm.add_checked(acc, a)?;
            }
            acc
        }
        FastOp::StrRepeat => {
            let n = int_arg(arg(0))?.unwrap_or(0.0);
            // A negative or infinite count is the host's RangeError.
            if !(0.0..=f64::from(u32::MAX)).contains(&n) {
                return None;
            }
            let n = n as usize;
            let rope = str_parts(realm, h)?;
            let bytes = rope.as_leaf_bytes()?;
            if bytes.len().checked_mul(n)? > realm.limits.max_string_len
                || rope.utf16_len().checked_mul(n)? > realm.limits.max_string_len
            {
                return None;
            }
            if n == 1 || bytes.is_empty() {
                return Some(Ok(recv));
            }
            if n == 0 {
                return Some(Ok(NanBox::handle(realm.new_string("").to_raw())));
            }
            let mut out = alloc::vec::Vec::with_capacity(bytes.len() * n);
            for _ in 0..n {
                crate::wtf8::append(&mut out, bytes);
            }
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::StrLastIndexOf => {
            // An ASCII receiver and needle (byte positions are code-unit
            // positions), and a Number or absent position.
            if !realm.typeof_is_string(h) {
                return None;
            }
            let needle = arg(0).as_handle().map(Handle::from_raw)?;
            if !realm.typeof_is_string(needle) {
                return None;
            }
            let pos = arg(1);
            let pos = match pos.as_number() {
                // NaN (and `undefined`) mean +Infinity.
                Some(n) if !n.is_nan() => n,
                Some(_) => f64::INFINITY,
                None if pos.is_undefined() => f64::INFINITY,
                None => return None,
            };
            let rope = str_parts(realm, h)?;
            let pat = str_parts(realm, needle)?;
            let (hay, pat) = (rope.as_leaf_bytes()?, pat.as_leaf_bytes()?);
            if !hay.is_ascii() || !pat.is_ascii() {
                return None;
            }
            let len = hay.len();
            let start = if pos <= 0.0 {
                0
            } else {
                (pos as usize).min(len)
            };
            if pat.len() > len {
                return Some(Ok(NanBox::number(-1.0)));
            }
            let top = start.min(len - pat.len());
            let found = (0..=top).rev().find(|&j| &hay[j..j + pat.len()] == pat);
            NanBox::number(found.map_or(-1.0, |j| j as f64))
        }
        FastOp::ArrFlat => {
            // `ArraySpeciesCreate` must be the plain `ArrayCreate`, and the
            // depth a Number (or `undefined`, depth 1).
            let depth = match int_arg(arg(0))? {
                None => 1,
                Some(d) if d <= 16.0 => d.max(0.0) as usize,
                // A deeper (or infinite) flattening is the host's.
                Some(_) => return None,
            };
            if !super::array_hof::default_species_armed(realm, h) {
                return None;
            }
            let mut out = alloc::vec::Vec::new();
            flatten_dense(realm, h, depth, &mut out)?;
            NanBox::handle(realm.new_array(out).to_raw())
        }
        FastOp::StrCase(upper) => {
            let rope = str_parts(realm, h)?;
            let bytes = rope.as_leaf_bytes()?;
            if ascii_case_enabled() && bytes.is_ascii() {
                // ASCII text: mapped byte by byte, the receiver itself when
                // no byte changes (a string is a value: which cell holds it
                // is not observable), with its UTF-16 length known.
                let changes = |b: &u8| {
                    if upper {
                        b.is_ascii_lowercase()
                    } else {
                        b.is_ascii_uppercase()
                    }
                };
                if !bytes.iter().any(changes) {
                    return Some(Ok(recv));
                }
                // One character (`w[0].toUpperCase()`): its interned cell.
                if let [b] = *bytes
                    && unit_case_enabled()
                {
                    let m = if upper {
                        b.to_ascii_uppercase()
                    } else {
                        b.to_ascii_lowercase()
                    };
                    return Some(Ok(NanBox::handle(realm.unit_string(u16::from(m)).to_raw())));
                }
                let out = crate::rope::Rope::from_ascii_mapped(bytes, upper);
                return Some(Ok(NanBox::handle(realm.new_string_from_rope(out).to_raw())));
            }
            let out = crate::nbexec::case_map_wtf8(bytes, upper);
            NanBox::handle(realm.new_string_wtf8(out).to_raw())
        }
        FastOp::StrTrim(start, end) => {
            // The kept part shares the receiver's bytes or subtrees; a
            // `Concat` too deep to walk is flattened first.
            if !realm.typeof_is_string(h) {
                return None;
            }
            let rope = realm.string_rope(h)?;
            let walk = !rope.is_concat() || realm.rope_walk_ok(&rope);
            let out = match walk
                .then(|| crate::nbexec::str_rope::trim_rope(&rope, start, end))
                .flatten()
            {
                Some(out) => out,
                None => {
                    let rope = str_parts(realm, h)?;
                    crate::nbexec::str_rope::trim_rope(&rope, start, end)?
                }
            };
            if out.ptr_eq(&rope) {
                recv
            } else {
                NanBox::handle(realm.new_string_from_rope(out).to_raw())
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
            // The receiver as it is (a `Concat` is not flattened): the result
            // joins it to the filler.
            if !realm.typeof_is_string(h) {
                return None;
            }
            let rope = realm.string_rope(h)?;
            // A huge target is the host's (its length check throws).
            if target > rope.utf16_len() + (1 << 20) {
                return None;
            }
            let pad: &[u8] = match &fill {
                None => b" ",
                Some(f) => f.as_leaf_bytes()?,
            };
            let out = crate::nbexec::str_rope::pad_rope(&rope, target, pad, at_start)?;
            if out.ptr_eq(&rope) {
                recv
            } else {
                NanBox::handle(realm.new_string_from_rope(out).to_raw())
            }
        }
        FastOp::StrSplit => {
            // A string split by a RegExp the host can split in one step.
            let Some(sep) = str_arg(realm, arg(0)) else {
                return split_by_regexp(ctx, recv, h, arg(0), arg(1));
            };
            // A non-empty string separator that cannot split a surrogate pair,
            // and no limit.
            if !arg(1).is_undefined() {
                return None;
            }
            let sep_b = sep.as_leaf_bytes()?;
            if sep_b.is_empty() {
                // `s.split('')`: one piece per code unit — of an ASCII text,
                // each byte's interned one-unit string.
                let rope = str_parts(realm, h)?;
                let bytes = rope.as_leaf_bytes()?;
                if !bytes.is_ascii() || bytes.len() > realm.limits.max_array_len {
                    return None;
                }
                let parts: alloc::vec::Vec<NanBox> = bytes
                    .iter()
                    .map(|&b| NanBox::handle(realm.unit_string(u16::from(b)).to_raw()))
                    .collect();
                return Some(Ok(NanBox::handle(realm.new_array(parts).to_raw())));
            }
            if crate::nbexec::split_sensitive(sep_b) {
                return None;
            }
            let rope = str_parts(realm, h)?;
            // `split_units`' byte scan (exact for a separator that cannot
            // split a pair), with each piece copied once, straight into its
            // string (inline when short).
            let bytes = rope.as_leaf_bytes()?;
            let n = sep_b.len();
            let lean = ascii_case_enabled();
            // A one-byte separator the text does not hold: the receiver as
            // the one piece, found by one scan (no piece list).
            if lean && n == 1 && str_parts_lean_on() && !bytes.contains(&sep_b[0]) {
                return Some(Ok(NanBox::handle(
                    realm.new_array(alloc::vec![recv]).to_raw(),
                )));
            }
            let mut parts = if lean {
                alloc::vec::Vec::with_capacity(4)
            } else {
                alloc::vec::Vec::new()
            };
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
            if lean && start == 0 {
                // No separator: the one piece is the receiver's whole text
                // (a string is a value: which cell holds it is not
                // observable).
                parts.push(recv);
            } else {
                let piece = crate::rope::Rope::from_bytes(&bytes[start..]);
                parts.push(NanBox::handle(realm.new_string_from_rope(piece).to_raw()));
            }
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
        FastOp::ObjNames => {
            // An object whose own properties are exactly its layout's
            // enumerable data keys, none a Symbol or other internal key:
            // its own String keys are its `Object.keys` (the same shared
            // key Strings, see `plain_enumerable_key_strings`).
            // So are those of a plain dictionary (no non-enumerable key).
            let o = arg(0).as_handle().map(Handle::from_raw)?;
            let strs = match realm.plain_enumerable_key_strings(o) {
                Some(strs) => strs,
                None if realm.is_plain_dict_object(o) => realm.dict_enumerable_key_strings(o)?,
                None => return None,
            };
            NanBox::handle(realm.new_array(strs.to_vec()).to_raw())
        }
        FastOp::ObjGetProto => {
            // An ordinary object or a plain array (no proxy, no host hook):
            // `[[GetPrototypeOf]]` is the ordinary one, the stored link.
            // (`ToObject` of a primitive, and `Reflect`'s TypeError for one,
            // are the host's.)
            let o = arg(0).as_handle().map(Handle::from_raw)?;
            let p = if let Some(p) = realm.plain_array_root_proto(o) {
                Some(p)
            } else if realm.is_object_cell(o) {
                realm.object_proto(o)
            } else {
                return None;
            };
            if realm.is_host_exotic(o) {
                return None;
            }
            p.map_or(NanBox::null(), |p| NanBox::handle(p.to_raw()))
        }
        FastOp::ReflectGet => {
            // `target.[[Get]](key, receiver)` over an ordinary chain that
            // finds a data property or nothing (the receiver plays no part).
            let t = arg(0).as_handle().map(Handle::from_raw)?;
            let key = reflect_key(realm, arg(1))?;
            match realm.ordinary_chain_get(t, &key)?.0 {
                crate::realm::ChainHit::Data(v) => v,
                crate::realm::ChainHit::Absent => NanBox::undefined(),
                crate::realm::ChainHit::Accessor(..) => return None,
            }
        }
        FastOp::ReflectSet => {
            // `target.[[Set]](key, value, receiver)` (OrdinarySet) over an
            // ordinary chain where the key is an own data property of the
            // target or on no object of it: the receiver's define — a plain
            // store on the target itself, or on a proxy receiver that
            // forwards the define to an ordinary object without a trap
            // (`Realm::proxy_receiver_define_quiet`).
            let t = arg(0).as_handle().map(Handle::from_raw)?;
            let key = reflect_key(realm, arg(1))?;
            match realm.ordinary_chain_get(t, &key)?.0 {
                crate::realm::ChainHit::Data(_) if realm.has_own(t, &key) => {}
                crate::realm::ChainHit::Absent => {}
                _ => return None,
            }
            let value = arg(2);
            let receiver = if args.len() > 3 { arg(3) } else { arg(0) };
            let r = Handle::from_raw(receiver.as_handle()?);
            if r == t {
                if !realm.create_data_property_plain(t, &key, value) {
                    return None;
                }
            } else if realm.proxy_at(r).is_some() {
                realm.proxy_receiver_define_quiet(r, &key, value)?;
            } else {
                return None;
            }
            NanBox::boolean(true)
        }
        FastOp::ReflectHas => {
            // `HasProperty(target, key)` for a String key that is its own
            // storage key, along a chain of ordinary objects and plain arrays
            // (no proxy `has` trap, no host hook, no index side table): each
            // link's own-key test, in turn.
            let t = arg(0).as_handle().map(Handle::from_raw)?;
            let kh = arg(1).as_handle().map(Handle::from_raw)?;
            let b = realm.string_leaf_bytes(kh)?;
            if b.first() == Some(&0) {
                return None;
            }
            let key = core::str::from_utf8(b).ok()?;
            let mut cur = Some(t);
            let mut found = false;
            for _ in 0..64 {
                let Some(c) = cur else {
                    break;
                };
                let next = if let Some(p) = realm.plain_array_root_proto(c) {
                    Some(p)
                } else if realm.is_object_cell(c) {
                    realm.object_proto(c)
                } else {
                    return None;
                };
                if realm.is_host_exotic(c) {
                    return None;
                }
                if realm.has_own(c, key) {
                    found = true;
                    break;
                }
                cur = next;
            }
            if cur.is_some() && !found {
                return None;
            }
            NanBox::boolean(found)
        }
        FastOp::ObjKeys | FastOp::ObjEntries | FastOp::ObjValues => {
            let o = arg(0).as_handle().map(Handle::from_raw)?;
            if matches!(op, FastOp::ObjKeys)
                && let Some(keys) = realm.plain_array_index_keys(o)
            {
                return Some(Ok(NanBox::handle(realm.new_array(keys).to_raw())));
            }
            // A plain shaped object: its layout's remembered key Strings,
            // and its values read from their slots. A plain dictionary:
            // its remembered key Strings, its values in the same order.
            let (strs, vals) = if let Some((strs, slots)) = realm.plain_enumerable_key_slots(o) {
                let vals = match op {
                    FastOp::ObjKeys => None,
                    _ => Some(realm.object_slot_values(o, &slots)?),
                };
                (strs, vals)
            } else if matches!(op, FastOp::ObjValues) {
                // (No key String is needed for the values alone.)
                if realm.is_host_exotic(o) {
                    return None;
                }
                let vals = realm.plain_enum_values(o)?;
                return Some(Ok(NanBox::handle(realm.new_array(vals).to_raw())));
            } else {
                let strs = realm.dict_enumerable_key_strings(o)?;
                let vals = match op {
                    FastOp::ObjKeys => None,
                    _ => Some(realm.plain_enum_values(o)?),
                };
                (strs, vals)
            };
            match (op, vals) {
                (FastOp::ObjValues, Some(vals)) => NanBox::handle(realm.new_array(vals).to_raw()),
                (FastOp::ObjEntries, Some(vals)) if vals.len() == strs.len() => {
                    let mut out = alloc::vec::Vec::with_capacity(vals.len());
                    for (&k, v) in strs.iter().zip(vals) {
                        out.push(NanBox::handle(realm.new_array(alloc::vec![k, v]).to_raw()));
                    }
                    NanBox::handle(realm.new_array(out).to_raw())
                }
                (FastOp::ObjKeys, _) => NanBox::handle(realm.new_array(strs.to_vec()).to_raw()),
                _ => return None,
            }
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
        N_BIGINT, N_BOOLEAN, N_IS_FINITE, N_IS_NAN, N_NUMBER, N_PARSE_FLOAT, N_PARSE_INT, N_STRING,
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
        // `BigInt(v)` of a BigInt is `v`; of a Boolean, `0n` / `1n`; of an
        // integral Number (NumberToBigInt) its exact value — below `2^127`
        // in magnitude here; anything else (a RangeError, a string's parse,
        // an object's conversion) is the host's.
        N_BIGINT
            if a0
                .as_handle()
                .is_some_and(|h| realm.is_bigint(Handle::from_raw(h))) =>
        {
            a0
        }
        N_BIGINT if matches!(a0.unpack(), Unpacked::Bool(_) | Unpacked::Number(_)) => {
            let v = match a0.unpack() {
                Unpacked::Bool(b) => i128::from(b),
                // (A NaN or an infinity fails the bound; the cast back is
                // exact just when `n` is an integer.)
                Unpacked::Number(n) if n.abs() < 1.7e38 && (n as i128) as f64 == n => n as i128,
                _ => return None,
            };
            NanBox::handle(
                realm
                    .new_bigint(crate::bignum::BigInt::from_i128(v))
                    .to_raw(),
            )
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

/// Whether the method value `v` is the built-in a direct call site of kind
/// `kind` (`crate::jit::direct_call`) serves: what its proof
/// (`MethodSite::fill_direct`) records.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn direct_proves(realm: &Realm, v: NanBox, kind: u8) -> bool {
    kind != 0 && direct_kind(realm, v) == kind
}

/// The direct kind (`crate::jit::direct_call`) of the function value `v`,
/// or `0` when it is none of those built-ins.
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn direct_kind(realm: &Realm, v: NanBox) -> u8 {
    use crate::jit::direct_call as d;
    match classify(realm, v) {
        FastOp::ArrayIsArray => d::K_IS_ARRAY,
        FastOp::NumPred(NumPred::Integer) => d::K_IS_INTEGER,
        FastOp::NumPred(NumPred::SafeInteger) => d::K_IS_SAFE_INTEGER,
        FastOp::NumPred(NumPred::Finite) => d::K_IS_FINITE,
        FastOp::NumPred(NumPred::NaN) => d::K_IS_NAN,
        FastOp::ObjStaticHasOwn => d::K_HAS_OWN,
        FastOp::ObjHasOwn => d::K_PROTO_HAS_OWN,
        FastOp::StrFromCharCode => d::K_FROM_CHAR_CODE,
        FastOp::HostDirect => match v
            .as_handle()
            .and_then(|f| realm.native_at(Handle::from_raw(f)))
        {
            Some(crate::nbexec::N_OBJECT_GET_PROTO) => d::K_GET_PROTO,
            Some(crate::nbexec::N_OBJECT_IS) => d::K_OBJECT_IS,
            _ => 0,
        },
        _ => 0,
    }
}

/// A direct built-in call (`crate::jit::direct_call`) of kind `kind` on the
/// arguments `a0`, `a1` (`undefined` for a missing one): its result when the
/// built-in's answer for them runs no code, allocates nothing and cannot
/// throw; `None`, having done nothing, for anything else (the site's general
/// call then makes it).
#[cfg(all(feature = "jit", jit_native))]
pub(super) fn direct_run(realm: &mut Realm, kind: u8, a0: NanBox, a1: NanBox) -> Option<NanBox> {
    use crate::jit::direct_call as d;
    match kind {
        d::K_IS_ARRAY => {
            let a = Handle::from_raw(a0.as_handle()?);
            // A proxy is unwrapped (and may be revoked): the general path's.
            if realm.proxy_at(a).is_some() {
                return None;
            }
            Some(NanBox::boolean(
                realm.is_array(a) && !realm.is_vm_function(a),
            ))
        }
        d::K_IS_INTEGER => Some(NanBox::boolean(NumPred::Integer.test(a0))),
        d::K_IS_SAFE_INTEGER => Some(NanBox::boolean(NumPred::SafeInteger.test(a0))),
        d::K_IS_FINITE => Some(NanBox::boolean(NumPred::Finite.test(a0))),
        d::K_IS_NAN => Some(NanBox::boolean(NumPred::NaN.test(a0))),
        // (`o.hasOwnProperty(k)`: ToPropertyKey before ToObject, which is
        // unobservable when neither runs code or throws.)
        d::K_HAS_OWN | d::K_PROTO_HAS_OWN => {
            let o = Handle::from_raw(a0.as_handle()?);
            Some(NanBox::boolean(obj_has_own_fast(realm, o, a1)?))
        }
        d::K_FROM_CHAR_CODE => {
            // One code unit given as itself whose interned string exists
            // (`Realm::unit_string`, which then allocates nothing).
            let n = a0.as_number()?;
            if !(0.0..256.0).contains(&n) {
                return None;
            }
            Some(NanBox::handle(
                realm.interned_unit_string(n as u16)?.to_raw(),
            ))
        }
        // (SameValue reads values only: a String's or BigInt's contents.)
        d::K_OBJECT_IS => Some(NanBox::boolean(realm.same_value(a0, a1))),
        d::K_GET_PROTO => {
            // An ordinary object (not host-exotic) or an array: its
            // `[[Prototype]]` link (ToObject is the identity, and no trap
            // runs).
            let o = Handle::from_raw(a0.as_handle()?);
            if !(realm.jit_object(o).is_some() || realm.is_array(o)) || realm.is_host_exotic(o) {
                return None;
            }
            Some(
                realm
                    .object_proto(o)
                    .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw())),
            )
        }
        _ => None,
    }
}

/// `Function.prototype.bind` of the VM function `t` with `this` and the
/// leading arguments `bound` (BoundFunctionCreate, then the `length` and
/// `name` steps), built in the realm as the host's `make_bound_function`
/// builds it — the same hidden slots and properties, in the same order —
/// when every read it makes is a plain data read that runs no code: a
/// single-realm run, an ordinary (non-proxy) target whose own `length` and
/// `name` are data properties held in its object. `None`, having made
/// nothing, for anything else (the host's `bind` then runs).
fn fn_bind(
    realm: &mut Realm,
    t: Handle,
    this: NanBox,
    bound: alloc::vec::Vec<NanBox>,
) -> Option<NanBox> {
    use crate::nbexec::{BOUND_ARGS, BOUND_TARGET, BOUND_THIS};
    if realm.multi_realm || !realm.is_vm_function(t) || realm.proxy_at(t).is_some() {
        return None;
    }
    // `HasOwnProperty(Target, "length")` and `Get(Target, "length")`, and
    // `Get(Target, "name")`: own data properties only.
    let own_data = |realm: &Realm, key: &str| {
        (realm.has_own(t, key) && realm.accessor(t, key).is_none())
            .then(|| realm.get_property(t, key))
            .flatten()
    };
    let len = own_data(realm, "length")?;
    let name = own_data(realm, "name")?;
    let l = match len.as_number() {
        Some(n) if !n.is_nan() => (crate::common::FloatExt::trunc(n) - bound.len() as f64).max(0.0),
        _ => 0.0,
    };
    let prefix = NanBox::handle(realm.new_string("bound ").to_raw());
    let bound_name = match name.as_handle().map(Handle::from_raw) {
        Some(nh) if realm.typeof_is_string(nh) => realm.add_checked(prefix, name)?,
        _ => prefix,
    };
    let proto = realm.object_proto(t);
    let obj = realm.new_object();
    realm.set_object_proto(obj, proto);
    realm.set_hidden_property(obj, BOUND_TARGET, NanBox::handle(t.to_raw()));
    realm.set_hidden_property(obj, BOUND_THIS, this);
    let arr = realm.new_array(bound);
    realm.set_hidden_property(obj, BOUND_ARGS, NanBox::handle(arr.to_raw()));
    realm.set_property(obj, "length", NanBox::number(l));
    realm.mark_hidden(obj, "length");
    realm.set_readonly_property(obj, "length");
    realm.set_property(obj, "name", bound_name);
    realm.mark_hidden(obj, "name");
    realm.set_readonly_property(obj, "name");
    Some(NanBox::handle(obj.to_raw()))
}
