//! The VM's allocation-free path for the array iteration protocol.
//!
//! `for (x of arr)`, `[...arr]`, `f(...arr)` and `[a, b] = arr` all run
//! `GetIterator` → `IteratorStep`* → (maybe) `IteratorClose`. When `arr` is a
//! plain array whose `@@iterator` resolves to the intrinsic
//! `%Array.prototype.values%`, and the fresh iterator's `next` would be the
//! intrinsic `%ArrayIteratorPrototype%.next`, nothing the program can observe
//! depends on the iterator object or on the `{ value, done }` result objects:
//! the iterator never escapes the compiled loop. So [`Op::IterOpen`] keeps the
//! *state* of that iterator in its two registers instead of allocating it:
//!
//! - `iter` holds the cursor as a Number (a real iterator is always an Object,
//!   so a Number there unambiguously marks this mode), and
//! - `next` (always `iter + 1` in this mode; checked when entering it) holds
//!   the array.
//!
//! Each step then does exactly what `CreateArrayIterator`'s closure does —
//! re-read the live `length`, stop at or past it, otherwise `Get(array, k)` —
//! reading a present element of a plain dense array directly and anything else
//! (a hole, an index accessor) through the host's `[[Get]]`. The `done`
//! register latches exhaustion as it does for a real iterator.
//!
//! The rare consumers that need the object itself — `IteratorClose` when some
//! `return` method is reachable from `%ArrayIteratorPrototype%`, and `yield*`,
//! which hands the iterator its own calls — first [`materialize`] it: the host
//! builds the very live `%ArrayIterator%` the fast path stood in for, positioned
//! at the cursor.
//!
//! A **string** iterates the same way when its `@@iterator` (from
//! `%String.prototype%`) is the intrinsic one and the fresh String Iterator's
//! `next` would be the intrinsic `%StringIteratorPrototype%.next`
//! ([`open_string`]): `next` then holds the (flattened) string and `iter` the
//! byte offset of the next code point in its WTF-8 storage, which is exactly
//! the state of the lazy String Iterator it stands in for. A string never
//! changes, so the step only decodes one code point.
//!
//! Every check is an identity comparison against handles the host recorded
//! while installing the intrinsics ([`crate::realm::VmArrayIterIntrinsics`]),
//! made at each `GetIterator`, so a later `Array.prototype[Symbol.iterator] =
//! …`, an own `@@iterator`, a replaced or deleted `%ArrayIteratorPrototype%.next`,
//! or a getter / proxy anywhere on those lookups simply takes the protocol path.

use super::{Ctx, HostError, Reg, VmError, with_host};
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::Realm;

/// The ordinary `[[Get]]` of the non-index key `key` from `start` up its
/// prototype chain, decided without running any code: `Some(Some(v))` for a
/// data property, `Some(None)` when no object on the chain has it, and `None`
/// when that cannot be told without side effects (an accessor, a proxy, an
/// exotic cell).
pub(super) fn plain_lookup(realm: &Realm, start: Handle, key: &str) -> Option<Option<NanBox>> {
    let mut cur = Some(start);
    // A prototype chain is acyclic, but stay bounded on a pathological one.
    for _ in 0..64 {
        let Some(c) = cur else {
            return Some(None);
        };
        if !realm.is_ordinary_get_cell(c) || realm.accessor(c, key).is_some() {
            return None;
        }
        if let Some(v) = realm.get_property(c, key) {
            return Some(Some(v));
        }
        cur = realm.object_proto(c);
    }
    None
}

/// [`plain_lookup`] through the realm's shared chain memo `slot` (a fixed
/// lookup site): a hit is the same answer without the walk.
fn plain_lookup_cached(
    realm: &Realm,
    slot: usize,
    start: Handle,
    key: &str,
) -> Option<Option<NanBox>> {
    realm
        .shared_chain_lookup(slot, start, key)
        .or_else(|| plain_lookup(realm, start, key))
}

/// The memo slots of this module's fixed lookups.
const SLOT_ITERATOR: usize = 0;
const SLOT_NEXT: usize = 1;
const SLOT_RETURN: usize = 2;

/// Whether `GetIterator(v)` would yield a built-in `%ArrayIterator%` over the
/// array `v` whose `next` is the intrinsic one, so that [`step`] may stand in
/// for it.
///
/// The two lookups behind the answer are memoized in the realm's
/// [`ArrayIterMemo`](crate::realm::builtins_fast::ArrayIterMemo), stamped with
/// the prototype epoch: while it is current, an array with no own named
/// properties whose prototype is the memo's needs only the two found values
/// re-read.
pub(super) fn can_open(realm: &mut Realm, v: NanBox) -> bool {
    can_open_array(realm, v)
}

/// Whether `GetIterator(v)` would yield a built-in String Iterator over the
/// string primitive `v` whose `next` is the intrinsic one, so that [`step`]
/// may stand in for it (see the module docs): then `v` is flattened in place
/// (its contents unchanged) for the steps to decode.
pub(super) fn open_string(realm: &mut Realm, v: NanBox) -> bool {
    let Some(h) = v.as_handle().map(Handle::from_raw) else {
        return false;
    };
    if realm.multi_realm || !realm.typeof_is_string(h) || !string_open_ok(realm) {
        return false;
    }
    realm.flatten_string(h);
    realm.string_leaf_bytes(h).is_some()
}

/// Whether `%String.prototype%[@@iterator]` is (by an ordinary data-property
/// lookup) the intrinsic `String.prototype[Symbol.iterator]` and
/// `%StringIteratorPrototype%.next` the intrinsic `next`: memoized at the
/// prototype epoch with both chains watched, so a hit re-reads the two found
/// values only.
fn string_open_ok(realm: &mut Realm) -> bool {
    let m = &realm.array_iter_memo;
    if m.str_stamp == crate::object::proto_epoch()
        && let Some((iter_at, next_at)) = &m.str_open
        && let (Some(f), Some(n)) = (realm.stamped_slot(iter_at), realm.stamped_slot(next_at))
    {
        return string_pair_ok(realm, f, n);
    }
    let ai = realm.vm_array_iter();
    let (Some(sp), Some(proto), Some(key)) = (
        realm.string_proto_intrinsic,
        ai.str_iter_proto,
        ai.iter_key.clone(),
    ) else {
        return false;
    };
    let (Some(Some(f)), Some(Some(n))) = (
        plain_lookup(realm, sp, &key),
        plain_lookup(realm, proto, "next"),
    ) else {
        return false;
    };
    if !string_pair_ok(realm, f, n) {
        return false;
    }
    realm.array_iter_memo.str_stamp = 0;
    if let Some(Some(iter_at)) = realm.watch_get_chain(sp, &key)
        && let Some(Some(next_at)) = realm.watch_get_chain(proto, "next")
    {
        let m = &mut realm.array_iter_memo;
        m.str_open = Some((iter_at, next_at));
        m.str_stamp = crate::object::proto_epoch();
    }
    true
}

/// Whether `f` is the intrinsic `String.prototype[Symbol.iterator]` and `n`
/// the intrinsic `%StringIteratorPrototype%.next`.
fn string_pair_ok(realm: &Realm, f: NanBox, n: NanBox) -> bool {
    f.as_handle().is_some_and(|fh| {
        realm.native_at(Handle::from_raw(fh)) == Some(crate::nbexec::N_STRING_PROTO_ITER)
    }) && realm
        .vm_array_iter()
        .str_next_fn
        .is_some_and(|nf| n.as_handle() == Some(nf.to_raw()))
}

/// The string of a fast-mode iterator (`regs[iter]` a Number), when it is a
/// string iteration rather than an array's.
fn string_of(realm: &Realm, regs: &[NanBox], iter: Reg) -> Option<Handle> {
    let h = Handle::from_raw(regs.get(iter as usize + 1)?.as_handle()?);
    realm.is_string_handle(h).then_some(h)
}

/// One step of a by-index string iteration over the string `s` (its byte
/// cursor in `regs[iter]`): the next code point as a string (a lone
/// surrogate as its own one-unit string), or `None` at the end, latching
/// `done`.
fn string_step(
    realm: &mut Realm,
    regs: &mut [NanBox],
    done: Reg,
    iter: Reg,
    s: Handle,
) -> Result<Option<NanBox>, VmError> {
    let mut idx = regs[iter as usize].as_number().unwrap_or(0.0) as usize;
    match string_code_point(realm, s, &mut idx)? {
        Some(v) => {
            regs[iter as usize] = NanBox::number(idx as f64);
            Ok(Some(v))
        }
        None => {
            regs[done as usize] = NanBox::boolean(true);
            Ok(None)
        }
    }
}

/// The code point of the flat string `s` at the byte offset `idx` (moved
/// past it) as a string — what the String Iterator's step yields, a lone
/// surrogate as its own one-unit string — or `None` at the end.
pub(super) fn string_code_point(
    realm: &mut Realm,
    s: Handle,
    idx: &mut usize,
) -> Result<Option<NanBox>, VmError> {
    let i = *idx;
    // (Flattened at the open; a string never becomes a rope again.)
    let Some(bytes) = realm.string_leaf_bytes(s) else {
        return Err(VmError::Unsupported);
    };
    let Some(&b0) = bytes.get(i) else {
        return Ok(None);
    };
    let v = if b0 < 0x80 {
        *idx = i + 1;
        realm.unit_string(u16::from(b0))
    } else {
        let (cp, n) = crate::wtf8::decode_code_point(bytes, i);
        let mut one = [0u8; 4];
        one[..n].copy_from_slice(&bytes[i..i + n]);
        *idx = i + n;
        if cp < 256 {
            realm.unit_string(cp as u16)
        } else {
            realm.new_string_from_rope(crate::rope::Rope::from_bytes(&one[..n]))
        }
    };
    Ok(Some(NanBox::handle(v.to_raw())))
}

/// `GetIterator(v)` of an iterator object whose `@@iterator` is
/// `%IteratorPrototype%[@@iterator]` (which returns `this`), found along a
/// chain of ordinary data properties: the iterator is `v` itself, and the
/// returned value is `Get(v, "next")` (also a data-property read). `None`
/// for anything whose lookups could run code or be exotic (an accessor, a
/// proxy), or another `@@iterator`: the host's `GetIterator` then runs.
pub(super) fn open_self_iterator(realm: &Realm, v: NanBox) -> Option<NanBox> {
    let h = Handle::from_raw(v.as_handle()?);
    if !realm.is_plain_object(h) {
        return None;
    }
    let key = realm.vm_array_iter().iter_key.as_deref()?;
    let f = super::array_hof::chain_data(realm, h, key).ok()??;
    if realm.native_at(Handle::from_raw(f.as_handle()?))
        != Some(crate::nbexec::N_ITERATOR_PROTO_SELF)
    {
        return None;
    }
    Some(
        super::array_hof::chain_data(realm, h, "next")
            .ok()?
            .unwrap_or(NanBox::undefined()),
    )
}

/// The first half of `GetIterator(v)` (ECMA-262 7.4.2, sync) for an
/// ordinary object whose `@@iterator`, along a chain of ordinary data
/// properties, is a VM function (a user-written `[Symbol.iterator]()`,
/// often a generator method): the method called on `v` in the VM and its
/// result checked to be an object — the iterator, which the caller roots in
/// its register before [`open_vm_next`] reads its `next`. `regs` is the
/// calling frame's register file, published for the call as
/// `call_closure_pub` does. `None` (nothing run) for anything else, which
/// the host's `GetIterator` then runs.
pub(super) fn open_vm_call(
    ctx: &mut Ctx,
    funcs: &[super::FnProto],
    regs: Option<&[NanBox]>,
    v: NanBox,
) -> Option<Result<NanBox, VmError>> {
    let h = Handle::from_raw(v.as_handle()?);
    if !ctx.realm.is_plain_object(h) {
        return None;
    }
    let key = ctx.realm.vm_array_iter().iter_key.as_deref()?;
    let f = super::array_hof::chain_data(&ctx.realm, h, key).ok()??;
    ctx.realm.vm_function(Handle::from_raw(f.as_handle()?))?;
    let it = match super::call_closure_compiled(ctx, funcs, regs, f, &[], v) {
        Ok(it) => it,
        Err(e) => return Some(Err(e)),
    };
    if !super::is_object_value(&ctx.realm, it) {
        let e = super::vm_error(
            ctx,
            "TypeError",
            "Result of the Symbol.iterator method is not an object",
        );
        return Some(Err(VmError::Thrown(e)));
    }
    Some(Ok(it))
}

/// IteratorStep (ECMA-262 7.4.8) of the iterator `it` whose `next` method
/// `nx` is a VM function (a hand-written iterator's): `next` called in the
/// VM with no argument (`regs`, the calling frame's register file, published
/// for it), its result required to be an object, then `Get(result, "done")`
/// and — while not done — `Get(result, "value")`, each a plain data read
/// when it is one and the host's `[[Get]]` otherwise (whose getters run
/// without lending registers: the result object, held here, needs no root
/// while nothing collects). `Some(Ok(None))` once done; `None` (nothing
/// run) when `nx` is not a VM function.
pub(super) fn vm_next_step(
    ctx: &mut Ctx,
    funcs: &[super::FnProto],
    regs: &[NanBox],
    it: NanBox,
    nx: NanBox,
) -> Option<Result<Option<NanBox>, VmError>> {
    ctx.realm.vm_function(Handle::from_raw(nx.as_handle()?))?;
    Some((|| {
        let res = super::call_closure_compiled(ctx, funcs, Some(regs), nx, &[], it)?;
        if !super::is_object_value(&ctx.realm, res) {
            let e = super::vm_error(ctx, "TypeError", "Iterator result is not an object");
            return Err(VmError::Thrown(e));
        }
        let d = super::get_str_plain(ctx, res, "done")?;
        if ctx.realm.truthy(d) {
            return Ok(None);
        }
        super::get_str_plain(ctx, res, "value").map(Some)
    })())
}

/// The second half of [`open_vm_call`]'s `GetIterator`: `Get(it, "next")`
/// of the iterator `it` it returned — a VM generator's straight from its
/// memoized prototype chain (`gen_fast::open`), anything else through
/// `[[Get]]`.
pub(super) fn open_vm_next(
    ctx: &mut Ctx,
    funcs: &[super::FnProto],
    it: NanBox,
) -> Result<NanBox, VmError> {
    if let Some(n) = super::gen_fast::open(ctx, funcs, it) {
        return Ok(n);
    }
    super::get_str_plain(ctx, it, "next")
}

/// `GetIterator(v)` of a Map or Set whose `@@iterator` is, along a chain of
/// ordinary data properties, the intrinsic `entries` (a Map's) / `values`
/// (a Set's) of its kind: the live iterator that method makes
/// (`Realm::new_live_collection_iterator`) and its `next` (a data-property
/// read). `None` (nothing made) for anything else, which the host's
/// `GetIterator` then runs.
pub(super) fn open_collection(realm: &mut Realm, v: NanBox) -> Option<(NanBox, NanBox)> {
    let h = Handle::from_raw(v.as_handle()?);
    let set = realm.collection_is_set(h)?;
    if realm.collection_is_weak(h) {
        return None;
    }
    let key = realm.vm_array_iter().iter_key.as_deref()?;
    // A collection's own properties (in its auxiliary object) are not an
    // ordinary object's: none of that key, then the walk from its prototype.
    if realm.is_host_exotic(h)
        || realm.accessor(h, key).is_some()
        || realm.get_property(h, key).is_some()
    {
        return None;
    }
    let f = super::array_hof::chain_data(realm, realm.object_proto(h)?, key).ok()??;
    let (id, name) = realm.bound_native_at(Handle::from_raw(f.as_handle()?))?;
    let want: &[u8] = if set { b"values" } else { b"entries" };
    let want_id = if set {
        crate::nbexec::N_SET_PROTO_FN
    } else {
        crate::nbexec::N_MAP_PROTO_FN
    };
    if id != want_id || realm.string_leaf_bytes(name) != Some(want) {
        return None;
    }
    let it = realm.new_live_collection_iterator(h, if set { 1 } else { 2 })?;
    let next = super::array_hof::chain_data(realm, it, "next")
        .ok()?
        .unwrap_or(NanBox::undefined());
    Some((NanBox::handle(it.to_raw()), next))
}

/// [`can_open`]'s body.
fn can_open_array(realm: &mut Realm, v: NanBox) -> bool {
    let Some(h) = v.as_handle().map(Handle::from_raw) else {
        return false;
    };
    if !realm.is_array(h) {
        return false;
    }
    // An array whose two sticky flags are clear owns no named property and
    // has the first `%Array.prototype%` (no side-table probe needed).
    let root = realm.plain_array_root_proto(h);
    let (own_named, proto) = match root {
        Some(p) => (false, Some(p)),
        None => (realm.may_own_named(h), realm.object_proto(h)),
    };
    if !own_named
        && let Some(p) = proto
        && let Some((ok, f, n)) = open_memo_hit(realm, p)
    {
        // For a plain array, publish the open to compiled code.
        if ok && root.is_some() {
            publish_open(realm, f, n);
        }
        return ok;
    }
    if !can_open_walk(realm, h) {
        return false;
    }
    // Memoize the answer for arrays like this one (no own named properties,
    // this prototype), watching both chains.
    if !own_named && let Some(p) = proto {
        arm_open_memo(realm, p);
    }
    true
}

/// The values `[...v]` would iterate, read without running code, when `v`
/// is a plain array (no side-table state, the first `%Array.prototype%`)
/// whose iteration [`can_open`]'s memo currently answers as the built-in
/// one, and none of whose indices below `length` is a hole: then every step
/// is `Get(v, k)` of a present own data element. `None` means "ask
/// [`can_open`]" (a cold or stale memo, any other array or value), never
/// "not plain". Unlike [`can_open`] it publishes nothing to compiled code.
#[inline]
pub(super) fn plain_spread_elems(realm: &Realm, v: NanBox) -> Option<&[NanBox]> {
    let (p, elems) = realm.plain_root_array_elems(Handle::from_raw(v.as_handle()?))?;
    let (ok, _, _) = open_memo_hit(realm, p)?;
    (ok && !elems.iter().any(|e| e.is_hole())).then_some(elems)
}

/// Whether `Get(iterator, "next")` on a fresh built-in `%ArrayIterator%`
/// (one with no own properties) yields the intrinsic
/// `%ArrayIteratorPrototype%.next`.
pub(super) fn array_iter_next_intact(realm: &Realm) -> bool {
    let ai = realm.vm_array_iter();
    let (Some(proto), Some(next_fn)) = (ai.iter_proto, ai.next_fn) else {
        return false;
    };
    matches!(plain_lookup_cached(realm, SLOT_NEXT, proto, "next"),
        Some(Some(n)) if n.as_handle() == Some(next_fn.to_raw()))
}

/// The values iterating `v` would yield, read without running code: `v` a
/// plain array with no hole whose iteration is the built-in one ([`can_open`],
/// [`plain_spread_elems`]). `None` for anything else.
pub(super) fn plain_array_values(realm: &mut Realm, v: NanBox) -> Option<&[NanBox]> {
    if plain_spread_elems(realm, v).is_none() && !can_open(realm, v) {
        return None;
    }
    plain_spread_elems(realm, v)
}

/// Publishes the memo's open — found valid now for the first
/// `%Array.prototype%`, with the values `f` (`@@iterator`) and `n` (`next`)
/// — to compiled code ([`ArrayIterJit`](crate::realm::builtins_fast::ArrayIterJit)),
/// with whether a fresh array's destructuring is plain at this epoch too.
fn publish_open(realm: &mut Realm, f: NanBox, n: NanBox) {
    #[cfg(all(feature = "jit", feature = "alloc", jit_native))]
    {
        let m = &realm.array_iter_memo;
        let stamp = m.open_stamp;
        // A fresh array's prototype is the memo's (the first one), and no
        // `return` is reachable for its close (see `fresh_pair_is_plain`).
        let fresh = if m.close_stamp == stamp
            && realm.root_array_proto().is_some()
            && realm.fresh_array_proto() == realm.root_array_proto()
            && realm.root_array_proto().map(Handle::to_raw) == Some(m.open_proto)
        {
            stamp
        } else {
            u64::MAX
        };
        if let Some((iter_at, next_at)) = &realm.array_iter_memo.open
            && let Some(iter_ptr) = realm
                .jit_object(Handle::from_raw(iter_at.0))
                .and_then(|o| o.jit_slot_ptr(iter_at.2))
            && let Some(next_ptr) = realm
                .jit_object(Handle::from_raw(next_at.0))
                .and_then(|o| o.jit_slot_ptr(next_at.2))
        {
            realm.array_iter_jit = crate::realm::builtins_fast::ArrayIterJit {
                stamp,
                iter_ptr: iter_ptr as u64,
                iter_fn: f.to_bits(),
                next_ptr: next_ptr as u64,
                next_fn: n.to_bits(),
                fresh,
            };
        }
    }
    #[cfg(not(all(feature = "jit", feature = "alloc", jit_native)))]
    let _ = (realm, f, n);
}

/// The memoized [`can_open`] answer for an array with no own named
/// properties and the prototype `p`, while the memo's stamp is current:
/// whether both found values are still the intrinsics, and those values.
fn open_memo_hit(realm: &Realm, p: Handle) -> Option<(bool, NanBox, NanBox)> {
    let m = &realm.array_iter_memo;
    if m.open_stamp != crate::object::proto_epoch() || m.open_proto != p.to_raw() {
        return None;
    }
    let (iter_at, next_at) = m.open.as_ref()?;
    let (f, n) = (realm.stamped_slot(iter_at)?, realm.stamped_slot(next_at)?);
    let ai = realm.vm_array_iter();
    let ok = f
        .as_handle()
        .is_some_and(|fh| ai.values_fns.iter().any(|x| x.to_raw() == fh))
        && ai
            .next_fn
            .is_some_and(|nf| n.as_handle() == Some(nf.to_raw()));
    Some((ok, f, n))
}

/// Memoizes the lookups behind [`can_open`] for arrays with no own named
/// properties and the prototype `p`, watching both chains.
fn arm_open_memo(realm: &mut Realm, p: Handle) {
    realm.array_iter_memo.open_stamp = 0;
    realm.array_iter_jit = crate::realm::builtins_fast::ArrayIterJit::default();
    let ai = realm.vm_array_iter();
    let (Some(iter_proto), Some(key)) = (ai.iter_proto, ai.iter_key.clone()) else {
        return;
    };
    if let Some(Some(iter_at)) = realm.watch_get_chain(p, &key)
        && let Some(Some(next_at)) = realm.watch_get_chain(iter_proto, "next")
    {
        let m = &mut realm.array_iter_memo;
        m.open = Some((iter_at, next_at));
        m.open_proto = p.to_raw();
        m.open_stamp = crate::object::proto_epoch();
    }
}

/// [`can_open`] by the lookups themselves.
fn can_open_walk(realm: &Realm, h: Handle) -> bool {
    let ai = realm.vm_array_iter();
    let (Some(proto), Some(next_fn), Some(key)) =
        (ai.iter_proto, ai.next_fn, ai.iter_key.as_deref())
    else {
        return false;
    };
    // `GetMethod(v, @@iterator)` is an intrinsic `%Array.prototype.values%`,
    // An array's own named properties are not shared, so only the walk from
    // its prototype is memoized.
    let found = if realm.may_own_named(h) && realm.has_own(h, key) {
        plain_lookup(realm, h, key)
    } else {
        match realm.object_proto(h) {
            Some(p) => plain_lookup_cached(realm, SLOT_ITERATOR, p, key),
            None => Some(None),
        }
    };
    let Some(Some(f)) = found else {
        return false;
    };
    if !f
        .as_handle()
        .is_some_and(|fh| ai.values_fns.iter().any(|x| x.to_raw() == fh))
    {
        return false;
    }
    // and `Get(iterator, "next")` on the fresh iterator (no own properties of
    // that name) is the intrinsic `next`.
    matches!(plain_lookup_cached(realm, SLOT_NEXT, proto, "next"),
        Some(Some(n)) if n.as_handle() == Some(next_fn.to_raw()))
}

/// Whether destructuring a fresh two-element array (as a Map/Set `entries`
/// step would make) with `[a, b]` runs no code and so can be skipped: its
/// `@@iterator` (from the fresh array's prototype) is the intrinsic
/// `%Array.prototype.values%`, the iterator's `next` is the intrinsic one,
/// and no `return` is reachable for the close after the two steps — exactly
/// what [`can_open`] and [`close`] would find for that array.
///
/// While the prototype epoch is unchanged, the answer comes from the same
/// stamped memo [`can_open`] keeps (plus [`close`]'s "no `return`" stamp),
/// re-reading only the two found values; a positive walk arms both.
fn fresh_pair_is_plain(realm: &mut Realm) -> bool {
    let Some(array_proto) = realm.fresh_array_proto() else {
        return false;
    };
    if realm.array_iter_memo.close_stamp == crate::object::proto_epoch()
        && let Some((ok, f, n)) = open_memo_hit(realm, array_proto)
    {
        // Published for compiled code's own test (`Op::FreshArrayIterPlain`)
        // when the memo is the first `%Array.prototype%`'s.
        if ok && realm.root_array_proto() == Some(array_proto) {
            publish_open(realm, f, n);
        }
        return ok;
    }
    if !fresh_pair_walk(realm) {
        return false;
    }
    arm_open_memo(realm, array_proto);
    if let Some(proto) = realm.vm_array_iter().iter_proto
        && let Some(None) = realm.watch_get_chain(proto, "return")
    {
        realm.array_iter_memo.close_stamp = crate::object::proto_epoch();
    }
    true
}

/// Whether destructuring a fresh array literal (all elements present) with
/// `[t1, …, tn]` runs no code: what [`fresh_pair_is_plain`] decides, which
/// depends on neither length (the steps read present own elements, and a
/// close finds no `return`).
pub(super) fn fresh_array_is_plain(realm: &mut Realm) -> bool {
    fresh_pair_is_plain(realm)
}

/// [`fresh_pair_is_plain`] by the lookups themselves.
fn fresh_pair_walk(realm: &Realm) -> bool {
    let ai = realm.vm_array_iter();
    let (Some(proto), Some(next_fn), Some(key), Some(array_proto)) = (
        ai.iter_proto,
        ai.next_fn,
        ai.iter_key.as_deref(),
        realm.fresh_array_proto(),
    ) else {
        return false;
    };
    let Some(Some(f)) = plain_lookup_cached(realm, SLOT_ITERATOR, array_proto, key) else {
        return false;
    };
    if !f
        .as_handle()
        .is_some_and(|fh| ai.values_fns.iter().any(|x| x.to_raw() == fh))
    {
        return false;
    }
    if !matches!(plain_lookup_cached(realm, SLOT_NEXT, proto, "next"),
        Some(Some(n)) if n.as_handle() == Some(next_fn.to_raw()))
    {
        return false;
    }
    match plain_lookup_cached(realm, SLOT_RETURN, proto, "return") {
        Some(None) => true,
        Some(Some(v)) => v.is_undefined() || v.is_null(),
        None => false,
    }
}

/// [`super::Op::IterNextEntry`]'s fused step: `Some(Some((key, value)))` for
/// the next entry of a built-in Map/Set `entries` iterator, `Some(None)`
/// once it is done (latching `done`), and `None` — having changed nothing —
/// when the step cannot skip the `[key, value]` array (another iterator, a
/// replaced `next`, an observable destructuring).
pub(super) fn entry_step(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    done: Reg,
    iter: Reg,
    next: Reg,
) -> Option<Option<(NanBox, NanBox)>> {
    let (it, nx) = (regs[iter as usize], regs[next as usize]);
    if it.is_number() || regs[done as usize].as_boolean() == Some(true) {
        return None;
    }
    if !fresh_pair_is_plain(&mut ctx.realm) {
        return None;
    }
    let step = crate::nbexec::realm_builtin_entry_step(&mut ctx.realm, nx, it)?;
    if step.is_none() {
        regs[done as usize] = NanBox::boolean(true);
    }
    Some(step)
}

/// The array of a fast-mode iterator whose cursor is in `iter`.
fn array_of(regs: &[NanBox], iter: Reg) -> Result<NanBox, VmError> {
    let arr = regs
        .get(iter as usize + 1)
        .copied()
        .ok_or(VmError::Unsupported)?;
    if arr.as_handle().is_none() {
        return Err(VmError::Unsupported);
    }
    Ok(arr)
}

/// One `IteratorStep` of a fast-mode iterator (`regs[iter]` is a Number):
/// `Ok(None)` once exhausted, latching `done`.
pub(super) fn step(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    done: Reg,
    iter: Reg,
) -> Result<Option<NanBox>, VmError> {
    if regs[done as usize].as_boolean() == Some(true) {
        return Ok(None);
    }
    if let Some(s) = string_of(&ctx.realm, regs, iter) {
        return string_step(&mut ctx.realm, regs, done, iter, s);
    }
    let arr_v = array_of(regs, iter)?;
    let arr = Handle::from_raw(arr_v.as_handle().unwrap_or_default());
    let idx = regs[iter as usize].as_number().unwrap_or(0.0) as usize;
    // `len = LengthOfArrayLike(array)`, re-read each step; `index ≥ len` ends
    // the iteration for good.
    let len = ctx.realm.array_length(arr).unwrap_or(0);
    if idx >= len {
        regs[done as usize] = NanBox::boolean(true);
        return Ok(None);
    }
    regs[iter as usize] = NanBox::number((idx + 1) as f64);
    // `Get(array, ToString(index))`: a present element of a dense array with no
    // index overrides is the element itself.
    if !ctx.realm.array_has_index_overrides(arr)
        && let Some(&v) = ctx.realm.array_elements(arr).and_then(|e| e.get(idx))
        && !v.is_hole()
    {
        return Ok(Some(v));
    }
    // A hole (inherited lookup) or an index accessor: the host's `[[Get]]`. A
    // throw there ends the iterator (`done` stays set: it must not be closed).
    regs[done as usize] = NanBox::boolean(true);
    let key = NanBox::number(idx as f64);
    match with_host(ctx, |h| h.get_member(arr_v, key)) {
        Some(Ok(v)) => {
            regs[done as usize] = NanBox::boolean(false);
            Ok(Some(v))
        }
        Some(Err(e)) => Err(e.into()),
        None => Err(VmError::Unsupported),
    }
}

/// The run of [`step`]s a rest/spread drain would take from a fast-mode
/// iterator (`regs[iter]` is a Number) that each read a present element of a
/// plain dense array directly: those elements are appended at once, to the
/// plain array `into` when given (the array literal or argument list the
/// spread feeds, which nothing can observe until it is complete) or else to
/// `out`, and the cursor moves past them. Nothing between such steps can
/// observe or change the array (they run no code), so the run is the slice
/// up to the first index that a step would hand to the host (a hole, an
/// index override, the end of the dense store); the caller's ordinary steps
/// take over from there, and latch `done` at the end.
pub(super) fn drain_plain_prefix(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    done: Reg,
    iter: Reg,
    into: Option<Handle>,
    out: &mut alloc::vec::Vec<NanBox>,
) {
    if regs[done as usize].as_boolean() == Some(true) {
        return;
    }
    // A string's steps run no code either: every remaining code point.
    if let Some(s) = string_of(&ctx.realm, regs, iter) {
        while let Ok(Some(v)) = string_step(&mut ctx.realm, regs, done, iter, s) {
            out.push(v);
        }
        return;
    }
    let Ok(arr_v) = array_of(regs, iter) else {
        return;
    };
    let arr = Handle::from_raw(arr_v.as_handle().unwrap_or_default());
    if ctx.realm.array_has_index_overrides(arr) {
        return;
    }
    let idx = regs[iter as usize].as_number().unwrap_or(0.0) as usize;
    let len = ctx.realm.array_length(arr).unwrap_or(0);
    if let Some(dst) = into
        && let Some(run) = ctx.realm.array_append_run_plain(dst, arr, idx, len, false)
    {
        regs[iter as usize] = NanBox::number((idx + run) as f64);
        return;
    }
    let Some(elems) = ctx.realm.array_elements(arr) else {
        return;
    };
    let end = len.min(elems.len());
    let Some(tail) = elems.get(idx..end) else {
        return;
    };
    let run = tail.iter().position(|e| e.is_hole()).unwrap_or(tail.len());
    out.extend_from_slice(&tail[..run]);
    regs[iter as usize] = NanBox::number((idx + run) as f64);
}

/// Replaces a fast-mode iterator in `regs[iter]`/`regs[iter + 1]` with the
/// real `%ArrayIterator%` it stood in for (and the intrinsic `next`), returning
/// the iterator.
pub(super) fn materialize(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    iter: Reg,
) -> Result<NanBox, VmError> {
    let arr = array_of(regs, iter)?;
    let ai = ctx.realm.vm_array_iter();
    let next_fn = if string_of(&ctx.realm, regs, iter).is_some() {
        ai.str_next_fn
    } else {
        ai.next_fn
    }
    .ok_or(VmError::Unsupported)?;
    let idx = regs[iter as usize].as_number().unwrap_or(0.0) as usize;
    let it = with_host(ctx, |h| h.array_iterator_at(arr, idx)).ok_or(VmError::Unsupported)?;
    if it.as_handle().is_none() {
        return Err(VmError::Unsupported);
    }
    regs[iter as usize] = it;
    regs[iter as usize + 1] = NanBox::handle(next_fn.to_raw());
    Ok(it)
}

/// `IteratorClose` of a fast-mode iterator: `GetMethod(iterator, "return")`
/// resolves on `%ArrayIteratorPrototype%`'s chain. Absent (the built-in case),
/// there is nothing to call; otherwise the iterator is made real and closed
/// through the host.
pub(super) fn close(ctx: &mut Ctx, regs: &mut [NanBox], iter: Reg) -> Result<(), HostError> {
    if string_of(&ctx.realm, regs, iter).is_some() {
        return close_string(ctx, regs, iter);
    }
    // No `return` anywhere on the chain, validated at the current epoch.
    if ctx.realm.array_iter_memo.close_stamp == crate::object::proto_epoch() {
        return Ok(());
    }
    if let Some(proto) = ctx.realm.vm_array_iter().iter_proto {
        match plain_lookup_cached(&ctx.realm, SLOT_RETURN, proto, "return") {
            Some(None) => {
                if let Some(None) = ctx.realm.watch_get_chain(proto, "return") {
                    ctx.realm.array_iter_memo.close_stamp = crate::object::proto_epoch();
                }
                return Ok(());
            }
            Some(Some(v)) if v.is_undefined() || v.is_null() => return Ok(()),
            _ => {}
        }
    }
    let it = match materialize(ctx, regs, iter) {
        Ok(it) => it,
        Err(VmError::Thrown(e)) => return Err(HostError::Thrown(e)),
        Err(_) => return Err(HostError::Fault),
    };
    with_host(ctx, |h| h.iter_close(it)).unwrap_or(Err(HostError::Fault))
}

/// Whether `IteratorClose` of a built-in `%ArrayIterator%` left unfinished
/// would call nothing (no `return` reachable from
/// `%ArrayIteratorPrototype%`), decided without running code: [`close`]'s
/// own first test (which arms the same memo), for a caller that must know
/// before it starts.
pub(super) fn close_is_noop(realm: &mut Realm) -> bool {
    if realm.array_iter_memo.close_stamp == crate::object::proto_epoch() {
        return true;
    }
    let Some(proto) = realm.vm_array_iter().iter_proto else {
        return false;
    };
    match plain_lookup_cached(realm, SLOT_RETURN, proto, "return") {
        Some(None) => {
            if let Some(None) = realm.watch_get_chain(proto, "return") {
                realm.array_iter_memo.close_stamp = crate::object::proto_epoch();
            }
            true
        }
        Some(Some(v)) => v.is_undefined() || v.is_null(),
        None => false,
    }
}

/// [`close`] of a by-index string iteration: `GetMethod(iterator, "return")`
/// resolves on `%StringIteratorPrototype%`'s chain, absent in the built-in
/// case; otherwise the iterator is made real and closed through the host.
fn close_string(ctx: &mut Ctx, regs: &mut [NanBox], iter: Reg) -> Result<(), HostError> {
    if ctx.realm.array_iter_memo.str_close_stamp == crate::object::proto_epoch() {
        return Ok(());
    }
    if let Some(proto) = ctx.realm.vm_array_iter().str_iter_proto {
        match plain_lookup(&ctx.realm, proto, "return") {
            Some(None) => {
                if let Some(None) = ctx.realm.watch_get_chain(proto, "return") {
                    ctx.realm.array_iter_memo.str_close_stamp = crate::object::proto_epoch();
                }
                return Ok(());
            }
            Some(Some(v)) if v.is_undefined() || v.is_null() => return Ok(()),
            _ => {}
        }
    }
    let it = match materialize(ctx, regs, iter) {
        Ok(it) => it,
        Err(VmError::Thrown(e)) => return Err(HostError::Thrown(e)),
        Err(_) => return Err(HostError::Fault),
    };
    with_host(ctx, |h| h.iter_close(it)).unwrap_or(Err(HostError::Fault))
}

/// Whether `GetMethod(iterator, "return")` of a fresh String Iterator finds
/// nothing (nothing on `%StringIteratorPrototype%`'s chain), decided without
/// running code: `false` when something is found or it cannot be told.
pub(super) fn string_return_absent(realm: &Realm) -> bool {
    if realm.array_iter_memo.str_close_stamp == crate::object::proto_epoch() {
        return true;
    }
    realm
        .vm_array_iter()
        .str_iter_proto
        .is_some_and(|p| matches!(plain_lookup(realm, p, "return"), Some(None)))
}
