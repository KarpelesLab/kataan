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
fn plain_lookup(realm: &Realm, start: Handle, key: &str) -> Option<Option<NanBox>> {
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

/// Whether `GetIterator(v)` would yield a built-in `%ArrayIterator%` over the
/// array `v` whose `next` is the intrinsic one, so that [`step`] may stand in
/// for it.
pub(super) fn can_open(realm: &Realm, v: NanBox) -> bool {
    let Some(h) = v.as_handle().map(Handle::from_raw) else {
        return false;
    };
    if !realm.is_array(h) {
        return false;
    }
    let ai = realm.vm_array_iter();
    let (Some(proto), Some(next_fn), Some(key)) =
        (ai.iter_proto, ai.next_fn, ai.iter_key.as_deref())
    else {
        return false;
    };
    // `GetMethod(v, @@iterator)` is an intrinsic `%Array.prototype.values%`,
    let Some(Some(f)) = plain_lookup(realm, h, key) else {
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
    matches!(plain_lookup(realm, proto, "next"),
        Some(Some(n)) if n.as_handle() == Some(next_fn.to_raw()))
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

/// Replaces a fast-mode iterator in `regs[iter]`/`regs[iter + 1]` with the
/// real `%ArrayIterator%` it stood in for (and the intrinsic `next`), returning
/// the iterator.
pub(super) fn materialize(
    ctx: &mut Ctx,
    regs: &mut [NanBox],
    iter: Reg,
) -> Result<NanBox, VmError> {
    let arr = array_of(regs, iter)?;
    let next_fn = ctx
        .realm
        .vm_array_iter()
        .next_fn
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
    if let Some(proto) = ctx.realm.vm_array_iter().iter_proto {
        match plain_lookup(&ctx.realm, proto, "return") {
            Some(None) => return Ok(()),
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
