//! The lean form of the `Map`/`Set` methods a memoized call site runs
//! (`nbvm::native_call`): each one checks the receiver's kind and does the
//! hash probe with as few heap reads as the operation needs, rather than the
//! separate `collection_is_set` / `collection_is_weak` / `collection_find`
//! round trips of the general helpers. The semantics are exactly those
//! helpers' (`SameValueZero` keys, `-0` stored as `+0`, insertion order,
//! tombstoned deletes that live iterators step over).

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;

/// Whether the lean hash-index paths are used: a short String key hashed
/// from its node's words, the probe comparing String texts with one heap
/// read per candidate, and the last lookup's entry reused by the update
/// that follows it (`KATAAN_COLLFAST=0` turns them off; read once per
/// process).
pub(crate) fn lean_enabled() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("KATAAN_COLLFAST").is_none_or(|v| v != "0"))
    }
    #[cfg(not(feature = "std"))]
    true
}

/// A `Map.prototype` / `Set.prototype` method served by [`Realm::coll_fast`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CollOp {
    MapGet,
    MapSet,
    MapHas,
    MapDelete,
    SetAdd,
    SetHas,
    SetDelete,
}

impl CollOp {
    /// Whether the method is a `Set.prototype` one (its receiver must be a Set).
    #[inline]
    pub(crate) fn on_set(self) -> bool {
        matches!(self, CollOp::SetAdd | CollOp::SetHas | CollOp::SetDelete)
    }
}

impl Realm {
    /// Whether the cell at `h` is a strong `Set` (`Some(true)`) or `Map`
    /// (`Some(false)`); `None` for anything else.
    #[inline]
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn collection_kind(&self, h: Handle) -> Option<bool> {
        match self.heap.get(h)? {
            Cell::Collection {
                is_set,
                is_weak: false,
                ..
            } => Some(*is_set),
            _ => None,
        }
    }

    /// Whether the stored key `k` is the lookup key `a` by `SameValueZero`
    /// on the cheap: the very bits, or two Strings of the same text. (A
    /// `false` may be a miss of another kind of equal key: callers then
    /// probe.)
    #[inline]
    fn coll_same_key(&self, k: NanBox, a: NanBox) -> bool {
        if k.to_bits() == a.to_bits() {
            return !k.is_hole();
        }
        match (k.as_handle(), a.as_handle()) {
            (Some(x), Some(y)) => matches!(
                (self.heap.get_hot(Handle::from_raw(x)), self.heap.get_hot(Handle::from_raw(y))),
                (Some(Cell::Str(p)), Some(Cell::Str(q))) if p.same_bytes(q)
            ),
            _ => false,
        }
    }

    /// `op` on the receiver `h` with arguments `a` and `b` (`undefined` when
    /// absent): its result, or `None`, having done nothing, when `h` is not a
    /// strong collection of the method's kind (a WeakMap, a Set for a `Map`
    /// method, not a collection at all, …), which the general path then
    /// rejects with its `TypeError`.
    pub(crate) fn coll_fast(
        &mut self,
        h: Handle,
        op: CollOp,
        a: NanBox,
        b: NanBox,
    ) -> Option<NanBox> {
        let lean = lean_enabled();
        // One read of the cell: its kind, and for a collection with a hash
        // index a lookup or update by the lean probe (`coll_probe_lean`):
        // the position found, the key's hash and the value there.
        let (indexed, hinted, probed) = match self.heap.get_hot(h)? {
            Cell::Collection {
                is_set,
                is_weak: false,
                index,
                entries,
                ..
            } if *is_set == op.on_set() => {
                // The entry the last lookup found, when it is this
                // collection's entry for this key (see `Realm::coll_hint`).
                let hinted = (lean
                    && matches!(op, CollOp::MapSet | CollOp::SetAdd)
                    && self.coll_hint.0 == h.to_raw()
                    && self.coll_hint.1 == a.to_bits())
                .then_some(self.coll_hint.2 as usize)
                .filter(|&i| {
                    entries
                        .get(i)
                        .is_some_and(|(k, _)| self.coll_same_key(*k, a))
                });
                let probed = match index {
                    Some(map)
                        if lean
                            && hinted.is_none()
                            && !matches!(op, CollOp::MapDelete | CollOp::SetDelete) =>
                    {
                        self.coll_probe_lean(map, entries, a).map(|(pos, hash)| {
                            (pos, hash, pos.and_then(|i| entries.get(i)).map(|e| e.1))
                        })
                    }
                    _ => None,
                };
                (index.is_some(), hinted, probed)
            }
            _ => return None,
        };
        if let Some(i) = hinted {
            // `set` replaces the hinted entry's value, `add` changes nothing.
            if op == CollOp::MapSet {
                if let Some(Cell::Collection { entries, .. }) = self.heap.get_mut(h)
                    && let Some(e) = entries.get_mut(i)
                {
                    e.1 = b;
                }
                self.write_barrier(h, b);
            }
            return Some(NanBox::handle(h.to_raw()));
        }
        if let Some((pos, hash, found)) = probed {
            let hint = |i: usize| u32::try_from(i).ok().map(|i| (h.to_raw(), a.to_bits(), i));
            match op {
                CollOp::MapGet | CollOp::MapHas | CollOp::SetHas => {
                    if let Some(t) = pos.and_then(hint) {
                        self.coll_hint = t;
                    }
                    return Some(match op {
                        CollOp::MapGet => found.unwrap_or(NanBox::undefined()),
                        _ => NanBox::boolean(pos.is_some()),
                    });
                }
                CollOp::MapSet | CollOp::SetAdd => {
                    // As `collection_set`: a `-0` key stored as `+0`, a
                    // Set's value its key; a key found has its value
                    // replaced (a Set's left alone), else the entry is
                    // appended (the barrier's tail form) and indexed.
                    let key = match a.as_number() {
                        Some(n) if n == 0.0 && n.is_sign_negative() => NanBox::number(0.0),
                        _ => a,
                    };
                    let value = if op == CollOp::SetAdd { key } else { b };
                    if let Some(i) = pos {
                        if op == CollOp::MapSet {
                            if let Some(Cell::Collection { entries, .. }) = self.heap.get_mut(h)
                                && let Some(e) = entries.get_mut(i)
                            {
                                e.1 = value;
                            }
                            self.write_barrier(h, value);
                        }
                        if let Some(t) = hint(i) {
                            self.coll_hint = t;
                        }
                        return Some(NanBox::handle(h.to_raw()));
                    }
                    let Some(Cell::Collection {
                        entries,
                        live,
                        index: Some(map),
                        ..
                    }) = self.heap.get_mut_append(h, Cell::ref_len)
                    else {
                        return Some(NanBox::handle(h.to_raw()));
                    };
                    let i = entries.len();
                    entries.push((key, value));
                    *live += 1;
                    map.insert(hash, i);
                    self.shade_stored(key);
                    self.shade_stored(value);
                    if let Some(t) = hint(i) {
                        self.coll_hint = t;
                    }
                    return Some(NanBox::handle(h.to_raw()));
                }
                CollOp::MapDelete | CollOp::SetDelete => {}
            }
        }
        match op {
            CollOp::MapSet => {
                self.collection_set(h, a, b);
                return Some(NanBox::handle(h.to_raw()));
            }
            CollOp::SetAdd => {
                self.collection_set(h, a, a);
                return Some(NanBox::handle(h.to_raw()));
            }
            CollOp::MapDelete | CollOp::SetDelete => {
                return Some(NanBox::boolean(self.collection_delete(h, a)));
            }
            CollOp::MapGet | CollOp::MapHas | CollOp::SetHas => {}
        }
        let pos = match self.collection_find_ro(h, a) {
            Some(pos) => pos,
            None => {
                if !indexed {
                    self.collection_build_index(h);
                }
                let hash = self.collection_key_hash(a);
                self.collection_find_hashed(h, a, hash)
            }
        };
        if let Some(i) = pos.and_then(|i| u32::try_from(i).ok())
            && lean_enabled()
        {
            self.coll_hint = (h.to_raw(), a.to_bits(), i);
        }
        Some(match op {
            CollOp::MapGet => match (pos, self.heap.get(h)) {
                (Some(i), Some(Cell::Collection { entries, .. })) => {
                    entries.get(i).map_or(NanBox::undefined(), |e| e.1)
                }
                _ => NanBox::undefined(),
            },
            _ => NanBox::boolean(pos.is_some()),
        })
    }
}
