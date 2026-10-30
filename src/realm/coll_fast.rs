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
    fn on_set(self) -> bool {
        matches!(self, CollOp::SetAdd | CollOp::SetHas | CollOp::SetDelete)
    }
}

impl Realm {
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
        let indexed = match self.heap.get(h)? {
            Cell::Collection {
                is_set,
                is_weak: false,
                index,
                ..
            } if *is_set == op.on_set() => index.is_some(),
            _ => return None,
        };
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
        if !indexed {
            self.collection_build_index(h);
        }
        let hash = self.collection_key_hash(a);
        let pos = self.collection_find_hashed(h, a, hash);
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
