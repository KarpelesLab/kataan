//! Sparse-aware stepping for the per-index `Array.prototype` algorithms.
//!
//! The spec algorithms of `indexOf`, `splice`, `copyWithin`, `reverse`, … walk
//! every index below `length`, which may be up to `2**32 - 1` on a real array
//! whose `length` was set past its dense store (`a.length = 2**32 - 1`). A
//! visit to an index the array does not own is a `HasProperty` that answers
//! `false` (or a `Get` that answers `undefined`) with no other effect — as long
//! as nothing on the prototype chain can answer for that index (an inherited
//! index property, a proxy, an exotic). These helpers find the next index a
//! loop must actually visit, re-deriving it after every step so that user code
//! run by a getter, setter or callback (which may add or remove indices, or
//! change the prototype chain) is honored: when the chain is not provably
//! index-free, or the receiver is not a real array, they skip nothing.
//!
//! Inside the dense store every index is visited as before; only the region
//! past it (where the elements live in the auxiliary store, so they are few)
//! is skipped.

use super::*;

impl Interp {
    /// The first index in `k..end` that a per-index algorithm must visit on
    /// `handle` (`end` if none): `k` itself unless `k` lies past the dense store
    /// of a real array whose absent indices are unobservable, in which case the
    /// smallest own index at or after `k`.
    pub(crate) fn sparse_next_index(&self, handle: Handle, k: usize, end: usize) -> usize {
        if k >= end {
            return end;
        }
        if !self.sparse_skip_possible(handle, k) {
            return k;
        }
        self.realm
            .array_aux_indices(handle)
            .into_iter()
            .filter(|&i| i >= k && i < end)
            .min()
            .unwrap_or(end)
    }

    /// The last index in `lo..=k` that a descending per-index algorithm must
    /// visit on `handle` (`None` if none): the mirror of
    /// [`sparse_next_index`](Self::sparse_next_index).
    pub(crate) fn sparse_prev_index(&self, handle: Handle, k: usize, lo: usize) -> Option<usize> {
        if k < lo {
            return None;
        }
        if !self.sparse_skip_possible(handle, k) {
            return Some(k);
        }
        let dense = self.realm.array_elements(handle).map_or(0, <[_]>::len);
        // `k` is past the dense store, so its last slot (if in range) is below `k`.
        let in_dense = (dense > lo).then(|| dense - 1);
        let in_aux = self
            .realm
            .array_aux_indices(handle)
            .into_iter()
            .filter(|&i| i >= lo && i <= k)
            .max();
        in_aux.max(in_dense)
    }

    /// The first position `p` in `k..end` for which index `p + a` or `p + b`
    /// must be visited (a copy loop reading `p + a` and writing `p + b`).
    pub(crate) fn sparse_next_pair(
        &self,
        handle: Handle,
        k: usize,
        end: usize,
        a: usize,
        b: usize,
    ) -> usize {
        let na = self.sparse_next_index(handle, k + a, end + a) - a;
        let nb = self.sparse_next_index(handle, k + b, end + b) - b;
        na.min(nb)
    }

    /// The last position `p` in `lo..=k` for which index `p + a` or `p + b` must
    /// be visited (the descending form of
    /// [`sparse_next_pair`](Self::sparse_next_pair)).
    pub(crate) fn sparse_prev_pair(
        &self,
        handle: Handle,
        k: usize,
        lo: usize,
        a: usize,
        b: usize,
    ) -> Option<usize> {
        let pa = self.sparse_prev_index(handle, k + a, lo + a).map(|i| i - a);
        let pb = self.sparse_prev_index(handle, k + b, lo + b).map(|i| i - b);
        pa.max(pb)
    }

    /// Whether index `k` may be skipped over: `handle` is a real (non-typed)
    /// array, `k` lies past its dense store, and no prototype can answer for an
    /// index.
    fn sparse_skip_possible(&self, handle: Handle, k: usize) -> bool {
        self.realm.typed_kind(handle).is_none()
            && self
                .realm
                .array_elements(handle)
                .is_some_and(|a| k >= a.len())
            && self.realm.proto_chain_index_free(handle)
    }

    /// Whether `handle` is a real array whose `length` lies past its dense
    /// store (`a.length = 2**32 - 1`, or an index set past the dense cap). The
    /// element arms of the array built-ins index the dense store and take its
    /// size for `length`, so such an array runs the per-index algorithms.
    pub(crate) fn array_length_past_dense(&self, handle: Handle) -> bool {
        self.realm.typed_kind(handle).is_none()
            && self
                .realm
                .array_elements(handle)
                .map(<[_]>::len)
                .is_some_and(|dense| self.realm.array_length(handle) != Some(dense))
    }

    /// `Array.prototype.slice` by its index-by-index algorithm (23.1.3.28):
    /// `HasProperty`/`Get` of each index in `[k, final)` (skipping the absent
    /// region past the dense store), `CreateDataPropertyOrThrow` into the
    /// species result, then `Set(A, "length", count, true)`.
    pub(crate) fn array_slice_generic(
        &mut self,
        handle: Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        let len = self.array_like_length(handle)?;
        let start = self.relative_index(arg(0), len, 0)?;
        let fin = self.relative_index(arg(1), len, len)?;
        let count = fin.saturating_sub(start);
        let out_v = self.array_species_create(handle, count)?;
        let Some(out) = out_v.as_handle().map(Handle::from_raw) else {
            return Err(self.type_error("Array species did not return an object"));
        };
        let mut k = start;
        while k < fin {
            k = self.sparse_next_index(handle, k, fin);
            if k >= fin {
                break;
            }
            let key = alloc::format!("{k}");
            if self.has_property_proxied(handle, &key)? {
                let v = self.read_member(handle, &key)?;
                self.create_data_property_or_throw(out, k - start, v)?;
            }
            k += 1;
        }
        let len_key = self.new_str("length");
        let saved_strict = self.strict;
        self.strict = true;
        let r = self.assign_member_value(out, len_key, NanBox::number(count as f64));
        self.strict = saved_strict;
        r?;
        Ok(out_v)
    }
}
