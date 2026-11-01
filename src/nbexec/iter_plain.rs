//! Built-in consumers of an iterable (`new Map(iterable)`, `new Set(iterable)`,
//! `Array.from(iterable)`) over a plain array, without the iterator object.
//!
//! When `GetIterator(array)` would produce the built-in `%ArrayIterator%`
//! whose `next` is the intrinsic one — which [`crate::nbvm::array_iteration_is_plain`]
//! decides without running code — nothing the program can observe depends on
//! the iterator object or on its `{ value, done }` results: each step is
//! "re-read the live `length`, stop at or past it, otherwise `Get(array, k)`".
//! The consumers here run exactly those steps by index, reading a present
//! element of a plain dense array directly. At the first index whose `Get`
//! could run code or consult the prototype chain (a hole, an index accessor,
//! the store having been made sparse), or where the consumer would need the
//! iterator itself (an abrupt completion closes it), they hand the rest to
//! the protocol path through a real live `%ArrayIterator%` positioned at that
//! index — the very iterator the elided one would have been by then.

use super::*;

impl Interp {
    /// `items` as an array whose iteration is the plain built-in one.
    pub(crate) fn plain_iterable_array(&mut self, items: NanBox) -> Option<Handle> {
        let h = items.as_handle().map(Handle::from_raw)?;
        (self.realm.is_array(h) && crate::nbvm::array_iteration_is_plain(&mut self.realm, items))
            .then_some(h)
    }

    /// The step `k` of the elided iterator over `arr` when it needs no code:
    /// `Some(Some(v))` for a present element of the dense store (no index side
    /// table), `Some(None)` once `k` is at or past the live length, `None`
    /// when the step must take the full `[[Get]]`.
    pub(crate) fn plain_array_step(&self, arr: Handle, k: usize) -> Option<Option<NanBox>> {
        let len = self.realm.array_length(arr)?;
        if k >= len {
            return Some(None);
        }
        if self.realm.array_has_index_side_table(arr) {
            return None;
        }
        let v = *self.realm.array_elements(arr)?.get(k)?;
        (!v.is_hole()).then_some(Some(v))
    }

    /// A live `%ArrayIterator%` over `arr` (values) positioned at `k`, with
    /// its `next`: where the elided iterator stood.
    pub(crate) fn plain_array_iterator_at(
        &mut self,
        arr: Handle,
        k: usize,
    ) -> Result<(Handle, NanBox), ExecError> {
        let it = self.make_live_array_iterator(arr, 1);
        let ih = it
            .as_handle()
            .map(Handle::from_raw)
            .ok_or_else(|| self.type_error("value is not iterable"))?;
        self.realm
            .set_hidden_property(ih, GEN_IDX, NanBox::number(k as f64));
        let next = self.read_member(ih, "next")?;
        Ok((ih, next))
    }

    /// `AddEntriesFromIterable` of `new Map(arr)` / `new Set(arr)` (`is_set`)
    /// into the fresh collection `coll`, whose adder is the intrinsic
    /// `set` / `add`, over the elided iterator of the plain array `arr`: the
    /// index to resume from with a real iterator, or `None` once done. A Map
    /// entry is taken in place only when it is itself a plain array holding
    /// both `0` and `1` (a non-object entry, or any other shape, resumes
    /// there, where the protocol path reads it or throws and closes).
    pub(crate) fn collection_fill_plain(
        &mut self,
        coll: Handle,
        arr: Handle,
        is_set: bool,
    ) -> Option<usize> {
        let mut k = 0usize;
        loop {
            let Some(step) = self.plain_array_step(arr, k) else {
                return Some(k);
            };
            let entry = step?;
            if is_set {
                self.realm.collection_set(coll, entry, entry);
            } else {
                let Some(eh) = entry.as_handle().map(Handle::from_raw) else {
                    return Some(k);
                };
                let pair = (!self.realm.array_has_index_side_table(eh))
                    .then(|| self.realm.array_elements(eh))
                    .flatten()
                    .filter(|e| e.len() >= 2 && !e[0].is_hole() && !e[1].is_hole())
                    .map(|e| (e[0], e[1]));
                let Some((key, val)) = pair else {
                    return Some(k);
                };
                self.realm.collection_set(coll, key, val);
            }
            k += 1;
        }
    }
}
