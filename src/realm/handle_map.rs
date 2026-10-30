//! A handle-keyed side table with an O(1) negative lookup.
//!
//! Several [`Realm`](super::Realm) side tables are keyed by a raw handle
//! (`generation << 32 | index`). Most lookups are *misses* — "does this array
//! have named properties?" is asked on every built-in dispatch, and the answer
//! is almost always no. [`HandleKeyedMap`] keeps a bitset over slot indices: a
//! clear bit is a definite miss, a set bit falls through to the table.
//!
//! The bitset is *conservative*: a bit is set on every insert and never cleared
//! by `remove`/`retain` (another generation of the same slot may still be
//! present, and a stale bit only costs the probe it would have done anyway).
//! It is rebuilt whenever the whole key set is replaced.
//!
//! The entries live in an open-addressed hash table (linear probing, kept at
//! most half full, backward-shift deletion), so a hit or an insert costs a
//! multiply and a probe or two. The table is the only store: iteration
//! (`iter`, `values`, `into_iter`) visits the entries in table order, which
//! is unspecified. Every caller iterates only to collect or mark (the GC's
//! roots and ephemeron passes, a compaction's forwarding), where order does
//! not matter. (An ordered `BTreeMap` copy used to be kept for that, and its
//! insert was most of the cost of creating an array's or a RegExp's side
//! object.)

use alloc::vec::Vec;

/// A map keyed by raw handles, plus a slot-index bitset that answers most
/// misses without probing the table.
pub(crate) struct HandleKeyedMap<V> {
    /// Bit `i` set ⇒ some key with slot index `i` may be present.
    bits: Vec<u64>,
    /// The entries, open-addressed (a power of two in length, at most half
    /// full; empty until the first insert).
    index: Vec<Option<(u64, V)>>,
    /// How many entries `index` holds.
    len: usize,
    /// For a map made by [`new_logged`](HandleKeyedMap::new_logged): every
    /// key inserted since [`clear_fresh`](HandleKeyedMap::clear_fresh), in no
    /// particular order and possibly repeated (the realm's minor collections
    /// visit only these; see `Realm::gc_side_log`). `None` for a map that
    /// keeps no log.
    fresh: Option<Vec<u64>>,
}

impl<V: Copy> Default for HandleKeyedMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn slot_of(raw: u64) -> usize {
    (raw & 0xffff_ffff) as usize
}

/// The home bucket of `raw` in an index of `mask + 1` buckets.
#[inline]
fn bucket(raw: u64, mask: usize) -> usize {
    (raw.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & mask
}

impl<V: Copy> HandleKeyedMap<V> {
    /// An empty map that logs the keys inserted into it (see `fresh`).
    pub(crate) fn new_logged() -> Self {
        let mut m = Self::new();
        m.fresh = Some(Vec::new());
        m
    }

    /// The keys inserted since the last [`clear_fresh`](Self::clear_fresh)
    /// (empty for a map that keeps no log).
    pub(crate) fn fresh_keys(&self) -> &[u64] {
        self.fresh.as_deref().unwrap_or(&[])
    }

    /// Takes the logged keys out (leaving the log empty).
    pub(crate) fn take_fresh(&mut self) -> Vec<u64> {
        self.fresh.as_mut().map(core::mem::take).unwrap_or_default()
    }

    /// Forgets the logged keys.
    pub(crate) fn clear_fresh(&mut self) {
        if let Some(f) = &mut self.fresh {
            f.clear();
        }
    }

    /// An empty map.
    pub(crate) const fn new() -> Self {
        Self {
            fresh: None,
            bits: Vec::new(),
            index: Vec::new(),
            len: 0,
        }
    }

    /// The index bucket holding `raw`, if present.
    #[inline]
    fn find(&self, raw: u64) -> Option<usize> {
        if self.index.is_empty() {
            return None;
        }
        let mask = self.index.len() - 1;
        let mut i = bucket(raw, mask);
        loop {
            match &self.index[i] {
                None => return None,
                Some((k, _)) if *k == raw => return Some(i),
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    /// Puts `(raw, v)` in the table (replacing an entry for `raw`, which it
    /// returns), growing it first if a new entry would make it more than half
    /// full.
    fn index_put(&mut self, raw: u64, v: V) -> Option<V> {
        if (self.len + 1) * 2 > self.index.len() {
            self.rebuild_index((self.len + 1) * 2);
        }
        let mask = self.index.len() - 1;
        let mut i = bucket(raw, mask);
        loop {
            match &mut self.index[i] {
                slot @ None => {
                    *slot = Some((raw, v));
                    self.len += 1;
                    return None;
                }
                Some((k, val)) if *k == raw => {
                    return Some(core::mem::replace(val, v));
                }
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    /// Removes `raw` from the index, shifting later entries of its probe run
    /// back so no lookup stops early at the hole.
    fn index_remove(&mut self, raw: u64) -> Option<V> {
        let mut hole = self.find(raw)?;
        let mask = self.index.len() - 1;
        let old = self.index[hole].take().map(|(_, v)| v);
        self.len -= 1;
        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            let Some((k, _)) = self.index[j] else {
                return old;
            };
            let home = bucket(k, mask);
            // The entry at `j` may fill the hole iff its home bucket is not
            // cyclically within (hole, j].
            let movable = if hole <= j {
                home <= hole || home > j
            } else {
                home <= hole && home > j
            };
            if movable {
                self.index[hole] = self.index[j].take();
                hole = j;
            }
        }
    }

    /// Re-places every entry into a table with room for at least
    /// `min_buckets` (and at least twice the entry count).
    fn rebuild_index(&mut self, min_buckets: usize) {
        let n = min_buckets.max(self.len * 2).max(16).next_power_of_two();
        let old = core::mem::replace(&mut self.index, alloc::vec![None; n]);
        let mask = n - 1;
        for (raw, v) in old.into_iter().flatten() {
            let mut i = bucket(raw, mask);
            while self.index[i].is_some() {
                i = (i + 1) & mask;
            }
            self.index[i] = Some((raw, v));
        }
    }

    /// Makes room for `additional` more entries without the table growing.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let want = (self.len + additional) * 2;
        if want > self.index.len() {
            self.rebuild_index(want);
        }
    }

    #[inline]
    fn may_contain(&self, raw: u64) -> bool {
        let i = slot_of(raw);
        self.bits
            .get(i / 64)
            .is_some_and(|w| (w >> (i % 64)) & 1 != 0)
    }

    fn mark(&mut self, raw: u64) {
        let i = slot_of(raw);
        let w = i / 64;
        if w >= self.bits.len() {
            self.bits.resize(w + 1, 0);
        }
        self.bits[w] |= 1 << (i % 64);
    }

    /// The value for `raw`, if present.
    #[inline]
    pub(crate) fn get(&self, raw: &u64) -> Option<&V> {
        if !self.may_contain(*raw) {
            return None;
        }
        self.find(*raw)
            .and_then(|i| self.index[i].as_ref())
            .map(|(_, v)| v)
    }

    /// Whether `raw` has an entry.
    #[inline]
    pub(crate) fn contains_key(&self, raw: &u64) -> bool {
        self.may_contain(*raw) && self.find(*raw).is_some()
    }

    /// Inserts `v` under `raw`, returning the value it replaced.
    pub(crate) fn insert(&mut self, raw: u64, v: V) -> Option<V> {
        if let Some(f) = &mut self.fresh {
            f.push(raw);
        }
        self.mark(raw);
        self.index_put(raw, v)
    }

    /// Removes and returns the entry for `raw`.
    pub(crate) fn remove(&mut self, raw: &u64) -> Option<V> {
        if !self.may_contain(*raw) {
            return None;
        }
        self.index_remove(*raw)
    }

    /// Keeps only the entries for which `f` returns `true`.
    pub(crate) fn retain(&mut self, mut f: impl FnMut(&u64, &mut V) -> bool) {
        // Dropping entries breaks probe runs, so the table is rebuilt.
        for slot in &mut self.index {
            if let Some((k, v)) = slot
                && !f(k, v)
            {
                *slot = None;
                self.len -= 1;
            }
        }
        self.rebuild_index(0);
    }

    /// The number of entries.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The entries, in table order.
    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&u64, &V)> {
        self.index.iter().flatten().map(|(k, v)| (k, v))
    }

    /// The values, in table order.
    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.index.iter().flatten().map(|(_, v)| v)
    }

    /// Exchanges whatever entries `a` and `b` hold (either, both or neither).
    pub(crate) fn swap_entries(&mut self, a: u64, b: u64) {
        let va = self.remove(&a);
        let vb = self.remove(&b);
        if let Some(v) = vb {
            self.insert(a, v);
        }
        if let Some(v) = va {
            self.insert(b, v);
        }
    }
}

impl<V> IntoIterator for HandleKeyedMap<V> {
    type Item = (u64, V);
    type IntoIter = core::iter::Flatten<alloc::vec::IntoIter<Option<(u64, V)>>>;
    fn into_iter(self) -> Self::IntoIter {
        self.index.into_iter().flatten()
    }
}

impl<'a, V> IntoIterator for &'a HandleKeyedMap<V> {
    type Item = (&'a u64, &'a V);
    type IntoIter = core::iter::Map<
        core::iter::Flatten<core::slice::Iter<'a, Option<(u64, V)>>>,
        fn(&'a (u64, V)) -> (&'a u64, &'a V),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.index.iter().flatten().map(|(k, v)| (k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::HandleKeyedMap;

    #[test]
    fn misses_hits_and_generations() {
        let mut m = HandleKeyedMap::new();
        assert_eq!(m.get(&5), None);
        m.insert(5, 'a');
        m.insert((3 << 32) | 200, 'b');
        assert_eq!(m.get(&5), Some(&'a'));
        assert_eq!(m.get(&((3 << 32) | 200)), Some(&'b'));
        // Same slot, other generation: the bit is set, the map still decides.
        assert_eq!(m.get(&((1 << 32) | 5)), None);
        assert!(!m.contains_key(&6));
        m.swap_entries(5, 6);
        assert_eq!(m.get(&5), None);
        assert_eq!(m.get(&6), Some(&'a'));
        m.retain(|_, v| *v != 'a');
        assert!(!m.contains_key(&6));
        assert_eq!(m.len(), 1);
        assert!(m.fresh_keys().is_empty(), "an unlogged map logs nothing");
    }

    #[test]
    fn a_logged_map_logs_every_insert() {
        let mut m = HandleKeyedMap::new_logged();
        m.insert(5, 'a');
        m.insert(7, 'b');
        m.swap_entries(5, 6);
        assert_eq!(m.fresh_keys(), &[5, 7, 6]);
        assert_eq!(m.take_fresh(), alloc::vec![5, 7, 6]);
        assert!(m.fresh_keys().is_empty());
        m.insert(9, 'c');
        m.clear_fresh();
        assert!(m.fresh_keys().is_empty());
        assert_eq!(m.get(&9), Some(&'c'));
    }

    #[test]
    fn the_table_tracks_a_reference_map() {
        // A deterministic mix of inserts, overwrites and removes, checked
        // against a reference map after every step (exercises growth and the
        // backward-shift deletion across wrapped probe runs).
        let mut m = HandleKeyedMap::new();
        let mut r = alloc::collections::BTreeMap::new();
        let mut x: u64 = 0x1234_5678;
        for step in 0..4000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let raw = ((x >> 40) & 3) << 32 | (x & 511);
            if x.is_multiple_of(3) {
                assert_eq!(m.remove(&raw), r.remove(&raw));
            } else {
                assert_eq!(m.insert(raw, step), r.insert(raw, step));
            }
            if step % 97 == 0 {
                let keep = |k: &u64, v: &mut u64| {
                    *v += 1;
                    !k.is_multiple_of(5)
                };
                m.retain(keep);
                r.retain(keep);
            }
            for k in [raw, raw ^ 1, raw.wrapping_add(1 << 32)] {
                assert_eq!(m.get(&k), r.get(&k), "key {k:#x} at step {step}");
                assert_eq!(m.contains_key(&k), r.contains_key(&k));
            }
            assert_eq!(m.len(), r.len());
        }
        let mut all: alloc::vec::Vec<(u64, u64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        all.sort_unstable();
        assert_eq!(all, r.into_iter().collect::<alloc::vec::Vec<_>>());
    }
}
