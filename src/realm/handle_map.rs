//! A handle-keyed side table with an O(1) negative lookup.
//!
//! Several [`Realm`](super::Realm) side tables are `BTreeMap`s keyed by a raw
//! handle (`generation << 32 | index`). Most lookups are *misses* — "does this
//! array have named properties?" is asked on every built-in dispatch, and the
//! answer is almost always no — and a miss in a map holding an entry for every
//! built-in function walks several B-tree nodes. [`HandleKeyedMap`] keeps a
//! bitset over slot indices beside the map: a clear bit is a definite miss, a
//! set bit falls through to the map.
//!
//! The bitset is *conservative*: a bit is set on every insert and never cleared
//! by `remove`/`retain` (another generation of the same slot may still be
//! present, and a stale bit only costs the map lookup it would have done
//! anyway). It is rebuilt whenever the whole key set is replaced.
//!
//! A *hit* is answered by an open-addressed hash index over the same entries
//! (linear probing, kept at most half full, backward-shift deletion), so a
//! present key costs a multiply and a probe or two instead of a descent
//! through the B-tree. The tree stays the source of truth for ordered
//! iteration (`Deref`).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A `BTreeMap<u64, V>` keyed by raw handles, plus a slot-index bitset that
/// answers most misses without searching the tree. Reads that are not
/// point lookups (`iter`, `values`, `len`, …) go through `Deref`.
pub(crate) struct HandleKeyedMap<V> {
    map: BTreeMap<u64, V>,
    /// Bit `i` set ⇒ some key with slot index `i` may be present.
    bits: Vec<u64>,
    /// The hash index: exactly the entries of `map`, open-addressed (a power
    /// of two in length, at most half full; empty until the first insert).
    index: Vec<Option<(u64, V)>>,
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
    /// An empty map.
    pub(crate) const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            bits: Vec::new(),
            index: Vec::new(),
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

    /// Puts `(raw, v)` in the index (replacing an entry for `raw`), growing it
    /// first if that would make it more than half full.
    fn index_put(&mut self, raw: u64, v: V) {
        if (self.map.len() + 1) * 2 > self.index.len() {
            self.rebuild_index((self.map.len() + 1) * 2);
        }
        let mask = self.index.len() - 1;
        let mut i = bucket(raw, mask);
        loop {
            match &mut self.index[i] {
                slot @ None => {
                    *slot = Some((raw, v));
                    return;
                }
                Some((k, val)) if *k == raw => {
                    *val = v;
                    return;
                }
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    /// Removes `raw` from the index, shifting later entries of its probe run
    /// back so no lookup stops early at the hole.
    fn index_remove(&mut self, raw: u64) {
        let Some(mut hole) = self.find(raw) else {
            return;
        };
        let mask = self.index.len() - 1;
        self.index[hole] = None;
        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            let Some((k, _)) = self.index[j] else {
                return;
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

    /// Rebuilds the index from `map` with room for at least `min_buckets`.
    fn rebuild_index(&mut self, min_buckets: usize) {
        let n = min_buckets
            .max(self.map.len() * 2)
            .max(16)
            .next_power_of_two();
        self.index.clear();
        self.index.resize(n, None);
        let mask = n - 1;
        for (&raw, &v) in &self.map {
            let mut i = bucket(raw, mask);
            while self.index[i].is_some() {
                i = (i + 1) & mask;
            }
            self.index[i] = Some((raw, v));
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
        self.mark(raw);
        self.index_put(raw, v);
        self.map.insert(raw, v)
    }

    /// Removes and returns the entry for `raw`.
    pub(crate) fn remove(&mut self, raw: &u64) -> Option<V> {
        if !self.may_contain(*raw) {
            return None;
        }
        let old = self.map.remove(raw);
        if old.is_some() {
            self.index_remove(*raw);
        }
        old
    }

    /// Keeps only the entries for which `f` returns `true`.
    pub(crate) fn retain(&mut self, f: impl FnMut(&u64, &mut V) -> bool) {
        // `f` may also rewrite values, so the index is rebuilt either way.
        self.map.retain(f);
        self.rebuild_index(0);
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

impl<V> core::ops::Deref for HandleKeyedMap<V> {
    type Target = BTreeMap<u64, V>;
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<V> IntoIterator for HandleKeyedMap<V> {
    type Item = (u64, V);
    type IntoIter = alloc::collections::btree_map::IntoIter<u64, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, V> IntoIterator for &'a HandleKeyedMap<V> {
    type Item = (&'a u64, &'a V);
    type IntoIter = alloc::collections::btree_map::Iter<'a, u64, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
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
    }

    #[test]
    fn the_hash_index_tracks_the_tree() {
        // A deterministic mix of inserts, overwrites and removes, checked
        // against the tree after every step (exercises growth and the
        // backward-shift deletion across wrapped probe runs).
        let mut m = HandleKeyedMap::new();
        let mut x: u64 = 0x1234_5678;
        for step in 0..4000u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let raw = ((x >> 40) & 3) << 32 | (x & 511);
            if x.is_multiple_of(3) {
                m.remove(&raw);
            } else {
                m.insert(raw, step);
            }
            if step % 97 == 0 {
                m.retain(|k, v| {
                    *v += 1;
                    k % 5 != 0
                });
            }
            for k in [raw, raw ^ 1, raw.wrapping_add(1 << 32)] {
                assert_eq!(m.get(&k), m.map.get(&k), "key {k:#x} at step {step}");
                assert_eq!(m.contains_key(&k), m.map.contains_key(&k));
            }
        }
        for (k, v) in m.map.clone() {
            assert_eq!(m.get(&k), Some(&v));
        }
    }
}
