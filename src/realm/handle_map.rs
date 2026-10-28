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

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A `BTreeMap<u64, V>` keyed by raw handles, plus a slot-index bitset that
/// answers most misses without searching the tree. Reads that are not
/// point lookups (`iter`, `values`, `len`, …) go through `Deref`.
pub(crate) struct HandleKeyedMap<V> {
    map: BTreeMap<u64, V>,
    /// Bit `i` set ⇒ some key with slot index `i` may be present.
    bits: Vec<u64>,
}

impl<V> Default for HandleKeyedMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn slot_of(raw: u64) -> usize {
    (raw & 0xffff_ffff) as usize
}

impl<V> HandleKeyedMap<V> {
    /// An empty map.
    pub(crate) const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            bits: Vec::new(),
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
        self.map.get(raw)
    }

    /// Whether `raw` has an entry.
    #[inline]
    pub(crate) fn contains_key(&self, raw: &u64) -> bool {
        self.may_contain(*raw) && self.map.contains_key(raw)
    }

    /// Inserts `v` under `raw`, returning the value it replaced.
    pub(crate) fn insert(&mut self, raw: u64, v: V) -> Option<V> {
        self.mark(raw);
        self.map.insert(raw, v)
    }

    /// Removes and returns the entry for `raw`.
    pub(crate) fn remove(&mut self, raw: &u64) -> Option<V> {
        if !self.may_contain(*raw) {
            return None;
        }
        self.map.remove(raw)
    }

    /// Keeps only the entries for which `f` returns `true`.
    pub(crate) fn retain(&mut self, f: impl FnMut(&u64, &mut V) -> bool) {
        self.map.retain(f);
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
}
