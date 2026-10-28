//! The lookup index of a `Map`/`Set` (`Cell::Collection`): an open-addressing
//! hash table from a key hash to the key's position in the collection's
//! insertion-ordered entry list.
//!
//! Each slot packs `(hash32 << 32) | (position + 1)` into one `u64`; `0` is an
//! empty slot and [`GONE`] a removed one (so probe chains stay intact). The
//! table never needs the keys themselves: lookups re-check candidates through a
//! caller-supplied equality, and growth re-places slots from their stored hash
//! bits, so neither ever touches the heap. Probing is linear from the slot the
//! hash's high bits pick, and the table doubles once live plus removed slots
//! reach half its size, so probe runs stay short.
//!
//! The hash only has to be *consistent* with the key equality
//! (`SameValueZero`): collisions cost a comparison, never a wrong answer.

use alloc::vec::Vec;

/// A removed slot: skipped by lookups, reused by inserts.
const GONE: u64 = u64::MAX;

/// The smallest table (slots) once one is built.
const MIN_SLOTS: usize = 8;

/// See the module docs.
#[derive(Clone, Debug, Default)]
pub struct CollIndex {
    /// The slots; empty, or a power-of-two length.
    slots: Vec<u64>,
    /// Occupied (live) slots.
    used: usize,
    /// [`GONE`] slots.
    gone: usize,
}

/// Folds a 64-bit key hash into the 32 stored bits (never all-ones together
/// with a position of `u32::MAX`, so a packed slot never equals [`GONE`]).
#[inline]
fn fold(hash: u64) -> u32 {
    (hash >> 32) as u32 ^ hash as u32
}

/// The home slot of folded hash `h` in a table of `mask + 1` slots.
#[inline]
fn home(h: u32, mask: usize) -> usize {
    // Fibonacci scrambling, so hashes differing only in high bits spread too.
    (h.wrapping_mul(0x9E37_79B9) as usize) & mask
}

impl CollIndex {
    /// An index with room for `n` entries before it grows.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let mut slots = MIN_SLOTS;
        while slots < n.saturating_mul(2) {
            slots *= 2;
        }
        CollIndex {
            slots: alloc::vec![0; slots],
            used: 0,
            gone: 0,
        }
    }

    /// The position of the first indexed entry hashing to `hash` for which
    /// `is_key(position)` holds.
    #[inline]
    pub fn find(&self, hash: u64, mut is_key: impl FnMut(usize) -> bool) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        let h = fold(hash);
        let mask = self.slots.len() - 1;
        let mut i = home(h, mask);
        loop {
            let s = self.slots[i];
            if s == 0 {
                return None;
            }
            if s != GONE && (s >> 32) as u32 == h {
                let pos = (s as u32 - 1) as usize;
                if is_key(pos) {
                    return Some(pos);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Records that the entry at `pos` (not yet indexed) has key hash `hash`.
    pub fn insert(&mut self, hash: u64, pos: usize) {
        if (self.used + self.gone + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let h = fold(hash);
        let packed = (u64::from(h) << 32) | (pos as u64 + 1);
        let mask = self.slots.len() - 1;
        let mut i = home(h, mask);
        loop {
            match self.slots[i] {
                0 => break,
                GONE => {
                    self.gone -= 1;
                    break;
                }
                _ => i = (i + 1) & mask,
            }
        }
        self.slots[i] = packed;
        self.used += 1;
    }

    /// Drops the slot recording that the entry at `pos` has key hash `hash`.
    pub fn remove(&mut self, hash: u64, pos: usize) {
        if self.slots.is_empty() {
            return;
        }
        let h = fold(hash);
        let packed = (u64::from(h) << 32) | (pos as u64 + 1);
        let mask = self.slots.len() - 1;
        let mut i = home(h, mask);
        loop {
            match self.slots[i] {
                0 => return,
                s if s == packed => {
                    self.slots[i] = GONE;
                    self.used -= 1;
                    self.gone += 1;
                    return;
                }
                _ => i = (i + 1) & mask,
            }
        }
    }

    /// Rebuilds without the removed slots, doubling until the live ones fill at
    /// most a quarter of the table (so a table full of removals is just
    /// re-placed at its size).
    fn grow(&mut self) {
        let mut len = self.slots.len().max(MIN_SLOTS);
        while (self.used + 1) * 4 > len {
            len *= 2;
        }
        let old = core::mem::replace(&mut self.slots, alloc::vec![0; len]);
        let mask = len - 1;
        for s in old {
            if s == 0 || s == GONE {
                continue;
            }
            let mut i = home((s >> 32) as u32, mask);
            while self.slots[i] != 0 {
                i = (i + 1) & mask;
            }
            self.slots[i] = s;
        }
        self.gone = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::CollIndex;

    #[test]
    fn insert_find_remove_and_grow() {
        let mut ix = CollIndex::default();
        // Many colliding hashes plus distinct ones, across several growths.
        for p in 0..1000usize {
            ix.insert((p % 7) as u64 * 0x1_0000_0001, p);
        }
        for p in 0..1000usize {
            let h = (p % 7) as u64 * 0x1_0000_0001;
            assert_eq!(ix.find(h, |q| q == p), Some(p));
        }
        for p in (0..1000usize).step_by(2) {
            ix.remove((p % 7) as u64 * 0x1_0000_0001, p);
        }
        for p in 0..1000usize {
            let h = (p % 7) as u64 * 0x1_0000_0001;
            let want = (p % 2 == 1).then_some(p);
            assert_eq!(ix.find(h, |q| q == p), want);
        }
        // Removed slots are reused without losing the survivors.
        for p in 1000..3000usize {
            ix.insert(p as u64, p);
        }
        assert_eq!(ix.find(0x1_0000_0001, |q| q == 1), Some(1));
        assert_eq!(ix.find(2999, |q| q == 2999), Some(2999));
        assert_eq!(ix.find(12345, |_| true), None);
    }
}
