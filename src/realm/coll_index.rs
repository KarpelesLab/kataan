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
    /// How many entries at the front of the collection's entry list are all
    /// tombstones (each of `entries[..head]` is one; 0 is always true): where
    /// a search for the first live entry may start, so a queue that deletes
    /// its oldest entry each time (`s.delete(s.values().next().value)`)
    /// finds the next one in O(1) amortized rather than rescanning the dead
    /// prefix. Kept here, with the index, because it goes stale exactly when
    /// the index does (a compaction renumbers the entries).
    pub head: u32,
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

/// Where compiled code finds an index's slots (`crate::jit::coll`'s inline
/// probe, which follows [`CollIndex::find`]: [`fold`], [`home`], linear
/// probing, `0` empty, [`GONE`] removed): the byte offsets, from the
/// index, of its slot buffer's pointer word and its slot count word
/// (`slots`, a `Vec<u64>`, whose words are found on a probe). `None` when
/// they cannot be told apart.
#[cfg(all(feature = "jit", jit_x64))]
#[must_use]
pub(crate) fn jit_slots_layout() -> Option<(usize, usize)> {
    use core::mem::{offset_of, size_of};
    if size_of::<Vec<u64>>() != 24 {
        return None;
    }
    let mut v: Vec<u64> = Vec::with_capacity(16);
    v.extend_from_slice(&[7, 8, 9]);
    // SAFETY: a `Vec` is three initialized `usize` words (checked above).
    #[allow(unsafe_code)]
    let w: [usize; 3] = unsafe { core::ptr::read((&v as *const Vec<u64>).cast::<[usize; 3]>()) };
    let one = |x: usize| {
        let mut k = (0..3).filter(|&k| w[k] == x);
        match (k.next(), k.next()) {
            (Some(k), None) => Some(k * 8),
            _ => None,
        }
    };
    let (ptr, len) = (one(v.as_ptr() as usize)?, one(v.len())?);
    one(v.capacity())?;
    let s = offset_of!(CollIndex, slots);
    Some((s + ptr, s + len))
}

/// The byte offsets, from an index, of its `used` and `gone` counts (both
/// `usize`), which compiled code's inline insert (`crate::jit::coll`) reads
/// for [`CollIndex::insert_at`]'s growth test and bumps as it does.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn jit_counts_layout() -> (usize, usize) {
    (
        core::mem::offset_of!(CollIndex, used),
        core::mem::offset_of!(CollIndex, gone),
    )
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
            head: 0,
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

    /// [`find`](Self::find), and on a miss where [`insert`](Self::insert)
    /// would then place an entry hashing to `hash` were the table not to
    /// grow first: the first removed or empty slot of the probe sequence
    /// (`usize::MAX` for an empty table). For
    /// [`insert_at`](Self::insert_at), with nothing changed in between.
    #[inline]
    pub fn find_free(
        &self,
        hash: u64,
        mut is_key: impl FnMut(usize) -> bool,
    ) -> (Option<usize>, usize) {
        if self.slots.is_empty() {
            return (None, usize::MAX);
        }
        let h = fold(hash);
        let mask = self.slots.len() - 1;
        let mut i = home(h, mask);
        let mut free = usize::MAX;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return (None, if free == usize::MAX { i } else { free });
            }
            if s == GONE {
                if free == usize::MAX {
                    free = i;
                }
            } else if (s >> 32) as u32 == h {
                let pos = (s as u32 - 1) as usize;
                if is_key(pos) {
                    return (Some(pos), usize::MAX);
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// [`insert`](Self::insert), into slot `free` as [`find_free`]
    /// (Self::find_free) found it when the table need not grow (the slot
    /// `insert` would take); otherwise `insert` itself.
    #[inline]
    pub fn insert_at(&mut self, free: usize, hash: u64, pos: usize) {
        if free >= self.slots.len() || (self.used + self.gone + 1) * 2 > self.slots.len() {
            self.insert(hash, pos);
            return;
        }
        let h = fold(hash);
        match self.slots[free] {
            0 => {}
            GONE => self.gone -= 1,
            _ => {
                self.insert(hash, pos);
                return;
            }
        }
        self.slots[free] = (u64::from(h) << 32) | (pos as u64 + 1);
        self.used += 1;
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
