//! A generational handle table — the managed heap that [`NanBox`] handles point
//! into (`ROADMAP.md` §3, the object model & GC).
//!
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! Heap objects are addressed by **handle**, not by raw pointer, so the
//! collector can relocate an object (compaction, generational promotion)
//! without rewriting every value that refers to it — it updates this table's
//! slot instead. Each handle carries a **generation** that is bumped when its
//! slot is freed, so a stale handle to a since-reclaimed object is detected
//! rather than silently aliasing whatever now occupies the slot (the classic
//! use-after-free a `slotmap` prevents).
//!
//! A handle packs into 48 bits (32-bit slot index + 16-bit generation), exactly
//! the payload width of a NaN-boxed handle, so `Handle::to_raw`/`from_raw` move
//! losslessly between the two.
//!
//! This is pure, safe `alloc`-only Rust (a `Vec` of slots plus a free list); the
//! tracing/compaction policy layers on top later.

use alloc::vec::Vec;

/// A stable reference to a heap slot: an index plus the generation it was live
/// in. Comparing the stored generation against the slot's detects staleness.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Handle {
    index: u32,
    generation: u16,
}

impl Handle {
    /// Packs the handle into its 48-bit form (for a NaN-boxed handle payload):
    /// `generation << 32 | index`.
    #[must_use]
    pub const fn to_raw(self) -> u64 {
        ((self.generation as u64) << 32) | self.index as u64
    }

    /// Unpacks a handle from its 48-bit form (the inverse of [`Self::to_raw`]).
    /// Bits above 48 are ignored.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self {
            index: (raw & 0xffff_ffff) as u32,
            generation: ((raw >> 32) & 0xffff) as u16,
        }
    }
}

/// A heap slot: either holds a live value or is free and awaiting reuse. Both
/// carry the generation so a freed-then-reallocated slot invalidates old
/// handles.
enum Slot<T> {
    Occupied { generation: u16, value: T },
    Free { generation: u16 },
}

/// A generational arena of `T`, addressed by [`Handle`].
pub struct Heap<T> {
    slots: Vec<Slot<T>>,
    /// Indices of free slots, reused before growing.
    free: Vec<u32>,
    /// The number of live (occupied) slots.
    live: usize,
}

impl<T> Default for Heap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Heap<T> {
    /// Creates an empty heap.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// The number of live objects.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live
    }

    /// Whether the heap holds no live objects.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Allocates `value`, reusing a freed slot when one is available, and
    /// returns a handle to it.
    pub fn alloc(&mut self, value: T) -> Handle {
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            // A freed slot already had its generation bumped; reuse it as-is.
            let generation = match slot {
                Slot::Free { generation } => *generation,
                Slot::Occupied { .. } => unreachable!("free list pointed at a live slot"),
            };
            *slot = Slot::Occupied { generation, value };
            Handle { index, generation }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot::Occupied {
                generation: 0,
                value,
            });
            Handle {
                index,
                generation: 0,
            }
        }
    }

    /// Borrows the value behind `handle`, or `None` if the handle is stale
    /// (its slot was freed/reused) or out of range.
    #[must_use]
    pub fn get(&self, handle: Handle) -> Option<&T> {
        match self.slots.get(handle.index as usize)? {
            Slot::Occupied { generation, value } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Mutably borrows the value behind `handle`, with the same staleness check
    /// as [`get`](Heap::get).
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize)? {
            Slot::Occupied { generation, value } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Whether `handle` still refers to a live object.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.get(handle).is_some()
    }

    /// Frees the slot behind `handle`, returning its value. The slot's
    /// generation is bumped so existing handles to it become stale. A stale or
    /// out-of-range handle frees nothing and returns `None`.
    pub fn free(&mut self, handle: Handle) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        match slot {
            Slot::Occupied { generation, .. } if *generation == handle.generation => {
                // Bump the generation (wrapping) and take the value out.
                let next_gen = generation.wrapping_add(1);
                let Slot::Occupied { value, .. } = core::mem::replace(
                    slot,
                    Slot::Free {
                        generation: next_gen,
                    },
                ) else {
                    unreachable!()
                };
                self.free.push(handle.index);
                self.live -= 1;
                Some(value)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_get_and_len() {
        let mut h: Heap<i32> = Heap::new();
        assert!(h.is_empty());
        let a = h.alloc(10);
        let b = h.alloc(20);
        assert_eq!(h.len(), 2);
        assert_eq!(h.get(a), Some(&10));
        assert_eq!(h.get(b), Some(&20));
        *h.get_mut(a).unwrap() = 11;
        assert_eq!(h.get(a), Some(&11));
    }

    #[test]
    fn free_returns_value_and_invalidates_handle() {
        let mut h: Heap<&str> = Heap::new();
        let a = h.alloc("hello");
        assert!(h.is_live(a));
        assert_eq!(h.free(a), Some("hello"));
        assert!(!h.is_live(a));
        assert_eq!(h.get(a), None);
        assert_eq!(h.len(), 0);
        // Double free is a no-op.
        assert_eq!(h.free(a), None);
    }

    #[test]
    fn reused_slot_makes_old_handle_stale() {
        let mut h: Heap<i32> = Heap::new();
        let a = h.alloc(1);
        h.free(a);
        // The next alloc reuses a's slot but with a bumped generation.
        let b = h.alloc(2);
        assert_eq!(b.to_raw() & 0xffff_ffff, a.to_raw() & 0xffff_ffff); // same index
        assert_ne!(a, b); // different generation
        assert_eq!(h.get(b), Some(&2));
        assert_eq!(h.get(a), None); // the old handle no longer resolves
    }

    #[test]
    fn free_list_reuses_before_growing() {
        let mut h: Heap<i32> = Heap::new();
        let a = h.alloc(1);
        let b = h.alloc(2);
        let c = h.alloc(3);
        h.free(b);
        // Reuses b's slot rather than allocating a fourth.
        let d = h.alloc(4);
        assert_eq!(h.slots.len(), 3);
        assert_eq!(h.len(), 3);
        assert_eq!(h.get(a), Some(&1));
        assert_eq!(h.get(c), Some(&3));
        assert_eq!(h.get(d), Some(&4));
    }

    #[test]
    fn handle_raw_round_trips() {
        for &(index, generation) in &[(0u32, 0u16), (1, 0), (42, 7), (0xffff_ffff, 0xffff)] {
            let raw = Handle { index, generation }.to_raw();
            assert!(raw <= 0x0000_ffff_ffff_ffff, "fits in 48 bits");
            let back = Handle::from_raw(raw);
            assert_eq!(back, Handle { index, generation });
        }
    }

    #[test]
    fn handles_index_into_the_heap_via_raw() {
        // The round-trip a NaN-boxed handle would take: store the raw payload,
        // reconstruct the handle, resolve the object.
        let mut h: Heap<i32> = Heap::new();
        let handle = h.alloc(99);
        let raw = handle.to_raw();
        let reconstructed = Handle::from_raw(raw);
        assert_eq!(h.get(reconstructed), Some(&99));
    }
}
