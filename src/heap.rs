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

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// The base of the high generation range reserved for compaction-produced
/// slots, kept disjoint from the low generations that free/reuse produce.
const COMPACT_GEN_BASE: u16 = 0x8000;

/// A stable reference to a heap slot: an index plus the generation it was live
/// in. Comparing the stored generation against the slot's detects staleness.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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
    Occupied {
        generation: u16,
        /// Collections this object has survived (for generational GC): `0` is
        /// the young generation; promoted objects age up.
        age: u8,
        value: T,
    },
    Free {
        generation: u16,
    },
}

/// A generational arena of `T`, addressed by [`Handle`].
pub struct Heap<T> {
    slots: Vec<Slot<T>>,
    /// Indices of free slots, reused before growing.
    free: Vec<u32>,
    /// The number of live (occupied) slots.
    live: usize,
    /// The generation stamped on slots produced by the next compaction. Drawn
    /// from a high range (`>= COMPACT_GEN_BASE`) disjoint from the low
    /// generations free/reuse produces, so a *new* (forwarded) handle can never
    /// equal an *old* handle key — which keeps reference fix-up idempotent even
    /// for shared, off-heap structures (closure scopes) visited more than once.
    compact_gen: u16,
    /// The **remembered set**: slot indices of old-generation objects that have
    /// been written with a young pointer since the last minor collection (the
    /// write barrier records them). A minor collection scans only these as old
    /// roots instead of the whole old generation.
    remembered: BTreeSet<u32>,
}

impl<T> Default for Heap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Heap<T> {
    /// Creates an empty heap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            compact_gen: COMPACT_GEN_BASE,
            remembered: BTreeSet::new(),
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
            *slot = Slot::Occupied {
                generation,
                age: 0,
                value,
            };
            Handle { index, generation }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot::Occupied {
                generation: 0,
                age: 0,
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
            Slot::Occupied {
                generation, value, ..
            } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Mutably borrows the value behind `handle`, with the same staleness check
    /// as [`get`](Heap::get).
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize)? {
            Slot::Occupied {
                generation, value, ..
            } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Whether `handle` still refers to a live object.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.get(handle).is_some()
    }

    /// A handle to every currently-live slot (used by the collector to find the
    /// sweep set). Order is by slot index.
    #[must_use]
    pub fn live_handles(&self) -> Vec<Handle> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { generation, .. } => Some(Handle {
                    index: i as u32,
                    generation: *generation,
                }),
                Slot::Free { .. } => None,
            })
            .collect()
    }

    /// The generational age of `handle`'s object (collections survived), or
    /// `None` if stale. `0` is the young generation.
    #[must_use]
    pub fn age(&self, handle: Handle) -> Option<u8> {
        match self.slots.get(handle.index as usize)? {
            Slot::Occupied {
                generation, age, ..
            } if *generation == handle.generation => Some(*age),
            _ => None,
        }
    }

    /// Promotes `handle`'s object by one generation (saturating), marking it as
    /// having survived another collection.
    pub fn tenure(&mut self, handle: Handle) {
        if let Some(Slot::Occupied {
            generation, age, ..
        }) = self.slots.get_mut(handle.index as usize)
            && *generation == handle.generation
        {
            *age = age.saturating_add(1);
        }
    }

    /// Handles to live objects whose age satisfies `keep` — e.g. the young
    /// generation (`|a| a == 0`) for a minor collection, or the old generation
    /// (`|a| a >= 1`) for the remembered root scan.
    #[must_use]
    pub fn handles_where(&self, keep: impl Fn(u8) -> bool) -> Vec<Handle> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied {
                    generation, age, ..
                } if keep(*age) => Some(Handle {
                    index: i as u32,
                    generation: *generation,
                }),
                _ => None,
            })
            .collect()
    }

    /// The **write barrier**: records `container` in the remembered set when an
    /// old-generation `container` is made to point at a young-generation
    /// `target` (an old→young edge a minor collection must treat as a root).
    /// A no-op for young containers or old→old edges.
    pub fn record_edge(&mut self, container: Handle, target: Handle, old_age: u8) {
        if self.age(container).is_some_and(|a| a >= old_age)
            && self.age(target).is_some_and(|a| a < old_age)
        {
            self.remembered.insert(container.index);
        }
    }

    /// Handles to the remembered-set objects that are still live (the old→young
    /// roots for a minor collection).
    #[must_use]
    pub fn remembered_roots(&self) -> Vec<Handle> {
        self.remembered
            .iter()
            .filter_map(|&index| match self.slots.get(index as usize) {
                Some(Slot::Occupied { generation, .. }) => Some(Handle {
                    index,
                    generation: *generation,
                }),
                _ => None,
            })
            .collect()
    }

    /// Clears the remembered set (after a collection re-establishes the
    /// generation boundary).
    pub fn clear_remembered(&mut self) {
        self.remembered.clear();
    }

    /// Compacts the `keep` (live) objects to the front of the slot table,
    /// discarding everything else, and returns the forwarding map
    /// `(old_handle → new_handle)`. Slot generations reset to `0`, so every
    /// pre-compaction handle is invalid *except* via the returned map — the
    /// moving collector uses it to fix up references. Ages (generations
    /// survived) are preserved; the free list and remembered set are cleared.
    pub fn compact_to(&mut self, keep: &BTreeSet<Handle>) -> Vec<(Handle, Handle)> {
        // A fresh high-range generation for this compaction's slots, disjoint
        // from any old handle's generation.
        let new_gen = self.compact_gen;
        self.compact_gen = self.compact_gen.wrapping_add(1).max(COMPACT_GEN_BASE);

        let old_slots = core::mem::take(&mut self.slots);
        self.free.clear();
        self.remembered.clear();
        let mut new_slots: Vec<Slot<T>> = Vec::new();
        let mut map: Vec<(Handle, Handle)> = Vec::new();
        for (i, slot) in old_slots.into_iter().enumerate() {
            if let Slot::Occupied {
                generation,
                age,
                value,
            } = slot
            {
                let old = Handle {
                    index: i as u32,
                    generation,
                };
                if keep.contains(&old) {
                    let new = Handle {
                        index: new_slots.len() as u32,
                        generation: new_gen,
                    };
                    new_slots.push(Slot::Occupied {
                        generation: new_gen,
                        age,
                        value,
                    });
                    map.push((old, new));
                }
            }
        }
        self.slots = new_slots;
        self.live = self.slots.len();
        map
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
