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

    /// The slot index this handle addresses.
    #[must_use]
    pub(crate) const fn index(self) -> u32 {
        self.index
    }

    /// The slot generation this handle was live in.
    #[must_use]
    pub(crate) const fn generation(self) -> u16 {
        self.generation
    }

    /// Reassembles a handle from its parts (the inverse of
    /// [`index`](Self::index) / [`generation`](Self::generation)).
    #[must_use]
    pub(crate) const fn from_parts(index: u32, generation: u16) -> Self {
        Self { index, generation }
    }
}

/// A heap slot: either holds a live value or is free and awaiting reuse. Both
/// carry the generation so a freed-then-reallocated slot invalidates old
/// handles.
///
/// A struct rather than an enum so the slot adds only its few header bytes to
/// the value: `Option<T>` takes the value's own niche (a `Cell`'s tag), and
/// the generation, age and occupancy byte share one word. Compiled code reads
/// all of them at probed offsets (`jit_slot_layout`).
struct Slot<T> {
    /// The value, or `None` for a free slot.
    value: Option<T>,
    /// The generation: a live slot's handles carry it; a free slot holds the
    /// one its next occupant will get.
    generation: u16,
    /// Collections this object has survived (for generational GC): `0` is
    /// the young generation; promoted objects age up.
    age: u8,
    /// [`SLOT_OCCUPIED`] while `value` is `Some`, else `0` — what compiled
    /// code tests (it cannot decode the `Option`).
    occupied: u8,
    /// The slot's **remembered-set** state (see [`Heap::remembered`]):
    /// [`CLEAN`], [`DIRTY`] (an old object mutably borrowed since the last
    /// collection) or [`STICKY`] (an old object whose value hides interior
    /// mutability, rescanned by every minor collection). Non-zero exactly
    /// while the slot's index is in the list. A free slot is always `CLEAN`,
    /// so an allocation — compiled code's inline one included — never has to
    /// write it.
    dirty: u8,
}

/// [`Slot::dirty`]: not in the remembered set.
const CLEAN: u8 = 0;
/// [`Slot::dirty`]: in the remembered set until the next collection.
const DIRTY: u8 = 1;
/// [`Slot::dirty`]: in the remembered set for as long as it lives (its value's
/// [`Trace::interior_mutable`](crate::gc::Trace::interior_mutable) holds).
const STICKY: u8 = 2;

/// [`Slot::occupied`] of a live slot.
const SLOT_OCCUPIED: u8 = 1;

impl<T> Slot<T> {
    /// A live slot holding `value`.
    #[inline]
    const fn live(generation: u16, age: u8, value: T) -> Self {
        Self {
            value: Some(value),
            generation,
            age,
            occupied: SLOT_OCCUPIED,
            dirty: CLEAN,
        }
    }

    /// The value, if the slot is live in `generation`.
    #[inline]
    fn get(&self, generation: u16) -> Option<&T> {
        if self.generation == generation {
            self.value.as_ref()
        } else {
            None
        }
    }

    /// The value, mutably, if the slot is live in `generation`.
    #[inline]
    fn get_mut(&mut self, generation: u16) -> Option<&mut T> {
        if self.generation == generation {
            self.value.as_mut()
        } else {
            None
        }
    }

    /// Whether the slot holds a value.
    #[inline]
    const fn is_live(&self) -> bool {
        self.value.is_some()
    }

    /// Empties the slot, giving it `generation`, and returns the value it
    /// held (for the sweep to release).
    #[inline]
    fn vacate(&mut self, generation: u16) -> Option<T> {
        self.occupied = 0;
        self.dirty = CLEAN;
        self.generation = generation;
        self.value.take()
    }
}

/// Runtime-probed byte layout of `Slot<T>` for the generic-JIT inline
/// property-get fast path. Every field is derived by pointer arithmetic on a real
/// `Slot::Occupied` instance (never hand-baked), so a layout change is caught by
/// the JIT layout verification test rather than silently corrupting a raw read.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotLayout {
    /// `size_of::<Slot<T>>()` — the arena stride (`base + index * stride`).
    pub stride: usize,
    /// The occupancy byte's value in a live slot.
    pub occupied_disc: u8,
    /// Byte offset of the occupancy byte within the slot.
    pub occupied_off: usize,
    /// Byte offset of the `generation: u16` field within the slot.
    pub generation_off: usize,
    /// Byte offset of the payload value (`T`) within the slot.
    pub value_off: usize,
    /// Byte offset of the `age: u8` field (collections survived) within the slot.
    pub age_off: usize,
}

/// Derives [`SlotLayout`] for `Slot<T>` from a real live instance. `value`
/// supplies a `T` so the slot can be built; only its byte layout is inspected.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn jit_slot_layout<T>(value: T) -> SlotLayout {
    let slot: Slot<T> = Slot::live(0, 0, value);
    let base = core::ptr::addr_of!(slot) as usize;
    let value = slot.value.as_ref().expect("just built a live slot");
    SlotLayout {
        stride: core::mem::size_of::<Slot<T>>(),
        occupied_disc: SLOT_OCCUPIED,
        occupied_off: core::ptr::addr_of!(slot.occupied) as usize - base,
        generation_off: core::ptr::addr_of!(slot.generation) as usize - base,
        // The payload of `Some(value)` — where the `T` itself lies (`Option<T>`
        // keeps no separate tag for a niche-carrying `T`).
        value_off: core::ptr::addr_of!(*value) as usize - base,
        age_off: core::ptr::addr_of!(slot.age) as usize - base,
    }
}

/// The byte offsets, within the object heap (`Heap<Cell>`), of its slot
/// array's data-pointer and length words: what compiled code reads instead of
/// calling [`Heap::slots_raw`]. The field offset is `offset_of!`; the `Vec`'s
/// word order (unspecified) is probed on a live `Vec<Slot<Cell>>` whose
/// length, capacity and pointer all differ. `(ptr_off, len_off)`.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn jit_heap_slots_offsets() -> (usize, usize) {
    use crate::cell::Cell;
    let mut v: Vec<Slot<Cell>> = Vec::with_capacity(8);
    for _ in 0..3 {
        v.push(Slot::live(
            0,
            0,
            Cell::Object(crate::object::Object::new(crate::shape::Shape::root())),
        ));
    }
    let vec = core::ptr::addr_of!(v) as usize;
    let word = core::mem::size_of::<usize>();
    let (mut ptr_off, mut len_off) = (usize::MAX, usize::MAX);
    for i in 0..core::mem::size_of::<Vec<Slot<Cell>>>() / word {
        // SAFETY: the i-th word of the live `Vec`, in bounds.
        #[allow(unsafe_code)]
        let w = unsafe { *((vec + i * word) as *const usize) };
        if w == v.as_ptr() as usize {
            ptr_off = i * word;
        } else if w == v.len() {
            len_off = i * word;
        }
    }
    assert!(
        ptr_off != usize::MAX && len_off != usize::MAX,
        "could not derive the heap's slot Vec offsets"
    );
    let field = core::mem::offset_of!(Heap<Cell>, slots);
    (field + ptr_off, field + len_off)
}

/// Where compiled code finds what [`Heap::alloc`]'s free-list path touches,
/// as byte offsets within a `Heap<Cell>` (see [`jit_heap_alloc_offsets`]).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeapAllocOffsets {
    /// The free list's data pointer.
    pub free_ptr: usize,
    /// The free list's allocation cursor (`Heap::free_top`): the number of
    /// still-free entries.
    pub free_len: usize,
    /// `live`.
    pub live: usize,
    /// `allocs_since_gc`.
    pub allocs_since_gc: usize,
    /// The `extra`-part pool's (`Heap::extra_pool`) data pointer.
    pub pool_ptr: usize,
    /// The `extra`-part pool's length.
    pub pool_len: usize,
}

/// The [`HeapAllocOffsets`], probed (a `Vec`'s word order is unspecified).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[must_use]
pub(crate) fn jit_heap_alloc_offsets() -> HeapAllocOffsets {
    use crate::cell::Cell;
    // Length 1, capacity 16: the three words all differ.
    let mut v: Vec<u32> = Vec::with_capacity(16);
    v.push(1);
    let vec = core::ptr::addr_of!(v) as usize;
    let word = core::mem::size_of::<usize>();
    let mut ptr_off = usize::MAX;
    let mut len_off = usize::MAX;
    for i in 0..core::mem::size_of::<Vec<u32>>() / word {
        // SAFETY: the i-th word of the live `Vec`, in bounds.
        #[allow(unsafe_code)]
        let w = unsafe { *((vec + i * word) as *const usize) };
        if w == v.as_ptr() as usize {
            ptr_off = i * word;
        } else if w == 1 {
            len_off = i * word;
        }
    }
    assert!(
        ptr_off != usize::MAX && len_off != usize::MAX,
        "could not derive the heap's free-list Vec offsets"
    );
    let field = core::mem::offset_of!(Heap<Cell>, free);
    let pool = core::mem::offset_of!(Heap<Cell>, extra_pool);
    let (pool_ptr, pool_len) = crate::object::ExtraPool::jit_offsets(ptr_off, len_off);
    HeapAllocOffsets {
        free_ptr: field + ptr_off,
        // The allocation cursor, not the `Vec`'s length (see `Heap::free_top`).
        free_len: core::mem::offset_of!(Heap<Cell>, free_top),
        live: core::mem::offset_of!(Heap<Cell>, live),
        allocs_since_gc: core::mem::offset_of!(Heap<Cell>, allocs_since_gc),
        pool_ptr: pool + pool_ptr,
        pool_len: pool + pool_len,
    }
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
    /// Entries `free[..free_top]` are the slots still free; `free[free_top..]`
    /// are the free slots allocated since the last collection — popping only
    /// moves this cursor, so the collection that follows knows which old slot
    /// indices now hold young objects ([`young_floor`](Heap::young_floor)
    /// covers the slots pushed since). Compiled code's inline allocation
    /// decrements it (`jit_heap_alloc_offsets`).
    free_top: usize,
    /// `slots.len()` at the end of the last collection: every slot at or past
    /// it was pushed since, so holds a young object (or nothing).
    young_floor: usize,
    /// The **remembered set**: slot indices of old-generation objects a minor
    /// collection must scan as roots, because they may point at young ones.
    /// The write barrier is the mutable borrow itself: [`get_mut`](Heap::get_mut),
    /// [`peek_mut`](Heap::peek_mut) and [`swap`](Heap::swap) enter an old slot
    /// the first time it is borrowed after a collection — every store into a
    /// cell goes through one of them — and the collector itself adds old
    /// objects with interior mutability ([`STICKY`]). Compiled code, which
    /// stores into cells directly, takes the helper for an old container and a
    /// young target (`p_barrier_check`), and the helper's store borrows.
    remembered: Vec<u32>,
    /// Objects allocated since the last collection — the **allocation-pressure**
    /// signal an allocation-triggered collector polls at its safepoints (see
    /// `Realm::maybe_collect`). Reading it is a field load, so the check on the
    /// allocation path costs nothing when it does not fire.
    allocs_since_gc: usize,
    /// **GC-verification mode** (`KATAAN_GC_VERIFY=1`): instead of actually
    /// freeing a swept slot, record it here and keep the object alive. Any later
    /// [`get`](Heap::get)/[`get_mut`](Heap::get_mut) of a recorded handle means
    /// the collector reclaimed something the program still uses — i.e. the root
    /// set is incomplete — and is counted in `hits` (and, with
    /// `KATAAN_GC_VERIFY=panic`, aborts). `None` (the default) in every ordinary
    /// build/run, so the extra check is a single null test on a cold field.
    poison: Option<alloc::boxed::Box<PoisonState>>,
    /// Swept objects' reusable `extra` parts (see
    /// [`ExtraPool`](crate::object::ExtraPool)), refilled by the sweeps.
    extra_pool: crate::object::ExtraPool,
}

/// The bookkeeping behind [`Heap`]'s GC-verification mode (see `Heap::poison`).
struct PoisonState {
    /// Handles the collector wanted to free (but did not, so a use-after-free is
    /// observable as a *hit* rather than as corruption).
    freed: BTreeSet<Handle>,
    /// How many times a recorded handle was resolved after being "freed"
    /// (counted from the `&self` read path, hence the [`core::cell::Cell`]).
    hits: core::cell::Cell<usize>,
    /// Abort on the first hit rather than only counting it.
    panic_on_hit: bool,
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
            free_top: 0,
            young_floor: 0,
            remembered: Vec::new(),
            allocs_since_gc: 0,
            poison: None,
            extra_pool: crate::object::ExtraPool::default(),
        }
    }

    /// Objects allocated since the last [`reset_alloc_pressure`](Heap::reset_alloc_pressure).
    #[must_use]
    pub const fn allocs_since_gc(&self) -> usize {
        self.allocs_since_gc
    }

    /// The offset of [`allocs_since_gc`](Heap::allocs_since_gc)'s counter in
    /// the heap (compiled code reads it inline).
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) const fn allocs_since_gc_offset() -> usize {
        core::mem::offset_of!(Self, allocs_since_gc)
    }

    /// Clears the allocation-pressure counter (called when a collection runs).
    pub const fn reset_alloc_pressure(&mut self) {
        self.allocs_since_gc = 0;
    }

    /// Turns on **GC-verification mode**: swept slots are recorded but not freed,
    /// and resolving a recorded handle afterwards is counted (see `Heap::poison`).
    /// `panic_on_hit` aborts on the first such use instead.
    pub fn enable_gc_verify(&mut self, panic_on_hit: bool) {
        self.poison = Some(alloc::boxed::Box::new(PoisonState {
            freed: BTreeSet::new(),
            hits: core::cell::Cell::new(0),
            panic_on_hit,
        }));
    }

    /// How many times a swept-but-retained handle has been resolved in
    /// GC-verification mode (`0` when the mode is off). Non-zero means the
    /// collector's root set is incomplete.
    #[must_use]
    pub fn gc_verify_hits(&self) -> usize {
        self.poison.as_ref().map_or(0, |p| p.hits.get())
    }

    /// Records (in verification mode) a broken collector invariant found by
    /// a check such as [`remembered_violation`](Heap::remembered_violation):
    /// counted like a use of a collected handle, and fatal with
    /// `KATAAN_GC_VERIFY=panic`.
    #[cold]
    pub fn report_gc_violation(&self, what: core::fmt::Arguments<'_>) {
        if let Some(p) = &self.poison {
            p.hits.set(p.hits.get() + 1);
            assert!(!p.panic_on_hit, "kataan GC verify: {what}");
        }
    }

    /// Whether GC-verification mode is on.
    #[must_use]
    pub const fn gc_verify(&self) -> bool {
        self.poison.is_some()
    }

    /// Records (in verification mode) that `handle` was resolved after the
    /// collector wanted it freed — an incomplete root set. Cold: only reached
    /// when the mode is on.
    #[cold]
    fn note_poison_use(&self, handle: Handle) {
        if let Some(p) = &self.poison
            && p.freed.contains(&handle)
        {
            p.hits.set(p.hits.get() + 1);
            assert!(
                !p.panic_on_hit,
                "kataan GC verify: use of collected handle {handle:?}"
            );
        }
    }

    /// The number of live objects.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live
    }

    /// The raw base pointer and length of the backing slot array, for the
    /// generic-JIT inline property-get fast path. The pointer is `slots.as_ptr()`
    /// (arena base) and the length is `slots.len()` (the index bound, **not**
    /// [`len`](Heap::len) which counts only live slots). Both are re-read on every
    /// fast-path entry because [`alloc`](Heap::alloc)'s `push` may reallocate the
    /// `Vec`; the JIT never caches this across a call.
    /// The `extra`-part pool's data pointer and length, for the layout test.
    #[cfg(all(test, feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    pub(crate) fn extra_pool_raw(&self) -> (usize, usize) {
        self.extra_pool.raw_parts()
    }

    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[must_use]
    pub(crate) fn slots_raw(&self) -> (*const u8, usize) {
        (self.slots.as_ptr().cast::<u8>(), self.slots.len())
    }

    /// Whether the heap holds no live objects.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Allocates `value`, reusing a freed slot when one is available, and
    /// returns a handle to it.
    /// Makes room for at least `additional` more freshly allocated cells
    /// without the slot array growing (and moving every cell) on the way.
    pub fn reserve(&mut self, additional: usize) {
        if self.slots.capacity() - self.slots.len() < additional {
            // The reserve moves every cell, as a growing push would.
            crate::object::bump_proto_epoch();
            self.slots.reserve(additional);
        }
    }

    pub fn alloc(&mut self, value: T) -> Handle {
        self.live += 1;
        self.allocs_since_gc += 1;
        if self.free_top > 0 {
            self.free_top -= 1;
            let index = self.free[self.free_top];
            let slot = &mut self.slots[index as usize];
            // A freed slot already had its generation bumped; reuse it as-is.
            debug_assert!(!slot.is_live(), "free list pointed at a live slot");
            let generation = slot.generation;
            slot.age = 0;
            slot.occupied = SLOT_OCCUPIED;
            // The slot is empty, so there is nothing to drop: `forget` the
            // `None` rather than run the drop glue over it.
            core::mem::forget(slot.value.replace(value));
            Handle { index, generation }
        } else {
            // Slot index `u32::MAX` is never handed out: its handle bits are
            // the JIT's empty receiver-cache key (`jit::RECV_CACHE_EMPTY`).
            let index = u32::try_from(self.slots.len())
                .ok()
                .filter(|&i| i != u32::MAX)
                .expect("heap slot space exhausted");
            if self.slots.len() == self.slots.capacity() {
                // The push moves every cell: compiled code caching a cell's
                // address under the prototype epoch must not use it any more
                // (see `jit::props::MethodEntry::value`).
                crate::object::bump_proto_epoch();
            }
            self.slots.push(Slot::live(0, 0, value));
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
        if self.poison.is_some() {
            self.note_poison_use(handle);
        }
        self.slots
            .get(handle.index as usize)?
            .get(handle.generation)
    }

    /// Like [`get`](Heap::get), but exempt from the GC-verification poison check.
    ///
    /// For code that *enumerates* the heap — [`live_handles`](Heap::live_handles)
    /// and friends — rather than following a handle the program handed it. Such a
    /// scan naturally visits objects the collector wanted to free (verification
    /// mode keeps them), which is not a use-after-free and must not be reported as
    /// one.
    #[must_use]
    pub fn peek(&self, handle: Handle) -> Option<&T> {
        self.slots
            .get(handle.index as usize)?
            .get(handle.generation)
    }

    /// Mutably borrows the value behind `handle`, with the same staleness check
    /// as [`get`](Heap::get).
    ///
    /// This is the generational **write barrier**: an old object borrowed
    /// mutably enters the remembered set (once per collection cycle), since
    /// the borrow may store a young reference into it.
    #[inline]
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        if self.poison.is_some() {
            self.note_poison_use(handle);
        }
        self.peek_mut(handle)
    }

    /// The mutable counterpart of [`peek`](Heap::peek): exempt from the
    /// GC-verification poison check, for code driven by a heap enumeration.
    /// Applies the write barrier as [`get_mut`](Heap::get_mut) does.
    #[inline]
    pub fn peek_mut(&mut self, handle: Handle) -> Option<&mut T> {
        let Self {
            slots, remembered, ..
        } = self;
        let slot = slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.as_mut()?;
        if slot.dirty == CLEAN && slot.age >= crate::gc::OLD_AGE {
            slot.dirty = DIRTY;
            remembered.push(handle.index);
        }
        Some(value)
    }

    /// Enters slot `index` into the remembered set if it holds an old object
    /// that is not there yet (the barrier of [`peek_mut`](Heap::peek_mut)).
    fn remember(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index)
            && slot.is_live()
            && slot.dirty == CLEAN
            && slot.age >= crate::gc::OLD_AGE
        {
            slot.dirty = DIRTY;
            self.remembered.push(index as u32);
        }
    }

    /// Swaps the *values* held by two live handles, leaving each slot's identity
    /// (generation/age) — and therefore each handle's validity — intact. After
    /// the swap `a` refers to what `b` held and vice-versa. No-op if either
    /// handle is stale or they alias the same slot. Used to transplant a
    /// freshly-built native cell into a placeholder instance handle (a class that
    /// `extends` a native builds the real cell from the `super(...)` arguments,
    /// then moves it into the already-threaded `this` handle).
    pub fn swap(&mut self, a: Handle, b: Handle) {
        if a.index == b.index {
            return;
        }
        let (i, j) = (a.index as usize, b.index as usize);
        if i >= self.slots.len() || j >= self.slots.len() {
            return;
        }
        // `split_at_mut` forces a low/high split, so re-associate the two borrows
        // with `a` and `b` before comparing generations. Getting this backwards
        // whenever `a.index > b.index` checked each handle's generation against the
        // *other* slot — harmless for as long as every slot sat at generation 0
        // (nothing was ever freed), but once the collector recycles slots the
        // generations diverge, the check fails, and the swap silently no-ops:
        // a `class X extends Map/TypedArray/...` instance would stay the bare
        // placeholder object it was allocated as.
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let (left, right) = self.slots.split_at_mut(hi);
        let (slot_lo, slot_hi) = (&mut left[lo], &mut right[0]);
        let (sa, sb) = if i < j {
            (slot_lo, slot_hi)
        } else {
            (slot_hi, slot_lo)
        };
        if let (Some(va), Some(vb)) = (sa.get_mut(a.generation), sb.get_mut(b.generation)) {
            core::mem::swap(va, vb);
            // Each slot now holds what the other did: both may point at young
            // objects (or be interior-mutable) — the write barrier.
            self.remember(i);
            self.remember(j);
        }
    }

    /// Whether `handle` still refers to a live object.
    ///
    /// Deliberately bypasses the GC-verification poison check: the collector asks
    /// this about handles it finds in weak side tables, and a *liveness query* on
    /// a collected handle is correct behaviour, not a use-after-free. Only an
    /// actual [`get`](Heap::get)/[`get_mut`](Heap::get_mut) counts as a use.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.slots
            .get(handle.index as usize)
            .is_some_and(|s| s.is_live() && s.generation == handle.generation)
    }

    /// A handle to every currently-live slot (used by the collector to find the
    /// sweep set). Order is by slot index.
    #[must_use]
    pub fn live_handles(&self) -> Vec<Handle> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_live())
            .map(|(i, slot)| Handle {
                index: i as u32,
                generation: slot.generation,
            })
            .collect()
    }

    /// The number of slots (live or free) — the bound on any handle's index.
    #[must_use]
    pub fn slot_capacity(&self) -> usize {
        self.slots.len()
    }

    /// The sweep of a full collection, in one pass over the slot table: every
    /// live object for which `keep(handle, value)` holds is promoted one
    /// generation; every other one is freed (dropped). Returns the number
    /// freed. Freeing follows [`free`](Heap::free) exactly — the same
    /// generation bump, retirement and free-list order, and GC-verification
    /// mode still only records the handle — but drops each doomed value where
    /// it lies instead of moving it out of the slot first.
    ///
    /// The remembered set is rebuilt from scratch: a survivor for which
    /// `sticky(value)` holds (interior mutability the write barrier cannot
    /// see) enters it for good, every other one leaves it.
    pub fn sweep(
        &mut self,
        mut keep: impl FnMut(Handle, &T) -> bool,
        sticky: impl Fn(&T) -> bool,
    ) -> usize
    where
        T: crate::gc::Trace,
    {
        let Self {
            slots,
            free,
            free_top,
            young_floor,
            remembered,
            live,
            poison,
            extra_pool,
            ..
        } = self;
        // Every slot is visited below, so the record of which ones were
        // allocated since the last collection is not needed.
        free.truncate(*free_top);
        remembered.clear();
        let mut swept = 0;
        for (i, slot) in slots.iter_mut().enumerate() {
            let Some(value) = &slot.value else {
                continue;
            };
            let h = Handle {
                index: i as u32,
                generation: slot.generation,
            };
            if keep(h, value) {
                slot.age = slot.age.saturating_add(1);
                if sticky(value) {
                    slot.dirty = STICKY;
                    remembered.push(i as u32);
                } else {
                    slot.dirty = CLEAN;
                }
                continue;
            }
            swept += 1;
            if let Some(p) = poison {
                p.freed.insert(h);
                // Kept (unreachable) for the verification, as an old object:
                // no later minor collection looks for it among the young.
                slot.age = slot.age.max(crate::gc::OLD_AGE);
                slot.dirty = CLEAN;
                continue;
            }
            // See `free` for why a slot leaving the low generation range is
            // retired rather than reused.
            let next_gen = slot.generation.wrapping_add(1);
            if let Some(v) = slot.vacate(next_gen) {
                v.release(extra_pool);
            }
            if next_gen < COMPACT_GEN_BASE {
                free.push(i as u32);
            }
            *live -= 1;
        }
        *free_top = free.len();
        *young_floor = slots.len();
        swept
    }

    /// Whether `handle` is a live object of the **young** generation
    /// (allocated since the last collection).
    #[must_use]
    #[inline]
    pub fn is_young(&self, handle: Handle) -> bool {
        self.slots.get(handle.index as usize).is_some_and(|s| {
            s.is_live() && s.generation == handle.generation && s.age < crate::gc::OLD_AGE
        })
    }

    /// A minor collection's mark: promotes `handle`'s object to the old
    /// generation if it is a live young one, returning whether it was (so
    /// whether it still has to be traced). The age *is* the mark bit — every
    /// young object the marking does not reach stays at age `0` and is freed
    /// by [`sweep_young`](Heap::sweep_young).
    #[inline]
    pub fn promote_young(&mut self, handle: Handle) -> bool {
        match self.slots.get_mut(handle.index as usize) {
            Some(s)
                if s.is_live()
                    && s.generation == handle.generation
                    && s.age < crate::gc::OLD_AGE =>
            {
                s.age = crate::gc::OLD_AGE;
                true
            }
            _ => false,
        }
    }

    /// Takes the remembered set's slot indices (for a minor collection to
    /// scan); [`sweep_young`](Heap::sweep_young) settles each one's state.
    pub fn take_remembered(&mut self) -> Vec<u32> {
        core::mem::take(&mut self.remembered)
    }

    /// The handle of the live object in slot `index`, if any.
    #[must_use]
    pub fn live_handle_at(&self, index: u32) -> Option<Handle> {
        let slot = self.slots.get(index as usize)?;
        slot.is_live().then_some(Handle {
            index,
            generation: slot.generation,
        })
    }

    /// The sweep of a **minor** collection: frees every young object the
    /// marking left at age `0`, visiting only the slots allocated since the
    /// last collection (the free-list entries consumed since, and the slots
    /// pushed since), so its cost is the nursery's, not the heap's. Returns
    /// the number freed.
    ///
    /// `rescanned` is what [`take_remembered`](Heap::take_remembered) handed
    /// out: those slots leave the remembered set unless `sticky(value)` holds,
    /// as do the surviving young objects for which it holds. Afterwards no
    /// object is young, so no old-to-young edge exists.
    pub fn sweep_young(&mut self, rescanned: Vec<u32>, sticky: impl Fn(&T) -> bool) -> usize
    where
        T: crate::gc::Trace,
    {
        let Self {
            slots,
            free,
            free_top,
            young_floor,
            remembered,
            live,
            poison,
            extra_pool,
            ..
        } = self;
        for index in rescanned {
            let slot = &mut slots[index as usize];
            match &slot.value {
                Some(value) if sticky(value) => {
                    slot.dirty = STICKY;
                    remembered.push(index);
                }
                _ => slot.dirty = CLEAN,
            }
        }
        let floor = *young_floor;
        let mut swept = 0;
        // Counted locally (and the tables walked as plain slices), so the
        // loops below keep their state in registers.
        let mut live_n = *live;
        let mut poison = poison.as_deref_mut();
        // The free-list entries consumed since the last collection: each one
        // freed again is written back over the consumed part, which never
        // runs ahead of the reading.
        let mut kept = *free_top;
        {
            let slots = slots.as_mut_slice();
            let free = free.as_mut_slice();
            let start = kept;
            for r in start..free.len() {
                let i = free[r];
                let i_us = i as usize;
                let slot = &mut slots[i_us];
                let p = poison.as_deref_mut();
                if settle_young(
                    slot,
                    i_us,
                    &sticky,
                    remembered,
                    &mut swept,
                    &mut live_n,
                    p,
                    extra_pool,
                ) {
                    free[kept] = i;
                    kept += 1;
                }
            }
        }
        free.truncate(kept);
        // The slots pushed since.
        for (i, slot) in slots.iter_mut().enumerate().skip(floor) {
            let p = poison.as_deref_mut();
            if settle_young(
                slot,
                i,
                &sticky,
                remembered,
                &mut swept,
                &mut live_n,
                p,
                extra_pool,
            ) {
                free.push(i as u32);
            }
        }
        *live = live_n;
        *free_top = free.len();
        *young_floor = slots.len();
        swept
    }

    /// **GC verification** of the generational invariants, for a minor
    /// collection about to run: every old object outside the remembered set
    /// points at no young object (`edges(value, out)` pushes a value's
    /// outgoing handles) nor is `sticky`, and every young object is one
    /// [`sweep_young`](Heap::sweep_young) will visit. Returns the first
    /// offending `(container, young target)` — the target equal to the
    /// container for an untracked young object. Walks the whole heap.
    pub fn remembered_violation(
        &self,
        mut edges: impl FnMut(&T, &mut Vec<Handle>),
        sticky: impl Fn(&T) -> bool,
    ) -> Option<(Handle, Handle)> {
        let mut tracked = alloc::vec![false; self.slots.len()];
        for &i in &self.free[self.free_top..] {
            tracked[i as usize] = true;
        }
        for t in tracked.iter_mut().skip(self.young_floor) {
            *t = true;
        }
        let mut out = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(value) = &slot.value else {
                continue;
            };
            let h = Handle {
                index: i as u32,
                generation: slot.generation,
            };
            if slot.age < crate::gc::OLD_AGE {
                if !tracked[i] {
                    return Some((h, h));
                }
                continue;
            }
            if slot.dirty != CLEAN || self.poison.as_ref().is_some_and(|p| p.freed.contains(&h)) {
                // Remembered, or collected (only kept for the verification).
                continue;
            }
            if sticky(value) {
                return Some((h, h));
            }
            out.clear();
            edges(value, &mut out);
            if let Some(&t) = out.iter().find(|&&t| self.is_young(t)) {
                return Some((h, t));
            }
        }
        None
    }

    /// The generational age of `handle`'s object (collections survived), or
    /// `None` if stale. `0` is the young generation.
    #[must_use]
    pub fn age(&self, handle: Handle) -> Option<u8> {
        let slot = self.slots.get(handle.index as usize)?;
        (slot.is_live() && slot.generation == handle.generation).then_some(slot.age)
    }

    /// Handles to live objects whose age satisfies `keep` — e.g. the young
    /// generation (`|a| a == 0`).
    #[must_use]
    pub fn handles_where(&self, keep: impl Fn(u8) -> bool) -> Vec<Handle> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_live() && keep(slot.age))
            .map(|(i, slot)| Handle {
                index: i as u32,
                generation: slot.generation,
            })
            .collect()
    }

    /// The explicit **write barrier**: enters `container` into the remembered
    /// set when an old-generation `container` is made to point at a
    /// young-generation `target` (an old→young edge a minor collection must
    /// treat as a root). A no-op for young containers or old→old edges. (A
    /// store through [`get_mut`](Heap::get_mut) needs no call: the borrow is
    /// the barrier.)
    pub fn record_edge(&mut self, container: Handle, target: Handle, old_age: u8) {
        if self.age(container).is_some_and(|a| a >= old_age)
            && self.age(target).is_some_and(|a| a < old_age)
        {
            self.remember(container.index as usize);
        }
    }

    /// Handles to the remembered-set objects that are still live.
    #[must_use]
    pub fn remembered_roots(&self) -> Vec<Handle> {
        self.remembered
            .iter()
            .filter_map(|&index| self.live_handle_at(index))
            .collect()
    }

    /// Compacts the `keep` (live) objects to the front of the slot table,
    /// discarding everything else, and returns the forwarding map
    /// `(old_handle → new_handle)`. Slot generations reset to `0`, so every
    /// pre-compaction handle is invalid *except* via the returned map — the
    /// moving collector uses it to fix up references. Ages (generations
    /// survived) are preserved; the free list and remembered set are cleared.
    pub fn compact_to(&mut self, keep: &crate::gc::MarkSet) -> Vec<(Handle, Handle)> {
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
            let Slot {
                value: Some(value),
                generation,
                age,
                ..
            } = slot
            else {
                continue;
            };
            let old = Handle {
                index: i as u32,
                generation,
            };
            if keep.contains(&old) {
                let new = Handle {
                    index: new_slots.len() as u32,
                    generation: new_gen,
                };
                new_slots.push(Slot::live(new_gen, age, value));
                map.push((old, new));
            }
        }
        self.slots = new_slots;
        self.live = self.slots.len();
        // Ages are kept, so a young survivor stays young: every slot is a
        // candidate for the next minor collection's sweep (which skips the old
        // ones), and every old one is remembered, which also settles the
        // sticky ones there.
        self.free_top = 0;
        self.young_floor = 0;
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.age >= crate::gc::OLD_AGE {
                slot.dirty = DIRTY;
                self.remembered.push(i as u32);
            }
        }
        map
    }

    /// Frees the slot behind `handle`, returning its value. The slot's
    /// generation is bumped so existing handles to it become stale. A stale or
    /// out-of-range handle frees nothing and returns `None`.
    pub fn free(&mut self, handle: Handle) -> Option<T> {
        // GC-verification mode keeps the object alive and only records the
        // intent, so a root-set gap shows up as a counted *use* of a collected
        // handle instead of as silent corruption.
        if let Some(p) = &mut self.poison {
            p.freed.insert(handle);
            return None;
        }
        let slot = self.slots.get_mut(handle.index as usize)?;
        match slot.generation {
            generation if slot.is_live() && generation == handle.generation => {
                // Bump the generation and take the value out. The bump must stay
                // strictly below the compaction-reserved range, so free/reuse
                // generations remain disjoint from those `compact_to` hands out
                // (>= COMPACT_GEN_BASE).
                //
                // A slot whose generation would leave that range is **retired**
                // instead of wrapping back to 0: it is dropped from the free list
                // and never reallocated. Wrapping was sound while the slot's own
                // handles were the only consumers, but the realm keeps several
                // handle-*keyed* side tables (`aux_props`, `fn_source`, the array
                // integrity flag sets, …) whose entries are only pruned
                // opportunistically; a wrapped generation could make a brand-new
                // object collide with a stale entry (inheriting, say, another
                // array's "frozen" flag). Retiring costs one dead slot per 32767
                // reuses of that slot and removes the hazard outright.
                let next_gen = generation.wrapping_add(1);
                let retire = next_gen >= COMPACT_GEN_BASE;
                let value = slot.value.take().expect("a live slot holds a value");
                slot.occupied = 0;
                slot.generation = next_gen;
                if slot.dirty != CLEAN {
                    // Each remembered slot is listed exactly once, and only
                    // while it is non-`CLEAN`.
                    slot.dirty = CLEAN;
                    self.remembered.retain(|&i| i != handle.index);
                }
                if !retire {
                    // Among the still-free entries: those past `free_top` are
                    // the record of what was allocated since the collection.
                    self.free.insert(self.free_top, handle.index);
                    self.free_top += 1;
                }
                self.live -= 1;
                Some(value)
            }
            _ => None,
        }
    }
}

/// One young slot's fate in [`Heap::sweep_young`]: a promoted (marked)
/// object stays — entering the remembered set if `sticky` — and an unmarked
/// one is freed (or, in GC-verification mode, recorded and kept). Returns
/// whether slot `i` goes back on the free list.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // the sweep's split borrows
fn settle_young<T: crate::gc::Trace>(
    slot: &mut Slot<T>,
    i: usize,
    sticky: &impl Fn(&T) -> bool,
    remembered: &mut Vec<u32>,
    swept: &mut usize,
    live: &mut usize,
    poison: Option<&mut PoisonState>,
    pool: &mut crate::object::ExtraPool,
) -> bool {
    let Some(value) = &slot.value else {
        return false;
    };
    if slot.age >= crate::gc::OLD_AGE {
        // Marked (promoted), or a slot listed twice.
        if slot.dirty == CLEAN && sticky(value) {
            slot.dirty = STICKY;
            remembered.push(i as u32);
        }
        return false;
    }
    *swept += 1;
    if let Some(p) = poison {
        p.freed.insert(Handle {
            index: i as u32,
            generation: slot.generation,
        });
        slot.age = crate::gc::OLD_AGE;
        return false;
    }
    *live -= 1;
    let next_gen = slot.generation.wrapping_add(1);
    if let Some(v) = slot.vacate(next_gen) {
        v.release(pool);
    }
    next_gen < COMPACT_GEN_BASE
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
    fn swap_works_in_both_index_orders_with_differing_generations() {
        // Regression: `swap` compared each handle's generation against the *other*
        // slot whenever `a.index > b.index`. That was invisible while nothing was
        // ever freed (every slot sat at generation 0) but silently no-ops once the
        // collector recycles slots and generations diverge.
        for reversed in [false, true] {
            let mut h: Heap<i32> = Heap::new();
            // Give slot 0 a generation of 2 and slot 1 a generation of 1, so the
            // crossed comparison cannot accidentally succeed.
            let s0 = h.alloc(0);
            h.free(s0);
            let s0 = h.alloc(0);
            h.free(s0);
            let low = h.alloc(10);
            let scratch = h.alloc(0);
            h.free(scratch);
            let high = h.alloc(20);
            assert_ne!(low.generation, high.generation, "generations must differ");
            assert!(low.index < high.index || high.index < low.index);

            let (a, b) = if reversed { (high, low) } else { (low, high) };
            h.swap(a, b);
            assert_eq!(h.get(a), Some(&if reversed { 10 } else { 20 }));
            assert_eq!(h.get(b), Some(&if reversed { 20 } else { 10 }));
        }
    }

    #[test]
    fn a_slot_is_retired_rather_than_wrapping_its_generation() {
        // Handle identity must be globally unique for the lifetime of the heap:
        // the realm keeps handle-keyed side tables that are only pruned
        // opportunistically, so a wrapped generation could let a brand-new object
        // inherit a dead one's entry. The slot is dropped instead.
        let mut h: Heap<i32> = Heap::new();
        let mut last = h.alloc(1);
        // Cycle one slot until it retires; every handle handed out must be distinct.
        let mut seen = BTreeSet::new();
        for _ in 0..(COMPACT_GEN_BASE as usize + 4) {
            assert!(seen.insert(last), "a handle was handed out twice: {last:?}");
            h.free(last);
            last = h.alloc(1);
        }
        // Retirement means the table grew past the single recycled slot.
        assert!(h.slots.len() > 1, "the exhausted slot should have retired");
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

    #[test]
    fn a_slot_adds_one_word_to_its_cell() {
        // `Option<Cell>` takes the cell's niche, and the generation, age and
        // occupancy byte share the one word the slot adds.
        use crate::cell::Cell;
        use core::mem::size_of;
        assert_eq!(size_of::<Option<Cell>>(), size_of::<Cell>());
        assert_eq!(size_of::<Slot<Cell>>(), size_of::<Cell>() + 8);
    }
}
