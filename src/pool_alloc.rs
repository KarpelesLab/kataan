//! A size-class pooling allocator for the `std` builds (the `pool-alloc`
//! feature): [`PoolAlloc`](crate::pool_alloc::PoolAlloc), meant to be installed as the process
//! `#[global_allocator]` by an executable (the `kataan` CLI and the Test262
//! runner do; the library itself never installs one, so an embedder keeps
//! whatever allocator it chose).
//!
//! # Why
//!
//! An engine allocates and frees a great many small, short-lived blocks: string
//! leaves, `Rc` nodes, argument vectors, property tables. The garbage collector
//! also frees them in large batches, which overflows the system allocator's
//! small per-thread caches and sends it down its slower general paths. A pool
//! of fixed-size blocks per size class, cached per thread, makes the common
//! allocation a pointer pop and the common free a pointer push.
//!
//! # Design
//!
//! * **Size classes.** Requests of at most [`MAX_SMALL`](crate::pool_alloc::MAX_SMALL) bytes with an
//!   alignment of at most 16 are rounded up to a multiple of 16 bytes, giving
//!   [`NCLASS`](crate::pool_alloc::NCLASS) classes. Everything else goes straight to the system allocator.
//!   The class of a block is recomputed from the `Layout` that `dealloc`
//!   receives (the `GlobalAlloc` contract guarantees it is the allocation's
//!   layout), so blocks carry no header.
//! * **Thread cache.** Each thread owns, per class, an intrusive singly-linked
//!   free list (the first word of a free block is the next pointer), plus one
//!   bump arena shared by all classes, refilled in `ARENA`-byte chunks from
//!   the system allocator. An allocation pops the class's free list, else bumps
//!   the arena; a free pushes the block on the *freeing* thread's list.
//! * **Global exchange.** A spin-locked stash holds per-class lists that
//!   threads hand back: a thread whose list for a class grows past
//!   `SPILL_BYTES` moves it to the stash, a thread whose list is empty takes
//!   the stash's list before carving new memory, and a thread that exits
//!   returns all its lists (and the unused tail of its arena). So memory freed
//!   on one thread is reusable by another, and a worker thread that exits does
//!   not strand its blocks.
//! * **Memory is never returned to the system.** Pool chunks live for the rest
//!   of the process, like most of what a `malloc` keeps in its free lists. A
//!   block from the system allocator (the bootstrap fallback below) may also
//!   end up on a free list; that is harmless, since small blocks are never
//!   handed back to the system allocator, whatever their origin.
//! * **Bootstrap / teardown.** While a thread's cache is being set up, or after
//!   it has been torn down (other thread-local destructors may still free
//!   memory), the thread works directly on the stash under its lock, and
//!   falls back to the system allocator for a fresh block.
//!
//! All `unsafe` in this file is confined to raw-pointer free-list manipulation
//! and calls into [`std::alloc::System`]; each block states its invariant.

#![allow(unsafe_code)]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};
use std::alloc::System;

/// Size-class granularity in bytes (and the alignment every pool block has).
const GRAN: usize = 16;
/// The largest request served by the pool; larger ones go to the system.
pub const MAX_SMALL: usize = 1024;
/// Number of size classes: class `c` holds blocks of `(c + 1) * GRAN` bytes.
pub const NCLASS: usize = MAX_SMALL / GRAN;
/// Bytes requested from the system per arena refill.
const ARENA: usize = 256 * 1024;
/// A thread's free list for one class is moved to the stash once it holds
/// more than this many bytes, bounding how much one thread can hoard.
const SPILL_BYTES: usize = 4 * 1024 * 1024;

/// A free block's link word.
struct Node {
    next: *mut Node,
}

/// One class's free list: head, tail (valid while `head` is non-null, so a
/// whole list splices in O(1)) and length.
#[derive(Clone, Copy)]
struct List {
    head: *mut Node,
    tail: *mut Node,
    len: usize,
}

impl List {
    const EMPTY: List = List {
        head: ptr::null_mut(),
        tail: ptr::null_mut(),
        len: 0,
    };

    /// Pushes block `p` (at least `GRAN` bytes, 16-aligned, not in any list).
    #[inline]
    fn push(&mut self, p: *mut u8) {
        let n = p.cast::<Node>();
        // SAFETY: `p` is a free pool block of at least `GRAN` (>= 8) bytes,
        // 16-byte aligned, exclusively owned by the caller; its first word is
        // ours to use as the link.
        unsafe { n.write(Node { next: self.head }) };
        if self.head.is_null() {
            self.tail = n;
        }
        self.head = n;
        self.len += 1;
    }

    /// Pops a block, or null when empty.
    #[inline]
    fn pop(&mut self) -> *mut u8 {
        let n = self.head;
        if n.is_null() {
            return ptr::null_mut();
        }
        // SAFETY: `n` is the head of this list, so it is a free block whose
        // first word was written by `push` (or `append`) as its link.
        self.head = unsafe { (*n).next };
        self.len -= 1;
        n.cast()
    }

    /// Moves every block of `other` onto the front of `self`.
    #[inline]
    fn append(&mut self, other: List) {
        if other.head.is_null() {
            return;
        }
        // SAFETY: `other` is non-empty, so `other.tail` is its last block and
        // a valid link word we own.
        unsafe { (*other.tail).next = self.head };
        if self.head.is_null() {
            self.tail = other.tail;
        }
        self.head = other.head;
        self.len += other.len;
    }
}

/// The process-wide exchange of free blocks, guarded by a spin lock (a lock
/// that never allocates, so it is usable from inside the allocator).
struct Stash {
    lock: AtomicBool,
    lists: UnsafeCell<[List; NCLASS]>,
    /// Unused arena tails of exited threads: a list of regions, each starting
    /// with a `Spare` header.
    spare: UnsafeCell<*mut Spare>,
}

/// The smallest unused arena tail worth handing back on thread exit (any
/// shorter one is abandoned). At least one largest block, so a spare region
/// usually serves whatever class takes it; `refill` re-checks regardless.
const SPARE_MIN: usize = MAX_SMALL;

/// Header written at the start of an unused arena tail.
struct Spare {
    next: *mut Spare,
    end: *mut u8,
}

// SAFETY: every access to the `UnsafeCell` contents goes through
// `Stash::with`, which holds the spin lock for its duration.
unsafe impl Sync for Stash {}

/// Per class, whether the stash may hold blocks: set (under the lock)
/// whenever blocks are added, cleared when a thread takes the list. Read
/// without the lock, so a thread whose list runs dry only takes the lock when
/// there is likely something to take, rather than on every arena bump; a
/// stale value only costs one wasted lock or one bump instead of a reuse.
static STASH_HAS: [AtomicBool; NCLASS] = [const { AtomicBool::new(false) }; NCLASS];

static STASH: Stash = Stash {
    lock: AtomicBool::new(false),
    lists: UnsafeCell::new([List::EMPTY; NCLASS]),
    spare: UnsafeCell::new(ptr::null_mut()),
};

impl Stash {
    /// Runs `f` with exclusive access to the stash.
    #[inline]
    fn with<R>(&self, f: impl FnOnce(&mut [List; NCLASS], &mut *mut Spare) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.lock.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
        // SAFETY: the lock is held, so no other thread touches the cells
        // until the release store below; `f` cannot re-enter `with` because
        // nothing it is given can reach the stash.
        let r = unsafe { f(&mut *self.lists.get(), &mut *self.spare.get()) };
        self.lock.store(false, Ordering::Release);
        r
    }
}

/// A thread's cache: per-class free lists plus a bump arena.
///
/// Its lifecycle is folded into data the fast paths read anyway, so neither
/// tests it: until the cache is set up (and after it is torn down) every free
/// list is empty, so an allocation finds nothing to pop, and every `limit` is
/// zero, so a free finds its list "full". Both then take the cold path, which
/// is where [`state`](Cache::state) is looked at.
struct Cache {
    lists: [List; NCLASS],
    /// Per class, the list length at which a free leaves the fast path:
    /// `SPILL_BYTES / class size` while [`LIVE`], `0` otherwise.
    limit: [u32; NCLASS],
    bump: *mut u8,
    end: *mut u8,
    /// [`UNINIT`], [`INIT`] (the thread-exit hook is being registered),
    /// [`LIVE`] or [`DEAD`] (torn down).
    state: u8,
}

impl Cache {
    const fn new() -> Self {
        Cache {
            lists: [List::EMPTY; NCLASS],
            limit: [0; NCLASS],
            bump: ptr::null_mut(),
            end: ptr::null_mut(),
            state: UNINIT,
        }
    }

    /// The slow path of an allocation of class `c` whose free list is empty.
    #[cold]
    fn refill(&mut self, c: usize) -> *mut u8 {
        let size = class_size(c);
        let stolen = if STASH_HAS[c].load(Ordering::Relaxed) {
            STASH.with(|lists, _| {
                STASH_HAS[c].store(false, Ordering::Relaxed);
                core::mem::replace(&mut lists[c], List::EMPTY)
            })
        } else {
            List::EMPTY
        };
        if !stolen.head.is_null() {
            self.lists[c] = stolen;
            return self.lists[c].pop();
        }
        // A spare region handed back by an exited thread can be shorter than
        // one block of this class, so re-check after every replacement.
        while (self.end as usize) - (self.bump as usize) < size {
            self.new_arena();
            if self.bump.is_null() {
                return ptr::null_mut();
            }
        }
        let p = self.bump;
        // SAFETY: `p + size <= end`, both inside the current arena.
        self.bump = unsafe { p.add(size) };
        p
    }

    /// Replaces the bump arena with a spare region left by an exited thread,
    /// or a fresh chunk from the system. The old arena's tail (less than one
    /// block of the class that needed it) is abandoned.
    fn new_arena(&mut self) {
        let spare = STASH.with(|_, spare| {
            let s = *spare;
            if !s.is_null() {
                // SAFETY: `s` heads the spare list (under the lock); its
                // header was written by `Cache::tear_down`.
                *spare = unsafe { (*s).next };
            }
            s
        });
        if !spare.is_null() {
            self.bump = spare.cast();
            // SAFETY: as above; the header is read before the region is reused.
            self.end = unsafe { (*spare).end };
            return;
        }
        // SAFETY: `ARENA` is non-zero and `GRAN` a power of two.
        let p = unsafe { System.alloc(Layout::from_size_align_unchecked(ARENA, GRAN)) };
        if p.is_null() {
            self.bump = ptr::null_mut();
            self.end = ptr::null_mut();
            return;
        }
        self.bump = p;
        // SAFETY: `p` heads an `ARENA`-byte allocation.
        self.end = unsafe { p.add(ARENA) };
    }

    /// A free whose list is at its limit, or into a cache that is not live:
    /// moves the full list to the stash, or routes the block around the cache.
    #[cold]
    fn dealloc_slow(&mut self, p: *mut u8, c: usize) {
        if self.state != LIVE {
            return dealloc_uncached(p, c);
        }
        let l = &mut self.lists[c];
        l.push(p);
        let full = core::mem::replace(l, List::EMPTY);
        STASH.with(|lists, _| {
            lists[c].append(full);
            STASH_HAS[c].store(true, Ordering::Relaxed);
        });
    }

    /// Hands every list and the unused arena tail back to the stash and marks
    /// the cache [`DEAD`], so whatever this thread frees or allocates later
    /// (in other thread-local destructors) goes through the stash.
    fn tear_down(&mut self) {
        self.state = DEAD;
        self.limit = [0; NCLASS];
        let lists = core::mem::replace(&mut self.lists, [List::EMPTY; NCLASS]);
        let (bump, end) = (self.bump, self.end);
        self.bump = ptr::null_mut();
        self.end = ptr::null_mut();
        STASH.with(|stash, spare| {
            for ((dst, src), has) in stash.iter_mut().zip(lists).zip(&STASH_HAS) {
                if !src.head.is_null() {
                    dst.append(src);
                    has.store(true, Ordering::Relaxed);
                }
            }
            if (end as usize) - (bump as usize) >= SPARE_MIN {
                let s = bump.cast::<Spare>();
                // SAFETY: `[bump, end)` is unused arena memory of this thread,
                // 16-aligned and large enough for the header.
                unsafe { s.write(Spare { next: *spare, end }) };
                *spare = s;
            }
        });
    }
}

const UNINIT: u8 = 0;
const INIT: u8 = 1;
const LIVE: u8 = 2;
const DEAD: u8 = 3;

/// Registered (by touching [`EXIT`]) when a thread's cache goes live; its
/// destructor tears the cache down when the thread exits.
struct ExitHook;

impl Drop for ExitHook {
    fn drop(&mut self) {
        // SAFETY: a thread-local destructor runs on its own thread, outside
        // any allocator call, so nothing else holds a reference to the cache.
        unsafe { (*cache_ptr()).tear_down() };
    }
}

std::thread_local! {
    /// The cache. It has no destructor, so reaching it is a plain
    /// thread-local address computation, valid for the whole life of the
    /// thread (no registration check on the fast path).
    static CACHE: UnsafeCell<Cache> = const { UnsafeCell::new(Cache::new()) };
    /// The thread-exit hook (see [`ExitHook`]).
    static EXIT: ExitHook = const { ExitHook };
}

/// This thread's cache. Every use is a short exclusive access from inside an
/// allocator call (never re-entered while it holds one: the cache paths call
/// only `System` and the stash, and the one call that may allocate, the
/// exit-hook registration, holds no reference) or from the exit hook.
#[inline(always)]
fn cache_ptr() -> *mut Cache {
    CACHE.with(UnsafeCell::get)
}

/// The class of a request, or `None` for one the pool does not serve (too
/// large, over-aligned, or of size zero).
#[inline(always)]
fn small_class(layout: Layout) -> Option<usize> {
    // Size zero wraps to a huge class, so the one comparison rejects it too.
    let c = layout.size().wrapping_sub(1) / GRAN;
    (c < NCLASS && layout.align() <= GRAN).then_some(c)
}

#[inline]
fn class_size(c: usize) -> usize {
    (c + 1) * GRAN
}

/// The slow path of an allocation of class `c` whose free list is empty:
/// sets the cache up on first use, refills it, or works without it.
#[cold]
fn alloc_slow(cache: *mut Cache, c: usize) -> *mut u8 {
    // SAFETY (every dereference of `cache` below): see `cache_ptr`; no
    // reference to the cache is live across the exit-hook registration.
    let state = unsafe { (*cache).state };
    match state {
        LIVE => {}
        UNINIT => {
            // SAFETY: as above.
            unsafe { (*cache).state = INIT };
            // First touch of `EXIT` registers its destructor, which may itself
            // allocate: that re-entry sees `INIT` and falls back below.
            let ok = EXIT.try_with(|_| ()).is_ok();
            // SAFETY: as above; the registration has returned.
            let cache = unsafe { &mut *cache };
            if !ok {
                cache.state = DEAD;
                return alloc_uncached(c);
            }
            cache.state = LIVE;
            for (k, l) in cache.limit.iter_mut().enumerate() {
                *l = (SPILL_BYTES / class_size(k)) as u32;
            }
        }
        _ => return alloc_uncached(c),
    }
    // SAFETY: as above (a live cache; `refill` does not re-enter).
    unsafe { (*cache).refill(c) }
}

/// Fallback allocation of class `c` without a thread cache.
#[cold]
fn alloc_uncached(c: usize) -> *mut u8 {
    let p = STASH.with(|lists, _| lists[c].pop());
    if !p.is_null() {
        return p;
    }
    // SAFETY: a class size is non-zero and `GRAN` a power of two.
    unsafe { System.alloc(Layout::from_size_align_unchecked(class_size(c), GRAN)) }
}

/// Fallback free of class `c` without a thread cache.
#[cold]
fn dealloc_uncached(p: *mut u8, c: usize) {
    STASH.with(|lists, _| {
        lists[c].push(p);
        STASH_HAS[c].store(true, Ordering::Relaxed);
    });
}

/// Allocates a block of class `c` (`< NCLASS`): the common case pops the
/// thread's list.
#[inline]
fn alloc_class(c: usize) -> *mut u8 {
    let cache = cache_ptr();
    // SAFETY: see `cache_ptr`; `c < NCLASS`.
    let p = unsafe { (*cache).lists.get_unchecked_mut(c).pop() };
    if p.is_null() { alloc_slow(cache, c) } else { p }
}

/// Frees block `p` of class `c` (`< NCLASS`): the common case pushes it on
/// the thread's list.
#[inline]
fn dealloc_class(p: *mut u8, c: usize) {
    // SAFETY: see `cache_ptr`; `c < NCLASS`.
    unsafe {
        let cache = &mut *cache_ptr();
        let limit = *cache.limit.get_unchecked(c) as usize;
        let l = cache.lists.get_unchecked_mut(c);
        if l.len < limit {
            l.push(p);
        } else {
            cache.dealloc_slow(p, c);
        }
    }
}

/// The pooling allocator. A zero-sized handle; all state is per-thread or in
/// the process-wide stash. See the [module docs](crate::pool_alloc).
#[derive(Debug, Default, Clone, Copy)]
pub struct PoolAlloc;

// SAFETY: small blocks come from a class whose block size (a multiple of 16)
// is at least the requested size, and are 16-aligned (arenas are 16-aligned
// and every class size is a multiple of 16), which satisfies any alignment up
// to 16; every other request is forwarded to `System` unchanged. A block is
// only ever on one free list, so no two live allocations overlap.
//
// The entry points are kept out of line: inlined into every allocation site
// of the program, the fast paths grew the executable by ~4% (and its
// resident size with it) for ~1% fewer instructions.
unsafe impl GlobalAlloc for PoolAlloc {
    #[inline(never)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match small_class(layout) {
            Some(c) => alloc_class(c),
            // SAFETY: forwarded with the caller's (valid, non-zero) layout.
            None => unsafe { System.alloc(layout) },
        }
    }

    #[inline(never)]
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        match small_class(layout) {
            Some(c) => dealloc_class(p, c),
            // SAFETY: a non-small layout was allocated by `System` with this
            // same layout.
            None => unsafe { System.dealloc(p, layout) },
        }
    }

    #[inline(never)]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let Some(c) = small_class(layout) else {
            // SAFETY: forwarded with the caller's layout.
            return unsafe { System.alloc_zeroed(layout) };
        };
        let p = alloc_class(c);
        if !p.is_null() {
            // SAFETY: `p` is valid for `layout.size()` bytes.
            unsafe { ptr::write_bytes(p, 0, layout.size()) };
        }
        p
    }

    #[inline(never)]
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `GlobalAlloc::realloc` requires `new_size` with the old
        // alignment to form a valid layout.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        match (small_class(layout), small_class(new_layout)) {
            // SAFETY: both sides are `System` allocations.
            (None, None) => return unsafe { System.realloc(p, layout, new_size) },
            (Some(a), Some(b)) if a == b => return p,
            _ => {}
        }
        // SAFETY: `new_layout` is valid (see above).
        let q = unsafe { self.alloc(new_layout) };
        if !q.is_null() {
            // SAFETY: `p` holds `layout.size()` bytes, `q` at least
            // `new_size`; distinct allocations cannot overlap.
            unsafe { ptr::copy_nonoverlapping(p, q, layout.size().min(new_size)) };
            // SAFETY: `p` was allocated by this allocator with `layout`.
            unsafe { self.dealloc(p, layout) };
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn lay(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn classes_round_up() {
        let class_of = |size| small_class(lay(size, 8));
        assert_eq!(class_of(1), Some(0));
        assert_eq!(class_of(16), Some(0));
        assert_eq!(class_of(17), Some(1));
        assert_eq!(class_of(MAX_SMALL).map(class_size), Some(MAX_SMALL));
        assert_eq!(class_of(MAX_SMALL), Some(NCLASS - 1));
        assert_eq!(class_of(MAX_SMALL + 1), None);
        assert_eq!(small_class(lay(0, 1)), None);
        assert_eq!(small_class(lay(64, 32)), None);
    }

    /// Fills each block with a per-block byte pattern and checks it survives
    /// interleaved allocations and frees (overlap / corruption detector).
    fn churn(seed: usize, rounds: usize) {
        let a = PoolAlloc;
        let mut live: Vec<(*mut u8, Layout, u8)> = Vec::new();
        let mut x = seed.wrapping_mul(2_654_435_761) | 1;
        for i in 0..rounds {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let size = match x % 8 {
                0 => 1 + x % 16,
                1..=4 => 1 + x % 200,
                5 | 6 => 1 + x % MAX_SMALL,
                _ => 1 + x % 5000,
            };
            let align = [1, 2, 8, 16, 32][(x >> 20) % 5];
            if !x.is_multiple_of(3) || live.is_empty() {
                let l = lay(size, align);
                // SAFETY: test of the allocator contract; non-zero size.
                let p = unsafe { a.alloc(l) };
                assert!(!p.is_null());
                assert_eq!(p as usize % align, 0);
                let tag = (i % 251) as u8;
                // SAFETY: `p` is valid for `size` bytes.
                unsafe { ptr::write_bytes(p, tag, size) };
                live.push((p, l, tag));
            } else {
                let (p, l, tag) = live.swap_remove(x % live.len());
                // SAFETY: `p` is a live block of `l.size()` bytes.
                let s = unsafe { core::slice::from_raw_parts(p, l.size()) };
                assert!(s.iter().all(|&b| b == tag), "block corrupted");
                if x.is_multiple_of(5) {
                    let ns = 1 + (x >> 8) % 3000;
                    // SAFETY: `p` was allocated with `l`; `ns` is non-zero.
                    let q = unsafe { a.realloc(p, l, ns) };
                    assert!(!q.is_null());
                    let nl = lay(ns, l.align());
                    // SAFETY: `q` holds the preserved prefix.
                    let s = unsafe { core::slice::from_raw_parts(q, l.size().min(ns)) };
                    assert!(s.iter().all(|&b| b == tag), "realloc lost data");
                    // SAFETY: `q` is valid for `ns` bytes.
                    unsafe { ptr::write_bytes(q, tag, ns) };
                    live.push((q, nl, tag));
                } else {
                    // SAFETY: allocated by `a` with `l`.
                    unsafe { a.dealloc(p, l) };
                }
            }
        }
        for (p, l, tag) in live {
            // SAFETY: as above.
            let s = unsafe { core::slice::from_raw_parts(p, l.size()) };
            assert!(s.iter().all(|&b| b == tag), "block corrupted");
            // SAFETY: allocated by `a` with `l`.
            unsafe { a.dealloc(p, l) };
        }
    }

    #[test]
    fn single_thread_churn() {
        churn(1, 200_000);
    }

    #[test]
    fn zeroed_is_zero() {
        let a = PoolAlloc;
        for size in [1, 15, 16, 100, 1024, 4096] {
            let l = lay(size, 8);
            // SAFETY: non-zero size; freed below with the same layout.
            unsafe {
                let p = a.alloc(l);
                ptr::write_bytes(p, 0xAB, size);
                a.dealloc(p, l);
                let z = a.alloc_zeroed(l);
                assert!(core::slice::from_raw_parts(z, size).iter().all(|&b| b == 0));
                a.dealloc(z, l);
            }
        }
    }

    /// A spare arena tail shorter than the block a fresh thread needs must be
    /// skipped, not bump-allocated past its end (this overran glibc's heap in
    /// the Atomics agent tests before `refill` re-checked).
    #[test]
    fn short_spare_region_is_not_overrun() {
        std::thread::spawn(|| {
            let region = Layout::from_size_align(32, GRAN).unwrap();
            // SAFETY: a 32-byte, 16-aligned system block becomes a spare
            // region with a valid header; it is never handed back.
            let r = unsafe { System.alloc(region) };
            assert!(!r.is_null());
            STASH.with(|_, spare| {
                // SAFETY: `r` holds 32 bytes, enough for the header.
                unsafe {
                    r.cast::<Spare>().write(Spare {
                        next: *spare,
                        end: r.add(32),
                    });
                }
                *spare = r.cast();
            });
            let a = PoolAlloc;
            let l = lay(MAX_SMALL, 8);
            // SAFETY: non-zero size; freed below.
            let p = unsafe { a.alloc(l) };
            assert!(!p.is_null());
            let (lo, hi) = (r as usize, r as usize + 32);
            assert!(p as usize + MAX_SMALL <= lo || p as usize >= hi);
            // SAFETY: `p` is valid for `MAX_SMALL` bytes.
            unsafe {
                ptr::write_bytes(p, 1, MAX_SMALL);
                a.dealloc(p, l);
            }
        })
        .join()
        .unwrap();
    }

    /// Many threads churning concurrently, plus blocks allocated on one
    /// thread and freed on another, plus threads exiting with full caches.
    #[test]
    fn multi_thread_stress() {
        let a = PoolAlloc;
        let handles: Vec<_> = (0..8)
            .map(|t| std::thread::spawn(move || churn(t + 7, 60_000)))
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        // Cross-thread frees: a producer allocates, consumers free.
        for round in 0..4 {
            let l = lay(48 + round * 16, 8);
            let blocks: Vec<usize> = (0..20_000)
                .map(|i| {
                    // SAFETY: non-zero size.
                    let p = unsafe { a.alloc(l) };
                    // SAFETY: `p` is valid for `l.size()` bytes.
                    unsafe { ptr::write_bytes(p, i as u8, l.size()) };
                    p as usize
                })
                .collect();
            let chunks: Vec<Vec<usize>> = blocks.chunks(5000).map(<[usize]>::to_vec).collect();
            let hs: Vec<_> = chunks
                .into_iter()
                .map(|c| {
                    std::thread::spawn(move || {
                        for p in c {
                            // SAFETY: allocated with `l` on the producer.
                            unsafe { a.dealloc(p as *mut u8, l) };
                        }
                        churn(round, 5_000);
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
        }
        churn(99, 50_000);
    }
}
