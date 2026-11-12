//! The shared function table: an append-only store of [`FnProto`]s that
//! dynamic code extends in amortized O(1) per function.
//!
//! A function value records only its index in the table, so every script,
//! `eval`, `Function(…)` body and module of one interpreter compiles into one
//! growth chain, and a running frame keeps the snapshot it started with. When
//! each growth copied the whole table (`table.to_vec()` plus the new tail), N
//! `Function(…)` calls cost O(N²): 32 000 of them took 84 s. Here the
//! snapshots of a chain share one buffer, the way [`crate::rope`]'s `s += x`
//! strings do: each holds a prefix of it (`len` entries), and extending the
//! snapshot that ends where the buffer's written part ends (`init`) writes the
//! new entries into the spare capacity behind it. Entries below `init` are
//! never written again, so every snapshot's view is fixed; extending any other
//! snapshot, or past the capacity, copies into a fresh buffer of twice the
//! size, so the copying is amortized O(1) per appended function.

use super::FnProto;
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, UnsafeCell};
use core::mem::MaybeUninit;

/// The buffer behind a chain of [`FnTable`] snapshots.
struct FnBuf {
    /// How many leading slots are written (and immutable from then on).
    init: Cell<usize>,
    /// The slots; `slots.len()` is the capacity.
    slots: Box<[UnsafeCell<MaybeUninit<FnProto>>]>,
}

impl FnBuf {
    /// An empty buffer with room for `cap` functions.
    fn with_capacity(cap: usize) -> Rc<FnBuf> {
        Rc::new(FnBuf {
            init: Cell::new(0),
            slots: (0..cap)
                .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
                .collect(),
        })
    }

    /// The first `len` entries (`len <= init`).
    #[inline]
    fn prefix(&self, len: usize) -> &[FnProto] {
        let len = len.min(self.init.get());
        // SAFETY: `UnsafeCell<MaybeUninit<T>>` has `T`'s layout (both are
        // `repr(transparent)`); the slots below `init` are initialized and never
        // written again (`push_at` writes only at and past `init`), so nothing
        // mutates what this borrow covers.
        #[allow(unsafe_code)]
        unsafe {
            core::slice::from_raw_parts(
                UnsafeCell::raw_get(self.slots.as_ptr())
                    .cast::<FnProto>()
                    .cast_const(),
                len,
            )
        }
    }

    /// Writes `items` at `at` when `at` is where the written part ends and
    /// they fit the spare capacity; hands them back otherwise.
    fn push_at(&self, at: usize, items: Vec<FnProto>) -> Result<(), Vec<FnProto>> {
        if at != self.init.get() || self.slots.len() - at < items.len() {
            return Err(items);
        }
        for (i, p) in items.into_iter().enumerate() {
            // SAFETY: slot `at + i` is inside `slots` (checked above) and at or
            // past `init`, which no borrow from `prefix` reaches; it holds no
            // value yet, so overwriting it leaks nothing.
            #[allow(unsafe_code)]
            unsafe {
                self.slots[at + i].get().write(MaybeUninit::new(p));
            }
            // Counted one at a time, so `Drop` sees exactly the written slots.
            self.init.set(at + i + 1);
        }
        Ok(())
    }
}

impl Drop for FnBuf {
    fn drop(&mut self) {
        let init = self.init.get();
        for slot in &mut self.slots[..init] {
            // SAFETY: the slots below `init` were written by `push_at` and are
            // dropped exactly once, here.
            #[allow(unsafe_code)]
            unsafe {
                slot.get_mut().assume_init_drop();
            }
        }
    }
}

/// A snapshot of the shared function table: the first `len` functions of an
/// append-only buffer. Cloning is O(1); it dereferences to `[FnProto]`.
#[derive(Clone)]
pub struct FnTable {
    buf: Rc<FnBuf>,
    len: usize,
}

impl FnTable {
    /// The empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: FnBuf::with_capacity(0),
            len: 0,
        }
    }

    /// This table followed by `tail` (whose functions take the ids from
    /// `self.len()` on). In place when this snapshot is the newest of its
    /// buffer and the tail fits; otherwise copied into a buffer with room to
    /// double, so a chain of extensions costs amortized O(1) per function.
    #[must_use]
    pub fn extended(&self, tail: Vec<FnProto>) -> Self {
        if tail.is_empty() {
            return self.clone();
        }
        let len = self.len + tail.len();
        match self.buf.push_at(self.len, tail) {
            Ok(()) => Self {
                buf: Rc::clone(&self.buf),
                len,
            },
            Err(tail) => {
                let buf = FnBuf::with_capacity(len.max(self.len * 2).max(16));
                let copied = buf.push_at(0, self.to_vec()).is_ok();
                let appended = buf.push_at(self.len, tail).is_ok();
                debug_assert!(copied && appended, "a fresh buffer holds both");
                Self { buf, len }
            }
        }
    }

    /// Whether `a` and `b` are the same snapshot (one buffer, one length).
    #[must_use]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Rc::ptr_eq(&a.buf, &b.buf) && a.len == b.len
    }
}

impl Default for FnTable {
    fn default() -> Self {
        Self::new()
    }
}

impl core::ops::Deref for FnTable {
    type Target = [FnProto];
    #[inline]
    fn deref(&self) -> &[FnProto] {
        self.buf.prefix(self.len)
    }
}

impl From<Vec<FnProto>> for FnTable {
    fn from(v: Vec<FnProto>) -> Self {
        let len = v.len();
        let buf = FnBuf::with_capacity(len);
        let ok = buf.push_at(0, v).is_ok();
        debug_assert!(ok, "a fresh buffer holds its contents");
        Self { buf, len }
    }
}

impl core::fmt::Debug for FnTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The functions one compilation adds to a table that already holds `base`
/// of them: indexed by table id (`base..`), but storing only the new ones, so
/// a compile into a large table copies none of it.
#[derive(Debug, Default)]
pub struct ProtoVec {
    base: usize,
    items: Vec<FnProto>,
}

impl ProtoVec {
    /// An empty tail whose first function will take id `base`.
    #[must_use]
    pub fn after(base: usize) -> Self {
        Self {
            base,
            items: Vec::new(),
        }
    }

    /// The table length this tail extends to (`base` plus the new functions).
    #[must_use]
    pub fn len(&self) -> usize {
        self.base + self.items.len()
    }

    /// Whether the table it extends is empty and nothing was added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The id the first new function takes.
    #[must_use]
    pub fn base(&self) -> usize {
        self.base
    }

    /// Appends a function (it takes id `len()`).
    pub fn push(&mut self, p: FnProto) {
        self.items.push(p);
    }

    /// Drops the functions from id `len` on (never below `base`).
    pub fn truncate(&mut self, len: usize) {
        self.items.truncate(len.saturating_sub(self.base));
    }

    /// The new functions, in id order.
    #[must_use]
    pub fn into_tail(self) -> Vec<FnProto> {
        self.items
    }
}

impl Extend<FnProto> for ProtoVec {
    fn extend<I: IntoIterator<Item = FnProto>>(&mut self, iter: I) {
        self.items.extend(iter);
    }
}

impl From<Vec<FnProto>> for ProtoVec {
    fn from(items: Vec<FnProto>) -> Self {
        Self { base: 0, items }
    }
}

impl core::ops::Index<usize> for ProtoVec {
    type Output = FnProto;
    #[inline]
    fn index(&self, id: usize) -> &FnProto {
        &self.items[id - self.base]
    }
}

impl core::ops::IndexMut<usize> for ProtoVec {
    #[inline]
    fn index_mut(&mut self, id: usize) -> &mut FnProto {
        &mut self.items[id - self.base]
    }
}

impl core::ops::Index<core::ops::RangeFrom<usize>> for ProtoVec {
    type Output = [FnProto];
    #[inline]
    fn index(&self, r: core::ops::RangeFrom<usize>) -> &[FnProto] {
        &self.items[r.start - self.base..]
    }
}

impl core::ops::IndexMut<core::ops::RangeFrom<usize>> for ProtoVec {
    #[inline]
    fn index_mut(&mut self, r: core::ops::RangeFrom<usize>) -> &mut [FnProto] {
        &mut self.items[r.start - self.base..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proto(name: &str) -> FnProto {
        FnProto {
            name: alloc::string::String::from(name),
            ..super::super::placeholder_proto()
        }
    }

    #[test]
    fn snapshots_share_a_buffer_and_keep_their_view() {
        let a = FnTable::from(alloc::vec![proto("a")]);
        let b = a.extended(alloc::vec![proto("b")]);
        let c = b.extended(alloc::vec![proto("c"), proto("d")]);
        // `b` extended in place again would clobber nothing: `c` took the
        // slots after it, so extending `b` now must copy.
        let b2 = b.extended(alloc::vec![proto("x")]);
        let names = |t: &FnTable| t.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&a), ["a"]);
        assert_eq!(names(&b), ["a", "b"]);
        assert_eq!(names(&c), ["a", "b", "c", "d"]);
        assert_eq!(names(&b2), ["a", "b", "x"]);
        assert!(Rc::ptr_eq(&b.buf, &c.buf));
        assert!(!Rc::ptr_eq(&b.buf, &b2.buf));
        // A long chain reallocates only O(log n) times.
        let mut t = FnTable::new();
        let mut bufs = 0;
        for i in 0..1000 {
            let next = t.extended(alloc::vec![proto("f")]);
            if !Rc::ptr_eq(&next.buf, &t.buf) {
                bufs += 1;
            }
            t = next;
            assert_eq!(t.len(), i + 1);
        }
        assert!(bufs <= 8, "{bufs} reallocations");
    }

    #[test]
    fn proto_vec_indexes_by_table_id() {
        let mut v = ProtoVec::after(10);
        v.push(proto("p"));
        v.push(proto("q"));
        assert_eq!(v.len(), 12);
        assert_eq!(v[11].name, "q");
        v[10].name = alloc::string::String::from("r");
        assert_eq!(v[10..].len(), 2);
        v.truncate(11);
        assert_eq!(v.into_tail().len(), 1);
    }
}
