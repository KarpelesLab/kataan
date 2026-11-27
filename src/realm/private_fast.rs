//! The inline cache of a private-element site (`o.#x`, `o.#x = v`, `#x in o`).
//!
//! A private name is a per-class-evaluation key string (`"\0#x@v7f"`, its last
//! byte the element kind), and a private field is stored on its object as an
//! ordinary (hidden) data slot under that key. Objects built by one
//! constructor add their fields in one order, so a site sees few layouts: a
//! [`PrivCache`] remembers the last few (a base-class method reached by its
//! subclasses' instances sees one per subclass), as each layout and the
//! key's slot in it, plus the key itself, since the same site runs with another class
//! evaluation's name when a class is evaluated more than once. A hit — the
//! object is an ordinary shaped object of a recorded layout, and the site's
//! private name is the recorded one, compared by its text — is exactly the
//! slow path's: the layout fixes that the object has the key (so the brand
//! check passes) and in which slot. Anything else (another layout, a
//! dictionary-mode object, an accessor or method element, a proxy or other
//! cell) is a miss, for the caller's general path, which keeps every error.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;

/// A private field's kind byte (the private name's last byte).
const PRIV_FIELD: u8 = b'f';
/// A private method's kind byte.
const PRIV_METHOD: u8 = b'm';
/// A private accessor's kind byte.
const PRIV_ACCESSOR: u8 = b'a';

/// One private-element site's memo (see the module docs). Empty when cold,
/// and when copied (a fresh copy of the bytecode starts cold).
#[derive(Default)]
pub struct PrivCache(
    core::cell::Cell<Option<Box<PrivEntry>>>,
    core::cell::OnceCell<Box<JitPrivWay>>,
);

/// How many layouts a [`JitPrivWay`] keeps (a base-class method reached by
/// its subclasses' instances sees one per subclass).
pub const JIT_PRIV_WAYS: usize = 4;

/// A private-element site's memo for compiled code (`jit.rs`, the inline
/// `o.#x` / `o.#x = v`): the accesses it served last — the private name's
/// handle bits and the heap epoch they were taken under (while it is
/// unchanged the handle still names that string), and up to
/// [`JIT_PRIV_WAYS`] layouts with the name's slot in each. Compiled code
/// reads it in place (made when the site is compiled, at a fixed address),
/// so its fields are at fixed offsets.
#[derive(Default)]
#[repr(C)]
pub struct JitPrivWay {
    pub(crate) key_bits: core::cell::Cell<u64>,
    pub(crate) epoch: core::cell::Cell<u64>,
    pub(crate) ways: [JitPrivEntry; JIT_PRIV_WAYS],
    next: core::cell::Cell<u64>,
}

/// One layout of a [`JitPrivWay`]: `shape` is the layout's `Rc` word as an
/// object's shape field holds it (`None` is zero, which no object's is),
/// keeping that layout alive while the memo names it.
#[derive(Default)]
#[repr(C)]
pub struct JitPrivEntry {
    pub(crate) shape: core::cell::Cell<Option<Rc<Shape>>>,
    pub(crate) slot: core::cell::Cell<u64>,
}

impl PrivCache {
    /// The site's memo for compiled code (made on first use: only a
    /// compiled copy of the site asks for it).
    #[must_use]
    pub fn jit_way(&self) -> &JitPrivWay {
        self.1.get_or_init(Box::default)
    }

    /// Records in [`jit_way`](Self::jit_way), when compiled code has one,
    /// an access the site served: name `k` (its handle under heap epoch
    /// `epoch`) in layout `shape` at `slot`.
    fn note_jit(&self, k: NanBox, epoch: u32, shape: &Rc<Shape>, slot: u32) {
        let Some(w) = self.1.get() else {
            return;
        };
        if w.key_bits.get() != k.to_bits() || w.epoch.get() != u64::from(epoch) {
            for e in &w.ways {
                drop(e.shape.take());
            }
            w.key_bits.set(k.to_bits());
            w.epoch.set(u64::from(epoch));
        }
        for e in &w.ways {
            let s = e.shape.take();
            let hit = s.as_ref().is_some_and(|s| Rc::ptr_eq(s, shape));
            e.shape.set(s);
            if hit {
                e.slot.set(u64::from(slot));
                return;
            }
        }
        let i = (w.next.get() as usize) % JIT_PRIV_WAYS;
        w.ways[i].shape.set(Some(Rc::clone(shape)));
        w.ways[i].slot.set(u64::from(slot));
        w.next.set(((i + 1) % JIT_PRIV_WAYS) as u64);
    }
}

struct PrivEntry {
    /// The layouts seen with the key's slot in each, most recently filled
    /// at `next - 1` (a site reached by the instances of a few classes that
    /// share a base — each subclass's instances have their own layout —
    /// keeps them all).
    ways: [Option<(Rc<Shape>, u32)>; PRIV_WAYS],
    /// Where the next layout goes (round robin).
    next: u8,
    key: Box<[u8]>,
    /// The key's string handle bits as last seen naming `key`, and the heap
    /// epoch then (`Heap::epoch`: while it is unchanged the handle still
    /// names that string), so a run with the same name skips reading and
    /// comparing its text.
    key_bits: u64,
    key_epoch: u32,
}

/// How many layouts a [`PrivCache`] keeps.
const PRIV_WAYS: usize = 4;

impl Clone for PrivCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl core::fmt::Debug for PrivCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PrivCache")
    }
}

impl Realm {
    /// The slot of the private name `k` (a string) in the ordinary shaped
    /// object `o`, through `cache`; `None` when `o` is anything else, or its
    /// layout lacks the key. `kinds` lists the element kinds served.
    fn private_slot(&self, o: NanBox, k: NanBox, cache: &PrivCache, kinds: &[u8]) -> Option<u32> {
        let Some(Cell::Object(obj)) = self.heap.get(Handle::from_raw(o.as_handle()?)) else {
            return None;
        };
        let shape = obj.shaped_shape()?;
        let mut memo = cache.0.take();
        // The same name as last time (its handle, under the same epoch):
        // the recorded key, without reading the string.
        let epoch = self.heap.epoch();
        if let Some(e) = &memo
            && e.key_bits == k.to_bits()
            && e.key_epoch == epoch
            && e.key.last().is_some_and(|b| kinds.contains(b))
            && let Some(slot) = e
                .ways
                .iter()
                .flatten()
                .find(|(s, _)| Rc::ptr_eq(s, shape))
                .map(|(_, slot)| *slot)
        {
            cache.0.set(memo);
            return Some(slot);
        }
        let Some(key) = k
            .as_handle()
            .and_then(|h| self.string_leaf_bytes(Handle::from_raw(h)))
            .filter(|key| key.last().is_some_and(|b| kinds.contains(b)))
        else {
            cache.0.set(memo);
            return None;
        };
        if let Some(e) = &mut memo
            && *e.key == *key
        {
            e.key_bits = k.to_bits();
            e.key_epoch = epoch;
            if let Some(slot) = e
                .ways
                .iter()
                .flatten()
                .find(|(s, _)| Rc::ptr_eq(s, shape))
                .map(|(_, slot)| *slot)
            {
                cache.0.set(memo);
                return Some(slot);
            }
        }
        let slot = core::str::from_utf8(key).ok().and_then(|t| shape.lookup(t));
        if let Some(slot) = slot {
            match &mut memo {
                Some(e) if *e.key == *key => {
                    let i = usize::from(e.next) % PRIV_WAYS;
                    e.ways[i] = Some((Rc::clone(shape), slot));
                    e.next = ((i + 1) % PRIV_WAYS) as u8;
                }
                _ => {
                    let mut ways: [Option<(Rc<Shape>, u32)>; PRIV_WAYS] = Default::default();
                    ways[0] = Some((Rc::clone(shape), slot));
                    memo = Some(Box::new(PrivEntry {
                        ways,
                        next: 1,
                        key: key.into(),
                        key_bits: k.to_bits(),
                        key_epoch: epoch,
                    }));
                }
            }
        }
        cache.0.set(memo);
        slot
    }

    /// `o.#k` for a private field or method `k` found through `cache` (see
    /// the module docs); `None`: take the general path.
    #[must_use]
    pub fn private_cached_get(&self, o: NanBox, k: NanBox, cache: &PrivCache) -> Option<NanBox> {
        let slot = self.private_slot(o, k, cache, &[PRIV_FIELD, PRIV_METHOD])?;
        let Some(Cell::Object(obj)) = self.heap.get(Handle::from_raw(o.as_handle()?)) else {
            return None;
        };
        let v = obj.slot(slot)?;
        // The read compiled code may repeat (`JitPrivWay`).
        if let Some(shape) = obj.shaped_shape() {
            cache.note_jit(k, self.heap.epoch(), shape, slot);
        }
        Some(v)
    }

    /// `o.#k = v` for a private field `k` found through `cache` (see the
    /// module docs): the in-place store (a private element is not a property,
    /// so neither freezing nor a read-only flag applies). `false`, having
    /// changed nothing: take the general path.
    pub fn private_cached_set(
        &mut self,
        o: NanBox,
        k: NanBox,
        v: NanBox,
        cache: &PrivCache,
    ) -> bool {
        let (Some(slot), Some(raw)) =
            (self.private_slot(o, k, cache, &[PRIV_FIELD]), o.as_handle())
        else {
            return false;
        };
        let h = Handle::from_raw(raw);
        let epoch = self.heap.epoch();
        let Some(Cell::Object(obj)) = self.heap.get_mut(h) else {
            return false;
        };
        if !obj.set_slot(slot, v) {
            return false;
        }
        // The write compiled code may repeat (`JitPrivWay`).
        if let Some(shape) = obj.shaped_shape() {
            cache.note_jit(k, epoch, shape, slot);
        }
        self.write_barrier(h, v);
        true
    }

    /// `#k in o` when `cache` (see the module docs) finds `k` in the ordinary
    /// shaped object `o`: `Some(true)`; `None` otherwise (the general path).
    #[must_use]
    pub fn private_cached_has(&self, o: NanBox, k: NanBox, cache: &PrivCache) -> Option<bool> {
        self.private_slot(o, k, cache, &[PRIV_FIELD, PRIV_METHOD, PRIV_ACCESSOR])
            .map(|_| true)
    }
}
