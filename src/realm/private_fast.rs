//! The inline cache of a private-element site (`o.#x`, `o.#x = v`, `#x in o`).
//!
//! A private name is a per-class-evaluation key string (`"\0#x@v7f"`, its last
//! byte the element kind), and a private field is stored on its object as an
//! ordinary (hidden) data slot under that key. Objects built by one
//! constructor add their fields in one order, so a site sees few layouts: a
//! [`PrivCache`] remembers the last one, as the layout and the key's slot in
//! it, plus the key itself, since the same site runs with another class
//! evaluation's name when a class is evaluated more than once. A hit — the
//! object is an ordinary shaped object of the recorded layout, and the site's
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
pub struct PrivCache(core::cell::Cell<Option<Box<PrivEntry>>>);

struct PrivEntry {
    shape: Rc<Shape>,
    slot: u32,
    key: Box<[u8]>,
}

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
        let key = self.string_leaf_bytes(Handle::from_raw(k.as_handle()?))?;
        if !key.last().is_some_and(|b| kinds.contains(b)) {
            return None;
        }
        let Some(Cell::Object(obj)) = self.heap.get(Handle::from_raw(o.as_handle()?)) else {
            return None;
        };
        let shape = obj.shaped_shape()?;
        let memo = cache.0.take();
        if let Some(e) = &memo
            && Rc::ptr_eq(&e.shape, shape)
            && *e.key == *key
        {
            let slot = e.slot;
            cache.0.set(memo);
            return Some(slot);
        }
        let slot = shape.lookup(core::str::from_utf8(key).ok()?);
        match slot {
            Some(slot) => cache.0.set(Some(Box::new(PrivEntry {
                shape: Rc::clone(shape),
                slot,
                key: key.into(),
            }))),
            None => cache.0.set(memo),
        }
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
        obj.slot(slot)
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
        let Some(Cell::Object(obj)) = self.heap.get_mut(h) else {
            return false;
        };
        if !obj.set_slot(slot, v) {
            return false;
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
