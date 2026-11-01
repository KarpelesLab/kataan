//! The `for-in` key-list cache.
//!
//! `for (k in o)` takes the list of `o`'s enumerable String keys, its own and
//! inherited ones (EnumerateObjectProperties, §14.7.5.9), every time the loop
//! is entered. Building it walks the whole prototype chain and every key of
//! every link (each own key shadows a like-named inherited one, enumerable or
//! not), so for a small object most of the work is `Object.prototype`'s
//! non-enumerable methods. That list depends on nothing but the receiver's
//! own keys and the chain's, so it is memoized here.
//!
//! An entry is keyed on the receiver's shape. A **plain shaped** ordinary
//! object (only data properties, all enumerable: see
//! [`Object::is_plain_shaped`](crate::object::Object::is_plain_shaped)) has
//! exactly the keys its shape lists, in its order, so every receiver with
//! that shape has the same own share of the list. Each prototype-chain link
//! is recorded by handle (a handle carries its slot's generation, so a freed
//! and reused slot never matches) and key epoch, which changes whenever the
//! link's own keys, or their enumerability, may have changed; a different
//! `[[Prototype]]` shows as a different next handle. Only ordinary,
//! non-wrapper, non-host-exotic objects take part (never a proxy, an array,
//! a typed array, a String wrapper, the global object or a function), and a
//! moved cell (`Realm::swap_cell_state`, compaction) retires every entry.
//!
//! The cached key Strings are handed to the loop as they are (a String is
//! immutable, and its identity is not observable), so they are GC roots.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// How many receiver shapes the cache remembers (direct-mapped).
const WAYS: usize = 8;

/// The longest prototype chain an entry records.
const MAX_LINKS: usize = 8;

/// One memoized key list (see the module docs).
struct Entry {
    /// The receiver's shape (held, so its address is not reused).
    shape: Rc<Shape>,
    /// `(handle, key epoch)` per prototype-chain link, nearest first; the
    /// last link's `[[Prototype]]` is `null`.
    links: Vec<(u64, u32)>,
    /// `Realm::cell_swaps` when the entry was made.
    swaps: u32,
    /// The `for-in` key list.
    keys: Vec<NanBox>,
}

/// The realm's `for-in` key-list cache (see the module docs).
#[derive(Default)]
pub(crate) struct ForInCache {
    entries: [Option<Entry>; WAYS],
}

/// The slot an entry for `shape` lives in.
fn way(shape: &Rc<Shape>) -> usize {
    let p = Rc::as_ptr(shape) as usize;
    (p >> 4 ^ p >> 9) % WAYS
}

impl Realm {
    /// The receiver's shape, when `h` is a plain shaped ordinary object
    /// whose `for-in` share is decided by that shape alone.
    fn for_in_receiver_shape(&self, h: Handle) -> Option<&Rc<Shape>> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if !o.is_plain_shaped() || o.is_prim_wrapper() || self.is_host_exotic(h) {
            return None;
        }
        o.shaped_shape()
    }

    /// One prototype-chain link's `(key epoch, [[Prototype]])`, when it is an
    /// ordinary object whose `for-in` share its own keys decide.
    fn for_in_link(&self, h: Handle) -> Option<(u32, Option<Handle>)> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if o.is_prim_wrapper()
            || o.jit_flags() & crate::object::JIT_FLAG_GLOBAL != 0
            || self.is_host_exotic(h)
        {
            return None;
        }
        Some((o.key_epoch(), o.proto()))
    }

    /// The memoized `for-in` key list of `h`, when an entry still describes
    /// it exactly.
    #[must_use]
    pub(crate) fn for_in_cached(&self, h: Handle) -> Option<&[NanBox]> {
        let cache = self.for_in_cache.as_deref()?;
        let shape = self.for_in_receiver_shape(h)?;
        let e = cache.entries[way(shape)].as_ref()?;
        if !Rc::ptr_eq(&e.shape, shape) || e.swaps != self.cell_swaps {
            return None;
        }
        let mut cur = self.object_proto(h);
        for &(raw, epoch) in &e.links {
            let link = cur?;
            if link.to_raw() != raw {
                return None;
            }
            let (now, next) = self.for_in_link(link)?;
            if now != epoch {
                return None;
            }
            cur = next;
        }
        cur.is_none().then_some(e.keys.as_slice())
    }

    /// A new array of the memoized `for-in` key list of `h` (see
    /// [`for_in_cached`](Self::for_in_cached)), whose liveness object is `h`
    /// itself. `None` on a miss.
    pub(crate) fn for_in_cached_array(&mut self, h: Handle) -> Option<NanBox> {
        let keys = self.for_in_cached(h)?.to_vec();
        Some(NanBox::handle(self.new_array(keys).to_raw()))
    }

    /// Records `keys` as the `for-in` key list of `h`, when `h` and its
    /// prototype chain are ones an entry can describe (see the module docs).
    pub(crate) fn for_in_remember(&mut self, h: Handle, keys: &[NanBox]) {
        let Some(shape) = self.for_in_receiver_shape(h).cloned() else {
            return;
        };
        let mut links = Vec::new();
        let mut cur = self.object_proto(h);
        while let Some(link) = cur {
            if links.len() == MAX_LINKS {
                return;
            }
            let Some((epoch, next)) = self.for_in_link(link) else {
                return;
            };
            links.push((link.to_raw(), epoch));
            cur = next;
        }
        let entry = Entry {
            links,
            swaps: self.cell_swaps,
            keys: keys.to_vec(),
            shape,
        };
        let w = way(&entry.shape);
        self.for_in_cache
            .get_or_insert_with(Default::default)
            .entries[w] = Some(entry);
    }

    /// The cached key Strings, which are GC roots.
    pub(crate) fn for_in_cache_roots(&self, out: &mut Vec<Handle>) {
        if let Some(c) = self.for_in_cache.as_deref() {
            for e in c.entries.iter().flatten() {
                out.extend(
                    e.keys
                        .iter()
                        .filter_map(|k| k.as_handle().map(Handle::from_raw)),
                );
            }
        }
    }
}
