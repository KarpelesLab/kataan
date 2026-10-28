//! Direct slot access to the internal (hidden) state of built-in objects whose
//! layout is fixed by the engine, such as a live Set/Map iterator.
//!
//! Such an object keeps its state in hidden data properties (`\0`-prefixed
//! keys) that the engine reads and rewrites on every step. Each by-name access
//! is a shape walk with string compares, which made one `for (x of set)` step
//! cost ~1.5k instructions. An object built by the same code goes through the
//! same keys in the same order and so reaches the same [`Shape`]. A
//! [`HiddenLayout`] records that shape and the slot of each key once. Any
//! object found with exactly that shape (pointer identity) can then be read
//! and written by slot. The answer is the same as by name: the shape decides
//! which slot holds which key.
//!
//! The by-name writes ([`Realm::set_hidden_property`]) are skipped for a
//! frozen object or a read-only key; a direct write is only offered when
//! neither can apply. Anything else — a different shape, dictionary mode, a
//! non-object — answers `None`, and the caller takes its by-name path.

use super::{Cell, Handle, NanBox, Realm};
use crate::object::JIT_FLAG_READONLY;
use crate::shape::Shape;
use alloc::rc::Rc;

/// A fixed hidden-state layout: a shape and the slots of `N` keys under it
/// (see the module docs).
pub(crate) struct HiddenLayout<const N: usize> {
    shape: Rc<Shape>,
    slots: [u32; N],
}

impl<const N: usize> HiddenLayout<N> {
    /// The layout of the object at `handle` for `keys`, provided it is a
    /// shaped ordinary object holding every key in `keys` and none of
    /// `absent` (keys whose presence changes the object's meaning, such as a
    /// "done" marker, so the layout also answers "absent" for them).
    pub(crate) fn learn(
        realm: &Realm,
        handle: Handle,
        keys: [&str; N],
        absent: &[&str],
    ) -> Option<Self> {
        let Some(Cell::Object(o)) = realm.heap.get(handle) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        if absent.iter().any(|k| shape.contains(k)) {
            return None;
        }
        let mut slots = [0u32; N];
        for (slot, key) in slots.iter_mut().zip(keys) {
            *slot = shape.lookup(key)?;
        }
        Some(Self {
            shape: Rc::clone(shape),
            slots,
        })
    }

    /// The values of the layout's keys on the object at `handle`, in `keys`
    /// order, when it has exactly the layout's shape and its slots may be
    /// rewritten directly (not frozen, no read-only property). `None`
    /// otherwise.
    pub(crate) fn read(&self, realm: &Realm, handle: Handle) -> Option<[NanBox; N]> {
        let Some(Cell::Object(o)) = realm.heap.get(handle) else {
            return None;
        };
        if !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, &self.shape))
            || o.is_frozen()
            || o.jit_flags() & JIT_FLAG_READONLY != 0
        {
            return None;
        }
        let mut out = [NanBox::undefined(); N];
        for (v, &slot) in out.iter_mut().zip(&self.slots) {
            *v = o.slot(slot)?;
        }
        Some(out)
    }

    /// Writes `value` to the layout's key `i` on the object at `handle` — what
    /// [`Realm::set_hidden_property`] does for that existing key. Only for an
    /// object [`read`](Self::read) just accepted, with nothing in between that
    /// could have changed its shape or flags.
    pub(crate) fn write(&self, realm: &mut Realm, handle: Handle, i: usize, value: NanBox) {
        if let Some(Cell::Object(o)) = realm.heap.get_mut(handle) {
            o.set_slot(self.slots[i], value);
        }
        realm.write_barrier(handle, value);
    }
}
