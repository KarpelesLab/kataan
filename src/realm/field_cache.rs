//! The transition cache of a class-field definition site (`Op::DefineData`
//! with the field attributes, `Op::PrivateDefine` of a private field).
//!
//! A constructor adds its instance fields to objects of one layout in one
//! order, so each field's define sees one incoming layout and makes one new
//! one. A [`FieldCache`] remembers that step: the layout made (whose parent is
//! the incoming one) and the key, compared exactly, since one site runs with
//! another class evaluation's private name or computed key when a class is
//! evaluated more than once. A hit is the step the general path takes for a
//! plain add of an absent key to an ordinary extensible object: the new layout,
//! the value pushed into the new slot, the key epoch bumped — and for a
//! private field, the key marked non-enumerable as `set_hidden_property` marks
//! it. Anything else (another layout or key, a dictionary-mode, non-extensible
//! or watched-write object, an accessor or attribute mark under the key, an
//! insertion-order log) is a miss for the caller's general path, which keeps
//! every check and error.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::rc::Rc;

/// One field-definition site's memo (see the module docs). Empty when cold,
/// and when copied (a fresh copy of the bytecode starts cold).
#[derive(Default)]
pub struct FieldCache(core::cell::Cell<Option<Box<FieldEntry>>>);

struct FieldEntry {
    /// The layout the add made; its parent is the incoming one.
    to: Rc<Shape>,
    /// The key, as its text (always valid UTF-8).
    key: Box<str>,
}

impl Clone for FieldCache {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl core::fmt::Debug for FieldCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FieldCache")
    }
}

impl Realm {
    /// The define of `key` (the string `k`) with value `v` on the object `o`
    /// through `cache` (see the module docs): `true` when it hit and the
    /// property (for `private`, the private field) was added; `false`,
    /// having changed nothing, otherwise.
    pub(crate) fn field_cached_add(
        &mut self,
        o: NanBox,
        k: NanBox,
        v: NanBox,
        cache: &FieldCache,
        private: bool,
    ) -> bool {
        let (Some(raw), Some(kraw)) = (o.as_handle(), k.as_handle()) else {
            return false;
        };
        let memo = cache.0.take();
        let Some(e) = &memo else {
            return false;
        };
        let h = Handle::from_raw(raw);
        // The key first: the leaf's bytes, compared with the memo's.
        let same_key = self
            .string_leaf_bytes(Handle::from_raw(kraw))
            .is_some_and(|b| b == e.key.as_bytes());
        let dict_threshold = self.limits.object_dictionary_threshold;
        let args_marker = !private && self.args_cells_attached;
        let hit = same_key
            && !args_marker
            && match self.heap.get_mut(h) {
                Some(Cell::Object(obj)) => {
                    obj.field_add_by_transition(&e.to, &e.key, v, dict_threshold, private)
                }
                _ => false,
            };
        cache.0.set(memo);
        if hit {
            self.write_barrier(h, v);
        }
        hit
    }

    /// The layout of the ordinary object `o`, when it has one (for
    /// [`field_arm`](Self::field_arm) after a general-path define).
    #[must_use]
    pub(crate) fn field_layout(&self, o: NanBox) -> Option<Rc<Shape>> {
        match self.heap.get(Handle::from_raw(o.as_handle()?)) {
            Some(Cell::Object(obj)) => obj.shaped_shape().cloned(),
            _ => None,
        }
    }

    /// Arms `cache` after a general-path define of `key` on `o` that took it
    /// from the layout `before` to a child layout adding exactly `key`.
    pub(crate) fn field_arm(
        &self,
        o: NanBox,
        before: Option<Rc<Shape>>,
        key: &str,
        cache: &FieldCache,
    ) {
        let Some(before) = before else {
            return;
        };
        if let Some(to) = self.field_layout(o)
            && to.parent().is_some_and(|p| Rc::ptr_eq(p, &before))
            && to.last_key() == Some(key)
        {
            cache.0.set(Some(Box::new(FieldEntry {
                to,
                key: key.into(),
            })));
        }
    }
}
