//! The own enumerable data properties of a plain object, read in one pass.
//!
//! Object spread (`{ ...src, k: v }`) and object rest run
//! `CopyDataProperties`, whose general path (`Interp::copy_data_properties`)
//! asks `[[GetOwnProperty]]` for each key by building a fresh descriptor
//! object (four property writes) only to read its `enumerable` back, and then
//! reads the value through the full member read: about 16,000 instructions
//! to spread a three-property object, so an immutable-update reducer
//! (`state = { ...state, n: state.n + 1 }`) spent nearly all its time there.
//!
//! For an ordinary object cell with no accessor properties and no internal
//! (`\0`-prefixed) keys — those mark wrappers, `arguments` objects and other
//! exotic behaviour, and hold the symbol-keyed properties — every own key is a
//! plain data property: its `[[GetOwnProperty]]` runs no code, its
//! `[[Enumerable]]` is the slot's own flag and its value is the slot's value.
//! [`Realm::plain_data_entries`] reads exactly those, in `[[OwnPropertyKeys]]`
//! order (integer keys ascending, then insertion order); anything else answers
//! `None` and the caller takes the general path.
//!
//! Copying key by key is still O(n^2) for an n-key source: each add looks
//! its key up in the target's growing layout. When the target is empty (an
//! object literal that starts with the spread, or a rest pattern's object)
//! and the source's own properties are all enumerable data properties under
//! copyable keys (names or Symbols), [`Realm::spread_plain`] gives the target
//! the source's layout and values in one step (a dictionary-mode source: a
//! copy of its dictionary, sharing the keys), leaving out a rest pattern's
//! excluded keys. The target's key order is then the source's, which
//! `[[OwnPropertyKeys]]` re-sorts the same way for both. `Object.values` of
//! such an object reads its slots the same way
//! ([`Realm::plain_enum_values`]).

use super::{Cell, Handle, NanBox, Realm};
use alloc::string::String;
use alloc::vec::Vec;

impl Realm {
    /// `CopyDataProperties(target, src, excluded)` when it is a wholesale
    /// copy (see the module docs): `target` an empty ordinary object (an
    /// object literal's or a rest pattern's own fresh object) and `src` an
    /// ordinary object whose own properties are all enumerable data
    /// properties. `target` then takes `src`'s layout (or a copy of its
    /// dictionary) and values in one step, instead of re-adding the keys one
    /// at a time: each add looks its key up in the growing layout first, so
    /// a key-by-key copy of an n-key object was O(n^2). Performed, returning
    /// `true`; `false`, having changed nothing, for any other pair.
    pub(crate) fn spread_plain(
        &mut self,
        target: Handle,
        src: Handle,
        excluded: &[String],
    ) -> bool {
        if self.is_host_exotic(src) || self.is_function_proto_intrinsic(src) {
            return false;
        }
        let Some(Cell::Object(s)) = self.heap.get(src) else {
            return false;
        };
        let Some(copy) = s.plain_copy(excluded) else {
            return false;
        };
        if self.is_host_exotic(target) {
            return false;
        }
        let threshold = self.limits.object_dictionary_threshold;
        let Some(Cell::Object(t)) = self.heap.get_mut(target) else {
            return false;
        };
        if copy.is_empty() {
            return true;
        }
        let mut vals: Vec<NanBox> = Vec::new();
        copy.for_each_value(|v| {
            if v.as_handle().is_some() {
                vals.push(v);
            }
        });
        if t.adopt_plain(copy, threshold).is_err() {
            return false;
        }
        for v in vals {
            self.write_barrier(target, v);
        }
        true
    }

    /// `Object.values(h)` when `h` is an ordinary object no step of whose
    /// `EnumerableOwnProperties` can run code (see
    /// [`Object::plain_enum_values`](crate::object::Object::plain_enum_values)).
    #[must_use]
    pub(crate) fn plain_enum_values(&self, h: Handle) -> Option<Vec<NanBox>> {
        if self.is_function_proto_intrinsic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => o.plain_enum_values(),
            _ => None,
        }
    }

    /// The own enumerable `(key, value)` pairs of `h`, in `[[OwnPropertyKeys]]`
    /// order, when `h` is an ordinary object whose own properties are all
    /// plain string-keyed data properties (see the module docs).
    pub(crate) fn plain_data_entries(&self, h: Handle) -> Option<Vec<(String, NanBox)>> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if !o.accessor_keys().is_empty() {
            return None;
        }
        let keys = o.ordered_keys();
        let mut out = Vec::with_capacity(keys.len());
        for k in keys {
            if k.as_bytes().first() == Some(&0) {
                return None;
            }
            if o.is_hidden(k) {
                continue;
            }
            out.push((String::from(k), o.get(k)?));
        }
        Some(out)
    }
}
