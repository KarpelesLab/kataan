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

use super::{Cell, Handle, NanBox, Realm};
use alloc::string::String;
use alloc::vec::Vec;

impl Realm {
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
