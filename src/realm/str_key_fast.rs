//! Computed-key operations decided from the heap alone, for a String key.
//!
//! `o[k]` and `k in o` with a String `k` convert the key to its storage form
//! (`realm::user_key_wtf8`), an owned copy, before looking it up. A flat
//! leaf holding well-formed UTF-8 that does not start with NUL is already
//! its own storage key, so these read it in place. They answer only for
//! ordinary objects whose own properties are exactly their slots and
//! accessors (no proxy, host-exotic object, primitive wrapper or mapped
//! `arguments` object), which is what the general paths would read there
//! too; anything else is `None` and goes the general way.

use super::Realm;
use crate::cell::Cell;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::object::Object;

impl Realm {
    /// The storage key of the String `v`, borrowed: `None` unless it is a
    /// flat well-formed UTF-8 leaf that needs no escape (see the module
    /// docs).
    #[must_use]
    #[inline]
    pub(crate) fn string_key_in_place(&self, v: NanBox) -> Option<&str> {
        let raw = v.as_handle()?;
        let b = self
            .heap
            .get(Handle::from_raw(raw))?
            .as_str()?
            .as_leaf_bytes()?;
        // (A string cell with an auxiliary object may carry a function
        // marker, which changes its `typeof`: the general path decides.)
        if b.first() == Some(&0) || self.aux_props.contains_key(&raw) {
            return None;
        }
        core::str::from_utf8(b).ok()
    }

    /// The object at `h` when its own properties are exactly its slots and
    /// accessors (see the module docs).
    #[inline]
    fn ordinary_props(&self, h: Handle) -> Option<&Object> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if o.is_prim_wrapper() || self.is_host_exotic(h) || self.has_argument_map_marker(h) {
            return None;
        }
        Some(o)
    }

    /// `obj[key]` for a String `key` naming an own **data** property of the
    /// ordinary object `obj` (no like-named accessor): its value. `None`
    /// for anything else.
    #[must_use]
    pub(crate) fn own_data_by_string(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        let k = self.string_key_in_place(key)?;
        let o = self.ordinary_props(Handle::from_raw(obj.as_handle()?))?;
        if o.accessor(k).is_some() {
            return None;
        }
        o.get(k)
    }

    /// `key in obj` (`[[HasProperty]]`) for a String `key`, when every object
    /// on `obj`'s prototype chain up to the answer is an ordinary one (see
    /// the module docs). `None` when that cannot be told here.
    #[must_use]
    pub(crate) fn has_property_by_string(&self, key: NanBox, obj: NanBox) -> Option<bool> {
        let k = self.string_key_in_place(key)?;
        let mut cur = Some(Handle::from_raw(obj.as_handle()?));
        // (An ordinary chain has no cycle; the bound is a guard.)
        for _ in 0..64 {
            let Some(h) = cur else {
                return Some(false);
            };
            let o = self.ordinary_props(h)?;
            if o.has_own_key(k) {
                return Some(true);
            }
            cur = o.proto();
        }
        None
    }
}
