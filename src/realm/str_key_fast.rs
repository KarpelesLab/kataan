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
    /// (A mapped `arguments` object owns its index keys like any other, so
    /// only a value read, `args_too`, has to leave it to the general path.)
    #[inline]
    fn ordinary_props(&self, h: Handle, args_too: bool) -> Option<&Object> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if o.is_prim_wrapper()
            || self.is_host_exotic(h)
            || (args_too && self.has_argument_map_marker(h))
        {
            return None;
        }
        Some(o)
    }

    /// An ordinary object's key epoch and `[[Prototype]]`, without the
    /// checks of [`ordinary_link`](Self::ordinary_link) (for revalidating a
    /// link that passed them, whose key epoch fixes its keys).
    #[must_use]
    #[inline]
    pub(crate) fn link_epoch(&self, h: Handle) -> Option<(u32, Option<Handle>)> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if self.is_host_exotic(h) {
            return None;
        }
        Some((o.key_epoch(), o.proto()))
    }

    /// A prototype-chain link [`has_property_named`](Self::has_property_named)
    /// walks: its key epoch and `[[Prototype]]`, or `None` when it is not
    /// one it decides for.
    #[must_use]
    pub(crate) fn ordinary_link(&self, h: Handle) -> Option<(u32, Option<Handle>)> {
        let o = self.ordinary_props(h, false)?;
        Some((o.key_epoch(), o.proto()))
    }

    /// `obj[key]` for a `for-in` key String `key` on an object of its key
    /// list's layout (`forin_cache`): its slot's value. `None` for anything
    /// else (what [`own_data_by_string`](Self::own_data_by_string) answers
    /// first).
    #[must_use]
    #[inline]
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn for_in_key_value(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        self.for_in_key_slot(obj, key)
            .and_then(|(o, slot)| o.slot(slot))
    }

    /// `obj[key]` for a String `key` naming an own **data** property of the
    /// ordinary object `obj` (no like-named accessor): its value. `None`
    /// for anything else.
    #[must_use]
    pub(crate) fn own_data_by_string(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        let k = self.string_key_in_place(key)?;
        let o = self.ordinary_props(Handle::from_raw(obj.as_handle()?), true)?;
        if o.accessor(k).is_some() {
            return None;
        }
        o.get(k)
    }

    /// The ordinary object `h`'s own **data** property `key` (no like-named
    /// accessor): what `[[Get]]` returns there without running anything.
    /// `None` for anything else.
    #[must_use]
    pub(crate) fn own_data_named(&self, h: Handle, key: &str) -> Option<NanBox> {
        let Some(Cell::Object(o)) = self.heap.get(h) else {
            return None;
        };
        if self.is_host_exotic(h) || o.accessor(key).is_some() {
            return None;
        }
        o.get(key)
    }

    /// `key in obj` (`[[HasProperty]]`) for a String `key`, when every object
    /// on `obj`'s prototype chain up to the answer is an ordinary one (see
    /// the module docs). `None` when that cannot be told here.
    #[must_use]
    pub(crate) fn has_property_by_string(&self, key: NanBox, obj: NanBox) -> Option<bool> {
        let k = self.string_key_in_place(key)?;
        self.has_property_named(Handle::from_raw(obj.as_handle()?), k)
    }

    /// [`has_property_by_string`](Self::has_property_by_string) where the
    /// key is likely a `for-in`'s (the loop's per-key liveness check): a key
    /// String of its list on an object of the list's layout is an own key
    /// (`forin_cache`), tried first.
    #[must_use]
    #[cfg_attr(
        not(all(feature = "jit", target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn has_property_by_for_in_key(&self, key: NanBox, obj: NanBox) -> Option<bool> {
        if self.for_in_key_slot(obj, key).is_some() {
            return Some(true);
        }
        self.has_property_by_string(key, obj)
    }

    /// `HasProperty(h, key)` for a storage key `key`, when every object on
    /// `h`'s prototype chain up to the answer is an ordinary one (see the
    /// module docs). `None` when that cannot be told here.
    #[must_use]
    pub(crate) fn has_property_named(&self, h: Handle, k: &str) -> Option<bool> {
        let mut cur = Some(h);
        // (An ordinary chain has no cycle; the bound is a guard.)
        for _ in 0..64 {
            let Some(h) = cur else {
                return Some(false);
            };
            let o = self.ordinary_props(h, false)?;
            if o.has_own_key(k) {
                return Some(true);
            }
            cur = o.proto();
        }
        None
    }
}
