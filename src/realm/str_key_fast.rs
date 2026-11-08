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
        if let Some((o, slot)) = self.for_in_key_slot(obj, key) {
            return o.slot(slot);
        }
        // An own key of a dictionary receiver's memoized list: its value,
        // unless it is an accessor.
        let (o, name) = self.for_in_dict_own_key(obj, key)?;
        if o.accessor(name).is_some() {
            return None;
        }
        o.get(name)
    }

    /// `obj[key] = value` for a String `key` naming an existing own data
    /// property of the ordinary shaped object `obj` whose write is just its
    /// slot's store (`Object::plain_slot_store_ok`: writable, no accessor,
    /// nothing the host must see): that store, with its write barrier, and
    /// `true`. `false`, having done nothing, otherwise (an add included).
    /// The key's slot comes from the String-key slot memo, which a miss here
    /// feeds as a read's does.
    #[inline]
    pub(crate) fn own_data_set_by_string(
        &mut self,
        obj: NanBox,
        key: NanBox,
        value: NanBox,
    ) -> bool {
        let slot = match self.key_memo_probe(obj, key) {
            Some((o, slot)) => o.plain_slot_store_ok().then_some(slot),
            None => self.own_data_set_missed(obj, key),
        };
        let (Some(slot), Some(raw)) = (slot, obj.as_handle()) else {
            return false;
        };
        let h = Handle::from_raw(raw);
        let wrote = self
            .heap
            .get_mut(h)
            .and_then(Cell::as_object_mut)
            .is_some_and(|o| o.set_slot(slot, value));
        if wrote {
            self.write_barrier(h, value);
        }
        wrote
    }

    /// [`own_data_set_by_string`](Self::own_data_set_by_string)'s slot once
    /// the memo missed (noted for the next write or read).
    #[inline(never)]
    fn own_data_set_missed(&self, obj: NanBox, key: NanBox) -> Option<u32> {
        let k = self.string_key_in_place(key)?;
        // (A mapped `arguments` object's index write aliases a parameter.)
        let o = self.ordinary_props(Handle::from_raw(obj.as_handle()?), true)?;
        if !o.plain_slot_store_ok() {
            return None;
        }
        let shape = o.shaped_shape()?;
        let slot = shape.lookup(k)?;
        self.key_memo_note(key, shape, slot);
        Some(slot)
    }

    /// `obj[key]` for a String `key` naming an own **data** property of the
    /// ordinary object `obj` (no like-named accessor): its value. `None`
    /// for anything else.
    #[must_use]
    pub(crate) fn own_data_by_string(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        match self.key_memo_probe(obj, key) {
            Some((o, slot)) => o.slot(slot),
            None => self.own_data_missed(obj, key),
        }
    }

    /// [`own_data_by_string`](Self::own_data_by_string) once the memo
    /// missed.
    #[must_use]
    #[inline(never)]
    pub(crate) fn own_data_missed(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        let k = self.string_key_in_place(key)?;
        let o = self.ordinary_props(Handle::from_raw(obj.as_handle()?), true)?;
        if o.accessor(k).is_some() {
            return None;
        }
        // A shaped object's slot of `k` is memoized for the next read
        // (`key_memo`): its layout is no wrapper's or mapped `arguments`
        // object's, as `ordinary_props` checked.
        if let Some(shape) = o.shaped_shape() {
            let slot = shape.lookup(k)?;
            if o.has_no_accessors() {
                self.key_memo_note(key, shape, slot);
            }
            return o.slot(slot);
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

    /// The array index a String key names, when its text is that index's
    /// canonical form (`"0"`, `"17"`; not `"01"`, `"-1"`, `"1.0"` or
    /// `"4294967295"`): the property key the Number index would convert to.
    #[must_use]
    pub(crate) fn string_array_index(&self, key: NanBox) -> Option<usize> {
        let b = self
            .heap
            .get(Handle::from_raw(key.as_handle()?))?
            .as_str()?
            .as_leaf_bytes()?;
        if b.is_empty() || b.len() > 10 || (b.len() > 1 && b[0] == b'0') {
            return None;
        }
        let mut n: u64 = 0;
        for &c in b {
            if !c.is_ascii_digit() {
                return None;
            }
            n = n * 10 + u64::from(c - b'0');
        }
        (n < u64::from(u32::MAX)).then_some(n as usize)
    }

    /// `obj[key]` for a key (a Number, or a String in an array index's
    /// canonical form) naming a present element of a plain dense array
    /// (`plain_array_get`): the element. `None` for anything else.
    #[must_use]
    #[inline]
    pub(crate) fn plain_array_get_key(&self, obj: NanBox, key: NanBox) -> Option<NanBox> {
        let h = Handle::from_raw(obj.as_handle()?);
        // (The receiver first: most keys reach here for other objects.)
        if !matches!(self.heap.get(h), Some(Cell::Array(_))) {
            return None;
        }
        let n = match key.as_number() {
            Some(n) => n,
            None => self.string_array_index(key)? as f64,
        };
        self.plain_array_get(h, n)
    }

    /// `key in obj` (`[[HasProperty]]`) for a String `key`, when every object
    /// on `obj`'s prototype chain up to the answer is an ordinary one (see
    /// the module docs). `None` when that cannot be told here.
    #[must_use]
    pub(crate) fn has_property_by_string(&self, key: NanBox, obj: NanBox) -> Option<bool> {
        if self.key_memo_probe(obj, key).is_some() {
            return Some(true);
        }
        let k = self.string_key_in_place(key)?;
        let h = Handle::from_raw(obj.as_handle()?);
        // The receiver itself (`has_property_named`'s first link): an own
        // data slot is memoized, as `own_data_by_string` does. (A present
        // element of a plain dense array is an own property.)
        let Some(o) = self.ordinary_props(h, false) else {
            return self.plain_array_get_key(obj, key).map(|_| true);
        };
        if let Some(shape) = o.shaped_shape() {
            if let Some(slot) = shape.lookup(k) {
                if o.has_no_accessors() && !shape.is_wrapper() && !self.has_argument_map_marker(h) {
                    self.key_memo_note(key, shape, slot);
                }
                return Some(true);
            }
            if o.accessor(k).is_some() {
                return Some(true);
            }
        } else if o.has_own_key(k) {
            return Some(true);
        }
        match o.proto() {
            Some(p) => self.has_property_named(p, k),
            None => Some(false),
        }
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
        if self.for_in_key_slot(obj, key).is_some() || self.for_in_dict_own_key(obj, key).is_some()
        {
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
