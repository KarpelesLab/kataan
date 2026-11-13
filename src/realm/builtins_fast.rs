//! Narrow, read-only views of the heap for the built-in library's fast paths
//! (`JSON.stringify` and friends): each answers one question exactly or says
//! "don't know", so a caller can skip the generic `[[Get]]` machinery only when
//! the answer is provably the same.

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

/// A layout's `Object.keys` Strings and each key's slot (see
/// [`Realm::plain_enumerable_key_slots`]).
pub(crate) type KeySlots = (Rc<[NanBox]>, Rc<[u32]>);

impl Realm {
    /// The shape of the object at `handle` when its own properties are exactly
    /// that shape's data keys, all enumerable, in shape order (see
    /// [`Object::keys_follow_shape`](crate::object::Object::keys_follow_shape)),
    /// and no host hook intercepts its reads (it is not host-exotic). `None`
    /// for anything else.
    #[must_use]
    pub fn plain_keyed_shape(&self, handle: Handle) -> Option<&Rc<Shape>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if !o.keys_follow_shape() || self.is_host_exotic(handle) {
            return None;
        }
        o.shaped_shape()
    }

    /// `EnumerableOwnProperties(O, key)` for an object whose own properties are
    /// all plain enumerable data properties (see
    /// [`plain_keyed_shape`](Self::plain_keyed_shape)) and which is no primitive
    /// wrapper (a String wrapper owns its indices exotically): its string keys
    /// in property order. `None` for anything else.
    #[must_use]
    pub fn plain_enumerable_keys(
        &self,
        handle: Handle,
    ) -> Option<alloc::vec::Vec<alloc::string::String>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.is_prim_wrapper() {
            return None;
        }
        self.plain_keyed_shape(handle)?;
        self.object_keys(handle)
    }

    /// [`plain_enumerable_keys`](Self::plain_enumerable_keys) as shared ropes.
    /// For such an object the key list is a function of its shape alone, so
    /// the last shape's list is remembered: listing the keys of objects built
    /// alike (a loop over records) then allocates nothing but the result's
    /// string cells, which share these immutable ropes.
    #[must_use]
    pub fn plain_enumerable_key_ropes(&self, handle: Handle) -> Option<Rc<[crate::rope::Rope]>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.is_prim_wrapper() {
            return None;
        }
        let shape = self.plain_keyed_shape(handle)?;
        if let Some((s, keys)) = &*self.key_ropes.borrow()
            && Rc::ptr_eq(s, shape)
        {
            return Some(keys.clone());
        }
        let shape = shape.clone();
        let names = self.object_keys(handle)?;
        // An escaped user key (`"\0\0…"`, see `realm::is_internal_key`) is not
        // its own text: leave such an object to the general path.
        if names.iter().any(|k| k.starts_with('\0')) {
            return None;
        }
        let keys: Rc<[crate::rope::Rope]> =
            names.iter().map(|k| crate::rope::Rope::leaf(k)).collect();
        *self.key_ropes.borrow_mut() = Some((shape, keys.clone()));
        Some(keys)
    }

    /// [`plain_enumerable_key_ropes`](Self::plain_enumerable_key_ropes) as
    /// String values, for `Object.keys`: the same cells for every object of
    /// a layout, remembered per layout (a String is immutable and its
    /// identity is not observable), so listing the keys of objects built
    /// alike allocates only the result array, and a read `o[k]` of a listed
    /// key sees the same key String every time (which the String-key slot
    /// memo, `key_memo`, then remembers).
    pub fn plain_enumerable_key_strings(&mut self, handle: Handle) -> Option<Rc<[NanBox]>> {
        self.plain_enumerable_key_slots(handle).map(|(ks, _)| ks)
    }

    /// [`plain_enumerable_key_strings`](Self::plain_enumerable_key_strings)
    /// with each key's slot in the object's layout, remembered with them: so
    /// `Object.values` / `Object.entries` of such an object read its slots
    /// in key order (integer keys ascending first, then the rest in
    /// insertion order), with no key lookup.
    pub(crate) fn plain_enumerable_key_slots(&mut self, handle: Handle) -> Option<KeySlots> {
        // The entry is the one the layout's own hint names, confirmed by
        // its layout (as the `for-in` cache finds its entries): any `CAP`
        // layouts in use stay remembered together, whatever their addresses.
        // (A memo indexed by the layout's address let two layouts in use
        // evict each other or not depending on where they were allocated.)
        /// The number of layouts remembered.
        const CAP: usize = 16;
        let shape = Rc::clone(self.plain_keyed_shape(handle)?);
        let hinted = usize::from(shape.keys_hint()).checked_sub(1);
        if let Some(i) = hinted
            && let Some(Some((s, ks, slots))) = self.key_strings.get(i)
            && Rc::ptr_eq(s, &shape)
        {
            return Some((Rc::clone(ks), Rc::clone(slots)));
        }
        if self.is_function_proto_intrinsic(handle) {
            return None;
        }
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.is_prim_wrapper() {
            return None;
        }
        let names = self.object_keys(handle)?;
        // An escaped user key (`"\0\0…"`, see `realm::is_internal_key`) is not
        // its own text: leave such an object to the general path.
        if names.iter().any(|k| k.starts_with('\0')) {
            return None;
        }
        let slots: Rc<[u32]> = names
            .iter()
            .map(|k| shape.lookup(k))
            .collect::<Option<_>>()?;
        let ks: Rc<[NanBox]> = names
            .iter()
            .map(|k| NanBox::handle(self.new_string(k).to_raw()))
            .collect();
        if self.key_strings.is_empty() {
            self.key_strings.resize(CAP, None);
        }
        // An empty entry, else the next one round.
        let i = self
            .key_strings
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                let i = self.key_strings_next % CAP;
                self.key_strings_next = (i + 1) % CAP;
                i
            });
        shape.set_keys_hint(u16::try_from(i + 1).unwrap_or(0));
        self.key_strings[i] = Some((shape, Rc::clone(&ks), Rc::clone(&slots)));
        Some((ks, slots))
    }

    /// The values of slots `slots` of the object at `handle`, in that order:
    /// with [`plain_enumerable_key_slots`](Self::plain_enumerable_key_slots)
    /// just answered for it, its `Object.values`.
    pub(crate) fn object_slot_values(&self, handle: Handle, slots: &[u32]) -> Option<Vec<NanBox>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        slots.iter().map(|&s| o.slot(s)).collect()
    }

    /// Whether `h` is an ordinary dictionary-mode object holding only plain
    /// writable, enumerable data properties
    /// ([`Object::is_plain_dict`](crate::object::Object::is_plain_dict)).
    #[must_use]
    pub(crate) fn is_plain_dict_object(&self, h: Handle) -> bool {
        matches!(self.heap.get(h), Some(Cell::Object(o)) if o.is_plain_dict())
    }

    /// `Object.keys(h)` of a plain dictionary-mode object (only data
    /// properties, which keys are not engine-internal): its key Strings,
    /// remembered per object while its key epoch is unchanged (no key added,
    /// removed or made non-enumerable since), so listing a large dictionary's
    /// keys again allocates only the result array. `None` for anything else.
    pub(crate) fn dict_enumerable_key_strings(&mut self, handle: Handle) -> Option<Rc<[NanBox]>> {
        /// The number of objects remembered (a power of two).
        const WAYS: usize = 4;
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if !o.is_dict() || self.is_host_exotic(handle) || self.is_function_proto_intrinsic(handle) {
            return None;
        }
        let epoch = o.key_epoch();
        let way = (handle.to_raw() as usize) & (WAYS - 1);
        if let Some(Some((h, e, ks))) = self.dict_key_strings.get(way)
            && *h == handle.to_raw()
            && *e == epoch
        {
            return Some(Rc::clone(ks));
        }
        let names: Vec<String> = o
            .plain_enum_entries()?
            .into_iter()
            .map(|(k, _)| String::from(k))
            .collect();
        let ks: Rc<[NanBox]> = names
            .iter()
            .map(|k| NanBox::handle(self.new_string(k).to_raw()))
            .collect();
        if self.dict_key_strings.is_empty() {
            self.dict_key_strings.resize(WAYS, None);
        }
        self.dict_key_strings[way] = Some((handle.to_raw(), epoch, Rc::clone(&ks)));
        Some(ks)
    }

    /// Slot `slot` of the object at `handle`, provided the object still has
    /// shape `shape` and still satisfies
    /// [`plain_keyed_shape`](Self::plain_keyed_shape)'s conditions — so the slot
    /// holds the data property the caller resolved under that shape.
    #[must_use]
    pub fn plain_slot_under(&self, handle: Handle, shape: &Rc<Shape>, slot: u32) -> Option<NanBox> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if !o.keys_follow_shape() || !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, shape)) {
            return None;
        }
        o.slot(slot)
    }

    /// The `[[Prototype]]` of the RegExp at `handle` when it owns no
    /// property (it has no auxiliary object) and was never explicitly
    /// linked, so it inherits from the realm's default RegExp prototype
    /// (see `link_regexp_proto`): that prototype. `None` for anything else.
    #[must_use]
    pub(crate) fn plain_regexp_default_proto(&self, handle: Handle) -> Option<Handle> {
        let raw = handle.to_raw();
        if !matches!(self.heap.get(handle), Some(Cell::RegExp { .. }))
            || self.aux_props.contains_key(&raw)
            || self.native_protos.contains_key(&raw)
            || self.has_explicit_null_proto(handle)
        {
            return None;
        }
        self.regexp_default_proto
    }

    /// The WTF-8 bytes of two string values, each flattened first (a
    /// concatenation is joined once, then served as a leaf): `None` unless
    /// both are string cells with no auxiliary object (no function marker,
    /// so `typeof` is `"string"` and a relational operator converts
    /// nothing).
    pub(crate) fn plain_string_pair(&mut self, a: Handle, b: Handle) -> Option<(&[u8], &[u8])> {
        for h in [a, b] {
            match self.heap.get(h) {
                Some(Cell::Str(r)) if r.as_leaf_bytes().is_some() => {}
                Some(Cell::Str(_)) => self.flatten_string(h),
                _ => return None,
            }
        }
        // (Only a string cell given an auxiliary object is no plain string;
        // until one is, `str_special` stays clear and no lookup is needed.)
        if self.str_special
            && (self.aux_props.contains_key(&a.to_raw())
                || self.aux_props.contains_key(&b.to_raw()))
        {
            return None;
        }
        let x = self.heap.get(a)?.as_str()?.as_leaf_bytes()?;
        let y = self.heap.get(b)?.as_str()?.as_leaf_bytes()?;
        Some((x, y))
    }

    /// Own **data** property `key` of the ordinary (not host-exotic) object
    /// at `handle`, with where [`own_data_reread`](Self::own_data_reread)
    /// finds it again while the object's key epoch is unchanged: its slot,
    /// or `u32::MAX` in dictionary mode. `None` for an accessor, an absent
    /// key or any other cell.
    #[must_use]
    pub(crate) fn own_data_probe(&self, handle: Handle, key: &str) -> Option<(u32, NanBox)> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        if o.accessor(key).is_some() || self.is_host_exotic(handle) {
            return None;
        }
        if o.is_dict() {
            return Some((u32::MAX, o.get(key)?));
        }
        let slot = o.shape().lookup(key)?;
        Some((slot, o.slot(slot)?))
    }

    /// Whether, for each `i` with bit `i` of `mask` set, the data property
    /// `keys[i]` that [`own_data_probe`](Self::own_data_probe) found at
    /// `at[i].0` still holds the value with bits `at[i].1` — for a caller that
    /// has checked the object's key epoch (so its layout) is the same.
    #[must_use]
    pub(crate) fn own_data_all_hold(
        &self,
        handle: Handle,
        at: &[(u32, u64)],
        keys: &[alloc::rc::Rc<str>],
        mask: u32,
    ) -> bool {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return false;
        };
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            let (Some(&(slot, bits)), Some(key)) = (at.get(i), keys.get(i)) else {
                continue;
            };
            let v = if slot == u32::MAX {
                o.get(key)
            } else {
                o.slot(slot)
            };
            if v.is_none_or(|v| v.to_bits() != bits) {
                return false;
            }
        }
        true
    }

    /// The flags of the RegExp at `handle`, borrowed (no source rendering).
    #[must_use]
    pub fn regexp_flags(&self, handle: Handle) -> Option<&str> {
        match self.heap.get(handle)? {
            Cell::RegExp { flags, .. } => Some(flags),
            _ => None,
        }
    }

    /// `CreateDataProperty(handle, key, value)` for a named key on a fresh
    /// array (or other cell whose named properties live in an auxiliary
    /// object), through `cache` as for an object literal's own properties
    /// ([`init_data_property_cached`](Self::init_data_property_cached)), so a
    /// site that always adds the same keys reuses one shape path. Falls back to
    /// [`set_property`](Self::set_property).
    pub fn init_aux_data_cached(
        &mut self,
        handle: Handle,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) {
        if self.aux_eligible(handle) {
            let aux = self.aux_object(handle);
            if self.init_data_property_cached(aux, key, value, cache) {
                return;
            }
        }
        self.set_property(handle, key, value);
    }

    /// A fresh array holding `elems` whose named own properties are `keys`
    /// (plain data properties, added in that order) with `values` — what
    /// `new_array` followed by [`init_aux_data_cached`](Self::init_aux_data_cached)
    /// of each key leaves. Once `cache` has seen the auxiliary object's
    /// layout, later arrays get theirs in one step: the object is created
    /// with that shape and its slots already filled, rather than walking the
    /// transitions one key at a time.
    pub fn new_array_with_named(
        &mut self,
        elems: alloc::vec::Vec<NanBox>,
        keys: &[&str],
        values: &[NanBox],
        cache: &mut AuxShapeCache,
    ) -> Handle {
        let arr = self.new_array(elems);
        if let Some((root, shape)) = &cache.0
            && Rc::ptr_eq(root, &self.root_shape)
            && values.len() <= self.limits.object_dictionary_threshold
            && let Some(mut obj) = crate::object::Object::with_data(Rc::clone(shape), values)
        {
            // As `new_object` would: the default `[[Prototype]]`, shaded for
            // an incremental cycle, then each stored value through the
            // barrier as `init_data_property_cached` would.
            let proto = self.default_object_proto;
            obj.set_proto(proto);
            let aux = self.heap.alloc(Cell::Object(obj));
            if let (Some(p), Some(marker)) = (proto, self.incremental.as_mut()) {
                marker.mark_grey(p);
            }
            for v in values {
                self.write_barrier(aux, *v);
            }
            self.cell_link_changed(arr);
            self.aux_props.insert(arr.to_raw(), aux);
            self.mark_array_side_state(arr);
            return arr;
        }
        let mut ic = crate::ic::PropertyCache::default();
        for (k, v) in keys.iter().zip(values) {
            self.init_aux_data_cached(arr, k, *v, &mut ic);
        }
        if let Some(&aux) = self.aux_props.get(&arr.to_raw())
            && let Some(Cell::Object(o)) = self.heap.get(aux)
            && o.is_plain_shaped()
            && o.len() as usize == values.len()
            && o.proto() == self.default_object_proto
        {
            cache.0 = Some((Rc::clone(&self.root_shape), Rc::clone(o.shape())));
        }
        arr
    }

    /// A fresh ordinary object of shape `shape` (grown from this realm's root
    /// shape `root` by plain data adds) holding `values` in slot order —
    /// what [`new_object`](Self::new_object) followed by an
    /// [`init_data_property_cached`](Self::init_data_property_cached) of each
    /// of the shape's keys leaves (an object literal's whole build, see
    /// `nbvm::objlit`). `None`, having allocated nothing, when `root` is not
    /// this realm's root shape or the shape does not fit the values.
    pub fn new_object_in_shape(
        &mut self,
        root: &Rc<Shape>,
        shape: &Rc<Shape>,
        values: &[NanBox],
    ) -> Option<Handle> {
        if !Rc::ptr_eq(root, &self.root_shape)
            || values.len() >= self.limits.object_dictionary_threshold
        {
            return None;
        }
        let mut obj = crate::object::Object::with_data(Rc::clone(shape), values)?;
        // As `new_object` would: the default `[[Prototype]]`, shaded for an
        // incremental cycle, then each stored value through the barrier as
        // `init_data_property_cached` would.
        let proto = self.default_object_proto;
        obj.set_proto(proto);
        let h = self.heap.alloc(Cell::Object(obj));
        if let (Some(p), Some(marker)) = (proto, self.incremental.as_mut()) {
            marker.mark_grey(p);
        }
        for v in values {
            self.write_barrier(h, *v);
        }
        Some(h)
    }

    /// The shape of the object at `handle` when it is a plain shaped data
    /// object with the default `[[Prototype]]` whose `i`th key is `keys[i]`
    /// in slot `i` — one [`new_object_in_shape`](Self::new_object_in_shape)
    /// can remake from its keys' values.
    #[must_use]
    pub fn plain_literal_shape(
        &self,
        handle: Handle,
        keys: &[alloc::string::String],
    ) -> Option<Rc<Shape>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        (o.is_plain_fresh_data()
            && o.len() as usize == keys.len()
            && o.proto() == self.default_object_proto
            && keys
                .iter()
                .enumerate()
                .all(|(i, k)| shape.lookup(k) == Some(i as u32)))
        .then(|| Rc::clone(shape))
    }

    /// The auxiliary object holding the named own properties of the VM
    /// closure at `handle` (a [`Cell::VmFunction`]), when it has one and `key`
    /// is an ordinary key there: not one of the keys a function synthesizes
    /// or keeps elsewhere (`name`/`length` from its definition, `prototype`,
    /// the legacy `caller`/`arguments`), nor an internal (`\0`) key, and the
    /// closure is not host-exotic. On such a key the function's own property
    /// is exactly the auxiliary object's.
    fn vm_fn_plain_aux(&self, handle: Handle, key: &str) -> Option<Handle> {
        if matches!(
            key,
            "name" | "length" | "prototype" | "caller" | "arguments"
        ) || key.starts_with('\0')
        {
            return None;
        }
        let Some(Cell::VmFunction { .. }) = self.heap.get(handle) else {
            return None;
        };
        if self.is_host_exotic(handle) {
            return None;
        }
        self.aux_props.get(&handle.to_raw()).copied()
    }

    /// `[[Get]]` of `key` on the VM closure at `handle` when it is an own
    /// data property in the closure's auxiliary object (see
    /// `vm_fn_plain_aux`), through the site's inline
    /// cache; `None` (the caller's general path) otherwise.
    #[must_use]
    pub fn vm_fn_cached_get(
        &self,
        handle: Handle,
        key: &str,
        cache: &mut crate::ic::PropertyCache,
    ) -> Option<NanBox> {
        let aux = self.vm_fn_plain_aux(handle, key)?;
        let o = self.heap.get(aux)?.as_object()?;
        if !o.has_no_accessors() && o.accessor(key).is_some() {
            return None;
        }
        o.cached_get(key, cache)
    }

    /// `[[Set]]` of `key` on the VM closure at `handle` when `key` is an
    /// existing writable own data property in the closure's auxiliary object
    /// (see `vm_fn_plain_aux`): an in-place store
    /// through the site's inline cache (which declines a frozen object, a
    /// read-only key and a dictionary), with the write barrier — what the
    /// general path's `set_property` does for such a key, since an own data
    /// property shadows anything on the prototype chain. `false`, having
    /// changed nothing, otherwise.
    pub fn vm_fn_cached_set(
        &mut self,
        handle: Handle,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) -> bool {
        let Some(aux) = self.vm_fn_plain_aux(handle, key) else {
            return false;
        };
        let Some(o) = self.heap.get_mut(aux).and_then(Cell::as_object_mut) else {
            return false;
        };
        if (!o.has_no_accessors() && o.accessor(key).is_some()) || !o.cached_set(key, value, cache)
        {
            return false;
        }
        self.write_barrier(aux, value);
        true
    }

    /// `[[Set]]` of `key` on the VM closure at `handle` when it adds a new
    /// plain data property: `key` is an ordinary key (see
    /// `vm_fn_plain_aux`) the closure does not own,
    /// the closure is extensible, and the site's prototype-chain cache proves
    /// that no object on its prototype chain has `key` (so no setter,
    /// read-only property or proxy can intercept the write). Then the
    /// property is added as `set_property` adds it (through the site's
    /// transition cache when the auxiliary object exists: a fresh writable,
    /// enumerable, configurable data property is the same whether made by
    /// `[[Set]]` or `CreateDataProperty`). `false`, having changed nothing but
    /// the site's chain cache, otherwise.
    pub fn vm_fn_cached_add(
        &mut self,
        handle: Handle,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) -> bool {
        if matches!(
            key,
            "name" | "length" | "prototype" | "caller" | "arguments"
        ) || key.starts_with('\0')
        {
            return false;
        }
        let Some(Cell::VmFunction { .. }) = self.heap.get(handle) else {
            return false;
        };
        if self.is_host_exotic(handle) {
            return false;
        }
        let aux = self.aux_props.get(&handle.to_raw()).copied();
        if let Some(a) = aux {
            let Some(o) = self.heap.get(a).and_then(Cell::as_object) else {
                return false;
            };
            if !o.is_extensible() || o.contains(key) || o.accessor(key).is_some() {
                return false;
            }
        }
        if let Some(p) = self.object_proto(handle) {
            match self.chain_cached_get(p, key, cache) {
                Some(super::ChainHit::Absent) => {}
                Some(_) => return false,
                None => {
                    self.chain_cache_fill(p, key, cache);
                    return false;
                }
            }
        }
        match aux {
            Some(a) => {
                if !self.init_data_property_cached(a, key, value, cache) {
                    self.set_property(handle, key, value);
                }
            }
            None => {
                self.set_property(handle, key, value);
            }
        }
        true
    }

    /// The shape of the object at `handle` when it is a plain shaped data
    /// object with the default `[[Prototype]]` holding exactly `n` keys,
    /// every one added once and none removed: then that shape's keys, in slot
    /// order, are the keys in the order they were added, and
    /// [`new_object_in_shape`](Self::new_object_in_shape) of the same keys'
    /// values remakes the same object (`JSON.parse`'s record templates).
    #[must_use]
    pub fn plain_fresh_shape(&self, handle: Handle, n: usize) -> Option<Rc<Shape>> {
        let Some(Cell::Object(o)) = self.heap.get(handle) else {
            return None;
        };
        let shape = o.shaped_shape()?;
        (o.is_plain_fresh_data() && o.len() as usize == n && o.proto() == self.default_object_proto)
            .then(|| Rc::clone(shape))
    }

    /// This realm's root shape (the one every ordinary object grows from).
    #[must_use]
    pub fn root_shape_rc(&self) -> Rc<Shape> {
        Rc::clone(&self.root_shape)
    }

    /// For a string cell, the UTF-16 code unit at index `i` (`None` when `i` is
    /// `None` or out of range); `None` for any other cell. In-order reads of a
    /// non-ASCII string stay linear (see [`Realm::string_unit_at_seq`]).
    pub fn string_unit_at_index(
        &mut self,
        handle: Handle,
        i: Option<usize>,
    ) -> Option<Option<u16>> {
        self.heap.get(handle)?.as_str()?;
        Some(i.and_then(|i| self.string_unit_at_seq(handle, i)))
    }

    /// Whether the cell at `handle` may own a named (non-index, non-`length`)
    /// property: `false` only for an array with no auxiliary object, the one
    /// place an array keeps such properties.
    #[must_use]
    pub fn may_own_named(&self, handle: Handle) -> bool {
        match self.heap.get(handle) {
            Some(Cell::Array(_)) => self.aux_props.get(&handle.to_raw()).is_some(),
            _ => true,
        }
    }

    /// Whether `handle` is a [`Cell::Array`].
    #[must_use]
    pub fn is_plain_array_cell(&self, handle: Handle) -> bool {
        matches!(self.heap.get(handle), Some(Cell::Array(_)))
    }

    /// The identity of one prototype-chain link for a [`ChainKeyCache`]: the
    /// key epoch that changes with the link's set of own named keys (an
    /// array's, a RegExp's or a Map/Set's live in its auxiliary object, which
    /// may come and go), and the next link. `None` for anything other than an
    /// ordinary object, an array, a RegExp, a collection or a built-in
    /// function, and for a host-exotic one.
    pub(super) fn key_link(&self, h: Handle) -> Option<(Option<u64>, u32, Option<Handle>)> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => Some((None, o.key_epoch(), o.proto())),
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. } => {
                let (aux, epoch) = match self.aux_props.get(&h.to_raw()) {
                    Some(a) => (
                        Some(a.to_raw()),
                        self.heap.get(*a)?.as_object()?.key_epoch(),
                    ),
                    None => (None, 0),
                };
                Some((aux, epoch, self.object_proto(h)))
            }
            _ => None,
        }
    }

    /// For a method call `recv.key(...)` on the heap value `h`: the object the
    /// ordinary `[[Get]]` of the named key `key` (not an index, not `length`,
    /// not `lastIndex`) continues at once `h` itself is known not to own it —
    /// its `[[Prototype]]`, possibly `None`. `None` when `h` does own `key`, or
    /// is anything other than a string primitive, an ordinary non-wrapper
    /// object, an array, a RegExp, a Map/Set or a function (or is
    /// host-exotic), whose own properties this cannot rule out (a built-in
    /// function's — a bound one's too, `Array.prototype.slice` — all live in
    /// its auxiliary object). `recv_shape` memoizes one receiver
    /// shape already known to lack `key` (a shape fixes its data keys, so an
    /// accessor-free object of that shape lacks it too).
    fn method_lookup_start(
        &self,
        h: Handle,
        key: &str,
        recv_shape: &mut Option<Rc<Shape>>,
    ) -> Option<Option<Handle>> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => {
                if o.is_prim_wrapper() {
                    return None;
                }
                let shape = o.shaped_shape().filter(|_| o.has_no_accessors());
                if let (Some(s), Some(m)) = (shape, recv_shape.as_ref())
                    && Rc::ptr_eq(s, m)
                {
                    return Some(o.proto());
                }
                if o.has_own_key(key) {
                    return None;
                }
                if let Some(s) = shape {
                    *recv_shape = Some(Rc::clone(s));
                }
                Some(o.proto())
            }
            // A string primitive owns only its indices and `length`; one with an
            // auxiliary object is a function-like marker cell, not a plain string.
            Cell::Str(_) => {
                if self.aux_props.contains_key(&h.to_raw()) {
                    return None;
                }
                Some(self.object_proto(h))
            }
            // A VM closure synthesizes `name` and `length` (and the legacy
            // `caller`/`arguments` are the VM's); its other own properties live
            // in its auxiliary object, as a native function's do.
            Cell::VmFunction { .. } if matches!(key, "name" | "caller" | "arguments") => None,
            // An array that never had side-table state owns no named property
            // (so no auxiliary object), and one whose `[[Prototype]]` was never
            // set has the realm's first `%Array.prototype%`: what the side-table
            // lookups below would find (see `plain_array_root_proto`).
            Cell::Array(a) if !a.has_side_state() && !a.has_proto_set() => {
                Some(self.root_array_proto)
            }
            // A Map/Set with its default `[[Prototype]]` (no explicit link
            // since, see `Realm::clear_collection_default_proto`).
            Cell::Collection {
                default_proto: true,
                is_set,
                ..
            } => {
                if self.aux_obj(h).is_some_and(|o| o.has_own_key(key)) {
                    return None;
                }
                Some(self.coll_default[usize::from(*is_set)].map(|(_, p)| p))
            }
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. }
            | Cell::VmFunction { .. } => {
                // (An owing closure's own keys are its owed ones.)
                if self.aux_obj(h).is_some_and(|o| o.has_own_key(key)) {
                    return None;
                }
                Some(self.object_proto(h))
            }
            _ => None,
        }
    }

    /// For a string cell holding one flat all-ASCII leaf, the code unit at
    /// ToIntegerOrInfinity(`n`) (`n` a Number: NaN is 0, truncated toward
    /// zero), `Some(None)` when that is out of range; `None` for any other
    /// cell, which the general read handles.
    #[inline]
    pub(crate) fn ascii_unit_at(&self, h: Handle, n: f64) -> Option<Option<u16>> {
        let rope = self.heap.get(h)?.as_str()?;
        let bytes = rope.as_leaf_bytes()?;
        if !rope.is_ascii() {
            return None;
        }
        let n = if n.is_nan() { 0.0 } else { n };
        // `n` truncates to a non-negative index exactly when it exceeds -1.
        Some(if n > -1.0 {
            bytes.get(n as usize).map(|b| u16::from(*b))
        } else {
            None
        })
    }

    /// `%String.prototype%` when `h` is a plain string cell (no auxiliary
    /// object, which would make it a function-like marker, and not
    /// host-exotic): where `[[Get]]` of any named key other than `length`
    /// continues, as [`method_lookup_start`](Self::method_lookup_start)
    /// answers for it. `None` for anything else.
    ///
    /// Likewise the realm's first `%Array.prototype%` for an array that never
    /// had side-table state nor a `[[Prototype]]` set: it owns no named
    /// property, so its lookup continues there (`method_lookup_start`'s
    /// answer for it too).
    #[inline]
    pub(crate) fn plain_recv_lookup_start(&self, h: Handle) -> Option<Handle> {
        let start = match self.heap.get(h)? {
            Cell::Str(_) => {
                if self.str_special && self.aux_props.contains_key(&h.to_raw()) {
                    return None;
                }
                self.string_proto_intrinsic?
            }
            Cell::Array(a) if !a.has_side_state() && !a.has_proto_set() => self.root_array_proto?,
            _ => return None,
        };
        (!self.is_host_exotic(h)).then_some(start)
    }

    /// The object holding the named properties of the chain link `h`: `h`
    /// itself for an ordinary object, else its auxiliary object.
    fn named_props(&self, h: Handle) -> Option<&crate::object::Object> {
        match self.heap.get(h)? {
            Cell::Object(o) => Some(o),
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. } => self
                .heap
                .get(*self.aux_props.get(&h.to_raw())?)?
                .as_object(),
            _ => None,
        }
    }

    /// The current value of the method `recv[key]` (a named key: not an
    /// index, not `length`, not `lastIndex`) when `cache` still proves where
    /// its `[[Get]]` lands: `recv` does not own `key` (see
    /// `method_lookup_start`), and from its prototype on the chain is exactly
    /// the recorded objects with their recorded key epochs, the last holding
    /// `key` as a data property (read now). `None` is a miss.
    #[must_use]
    pub fn method_site_get(
        &self,
        recv: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        if !cache.armed || cache.swaps != self.cell_swaps || cache.prim {
            return None;
        }
        let epoch = crate::object::proto_epoch();
        if cache.stamp == epoch && cache.recv_stamp == epoch && cache.recv_memo == recv.to_raw() {
            let first = cache.links.first().map(|l| Handle::from_raw(l.0));
            return self.site_stamped_get(first, key, cache);
        }
        let cur = if cache.own {
            Some(recv)
        } else {
            self.method_lookup_start(recv, key, &mut cache.recv_shape)?
        };
        if cache.stamp == crate::object::proto_epoch() {
            return self.site_stamped_get(cur, key, cache);
        }
        self.site_links_get_stamping(cur, key, cache)
    }

    /// `recv.key(a, b)` at a method site whose memo `cache` names the
    /// `Map.prototype` / `Set.prototype` built-in `method` that `op` is, when
    /// the site still proves that `recv[key]` is that very function: either
    /// `recv` is the receiver the site noted at the current prototype epoch
    /// ([`method_site_note_recv`](Self::method_site_note_recv)), or it is any
    /// collection cell that owns no named property (it has no auxiliary
    /// object, where a collection's named properties live) and is not
    /// host-exotic, so the `[[Get]]` of the key continues at its
    /// `[[Prototype]]`; that is the cache's first link, every link of which
    /// is unchanged since it was validated with all of them watched (the
    /// stamp is the current prototype epoch); and the holder's slot still
    /// holds `method`. An interpreter that keeps making environments of fresh
    /// Maps (`env.vars.get(name)` up a scope chain) calls each method on ever
    /// new receivers, which a one-receiver memo never serves. Then the
    /// method's lean form ([`coll_fast`](Self::coll_fast)); a lookup in a
    /// collection without an index and at most `COLL_SCAN_MAX` entries is
    /// answered on the same read of the receiver's cell. `None`, having done
    /// nothing, for anything else (the caller's general path, which also
    /// rejects a receiver of the wrong kind).
    #[inline]
    pub(crate) fn coll_site_call(
        &mut self,
        recv: Handle,
        cache: &mut MethodSiteCache,
        method: u64,
        op: super::coll_fast::CollOp,
        a: NanBox,
        b: NanBox,
    ) -> Option<NanBox> {
        use super::coll_fast::CollOp;
        let epoch = crate::object::proto_epoch();
        if !cache.armed
            || cache.prim
            || cache.own
            || cache.stamp != epoch
            || cache.swaps != self.cell_swaps
        {
            return None;
        }
        // The holder's own value (a store to an existing data property does
        // not move the epoch): read under its recorded shape.
        match self.heap.get_hot(Handle::from_raw(cache.holder_props)) {
            Some(Cell::Object(o))
                if o.shaped_shape()
                    .zip(cache.holder_shape.as_ref())
                    .is_some_and(|(s, m)| Rc::ptr_eq(s, m))
                    && o.slot(cache.holder_slot)
                        .is_some_and(|v| v.to_bits() == method) => {}
            _ => return None,
        }
        let Some(Cell::Collection {
            is_set,
            is_weak: false,
            default_proto,
            entries,
            live,
            index,
            ..
        }) = self.heap.get_hot(recv)
        else {
            return None;
        };
        if *is_set != op.on_set() {
            return None;
        }
        let unnoted = cache.recv_memo != recv.to_raw() || cache.recv_stamp != epoch;
        if unnoted {
            // No auxiliary object (no own named property), not host-exotic,
            // and the first link as its `[[Prototype]]`: the default one
            // (`default_proto` means no explicit link was made since) or
            // another found the general way.
            let first = cache.links.first()?;
            if self.aux_props.contains_key(&recv.to_raw()) || self.is_host_exotic(recv) {
                return None;
            }
            let proto = if *default_proto {
                self.coll_default[usize::from(*is_set)].map(|(_, p)| p)
            } else {
                self.coll_recv_proto(recv)
            };
            if proto.map(Handle::to_raw) != Some(first.0) {
                return None;
            }
        }
        let lookup = matches!(op, CollOp::MapGet | CollOp::MapHas | CollOp::SetHas);
        // A lookup answered on this read of the cell: an empty collection,
        // one with an index (its probe), or a small one (a scan).
        let found = if !lookup {
            None
        } else if *live == 0 {
            Some(None)
        } else if let Some(map) = index {
            let hash = self.collection_key_hash(a);
            Some(self.collection_probe(map, entries, a, hash))
        } else if entries.len() <= super::COLL_SCAN_MAX {
            self.collection_scan(entries, a)
        } else {
            None
        };
        let answer = found.map(|pos| {
            if op == CollOp::MapGet {
                pos.and_then(|i| entries.get(i))
                    .map_or(NanBox::undefined(), |e| e.1)
            } else {
                NanBox::boolean(pos.is_some())
            }
        });
        if unnoted {
            // A receiver proved twice in a row is noted, so the next calls
            // on it skip its probe (one that keeps changing is not: noting
            // watches it, a cost every receiver would pay).
            if cache.last_recv == recv.to_raw() && cache.recv_notes < RECV_NOTES {
                self.coll_site_note(recv, cache);
            }
            cache.last_recv = recv.to_raw();
        }
        // (The receiver is a strong collection of the method's kind.)
        match (answer, op) {
            (Some(v), _) => Some(v),
            _ => self.coll_site_update(recv, op, a, b),
        }
    }

    /// Whether `recv[key]` is still `method` at a method site whose memo
    /// `cache` names that `Map.prototype` / `Set.prototype` built-in, for a
    /// Map/Set receiver with its default `[[Prototype]]`: the proof of
    /// [`coll_site_call`](Self::coll_site_call) (the holder's slot still
    /// holds `method`, every link unchanged since it was validated with all
    /// of them watched, and `recv` owns no named property and continues at
    /// the first link), without the call. A compiled `m.get(k)` reads its
    /// method on its own (`Op::GetMethod`, before the arguments) on ever new
    /// receivers (`dist.get(…)` on each call's fresh Map), which the general
    /// [`method_site_get`](Self::method_site_get) serves only after a
    /// receiver walk. `false` for anything else (the caller's general path).
    #[inline]
    pub(crate) fn coll_site_method_hit(
        &self,
        recv: Handle,
        cache: &MethodSiteCache,
        method: u64,
    ) -> bool {
        let epoch = crate::object::proto_epoch();
        if !cache.armed
            || cache.prim
            || cache.own
            || cache.stamp != epoch
            || cache.swaps != self.cell_swaps
        {
            return false;
        }
        match self.heap.get_hot(Handle::from_raw(cache.holder_props)) {
            Some(Cell::Object(o))
                if o.shaped_shape()
                    .zip(cache.holder_shape.as_ref())
                    .is_some_and(|(s, m)| Rc::ptr_eq(s, m))
                    && o.slot(cache.holder_slot)
                        .is_some_and(|v| v.to_bits() == method) => {}
            _ => return false,
        }
        let Some(Cell::Collection {
            is_set,
            is_weak: false,
            default_proto: true,
            ..
        }) = self.heap.get_hot(recv)
        else {
            return false;
        };
        if cache.recv_memo == recv.to_raw() && cache.recv_stamp == epoch {
            return true;
        }
        let Some(first) = cache.links.first() else {
            return false;
        };
        !self.aux_props.contains_key(&recv.to_raw())
            && !self.is_host_exotic(recv)
            && self.coll_default[usize::from(*is_set)].map(|(_, p)| p.to_raw()) == Some(first.0)
    }

    /// [`object_proto`](Self::object_proto) of a collection without the
    /// default `[[Prototype]]` (out of [`coll_site_call`](Self::coll_site_call)'s
    /// line).
    #[inline(never)]
    fn coll_recv_proto(&self, recv: Handle) -> Option<Handle> {
        self.object_proto(recv)
    }

    /// [`coll_site_call`](Self::coll_site_call)'s update, or lookup the
    /// one read of the cell could not answer (out of its line).
    #[inline(never)]
    fn coll_site_update(
        &mut self,
        recv: Handle,
        op: super::coll_fast::CollOp,
        a: NanBox,
        b: NanBox,
    ) -> Option<NanBox> {
        use super::coll_fast::CollOp;
        Some(match op {
            CollOp::MapSet => {
                self.collection_set(recv, a, b);
                NanBox::handle(recv.to_raw())
            }
            CollOp::SetAdd => {
                self.collection_set(recv, a, a);
                NanBox::handle(recv.to_raw())
            }
            CollOp::MapDelete | CollOp::SetDelete => {
                NanBox::boolean(self.collection_delete(recv, a))
            }
            // A lookup in a larger collection without an index (which
            // builds it).
            _ => return self.coll_fast(recv, op, a, b),
        })
    }

    /// [`coll_site_call`](Self::coll_site_call)'s noting of a receiver
    /// (out of its line).
    #[cold]
    #[inline(never)]
    fn coll_site_note(&mut self, recv: Handle, cache: &mut MethodSiteCache) {
        self.method_site_note_recv(recv, cache, false);
    }

    /// `Object.keys(a)` of an array that never had side-table state (so no
    /// named property, no index accessor, no non-writable `length`) whose
    /// length is its dense length: the indices of its present elements, in
    /// ascending order (a hole is no own property), as fresh index Strings.
    /// `None` for anything else.
    pub(crate) fn plain_array_index_keys(&mut self, h: Handle) -> Option<alloc::vec::Vec<NanBox>> {
        if self.is_host_exotic(h) {
            return None;
        }
        let present: alloc::vec::Vec<usize> = match self.heap.get(h)? {
            Cell::Array(a) if !a.has_side_state() => {
                (0..a.len()).filter(|&i| !a[i].is_hole()).collect()
            }
            _ => return None,
        };
        if self.array_length(h) != self.array_elements(h).map(<[NanBox]>::len) {
            return None;
        }
        Some(
            present
                .into_iter()
                .map(|i| {
                    NanBox::handle(self.new_primitive_string(NanBox::number(i as f64)).to_raw())
                })
                .collect(),
        )
    }

    /// [`has_own`](Self::has_own) for an ordinary object that is no
    /// primitive wrapper (whose own properties are exactly its data keys and
    /// accessors); `None` for any other cell.
    #[inline]
    pub(crate) fn ordinary_has_own(&self, h: Handle, key: &str) -> Option<bool> {
        match self.heap.get(h)? {
            Cell::Object(o) if !o.is_prim_wrapper() => {
                Some(o.contains(key) || (!o.has_no_accessors() && o.accessor(key).is_some()))
            }
            _ => None,
        }
    }

    /// The static-method fast form of [`method_site_get`](Self::method_site_get)
    /// (`Object.hasOwn`, `Number.isInteger`, `JSON.stringify`): whether `cache`
    /// was filled for a receiver that owns the method itself (its one link,
    /// the holder) and that receiver is `recv`, with the link unchanged since
    /// it was validated with every link watched (the prototype epoch is still
    /// the stamp: no own key added, removed or turned into an accessor) and
    /// the holder's slot still `method` (a plain store to the property does
    /// not move the epoch, so the value is re-read). `false` sends the caller
    /// to the general check. A receiver handle carries its generation, so a
    /// collected-and-reused slot cannot match the recorded link.
    #[inline]
    pub(crate) fn method_site_own_hit(
        &self,
        recv: Handle,
        cache: &MethodSiteCache,
        method: u64,
    ) -> bool {
        if !cache.armed
            || !cache.own
            || cache.stamp != crate::object::proto_epoch()
            || cache.swaps != self.cell_swaps
            || cache.links.first().is_none_or(|l| l.0 != recv.to_raw())
        {
            return false;
        }
        if let Some(Cell::Object(o)) = self.heap.get(Handle::from_raw(cache.holder_props))
            && let (Some(s), Some(m)) = (o.shaped_shape(), cache.holder_shape.as_ref())
            && Rc::ptr_eq(s, m)
        {
            return o
                .slot(cache.holder_slot)
                .is_some_and(|v| v.to_bits() == method);
        }
        false
    }

    /// After a [`method_site_get`](Self::method_site_get) hit on the Map/Set
    /// `recv` with every link watched at the current prototype epoch: watches
    /// `recv` (its cell and auxiliary object, so a new own property or a
    /// `[[Prototype]]` change bumps the epoch) and remembers that, at this
    /// epoch, its lookup continues at the first link, so the next hits on it
    /// skip the receiver probe.
    ///
    /// `objects`: an ordinary object receiver is noted too (a site calling a
    /// generator's `next`, whose receivers are generator objects: rarely
    /// reshaped, so watching them costs nothing).
    pub fn method_site_note_recv(
        &mut self,
        recv: Handle,
        cache: &mut MethodSiteCache,
        objects: bool,
    ) {
        let epoch = crate::object::proto_epoch();
        if cache.own
            || cache.prim
            || cache.stamp != epoch
            || (cache.recv_stamp == epoch && cache.recv_memo == recv.to_raw())
            || !match self.heap.get(recv) {
                Some(Cell::Collection { .. }) => true,
                Some(Cell::Object(_)) => objects,
                _ => false,
            }
            || cache.recv_notes >= RECV_NOTES
            || self.is_host_exotic(recv)
        {
            return;
        }
        cache.recv_notes += 1;
        let aux = self.aux_props.get(&recv.to_raw()).map(|a| a.to_raw());
        self.watch_link(recv.to_raw(), aux);
        if crate::object::proto_epoch() == epoch {
            cache.recv_memo = recv.to_raw();
            cache.recv_stamp = epoch;
        }
    }

    /// [`site_links_get`](Self::site_links_get) for a cache whose stamp is
    /// current: the chain from `cur` is the validated one as long as `cur` is
    /// its first link, so only the holder's value is read.
    fn site_stamped_get(
        &self,
        cur: Option<Handle>,
        key: &str,
        cache: &MethodSiteCache,
    ) -> Option<NanBox> {
        let first = cache.links.first()?;
        if cur?.to_raw() != first.0 {
            return None;
        }
        let holder = Handle::from_raw(cache.links.last()?.0);
        if let Some(Cell::Object(o)) = self.heap.get(Handle::from_raw(cache.holder_props))
            && let (Some(s), Some(m)) = (o.shaped_shape(), cache.holder_shape.as_ref())
            && Rc::ptr_eq(s, m)
        {
            return o.slot(cache.holder_slot);
        }
        self.holder_data(holder, key)
    }

    /// [`site_links_get`](Self::site_links_get), stamping the cache with the
    /// current prototype epoch when the walk hits over watched links only.
    fn site_links_get_stamping(
        &self,
        cur: Option<Handle>,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        let v = self.site_links_get(cur, key, cache)?;
        if cache
            .links
            .iter()
            .all(|&(raw, aux, _)| self.link_watched(raw, aux))
        {
            cache.stamp = crate::object::proto_epoch();
        }
        Some(v)
    }

    /// [`method_site_get`](Self::method_site_get) for a *primitive* receiver
    /// (a Number, a Boolean, …), whose lookup starts at its prototype
    /// intrinsic `start` ([`primitive_proto`](Self::primitive_proto)) — a
    /// primitive owns no named property. Only a site filled by
    /// [`method_site_fill_prim`](Self::method_site_fill_prim) hits.
    #[must_use]
    pub fn method_site_get_prim(
        &self,
        start: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        if !cache.armed || cache.swaps != self.cell_swaps || !cache.prim {
            return None;
        }
        if cache.stamp == crate::object::proto_epoch() {
            return self.site_stamped_get(Some(start), key, cache);
        }
        self.site_links_get_stamping(Some(start), key, cache)
    }

    /// The walk of [`method_site_get`](Self::method_site_get) from the first
    /// chain link `cur`.
    fn site_links_get(
        &self,
        mut cur: Option<Handle>,
        key: &str,
        cache: &MethodSiteCache,
    ) -> Option<NanBox> {
        let last = cache.links.len().checked_sub(1)?;
        for (i, &(raw, aux, epoch)) in cache.links.iter().enumerate() {
            let h = cur?;
            if h.to_raw() != raw {
                return None;
            }
            // The usual link, an ordinary object (a prototype): the checks of
            // `key_link`/`named_props` on one heap read.
            if aux.is_none()
                && let Some(Cell::Object(o)) = self.heap.get(h)
            {
                if o.key_epoch() != epoch || self.is_host_exotic(h) {
                    return None;
                }
                if i == last {
                    if let (Some(s), Some(m)) = (o.shaped_shape(), cache.holder_shape.as_ref())
                        && Rc::ptr_eq(s, m)
                    {
                        return o.slot(cache.holder_slot);
                    }
                    return self.holder_data(h, key);
                }
                cur = o.proto();
                continue;
            }
            let (a, e, next) = self.key_link(h)?;
            if a != aux || e != epoch {
                return None;
            }
            if i == last {
                // Same key set as when armed: a data property, at the
                // recorded slot while the layout is the recorded one.
                let o = self.named_props(h)?;
                if let (Some(s), Some(m)) = (o.shaped_shape(), cache.holder_shape.as_ref())
                    && Rc::ptr_eq(s, m)
                {
                    return o.slot(cache.holder_slot);
                }
                return self.holder_data(h, key);
            }
            cur = next;
        }
        None
    }

    /// Arms `cache` for `recv[key]` (see [`method_site_get`](Self::method_site_get))
    /// and returns the method's value, when the walk from `recv`'s prototype
    /// meets only ordinary objects, arrays, RegExps and collections (none
    /// host-exotic, at most eight deep) before one owning `key` as a data
    /// property. Leaves it unarmed (and returns `None`) otherwise.
    pub fn method_site_fill(
        &mut self,
        recv: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        cache.armed = false;
        cache.stamp = 0;
        cache.recv_stamp = 0;
        cache.links.clear();
        cache.holder_shape = None;
        cache.own = false;
        cache.prim = false;
        cache.swaps = self.cell_swaps;
        let cur = match self.method_lookup_start(recv, key, &mut cache.recv_shape) {
            Some(start) => start,
            // A static (`String.fromCharCode`): the receiver itself, an
            // ordinary object or a native function, holds the method. It is
            // then the chain's one link, so a hit requires this very receiver.
            None if matches!(self.heap.get(recv), Some(Cell::Object(_) | Cell::Native(_)))
                && self.named_props(recv).is_some_and(|o| !o.is_prim_wrapper())
                && self.has_own(recv, key) =>
            {
                cache.own = true;
                Some(recv)
            }
            None => return None,
        };
        self.site_links_fill(cur, key, cache)
    }

    /// [`method_site_fill`](Self::method_site_fill) for a primitive receiver
    /// whose lookup starts at `start` (see
    /// [`method_site_get_prim`](Self::method_site_get_prim)).
    pub fn method_site_fill_prim(
        &mut self,
        start: Handle,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        cache.armed = false;
        cache.stamp = 0;
        cache.links.clear();
        cache.holder_shape = None;
        cache.own = false;
        cache.prim = true;
        cache.swaps = self.cell_swaps;
        self.site_links_fill(Some(start), key, cache)
    }

    /// The walk of [`method_site_fill`](Self::method_site_fill) from the first
    /// chain link `cur`.
    fn site_links_fill(
        &mut self,
        mut cur: Option<Handle>,
        key: &str,
        cache: &mut MethodSiteCache,
    ) -> Option<NanBox> {
        for _ in 0..8 {
            let h = cur?;
            let (aux, epoch, next) = self.key_link(h)?;
            cache.links.push((h.to_raw(), aux, epoch));
            if self.has_own(h, key) {
                let v = self.holder_data(h, key)?;
                let o = self.named_props(h)?;
                if let Some(s) = o.shaped_shape()
                    && let Some(slot) = s.lookup(key)
                {
                    cache.holder_shape = Some(Rc::clone(s));
                    cache.holder_slot = slot;
                }
                cache.holder_props = aux.unwrap_or(h.to_raw());
                cache.armed = true;
                for &(raw, aux, _) in &cache.links {
                    self.watch_link(raw, aux);
                }
                cache.stamp = crate::object::proto_epoch();
                return Some(v);
            }
            cur = next;
        }
        None
    }

    /// [`watch_link`](Self::watch_link) of the cell `raw` with its own
    /// auxiliary object, if any.
    pub(crate) fn watch_cell(&mut self, raw: u64) {
        let aux = self.aux_props.get(&raw).map(|a| a.to_raw());
        self.watch_link(raw, aux);
    }

    /// Watches a recorded chain link (see `crate::object::JIT_FLAG_WATCHED`):
    /// the link `raw` itself when it is an ordinary object, else the cell (in
    /// the realm's watched set, so a change to its auxiliary object or
    /// `[[Prototype]]` bumps the prototype epoch) and its auxiliary object
    /// `aux`, if any.
    pub(crate) fn watch_link(&mut self, raw: u64, aux: Option<u64>) {
        let h = Handle::from_raw(raw);
        if let Some(Cell::Object(o)) = self.heap.get_mut(h) {
            o.watch();
            return;
        }
        self.watched_cells.insert(raw, ());
        if let Some(a) = aux
            && let Some(Cell::Object(o)) = self.heap.get_mut(Handle::from_raw(a))
        {
            o.watch();
        }
    }

    /// Whether [`watch_link`](Self::watch_link) holds for the link `raw`
    /// (with auxiliary object `aux`) as it is now.
    pub(crate) fn link_watched(&self, raw: u64, aux: Option<u64>) -> bool {
        let watched = |raw: u64| match self.heap.get(Handle::from_raw(raw)) {
            Some(Cell::Object(o)) => o.jit_flags() & crate::object::JIT_FLAG_WATCHED != 0,
            _ => false,
        };
        if matches!(self.heap.get(Handle::from_raw(raw)), Some(Cell::Object(_))) {
            return watched(raw);
        }
        self.watched_cells.contains_key(&raw) && aux.is_none_or(watched)
    }

    /// The `[[Get]]` of `cache`'s key from `start` when `cache` still proves
    /// its outcome: `Some(None)` when no object on the chain owns the key,
    /// `Some(Some(v))` when the recorded holder still owns it as a data
    /// property whose current value is `v`. `None` is a miss. The live chain
    /// must still be exactly the recorded objects, each with its recorded key
    /// epoch (so none has gained or lost an own key, or turned one into an
    /// accessor), in which case the ordinary `[[Get]]` walk visits the same
    /// objects and stops at the same place.
    #[must_use]
    pub fn chain_key_cached(&self, start: Handle, cache: &ChainKeyCache) -> Option<Option<NanBox>> {
        if !cache.armed || cache.swaps != self.cell_swaps {
            return None;
        }
        let mut cur = Some(start);
        let last = cache.links.len().checked_sub(1);
        for (i, &(raw, aux, epoch)) in cache.links.iter().enumerate() {
            let h = cur?;
            if h.to_raw() != raw {
                return None;
            }
            // An ordinary-object link (the usual case): one heap read.
            if aux.is_none()
                && !(cache.holder && Some(i) == last)
                && let Some(Cell::Object(o)) = self.heap.get(h)
            {
                if o.key_epoch() != epoch || self.is_host_exotic(h) {
                    return None;
                }
                cur = o.proto();
                continue;
            }
            match self.key_link(h) {
                Some((a, e, next)) if a == aux && e == epoch => {
                    if cache.holder && Some(i) == last {
                        // Same key set as when armed, so still a data
                        // property: at the recorded slot while the layout is
                        // the recorded one.
                        if let Some(m) = cache.holder_shape.as_ref()
                            && let Some(o) = self.named_props(h)
                            && let Some(s) = o.shaped_shape()
                            && Rc::ptr_eq(s, m)
                        {
                            return o.slot(cache.holder_slot).map(Some);
                        }
                        return self.holder_data(h, &cache.key).map(Some);
                    }
                    cur = next;
                }
                _ => return None,
            }
        }
        cur.is_none().then_some(None)
    }

    /// The own data property `key` of a [`ChainKeyCache`] holder: an ordinary
    /// object's slot, or the named property in an array's or RegExp's
    /// auxiliary object. `None` when it is an accessor or absent.
    fn holder_data(&self, h: Handle, key: &str) -> Option<NanBox> {
        let o = match self.heap.get(h)? {
            Cell::Object(o) => o,
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. } => self
                .heap
                .get(*self.aux_props.get(&h.to_raw())?)?
                .as_object()?,
            _ => return None,
        };
        if o.accessor(key).is_some() {
            return None;
        }
        o.get(key)
    }

    /// The ordinary `[[Get]]` of the named key `key` (not an index or
    /// `length`) from `start`, answered from the realm's `slot`-th shared
    /// `ChainKeyCache` when it still proves the outcome (see
    /// [`chain_key_cached`](Self::chain_key_cached)): `Some(None)` absent,
    /// `Some(Some(v))` a data property. `None` on a miss, after re-arming the
    /// cache; the caller then does the lookup itself.
    #[must_use]
    pub fn shared_chain_lookup(
        &self,
        slot: usize,
        start: Handle,
        key: &str,
    ) -> Option<Option<NanBox>> {
        let mut caches = self.chain_lookups.borrow_mut();
        if caches.len() <= slot {
            caches.resize_with(slot + 1, ChainKeyCache::default);
        }
        let cache = &mut caches[slot];
        if &*cache.key == key
            && let Some(hit) = self.chain_key_cached(start, cache)
        {
            return Some(hit);
        }
        // A different key re-arms from scratch (and gets its own fill budget).
        if &*cache.key != key {
            *cache = ChainKeyCache::default();
        }
        self.chain_key_fill(start, key, cache);
        None
    }

    /// Whether `cache` proves that no object on the chain from `start` owns
    /// its key (see [`chain_key_cached`](Self::chain_key_cached)).
    #[must_use]
    pub fn chain_absent_cached(&self, start: Handle, cache: &ChainKeyCache) -> bool {
        matches!(self.chain_key_cached(start, cache), Some(None))
    }

    /// Arms `cache` for the named key `key` (not an index, not `length`, not
    /// `lastIndex`) from `start` when every object on the chain up to the
    /// first that owns `key` is an ordinary object, an array or a RegExp (none
    /// host-exotic, at most eight deep), and that owner, if any, is an ordinary
    /// object holding `key` as a data property. Leaves it unarmed otherwise.
    /// Re-arming stops after a few fills, so a site whose chain keeps changing
    /// stops paying for it.
    pub fn chain_key_fill(&self, start: Handle, key: &str, cache: &mut ChainKeyCache) {
        cache.armed = false;
        if cache.fills >= 16 {
            return;
        }
        cache.fills += 1;
        cache.links.clear();
        cache.swaps = self.cell_swaps;
        cache.holder = false;
        cache.holder_shape = None;
        if &*cache.key != key {
            cache.key = alloc::boxed::Box::from(key);
        }
        let mut cur = Some(start);
        for _ in 0..8 {
            let Some(h) = cur else {
                cache.armed = true;
                return;
            };
            let Some((aux, epoch, next)) = self.key_link(h) else {
                return;
            };
            if self.has_own(h, key) {
                if self.holder_data(h, key).is_none() {
                    return;
                }
                if let Some(o) = self.named_props(h)
                    && let Some(s) = o.shaped_shape()
                    && let Some(slot) = s.lookup(key)
                {
                    cache.holder_shape = Some(Rc::clone(s));
                    cache.holder_slot = slot;
                }
                cache.links.push((h.to_raw(), aux, epoch));
                cache.holder = true;
                cache.armed = true;
                return;
            }
            cache.links.push((h.to_raw(), aux, epoch));
            cur = next;
        }
    }
}

/// Where a watched chain's `[[Get]]` found a data property: the holder's
/// named-property object (raw handle), its shape then and the slot.
pub(crate) type StampedSlot = (u64, Rc<Shape>, u32);

/// [`ArrayIterMemo`]'s `open`, as last found valid for a plain array (no
/// named property, the realm's first `%Array.prototype%`), flat so compiled
/// code can make the same test inline: the prototype epoch it holds at, and
/// per lookup the address of the found slot's value in its (watched) holder
/// and the value found there. While the epoch stands, neither holder has
/// changed its layout or moved (`Object::jit_slot_ptr`), so compiled code
/// re-reads both values through those addresses and compares them (a value
/// is no part of what the epoch watches). It never outlives the memo entry
/// it copies (every change to the memo empties it). `stamp == u64::MAX`:
/// empty.
///
/// `fresh` is `stamp` when, besides, a fresh array (the running realm's
/// `%Array.prototype%`, which is then the first one) destructures without
/// running code — the memo's "no reachable `return`" stamp
/// (`ArrayIterMemo::close_stamp`) holds at that epoch too — as
/// `iter_fast::fresh_array_is_plain` decides it; else `u64::MAX`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ArrayIterJit {
    pub(crate) stamp: u64,
    pub(crate) iter_ptr: u64,
    pub(crate) iter_fn: u64,
    pub(crate) next_ptr: u64,
    pub(crate) next_fn: u64,
    pub(crate) fresh: u64,
}

impl Default for ArrayIterJit {
    fn default() -> Self {
        Self {
            stamp: u64::MAX,
            iter_ptr: 0,
            iter_fn: 0,
            next_ptr: 0,
            next_fn: 0,
            fresh: u64::MAX,
        }
    }
}

/// The memo of `ArraySpeciesCreate`'s default-case proof for arrays with no
/// own named properties (`nbvm::array_hof::default_species`): the
/// `[[Get]]` of `constructor` from the array prototype `proto` finds the data
/// property `at` (every link on the way watched at prototype epoch `stamp`),
/// which then held the `%Array%` intrinsic `ctor`. A hit re-reads that slot
/// and `ctor`'s `@@species` getter, which an in-place write could change
/// without moving the epoch.
pub(crate) struct ArraySpeciesMemo {
    pub(crate) proto: u64,
    pub(crate) stamp: u64,
    pub(crate) at: StampedSlot,
    pub(crate) ctor: u64,
}

/// The VM's memo of the fixed lookups of its array-iteration fast path
/// (`nbvm::iter_fast`), each validated at a prototype epoch
/// (`crate::object::proto_epoch`) with every link of its chain watched, so
/// that while the epoch is unchanged a hit re-reads only the found value.
#[derive(Default)]
pub(crate) struct ArrayIterMemo {
    /// The epoch at which `open` was validated (`0`: never).
    pub(crate) open_stamp: u64,
    /// The array prototype (raw handle) `open` is for.
    pub(crate) open_proto: u64,
    /// `[[Get]](open_proto, @@iterator)` and `[[Get]](%ArrayIteratorPrototype%,
    /// "next")`.
    pub(crate) open: Option<(StampedSlot, StampedSlot)>,
    /// The epoch at which `GetMethod(%ArrayIteratorPrototype%, "return")` was
    /// validated to find nothing (`0`: never).
    pub(crate) close_stamp: u64,
}

impl Realm {
    /// Watches every link of the `[[Get]]` of the named key `key` from
    /// `start` (see [`watch_link`](Self::watch_link)) up to the first that
    /// owns it, when that walk meets only ordinary objects, arrays, RegExps,
    /// collections and natives (none host-exotic, at most eight deep): the
    /// holder's shaped data property ([`StampedSlot`]) as `Some(Some(_))`,
    /// `Some(None)` when no link owns the key. `None` (some links perhaps
    /// watched) for anything else: an accessor, a dictionary-mode holder, a
    /// deeper or other chain.
    pub(crate) fn watch_get_chain(
        &mut self,
        start: Handle,
        key: &str,
    ) -> Option<Option<StampedSlot>> {
        let mut cur = Some(start);
        for _ in 0..8 {
            let Some(h) = cur else {
                return Some(None);
            };
            let (aux, _, next) = self.key_link(h)?;
            self.watch_link(h.to_raw(), aux);
            if self.has_own(h, key) {
                self.holder_data(h, key)?;
                let o = self.named_props(h)?;
                let s = o.shaped_shape()?;
                let slot = s.lookup(key)?;
                return Some(Some((aux.unwrap_or(h.to_raw()), Rc::clone(s), slot)));
            }
            cur = next;
        }
        None
    }

    /// The current value at a [`StampedSlot`], while its holder still has the
    /// recorded shape.
    pub(crate) fn stamped_slot(&self, at: &StampedSlot) -> Option<NanBox> {
        let o = self.heap.get(Handle::from_raw(at.0))?.as_object()?;
        if !o.shaped_shape().is_some_and(|s| Rc::ptr_eq(s, &at.1)) {
            return None;
        }
        o.slot(at.2)
    }
}

/// A memo of one named key's `[[Get]]` outcome along one prototype chain —
/// absent everywhere, or a data property of a particular holder — validated by
/// object identity (a handle carries its slot's generation, so a recycled slot
/// never matches) and each object's key epoch
/// ([`Object::key_epoch`](crate::object::Object::key_epoch), bumped on every
/// change to its set of own keys). Unlike [`crate::ic::ChainCache`] it does not
/// need shaped objects, so it also covers dictionary-mode, array and RegExp
/// links. The holder's value is re-read on every hit.
#[derive(Default)]
pub struct ChainKeyCache {
    key: alloc::boxed::Box<str>,
    /// `(handle, auxiliary object handle, key epoch)` per link.
    links: alloc::vec::Vec<(u64, Option<u64>, u32)>,
    /// Whether the last link owns the key (else no link does).
    holder: bool,
    /// The holder's shape when armed and the key's slot in it, so a hit
    /// under that same layout reads the slot without a name lookup.
    holder_shape: Option<Rc<Shape>>,
    holder_slot: u32,
    armed: bool,
    fills: u16,
    swaps: u32,
}

/// The most receivers one method site watches for its receiver memo (see
/// [`Realm::method_site_note_recv`]).
const RECV_NOTES: u8 = 32;

/// A method-call site's memo of where `recv[key]` resolves (see
/// [`Realm::method_site_get`]): like [`ChainKeyCache`] from the receiver's
/// prototype on, plus the holder's slot under its recorded shape and one
/// receiver shape known not to own the key, so neither end repeats a name
/// lookup on a hit.
#[derive(Default)]
pub struct MethodSiteCache {
    /// `(handle, auxiliary object handle, key epoch)` per link, the holder last.
    links: alloc::vec::Vec<(u64, Option<u64>, u32)>,
    holder_shape: Option<Rc<Shape>>,
    holder_slot: u32,
    /// The object holding the holder link's named properties (the link
    /// itself, or its auxiliary object), as a raw handle.
    holder_props: u64,
    /// The prototype epoch (`crate::object::proto_epoch`) at which the links
    /// were last validated with every link watched (see
    /// [`Realm::watch_link`]); `0` for never. While the epoch is still
    /// `stamp`, no link changed, so a hit needs only the receiver's side.
    stamp: u64,
    recv_shape: Option<Rc<Shape>>,
    /// A Map/Set receiver (raw handle) known, at the prototype epoch
    /// `recv_stamp`, not to own the key and to have the first link as its
    /// `[[Prototype]]`, with the cell (and its auxiliary object, if any)
    /// watched: while the epoch is unchanged, neither can have changed (see
    /// [`Realm::method_site_note_recv`]).
    recv_memo: u64,
    recv_stamp: u64,
    /// The last receiver [`Realm::coll_site_call`] proved without the memo
    /// (raw handle), so that one met twice in a row gets noted.
    last_recv: u64,
    /// How many receivers this site has watched for `recv_memo` (capped at
    /// [`RECV_NOTES`], so a site that sees ever new Maps adds a bounded
    /// number of entries to the realm's never-pruned watched set).
    recv_notes: u8,
    /// Whether the receiver itself is the holder (the only link).
    own: bool,
    /// Whether the site was filled for a primitive receiver (the walk starts
    /// at its prototype intrinsic).
    prim: bool,
    armed: bool,
    swaps: u32,
}

/// The program slot of `Realm::regexp_literal_memo`.
#[cfg(feature = "regex")]
pub(crate) type MemoProgram = core::cell::RefCell<Option<Rc<crate::regex::Regex>>>;
/// The program slot of `Realm::regexp_literal_memo` (none without `regex`).
#[cfg(not(feature = "regex"))]
pub(crate) type MemoProgram = ();

/// A new RegExp's compiled-program cache, from the memo's program slot.
#[cfg(feature = "regex")]
pub(crate) fn memo_program(p: &MemoProgram) -> Option<Rc<crate::regex::Regex>> {
    p.borrow().clone()
}

/// A new RegExp's compiled-program cache (none without `regex`).
#[cfg(not(feature = "regex"))]
pub(crate) fn memo_program(_: &MemoProgram) {}

impl Realm {
    /// Records `re` as the program of the memoized literal source `source`
    /// (the very allocation), so the next RegExp made from it starts
    /// compiled.
    #[cfg(feature = "regex")]
    pub(crate) fn regexp_literal_memo_note(&self, source: &Rc<[u8]>, re: &Rc<crate::regex::Regex>) {
        for (s, _, p) in self.regexp_literal_memo.iter().flatten() {
            if Rc::ptr_eq(s, source) {
                *p.borrow_mut() = Some(re.clone());
            }
        }
    }
}

/// One remembered compiled regex program (see `Realm::regex_programs`).
#[cfg(feature = "regex")]
pub(crate) struct RegexProgram {
    source: alloc::boxed::Box<[u8]>,
    flags: alloc::boxed::Box<str>,
    program: Rc<crate::regex::Regex>,
}

/// How many compiled programs a realm remembers.
#[cfg(feature = "regex")]
const REGEX_PROGRAMS: usize = 32;

#[cfg(feature = "regex")]
impl Realm {
    /// The remembered program for `(source, flags)`, if any (moved to the
    /// most-recently-used end).
    pub(super) fn regex_program_lookup(
        &self,
        source: &[u8],
        flags: &str,
    ) -> Option<Rc<crate::regex::Regex>> {
        let mut progs = self.regex_programs.borrow_mut();
        let i = progs
            .iter()
            .rposition(|p| &*p.source == source && &*p.flags == flags)?;
        // Already the most recent (a literal evaluated in a loop): no move.
        if i + 1 != progs.len() {
            let p = progs.remove(i);
            progs.push(p);
        }
        progs.last().map(|p| Rc::clone(&p.program))
    }

    /// The compiled program for the WTF-8 pattern `source` under `flags`:
    /// the remembered one, else compiled now and remembered, so that
    /// validating a pattern (the `RegExp` constructor, `compile`) and the
    /// first match of the RegExp it makes share one compilation. `None` for
    /// an invalid pattern or flags.
    pub(crate) fn regex_program_for(
        &self,
        source: &[u8],
        flags: &str,
    ) -> Option<Rc<crate::regex::Regex>> {
        if let Some(re) = self.regex_program_lookup(source, flags) {
            return Some(re);
        }
        let re = Rc::new(
            crate::regex::Regex::new(&crate::regex::pattern_from_wtf8(source, flags), flags)
                .ok()?,
        );
        self.regex_program_remember(source, flags, &re);
        Some(re)
    }

    /// Remembers `program` as the compilation of `(source, flags)`, dropping
    /// the least recently used entry when full.
    pub(super) fn regex_program_remember(
        &self,
        source: &[u8],
        flags: &str,
        program: &Rc<crate::regex::Regex>,
    ) {
        let mut progs = self.regex_programs.borrow_mut();
        if progs.len() >= REGEX_PROGRAMS {
            progs.remove(0);
        }
        progs.push(RegexProgram {
            source: source.into(),
            flags: flags.into(),
            program: Rc::clone(program),
        });
    }
}

/// Appends `Number::toString(n)` to the WTF-8 buffer `out`.
pub(crate) fn push_number_bytes(out: &mut alloc::vec::Vec<u8>, n: f64) {
    let mut buf = [0u8; super::ryu::NUMBER_TEXT_MAX];
    let len = super::ryu::write_js_number(n, &mut buf);
    // The whole buffer, then the excess dropped: a copy of a known size
    // (a few moves) rather than a call to copy `len` bytes.
    let start = out.len();
    out.extend_from_slice(&buf);
    out.truncate(start + len);
}

/// `ToString(n)` as a fresh rope, built from the text without an
/// intermediate `String`.
fn number_rope_uncached(n: f64) -> crate::rope::Rope {
    // An integer below 2^53 in magnitude prints as its plain digits (`-0`
    // as `0`): written straight into an inline rope.
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    if n > -LIMIT && n < LIMIT && (n as i64) as f64 == n {
        return crate::rope::Rope::from_int(n as i64);
    }
    let mut buf = [0u8; super::ryu::NUMBER_TEXT_MAX];
    let len = super::ryu::write_js_number(n, &mut buf);
    // (A number's text is all ASCII: its UTF-16 length is its length.)
    crate::rope::Rope::from_ascii_bytes(&buf[..len])
}

/// How many small non-negative integers keep a shared string rope.
const NUMBER_ROPES: usize = 256;

impl Realm {
    /// `ToString(n)` as a rope. The integers `0..256` share one rope each
    /// (a rope is immutable, so sharing is invisible), which spares
    /// `` `${i % 10}` `` and `"" + digit` an allocation per evaluation.
    pub(super) fn number_rope(&self, n: f64) -> crate::rope::Rope {
        if n >= 0.0 && n < NUMBER_ROPES as f64 && n == (n as usize) as f64 {
            let i = n as usize;
            let mut cache = self.number_ropes.borrow_mut();
            if cache.is_empty() {
                cache.resize(NUMBER_ROPES, None);
            }
            return cache[i]
                .get_or_insert_with(|| number_rope_uncached(n))
                .clone();
        }
        number_rope_uncached(n)
    }

    /// A new string cell holding `ToString(v)` for a primitive `v` (a number
    /// shares its rope through `number_rope`).
    pub fn new_primitive_string(&mut self, v: NanBox) -> Handle {
        // A single digit is a one-unit string: the realm's interned cell.
        if let Some(n) = v.as_number()
            && (0.0..10.0).contains(&n)
            && n == (n as u8) as f64
        {
            return self.unit_string(u16::from(b'0' + n as u8));
        }
        let rope = match v.as_number() {
            Some(n) => self.number_rope(n),
            None => crate::rope::Rope::from(self.to_display_string(v).as_str()),
        };
        self.heap.alloc(Cell::Str(rope))
    }

    /// The string value of an evaluation of a string literal whose text is `s`
    /// (borrowed from the bytecode). Strings are immutable values compared by
    /// contents, so every evaluation of one literal may share one cell: a small
    /// direct-mapped memo keyed by the text's address hands it back. A hit is
    /// re-checked against the text itself (a reused address — freed bytecode —
    /// can never return another string) and the cell against the memo's rope
    /// (an `Rc` the memo keeps alive, so its address cannot be reused either).
    /// A one-unit Latin-1 literal is the realm's interned unit string, the same
    /// cell `s[i]` and `charAt` return.
    pub fn new_literal_string(&mut self, s: &str) -> Handle {
        const SLOTS: usize = 256;
        let b = s.as_bytes();
        match *b {
            // (An ASCII unit's interned cell, checked against the literal's
            // own byte: `unit_string`'s hit without re-encoding the unit.)
            [c] if c < 0x80
                && let Some(Some(h)) = self.unit_strings.get(usize::from(c))
                && self
                    .heap
                    .get(*h)
                    .and_then(Cell::as_str)
                    .and_then(crate::rope::Rope::as_leaf_bytes)
                    == Some(b) =>
            {
                return *h;
            }
            [c] => return self.unit_string(u16::from(c)),
            [c0 @ 0xC2..=0xC3, c1] => {
                return self.unit_string(u16::from(c0 & 0x1F) << 6 | u16::from(c1 & 0x3F));
            }
            _ => {}
        }
        let key = s.as_ptr() as usize;
        let slot = (key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56;
        let slot = slot as usize % SLOTS;
        if self.literal_ropes.is_empty() {
            self.literal_ropes.resize(SLOTS, None);
        }
        if let Some((k, r, h)) = &self.literal_ropes[slot]
            && *k == key
            && r.as_leaf_bytes() == Some(b)
        {
            if self
                .heap
                .get(*h)
                .and_then(Cell::as_str)
                .is_some_and(|c| c.ptr_eq(r))
            {
                return *h;
            }
            let r = r.clone();
            let h = self.heap.alloc(Cell::Str(r.clone()));
            self.literal_ropes[slot] = Some((key, r, h));
            return h;
        }
        let r = crate::rope::Rope::from(s);
        let h = self.heap.alloc(Cell::Str(r.clone()));
        self.literal_ropes[slot] = Some((key, r, h));
        h
    }

    /// The string concatenation of `parts`, each a string or a primitive
    /// (Number, Boolean, `null`, `undefined`) that is `ToString`ed here — what
    /// `Op::Concat` (a template literal's pieces) computes in one step instead
    /// of one `+` and one new cell per piece. `None` when a part is any other
    /// value (the caller then `+`-adds the parts in order); `Some(None)` when
    /// the result would exceed the string length cap (a `RangeError`).
    ///
    /// Short flat results are written straight into one buffer (joining a lone
    /// high surrogate at a piece's end with a lone low one at the next's start,
    /// exactly as [`Rope::try_concat`](crate::rope::Rope::try_concat) does);
    /// anything longer, or with a `Concat` rope part, is a rope of the parts.
    pub fn concat_parts(&mut self, parts: &[NanBox]) -> Option<Option<Handle>> {
        if let Some(rope) = self.concat_parts_flat(parts) {
            return Some(Some(self.heap.alloc(Cell::Str(rope))));
        }
        // A piece past this many bytes is shared, not copied.
        const FLAT_MAX: usize = 256;
        let mut bound = 0usize;
        let mut flat = true;
        for &p in parts {
            bound += match p.unpack() {
                crate::nanbox::Unpacked::Number(_) => 25,
                crate::nanbox::Unpacked::Bool(_)
                | crate::nanbox::Unpacked::Null
                | crate::nanbox::Unpacked::Undefined => 9,
                crate::nanbox::Unpacked::Handle(raw) => {
                    let r = self.heap.get(Handle::from_raw(raw))?.as_str()?;
                    flat &= r.as_leaf_bytes().is_some();
                    r.len()
                }
            };
            if bound > FLAT_MAX {
                flat = false;
            }
        }
        let rope = if flat {
            let mut buf: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(bound);
            for &p in parts {
                match p.unpack() {
                    crate::nanbox::Unpacked::Number(n) => push_number_bytes(&mut buf, n),
                    crate::nanbox::Unpacked::Handle(raw) => {
                        let piece = self
                            .heap
                            .get(Handle::from_raw(raw))
                            .and_then(Cell::as_str)
                            .and_then(crate::rope::Rope::as_leaf_bytes)
                            .unwrap_or_default();
                        push_wtf8_joined(&mut buf, piece);
                    }
                    _ => buf.extend_from_slice(self.to_display_string(p).as_bytes()),
                }
            }
            crate::rope::Rope::from_wtf8(buf)
        } else {
            let mut acc = crate::rope::Rope::new();
            for &p in parts {
                let piece = match p.unpack() {
                    crate::nanbox::Unpacked::Number(n) => self.number_rope(n),
                    crate::nanbox::Unpacked::Handle(raw) => self
                        .heap
                        .get(Handle::from_raw(raw))
                        .and_then(Cell::as_str)?
                        .clone(),
                    _ => crate::rope::Rope::from(self.to_display_string(p).as_str()),
                };
                match acc.try_concat(&piece) {
                    Some(r) => acc = r,
                    None => return Some(None),
                }
            }
            acc
        };
        Some(Some(self.heap.alloc(Cell::Str(rope))))
    }

    /// [`concat_parts`](Self::concat_parts) of a few parts making a short
    /// result, each a flat string or a primitive with no conversion to run
    /// (Number, Boolean, `null`, `undefined`): their text written once into
    /// a buffer and copied into one new flat string, with its UTF-16 length
    /// recorded when every string part's is known (joining a lone high
    /// surrogate with a lone low one changes the bytes, not the units).
    /// `None`, having done nothing, for anything else.
    fn concat_parts_flat(&self, parts: &[NanBox]) -> Option<crate::rope::Rope> {
        use crate::nanbox::Unpacked;
        use crate::rope::Rope;
        /// The most parts, and the longest result (bytes), served here.
        const PARTS_MAX: usize = 16;
        const FLAT_MAX: usize = 1024;
        /// A result this short is written on the stack.
        const STACK_MAX: usize = 64;
        if parts.len() > PARTS_MAX {
            return None;
        }
        // Each string part's rope (looked up once), and a bound on the
        // result's length (a Number's text is at most 25 bytes).
        let mut ropes: [Option<&Rope>; PARTS_MAX] = [None; PARTS_MAX];
        let mut bound = 0usize;
        for (slot, &p) in ropes.iter_mut().zip(parts) {
            bound += match p.unpack() {
                Unpacked::Number(_) => 25,
                Unpacked::Bool(_) | Unpacked::Null | Unpacked::Undefined => 9,
                Unpacked::Handle(raw) => {
                    let r = self.heap.get(Handle::from_raw(raw))?.as_str()?;
                    *slot = Some(r);
                    r.len()
                }
            };
        }
        if bound > FLAT_MAX {
            return None;
        }
        // (Room past the bound for a Number written as a whole buffer.)
        let fill = |out: &mut [u8]| -> Option<(usize, Option<usize>)> {
            let mut n = 0usize;
            let mut units = Some(0usize);
            for (r, &p) in ropes.iter().zip(parts) {
                let at = n;
                if let Some(r) = r {
                    let b = r.as_leaf_bytes()?;
                    if b.first() == Some(&0xED)
                        && let Some(hi) = crate::wtf8::trailing_high_surrogate(&out[..n])
                        && let Some(lo) = crate::wtf8::leading_low_surrogate(b)
                    {
                        // A lone high surrogate then a lone low one: the
                        // astral code point they denote (four bytes for six).
                        let cp =
                            0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
                        let c = char::from_u32(cp)?;
                        n -= 3;
                        n += c.encode_utf8(&mut out[n..n + 4]).len();
                        out[n..n + b.len() - 3].copy_from_slice(&b[3..]);
                        n += b.len() - 3;
                    } else {
                        out[n..n + b.len()].copy_from_slice(b);
                        n += b.len();
                    }
                    units = units.zip(r.utf16_len_known()).map(|(u, k)| u + k);
                    continue;
                }
                let text: &[u8] = match p.unpack() {
                    Unpacked::Number(x) => {
                        let dst: &mut [u8; super::ryu::NUMBER_TEXT_MAX] = (&mut out
                            [n..n + super::ryu::NUMBER_TEXT_MAX])
                            .try_into()
                            .ok()?;
                        n += super::ryu::write_js_number(x, dst);
                        units = units.map(|u| u + (n - at));
                        continue;
                    }
                    Unpacked::Bool(true) => b"true",
                    Unpacked::Bool(false) => b"false",
                    Unpacked::Null => b"null",
                    Unpacked::Undefined => b"undefined",
                    Unpacked::Handle(_) => return None,
                };
                out[n..n + text.len()].copy_from_slice(text);
                n += text.len();
                units = units.map(|u| u + text.len());
            }
            Some((n, units))
        };
        let make = |bytes: &[u8], units: Option<usize>| match units {
            Some(u) => Rope::from_bytes_units(bytes, u),
            None => Rope::from_bytes(bytes),
        };
        if bound <= STACK_MAX {
            let mut out = [0u8; STACK_MAX + super::ryu::NUMBER_TEXT_MAX];
            let (n, units) = fill(&mut out)?;
            Some(make(&out[..n], units))
        } else {
            let mut out = alloc::vec![0u8; bound + super::ryu::NUMBER_TEXT_MAX];
            let (n, units) = fill(&mut out)?;
            Some(make(&out[..n], units))
        }
    }

    /// `a + b` for a string `a` held as one short leaf and a number `b` (or the
    /// mirror case when `number_first`): the joined text written straight into
    /// one new leaf, with no intermediate string for the number. `None` when
    /// the operands are not of that form.
    pub(super) fn add_leaf_number(
        &self,
        s: NanBox,
        n: NanBox,
        number_first: bool,
    ) -> Option<crate::rope::Rope> {
        let n = n.as_number()?;
        let rope = self.heap.get(Handle::from_raw(s.as_handle()?))?.as_str()?;
        let bytes = rope.as_leaf_bytes()?;
        if bytes.len() > 40 || bytes.is_empty() {
            return None;
        }
        let mut buf = [0u8; super::ryu::NUMBER_TEXT_MAX];
        let len = super::ryu::write_js_number(n, &mut buf);
        let digits = &buf[..len];
        // At most 40 + 25 bytes: joined on the stack, then copied once into the
        // new rope (inline when short).
        let mut out = [0u8; 64];
        let (first, second) = if number_first {
            (digits, bytes)
        } else {
            (bytes, digits)
        };
        out[..first.len()].copy_from_slice(first);
        out[first.len()..first.len() + second.len()].copy_from_slice(second);
        Some(crate::rope::Rope::from_bytes(
            &out[..first.len() + second.len()],
        ))
    }
}

/// Appends the WTF-8 `piece` to `buf`, pairing a lone high surrogate at the
/// end of `buf` with a lone low surrogate at the start of `piece` into the one
/// astral code point they denote, so the result stays canonical WTF-8.
fn push_wtf8_joined(buf: &mut alloc::vec::Vec<u8>, piece: &[u8]) {
    if let Some(hi) = crate::wtf8::trailing_high_surrogate(buf)
        && let Some(lo) = crate::wtf8::leading_low_surrogate(piece)
    {
        buf.truncate(buf.len() - 3);
        let cp = 0x1_0000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
        crate::wtf8::encode_code_point(cp, buf);
        buf.extend_from_slice(&piece[3..]);
    } else {
        buf.extend_from_slice(piece);
    }
}

/// The layout memo of [`Realm::new_array_with_named`]: the root shape it was
/// derived from, and the shape the auxiliary object reaches from it.
#[derive(Default)]
pub struct AuxShapeCache(Option<(Rc<Shape>, Rc<Shape>)>);
