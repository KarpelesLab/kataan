//! Narrow, read-only views of the heap for the built-in library's fast paths
//! (`JSON.stringify` and friends): each answers one question exactly or says
//! "don't know", so a caller can skip the generic `[[Get]]` machinery only when
//! the answer is provably the same.

use super::{Cell, Handle, NanBox, Realm};
use crate::shape::Shape;
use alloc::rc::Rc;

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
    /// ordinary object, an array, a RegExp or a collection, and for a
    /// host-exotic one.
    fn key_link(&self, h: Handle) -> Option<(Option<u64>, u32, Option<Handle>)> {
        if self.is_host_exotic(h) {
            return None;
        }
        match self.heap.get(h)? {
            Cell::Object(o) => Some((None, o.key_epoch(), o.proto())),
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
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
    /// object, an array, a RegExp or a Map/Set (or is host-exotic), whose own
    /// properties this cannot rule out. `recv_shape` memoizes one receiver
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
            Cell::Array(_)
            | Cell::RegExp { .. }
            | Cell::Collection { .. }
            | Cell::Native(_)
            | Cell::VmFunction { .. } => {
                if let Some(a) = self.aux_props.get(&h.to_raw())
                    && self.heap.get(*a)?.as_object()?.has_own_key(key)
                {
                    return None;
                }
                Some(self.object_proto(h))
            }
            _ => None,
        }
    }

    /// The object holding the named properties of the chain link `h`: `h`
    /// itself for an ordinary object, else its auxiliary object.
    fn named_props(&self, h: Handle) -> Option<&crate::object::Object> {
        match self.heap.get(h)? {
            Cell::Object(o) => Some(o),
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
                self.heap
                    .get(*self.aux_props.get(&h.to_raw())?)?
                    .as_object()
            }
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

    /// The one-receiver fast form of [`method_site_get`](Self::method_site_get):
    /// whether `cache` proves that `recv[key]` is the function cell `method`
    /// because `recv` is the Map/Set it noted at the current prototype epoch
    /// ([`method_site_note_recv`](Self::method_site_note_recv)), with every
    /// link unchanged since and the holder's slot still `method`. `false`
    /// sends the caller to the general check.
    #[inline]
    pub(crate) fn method_site_memo_hit(
        &self,
        recv: Handle,
        cache: &MethodSiteCache,
        method: u64,
    ) -> bool {
        let epoch = crate::object::proto_epoch();
        if !cache.armed
            || cache.prim
            || cache.own
            || cache.recv_memo != recv.to_raw()
            || cache.recv_stamp != epoch
            || cache.stamp != epoch
            || cache.swaps != self.cell_swaps
        {
            return false;
        }
        // The holder's own value (a store to an existing data property does
        // not move the epoch): read under its recorded shape.
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
    pub fn method_site_note_recv(&mut self, recv: Handle, cache: &mut MethodSiteCache) {
        let epoch = crate::object::proto_epoch();
        if cache.own
            || cache.prim
            || cache.stamp != epoch
            || (cache.recv_stamp == epoch && cache.recv_memo == recv.to_raw())
            || !matches!(self.heap.get(recv), Some(Cell::Collection { .. }))
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
                for i in 0..cache.links.len() {
                    let (raw, aux, _) = cache.links[i];
                    self.watch_link(raw, aux);
                }
                cache.stamp = crate::object::proto_epoch();
                return Some(v);
            }
            cur = next;
        }
        None
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
    fn link_watched(&self, raw: u64, aux: Option<u64>) -> bool {
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
            Cell::Array(_) | Cell::RegExp { .. } | Cell::Collection { .. } | Cell::Native(_) => {
                self.heap
                    .get(*self.aux_props.get(&h.to_raw())?)?
                    .as_object()?
            }
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
/// per lookup the holder (raw handle), its shape word, the slot and the value
/// found there. Compiled code re-reads both slots and compares the values (a
/// value is no part of what the epoch watches). It never outlives the memo
/// entry it copies (every change to the memo empties it), whose `Rc`s keep
/// the two shapes alive. `stamp == u64::MAX`: empty.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ArrayIterJit {
    pub(crate) stamp: u64,
    pub(crate) iter_holder: u64,
    pub(crate) iter_shape: u64,
    pub(crate) iter_slot: u64,
    pub(crate) iter_fn: u64,
    pub(crate) next_holder: u64,
    pub(crate) next_shape: u64,
    pub(crate) next_slot: u64,
    pub(crate) next_fn: u64,
}

impl Default for ArrayIterJit {
    fn default() -> Self {
        Self {
            stamp: u64::MAX,
            iter_holder: 0,
            iter_shape: 0,
            iter_slot: 0,
            iter_fn: 0,
            next_holder: 0,
            next_shape: 0,
            next_slot: 0,
            next_fn: 0,
        }
    }
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

/// The decimal digits of `Number::toString(n)` when `n` is an integer of
/// magnitude below 2^53 (`-0` reads as `0`): a buffer and the start of the
/// text in it. `None` for any other number.
fn small_integer_digits(n: f64) -> Option<([u8; 21], usize)> {
    const LIMIT: f64 = 9_007_199_254_740_992.0;
    if !(n > -LIMIT && n < LIMIT) {
        return None;
    }
    let i = n as i64;
    if i as f64 != n {
        return None;
    }
    let mut buf = [0u8; 21];
    let mut pos = buf.len();
    let mut u = i.unsigned_abs();
    loop {
        pos -= 1;
        buf[pos] = b'0' + (u % 10) as u8;
        u /= 10;
        if u == 0 {
            break;
        }
    }
    if i < 0 {
        pos -= 1;
        buf[pos] = b'-';
    }
    Some((buf, pos))
}

/// Appends `Number::toString(n)` to `out` when `n` is an integer of magnitude
/// below 2^53 (written as plain decimal digits, `-0` as `0`) and returns
/// `true`; returns `false`, having written nothing, for any other number.
pub(crate) fn push_small_integer(out: &mut alloc::string::String, n: f64) -> bool {
    let Some((buf, pos)) = small_integer_digits(n) else {
        return false;
    };
    // Digits are ASCII, so the slice is valid UTF-8; one reserve-and-copy.
    out.push_str(core::str::from_utf8(&buf[pos..]).unwrap_or_default());
    true
}

/// Appends `Number::toString(n)` to the WTF-8 buffer `out` — the digits
/// directly for a small integer, else the general rendering.
pub(crate) fn push_number_bytes(out: &mut alloc::vec::Vec<u8>, n: f64) {
    if let Some((buf, pos)) = small_integer_digits(n) {
        out.extend_from_slice(&buf[pos..]);
    } else {
        out.extend_from_slice(super::js_number_string(n).as_bytes());
    }
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
                .get_or_insert_with(|| crate::rope::Rope::from(super::js_number_string(n).as_str()))
                .clone();
        }
        crate::rope::Rope::from(super::js_number_string(n).as_str())
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
                    crate::nanbox::Unpacked::Number(n) => match small_integer_digits(n) {
                        Some((d, pos)) => buf.extend_from_slice(&d[pos..]),
                        None => buf.extend_from_slice(super::js_number_string(n).as_bytes()),
                    },
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
        let (buf, pos) = small_integer_digits(n)?;
        let digits = &buf[pos..];
        // At most 40 + 21 bytes: joined on the stack, then copied once into the
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
