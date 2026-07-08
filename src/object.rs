//! The performance-era object: a [`Shape`] (hidden class) paired with a flat
//! vector of [`NanBox`] value slots (`ROADMAP.md` §3).
//!
//! This composes the three object-model pillars. An object holds:
//! - a shared, immutable [`Shape`] describing *where* each property lives, and
//! - a dense `Vec<NanBox>` holding the property *values* by slot index.
//!
//! Reading a property is a shape lookup (cacheable on the shape pointer) plus a
//! slot load; adding one transitions the shape and pushes a slot. Objects of the
//! same structure share a shape, so the per-object cost is just the value
//! vector — and a handle into a [`Heap`] is how other values point at it.
//!
//! An object that accumulates more own properties than the realm's
//! `object_dictionary_threshold` switches, behind this same method API, from the
//! shaped representation to a **dictionary** (`ObjectData::Dict`): an
//! insertion-ordered map that creates no further shape transitions. This bounds
//! the shape transition-tree's growth for programs that pile up unbounded unique
//! keys (MEM-3), at the cost of the per-shape inline-cache fast path for those
//! (now-atypical) objects.
//!
//! [`Shape`]: crate::shape::Shape
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Heap`]: crate::heap::Heap
//!
//! Pure, safe `alloc`-only Rust; this is the representation the bytecode VM will
//! migrate onto once the GC that manages the heap lands.

use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec::Vec;

/// The numeric value of `k` if it is a canonical array-index key (a non-negative
/// integer `< 2^32 - 1` whose decimal form is exactly `k`), else `None`.
fn array_index(k: &str) -> Option<u32> {
    k.parse::<u32>()
        .ok()
        .filter(|n| *n < u32::MAX && n.to_string() == k)
}

/// The internal storage representation of an [`Object`]'s data properties.
///
/// Objects start in `Shaped` mode — a shared hidden-class
/// [`Shape`] plus a dense slot vector — which is the zero-overhead representation
/// for normal objects whose structure is shared across many instances. An object
/// that accumulates more own properties than the realm's
/// `object_dictionary_threshold` converts in place to
/// `Dict` mode, an insertion-ordered map that adds **no**
/// shape transitions: this bounds the shape transition-tree's growth for
/// programs that pile up unbounded unique keys (MEM-3).
#[repr(u8)] // JIT inline fast path bakes the Shaped discriminant + field offsets; verified
enum ObjectData {
    /// Default representation: a hidden-class shape and a dense value vector
    /// indexed by the shape's slot numbers.
    Shaped {
        shape: Rc<Shape>,
        slots: Vec<NanBox>,
    },
    /// Dictionary representation: values keyed by name, with a parallel list
    /// preserving insertion order. Carries no live `Rc<Shape>`, so it creates no
    /// transitions; `order` and `map` are kept in sync (every key in `order` is
    /// a key in `map` and vice versa).
    Dict {
        order: Vec<Box<str>>,
        map: BTreeMap<Box<str>, NanBox>,
    },
}

/// A property-bearing object: a hidden-class shape plus its value slots (or a
/// dictionary once it grows past the threshold), with an optional side list of
/// accessor (getter/setter) properties.
#[repr(C)] // JIT inline fast path bakes `data` at offset 0; verified in jit layout test
pub struct Object {
    /// The data-property storage: shaped (default) or dictionary mode.
    data: ObjectData,
    /// An empty sentinel shape returned by [`shape`](Object::shape) while in
    /// dictionary mode, so inline caches keyed on the shape pointer always miss
    /// (an empty shape resolves no key) and never bind a dictionary object.
    dict_shape: Option<Rc<Shape>>,
    /// Accessor properties: `(name, getter, setter)`, both held as value
    /// handles (`undefined` when absent). Kept out of the shape's slot layout.
    accessors: Vec<(alloc::boxed::Box<str>, NanBox, NanBox)>,
    /// Own keys that are **non-enumerable** (e.g. class methods): present in the
    /// slots and readable, but hidden from `Object.keys`/spread/`for-in`/JSON.
    hidden: Vec<alloc::boxed::Box<str>>,
    /// Own keys that are **non-writable** (`defineProperty` with
    /// `writable: false`): writes are silently ignored.
    readonly: Vec<alloc::boxed::Box<str>>,
    /// Own keys that are **non-configurable** (`defineProperty` with
    /// `configurable: false`): they cannot be deleted.
    non_configurable: Vec<alloc::boxed::Box<str>>,
    /// Whether the object is frozen (`Object.freeze`): no new properties and no
    /// writes to existing ones.
    frozen: bool,
    /// Whether new properties may be added (`Object.preventExtensions` clears it).
    extensible: bool,
    /// Whether the object is sealed (`Object.seal`): no new properties and no
    /// deletions, but existing writable properties may still change.
    sealed: bool,
    /// The class this object was instantiated from (for `instanceof`), if any.
    class_tag: Option<u32>,
    /// The `[[Prototype]]` link (`Object.create`/`getPrototypeOf`), if any. A
    /// property miss walks this chain.
    proto: Option<crate::heap::Handle>,
}

impl Object {
    /// Creates an empty object whose layout starts at `root` (the shared root
    /// shape of the owning realm/heap, so identically-structured objects share
    /// shapes).
    #[must_use]
    pub fn new(root: Rc<Shape>) -> Self {
        Self {
            data: ObjectData::Shaped {
                shape: root,
                slots: Vec::new(),
            },
            dict_shape: None,
            accessors: Vec::new(),
            hidden: Vec::new(),
            readonly: Vec::new(),
            non_configurable: Vec::new(),
            frozen: false,
            extensible: true,
            sealed: false,
            class_tag: None,
            proto: None,
        }
    }

    /// The `[[Prototype]]` handle, if any.
    #[must_use]
    pub fn proto(&self) -> Option<crate::heap::Handle> {
        self.proto
    }

    /// Sets the `[[Prototype]]` link (`None` clears it to a null prototype).
    pub fn set_proto(&mut self, proto: Option<crate::heap::Handle>) {
        self.proto = proto;
    }

    /// Tags this object with the class it was constructed from.
    pub fn set_class_tag(&mut self, class_id: u32) {
        self.class_tag = Some(class_id);
    }

    /// The class this object was constructed from, if any.
    #[must_use]
    pub fn class_tag(&self) -> Option<u32> {
        self.class_tag
    }

    /// Defines an accessor property `name` with `getter`/`setter` (either may be
    /// `undefined`). Replaces an existing accessor of the same name.
    pub fn define_accessor(&mut self, name: &str, getter: NanBox, setter: NanBox) {
        if let Some(a) = self
            .accessors
            .iter_mut()
            .find(|(k, _, _)| k.as_ref() == name)
        {
            if !matches!(getter.unpack(), crate::nanbox::Unpacked::Undefined) {
                a.1 = getter;
            }
            if !matches!(setter.unpack(), crate::nanbox::Unpacked::Undefined) {
                a.2 = setter;
            }
        } else {
            self.accessors
                .push((alloc::boxed::Box::from(name), getter, setter));
        }
    }

    /// The names of this object's accessor (getter/setter) properties.
    #[must_use]
    pub fn accessor_keys(&self) -> Vec<&str> {
        self.accessors.iter().map(|(k, _, _)| k.as_ref()).collect()
    }

    /// The `(getter, setter)` of accessor `name`, if defined.
    #[must_use]
    pub fn accessor(&self, name: &str) -> Option<(NanBox, NanBox)> {
        self.accessors
            .iter()
            .find(|(k, _, _)| k.as_ref() == name)
            .map(|(_, g, s)| (*g, *s))
    }

    /// The object's current shape (its hidden class). In dictionary mode this is
    /// an empty sentinel shape, so any inline cache keyed on the returned pointer
    /// resolves no key and misses — a dictionary object never binds an IC.
    #[must_use]
    pub fn shape(&self) -> &Rc<Shape> {
        match &self.data {
            ObjectData::Shaped { shape, .. } => shape,
            ObjectData::Dict { .. } => self
                .dict_shape
                .as_ref()
                .expect("dictionary objects carry a sentinel shape"),
        }
    }

    /// The number of own data properties (excludes side-list accessors).
    #[must_use]
    pub fn len(&self) -> u32 {
        match &self.data {
            ObjectData::Shaped { shape, .. } => shape.len(),
            ObjectData::Dict { order, .. } => order.len() as u32,
        }
    }

    /// Whether the object has no own data properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.data {
            ObjectData::Shaped { shape, .. } => shape.is_empty(),
            ObjectData::Dict { order, .. } => order.is_empty(),
        }
    }

    /// The value of own property `key`, or `None` if absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<NanBox> {
        match &self.data {
            ObjectData::Shaped { shape, slots } => {
                let slot = shape.lookup(key)?;
                slots.get(slot as usize).copied()
            }
            ObjectData::Dict { map, .. } => map.get(key).copied(),
        }
    }

    /// Reads slot `slot` directly, without a name lookup — the inline-cache fast
    /// path. Returns `None` if the object is in dictionary mode (which has no slot
    /// vector) or the index is out of range. The caller must have obtained `slot`
    /// from a cache armed on *this object's current shape* (see
    /// [`cached_get`](Object::cached_get)); a slot from a stale shape would read a
    /// different property.
    #[must_use]
    pub fn slot(&self, slot: u32) -> Option<NanBox> {
        match &self.data {
            ObjectData::Shaped { slots, .. } => slots.get(slot as usize).copied(),
            ObjectData::Dict { .. } => None,
        }
    }

    /// Reads own data property `key`, consulting `cache` so a repeat access on the
    /// same shape skips the name lookup (a shape-pointer compare plus a slot
    /// load). Returns `None` if the property is absent (a cache miss).
    ///
    /// This is the bytecode VM's `GetProp` fast path for a plain shaped object. A
    /// dictionary-mode object resolves through its empty sentinel shape, so the
    /// cache always misses and this falls back to a normal lookup — never binding
    /// a dictionary to the cache. Accessors live outside the slot layout and are
    /// the caller's responsibility (checked before this is reached), so a hit here
    /// is always a genuine own data slot.
    pub fn cached_get(&self, key: &str, cache: &mut crate::ic::PropertyCache) -> Option<NanBox> {
        match &self.data {
            ObjectData::Shaped { shape, slots } => {
                let slot = cache.lookup(shape, key)?;
                slots.get(slot as usize).copied()
            }
            // Dictionary mode carries an empty sentinel shape; route around the
            // cache entirely so it never binds (and never goes stale on a later
            // dictionary mutation that does not change the sentinel pointer).
            ObjectData::Dict { map, .. } => map.get(key).copied(),
        }
    }

    /// Writes `value` to own data property `key` *in place*, consulting `cache`,
    /// and reports whether the write happened. Returns `false` when `key` is not
    /// an existing own data property (a new property is a shape transition and must
    /// go through [`set`](Object::set)'s slow path), when the object is in
    /// dictionary mode, or when the property is frozen/read-only — in every such
    /// case the caller falls back to the slow path.
    ///
    /// This only ever rewrites a slot that already exists on the current shape, so
    /// it performs no transition and the cache stays valid; a transition would
    /// produce a new shape pointer that simply misses next time.
    pub fn cached_set(
        &mut self,
        key: &str,
        value: NanBox,
        cache: &mut crate::ic::PropertyCache,
    ) -> bool {
        if self.frozen || self.is_readonly(key) {
            return false;
        }
        match &mut self.data {
            ObjectData::Shaped { shape, slots } => match cache.lookup(shape, key) {
                Some(slot) => match slots.get_mut(slot as usize) {
                    Some(cell) => {
                        *cell = value;
                        true
                    }
                    None => false,
                },
                // Absent: a new property (transition) — let the slow path add it.
                None => false,
            },
            ObjectData::Dict { .. } => false,
        }
    }

    /// Whether the object has own property `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        match &self.data {
            ObjectData::Shaped { shape, .. } => shape.contains(key),
            ObjectData::Dict { map, .. } => map.contains_key(key),
        }
    }

    /// Sets own property `key` to `value`: updates the value in place if the
    /// property exists, otherwise adds it. Works transparently in either storage
    /// mode (shaped or dictionary). A no-op on a frozen object (matching
    /// `Object.freeze` semantics in non-strict code).
    ///
    /// Conversion to dictionary mode is driven by the realm via
    /// [`maybe_convert_to_dict`](Object::maybe_convert_to_dict), called just
    /// before adding a new property; once converted, this method appends into the
    /// dictionary and creates no shape transitions.
    pub fn set(&mut self, key: &str, value: NanBox) {
        if self.frozen || self.is_readonly(key) {
            return;
        }
        match &mut self.data {
            ObjectData::Shaped { shape, slots } => {
                if let Some(slot) = shape.lookup(key) {
                    slots[slot as usize] = value;
                } else if self.extensible {
                    *shape = shape.transition(key);
                    slots.push(value);
                }
                // A non-extensible object silently ignores new keys.
            }
            ObjectData::Dict { order, map } => {
                if let Some(v) = map.get_mut(key) {
                    *v = value;
                } else if self.extensible {
                    order.push(Box::from(key));
                    map.insert(Box::from(key), value);
                }
            }
        }
    }

    /// Like [`set`](Object::set) but bypasses the extensibility / frozen /
    /// readonly guards: it always stores (creating a new slot if needed). Used by
    /// `[[DefineOwnProperty]]`, which validates configurability/extensibility
    /// itself and must be able to materialize a data slot even on a non-extensible
    /// object (e.g. converting an existing accessor property to a data property, or
    /// changing the value of a configurable-but-non-extensible property).
    pub fn force_set(&mut self, key: &str, value: NanBox) {
        match &mut self.data {
            ObjectData::Shaped { shape, slots } => {
                if let Some(slot) = shape.lookup(key) {
                    slots[slot as usize] = value;
                } else {
                    *shape = shape.transition(key);
                    slots.push(value);
                }
            }
            ObjectData::Dict { order, map } => {
                if let Some(v) = map.get_mut(key) {
                    *v = value;
                } else {
                    order.push(Box::from(key));
                    map.insert(Box::from(key), value);
                }
            }
        }
    }

    /// If adding `key` would be a *new* own property that pushes the own-data
    /// count past `threshold`, converts the object to dictionary mode in place so
    /// the subsequent [`set`](Object::set) — and every later add — creates no
    /// shape transitions. A no-op if `key` already exists, the object is already a
    /// dictionary, the object is non-extensible/frozen, or the count is still
    /// within the threshold. The realm calls this immediately before
    /// [`set`](Object::set) on a property add (MEM-3: bounds shape-tree growth).
    pub fn maybe_convert_to_dict(&mut self, key: &str, threshold: usize) {
        if let ObjectData::Shaped { shape, .. } = &self.data
            && self.extensible
            && !self.frozen
            && !self.is_readonly(key)
            && shape.lookup(key).is_none()
            && shape.len() as usize >= threshold
        {
            self.convert_to_dict();
        }
    }

    /// Converts a shaped object to dictionary mode in place, preserving the
    /// current properties in insertion order and dropping the live `Rc<Shape>`
    /// so no further transitions are created. A no-op if already a dictionary.
    fn convert_to_dict(&mut self) {
        let ObjectData::Shaped { shape, slots } = &self.data else {
            return;
        };
        let keys = shape.keys();
        let mut order: Vec<Box<str>> = Vec::with_capacity(keys.len());
        let mut map: BTreeMap<Box<str>, NanBox> = BTreeMap::new();
        for k in keys {
            let slot = shape.lookup(k).expect("shape key resolves");
            let v = slots[slot as usize];
            order.push(Box::from(k));
            map.insert(Box::from(k), v);
        }
        // A fresh empty shape so `shape()` returns a pointer that resolves no
        // key (inline caches keyed on it always miss for this object).
        self.dict_shape = Some(Shape::root());
        self.data = ObjectData::Dict { order, map };
    }

    /// The own property names, in insertion order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        match &self.data {
            ObjectData::Shaped { shape, .. } => shape.keys(),
            ObjectData::Dict { order, .. } => order.iter().map(Box::as_ref).collect(),
        }
    }

    /// All own property names — data and accessor, including non-enumerable — in
    /// insertion order. For reflection that ignores enumerability (e.g.
    /// `Object.getOwnPropertySymbols`).
    #[must_use]
    pub fn all_keys(&self) -> Vec<&str> {
        let mut keys = self.keys();
        keys.extend(self.accessors.iter().map(|(k, _, _)| k.as_ref()));
        keys
    }

    /// All own property names (data + accessor, **including** non-enumerable) in spec
    /// `[[OwnPropertyKeys]]` order: integer-index keys ascending, then the rest in
    /// insertion order. Used by `getOwnPropertyNames` / `Reflect.ownKeys`.
    #[must_use]
    pub fn ordered_keys(&self) -> Vec<&str> {
        let keys = self.keys();
        let mut ints: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| array_index(k).is_some())
            .collect();
        ints.sort_by_key(|k| array_index(k).unwrap());
        let strs = keys.iter().copied().filter(|k| array_index(k).is_none());
        let acc = self.accessors.iter().map(|(k, _, _)| k.as_ref());
        ints.into_iter().chain(strs).chain(acc).collect()
    }

    /// The own **enumerable** property names (excludes keys marked hidden), in
    /// spec order: integer-index keys ascending, then the rest in insertion order.
    #[must_use]
    pub fn enumerable_keys(&self) -> Vec<&str> {
        let keys: Vec<&str> = self
            .keys()
            .into_iter()
            .filter(|k| !self.is_hidden(k))
            .collect();
        let mut ints: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| array_index(k).is_some())
            .collect();
        ints.sort_by_key(|k| array_index(k).unwrap());
        let strs = keys.into_iter().filter(|k| array_index(k).is_none());
        // Enumerable accessor (getter/setter) properties live outside the shape;
        // include those not marked hidden, after the data keys.
        let acc = self
            .accessors
            .iter()
            .map(|(k, _, _)| k.as_ref())
            .filter(|k| !self.is_hidden(k));
        ints.into_iter().chain(strs).chain(acc).collect()
    }

    /// Marks own property `key` non-enumerable (idempotent).
    pub fn set_hidden(&mut self, key: &str) {
        if !self.is_hidden(key) {
            self.hidden.push(alloc::boxed::Box::from(key));
        }
    }

    /// Whether own property `key` is non-enumerable.
    #[must_use]
    pub fn is_hidden(&self, key: &str) -> bool {
        self.hidden.iter().any(|k| k.as_ref() == key)
    }

    /// Clears the non-enumerable mark for `key` (a `defineProperty` redefining a
    /// configurable property to `enumerable: true`).
    pub fn clear_hidden(&mut self, key: &str) {
        self.hidden.retain(|k| k.as_ref() != key);
    }

    /// Marks own property `key` non-writable (idempotent).
    pub fn set_readonly(&mut self, key: &str) {
        if !self.is_readonly(key) {
            self.readonly.push(alloc::boxed::Box::from(key));
        }
    }

    /// Whether own property `key` is non-writable.
    #[must_use]
    pub fn is_readonly(&self, key: &str) -> bool {
        self.readonly.iter().any(|k| k.as_ref() == key)
    }

    /// Clears the non-writable mark for `key` (a `defineProperty` redefines
    /// attributes from scratch, so the prior `writable: false` is dropped first).
    pub fn clear_readonly(&mut self, key: &str) {
        self.readonly.retain(|k| k.as_ref() != key);
    }

    /// Marks own property `key` non-configurable (idempotent).
    pub fn set_non_configurable(&mut self, key: &str) {
        if !self.is_non_configurable(key) {
            self.non_configurable.push(alloc::boxed::Box::from(key));
        }
    }

    /// Whether own property `key` is non-configurable (cannot be deleted).
    #[must_use]
    pub fn is_non_configurable(&self, key: &str) -> bool {
        self.non_configurable.iter().any(|k| k.as_ref() == key)
    }

    /// Clears the non-configurable mark for `key` (a `defineProperty` redefining a
    /// configurable property whose new descriptor keeps it configurable).
    pub fn clear_non_configurable(&mut self, key: &str) {
        self.non_configurable.retain(|k| k.as_ref() != key);
    }

    /// Whether *no* own property is marked non-configurable — a cheap probe used to
    /// skip the per-index ArraySetLength stop scan on the common (unrestricted) array.
    #[must_use]
    pub fn non_configurable_is_empty(&self) -> bool {
        self.non_configurable.is_empty()
    }

    /// Whether `key` is an own property (data slot or accessor).
    #[must_use]
    pub fn has_own_key(&self, key: &str) -> bool {
        self.contains(key) || self.accessors.iter().any(|(k, _, _)| k.as_ref() == key)
    }

    /// Marks the object frozen (`Object.freeze`) — implies sealed + non-extensible.
    pub fn freeze(&mut self) {
        self.frozen = true;
        self.sealed = true;
        self.extensible = false;
    }

    /// Prevents new properties (`Object.preventExtensions`).
    pub fn prevent_extensions(&mut self) {
        self.extensible = false;
    }

    /// Seals the object (`Object.seal`): no new props, no deletions.
    pub fn seal(&mut self) {
        self.sealed = true;
        self.extensible = false;
    }

    /// Whether new properties may be added.
    #[must_use]
    pub fn is_extensible(&self) -> bool {
        self.extensible
    }

    /// Whether the object is sealed (or frozen).
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed || self.frozen
    }

    /// Whether the object is frozen.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Removes any accessor (getter/setter) for `key`, leaving data properties
    /// intact — used when a `defineProperty` replaces an accessor with data.
    pub fn clear_accessor(&mut self, key: &str) {
        self.accessors.retain(|(k, _, _)| k.as_ref() != key);
    }

    /// Deletes only the *data slot* for `key`, rebuilding the shape/slots from
    /// `root` without it, and leaving any same-named accessor untouched. Used when
    /// `defineProperty` converts a data property into an accessor.
    pub fn delete_data(&mut self, root: Rc<Shape>, key: &str) {
        match &mut self.data {
            ObjectData::Shaped { shape, slots } => {
                if !shape.contains(key) {
                    return;
                }
                let kept: Vec<(alloc::string::String, NanBox)> = shape
                    .keys()
                    .into_iter()
                    .filter(|k| *k != key)
                    .map(|k| {
                        let slot = shape.lookup(k).expect("shape key resolves");
                        (alloc::string::String::from(k), slots[slot as usize])
                    })
                    .collect();
                let mut new_shape = root;
                let mut new_slots = Vec::with_capacity(kept.len());
                for (k, v) in kept {
                    new_shape = new_shape.transition(&k);
                    new_slots.push(v);
                }
                *shape = new_shape;
                *slots = new_slots;
            }
            ObjectData::Dict { order, map } => {
                if map.remove(key).is_some() {
                    order.retain(|k| k.as_ref() != key);
                }
            }
        }
    }

    /// Deletes own property `key`, rebuilding the shape/slots from `root` without
    /// it (also drops a same-named accessor). Returns whether anything was
    /// removed.
    pub fn delete(&mut self, root: Rc<Shape>, key: &str) -> bool {
        let had_accessor = self.accessors.iter().any(|(k, _, _)| k.as_ref() == key);
        self.accessors.retain(|(k, _, _)| k.as_ref() != key);
        // Clear the property's attribute flags so a later re-add of the same key
        // starts from defaults — otherwise a stale `readonly` flag makes `set` a
        // silent no-op, and a stale `hidden`/`non_configurable` flag wrongly
        // carries over to the new property (e.g. re-defining a deleted, formerly
        // non-writable `name`/`length`, or NamedEvaluation re-installing `name`).
        self.hidden.retain(|k| k.as_ref() != key);
        self.readonly.retain(|k| k.as_ref() != key);
        self.non_configurable.retain(|k| k.as_ref() != key);
        match &mut self.data {
            ObjectData::Shaped { shape, slots } => {
                if !shape.contains(key) {
                    return had_accessor;
                }
                let kept: Vec<(alloc::string::String, NanBox)> = shape
                    .keys()
                    .into_iter()
                    .filter(|k| *k != key)
                    .map(|k| {
                        let slot = shape.lookup(k).expect("shape key resolves");
                        (alloc::string::String::from(k), slots[slot as usize])
                    })
                    .collect();
                let mut new_shape = root;
                let mut new_slots = Vec::with_capacity(kept.len());
                for (k, v) in kept {
                    new_shape = new_shape.transition(&k);
                    new_slots.push(v);
                }
                *shape = new_shape;
                *slots = new_slots;
                true
            }
            ObjectData::Dict { order, map } => {
                if map.remove(key).is_none() {
                    return had_accessor;
                }
                order.retain(|k| k.as_ref() != key);
                true
            }
        }
    }

    /// Rewrites every outgoing handle through `forward` — the mutating mirror of
    /// [`trace_handles`](Object::trace_handles), used by a moving collector to
    /// fix up references after relocation.
    pub fn relocate_handles(
        &mut self,
        forward: &dyn Fn(crate::heap::Handle) -> crate::heap::Handle,
    ) {
        let fwd = |v: &mut NanBox| {
            if let Some(raw) = v.as_handle() {
                *v = NanBox::handle(forward(crate::heap::Handle::from_raw(raw)).to_raw());
            }
        };
        match &mut self.data {
            ObjectData::Shaped { slots, .. } => {
                for slot in slots {
                    fwd(slot);
                }
            }
            ObjectData::Dict { map, .. } => {
                for v in map.values_mut() {
                    fwd(v);
                }
            }
        }
        for (_, g, s) in &mut self.accessors {
            fwd(g);
            fwd(s);
        }
        if let Some(p) = self.proto {
            self.proto = Some(forward(p));
        }
    }

    /// Calls `visit` for every heap [`Handle`](crate::heap::Handle) this object
    /// references through a slot — the outgoing edges a tracing collector
    /// follows.
    pub fn trace_handles(&self, mut visit: impl FnMut(crate::heap::Handle)) {
        let trace_value = |v: &NanBox, visit: &mut dyn FnMut(crate::heap::Handle)| {
            if let Some(raw) = v.as_handle() {
                visit(crate::heap::Handle::from_raw(raw));
            }
        };
        match &self.data {
            ObjectData::Shaped { slots, .. } => {
                for slot in slots {
                    trace_value(slot, &mut visit);
                }
            }
            ObjectData::Dict { map, .. } => {
                for v in map.values() {
                    trace_value(v, &mut visit);
                }
            }
        }
        for (_, g, s) in &self.accessors {
            for v in [g, s] {
                if let Some(raw) = v.as_handle() {
                    visit(crate::heap::Handle::from_raw(raw));
                }
            }
        }
        if let Some(p) = self.proto {
            visit(p);
        }
    }
}

impl crate::gc::Trace for Object {
    fn trace(&self, visit: &mut dyn FnMut(crate::heap::Handle)) {
        self.trace_handles(visit);
    }
}

/// Runtime-probed byte layout of [`Object`] / `ObjectData::Shaped` (and the inner
/// `Vec<NanBox>` slot store) for the generic-JIT inline property-get fast path.
/// Every field is derived by pointer arithmetic on a real, populated shaped
/// object (never hand-baked); the JIT layout verification test cross-checks each
/// against a safe read, so a layout drift fails loudly instead of corrupting a
/// raw memory read.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ObjectLayout {
    /// Byte offset of the `data: ObjectData` field within `Object` (repr(C) → 0).
    pub data_off: usize,
    /// The `ObjectData::Shaped` discriminant byte value (tag at offset 0).
    pub shaped_disc: u8,
    /// Byte offset of the `shape: Rc<Shape>` field within `ObjectData`. The 8-byte
    /// word there is the `Rc`'s stored `NonNull<RcBox>` (the shape identity the IC
    /// compares against).
    pub shape_off: usize,
    /// Byte offset of the `slots: Vec<NanBox>` field within `ObjectData`.
    pub slots_off: usize,
    /// Byte offset of the data pointer within a `Vec<NanBox>`.
    pub vec_ptr_off: usize,
    /// Byte offset of the length within a `Vec<NanBox>`.
    pub vec_len_off: usize,
}

/// Derives [`ObjectLayout`] from a real shaped object with several own data
/// properties (so the slot `Vec` is populated, its length distinct from its
/// capacity and data pointer, making the by-value word scan unambiguous).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn jit_object_layout() -> ObjectLayout {
    let mut obj = Object::new(Shape::root());
    // Three properties: len == 3, cap != 3 (first growth → 4), ptr is a heap
    // address — so exactly one Vec word equals 3 (the length).
    obj.set("a", NanBox::number(1.0));
    obj.set("b", NanBox::number(2.0));
    obj.set("c", NanBox::number(3.0));

    let obj_base = core::ptr::addr_of!(obj) as usize;
    let data_off = core::ptr::addr_of!(obj.data) as usize - obj_base;
    let od = &obj.data;
    let od_base = core::ptr::addr_of!(*od) as usize;
    // SAFETY: `od` is a live, initialized `ObjectData`; its repr(u8) tag byte is
    // in bounds and always a valid `u8`.
    #[allow(unsafe_code)]
    let shaped_disc = unsafe { *(od_base as *const u8) };
    let ObjectData::Shaped { shape, slots } = od else {
        unreachable!("a fresh object is in Shaped mode");
    };
    let shape_off = core::ptr::addr_of!(*shape) as usize - od_base;
    let slots_off = core::ptr::addr_of!(*slots) as usize - od_base;

    // Locate the data-pointer and length words inside the `Vec<NanBox>` by
    // matching their known values against each 8-byte word of the Vec — a pure
    // derivation from a real instance (no assumed field order).
    let vec_base = core::ptr::addr_of!(*slots) as usize;
    let want_ptr = slots.as_ptr() as usize;
    let want_len = slots.len(); // == 3
    let words = core::mem::size_of::<Vec<NanBox>>() / core::mem::size_of::<usize>();
    let mut vec_ptr_off = usize::MAX;
    let mut vec_len_off = usize::MAX;
    for i in 0..words {
        // SAFETY: reading the i-th usize-word of a live `Vec<NanBox>`, in bounds.
        #[allow(unsafe_code)]
        let w = unsafe { *((vec_base + i * core::mem::size_of::<usize>()) as *const usize) };
        if w == want_ptr {
            vec_ptr_off = i * core::mem::size_of::<usize>();
        } else if w == want_len {
            vec_len_off = i * core::mem::size_of::<usize>();
        }
    }
    assert!(
        vec_ptr_off != usize::MAX && vec_len_off != usize::MAX && vec_ptr_off != vec_len_off,
        "could not derive Vec ptr/len offsets"
    );

    ObjectLayout {
        data_off,
        shaped_disc,
        shape_off,
        slots_off,
        vec_ptr_off,
        vec_len_off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::Heap;
    use crate::nanbox::Unpacked;

    fn n(x: f64) -> NanBox {
        NanBox::number(x)
    }

    #[test]
    fn set_get_and_update() {
        let mut o = Object::new(Shape::root());
        assert!(o.is_empty());
        o.set("x", n(1.0));
        o.set("y", n(2.0));
        assert_eq!(o.len(), 2);
        assert_eq!(o.get("x").unwrap().unpack(), Unpacked::Number(1.0));
        assert_eq!(o.get("y").unwrap().unpack(), Unpacked::Number(2.0));
        assert_eq!(o.get("z"), None);
        // Updating an existing property keeps the shape and reuses the slot.
        let shape_before = Rc::clone(o.shape());
        o.set("x", n(9.0));
        assert!(Rc::ptr_eq(o.shape(), &shape_before));
        assert_eq!(o.get("x").unwrap().unpack(), Unpacked::Number(9.0));
        assert_eq!(o.len(), 2);
        assert_eq!(o.keys(), ["x", "y"]);
    }

    #[test]
    fn same_structure_objects_share_a_shape() {
        let root = Shape::root();
        let mut a = Object::new(Rc::clone(&root));
        let mut b = Object::new(Rc::clone(&root));
        a.set("p", n(1.0));
        a.set("q", n(2.0));
        b.set("p", n(10.0));
        b.set("q", n(20.0));
        // Distinct values, one shared hidden class.
        assert!(Rc::ptr_eq(a.shape(), b.shape()));
        assert_eq!(a.get("p").unwrap().unpack(), Unpacked::Number(1.0));
        assert_eq!(b.get("p").unwrap().unpack(), Unpacked::Number(10.0));
    }

    #[test]
    fn mixed_value_kinds_in_slots() {
        let mut o = Object::new(Shape::root());
        o.set("a", NanBox::number(3.5));
        o.set("b", NanBox::boolean(true));
        o.set("c", NanBox::null());
        o.set("d", NanBox::handle(42));
        assert_eq!(o.get("a").unwrap().unpack(), Unpacked::Number(3.5));
        assert_eq!(o.get("b").unwrap().unpack(), Unpacked::Bool(true));
        assert_eq!(o.get("c").unwrap().unpack(), Unpacked::Null);
        assert_eq!(o.get("d").unwrap().unpack(), Unpacked::Handle(42));
    }

    #[test]
    fn objects_live_in_the_heap_and_reference_each_other() {
        // An object graph: `parent.child` holds a handle to another object in
        // the same heap — the value representation the GC will manage.
        let root = Shape::root();
        let mut heap: Heap<Object> = Heap::new();

        let mut child = Object::new(Rc::clone(&root));
        child.set("value", n(7.0));
        let child_handle = heap.alloc(child);

        let mut parent = Object::new(Rc::clone(&root));
        parent.set("child", NanBox::handle(child_handle.to_raw()));
        let parent_handle = heap.alloc(parent);

        // Walk parent -> child by resolving the handle stored in the slot.
        let parent_ref = heap.get(parent_handle).unwrap();
        let raw = parent_ref.get("child").unwrap().as_handle().unwrap();
        let resolved = crate::heap::Handle::from_raw(raw);
        let child_ref = heap.get(resolved).unwrap();
        assert_eq!(
            child_ref.get("value").unwrap().unpack(),
            Unpacked::Number(7.0)
        );
    }

    /// Adds `key=value` the way the realm does: convert-if-needed, then set.
    fn add(o: &mut Object, key: &str, value: NanBox, threshold: usize) {
        o.maybe_convert_to_dict(key, threshold);
        o.set(key, value);
    }

    #[test]
    fn converts_to_dictionary_past_threshold_and_preserves_semantics() {
        // Threshold 4: the 5th distinct key triggers conversion.
        let mut o = Object::new(Shape::root());
        let threshold = 4;
        for i in 0..4 {
            add(&mut o, &alloc::format!("k{i}"), n(f64::from(i)), threshold);
        }
        // Still shaped: the shape resolves a real key.
        assert_eq!(o.shape().lookup("k0"), Some(0));
        assert_eq!(o.len(), 4);

        // The 5th add converts to dictionary mode.
        add(&mut o, "k4", n(4.0), threshold);
        assert_eq!(o.len(), 5);
        // In dictionary mode the sentinel shape resolves nothing (ICs miss).
        assert_eq!(o.shape().lookup("k0"), None);
        assert_eq!(o.shape().lookup("k4"), None);

        // Keep adding well past the threshold.
        for i in 5..300 {
            add(&mut o, &alloc::format!("k{i}"), n(f64::from(i)), threshold);
        }
        assert_eq!(o.len(), 300);

        // get across the conversion boundary.
        assert_eq!(o.get("k0").unwrap().unpack(), Unpacked::Number(0.0));
        assert_eq!(o.get("k3").unwrap().unpack(), Unpacked::Number(3.0));
        assert_eq!(o.get("k150").unwrap().unpack(), Unpacked::Number(150.0));
        assert_eq!(o.get("k299").unwrap().unpack(), Unpacked::Number(299.0));
        assert_eq!(o.get("nope"), None);

        // Insertion order is preserved across the boundary.
        let keys = o.keys();
        assert_eq!(keys.len(), 300);
        assert_eq!(keys[0], "k0");
        assert_eq!(keys[299], "k299");
        assert!(o.contains("k200"));
        assert!(o.has_own_key("k200"));

        // Update-in-place in dictionary mode.
        add(&mut o, "k150", n(-1.0), threshold);
        assert_eq!(o.len(), 300, "updating an existing key does not grow");
        assert_eq!(o.get("k150").unwrap().unpack(), Unpacked::Number(-1.0));

        // Delete in dictionary mode.
        assert!(o.delete(Shape::root(), "k0"));
        assert_eq!(o.get("k0"), None);
        assert_eq!(o.len(), 299);
        assert_eq!(o.keys()[0], "k1");
        // Deleting an absent key is a no-op (returns false).
        assert!(!o.delete(Shape::root(), "k0"));
    }

    #[test]
    fn dictionary_ordered_keys_put_integer_indices_first() {
        // Mirror the `{b, 2, 1, a}` example but force dictionary mode with a low
        // threshold so the integer-index ordering is exercised in the dict path.
        let mut o = Object::new(Shape::root());
        let threshold = 0; // convert on the very first add
        add(&mut o, "b", n(1.0), threshold);
        add(&mut o, "2", n(1.0), threshold);
        add(&mut o, "1", n(1.0), threshold);
        add(&mut o, "a", n(1.0), threshold);
        // Confirm we are in dictionary mode.
        assert_eq!(o.shape().lookup("b"), None);
        // [[OwnPropertyKeys]] order: ascending integer indices, then strings in
        // insertion order.
        assert_eq!(o.ordered_keys(), ["1", "2", "b", "a"]);
        assert_eq!(o.enumerable_keys(), ["1", "2", "b", "a"]);
        // Raw insertion order is unchanged.
        assert_eq!(o.keys(), ["b", "2", "1", "a"]);
    }

    #[test]
    fn dictionary_traces_handle_values() {
        // GC must trace handle values stored in a dictionary-mode object.
        let mut o = Object::new(Shape::root());
        let threshold = 0;
        add(&mut o, "h1", NanBox::handle(7), threshold);
        add(&mut o, "n", n(1.0), threshold);
        add(&mut o, "h2", NanBox::handle(9), threshold);
        let mut seen: Vec<u64> = Vec::new();
        o.trace_handles(|h| seen.push(h.to_raw()));
        seen.sort_unstable();
        assert_eq!(seen, [7, 9]);
    }
}
