//! The object-model context that ties the foundation together (`ROADMAP.md`
//! §3).
//!
//! [`nanbox`](crate::nanbox), [`shape`](crate::shape), [`object`](crate::object),
//! [`heap`](crate::heap), [`atom`](crate::atom), and [`gc`](crate::gc) are
//! independent pieces; a running engine needs them bundled behind one handle. A
//! `Realm` owns:
//! - the managed [`Heap`] of objects,
//! - the shared **root [`Shape`]** every object's layout descends from (so
//!   identically-structured objects share hidden classes across the realm), and
//! - the realm's [`AtomTable`] for interned property keys.
//!
//! It exposes the operations a VM performs on the heap — allocate an object, get
//! and set properties (as [`NanBox`] values addressed by [`Handle`]), and run a
//! collection from a root set — so the eventual VM migration targets this one
//! API rather than wiring the pieces together itself.
//!
//! Pure, safe `alloc`-only Rust.
//!
//! [`Heap`]: crate::heap::Heap
//! [`Shape`]: crate::shape::Shape
//! [`AtomTable`]: crate::atom::AtomTable
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Handle`]: crate::heap::Handle

use crate::atom::{Atom, AtomTable};
use crate::cell::Cell;
use crate::gc::{self, Stats};
use crate::heap::{Handle, Heap};
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::rope::Rope;
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// An object-model context: the heap, the shared root shape, and the atom table.
pub struct Realm {
    heap: Heap<Cell>,
    root_shape: Rc<Shape>,
    atoms: AtomTable,
    /// The in-flight incremental collection's marker, present only while an
    /// incremental marking cycle is running. The write barrier shades stored
    /// references into it so concurrent mutation stays sound.
    incremental: Option<gc::IncrementalMarker>,
}

impl Default for Realm {
    fn default() -> Self {
        Self::new()
    }
}

impl Realm {
    /// Creates an empty realm.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: Heap::new(),
            root_shape: Shape::root(),
            atoms: AtomTable::new(),
            incremental: None,
        }
    }

    /// Allocates a fresh empty object in the heap and returns its handle.
    pub fn new_object(&mut self) -> Handle {
        let obj = Object::new(Rc::clone(&self.root_shape));
        self.heap.alloc(Cell::Object(obj))
    }

    /// Allocates a string value in the heap and returns its handle.
    pub fn new_string(&mut self, s: &str) -> Handle {
        self.heap.alloc(Cell::Str(Rope::from(s)))
    }

    /// Allocates an array of `elements` in the heap and returns its handle.
    pub fn new_array(&mut self, elements: Vec<NanBox>) -> Handle {
        self.heap.alloc(Cell::Array(elements))
    }

    /// Allocates a closure: a function-table index plus its captured scope.
    pub fn new_function(&mut self, func_id: u32, env: crate::env::Scope) -> Handle {
        self.heap.alloc(Cell::Function { func_id, env })
    }

    /// The `(func_id, captured env)` of the function at `handle`, or `None` if it
    /// is not callable.
    #[must_use]
    pub fn function_at(&self, handle: Handle) -> Option<(u32, crate::env::Scope)> {
        let (id, env) = self.heap.get(handle)?.as_function()?;
        Some((id, env.clone()))
    }

    /// Allocates a built-in (native) function with the given id.
    pub fn new_native(&mut self, id: u16) -> Handle {
        self.heap.alloc(Cell::Native(id))
    }

    /// Allocates a class value (a class-table index plus its captured scope).
    pub fn new_class(&mut self, class_id: u32, env: crate::env::Scope) -> Handle {
        self.heap.alloc(Cell::Class { class_id, env })
    }

    /// The `(class_id, captured env)` of the class at `handle`, or `None`.
    #[must_use]
    pub fn class_at(&self, handle: Handle) -> Option<(u32, crate::env::Scope)> {
        let (id, env) = self.heap.get(handle)?.as_class()?;
        Some((id, env.clone()))
    }

    /// Allocates an empty `Map` (`is_set = false`) or `Set` (`is_set = true`).
    pub fn new_collection(&mut self, is_set: bool) -> Handle {
        self.heap.alloc(Cell::Collection {
            is_set,
            entries: Vec::new(),
        })
    }

    /// Sets `key → value` in the collection at `handle` (inserting or updating,
    /// by strict-equality key match). Returns `false` if not a collection.
    pub fn collection_set(&mut self, handle: Handle, key: NanBox, value: NanBox) -> bool {
        // Find an existing key first (immutable strict_equals borrow), then write.
        let pos = match self.heap.get(handle).and_then(Cell::as_collection) {
            Some((_, entries)) => entries
                .iter()
                .position(|(k, _)| self.strict_equals(*k, key)),
            None => return false,
        };
        let Some((_, entries)) = self.heap.get_mut(handle).and_then(Cell::as_collection_mut) else {
            return false;
        };
        match pos {
            Some(i) => entries[i].1 = value,
            None => entries.push((key, value)),
        }
        self.write_barrier(handle, key);
        self.write_barrier(handle, value);
        true
    }

    /// The value for `key` in the collection, or `None` if absent / not a
    /// collection.
    #[must_use]
    pub fn collection_get(&self, handle: Handle, key: NanBox) -> Option<NanBox> {
        let (_, entries) = self.heap.get(handle)?.as_collection()?;
        entries
            .iter()
            .find(|(k, _)| self.strict_equals(*k, key))
            .map(|(_, v)| *v)
    }

    /// Whether the collection contains `key`.
    #[must_use]
    pub fn collection_has(&self, handle: Handle, key: NanBox) -> bool {
        self.heap
            .get(handle)
            .and_then(Cell::as_collection)
            .is_some_and(|(_, e)| e.iter().any(|(k, _)| self.strict_equals(*k, key)))
    }

    /// Removes `key`; returns whether it was present.
    pub fn collection_delete(&mut self, handle: Handle, key: NanBox) -> bool {
        let pos = match self.heap.get(handle).and_then(Cell::as_collection) {
            Some((_, e)) => e.iter().position(|(k, _)| self.strict_equals(*k, key)),
            None => return false,
        };
        if let Some(i) = pos
            && let Some((_, e)) = self.heap.get_mut(handle).and_then(Cell::as_collection_mut)
        {
            e.remove(i);
            return true;
        }
        false
    }

    /// The number of entries, or `None` if not a collection.
    #[must_use]
    pub fn collection_size(&self, handle: Handle) -> Option<usize> {
        Some(self.heap.get(handle)?.as_collection()?.1.len())
    }

    /// A snapshot of the collection's entries (for iteration / `forEach`).
    #[must_use]
    pub fn collection_entries(&self, handle: Handle) -> Option<Vec<(NanBox, NanBox)>> {
        Some(self.heap.get(handle)?.as_collection()?.1.to_vec())
    }

    /// Whether the collection at `handle` is a `Set` (vs a `Map`).
    #[must_use]
    pub fn collection_is_set(&self, handle: Handle) -> Option<bool> {
        Some(self.heap.get(handle)?.as_collection()?.0)
    }

    /// The native-function id at `handle`, or `None` if it is not a native.
    #[must_use]
    pub fn native_at(&self, handle: Handle) -> Option<u16> {
        self.heap.get(handle)?.as_native()
    }

    /// The `(id, target)` of a bound native at `handle`.
    #[must_use]
    pub fn bound_native_at(&self, handle: Handle) -> Option<(u16, Handle)> {
        self.heap.get(handle)?.as_bound_native()
    }

    /// Allocates a bound native function (e.g. a promise resolve/reject).
    pub fn new_bound_native(&mut self, id: u16, target: Handle) -> Handle {
        self.heap.alloc(Cell::BoundNative { id, target })
    }

    /// Allocates a `Date` from a millisecond timestamp.
    pub fn new_date(&mut self, ms: f64) -> Handle {
        self.heap.alloc(Cell::Date(ms))
    }

    /// The timestamp (ms) of the `Date` at `handle`, if it is one.
    #[must_use]
    pub fn date_at(&self, handle: Handle) -> Option<f64> {
        self.heap.get(handle)?.as_date()
    }

    /// Allocates a `RegExp` from its source and flags.
    pub fn new_regexp(&mut self, source: &str, flags: &str) -> Handle {
        self.heap.alloc(Cell::RegExp {
            source: alloc::boxed::Box::from(source),
            flags: alloc::boxed::Box::from(flags),
        })
    }

    /// The `(source, flags)` of the `RegExp` at `handle` (owned), if it is one.
    #[must_use]
    pub fn regexp_at(
        &self,
        handle: Handle,
    ) -> Option<(alloc::string::String, alloc::string::String)> {
        let (s, f) = self.heap.get(handle)?.as_regexp()?;
        Some((
            alloc::string::String::from(s),
            alloc::string::String::from(f),
        ))
    }

    /// Allocates a pending `Promise`.
    pub fn new_promise(&mut self) -> Handle {
        self.heap
            .alloc(Cell::Promise(alloc::rc::Rc::new(core::cell::RefCell::new(
                crate::cell::PromiseState {
                    status: crate::cell::PromiseStatus::Pending,
                    value: NanBox::undefined(),
                    reactions: alloc::vec::Vec::new(),
                },
            ))))
    }

    /// The shared promise state at `handle`, if it is a promise.
    #[must_use]
    pub fn promise_state(
        &self,
        handle: Handle,
    ) -> Option<alloc::rc::Rc<core::cell::RefCell<crate::cell::PromiseState>>> {
        self.heap.get(handle)?.as_promise().cloned()
    }

    /// The string at `handle` as a `String`, or `None` if it is not a string
    /// (or the handle is stale).
    #[must_use]
    pub fn string_value(&self, handle: Handle) -> Option<alloc::string::String> {
        Some(self.heap.get(handle)?.as_str()?.materialize())
    }

    /// The array elements at `handle`, or `None` if it is not an array.
    #[must_use]
    pub fn array_elements(&self, handle: Handle) -> Option<&[NanBox]> {
        self.heap.get(handle)?.as_array()
    }

    /// Whether `handle` refers to an array.
    #[must_use]
    pub fn is_array(&self, handle: Handle) -> bool {
        self.heap.get(handle).and_then(Cell::as_array).is_some()
    }

    /// The own property names of the object at `handle`, in insertion order, or
    /// `None` if it is not an object.
    #[must_use]
    pub fn object_keys(&self, handle: Handle) -> Option<Vec<alloc::string::String>> {
        let obj = self.heap.get(handle)?.as_object()?;
        Some(
            obj.enumerable_keys()
                .iter()
                // Private fields are stored under a `#`-prefixed key and are
                // never enumerable (so they stay out of `Object.keys`, spread,
                // `for-in`, and JSON). Methods are marked hidden via
                // `enumerable_keys`.
                .filter(|s| !s.starts_with('#'))
                .map(|s| alloc::string::String::from(*s))
                .collect(),
        )
    }

    /// All own string property names (including non-enumerable ones such as
    /// methods, but not private `#` fields) — for `Object.getOwnPropertyNames`.
    pub fn own_property_names(&self, handle: Handle) -> Option<Vec<alloc::string::String>> {
        let obj = self.heap.get(handle)?.as_object()?;
        Some(
            obj.keys()
                .iter()
                .filter(|s| !s.starts_with('#') && !s.starts_with('\u{0}'))
                .map(|s| alloc::string::String::from(*s))
                .collect(),
        )
    }

    /// Freezes the object at `handle` (`Object.freeze`); returns whether it was
    /// an object.
    pub fn freeze_object(&mut self, handle: Handle) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.freeze();
                true
            }
            None => false,
        }
    }

    /// Whether the value at `handle` is a frozen object.
    #[must_use]
    pub fn is_frozen(&self, handle: Handle) -> bool {
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .is_some_and(crate::object::Object::is_frozen)
    }

    /// The length of the array at `handle`, or `None` if it is not an array.
    #[must_use]
    pub fn array_length(&self, handle: Handle) -> Option<usize> {
        Some(self.heap.get(handle)?.as_array()?.len())
    }

    /// `arr[index]` — the element at `index`, or `undefined` if out of range or
    /// the cell is not an array.
    #[must_use]
    pub fn get_element(&self, handle: Handle, index: usize) -> NanBox {
        self.heap
            .get(handle)
            .and_then(Cell::as_array)
            .and_then(|a| a.get(index).copied())
            .unwrap_or(NanBox::undefined())
    }

    /// `arr[index] = value` — grows the array with `undefined` holes if `index`
    /// is past the end (per JS). Returns `false` if the cell is not an array.
    pub fn set_element(&mut self, handle: Handle, index: usize, value: NanBox) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_array_mut) {
            Some(a) => {
                if index >= a.len() {
                    a.resize(index + 1, NanBox::undefined());
                }
                a[index] = value;
                self.write_barrier(handle, value);
                true
            }
            None => false,
        }
    }

    /// `arr.push(value)` — appends, returning the new length, or `None` if the
    /// cell is not an array.
    pub fn array_push(&mut self, handle: Handle, value: NanBox) -> Option<usize> {
        let a = self.heap.get_mut(handle).and_then(Cell::as_array_mut)?;
        a.push(value);
        Some(a.len())
    }

    /// `arr.pop()` — removes and returns the last element (`undefined` if empty
    /// or not an array).
    pub fn array_pop(&mut self, handle: Handle) -> NanBox {
        self.heap
            .get_mut(handle)
            .and_then(Cell::as_array_mut)
            .and_then(Vec::pop)
            .unwrap_or(NanBox::undefined())
    }

    /// The `typeof` string for the heap value at `handle` (`"string"`/`"object"`),
    /// or `None` if the handle is stale.
    #[must_use]
    pub fn type_of(&self, handle: Handle) -> Option<&'static str> {
        Some(self.heap.get(handle)?.type_of())
    }

    /// The value of own property `key` on the object at `handle`, or `None` if
    /// the property is absent, the cell is not an object, or the handle is stale.
    #[must_use]
    pub fn get_property(&self, handle: Handle, key: &str) -> Option<NanBox> {
        self.heap.get(handle)?.as_object()?.get(key)
    }

    /// Tags the object at `handle` with the class it was constructed from.
    pub fn set_class_tag(&mut self, handle: Handle, class_id: u32) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.set_class_tag(class_id);
        }
    }

    /// The class tag of the object at `handle`, if any.
    #[must_use]
    pub fn class_tag(&self, handle: Handle) -> Option<u32> {
        self.heap.get(handle)?.as_object()?.class_tag()
    }

    /// Deletes own property `key` from the object at `handle`; returns whether
    /// anything was removed.
    pub fn delete_property(&mut self, handle: Handle, key: &str) -> bool {
        let root = Rc::clone(&self.root_shape);
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(o) => o.delete(root, key),
            None => false,
        }
    }

    /// Whether the object at `handle` has an own property `key` (including
    /// accessors) — the `in` operator.
    #[must_use]
    pub fn has_own(&self, handle: Handle, key: &str) -> bool {
        self.heap
            .get(handle)
            .and_then(Cell::as_object)
            .is_some_and(|o| o.contains(key) || o.accessor(key).is_some())
    }

    /// Defines an accessor (getter/setter) property on the object at `handle`.
    pub fn define_accessor(&mut self, handle: Handle, key: &str, getter: NanBox, setter: NanBox) {
        if let Some(o) = self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            o.define_accessor(key, getter, setter);
        }
    }

    /// The `(getter, setter)` of accessor `key` on `handle`, if defined.
    #[must_use]
    pub fn accessor(&self, handle: Handle, key: &str) -> Option<(NanBox, NanBox)> {
        self.heap.get(handle)?.as_object()?.accessor(key)
    }

    /// Sets own property `key` to `value` on the object at `handle`. Returns
    /// `false` if the handle is stale or the cell is not an object.
    pub fn set_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.set(key, value);
                self.write_barrier(handle, value);
                true
            }
            None => false,
        }
    }

    /// Sets own property `key` to `value` but marks it **non-enumerable** — used
    /// for class methods, which are callable but must stay out of `Object.keys`,
    /// spread, `for-in`, and JSON.
    pub fn set_hidden_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.set(key, value);
                obj.set_hidden(key);
                self.write_barrier(handle, value);
                true
            }
            None => false,
        }
    }

    /// The generational write barrier: records an old→young edge (container is
    /// in the old generation, `value` is a young heap object) in the heap's
    /// remembered set, so a minor collection keeps the young object alive.
    fn write_barrier(&mut self, container: Handle, value: NanBox) {
        if let Some(raw) = value.as_handle() {
            let target = Handle::from_raw(raw);
            self.heap.record_edge(container, target, gc::OLD_AGE);
            // Incremental (Dijkstra) barrier: shade a reference stored during a
            // marking cycle so it cannot be missed.
            if let Some(marker) = self.incremental.as_mut() {
                marker.mark_grey(target);
            }
        }
    }

    /// Interns `key`, so callers can hold a `Copy` [`Atom`] for hot property
    /// names.
    pub fn intern(&mut self, key: &str) -> Atom {
        self.atoms.intern(key)
    }

    /// The number of live objects in the heap.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.heap.len()
    }

    /// Whether `handle` still refers to a live object.
    #[must_use]
    pub fn is_live(&self, handle: Handle) -> bool {
        self.heap.is_live(handle)
    }

    /// Runs a full (**major**) garbage collection, keeping everything reachable
    /// from `roots` and freeing the rest (including cycles). Survivors are
    /// promoted toward the old generation. Returns the collection statistics.
    pub fn collect(&mut self, roots: &[Handle]) -> Stats {
        gc::collect(&mut self.heap, roots)
    }

    /// Runs a **minor** (generational) collection — reclaims only short-lived
    /// objects in the young generation, treating the old generation as roots.
    /// Cheap when most allocation is short-lived. Returns the statistics.
    pub fn collect_minor(&mut self, roots: &[Handle]) -> Stats {
        gc::collect_minor(&mut self.heap, roots)
    }

    /// Runs a **moving (compacting)** collection: keeps everything reachable from
    /// `roots`, relocates the survivors to the front of the heap (defragmenting
    /// the slot table), and rewrites every reference — including the caller's
    /// `roots`, updated in place — to the new locations. Returns the statistics.
    pub fn compact(&mut self, roots: &mut [Handle]) -> Stats {
        gc::compact(&mut self.heap, roots)
    }

    /// Runs an **incremental** collection: marks in step-bounded slices of at
    /// most `step_budget` objects each (rather than one stop-the-world pass),
    /// then sweeps. Equivalent result to [`collect`](Realm::collect); the
    /// step-based [`IncrementalMarker`](gc::IncrementalMarker) is what lets the
    /// pause be bounded / interleaved with execution.
    pub fn collect_incremental(&mut self, roots: &[Handle], step_budget: usize) -> Stats {
        let before = self.heap.len();
        let mut marker = gc::IncrementalMarker::new(roots);
        while !marker.step(&self.heap, step_budget.max(1)) {}
        let swept = marker.sweep(&mut self.heap);
        Stats {
            marked: before - swept,
            swept,
        }
    }

    /// Begins an **interleaved** incremental collection: installs a marker
    /// seeded from `roots` whose write barrier is now active, so mutation
    /// (`set_property`/`set_element`/…) between steps stays sound. Pair with
    /// [`incremental_step`](Realm::incremental_step) and
    /// [`incremental_finish`](Realm::incremental_finish).
    pub fn incremental_start(&mut self, roots: &[Handle]) {
        self.incremental = Some(gc::IncrementalMarker::new(roots));
    }

    /// Advances the active interleaved marker by up to `step_budget` objects.
    /// Returns `true` when marking is complete. A no-op (`true`) if no cycle is
    /// active.
    pub fn incremental_step(&mut self, step_budget: usize) -> bool {
        // Split the borrow: the marker scans the heap immutably.
        let Some(mut marker) = self.incremental.take() else {
            return true;
        };
        let done = marker.step(&self.heap, step_budget.max(1));
        self.incremental = Some(marker);
        done
    }

    /// Finishes the interleaved cycle: sweeps everything marking did not reach,
    /// clears the active marker, and returns the statistics. A no-op (empty
    /// stats) if no cycle is active.
    pub fn incremental_finish(&mut self) -> Stats {
        let Some(marker) = self.incremental.take() else {
            return Stats::default();
        };
        let before = self.heap.len();
        let swept = marker.sweep(&mut self.heap);
        Stats {
            marked: before - swept,
            swept,
        }
    }

    // --- value operations (the VM's `+`, `ToString`, `===` over heap values) ---

    /// Whether `v` is a heap string.
    #[must_use]
    fn is_string(&self, v: NanBox) -> bool {
        v.as_handle()
            .and_then(|raw| self.heap.get(Handle::from_raw(raw)))
            .is_some_and(|c| c.as_str().is_some())
    }

    /// The rope view of `v` for concatenation: a string cell's own rope (shared,
    /// so concatenation stays O(1)), or a fresh leaf from its `ToString`.
    fn rope_of(&self, v: NanBox) -> Rope {
        if let Some(raw) = v.as_handle()
            && let Some(rope) = self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str)
        {
            return rope.clone();
        }
        Rope::from(self.to_display_string(v).as_str())
    }

    /// ECMAScript `ToString` for display: numbers/booleans/null/undefined render
    /// directly; a string yields its text; an array joins its elements with
    /// `","`; a plain object is `"[object Object]"`.
    #[must_use]
    pub fn to_display_string(&self, v: NanBox) -> alloc::string::String {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Undefined => "undefined".into(),
            Unpacked::Null => "null".into(),
            Unpacked::Bool(b) => {
                if b {
                    "true".into()
                } else {
                    "false".into()
                }
            }
            Unpacked::Number(n) => js_number_string(n),
            Unpacked::Handle(raw) => match self.heap.get(Handle::from_raw(raw)) {
                Some(Cell::Str(r)) => r.materialize(),
                Some(Cell::Array(elems)) => {
                    let parts: Vec<alloc::string::String> = elems
                        .iter()
                        .map(|e| {
                            // A hole-ish nullish element renders empty, per `Array#join`.
                            if matches!(e.unpack(), Unpacked::Undefined | Unpacked::Null) {
                                alloc::string::String::new()
                            } else {
                                self.to_display_string(*e)
                            }
                        })
                        .collect();
                    parts.join(",")
                }
                Some(Cell::Object(_)) => "[object Object]".into(),
                Some(Cell::Function { .. } | Cell::Native(_) | Cell::Class { .. }) => {
                    "function () { … }".into()
                }
                Some(Cell::Collection { is_set, .. }) => {
                    if *is_set {
                        "[object Set]".into()
                    } else {
                        "[object Map]".into()
                    }
                }
                Some(Cell::BoundNative { .. }) => "function () { … }".into(),
                Some(Cell::Promise(_)) => "[object Promise]".into(),
                Some(Cell::Date(ms)) => date_to_iso(*ms),
                Some(Cell::RegExp { source, flags }) => alloc::format!("/{source}/{flags}"),
                None => "undefined".into(), // stale handle
            },
        }
    }

    /// The ECMAScript `+` operator (the cases this model covers): number + number
    /// is numeric addition; if either side is a string it is string
    /// concatenation, producing a new heap string built by joining ropes (so a
    /// loop of `+` stays O(1) per step). Other combinations coerce to string for
    /// now (full `ToPrimitive` arrives with the boxed primitives).
    pub fn add(&mut self, a: NanBox, b: NanBox) -> NanBox {
        if let (Some(x), Some(y)) = (a.as_number(), b.as_number()) {
            return NanBox::number(x + y);
        }
        // A string operand keeps the O(1) rope concatenation path.
        if self.is_string(a) || self.is_string(b) {
            let combined = self.rope_of(a).concat(&self.rope_of(b));
            let handle = self.heap.alloc(Cell::Str(combined));
            return NanBox::handle(handle.to_raw());
        }
        // Any other heap value (array, object): `+` is string concatenation
        // after `ToPrimitive` — our arrays/objects stringify (`[1,2] + [3,4]`
        // → "1,23,4", `{} + "!"` → "[object Object]!").
        if a.as_handle().is_some() || b.as_handle().is_some() {
            let mut combined = self.to_display_string(a);
            combined.push_str(&self.to_display_string(b));
            let handle = self.heap.alloc(Cell::Str(Rope::from(combined.as_str())));
            return NanBox::handle(handle.to_raw());
        }
        // Primitives only (bool/null/undefined): numeric.
        NanBox::number(self.to_number(a) + self.to_number(b))
    }

    /// ECMAScript `ToNumber` (the cases this model covers): numbers pass
    /// through; `true`/`false` → `1`/`0`; `null` → `0`; `undefined` → `NaN`; a
    /// string is parsed (blank → `0`, a numeric literal → its value, else
    /// `NaN`); objects/arrays → `NaN` (full `ToPrimitive` arrives later).
    #[must_use]
    pub fn to_number(&self, v: NanBox) -> f64 {
        use crate::nanbox::Unpacked;
        match v.unpack() {
            Unpacked::Number(n) => n,
            Unpacked::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Unpacked::Null => 0.0,
            Unpacked::Undefined => f64::NAN,
            Unpacked::Handle(raw) => {
                match self.heap.get(Handle::from_raw(raw)).and_then(Cell::as_str) {
                    Some(rope) => {
                        let s = rope.materialize();
                        let t = s.trim();
                        if t.is_empty() {
                            0.0
                        } else {
                            t.parse::<f64>().unwrap_or(f64::NAN)
                        }
                    }
                    None => f64::NAN, // object/array/stale
                }
            }
        }
    }

    /// The ECMAScript abstract relational comparison `a < b`: if *both* operands
    /// are strings they compare lexicographically by code point; otherwise both
    /// are coerced with `ToNumber`. Returns `None` when the result is undefined
    /// (a `NaN` operand) — the caller turns that into `false`.
    #[must_use]
    fn compare(&self, a: NanBox, b: NanBox) -> Option<core::cmp::Ordering> {
        if self.is_string(a) && self.is_string(b) {
            let sa = self.to_display_string(a);
            let sb = self.to_display_string(b);
            return Some(sa.cmp(&sb));
        }
        let (x, y) = (self.to_number(a), self.to_number(b));
        x.partial_cmp(&y) // None on NaN
    }

    /// `a < b` (boolean).
    #[must_use]
    pub fn less_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Less))
    }

    /// `a <= b` (boolean).
    #[must_use]
    pub fn less_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
        ))
    }

    /// `a > b` (boolean).
    #[must_use]
    pub fn greater_than(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(self.compare(a, b) == Some(core::cmp::Ordering::Greater))
    }

    /// `a >= b` (boolean).
    #[must_use]
    pub fn greater_equal(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::boolean(matches!(
            self.compare(a, b),
            Some(core::cmp::Ordering::Greater | core::cmp::Ordering::Equal)
        ))
    }

    /// `a - b` (numeric, both coerced with `ToNumber`).
    #[must_use]
    pub fn sub(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) - self.to_number(b))
    }

    /// `a * b` (numeric).
    #[must_use]
    pub fn mul(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) * self.to_number(b))
    }

    /// `a / b` (numeric; division by zero yields ±∞ / `NaN` per IEEE-754).
    #[must_use]
    pub fn div(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) / self.to_number(b))
    }

    /// `a % b` (the ECMAScript remainder, which follows the sign of the
    /// dividend — Rust's `%` on `f64` matches).
    #[must_use]
    pub fn rem(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a) % self.to_number(b))
    }

    /// `a ** b` (exponentiation). Needs `std` for the float `powf` intrinsic
    /// (the alloc-only core omits it, like the rest of the engine's float math).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn pow(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(self.to_number(a).powf(self.to_number(b)))
    }

    /// Unary `-a` (numeric negation).
    #[must_use]
    pub fn neg(&self, a: NanBox) -> NanBox {
        NanBox::number(-self.to_number(a))
    }

    /// Unary `!a` (logical negation via `ToBoolean`).
    #[must_use]
    pub fn logical_not(&self, a: NanBox) -> NanBox {
        NanBox::boolean(!a.to_boolean())
    }

    /// The `typeof` string for any value: primitives via the box
    /// (`"undefined"`/`"boolean"`/`"number"`/`"object"` for null), and heap
    /// values via their cell (`"string"` for strings, `"object"` otherwise).
    #[must_use]
    pub fn type_of_value(&self, v: NanBox) -> &'static str {
        match v.as_handle() {
            Some(raw) => self
                .heap
                .get(Handle::from_raw(raw))
                .map_or("undefined", Cell::type_of),
            None => v.type_of(),
        }
    }

    /// ECMAScript `ToInt32`. Needs `std` for the `trunc` float intrinsic.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn to_int32(&self, v: NanBox) -> i32 {
        let n = self.to_number(v);
        if !n.is_finite() || n == 0.0 {
            return 0;
        }
        // Reduce trunc(n) modulo 2^32 into [0, 2^32), then reinterpret as i32.
        let m = n.trunc().rem_euclid(4_294_967_296.0);
        m as u32 as i32
    }

    /// ECMAScript `ToUint32`. Needs `std` for the `trunc` float intrinsic.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn to_uint32(&self, v: NanBox) -> u32 {
        self.to_int32(v) as u32
    }

    /// `a & b` (bitwise AND over `ToInt32`). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_and(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) & self.to_int32(b)))
    }

    /// `a | b` (bitwise OR). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_or(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) | self.to_int32(b)))
    }

    /// `a ^ b` (bitwise XOR). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_xor(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) ^ self.to_int32(b)))
    }

    /// `~a` (bitwise NOT). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn bit_not(&self, a: NanBox) -> NanBox {
        NanBox::number(f64::from(!self.to_int32(a)))
    }

    /// `a << b` (left shift; `b` masked to 5 bits, per spec). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn shl(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) << (self.to_uint32(b) & 31)))
    }

    /// `a >> b` (sign-propagating right shift). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn shr(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_int32(a) >> (self.to_uint32(b) & 31)))
    }

    /// `a >>> b` (zero-fill right shift; result is unsigned). Needs `std`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn ushr(&self, a: NanBox, b: NanBox) -> NanBox {
        NanBox::number(f64::from(self.to_uint32(a) >> (self.to_uint32(b) & 31)))
    }

    /// ECMAScript abstract equality (`==`) — strict equality plus coercion:
    /// `null == undefined`; a boolean is compared as its number; a number and a
    /// string compare numerically; two references use [`strict_equals`] (string
    /// value / object identity). (Full `ToPrimitive` on objects arrives later.)
    ///
    /// [`strict_equals`]: Realm::strict_equals
    #[must_use]
    pub fn loose_equals(&self, a: NanBox, b: NanBox) -> bool {
        use crate::nanbox::Unpacked::{Bool, Handle as H, Null, Undefined};
        let (ua, ub) = (a.unpack(), b.unpack());
        let nullish = |u| matches!(u, Undefined | Null);
        // null/undefined are == to each other and to nothing else.
        if nullish(ua) || nullish(ub) {
            return nullish(ua) && nullish(ub);
        }
        // Two references: strings by value, objects by identity.
        if matches!(ua, H(_)) && matches!(ub, H(_)) {
            return self.strict_equals(a, b);
        }
        // Booleans compare as their numeric value.
        if matches!(ua, Bool(_)) {
            return self.loose_equals(NanBox::number(self.to_number(a)), b);
        }
        if matches!(ub, Bool(_)) {
            return self.loose_equals(a, NanBox::number(self.to_number(b)));
        }
        // Remaining: number vs number, or number vs string — compare numerically.
        self.to_number(a) == self.to_number(b)
    }

    /// ECMAScript strict equality (`===`) over heap values: primitives compare
    /// by value; two strings compare by *content* (strings are primitives, so
    /// distinct allocations of `"ab"` are equal); other references compare by
    /// identity.
    #[must_use]
    pub fn strict_equals(&self, a: NanBox, b: NanBox) -> bool {
        match (a.as_handle(), b.as_handle()) {
            (Some(ha), Some(hb)) => {
                if ha == hb {
                    return true; // same heap cell
                }
                let sa = self.heap.get(Handle::from_raw(ha)).and_then(Cell::as_str);
                let sb = self.heap.get(Handle::from_raw(hb)).and_then(Cell::as_str);
                match (sa, sb) {
                    (Some(ra), Some(rb)) => ra.materialize() == rb.materialize(),
                    _ => false, // distinct non-string references
                }
            }
            // At least one primitive: decided by the boxed value itself.
            _ => a.strict_equals(b),
        }
    }
}

/// Renders a number as ECMAScript `ToString` would for the cases the engine
/// produces: `±Infinity` (not Rust's `inf`) and `NaN`; finite values use Rust's
/// `Display` (which omits a trailing `.0` for integers).
#[must_use]
pub(crate) fn js_number_string(n: f64) -> alloc::string::String {
    if n.is_infinite() {
        if n > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    } else {
        alloc::format!("{n}")
    }
}

/// The civil date `(year, month [1-12], day [1-31])` for a day count `z` since
/// the Unix epoch — Howard Hinnant's `civil_from_days` algorithm (pure integer
/// arithmetic, so `core`-clean).
#[must_use]
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Renders a millisecond timestamp as an ISO-8601 UTC string.
#[must_use]
pub(crate) fn date_to_iso(ms: f64) -> alloc::string::String {
    let total_ms = ms as i64;
    let day = total_ms.div_euclid(86_400_000);
    let tod = total_ms.rem_euclid(86_400_000); // [0, 86_400_000)
    let (y, mo, d) = civil_from_days(day);
    let (h, min, s, milli) = (
        tod / 3_600_000,
        (tod / 60_000) % 60,
        (tod / 1000) % 60,
        tod % 1000,
    );
    alloc::format!("{y:04}-{mo:02}-{d:02}T{h:02}:{min:02}:{s:02}.{milli:03}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_access_properties() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.set_property(obj, "y", NanBox::number(2.0));
        assert_eq!(realm.get_property(obj, "x"), Some(NanBox::number(1.0)));
        assert_eq!(realm.get_property(obj, "y"), Some(NanBox::number(2.0)));
        assert_eq!(realm.get_property(obj, "z"), None);
        assert_eq!(realm.object_count(), 1);
    }

    #[test]
    fn objects_share_hidden_classes_across_the_realm() {
        let mut realm = Realm::new();
        let a = realm.new_object();
        let b = realm.new_object();
        realm.set_property(a, "p", NanBox::number(1.0));
        realm.set_property(b, "p", NanBox::number(9.0));
        // Same structure built from the realm's shared root → same shape.
        let sa = Rc::clone(realm.heap.get(a).unwrap().as_object().unwrap().shape());
        let sb = Rc::clone(realm.heap.get(b).unwrap().as_object().unwrap().shape());
        assert!(Rc::ptr_eq(&sa, &sb));
    }

    #[test]
    fn gc_reclaims_unreachable_objects_including_cycles() {
        let mut realm = Realm::new();
        // root -> a; b <-> c form an unreachable cycle.
        let root = realm.new_object();
        let a = realm.new_object();
        realm.set_property(root, "a", NanBox::handle(a.to_raw()));

        let b = realm.new_object();
        let c = realm.new_object();
        realm.set_property(b, "c", NanBox::handle(c.to_raw()));
        realm.set_property(c, "b", NanBox::handle(b.to_raw()));
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[root]);
        assert_eq!(stats.marked, 2); // root + a
        assert_eq!(stats.swept, 2); // the b<->c cycle
        assert!(realm.is_live(root) && realm.is_live(a));
        assert!(!realm.is_live(b) && !realm.is_live(c));
        assert_eq!(realm.object_count(), 2);
    }

    #[test]
    fn stale_handle_access_is_safe() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        realm.set_property(obj, "x", NanBox::number(1.0));
        realm.collect(&[]); // frees obj (no roots)
        assert!(!realm.is_live(obj));
        assert_eq!(realm.get_property(obj, "x"), None);
        assert!(!realm.set_property(obj, "x", NanBox::number(2.0)));
    }

    #[test]
    fn interning_is_shared() {
        let mut realm = Realm::new();
        let a = realm.intern("length");
        let b = realm.intern("length");
        assert_eq!(a, b);
    }

    #[test]
    fn strings_and_arrays_are_heap_values() {
        let mut realm = Realm::new();
        let s = realm.new_string("hello");
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        let obj = realm.new_object();
        assert_eq!(realm.string_value(s).as_deref(), Some("hello"));
        assert_eq!(realm.array_elements(arr).map(<[_]>::len), Some(2));
        assert_eq!(realm.type_of(s), Some("string"));
        assert_eq!(realm.type_of(arr), Some("object"));
        assert_eq!(realm.type_of(obj), Some("object"));
        // A property op on a string cell is rejected (not an object).
        assert!(!realm.set_property(s, "x", NanBox::number(1.0)));
        assert_eq!(realm.get_property(s, "x"), None);
    }

    #[test]
    fn add_numbers_and_concatenates_strings() {
        let mut realm = Realm::new();
        // number + number → numeric.
        let n = realm.add(NanBox::number(2.0), NanBox::number(3.0));
        assert_eq!(n.as_number(), Some(5.0));
        // string + number → concatenation (coerced).
        let hi = realm.new_string("count: ");
        let r = realm.add(NanBox::handle(hi.to_raw()), NanBox::number(42.0));
        assert_eq!(realm.to_display_string(r), "count: 42");
        // string + string.
        let a = realm.new_string("foo");
        let b = realm.new_string("bar");
        let ab = realm.add(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw()));
        assert_eq!(realm.to_display_string(ab), "foobar");
    }

    #[test]
    fn string_append_loop_builds_correctly() {
        // The `s += part` shape, now through ropes in the heap.
        let mut realm = Realm::new();
        let mut acc = NanBox::handle(realm.new_string("").to_raw());
        let mut expected = alloc::string::String::new();
        for i in 0..30 {
            let part = realm.new_string(&alloc::format!("{i},"));
            acc = realm.add(acc, NanBox::handle(part.to_raw()));
            expected.push_str(&alloc::format!("{i},"));
        }
        assert_eq!(realm.to_display_string(acc), expected);
    }

    #[test]
    fn to_display_string_covers_kinds() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_display_string(NanBox::undefined()), "undefined");
        assert_eq!(realm.to_display_string(NanBox::null()), "null");
        assert_eq!(realm.to_display_string(NanBox::boolean(true)), "true");
        assert_eq!(realm.to_display_string(NanBox::number(3.5)), "3.5");
        let arr = realm.new_array(alloc::vec![
            NanBox::number(1.0),
            NanBox::number(2.0),
            NanBox::number(3.0),
        ]);
        assert_eq!(
            realm.to_display_string(NanBox::handle(arr.to_raw())),
            "1,2,3"
        );
        let obj = realm.new_object();
        assert_eq!(
            realm.to_display_string(NanBox::handle(obj.to_raw())),
            "[object Object]"
        );
    }

    #[test]
    fn strict_equals_strings_by_value_objects_by_identity() {
        let mut realm = Realm::new();
        // Primitives.
        assert!(realm.strict_equals(NanBox::number(1.0), NanBox::number(1.0)));
        assert!(!realm.strict_equals(NanBox::number(1.0), NanBox::null()));
        // Two distinct string allocations with equal content are ===.
        let a = realm.new_string("hello");
        let b = realm.new_string("hello");
        let c = realm.new_string("world");
        assert_ne!(a.to_raw(), b.to_raw()); // genuinely different cells
        assert!(realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(b.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(a.to_raw()), NanBox::handle(c.to_raw())));
        // Objects compare by identity.
        let o1 = realm.new_object();
        let o2 = realm.new_object();
        assert!(realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o1.to_raw())));
        assert!(!realm.strict_equals(NanBox::handle(o1.to_raw()), NanBox::handle(o2.to_raw())));
    }

    #[test]
    fn to_number_coerces() {
        let mut realm = Realm::new();
        assert_eq!(realm.to_number(NanBox::number(3.5)), 3.5);
        assert_eq!(realm.to_number(NanBox::boolean(true)), 1.0);
        assert_eq!(realm.to_number(NanBox::boolean(false)), 0.0);
        assert_eq!(realm.to_number(NanBox::null()), 0.0);
        assert!(realm.to_number(NanBox::undefined()).is_nan());
        let s = realm.new_string("  42  ");
        assert_eq!(realm.to_number(NanBox::handle(s.to_raw())), 42.0);
        let blank = realm.new_string("");
        assert_eq!(realm.to_number(NanBox::handle(blank.to_raw())), 0.0);
        let bad = realm.new_string("nope");
        assert!(realm.to_number(NanBox::handle(bad.to_raw())).is_nan());
    }

    #[test]
    fn relational_operators_on_numbers_and_strings() {
        let mut realm = Realm::new();
        let one = NanBox::number(1.0);
        let two = NanBox::number(2.0);
        assert_eq!(realm.less_than(one, two).as_boolean(), Some(true));
        assert_eq!(realm.less_than(two, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(two, one).as_boolean(), Some(true));
        assert_eq!(realm.less_equal(two, two).as_boolean(), Some(true));
        assert_eq!(realm.greater_equal(two, two).as_boolean(), Some(true));
        // A NaN operand makes every comparison false.
        let nan = NanBox::number(f64::NAN);
        assert_eq!(realm.less_than(nan, one).as_boolean(), Some(false));
        assert_eq!(realm.greater_than(nan, one).as_boolean(), Some(false));
        // Two strings compare lexicographically: "10" < "9" (string order).
        let s10 = realm.new_string("10");
        let s9 = realm.new_string("9");
        assert_eq!(
            realm
                .less_than(NanBox::handle(s10.to_raw()), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(true)
        );
        // Mixed string/number coerces to numeric: 10 < "9" is false (9 < 10).
        assert_eq!(
            realm
                .less_than(NanBox::number(10.0), NanBox::handle(s9.to_raw()))
                .as_boolean(),
            Some(false)
        );
    }

    #[test]
    fn arithmetic_operators() {
        let realm = Realm::new();
        let n = NanBox::number;
        assert_eq!(realm.sub(n(5.0), n(3.0)).as_number(), Some(2.0));
        assert_eq!(realm.mul(n(4.0), n(2.5)).as_number(), Some(10.0));
        assert_eq!(realm.div(n(9.0), n(2.0)).as_number(), Some(4.5));
        assert_eq!(realm.rem(n(7.0), n(3.0)).as_number(), Some(1.0));
        assert_eq!(realm.rem(n(-7.0), n(3.0)).as_number(), Some(-1.0)); // sign of dividend
        assert_eq!(realm.neg(n(3.0)).as_number(), Some(-3.0));
        // Division by zero is ±Infinity.
        assert_eq!(realm.div(n(1.0), n(0.0)).as_number(), Some(f64::INFINITY));
        #[cfg(feature = "std")]
        assert_eq!(realm.pow(n(2.0), n(10.0)).as_number(), Some(1024.0));
    }

    #[test]
    fn abstract_equality_coerces() {
        let mut realm = Realm::new();
        let n = NanBox::number;
        // null == undefined, but not == 0.
        assert!(realm.loose_equals(NanBox::null(), NanBox::undefined()));
        assert!(!realm.loose_equals(NanBox::null(), n(0.0)));
        // number == string by numeric coercion.
        let s1 = NanBox::handle(realm.new_string("1").to_raw());
        assert!(realm.loose_equals(n(1.0), s1));
        // boolean coerces to number: true == 1, false == 0.
        assert!(realm.loose_equals(NanBox::boolean(true), n(1.0)));
        assert!(realm.loose_equals(NanBox::boolean(false), n(0.0)));
        assert!(!realm.loose_equals(NanBox::boolean(true), n(2.0)));
        // strings by value; objects by identity.
        let a = NanBox::handle(realm.new_string("x").to_raw());
        let b = NanBox::handle(realm.new_string("x").to_raw());
        assert!(realm.loose_equals(a, b));
        let o1 = NanBox::handle(realm.new_object().to_raw());
        assert!(!realm.loose_equals(o1, n(0.0))); // object != 0 (ToNumber→NaN)
    }

    #[test]
    fn typeof_and_logical_not() {
        let mut realm = Realm::new();
        assert_eq!(realm.type_of_value(NanBox::undefined()), "undefined");
        assert_eq!(realm.type_of_value(NanBox::null()), "object");
        assert_eq!(realm.type_of_value(NanBox::boolean(true)), "boolean");
        assert_eq!(realm.type_of_value(NanBox::number(1.0)), "number");
        let s = NanBox::handle(realm.new_string("hi").to_raw());
        assert_eq!(realm.type_of_value(s), "string");
        let o = NanBox::handle(realm.new_object().to_raw());
        assert_eq!(realm.type_of_value(o), "object");
        // ToBoolean-based negation.
        assert_eq!(
            realm.logical_not(NanBox::number(0.0)).as_boolean(),
            Some(true)
        );
        assert_eq!(
            realm.logical_not(NanBox::number(1.0)).as_boolean(),
            Some(false)
        );
        assert_eq!(realm.logical_not(s).as_boolean(), Some(false)); // objects truthy
    }

    #[cfg(feature = "std")]
    #[test]
    fn bitwise_operators() {
        let realm = Realm::new();
        let n = NanBox::number;
        assert_eq!(realm.bit_and(n(12.0), n(10.0)).as_number(), Some(8.0));
        assert_eq!(realm.bit_or(n(12.0), n(10.0)).as_number(), Some(14.0));
        assert_eq!(realm.bit_xor(n(12.0), n(10.0)).as_number(), Some(6.0));
        assert_eq!(realm.bit_not(n(0.0)).as_number(), Some(-1.0));
        assert_eq!(realm.shl(n(1.0), n(4.0)).as_number(), Some(16.0));
        assert_eq!(realm.shr(n(-8.0), n(1.0)).as_number(), Some(-4.0)); // sign-propagating
        assert_eq!(realm.ushr(n(-1.0), n(0.0)).as_number(), Some(4294967295.0)); // zero-fill
        // ToInt32 truncates fractional and wraps modulo 2^32.
        assert_eq!(realm.to_int32(n(3.9)), 3);
        assert_eq!(realm.to_int32(n(4294967297.0)), 1);
        assert_eq!(realm.to_int32(n(f64::NAN)), 0);
    }

    #[test]
    fn array_index_length_and_push() {
        let mut realm = Realm::new();
        let arr = realm.new_array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        assert_eq!(realm.array_length(arr), Some(2));
        assert_eq!(realm.get_element(arr, 0).as_number(), Some(1.0));
        assert_eq!(realm.get_element(arr, 1).as_number(), Some(2.0));
        // Out of range reads undefined.
        assert!(realm.get_element(arr, 5).is_undefined());
        // Setting past the end grows with holes.
        assert!(realm.set_element(arr, 4, NanBox::number(9.0)));
        assert_eq!(realm.array_length(arr), Some(5));
        assert!(realm.get_element(arr, 3).is_undefined()); // a hole
        assert_eq!(realm.get_element(arr, 4).as_number(), Some(9.0));
        // Push returns the new length.
        assert_eq!(realm.array_push(arr, NanBox::number(7.0)), Some(6));
        assert_eq!(realm.get_element(arr, 5).as_number(), Some(7.0));
        // join renders the array (holes empty).
        assert_eq!(
            realm.to_display_string(NanBox::handle(arr.to_raw())),
            "1,2,,,9,7"
        );
        // Array ops on a non-array are rejected.
        let obj = realm.new_object();
        assert_eq!(realm.array_length(obj), None);
        assert!(!realm.set_element(obj, 0, NanBox::number(1.0)));
    }

    #[test]
    fn gc_keeps_a_mixed_object_array_string_graph() {
        let mut realm = Realm::new();
        // obj.name -> string; obj.items -> array; array[0] -> obj (a cycle).
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        let _unreachable = realm.new_string("garbage");
        assert_eq!(realm.object_count(), 4);

        let stats = realm.collect(&[obj]);
        assert_eq!(stats.marked, 3); // obj, name, items (cycle obj<-items kept)
        assert_eq!(stats.swept, 1); // the unreachable string
        assert!(realm.is_live(obj) && realm.is_live(name) && realm.is_live(items));
        assert_eq!(realm.string_value(name).as_deref(), Some("widget"));
    }

    #[test]
    fn compaction_defragments_and_fixes_up_the_object_graph() {
        let mut realm = Realm::new();
        // obj.name -> string; obj.items -> array; array[0] -> obj (a cycle),
        // with unreachable garbage interleaved to create gaps.
        let _g0 = realm.new_string("garbage0");
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let _g1 = realm.new_object();
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        assert_eq!(realm.object_count(), 5);

        let mut roots = [obj];
        let stats = realm.compact(&mut roots);
        assert_eq!(stats.marked, 3); // obj, name, items
        assert_eq!(stats.swept, 2); // the two garbage objects
        assert_eq!(realm.object_count(), 3); // slot table defragmented

        // The root was rewritten; the whole graph resolves through new handles.
        let obj2 = roots[0];
        let name2 = realm.get_property(obj2, "name").unwrap();
        assert_eq!(
            realm
                .string_value(Handle::from_raw(name2.as_handle().unwrap()))
                .as_deref(),
            Some("widget")
        );
        let items2 = realm.get_property(obj2, "items").unwrap();
        let arr = Handle::from_raw(items2.as_handle().unwrap());
        // array[0] still points back at the (relocated) object — the cycle held.
        assert_eq!(realm.get_element(arr, 0).as_handle(), Some(obj2.to_raw()));
    }

    #[test]
    fn interleaved_incremental_barrier_keeps_a_mid_cycle_store() {
        let mut realm = Realm::new();
        let root = realm.new_object();

        // Start an interleaved cycle and mark the root fully (it has no edges
        // yet, so marking completes with `root` black).
        realm.incremental_start(&[root]);
        while !realm.incremental_step(1) {}

        // Mid-cycle, the mutator allocates a new object (white) and stores it on
        // the already-black root. The integrated write barrier shades it.
        let child = realm.new_object();
        realm.set_property(root, "child", NanBox::handle(child.to_raw()));

        // Drain the re-greyed work, then finish.
        while !realm.incremental_step(4) {}
        let stats = realm.incremental_finish();

        assert_eq!(stats.swept, 0, "the barrier-shaded child must survive");
        assert!(realm.is_live(root) && realm.is_live(child));
    }

    #[test]
    fn incremental_collection_matches_full_over_an_object_graph() {
        let mut realm = Realm::new();
        let obj = realm.new_object();
        let name = realm.new_string("widget");
        let items = realm.new_array(alloc::vec![NanBox::handle(obj.to_raw())]);
        realm.set_property(obj, "name", NanBox::handle(name.to_raw()));
        realm.set_property(obj, "items", NanBox::handle(items.to_raw()));
        let _garbage = realm.new_string("garbage");
        assert_eq!(realm.object_count(), 4);

        // Tiny step budget → many incremental slices; same result as a full GC.
        let stats = realm.collect_incremental(&[obj], 1);
        assert_eq!(stats.marked, 3); // obj, name, items (cycle held)
        assert_eq!(stats.swept, 1); // the unreachable string
        assert!(realm.is_live(obj) && realm.is_live(name) && realm.is_live(items));
        assert_eq!(realm.string_value(name).as_deref(), Some("widget"));
    }

    #[test]
    fn minor_collection_with_write_barrier_keeps_old_to_young_edge() {
        let mut realm = Realm::new();
        // Create an object and promote it to the old generation via a major GC.
        let parent = realm.new_object();
        realm.collect(&[parent]);

        // Now attach a freshly-allocated (young) string — the `set_property`
        // write barrier records the old→young edge.
        let child = realm.new_string("attached");
        realm.set_property(parent, "child", NanBox::handle(child.to_raw()));
        let _garbage = realm.new_string("garbage"); // young, unreferenced

        // A minor collection frees only the young garbage; `child`, reachable
        // solely through the old `parent`, survives thanks to the barrier.
        let stats = realm.collect_minor(&[parent]);
        assert_eq!(stats.swept, 1);
        assert!(realm.is_live(parent) && realm.is_live(child));
        assert_eq!(realm.string_value(child).as_deref(), Some("attached"));
    }
}
