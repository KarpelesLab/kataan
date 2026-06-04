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

    /// Sets own property `key` to `value` on the object at `handle`. Returns
    /// `false` if the handle is stale or the cell is not an object.
    pub fn set_property(&mut self, handle: Handle, key: &str, value: NanBox) -> bool {
        match self.heap.get_mut(handle).and_then(Cell::as_object_mut) {
            Some(obj) => {
                obj.set(key, value);
                true
            }
            None => false,
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

    /// Runs a garbage collection, keeping everything reachable from `roots` and
    /// freeing the rest (including cycles). Returns the collection statistics.
    pub fn collect(&mut self, roots: &[Handle]) -> Stats {
        gc::collect(&mut self.heap, roots)
    }
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
}
