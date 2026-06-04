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
//! [`Shape`]: crate::shape::Shape
//! [`NanBox`]: crate::nanbox::NanBox
//! [`Heap`]: crate::heap::Heap
//!
//! Pure, safe `alloc`-only Rust; this is the representation the bytecode VM will
//! migrate onto once the GC that manages the heap lands.

use crate::nanbox::NanBox;
use crate::shape::Shape;
use alloc::rc::Rc;
use alloc::vec::Vec;

/// A property-bearing object: a hidden-class shape plus its value slots.
pub struct Object {
    shape: Rc<Shape>,
    slots: Vec<NanBox>,
}

impl Object {
    /// Creates an empty object whose layout starts at `root` (the shared root
    /// shape of the owning realm/heap, so identically-structured objects share
    /// shapes).
    #[must_use]
    pub fn new(root: Rc<Shape>) -> Self {
        Self {
            shape: root,
            slots: Vec::new(),
        }
    }

    /// The object's current shape (its hidden class).
    #[must_use]
    pub fn shape(&self) -> &Rc<Shape> {
        &self.shape
    }

    /// The number of own properties.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.shape.len()
    }

    /// Whether the object has no own properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// The value of own property `key`, or `None` if absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<NanBox> {
        let slot = self.shape.lookup(key)?;
        self.slots.get(slot as usize).copied()
    }

    /// Whether the object has own property `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.shape.contains(key)
    }

    /// Sets own property `key` to `value`: updates the slot in place if the
    /// property exists, otherwise transitions the shape and appends a slot.
    pub fn set(&mut self, key: &str, value: NanBox) {
        if let Some(slot) = self.shape.lookup(key) {
            self.slots[slot as usize] = value;
        } else {
            self.shape = self.shape.transition(key);
            self.slots.push(value);
        }
    }

    /// The own property names, in insertion (slot) order.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        self.shape.keys()
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
}
