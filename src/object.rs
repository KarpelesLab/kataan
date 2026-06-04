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

/// A property-bearing object: a hidden-class shape plus its value slots, with an
/// optional side list of accessor (getter/setter) properties.
pub struct Object {
    shape: Rc<Shape>,
    slots: Vec<NanBox>,
    /// Accessor properties: `(name, getter, setter)`, both held as value
    /// handles (`undefined` when absent). Kept out of the shape's slot layout.
    accessors: Vec<(alloc::boxed::Box<str>, NanBox, NanBox)>,
    /// Own keys that are **non-enumerable** (e.g. class methods): present in the
    /// slots and readable, but hidden from `Object.keys`/spread/`for-in`/JSON.
    hidden: Vec<alloc::boxed::Box<str>>,
    /// The class this object was instantiated from (for `instanceof`), if any.
    class_tag: Option<u32>,
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
            accessors: Vec::new(),
            hidden: Vec::new(),
            class_tag: None,
        }
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

    /// The `(getter, setter)` of accessor `name`, if defined.
    #[must_use]
    pub fn accessor(&self, name: &str) -> Option<(NanBox, NanBox)> {
        self.accessors
            .iter()
            .find(|(k, _, _)| k.as_ref() == name)
            .map(|(_, g, s)| (*g, *s))
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

    /// The own **enumerable** property names (excludes keys marked hidden).
    #[must_use]
    pub fn enumerable_keys(&self) -> Vec<&str> {
        self.shape
            .keys()
            .into_iter()
            .filter(|k| !self.is_hidden(k))
            .collect()
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

    /// Deletes own property `key`, rebuilding the shape/slots from `root` without
    /// it (also drops a same-named accessor). Returns whether anything was
    /// removed.
    pub fn delete(&mut self, root: Rc<Shape>, key: &str) -> bool {
        let had_accessor = self.accessors.iter().any(|(k, _, _)| k.as_ref() == key);
        self.accessors.retain(|(k, _, _)| k.as_ref() != key);
        if !self.shape.contains(key) {
            return had_accessor;
        }
        let kept: Vec<(alloc::string::String, NanBox)> = self
            .shape
            .keys()
            .into_iter()
            .filter(|k| *k != key)
            .map(|k| (alloc::string::String::from(k), self.get(k).unwrap()))
            .collect();
        self.shape = root;
        self.slots.clear();
        for (k, v) in kept {
            self.set(&k, v);
        }
        true
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
        for slot in &mut self.slots {
            fwd(slot);
        }
        for (_, g, s) in &mut self.accessors {
            fwd(g);
            fwd(s);
        }
    }

    /// Calls `visit` for every heap [`Handle`](crate::heap::Handle) this object
    /// references through a slot — the outgoing edges a tracing collector
    /// follows.
    pub fn trace_handles(&self, mut visit: impl FnMut(crate::heap::Handle)) {
        for slot in &self.slots {
            if let Some(raw) = slot.as_handle() {
                visit(crate::heap::Handle::from_raw(raw));
            }
        }
        for (_, g, s) in &self.accessors {
            for v in [g, s] {
                if let Some(raw) = v.as_handle() {
                    visit(crate::heap::Handle::from_raw(raw));
                }
            }
        }
    }
}

impl crate::gc::Trace for Object {
    fn trace(&self, visit: &mut dyn FnMut(crate::heap::Handle)) {
        self.trace_handles(visit);
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
