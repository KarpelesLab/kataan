//! Hidden classes ("shapes") — shared property-layout descriptors
//! (`ROADMAP.md` §3, the object model).
//!
//! In the interpreter era an object stores its own `Vec<(name, value)>` and
//! looks properties up by linear scan. The performance object model instead
//! separates an object's **layout** from its **values**: objects with the same
//! set of properties added in the same order share one immutable `Shape` that
//! maps each property name to a dense slot index, and the object stores only a
//! flat `Vec` of values indexed by that slot. This makes property access a slot
//! load (cacheable by an inline cache keyed on the shape) and shrinks objects.
//!
//! Shapes form a **transition tree**: the root (`Shape::root`) is the empty
//! layout, and adding a property transitions to a child shape (creating it once
//! and caching it, so every `{x, y}` built `x`-then-`y` ends up at the same
//! shape). The chain of parents records the property order; each node adds
//! exactly one property at the next slot.
//!
//! Pure, safe `alloc`-only Rust (`Rc` + `RefCell`); the object that pairs a
//! shape with a value vector, and inline caches over shapes, build on top.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// An immutable description of an object's property layout: which names map to
/// which slot indices, shared by every object with that structure. Build shapes
/// from [`Shape::root`] via [`Shape::transition`].
pub struct Shape {
    /// The shape this one extends (the layout without the last property), or
    /// `None` for the root.
    parent: Option<Rc<Shape>>,
    /// The property this shape adds over its parent (`None` for the root).
    key: Option<Box<str>>,
    /// The slot index of `key` (and `= parent.len()`); meaningless for the root.
    slot: u32,
    /// The number of properties in this layout.
    len: u32,
    /// Cached child shapes, one per property added from here — so a repeated
    /// `+key` reuses the same child rather than forking the tree.
    transitions: RefCell<Vec<(Box<str>, Rc<Shape>)>>,
}

impl Shape {
    /// The root (empty) shape. Every object's layout descends from it.
    #[must_use]
    pub fn root() -> Rc<Shape> {
        Rc::new(Shape {
            parent: None,
            key: None,
            slot: 0,
            len: 0,
            transitions: RefCell::new(Vec::new()),
        })
    }

    /// The number of properties in this layout.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether this is the empty (root) layout.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The shape reached by adding property `key` to this one. If `key` is
    /// already present this shape is returned unchanged; otherwise the child
    /// shape is created once and cached, so the same structure always maps to
    /// the same shape (structural sharing). The new property takes the next
    /// slot (`self.len()`).
    pub fn transition(self: &Rc<Self>, key: &str) -> Rc<Shape> {
        // Already present: no layout change.
        if self.lookup(key).is_some() {
            return Rc::clone(self);
        }
        // Reuse the cached transition if this `+key` was taken before.
        if let Some((_, child)) = self
            .transitions
            .borrow()
            .iter()
            .find(|(k, _)| k.as_ref() == key)
        {
            return Rc::clone(child);
        }
        let child = Rc::new(Shape {
            parent: Some(Rc::clone(self)),
            key: Some(Box::from(key)),
            slot: self.len,
            len: self.len + 1,
            transitions: RefCell::new(Vec::new()),
        });
        self.transitions
            .borrow_mut()
            .push((Box::from(key), Rc::clone(&child)));
        child
    }

    /// The slot index of `key` in this layout, or `None` if absent. Walks the
    /// parent chain (newest property first).
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<u32> {
        let mut node = self;
        loop {
            if let Some(k) = &node.key
                && k.as_ref() == key
            {
                return Some(node.slot);
            }
            node = node.parent.as_deref()?;
        }
    }

    /// Whether `key` is present in this layout.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.lookup(key).is_some()
    }

    /// The property names in slot order (slot 0 first).
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::with_capacity(self.len as usize);
        let mut node = self;
        loop {
            if let Some(k) = &node.key {
                out.push(k.as_ref());
            }
            match node.parent.as_deref() {
                Some(p) => node = p,
                None => break,
            }
        }
        out.reverse(); // collected newest-first; present in insertion order
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_is_empty() {
        let root = Shape::root();
        assert!(root.is_empty());
        assert_eq!(root.len(), 0);
        assert_eq!(root.lookup("x"), None);
        assert!(root.keys().is_empty());
    }

    #[test]
    fn transition_assigns_sequential_slots() {
        let s = Shape::root();
        let s = s.transition("x");
        let s = s.transition("y");
        let s = s.transition("z");
        assert_eq!(s.len(), 3);
        assert_eq!(s.lookup("x"), Some(0));
        assert_eq!(s.lookup("y"), Some(1));
        assert_eq!(s.lookup("z"), Some(2));
        assert_eq!(s.lookup("w"), None);
        assert_eq!(s.keys(), ["x", "y", "z"]);
    }

    #[test]
    fn same_structure_shares_one_shape() {
        // Two objects built {a, b} from the same root reach the *same* shape
        // (structural sharing holds within a root's transition tree).
        let root = Shape::root();
        let o1 = root.transition("a").transition("b");
        let o2 = root.transition("a").transition("b");
        assert!(Rc::ptr_eq(&o1, &o2));
    }

    #[test]
    fn divergent_structure_forks_the_tree() {
        let base = Shape::root().transition("a");
        let with_b = base.transition("b");
        let with_c = base.transition("c");
        assert!(!Rc::ptr_eq(&with_b, &with_c));
        assert_eq!(with_b.lookup("b"), Some(1));
        assert_eq!(with_b.lookup("c"), None);
        assert_eq!(with_c.lookup("c"), Some(1));
        // The shared prefix is the same node.
        assert_eq!(with_b.lookup("a"), Some(0));
        assert_eq!(with_c.lookup("a"), Some(0));
    }

    #[test]
    fn re_adding_an_existing_key_is_a_no_op() {
        let s = Shape::root().transition("x").transition("y");
        let again = s.transition("x");
        assert!(Rc::ptr_eq(&s, &again));
        assert_eq!(again.len(), 2);
    }

    #[test]
    fn property_order_differs_by_insertion_order() {
        let xy = Shape::root().transition("x").transition("y");
        let yx = Shape::root().transition("y").transition("x");
        // Different layouts (different slot assignment), so different shapes.
        assert!(!Rc::ptr_eq(&xy, &yx));
        assert_eq!(xy.lookup("x"), Some(0));
        assert_eq!(yx.lookup("x"), Some(1));
        assert_eq!(xy.keys(), ["x", "y"]);
        assert_eq!(yx.keys(), ["y", "x"]);
    }
}
