//! Heap cells — the reference types the managed heap holds (`ROADMAP.md` §3).
//!
//! A NaN-boxed handle can point at any *reference* value, not just a plain
//! object: a string, an array, a function, … all live on the heap. `Cell` is
//! that sum — what a [`heap::Heap`](crate::heap::Heap) slot actually stores in
//! the performance model. This turn covers the three core kinds:
//! - `Cell::Object` — a shape + slots object,
//! - `Cell::Str` — a string *value*, backed by a [`rope::Rope`](crate::rope::Rope)
//!   so concatenation stays O(1),
//! - `Cell::Array` — a dense vector of element values.
//!
//! A cell knows how to enumerate its outgoing handles, so the collector traces a
//! mixed object/array/string graph uniformly: objects report their property
//! slots, arrays their elements, strings nothing.
//!
//! Pure, safe `alloc`-only Rust.

use crate::env::Scope;
use crate::gc::Trace;
use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::object::Object;
use crate::rope::Rope;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// A `Promise`'s settlement status.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PromiseStatus {
    /// Not yet settled.
    Pending,
    /// Settled with a value.
    Fulfilled,
    /// Settled with a reason.
    Rejected,
}

/// A `then` reaction: the handlers to run on settlement and the dependent
/// promise to settle with their result.
pub struct Reaction {
    /// Called on fulfillment (`undefined` to pass through).
    pub on_fulfilled: NanBox,
    /// Called on rejection (`undefined` to pass through).
    pub on_rejected: NanBox,
    /// The promise produced by `then`, settled with the handler's result.
    pub result: Handle,
}

/// A `Promise`'s internal state, shared (`Rc`) between the promise and its
/// resolve/reject functions.
pub struct PromiseState {
    /// The settlement status.
    pub status: PromiseStatus,
    /// The fulfillment value or rejection reason.
    pub value: NanBox,
    /// Reactions registered while pending, drained on settlement.
    pub reactions: Vec<Reaction>,
}

/// A heap-allocated reference value.
pub enum Cell {
    /// An ordinary object (shape + value slots).
    Object(Object),
    /// A string value (rope-backed for O(1) concatenation).
    Str(Rope),
    /// A dense array of element values.
    Array(Vec<NanBox>),
    /// A closure: an index into the interpreter's function table plus the
    /// lexical scope it captured at definition.
    Function {
        /// Index into the owning interpreter's function table (the AST body).
        func_id: u32,
        /// The captured lexical environment.
        env: Scope,
    },
    /// A built-in (native) function, identified by an id the interpreter maps to
    /// a Rust implementation.
    Native(u16),
    /// A native function bound to a heap value (e.g. a promise's resolve/reject,
    /// which carry the promise they settle).
    BoundNative {
        /// The built-in id.
        id: u16,
        /// The bound receiver (e.g. the promise to settle).
        target: Handle,
    },
    /// A `Promise` (its shared settlement state).
    Promise(Rc<RefCell<PromiseState>>),
    /// A class constructor: an index into the interpreter's class table plus the
    /// scope it was defined in.
    Class {
        /// Index into the owning interpreter's class table (the AST body).
        class_id: u32,
        /// The captured lexical environment.
        env: Scope,
    },
    /// A `Map` (`is_set = false`) or `Set` (`is_set = true`): insertion-ordered
    /// key/value entries (for a `Set`, the value equals the key).
    Collection {
        /// Whether this is a `Set` (vs a `Map`).
        is_set: bool,
        /// The entries, in insertion order, compared by `SameValueZero`-ish
        /// strict equality.
        entries: Vec<(NanBox, NanBox)>,
    },
}

impl Cell {
    /// The object, if this cell is one.
    #[must_use]
    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Cell::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The object mutably, if this cell is one.
    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Cell::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The string, if this cell is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&Rope> {
        match self {
            Cell::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The array elements, if this cell is one.
    #[must_use]
    pub fn as_array(&self) -> Option<&[NanBox]> {
        match self {
            Cell::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The array elements mutably, if this cell is one.
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<NanBox>> {
        match self {
            Cell::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The function's `(func_id, captured env)`, if this cell is a function.
    #[must_use]
    pub fn as_function(&self) -> Option<(u32, &Scope)> {
        match self {
            Cell::Function { func_id, env } => Some((*func_id, env)),
            _ => None,
        }
    }

    /// The built-in id, if this cell is a native function.
    #[must_use]
    pub fn as_native(&self) -> Option<u16> {
        match self {
            Cell::Native(id) => Some(*id),
            _ => None,
        }
    }

    /// The `(id, target)` if this cell is a bound native function.
    #[must_use]
    pub fn as_bound_native(&self) -> Option<(u16, Handle)> {
        match self {
            Cell::BoundNative { id, target } => Some((*id, *target)),
            _ => None,
        }
    }

    /// The shared promise state, if this cell is a promise.
    #[must_use]
    pub fn as_promise(&self) -> Option<&Rc<RefCell<PromiseState>>> {
        match self {
            Cell::Promise(p) => Some(p),
            _ => None,
        }
    }

    /// The `(class_id, captured env)`, if this cell is a class.
    #[must_use]
    pub fn as_class(&self) -> Option<(u32, &Scope)> {
        match self {
            Cell::Class { class_id, env } => Some((*class_id, env)),
            _ => None,
        }
    }

    /// The `(is_set, entries)` of a collection, mutably.
    pub fn as_collection_mut(&mut self) -> Option<(bool, &mut Vec<(NanBox, NanBox)>)> {
        match self {
            Cell::Collection { is_set, entries } => Some((*is_set, entries)),
            _ => None,
        }
    }

    /// The `(is_set, entries)` of a collection.
    #[must_use]
    pub fn as_collection(&self) -> Option<(bool, &[(NanBox, NanBox)])> {
        match self {
            Cell::Collection { is_set, entries } => Some((*is_set, entries)),
            _ => None,
        }
    }

    /// The `typeof` string for this reference value (`"string"` for strings,
    /// `"function"` for functions, `"object"` for objects and arrays — JS has no
    /// array primitive type).
    #[must_use]
    pub fn type_of(&self) -> &'static str {
        match self {
            Cell::Str(_) => "string",
            Cell::Function { .. }
            | Cell::Native(_)
            | Cell::BoundNative { .. }
            | Cell::Class { .. } => "function",
            Cell::Object(_) | Cell::Array(_) | Cell::Collection { .. } | Cell::Promise(_) => {
                "object"
            }
        }
    }
}

impl Trace for Cell {
    fn trace(&self, visit: &mut dyn FnMut(Handle)) {
        match self {
            Cell::Object(o) => o.trace_handles(visit),
            Cell::Array(elems) => {
                for e in elems {
                    if let Some(raw) = e.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                }
            }
            // A closure (or class) keeps its captured environment alive.
            Cell::Function { env, .. } | Cell::Class { env, .. } => env.for_each_handle(visit),
            // A collection's keys and values are reachable.
            Cell::Collection { entries, .. } => {
                for (k, v) in entries {
                    if let Some(raw) = k.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                    if let Some(raw) = v.as_handle() {
                        visit(Handle::from_raw(raw));
                    }
                }
            }
            // A promise keeps its value and reaction handlers reachable.
            Cell::Promise(p) => {
                let s = p.borrow();
                if let Some(raw) = s.value.as_handle() {
                    visit(Handle::from_raw(raw));
                }
                for r in &s.reactions {
                    for h in [r.on_fulfilled, r.on_rejected] {
                        if let Some(raw) = h.as_handle() {
                            visit(Handle::from_raw(raw));
                        }
                    }
                    visit(r.result);
                }
            }
            // A bound native keeps its target reachable.
            Cell::BoundNative { target, .. } => visit(*target),
            // Strings and native functions reference no handles.
            Cell::Str(_) | Cell::Native(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heap::Heap;
    use crate::shape::Shape;
    use alloc::rc::Rc;

    #[test]
    fn accessors_select_the_right_kind() {
        let obj = Cell::Object(Object::new(Shape::root()));
        assert!(obj.as_object().is_some());
        assert!(obj.as_str().is_none());
        assert!(obj.as_array().is_none());
        assert_eq!(obj.type_of(), "object");

        let s = Cell::Str(Rope::from("hi"));
        assert_eq!(s.as_str().map(Rope::materialize).as_deref(), Some("hi"));
        assert_eq!(s.type_of(), "string");

        let a = Cell::Array(alloc::vec![NanBox::number(1.0), NanBox::number(2.0)]);
        assert_eq!(a.as_array().map(<[_]>::len), Some(2));
        assert_eq!(a.type_of(), "object");
    }

    #[test]
    fn gc_traces_arrays_and_objects_uniformly() {
        // root array -> object -> (handle to) a leaf; plus an unreachable string.
        let mut heap: Heap<Cell> = Heap::new();
        let root_shape = Shape::root();

        let leaf = heap.alloc(Cell::Object(Object::new(Rc::clone(&root_shape))));
        let mut mid = Object::new(Rc::clone(&root_shape));
        mid.set("leaf", NanBox::handle(leaf.to_raw()));
        let mid_h = heap.alloc(Cell::Object(mid));
        let arr = heap.alloc(Cell::Array(alloc::vec![NanBox::handle(mid_h.to_raw())]));
        let _garbage = heap.alloc(Cell::Str(Rope::from("unreferenced")));
        assert_eq!(heap.len(), 4);

        let stats = crate::gc::collect(&mut heap, &[arr]);
        assert_eq!(stats.marked, 3); // arr -> mid -> leaf
        assert_eq!(stats.swept, 1); // the string
        assert!(heap.is_live(arr) && heap.is_live(mid_h) && heap.is_live(leaf));
    }
}
