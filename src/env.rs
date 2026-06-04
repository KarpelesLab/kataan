//! Lexical environments (scope chains) for the tree-walker over the new model
//! (`ROADMAP.md` §3 → Phase D migration, the function/closure piece).
//!
//! A `Scope` is a reference-counted frame of `name → `[`NanBox`] bindings with
//! a link to its enclosing scope. A function **closes over** the scope it was
//! defined in by keeping an `Rc` to it, so its captured variables stay live and
//! shared after the defining call returns — the property a flat stack of scopes
//! cannot provide. Resolution walks the parent chain inner-first.
//!
//! Bindings hold only [`NanBox`] values (heap references are handles), so a
//! scope is `'static` and the GC can trace a closure's captured handles via
//! `Scope::for_each_handle`.
//!
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! Pure, safe `alloc`-only Rust.

use crate::heap::Handle;
use crate::nanbox::NanBox;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use core::cell::RefCell;

/// A reference-counted lexical scope: its own bindings plus a link to the
/// enclosing scope. Cloning shares the same frame (closures capture by sharing).
#[derive(Clone)]
pub struct Scope(Rc<RefCell<ScopeData>>);

struct ScopeData {
    vars: BTreeMap<String, NanBox>,
    parent: Option<Scope>,
}

impl Scope {
    /// A new root scope (no parent).
    #[must_use]
    pub fn root() -> Self {
        Scope(Rc::new(RefCell::new(ScopeData {
            vars: BTreeMap::new(),
            parent: None,
        })))
    }

    /// A new child scope nested inside `self`.
    #[must_use]
    pub fn child(&self) -> Self {
        Scope(Rc::new(RefCell::new(ScopeData {
            vars: BTreeMap::new(),
            parent: Some(self.clone()),
        })))
    }

    /// Declares (or redeclares) `name` in *this* scope.
    pub fn declare(&self, name: &str, value: NanBox) {
        self.0.borrow_mut().vars.insert(String::from(name), value);
    }

    /// Looks up `name`, walking outward through enclosing scopes.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<NanBox> {
        let data = self.0.borrow();
        if let Some(v) = data.vars.get(name) {
            return Some(*v);
        }
        data.parent.as_ref().and_then(|p| p.get(name))
    }

    /// Assigns to the nearest existing binding of `name`; returns `false` if it
    /// is not declared anywhere in the chain.
    pub fn set(&self, name: &str, value: NanBox) -> bool {
        let mut data = self.0.borrow_mut();
        if let Some(slot) = data.vars.get_mut(name) {
            *slot = value;
            return true;
        }
        match &data.parent {
            Some(p) => p.set(name, value),
            None => false,
        }
    }

    /// Visits every heap [`Handle`] reachable from this scope chain's bindings
    /// (for GC tracing of a closure's captured values).
    pub fn for_each_handle(&self, visit: &mut dyn FnMut(Handle)) {
        let data = self.0.borrow();
        for v in data.vars.values() {
            if let Some(raw) = v.as_handle() {
                visit(Handle::from_raw(raw));
            }
        }
        if let Some(p) = &data.parent {
            p.for_each_handle(visit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_and_resolve_inner_first() {
        let root = Scope::root();
        root.declare("x", NanBox::number(1.0));
        root.declare("y", NanBox::number(2.0));
        let child = root.child();
        child.declare("x", NanBox::number(99.0)); // shadows

        assert_eq!(child.get("x").unwrap().as_number(), Some(99.0)); // inner
        assert_eq!(child.get("y").unwrap().as_number(), Some(2.0)); // from parent
        assert_eq!(root.get("x").unwrap().as_number(), Some(1.0)); // parent unaffected
        assert!(child.get("z").is_none());
    }

    #[test]
    fn set_targets_the_nearest_binding() {
        let root = Scope::root();
        root.declare("x", NanBox::number(1.0));
        let child = root.child();
        // No local `x`: assignment reaches the parent's binding.
        assert!(child.set("x", NanBox::number(5.0)));
        assert_eq!(root.get("x").unwrap().as_number(), Some(5.0));
        // A local shadow captures the assignment.
        child.declare("x", NanBox::number(10.0));
        assert!(child.set("x", NanBox::number(20.0)));
        assert_eq!(child.get("x").unwrap().as_number(), Some(20.0));
        assert_eq!(root.get("x").unwrap().as_number(), Some(5.0)); // parent unchanged
        // Assigning an undeclared name fails.
        assert!(!child.set("nope", NanBox::number(0.0)));
    }

    #[test]
    fn captured_scope_outlives_and_shares() {
        // A child keeps its parent alive and sees the parent's later mutations
        // (the closure-capture property).
        let captured = {
            let outer = Scope::root();
            outer.declare("count", NanBox::number(0.0));
            let inner = outer.child();
            outer.set("count", NanBox::number(7.0)); // mutate after capture
            inner // `outer` drops here, but `inner` holds an Rc to it
        };
        assert_eq!(captured.get("count").unwrap().as_number(), Some(7.0));
    }

    #[test]
    fn for_each_handle_visits_chain() {
        let root = Scope::root();
        root.declare("a", NanBox::handle(1));
        root.declare("n", NanBox::number(5.0)); // not a handle
        let child = root.child();
        child.declare("b", NanBox::handle(2));

        let mut seen = alloc::vec::Vec::new();
        child.for_each_handle(&mut |h| seen.push(h.to_raw()));
        seen.sort_unstable();
        assert_eq!(seen, alloc::vec![1, 2]);
    }
}
