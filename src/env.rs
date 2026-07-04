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
    /// Names declared `const` in this frame (reassignment is a TypeError).
    consts: alloc::collections::BTreeSet<String>,
    parent: Option<Scope>,
    /// Explicit-resource-management disposers recorded by `using` / `await using`
    /// declarations bound *in this frame*, in declaration order. Each entry is
    /// `(resourceValue, disposeMethod, isAsync)`; `disposeMethod` is `undefined`
    /// for a `null`/`undefined` resource (a recorded no-op). They are run in
    /// reverse (LIFO) order when this scope is exited (see the disposal driver in
    /// `nbexec`). `None` until the first `using` is recorded, so an ordinary
    /// scope carries no extra allocation (the fast path).
    disposers: Option<alloc::vec::Vec<(NanBox, NanBox, bool)>>,
    /// The object environment record of a `with (obj)` statement, set ONLY on the
    /// child scope that the `with` body runs in (`None` on every other frame).
    /// Resolution interleaves this frame's object with the surrounding lexical
    /// frames (see `Interp::with_binding`), so an inner local binding shadows the
    /// object and the object shadows an outer binding. Because the body runs in
    /// this child, a closure created inside a `with` captures the object
    /// naturally — `with` is *lexically* scoped: a function defined elsewhere and
    /// merely *called* inside the `with` does not see the object.
    with_obj: Option<NanBox>,
    /// Set ONLY on a module's top-level scope: that module's import-alias table
    /// (`local name → (exporting module scope, exported local name)`). A function
    /// defined in the module captures this scope, so `Interp::invoke_inner` can
    /// restore the correct `module_imports` when the function runs — even when
    /// called from *another* module — by walking the closure's scope chain to the
    /// nearest module frame (`Scope::module_imports`). `None` on ordinary frames.
    #[allow(clippy::type_complexity)]
    module_imports: Option<Rc<BTreeMap<String, (Scope, String)>>>,
}

impl Scope {
    /// A new root scope (no parent).
    #[must_use]
    pub fn root() -> Self {
        Scope(Rc::new(RefCell::new(ScopeData {
            vars: BTreeMap::new(),
            consts: alloc::collections::BTreeSet::new(),
            parent: None,
            disposers: None,
            with_obj: None,
            module_imports: None,
        })))
    }

    /// A new child scope nested inside `self`.
    #[must_use]
    pub fn child(&self) -> Self {
        Scope(Rc::new(RefCell::new(ScopeData {
            vars: BTreeMap::new(),
            consts: alloc::collections::BTreeSet::new(),
            parent: Some(self.clone()),
            disposers: None,
            with_obj: None,
            module_imports: None,
        })))
    }

    /// A new child scope that enters a `with (obj)` object environment. The body
    /// of a `with` statement runs in this child so the object is captured
    /// lexically by any closure created inside it.
    #[must_use]
    pub fn child_with(&self, obj: NanBox) -> Self {
        let child = self.child();
        child.0.borrow_mut().with_obj = Some(obj);
        child
    }

    /// The `with` object introduced *at this frame* (`None` unless this is the
    /// child scope of a `with` statement).
    #[must_use]
    pub fn with_obj(&self) -> Option<NanBox> {
        self.0.borrow().with_obj
    }

    /// Records this frame as a module's top-level scope carrying `imports` (its
    /// import-alias table). Set once, at link time.
    #[allow(clippy::type_complexity)]
    pub fn set_module_imports(&self, imports: Rc<BTreeMap<String, (Scope, String)>>) {
        self.0.borrow_mut().module_imports = Some(imports);
    }

    /// The import-alias table of the nearest enclosing module scope (this frame or
    /// an ancestor), or `None` if this scope chain is not inside a module — used to
    /// restore `module_imports` when a module function runs (so a named import read
    /// inside it resolves even when it is called from another module).
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn module_imports(&self) -> Option<Rc<BTreeMap<String, (Scope, String)>>> {
        let data = self.0.borrow();
        if let Some(mi) = &data.module_imports {
            return Some(mi.clone());
        }
        let parent = data.parent.clone();
        drop(data);
        parent.and_then(|p| p.module_imports())
    }

    /// Declares (or redeclares) `name` in *this* scope.
    pub fn declare(&self, name: &str, value: NanBox) {
        self.0.borrow_mut().vars.insert(String::from(name), value);
    }

    /// Declares `name` as a `const` binding in *this* scope (reassignment fails).
    pub fn declare_const(&self, name: &str, value: NanBox) {
        let mut data = self.0.borrow_mut();
        data.vars.insert(String::from(name), value);
        data.consts.insert(String::from(name));
    }

    /// Whether the nearest binding of `name` was declared `const`.
    #[must_use]
    pub fn is_const(&self, name: &str) -> bool {
        let data = self.0.borrow();
        if data.vars.contains_key(name) {
            return data.consts.contains(name);
        }
        data.parent.as_ref().is_some_and(|p| p.is_const(name))
    }

    /// Whether `name` is bound in *this* scope (not the enclosing chain).
    #[must_use]
    pub fn has_local(&self, name: &str) -> bool {
        self.0.borrow().vars.contains_key(name)
    }

    /// The scope frame that currently binds `name` (the innermost one whose own
    /// bindings include it), walking outward. `None` if no lexical frame binds it
    /// (e.g. it lives only as a global-object property). Used to capture a
    /// declarative reference *before* a side-effecting RHS, so a binding created
    /// mid-RHS (by direct `eval`) cannot hijack the write target.
    #[must_use]
    pub fn owner_frame(&self, name: &str) -> Option<Scope> {
        if self.has_local(name) {
            return Some(self.clone());
        }
        self.parent().and_then(|p| p.owner_frame(name))
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

    /// This frame's own `(name, value, is_const)` bindings (not the parent
    /// chain) — for snapshotting a closure's captured environment.
    #[must_use]
    pub fn local_bindings(&self) -> alloc::vec::Vec<(String, NanBox, bool)> {
        let data = self.0.borrow();
        data.vars
            .iter()
            .map(|(k, v)| (k.clone(), *v, data.consts.contains(k)))
            .collect()
    }

    /// This scope's enclosing scope, if any.
    #[must_use]
    pub fn parent(&self) -> Option<Scope> {
        self.0.borrow().parent.clone()
    }

    /// Records a `using` / `await using` disposer in *this* frame:
    /// `(resourceValue, disposeMethod, isAsync)`. Disposers are run in reverse
    /// declaration order when the scope is exited.
    pub fn add_disposer(&self, value: NanBox, method: NanBox, is_async: bool) {
        self.0
            .borrow_mut()
            .disposers
            .get_or_insert_with(alloc::vec::Vec::new)
            .push((value, method, is_async));
    }

    /// Whether this frame has any recorded `using` disposers.
    #[must_use]
    pub fn has_disposers(&self) -> bool {
        self.0
            .borrow()
            .disposers
            .as_ref()
            .is_some_and(|d| !d.is_empty())
    }

    /// Removes and returns this frame's recorded disposers (in declaration
    /// order), leaving the frame with none — so disposal runs exactly once even
    /// if the scope is revisited.
    #[must_use]
    pub fn take_disposers(&self) -> alloc::vec::Vec<(NanBox, NanBox, bool)> {
        self.0.borrow_mut().disposers.take().unwrap_or_default()
    }

    /// Whether `self` and `other` are the *same* scope record (identity, not
    /// contents) — used to detect when execution is running directly in the
    /// global scope (so a `var`/`function` declaration there also publishes a
    /// property on the global object).
    #[must_use]
    pub fn ptr_eq(&self, other: &Scope) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
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
        // Pending `using` disposers also root their resource value and method.
        if let Some(disposers) = &data.disposers {
            for (value, method, _) in disposers {
                if let Some(raw) = value.as_handle() {
                    visit(Handle::from_raw(raw));
                }
                if let Some(raw) = method.as_handle() {
                    visit(Handle::from_raw(raw));
                }
            }
        }
        // A `with` object is rooted by the scope that introduced it.
        if let Some(obj) = &data.with_obj
            && let Some(raw) = obj.as_handle()
        {
            visit(Handle::from_raw(raw));
        }
        if let Some(p) = &data.parent {
            p.for_each_handle(visit);
        }
    }

    /// Rewrites every handle binding in this scope chain through `forward` — the
    /// mutating mirror of [`for_each_handle`](Scope::for_each_handle), for a
    /// moving collector. (A shared parent is visited once per referrer, which is
    /// idempotent: `forward` maps an already-forwarded handle to itself.)
    pub fn relocate_handles(&self, forward: &dyn Fn(Handle) -> Handle) {
        let mut data = self.0.borrow_mut();
        for v in data.vars.values_mut() {
            if let Some(raw) = v.as_handle() {
                *v = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
            }
        }
        if let Some(disposers) = &mut data.disposers {
            for (value, method, _) in disposers.iter_mut() {
                if let Some(raw) = value.as_handle() {
                    *value = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
                }
                if let Some(raw) = method.as_handle() {
                    *method = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
                }
            }
        }
        if let Some(obj) = &mut data.with_obj
            && let Some(raw) = obj.as_handle()
        {
            *obj = NanBox::handle(forward(Handle::from_raw(raw)).to_raw());
        }
        if let Some(p) = &data.parent {
            p.relocate_handles(forward);
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
