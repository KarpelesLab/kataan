//! Lexical environments (scope chains) for the tree-walking interpreter.

use super::value::Value;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use core::cell::RefCell;

/// A reference-counted, shareable lexical scope. Closures capture an `Env`,
/// keeping their defining scope alive.
pub type Env<'a> = Rc<Scope<'a>>;

/// One lexical scope: a set of bindings plus a link to its enclosing scope.
pub struct Scope<'a> {
    bindings: RefCell<BTreeMap<Box<str>, Binding<'a>>>,
    parent: Option<Env<'a>>,
}

/// A single binding and whether it may be reassigned (`const` cannot).
struct Binding<'a> {
    value: Value<'a>,
    mutable: bool,
}

impl<'a> Scope<'a> {
    /// Creates a new top-level (global) scope.
    #[must_use]
    pub fn new_global() -> Env<'a> {
        Rc::new(Scope {
            bindings: RefCell::new(BTreeMap::new()),
            parent: None,
        })
    }

    /// Creates a child scope nested inside `parent`.
    #[must_use]
    pub fn child(parent: &Env<'a>) -> Env<'a> {
        Rc::new(Scope {
            bindings: RefCell::new(BTreeMap::new()),
            parent: Some(Rc::clone(parent)),
        })
    }

    /// Declares (or redeclares) a binding in *this* scope.
    pub fn declare(&self, name: &str, value: Value<'a>, mutable: bool) {
        self.bindings
            .borrow_mut()
            .insert(name.into(), Binding { value, mutable });
    }

    /// Looks a name up through the scope chain.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Value<'a>> {
        if let Some(b) = self.bindings.borrow().get(name) {
            return Some(b.value.clone());
        }
        self.parent.as_ref().and_then(|p| p.get(name))
    }

    /// Whether `name` is bound anywhere in the chain.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.bindings.borrow().contains_key(name)
            || self.parent.as_ref().is_some_and(|p| p.has(name))
    }

    /// Assigns to an existing binding, searching outward.
    pub fn assign(&self, name: &str, value: Value<'a>) -> AssignOutcome {
        if let Some(b) = self.bindings.borrow_mut().get_mut(name) {
            if !b.mutable {
                return AssignOutcome::Immutable;
            }
            b.value = value;
            return AssignOutcome::Assigned;
        }
        match &self.parent {
            Some(p) => p.assign(name, value),
            None => AssignOutcome::Unbound,
        }
    }
}

/// The result of [`Scope::assign`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignOutcome {
    /// The binding was found and updated.
    Assigned,
    /// No binding with that name exists in the chain.
    Unbound,
    /// The binding exists but is `const` (immutable).
    Immutable,
}
