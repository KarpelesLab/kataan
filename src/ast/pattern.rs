//! Binding targets and destructuring patterns — what a declaration, parameter,
//! `catch` clause, or `for`-head binds to.

use super::{Expr, Ident, PropertyKey};
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A binding target: a plain name or a destructuring pattern.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum BindingTarget {
    /// A simple identifier binding.
    Ident(Ident),
    /// An array destructuring pattern: `[a, , b = 1, ...rest]`.
    Array(ArrayPattern),
    /// An object destructuring pattern: `{ a, b: c, d = 1, ...rest }`.
    Object(ObjectPattern),
}

/// An array destructuring pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayPattern {
    /// The pattern elements, in order (including elisions).
    pub elements: Vec<ArrayPatternElement>,
    /// The span of the whole pattern.
    pub span: Span,
}

/// One element of an [`ArrayPattern`].
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ArrayPatternElement {
    /// An elision (a hole), as in `[a, , b]`.
    Hole,
    /// A bound element with an optional default: `a` or `a = 1`.
    Item {
        target: BindingTarget,
        default: Option<Expr>,
        span: Span,
    },
    /// A trailing rest element `...target` (must be last, no default).
    Rest { target: BindingTarget, span: Span },
}

/// An object destructuring pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectPattern {
    /// The bound properties.
    pub properties: Vec<ObjectPatternProp>,
    /// A trailing rest binding `...rest`, if present.
    pub rest: Option<Box<BindingTarget>>,
    /// The span of the whole pattern.
    pub span: Span,
}

/// One property of an [`ObjectPattern`], e.g. `a`, `b: c`, or `d = 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectPatternProp {
    /// The property key being read from the source object.
    pub key: PropertyKey,
    /// The binding target the value is bound to.
    pub value: BindingTarget,
    /// A default applied when the read value is `undefined`.
    pub default: Option<Expr>,
    /// Whether this used the shorthand form `{ a }` (vs `{ a: target }`).
    pub shorthand: bool,
    /// The span of the property.
    pub span: Span,
}
