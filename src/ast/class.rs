//! Class declaration/expression nodes.

use super::{Expr, Function, Ident, PropertyKey, Stmt};
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A class declaration or expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Class {
    /// The class name (required for declarations, optional for expressions).
    pub id: Option<Ident>,
    /// The `extends` superclass expression, if any.
    pub super_class: Option<Box<Expr>>,
    /// The class body members.
    pub body: Vec<ClassMember>,
    /// The span of the whole class.
    pub span: Span,
}

/// A member of a [`Class`] body.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ClassMember {
    /// A method, getter, setter, or constructor.
    Method(ClassMethod),
    /// A field (instance or static), with an optional initializer.
    Field(ClassField),
    /// A `static { … }` initialization block.
    StaticBlock { body: Vec<Stmt>, span: Span },
}

/// A method-like class member.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassMethod {
    /// The method key (identifier name, string/number literal, computed, or
    /// private name).
    pub key: PropertyKey,
    /// What kind of method this is.
    pub kind: MethodKind,
    /// The method's function (parameters, body, async/generator flags).
    pub value: Function,
    /// Whether this is a `static` method.
    pub is_static: bool,
    /// The span of the member.
    pub span: Span,
}

/// A class field declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassField {
    /// The field key.
    pub key: PropertyKey,
    /// The initializer expression, if any.
    pub value: Option<Expr>,
    /// Whether this is a `static` field.
    pub is_static: bool,
    /// The span of the member.
    pub span: Span,
}

/// The kind of a [`ClassMethod`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum MethodKind {
    /// An ordinary method.
    Method,
    /// A getter (`get x() { … }`).
    Get,
    /// A setter (`set x(v) { … }`).
    Set,
    /// The `constructor`.
    Constructor,
}
