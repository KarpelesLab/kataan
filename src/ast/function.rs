//! Function, parameter, and arrow nodes — shared by both the statement form
//! (function declarations) and the expression forms (function expressions and
//! arrows).

use super::{BindingTarget, Expr, Ident, Stmt};
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A function declaration or expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    /// The function name. Required for declarations, optional for expressions.
    pub id: Option<Ident>,
    /// The parameter list.
    pub params: Vec<Param>,
    /// The body's statement list.
    pub body: Vec<Stmt>,
    /// Whether this is an `async function`.
    pub is_async: bool,
    /// Whether this is a generator (`function*`).
    pub is_generator: bool,
    /// The span of the whole function.
    pub span: Span,
}

/// One formal parameter of a function or arrow.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// The binding target (an identifier for now; destructuring patterns are
    /// added with the pattern grammar).
    pub target: BindingTarget,
    /// A default-value expression (`= expr`), if any. Mutually exclusive with
    /// [`rest`](Self::rest).
    pub default: Option<Expr>,
    /// Whether this is a rest parameter (`...x`). A rest parameter must be last
    /// and may not have a default.
    pub rest: bool,
    /// The span of the parameter.
    pub span: Span,
}

/// An arrow function `params => body`.
#[derive(Clone, Debug, PartialEq)]
pub struct Arrow {
    /// The parameter list.
    pub params: Vec<Param>,
    /// The body — either a concise expression or a block.
    pub body: ArrowBody,
    /// Whether this is an `async` arrow.
    pub is_async: bool,
    /// The span of the whole arrow.
    pub span: Span,
}

/// The body of an [`Arrow`].
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ArrowBody {
    /// A concise body: `x => x + 1`.
    Expr(Box<Expr>),
    /// A block body: `x => { return x; }`.
    Block(Vec<Stmt>),
}
