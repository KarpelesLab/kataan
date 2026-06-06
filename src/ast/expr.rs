//! Expression nodes.

use super::{Arrow, AssignOp, BinaryOp, Class, Function, Ident, LogicalOp, UnaryOp, UpdateOp};
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A JavaScript expression.
///
/// Every variant carries its source [`Span`]; use [`Expr::span`] to retrieve it
/// uniformly. Literal variants store the *cooked* value (numbers as `f64`,
/// strings with escapes decoded) rather than the raw source text.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)] // variant/field names mirror the grammar
pub enum Expr {
    /// The `null` literal.
    Null(Span),
    /// A `true` / `false` literal.
    Bool { value: bool, span: Span },
    /// A numeric literal, cooked to an IEEE-754 double.
    Number { value: f64, span: Span },
    /// A `BigInt` literal. The value is kept as its normalized digit string
    /// (separators and the `n` suffix removed, any radix prefix retained) until
    /// the bignum runtime type exists to hold it.
    BigInt { digits: Box<str>, span: Span },
    /// A string literal with escapes decoded.
    Str { value: Box<str>, span: Span },
    /// A regular-expression literal `/pattern/flags` (kept as source text; the
    /// pattern is compiled by the regex engine, not here).
    Regex {
        pattern: Box<str>,
        flags: Box<str>,
        span: Span,
    },
    /// A template literal, e.g. `` `a${x}b` ``.
    Template(TemplateLiteral),
    /// A tagged template, e.g. `` tag`a${x}b` ``.
    TaggedTemplate {
        tag: Box<Expr>,
        quasi: TemplateLiteral,
        span: Span,
    },

    /// An identifier reference.
    Ident(Ident),
    /// A bare private name `#x`, valid only as the left of `#x in obj` (the
    /// ergonomic brand check).
    PrivateName(Box<str>, Span),
    /// The `this` keyword.
    This(Span),
    /// The `super` keyword (only valid inside methods; validated later).
    Super(Span),

    /// An array literal, e.g. `[1, , ...xs]`. Elisions are
    /// [`ArrayElement::Hole`].
    Array {
        elements: Vec<ArrayElement>,
        span: Span,
    },
    /// An object literal, e.g. `{ a: 1, b, ...rest }`.
    Object {
        members: Vec<ObjectMember>,
        span: Span,
    },

    /// A member access: `obj.prop`, `obj[expr]`, `obj?.prop`, `obj.#priv`.
    Member {
        object: Box<Expr>,
        property: PropertyKey,
        /// True for the optional-chaining forms (`?.`, `?.[`).
        optional: bool,
        span: Span,
    },
    /// A function call: `callee(args)` or `callee?.(args)`.
    Call {
        callee: Box<Expr>,
        arguments: Vec<Argument>,
        /// True for the optional-chaining call `callee?.(...)`.
        optional: bool,
        span: Span,
    },
    /// A `new` expression: `new Callee(args)` / `new Callee`.
    New {
        callee: Box<Expr>,
        arguments: Vec<Argument>,
        span: Span,
    },

    /// An optional-chain boundary wrapping a postfix chain that contains at least
    /// one `?.`. It defines the *short-circuit target*: if any `?.` link inside
    /// `expr` has a nullish base, the whole chain evaluates to `undefined`
    /// (subsequent non-optional `.`/`[]`/`()` links are skipped). A non-optional
    /// access on a value that merely *happens* to be nullish still throws.
    OptChain {
        /// the wrapped chain expression
        expr: Box<Expr>,
        span: Span,
    },

    /// A prefix unary operation: `!x`, `-x`, `typeof x`, …
    Unary {
        op: UnaryOp,
        argument: Box<Expr>,
        span: Span,
    },
    /// A `++` / `--` update, prefix (`++x`) or postfix (`x++`).
    Update {
        op: UpdateOp,
        prefix: bool,
        argument: Box<Expr>,
        span: Span,
    },
    /// A binary operation (arithmetic, bitwise, comparison, `in`,
    /// `instanceof`).
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// A short-circuiting logical operation (`&&`, `||`, `??`).
    Logical {
        op: LogicalOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// A ternary conditional: `test ? consequent : alternate`.
    Conditional {
        test: Box<Expr>,
        consequent: Box<Expr>,
        alternate: Box<Expr>,
        span: Span,
    },
    /// An assignment: `target = value` or a compound form.
    Assign {
        op: AssignOp,
        target: Box<Expr>,
        value: Box<Expr>,
        span: Span,
    },
    /// A comma sequence: `a, b, c`.
    Sequence { expressions: Vec<Expr>, span: Span },

    /// A function expression: `function f() { … }`.
    Function(Function),
    /// An arrow function: `x => x + 1`.
    Arrow(Arrow),
    /// A class expression: `class { … }`.
    Class(Class),

    /// A `yield` / `yield*` expression (only inside a generator).
    Yield {
        /// The yielded operand, if any (`yield;` has none).
        argument: Option<Box<Expr>>,
        /// Whether this is `yield*` (delegating).
        delegate: bool,
        span: Span,
    },
    /// An `await` expression (only inside an async function).
    Await { argument: Box<Expr>, span: Span },
}

impl Expr {
    /// The source span covered by this expression.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Expr::Null(span)
            | Expr::PrivateName(_, span)
            | Expr::This(span)
            | Expr::Super(span)
            | Expr::Bool { span, .. }
            | Expr::Number { span, .. }
            | Expr::BigInt { span, .. }
            | Expr::Str { span, .. }
            | Expr::Regex { span, .. }
            | Expr::TaggedTemplate { span, .. }
            | Expr::Array { span, .. }
            | Expr::Object { span, .. }
            | Expr::Member { span, .. }
            | Expr::Call { span, .. }
            | Expr::New { span, .. }
            | Expr::OptChain { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Update { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Logical { span, .. }
            | Expr::Conditional { span, .. }
            | Expr::Assign { span, .. }
            | Expr::Sequence { span, .. }
            | Expr::Yield { span, .. }
            | Expr::Await { span, .. } => *span,
            Expr::Template(t) => t.span,
            Expr::Ident(id) => id.span,
            Expr::Function(f) => f.span,
            Expr::Arrow(a) => a.span,
            Expr::Class(c) => c.span,
        }
    }

    /// Whether this expression is a valid target for assignment / update — an
    /// identifier, a member access, or an array/object **destructuring** pattern
    /// whose elements are themselves valid targets (with optional defaults).
    #[must_use]
    pub fn is_assignment_target(&self) -> bool {
        match self {
            Expr::Ident(_) | Expr::Member { .. } => true,
            Expr::Array { elements, .. } => elements.iter().all(|el| match el {
                ArrayElement::Hole => true,
                ArrayElement::Item(e) | ArrayElement::Spread(e) => e.is_assignment_target(),
            }),
            Expr::Object { members, .. } => members.iter().all(|m| match m {
                ObjectMember::Property { value, .. } => value.is_assignment_target(),
                ObjectMember::Spread { value, .. } => value.is_assignment_target(),
                ObjectMember::Accessor { .. } => false,
            }),
            // A default in a destructuring pattern: `[a = 1] = …`.
            Expr::Assign {
                op: crate::ast::AssignOp::Assign,
                target,
                ..
            } => target.is_assignment_target(),
            _ => false,
        }
    }
}

/// An element of an array literal.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ArrayElement {
    /// An elision (a hole), as in `[1, , 3]`.
    Hole,
    /// An ordinary element expression.
    Item(Expr),
    /// A spread element `...expr`.
    Spread(Expr),
}

/// An argument in a call or `new` argument list.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum Argument {
    /// An ordinary argument expression.
    Item(Expr),
    /// A spread argument `...expr`.
    Spread(Expr),
}

/// The key of a member access or object-literal property.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum PropertyKey {
    /// A static identifier key: `obj.foo` / `{ foo: ... }`.
    Ident(Box<str>),
    /// A private name key: `obj.#foo`.
    Private(Box<str>),
    /// A string-literal key in an object literal: `{ "a-b": ... }`.
    Str(Box<str>),
    /// A numeric-literal key in an object literal: `{ 0: ... }`.
    Number(f64),
    /// A computed key: `obj[expr]` / `{ [expr]: ... }`.
    Computed(Box<Expr>),
}

/// A member of an object literal.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ObjectMember {
    /// A `key: value` or shorthand (`{ x }`) property.
    Property {
        key: PropertyKey,
        value: Box<Expr>,
        /// True for the shorthand form `{ x }` (where `value` is the ident).
        shorthand: bool,
        span: Span,
    },
    /// A spread member `...expr`.
    Spread { value: Box<Expr>, span: Span },
    /// A getter/setter accessor: `get x() { … }` / `set x(v) { … }`.
    Accessor {
        /// True for a getter, false for a setter.
        is_getter: bool,
        key: PropertyKey,
        value: super::Function,
        span: Span,
    },
}

/// A template literal: alternating string *quasis* and interpolated
/// *expressions*. There is always exactly one more quasi than expression.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateLiteral {
    /// The literal string segments (cooked + raw).
    pub quasis: Vec<TemplateElement>,
    /// The interpolated `${ … }` expressions.
    pub expressions: Vec<Expr>,
    /// The full span of the template, backtick to backtick.
    pub span: Span,
}

/// One string segment of a [`TemplateLiteral`].
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateElement {
    /// The raw source text of the segment (escapes *not* decoded), as exposed
    /// to tag functions via `String.raw`.
    pub raw: Box<str>,
    /// The cooked text (escapes decoded). `None` if the segment contains an
    /// invalid escape sequence (legal only in tagged templates).
    pub cooked: Option<Box<str>>,
    /// The span of the segment, excluding the delimiters.
    pub span: Span,
}
