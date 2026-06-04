//! The abstract syntax tree produced by the [`parser`](crate::parser).
//!
//! The AST is a conventional boxed-enum tree: every node carries a source
//! [`Span`], and child nodes are owned via [`Box`]/[`Vec`]. It is deliberately
//! *transient* — the compiler lowers it to position-independent bytecode (which
//! is the artifact that gets serialized and cached, see `ROADMAP.md` §2.2), so
//! the AST itself is thrown away after compilation and is optimized for clarity
//! over compactness.
//!
//! This module models the full ECMAScript grammar (Phase B): expressions,
//! statements and declarations, functions and arrows, destructuring patterns,
//! classes, and module `import`/`export`.

mod class;
mod expr;
mod function;
mod module;
mod pattern;
mod stmt;

pub use class::{Class, ClassField, ClassMember, ClassMethod, MethodKind};
pub use expr::{
    Argument, ArrayElement, Expr, ObjectMember, PropertyKey, TemplateElement, TemplateLiteral,
};
pub use function::{Arrow, ArrowBody, Function, Param};
pub use module::{ExportDecl, ExportSpecifier, ImportDecl, ImportSpecifier, ModuleExportName};
pub use pattern::{
    ArrayPattern, ArrayPatternElement, BindingTarget, ObjectPattern, ObjectPatternProp,
};
pub use stmt::{
    CatchClause, ForInit, ForLeft, Program, SourceType, Stmt, SwitchCase, VarDecl, VarDeclKind,
    VarDeclarator,
};

use crate::common::Span;
use alloc::boxed::Box;

/// An identifier reference or binding name, e.g. `foo` in `foo.bar` or `x` in
/// `let x`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ident {
    /// The identifier's textual name, with any Unicode escapes already
    /// decoded by the parser.
    pub name: Box<str>,
    /// The source span of the identifier.
    pub span: Span,
}

impl Ident {
    /// Builds an identifier node.
    #[must_use]
    pub fn new(name: impl Into<Box<str>>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }
}

/// A binary operator (arithmetic, bitwise, comparison, and the relational
/// keyword operators `in` / `instanceof`). The short-circuiting `&&`, `||`, and
/// `??` are modeled separately as [`LogicalOp`] because their evaluation
/// semantics differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)] // operator names are self-describing
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// `**`
    Exp,
    Lt,
    Gt,
    LtEq,
    GtEq,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `===`
    EqEqEq,
    /// `!==`
    NotEqEq,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `>>>`
    Ushr,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    In,
    Instanceof,
}

impl BinaryOp {
    /// The canonical source spelling of the operator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        use BinaryOp::*;
        match self {
            Add => "+",
            Sub => "-",
            Mul => "*",
            Div => "/",
            Mod => "%",
            Exp => "**",
            Lt => "<",
            Gt => ">",
            LtEq => "<=",
            GtEq => ">=",
            EqEq => "==",
            NotEq => "!=",
            EqEqEq => "===",
            NotEqEq => "!==",
            Shl => "<<",
            Shr => ">>",
            Ushr => ">>>",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            In => "in",
            Instanceof => "instanceof",
        }
    }
}

/// A short-circuiting logical operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum LogicalOp {
    /// `&&`
    And,
    /// `||`
    Or,
    /// `??`
    Nullish,
}

impl LogicalOp {
    /// The canonical source spelling of the operator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LogicalOp::And => "&&",
            LogicalOp::Or => "||",
            LogicalOp::Nullish => "??",
        }
    }
}

/// A prefix unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum UnaryOp {
    /// unary `+`
    Plus,
    /// unary `-`
    Minus,
    /// `!`
    Not,
    /// `~`
    BitNot,
    Typeof,
    Void,
    Delete,
}

impl UnaryOp {
    /// The canonical source spelling of the operator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            UnaryOp::Plus => "+",
            UnaryOp::Minus => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
            UnaryOp::Typeof => "typeof",
            UnaryOp::Void => "void",
            UnaryOp::Delete => "delete",
        }
    }
}

/// The `++` / `--` update operator (used in both prefix and postfix position;
/// position is recorded on the [`Expr::Update`] node).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum UpdateOp {
    /// `++`
    Inc,
    /// `--`
    Dec,
}

impl UpdateOp {
    /// The canonical source spelling of the operator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            UpdateOp::Inc => "++",
            UpdateOp::Dec => "--",
        }
    }
}

/// An assignment operator: plain `=` or one of the compound forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
    /// `%=`
    ModAssign,
    /// `**=`
    ExpAssign,
    /// `<<=`
    ShlAssign,
    /// `>>=`
    ShrAssign,
    /// `>>>=`
    UshrAssign,
    /// `&=`
    BitAndAssign,
    /// `|=`
    BitOrAssign,
    /// `^=`
    BitXorAssign,
    /// `&&=`
    AndAssign,
    /// `||=`
    OrAssign,
    /// `??=`
    NullishAssign,
}

impl AssignOp {
    /// The canonical source spelling of the operator.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        use AssignOp::*;
        match self {
            Assign => "=",
            AddAssign => "+=",
            SubAssign => "-=",
            MulAssign => "*=",
            DivAssign => "/=",
            ModAssign => "%=",
            ExpAssign => "**=",
            ShlAssign => "<<=",
            ShrAssign => ">>=",
            UshrAssign => ">>>=",
            BitAndAssign => "&=",
            BitOrAssign => "|=",
            BitXorAssign => "^=",
            AndAssign => "&&=",
            OrAssign => "||=",
            NullishAssign => "??=",
        }
    }

    /// Whether this is the plain `=` assignment (as opposed to a compound
    /// assignment that also reads the target).
    #[must_use]
    pub fn is_plain(self) -> bool {
        matches!(self, AssignOp::Assign)
    }
}
