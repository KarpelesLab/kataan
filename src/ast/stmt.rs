//! Statement, declaration, and program nodes.

use super::{BindingTarget, Class, Expr, Function, Ident};
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A parsed compilation unit: a sequence of top-level statements plus whether
/// it was parsed as a script or a module.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The top-level statement list.
    pub body: Vec<Stmt>,
    /// Whether this was parsed as a script or an ES module.
    pub source_type: SourceType,
    /// The span covering the whole program.
    pub span: Span,
}

/// Whether a [`Program`] is a script or an ECMAScript module. (Module
/// `import`/`export` syntax is added in a later increment; the distinction is
/// recorded now so the goal-symbol choice is explicit.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceType {
    /// A non-module script.
    Script,
    /// An ECMAScript module.
    Module,
}

/// A JavaScript statement.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)] // variant/field names mirror the grammar
pub enum Stmt {
    /// An expression evaluated for its side effects: `expr;`.
    Expr { expression: Box<Expr>, span: Span },
    /// A block `{ … }` introducing a new lexical scope.
    Block { body: Vec<Stmt>, span: Span },
    /// An empty statement `;`.
    Empty { span: Span },
    /// A `var` / `let` / `const` declaration.
    Var(VarDecl),
    /// `if (test) consequent else alternate`.
    If {
        test: Box<Expr>,
        consequent: Box<Stmt>,
        alternate: Option<Box<Stmt>>,
        span: Span,
    },
    /// A C-style `for (init; test; update) body`.
    For {
        init: Option<ForInit>,
        test: Option<Box<Expr>>,
        update: Option<Box<Expr>>,
        body: Box<Stmt>,
        span: Span,
    },
    /// A `for (left in right) body` loop.
    ForIn {
        left: ForLeft,
        right: Box<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    /// A `for (left of right) body` loop.
    ForOf {
        left: ForLeft,
        right: Box<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    /// `while (test) body`.
    While {
        test: Box<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    /// `do body while (test)`.
    DoWhile {
        body: Box<Stmt>,
        test: Box<Expr>,
        span: Span,
    },
    /// `switch (discriminant) { cases }`.
    Switch {
        discriminant: Box<Expr>,
        cases: Vec<SwitchCase>,
        span: Span,
    },
    /// `try { … } catch (e) { … } finally { … }`.
    Try {
        block: Vec<Stmt>,
        handler: Option<CatchClause>,
        finalizer: Option<Vec<Stmt>>,
        span: Span,
    },
    /// `return argument?;`.
    Return {
        argument: Option<Box<Expr>>,
        span: Span,
    },
    /// `break label?;`.
    Break { label: Option<Ident>, span: Span },
    /// `continue label?;`.
    Continue { label: Option<Ident>, span: Span },
    /// `throw argument;`.
    Throw { argument: Box<Expr>, span: Span },
    /// A labeled statement: `label: body`.
    Labeled {
        label: Ident,
        body: Box<Stmt>,
        span: Span,
    },
    /// The `debugger;` statement.
    Debugger { span: Span },
    /// A `with (object) body` statement (sloppy mode only; validated later).
    With {
        object: Box<Expr>,
        body: Box<Stmt>,
        span: Span,
    },
    /// A function declaration: `function f() { … }`.
    Function(Function),
    /// A class declaration: `class C { … }`.
    Class(Class),
}

impl Stmt {
    /// The source span covered by this statement.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Stmt::Expr { span, .. }
            | Stmt::Block { span, .. }
            | Stmt::Empty { span }
            | Stmt::If { span, .. }
            | Stmt::For { span, .. }
            | Stmt::ForIn { span, .. }
            | Stmt::ForOf { span, .. }
            | Stmt::While { span, .. }
            | Stmt::DoWhile { span, .. }
            | Stmt::Switch { span, .. }
            | Stmt::Try { span, .. }
            | Stmt::Return { span, .. }
            | Stmt::Break { span, .. }
            | Stmt::Continue { span, .. }
            | Stmt::Throw { span, .. }
            | Stmt::Labeled { span, .. }
            | Stmt::Debugger { span }
            | Stmt::With { span, .. } => *span,
            Stmt::Var(decl) => decl.span,
            Stmt::Function(f) => f.span,
            Stmt::Class(c) => c.span,
        }
    }
}

/// A `var` / `let` / `const` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct VarDecl {
    /// Which declaration keyword introduced it.
    pub kind: VarDeclKind,
    /// The individual declarators (`x = 1` in `let x = 1, y = 2`).
    pub declarations: Vec<VarDeclarator>,
    /// The span of the whole declaration.
    pub span: Span,
}

/// The declaration keyword of a [`VarDecl`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum VarDeclKind {
    Var,
    Let,
    Const,
}

impl VarDeclKind {
    /// The keyword spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            VarDeclKind::Var => "var",
            VarDeclKind::Let => "let",
            VarDeclKind::Const => "const",
        }
    }
}

/// A single binding in a [`VarDecl`], e.g. `x = 1`.
#[derive(Clone, Debug, PartialEq)]
pub struct VarDeclarator {
    /// The binding target.
    pub target: BindingTarget,
    /// The optional initializer expression.
    pub init: Option<Expr>,
    /// The span of this declarator.
    pub span: Span,
}

/// The initializer of a C-style `for` loop header.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ForInit {
    /// A `var`/`let`/`const` declaration: `for (let i = 0; …)`.
    Var(VarDecl),
    /// An expression: `for (i = 0; …)`.
    Expr(Box<Expr>),
}

/// The left-hand side of a `for-in` / `for-of` loop.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ForLeft {
    /// A declaration head: `for (const x of …)`.
    Decl {
        kind: VarDeclKind,
        target: BindingTarget,
        span: Span,
    },
    /// An existing assignment target: `for (x of …)`.
    Target(Box<Expr>),
}

/// One `case` / `default` clause of a `switch`.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchCase {
    /// The case test, or `None` for the `default` clause.
    pub test: Option<Expr>,
    /// The statements of the clause.
    pub body: Vec<Stmt>,
    /// The span of the clause.
    pub span: Span,
}

/// The `catch` clause of a `try` statement.
#[derive(Clone, Debug, PartialEq)]
pub struct CatchClause {
    /// The optional catch binding (absent for `catch { … }`).
    pub param: Option<BindingTarget>,
    /// The catch block's statements.
    pub body: Vec<Stmt>,
    /// The span of the clause.
    pub span: Span,
}
