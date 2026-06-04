//! ECMAScript module `import` / `export` declaration nodes.

use super::Ident;
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// An `import` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportDecl {
    /// The imported bindings (empty for a bare `import "mod";`).
    pub specifiers: Vec<ImportSpecifier>,
    /// The module specifier string.
    pub source: Box<str>,
    /// The span of the whole declaration.
    pub span: Span,
}

/// A single binding introduced by an [`ImportDecl`].
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ImportSpecifier {
    /// `import x from …` — the default binding.
    Default(Ident),
    /// `import * as ns from …` — the namespace binding.
    Namespace(Ident),
    /// `import { a as b } from …` — a named binding.
    Named {
        imported: ModuleExportName,
        local: Ident,
    },
}

/// A name used in an import/export clause — either an identifier name or a
/// string literal (the arbitrary-module-namespace-names proposal).
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ModuleExportName {
    Ident(Box<str>),
    Str(Box<str>),
}

/// An `export` declaration.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum ExportDecl {
    /// `export { a, b as c }` or `export { … } from "mod"` (re-export).
    Named {
        specifiers: Vec<ExportSpecifier>,
        source: Option<Box<str>>,
        span: Span,
    },
    /// `export * from "mod"` / `export * as ns from "mod"`.
    All {
        exported: Option<ModuleExportName>,
        source: Box<str>,
        span: Span,
    },
    /// `export default …` (the declaration is a function/class declaration or
    /// an expression statement).
    Default {
        declaration: Box<super::Stmt>,
        span: Span,
    },
    /// `export <declaration>` — a `var`/`let`/`const`/function/class
    /// declaration that is also exported.
    Decl {
        declaration: Box<super::Stmt>,
        span: Span,
    },
}

impl ExportDecl {
    /// The source span of this export declaration.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            ExportDecl::Named { span, .. }
            | ExportDecl::All { span, .. }
            | ExportDecl::Default { span, .. }
            | ExportDecl::Decl { span, .. } => *span,
        }
    }
}

/// One specifier of an `export { … }` clause.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportSpecifier {
    /// The local name being exported.
    pub local: ModuleExportName,
    /// The name it is exported as (equal to `local` when there is no `as`).
    pub exported: ModuleExportName,
    /// The span of the specifier.
    pub span: Span,
}
