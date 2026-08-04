//! ECMAScript module `import` / `export` declaration nodes.

use super::Ident;
use crate::common::Span;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A single `key: "value"` entry of an import `with { … }` / `assert { … }`
/// clause (an *import attribute*). The key is an IdentifierName or StringLiteral
/// (stored cooked); the value is always a StringLiteral (stored cooked).
pub type ImportAttribute = (Box<str>, Box<str>);

/// An `import` declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportDecl {
    /// The imported bindings (empty for a bare `import "mod";`).
    pub specifiers: Vec<ImportSpecifier>,
    /// The module specifier string.
    pub source: Box<str>,
    /// `import defer * as ns from …` — the *defer* phase (import-defer proposal):
    /// the module is loaded and linked but not evaluated until the namespace is
    /// first accessed. Only ever set together with a single
    /// [`ImportSpecifier::Namespace`].
    pub deferred: bool,
    /// The `with { … }` / (legacy) `assert { … }` import attributes, if any
    /// (the import-attributes proposal). Empty when the clause is absent.
    pub attributes: Vec<ImportAttribute>,
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
    /// `import source x from …` — the *source* phase (source-phase-imports
    /// proposal): the binding holds the dependency's `[[ModuleSource]]` object
    /// rather than any of its exports. It is always the only specifier of its
    /// declaration (the grammar takes a bare `ImportedBinding`, with no default,
    /// namespace, or named clause alongside it).
    Source(Ident),
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
        /// `with { … }` import attributes on the re-export's `from` clause
        /// (only meaningful when `source` is `Some`).
        attributes: Vec<ImportAttribute>,
        span: Span,
    },
    /// `export * from "mod"` / `export * as ns from "mod"`.
    All {
        exported: Option<ModuleExportName>,
        source: Box<str>,
        /// `with { … }` import attributes on the `from` clause.
        attributes: Vec<ImportAttribute>,
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
