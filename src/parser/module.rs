//! Parsing of module `import` / `export` declarations. Methods on
//! [`Parser`](super::Parser).
//!
//! Dynamic `import(…)` and `import.meta` are expression-level and handled
//! elsewhere (a later increment); this module covers the static declarations.

use super::{Parser, cook};
use crate::ast::{
    ExportDecl, ExportSpecifier, Ident, ImportDecl, ImportSpecifier, ModuleExportName, Stmt,
};
use crate::error::Result;
use crate::lexer::{Keyword as Kw, TokenKind};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

impl<'src> Parser<'src> {
    // --- import ---------------------------------------------------------

    /// Parses an `import` declaration (the cursor is at `import`).
    pub(super) fn parse_import(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `import`

        // Bare side-effect import: `import "mod";`.
        if self.at(TokenKind::String) {
            let source = self.parse_module_specifier()?;
            self.semicolon()?;
            return Ok(Stmt::Import(ImportDecl {
                specifiers: Vec::new(),
                source,
                span: start.to(self.prev_span()),
            }));
        }

        let mut specifiers = Vec::new();

        // Default binding, optionally followed by `, namespace|named`.
        if self.at_binding_ident() {
            let local = self.parse_binding_ident()?;
            specifiers.push(ImportSpecifier::Default(local));
            if self.eat(TokenKind::Comma) {
                self.parse_import_tail(&mut specifiers)?;
            }
        } else {
            self.parse_import_tail(&mut specifiers)?;
        }

        self.expect_contextual(Kw::From, "from")?;
        let source = self.parse_module_specifier()?;
        self.semicolon()?;
        Ok(Stmt::Import(ImportDecl {
            specifiers,
            source,
            span: start.to(self.prev_span()),
        }))
    }

    /// Parses the namespace (`* as ns`) or named (`{ … }`) part of an import
    /// clause.
    fn parse_import_tail(&mut self, specifiers: &mut Vec<ImportSpecifier>) -> Result<()> {
        if self.eat(TokenKind::Star) {
            self.expect_contextual(Kw::As, "as")?;
            let local = self.parse_binding_ident()?;
            specifiers.push(ImportSpecifier::Namespace(local));
        } else if self.at(TokenKind::LBrace) {
            self.bump();
            while !self.at(TokenKind::RBrace) {
                let imported = self.parse_module_export_name()?;
                let local = if self.eat(TokenKind::Keyword(Kw::As)) {
                    self.parse_binding_ident()?
                } else {
                    match &imported {
                        ModuleExportName::Ident(name) => Ident::new(name.clone(), self.prev_span()),
                        ModuleExportName::Str(_) => {
                            return Err(self.err("a string-named import must be bound with `as`"));
                        }
                    }
                };
                specifiers.push(ImportSpecifier::Named { imported, local });
                if !self.eat(TokenKind::Comma) {
                    break;
                }
            }
            self.expect(TokenKind::RBrace)?;
        } else {
            return Err(self.err("expected an import clause"));
        }
        Ok(())
    }

    // --- export ---------------------------------------------------------

    /// Parses an `export` declaration (the cursor is at `export`).
    pub(super) fn parse_export(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `export`

        // `export * [as name] from "mod";`
        if self.eat(TokenKind::Star) {
            let exported = if self.eat(TokenKind::Keyword(Kw::As)) {
                Some(self.parse_module_export_name()?)
            } else {
                None
            };
            self.expect_contextual(Kw::From, "from")?;
            let source = self.parse_module_specifier()?;
            self.semicolon()?;
            return Ok(Stmt::Export(ExportDecl::All {
                exported,
                source,
                span: start.to(self.prev_span()),
            }));
        }

        // `export default …`
        if self.eat(TokenKind::Keyword(Kw::Default)) {
            let declaration = if self.at(TokenKind::Keyword(Kw::Function))
                || (self.at(TokenKind::Keyword(Kw::Async))
                    && self.nth_kind(1) == TokenKind::Keyword(Kw::Function)
                    && !self.nth_newline(1))
            {
                self.parse_default_function()?
            } else if self.at(TokenKind::Keyword(Kw::Class)) {
                self.parse_default_class()?
            } else {
                let expr = self.parse_assignment()?;
                let espan = expr.span();
                self.semicolon()?;
                Stmt::Expr {
                    expression: Box::new(expr),
                    span: espan,
                }
            };
            return Ok(Stmt::Export(ExportDecl::Default {
                declaration: Box::new(declaration),
                span: start.to(self.prev_span()),
            }));
        }

        // `export { … } [from "mod"];`
        if self.at(TokenKind::LBrace) {
            let specifiers = self.parse_export_specifiers()?;
            let source = if self.eat(TokenKind::Keyword(Kw::From)) {
                Some(self.parse_module_specifier()?)
            } else {
                None
            };
            self.semicolon()?;
            return Ok(Stmt::Export(ExportDecl::Named {
                specifiers,
                source,
                span: start.to(self.prev_span()),
            }));
        }

        // `export <declaration>` — var/let/const/function/class.
        let declaration = self.parse_statement()?;
        Ok(Stmt::Export(ExportDecl::Decl {
            span: start.to(declaration.span()),
            declaration: Box::new(declaration),
        }))
    }

    fn parse_export_specifiers(&mut self) -> Result<Vec<ExportSpecifier>> {
        self.expect(TokenKind::LBrace)?;
        let mut specifiers = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let start = self.cur_span();
            let local = self.parse_module_export_name()?;
            let exported = if self.eat(TokenKind::Keyword(Kw::As)) {
                self.parse_module_export_name()?
            } else {
                local.clone()
            };
            specifiers.push(ExportSpecifier {
                local,
                exported,
                span: start.to(self.prev_span()),
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(specifiers)
    }

    // --- shared ---------------------------------------------------------

    /// A module specifier string literal (e.g. `"./mod.js"`).
    fn parse_module_specifier(&mut self) -> Result<Box<str>> {
        let tok = self.expect(TokenKind::String)?;
        Ok(cook::string(tok.text(self.source), tok.span)?.into())
    }

    /// An import/export name — an identifier name or a string literal.
    fn parse_module_export_name(&mut self) -> Result<ModuleExportName> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::String => {
                self.bump();
                Ok(ModuleExportName::Str(
                    cook::string(tok.text(self.source), tok.span)?.into(),
                ))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok(ModuleExportName::Ident(tok.text(self.source).into()))
            }
            TokenKind::Keyword(kw) => {
                self.bump();
                Ok(ModuleExportName::Ident(kw.as_str().into()))
            }
            _ => Err(self.err(format!("expected a module name, found {:?}", tok.kind))),
        }
    }

    /// Consumes a contextual keyword (`from` / `as`), reporting `what` if it is
    /// missing.
    fn expect_contextual(&mut self, kw: Kw, what: &str) -> Result<()> {
        if self.eat(TokenKind::Keyword(kw)) {
            Ok(())
        } else {
            Err(self.err(format!("expected `{what}`, found {:?}", self.peek())))
        }
    }
}
