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
    /// Whether the current token is an `Identifier` whose name is exactly `word`
    /// (used for the contextual `defer` phase keyword, which is not a reserved
    /// word in the lexer).
    fn at_contextual_ident(&self, word: &str) -> bool {
        self.peek() == TokenKind::Identifier && self.ident_name(self.peek_tok()) == word
    }

    /// Whether the token `n` positions ahead could start a `BindingIdentifier`
    /// (the lookahead form of [`Parser::at_binding_ident`]).
    fn nth_starts_binding_ident(&self, n: usize) -> bool {
        match self.nth_kind(n) {
            TokenKind::Identifier => true,
            TokenKind::Keyword(kw) => self.keyword_is_binding_ident(kw),
            _ => false,
        }
    }

    // --- import ---------------------------------------------------------

    /// Parses an `import` declaration (the cursor is at `import`).
    pub(super) fn parse_import(&mut self) -> Result<Stmt> {
        // An `import` declaration is a `ModuleItem`: legal only at the module top
        // level. (Dynamic `import(…)` / `import.meta` are expressions, dispatched
        // separately and allowed anywhere.)
        if !self.module_top_level {
            return Err(self.err("`import` is only allowed at the top level of a module"));
        }
        let start = self.bump().span; // `import`

        // Bare side-effect import: `import "mod";`.
        if self.at(TokenKind::String) {
            let source = self.parse_module_specifier()?;
            let attributes = self.parse_import_attributes()?;
            self.semicolon()?;
            return Ok(Stmt::Import(ImportDecl {
                specifiers: Vec::new(),
                source,
                deferred: false,
                attributes,
                span: start.to(self.prev_span()),
            }));
        }

        let mut specifiers = Vec::new();

        // `import defer * as ns from "mod";` (import-defer proposal). `defer` is
        // contextual: it is the *defer phase* only when immediately followed by a
        // `*` namespace clause; everywhere else (`import defer from …`,
        // `import defer, {x} from …`) it is an ordinary identifier (a default
        // binding named `defer`). A `\u`-escaped spelling is not the keyword.
        let deferred = self.at_contextual_ident("defer")
            && !self.peek_tok().had_escape
            && self.nth_kind(1) == TokenKind::Star;
        // `import source x from "mod";` (source-phase-imports proposal). Like
        // `defer`, `source` is contextual and unescaped-only, but it is *not*
        // disambiguated by a following `*`: the phase form takes a plain
        // `ImportedBinding`. The clashing shapes are
        //   `import source from "mod";`      — a default binding named `source`
        //   `import source, { x } from "m";` — ditto, with a named clause
        //   `import source x    from "mod";` — the source phase, binding `x`
        //   `import source from from "mod";` — the source phase, binding `from`
        // so it is the source phase exactly when a `BindingIdentifier` follows,
        // *except* when that identifier is the `from` of a `FromClause` — i.e.
        // `from` immediately followed by the module specifier string.
        let source_phase = self.at_contextual_ident("source")
            && !self.peek_tok().had_escape
            && self.nth_starts_binding_ident(1)
            && !(self.nth_kind(1) == TokenKind::Keyword(Kw::From)
                && self.nth_kind(2) == TokenKind::String);
        if source_phase {
            self.bump(); // `source`
            let local = self.parse_binding_ident()?;
            specifiers.push(ImportSpecifier::Source(local));
        } else if deferred {
            self.bump(); // `defer`
            self.parse_import_tail(&mut specifiers)?; // requires `* as ns`
        } else if self.at_binding_ident() {
            // Default binding, optionally followed by `, namespace|named`.
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
        let attributes = self.parse_import_attributes()?;
        self.semicolon()?;
        Ok(Stmt::Import(ImportDecl {
            specifiers,
            source,
            deferred,
            attributes,
            span: start.to(self.prev_span()),
        }))
    }

    /// Parses an optional `WithClause` (`with { … }`) or the legacy
    /// `AssertClause` (`assert { … }`) that may trail a module specifier
    /// (import-attributes proposal). Returns the cooked key/value attribute
    /// pairs, or an empty vec when no clause is present.
    ///
    /// Grammar:
    /// ```text
    /// WithClause : AttributesKeyword { WithEntries_opt ,opt }
    /// AttributesKeyword : with | [no LineTerminator here] assert
    /// WithEntries : AttributeKey : StringLiteral (, WithEntries)?
    /// AttributeKey : IdentifierName | StringLiteral
    /// ```
    ///
    /// A duplicate `AttributeKey` is an early SyntaxError.
    fn parse_import_attributes(&mut self) -> Result<Vec<(Box<str>, Box<str>)>> {
        // `with` may be preceded by a line terminator; the legacy `assert`
        // keyword (a contextual identifier) may not, and an escaped spelling is
        // never the keyword.
        let is_with = self.at(TokenKind::Keyword(Kw::With));
        let is_assert = self.at_contextual_ident("assert")
            && !self.peek_tok().newline_before
            && !self.peek_tok().had_escape;
        if !is_with && !is_assert {
            return Ok(Vec::new());
        }
        self.bump(); // `with` / `assert`
        self.expect(TokenKind::LBrace)?;
        let mut attributes: Vec<(Box<str>, Box<str>)> = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let (key, key_span) = self.parse_attribute_key()?;
            self.expect(TokenKind::Colon)?;
            // The value must be a StringLiteral.
            let vtok = self.expect(TokenKind::String)?;
            let value: Box<str> = cook::string_key(vtok.text(self.source), vtok.span)?.into();
            // Early error: duplicate AttributeKey.
            if attributes.iter().any(|(k, _)| **k == *key) {
                return Err(
                    self.err_at(key_span, format!("duplicate import attribute key '{key}'"))
                );
            }
            attributes.push((key, value));
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(attributes)
    }

    /// An `AttributeKey`: an IdentifierName (including reserved words) or a
    /// StringLiteral. Returns the cooked key text and its span.
    fn parse_attribute_key(&mut self) -> Result<(Box<str>, crate::common::Span)> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::String => {
                self.bump();
                Ok((
                    cook::string_key(tok.text(self.source), tok.span)?.into(),
                    tok.span,
                ))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok((self.ident_name(tok).into(), tok.span))
            }
            TokenKind::Keyword(kw) => {
                self.bump();
                Ok((kw.as_str().into(), tok.span))
            }
            _ => Err(self.err(format!(
                "expected an import attribute key, found {:?}",
                tok.kind
            ))),
        }
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
        // An `export` declaration is a `ModuleItem`: legal only at the module top
        // level, never nested in a block, function, control body, or `switch`.
        if !self.module_top_level {
            return Err(self.err("`export` is only allowed at the top level of a module"));
        }
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
            let attributes = self.parse_import_attributes()?;
            self.semicolon()?;
            return Ok(Stmt::Export(ExportDecl::All {
                exported,
                source,
                attributes,
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
            } else if self.at(TokenKind::At) {
                // `export default @dec … class { … }` — a decorated default
                // class. Decorators are parsed and discarded (no-op).
                self.parse_decorators()?;
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
            let (source, attributes) = if self.eat(TokenKind::Keyword(Kw::From)) {
                let src = self.parse_module_specifier()?;
                let attrs = self.parse_import_attributes()?;
                (Some(src), attrs)
            } else {
                (None, Vec::new())
            };
            self.semicolon()?;
            return Ok(Stmt::Export(ExportDecl::Named {
                specifiers,
                source,
                attributes,
                span: start.to(self.prev_span()),
            }));
        }

        // `export <declaration>` — only a `HoistableDeclaration`
        // (function/generator/async/async-generator), a `ClassDeclaration`, a
        // `VariableStatement` (`var`), or a `LexicalDeclaration` (`let`/`const`)
        // may follow. Any other statement (`if`, `for`, `while`, `try`, a block,
        // a labeled statement, a bare expression, a method/getter shorthand, …)
        // is a SyntaxError at this position. Guard *before* parsing so we reject
        // at the right place rather than accepting an arbitrary statement.
        let decl_ok = matches!(
            self.peek(),
            TokenKind::Keyword(Kw::Function | Kw::Class | Kw::Var | Kw::Let | Kw::Const)
        ) || self.at(TokenKind::At)
            || (self.at(TokenKind::Keyword(Kw::Async))
                && self.nth_kind(1) == TokenKind::Keyword(Kw::Function)
                && !self.nth_newline(1));
        if !decl_ok {
            return Err(
                self.err("`export` must be followed by a declaration, `default`, `*`, or `{ … }`")
            );
        }
        // This is a declaration position, so a leading `let` is a
        // `LexicalDeclaration` (parse it as a `StatementListItem`).
        let declaration = self.parse_statement_item()?;
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
        Ok(cook::string_key(tok.text(self.source), tok.span)?.into())
    }

    /// An import/export name — an identifier name or a string literal.
    fn parse_module_export_name(&mut self) -> Result<ModuleExportName> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::String => {
                self.bump();
                // A `ModuleExportName : StringLiteral` must be well-formed Unicode
                // (no lone UTF-16 surrogate). Check the WTF-8 cooked bytes, which
                // preserve surrogates, before the lossy `string_key` collapses them.
                let bytes = cook::string(tok.text(self.source), tok.span)?;
                if !crate::wtf8::is_well_formed_utf16(&bytes) {
                    return Err(self.err_at(
                        tok.span,
                        "a module export name string literal may not contain a lone surrogate",
                    ));
                }
                Ok(ModuleExportName::Str(
                    crate::wtf8::to_string_lossy(&bytes).into(),
                ))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok(ModuleExportName::Ident(self.ident_name(tok).into()))
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
