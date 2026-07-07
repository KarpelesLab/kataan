//! Statement, declaration, and program parsing, plus Automatic Semicolon
//! Insertion. These are methods on [`Parser`](super::Parser); the expression
//! grammar lives in the parent module.

use super::{Parser, cook};
use crate::ast::{
    ArrayPattern, ArrayPatternElement, BindingTarget, CatchClause, Expr, ForInit, ForLeft, Ident,
    ObjectPattern, ObjectPatternProp, Program, PropertyKey, SourceType, Stmt, SwitchCase, VarDecl,
    VarDeclKind, VarDeclarator,
};
use crate::common::Span;
use crate::error::Result;
use crate::lexer::{Keyword as Kw, TokenKind};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

/// The shape of a `for`-loop header, determined while the `no_in` restriction
/// is in effect, then acted on once it has been lifted.
enum ForHead {
    /// `for (;` — no initializer.
    Empty,
    /// `for (init;` — a classic C-style loop (the first `;` is consumed).
    Classic(Option<ForInit>),
    /// `for (left in/of` — an iteration loop.
    InOf { left: ForLeft, is_of: bool },
}

impl<'src> Parser<'src> {
    /// Parses a whole compilation unit. The goal symbol is inferred: a unit
    /// containing a top-level `import`/`export` is a module, otherwise a
    /// script. (Strict-mode and top-level-only semantics are validated in a
    /// later phase.)
    pub fn parse_program(source: &'src str) -> Result<Program> {
        let mut p = Parser::new(source)?;
        // The program top level is a `ModuleItem` position (a top-level
        // `import`/`export` makes the unit a module); nested `import`/`export`
        // remain illegal. A pure script has no such tokens, so the flag is inert.
        p.module_top_level = true;
        let body = p.parse_statement_list(TokenKind::Eof)?;
        p.expect(TokenKind::Eof)?;
        let source_type = if body
            .iter()
            .any(|s| matches!(s, Stmt::Import(_) | Stmt::Export(_)))
        {
            SourceType::Module
        } else {
            SourceType::Script
        };
        let program = Program {
            body,
            source_type,
            span: Span::new(0, source.len() as u32),
        };
        // Static-semantics early errors (private names, lexical redeclaration,
        // strict-mode rules, …) are enforced as a post-parse pass so they surface
        // as a parse-phase `SyntaxError`.
        super::validate::validate_program(&program)?;
        Ok(program)
    }

    /// Parses a **direct-eval** program that inherits a `super` context from the
    /// calling code. Identical to [`Parser::parse_program`] except the static
    /// `super`-reference check is relaxed per the caller's home-object context
    /// (`allow_super_property` inside a method/accessor/constructor/field
    /// initializer/static block; `allow_super_call` inside a derived-class
    /// constructor).
    ///
    /// # Errors
    /// Returns a parse-phase `SyntaxError` on malformed input.
    pub fn parse_eval_program(
        source: &'src str,
        allow_super_property: bool,
        allow_super_call: bool,
        allow_new_target: bool,
        inherited_strict: bool,
    ) -> Result<Program> {
        let mut p = Parser::new(source)?;
        p.module_top_level = true;
        let body = p.parse_statement_list(TokenKind::Eof)?;
        p.expect(TokenKind::Eof)?;
        let source_type = if body
            .iter()
            .any(|s| matches!(s, Stmt::Import(_) | Stmt::Export(_)))
        {
            SourceType::Module
        } else {
            SourceType::Script
        };
        let program = Program {
            body,
            source_type,
            span: Span::new(0, source.len() as u32),
        };
        super::validate::validate_program_with(
            &program,
            allow_super_property,
            allow_super_call,
            allow_new_target,
            inherited_strict,
        )?;
        Ok(program)
    }

    /// Parses a source as an ECMAScript **module** (the `Module` goal symbol):
    /// the top level is module-strict and permits top-level `await` (the module
    /// body is the outermost async context), and `import`/`export` declarations
    /// are allowed. The result's `source_type` is always [`SourceType::Module`].
    ///
    /// Used by the module loader (`Parser::parse_program` infers the goal from
    /// the presence of `import`/`export`, but cannot enable top-level `await`
    /// before it has parsed — a module known to be a module up front does).
    ///
    /// # Errors
    /// Returns a parse-phase `SyntaxError` on malformed input.
    pub fn parse_module(source: &'src str) -> Result<Program> {
        let mut p = Parser::with_goal(source, true)?;
        // The module body is the outermost async context, so top-level `await`
        // is an operator (not an identifier).
        p.in_async = true;
        // The top statement list is the only `ModuleItem` position where `import`
        // / `export` are legal.
        p.module_top_level = true;
        let body = p.parse_statement_list(TokenKind::Eof)?;
        p.expect(TokenKind::Eof)?;
        let program = Program {
            body,
            source_type: SourceType::Module,
            span: Span::new(0, source.len() as u32),
        };
        super::validate::validate_program(&program)?;
        Ok(program)
    }

    /// Parses statements until `terminator` (or `Eof`). Items in a statement
    /// list are `StatementListItem`s, so declarations (`let`/`const`/`class`/…)
    /// are permitted here.
    fn parse_statement_list(&mut self, terminator: TokenKind) -> Result<Vec<Stmt>> {
        // A brace-delimited list (block, function body, `switch` body) is a nested
        // context, never the module top level, so `import` / `export` are illegal
        // inside it. The `Eof`-terminated list is the program/module top level and
        // keeps the flag as the caller set it.
        let saved = self.module_top_level;
        if terminator == TokenKind::RBrace {
            self.module_top_level = false;
        }
        let mut body = Vec::new();
        while !self.at(terminator) && !self.at(TokenKind::Eof) {
            body.push(self.parse_statement_item()?);
        }
        self.module_top_level = saved;
        Ok(body)
    }

    /// Parses a `StatementListItem` — a statement *or* a declaration. Used at
    /// every position where the grammar admits a declaration (program body,
    /// block, function body, `switch` case clauses, …).
    pub(crate) fn parse_statement_item(&mut self) -> Result<Stmt> {
        let guard = self.enter_recursion()?;
        guard.parser.parse_statement_inner(true)
    }

    /// Parses a single `Statement` in *single-statement position* — the body of
    /// an `if`/`else`, a loop, a `with`, or a labeled statement. A declaration
    /// is **not** a `Statement`, so at such a position a leading `let` is the
    /// identifier `let` (an `ExpressionStatement`), not a `LexicalDeclaration`.
    pub(crate) fn parse_statement(&mut self) -> Result<Stmt> {
        let guard = self.enter_recursion()?;
        // A single-statement position (control-flow / labeled body) is never the
        // module top level, so `import` / `export` are not legal here.
        let saved = guard.parser.module_top_level;
        guard.parser.module_top_level = false;
        let r = guard.parser.parse_statement_inner(false);
        guard.parser.module_top_level = saved;
        r
    }

    /// The body of the statement dispatchers, run inside the recursion guard.
    /// `decl_ok` is true at `StatementListItem` positions (declarations allowed)
    /// and false in single-statement position.
    fn parse_statement_inner(&mut self, decl_ok: bool) -> Result<Stmt> {
        // A leading `let` only introduces a `LexicalDeclaration` at a
        // `StatementListItem` position. In single-statement position a
        // declaration is not a `Statement`, so `let` is the ordinary identifier
        // `let` and the construct is an `ExpressionStatement` (e.g.
        // `if (x) let\nx = 1;` is `let; x = 1;` via ASI).
        //
        // The one exception is the `ExpressionStatement` lookahead restriction
        // `[lookahead ∉ { … let [ }]`: a `let` immediately followed by `[` can
        // begin neither an `ExpressionStatement` nor (here) a declaration, so it
        // is a `SyntaxError`. The restriction is on the token pair `let` `[`; an
        // intervening line terminator does not lift it.
        if !decl_ok && self.at(TokenKind::Keyword(Kw::Let)) {
            if self.nth_kind(1) == TokenKind::LBracket {
                return Err(
                    self.err("`let [` may not begin a statement in single-statement position")
                );
            }
            return self.parse_expression_statement();
        }
        match self.peek() {
            TokenKind::LBrace => self.parse_block(),
            TokenKind::Semicolon => {
                let span = self.bump().span;
                Ok(Stmt::Empty { span })
            }
            TokenKind::Keyword(Kw::Var | Kw::Let | Kw::Const) => self.parse_var_statement(),
            // `using x = …` / `await using x = …` — explicit-resource-management
            // declarations (only at a `StatementListItem` position). `using` and
            // `await` are otherwise ordinary identifiers, so this is gated on a
            // tight lookahead (see `at_using_decl` / `at_await_using_decl`).
            TokenKind::Identifier if decl_ok && self.at_using_decl() => {
                self.parse_using_statement(false)
            }
            TokenKind::Keyword(Kw::Await) if decl_ok && self.at_await_using_decl() => {
                self.parse_using_statement(true)
            }
            TokenKind::Keyword(Kw::If) => self.parse_if(),
            TokenKind::Keyword(Kw::For) => self.parse_for(),
            TokenKind::Keyword(Kw::While) => self.parse_while(),
            TokenKind::Keyword(Kw::Do) => self.parse_do_while(),
            TokenKind::Keyword(Kw::Switch) => self.parse_switch(),
            TokenKind::Keyword(Kw::Try) => self.parse_try(),
            TokenKind::Keyword(Kw::Return) => self.parse_return(),
            TokenKind::Keyword(Kw::Break) => self.parse_break_continue(true),
            TokenKind::Keyword(Kw::Continue) => self.parse_break_continue(false),
            TokenKind::Keyword(Kw::Throw) => self.parse_throw(),
            TokenKind::Keyword(Kw::Debugger) => {
                let span = self.bump().span;
                self.semicolon()?;
                Ok(Stmt::Debugger { span })
            }
            TokenKind::Keyword(Kw::With) => self.parse_with(),
            TokenKind::Keyword(Kw::Function) => self.parse_function_declaration(),
            TokenKind::Keyword(Kw::Async)
                if self.nth_kind(1) == TokenKind::Keyword(Kw::Function) && !self.nth_newline(1) =>
            {
                self.parse_function_declaration()
            }
            TokenKind::Keyword(Kw::Class) => self.parse_class_declaration(),
            // `@dec … class C { … }` — a decorated class declaration. The
            // decorators are parsed and discarded (applied as no-ops).
            TokenKind::At => {
                self.parse_decorators()?;
                self.parse_class_declaration()
            }
            // `import(` / `import.` are expression forms (dynamic import /
            // import.meta), handled as expressions, not import declarations.
            TokenKind::Keyword(Kw::Import)
                if !matches!(self.nth_kind(1), TokenKind::LParen | TokenKind::Dot) =>
            {
                self.parse_import()
            }
            TokenKind::Keyword(Kw::Export) => self.parse_export(),
            // Labeled statement: `ident :`.
            TokenKind::Identifier if self.nth_kind(1) == TokenKind::Colon => self.parse_labeled(),
            TokenKind::Keyword(kw)
                if kw.is_contextual() && self.nth_kind(1) == TokenKind::Colon =>
            {
                self.parse_labeled()
            }
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_block(&mut self) -> Result<Stmt> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let body = self.parse_statement_list(TokenKind::RBrace)?;
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Stmt::Block {
            body,
            span: start.to(end),
        })
    }

    /// Parses a brace-delimited block and returns just its statement list (for
    /// `try`/`catch`/`finally` and function bodies).
    pub(super) fn parse_block_body(&mut self) -> Result<Vec<Stmt>> {
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_statement_list(TokenKind::RBrace)?;
        self.expect(TokenKind::RBrace)?;
        Ok(body)
    }

    fn parse_expression_statement(&mut self) -> Result<Stmt> {
        let expression = self.parse_expression()?;
        let start = expression.span();
        self.semicolon()?;
        Ok(Stmt::Expr {
            expression: Box::new(expression),
            span: start.to(self.prev_span()),
        })
    }

    fn parse_labeled(&mut self) -> Result<Stmt> {
        let tok = self.bump();
        let name: Box<str> = match tok.kind {
            TokenKind::Identifier => self.checked_ident_name(tok)?.into(),
            TokenKind::Keyword(kw) => kw.as_str().into(),
            _ => unreachable!("labeled dispatch guaranteed an identifier"),
        };
        let label = Ident::new(name, tok.span);
        self.expect(TokenKind::Colon)?;
        let body = self.parse_statement()?;
        let span = tok.span.to(body.span());
        Ok(Stmt::Labeled {
            label,
            body: Box::new(body),
            span,
        })
    }

    // --- declarations ---------------------------------------------------

    fn parse_var_statement(&mut self) -> Result<Stmt> {
        let kw = self.bump();
        let kind = var_kind(kw.kind).expect("dispatched on a declaration keyword");
        let first = self.parse_declarator()?;
        let decl = self.parse_declarator_tail(kind, kw.span, first)?;
        self.semicolon()?;
        Ok(Stmt::Var(decl))
    }

    /// Whether the cursor begins a `using` declaration: the contextual keyword
    /// `using`, then — with no intervening line terminator (a restricted
    /// production) — a `BindingIdentifier`. The follower must be a binding name
    /// and not itself `using`; a bare `using` (e.g. `using;`, `using = 1`,
    /// `using\n x`) remains an ordinary identifier expression.
    fn at_using_decl(&self) -> bool {
        self.peek_tok().text(self.source) == "using"
            && !self.nth_newline(1)
            && self.nth_is_binding_ident(1)
    }

    /// Whether the cursor begins an `await using` declaration: `await`, then
    /// `using` (no line terminator between `using` and the binding name), then a
    /// `BindingIdentifier`.
    fn at_await_using_decl(&self) -> bool {
        self.nth_kind(1) == TokenKind::Identifier
            && self
                .tokens
                .get(self.pos + 1)
                .is_some_and(|t| t.text(self.source) == "using")
            && !self.nth_newline(2)
            && self.nth_is_binding_ident(2)
    }

    /// Whether the cursor begins a `using` declaration **in a `for` head**. This
    /// is the same as [`at_using_decl`], except a binding named `of` is treated
    /// specially.
    ///
    /// In the `for-of`/`for-await-of` head, the explicit-resource-management
    /// grammar excludes a `using ForBinding` whose name is `of`, so
    /// `for (using of <iterable>)` is interpreted as the identifier `using` being
    /// the for-of left-hand side (the following `of` being the `for-of` keyword);
    /// likewise `for (using of of x)` reads as `using` (LHS) `of` (keyword)
    /// `of[/x]` (iterable). That exclusion only applies to the `for-of` form.
    ///
    /// In a *classic* `for( ; ; )` statement there is no such restriction: a
    /// `using` declaration may bind `of`, handled like `for (let of = null;;)`.
    /// We detect the classic case by the token following the binding `of`: a
    /// `=`/`;`/`,` can only continue a classic-for declarator, never a for-of
    /// head (whose binding would be immediately followed by the `of` keyword).
    fn at_for_using_decl(&self) -> bool {
        if !self.at_using_decl() {
            return false;
        }
        if self.nth_is_named(1, "of") {
            // `using of` — a declaration binding `of` only in the classic-for
            // form (`using of = … ;` / `using of ; …` / `using of , …`).
            return matches!(
                self.nth_kind(2),
                TokenKind::Eq | TokenKind::Semicolon | TokenKind::Comma
            );
        }
        true
    }

    /// Whether the cursor begins an `await using` declaration in a `for` head.
    ///
    /// Unlike plain `using` (see [`at_for_using_decl`]), the binding name *may*
    /// be `of`: `for (await using of of x)` is interpreted as an `await using`
    /// declaration binding `of`, iterating over `x` (the second `of` is the
    /// `for-of` keyword). The `of`-exclusion exists for plain `using` only
    /// because `for (using of …)` is otherwise ambiguous with `using` being an
    /// ordinary LHS identifier — `await using` has no such ambiguity, since
    /// `await using` can only begin a declaration here.
    fn at_for_await_using_decl(&self) -> bool {
        self.at_await_using_decl()
    }

    /// Whether the token `n` ahead is an identifier (or contextual keyword)
    /// spelled exactly `name`.
    fn nth_is_named(&self, n: usize, name: &str) -> bool {
        self.tokens
            .get(self.pos + n)
            .is_some_and(|t| t.text(self.source) == name)
    }

    /// Whether the token `n` ahead is a `BindingIdentifier` that may follow
    /// `using` — an identifier (but not `using` itself) or `yield`/`await` used
    /// as a name. Patterns (`[`/`{`) are not permitted after `using`.
    fn nth_is_binding_ident(&self, n: usize) -> bool {
        match self.nth_kind(n) {
            TokenKind::Identifier => self
                .tokens
                .get(self.pos + n)
                .is_some_and(|t| t.text(self.source) != "using"),
            TokenKind::Keyword(kw) => self.keyword_is_binding_ident(kw),
            _ => false,
        }
    }

    /// Parses a `using` / `await using` declaration (the cursor is at `using`
    /// or `await`). Each binding must be a plain identifier with an initializer;
    /// destructuring patterns are not permitted.
    fn parse_using_statement(&mut self, is_await: bool) -> Result<Stmt> {
        let start = self.cur_span();
        if is_await {
            self.bump(); // `await`
        }
        self.bump(); // `using`
        let kind = if is_await {
            VarDeclKind::AwaitUsing
        } else {
            VarDeclKind::Using
        };
        let mut declarations = Vec::new();
        loop {
            let d_start = self.cur_span();
            let name = self.parse_binding_ident()?;
            // A `using` / `await using` declaration (outside a `for-of` head)
            // requires an initializer, like `const`.
            if !self.eat(TokenKind::Eq) {
                return Err(self.err_at(
                    d_start.to(self.prev_span()),
                    "a `using` declaration must be initialized",
                ));
            }
            let init = self.parse_assignment()?;
            declarations.push(VarDeclarator {
                target: BindingTarget::Ident(name),
                init: Some(init),
                span: d_start.to(self.prev_span()),
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.semicolon()?;
        Ok(Stmt::Var(VarDecl {
            kind,
            declarations,
            span: start.to(self.prev_span()),
        }))
    }

    /// Parses one `target (= init)?` declarator.
    fn parse_declarator(&mut self) -> Result<VarDeclarator> {
        let start = self.cur_span();
        let target = self.parse_binding_target()?;
        let init = if self.eat(TokenKind::Eq) {
            Some(self.parse_assignment()?)
        } else {
            None
        };
        Ok(VarDeclarator {
            target,
            init,
            span: start.to(self.prev_span()),
        })
    }

    /// Given an already-parsed first declarator, parses any comma-separated
    /// rest and assembles the [`VarDecl`], enforcing that `const` bindings are
    /// initialized.
    fn parse_declarator_tail(
        &mut self,
        kind: VarDeclKind,
        start: Span,
        first: VarDeclarator,
    ) -> Result<VarDecl> {
        let mut declarations = alloc::vec![first];
        while self.eat(TokenKind::Comma) {
            declarations.push(self.parse_declarator()?);
        }
        if kind == VarDeclKind::Const {
            for d in &declarations {
                if d.init.is_none() {
                    return Err(self.err_at(d.span, "`const` declaration must be initialized"));
                }
            }
        }
        Ok(VarDecl {
            kind,
            declarations,
            span: start.to(self.prev_span()),
        })
    }

    /// A binding target: an identifier or an array/object destructuring
    /// pattern.
    pub(super) fn parse_binding_target(&mut self) -> Result<BindingTarget> {
        let guard = self.enter_recursion()?;
        guard.parser.parse_binding_target_inner()
    }

    /// The body of [`Self::parse_binding_target`], run inside the recursion
    /// guard.
    fn parse_binding_target_inner(&mut self) -> Result<BindingTarget> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Identifier => {
                self.bump();
                Ok(BindingTarget::Ident(Ident::new(
                    self.checked_ident_name(tok)?,
                    tok.span,
                )))
            }
            TokenKind::Keyword(kw) if self.keyword_is_binding_ident(kw) => {
                self.bump();
                Ok(BindingTarget::Ident(Ident::new(kw.as_str(), tok.span)))
            }
            TokenKind::LBracket => self.parse_array_pattern(),
            TokenKind::LBrace => self.parse_object_pattern(),
            _ => Err(self.err(format!("expected a binding name, found {:?}", tok.kind))),
        }
    }

    /// An array destructuring pattern: `[a, , b = 1, ...rest]`.
    fn parse_array_pattern(&mut self) -> Result<BindingTarget> {
        let start = self.expect(TokenKind::LBracket)?.span;
        let mut elements = Vec::new();
        while !self.at(TokenKind::RBracket) {
            if self.at(TokenKind::Comma) {
                self.bump();
                elements.push(ArrayPatternElement::Hole);
                continue;
            }
            if self.at(TokenKind::DotDotDot) {
                let rest_start = self.bump().span;
                let target = self.parse_binding_target()?;
                elements.push(ArrayPatternElement::Rest {
                    span: rest_start.to(self.prev_span()),
                    target,
                });
                break; // a rest element must be last
            }
            let el_start = self.cur_span();
            let target = self.parse_binding_target()?;
            let default = self.parse_optional_default()?;
            elements.push(ArrayPatternElement::Item {
                target,
                default,
                span: el_start.to(self.prev_span()),
            });
            if !self.at(TokenKind::RBracket) {
                self.expect(TokenKind::Comma)?;
            }
        }
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(BindingTarget::Array(ArrayPattern {
            elements,
            span: start.to(end),
        }))
    }

    /// An object destructuring pattern: `{ a, b: c, d = 1, [k]: e, ...rest }`.
    fn parse_object_pattern(&mut self) -> Result<BindingTarget> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let mut properties = Vec::new();
        let mut rest = None;
        while !self.at(TokenKind::RBrace) {
            if self.at(TokenKind::DotDotDot) {
                self.bump();
                rest = Some(Box::new(self.parse_binding_target()?));
                break; // a rest element must be last
            }
            properties.push(self.parse_object_pattern_prop()?);
            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(BindingTarget::Object(ObjectPattern {
            properties,
            rest,
            span: start.to(end),
        }))
    }

    fn parse_object_pattern_prop(&mut self) -> Result<ObjectPatternProp> {
        let start = self.cur_span();

        // Computed key `[expr]: target`.
        if self.at(TokenKind::LBracket) {
            self.bump();
            let key_expr = self.without_no_in(Self::parse_assignment)?;
            self.expect(TokenKind::RBracket)?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_binding_target()?;
            let default = self.parse_optional_default()?;
            return Ok(ObjectPatternProp {
                key: PropertyKey::Computed(Box::new(key_expr)),
                value,
                default,
                shorthand: false,
                span: start.to(self.prev_span()),
            });
        }

        let tok = self.peek_tok();
        // String/number literal key — always `key: target`.
        if matches!(tok.kind, TokenKind::String | TokenKind::Number) {
            self.bump();
            let key = if tok.kind == TokenKind::String {
                PropertyKey::Str(cook::string_key(tok.text(self.source), tok.span)?.into())
            } else {
                PropertyKey::Number(cook::number(tok.text(self.source)))
            };
            self.expect(TokenKind::Colon)?;
            let value = self.parse_binding_target()?;
            let default = self.parse_optional_default()?;
            return Ok(ObjectPatternProp {
                key,
                value,
                default,
                shorthand: false,
                span: start.to(self.prev_span()),
            });
        }

        // Identifier-name key: `name`, `name: target`, with optional default.
        let (name, can_shorthand): (Box<str>, bool) = match tok.kind {
            TokenKind::Identifier => (self.ident_name(tok).into(), true),
            TokenKind::Keyword(kw) if self.keyword_is_binding_ident(kw) => {
                (kw.as_str().into(), true)
            }
            TokenKind::Keyword(kw) => (kw.as_str().into(), false),
            _ => return Err(self.err(format!("expected a property key, found {:?}", tok.kind))),
        };
        self.bump();

        if self.eat(TokenKind::Colon) {
            let value = self.parse_binding_target()?;
            let default = self.parse_optional_default()?;
            return Ok(ObjectPatternProp {
                key: PropertyKey::Ident(name),
                value,
                default,
                shorthand: false,
                span: start.to(self.prev_span()),
            });
        }

        if !can_shorthand {
            return Err(self.err_at(tok.span, "reserved word cannot be a shorthand binding"));
        }
        // A shorthand binding is a `BindingIdentifier`; an escaped reserved word
        // is a Syntax Error here (unlike as a key).
        self.checked_ident_name(tok)?;
        let value = BindingTarget::Ident(Ident::new(name.clone(), tok.span));
        let default = self.parse_optional_default()?;
        Ok(ObjectPatternProp {
            key: PropertyKey::Ident(name),
            value,
            default,
            shorthand: true,
            span: start.to(self.prev_span()),
        })
    }

    /// Parses an optional `= default` initializer used in patterns.
    fn parse_optional_default(&mut self) -> Result<Option<Expr>> {
        if self.eat(TokenKind::Eq) {
            Ok(Some(self.parse_assignment()?))
        } else {
            Ok(None)
        }
    }

    // --- control flow ---------------------------------------------------

    fn parse_if(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `if`
        self.expect(TokenKind::LParen)?;
        let test = self.without_no_in(Self::parse_expression)?;
        self.expect(TokenKind::RParen)?;
        let consequent = self.parse_statement()?;
        let alternate = if self.eat(TokenKind::Keyword(Kw::Else)) {
            Some(Box::new(self.parse_statement()?))
        } else {
            None
        };
        let end = alternate
            .as_ref()
            .map_or_else(|| consequent.span(), |a| a.span());
        Ok(Stmt::If {
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate,
            span: start.to(end),
        })
    }

    fn parse_while(&mut self) -> Result<Stmt> {
        let start = self.bump().span;
        self.expect(TokenKind::LParen)?;
        let test = self.without_no_in(Self::parse_expression)?;
        self.expect(TokenKind::RParen)?;
        let body = self.parse_statement()?;
        let span = start.to(body.span());
        Ok(Stmt::While {
            test: Box::new(test),
            body: Box::new(body),
            span,
        })
    }

    fn parse_do_while(&mut self) -> Result<Stmt> {
        let start = self.bump().span;
        let body = self.parse_statement()?;
        self.expect(TokenKind::Keyword(Kw::While))?;
        self.expect(TokenKind::LParen)?;
        let test = self.without_no_in(Self::parse_expression)?;
        let end = self.expect(TokenKind::RParen)?.span;
        // A `do…while` permits (and ignores) a trailing semicolon regardless of
        // ASI.
        self.eat(TokenKind::Semicolon);
        Ok(Stmt::DoWhile {
            body: Box::new(body),
            test: Box::new(test),
            span: start.to(end),
        })
    }

    fn parse_for(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `for`
        // `for await (… of …)` — async iteration.
        let is_await = self.eat(TokenKind::Keyword(Kw::Await));
        self.expect(TokenKind::LParen)?;
        // The header is parsed with the `in`-as-operator restriction in force.
        let head = self.with_no_in(|p| p.parse_for_head(is_await))?;
        match head {
            ForHead::Empty => self.finish_for_classic(start, None),
            ForHead::Classic(init) => self.finish_for_classic(start, init),
            ForHead::InOf { left, is_of } => self.finish_for_in_of(start, left, is_of, is_await),
        }
    }

    fn parse_for_head(&mut self, is_await: bool) -> Result<ForHead> {
        if self.eat(TokenKind::Semicolon) {
            return Ok(ForHead::Empty);
        }

        // The plain (`[~Await]`) `for-of` head forbids a left-hand side beginning
        // with the tokens `async of` (`for ( [lookahead ∉ { let, async of }]
        // LeftHandSideExpr of … )`): `for (async of …)` is a Syntax Error (use
        // `for ((async) of …)` to iterate into the variable `async`). This
        // lookahead restriction does *not* apply to the `for await` (`[+Await]`)
        // production, whose head only excludes a leading `let`; there `async` is a
        // valid LHS identifier (`for await (async of x)`). A bare `async` followed
        // by `of` can only be the forbidden form, so reject it eagerly — but only
        // outside `for await`.
        if !is_await && self.peek_tok().text(self.source) == "async" && self.nth_is_named(1, "of") {
            return Err(self.err("`async` may not be the left-hand side of a `for-of` loop"));
        }

        // `for (using x …)` / `for (await using x …)` — explicit-resource
        // management bindings in a `for` head (classic or `for-of`). The
        // for-head variants exclude a binding named `of` so that
        // `for (using of …)` parses `using` as an ordinary LHS identifier.
        if self.at_for_using_decl() || self.at_for_await_using_decl() {
            return self.parse_for_using_head();
        }

        // `let` heads a lexical declaration in a `for` header only when followed
        // by a binding (`let x`, `let [`, `let {`); otherwise (sloppy code) it is
        // an ordinary `LeftHandSideExpression` identifier — `for (let in obj)`,
        // `for (let.x of y)`. (`var`/`const` always require a binding.) Strict-mode
        // misuse of bare `let` is caught later by the identifier-reference
        // validator. `let [` is always a declaration per the grammar lookahead.
        let kind = var_kind(self.peek()).filter(|k| {
            *k != VarDeclKind::Let
                || matches!(self.nth_kind(1), TokenKind::LBracket | TokenKind::LBrace)
                || self.nth_is_binding_ident(1)
        });
        if let Some(kind) = kind {
            let kw = self.bump();
            let target = self.parse_binding_target()?;
            if self.eat(TokenKind::Keyword(Kw::In)) {
                return Ok(self.decl_for_left(kind, target, kw.span, false));
            }
            if self.eat(TokenKind::Keyword(Kw::Of)) {
                return Ok(self.decl_for_left(kind, target, kw.span, true));
            }
            // Classic loop with a declaration initializer.
            let init = if self.eat(TokenKind::Eq) {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            let first = VarDeclarator {
                target,
                init,
                span: kw.span.to(self.prev_span()),
            };
            let decl = self.parse_declarator_tail(kind, kw.span, first)?;
            self.expect(TokenKind::Semicolon)?;
            return Ok(ForHead::Classic(Some(ForInit::Var(decl))));
        }

        // Expression initializer / iteration target.
        let expr = self.parse_expression()?;
        if self.eat(TokenKind::Keyword(Kw::In)) {
            return self.expr_for_left(expr, false);
        }
        if self.eat(TokenKind::Keyword(Kw::Of)) {
            return self.expr_for_left(expr, true);
        }
        self.expect(TokenKind::Semicolon)?;
        Ok(ForHead::Classic(Some(ForInit::Expr(Box::new(expr)))))
    }

    /// Parses a `using` / `await using` binding list in a `for` head (the
    /// cursor is at `using` or `await`). Supports the `for-of` form
    /// (`for (using x of it)`) and the classic form (`for (using x = e; …; …)`);
    /// `for-in` is not permitted with `using`.
    fn parse_for_using_head(&mut self) -> Result<ForHead> {
        let start = self.cur_span();
        let is_await = self.at_await_using_decl();
        if is_await {
            self.bump(); // `await`
        }
        self.bump(); // `using`
        let kind = if is_await {
            VarDeclKind::AwaitUsing
        } else {
            VarDeclKind::Using
        };
        let first_name = self.parse_binding_ident()?;
        // `for (using x of iterable)`.
        if self.eat(TokenKind::Keyword(Kw::Of)) {
            return Ok(ForHead::InOf {
                left: ForLeft::Decl {
                    kind,
                    target: BindingTarget::Ident(first_name),
                    span: start.to(self.prev_span()),
                },
                is_of: true,
            });
        }
        // Classic loop: `using x = e [, y = e2 …] ;`.
        let init = if self.eat(TokenKind::Eq) {
            Some(self.parse_assignment()?)
        } else {
            None
        };
        let mut declarations = alloc::vec![VarDeclarator {
            target: BindingTarget::Ident(first_name),
            init,
            span: start.to(self.prev_span()),
        }];
        while self.eat(TokenKind::Comma) {
            let d_start = self.cur_span();
            let name = self.parse_binding_ident()?;
            let d_init = if self.eat(TokenKind::Eq) {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            declarations.push(VarDeclarator {
                target: BindingTarget::Ident(name),
                init: d_init,
                span: d_start.to(self.prev_span()),
            });
        }
        self.expect(TokenKind::Semicolon)?;
        Ok(ForHead::Classic(Some(ForInit::Var(VarDecl {
            kind,
            declarations,
            span: start.to(self.prev_span()),
        }))))
    }

    fn decl_for_left(
        &self,
        kind: VarDeclKind,
        target: BindingTarget,
        kw_span: Span,
        is_of: bool,
    ) -> ForHead {
        ForHead::InOf {
            left: ForLeft::Decl {
                kind,
                target,
                span: kw_span.to(self.prev_span()),
            },
            is_of,
        }
    }

    fn expr_for_left(&self, expr: Expr, is_of: bool) -> Result<ForHead> {
        if !expr.is_assignment_target() {
            return Err(self.err_at(
                expr.span(),
                "invalid left-hand side in for-in/of (not an assignment target)",
            ));
        }
        Ok(ForHead::InOf {
            left: ForLeft::Target(Box::new(expr)),
            is_of,
        })
    }

    /// Parses the remainder of a classic `for` (the first `;` already
    /// consumed): `test? ; update? ) body`.
    fn finish_for_classic(&mut self, start: Span, init: Option<ForInit>) -> Result<Stmt> {
        let test = if self.at(TokenKind::Semicolon) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        self.expect(TokenKind::Semicolon)?;
        let update = if self.at(TokenKind::RParen) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        self.expect(TokenKind::RParen)?;
        let body = self.parse_statement()?;
        let span = start.to(body.span());
        Ok(Stmt::For {
            init,
            test,
            update,
            body: Box::new(body),
            span,
        })
    }

    /// Parses the remainder of a `for-in`/`for-of`: `right ) body`.
    fn finish_for_in_of(
        &mut self,
        start: Span,
        left: ForLeft,
        is_of: bool,
        is_await: bool,
    ) -> Result<Stmt> {
        // `for-of` iterates an AssignmentExpression; `for-in` an Expression.
        let right = if is_of {
            self.parse_assignment()?
        } else {
            self.parse_expression()?
        };
        self.expect(TokenKind::RParen)?;
        let body = self.parse_statement()?;
        let span = start.to(body.span());
        let right = Box::new(right);
        let body = Box::new(body);
        Ok(if is_of {
            Stmt::ForOf {
                left,
                right,
                body,
                is_await,
                span,
            }
        } else {
            Stmt::ForIn {
                left,
                right,
                body,
                span,
            }
        })
    }

    fn parse_switch(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `switch`
        self.expect(TokenKind::LParen)?;
        let discriminant = self.without_no_in(Self::parse_expression)?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::LBrace)?;
        // A `switch` body is a nested context: `import`/`export` are illegal in a
        // case clause.
        let saved_top = self.module_top_level;
        self.module_top_level = false;
        let mut cases = Vec::new();
        let mut seen_default = false;
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let case_start = self.cur_span();
            let test = if self.eat(TokenKind::Keyword(Kw::Case)) {
                Some(self.parse_expression()?)
            } else if self.eat(TokenKind::Keyword(Kw::Default)) {
                if seen_default {
                    return Err(self.err_at(case_start, "multiple `default` clauses in switch"));
                }
                seen_default = true;
                None
            } else {
                return Err(self.err("expected `case` or `default`"));
            };
            self.expect(TokenKind::Colon)?;
            let body = self.parse_case_body()?;
            cases.push(SwitchCase {
                test,
                body,
                span: case_start.to(self.prev_span()),
            });
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        self.module_top_level = saved_top;
        Ok(Stmt::Switch {
            discriminant: Box::new(discriminant),
            cases,
            span: start.to(end),
        })
    }

    /// Statements of a `case`/`default` clause, up to the next clause or `}`.
    fn parse_case_body(&mut self) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        while !matches!(
            self.peek(),
            TokenKind::Keyword(Kw::Case | Kw::Default) | TokenKind::RBrace | TokenKind::Eof
        ) {
            body.push(self.parse_statement_item()?);
        }
        Ok(body)
    }

    fn parse_try(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `try`
        let block = self.parse_block_body()?;
        let handler = if self.at(TokenKind::Keyword(Kw::Catch)) {
            Some(self.parse_catch()?)
        } else {
            None
        };
        let finalizer = if self.eat(TokenKind::Keyword(Kw::Finally)) {
            Some(self.parse_block_body()?)
        } else {
            None
        };
        if handler.is_none() && finalizer.is_none() {
            return Err(self.err("`try` must be followed by `catch` and/or `finally`"));
        }
        Ok(Stmt::Try {
            block,
            handler,
            finalizer,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_catch(&mut self) -> Result<CatchClause> {
        let start = self.bump().span; // `catch`
        let param = if self.eat(TokenKind::LParen) {
            let target = self.parse_binding_target()?;
            self.expect(TokenKind::RParen)?;
            Some(target)
        } else {
            None
        };
        let body = self.parse_block_body()?;
        Ok(CatchClause {
            param,
            body,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_with(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `with`
        self.expect(TokenKind::LParen)?;
        let object = self.without_no_in(Self::parse_expression)?;
        self.expect(TokenKind::RParen)?;
        let body = self.parse_statement()?;
        let span = start.to(body.span());
        Ok(Stmt::With {
            object: Box::new(object),
            body: Box::new(body),
            span,
        })
    }

    // --- jumps (restricted productions) ---------------------------------

    fn parse_return(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `return`
        let argument = if self.at_statement_end() {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        self.semicolon()?;
        Ok(Stmt::Return {
            argument,
            span: start.to(self.prev_span()),
        })
    }

    fn parse_throw(&mut self) -> Result<Stmt> {
        let start = self.bump().span; // `throw`
        // A line terminator after `throw` is always an error (the argument is
        // mandatory and may not be ASI-separated).
        if self.peek_tok().newline_before {
            return Err(self.err("illegal newline after `throw`"));
        }
        let argument = self.parse_expression()?;
        self.semicolon()?;
        let span = start.to(self.prev_span());
        Ok(Stmt::Throw {
            argument: Box::new(argument),
            span,
        })
    }

    /// Parses `break` (`is_break = true`) or `continue`, with an optional
    /// label that, per the restricted production, may not be separated from the
    /// keyword by a line terminator.
    fn parse_break_continue(&mut self, is_break: bool) -> Result<Stmt> {
        let start = self.bump().span;
        let label = if self.peek_tok().newline_before {
            None
        } else {
            self.try_parse_label()
        };
        self.semicolon()?;
        let span = start.to(self.prev_span());
        Ok(if is_break {
            Stmt::Break { label, span }
        } else {
            Stmt::Continue { label, span }
        })
    }

    /// Consumes an identifier label if one is present at the cursor.
    fn try_parse_label(&mut self) -> Option<Ident> {
        let tok = self.peek_tok();
        let name: Box<str> = match tok.kind {
            TokenKind::Identifier => self.ident_name(tok).into(),
            TokenKind::Keyword(kw) if self.keyword_is_binding_ident(kw) => kw.as_str().into(),
            _ => return None,
        };
        self.bump();
        Some(Ident::new(name, tok.span))
    }

    // --- ASI helpers ----------------------------------------------------

    /// Whether the cursor is at a point where a statement may end without an
    /// explicit `;` — at `}`, at `Eof`, or before a token preceded by a line
    /// terminator (used by the restricted `return`/`break`/`continue`
    /// productions).
    fn at_statement_end(&self) -> bool {
        self.at(TokenKind::Semicolon)
            || self.at(TokenKind::RBrace)
            || self.at(TokenKind::Eof)
            || self.peek_tok().newline_before
    }

    /// Consumes a statement-terminating `;`, applying Automatic Semicolon
    /// Insertion: a semicolon is implied before `}`, at `Eof`, or before any
    /// token that a line terminator precedes.
    pub(super) fn semicolon(&mut self) -> Result<()> {
        if self.eat(TokenKind::Semicolon) {
            return Ok(());
        }
        if self.at(TokenKind::RBrace) || self.at(TokenKind::Eof) || self.peek_tok().newline_before {
            return Ok(());
        }
        Err(self.err(format!(
            "expected a semicolon or newline, found {:?}",
            self.peek()
        )))
    }
}

/// Maps a declaration keyword token to its [`VarDeclKind`].
fn var_kind(kind: TokenKind) -> Option<VarDeclKind> {
    match kind {
        TokenKind::Keyword(Kw::Var) => Some(VarDeclKind::Var),
        TokenKind::Keyword(Kw::Let) => Some(VarDeclKind::Let),
        TokenKind::Keyword(Kw::Const) => Some(VarDeclKind::Const),
        _ => None,
    }
}
