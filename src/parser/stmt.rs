//! Statement, declaration, and program parsing, plus Automatic Semicolon
//! Insertion. These are methods on [`Parser`](super::Parser); the expression
//! grammar lives in the parent module.

use super::Parser;
use crate::ast::{
    BindingTarget, CatchClause, Expr, ForInit, ForLeft, Ident, Program, SourceType, Stmt,
    SwitchCase, VarDecl, VarDeclKind, VarDeclarator,
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
    /// Parses a whole compilation unit as a script.
    pub fn parse_program(source: &'src str) -> Result<Program> {
        let mut p = Parser::new(source)?;
        let body = p.parse_statement_list(TokenKind::Eof)?;
        p.expect(TokenKind::Eof)?;
        Ok(Program {
            body,
            source_type: SourceType::Script,
            span: Span::new(0, source.len() as u32),
        })
    }

    /// Parses statements until `terminator` (or `Eof`).
    fn parse_statement_list(&mut self, terminator: TokenKind) -> Result<Vec<Stmt>> {
        let mut body = Vec::new();
        while !self.at(terminator) && !self.at(TokenKind::Eof) {
            body.push(self.parse_statement()?);
        }
        Ok(body)
    }

    /// Parses a single statement, dispatching on the leading token.
    pub(crate) fn parse_statement(&mut self) -> Result<Stmt> {
        match self.peek() {
            TokenKind::LBrace => self.parse_block(),
            TokenKind::Semicolon => {
                let span = self.bump().span;
                Ok(Stmt::Empty { span })
            }
            TokenKind::Keyword(Kw::Var | Kw::Let | Kw::Const) => self.parse_var_statement(),
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
            TokenKind::Keyword(Kw::Function) => {
                Err(self.err("function declarations are added in a later increment"))
            }
            TokenKind::Keyword(Kw::Class) => {
                Err(self.err("class declarations are added in a later increment"))
            }
            TokenKind::Keyword(Kw::Import | Kw::Export) => {
                Err(self.err("module import/export is added in a later increment"))
            }
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
    /// `try`/`catch`/`finally` bodies).
    fn parse_block_body(&mut self) -> Result<Vec<Stmt>> {
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
            TokenKind::Identifier => tok.text(self.source).into(),
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

    /// A binding name. Array/object destructuring patterns are deferred.
    fn parse_binding_target(&mut self) -> Result<BindingTarget> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Identifier => {
                self.bump();
                Ok(BindingTarget::Ident(Ident::new(
                    tok.text(self.source),
                    tok.span,
                )))
            }
            TokenKind::Keyword(kw) if kw.is_contextual() => {
                self.bump();
                Ok(BindingTarget::Ident(Ident::new(kw.as_str(), tok.span)))
            }
            TokenKind::LBracket | TokenKind::LBrace => {
                Err(self.err("destructuring patterns are added in a later increment"))
            }
            _ => Err(self.err(format!("expected a binding name, found {:?}", tok.kind))),
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
        self.expect(TokenKind::LParen)?;
        // The header is parsed with the `in`-as-operator restriction in force.
        let head = self.with_no_in(Self::parse_for_head)?;
        match head {
            ForHead::Empty => self.finish_for_classic(start, None),
            ForHead::Classic(init) => self.finish_for_classic(start, init),
            ForHead::InOf { left, is_of } => self.finish_for_in_of(start, left, is_of),
        }
    }

    fn parse_for_head(&mut self) -> Result<ForHead> {
        if self.eat(TokenKind::Semicolon) {
            return Ok(ForHead::Empty);
        }

        if let Some(kind) = var_kind(self.peek()) {
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
    fn finish_for_in_of(&mut self, start: Span, left: ForLeft, is_of: bool) -> Result<Stmt> {
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
            body.push(self.parse_statement()?);
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
            TokenKind::Identifier => tok.text(self.source).into(),
            TokenKind::Keyword(kw) if kw.is_contextual() => kw.as_str().into(),
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
    fn semicolon(&mut self) -> Result<()> {
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
