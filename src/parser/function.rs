//! Parsing of functions (declarations and expressions) and arrow functions,
//! including the arrow cover-grammar lookahead. Methods on [`Parser`](super::Parser).

use super::Parser;
use crate::ast::{Arrow, ArrowBody, Expr, Function, Ident, Param, Stmt};
use crate::common::Span;
use crate::error::Result;
use crate::lexer::{Keyword as Kw, TokenKind};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

impl<'src> Parser<'src> {
    // --- functions ------------------------------------------------------

    /// Parses a function *declaration* (a statement). The cursor is at
    /// `function` or at the leading `async`.
    pub(super) fn parse_function_declaration(&mut self) -> Result<Stmt> {
        let start = self.cur_span();
        let is_async = self.eat(TokenKind::Keyword(Kw::Async));
        let function = self.parse_function(start, is_async, true)?;
        Ok(Stmt::Function(function))
    }

    /// Parses a function *expression*. `start` is the span of the `async`
    /// keyword (if any) or the `function` keyword; if `is_async`, the `async`
    /// has already been consumed and the cursor is at `function`.
    pub(super) fn parse_function_expr(&mut self, is_async: bool, start: Span) -> Result<Expr> {
        let function = self.parse_function(start, is_async, false)?;
        Ok(Expr::Function(function))
    }

    /// Shared function parser. Expects the cursor at `function`. `require_name`
    /// distinguishes declarations (name required) from expressions (optional).
    fn parse_function(
        &mut self,
        start: Span,
        is_async: bool,
        require_name: bool,
    ) -> Result<Function> {
        self.expect(TokenKind::Keyword(Kw::Function))?;
        let is_generator = self.eat(TokenKind::Star);
        let id = if self.at_binding_ident() {
            Some(self.parse_binding_ident()?)
        } else if require_name {
            return Err(self.err("a function declaration requires a name"));
        } else {
            None
        };
        self.expect(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen)?;
        let body = self.parse_block_body()?;
        Ok(Function {
            id,
            params,
            body,
            is_async,
            is_generator,
            span: start.to(self.prev_span()),
        })
    }

    /// Parses a comma-separated parameter list up to (but not consuming) the
    /// closing `)`. Supports defaults (`x = 1`) and a trailing rest (`...xs`).
    pub(super) fn parse_params(&mut self) -> Result<Vec<Param>> {
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) {
            let start = self.cur_span();
            let rest = self.eat(TokenKind::DotDotDot);
            let target = self.parse_binding_target()?;
            let default = if !rest && self.eat(TokenKind::Eq) {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            params.push(Param {
                target,
                default,
                rest,
                span: start.to(self.prev_span()),
            });
            if rest {
                // A rest parameter must be last; a trailing comma is illegal.
                break;
            }
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(params)
    }

    // --- arrows ---------------------------------------------------------

    /// Whether the cursor begins an arrow function. This is the cover-grammar
    /// lookahead: an `Identifier =>`, a parenthesized parameter list followed
    /// by `=>`, or either of those prefixed with `async` (on the same line).
    pub(super) fn at_arrow_head(&self) -> bool {
        match self.peek() {
            TokenKind::Identifier => self.nth_kind(1) == TokenKind::Arrow,
            TokenKind::LParen => self.paren_arrow_at(0),
            TokenKind::Keyword(Kw::Async) if !self.nth_newline(1) => match self.nth_kind(1) {
                TokenKind::Identifier => self.nth_kind(2) == TokenKind::Arrow,
                TokenKind::LParen => self.paren_arrow_at(1),
                // `async =>` — `async` itself is the parameter name.
                TokenKind::Arrow => true,
                _ => false,
            },
            TokenKind::Keyword(kw) if kw.is_contextual() => self.nth_kind(1) == TokenKind::Arrow,
            _ => false,
        }
    }

    /// Whether the `(` at `self.pos + offset` is matched by a `)` immediately
    /// followed by `=>`. Scans on paren depth alone, which is sufficient for a
    /// well-formed token stream (brackets and braces are independently
    /// balanced, so no stray `)` appears inside them).
    fn paren_arrow_at(&self, offset: usize) -> bool {
        let mut i = self.pos + offset;
        let mut depth = 0usize;
        loop {
            match self.tokens.get(i).map(|t| t.kind) {
                Some(TokenKind::LParen) => depth += 1,
                Some(TokenKind::RParen) => {
                    depth -= 1;
                    if depth == 0 {
                        return self.tokens.get(i + 1).map(|t| t.kind) == Some(TokenKind::Arrow);
                    }
                }
                None | Some(TokenKind::Eof) => return false,
                _ => {}
            }
            i += 1;
        }
    }

    /// Parses an arrow function (the head is known to be an arrow via
    /// [`Self::at_arrow_head`]).
    pub(super) fn parse_arrow(&mut self) -> Result<Expr> {
        let start = self.cur_span();
        let is_async = self.at(TokenKind::Keyword(Kw::Async)) && {
            // Only consume `async` as the arrow marker, not as a sole `async =>`
            // parameter name.
            matches!(self.nth_kind(1), TokenKind::Identifier | TokenKind::LParen)
        };
        if is_async {
            self.bump();
        }

        let params = if self.at(TokenKind::LParen) {
            self.bump();
            let params = self.parse_params()?;
            self.expect(TokenKind::RParen)?;
            params
        } else {
            let pstart = self.cur_span();
            let target = self.parse_binding_target()?;
            alloc::vec![Param {
                target,
                default: None,
                rest: false,
                span: pstart.to(self.prev_span()),
            }]
        };

        self.expect(TokenKind::Arrow)?;
        let body = if self.at(TokenKind::LBrace) {
            ArrowBody::Block(self.parse_block_body()?)
        } else {
            ArrowBody::Expr(Box::new(self.parse_assignment()?))
        };
        let span = start.to(self.prev_span());
        Ok(Expr::Arrow(Arrow {
            params,
            body,
            is_async,
            span,
        }))
    }

    // --- shared ---------------------------------------------------------

    /// Whether the cursor is a binding identifier (a plain identifier or a
    /// contextual keyword used as a name).
    fn at_binding_ident(&self) -> bool {
        matches!(self.peek(), TokenKind::Identifier)
            || matches!(self.peek(), TokenKind::Keyword(kw) if kw.is_contextual())
    }

    /// Parses a binding identifier into an [`Ident`].
    fn parse_binding_ident(&mut self) -> Result<Ident> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Identifier => {
                self.bump();
                Ok(Ident::new(tok.text(self.source), tok.span))
            }
            TokenKind::Keyword(kw) if kw.is_contextual() => {
                self.bump();
                Ok(Ident::new(kw.as_str(), tok.span))
            }
            _ => Err(self.err(format!("expected an identifier, found {:?}", tok.kind))),
        }
    }
}
