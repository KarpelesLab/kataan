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

    /// Parses a function declaration whose name is *optional* (for
    /// `export default function …`).
    pub(super) fn parse_default_function(&mut self) -> Result<Stmt> {
        let start = self.cur_span();
        let is_async = self.eat(TokenKind::Keyword(Kw::Async));
        let function = self.parse_function(start, is_async, false)?;
        Ok(Stmt::Function(function))
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
        // A *function expression*'s name uses the *function's own* `[Yield,
        // Await]` context, not the enclosing one: it is reserved iff this
        // function is a generator (for `yield`) / async (for `await`). So
        // `(function yield(){})` inside a generator is valid (the inner function
        // is not a generator), while `(function* yield(){})` and
        // `(async function await(){})` are Syntax Errors. A *declaration*'s name
        // uses the surrounding context, which is already in effect.
        let id = if !require_name {
            let (sg, sa) = (self.in_generator, self.in_async);
            self.in_generator = is_generator;
            self.in_async = is_async;
            let id = if self.at_binding_ident() {
                Some(self.parse_binding_ident()?)
            } else {
                None
            };
            self.in_generator = sg;
            self.in_async = sa;
            id
        } else if self.at_binding_ident() {
            Some(self.parse_binding_ident()?)
        } else {
            return Err(self.err("a function declaration requires a name"));
        };
        self.expect(TokenKind::LParen)?;
        // Parameters of a generator/async function are parsed in the function's
        // own `[?Yield, ?Await]` context: a generator's params may not bind or
        // reference `yield`, and an async function's may not use `await` (so
        // `function*(yield){}` / `function* g(x = yield){}` are Syntax Errors).
        let params = self.in_function_context(is_generator, is_async, Self::parse_params)?;
        self.expect(TokenKind::RParen)?;
        let body = self.in_function_context(is_generator, is_async, Self::parse_block_body)?;
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
                // `async of => …` — a contextual keyword (`of`, `as`, `get`, …) is a
                // valid single async-arrow parameter name.
                TokenKind::Keyword(kw) if kw.is_contextual() => {
                    self.nth_kind(2) == TokenKind::Arrow
                }
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
            // parameter name. A contextual keyword (`async of => …`) is a valid
            // single parameter, so `async` is the marker there too.
            matches!(self.nth_kind(1), TokenKind::Identifier | TokenKind::LParen)
                || matches!(self.nth_kind(1), TokenKind::Keyword(k) if k.is_contextual())
        };
        if is_async {
            self.bump();
        }

        // An async arrow's parameters are in a `[+Await]` context (so `await` may
        // not be a parameter name); `yield` follows the enclosing context.
        let saved_async = self.in_async;
        self.in_async = is_async || self.in_async;
        let params_result = (|| -> Result<Vec<Param>> {
            if self.at(TokenKind::LParen) {
                self.bump();
                let params = self.parse_params()?;
                self.expect(TokenKind::RParen)?;
                Ok(params)
            } else {
                let pstart = self.cur_span();
                let target = self.parse_binding_target()?;
                Ok(alloc::vec![Param {
                    target,
                    default: None,
                    rest: false,
                    span: pstart.to(self.prev_span()),
                }])
            }
        })();
        self.in_async = saved_async;
        let params = params_result?;

        // `ArrowFunction : ArrowParameters [no LineTerminator here] =>` — a line
        // terminator between the parameters and `=>` is a Syntax Error (ASI would
        // otherwise turn `x \n => y` into an incomplete statement).
        if self.at(TokenKind::Arrow) && self.nth_newline(0) {
            return Err(self.err("no line terminator is allowed before `=>`"));
        }
        self.expect(TokenKind::Arrow)?;
        // An arrow is never a generator. Only an *async* arrow's `ConciseBody` is
        // in a `[+Await]` context; a plain arrow's body is `ConciseBody[?In]` with
        // no `[Await]` (per the grammar), so `await` is an ordinary identifier
        // there even inside an enclosing async function or class static block
        // (`static { (() => { var await; }); }` is valid).
        let body_async = is_async;
        let body = self.in_function_context(false, body_async, |p| {
            if p.at(TokenKind::LBrace) {
                Ok(ArrowBody::Block(p.parse_block_body()?))
            } else {
                Ok(ArrowBody::Expr(Box::new(p.parse_assignment()?)))
            }
        })?;
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
    /// keyword that is usable as a name in the current context).
    pub(super) fn at_binding_ident(&self) -> bool {
        match self.peek() {
            TokenKind::Identifier => true,
            TokenKind::Keyword(kw) => self.keyword_is_binding_ident(kw),
            _ => false,
        }
    }

    /// Whether a keyword may be used as a `BindingIdentifier` here. The
    /// contextual keywords (`async`, `of`, `get`, …) always may. `yield` may
    /// outside a generator body and `await` outside an async body — both are
    /// then ordinary identifiers (in sloppy code; strict-mode use is rejected by
    /// the post-parse validator, which is where strict mode is known).
    pub(super) fn keyword_is_binding_ident(&self, kw: Kw) -> bool {
        kw.is_contextual()
            || (kw == Kw::Yield && !self.in_generator)
            || (kw == Kw::Await && !self.in_async)
            // Strict-mode future-reserved words are ordinary identifiers in
            // sloppy code; the post-parse validator rejects them in strict mode.
            || matches!(
                kw,
                Kw::Let
                    | Kw::Static
                    | Kw::Implements
                    | Kw::Interface
                    | Kw::Package
                    | Kw::Private
                    | Kw::Protected
                    | Kw::Public
            )
    }

    /// Parses a binding identifier into an [`Ident`].
    pub(super) fn parse_binding_ident(&mut self) -> Result<Ident> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Identifier => {
                self.bump();
                Ok(Ident::new(self.checked_ident_name(tok)?, tok.span))
            }
            TokenKind::Keyword(kw) if self.keyword_is_binding_ident(kw) => {
                self.bump();
                Ok(Ident::new(kw.as_str(), tok.span))
            }
            _ => Err(self.err(format!("expected an identifier, found {:?}", tok.kind))),
        }
    }
}
