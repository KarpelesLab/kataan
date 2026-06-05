//! Parsing of class declarations and expressions. Methods on
//! [`Parser`](super::Parser).

use super::{Parser, cook};
use crate::ast::{
    Class, ClassField, ClassMember, ClassMethod, Expr, Function, MethodKind, PropertyKey, Stmt,
};
use crate::error::Result;
use crate::lexer::{Keyword as Kw, TokenKind};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

impl<'src> Parser<'src> {
    /// Parses a class *declaration* (the cursor is at `class`).
    pub(super) fn parse_class_declaration(&mut self) -> Result<Stmt> {
        let class = self.parse_class(true)?;
        Ok(Stmt::Class(class))
    }

    /// Parses a class *expression* (the cursor is at `class`).
    pub(super) fn parse_class_expr(&mut self) -> Result<Expr> {
        let class = self.parse_class(false)?;
        Ok(Expr::Class(class))
    }

    /// Parses a class declaration whose name is *optional* (for
    /// `export default class …`).
    pub(super) fn parse_default_class(&mut self) -> Result<Stmt> {
        let class = self.parse_class(false)?;
        Ok(Stmt::Class(class))
    }

    /// Shared class parser. `require_name` distinguishes declarations from
    /// expressions.
    fn parse_class(&mut self, require_name: bool) -> Result<Class> {
        let start = self.expect(TokenKind::Keyword(Kw::Class))?.span;

        // An optional name: present unless `extends` or `{` follows.
        let id = if self.at_binding_ident()
            && !self.at(TokenKind::Keyword(Kw::Extends))
            && !self.at(TokenKind::LBrace)
        {
            Some(self.parse_binding_ident()?)
        } else if require_name {
            return Err(self.err("a class declaration requires a name"));
        } else {
            None
        };

        let super_class = if self.eat(TokenKind::Keyword(Kw::Extends)) {
            Some(Box::new(self.parse_lhs()?))
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if self.eat(TokenKind::Semicolon) {
                continue; // a stray `;` between members is allowed
            }
            body.push(self.parse_class_member()?);
        }
        let end = self.expect(TokenKind::RBrace)?.span;

        Ok(Class {
            id,
            super_class,
            body,
            span: start.to(end),
        })
    }

    fn parse_class_member(&mut self) -> Result<ClassMember> {
        let start = self.cur_span();

        // `static` — a modifier, unless it is itself the member name or begins
        // a static initialization block.
        let is_static = if self.at(TokenKind::Keyword(Kw::Static)) {
            match self.nth_kind(1) {
                // `static { … }` — static block.
                TokenKind::LBrace => {
                    self.bump(); // `static`
                    let body = self.parse_block_body()?;
                    return Ok(ClassMember::StaticBlock {
                        body,
                        span: start.to(self.prev_span()),
                    });
                }
                // `static` used as a field/method name.
                TokenKind::LParen | TokenKind::Eq | TokenKind::Semicolon | TokenKind::RBrace => {
                    false
                }
                _ => {
                    self.bump(); // consume the `static` modifier
                    true
                }
            }
        } else {
            false
        };

        // Method modifiers: `async`, generator `*`, and `get`/`set`.
        let is_async = self.at(TokenKind::Keyword(Kw::Async))
            && !self.nth_newline(1)
            && !self.modifier_is_name(1);
        if is_async {
            self.bump();
        }
        let is_generator = self.eat(TokenKind::Star);

        let accessor = if !is_async
            && !is_generator
            && matches!(self.peek(), TokenKind::Keyword(Kw::Get | Kw::Set))
            && !self.modifier_is_name(1)
        {
            let k = if self.at(TokenKind::Keyword(Kw::Get)) {
                MethodKind::Get
            } else {
                MethodKind::Set
            };
            self.bump();
            Some(k)
        } else {
            None
        };

        let key = self.parse_class_key()?;

        // A `(` makes this a method; otherwise it is a field.
        if self.at(TokenKind::LParen) {
            let value = self.parse_method_tail(is_async, is_generator)?;
            let kind = match accessor {
                Some(k) => k,
                None if is_constructor_key(&key, is_static, is_async, is_generator) => {
                    MethodKind::Constructor
                }
                None => MethodKind::Method,
            };
            return Ok(ClassMember::Method(ClassMethod {
                key,
                kind,
                value,
                is_static,
                span: start.to(self.prev_span()),
            }));
        }

        // A field: get/set/async/* are not valid here.
        if accessor.is_some() || is_async || is_generator {
            return Err(self.err_at(start, "expected `(` after method modifier"));
        }
        let value = if self.eat(TokenKind::Eq) {
            Some(self.parse_assignment()?)
        } else {
            None
        };
        self.semicolon()?;
        Ok(ClassMember::Field(ClassField {
            key,
            value,
            is_static,
            span: start.to(self.prev_span()),
        }))
    }

    /// A class member key: a private name, a computed `[expr]`, a string/number
    /// literal, or any identifier name.
    pub(super) fn parse_class_key(&mut self) -> Result<PropertyKey> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::PrivateName => {
                self.bump();
                Ok(PropertyKey::Private(tok.text(self.source)[1..].into()))
            }
            TokenKind::LBracket => {
                self.bump();
                let expr = self.without_no_in(Self::parse_assignment)?;
                self.expect(TokenKind::RBracket)?;
                Ok(PropertyKey::Computed(Box::new(expr)))
            }
            TokenKind::String => {
                self.bump();
                Ok(PropertyKey::Str(
                    cook::string(tok.text(self.source), tok.span)?.into(),
                ))
            }
            TokenKind::Number => {
                self.bump();
                Ok(PropertyKey::Number(cook::number(tok.text(self.source))))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok(PropertyKey::Ident(tok.text(self.source).into()))
            }
            TokenKind::Keyword(kw) => {
                self.bump();
                Ok(PropertyKey::Ident(kw.as_str().into()))
            }
            _ => Err(self.err(format!(
                "expected a class member name, found {:?}",
                tok.kind
            ))),
        }
    }

    /// Whether the token `n` ahead means the preceding contextual keyword is a
    /// member *name* rather than a modifier (i.e. it is directly followed by a
    /// method `(`, a field `=`, or a member terminator).
    pub(super) fn modifier_is_name(&self, n: usize) -> bool {
        matches!(
            self.nth_kind(n),
            TokenKind::LParen | TokenKind::Eq | TokenKind::Semicolon | TokenKind::RBrace
        )
    }
}

/// Whether a method key/flags identify the `constructor`.
fn is_constructor_key(
    key: &PropertyKey,
    is_static: bool,
    is_async: bool,
    is_generator: bool,
) -> bool {
    !is_static
        && !is_async
        && !is_generator
        && matches!(key, PropertyKey::Ident(name) if &**name == "constructor")
}

impl<'src> Parser<'src> {
    /// Parses `(params) { body }` for a class method or object method, building
    /// an anonymous [`Function`] carrying the async/generator flags.
    pub(super) fn parse_method_tail(
        &mut self,
        is_async: bool,
        is_generator: bool,
    ) -> Result<Function> {
        let start = self.cur_span();
        self.expect(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(TokenKind::RParen)?;
        let body = self.in_function_context(is_generator, is_async, Self::parse_block_body)?;
        Ok(Function {
            id: None,
            params,
            body,
            is_async,
            is_generator,
            span: start.to(self.prev_span()),
        })
    }
}
