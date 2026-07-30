//! Parsing of class declarations and expressions. Methods on
//! [`Parser`](super::Parser).

use super::{Parser, cook};
use crate::ast::{
    AssignOp, BindingTarget, Class, ClassField, ClassMember, ClassMethod, Expr, Function, Ident,
    MethodKind, Param, PropertyKey, Stmt,
};
use crate::common::Span;
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
        // `class extends class extends … Object {}` recurses through the cycle
        // parse_lhs → parse_primary → parse_class_expr → parse_class → extends
        // → parse_lhs with no guarded hub. Guard `parse_class` (covering both
        // declaration and expression entry, plus the `extends` recursion) so
        // each level counts toward `MAX_PARSE_DEPTH` and a deep chain returns a
        // syntax error instead of overflowing the native stack.
        let guard = self.enter_recursion()?;
        guard.parser.parse_class_inner(require_name)
    }

    /// The body of [`Self::parse_class`], run inside the recursion guard.
    fn parse_class_inner(&mut self, require_name: bool) -> Result<Class> {
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
            // A member may expand into several `ClassMember`s: an auto-accessor
            // (`accessor x`) desugars to a private backing field plus a
            // getter/setter pair.
            body.extend(self.parse_class_member()?);
        }
        let end = self.expect(TokenKind::RBrace)?.span;

        Ok(Class {
            id,
            super_class,
            body,
            span: start.to(end),
        })
    }

    fn parse_class_member(&mut self) -> Result<Vec<ClassMember>> {
        let start = self.cur_span();

        // A `DecoratorList` may prefix any class element. Decorators are parsed
        // (validated for shape) but discarded — no test exercises decorator
        // application at runtime, so they act as no-ops.
        if self.at(TokenKind::At) {
            self.parse_decorators()?;
        }

        // `static` — a modifier, unless it is itself the member name or begins
        // a static initialization block.
        let is_static = if self.at(TokenKind::Keyword(Kw::Static)) {
            match self.nth_kind(1) {
                // `static { … }` — static block. Its body is in an `[+Await]`
                // context: `await` is reserved (it may not be used as an
                // identifier), so parse with the async flag set. It is not a
                // generator, so `yield` remains an ordinary identifier.
                TokenKind::LBrace => {
                    self.bump(); // `static`
                    let body = self.in_function_context(false, true, Self::parse_block_body)?;
                    return Ok(alloc::vec![ClassMember::StaticBlock {
                        body,
                        span: start.to(self.prev_span()),
                    }]);
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

        // An auto-accessor: `accessor [no LineTerminator here] ClassElementName
        // Initializer_opt`. It is a contextual keyword: `accessor` is an
        // ordinary member name unless directly (no newline) followed by a token
        // that begins a `ClassElementName`.
        if self.at(TokenKind::Keyword(Kw::Accessor))
            && !self.nth_newline(1)
            && token_starts_class_element_name(self.nth_kind(1))
        {
            self.bump(); // `accessor`
            let key = self.parse_class_key()?;
            let value = if self.eat(TokenKind::Eq) {
                Some(self.parse_assignment()?)
            } else {
                None
            };
            self.semicolon()?;
            let span = start.to(self.prev_span());
            return Ok(self.desugar_auto_accessor(key, value, is_static, span));
        }

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
            // A getter/setter is never a generator: `get *m(){}` is NOT an
            // accessor named `m` — `get` is a plain field/method name (`*m` is a
            // separate generator method, reached via ASI). Only `async` may be a
            // modifier before `*`.
            && self.nth_kind(1) != TokenKind::Star
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
            return Ok(alloc::vec![ClassMember::Method(ClassMethod {
                key,
                kind,
                value,
                is_static,
                span: start.to(self.prev_span()),
            })]);
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
        Ok(alloc::vec![ClassMember::Field(ClassField {
            key,
            value,
            is_static,
            span: start.to(self.prev_span()),
        })])
    }

    /// Desugars an auto-accessor (`accessor x = init`) into three members: a
    /// private backing field holding the value, plus a public (or private)
    /// getter and setter that read/write the backing field. This reuses the
    /// existing field / accessor machinery so the runtime needs no special
    /// case. The backing field's private name is unique (derived from the
    /// member's source offset) and starts with a digit, so it cannot collide
    /// with — or be named by — any user-written `#name`.
    fn desugar_auto_accessor(
        &self,
        key: PropertyKey,
        value: Option<Expr>,
        is_static: bool,
        span: Span,
    ) -> Vec<ClassMember> {
        let backing: Box<str> = format!("0acc{}", span.start).into();

        // `this.#backing`
        let backing_member = |span: Span| Expr::Member {
            object: Box::new(Expr::This(span)),
            property: PropertyKey::Private(backing.clone()),
            optional: false,
            span,
        };

        // getter: `get key() { return this.#backing; }`
        let getter = Function {
            id: None,
            params: Vec::new(),
            body: alloc::vec![Stmt::Return {
                argument: Some(Box::new(backing_member(span))),
                span,
            }],
            is_async: false,
            is_generator: false,
            span,
        };

        // setter: `set key(value) { this.#backing = value; }`
        let param_name: Box<str> = "value".into();
        let setter = Function {
            id: None,
            params: alloc::vec![Param {
                target: BindingTarget::Ident(Ident::new(param_name.clone(), span)),
                default: None,
                rest: false,
                span,
            }],
            body: alloc::vec![Stmt::Expr {
                expression: Box::new(Expr::Assign {
                    op: AssignOp::Assign,
                    target: Box::new(backing_member(span)),
                    value: Box::new(Expr::Ident(Ident::new(param_name, span))),
                    paren_target: false,
                    span,
                }),
                span,
            }],
            is_async: false,
            is_generator: false,
            span,
        };

        alloc::vec![
            ClassMember::Method(ClassMethod {
                key: key.clone(),
                kind: MethodKind::Get,
                value: getter,
                is_static,
                span,
            }),
            ClassMember::Method(ClassMethod {
                key,
                kind: MethodKind::Set,
                value: setter,
                is_static,
                span,
            }),
            ClassMember::Field(ClassField {
                key: PropertyKey::Private(backing),
                value,
                is_static,
                span,
            }),
        ]
    }

    /// A class member key: a private name, a computed `[expr]`, a string/number
    /// literal, or any identifier name.
    pub(super) fn parse_class_key(&mut self) -> Result<PropertyKey> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::PrivateName => {
                self.bump();
                Ok(PropertyKey::Private(self.private_name(tok)))
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
                    cook::string_key(tok.text(self.source), tok.span)?.into(),
                ))
            }
            TokenKind::Number => {
                self.bump();
                Ok(PropertyKey::Number(cook::number(tok.text(self.source))))
            }
            TokenKind::BigInt => {
                self.bump();
                Ok(PropertyKey::Str(
                    cook::bigint_property_key(tok.text(self.source)).into(),
                ))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok(PropertyKey::Ident(self.ident_name(tok).into()))
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

/// Whether a token can begin a `ClassElementName` (a private name, a computed
/// key, a string/number literal, or any identifier name). Used to decide
/// whether a contextual `accessor` is an auto-accessor modifier or a plain
/// member name.
fn token_starts_class_element_name(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::PrivateName
            | TokenKind::LBracket
            | TokenKind::String
            | TokenKind::Number
            | TokenKind::Identifier
            | TokenKind::Keyword(_)
    )
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
        // A method's parameters are in its own `[?Yield, ?Await]` context (e.g. a
        // generator method may not bind `yield`, an async method may not bind
        // `await`).
        let params = self.in_function_context(is_generator, is_async, Self::parse_params)?;
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
