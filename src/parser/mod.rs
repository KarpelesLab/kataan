//! The parser: a token stream → an AST.
//!
//! Kataan uses a hand-written recursive-descent parser with a precedence
//! ladder for binary operators (predictable performance, good diagnostics, no
//! parser-generator dependency). It covers the full ECMAScript grammar —
//! expressions, statements and declarations, functions/arrows, destructuring
//! patterns, classes, generators/async, and module `import`/`export` — split
//! across the `stmt`, `function`, `class`, and `module` submodules.
//!
//! The parser pre-tokenizes the whole input via the [`Lexer`], so it walks a
//! `Vec<Token>` with O(1) lookahead. Literal values are decoded by the `cook`
//! submodule.

mod class;
mod cook;
mod function;
mod module;
mod stmt;

#[cfg(test)]
mod tests;

use crate::ast::{
    Argument, ArrayElement, AssignOp, BinaryOp, Expr, Ident, LogicalOp, ObjectMember, PropertyKey,
    TemplateElement, TemplateLiteral, UnaryOp, UpdateOp,
};
use crate::common::Span;
use crate::error::{Error, Result};
use crate::lexer::{Keyword as Kw, Lexer, Token, TokenKind};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

/// Maximum recursive-descent nesting depth. Deeply nested source (e.g. a long
/// run of unary operators, brackets, parens, or destructuring patterns) drives
/// the parser into unbounded native recursion; without a cap this overflows the
/// stack and `abort()`s — uncatchable by a library embedder linking the
/// `alloc`-only core on a default ~8 MiB stack. The guard turns that into a
/// recoverable syntax [`Error`].
///
/// This counts *logical* nesting levels at the recursion hubs, but a single
/// level can span the whole expression ladder (`parse_assignment` →
/// `parse_conditional` → … → `parse_unary` → `parse_primary` → `parse_paren` →
/// back to `parse_assignment`), so each level costs a fair chunk of native
/// stack. Measured against a release build, the deepest paths (`(((…`, `[[[…`)
/// use roughly one level per ~3 KiB of stack; 300 levels therefore stays inside
/// a 1 MiB stack with margin and leaves ~8x headroom on the stated 8 MiB target,
/// while sitting comfortably in the few-hundred-to-~1000 range engines use and
/// far above any realistic program's nesting.
const MAX_PARSE_DEPTH: u32 = 300;

/// A recursive-descent parser over a borrowed source string.
pub struct Parser<'src> {
    source: &'src str,
    tokens: Vec<Token>,
    /// Index of the current token in `tokens`. Never advances past the final
    /// [`TokenKind::Eof`].
    pos: usize,
    /// When set, the `in` keyword is not treated as a relational operator
    /// (used inside `for`-loop headers, in a later increment).
    no_in: bool,
    /// Whether the cursor is inside a generator body (enables `yield`).
    in_generator: bool,
    /// Whether the cursor is inside an async body (enables `await`).
    in_async: bool,
    /// Current recursive-descent nesting depth, bounded by [`MAX_PARSE_DEPTH`].
    depth: u32,
}

/// RAII guard that decrements [`Parser::depth`] when dropped, keeping the count
/// correct across the parser's many `?` early-returns. It owns the `&mut Parser`
/// borrow, so the guarded body runs against `guard.parser` and the level is
/// released on every exit path (success, error, or early return).
struct DepthGuard<'a, 'src> {
    parser: &'a mut Parser<'src>,
}

impl Drop for DepthGuard<'_, '_> {
    fn drop(&mut self) {
        self.parser.depth -= 1;
    }
}

impl<'src> Parser<'src> {
    /// Creates a parser over `source`, tokenizing it up front. Returns a
    /// lexical [`Error`] if the source does not tokenize.
    pub fn new(source: &'src str) -> Result<Self> {
        let tokens = Lexer::new(source).tokenize()?;
        Ok(Self {
            source,
            tokens,
            pos: 0,
            no_in: false,
            in_generator: false,
            in_async: false,
            depth: 0,
        })
    }

    /// Parses a complete expression (allowing the top-level comma operator) and
    /// asserts that the entire input was consumed. This is the entry point for
    /// expression-level tools such as `kataan parse -e`.
    pub fn parse_expression_entry(source: &'src str) -> Result<Expr> {
        let mut p = Parser::new(source)?;
        let expr = p.parse_expression()?;
        p.expect_eof()?;
        Ok(expr)
    }

    // --- cursor ---------------------------------------------------------

    #[inline]
    fn peek(&self) -> TokenKind {
        self.tokens[self.pos].kind
    }

    #[inline]
    fn peek_tok(&self) -> Token {
        self.tokens[self.pos]
    }

    /// The kind of the token `n` positions ahead (clamped to `Eof`).
    #[inline]
    fn nth_kind(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map_or(TokenKind::Eof, |t| t.kind)
    }

    /// Whether the token `n` positions ahead is preceded by a line terminator.
    #[inline]
    fn nth_newline(&self, n: usize) -> bool {
        self.tokens
            .get(self.pos + n)
            .is_some_and(|t| t.newline_before)
    }

    #[inline]
    fn at(&self, kind: TokenKind) -> bool {
        self.peek() == kind
    }

    #[inline]
    fn cur_span(&self) -> Span {
        self.tokens[self.pos].span
    }

    /// The span of the token just consumed (for computing end positions).
    #[inline]
    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1)].span
    }

    /// Consumes and returns the current token, never advancing past `Eof`.
    fn bump(&mut self) -> Token {
        let tok = self.tokens[self.pos];
        if tok.kind != TokenKind::Eof {
            self.pos += 1;
        }
        tok
    }

    /// Consumes the current token if it matches `kind`.
    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Consumes a token of the expected `kind` or reports a syntax error.
    fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        if self.at(kind) {
            Ok(self.bump())
        } else {
            Err(self.err(format!("expected {kind:?}, found {:?}", self.peek())))
        }
    }

    /// Asserts the input is fully consumed.
    fn expect_eof(&self) -> Result<()> {
        if self.at(TokenKind::Eof) {
            Ok(())
        } else {
            Err(self.err(format!("unexpected trailing token {:?}", self.peek())))
        }
    }

    fn err(&self, message: impl Into<alloc::string::String>) -> Error {
        Error::syntax(message, self.cur_span())
    }

    /// Enters one recursive-descent nesting level, returning a [`DepthGuard`]
    /// that holds the parser and releases the level when dropped. Returns a
    /// syntax [`Error`] (never a panic) when [`MAX_PARSE_DEPTH`] is exceeded.
    /// Called at the top of the parser's true recursion hubs so deeply nested
    /// input cannot overflow the native stack and `abort()`. The guarded body
    /// runs against the returned guard's `parser` field.
    fn enter_recursion(&mut self) -> Result<DepthGuard<'_, 'src>> {
        if self.depth >= MAX_PARSE_DEPTH {
            return Err(self.err("maximum parser nesting depth exceeded"));
        }
        self.depth += 1;
        Ok(DepthGuard { parser: self })
    }

    fn err_at(&self, span: Span, message: impl Into<alloc::string::String>) -> Error {
        Error::syntax(message, span)
    }

    // --- expressions ----------------------------------------------------

    /// `Expression : AssignmentExpression (`,` AssignmentExpression)*`
    pub(crate) fn parse_expression(&mut self) -> Result<Expr> {
        let first = self.parse_assignment()?;
        if !self.at(TokenKind::Comma) {
            return Ok(first);
        }
        let start = first.span();
        let mut expressions = alloc::vec![first];
        while self.eat(TokenKind::Comma) {
            expressions.push(self.parse_assignment()?);
        }
        let span = start.to(self.prev_span());
        Ok(Expr::Sequence { expressions, span })
    }

    /// `AssignmentExpression` — handles arrow functions (via cover-grammar
    /// lookahead), `=`, and the compound assignments (right-associative).
    fn parse_assignment(&mut self) -> Result<Expr> {
        let guard = self.enter_recursion()?;
        guard.parser.parse_assignment_inner()
    }

    /// The body of [`Self::parse_assignment`], run inside the recursion guard.
    fn parse_assignment_inner(&mut self) -> Result<Expr> {
        if self.in_generator && self.at(TokenKind::Keyword(Kw::Yield)) {
            return self.parse_yield();
        }
        if self.at_arrow_head() {
            return self.parse_arrow();
        }
        let left = self.parse_conditional()?;
        let Some(op) = assign_op(self.peek()) else {
            return Ok(left);
        };
        if !left.is_assignment_target() {
            return Err(self.err_at(left.span(), "invalid assignment target"));
        }
        self.bump();
        let value = self.parse_assignment()?;
        let span = left.span().to(value.span());
        Ok(Expr::Assign {
            op,
            target: Box::new(left),
            value: Box::new(value),
            span,
        })
    }

    /// `ConditionalExpression : ShortCircuit (`?` AssignmentExpression `:`
    /// AssignmentExpression)?`
    fn parse_conditional(&mut self) -> Result<Expr> {
        let test = self.parse_short_circuit()?;
        if !self.at(TokenKind::Question) {
            return Ok(test);
        }
        self.bump();
        // The branches are AssignmentExpressions; `in` is always allowed here.
        let consequent = self.without_no_in(Self::parse_assignment)?;
        self.expect(TokenKind::Colon)?;
        let alternate = self.parse_assignment()?;
        let span = test.span().to(alternate.span());
        Ok(Expr::Conditional {
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate: Box::new(alternate),
            span,
        })
    }

    /// The short-circuit band. Per the spec grammar, this is *either* a
    /// `LogicalOR` expression (which may contain `&&` and `||`) *or* a
    /// `Coalesce` expression (`??` chained over `BitwiseOR` operands) — the two
    /// are disjoint, so `??` cannot be mixed with `&&`/`||` without
    /// parentheses. Both branches start from a single `BitwiseOR` operand and
    /// then dispatch on the operator that follows.
    fn parse_short_circuit(&mut self) -> Result<Expr> {
        let first = self.parse_bitor()?;
        match self.peek() {
            TokenKind::QuestionQuestion => self.parse_coalesce(first),
            TokenKind::AmpAmp | TokenKind::PipePipe => self.parse_logical_or(first),
            _ => Ok(first),
        }
    }

    /// A `??` chain whose operands are `BitwiseOR` expressions. A trailing
    /// `&&`/`||` is a mixing error.
    fn parse_coalesce(&mut self, mut left: Expr) -> Result<Expr> {
        while self.at(TokenKind::QuestionQuestion) {
            self.bump();
            let right = self.parse_bitor()?;
            left = logical(LogicalOp::Nullish, left, right);
        }
        if self.at(TokenKind::AmpAmp) || self.at(TokenKind::PipePipe) {
            return Err(self.err("`??` cannot be mixed with `||`/`&&` without parentheses"));
        }
        Ok(left)
    }

    /// The `&&`/`||` chain, starting from an already-parsed `BitwiseOR`
    /// operand. A trailing `??` is a mixing error.
    fn parse_logical_or(&mut self, first: Expr) -> Result<Expr> {
        let mut left = self.continue_logical_and(first)?;
        while self.at(TokenKind::PipePipe) {
            self.bump();
            let right_first = self.parse_bitor()?;
            let right = self.continue_logical_and(right_first)?;
            left = logical(LogicalOp::Or, left, right);
        }
        if self.at(TokenKind::QuestionQuestion) {
            return Err(self.err("`??` cannot be mixed with `||`/`&&` without parentheses"));
        }
        Ok(left)
    }

    /// Continues a `&&` chain over `BitwiseOR` operands, starting from an
    /// already-parsed left operand.
    fn continue_logical_and(&mut self, mut left: Expr) -> Result<Expr> {
        while self.at(TokenKind::AmpAmp) {
            self.bump();
            let right = self.parse_bitor()?;
            left = logical(LogicalOp::And, left, right);
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<Expr> {
        let mut left = self.parse_bitxor()?;
        while self.at(TokenKind::Pipe) {
            self.bump();
            let right = self.parse_bitxor()?;
            left = binary(BinaryOp::BitOr, left, right);
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Expr> {
        let mut left = self.parse_bitand()?;
        while self.at(TokenKind::Caret) {
            self.bump();
            let right = self.parse_bitand()?;
            left = binary(BinaryOp::BitXor, left, right);
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Expr> {
        let mut left = self.parse_equality()?;
        while self.at(TokenKind::Amp) {
            self.bump();
            let right = self.parse_equality()?;
            left = binary(BinaryOp::BitAnd, left, right);
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr> {
        let mut left = self.parse_relational()?;
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinaryOp::EqEq,
                TokenKind::BangEq => BinaryOp::NotEq,
                TokenKind::EqEqEq => BinaryOp::EqEqEq,
                TokenKind::BangEqEq => BinaryOp::NotEqEq,
                _ => break,
            };
            self.bump();
            let right = self.parse_relational()?;
            left = binary(op, left, right);
        }
        Ok(left)
    }

    fn parse_relational(&mut self) -> Result<Expr> {
        let mut left = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::LtEq => BinaryOp::LtEq,
                TokenKind::GtEq => BinaryOp::GtEq,
                TokenKind::Keyword(Kw::Instanceof) => BinaryOp::Instanceof,
                TokenKind::Keyword(Kw::In) if !self.no_in => BinaryOp::In,
                _ => break,
            };
            self.bump();
            let right = self.parse_shift()?;
            left = binary(op, left, right);
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                TokenKind::Shl => BinaryOp::Shl,
                TokenKind::Shr => BinaryOp::Shr,
                TokenKind::Ushr => BinaryOp::Ushr,
                _ => break,
            };
            self.bump();
            let right = self.parse_additive()?;
            left = binary(op, left, right);
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.bump();
            let right = self.parse_multiplicative()?;
            left = binary(op, left, right);
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut left = self.parse_exponent()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.bump();
            let right = self.parse_exponent()?;
            left = binary(op, left, right);
        }
        Ok(left)
    }

    /// `ExponentiationExpression` — right-associative `**`. A *unary*
    /// expression on the left (e.g. `-2 ** 2`) is a syntax error and must be
    /// parenthesized; an update expression (`++x ** 2`) is allowed.
    fn parse_exponent(&mut self) -> Result<Expr> {
        let unary_lead = self.at_unary_operator();
        let base = self.parse_unary()?;
        if self.at(TokenKind::StarStar) {
            if unary_lead {
                return Err(self.err_at(
                    base.span(),
                    "unary operator before `**` must be parenthesized",
                ));
            }
            self.bump();
            let exp = self.parse_exponent()?;
            let span = base.span().to(exp.span());
            return Ok(Expr::Binary {
                op: BinaryOp::Exp,
                left: Box::new(base),
                right: Box::new(exp),
                span,
            });
        }
        Ok(base)
    }

    /// Whether the current token begins a prefix *unary* operator (not the
    /// `++`/`--` update operators, which are permitted before `**`).
    fn at_unary_operator(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Bang
                | TokenKind::Tilde
                | TokenKind::Keyword(Kw::Typeof)
                | TokenKind::Keyword(Kw::Void)
                | TokenKind::Keyword(Kw::Delete)
        )
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        // Prefix unary/update/`await` operators (`!~+-`, `typeof`/`void`/`delete`,
        // `++`/`--`, `await`) recurse straight back into `parse_unary` — and
        // `**` recurses through `parse_exponent` → `parse_unary` — *without*
        // re-entering `parse_assignment`, so this is an independent recursion hub
        // that must carry its own depth guard.
        let guard = self.enter_recursion()?;
        guard.parser.parse_unary_inner()
    }

    /// The body of [`Self::parse_unary`], run inside the recursion guard.
    fn parse_unary_inner(&mut self) -> Result<Expr> {
        let tok = self.peek_tok();
        if self.in_async && tok.kind == TokenKind::Keyword(Kw::Await) {
            self.bump();
            let argument = self.parse_unary()?;
            let span = tok.span.to(argument.span());
            return Ok(Expr::Await {
                argument: Box::new(argument),
                span,
            });
        }
        let unary = match tok.kind {
            TokenKind::Plus => Some(UnaryOp::Plus),
            TokenKind::Minus => Some(UnaryOp::Minus),
            TokenKind::Bang => Some(UnaryOp::Not),
            TokenKind::Tilde => Some(UnaryOp::BitNot),
            TokenKind::Keyword(Kw::Typeof) => Some(UnaryOp::Typeof),
            TokenKind::Keyword(Kw::Void) => Some(UnaryOp::Void),
            TokenKind::Keyword(Kw::Delete) => Some(UnaryOp::Delete),
            _ => None,
        };
        if let Some(op) = unary {
            self.bump();
            let argument = self.parse_unary()?;
            let span = tok.span.to(argument.span());
            return Ok(Expr::Unary {
                op,
                argument: Box::new(argument),
                span,
            });
        }

        if let Some(op) = update_op(tok.kind) {
            self.bump();
            let argument = self.parse_unary()?;
            let span = tok.span.to(argument.span());
            return Ok(Expr::Update {
                op,
                prefix: true,
                argument: Box::new(argument),
                span,
            });
        }

        self.parse_postfix()
    }

    /// Postfix `++` / `--`. Per ASI, a line terminator may not appear between
    /// the operand and the operator.
    fn parse_postfix(&mut self) -> Result<Expr> {
        let expr = self.parse_lhs()?;
        let tok = self.peek_tok();
        if !tok.newline_before
            && let Some(op) = update_op(tok.kind)
        {
            self.bump();
            let span = expr.span().to(tok.span);
            return Ok(Expr::Update {
                op,
                prefix: false,
                argument: Box::new(expr),
                span,
            });
        }
        Ok(expr)
    }

    // --- left-hand-side: new / call / member ----------------------------

    fn parse_lhs(&mut self) -> Result<Expr> {
        let head = if self.at(TokenKind::Keyword(Kw::New)) {
            self.parse_new()?
        } else {
            self.parse_primary()?
        };
        self.parse_tails(head, true)
    }

    /// `new Callee` / `new Callee(args)` — the callee is a member expression
    /// (no call), and the optional argument list binds to this `new`.
    fn parse_new(&mut self) -> Result<Expr> {
        let start = self.cur_span();
        self.bump(); // `new`
        if self.at(TokenKind::Dot) {
            self.bump(); // `.`
            // The only valid meta-property here is `new.target` (`target` is a
            // contextual keyword).
            if self.at(TokenKind::Keyword(Kw::Target)) {
                self.bump();
                return Ok(Expr::NewTarget(start.to(self.prev_span())));
            }
            return Err(self.err("expected `target` after `new.`"));
        }
        let callee = if self.at(TokenKind::Keyword(Kw::New)) {
            self.parse_new()?
        } else {
            let primary = self.parse_primary()?;
            self.parse_tails(primary, false)?
        };
        let arguments = if self.at(TokenKind::LParen) {
            self.parse_arguments()?
        } else {
            Vec::new()
        };
        let span = start.to(self.prev_span());
        Ok(Expr::New {
            callee: Box::new(callee),
            arguments,
            span,
        })
    }

    /// Consumes the member/call/optional-chain/tagged-template tails after a
    /// primary or `new` expression. `allow_call` is false while parsing a
    /// `new` callee (so the parens bind to the `new`, not to a call).
    fn parse_tails(&mut self, mut expr: Expr, allow_call: bool) -> Result<Expr> {
        let start = expr.span();
        let mut saw_optional = false;
        loop {
            match self.peek() {
                TokenKind::Dot => {
                    self.bump();
                    let (property, end) = self.parse_member_property()?;
                    let span = expr.span().to(end);
                    expr = Expr::Member {
                        object: Box::new(expr),
                        property,
                        optional: false,
                        span,
                    };
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.without_no_in(Self::parse_expression)?;
                    let end = self.expect(TokenKind::RBracket)?.span;
                    let span = expr.span().to(end);
                    expr = Expr::Member {
                        object: Box::new(expr),
                        property: PropertyKey::Computed(Box::new(index)),
                        optional: false,
                        span,
                    };
                }
                TokenKind::QuestionDot => {
                    self.bump();
                    saw_optional = true;
                    expr = self.parse_optional_tail(expr, allow_call)?;
                }
                TokenKind::LParen if allow_call => {
                    let arguments = self.parse_arguments()?;
                    let span = expr.span().to(self.prev_span());
                    expr = Expr::Call {
                        callee: Box::new(expr),
                        arguments,
                        optional: false,
                        span,
                    };
                }
                TokenKind::NoSubstitutionTemplate | TokenKind::TemplateHead if allow_call => {
                    let quasi = self.parse_template_literal()?;
                    let span = expr.span().to(quasi.span);
                    expr = Expr::TaggedTemplate {
                        tag: Box::new(expr),
                        quasi,
                        span,
                    };
                }
                _ => {
                    // A chain containing any `?.` becomes an `OptChain` so the
                    // evaluator knows the short-circuit boundary (a nullish `?.`
                    // base skips the rest of the chain → `undefined`).
                    if saw_optional {
                        let span = start.to(expr.span());
                        expr = Expr::OptChain {
                            expr: Box::new(expr),
                            span,
                        };
                    }
                    return Ok(expr);
                }
            }
        }
    }

    /// Parses the link after a `?.` token: `?.prop`, `?.[expr]`, or `?.(args)`.
    fn parse_optional_tail(&mut self, object: Expr, allow_call: bool) -> Result<Expr> {
        match self.peek() {
            TokenKind::LParen if allow_call => {
                let arguments = self.parse_arguments()?;
                let span = object.span().to(self.prev_span());
                Ok(Expr::Call {
                    callee: Box::new(object),
                    arguments,
                    optional: true,
                    span,
                })
            }
            TokenKind::LBracket => {
                self.bump();
                let index = self.without_no_in(Self::parse_expression)?;
                let end = self.expect(TokenKind::RBracket)?.span;
                let span = object.span().to(end);
                Ok(Expr::Member {
                    object: Box::new(object),
                    property: PropertyKey::Computed(Box::new(index)),
                    optional: true,
                    span,
                })
            }
            _ => {
                let (property, end) = self.parse_member_property()?;
                let span = object.span().to(end);
                Ok(Expr::Member {
                    object: Box::new(object),
                    property,
                    optional: true,
                    span,
                })
            }
        }
    }

    /// A property name after `.` (or `?.`): an `IdentifierName` (any keyword is
    /// allowed here, e.g. `obj.class`) or a private name (`obj.#x`). Returns
    /// the key and the end span.
    fn parse_member_property(&mut self) -> Result<(PropertyKey, Span)> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Identifier => {
                self.bump();
                Ok((PropertyKey::Ident(tok.text(self.source).into()), tok.span))
            }
            TokenKind::Keyword(kw) => {
                self.bump();
                Ok((PropertyKey::Ident(kw.as_str().into()), tok.span))
            }
            TokenKind::PrivateName => {
                self.bump();
                // Strip the leading `#`.
                let name = &tok.text(self.source)[1..];
                Ok((PropertyKey::Private(name.into()), tok.span))
            }
            _ => Err(self.err(format!("expected a property name, found {:?}", tok.kind))),
        }
    }

    /// A parenthesized argument list `( … )` with optional spread arguments and
    /// trailing comma.
    fn parse_arguments(&mut self) -> Result<Vec<Argument>> {
        self.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        while !self.at(TokenKind::RParen) {
            if self.eat(TokenKind::DotDotDot) {
                args.push(Argument::Spread(self.parse_assignment()?));
            } else {
                args.push(Argument::Item(self.parse_assignment()?));
            }
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        self.expect(TokenKind::RParen)?;
        Ok(args)
    }

    // --- primary --------------------------------------------------------

    fn parse_primary(&mut self) -> Result<Expr> {
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Number => {
                self.bump();
                Ok(Expr::Number {
                    value: cook::number(tok.text(self.source)),
                    span: tok.span,
                })
            }
            TokenKind::BigInt => {
                self.bump();
                Ok(Expr::BigInt {
                    digits: cook::bigint(tok.text(self.source)).into(),
                    span: tok.span,
                })
            }
            TokenKind::String => {
                self.bump();
                Ok(Expr::Str {
                    value: cook::string(tok.text(self.source), tok.span)?.into(),
                    span: tok.span,
                })
            }
            TokenKind::Regex => {
                self.bump();
                let (pattern, flags) = split_regex(tok.text(self.source));
                Ok(Expr::Regex {
                    pattern: pattern.into(),
                    flags: flags.into(),
                    span: tok.span,
                })
            }
            TokenKind::NoSubstitutionTemplate | TokenKind::TemplateHead => {
                Ok(Expr::Template(self.parse_template_literal()?))
            }
            TokenKind::Keyword(Kw::True) => {
                self.bump();
                Ok(Expr::Bool {
                    value: true,
                    span: tok.span,
                })
            }
            TokenKind::Keyword(Kw::False) => {
                self.bump();
                Ok(Expr::Bool {
                    value: false,
                    span: tok.span,
                })
            }
            TokenKind::Keyword(Kw::Null) => {
                self.bump();
                Ok(Expr::Null(tok.span))
            }
            TokenKind::Keyword(Kw::This) => {
                self.bump();
                Ok(Expr::This(tok.span))
            }
            TokenKind::Keyword(Kw::Super) => {
                self.bump();
                Ok(Expr::Super(tok.span))
            }
            TokenKind::Identifier => {
                self.bump();
                Ok(Expr::Ident(Ident::new(tok.text(self.source), tok.span)))
            }
            // `async function …` is a function expression. This must precede the
            // contextual-keyword arm below, which would otherwise treat `async`
            // as a plain identifier reference.
            TokenKind::Keyword(Kw::Async)
                if self.nth_kind(1) == TokenKind::Keyword(Kw::Function) && !self.nth_newline(1) =>
            {
                self.bump(); // `async`
                self.parse_function_expr(true, tok.span)
            }
            TokenKind::Keyword(kw) if kw.is_contextual() => {
                self.bump();
                Ok(Expr::Ident(Ident::new(kw.as_str(), tok.span)))
            }
            TokenKind::LParen => self.parse_paren(),
            TokenKind::LBracket => self.parse_array(),
            TokenKind::LBrace => self.parse_object(),
            TokenKind::Keyword(Kw::Function) => self.parse_function_expr(false, tok.span),
            TokenKind::Keyword(Kw::Class) => self.parse_class_expr(),
            // A bare `#x` — only valid as the left of `#x in obj` (brand check).
            TokenKind::PrivateName => {
                self.bump();
                let name = &tok.text(self.source)[1..];
                Ok(Expr::PrivateName(name.into(), tok.span))
            }
            _ => Err(self.err(format!("unexpected token {:?}", tok.kind))),
        }
    }

    /// `( Expression )` — no parenthesized-expression node is retained; an
    /// empty `()` is only valid as an arrow head and is rejected here.
    fn parse_paren(&mut self) -> Result<Expr> {
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err("unexpected `)` (empty parentheses)"));
        }
        let expr = self.without_no_in(Self::parse_expression)?;
        self.expect(TokenKind::RParen)?;
        Ok(expr)
    }

    fn parse_array(&mut self) -> Result<Expr> {
        let start = self.expect(TokenKind::LBracket)?.span;
        let mut elements = Vec::new();
        while !self.at(TokenKind::RBracket) {
            if self.at(TokenKind::Comma) {
                self.bump();
                elements.push(ArrayElement::Hole);
                continue;
            }
            if self.eat(TokenKind::DotDotDot) {
                elements.push(ArrayElement::Spread(self.parse_assignment()?));
            } else {
                elements.push(ArrayElement::Item(self.parse_assignment()?));
            }
            if !self.at(TokenKind::RBracket) {
                self.expect(TokenKind::Comma)?;
            }
        }
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(Expr::Array {
            elements,
            span: start.to(end),
        })
    }

    fn parse_object(&mut self) -> Result<Expr> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let mut members = Vec::new();
        while !self.at(TokenKind::RBrace) {
            members.push(self.parse_object_member()?);
            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }
        let end = self.expect(TokenKind::RBrace)?.span;
        Ok(Expr::Object {
            members,
            span: start.to(end),
        })
    }

    fn parse_object_member(&mut self) -> Result<ObjectMember> {
        let start = self.cur_span();
        if self.eat(TokenKind::DotDotDot) {
            let value = self.parse_assignment()?;
            let span = start.to(value.span());
            return Ok(ObjectMember::Spread {
                value: Box::new(value),
                span,
            });
        }

        // `async method() { … }` / `async *gen() { … }` / `async [expr]() { … }`
        // (when `async` is not itself the property name).
        if self.at(TokenKind::Keyword(Kw::Async))
            && !self.nth_newline(1)
            && !self.modifier_is_name(1)
            // In an object literal, `,`/`:` after `async` mean it is the key.
            && !matches!(self.nth_kind(1), TokenKind::Comma | TokenKind::Colon)
        {
            self.bump(); // `async`
            let is_gen = self.eat(TokenKind::Star);
            let key = self.parse_class_key()?;
            let func = self.parse_method_tail(true, is_gen)?;
            return Ok(ObjectMember::Property {
                key,
                value: Box::new(Expr::Function(func)),
                shorthand: false,
                span: start.to(self.prev_span()),
            });
        }

        // `*key() { … }` / `*[expr]() { … }` — a generator method.
        if self.at(TokenKind::Star) {
            self.bump();
            let key = self.parse_class_key()?;
            let func = self.parse_method_tail(false, true)?;
            return Ok(ObjectMember::Property {
                key,
                value: Box::new(Expr::Function(func)),
                shorthand: false,
                span: start.to(self.prev_span()),
            });
        }

        // `get key() { … }` / `set key(v) { … }` — accessors. (When `get`/`set`
        // is directly followed by `(`, `:`, or a member terminator it is an
        // ordinary property/method *named* `get`/`set` instead.)
        if let TokenKind::Keyword(kw @ (Kw::Get | Kw::Set)) = self.peek()
            && !matches!(
                self.nth_kind(1),
                TokenKind::LParen
                    | TokenKind::Colon
                    | TokenKind::Comma
                    | TokenKind::RBrace
                    | TokenKind::Eq
            )
        {
            self.bump(); // `get` / `set`
            let key = self.parse_class_key()?;
            let func = self.parse_method_tail(false, false)?;
            return Ok(ObjectMember::Accessor {
                is_getter: kw == Kw::Get,
                key,
                value: func,
                span: start.to(self.prev_span()),
            });
        }

        // Computed key `[expr]: value` or computed method `[expr]() { … }`.
        if self.at(TokenKind::LBracket) {
            self.bump();
            let key_expr = self.without_no_in(Self::parse_assignment)?;
            self.expect(TokenKind::RBracket)?;
            // Computed method shorthand `[expr]() { … }`.
            if self.at(TokenKind::LParen) {
                let func = self.parse_method_tail(false, false)?;
                let span = start.to(self.prev_span());
                return Ok(ObjectMember::Property {
                    key: PropertyKey::Computed(Box::new(key_expr)),
                    value: Box::new(Expr::Function(func)),
                    shorthand: false,
                    span,
                });
            }
            self.expect(TokenKind::Colon)?;
            let value = self.parse_assignment()?;
            let span = start.to(value.span());
            return Ok(ObjectMember::Property {
                key: PropertyKey::Computed(Box::new(key_expr)),
                value: Box::new(value),
                shorthand: false,
                span,
            });
        }

        let tok = self.peek_tok();
        // A literal string/number key always takes `: value`.
        if matches!(tok.kind, TokenKind::String | TokenKind::Number) {
            self.bump();
            let key = if tok.kind == TokenKind::String {
                PropertyKey::Str(cook::string(tok.text(self.source), tok.span)?.into())
            } else {
                PropertyKey::Number(cook::number(tok.text(self.source)))
            };
            self.expect(TokenKind::Colon)?;
            let value = self.parse_assignment()?;
            let span = start.to(value.span());
            return Ok(ObjectMember::Property {
                key,
                value: Box::new(value),
                shorthand: false,
                span,
            });
        }

        // Identifier-name key: either `name: value` (any IdentifierName) or the
        // shorthand `{ name }` (only a real identifier reference).
        let (name, can_shorthand): (Box<str>, bool) = match tok.kind {
            TokenKind::Identifier => (tok.text(self.source).into(), true),
            TokenKind::Keyword(kw) if kw.is_contextual() => (kw.as_str().into(), true),
            TokenKind::Keyword(kw) => (kw.as_str().into(), false),
            _ => return Err(self.err(format!("expected a property key, found {:?}", tok.kind))),
        };
        self.bump();

        // Method shorthand `name() { … }` (getters/setters/async/generator
        // object methods are added later).
        if self.at(TokenKind::LParen) {
            let func = self.parse_method_tail(false, false)?;
            let span = start.to(self.prev_span());
            return Ok(ObjectMember::Property {
                key: PropertyKey::Ident(name),
                value: Box::new(Expr::Function(func)),
                shorthand: false,
                span,
            });
        }

        if self.eat(TokenKind::Colon) {
            let value = self.parse_assignment()?;
            let span = start.to(value.span());
            return Ok(ObjectMember::Property {
                key: PropertyKey::Ident(name),
                value: Box::new(value),
                shorthand: false,
                span,
            });
        }

        if !can_shorthand {
            return Err(self.err_at(
                tok.span,
                "reserved word cannot be used as a shorthand property",
            ));
        }
        let ident_expr = Expr::Ident(Ident::new(name.clone(), tok.span));
        // CoverInitializedName `{ name = default }` — only meaningful when the
        // object is reinterpreted as a destructuring assignment target; the
        // default lives in the property value as an assignment expression.
        if self.eat(TokenKind::Eq) {
            let default = self.parse_assignment()?;
            let span = start.to(default.span());
            return Ok(ObjectMember::Property {
                key: PropertyKey::Ident(name),
                value: Box::new(Expr::Assign {
                    op: crate::ast::AssignOp::Assign,
                    target: Box::new(ident_expr),
                    value: Box::new(default),
                    span,
                }),
                shorthand: true,
                span,
            });
        }
        // Shorthand `{ name }`.
        Ok(ObjectMember::Property {
            key: PropertyKey::Ident(name),
            value: Box::new(ident_expr),
            shorthand: true,
            span: tok.span,
        })
    }

    // --- templates ------------------------------------------------------

    /// Parses a template literal starting at a `NoSubstitutionTemplate` or
    /// `TemplateHead` token (the substitutions are full expressions).
    fn parse_template_literal(&mut self) -> Result<TemplateLiteral> {
        let start = self.cur_span();
        let mut quasis = Vec::new();
        let mut expressions = Vec::new();

        let head = self.bump();
        if head.kind == TokenKind::NoSubstitutionTemplate {
            quasis.push(self.template_element(head, TemplatePart::NoSub));
            return Ok(TemplateLiteral {
                quasis,
                expressions,
                span: start.to(head.span),
            });
        }
        // TemplateHead.
        quasis.push(self.template_element(head, TemplatePart::Head));
        loop {
            expressions.push(self.without_no_in(Self::parse_expression)?);
            let part = self.bump();
            match part.kind {
                TokenKind::TemplateMiddle => {
                    quasis.push(self.template_element(part, TemplatePart::Middle));
                }
                TokenKind::TemplateTail => {
                    quasis.push(self.template_element(part, TemplatePart::Tail));
                    return Ok(TemplateLiteral {
                        quasis,
                        expressions,
                        span: start.to(part.span),
                    });
                }
                _ => {
                    return Err(self.err_at(
                        part.span,
                        format!("expected a template continuation, found {:?}", part.kind),
                    ));
                }
            }
        }
    }

    /// Extracts the raw and cooked text of one template segment, stripping the
    /// surrounding delimiters according to which part it is.
    fn template_element(&self, tok: Token, part: TemplatePart) -> TemplateElement {
        let text = tok.text(self.source);
        let inner = match part {
            // `` `…` ``  and  `}…` ``  drop one delimiter each side / end.
            TemplatePart::NoSub | TemplatePart::Tail => &text[1..text.len() - 1],
            // `` `…${ ``  and  `}…${ ``  drop one leading and the trailing `${`.
            TemplatePart::Head | TemplatePart::Middle => &text[1..text.len() - 2],
        };
        let cooked = cook::decode_escapes(inner, tok.span).ok().map(Into::into);
        TemplateElement {
            raw: inner.into(),
            cooked,
            span: tok.span,
        }
    }

    // --- helpers --------------------------------------------------------

    /// Runs `f` with `no_in` cleared (e.g. inside brackets/parens, where a
    /// `for`-header's `in` restriction does not apply), restoring it after.
    fn without_no_in<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let saved = self.no_in;
        self.no_in = false;
        let r = f(self);
        self.no_in = saved;
        r
    }

    /// Runs `f` with `no_in` set, so a top-level `in` is *not* taken as a
    /// relational operator — used while parsing a `for`-loop header, where a
    /// bare `in` introduces a `for-in` loop instead.
    fn with_no_in<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let saved = self.no_in;
        self.no_in = true;
        let r = f(self);
        self.no_in = saved;
        r
    }

    /// Parses a `yield` / `yield*` expression (the caller has checked we are in
    /// a generator and the cursor is at `yield`).
    fn parse_yield(&mut self) -> Result<Expr> {
        let start = self.bump().span; // `yield`
        let delegate = self.eat(TokenKind::Star);
        // `yield*` always takes an operand; bare `yield` takes one unless a line
        // terminator or an expression-terminating token follows.
        let argument = if delegate || self.yield_has_argument() {
            Some(Box::new(self.parse_assignment()?))
        } else {
            None
        };
        let span = start.to(self.prev_span());
        Ok(Expr::Yield {
            argument,
            delegate,
            span,
        })
    }

    /// Whether a bare `yield` is followed by an operand.
    fn yield_has_argument(&self) -> bool {
        if self.peek_tok().newline_before {
            return false;
        }
        !matches!(
            self.peek(),
            TokenKind::RParen
                | TokenKind::RBracket
                | TokenKind::RBrace
                | TokenKind::Comma
                | TokenKind::Semicolon
                | TokenKind::Colon
                | TokenKind::Eof
        )
    }

    /// Runs `f` with the generator/async context set to the given values
    /// (used when entering a function/method body), restoring them afterward.
    pub(crate) fn in_function_context<T>(
        &mut self,
        is_generator: bool,
        is_async: bool,
        f: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let saved = (self.in_generator, self.in_async);
        self.in_generator = is_generator;
        self.in_async = is_async;
        let r = f(self);
        (self.in_generator, self.in_async) = saved;
        r
    }
}

/// Builds an `Expr::Binary` with a span covering both operands.
fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    let span = left.span().to(right.span());
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
        span,
    }
}

/// Builds an `Expr::Logical` with a span covering both operands.
fn logical(op: LogicalOp, left: Expr, right: Expr) -> Expr {
    let span = left.span().to(right.span());
    Expr::Logical {
        op,
        left: Box::new(left),
        right: Box::new(right),
        span,
    }
}

/// Maps an assignment-operator token to its [`AssignOp`].
fn assign_op(kind: TokenKind) -> Option<AssignOp> {
    Some(match kind {
        TokenKind::Eq => AssignOp::Assign,
        TokenKind::PlusEq => AssignOp::AddAssign,
        TokenKind::MinusEq => AssignOp::SubAssign,
        TokenKind::StarEq => AssignOp::MulAssign,
        TokenKind::SlashEq => AssignOp::DivAssign,
        TokenKind::PercentEq => AssignOp::ModAssign,
        TokenKind::StarStarEq => AssignOp::ExpAssign,
        TokenKind::ShlEq => AssignOp::ShlAssign,
        TokenKind::ShrEq => AssignOp::ShrAssign,
        TokenKind::UshrEq => AssignOp::UshrAssign,
        TokenKind::AmpEq => AssignOp::BitAndAssign,
        TokenKind::PipeEq => AssignOp::BitOrAssign,
        TokenKind::CaretEq => AssignOp::BitXorAssign,
        TokenKind::AmpAmpEq => AssignOp::AndAssign,
        TokenKind::PipePipeEq => AssignOp::OrAssign,
        TokenKind::QuestionQuestionEq => AssignOp::NullishAssign,
        _ => return None,
    })
}

/// Maps `++` / `--` tokens to an [`UpdateOp`].
fn update_op(kind: TokenKind) -> Option<UpdateOp> {
    match kind {
        TokenKind::PlusPlus => Some(UpdateOp::Inc),
        TokenKind::MinusMinus => Some(UpdateOp::Dec),
        _ => None,
    }
}

/// Splits a regex token's text `/pattern/flags` into its pattern and flags.
/// The closing slash is the last `/` in the token, since flag characters never
/// include `/`.
fn split_regex(text: &str) -> (&str, &str) {
    let close = text.rfind('/').unwrap_or(text.len() - 1);
    (&text[1..close], &text[close + 1..])
}

/// Which segment of a template literal a token represents (controls delimiter
/// stripping).
#[derive(Clone, Copy)]
enum TemplatePart {
    NoSub,
    Head,
    Middle,
    Tail,
}
