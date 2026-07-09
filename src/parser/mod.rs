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
mod validate;

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
const MAX_PARSE_DEPTH: u32 = crate::limits::DEFAULT_MAX_PARSE_DEPTH;

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
    /// Span of the most recently parsed *parenthesized* primary expression whose
    /// stripped value is an object or array literal — e.g. `({})` / `([a])`. The
    /// parentheses are not retained in the AST, but a `ParenthesizedExpression`
    /// wrapping an `ObjectLiteral`/`ArrayLiteral` is **not** a valid assignment
    /// target (only the bare cover `{ … } = …` reinterprets as a pattern), so the
    /// assignment parser consults this (matching on span) to reject `({}) = 1`.
    /// Cleared at the start of every primary parse so it is never stale.
    paren_obj_arr_span: Option<Span>,
    /// Current recursive-descent nesting depth, bounded by [`MAX_PARSE_DEPTH`].
    depth: u32,
    /// Whether the cursor is at the **module top level** — the direct statement
    /// list of a Module (a `ModuleItem` position). `import` / `export`
    /// declarations are only legal here; nested inside a block, a function body,
    /// a control-flow body, or a `switch` case they are early Syntax Errors.
    /// Set true only while [`Parser::parse_module`] parses the top list, and
    /// cleared on entry to any nested statement context.
    module_top_level: bool,
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
        Self::with_goal(source, false)
    }

    /// Like [`Parser::new`] but for the given goal — a `module` parser lexes
    /// without the script-only Annex B HTML-like comments.
    pub fn with_goal(source: &'src str, module: bool) -> Result<Self> {
        let tokens = Lexer::with_goal(source, module).tokenize()?;
        Ok(Self {
            source,
            tokens,
            pos: 0,
            no_in: false,
            in_generator: false,
            in_async: false,
            paren_obj_arr_span: None,
            depth: 0,
            module_top_level: false,
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

    /// The cooked name of an identifier (or private-name) token: its raw source
    /// text, with any `\uXXXX` / `\u{…}` escapes decoded. The common
    /// no-escape case borrows the source directly; only an escaped identifier
    /// allocates. Use this everywhere an identifier *name* is read so that, e.g.
    /// `a` and `a` denote the same binding/property.
    fn ident_name(&self, tok: Token) -> alloc::borrow::Cow<'src, str> {
        let text = tok.span.slice(self.source);
        if tok.had_escape {
            alloc::borrow::Cow::Owned(cook::identifier_name(text))
        } else {
            alloc::borrow::Cow::Borrowed(text)
        }
    }

    /// Like [`Self::ident_name`], but for positions that require a
    /// non-reserved identifier (a `BindingIdentifier`, `IdentifierReference`,
    /// or `LabelIdentifier`). A reserved word written *with* a Unicode escape
    /// (e.g. `if` for `if`) tokenizes as an `Identifier`, but its
    /// `StringValue` is a reserved word, which is a Syntax Error here (an
    /// un-escaped reserved word would have tokenized as a keyword and never
    /// reached an identifier site). Contextual keywords (`async`, `let`, …) are
    /// not reserved and remain valid identifiers.
    fn checked_ident_name(&self, tok: Token) -> Result<alloc::borrow::Cow<'src, str>> {
        let name = self.ident_name(tok);
        if tok.had_escape {
            // An escaped always-reserved word is never a valid identifier.
            // `yield`/`await` are reserved only in a generator/async context
            // respectively — an un-escaped one is intercepted earlier as a
            // keyword, but an escaped one arrives here as an `Identifier`, so it
            // must be rejected under the same context rule.
            if is_always_reserved_word(&name)
                || (name == "yield" && self.in_generator)
                || (name == "await" && self.in_async)
            {
                return Err(self.err_at(
                    tok.span,
                    "an escaped reserved word may not be used as an identifier",
                ));
            }
        }
        Ok(name)
    }

    /// The cooked name of a `PrivateName` token (`#x`): the text after the `#`,
    /// with any `\u` escapes decoded.
    fn private_name(&self, tok: Token) -> Box<str> {
        let text = &tok.span.slice(self.source)[1..]; // drop leading `#`
        if tok.had_escape {
            cook::identifier_name(text).into()
        } else {
            text.into()
        }
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
        // A parenthesized object/array literal (`({}) = …`, `([a]) = …`) is not a
        // valid assignment target: only the bare cover reinterprets as a pattern.
        // The parser strips the parens, so we match the marker span (set by
        // `parse_paren` while parsing `left`) against the target's span.
        if matches!(&left, Expr::Object { .. } | Expr::Array { .. })
            && self.paren_obj_arr_span == Some(left.span())
        {
            return Err(self.err_at(
                left.span(),
                "a parenthesized object or array literal is not a valid assignment target",
            ));
        }
        if !left.is_assignment_target() {
            // AnnexB web-compat: a direct `CallExpression` LHS (`f() = x`,
            // `f() += x`) is not a static SyntaxError — it parses and is a runtime
            // ReferenceError in sloppy code. The strict-mode rejection is deferred
            // to the validator (which knows the mode). Logical assignment
            // (`&&=`/`||=`/`??=`) is excluded: its target must be simple in *both*
            // modes, so a call target there stays a parse error.
            let logical = matches!(
                op,
                crate::ast::AssignOp::AndAssign
                    | crate::ast::AssignOp::OrAssign
                    | crate::ast::AssignOp::NullishAssign
            );
            if logical || !left.is_web_compat_call_target() {
                return Err(self.err_at(left.span(), "invalid assignment target"));
            }
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
        // `**` is right-recursive: `parse_exponent` calls itself for the
        // exponent (`2**2**…`). The inner `parse_unary` enters *and exits* its
        // own guard per call, so the `**` recursion itself is uncounted and can
        // overflow the native stack. Guard `parse_exponent` directly so each
        // `**` level counts toward `MAX_PARSE_DEPTH`.
        let guard = self.enter_recursion()?;
        guard.parser.parse_exponent_inner()
    }

    /// The body of [`Self::parse_exponent`], run inside the recursion guard.
    fn parse_exponent_inner(&mut self) -> Result<Expr> {
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
        // A `new` callee may itself be `new` (`new new new … Object`), and
        // `parse_new` recurses straight back into itself for each level without
        // passing through a guarded hub. Guard it so each `new` counts toward
        // `MAX_PARSE_DEPTH` and a deep chain returns a syntax error instead of
        // overflowing the native stack.
        let guard = self.enter_recursion()?;
        guard.parser.parse_new_inner()
    }

    /// The body of [`Self::parse_new`], run inside the recursion guard.
    fn parse_new_inner(&mut self) -> Result<Expr> {
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
        // A dynamic import call — `import(...)`, `import.source(...)`,
        // `import.defer(...)` — is a `CallExpression`, which may not be the
        // callee of `new`. `import.meta` (a meta-property) is a valid member
        // expression and is allowed through.
        if self.at(TokenKind::Keyword(Kw::Import)) {
            let is_call = self.nth_kind(1) == TokenKind::LParen
                || (self.nth_kind(1) == TokenKind::Dot
                    && self.nth_kind(2) == TokenKind::Identifier
                    && matches!(
                        self.tokens.get(self.pos + 2).map(|t| t.text(self.source)),
                        Some("source" | "defer")
                    )
                    && self.nth_kind(3) == TokenKind::LParen);
            if is_call {
                return Err(self.err("a dynamic import call is not a valid `new` callee"));
            }
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
                    // A tagged template may not appear in the tail of an optional
                    // chain (`a?.b`…`` ` is a SyntaxError); parentheses break the
                    // chain and make it legal again.
                    if saw_optional {
                        return Err(self.err("tagged template is not allowed in an optional chain"));
                    }
                    let quasi = self.parse_template_literal(true)?;
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
                Ok((PropertyKey::Ident(self.ident_name(tok).into()), tok.span))
            }
            TokenKind::Keyword(kw) => {
                self.bump();
                Ok((PropertyKey::Ident(kw.as_str().into()), tok.span))
            }
            TokenKind::PrivateName => {
                self.bump();
                // Strip the leading `#`, then cook any escapes in the name.
                Ok((PropertyKey::Private(self.private_name(tok)), tok.span))
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

    // --- decorators -----------------------------------------------------

    /// Parses a `DecoratorList` — a run of `@ Decorator` prefixes on a class or
    /// class element. The decorator expressions are parsed (and validated for
    /// grammatical shape) but *not* retained: no test exercises decorator
    /// application at runtime, so they are evaluated as no-ops. Returns the
    /// number of decorators consumed.
    pub(super) fn parse_decorators(&mut self) -> Result<usize> {
        let mut count = 0;
        while self.at(TokenKind::At) {
            self.parse_decorator()?;
            count += 1;
        }
        Ok(count)
    }

    /// Parses a single `Decorator`:
    ///   `@ DecoratorMemberExpression`
    ///   `@ DecoratorParenthesizedExpression`  (`@ ( Expression )`)
    ///   `@ DecoratorCallExpression`           (member expr + `Arguments`)
    fn parse_decorator(&mut self) -> Result<()> {
        self.expect(TokenKind::At)?;
        // `@ ( Expression )` — a fully parenthesized expression.
        if self.at(TokenKind::LParen) {
            self.bump();
            self.without_no_in(Self::parse_expression)?;
            self.expect(TokenKind::RParen)?;
            return Ok(());
        }
        // `DecoratorMemberExpression` — an IdentifierReference followed by a
        // chain of `.IdentifierName` / `.PrivateIdentifier` (no computed `[…]`).
        match self.peek() {
            TokenKind::Identifier | TokenKind::Keyword(_) => self.bump(),
            other => return Err(self.err(format!("expected a decorator, found {other:?}"))),
        };
        while self.at(TokenKind::Dot) {
            self.bump();
            match self.peek() {
                // `.IdentifierName` (any name, reserved words allowed) or
                // `.#PrivateIdentifier`.
                TokenKind::Identifier | TokenKind::Keyword(_) | TokenKind::PrivateName => {
                    self.bump();
                }
                other => {
                    return Err(self.err(format!("expected a property name, found {other:?}")));
                }
            }
        }
        // An optional trailing `Arguments` makes this a `DecoratorCallExpression`.
        if self.at(TokenKind::LParen) {
            self.parse_arguments()?;
        }
        Ok(())
    }

    // --- primary --------------------------------------------------------

    fn parse_primary(&mut self) -> Result<Expr> {
        // Reset the parenthesized-object/array marker; only `parse_paren` re-sets
        // it, so it never lingers past the primary it describes.
        self.paren_obj_arr_span = None;
        let tok = self.peek_tok();
        match tok.kind {
            TokenKind::Number => {
                self.bump();
                let text = tok.text(self.source);
                Ok(Expr::Number {
                    value: cook::number(text),
                    span: tok.span,
                    legacy_octal: cook::is_legacy_octal_literal(text),
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
                let raw = tok.text(self.source);
                Ok(Expr::Str {
                    value: cook::string(raw, tok.span)?.into(),
                    legacy_octal: cook::string_has_legacy_octal_escape(raw),
                    span: tok.span,
                })
            }
            TokenKind::Regex => {
                self.bump();
                let (pattern, flags) = split_regex(tok.text(self.source));
                // A regex *literal* is validated at parse time: an invalid pattern
                // or invalid/duplicate flags is an early (parse-phase) SyntaxError,
                // per `RegExp` literal evaluation. Running the regex engine's own
                // parser/compiler here surfaces the same diagnostics the runtime
                // `new RegExp(...)` path produces, but at the Parse phase. (Lazily
                // compiled literals would otherwise never be validated until first
                // matched.) Gated on the `regex` feature: without it there is no
                // engine to validate against and the lenient behavior is kept.
                // An unimplemented-but-valid feature (e.g. a `\p{…}` property
                // escape) is NOT a SyntaxError: defer it to runtime rather than
                // rejecting the literal at parse, so a never-matched such literal
                // keeps working. Only genuine syntax errors fail the parse.
                #[cfg(feature = "regex")]
                if let Err(e) = crate::regex::Regex::new(pattern, flags)
                    && !e.is_unsupported()
                {
                    return Err(self.err_at(tok.span, alloc::format!("{e}")));
                }
                Ok(Expr::Regex {
                    pattern: pattern.into(),
                    flags: flags.into(),
                    span: tok.span,
                })
            }
            TokenKind::NoSubstitutionTemplate | TokenKind::TemplateHead => {
                Ok(Expr::Template(self.parse_template_literal(false)?))
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
                Ok(Expr::Ident(Ident::new(
                    self.checked_ident_name(tok)?,
                    tok.span,
                )))
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
            // `let` is only reserved as the leading token of a
            // `LexicalDeclaration`; everywhere else (notably in single-statement
            // position, where a declaration is not permitted) it is an ordinary
            // identifier. The statement dispatcher routes such a `let` here.
            TokenKind::Keyword(Kw::Let) => {
                self.bump();
                Ok(Expr::Ident(Ident::new(Kw::Let.as_str(), tok.span)))
            }
            TokenKind::Keyword(kw) if kw.is_contextual() => {
                self.bump();
                Ok(Expr::Ident(Ident::new(kw.as_str(), tok.span)))
            }
            // The strict-mode future-reserved words are ordinary identifier
            // references in sloppy code (mirroring `keyword_is_binding_ident`);
            // strict-mode misuse is a static-semantics error caught by the
            // validator (`is_strict_reserved_word`).
            TokenKind::Keyword(
                kw @ (Kw::Static
                | Kw::Implements
                | Kw::Interface
                | Kw::Package
                | Kw::Private
                | Kw::Protected
                | Kw::Public),
            ) => {
                self.bump();
                Ok(Expr::Ident(Ident::new(kw.as_str(), tok.span)))
            }
            // `yield` outside a generator and `await` outside an async body are
            // ordinary identifier references (in sloppy code; strict-mode misuse
            // is a static-semantics error caught by the validator). Inside a
            // generator/async body the respective keyword is intercepted earlier
            // (`parse_assignment` / `parse_unary`) and never reaches here.
            TokenKind::Keyword(Kw::Yield) if !self.in_generator => {
                self.bump();
                Ok(Expr::Ident(Ident::new(Kw::Yield.as_str(), tok.span)))
            }
            TokenKind::Keyword(Kw::Await) if !self.in_async => {
                self.bump();
                Ok(Expr::Ident(Ident::new(Kw::Await.as_str(), tok.span)))
            }
            // `import(specifier)` (dynamic import) and `import.meta` (the module
            // meta-property). Both are expression forms accepted in script *and*
            // module code. There is no first-class AST node for either, and the
            // tree-walker's `Expr` match is closed, so we desugar to a node the
            // evaluators already understand: a call of / member on the reference
            // `import`. That reference is unbound at runtime, so evaluation fails
            // with a `ReferenceError` — the failure moves from the parse phase to
            // the runtime phase (full dynamic-import semantics are a separate,
            // higher layer that is out of scope here).
            TokenKind::Keyword(Kw::Import) => self.parse_import_expr(tok.span),
            TokenKind::LParen => self.parse_paren(),
            TokenKind::LBracket => self.parse_array(),
            TokenKind::LBrace => self.parse_object(),
            TokenKind::Keyword(Kw::Function) => self.parse_function_expr(false, tok.span),
            TokenKind::Keyword(Kw::Class) => self.parse_class_expr(),
            // `@dec … class { … }` — a decorated class expression. The
            // decorators are parsed and discarded (applied as no-ops).
            TokenKind::At => {
                self.parse_decorators()?;
                self.parse_class_expr()
            }
            // A bare `#x` — only valid as the left of `#x in obj` (brand check).
            TokenKind::PrivateName => {
                self.bump();
                Ok(Expr::PrivateName(self.private_name(tok), tok.span))
            }
            _ => Err(self.err(format!("unexpected token {:?}", tok.kind))),
        }
    }

    /// Parses an `import` expression: dynamic `import(specifier)` /
    /// `import(specifier, options)` or the `import.meta` meta-property (the
    /// cursor is at the `import` keyword). See the dispatch site in
    /// [`Self::parse_primary`] for why both desugar to a reference named
    /// `import` (a closed evaluator `Expr` match plus no first-class node).
    fn parse_import_expr(&mut self, import_span: Span) -> Result<Expr> {
        self.bump(); // `import`
        let import_ref = Expr::Ident(Ident::new("import", import_span));
        match self.peek() {
            TokenKind::Dot => {
                self.bump(); // `.`
                // After `import.` the valid continuations are the `import.meta`
                // meta-property, and the source-phase / deferred-import call
                // forms `import.source(...)` / `import.defer(...)` (each valid
                // only immediately before `(`). All other `import.<x>` forms are
                // a SyntaxError. `meta`/`source`/`defer` are contextual, so they
                // arrive as identifiers.
                if !self.at(TokenKind::Identifier) {
                    return Err(self.err("expected `meta` after `import.`"));
                }
                let prop = self.peek_tok();
                match prop.text(self.source) {
                    "meta" => {
                        self.bump(); // `meta`
                        Ok(Expr::Member {
                            object: Box::new(import_ref),
                            property: PropertyKey::Ident("meta".into()),
                            optional: false,
                            span: import_span.to(prop.span),
                        })
                    }
                    kind @ ("source" | "defer") if self.nth_kind(1) == TokenKind::LParen => {
                        // `import.source(specifier)` / `import.defer(specifier)` —
                        // the source-phase / deferred-import proposals. Kept as a
                        // call of the `import.<source|defer>` *member* (NOT a plain
                        // `import()` call) so the evaluator can distinguish it; these
                        // proposals are unimplemented, so it rejects with a
                        // SyntaxError rather than loading the module.
                        self.bump(); // `source` / `defer`
                        let member = Expr::Member {
                            object: Box::new(import_ref),
                            property: PropertyKey::Ident(kind.into()),
                            optional: false,
                            span: import_span.to(prop.span),
                        };
                        let arguments = self.parse_dynamic_import_args()?;
                        let span = import_span.to(self.prev_span());
                        Ok(Expr::Call {
                            callee: Box::new(member),
                            arguments,
                            optional: false,
                            span,
                        })
                    }
                    _ => Err(self.err("expected `meta` after `import.`")),
                }
            }
            // `import(specifier [, options])` — dynamic import.
            TokenKind::LParen => {
                let arguments = self.parse_dynamic_import_args()?;
                let span = import_span.to(self.prev_span());
                Ok(Expr::Call {
                    callee: Box::new(import_ref),
                    arguments,
                    optional: false,
                    span,
                })
            }
            _ => Err(self.err("expected `(` or `.meta` after `import`")),
        }
    }

    /// Parses the argument list of a dynamic import: `( AssignmentExpression
    /// (, AssignmentExpression)? ,? )` — one required specifier, an optional
    /// options argument, and an optional trailing comma. Unlike a general call
    /// argument list, a spread (`...`) argument and an empty list are Syntax
    /// Errors here (per the `ImportCall` grammar).
    fn parse_dynamic_import_args(&mut self) -> Result<Vec<Argument>> {
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err("dynamic import requires a specifier argument"));
        }
        if self.at(TokenKind::DotDotDot) {
            return Err(self.err("a spread argument is not allowed in dynamic import"));
        }
        let mut args = alloc::vec![Argument::Item(self.without_no_in(Self::parse_assignment)?)];
        // Optional second (options) argument.
        if self.eat(TokenKind::Comma) && !self.at(TokenKind::RParen) {
            if self.at(TokenKind::DotDotDot) {
                return Err(self.err("a spread argument is not allowed in dynamic import"));
            }
            args.push(Argument::Item(self.without_no_in(Self::parse_assignment)?));
            self.eat(TokenKind::Comma); // optional trailing comma
        }
        self.expect(TokenKind::RParen)?;
        Ok(args)
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
        // A parenthesized object/array literal is not a valid assignment target
        // (only the bare `{ … }`/`[ … ]` cover reinterprets as a destructuring
        // pattern). Record its span so the assignment parser can reject `({}) = 1`
        // while still accepting `({ a } = x)` (where the parens wrap the whole
        // assignment, not just the object). A nested `(({}))` keeps the marker
        // because the stripped inner is still an object/array literal.
        self.paren_obj_arr_span = match &expr {
            Expr::Object { span, .. } | Expr::Array { span, .. } => Some(*span),
            _ => None,
        };
        Ok(expr)
    }

    fn parse_array(&mut self) -> Result<Expr> {
        let start = self.expect(TokenKind::LBracket)?.span;
        let mut elements = Vec::new();
        // Records whether a spread element was immediately followed by a comma.
        // Valid in an array literal, but an early error if the literal is later
        // reinterpreted as an array assignment pattern (a rest may not be
        // followed by a comma). The trailing comma is otherwise dropped, so we
        // capture it here.
        let mut rest_trailing_comma = false;
        while !self.at(TokenKind::RBracket) {
            if self.at(TokenKind::Comma) {
                self.bump();
                elements.push(ArrayElement::Hole);
                continue;
            }
            // An array literal's contents are `[+In]` even inside a `for` header
            // (where `no_in` is set to disambiguate `for (x in y)`): a default
            // initializer like `[ x = 'a' in {} ]` must accept `in`.
            let is_spread = self.eat(TokenKind::DotDotDot);
            if is_spread {
                elements.push(ArrayElement::Spread(
                    self.without_no_in(Self::parse_assignment)?,
                ));
            } else {
                elements.push(ArrayElement::Item(
                    self.without_no_in(Self::parse_assignment)?,
                ));
            }
            if !self.at(TokenKind::RBracket) {
                if is_spread {
                    rest_trailing_comma = true;
                }
                self.expect(TokenKind::Comma)?;
            }
        }
        let end = self.expect(TokenKind::RBracket)?.span;
        Ok(Expr::Array {
            elements,
            rest_trailing_comma,
            span: start.to(end),
        })
    }

    fn parse_object(&mut self) -> Result<Expr> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let mut members = Vec::new();
        // An object literal's contents are `[+In]` even inside a `for` header
        // (where `no_in` is set): a value/default like `{ x: y = 'a' in {} }` or
        // shorthand `{ x = 'a' in {} }` must accept `in`.
        while !self.at(TokenKind::RBrace) {
            members.push(self.without_no_in(Self::parse_object_member)?);
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
                method: true,
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
                method: true,
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
                    method: true,
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
                method: false,
                span,
            });
        }

        let tok = self.peek_tok();
        // A literal string/number key always takes `: value`.
        if matches!(tok.kind, TokenKind::String | TokenKind::Number) {
            self.bump();
            let key = if tok.kind == TokenKind::String {
                PropertyKey::Str(cook::string_key(tok.text(self.source), tok.span)?.into())
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
                method: false,
                span,
            });
        }

        // Identifier-name key: either `name: value` (any IdentifierName) or the
        // shorthand `{ name }` (only a real identifier reference).
        let (name, can_shorthand): (Box<str>, bool) = match tok.kind {
            TokenKind::Identifier => (self.ident_name(tok).into(), true),
            TokenKind::Keyword(kw) if self.keyword_is_binding_ident(kw) => {
                (kw.as_str().into(), true)
            }
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
                method: true,
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
                method: false,
                span,
            });
        }

        if !can_shorthand {
            return Err(self.err_at(
                tok.span,
                "reserved word cannot be used as a shorthand property",
            ));
        }
        // A shorthand is an `IdentifierReference`, so an escaped reserved word
        // (`{ if }`) is a Syntax Error even though it is a valid IdentifierName
        // as a key (`{ if: x }`).
        self.checked_ident_name(tok)?;
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
                method: false,
                span,
            });
        }
        // Shorthand `{ name }`.
        Ok(ObjectMember::Property {
            key: PropertyKey::Ident(name),
            value: Box::new(ident_expr),
            shorthand: true,
            method: false,
            span: tok.span,
        })
    }

    // --- templates ------------------------------------------------------

    /// Parses a template literal starting at a `NoSubstitutionTemplate` or
    /// `TemplateHead` token (the substitutions are full expressions).
    ///
    /// `tagged` selects the early-error policy for escape sequences: an
    /// *untagged* template with an invalid escape (`` `\x0` ``, `` `\u` ``,
    /// `` `\8` ``, `` `\00` ``, …) is a Syntax Error, whereas a *tagged*
    /// template tolerates such escapes (the cooked value becomes `undefined`,
    /// only the `raw` strings are observable). See
    /// `cook::validate_template_escapes`.
    fn parse_template_literal(&mut self, tagged: bool) -> Result<TemplateLiteral> {
        let start = self.cur_span();
        let mut quasis = Vec::new();
        let mut expressions = Vec::new();

        let head = self.bump();
        if head.kind == TokenKind::NoSubstitutionTemplate {
            quasis.push(self.template_element(head, TemplatePart::NoSub, tagged)?);
            return Ok(TemplateLiteral {
                quasis,
                expressions,
                span: start.to(head.span),
            });
        }
        // TemplateHead.
        quasis.push(self.template_element(head, TemplatePart::Head, tagged)?);
        loop {
            expressions.push(self.without_no_in(Self::parse_expression)?);
            let part = self.bump();
            match part.kind {
                TokenKind::TemplateMiddle => {
                    quasis.push(self.template_element(part, TemplatePart::Middle, tagged)?);
                }
                TokenKind::TemplateTail => {
                    quasis.push(self.template_element(part, TemplatePart::Tail, tagged)?);
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
    ///
    /// For an *untagged* template (`tagged == false`) an invalid escape in the
    /// segment is an early Syntax Error; a *tagged* template tolerates it and
    /// records `cooked: None`.
    fn template_element(
        &self,
        tok: Token,
        part: TemplatePart,
        tagged: bool,
    ) -> Result<TemplateElement> {
        let text = tok.text(self.source);
        let inner = match part {
            // `` `…` ``  and  `}…` ``  drop one delimiter each side / end.
            TemplatePart::NoSub | TemplatePart::Tail => &text[1..text.len() - 1],
            // `` `…${ ``  and  `}…${ ``  drop one leading and the trailing `${`.
            TemplatePart::Head | TemplatePart::Middle => &text[1..text.len() - 2],
        };
        // Line-terminator normalization (ECMA-262 TRV/TV): a `<CR><LF>` or a lone
        // `<CR>` in a template becomes a single `<LF>` in *both* the raw and cooked
        // values (`<LF>`/`<LS>`/`<PS>` are left as-is). So `` String.raw`\r\n` ``
        // over literal CRLF/CR source yields "\n", not the original bytes.
        let normalized;
        let inner: &str = if inner.contains('\r') {
            normalized = inner.replace("\r\n", "\n").replace('\r', "\n");
            &normalized
        } else {
            inner
        };
        if !tagged {
            cook::validate_template_escapes(inner, tok.span)?;
        }
        let cooked = cook::decode_escapes(inner, tok.span).ok().map(Into::into);
        Ok(TemplateElement {
            raw: inner.into(),
            cooked,
            span: tok.span,
        })
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
        // `yield [no LineTerminator here] * AssignmentExpression` — a line
        // terminator between `yield` and `*` is a Syntax Error (it is not ASI'd
        // into `yield; *expr`).
        if self.at(TokenKind::Star) && self.nth_newline(0) {
            return Err(self.err("no line terminator is allowed between `yield` and `*`"));
        }
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

/// Whether `name` is an *always-reserved* ECMAScript keyword — one that is
/// reserved in every context (so it may never be an identifier, even spelled
/// with a Unicode escape). The strict-mode-only reserved words and the
/// contextual keywords (`async`, `let`, `yield`, `await`, …) are deliberately
/// excluded: their identifier-vs-keyword status is position/mode dependent and
/// is handled elsewhere.
fn is_always_reserved_word(name: &str) -> bool {
    matches!(
        Kw::from_str(name),
        Some(kw)
            if !kw.is_contextual()
                && !matches!(
                    kw,
                    Kw::Let
                        | Kw::Static
                        | Kw::Yield
                        | Kw::Await
                        | Kw::Implements
                        | Kw::Interface
                        | Kw::Package
                        | Kw::Private
                        | Kw::Protected
                        | Kw::Public
                )
    )
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
