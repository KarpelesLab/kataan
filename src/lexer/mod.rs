//! The lexer: ECMAScript source text → a stream of [`Token`]s.
//!
//! This is a hand-written, single-pass, allocation-light tokenizer. It is the
//! first stage of the pipeline and deliberately self-contained: it depends
//! only on `core`/`alloc` and produces tokens carrying [`Span`]s into the
//! original source.
//!
//! ## What it handles
//!
//! - All ECMAScript punctuators and assignment operators, optional chaining
//!   (`?.`) and nullish coalescing (`??`/`??=`).
//! - Keywords vs identifiers (Unicode identifier start/continue via a compact
//!   classifier), private names (`#field`).
//! - Numeric literals: decimal, hex/octal/binary, exponents, `BigInt` (`n`),
//!   and numeric separators (`1_000`).
//! - String literals with the full escape grammar, including line
//!   continuations.
//! - Template literals — including nested substitutions and nested templates —
//!   via an internal brace-kind stack, so the lexer is fully self-contained
//!   (no parser feedback needed for the common cases).
//! - The regex-vs-division ambiguity, resolved with the standard
//!   previous-significant-token heuristic.
//! - Line terminators and comments, recording for each token whether a line
//!   terminator preceded it (the signal the parser needs for Automatic
//!   Semicolon Insertion).
//!
//! ## What it defers
//!
//! Full template re-lexing driven by the parser (needed only for a few
//! pathological `}`-after-substitution cases the brace-stack cannot
//! disambiguate on its own) lands with the parser in Phase B.

mod token;

#[cfg(test)]
mod tests;

pub use token::{Keyword, Token, TokenKind};

use crate::common::Span;
use crate::error::{Error, Result};
use alloc::vec::Vec;

/// Tracks why we are inside a `{ … }`, so a `}` can be disambiguated between
/// "close a block/object" and "resume a template literal after `${ … }`".
#[derive(Clone, Copy, PartialEq, Eq)]
enum BraceKind {
    /// A `{` opening a *statement block* (function/loop/`if` body, bare block,
    /// `try`/`catch`/`finally`, …). After its closing `}` the cursor is at the
    /// start of a new statement, so a following `/` begins a **regex literal**.
    Block,
    /// A `{` opening an *object literal* (or class body / destructuring pattern
    /// used as a value). After its closing `}` the cursor is in expression
    /// position following a value, so a following `/` is **division**.
    ExprObject,
    /// The `{` of a template substitution `${ … }`; its `}` resumes the
    /// template body.
    TemplateSubstitution,
}

/// A streaming ECMAScript tokenizer over a borrowed source string.
///
/// Drive it with [`Lexer::next_token`] until it yields [`TokenKind::Eof`], or
/// collect everything at once with [`Lexer::tokenize`].
pub struct Lexer<'src> {
    /// The full source text, as **WTF-8 bytes**. Program text is not necessarily
    /// valid UTF-8: `eval` of a JS string may carry lone surrogates, which must
    /// survive into a regex/string literal's value (`eval("/\uD800/").source`).
    /// Scanning is byte-wise anyway (all of ECMAScript's punctuation is ASCII),
    /// so the byte form costs nothing and loses nothing.
    bytes: &'src [u8],
    /// Current byte offset into `bytes`.
    pos: usize,
    /// The kind of the previous significant (non-trivia) token, used to
    /// resolve the `/` regex-vs-division ambiguity. `None` at start of input.
    prev_significant: Option<TokenKind>,
    /// Open-brace stack for template-substitution tracking and block-vs-object
    /// disambiguation (so a `/` after `}` can be classified).
    brace_stack: Vec<BraceKind>,
    /// Whether the most recently emitted `}` token closed a *statement block*
    /// (`BraceKind::Block`). Consulted by [`Lexer::regex_allowed`] when the
    /// previous significant token is `}`: a `}` ending a block is followed by a
    /// statement (regex position), whereas a `}` ending an object literal is
    /// followed by an operator (division position).
    last_rbrace_closed_block: bool,
    /// Stack of pending `function` contexts: one entry per `function` keyword
    /// whose body `{` has not yet been seen, recording whether that `function`
    /// occurred in *statement* position (a `FunctionDeclaration`, so its body is
    /// a block and a `/` after the closing `}` is a regex) versus *expression*
    /// position (a `FunctionExpression`, a value, so a following `/` is
    /// division). The body `{` — recognized as the first `{` preceded by the
    /// param-list `)` — pops the matching entry. Nested function expressions in
    /// parameter defaults push/pop in LIFO order with their own bodies.
    func_stmt_stack: Vec<bool>,
    /// Stack of pending `class` contexts: one entry per `class` keyword whose
    /// body `{` has not yet been seen, recording whether that `class` occurred in
    /// *statement* position (a `ClassDeclaration`, so a `/` after the closing `}`
    /// is a regex) versus *expression* position (a `ClassExpression`, a value, so
    /// a following `/` is division), together with the brace and paren nesting
    /// depths at the keyword. The body `{` is the next `{` seen at those same
    /// depths, which is what distinguishes it from a brace inside the
    /// `extends` expression (`class C extends f({a: 1}) {}` — the object literal
    /// is one paren deeper).
    class_stack: Vec<(bool, usize, u32)>,
    /// Nesting depth of `(`/`[`, tracked only to place a class body `{`.
    paren_depth: u32,
    /// The `paren_depth` of each currently-open `for ( … )` head. `of` is the
    /// `for-of` keyword — and so precedes an expression, making a following `/`
    /// a regex — only directly inside such a head; everywhere else it is an
    /// ordinary identifier and `/` is division (`instance/of/g`).
    for_head_parens: Vec<u32>,
    /// Whether the tokens seen so far end in `for` or `for await`, so the next
    /// `(` opens a `for` head.
    pending_for: bool,
    /// The `paren_depth` of each currently-open `if (`/`for (`/`while (`/`with (`
    /// head. Such a `)` is followed by a *statement*, so a `/` after it starts a
    /// regex — unlike the `)` of a call or a parenthesized expression, which
    /// leaves a value and makes `/` division.
    stmt_head_parens: Vec<u32>,
    /// Whether the tokens seen so far end in a keyword whose `(` opens one of
    /// those statement heads.
    pending_stmt_head: bool,
    /// Whether the `)` just emitted closed a statement head (see
    /// [`Lexer::stmt_head_parens`]). Consulted by [`Lexer::regex_allowed`], in
    /// the same way `last_rbrace_closed_block` disambiguates `}`.
    last_rparen_closed_head: bool,
    /// Set while scanning an identifier (or private name) that contained a
    /// `\u` escape; consumed and cleared by [`Lexer::make`].
    cur_had_escape: bool,
    /// Whether the source is being lexed as a **Module** (vs a Script). Annex B
    /// HTML-like comments (`<!--`/`-->`) are a script-only web-compat feature and
    /// are NOT recognized in module code.
    module: bool,
}

impl<'src> Lexer<'src> {
    /// Creates a lexer over `source` (Script goal — Annex B HTML comments on).
    #[must_use]
    pub fn new(source: &'src str) -> Self {
        Self::with_goal(source, false)
    }

    /// Creates a lexer over `source` for the given goal (`module` disables the
    /// script-only Annex B HTML-like comments).
    #[must_use]
    pub fn with_goal(source: &'src str, module: bool) -> Self {
        Self::with_goal_bytes(source.as_bytes(), module)
    }

    /// Creates a lexer over **WTF-8** source bytes — the lossless entry point,
    /// used when the program text comes from a JS string that may hold lone
    /// surrogates (`eval`, the `Function` constructor).
    #[must_use]
    pub fn with_goal_bytes(bytes: &'src [u8], module: bool) -> Self {
        Self {
            bytes,
            pos: 0,
            prev_significant: None,
            brace_stack: Vec::new(),
            last_rbrace_closed_block: false,
            func_stmt_stack: Vec::new(),
            class_stack: Vec::new(),
            paren_depth: 0,
            for_head_parens: Vec::new(),
            pending_for: false,
            stmt_head_parens: Vec::new(),
            pending_stmt_head: false,
            last_rparen_closed_head: false,
            cur_had_escape: false,
            module,
        }
    }

    /// The source text this lexer is scanning, as WTF-8 bytes.
    #[inline]
    #[must_use]
    pub fn source(&self) -> &'src [u8] {
        self.bytes
    }

    /// Tokenizes the entire input into a vector ending with an
    /// [`TokenKind::Eof`] token. Returns the first lexical [`Error`]
    /// encountered, if any.
    pub fn tokenize(mut self) -> Result<Vec<Token>> {
        let mut out = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            out.push(tok);
            if is_eof {
                return Ok(out);
            }
        }
    }

    /// Produces the next token, consuming any leading whitespace/comments. The
    /// returned token's [`Token::newline_before`] records whether a line
    /// terminator was skipped before it (for ASI).
    pub fn next_token(&mut self) -> Result<Token> {
        // A hashbang (`#!…`) comment is only recognized as the very first thing
        // in the source (before any whitespace), and runs to the end of the
        // line. It is otherwise lexically a line comment.
        if self.pos == 0 && self.peek() == Some(b'#') && self.peek_at(1) == Some(b'!') {
            self.skip_line_comment(); // consumes `#!` and the rest of the line
        }
        let newline_before = self.skip_trivia()?;
        let start = self.pos;

        let Some(c) = self.peek() else {
            return Ok(self.make(TokenKind::Eof, start, newline_before));
        };

        let kind = match c {
            b'{' => {
                self.advance();
                let kind = if self.brace_is_block(newline_before) {
                    BraceKind::Block
                } else {
                    BraceKind::ExprObject
                };
                self.brace_stack.push(kind);
                TokenKind::LBrace
            }
            b'}' => {
                // A `}` that closes a template substitution resumes template
                // scanning rather than emitting a plain `}`.
                if matches!(
                    self.brace_stack.last(),
                    Some(BraceKind::TemplateSubstitution)
                ) {
                    self.brace_stack.pop();
                    return self.read_template_continuation(start, newline_before);
                }
                self.advance();
                // Record whether this `}` closed a statement block, so a `/`
                // immediately after it is classified as a regex (block) rather
                // than division (object literal). An unbalanced `}` (empty stack)
                // is a syntax error the parser will report; default to non-block.
                self.last_rbrace_closed_block =
                    matches!(self.brace_stack.pop(), Some(BraceKind::Block));
                TokenKind::RBrace
            }
            b'(' => {
                self.paren_depth += 1;
                // `for (` / `for await (` opens a head in which `of` is the
                // `for-of` keyword rather than an identifier.
                if self.pending_for {
                    self.for_head_parens.push(self.paren_depth);
                }
                if self.pending_stmt_head {
                    self.stmt_head_parens.push(self.paren_depth);
                }
                self.single(TokenKind::LParen)
            }
            b')' => {
                if self.for_head_parens.last() == Some(&self.paren_depth) {
                    self.for_head_parens.pop();
                }
                self.last_rparen_closed_head =
                    if self.stmt_head_parens.last() == Some(&self.paren_depth) {
                        self.stmt_head_parens.pop();
                        true
                    } else {
                        false
                    };
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.single(TokenKind::RParen)
            }
            b'[' => {
                self.paren_depth += 1;
                self.single(TokenKind::LBracket)
            }
            b']' => {
                self.paren_depth = self.paren_depth.saturating_sub(1);
                self.single(TokenKind::RBracket)
            }
            b';' => self.single(TokenKind::Semicolon),
            b',' => self.single(TokenKind::Comma),
            b'~' => self.single(TokenKind::Tilde),
            b'@' => self.single(TokenKind::At),
            b':' => self.single(TokenKind::Colon),
            b'?' => self.read_question(),
            // `.` may begin a number (`.5`), the spread `...`, or a member `.`.
            b'.' if matches!(self.peek_at(1), Some(b'0'..=b'9')) => self.read_number()?,
            b'.' => self.read_dot(),
            b'<' => self.read_lt(),
            b'>' => self.read_gt(),
            b'=' => self.read_eq(),
            b'!' => self.read_bang(),
            b'+' => self.read_plus(),
            b'-' => self.read_minus(),
            b'*' => self.read_star(),
            b'%' => self.read_percent(),
            b'&' => self.read_amp(),
            b'|' => self.read_pipe(),
            b'^' => self.read_caret(),
            b'/' => return self.read_slash(start, newline_before),
            b'"' | b'\'' => self.read_string(c)?,
            b'`' => return self.read_template_start(start, newline_before),
            b'#' => self.read_private_name()?,
            b'0'..=b'9' => self.read_number()?,
            // A `\` here can only legally begin an identifier via a `\u` escape
            // whose code point is an identifier-start char (e.g. `a` for
            // `a`); `read_identifier_or_keyword` validates that.
            b'\\' if self.peek_at(1) == Some(b'u') => self.read_identifier_or_keyword()?,
            _ => {
                if is_identifier_start_byte(c)
                    || (c >= 0x80 && self.peek_char().is_some_and(is_identifier_start_char))
                {
                    self.read_identifier_or_keyword()?
                } else {
                    // Consume the whole (possibly multi-byte) char before
                    // reporting, so the span is correct and `advance` isn't
                    // mid-codepoint.
                    let ch = self.peek_char().unwrap_or(c as char);
                    self.advance_char(ch);
                    return Err(Error::syntax(
                        alloc::format!("unexpected character {ch:?}"),
                        Span::new(start as u32, self.pos as u32),
                    ));
                }
            }
        };

        Ok(self.make(kind, start, newline_before))
    }

    // --- trivia ---------------------------------------------------------

    /// Skips whitespace and comments. Returns whether at least one line
    /// terminator was crossed. Errors only on an unterminated block comment.
    fn skip_trivia(&mut self) -> Result<bool> {
        let mut newline = false;
        loop {
            let Some(c) = self.peek() else {
                return Ok(newline);
            };
            match c {
                // ASCII whitespace.
                b' ' | b'\t' | 0x0b | 0x0c => self.advance(),
                // Line terminators (LF, CR). CRLF counts once.
                b'\n' => {
                    newline = true;
                    self.advance();
                }
                b'\r' => {
                    newline = true;
                    self.advance();
                    if self.peek() == Some(b'\n') {
                        self.advance();
                    }
                }
                b'/' => match self.peek_at(1) {
                    Some(b'/') => self.skip_line_comment(),
                    Some(b'*') => newline |= self.skip_block_comment()?,
                    _ => return Ok(newline),
                },
                // Annex B B.1.3 HTML-like comments (script/web reality). `<!--`
                // opens a single-line comment anywhere; `-->` opens one only at a
                // *line start* — a line terminator (tracked by `newline`) or the
                // input start (no significant token yet) precedes it, so `x-->0`
                // (postfix `--` then `>`) is unaffected.
                b'<' if !self.module
                    && self.peek_at(1) == Some(b'!')
                    && self.peek_at(2) == Some(b'-')
                    && self.peek_at(3) == Some(b'-') =>
                {
                    self.advance();
                    self.advance();
                    self.advance();
                    self.advance();
                    self.skip_rest_of_line();
                }
                b'-' if !self.module
                    && (newline || self.prev_significant.is_none())
                    && self.peek_at(1) == Some(b'-')
                    && self.peek_at(2) == Some(b'>') =>
                {
                    self.advance();
                    self.advance();
                    self.advance();
                    self.skip_rest_of_line();
                }
                // Non-ASCII whitespace / line terminators (NBSP, BOM, U+2028,
                // U+2029, the Zs category…). Decode one char to classify.
                _ if c >= 0x80 => {
                    let ch = self.peek_char().expect("non-empty");
                    if is_unicode_line_terminator(ch) {
                        newline = true;
                        self.advance_char(ch);
                    } else if is_unicode_whitespace(ch) {
                        self.advance_char(ch);
                    } else {
                        return Ok(newline);
                    }
                }
                _ => return Ok(newline),
            }
        }
    }

    fn skip_line_comment(&mut self) {
        // Consume `//` then everything up to (not including) a line terminator.
        self.advance();
        self.advance();
        self.skip_rest_of_line();
    }

    /// Consumes everything up to (not including) the next line terminator — the
    /// shared body of `//` and the Annex B `<!--` / `-->` HTML-like comments.
    fn skip_rest_of_line(&mut self) {
        while let Some(c) = self.peek() {
            if c == b'\n' || c == b'\r' {
                break;
            }
            if c >= 0x80 {
                let ch = self.peek_char().expect("non-empty");
                if is_unicode_line_terminator(ch) {
                    break;
                }
                self.advance_char(ch);
            } else {
                self.advance();
            }
        }
    }

    /// Skips a `/* … */` comment. Returns whether it contained a line
    /// terminator (which, per spec, makes it act as one for ASI). An EOF reached
    /// before the closing `*/` is an unterminated comment — a Syntax Error.
    fn skip_block_comment(&mut self) -> Result<bool> {
        let start = self.pos;
        self.advance();
        self.advance();
        let mut newline = false;
        while let Some(c) = self.peek() {
            if c == b'*' && self.peek_at(1) == Some(b'/') {
                self.advance();
                self.advance();
                return Ok(newline);
            }
            if c == b'\n' || c == b'\r' {
                newline = true;
                self.advance();
            } else if c >= 0x80 {
                let ch = self.peek_char().expect("non-empty");
                if is_unicode_line_terminator(ch) {
                    newline = true;
                }
                self.advance_char(ch);
            } else {
                self.advance();
            }
        }
        Err(Error::syntax(
            "unterminated block comment",
            Span::new(start as u32, self.pos as u32),
        ))
    }

    // --- multi-character punctuators ------------------------------------

    fn read_question(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            // `?.` but only as optional chaining, not `?.5` (which is `?`
            // then `.5`). The spec carves out a digit after `?.`.
            Some(b'.') if !matches!(self.peek_at(1), Some(b'0'..=b'9')) => {
                self.advance();
                TokenKind::QuestionDot
            }
            Some(b'?') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    TokenKind::QuestionQuestionEq
                } else {
                    TokenKind::QuestionQuestion
                }
            }
            _ => TokenKind::Question,
        }
    }

    /// `.` — member access or the `...` spread. The `.5` numeric case is
    /// routed to [`Self::read_number`] by the caller before reaching here.
    fn read_dot(&mut self) -> TokenKind {
        self.advance();
        if self.peek() == Some(b'.') && self.peek_at(1) == Some(b'.') {
            self.advance();
            self.advance();
            TokenKind::DotDotDot
        } else {
            TokenKind::Dot
        }
    }

    fn read_lt(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'=') => self.single(TokenKind::LtEq),
            Some(b'<') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.single(TokenKind::ShlEq)
                } else {
                    TokenKind::Shl
                }
            }
            _ => TokenKind::Lt,
        }
    }

    fn read_gt(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'=') => self.single(TokenKind::GtEq),
            Some(b'>') => {
                self.advance();
                match self.peek() {
                    Some(b'=') => self.single(TokenKind::ShrEq),
                    Some(b'>') => {
                        self.advance();
                        if self.peek() == Some(b'=') {
                            self.single(TokenKind::UshrEq)
                        } else {
                            TokenKind::Ushr
                        }
                    }
                    _ => TokenKind::Shr,
                }
            }
            _ => TokenKind::Gt,
        }
    }

    fn read_eq(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'=') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.single(TokenKind::EqEqEq)
                } else {
                    TokenKind::EqEq
                }
            }
            Some(b'>') => self.single(TokenKind::Arrow),
            _ => TokenKind::Eq,
        }
    }

    fn read_bang(&mut self) -> TokenKind {
        self.advance();
        if self.peek() == Some(b'=') {
            self.advance();
            if self.peek() == Some(b'=') {
                self.single(TokenKind::BangEqEq)
            } else {
                TokenKind::BangEq
            }
        } else {
            TokenKind::Bang
        }
    }

    fn read_plus(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'+') => self.single(TokenKind::PlusPlus),
            Some(b'=') => self.single(TokenKind::PlusEq),
            _ => TokenKind::Plus,
        }
    }

    fn read_minus(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'-') => self.single(TokenKind::MinusMinus),
            Some(b'=') => self.single(TokenKind::MinusEq),
            _ => TokenKind::Minus,
        }
    }

    fn read_star(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'*') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.single(TokenKind::StarStarEq)
                } else {
                    TokenKind::StarStar
                }
            }
            Some(b'=') => self.single(TokenKind::StarEq),
            _ => TokenKind::Star,
        }
    }

    fn read_percent(&mut self) -> TokenKind {
        self.advance();
        if self.peek() == Some(b'=') {
            self.single(TokenKind::PercentEq)
        } else {
            TokenKind::Percent
        }
    }

    fn read_amp(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'&') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.single(TokenKind::AmpAmpEq)
                } else {
                    TokenKind::AmpAmp
                }
            }
            Some(b'=') => self.single(TokenKind::AmpEq),
            _ => TokenKind::Amp,
        }
    }

    fn read_pipe(&mut self) -> TokenKind {
        self.advance();
        match self.peek() {
            Some(b'|') => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.single(TokenKind::PipePipeEq)
                } else {
                    TokenKind::PipePipe
                }
            }
            Some(b'=') => self.single(TokenKind::PipeEq),
            _ => TokenKind::Pipe,
        }
    }

    fn read_caret(&mut self) -> TokenKind {
        self.advance();
        if self.peek() == Some(b'=') {
            self.single(TokenKind::CaretEq)
        } else {
            TokenKind::Caret
        }
    }

    /// `/` — either a comment (already handled in trivia), a division
    /// operator, or the start of a regular-expression literal, decided by the
    /// previous significant token.
    fn read_slash(&mut self, start: usize, newline_before: bool) -> Result<Token> {
        if self.regex_allowed() {
            return self.read_regex(start, newline_before);
        }
        self.advance();
        let kind = if self.peek() == Some(b'=') {
            self.single(TokenKind::SlashEq)
        } else {
            TokenKind::Slash
        };
        Ok(self.make(kind, start, newline_before))
    }

    // --- literals -------------------------------------------------------

    fn read_string(&mut self, quote: u8) -> Result<TokenKind> {
        let start = self.pos;
        self.advance(); // opening quote
        loop {
            let Some(c) = self.peek() else {
                return Err(Error::syntax(
                    "unterminated string literal",
                    Span::new(start as u32, self.pos as u32),
                ));
            };
            match c {
                _ if c == quote => {
                    self.advance();
                    return Ok(TokenKind::String);
                }
                b'\\' => {
                    self.advance();
                    self.consume_escape_tail(start)?;
                }
                b'\n' | b'\r' => {
                    return Err(Error::syntax(
                        "unterminated string literal (line terminator in string)",
                        Span::new(start as u32, self.pos as u32),
                    ));
                }
                _ => self.advance_any(),
            }
        }
    }

    /// Consumes the character(s) after a `\` inside a string/template. We do
    /// not decode the value here (that is the parser's cooked-value step); we
    /// only consume the right number of source bytes so scanning stays in
    /// sync, while validating the few escapes that have a fixed shape.
    fn consume_escape_tail(&mut self, start: usize) -> Result<()> {
        let Some(c) = self.peek() else {
            return Err(Error::syntax(
                "unterminated escape sequence",
                Span::new(start as u32, self.pos as u32),
            ));
        };
        match c {
            // Line continuation: `\` followed by a line terminator.
            b'\n' => self.advance(),
            b'\r' => {
                self.advance();
                if self.peek() == Some(b'\n') {
                    self.advance();
                }
            }
            b'x' => {
                self.advance();
                for _ in 0..2 {
                    if !self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                        return Err(Error::syntax(
                            "invalid hexadecimal escape sequence",
                            Span::new(start as u32, self.pos as u32),
                        ));
                    }
                    self.advance();
                }
            }
            b'u' => {
                self.advance();
                self.consume_unicode_escape(start)?;
            }
            _ => self.advance_any(),
        }
        Ok(())
    }

    /// Consumes the body of a `\u` escape: either `\uXXXX` or `\u{ … }`.
    fn consume_unicode_escape(&mut self, start: usize) -> Result<()> {
        if self.peek() == Some(b'{') {
            self.advance();
            let mut any = false;
            while self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                any = true;
                self.advance();
            }
            if !any || self.peek() != Some(b'}') {
                return Err(Error::syntax(
                    "invalid Unicode code-point escape",
                    Span::new(start as u32, self.pos as u32),
                ));
            }
            self.advance(); // `}`
        } else {
            for _ in 0..4 {
                if !self.peek().is_some_and(|b| b.is_ascii_hexdigit()) {
                    return Err(Error::syntax(
                        "invalid Unicode escape sequence",
                        Span::new(start as u32, self.pos as u32),
                    ));
                }
                self.advance();
            }
        }
        Ok(())
    }

    /// Scans from a backtick: emits either a [`TokenKind::NoSubstitutionTemplate`]
    /// (no `${`) or a [`TokenKind::TemplateHead`] (up to and including the first
    /// `${`), pushing a substitution marker so the matching `}` resumes here.
    fn read_template_start(&mut self, start: usize, newline_before: bool) -> Result<Token> {
        self.advance(); // opening backtick
        let kind = self.scan_template_body(start)?;
        if kind == TokenKind::TemplateHead {
            self.brace_stack.push(BraceKind::TemplateSubstitution);
        }
        Ok(self.make(kind, start, newline_before))
    }

    /// Scans from the `}` that closes a substitution: emits either a
    /// [`TokenKind::TemplateMiddle`] (another `${` follows) or a
    /// [`TokenKind::TemplateTail`] (closing backtick).
    fn read_template_continuation(&mut self, start: usize, newline_before: bool) -> Result<Token> {
        self.advance(); // the `}`
        let kind = match self.scan_template_body(start)? {
            TokenKind::NoSubstitutionTemplate => TokenKind::TemplateTail,
            TokenKind::TemplateHead => {
                self.brace_stack.push(BraceKind::TemplateSubstitution);
                TokenKind::TemplateMiddle
            }
            other => other,
        };
        Ok(self.make(kind, start, newline_before))
    }

    /// Shared template-body scanner. Assumes the introducer (backtick or `}`)
    /// has been consumed. Returns [`TokenKind::NoSubstitutionTemplate`] if it
    /// reached a closing backtick, or [`TokenKind::TemplateHead`] if it
    /// reached a `${`.
    fn scan_template_body(&mut self, start: usize) -> Result<TokenKind> {
        loop {
            let Some(c) = self.peek() else {
                return Err(Error::syntax(
                    "unterminated template literal",
                    Span::new(start as u32, self.pos as u32),
                ));
            };
            match c {
                b'`' => {
                    self.advance();
                    return Ok(TokenKind::NoSubstitutionTemplate);
                }
                b'$' if self.peek_at(1) == Some(b'{') => {
                    self.advance();
                    self.advance();
                    return Ok(TokenKind::TemplateHead);
                }
                b'\\' => {
                    self.advance();
                    // A `\` escapes the next char (incl. backtick and `$`); we
                    // just consume one unit so scanning stays in sync.
                    self.advance_any();
                }
                _ => self.advance_any(),
            }
        }
    }

    /// A regular-expression literal `/pattern/flags`. Handles character
    /// classes (`[...]`, inside which `/` is literal) and escapes.
    fn read_regex(&mut self, start: usize, newline_before: bool) -> Result<Token> {
        self.advance(); // opening `/`
        let mut in_class = false;
        loop {
            let Some(c) = self.peek() else {
                return Err(Error::syntax(
                    "unterminated regular expression literal",
                    Span::new(start as u32, self.pos as u32),
                ));
            };
            match c {
                b'\n' | b'\r' => {
                    return Err(Error::syntax(
                        "unterminated regular expression literal (line terminator)",
                        Span::new(start as u32, self.pos as u32),
                    ));
                }
                b'\\' => {
                    self.advance();
                    // A `\` must be followed by a RegularExpressionNonTerminator,
                    // i.e. any SourceCharacter but not a LineTerminator (LF, CR,
                    // U+2028 LS, U+2029 PS) — an early SyntaxError otherwise.
                    match self.peek() {
                        Some(b'\n') | Some(b'\r') => {
                            return Err(Error::syntax(
                                "unterminated regular expression literal",
                                Span::new(start as u32, self.pos as u32),
                            ));
                        }
                        Some(c)
                            if c >= 0x80
                                && self.peek_char().is_some_and(is_unicode_line_terminator) =>
                        {
                            return Err(Error::syntax(
                                "regular expression literal may not contain a line terminator",
                                Span::new(start as u32, self.pos as u32),
                            ));
                        }
                        _ => self.advance_any(),
                    }
                }
                b'[' => {
                    in_class = true;
                    self.advance();
                }
                b']' => {
                    in_class = false;
                    self.advance();
                }
                b'/' if !in_class => {
                    self.advance();
                    break;
                }
                _ => {
                    // A RegularExpressionChar is a SourceCharacter but not a
                    // LineTerminator; U+2028 (LS) and U+2029 (PS) are multibyte
                    // line terminators the ASCII branches above cannot catch.
                    if c >= 0x80 && self.peek_char().is_some_and(is_unicode_line_terminator) {
                        return Err(Error::syntax(
                            "regular expression literal may not contain a line terminator",
                            Span::new(start as u32, self.pos as u32),
                        ));
                    }
                    self.advance_any();
                }
            }
        }
        // Flags: identifier-continue characters immediately after the closing
        // slash.
        while let Some(c) = self.peek() {
            if c < 0x80 {
                if is_identifier_part_byte(c) {
                    self.advance();
                } else {
                    break;
                }
            } else {
                let ch = self.peek_char().expect("non-empty");
                if is_identifier_part_char(ch) {
                    self.advance_char(ch);
                } else {
                    break;
                }
            }
        }
        Ok(self.make(TokenKind::Regex, start, newline_before))
    }

    fn read_private_name(&mut self) -> Result<TokenKind> {
        let start = self.pos;
        self.advance(); // `#`
        // The first char of a private name follows the identifier-start rules,
        // and may itself be a `\u` escape.
        if self.peek() == Some(b'\\') {
            let cp = self.read_ident_unicode_escape(start)?;
            if !is_identifier_start_char(cp) {
                return Err(Error::syntax(
                    "escape sequence is not a valid identifier start",
                    Span::new(start as u32, self.pos as u32),
                ));
            }
            self.cur_had_escape = true;
        } else {
            match self.peek() {
                Some(c)
                    if is_identifier_start_byte(c)
                        || (c >= 0x80
                            && self.peek_char().is_some_and(is_identifier_start_char)) =>
                {
                    self.advance_any();
                }
                _ => {
                    return Err(Error::syntax(
                        "expected an identifier after `#`",
                        Span::new(start as u32, self.pos as u32),
                    ));
                }
            }
        }
        let had_escape = self.read_identifier_tail(start)?;
        if had_escape {
            self.cur_had_escape = true;
        }
        Ok(TokenKind::PrivateName)
    }

    fn read_number(&mut self) -> Result<TokenKind> {
        let start = self.pos;
        let first = self.peek().expect("called with a digit or dot");

        if first == b'0' {
            match self.peek_at(1) {
                Some(b'x' | b'X') => return self.read_radix_number(16, start),
                Some(b'o' | b'O') => return self.read_radix_number(8, start),
                Some(b'b' | b'B') => return self.read_radix_number(2, start),
                // A `0` immediately followed by a decimal digit is a
                // `LegacyOctalIntegerLiteral` (`0123`) or a
                // `NonOctalDecimalIntegerLiteral` (`08`, `09`). Neither
                // production admits a `NumericLiteralSeparator`, so `0`-prefixed
                // integers with a `_` (e.g. `0_0`, `01_0`, `08_0`) are a parse
                // error. A `0` directly followed by `_` is likewise invalid: the
                // leading `0` is already a complete `DecimalIntegerLiteral`.
                Some(b'0'..=b'9') => return self.read_legacy_zero_prefixed_number(start),
                Some(b'_') => return Err(self.sep_error(start)),
                _ => {}
            }
        }

        // Integer part (decimal), allowing numeric separators.
        if first == b'.' {
            self.advance(); // `.`
            self.read_decimal_digits()?;
            self.read_exponent()?;
            return Ok(TokenKind::Number);
        }

        self.read_decimal_digits()?;

        // BigInt suffix is only valid for an integer with no fraction/exponent.
        if self.peek() == Some(b'n') {
            self.advance();
            return Ok(TokenKind::BigInt);
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.advance();
            // Fractional digits are optional (`1.`).
            if self.peek().is_some_and(|b| b.is_ascii_digit()) {
                self.read_decimal_digits()?;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.read_exponent()?;
        }

        let _ = is_float; // both fold to TokenKind::Number for now
        self.reject_identifier_after_number(start)?;
        Ok(TokenKind::Number)
    }

    /// Reads a `0`-prefixed legacy integer literal: a `LegacyOctalIntegerLiteral`
    /// (`0` followed by octal digits, e.g. `0123`) or a
    /// `NonOctalDecimalIntegerLiteral` (a `0`-prefixed run containing an `8` or
    /// `9`, e.g. `08`, `0192`). The cursor is at the leading `0`, which is known
    /// to be followed by a decimal digit. Neither production admits a numeric
    /// separator, so an embedded `_` is a parse error. A legacy-octal literal is
    /// integer-only — no fraction, exponent, or BigInt suffix. A non-octal one is
    /// a `DecimalIntegerLiteral`, so it may carry a fraction/exponent (`08.5`,
    /// `08e2`) but still never a BigInt suffix.
    fn read_legacy_zero_prefixed_number(&mut self, start: usize) -> Result<TokenKind> {
        self.advance(); // leading `0`
        let mut has_nonoctal = false; // saw an `8` or `9`
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'7' => self.advance(),
                b'8' | b'9' => {
                    has_nonoctal = true;
                    self.advance();
                }
                // Separators are not part of the legacy productions.
                b'_' => return Err(self.sep_error(start)),
                _ => break,
            }
        }

        // A BigInt suffix is never valid on a legacy-octal / non-octal-decimal
        // literal (`00n`, `08n`, `0123n` are all errors).
        if self.peek() == Some(b'n') {
            self.advance();
            return Err(Error::syntax(
                "a BigInt literal may not have a leading zero",
                Span::new(start as u32, self.pos as u32),
            ));
        }

        // Only a non-octal-decimal literal (an `8`/`9` present) is a
        // `DecimalIntegerLiteral` and may carry a fraction or exponent; a
        // legacy-octal literal is integer-only.
        if has_nonoctal {
            if self.peek() == Some(b'.') {
                self.advance();
                if self.peek().is_some_and(|b| b.is_ascii_digit()) {
                    self.read_decimal_digits()?;
                }
            }
            self.read_exponent()?;
        }

        self.reject_identifier_after_number(start)?;
        Ok(TokenKind::Number)
    }

    fn read_radix_number(&mut self, radix: u32, start: usize) -> Result<TokenKind> {
        self.advance(); // `0`
        self.advance(); // radix marker
        let mut any = false;
        let mut last_was_sep = false;
        while let Some(c) = self.peek() {
            if c == b'_' {
                if !any || last_was_sep {
                    return Err(self.sep_error(start));
                }
                last_was_sep = true;
                self.advance();
            } else if (c as char).is_digit(radix) {
                any = true;
                last_was_sep = false;
                self.advance();
            } else {
                break;
            }
        }
        if !any || last_was_sep {
            return Err(Error::syntax(
                "missing digits in numeric literal",
                Span::new(start as u32, self.pos as u32),
            ));
        }
        if self.peek() == Some(b'n') {
            self.advance();
            return Ok(TokenKind::BigInt);
        }
        self.reject_identifier_after_number(start)?;
        Ok(TokenKind::Number)
    }

    fn read_decimal_digits(&mut self) -> Result<()> {
        let start = self.pos;
        let mut last_was_sep = false;
        let mut any = false;
        while let Some(c) = self.peek() {
            if c == b'_' {
                if !any || last_was_sep {
                    return Err(self.sep_error(start));
                }
                last_was_sep = true;
                self.advance();
            } else if c.is_ascii_digit() {
                any = true;
                last_was_sep = false;
                self.advance();
            } else {
                break;
            }
        }
        if last_was_sep {
            return Err(self.sep_error(start));
        }
        Ok(())
    }

    fn read_exponent(&mut self) -> Result<()> {
        if !matches!(self.peek(), Some(b'e' | b'E')) {
            return Ok(());
        }
        let start = self.pos;
        self.advance(); // `e`
        if matches!(self.peek(), Some(b'+' | b'-')) {
            self.advance();
        }
        if !self.peek().is_some_and(|b| b.is_ascii_digit()) {
            return Err(Error::syntax(
                "missing exponent in numeric literal",
                Span::new(start as u32, self.pos as u32),
            ));
        }
        self.read_decimal_digits()
    }

    /// Per spec, an identifier may not immediately follow a numeric literal
    /// (`3in` is an error, not `3 in`).
    fn reject_identifier_after_number(&mut self, start: usize) -> Result<()> {
        if let Some(c) = self.peek()
            && (is_identifier_start_byte(c)
                || (c >= 0x80 && self.peek_char().is_some_and(is_identifier_start_char)))
        {
            return Err(Error::syntax(
                "identifier directly after numeric literal",
                Span::new(start as u32, self.pos as u32),
            ));
        }
        Ok(())
    }

    fn sep_error(&self, start: usize) -> Error {
        Error::syntax(
            "misplaced numeric separator `_`",
            Span::new(start as u32, self.pos as u32),
        )
    }

    // --- identifiers & keywords -----------------------------------------

    fn read_identifier_or_keyword(&mut self) -> Result<TokenKind> {
        let start = self.pos;
        // The leading char: either a `\u` escape or a literal identifier-start.
        if self.peek() == Some(b'\\') {
            let cp = self.read_ident_unicode_escape(start)?;
            if !is_identifier_start_char(cp) {
                return Err(Error::syntax(
                    "escape sequence is not a valid identifier start",
                    Span::new(start as u32, self.pos as u32),
                ));
            }
            self.cur_had_escape = true;
        } else {
            // A literal start char (validated by the caller); consume it.
            self.advance_any();
        }
        let had_escape = self.read_identifier_tail(start)?;

        // A keyword spelled with any escape (e.g. `if`) is *not* the
        // keyword token — it is an ordinary `IdentifierName` whose cooked value
        // happens to match a reserved word. Position-sensitive reserved-word
        // rules are enforced by the parser/validator.
        if had_escape || self.cur_had_escape {
            self.cur_had_escape = true;
            return Ok(TokenKind::Identifier);
        }
        // An escape-free identifier is by definition all `IdentifierPart` code
        // points, none of which is a surrogate — so this is always valid UTF-8.
        let text = crate::wtf8::as_str(&self.bytes[start..self.pos]).unwrap_or("");
        Ok(match Keyword::from_str(text) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Identifier,
        })
    }

    /// Consumes identifier-continue characters (including `\u` escapes) from the
    /// current position. Returns whether any escape was seen. `name_start` is
    /// the offset of the whole identifier, used only for error spans.
    fn read_identifier_tail(&mut self, name_start: usize) -> Result<bool> {
        let mut had_escape = false;
        loop {
            match self.peek() {
                Some(b'\\') => {
                    let cp = self.read_ident_unicode_escape(name_start)?;
                    if !is_identifier_part_char(cp) {
                        return Err(Error::syntax(
                            "escape sequence is not a valid identifier part",
                            Span::new(name_start as u32, self.pos as u32),
                        ));
                    }
                    had_escape = true;
                }
                Some(c) if c < 0x80 => {
                    if is_identifier_part_byte(c) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                Some(_) => {
                    let ch = self.peek_char().expect("non-empty");
                    if is_identifier_part_char(ch) {
                        self.advance_char(ch);
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        Ok(had_escape)
    }

    /// Consumes a `\uXXXX` or `\u{…}` escape appearing inside an identifier and
    /// returns its decoded scalar value. The cursor must be at the `\`. A code
    /// point above U+10FFFF or a surrogate is rejected (surrogates are never
    /// valid identifier chars). `name_start` is used for error spans.
    fn read_ident_unicode_escape(&mut self, name_start: usize) -> Result<char> {
        debug_assert_eq!(self.peek(), Some(b'\\'));
        self.advance(); // `\`
        if self.peek() != Some(b'u') {
            return Err(Error::syntax(
                "expected a Unicode escape in identifier",
                Span::new(name_start as u32, self.pos as u32),
            ));
        }
        self.advance(); // `u`
        let value: u32 = if self.peek() == Some(b'{') {
            self.advance();
            let mut v: u32 = 0;
            let mut any = false;
            while let Some(b) = self.peek() {
                let Some(d) = (b as char).to_digit(16) else {
                    break;
                };
                any = true;
                v = v.saturating_mul(16).saturating_add(d);
                self.advance();
            }
            if !any || self.peek() != Some(b'}') {
                return Err(Error::syntax(
                    "invalid Unicode code-point escape in identifier",
                    Span::new(name_start as u32, self.pos as u32),
                ));
            }
            self.advance(); // `}`
            v
        } else {
            let mut v: u32 = 0;
            for _ in 0..4 {
                let Some(d) = self.peek().and_then(|b| (b as char).to_digit(16)) else {
                    return Err(Error::syntax(
                        "invalid Unicode escape sequence in identifier",
                        Span::new(name_start as u32, self.pos as u32),
                    ));
                };
                v = v * 16 + d;
                self.advance();
            }
            v
        };
        char::from_u32(value).ok_or_else(|| {
            Error::syntax(
                "invalid code point in identifier escape",
                Span::new(name_start as u32, self.pos as u32),
            )
        })
    }

    // --- regex-vs-division heuristic ------------------------------------

    /// Whether a `function` keyword (just about to be emitted) begins a
    /// `FunctionDeclaration` (statement position) rather than a
    /// `FunctionExpression` (expression position). Keyed on the previous
    /// significant token. Errs toward `false` (expression / division) for any
    /// ambiguous position, so the only behavior change versus the historical
    /// "always division after `}`" is for unambiguous declaration sites.
    fn function_at_statement_start(&self) -> bool {
        use TokenKind as T;
        let Some(prev) = self.prev_significant else {
            return true; // start of input
        };
        match prev {
            // A statement terminator / opener: the `function` starts a statement.
            T::Semicolon => true,
            // After a `}` that closed a statement block (`{} function f(){}`), or
            // a `{` that opened a block (`{ function f(){} }`): statement context.
            T::RBrace => self.last_rbrace_closed_block,
            T::LBrace => matches!(self.brace_stack.last(), Some(BraceKind::Block)),
            // The single-statement body of `if`/`for`/`while`/`with`/`else`/`do`
            // (Annex B sloppy `FunctionDeclaration`): `)` or these keywords.
            T::RParen => true,
            T::Keyword(Keyword::Else | Keyword::Do) => true,
            // `case x:` / `default:` / a label introduce a statement; a `:` is the
            // common spelling, but it is also a ternary/object separator, so keep
            // it ambiguous (expression) to avoid misclassifying value positions.
            // Everything else (`,`, `=`, `(`, `return`, operators, …) is an
            // expression position → a `FunctionExpression`.
            _ => false,
        }
    }

    /// Whether a `{` just consumed opens a *statement block* (vs an object
    /// literal). Mirrors the classic acorn `braceIsBlock` decision, keyed on the
    /// previous significant token. The result feeds the brace stack so a `/`
    /// after the matching `}` is classified correctly.
    ///
    /// The decision is deliberately conservative: it returns `true` (block) only
    /// for positions where a `{` unambiguously begins a statement (so a `/` after
    /// the block is read as a regex). Any expression position — `=`, `(`, `,`,
    /// `return`-value, an operator, … — yields an object literal, preserving the
    /// historical "`/` after `}` is division" behavior there. `newline_before` is
    /// the line-terminator flag of the `{` token (for the `return`⏎`{` ASI case).
    fn brace_is_block(&mut self, newline_before: bool) -> bool {
        use TokenKind as T;
        // A pending `class` whose nesting depths match: this `{` opens its body.
        if let Some(&(is_decl, braces, parens)) = self.class_stack.last()
            && braces == self.brace_stack.len()
            && parens == self.paren_depth
        {
            self.class_stack.pop();
            return is_decl;
        }
        let Some(prev) = self.prev_significant else {
            // Start of input: a `{` begins a (block) statement.
            return true;
        };
        match prev {
            // A `{` directly inside another `{ … }` is a block iff the enclosing
            // brace is itself a block (a statement list nests statements; an
            // object literal value position would have an intervening `:`/`,`).
            T::LBrace => matches!(self.brace_stack.last(), Some(BraceKind::Block)),
            // A `{` after `)` closes either a function's parameter list (its body
            // is a block for a declaration, a value for an expression) or a
            // control-flow head (`if`/`for`/`while`/`switch`/`catch`/`with` — a
            // block). The pending-`function` stack tells the two apart.
            T::RParen => self.func_stmt_stack.pop().unwrap_or(true),
            // Unambiguous statement boundaries: the next `{` opens a block.
            T::Semicolon | T::Arrow => true,
            // A `:` is ambiguous between a label colon (`L: { … }` — block), an
            // object property separator (`{ a: { … } }` — value), and a ternary
            // colon (`c ? a : { … }` — value). Only the label form opens a block,
            // and the three are not reliably distinguishable from the lexer
            // alone, so we keep the conservative object classification (a `/`
            // after the matching `}` stays division — the pre-existing behavior).
            T::Colon => false,
            // After a `}`: a block follows a block (`{}{}` — two statements), an
            // object follows an object value position.
            T::RBrace => self.last_rbrace_closed_block,
            // `return` / `throw` / `yield` / etc. that precede an expression make
            // the `{` an object literal — *unless* an ASI-triggering newline
            // intervenes after `return`/`throw`, which ends the statement so the
            // `{` starts a fresh (block) statement.
            T::Keyword(Keyword::Return | Keyword::Throw) => newline_before,
            // Keywords that introduce a statement whose body is a block.
            T::Keyword(Keyword::Else | Keyword::Do | Keyword::Try | Keyword::Finally) => true,
            // Other keywords: those that precede an expression (`typeof`, `new`,
            // `case`, …) open an object; the value-like ones (`this`, `super`,
            // literals) would be division anyway but a following `{` is an object.
            T::Keyword(kw) => !kw.before_expression(),
            // A value-producing token (identifier, literal, `)`, `]`, …) before a
            // `{` cannot legally be followed by a block in expression-free code;
            // treat as object literal (the conservative, pre-existing behavior).
            _ => false,
        }
    }

    /// Whether a `/` at the current position should begin a regex literal,
    /// based on the previous significant token. This is the standard heuristic
    /// used by hand-written JS lexers.
    fn regex_allowed(&self) -> bool {
        match self.prev_significant {
            // Start of input → expression position.
            None => true,
            Some(kind) => match kind {
                // After a value-producing token, `/` is division.
                TokenKind::Identifier
                | TokenKind::PrivateName
                | TokenKind::Number
                | TokenKind::BigInt
                | TokenKind::String
                | TokenKind::Regex
                | TokenKind::NoSubstitutionTemplate
                | TokenKind::TemplateTail
                | TokenKind::RBracket
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus => false,
                // A `)` is ambiguous the same way `}` is: closing an `if`/`for`/
                // `while`/`with` head puts the cursor at the start of that
                // statement's body (regex position), while closing a call or a
                // parenthesized expression leaves a value (division). Without
                // this, `if (x) /re/.test(s)` failed to parse.
                TokenKind::RParen => self.last_rparen_closed_head,
                // A `}` is ambiguous: closing a statement block puts the cursor
                // at the start of a new statement (regex position), while closing
                // an object literal leaves a value (division). The brace stack
                // recorded which it was when the `}` was emitted.
                TokenKind::RBrace => self.last_rbrace_closed_block,
                // `of` precedes an expression only as the `for-of` keyword, i.e.
                // directly inside a `for (` head. Elsewhere it is an ordinary
                // identifier that ends an expression, so `instance/of/g` is two
                // divisions, not a regex.
                TokenKind::Keyword(Keyword::Of) => {
                    self.for_head_parens.last() == Some(&self.paren_depth)
                }
                // The other contextual keywords (`as`, `async`, `from`, `get`,
                // `set`, `target`, `accessor`) are usable as plain identifiers and
                // never directly precede a regex, so a following `/` is division.
                TokenKind::Keyword(kw) if kw.is_contextual() => false,
                // Keywords that produce/precede a value vs. those that precede
                // an expression.
                TokenKind::Keyword(kw) => kw.before_expression(),
                // Everything else (operators, `(`, `,`, `=`, `return`, …) is an
                // expression position.
                _ => true,
            },
        }
    }

    // --- low-level cursor ------------------------------------------------

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    #[inline]
    fn peek_at(&self, n: usize) -> Option<u8> {
        self.bytes.get(self.pos + n).copied()
    }

    /// Decodes the full Unicode scalar at `pos` (for the non-ASCII paths).
    ///
    /// A stored **lone surrogate** is not a `char`; it decodes to U+FFFD, which
    /// is exactly as good here — a surrogate is neither whitespace, nor a line
    /// terminator, nor an `IdentifierStart`/`IdentifierPart`, so every predicate
    /// this feeds answers the same for both, and U+FFFD's UTF-8 length (3) equals
    /// the surrogate's WTF-8 length, so [`Lexer::advance_any`] still steps
    /// exactly one code point. The literal's *value* is taken from the raw bytes,
    /// never from this.
    #[inline]
    fn peek_char(&self) -> Option<char> {
        crate::wtf8::code_points(&self.bytes[self.pos..])
            .next()
            .map(|cp| char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER))
    }

    /// Advances one ASCII byte. Must not be called on a multi-byte lead byte.
    #[inline]
    fn advance(&mut self) {
        debug_assert!(self.bytes.get(self.pos).is_some_and(|b| *b < 0x80));
        self.pos += 1;
    }

    /// Advances over whatever is at `pos`, whether ASCII or a multi-byte char.
    #[inline]
    fn advance_any(&mut self) {
        match self.peek() {
            Some(c) if c < 0x80 => self.pos += 1,
            Some(_) => {
                let ch = self.peek_char().expect("non-empty");
                self.pos += ch.len_utf8();
            }
            None => {}
        }
    }

    /// Advances over a known decoded char.
    #[inline]
    fn advance_char(&mut self, ch: char) {
        self.pos += ch.len_utf8();
    }

    /// Consumes a single ASCII byte and returns `kind` — the common
    /// "two-character operator, second char matched" tail.
    #[inline]
    fn single(&mut self, kind: TokenKind) -> TokenKind {
        self.advance();
        kind
    }

    /// Finalizes a token spanning `[start, self.pos)`, updating the
    /// previous-significant-token state for the regex heuristic.
    fn make(&mut self, kind: TokenKind, start: usize, newline_before: bool) -> Token {
        // A `function` keyword opens a context whose body-brace classification
        // depends on whether the keyword begins a `FunctionDeclaration`
        // (statement position → body is a block, so a `/` after the closing `}`
        // is a regex) or a `FunctionExpression` (a value → a following `/` is
        // division). `function_at_statement_start` distinguishes them from the
        // previous token (computed before `prev_significant` is overwritten).
        if kind == TokenKind::Keyword(Keyword::Function) {
            let is_decl = self.function_at_statement_start();
            self.func_stmt_stack.push(is_decl);
        }
        // Likewise a `class` keyword: its body `{` opens a block for a
        // `ClassDeclaration` (so `class C {} /re/` lexes the `/` as a regex) and a
        // value for a `ClassExpression`. Recorded with the nesting depths so the
        // body brace can be told from one inside the `extends` expression.
        if kind == TokenKind::Keyword(Keyword::Class) {
            let is_decl = self.function_at_statement_start();
            self.class_stack
                .push((is_decl, self.brace_stack.len(), self.paren_depth));
        }
        if kind != TokenKind::Eof {
            // `for` (optionally followed by `await`) arms the next `(` as a
            // for-head. A `for` in member position (`a.for(…)`) is a property
            // name, not the statement keyword.
            let after_dot = matches!(
                self.prev_significant,
                Some(TokenKind::Dot | TokenKind::QuestionDot)
            );
            self.pending_for = match kind {
                TokenKind::Keyword(Keyword::For) => !after_dot,
                TokenKind::Keyword(Keyword::Await) => self.pending_for,
                _ => false,
            };
            // `if`/`for`/`while`/`with` arm the next `(` as a statement head, so
            // its `)` is followed by a statement rather than by an operator. In
            // member position (`a.while(…)`) these are property names, not
            // keywords, and the `)` closes an ordinary call.
            self.pending_stmt_head = match kind {
                TokenKind::Keyword(Keyword::If | Keyword::For | Keyword::While | Keyword::With) => {
                    !after_dot
                }
                TokenKind::Keyword(Keyword::Await) => self.pending_stmt_head,
                _ => false,
            };
            self.prev_significant = Some(kind);
        }
        let had_escape = core::mem::take(&mut self.cur_had_escape);
        Token {
            kind,
            span: Span::new(start as u32, self.pos as u32),
            newline_before,
            had_escape,
        }
    }
}

// --- identifier classification ------------------------------------------

/// ASCII identifier-start bytes (`$`, `_`, `A–Z`, `a–z`). Non-ASCII is handled
/// separately via [`is_identifier_part_char`].
#[inline]
fn is_identifier_start_byte(c: u8) -> bool {
    c == b'$' || c == b'_' || c.is_ascii_alphabetic()
}

/// ASCII identifier-continue bytes (start set plus digits).
#[inline]
fn is_identifier_part_byte(c: u8) -> bool {
    is_identifier_start_byte(c) || c.is_ascii_digit()
}

/// Whether a non-ASCII char may *start* an identifier (`ID_Start`: letters and
/// letter-numbers). With the `intl` feature this uses the Unicode property
/// tables; otherwise it falls back to a pragmatic `is_alphabetic` approximation.
#[inline]
pub(crate) fn is_identifier_start_char(ch: char) -> bool {
    if ch.is_ascii() {
        return is_identifier_start_byte(ch as u8);
    }
    if is_other_id_start(ch) {
        return true;
    }
    #[cfg(feature = "intl")]
    {
        // ECMAScript `UnicodeIDStart` is the Unicode `ID_Start` property. The
        // `intl` crate exposes the UAX #31 `XID_Start` set, which differs from a
        // bare general-category letter test in one crucial way: it excludes
        // `Pattern_Syntax` characters. For example U+2E2F VERTICAL TILDE is a
        // `Lm` (modifier letter) but is `Pattern_Syntax`, so it is NOT a valid
        // identifier character — `gc.is_letter()` wrongly accepted it.
        intl::unicode::is_xid_start(ch)
    }
    #[cfg(not(feature = "intl"))]
    {
        ch.is_alphabetic()
    }
}

/// The `Other_ID_Start` compatibility set — a small, Unicode-stability-fixed
/// list of code points that are `ID_Start` despite not being letters/Nl (so the
/// general-category test misses them). Required for spec-conformant identifiers.
#[inline]
fn is_other_id_start(ch: char) -> bool {
    matches!(
        ch,
        '\u{1885}' | '\u{1886}' | '\u{2118}' | '\u{212E}' | '\u{309B}' | '\u{309C}'
    )
}

/// The `Other_ID_Continue` compatibility set — code points that are
/// `ID_Continue` despite not falling in the marks/digits/connector categories.
/// Like `Other_ID_Start`, fixed by Unicode's identifier-stability guarantee.
#[inline]
fn is_other_id_continue(ch: char) -> bool {
    matches!(
        ch,
        '\u{00B7}' | '\u{0387}' | '\u{1369}'..='\u{1371}' | '\u{19DA}'
    )
}

/// Whether a non-ASCII char may continue an identifier (`ID_Continue`:
/// `ID_Start` plus marks, decimal digits, connector punctuation, and ZWNJ/ZWJ).
/// With the `intl` feature this uses the Unicode property tables; otherwise a
/// pragmatic `is_alphanumeric` approximation. ASCII is routed through the byte
/// classifiers.
#[inline]
fn is_identifier_part_char(ch: char) -> bool {
    if ch.is_ascii() {
        return is_identifier_part_byte(ch as u8);
    }
    if ch == '\u{200C}' || ch == '\u{200D}' {
        return true; // ZWNJ / ZWJ
    }
    // `ID_Continue` is a superset of `ID_Start`, so include the `Other_ID_Start`
    // set as well as the `Other_ID_Continue` set.
    if is_other_id_start(ch) || is_other_id_continue(ch) {
        return true;
    }
    #[cfg(feature = "intl")]
    {
        // ECMAScript `UnicodeIDContinue` is the Unicode `ID_Continue` property.
        // As with `ID_Start`, the `intl` `XID_Continue` set is the authoritative
        // UAX #31 set and excludes `Pattern_Syntax` characters that a bare
        // general-category test (letter/mark/Nd/Nl/Pc) would wrongly accept.
        intl::unicode::is_xid_continue(ch)
    }
    #[cfg(not(feature = "intl"))]
    {
        ch.is_alphanumeric()
    }
}

/// Whether a char is ECMAScript whitespace (the `WhiteSpace` production):
/// TAB, VT, FF, SP, NBSP, ZWNBSP/BOM, and the Unicode `Zs` category.
#[inline]
fn is_unicode_whitespace(ch: char) -> bool {
    matches!(ch, '\u{00A0}' | '\u{FEFF}') || ch.is_whitespace() && !is_unicode_line_terminator(ch)
}

/// Whether a char is an ECMAScript `LineTerminator`: LF, CR, LS, PS.
#[inline]
fn is_unicode_line_terminator(ch: char) -> bool {
    matches!(ch, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}
