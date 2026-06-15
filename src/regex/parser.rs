//! The regex pattern parser: source pattern → AST.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// The largest accepted explicit quantifier bound (`a{N}` / `a{N,M}`). Bounds
/// above this are rejected at parse time so a pattern like `a{99999999999}` can
/// never reach the compiler and blow up instruction count / memory (RE-3). The
/// compiler enforces a stricter *expanded-size* budget on top of this.
const MAX_QUANT: usize = crate::limits::DEFAULT_REGEX_MAX_QUANT;

/// Maximum nesting depth the parser will descend (groups / lookaround). Past this
/// the parser errors instead of recursing, so a pathological pattern such as
/// `"(".repeat(100_000)` cannot overflow the stack during parsing (RE-4). Each
/// nesting level costs several recursive frames (`parse_group` →
/// `parse_alt`/`concat`/`quantified`/`atom` → `parse_group`), so this is kept
/// well below the point that would exhaust a small (2 MiB) embedding stack while
/// remaining far above any realistic legitimate nesting.
const MAX_PARSE_DEPTH: u32 = crate::limits::DEFAULT_REGEX_MAX_PARSE_DEPTH;

/// A regex compilation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexError {
    message: String,
}

impl RegexError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid regular expression: {}", self.message)
    }
}

/// A regex AST node.
pub(crate) enum Node {
    /// Matches nothing (the empty pattern); always succeeds.
    Empty,
    /// A literal character, as a scalar code point. May be an astral value
    /// (`> 0xFFFF`) or a lone surrogate (`0xD800..=0xDFFF`, from a `\u` escape);
    /// the compiler turns it into the right code-unit sequence per the `u` flag.
    Char(u32),
    /// `.` — any character (subject to the dotall flag).
    Any,
    /// A character class `[ … ]`.
    Class { neg: bool, items: Vec<ClassItem> },
    /// `^`
    Start,
    /// `$`
    End,
    /// `\b` (or `\B` when `neg`).
    WordBoundary { neg: bool },
    /// A group; `index` is `Some` for a capturing group.
    Group {
        index: Option<usize>,
        inner: Box<Node>,
    },
    /// A lookahead `(?=…)` / `(?!…)` (`neg` for the negative form).
    Look { neg: bool, inner: Box<Node> },
    /// A lookbehind `(?<=…)` / `(?<!…)` (`neg` for the negative form).
    LookBehind { neg: bool, inner: Box<Node> },
    /// A backreference `\1`…`\9`.
    Backref(usize),
    /// A named backreference `\k<name>`, resolved to a group index at compile time
    /// (so it may reference a group declared later in the pattern).
    NamedBackref(alloc::string::String),
    /// A sequence of nodes.
    Concat(Vec<Node>),
    /// Alternation `a|b|…`.
    Alt(Vec<Node>),
    /// A quantified node.
    Repeat {
        inner: Box<Node>,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
}

/// One item inside a character class. Bounds are scalar code points (may be
/// astral or lone surrogates).
pub(crate) enum ClassItem {
    Char(u32),
    Range(u32, u32),
    Shorthand(Shorthand),
}

/// The `\d \w \s` (and negated) class shorthands.
#[derive(Clone, Copy)]
pub(crate) enum Shorthand {
    Digit,
    NotDigit,
    Word,
    NotWord,
    Space,
    NotSpace,
    /// A Unicode property escape `\p{…}` (or negated `\P{…}`). The boolean is the
    /// negation flag (`\P`).
    Property(super::props::PropEscape, bool),
}

/// `(group index, name)` pairs for named capture groups (`(?<name>…)`).
pub(crate) type GroupNames = Vec<(usize, alloc::string::String)>;

/// Parses `pattern` into an AST plus the number of capturing groups and the
/// `(group index, name)` pairs of any named groups (`(?<name>…)`).
pub(crate) fn parse(pattern: &str, unicode: bool) -> Result<(Node, usize, GroupNames), RegexError> {
    let chars: Vec<char> = pattern.chars().collect();
    // Whether the pattern contains `(?<name>…)` group syntax anywhere. Under the
    // `u` flag a `\k` is always a named backreference; without it, `\k<…>` is a
    // named backreference only when the pattern declares a named group (else it
    // is the Annex B identity escape `k`, with `<name>` matched literally).
    let has_named = unicode || scan_has_named_group(&chars);
    let mut p = Parser {
        chars,
        pos: 0,
        group_count: 0,
        group_names: Vec::new(),
        depth: 0,
        unicode,
        branch_path: Vec::new(),
        next_alt_id: 0,
        decl_paths: Vec::new(),
        named_refs: Vec::new(),
        numeric_refs: Vec::new(),
        pattern_has_named_groups: has_named,
    };
    let node = p.parse_alt()?;
    if p.pos != p.chars.len() {
        return Err(RegexError::new(alloc::format!(
            "unexpected `{}`",
            p.chars[p.pos]
        )));
    }
    // Every `\k<name>` must reference a declared group name (ES2018+). A pattern
    // with no named groups at all treats `\k` as a literal (Annex B), handled in
    // `parse_escape`, so any entry here means named-group syntax *was* present.
    for r in &p.named_refs {
        if !p.group_names.iter().any(|(_, n)| n == r) {
            return Err(RegexError::new(alloc::format!(
                "named backreference to undefined group `{r}`"
            )));
        }
    }
    // Under the `u` flag a numeric backreference must name an existing group; an
    // out-of-range `\n` is a Syntax Error (no Annex B legacy-octal fallback).
    if unicode {
        for &n in &p.numeric_refs {
            if n > p.group_count {
                return Err(RegexError::new(alloc::format!(
                    "backreference to nonexistent group \\{n}"
                )));
            }
        }
    }
    Ok((node, p.group_count, p.group_names))
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    group_count: usize,
    group_names: Vec<(usize, alloc::string::String)>,
    /// Current group/lookaround nesting depth, bounded by `MAX_PARSE_DEPTH`.
    depth: u32,
    /// Whether the `u` (unicode) flag is set: enables `\u` surrogate-pair
    /// combination so a pattern like `😀` parses as one astral char.
    unicode: bool,
    /// The current alternation path: `(alt_id, branch_index)` for each enclosing
    /// `Disjunction`. Two named groups may share a name only when they are in
    /// mutually-exclusive alternatives — i.e. some common `alt_id` selects a
    /// different branch for each (ES2025 duplicate named groups). Used to reject
    /// a duplicate within the *same* alternative (`(?<a>x)(?<a>y)`).
    branch_path: Vec<(u32, u32)>,
    /// Monotonic id handed to each `Disjunction` so distinct disjunctions never
    /// collide in `branch_path`.
    next_alt_id: u32,
    /// `(name, branch_path)` recorded at each `(?<name>…)` declaration, for the
    /// mutual-exclusion duplicate-name check.
    decl_paths: Vec<(alloc::string::String, Vec<(u32, u32)>)>,
    /// Every `\k<name>` reference, validated against declared names after parsing
    /// (a name may be referenced before it is declared).
    named_refs: Vec<alloc::string::String>,
    /// Every numeric backreference `\1`…`\n`, validated against the final group
    /// count after parsing (a backref may precede its group). Only enforced under
    /// the `u` flag; without it, an out-of-range escape is an Annex B legacy
    /// octal/literal and is left to the existing relaxed handling.
    numeric_refs: Vec<usize>,
    /// Whether the whole pattern contains named-group syntax (pre-scanned), or
    /// the `u` flag is set. Drives the `\k<…>` named-backreference vs. Annex B
    /// identity-escape decision.
    pattern_has_named_groups: bool,
}

/// Pre-scans for `(?<name>…)` group syntax (a `(?<` not immediately followed by
/// `=` or `!`, which would be a lookbehind). Determines whether `\k<…>` is a
/// named backreference (Annex B: only when a named group is present).
fn scan_has_named_group(chars: &[char]) -> bool {
    let mut i = 0;
    let mut in_class = false;
    while i < chars.len() {
        match chars[i] {
            // Skip an escaped char so an escaped paren/bracket is not misread.
            '\\' => {
                i += 2;
                continue;
            }
            '[' => in_class = true,
            ']' => in_class = false,
            '(' if !in_class
                && chars.get(i + 1) == Some(&'?')
                && chars.get(i + 2) == Some(&'<')
                && !matches!(chars.get(i + 3), Some('=') | Some('!')) =>
            {
                return true;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// Whether `a` and `b` alternation paths are mutually exclusive: some common
/// disjunction (`alt_id`) selects a different branch in each. Two named groups on
/// mutually-exclusive paths can never both participate, so they may share a name
/// (ES2025 duplicate named groups); otherwise a shared name is a Syntax Error.
fn paths_mutually_exclusive(a: &[(u32, u32)], b: &[(u32, u32)]) -> bool {
    for &(aid, abr) in a {
        if let Some(&(_, bbr)) = b.iter().find(|&&(bid, _)| bid == aid)
            && abr != bbr
        {
            return true;
        }
    }
    false
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// `alt := concat ('|' concat)*`
    ///
    /// Each disjunction gets a fresh `alt_id`; the current branch index is pushed
    /// onto `branch_path` while parsing each branch so named-group declarations
    /// can be checked for mutual exclusivity (duplicate-name rule).
    fn parse_alt(&mut self) -> Result<Node, RegexError> {
        let alt_id = self.next_alt_id;
        self.next_alt_id += 1;
        self.branch_path.push((alt_id, 0));
        let mut branches = alloc::vec![self.parse_concat()?];
        while self.eat('|') {
            if let Some(last) = self.branch_path.last_mut() {
                last.1 += 1;
            }
            branches.push(self.parse_concat()?);
        }
        self.branch_path.pop();
        Ok(if branches.len() == 1 {
            branches.pop().unwrap()
        } else {
            Node::Alt(branches)
        })
    }

    /// `concat := quantified*`
    fn parse_concat(&mut self) -> Result<Node, RegexError> {
        let mut nodes = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            nodes.push(self.parse_quantified()?);
        }
        Ok(match nodes.len() {
            0 => Node::Empty,
            1 => nodes.pop().unwrap(),
            _ => Node::Concat(nodes),
        })
    }

    /// `quantified := atom quantifier?`
    fn parse_quantified(&mut self) -> Result<Node, RegexError> {
        let atom = self.parse_atom()?;
        let (min, max) = match self.peek() {
            Some('*') => {
                self.pos += 1;
                (0, None)
            }
            Some('+') => {
                self.pos += 1;
                (1, None)
            }
            Some('?') => {
                self.pos += 1;
                (0, Some(1))
            }
            Some('{') => match self.try_parse_brace()? {
                Some(bounds) => bounds,
                // A `{` not forming a quantifier: a literal `{` under Annex B, but
                // a Syntax Error under the `u` flag (strict grammar).
                None if self.unicode => {
                    return Err(RegexError::new("lone `{` quantifier"));
                }
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        // A trailing `?` makes the quantifier lazy.
        let greedy = !self.eat('?');
        Ok(Node::Repeat {
            inner: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    /// Parses a `{n}` / `{n,}` / `{n,m}` quantifier, returning `Ok(None)` (without
    /// consuming) if it is not a well-formed bound (then `{` is a literal). An
    /// out-of-range bound (`parse_int` overflow) or an inverted range (`{5,2}`)
    /// is a hard `Err` rather than a silent fallback.
    fn try_parse_brace(&mut self) -> Result<Option<(usize, Option<usize>)>, RegexError> {
        let save = self.pos;
        self.pos += 1; // `{`
        let min = self.parse_int()?;
        let Some(min) = min else {
            self.pos = save;
            return Ok(None);
        };
        let max = if self.eat(',') {
            if self.peek() == Some('}') {
                None
            } else {
                match self.parse_int()? {
                    Some(m) => Some(m),
                    None => {
                        self.pos = save;
                        return Ok(None);
                    }
                }
            }
        } else {
            Some(min)
        };
        if !self.eat('}') {
            self.pos = save;
            return Ok(None);
        }
        // Reject an inverted range like `a{5,2}`.
        if let Some(m) = max
            && min > m
        {
            return Err(RegexError::new(
                "quantifier range out of order (min greater than max)",
            ));
        }
        Ok(Some((min, max)))
    }

    /// Parses a run of decimal digits as a quantifier bound. `Ok(None)` means no
    /// digits were present (so `{` is a literal); `Ok(Some(v))` is the value;
    /// `Err` is returned when the value exceeds `MAX_QUANT` — quantifier bounds
    /// must NOT silently saturate (a saturated giant bound would blow up the
    /// compiler / allocate enormously, RE-3), so an out-of-range bound is a hard
    /// compile error.
    fn parse_int(&mut self) -> Result<Option<usize>, RegexError> {
        let start = self.pos;
        let mut value: usize = 0;
        let mut overflow = false;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                value = value.saturating_mul(10).saturating_add(d as usize);
                if value > MAX_QUANT {
                    overflow = true;
                }
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            Ok(None)
        } else if overflow {
            Err(RegexError::new("quantifier bound too large"))
        } else {
            Ok(Some(value))
        }
    }

    fn parse_atom(&mut self) -> Result<Node, RegexError> {
        match self.peek() {
            Some('(') => self.parse_group(),
            Some('[') => self.parse_class(),
            Some('.') => {
                self.pos += 1;
                Ok(Node::Any)
            }
            Some('^') => {
                self.pos += 1;
                Ok(Node::Start)
            }
            Some('$') => {
                self.pos += 1;
                Ok(Node::End)
            }
            Some('\\') => self.parse_escape(),
            Some(c @ ('*' | '+' | '?')) => Err(RegexError::new(alloc::format!(
                "nothing to repeat before `{c}`"
            ))),
            // Under the `u` flag a bare `}`, `]`, or `{` is a Syntax Error (the
            // strict grammar admits them only when escaped or as class/quantifier
            // delimiters). Without `u`, Annex B treats them as literal
            // PatternCharacters. A bare `{` is consumed here only when it did not
            // open a quantifier (`parse_quantified` handled that case).
            Some(c @ ('}' | ']' | '{')) if self.unicode => Err(RegexError::new(alloc::format!(
                "lone `{c}` is not allowed in unicode mode"
            ))),
            Some(c) => {
                self.pos += 1;
                Ok(Node::Char(c as u32))
            }
            None => Ok(Node::Empty),
        }
    }

    /// Depth-guarded entry to group parsing: bounds nesting so deeply nested
    /// patterns (`"(".repeat(100_000)`) error out instead of overflowing the
    /// parser's own recursion (RE-4).
    fn parse_group(&mut self) -> Result<Node, RegexError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RegexError::new("pattern nested too deeply"));
        }
        let result = self.parse_group_inner();
        self.depth -= 1;
        result
    }

    fn parse_group_inner(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `(`
        let index = if self.peek() == Some('?') {
            // `(?: … )` non-capturing; `(?<name> … )` named capturing.
            if self.chars.get(self.pos + 1) == Some(&':') {
                self.pos += 2;
                None
            } else if matches!(self.chars.get(self.pos + 1), Some('=' | '!')) {
                // `(?= … )` / `(?! … )` — lookahead.
                let neg = self.chars.get(self.pos + 1) == Some(&'!');
                self.pos += 2; // `?=` or `?!`
                let inner = self.parse_alt()?;
                if !self.eat(')') {
                    return Err(RegexError::new("unterminated lookahead `(?=`"));
                }
                return Ok(Node::Look {
                    neg,
                    inner: Box::new(inner),
                });
            } else if self.chars.get(self.pos + 1) == Some(&'<')
                && matches!(self.chars.get(self.pos + 2), Some('=' | '!'))
            {
                // `(?<= … )` / `(?<! … )` — lookbehind.
                let neg = self.chars.get(self.pos + 2) == Some(&'!');
                self.pos += 3; // `?<=` or `?<!`
                let inner = self.parse_alt()?;
                if !self.eat(')') {
                    return Err(RegexError::new("unterminated lookbehind `(?<=`"));
                }
                return Ok(Node::LookBehind {
                    neg,
                    inner: Box::new(inner),
                });
            } else if self.chars.get(self.pos + 1) == Some(&'<') {
                // `(?<name> … )` — a capturing group with a name.
                self.pos += 2; // `?<`
                let name = self.parse_group_name()?;
                // Duplicate names are allowed only across mutually-exclusive
                // alternatives (ES2025); a duplicate reachable in the same
                // alternative as a prior declaration is a Syntax Error.
                for (prev, path) in &self.decl_paths {
                    if *prev == name && !paths_mutually_exclusive(path, &self.branch_path) {
                        return Err(RegexError::new(alloc::format!(
                            "duplicate capture group name `{name}`"
                        )));
                    }
                }
                self.decl_paths
                    .push((name.clone(), self.branch_path.clone()));
                self.group_count += 1;
                self.group_names.push((self.group_count, name));
                Some(self.group_count)
            } else {
                return Err(RegexError::new("unsupported group extension `(?…)`"));
            }
        } else {
            self.group_count += 1;
            Some(self.group_count)
        };
        let inner = self.parse_alt()?;
        if !self.eat(')') {
            return Err(RegexError::new("unterminated group `(`"));
        }
        Ok(Node::Group {
            index,
            inner: Box::new(inner),
        })
    }

    /// Parses a `GroupName` body (the leading `<` already consumed) up to and
    /// including the closing `>`, returning the decoded name. The name must be a
    /// non-empty `RegExpIdentifierName`: a `RegExpIdentifierStart` followed by
    /// `RegExpIdentifierPart`s, where a `\u` escape contributes its code point.
    /// An empty name, an invalid start/part character, or a missing `>` is a
    /// Syntax Error.
    fn parse_group_name(&mut self) -> Result<alloc::string::String, RegexError> {
        let mut name = alloc::string::String::new();
        let mut first = true;
        loop {
            let ch = match self.peek() {
                Some('>') => {
                    self.pos += 1;
                    break;
                }
                Some('\\') if self.chars.get(self.pos + 1) == Some(&'u') => {
                    self.pos += 2; // `\u`
                    let cp = self.parse_unicode_escape()?;
                    char::from_u32(cp)
                        .ok_or_else(|| RegexError::new("invalid code point in group name"))?
                }
                Some(c) => {
                    self.pos += 1;
                    c
                }
                None => return Err(RegexError::new("unterminated group name")),
            };
            let ok = if first {
                is_regexp_id_start(ch)
            } else {
                is_regexp_id_part(ch)
            };
            if !ok {
                return Err(RegexError::new("invalid character in capture group name"));
            }
            name.push(ch);
            first = false;
        }
        if name.is_empty() {
            return Err(RegexError::new("empty capture group name"));
        }
        Ok(name)
    }

    fn parse_escape(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `\`
        let Some(c) = self.bump() else {
            return Err(RegexError::new("trailing backslash"));
        };
        Ok(match c {
            'd' => class_shorthand(Shorthand::Digit),
            'D' => class_shorthand(Shorthand::NotDigit),
            'w' => class_shorthand(Shorthand::Word),
            'W' => class_shorthand(Shorthand::NotWord),
            's' => class_shorthand(Shorthand::Space),
            'S' => class_shorthand(Shorthand::NotSpace),
            'b' => Node::WordBoundary { neg: false },
            'B' => Node::WordBoundary { neg: true },
            // `\1`…`\n` — a (possibly multi-digit) backreference to a capture
            // group. Validated against the group count after parsing under the
            // `u` flag; without it, an out-of-range reference falls back to Annex
            // B legacy-octal/literal handling (see `decimal_escape_fallback`).
            d if d.is_ascii_digit() && d != '0' => {
                let mut n = (d as u8 - b'0') as usize;
                // Multi-digit references are read whole only under `u` (where the
                // grammar has no Annex B legacy-octal ambiguity); in non-`u` mode
                // a single digit is kept to preserve the historical behavior.
                if self.unicode {
                    while let Some(nd) = self.peek().and_then(|c| c.to_digit(10)) {
                        n = n.saturating_mul(10).saturating_add(nd as usize);
                        self.pos += 1;
                    }
                }
                self.numeric_refs.push(n);
                Node::Backref(n)
            }
            // `\k<name>` — a named backreference (resolved at compile time), but
            // only when the pattern contains named-group syntax (or the `u` flag
            // is set). Otherwise (Annex B) `\k` is the identity escape `k` and the
            // following `<name>` is matched literally. A bare `\k` not followed by
            // `<` is always the literal character `k`.
            'k' if self.pattern_has_named_groups && self.chars.get(self.pos) == Some(&'<') => {
                self.eat('<');
                let name = self.parse_group_name()?;
                self.named_refs.push(name.clone());
                Node::NamedBackref(name)
            }
            'u' => Node::Char(self.parse_unicode_escape()?),
            'x' => Node::Char(self.parse_hex_escape(2)?),
            // `\p` / `\P` are Unicode property escapes only under the `u` flag.
            // Without it they are an IdentityEscape (Annex B): the literal `p`/`P`.
            'p' if self.unicode => {
                class_shorthand(Shorthand::Property(self.parse_property()?, false))
            }
            'P' if self.unicode => {
                class_shorthand(Shorthand::Property(self.parse_property()?, true))
            }
            // `\cX` — a control escape (`X` an ASCII letter → U+0001..=U+001A).
            // Under `u`, anything else after `\c` is a Syntax Error; without it
            // Annex B treats `\c` not followed by a letter as the literal `\`
            // followed by `c` (handled by the identity fallback below).
            'c' if self.peek().is_some_and(|c| c.is_ascii_alphabetic()) => {
                let letter = self.bump().unwrap();
                Node::Char((letter.to_ascii_uppercase() as u32 - 'A' as u32) + 1)
            }
            'c' if self.unicode => {
                return Err(RegexError::new("invalid `\\c` control escape"));
            }
            // A control-letter-less `\c` (non-`u`): Annex B identity escape → `\`.
            'c' => Node::Char('\\' as u32),
            // ControlEscapes (`\f \n \r \t \v`) — valid CharacterEscapes in every
            // mode. (`\0` is handled below as a NUL escape.)
            'f' | 'n' | 'r' | 't' | 'v' => Node::Char(escape_char(c) as u32),
            // `\0` — the NUL character, provided it is not the start of a longer
            // legacy-octal/decimal sequence (a following digit would be one).
            '0' if !self.peek().is_some_and(|c| c.is_ascii_digit()) => Node::Char(0),
            // Any other escaped character is an IdentityEscape. Under `u` only the
            // SyntaxCharacters and `/` may be escaped this way; an arbitrary
            // `\letter` (e.g. `\M`) is a Syntax Error. Without `u`, Annex B allows
            // escaping any source character (the literal character).
            other => {
                if self.unicode && !is_u_identity_escape(other) {
                    return Err(RegexError::new("invalid identity escape in unicode mode"));
                }
                Node::Char(escape_char(other) as u32)
            }
        })
    }

    /// Parses a `\p{…}` body (the `\p`/`\P` already consumed) into a resolved
    /// [`PropEscape`](super::props::PropEscape). Handles both the lone-name form
    /// (`\p{Alphabetic}`, `\p{Lu}`) and the `name=value` form
    /// (`\p{Script=Greek}`). An unrecognised property/value, a malformed body, or
    /// an empty name/value is rejected with a `RegexError` so the caller raises a
    /// `SyntaxError` — the corpus's negative parse tests rely on this.
    fn parse_property(&mut self) -> Result<super::props::PropEscape, RegexError> {
        if !self.eat('{') {
            return Err(RegexError::new("expected `{` after `\\p`"));
        }
        let mut name = alloc::string::String::new();
        let mut value: Option<alloc::string::String> = None;
        loop {
            match self.bump() {
                Some('}') => break,
                // The first `=` separates the property name from its value; a
                // second `=` is part of the (then invalid) value text.
                Some('=') if value.is_none() => value = Some(alloc::string::String::new()),
                Some(c) => match &mut value {
                    Some(v) => v.push(c),
                    None => name.push(c),
                },
                None => return Err(RegexError::new("unterminated `\\p{…}`")),
            }
        }
        let resolved = match value {
            // `name=value` — both sides must be non-empty and resolve.
            Some(value) => {
                if name.is_empty() || value.is_empty() {
                    None
                } else {
                    super::props::resolve_pair(&name, &value)
                }
            }
            // Lone `name` — a binary property or a General_Category value.
            None => {
                if name.is_empty() {
                    None
                } else {
                    super::props::resolve_lone(&name)
                }
            }
        };
        resolved.ok_or_else(|| RegexError::new("unsupported `\\p{…}` property escape"))
    }

    /// Parses a `\uHHHH` or `\u{H…}` escape body (the `\u` already consumed),
    /// returning the raw scalar code point. A lone surrogate (from `\uHHHH`) is
    /// returned as-is so it can match a lone-surrogate code unit; in `u` mode a
    /// high surrogate immediately followed by `\u` low surrogate is combined into
    /// the astral code point it represents (ECMAScript surrogate-pair escapes).
    /// `\u{…}` rejects values above U+10FFFF.
    fn parse_unicode_escape(&mut self) -> Result<u32, RegexError> {
        if self.eat('{') {
            let mut v: u32 = 0;
            let mut any = false;
            while let Some(c) = self.peek() {
                if c == '}' {
                    self.pos += 1;
                    break;
                }
                let d = c
                    .to_digit(16)
                    .ok_or_else(|| RegexError::new("invalid `\\u{…}` escape"))?;
                v = v.saturating_mul(16).saturating_add(d);
                self.pos += 1;
                any = true;
            }
            if !any {
                return Err(RegexError::new("empty `\\u{}` escape"));
            }
            if v > 0x10_FFFF {
                return Err(RegexError::new("escape is not a valid code point"));
            }
            return Ok(v);
        }
        let hi = self.parse_hex_digits(4)?;
        // In `u` mode, fold a leading high surrogate together with a following
        // `\u`-escaped low surrogate into one astral code point.
        if self.unicode
            && (0xD800..=0xDBFF).contains(&hi)
            && self.chars.get(self.pos) == Some(&'\\')
            && self.chars.get(self.pos + 1) == Some(&'u')
        {
            let save = self.pos;
            self.pos += 2; // `\u`
            let lo = self.parse_hex_digits(4)?;
            if (0xDC00..=0xDFFF).contains(&lo) {
                return Ok(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00));
            }
            // Not a low surrogate: rewind and keep `hi` as a lone surrogate.
            self.pos = save;
        }
        Ok(hi)
    }

    /// Parses a `\xHH` escape body (the `\x` already consumed), returning the raw
    /// code point (always BMP, so always a valid scalar).
    fn parse_hex_escape(&mut self, n: usize) -> Result<u32, RegexError> {
        self.parse_hex_digits(n)
    }

    /// Reads exactly `n` hex digits as a code point.
    fn parse_hex_digits(&mut self, n: usize) -> Result<u32, RegexError> {
        let mut v: u32 = 0;
        for _ in 0..n {
            let c = self
                .bump()
                .ok_or_else(|| RegexError::new("incomplete hex escape"))?;
            let d = c
                .to_digit(16)
                .ok_or_else(|| RegexError::new("invalid hex digit in escape"))?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn parse_class(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `[`
        let neg = self.eat('^');
        let mut items = Vec::new();
        loop {
            match self.peek() {
                None => return Err(RegexError::new("unterminated character class `[`")),
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some('\\') => {
                    self.pos += 1;
                    let Some(e) = self.bump() else {
                        return Err(RegexError::new("trailing backslash in class"));
                    };
                    match e {
                        'd' => items.push(ClassItem::Shorthand(Shorthand::Digit)),
                        'D' => items.push(ClassItem::Shorthand(Shorthand::NotDigit)),
                        'w' => items.push(ClassItem::Shorthand(Shorthand::Word)),
                        'W' => items.push(ClassItem::Shorthand(Shorthand::NotWord)),
                        's' => items.push(ClassItem::Shorthand(Shorthand::Space)),
                        'S' => items.push(ClassItem::Shorthand(Shorthand::NotSpace)),
                        // `\p`/`\P` are property escapes only under `u`; otherwise
                        // they are the literal `p`/`P` (Annex B IdentityEscape).
                        'p' if self.unicode => items.push(ClassItem::Shorthand(
                            Shorthand::Property(self.parse_property()?, false),
                        )),
                        'P' if self.unicode => items.push(ClassItem::Shorthand(
                            Shorthand::Property(self.parse_property()?, true),
                        )),
                        'u' => {
                            let ch = self.parse_unicode_escape()?;
                            self.push_class_member(&mut items, ch);
                        }
                        'x' => {
                            let ch = self.parse_hex_escape(2)?;
                            self.push_class_member(&mut items, ch);
                        }
                        other => {
                            self.push_class_member(&mut items, escape_char(other) as u32);
                        }
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    self.push_class_member(&mut items, c as u32);
                }
            }
        }
        Ok(Node::Class { neg, items })
    }

    /// Pushes `c` (a scalar code point) into a class, forming a range if a `-`
    /// and another member follow.
    fn push_class_member(&mut self, items: &mut Vec<ClassItem>, c: u32) {
        if self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']') {
            self.pos += 1; // `-`
            let hi = if self.peek() == Some('\\') {
                self.pos += 1;
                match self.bump() {
                    Some('u') => self.parse_unicode_escape().unwrap_or(c),
                    Some('x') => self.parse_hex_escape(2).unwrap_or(c),
                    Some(e) => escape_char(e) as u32,
                    None => '\\' as u32,
                }
            } else {
                self.bump().map_or(c, |ch| ch as u32)
            };
            items.push(ClassItem::Range(c, hi));
        } else {
            items.push(ClassItem::Char(c));
        }
    }
}

fn class_shorthand(s: Shorthand) -> Node {
    Node::Class {
        neg: false,
        items: alloc::vec![ClassItem::Shorthand(s)],
    }
}

/// Whether `ch` may *start* a `RegExpIdentifierName` (a capture group name):
/// `UnicodeIDStart`, `$`, or `_`. Mirrors `IdentifierName` start.
fn is_regexp_id_start(ch: char) -> bool {
    ch == '$' || ch == '_' || crate::lexer::is_identifier_start_char(ch)
}

/// Whether `ch` may *continue* a `RegExpIdentifierName`: `UnicodeIDContinue`,
/// `$`, `_`, ZWNJ, or ZWJ.
fn is_regexp_id_part(ch: char) -> bool {
    ch == '$' || ch == '_' || ch == '\u{200C}' || ch == '\u{200D}' || is_id_continue(ch)
}

/// `ID_Continue` (letters, marks, decimal digits, connector punctuation, and
/// letter-numbers). With the `intl` feature this uses the Unicode property
/// tables; otherwise a pragmatic `is_alphanumeric` approximation. ASCII is
/// classified directly. Kept local to the regex crate so capture-group-name
/// validation does not depend on lexer internals.
fn is_id_continue(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_alphanumeric() || ch == '_';
    }
    #[cfg(feature = "intl")]
    {
        use intl::unicode::category::GeneralCategory as Gc;
        let gc = intl::unicode::general_category(ch);
        gc.is_letter()
            || gc.is_mark()
            || matches!(
                gc,
                Gc::LetterNumber | Gc::DecimalNumber | Gc::ConnectorPunctuation
            )
    }
    #[cfg(not(feature = "intl"))]
    {
        ch.is_alphanumeric()
    }
}

/// Whether `c` is a permitted `IdentityEscape` target under the `u` flag: a
/// `SyntaxCharacter` (`^ $ \ . * + ? ( ) [ ] { } |`) or `/`. Any other escaped
/// character is a Syntax Error in unicode mode (Annex B's "escape anything" rule
/// does not apply).
fn is_u_identity_escape(c: char) -> bool {
    matches!(
        c,
        '^' | '$' | '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '/'
    )
}

/// Resolves a single-character escape body to its literal character.
fn escape_char(c: char) -> char {
    match c {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        'f' => '\u{0C}',
        'v' => '\u{0B}',
        '0' => '\0',
        other => other, // `\.`, `\*`, `\\`, … are the literal character
    }
}
