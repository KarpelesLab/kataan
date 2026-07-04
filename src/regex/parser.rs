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
    /// `true` when the pattern is well-formed per spec but uses a feature this
    /// engine does not implement yet (e.g. a `\p{…}` property escape) — as
    /// opposed to a genuine *SyntaxError*. Parse-time literal validation defers
    /// such patterns to runtime rather than rejecting them at parse, so an
    /// unsupported-but-valid literal that is never matched does not break.
    unsupported: bool,
}

impl RegexError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            unsupported: false,
        }
    }

    /// A "valid syntax, unimplemented feature" error (see [`Self::unsupported`]).
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            unsupported: true,
        }
    }

    /// Whether this is an unimplemented-feature error rather than a `SyntaxError`.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        self.unsupported
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
    /// A `v`-mode extended character class `[ … ]` (ClassSetExpression): a tree of
    /// set operations over code points, plus a set of string alternatives (from
    /// `\q{…}` and properties of strings). `neg` is the leading `^`.
    ClassSet { neg: bool, set: Box<ClassSetExpr> },
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
    /// An inline-modifier group `(?ims-ims:…)`: the flag deltas applied to
    /// `inner`'s matching. Each field is `Some(true)` to turn the flag on,
    /// `Some(false)` to turn it off, or `None` to inherit the enclosing value.
    Modifier {
        ignore_case: Option<bool>,
        multiline: Option<bool>,
        dotall: Option<bool>,
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

/// A `v`-mode ClassSetExpression: a tree of set operations whose leaves are
/// code-point sets (and, in unions/operands, `\q{…}` string literals). Evaluated
/// at compile time into a code-point matcher plus the set of matchable strings.
pub(crate) enum ClassSetExpr {
    /// A union of operands (the default when items are juxtaposed): a code-point
    /// matches if any operand matches; the string set is the union of operands'.
    Union(Vec<ClassSetExpr>),
    /// `A&&B&&…` — intersection: a code point must match every operand.
    Intersection(Vec<ClassSetExpr>),
    /// `A--B--…` — difference: in the first operand and in none of the rest.
    Difference(Vec<ClassSetExpr>),
    /// A leaf set of single code points / ranges / shorthands (`[a-z\d\p{L}]`).
    Items(Vec<ClassItem>),
    /// A `\q{…}` alternation of strings; each string is a sequence of code points
    /// (length 1 strings act as ordinary characters).
    Strings(Vec<Vec<u32>>),
    /// A nested negated class `[^…]` used as an operand.
    Negated(Box<ClassSetExpr>),
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
pub(crate) fn parse(
    pattern: &str,
    unicode: bool,
    unicode_sets: bool,
) -> Result<(Node, usize, GroupNames), RegexError> {
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
        unicode_sets,
        in_negated_class: false,
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
    /// Whether the `v` (unicodeSets) flag is set: enables extended character
    /// classes (set operations, nested classes, `\q{…}` strings). Implies
    /// `unicode`.
    unicode_sets: bool,
    /// Whether the current parse position is inside a negated character class
    /// (`[^…]`, propagated through nesting). A *property of strings* (`\p{…}`
    /// naming e.g. `Basic_Emoji`) may not appear in a negated class — it is an
    /// early `SyntaxError` there.
    in_negated_class: bool,
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
        // A quantifier may not follow a non-quantifiable assertion. A lookbehind
        // can never be quantified; a lookahead is quantifiable only under Annex B
        // (non-`u`). `^`, `$`, `\b`, `\B` are likewise unquantifiable under `u`.
        match &atom {
            Node::LookBehind { .. } => {
                return Err(RegexError::new(
                    "a lookbehind assertion is not quantifiable",
                ));
            }
            Node::Look { .. } if self.unicode => {
                return Err(RegexError::new(
                    "a lookahead assertion is not quantifiable in unicode mode",
                ));
            }
            Node::Start | Node::End | Node::WordBoundary { .. } if self.unicode => {
                return Err(RegexError::new("this assertion is not quantifiable"));
            }
            _ => {}
        }
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
            Some('[') if self.unicode_sets => self.parse_class_set(),
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
            // A `{` at an atom position. Under the `u` flag a bare `{` is a
            // Syntax Error outright. Without `u`, the Annex B InvalidBracedQuantifier
            // production has higher precedence than the literal-`{` extension: a
            // `{` that forms a valid braced-quantifier shape (`{n}`, `{n,}`,
            // `{n,m}`) here has nothing to repeat and is a Syntax Error; any other
            // `{` is a literal PatternCharacter (`/{a}/`, `/{/`).
            Some('{') => {
                if self.unicode {
                    return Err(RegexError::new("lone `{` is not allowed in unicode mode"));
                }
                match self.try_parse_brace()? {
                    Some(_) => Err(RegexError::new("nothing to repeat before `{`")),
                    None => {
                        self.pos += 1;
                        Ok(Node::Char(u32::from('{')))
                    }
                }
            }
            // Under the `u` flag a bare `}` or `]` is a Syntax Error (the strict
            // grammar admits them only when escaped or as class delimiters).
            // Without `u`, Annex B treats them as literal PatternCharacters.
            Some(c @ ('}' | ']')) if self.unicode => Err(RegexError::new(alloc::format!(
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
            } else if matches!(self.chars.get(self.pos + 1), Some('i' | 'm' | 's' | '-')) {
                // `(?ims-ims: … )` — an inline-modifier group (ES2025). The flag
                // letters before the optional `-` are turned on, those after it
                // turned off, for the enclosed `Disjunction` only.
                return self.parse_modifier_group();
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
                // After `(?`, the only valid continuations are the forms handled
                // above (`(?:`, `(?=`, `(?!`, `(?<=`, `(?<!`, `(?<name>`, and an
                // inline-modifier group `(?ims-ims:…)`). Anything else — e.g. an
                // unknown flag letter such as `(?g:a)`, `(?d:a)`, `(?I:a)`, or a
                // non-flag code point — is a genuine Syntax Error per the grammar
                // `Atom :: ( ? RegularExpressionFlags (- RegularExpressionFlags)? :
                // Disjunction )`, not an unimplemented-but-valid feature; raise a
                // hard error so a regex *literal* is rejected at parse phase.
                return Err(RegexError::new("invalid group extension `(?…)`"));
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

    /// Parses an inline-modifier group `(?ims-ims:…)` (the `(` already consumed,
    /// `self.pos` pointing at the `?`). The grammar is
    /// `( ? AddModifiers ( - RemoveModifiers )? : Disjunction )` where each
    /// modifier set is a sequence of distinct flags from `{i, m, s}`.
    ///
    /// Syntax errors (each a `RegexError` → the host raises `SyntaxError`):
    /// * an unknown flag letter (only `i`, `m`, `s` are permitted),
    /// * a repeated flag within the add set or within the remove set,
    /// * a flag appearing in *both* the add and remove sets,
    /// * both sets empty (`(?-:…)` or `(?:…)`-via-this-path — the latter never
    ///   reaches here since `(?:` is handled earlier),
    /// * a `-` with no flags following it (`(?i-:…)`),
    /// * a missing `:` terminator (`(?ms-i)`, `(?i-)`).
    fn parse_modifier_group(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `?`
        // Parse a run of distinct flag letters until `-`, `:`, or end.
        let read_flags = |p: &mut Self| -> Result<(bool, bool, bool), RegexError> {
            let (mut i, mut m, mut s) = (false, false, false);
            loop {
                match p.peek() {
                    Some('i') => {
                        if i {
                            return Err(RegexError::new("duplicate modifier flag `i`"));
                        }
                        i = true;
                        p.pos += 1;
                    }
                    Some('m') => {
                        if m {
                            return Err(RegexError::new("duplicate modifier flag `m`"));
                        }
                        m = true;
                        p.pos += 1;
                    }
                    Some('s') => {
                        if s {
                            return Err(RegexError::new("duplicate modifier flag `s`"));
                        }
                        s = true;
                        p.pos += 1;
                    }
                    Some('-') | Some(':') => break,
                    _ => return Err(RegexError::new("invalid inline modifier")),
                }
            }
            Ok((i, m, s))
        };
        let (add_i, add_m, add_s) = read_flags(self)?;
        let had_dash = self.eat('-');
        let (rem_i, rem_m, rem_s) = if had_dash {
            read_flags(self)?
        } else {
            (false, false, false)
        };
        if !self.eat(':') {
            return Err(RegexError::new("expected `:` in inline modifier group"));
        }
        // RemoveFlags after `-` MAY be empty: `(?i-:a)` (add `i`, remove nothing) is
        // valid. Only the all-empty form `(?-:a)` is rejected, by the "empty inline
        // modifier group" check below.
        // A flag may not be both added and removed.
        if (add_i && rem_i) || (add_m && rem_m) || (add_s && rem_s) {
            return Err(RegexError::new("modifier flag both added and removed"));
        }
        // At least one modifier must be present overall (the bare `(?:…)` form is a
        // plain non-capturing group, handled earlier; reaching here with no flags
        // and no `-` would be `(?:` which never routes to this function).
        if !(add_i || add_m || add_s || rem_i || rem_m || rem_s) {
            return Err(RegexError::new("empty inline modifier group"));
        }
        let merge = |add: bool, rem: bool| -> Option<bool> {
            if add {
                Some(true)
            } else if rem {
                Some(false)
            } else {
                None
            }
        };
        let inner = self.parse_alt()?;
        if !self.eat(')') {
            return Err(RegexError::new("unterminated inline modifier group"));
        }
        Ok(Node::Modifier {
            ignore_case: merge(add_i, rem_i),
            multiline: merge(add_m, rem_m),
            dotall: merge(add_s, rem_s),
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
                class_shorthand(Shorthand::Property(self.parse_property(false)?, false))
            }
            'P' if self.unicode => {
                class_shorthand(Shorthand::Property(self.parse_property(true)?, true))
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
    fn parse_property(&mut self, negated: bool) -> Result<super::props::PropEscape, RegexError> {
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
        let is_lone = value.is_none();
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
        let v_mode = self.unicode_sets;
        let in_neg_class = self.in_negated_class;
        resolved.ok_or_else(|| {
            // A lone `\p{…}` naming a *property of strings* (e.g. `Basic_Emoji`)
            // is valid only in `v` mode and only un-negated: negating it (`\P`),
            // placing it in a negated `[^…]` class, or using it without the `v`
            // flag (e.g. under `u`) is an early `SyntaxError`. In a valid position
            // it is well-formed but matching is unimplemented here, so defer it to
            // runtime rather than reject the literal at parse (a never-matched such
            // literal keeps working). Every other unresolved escape — a bogus name,
            // an ES-unsupported Unicode property, a binary property given a value,
            // a non-binary property used without one, loose matching — is a genuine
            // `SyntaxError`.
            if is_lone && super::props::is_property_of_strings(&name) {
                if negated || in_neg_class || !v_mode {
                    RegexError::new(
                        "a property of strings is only valid, un-negated, with the `v` flag",
                    )
                } else {
                    RegexError::unsupported("unsupported `\\p{…}` property of strings")
                }
            } else {
                RegexError::new("invalid `\\p{…}` property escape")
            }
        })
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
                    // A class escape (`\d \D \w \W \s \S`, or `\p`/`\P` under `u`)
                    // is a *set*, not a single character, so it cannot be an
                    // endpoint of a range. Under the `u` flag `[\w-z]` (a class
                    // escape on the left of a `-` range) is a Syntax Error; in
                    // Annex B (non-`u`) sloppy mode the `-` is a literal dash, so
                    // only reject under `u`.
                    let is_class_escape = matches!(e, 'd' | 'D' | 'w' | 'W' | 's' | 'S')
                        || (self.unicode && matches!(e, 'p' | 'P'));
                    if self.unicode
                        && is_class_escape
                        && self.peek() == Some('-')
                        && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']')
                    {
                        return Err(RegexError::new(
                            "invalid character class range with a class-escape endpoint",
                        ));
                    }
                    match e {
                        'd' => items.push(ClassItem::Shorthand(Shorthand::Digit)),
                        'D' => items.push(ClassItem::Shorthand(Shorthand::NotDigit)),
                        'w' => items.push(ClassItem::Shorthand(Shorthand::Word)),
                        'W' => items.push(ClassItem::Shorthand(Shorthand::NotWord)),
                        's' => items.push(ClassItem::Shorthand(Shorthand::Space)),
                        'S' => items.push(ClassItem::Shorthand(Shorthand::NotSpace)),
                        // `\p`/`\P` are property escapes only under `u`; otherwise
                        // they are the literal `p`/`P` (Annex B IdentityEscape). A
                        // property class is a *set*, so a following `-X` range (the
                        // `-` is checked only after `{…}` is consumed, which the
                        // single-char `is_class_escape` peek above cannot see) is a
                        // Syntax Error: `[\p{Hex}-￿]`, `[\p{Hex}--]`.
                        'p' if self.unicode => {
                            items.push(ClassItem::Shorthand(Shorthand::Property(
                                self.parse_property(false)?,
                                false,
                            )));
                            self.reject_property_range_endpoint()?;
                        }
                        'P' if self.unicode => {
                            items.push(ClassItem::Shorthand(Shorthand::Property(
                                self.parse_property(true)?,
                                true,
                            )));
                            self.reject_property_range_endpoint()?;
                        }
                        'u' => {
                            let ch = self.parse_unicode_escape()?;
                            self.push_class_member(&mut items, ch)?;
                        }
                        'x' => {
                            let ch = self.parse_hex_escape(2)?;
                            self.push_class_member(&mut items, ch)?;
                        }
                        // Inside a character class, `\b` is a backspace (U+0008),
                        // not a word boundary (ClassEscape :: `b`).
                        'b' => {
                            self.push_class_member(&mut items, 0x08)?;
                        }
                        other => {
                            self.push_class_member(&mut items, escape_char(other) as u32)?;
                        }
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    self.push_class_member(&mut items, c as u32)?;
                }
            }
        }
        Ok(Node::Class { neg, items })
    }

    /// After a `\p{…}` / `\P{…}` property class under `u`, a following `-X`
    /// (where `X` is not the closing `]`) would make the property class a range
    /// endpoint, which is a Syntax Error — a property class is a set, not a
    /// single character. A trailing `-]` is a literal dash and is allowed.
    fn reject_property_range_endpoint(&self) -> Result<(), RegexError> {
        if self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']') {
            return Err(RegexError::new(
                "invalid character class range with a class-escape endpoint",
            ));
        }
        Ok(())
    }

    /// Pushes `c` (a scalar code point) into a class, forming a range if a `-`
    /// and another member follow. Returns `Err` for an invalid range whose high
    /// endpoint is a class escape under the `u` flag (`[a-\d]`); in Annex B
    /// (non-`u`) sloppy mode that `-` is a literal dash and never errors.
    fn push_class_member(&mut self, items: &mut Vec<ClassItem>, c: u32) -> Result<(), RegexError> {
        if self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']') {
            // Under `u`, a class escape (`\d`, `\w`, … or `\p`/`\P`) on the right
            // of a `-` range is a Syntax Error: it is a set, not a single
            // character, so it cannot serve as a range endpoint (`[a-\d]`).
            if self.unicode
                && self.chars.get(self.pos + 1) == Some(&'\\')
                && self
                    .chars
                    .get(self.pos + 2)
                    .is_some_and(|&n| matches!(n, 'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'p' | 'P'))
            {
                return Err(RegexError::new(
                    "invalid character class range with a class-escape endpoint",
                ));
            }
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
            // A character-class range whose low endpoint exceeds its high
            // endpoint (`[b-a]`, `[z-a]`) is a Syntax Error in both Annex B and
            // strict mode (NonemptyClassRanges early error). Both endpoints here
            // are single characters (a class-escape endpoint takes the literal-
            // dash path under Annex B, or already errored under `u`).
            if c > hi {
                return Err(RegexError::new(
                    "character class range out of order (low greater than high)",
                ));
            }
            items.push(ClassItem::Range(c, hi));
        } else {
            items.push(ClassItem::Char(c));
        }
        Ok(())
    }

    // --- `v`-mode extended character classes (ClassSetExpression) ---

    /// Parses a `v`-mode character class `[ … ]` into a [`Node::ClassSet`]. The
    /// body is a `ClassSetExpression`: a union of operands/ranges, an
    /// intersection (`A&&B`), or a difference (`A--B`).
    fn parse_class_set(&mut self) -> Result<Node, RegexError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            self.depth -= 1;
            return Err(RegexError::new("pattern nested too deeply"));
        }
        let r = self.parse_class_set_inner();
        self.depth -= 1;
        r
    }

    fn parse_class_set_inner(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `[`
        let neg = self.eat('^');
        // A property of strings may not appear in a negated class (propagated
        // through nesting): track it while parsing the class body.
        let saved_neg_class = self.in_negated_class;
        self.in_negated_class = saved_neg_class || neg;
        let set = self.parse_class_set_expression();
        self.in_negated_class = saved_neg_class;
        let set = set?;
        if !self.eat(']') {
            return Err(RegexError::new("unterminated character class `[`"));
        }
        Ok(Node::ClassSet {
            neg,
            set: Box::new(set),
        })
    }

    /// Parses a `ClassSetExpression` (the body between `[`…`]`, after any `^`).
    /// Decides union vs. intersection (`&&`) vs. difference (`--`) by the operator
    /// that follows the first operand.
    fn parse_class_set_expression(&mut self) -> Result<ClassSetExpr, RegexError> {
        // An empty class `[]` (or `[^]`) is a valid empty set.
        if self.peek() == Some(']') {
            return Ok(ClassSetExpr::Items(Vec::new()));
        }
        let first = self.parse_class_set_operand()?;
        match self.peek_set_operator() {
            SetOp::Intersection => {
                let mut operands = alloc::vec![first];
                while self.eat_set_operator(SetOp::Intersection)? {
                    operands.push(self.parse_class_set_operand()?);
                }
                Ok(ClassSetExpr::Intersection(operands))
            }
            SetOp::Difference => {
                let mut operands = alloc::vec![first];
                while self.eat_set_operator(SetOp::Difference)? {
                    operands.push(self.parse_class_set_operand()?);
                }
                Ok(ClassSetExpr::Difference(operands))
            }
            SetOp::Union => {
                // Union: the first operand, plus any further operands/ranges until
                // `]`. A `ClassSetRange` (`a-z`) is detected when a `-` (not a
                // `--`) separates two single-character operands.
                let mut union: Vec<ClassSetExpr> = Vec::new();
                let mut pending = Some(first);
                loop {
                    match self.peek() {
                        Some(']') | None => break,
                        // A single `-` between two characters forms a range; `--`
                        // is the difference operator (handled above) and never
                        // reaches a union, so a lone `-` here is a range dash.
                        Some('-') if self.chars.get(self.pos + 1) != Some(&'-') => {
                            // The left side must be a single code point.
                            let lo = match pending.take() {
                                Some(ClassSetExpr::Items(ref items))
                                    if matches!(items.as_slice(), [ClassItem::Char(_)]) =>
                                {
                                    if let [ClassItem::Char(c)] = items.as_slice() {
                                        *c
                                    } else {
                                        unreachable!()
                                    }
                                }
                                _ => return Err(RegexError::new("invalid class set range")),
                            };
                            self.pos += 1; // `-`
                            let hi = self.parse_class_set_char_operand()?;
                            if hi < lo {
                                return Err(RegexError::new("class set range out of order"));
                            }
                            union.push(ClassSetExpr::Items(alloc::vec![ClassItem::Range(lo, hi)]));
                        }
                        _ => {
                            if let Some(p) = pending.take() {
                                union.push(p);
                            }
                            pending = Some(self.parse_class_set_operand()?);
                        }
                    }
                }
                if let Some(p) = pending.take() {
                    union.push(p);
                }
                Ok(ClassSetExpr::Union(union))
            }
        }
    }

    /// Reports which set operator (if any) immediately follows, without consuming.
    fn peek_set_operator(&self) -> SetOp {
        if self.peek() == Some('&') && self.chars.get(self.pos + 1) == Some(&'&') {
            SetOp::Intersection
        } else if self.peek() == Some('-') && self.chars.get(self.pos + 1) == Some(&'-') {
            SetOp::Difference
        } else {
            SetOp::Union
        }
    }

    /// Consumes the given set operator if present, returning whether it was. A
    /// trailing operator with no following operand (`[a&&]`) is a Syntax Error.
    fn eat_set_operator(&mut self, op: SetOp) -> Result<bool, RegexError> {
        if self.peek_set_operator() == op {
            self.pos += 2; // `&&` or `--`
            if self.peek() == Some(']') {
                return Err(RegexError::new("missing operand after set operator"));
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Parses one `ClassSetOperand`: a nested class `[…]`, a `\q{…}` string set,
    /// a character-class escape / property escape, or a single `ClassSetCharacter`.
    fn parse_class_set_operand(&mut self) -> Result<ClassSetExpr, RegexError> {
        match self.peek() {
            Some('[') => {
                let nested = self.parse_class_set()?;
                let Node::ClassSet { neg, set } = nested else {
                    unreachable!("parse_class_set returns a ClassSet node");
                };
                Ok(if neg {
                    ClassSetExpr::Negated(set)
                } else {
                    *set
                })
            }
            Some('\\') => self.parse_class_set_escape(),
            _ => {
                let c = self.parse_class_set_char_operand()?;
                Ok(ClassSetExpr::Items(alloc::vec![ClassItem::Char(c)]))
            }
        }
    }

    /// Parses a backslash operand inside a `v`-mode class: a shorthand
    /// (`\d \w \s` …), a property escape (`\p`/`\P`), a `\q{…}` string set, or a
    /// character escape (`\u`, `\x`, control, identity). Returns the operand as a
    /// [`ClassSetExpr`].
    fn parse_class_set_escape(&mut self) -> Result<ClassSetExpr, RegexError> {
        self.pos += 1; // `\`
        let Some(e) = self.peek() else {
            return Err(RegexError::new("trailing backslash in class"));
        };
        match e {
            'q' => {
                self.pos += 1; // `q`
                self.parse_q_strings()
            }
            'd' | 'D' | 'w' | 'W' | 's' | 'S' => {
                self.pos += 1;
                let sh = match e {
                    'd' => Shorthand::Digit,
                    'D' => Shorthand::NotDigit,
                    'w' => Shorthand::Word,
                    'W' => Shorthand::NotWord,
                    's' => Shorthand::Space,
                    _ => Shorthand::NotSpace,
                };
                Ok(ClassSetExpr::Items(alloc::vec![ClassItem::Shorthand(sh)]))
            }
            'p' | 'P' => {
                self.pos += 1; // `p`/`P`
                let prop = self.parse_property(e == 'P')?;
                Ok(ClassSetExpr::Items(alloc::vec![ClassItem::Shorthand(
                    Shorthand::Property(prop, e == 'P'),
                )]))
            }
            _ => {
                // A single-character class escape (`\u`, `\x`, `\n`, identity, …).
                let c = self.parse_class_set_char_after_backslash()?;
                Ok(ClassSetExpr::Items(alloc::vec![ClassItem::Char(c)]))
            }
        }
    }

    /// Parses a `\q{…}` string-set body (the `\q` already consumed): a `|`
    /// alternation of `ClassStringDisjunction` strings, each a (possibly empty)
    /// sequence of `ClassSetCharacter`s. Returns a [`ClassSetExpr::Strings`].
    fn parse_q_strings(&mut self) -> Result<ClassSetExpr, RegexError> {
        if !self.eat('{') {
            return Err(RegexError::new("expected `{` after `\\q`"));
        }
        let mut strings: Vec<Vec<u32>> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        loop {
            match self.peek() {
                None => return Err(RegexError::new("unterminated `\\q{…}`")),
                Some('}') => {
                    self.pos += 1;
                    strings.push(core::mem::take(&mut cur));
                    break;
                }
                Some('|') => {
                    self.pos += 1;
                    strings.push(core::mem::take(&mut cur));
                }
                Some('\\') => {
                    let c = self.parse_class_set_char_after_backslash_consuming()?;
                    cur.push(c);
                }
                Some(_) => {
                    let c = self.parse_class_set_char_operand()?;
                    cur.push(c);
                }
            }
        }
        Ok(ClassSetExpr::Strings(strings))
    }

    /// Parses one `ClassSetCharacter`: a single (non-escaped) source character or
    /// an escaped character. In `v`-mode the ClassSetSyntaxCharacters
    /// (`( ) [ ] { } / - \ |`) must be escaped, and the doubled punctuators
    /// (`&&`, `!!`, …) are reserved; a bare one of these is a Syntax Error.
    fn parse_class_set_char_operand(&mut self) -> Result<u32, RegexError> {
        match self.peek() {
            Some('\\') => {
                self.pos += 1;
                self.parse_class_set_char_after_backslash()
            }
            Some(c) => {
                // Reserved/`ClassSetSyntaxCharacter`s may not appear unescaped.
                if matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '/' | '|') {
                    return Err(RegexError::new("unescaped reserved character in class set"));
                }
                // A reserved double punctuator (`&&`, `!!`, `##`, …) unescaped is
                // also forbidden; a single one of these chars is a literal.
                if is_class_set_reserved_double(c) && self.chars.get(self.pos + 1) == Some(&c) {
                    return Err(RegexError::new("reserved double punctuator in class set"));
                }
                self.pos += 1;
                Ok(c as u32)
            }
            None => Err(RegexError::new("unterminated character class `[`")),
        }
    }

    /// Parses a character escape body inside a `v`-mode class, the `\` already
    /// consumed (e.g. `u1234`, `x41`, `n`, `-`). Used for class-set characters.
    fn parse_class_set_char_after_backslash(&mut self) -> Result<u32, RegexError> {
        let Some(e) = self.bump() else {
            return Err(RegexError::new("trailing backslash in class"));
        };
        Ok(match e {
            'u' => self.parse_unicode_escape()?,
            'x' => self.parse_hex_escape(2)?,
            'f' | 'n' | 'r' | 't' | 'v' => escape_char(e) as u32,
            '0' if !self.peek().is_some_and(|c| c.is_ascii_digit()) => 0,
            'c' if self.peek().is_some_and(|c| c.is_ascii_alphabetic()) => {
                let letter = self.bump().unwrap();
                (letter.to_ascii_uppercase() as u32 - 'A' as u32) + 1
            }
            // In `v`-mode the ClassSetSyntaxCharacters and the reserved
            // punctuators may all be escaped to mean the literal character.
            'b' => 0x08, // `\b` is backspace inside a class
            other => {
                if is_class_set_identity_escape(other) {
                    other as u32
                } else {
                    return Err(RegexError::new("invalid class set escape"));
                }
            }
        })
    }

    /// Like [`parse_class_set_char_after_backslash`](Self::parse_class_set_char_after_backslash)
    /// but consumes the leading `\` itself (for `\q{…}` bodies).
    fn parse_class_set_char_after_backslash_consuming(&mut self) -> Result<u32, RegexError> {
        self.pos += 1; // `\`
        self.parse_class_set_char_after_backslash()
    }
}

/// A set operator joining `ClassSetExpression` operands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SetOp {
    Union,
    Intersection,
    Difference,
}

/// Whether `c` is one of the characters that, when doubled, forms a reserved
/// `ClassSetReservedDoublePunctuator` in `v`-mode (so an unescaped doubled run is
/// a Syntax Error). Source: ClassSetReservedDoublePunctuator production.
fn is_class_set_reserved_double(c: char) -> bool {
    matches!(
        c,
        '&' | '!'
            | '#'
            | '$'
            | '%'
            | '*'
            | '+'
            | ','
            | '.'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '^'
            | '`'
            | '~'
    )
}

/// Whether `c` may follow a backslash as a `ClassSetCharacter` identity escape in
/// `v`-mode: the ClassSetSyntaxCharacters and the reserved punctuators (so that
/// `\&`, `\-`, `\|`, … denote the literal character).
fn is_class_set_identity_escape(c: char) -> bool {
    matches!(
        c,
        '(' | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '/'
            | '-'
            | '\\'
            | '|'
            | '&'
            | '!'
            | '#'
            | '$'
            | '%'
            | '*'
            | '+'
            | ','
            | '.'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '^'
            | '`'
            | '~'
    )
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
