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
    /// A literal character.
    Char(char),
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

/// One item inside a character class.
pub(crate) enum ClassItem {
    Char(char),
    Range(char, char),
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
    /// A Unicode property escape `\p{…}` (or negated `\P{…}`).
    Property(PropKind, bool),
}

/// The Unicode general categories / binary properties supported by `\p{…}`,
/// matched via pure-Rust `char` methods (no Unicode tables of our own).
#[derive(Clone, Copy)]
pub(crate) enum PropKind {
    /// `L` / `Letter` / `Alphabetic`.
    Letter,
    /// `Lu` / `Uppercase`.
    Upper,
    /// `Ll` / `Lowercase`.
    Lower,
    /// `N` / `Nd` / `Number`.
    Number,
    /// `White_Space` / `space`.
    White,
    /// `Alphabetic` plus `Number` (`\w`-ish, but Unicode-aware).
    Alnum,
    /// A general category by its code: a single-letter group (`[b'L', 0]`) or a
    /// two-letter subcategory (`[b'L', b'u']`). Matched precisely via the `intl`
    /// Unicode tables when available, else by a `char`-method approximation.
    Gc([u8; 2]),
}

/// Maps a `\p{…}` property name (a 1–2 letter category code, or a long alias) to
/// its general-category code, or `None` if unrecognized.
pub(crate) fn general_category_code(name: &str) -> Option<[u8; 2]> {
    // Long-form aliases → their canonical code.
    let code = match name {
        "Mark" => "M",
        "Punctuation" => "P",
        "Symbol" => "S",
        "Separator" => "Z",
        "Titlecase_Letter" => "Lt",
        "Modifier_Letter" => "Lm",
        "Other_Letter" => "Lo",
        "Nonspacing_Mark" => "Mn",
        "Spacing_Mark" => "Mc",
        "Enclosing_Mark" => "Me",
        "Letter_Number" => "Nl",
        "Other_Number" => "No",
        "Connector_Punctuation" => "Pc",
        "Dash_Punctuation" => "Pd",
        "Open_Punctuation" => "Ps",
        "Close_Punctuation" => "Pe",
        "Initial_Punctuation" => "Pi",
        "Final_Punctuation" => "Pf",
        "Other_Punctuation" => "Po",
        "Math_Symbol" => "Sm",
        "Currency_Symbol" => "Sc",
        "Modifier_Symbol" => "Sk",
        "Other_Symbol" => "So",
        "Space_Separator" => "Zs",
        "Line_Separator" => "Zl",
        "Paragraph_Separator" => "Zp",
        "Control" | "cntrl" => "Cc",
        "Format" => "Cf",
        "Private_Use" => "Co",
        "Unassigned" => "Cn",
        other => other,
    };
    const SUBCATS: [&str; 30] = [
        "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
        "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
    ];
    let b = code.as_bytes();
    match code {
        "L" | "M" | "N" | "P" | "S" | "Z" | "C" => Some([b[0], 0]),
        _ if SUBCATS.contains(&code) => Some([b[0], b[1]]),
        _ => None,
    }
}

/// `(group index, name)` pairs for named capture groups (`(?<name>…)`).
pub(crate) type GroupNames = Vec<(usize, alloc::string::String)>;

/// Parses `pattern` into an AST plus the number of capturing groups and the
/// `(group index, name)` pairs of any named groups (`(?<name>…)`).
pub(crate) fn parse(pattern: &str) -> Result<(Node, usize, GroupNames), RegexError> {
    let mut p = Parser {
        chars: pattern.chars().collect(),
        pos: 0,
        group_count: 0,
        group_names: Vec::new(),
        depth: 0,
    };
    let node = p.parse_alt()?;
    if p.pos != p.chars.len() {
        return Err(RegexError::new(alloc::format!(
            "unexpected `{}`",
            p.chars[p.pos]
        )));
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
    fn parse_alt(&mut self) -> Result<Node, RegexError> {
        let mut branches = alloc::vec![self.parse_concat()?];
        while self.eat('|') {
            branches.push(self.parse_concat()?);
        }
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
                None => return Ok(atom), // a literal `{` not forming a quantifier
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
            Some(c) => {
                self.pos += 1;
                Ok(Node::Char(c))
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
                let mut name = alloc::string::String::new();
                while let Some(&c) = self.chars.get(self.pos) {
                    if c == '>' {
                        break;
                    }
                    name.push(c);
                    self.pos += 1;
                }
                if !self.eat('>') {
                    return Err(RegexError::new("unterminated group name `(?<`"));
                }
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
            // `\1`…`\9` — a backreference to a capture group.
            d if d.is_ascii_digit() && d != '0' => Node::Backref((d as u8 - b'0') as usize),
            // `\k<name>` — a named backreference (resolved at compile time). A bare
            // `\k` not followed by `<` is the literal character `k` (Annex B).
            'k' if self.chars.get(self.pos) == Some(&'<') => {
                self.eat('<');
                let mut name = alloc::string::String::new();
                while let Some(&c) = self.chars.get(self.pos) {
                    if c == '>' {
                        break;
                    }
                    name.push(c);
                    self.pos += 1;
                }
                if !self.eat('>') {
                    return Err(RegexError::new("unterminated `\\k<` group name"));
                }
                Node::NamedBackref(name)
            }
            'u' => Node::Char(self.parse_unicode_escape()?),
            'x' => Node::Char(self.parse_hex_escape(2)?),
            'p' => class_shorthand(Shorthand::Property(self.parse_property()?, false)),
            'P' => class_shorthand(Shorthand::Property(self.parse_property()?, true)),
            other => Node::Char(escape_char(other)),
        })
    }

    /// Parses a `\p{Name}` body (the `\p`/`\P` already consumed) into a
    /// `PropKind`. Unknown property names are rejected.
    fn parse_property(&mut self) -> Result<PropKind, RegexError> {
        if !self.eat('{') {
            return Err(RegexError::new("expected `{` after `\\p`"));
        }
        let mut name = alloc::string::String::new();
        loop {
            match self.bump() {
                Some('}') => break,
                Some(c) => name.push(c),
                None => return Err(RegexError::new("unterminated `\\p{…}`")),
            }
        }
        Ok(match name.as_str() {
            "L" | "Letter" | "Alphabetic" => PropKind::Letter,
            "Lu" | "Uppercase" | "Uppercase_Letter" => PropKind::Upper,
            "Ll" | "Lowercase" | "Lowercase_Letter" => PropKind::Lower,
            "N" | "Nd" | "Number" | "Decimal_Number" => PropKind::Number,
            "White_Space" | "space" => PropKind::White,
            "Alnum" => PropKind::Alnum,
            other => match general_category_code(other) {
                Some(code) => PropKind::Gc(code),
                None => {
                    return Err(RegexError::new(alloc::format!(
                        "unsupported \\p property `{other}`"
                    )));
                }
            },
        })
    }

    /// Parses a `\uHHHH` or `\u{H…}` escape body (the `\u` already consumed).
    fn parse_unicode_escape(&mut self) -> Result<char, RegexError> {
        let cp = if self.eat('{') {
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
            v
        } else {
            self.parse_hex_digits(4)?
        };
        char::from_u32(cp).ok_or_else(|| RegexError::new("escape is not a valid code point"))
    }

    /// Parses a `\xHH` escape body (the `\x` already consumed).
    fn parse_hex_escape(&mut self, n: usize) -> Result<char, RegexError> {
        let cp = self.parse_hex_digits(n)?;
        char::from_u32(cp).ok_or_else(|| RegexError::new("escape is not a valid code point"))
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
                        'p' => items.push(ClassItem::Shorthand(Shorthand::Property(
                            self.parse_property()?,
                            false,
                        ))),
                        'P' => items.push(ClassItem::Shorthand(Shorthand::Property(
                            self.parse_property()?,
                            true,
                        ))),
                        'u' => {
                            let ch = self.parse_unicode_escape()?;
                            self.push_class_member(&mut items, ch);
                        }
                        'x' => {
                            let ch = self.parse_hex_escape(2)?;
                            self.push_class_member(&mut items, ch);
                        }
                        other => {
                            self.push_class_member(&mut items, escape_char(other));
                        }
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    self.push_class_member(&mut items, c);
                }
            }
        }
        Ok(Node::Class { neg, items })
    }

    /// Pushes `c` into a class, forming a range if a `-` and another member
    /// follow.
    fn push_class_member(&mut self, items: &mut Vec<ClassItem>, c: char) {
        if self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']') {
            self.pos += 1; // `-`
            let hi = if self.peek() == Some('\\') {
                self.pos += 1;
                escape_char(self.bump().unwrap_or('\\'))
            } else {
                self.bump().unwrap_or(c)
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
