//! The regex pattern parser: source pattern → AST.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

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
}

/// Parses `pattern` into an AST plus the number of capturing groups.
pub(crate) fn parse(pattern: &str) -> Result<(Node, usize), RegexError> {
    let mut p = Parser {
        chars: pattern.chars().collect(),
        pos: 0,
        group_count: 0,
    };
    let node = p.parse_alt()?;
    if p.pos != p.chars.len() {
        return Err(RegexError::new(alloc::format!(
            "unexpected `{}`",
            p.chars[p.pos]
        )));
    }
    Ok((node, p.group_count))
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    group_count: usize,
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
            Some('{') => match self.try_parse_brace() {
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

    /// Parses a `{n}` / `{n,}` / `{n,m}` quantifier, returning `None` (without
    /// consuming) if it is not a well-formed bound (then `{` is a literal).
    fn try_parse_brace(&mut self) -> Option<(usize, Option<usize>)> {
        let save = self.pos;
        self.pos += 1; // `{`
        let min = self.parse_int();
        let Some(min) = min else {
            self.pos = save;
            return None;
        };
        let max = if self.eat(',') {
            if self.peek() == Some('}') {
                None
            } else {
                match self.parse_int() {
                    Some(m) => Some(m),
                    None => {
                        self.pos = save;
                        return None;
                    }
                }
            }
        } else {
            Some(min)
        };
        if !self.eat('}') {
            self.pos = save;
            return None;
        }
        Some((min, max))
    }

    fn parse_int(&mut self) -> Option<usize> {
        let start = self.pos;
        let mut value: usize = 0;
        while let Some(c) = self.peek() {
            if let Some(d) = c.to_digit(10) {
                value = value.saturating_mul(10).saturating_add(d as usize);
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start { None } else { Some(value) }
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

    fn parse_group(&mut self) -> Result<Node, RegexError> {
        self.pos += 1; // `(`
        let index = if self.peek() == Some('?') {
            // `(?: … )` non-capturing; other `(?…)` forms are unsupported.
            if self.chars.get(self.pos + 1) == Some(&':') {
                self.pos += 2;
                None
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
            other => Node::Char(escape_char(other)),
        })
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
