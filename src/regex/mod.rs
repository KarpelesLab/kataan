//! A small, in-house regular-expression engine (no foreign code).
//!
//! The pattern is parsed to an AST, compiled to a flat instruction list, and
//! executed by a backtracking virtual machine (the classic Thompson/Pike
//! "regex VM" shape — see Russ Cox, *Regular Expression Matching: the Virtual
//! Machine Approach*). Backtracking gives JavaScript's greedy/lazy and
//! capturing-group semantics directly.
//!
//! Supported today: literals, `.`, character classes `[...]` (ranges and the
//! `\d \w \s` shorthands, negation), anchors `^ $`, word boundaries `\b \B`,
//! capturing `( )`, non-capturing `(?: )` and named `(?<name> )` groups,
//! alternation `|`, the quantifiers `* + ? {n} {n,} {n,m}` (greedy and lazy),
//! backreferences `\1`, lookahead `(?= )`/`(?! )`, lookbehind `(?<= )`/`(?<! )`,
//! and the `i` (case-insensitive), `m` (multiline), and `s` (dotall) flags.
//! Positions are Unicode scalar (`char`) indices.
//!
//! Not yet: the Unicode property escapes `\p{…}` and the `u`/`y` flag semantics —
//! these land as the engine matures (see `ROADMAP.md`).

mod compile;
mod parser;
mod vm;

#[cfg(test)]
mod tests;

use alloc::string::String;
use alloc::vec::Vec;

pub use parser::RegexError;

/// A compiled regular expression.
pub struct Regex {
    prog: Vec<vm::Inst>,
    /// Number of capturing groups (excluding the whole-match group 0).
    group_count: usize,
    /// `(group index, name)` pairs for named capture groups (`(?<name>…)`).
    group_names: alloc::vec::Vec<(usize, alloc::string::String)>,
    flags: Flags,
}

/// The active regex flags.
#[derive(Clone, Copy, Default)]
pub struct Flags {
    /// `i` — case-insensitive matching.
    pub ignore_case: bool,
    /// `g` — global (affects iteration by the caller, not a single match).
    pub global: bool,
    /// `m` — multiline (`^`/`$` match at line terminators).
    pub multiline: bool,
    /// `s` — dotall (`.` matches line terminators).
    pub dotall: bool,
}

impl Flags {
    /// Parses a flags string (e.g. `"gi"`), erroring on unknown flags.
    pub fn parse(s: &str) -> Result<Flags, RegexError> {
        let mut f = Flags::default();
        for c in s.chars() {
            match c {
                'i' => f.ignore_case = true,
                'g' => f.global = true,
                'm' => f.multiline = true,
                's' => f.dotall = true,
                'u' | 'y' | 'd' => {} // accepted but not yet acted on
                other => return Err(RegexError::new(alloc::format!("unknown flag `{other}`"))),
            }
        }
        Ok(f)
    }
}

/// A single match: the whole-match span plus any capture-group spans (each
/// `Some((start, end))` in `char` indices, or `None` if the group did not
/// participate). Index 0 is the whole match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Captures {
    /// The group spans; `groups[0]` is the whole match.
    pub groups: Vec<Option<(usize, usize)>>,
}

impl Captures {
    /// The whole match span.
    #[must_use]
    pub fn whole(&self) -> (usize, usize) {
        self.groups[0].expect("group 0 is always set on a successful match")
    }

    /// The span of capture group `i` (1-based for user groups), if it
    /// participated.
    #[must_use]
    pub fn group(&self, i: usize) -> Option<(usize, usize)> {
        self.groups.get(i).copied().flatten()
    }
}

impl Regex {
    /// Compiles `pattern` with the given `flags` string.
    pub fn new(pattern: &str, flags: &str) -> Result<Regex, RegexError> {
        let flags = Flags::parse(flags)?;
        let (ast, _, group_names) = parser::parse(pattern)?;
        let (prog, group_count) = compile::compile(&ast);
        Ok(Regex {
            prog,
            group_count,
            group_names,
            flags,
        })
    }

    /// The number of capturing groups (excluding group 0).
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// The `(group index, name)` pairs of named capture groups (`(?<name>…)`).
    #[must_use]
    pub fn group_names(&self) -> &[(usize, alloc::string::String)] {
        &self.group_names
    }

    /// The flags this regex was compiled with.
    #[must_use]
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Whether the pattern matches anywhere in `text`.
    #[must_use]
    pub fn is_match(&self, text: &str) -> bool {
        self.captures_at(&text.chars().collect::<Vec<_>>(), 0)
            .is_some()
    }

    /// The first match at or after `char` index `start`, with captures.
    #[must_use]
    pub fn captures_from(&self, text: &str, start: usize) -> Option<Captures> {
        let chars: Vec<char> = text.chars().collect();
        self.captures_at(&chars, start)
    }

    /// The first match's whole span at or after `start` (char indices).
    #[must_use]
    pub fn find_from(&self, text: &str, start: usize) -> Option<(usize, usize)> {
        self.captures_from(text, start).map(|c| c.whole())
    }

    fn captures_at(&self, chars: &[char], start: usize) -> Option<Captures> {
        for s in start..=chars.len() {
            if let Some(groups) = vm::run(&self.prog, chars, s, self.group_count, self.flags) {
                return Some(Captures { groups });
            }
        }
        None
    }

    /// Replaces matches in `text` with `replacement`, honoring the `g` flag
    /// (otherwise only the first match). `$&` in the replacement inserts the
    /// whole match and `$1`..`$9` insert capture groups.
    #[must_use]
    pub fn replace(&self, text: &str, replacement: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::new();
        let mut pos = 0;
        while pos <= chars.len() {
            let Some(caps) = self.captures_at(&chars, pos) else {
                break;
            };
            let (ms, me) = caps.whole();
            out.extend(&chars[pos..ms]);
            expand_replacement(replacement, &chars, &caps, &mut out);
            if me > ms {
                pos = me;
            } else {
                // Empty match: emit one char and advance so we make progress.
                if me < chars.len() {
                    out.push(chars[me]);
                }
                pos = me + 1;
            }
            if !self.flags.global {
                break;
            }
        }
        out.extend(&chars[pos.min(chars.len())..]);
        out
    }
}

/// Expands a replacement template (`$&`, `$1`..`$9`) for one match.
fn expand_replacement(template: &str, chars: &[char], caps: &Captures, out: &mut String) {
    let t: Vec<char> = template.chars().collect();
    let mut i = 0;
    while i < t.len() {
        if t[i] == '$' && i + 1 < t.len() {
            match t[i + 1] {
                '&' => {
                    let (s, e) = caps.whole();
                    out.extend(&chars[s..e]);
                    i += 2;
                    continue;
                }
                d @ '1'..='9' => {
                    let idx = d as usize - '0' as usize;
                    if let Some((s, e)) = caps.group(idx) {
                        out.extend(&chars[s..e]);
                    }
                    i += 2;
                    continue;
                }
                '$' => {
                    out.push('$');
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        out.push(t[i]);
        i += 1;
    }
}
