//! The backtracking regex virtual machine.

use super::Flags;
use super::parser::Shorthand;
use alloc::vec::Vec;

/// A compiled instruction.
pub(crate) enum Inst {
    /// Match a specific character.
    Char(char),
    /// `.` — any character (subject to the dotall flag).
    Any,
    /// A character class.
    Class(Class),
    /// A successful match.
    Match,
    /// Unconditional jump.
    Jmp(usize),
    /// Try the first target, backtracking to the second.
    Split(usize, usize),
    /// Record the current position into capture slot `n`.
    Save(usize),
    /// A zero-width assertion.
    Assert(Assert),
}

/// A character-class instruction operand.
pub(crate) struct Class {
    pub neg: bool,
    pub members: Vec<ClassMember>,
}

/// One member of a compiled class.
pub(crate) enum ClassMember {
    Char(char),
    Range(char, char),
    Shorthand(Shorthand),
}

/// A zero-width assertion.
pub(crate) enum Assert {
    Start,
    End,
    WordBoundary,
    NotWordBoundary,
}

/// Runs `prog` against `input` starting at `start`, returning the capture slots
/// (`2 * (group_count + 1)` of them, as `(start, end)` pairs) on success.
pub(crate) fn run(
    prog: &[Inst],
    input: &[char],
    start: usize,
    group_count: usize,
    flags: Flags,
) -> Option<Vec<Option<(usize, usize)>>> {
    let mut saves = alloc::vec![None; 2 * (group_count + 1)];
    let ctx = Ctx { prog, input, flags };
    if backtrack(&ctx, 0, start, &mut saves) {
        // Pair the raw save slots into (start, end) spans per group.
        let mut groups = Vec::with_capacity(group_count + 1);
        for g in 0..=group_count {
            groups.push(match (saves[2 * g], saves[2 * g + 1]) {
                (Some(s), Some(e)) => Some((s, e)),
                _ => None,
            });
        }
        Some(groups)
    } else {
        None
    }
}

struct Ctx<'a> {
    prog: &'a [Inst],
    input: &'a [char],
    flags: Flags,
}

/// The recursive backtracking executor. `pc` is the program counter, `sp` the
/// position in the input. `saves` holds raw capture positions.
fn backtrack(ctx: &Ctx, mut pc: usize, mut sp: usize, saves: &mut Vec<Option<usize>>) -> bool {
    loop {
        match &ctx.prog[pc] {
            Inst::Match => return true,
            Inst::Char(c) => {
                if sp < ctx.input.len() && char_eq(ctx.input[sp], *c, ctx.flags) {
                    sp += 1;
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::Any => {
                if sp < ctx.input.len() && (ctx.flags.dotall || !is_line_term(ctx.input[sp])) {
                    sp += 1;
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::Class(class) => {
                if sp < ctx.input.len() && class_matches(class, ctx.input[sp], ctx.flags) {
                    sp += 1;
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::Jmp(target) => pc = *target,
            Inst::Split(a, b) => {
                // Try the first branch; on failure, fall through to the second.
                if backtrack(ctx, *a, sp, saves) {
                    return true;
                }
                pc = *b;
            }
            Inst::Save(slot) => {
                let old = saves[*slot];
                saves[*slot] = Some(sp);
                if backtrack(ctx, pc + 1, sp, saves) {
                    return true;
                }
                saves[*slot] = old;
                return false;
            }
            Inst::Assert(assert) => {
                if assert_ok(assert, ctx.input, sp, ctx.flags) {
                    pc += 1;
                } else {
                    return false;
                }
            }
        }
    }
}

fn char_eq(a: char, b: char, flags: Flags) -> bool {
    if a == b {
        return true;
    }
    if !flags.ignore_case {
        return false;
    }
    // Case-insensitive: compare by Unicode case folding (spec `Canonicalize`),
    // which catches pairs simple lowercasing misses (e.g. the Kelvin sign
    // U+212A ↔ `k`, long s U+017F ↔ `s`, final sigma ς ↔ σ).
    #[cfg(feature = "intl")]
    {
        intl::unicode::case::case_fold(a).eq(intl::unicode::case::case_fold(b))
    }
    #[cfg(not(feature = "intl"))]
    {
        a.eq_ignore_ascii_case(&b) || a.to_lowercase().eq(b.to_lowercase())
    }
}

fn is_line_term(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn class_matches(class: &Class, c: char, flags: Flags) -> bool {
    let mut hit = false;
    for m in &class.members {
        let matched = match m {
            ClassMember::Char(ch) => char_eq(c, *ch, flags),
            ClassMember::Range(lo, hi) => {
                (c >= *lo && c <= *hi)
                    || (flags.ignore_case && {
                        let cl = c.to_ascii_lowercase();
                        let cu = c.to_ascii_uppercase();
                        (cl >= *lo && cl <= *hi) || (cu >= *lo && cu <= *hi)
                    })
            }
            ClassMember::Shorthand(s) => shorthand_matches(*s, c),
        };
        if matched {
            hit = true;
            break;
        }
    }
    hit ^ class.neg
}

fn shorthand_matches(s: Shorthand, c: char) -> bool {
    match s {
        Shorthand::Digit => c.is_ascii_digit(),
        Shorthand::NotDigit => !c.is_ascii_digit(),
        Shorthand::Word => is_word(c),
        Shorthand::NotWord => !is_word(c),
        Shorthand::Space => c.is_whitespace(),
        Shorthand::NotSpace => !c.is_whitespace(),
    }
}

fn assert_ok(assert: &Assert, input: &[char], sp: usize, flags: Flags) -> bool {
    match assert {
        Assert::Start => sp == 0 || (flags.multiline && is_line_term(input[sp - 1])),
        Assert::End => sp == input.len() || (flags.multiline && is_line_term(input[sp])),
        Assert::WordBoundary => is_boundary(input, sp),
        Assert::NotWordBoundary => !is_boundary(input, sp),
    }
}

fn is_boundary(input: &[char], sp: usize) -> bool {
    let before = sp > 0 && is_word(input[sp - 1]);
    let after = sp < input.len() && is_word(input[sp]);
    before != after
}
