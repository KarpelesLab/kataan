//! The backtracking regex virtual machine.

use super::Flags;
use super::parser::{PropKind, Shorthand};
use alloc::vec::Vec;
use core::cell::Cell;

/// Backstop step budget for a whole find operation. The recursive backtracker
/// has no inherent bound, so a pathological pattern (e.g. `/(a+)+$/`) can explore
/// exponentially many paths. Once this many backtrack steps are taken the match
/// aborts cleanly (treated as "no match") rather than hanging the process. The
/// budget is scaled by input length so legitimate long-subject matches are not
/// starved, while keeping a fixed ceiling.
///
/// `STEP_BASE` was 10M, which is ~60-76 s of wall time on an adversarial
/// pattern; lowered ~33× to keep catastrophic backtracking well under a second
/// while still covering legitimate patterns (the regex suite passes). The
/// counter is created once per *find* (in `captures_at`) and shared across all
/// start positions via `run_shared`, so the bound covers the whole operation,
/// not each start independently (RE-8).
const STEP_BASE: u64 = 300_000;
const STEP_PER_CHAR: u64 = 1_000;

/// Maximum `backtrack` recursion depth before the match aborts cleanly. The VM
/// recurses once per `Split`/`Save`/lookaround exploration; on an 8 MiB stack a
/// few thousand frames is safe. Picked conservatively to avoid SIGSEGV on
/// adversarial input (e.g. `/a*/` on a very long string) while remaining far
/// above any realistic legitimate nesting/iteration depth.
const MAX_DEPTH: u32 = 2_000;

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
    /// A lookahead: run `prog` (a self-contained sub-program ending in `Match`)
    /// at the current position without consuming input. `neg` inverts the sense.
    Look { neg: bool, prog: Vec<Inst> },
    /// A lookbehind: `prog` must match some substring ending exactly at the
    /// current position. `neg` inverts the sense.
    LookBehind { neg: bool, prog: Vec<Inst> },
    /// A backreference: match the text previously captured by group `n`.
    Backref(usize),
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

/// The step budget for a subject of `input_len` chars: a fixed base plus a
/// per-char allowance so a legitimate long-subject match is not starved.
pub(crate) fn budget_for(input_len: usize) -> u64 {
    STEP_BASE.saturating_add(STEP_PER_CHAR.saturating_mul(input_len as u64))
}

/// Runs `prog` against `input` starting at `start`, returning the capture slots
/// (`2 * (group_count + 1)` of them, as `(start, end)` pairs) on success.
///
/// Threads a caller-owned step counter and budget so a multi-start find
/// ([`super::Regex::captures_at`]) shares one budget across all start positions
/// instead of resetting it per start (RE-8).
pub(crate) fn run_shared(
    prog: &[Inst],
    input: &[char],
    start: usize,
    group_count: usize,
    flags: Flags,
    steps: &Cell<u64>,
    budget: u64,
) -> Option<Vec<Option<(usize, usize)>>> {
    let mut saves = alloc::vec![None; 2 * (group_count + 1)];
    let ctx = Ctx {
        prog,
        input,
        flags,
        match_end: None,
        steps,
        budget,
    };
    let mut loops: Vec<(usize, usize)> = Vec::new();
    if backtrack(&ctx, 0, start, &mut saves, 0, &mut loops) {
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
    /// When `Some(p)`, `Match` succeeds only at position `p` (for lookbehind,
    /// which requires the sub-pattern to end exactly at the assertion point).
    match_end: Option<usize>,
    /// Backtrack steps consumed so far. A single shared `Cell` is referenced by
    /// the root context and every lookaround sub-context, so the budget bounds
    /// the *whole* match (sub-programs included), not each context separately.
    steps: &'a Cell<u64>,
    /// The step ceiling; once `steps` exceeds it the match aborts as no-match.
    budget: u64,
}

impl Ctx<'_> {
    /// Records one backtrack step; returns `false` once the budget is exhausted
    /// so the caller can abort the match cleanly (treated as no-match).
    #[inline]
    fn tick(&self) -> bool {
        let n = self.steps.get().saturating_add(1);
        self.steps.set(n);
        n <= self.budget
    }
}

/// The recursive backtracking executor. `pc` is the program counter, `sp` the
/// position in the input, `depth` the current recursion depth. `saves` holds raw
/// capture positions. Returns `false` (clean no-match) if the step budget or the
/// recursion-depth cap is exceeded, so an adversarial pattern can never hang or
/// overflow the stack.
fn backtrack(
    ctx: &Ctx,
    mut pc: usize,
    mut sp: usize,
    saves: &mut Vec<Option<usize>>,
    depth: u32,
    // Active loop-head positions on the current path: each `(split_pc, sp)` we
    // are currently iterating. Re-entering the same loop head at the same `sp`
    // means the body matched empty (`/()*/`, `/(a*)*/`) — that path is pruned.
    loops: &mut Vec<(usize, usize)>,
) -> bool {
    // Recursion-depth cap: abort cleanly before we can overflow the stack.
    if depth > MAX_DEPTH {
        return false;
    }
    loop {
        // Step backstop: every instruction visited counts toward the budget; once
        // exhausted the whole match unwinds as a no-match.
        if !ctx.tick() {
            return false;
        }
        match &ctx.prog[pc] {
            Inst::Match => return ctx.match_end.is_none_or(|p| sp == p),
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
                // Fast path for a *simple* quantifier loop whose body is a single
                // consuming instruction (`a*`, `.*`, `\d+`, `[…]*`, …): consume
                // iteratively instead of recursing once per repetition. Without
                // this, `/a*/` over a long subject would recurse one frame per
                // character and overflow the stack (RE-2a). Iterating keeps the
                // recursion depth proportional to pattern *nesting*, not input
                // length. A single-consume body always advances, so it can never
                // be zero-width — no loop-stack bookkeeping is needed here.
                if let Some((consume_pc, cont_pc, greedy)) = simple_loop(ctx.prog, pc) {
                    if greedy {
                        // Greedy: consume as many as possible, recording each
                        // position, then try the continuation from longest down.
                        let mut positions = alloc::vec![sp];
                        while consume_one(&ctx.prog[consume_pc], ctx.input, sp, ctx.flags) {
                            sp += 1;
                            positions.push(sp);
                            if !ctx.tick() {
                                return false;
                            }
                        }
                        while let Some(p) = positions.pop() {
                            if backtrack(ctx, cont_pc, p, saves, depth + 1, loops) {
                                return true;
                            }
                            if !ctx.tick() {
                                return false;
                            }
                        }
                        return false;
                    }
                    // Lazy: try the continuation first at each length, growing.
                    loop {
                        if backtrack(ctx, cont_pc, sp, saves, depth + 1, loops) {
                            return true;
                        }
                        if !consume_one(&ctx.prog[consume_pc], ctx.input, sp, ctx.flags) {
                            return false;
                        }
                        sp += 1;
                        if !ctx.tick() {
                            return false;
                        }
                    }
                }

                // Zero-width-progress guard: a `*`/`+` loop compiles to
                // `Split(body, exit); body…; Jmp(split)`, so the body loops back
                // to this Split. Track the loop heads currently being iterated;
                // if we re-enter *this* head at the *same* `sp`, the body matched
                // empty (`/()*/`, `/(a*)*/`) and re-entering can only recurse
                // forever, so prune straight to the exit branch.
                if let Some((body, exit)) = loop_head(ctx.prog, pc) {
                    if loops.contains(&(pc, sp)) {
                        pc = exit;
                        continue;
                    }
                    loops.push((pc, sp));
                    let took_body = backtrack(ctx, body, sp, saves, depth + 1, loops);
                    loops.pop();
                    if took_body {
                        return true;
                    }
                    pc = exit;
                    continue;
                }

                // A plain (non-loop) Split, e.g. alternation or `?`.
                if backtrack(ctx, *a, sp, saves, depth + 1, loops) {
                    return true;
                }
                pc = *b;
            }
            Inst::Save(slot) => {
                let old = saves[*slot];
                saves[*slot] = Some(sp);
                if backtrack(ctx, pc + 1, sp, saves, depth + 1, loops) {
                    return true;
                }
                saves[*slot] = old;
                return false;
            }
            Inst::Look { neg, prog } => {
                // Zero-width: run the sub-program at `sp` (captures discarded).
                let sub = Ctx {
                    prog,
                    input: ctx.input,
                    flags: ctx.flags,
                    match_end: None,
                    steps: ctx.steps,
                    budget: ctx.budget,
                };
                let mut sub_saves = alloc::vec![None; saves.len()];
                let mut sub_loops = Vec::new();
                let matched = backtrack(&sub, 0, sp, &mut sub_saves, depth + 1, &mut sub_loops);
                if matched != *neg {
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::LookBehind { neg, prog } => {
                // The sub-pattern must match some substring ending exactly at
                // `sp`; try every start position `j <= sp`.
                let sub = Ctx {
                    prog,
                    input: ctx.input,
                    flags: ctx.flags,
                    match_end: Some(sp),
                    steps: ctx.steps,
                    budget: ctx.budget,
                };
                let mut matched = false;
                for j in (0..=sp).rev() {
                    // Each rescan start counts as a step; the O(n) rescan is thus
                    // bounded by the shared budget (RE-6 backstop).
                    if !ctx.tick() {
                        return false;
                    }
                    let mut sub_saves = alloc::vec![None; saves.len()];
                    let mut sub_loops = Vec::new();
                    if backtrack(&sub, 0, j, &mut sub_saves, depth + 1, &mut sub_loops) {
                        matched = true;
                        break;
                    }
                }
                if matched != *neg {
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::Backref(g) => {
                match (
                    saves.get(2 * g).copied().flatten(),
                    saves.get(2 * g + 1).copied().flatten(),
                ) {
                    (Some(s), Some(e)) => {
                        let len = e - s;
                        if sp + len <= ctx.input.len()
                            && (0..len)
                                .all(|i| char_eq(ctx.input[sp + i], ctx.input[s + i], ctx.flags))
                        {
                            sp += len;
                            pc += 1;
                        } else {
                            return false;
                        }
                    }
                    // An unmatched group backreference matches the empty string.
                    _ => pc += 1,
                }
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

/// Recognizes a *simple* quantifier loop at `split_pc`: a `Split` whose body is a
/// single consuming instruction (`Char`/`Any`/`Class`) followed by `Jmp` back to
/// the `Split`. Returns `(consume_pc, continuation_pc, greedy)` so the executor
/// can iterate the repetition without recursing once per character.
///
/// Greedy form: `Split(body, exit); <consume>; Jmp(split)` → body is `split+1`.
/// Lazy form:   `Split(exit, body); <consume>; Jmp(split)` → body is `split+2`.
fn simple_loop(prog: &[Inst], split_pc: usize) -> Option<(usize, usize, bool)> {
    let Inst::Split(a, b) = &prog[split_pc] else {
        return None;
    };
    // Greedy: first target is the body (split+1), second is the exit.
    if *a == split_pc + 1
        && is_single_consume(&prog[split_pc + 1])
        && matches!(prog.get(split_pc + 2), Some(Inst::Jmp(t)) if *t == split_pc)
    {
        return Some((split_pc + 1, *b, true));
    }
    // Lazy: first target is the exit, second is the body (split+1).
    if *b == split_pc + 1
        && is_single_consume(&prog[split_pc + 1])
        && matches!(prog.get(split_pc + 2), Some(Inst::Jmp(t)) if *t == split_pc)
    {
        return Some((split_pc + 1, *a, false));
    }
    None
}

/// Whether `inst` consumes exactly one input character on success.
fn is_single_consume(inst: &Inst) -> bool {
    matches!(inst, Inst::Char(_) | Inst::Any | Inst::Class(_))
}

/// Recognizes a `*`/`+` loop head at `split_pc` (any body, not just a single
/// instruction). The compiler always emits the body immediately after the
/// `Split` (at `split_pc + 1`) and ends it with `Jmp(split_pc)` just before the
/// exit. Returns `(body_pc, exit_pc)` if so; greedy and lazy differ only in which
/// `Split` branch is the body. Used to drive the zero-width-progress guard.
fn loop_head(prog: &[Inst], split_pc: usize) -> Option<(usize, usize)> {
    let Inst::Split(a, b) = &prog[split_pc] else {
        return None;
    };
    let body = split_pc + 1;
    // The body branch is whichever target points just past the Split.
    let exit = if *a == body {
        *b
    } else if *b == body {
        *a
    } else {
        return None;
    };
    // A loop body is closed by `Jmp(split_pc)` sitting immediately before `exit`.
    if exit > 0 && matches!(prog.get(exit - 1), Some(Inst::Jmp(t)) if *t == split_pc) {
        Some((body, exit))
    } else {
        None
    }
}

/// Tries to consume one character with a single-consume instruction at `sp`.
fn consume_one(inst: &Inst, input: &[char], sp: usize, flags: Flags) -> bool {
    if sp >= input.len() {
        return false;
    }
    match inst {
        Inst::Char(c) => char_eq(input[sp], *c, flags),
        Inst::Any => flags.dotall || !is_line_term(input[sp]),
        Inst::Class(class) => class_matches(class, input[sp], flags),
        _ => false,
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
        Shorthand::Property(kind, neg) => property_matches(kind, c) ^ neg,
    }
}

/// Matches a `\p{…}` property using pure-Rust `char` predicates.
fn property_matches(kind: PropKind, c: char) -> bool {
    match kind {
        PropKind::Letter => c.is_alphabetic(),
        PropKind::Upper => c.is_uppercase(),
        PropKind::Lower => c.is_lowercase(),
        PropKind::Number => c.is_numeric(),
        PropKind::White => c.is_whitespace(),
        PropKind::Alnum => c.is_alphanumeric(),
        PropKind::Gc(code) => general_category_matches(code, c),
    }
}

/// Whether `c` belongs to the general category `code` (`[group, 0]` or
/// `[g, sub]`). With the `intl` feature this consults the Unicode tables for an
/// exact answer; otherwise it falls back to `char`-method approximations that
/// are correct for the common groups and cased/letter/number subcategories.
#[cfg(feature = "intl")]
fn general_category_matches(code: [u8; 2], c: char) -> bool {
    use intl::unicode::category::Group;
    let gc = intl::unicode::general_category(c);
    if code[1] == 0 {
        let want = match code[0] {
            b'L' => Group::Letter,
            b'M' => Group::Mark,
            b'N' => Group::Number,
            b'P' => Group::Punctuation,
            b'S' => Group::Symbol,
            b'Z' => Group::Separator,
            b'C' => Group::Other,
            _ => return false,
        };
        gc.group() == want
    } else {
        gc.abbr().as_bytes() == code
    }
}

#[cfg(not(feature = "intl"))]
fn general_category_matches(code: [u8; 2], c: char) -> bool {
    match &code {
        b"L\0" => c.is_alphabetic(),
        b"N\0" => c.is_numeric(),
        b"Z\0" => c == ' ' || (c.is_whitespace() && !c.is_control()),
        b"C\0" => c.is_control(),
        b"P\0" => c.is_ascii_punctuation(),
        b"Lu" => c.is_uppercase(),
        b"Ll" => c.is_lowercase(),
        // An uncased letter (e.g. CJK, scripts without case).
        b"Lo" => c.is_alphabetic() && !c.is_uppercase() && !c.is_lowercase(),
        b"Nd" => c.is_ascii_digit(),
        b"Cc" => c.is_control(),
        // Finer categories need the Unicode tables (the `intl` feature).
        _ => false,
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
