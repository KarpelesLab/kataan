//! The backtracking regex virtual machine.
//!
//! The VM matches over a subject of **UTF-16 code units** (`&[u16]`), matching
//! JavaScript string semantics: positions are code-unit indices and a lone
//! surrogate is a matchable unit. How many code units one "character" spans
//! depends on the `u` (unicode) flag:
//!
//! * **non-`u`** (the web-compat default): every primitive (`.`, classes,
//!   literals, quantifiers, backrefs) operates on a single code unit. An astral
//!   character is two units, so `.` matches one half of a surrogate pair.
//! * **`u`**: primitives operate on whole code points — a surrogate pair counts
//!   as one character and the engine advances by a full code point — but the
//!   reported positions remain code-unit indices (per the spec, indices into the
//!   UTF-16 string). A lone surrogate in `u` mode matches as a single unit.

use super::Flags;
use super::parser::Shorthand;
use alloc::boxed::Box;
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
const STEP_BASE: u64 = crate::limits::DEFAULT_REGEX_STEP_BASE;
const STEP_PER_CHAR: u64 = 1_000;

/// Maximum `backtrack` recursion depth before the match aborts cleanly. The VM
/// recurses once per `Split`/`Save`/lookaround exploration; on an 8 MiB stack a
/// few thousand frames is safe. Picked conservatively to avoid SIGSEGV on
/// adversarial input (e.g. `/a*/` on a very long string) while remaining far
/// above any realistic legitimate nesting/iteration depth.
const MAX_DEPTH: u32 = crate::limits::DEFAULT_REGEX_MAX_DEPTH;

/// A compiled instruction.
pub(crate) enum Inst {
    /// Match a specific code point (a scalar value; in non-`u` mode the compiler
    /// only ever emits BMP/lone-surrogate scalars since astral literals are split
    /// into two `Char` units up front).
    Char(u32),
    /// `.` — any character (subject to the dotall flag).
    Any,
    /// A character class.
    Class(Class),
    /// A `v`-mode extended character class matching a single code point. `neg`
    /// inverts membership. String alternatives are compiled separately (as an
    /// alternation), so this instruction always consumes exactly one code point.
    ClassSet { neg: bool, matcher: SetMatcher },
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
    /// Enter an inline-modifier scope `(?ims-ims:…)`: push the flag delta onto
    /// the flag stack so the enclosed instructions match under the adjusted
    /// `i`/`m`/`s` flags. Paired with a later [`Inst::PopFlags`].
    PushFlags(FlagDelta),
    /// Leave the innermost inline-modifier scope (restores the prior flags).
    PopFlags,
}

/// A flag delta for an inline-modifier group: each field, when `Some`, overrides
/// the enclosing value of that flag for the scoped subexpression.
#[derive(Clone, Copy)]
pub(crate) struct FlagDelta {
    pub ignore_case: Option<bool>,
    pub multiline: Option<bool>,
    pub dotall: Option<bool>,
}

impl FlagDelta {
    /// Applies this delta to `base`, returning the effective flags inside the
    /// modifier scope. Only `i`/`m`/`s` can be overridden; the rest pass through.
    fn apply(self, base: Flags) -> Flags {
        let mut f = base;
        if let Some(v) = self.ignore_case {
            f.ignore_case = v;
        }
        if let Some(v) = self.multiline {
            f.multiline = v;
        }
        if let Some(v) = self.dotall {
            f.dotall = v;
        }
        f
    }
}

/// A character-class instruction operand.
pub(crate) struct Class {
    pub neg: bool,
    pub members: Vec<ClassMember>,
}

/// A compiled `v`-mode extended character-class matcher: a tree of set
/// operations over code points. Strings (from `\q{…}` / properties of strings)
/// are handled separately by the compiler (as an alternation), so this only
/// concerns single-code-point membership.
pub(crate) enum SetMatcher {
    /// Union: matches if any child matches.
    Union(Vec<SetMatcher>),
    /// Intersection: matches if every child matches.
    Intersection(Vec<SetMatcher>),
    /// Difference: in the first child and in none of the rest.
    Difference(Vec<SetMatcher>),
    /// A negated sub-matcher (a nested `[^…]`).
    Negated(Box<SetMatcher>),
    /// A leaf set of chars/ranges/shorthands.
    Leaf(Vec<ClassMember>),
}

impl SetMatcher {
    /// Whether code point `c` is a member of this set, honoring `flags` (the `i`
    /// flag for case-insensitive char/range/property membership).
    pub(crate) fn matches(&self, c: u32, flags: Flags) -> bool {
        match self {
            SetMatcher::Union(kids) => kids.iter().any(|k| k.matches(c, flags)),
            SetMatcher::Intersection(kids) => kids.iter().all(|k| k.matches(c, flags)),
            SetMatcher::Difference(kids) => {
                let mut it = kids.iter();
                let Some(first) = it.next() else {
                    return false;
                };
                first.matches(c, flags) && !it.any(|k| k.matches(c, flags))
            }
            SetMatcher::Negated(inner) => !inner.matches(c, flags),
            SetMatcher::Leaf(members) => leaf_members_match(members, c, flags),
        }
    }
}

/// Membership of `c` in a flat list of class members (chars/ranges/shorthands),
/// honoring the `i` flag — shared by the legacy `Class` and `v`-mode leaves.
fn leaf_members_match(members: &[ClassMember], c: u32, flags: Flags) -> bool {
    for m in members {
        let matched = match m {
            ClassMember::Char(ch) => cp_eq(c, *ch, flags),
            ClassMember::Range(lo, hi) => {
                (c >= *lo && c <= *hi) || (flags.ignore_case && range_fold_hit(c, *lo, *hi))
            }
            ClassMember::Shorthand(s) => shorthand_matches(*s, c, flags),
        };
        if matched {
            return true;
        }
    }
    false
}

/// One member of a compiled class. Bounds are scalar values (code points).
pub(crate) enum ClassMember {
    Char(u32),
    Range(u32, u32),
    Shorthand(Shorthand),
}

/// A zero-width assertion.
pub(crate) enum Assert {
    Start,
    End,
    WordBoundary,
    NotWordBoundary,
}

/// The step budget for a subject of `input_len` code units: a fixed base plus a
/// per-unit allowance so a legitimate long-subject match is not starved.
pub(crate) fn budget_for(input_len: usize) -> u64 {
    STEP_BASE.saturating_add(STEP_PER_CHAR.saturating_mul(input_len as u64))
}

/// Reads the code point at unit index `sp` and the number of code units it
/// spans. In `u` mode a well-formed surrogate pair is decoded as one code point
/// (length 2); a lone surrogate, or any unit in non-`u` mode, is returned as its
/// own scalar with length 1. The returned scalar is suitable for comparing
/// against compiled `Char`/`Class` scalars.
fn read_cp(input: &[u16], sp: usize, unicode: bool) -> Option<(u32, usize)> {
    let u = *input.get(sp)? as u32;
    if unicode
        && (0xD800..=0xDBFF).contains(&u)
        && let Some(&lo) = input.get(sp + 1)
    {
        let lo = lo as u32;
        if (0xDC00..=0xDFFF).contains(&lo) {
            let cp = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            return Some((cp, 2));
        }
    }
    Some((u, 1))
}

/// Runs `prog` against `input` (UTF-16 code units) starting at unit index
/// `start`, returning the capture slots (`2 * (group_count + 1)` of them, as
/// `(start, end)` unit-index pairs) on success.
///
/// Threads a caller-owned step counter and budget so a multi-start find
/// ([`super::Regex::captures_at`]) shares one budget across all start positions
/// instead of resetting it per start (RE-8).
pub(crate) fn run_shared(
    prog: &[Inst],
    input: &[u16],
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
    // The flag stack: the base flags sit at the bottom; each entered
    // inline-modifier scope (`(?ims-ims:…)`) pushes the locally-adjusted flags.
    let mut flag_stack: Vec<Flags> = alloc::vec![flags];
    if backtrack(&ctx, 0, start, &mut saves, 0, &mut loops, &mut flag_stack) {
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
    input: &'a [u16],
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
/// position in the input (a code-unit index), `depth` the current recursion
/// depth. `saves` holds raw capture positions. Returns `false` (clean no-match)
/// if the step budget or the recursion-depth cap is exceeded, so an adversarial
/// pattern can never hang or overflow the stack.
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
    // The flag stack: the top entry is the flags in effect at this point, after
    // applying any enclosing inline-modifier scopes. `unicode` never changes
    // (only `i`/`m`/`s` can be modified), so it is read once below.
    flags_stack: &mut Vec<Flags>,
) -> bool {
    // Recursion-depth cap: abort cleanly before we can overflow the stack.
    if depth > MAX_DEPTH {
        return false;
    }
    let unicode = ctx.flags.unicode;
    loop {
        // The flags in effect for the current instruction (the flag-stack top).
        let cur = *flags_stack.last().unwrap_or(&ctx.flags);
        // Step backstop: every instruction visited counts toward the budget; once
        // exhausted the whole match unwinds as a no-match.
        if !ctx.tick() {
            return false;
        }
        match &ctx.prog[pc] {
            Inst::Match => return ctx.match_end.is_none_or(|p| sp == p),
            Inst::Char(c) => {
                if let Some((cp, len)) = read_cp(ctx.input, sp, unicode)
                    && cp_eq(cp, *c, cur)
                {
                    sp += len;
                    pc += 1;
                    continue;
                }
                return false;
            }
            Inst::Any => {
                if let Some((cp, len)) = read_cp(ctx.input, sp, unicode)
                    && (cur.dotall || !is_line_term(cp))
                {
                    sp += len;
                    pc += 1;
                    continue;
                }
                return false;
            }
            Inst::Class(class) => {
                if let Some((cp, len)) = read_cp(ctx.input, sp, unicode)
                    && class_matches(class, cp, cur)
                {
                    sp += len;
                    pc += 1;
                    continue;
                }
                return false;
            }
            Inst::ClassSet { neg, matcher } => {
                if let Some((cp, len)) = read_cp(ctx.input, sp, unicode)
                    && (matcher.matches(cp, cur) ^ *neg)
                {
                    sp += len;
                    pc += 1;
                    continue;
                }
                return false;
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
                        while let Some(adv) = consume_one(&ctx.prog[consume_pc], ctx.input, sp, cur)
                        {
                            sp += adv;
                            positions.push(sp);
                            if !ctx.tick() {
                                return false;
                            }
                        }
                        while let Some(p) = positions.pop() {
                            if backtrack(ctx, cont_pc, p, saves, depth + 1, loops, flags_stack) {
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
                        if backtrack(ctx, cont_pc, sp, saves, depth + 1, loops, flags_stack) {
                            return true;
                        }
                        let Some(adv) = consume_one(&ctx.prog[consume_pc], ctx.input, sp, cur)
                        else {
                            return false;
                        };
                        sp += adv;
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
                    let took_body = backtrack(ctx, body, sp, saves, depth + 1, loops, flags_stack);
                    loops.pop();
                    if took_body {
                        return true;
                    }
                    pc = exit;
                    continue;
                }

                // A plain (non-loop) Split, e.g. alternation or `?`.
                if backtrack(ctx, *a, sp, saves, depth + 1, loops, flags_stack) {
                    return true;
                }
                pc = *b;
            }
            Inst::Save(slot) => {
                let old = saves[*slot];
                saves[*slot] = Some(sp);
                if backtrack(ctx, pc + 1, sp, saves, depth + 1, loops, flags_stack) {
                    return true;
                }
                saves[*slot] = old;
                return false;
            }
            Inst::Look { neg, prog } => {
                // Zero-width: run the sub-program at `sp`. A *positive* lookahead
                // propagates the groups its sub-match captured to the outer match
                // (`/(?=(\d+))/.exec("123")` captures `"123"`); a *negative* one
                // contributes nothing. Seed the sub-saves from the current saves so
                // groups already bound outside (and backrefs to them) stay visible
                // inside the assertion.
                let sub = Ctx {
                    prog,
                    input: ctx.input,
                    flags: cur,
                    match_end: None,
                    steps: ctx.steps,
                    budget: ctx.budget,
                };
                let mut sub_saves = saves.clone();
                let mut sub_loops = Vec::new();
                // The sub-program starts under the flags in effect at the
                // assertion (so an enclosing `(?i:…(?=…)…)` is honored).
                let mut sub_flags = alloc::vec![cur];
                let matched = backtrack(
                    &sub,
                    0,
                    sp,
                    &mut sub_saves,
                    depth + 1,
                    &mut sub_loops,
                    &mut sub_flags,
                );
                if matched == *neg {
                    return false;
                }
                if *neg {
                    // Negative lookahead: zero-width, no captures contributed.
                    pc += 1;
                    continue;
                }
                // Positive: adopt the sub-match's captures, then continue. On a
                // later failure restore the originals so backtracking past the
                // assertion is sound.
                let saved = core::mem::replace(saves, sub_saves);
                if backtrack(ctx, pc + 1, sp, saves, depth + 1, loops, flags_stack) {
                    return true;
                }
                *saves = saved;
                return false;
            }
            Inst::LookBehind { neg, prog } => {
                // The sub-pattern must match some substring ending exactly at
                // `sp`; try every start position `j <= sp`. A *positive* lookbehind
                // propagates the captures of the matched substring; a *negative*
                // one contributes nothing.
                let sub = Ctx {
                    prog,
                    input: ctx.input,
                    flags: cur,
                    match_end: Some(sp),
                    steps: ctx.steps,
                    budget: ctx.budget,
                };
                let mut found: Option<Vec<Option<usize>>> = None;
                for j in (0..=sp).rev() {
                    // Each rescan start counts as a step; the O(n) rescan is thus
                    // bounded by the shared budget (RE-6 backstop).
                    if !ctx.tick() {
                        return false;
                    }
                    let mut sub_saves = saves.clone();
                    let mut sub_loops = Vec::new();
                    let mut sub_flags = alloc::vec![cur];
                    if backtrack(
                        &sub,
                        0,
                        j,
                        &mut sub_saves,
                        depth + 1,
                        &mut sub_loops,
                        &mut sub_flags,
                    ) {
                        found = Some(sub_saves);
                        break;
                    }
                }
                let matched = found.is_some();
                if matched == *neg {
                    return false;
                }
                if *neg {
                    pc += 1;
                    continue;
                }
                let saved = core::mem::replace(saves, found.unwrap());
                if backtrack(ctx, pc + 1, sp, saves, depth + 1, loops, flags_stack) {
                    return true;
                }
                *saves = saved;
                return false;
            }
            Inst::Backref(g) => {
                match (
                    saves.get(2 * g).copied().flatten(),
                    saves.get(2 * g + 1).copied().flatten(),
                ) {
                    (Some(s), Some(e)) => {
                        // Backrefs compare raw code units (a captured span is a
                        // run of units); case folding is applied unit-by-unit.
                        let len = e - s;
                        if sp + len <= ctx.input.len()
                            && (0..len).all(|i| unit_eq(ctx.input[sp + i], ctx.input[s + i], cur))
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
                if assert_ok(assert, ctx.input, sp, cur) {
                    pc += 1;
                } else {
                    return false;
                }
            }
            Inst::PushFlags(delta) => {
                // Enter an inline-modifier scope: push the adjusted flags and run
                // the remainder recursively so the push is undone on every exit
                // path (success, failure, or backtrack), keeping the stack sound.
                flags_stack.push(delta.apply(cur));
                let r = backtrack(ctx, pc + 1, sp, saves, depth + 1, loops, flags_stack);
                flags_stack.pop();
                return r;
            }
            Inst::PopFlags => {
                // Leave the innermost modifier scope: pop, run the remainder under
                // the restored flags, then push back so the caller's stack is
                // unchanged on return.
                let restored = flags_stack.pop();
                let r = backtrack(ctx, pc + 1, sp, saves, depth + 1, loops, flags_stack);
                if let Some(f) = restored {
                    flags_stack.push(f);
                }
                return r;
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

/// Whether `inst` consumes exactly one character (1 or 2 code units) on success.
fn is_single_consume(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. }
    )
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

/// Tries to consume one character with a single-consume instruction at `sp`,
/// returning the number of code units advanced (1 or 2) on success.
fn consume_one(inst: &Inst, input: &[u16], sp: usize, flags: Flags) -> Option<usize> {
    let (cp, len) = read_cp(input, sp, flags.unicode)?;
    let ok = match inst {
        Inst::Char(c) => cp_eq(cp, *c, flags),
        Inst::Any => flags.dotall || !is_line_term(cp),
        Inst::Class(class) => class_matches(class, cp, flags),
        Inst::ClassSet { neg, matcher } => matcher.matches(cp, flags) ^ *neg,
        _ => false,
    };
    ok.then_some(len)
}

/// Compares two scalar code points for equality, honoring the `i` flag.
fn cp_eq(a: u32, b: u32, flags: Flags) -> bool {
    if a == b {
        return true;
    }
    if !flags.ignore_case {
        return false;
    }
    match (char::from_u32(a), char::from_u32(b)) {
        (Some(ca), Some(cb)) => char_fold_eq(ca, cb),
        // Lone surrogates have no case; only exact equality (handled above).
        _ => false,
    }
}

/// Compares two raw code units for equality, honoring the `i` flag. Used by
/// backreference matching, which works unit-by-unit.
fn unit_eq(a: u16, b: u16, flags: Flags) -> bool {
    if a == b {
        return true;
    }
    if !flags.ignore_case {
        return false;
    }
    match (char::from_u32(a as u32), char::from_u32(b as u32)) {
        (Some(ca), Some(cb)) => char_fold_eq(ca, cb),
        _ => false,
    }
}

fn char_fold_eq(a: char, b: char) -> bool {
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

fn is_line_term(c: u32) -> bool {
    matches!(c, 0x0A | 0x0D | 0x2028 | 0x2029)
}

fn is_word(c: u32) -> bool {
    matches!(c, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A | 0x5F)
}

/// Whether `c` counts as a "word character" for `\w`/`\b`, honoring the spec's
/// `WordCharacters` carve-out: when **both** `i` and `u` are set, a code point
/// also counts if its case fold lands on an ASCII word character. The only two
/// non-word code points this admits are U+017F (ſ → s) and U+212A (K → k); they
/// must match `\w` / be word-boundary-relevant under `iu` (22.2.2.9.3).
fn is_word_flags(c: u32, flags: Flags) -> bool {
    if is_word(c) {
        return true;
    }
    if flags.ignore_case && flags.unicode {
        return matches!(c, 0x017F | 0x212A);
    }
    false
}

fn class_matches(class: &Class, c: u32, flags: Flags) -> bool {
    leaf_members_match(&class.members, c, flags) ^ class.neg
}

/// Case-insensitive class-range membership: a character matches `[lo-hi]` under
/// `i` if either case variant lands in the range. Restricted to ASCII letters,
/// matching the historical behavior.
fn range_fold_hit(c: u32, lo: u32, hi: u32) -> bool {
    let Some(ch) = char::from_u32(c) else {
        return false;
    };
    let cl = ch.to_ascii_lowercase() as u32;
    let cu = ch.to_ascii_uppercase() as u32;
    (cl >= lo && cl <= hi) || (cu >= lo && cu <= hi)
}

fn shorthand_matches(s: Shorthand, c: u32, flags: Flags) -> bool {
    match s {
        Shorthand::Digit => is_ascii_digit(c),
        Shorthand::NotDigit => !is_ascii_digit(c),
        Shorthand::Word => is_word_flags(c, flags),
        Shorthand::NotWord => !is_word_flags(c, flags),
        Shorthand::Space => is_space(c),
        Shorthand::NotSpace => !is_space(c),
        // A `\p{…}` / `\P{…}` Unicode property escape; `neg` is the `\P` form.
        // Resolution/validation happened at parse time, so here we only test
        // membership of the code point (see `super::props`). Under `i` the spec's
        // `Canonicalize` makes the (already-negated) set case-insensitive: `c`
        // matches if any of its case variants satisfies the negated membership.
        Shorthand::Property(prop, neg) => prop_matches_ci(&prop, neg, c, flags),
    }
}

/// Membership of a (possibly negated) property escape honoring the `i` flag. The
/// base predicate is `prop.matches(x) ^ neg` (so `\P{…}` negates the set first).
/// Under `i`, `c` matches if any of its simple case variants satisfies that
/// predicate — the spec's `Canonicalize`-based CharacterClass matching. Applying
/// `neg` *before* the variant search is what makes `\P{Lu}` match `A` under `i`
/// (the lowercase variant `a` is not `Lu`, so it is in the negated set).
fn prop_matches_ci(prop: &super::props::PropEscape, neg: bool, c: u32, flags: Flags) -> bool {
    let base = |x: u32| prop.matches(x) ^ neg;
    if base(c) {
        return true;
    }
    if !flags.ignore_case {
        return false;
    }
    let Some(ch) = char::from_u32(c) else {
        return false;
    };
    for v in case_variants(ch) {
        if v != c && base(v) {
            return true;
        }
    }
    false
}

/// The simple case variants of `ch` to test for case-insensitive set membership:
/// its ASCII/simple upper- and lowercase, and (with the `intl` feature) the
/// targets that share its case fold. Returns an iterator of code points.
fn case_variants(ch: char) -> impl Iterator<Item = u32> {
    let mut out: Vec<u32> = Vec::new();
    for u in ch.to_uppercase() {
        out.push(u as u32);
    }
    for l in ch.to_lowercase() {
        out.push(l as u32);
    }
    #[cfg(feature = "intl")]
    {
        // Add code points whose simple case fold equals `ch`'s, catching pairs the
        // plain to_upper/to_lower miss (ſ↔s, K↔k, ς↔σ, …).
        let target: Vec<char> = intl::unicode::case::case_fold(ch).collect();
        // The common single-scalar fold case: scan the BMP letters cheaply is too
        // costly; instead rely on the fact that to_uppercase/to_lowercase plus the
        // fold of `ch` itself cover the ECMAScript-relevant variants. Add the fold
        // result directly so `ch` matching a folded target works both ways.
        if target.len() == 1 {
            out.push(target[0] as u32);
        }
    }
    out.into_iter()
}

fn is_ascii_digit(c: u32) -> bool {
    (0x30..=0x39).contains(&c)
}

fn is_space(c: u32) -> bool {
    char::from_u32(c).is_some_and(|ch| ch.is_whitespace())
}

fn assert_ok(assert: &Assert, input: &[u16], sp: usize, flags: Flags) -> bool {
    match assert {
        Assert::Start => sp == 0 || (flags.multiline && is_line_term(input[sp - 1] as u32)),
        Assert::End => sp == input.len() || (flags.multiline && is_line_term(input[sp] as u32)),
        Assert::WordBoundary => is_boundary(input, sp, flags),
        Assert::NotWordBoundary => !is_boundary(input, sp, flags),
    }
}

fn is_boundary(input: &[u16], sp: usize, flags: Flags) -> bool {
    let before = sp > 0 && is_word_flags(input[sp - 1] as u32, flags);
    let after = sp < input.len() && is_word_flags(input[sp] as u32, flags);
    before != after
}
