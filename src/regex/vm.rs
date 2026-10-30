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
/// start positions via `run_with`, so the bound covers the whole operation,
/// not each start independently (RE-8).
const STEP_BASE: u64 = crate::limits::DEFAULT_REGEX_STEP_BASE;
const STEP_PER_CHAR: u64 = 1_000;

/// Maximum lookaround nesting depth of `backtrack` before the match aborts
/// cleanly. Choice points live on an explicit heap stack, so the executor
/// recurses only to run a lookaround sub-program; the pattern parser's nesting
/// cap already keeps that far below this backstop.
const MAX_DEPTH: u32 = crate::limits::DEFAULT_REGEX_MAX_DEPTH;

/// Maximum length of the executor's backtrack stack (choice points plus undo
/// records) before the match aborts cleanly, bounding its memory on an
/// adversarial subject (one frame is a few dozen bytes).
const MAX_STACK: usize = crate::limits::DEFAULT_REGEX_MAX_BACKTRACK_STACK;

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
    /// A named backreference `\k<name>` where `name` is a **duplicate** capture
    /// name (ES2025): references whichever of the listed groups participated (has a
    /// recorded span); if none is set it matches the empty string.
    BackrefMulti(alloc::vec::Vec<usize>),
    /// Reset the capture slots `from..=to` before an iteration of a quantified atom
    /// (ECMA-262 RepeatMatcher clears the atom's parentheses each repetition).
    /// Backtrack-safe: the prior values are restored if the iteration fails.
    ClearCaptures { from: usize, to: usize },
    /// Record the position at which an *optional* (min = 0) quantifier iteration
    /// begins, for the paired [`Inst::EmptyFail`]. Emitted only when the quantified
    /// atom can match the empty string.
    Mark,
    /// ECMA-262 RepeatMatcher step 2.b: *"If min = 0 and y's endIndex = x's
    /// endIndex, return failure."* Pops the position pushed by the matching
    /// [`Inst::Mark`] and fails the iteration if it consumed nothing, so the body
    /// backtracks into its other alternatives instead of the loop silently exiting
    /// with the empty iteration's captures still bound.
    EmptyFail,
    /// An instruction that never matches. Emitted for a quantifier whose minimum
    /// count provably cannot be satisfied by any subject the engine can hold (see
    /// `compile_repeat`).
    Fail,
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
    /// Membership (negation applied) of every Latin-1 code point when the `i`
    /// flag is off, one bit each: without case folding a member's match does
    /// not depend on the flags, so the per-character test for the commonest
    /// subjects is one bit lookup instead of a walk over `members`.
    latin1: [u64; 4],
}

impl Class {
    /// A class over `members` (negated when `neg`).
    pub(crate) fn new(neg: bool, members: Vec<ClassMember>) -> Self {
        let mut latin1 = [0u64; 4];
        for c in 0u32..256 {
            if leaf_members_match(&members, c, Flags::default()) ^ neg {
                latin1[(c >> 6) as usize] |= 1u64 << (c & 63);
            }
        }
        Class {
            neg,
            members,
            latin1,
        }
    }
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
                (c >= *lo && c <= *hi) || (flags.ignore_case && range_fold_hit(c, *lo, *hi, flags))
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

/// Reads the code point that **ends** at unit index `sp` (i.e. the character
/// immediately to the left of `sp`), returning it and the number of code units it
/// spans, for right-to-left (lookbehind) matching. In `u` mode a well-formed
/// surrogate pair ending at `sp` decodes to one code point (length 2); otherwise
/// the single unit at `sp - 1` is returned. `None` at the left edge (`sp == 0`).
fn read_cp_back(input: &[u16], sp: usize, unicode: bool) -> Option<(u32, usize)> {
    if sp == 0 {
        return None;
    }
    let lo = *input.get(sp - 1)? as u32;
    if unicode && (0xDC00..=0xDFFF).contains(&lo) && sp >= 2 {
        let hi = input[sp - 2] as u32;
        if (0xD800..=0xDBFF).contains(&hi) {
            let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
            return Some((cp, 2));
        }
    }
    Some((lo, 1))
}

/// Reads one character in the current match direction, returning its code point
/// and the resulting position. Forward: the character starting at `sp`, advancing
/// to `sp + len`. Reverse: the character ending at `sp`, retreating to `sp - len`.
fn read_dir(input: &[u16], sp: usize, unicode: bool, reverse: bool) -> Option<(u32, usize)> {
    if reverse {
        let (cp, len) = read_cp_back(input, sp, unicode)?;
        Some((cp, sp - len))
    } else {
        let (cp, len) = read_cp(input, sp, unicode)?;
        Some((cp, sp + len))
    }
}

/// The per-attempt working buffers, owned by the caller so one scan reuses them
/// across start positions instead of allocating four vectors per offset — an
/// attempt that fails on its first instruction still paid for all four.
#[derive(Default)]
pub(crate) struct Scratch {
    saves: Vec<Option<usize>>,
    marks: Vec<usize>,
    flag_stack: Vec<Flags>,
    stack: BtStack,
}

/// Runs `prog` against `input` (UTF-16 code units) starting at unit index
/// `start`, returning the capture slots (`2 * (group_count + 1)` of them, as
/// `(start, end)` unit-index pairs) on success. `scratch` is reset per attempt.
///
/// Threads a caller-owned step counter and budget so a multi-start find
/// ([`super::Regex::captures_at`]) shares one budget across all start positions
/// instead of resetting it per start (RE-8).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_with(
    scratch: &mut Scratch,
    prog: &[Inst],
    input: &[u16],
    start: usize,
    group_count: usize,
    flags: Flags,
    steps: &Cell<u64>,
    budget: u64,
) -> Option<Vec<Option<(usize, usize)>>> {
    let Scratch {
        saves,
        marks,
        flag_stack,
        stack,
    } = scratch;
    saves.clear();
    saves.resize(2 * (group_count + 1), None);
    marks.clear();
    // The flag stack: the base flags sit at the bottom; each entered
    // inline-modifier scope (`(?ims-ims:…)`) pushes the locally-adjusted flags.
    flag_stack.clear();
    flag_stack.push(flags);
    let ctx = Ctx {
        prog,
        input,
        flags,
        match_end: None,
        reverse: false,
        steps,
        budget,
    };
    if backtrack(&ctx, 0, start, saves, 0, marks, flag_stack, stack) {
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
    /// When `Some(p)`, `Match` succeeds only at position `p`. Currently unused by
    /// the reverse-lookbehind path (which imposes no end constraint) but retained
    /// for the forward sub-program contract.
    match_end: Option<usize>,
    /// Whether this context consumes the subject **right-to-left**. Set for a
    /// lookbehind sub-program (which is compiled in reverse): consuming
    /// instructions read the character *ending* at `sp` and decrement `sp`, and
    /// backreferences compare backward. Every other context runs forward.
    reverse: bool,
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

/// The backtracking executor's explicit stack. `frames` holds choice points
/// and undo records; variable-length payloads (saved capture slots, recorded
/// loop positions) live on the two side stacks in the same LIFO order, so a
/// [`Frame`] stays `Copy` and unwinding or clearing never runs drop glue.
#[derive(Default)]
pub(crate) struct BtStack {
    frames: Vec<Frame>,
    slots: Vec<Option<usize>>,
    positions: Vec<usize>,
}

/// One entry of the backtracking executor's explicit stack: either a choice
/// point to resume from, or the undo record of a state change made since the
/// choice point below it (undone, in order, while unwinding to that choice).
#[derive(Clone, Copy)]
enum Frame {
    /// The untried branch of a `Split`: resume at `pc` with position `sp`.
    Alt { pc: usize, sp: usize },
    /// Undo a `Save`: restore capture slot `slot` to `old`.
    Save { slot: usize, old: Option<usize> },
    /// Undo a `ClearCaptures`: restore the `n` slots starting at `lo` from the
    /// top of the slot side stack.
    Clear { lo: usize, n: usize },
    /// Undo a positive lookaround's adopted captures: restore every slot from
    /// the top of the slot side stack.
    Saves,
    /// Undo a `Mark`: pop the position it pushed.
    MarkPop,
    /// Undo an `EmptyFail`: push back the position it popped.
    MarkPush(usize),
    /// Undo a `PushFlags`: pop the scope's flags.
    FlagsPop,
    /// Undo a `PopFlags`: push back the scope's flags.
    FlagsPush(Flags),
    /// A greedy single-unit loop (outside `u` mode) that consumed the run
    /// `first..=p` (or `p..=first` in reverse): the continuation at `cont` is
    /// next tried at `p`, then at each position backing off toward `first`.
    Run {
        cont: usize,
        first: usize,
        p: usize,
        flags: Flags,
    },
    /// A greedy single-consume loop in `u` mode: the continuation at `cont` is
    /// next tried at the top of the position side stack, longest first, down
    /// to its entry `base`.
    Positions { cont: usize, base: usize },
    /// A lazy single-consume loop whose continuation at `cont` failed at `sp`:
    /// consume one more character with the instruction at `consume`, then retry.
    Lazy {
        consume: usize,
        cont: usize,
        sp: usize,
        flags: Flags,
    },
}

/// The backtracking executor. `pc` is the program counter, `sp` the position in
/// the input (a code-unit index), `depth` the lookaround nesting depth. `saves`
/// holds raw capture positions.
///
/// Choice points live on an explicit heap stack ([`Frame`]) rather than the
/// Rust call stack, so a quantified group iterated many thousands of times
/// (`/(?:ab)+c/` over a long subject) neither overflows the native stack nor
/// hits a recursion cap — that cap used to end such a match as a silent "no
/// match". Only lookaround sub-programs recurse, so the native depth follows
/// pattern nesting, not input length.
///
/// Returns `false` (clean no-match) if the step budget, the backtrack-stack cap
/// or the lookaround-depth cap is exceeded, so an adversarial pattern can never
/// hang, exhaust memory or overflow the stack.
#[allow(clippy::too_many_arguments)]
fn backtrack(
    ctx: &Ctx,
    mut pc: usize,
    mut sp: usize,
    saves: &mut Vec<Option<usize>>,
    depth: u32,
    // The positions at which the *optional* quantifier iterations enclosing this
    // point began, pushed by [`Inst::Mark`] and consumed by [`Inst::EmptyFail`].
    marks: &mut Vec<usize>,
    // The flag stack: the top entry is the flags in effect at this point, after
    // applying any enclosing inline-modifier scopes. `unicode` never changes
    // (only `i`/`m`/`s` can be modified), so it is read once below.
    flags_stack: &mut Vec<Flags>,
    // The backtrack stack, caller-owned so a scan reuses one allocation across
    // start positions; cleared on entry.
    bt: &mut BtStack,
) -> bool {
    // Lookaround-nesting cap: abort cleanly before we can overflow the stack.
    if depth > MAX_DEPTH {
        return false;
    }
    let unicode = ctx.flags.unicode;
    let BtStack {
        frames: stack,
        slots,
        positions,
    } = bt;
    stack.clear();
    slots.clear();
    positions.clear();
    // The flags in effect for the current instruction: the flag-stack top, kept
    // in step with every push and pop of that stack.
    let mut cur = *flags_stack.last().unwrap_or(&ctx.flags);
    loop {
        // Run forward from (pc, sp) until the path fails (`break`) or matches.
        loop {
            // Step backstop: every instruction visited counts toward the budget;
            // once exhausted the whole match unwinds as a no-match.
            if !ctx.tick() {
                return false;
            }
            match &ctx.prog[pc] {
                Inst::Match => {
                    if ctx.match_end.is_none_or(|p| sp == p) {
                        return true;
                    }
                    break;
                }
                Inst::Char(c) => match read_dir(ctx.input, sp, unicode, ctx.reverse) {
                    Some((cp, nsp)) if cp_eq(cp, *c, cur) => {
                        sp = nsp;
                        pc += 1;
                    }
                    _ => break,
                },
                Inst::Any => match read_dir(ctx.input, sp, unicode, ctx.reverse) {
                    Some((cp, nsp)) if cur.dotall || !is_line_term(cp) => {
                        sp = nsp;
                        pc += 1;
                    }
                    _ => break,
                },
                Inst::Class(class) => match read_dir(ctx.input, sp, unicode, ctx.reverse) {
                    Some((cp, nsp)) if class_matches(class, cp, cur) => {
                        sp = nsp;
                        pc += 1;
                    }
                    _ => break,
                },
                Inst::ClassSet { neg, matcher } => {
                    match read_dir(ctx.input, sp, unicode, ctx.reverse) {
                        Some((cp, nsp)) if matcher.matches(cp, cur) ^ *neg => {
                            sp = nsp;
                            pc += 1;
                        }
                        _ => break,
                    }
                }
                Inst::Jmp(target) => pc = *target,
                Inst::Split(a, b) => {
                    // Fast path for a *simple* quantifier loop whose body is a single
                    // consuming instruction (`a*`, `.*`, `\d+`, `[…]*`, …): consume
                    // the whole run iteratively and leave one frame that backs off
                    // through it, instead of one choice point per repetition. A
                    // single-consume body always advances, so it can never be
                    // zero-width — no loop-stack bookkeeping is needed here.
                    if let Some((consume_pc, cont_pc, greedy)) = simple_loop(ctx.prog, pc) {
                        if greedy && !unicode {
                            // Outside `u` mode every consume is exactly one code
                            // unit, so the candidate positions are the contiguous
                            // run from the start to the furthest point reached.
                            let first = sp;
                            while let Some(nsp) =
                                consume_one(&ctx.prog[consume_pc], ctx.input, sp, cur, ctx.reverse)
                            {
                                sp = nsp;
                                if !ctx.tick() {
                                    return false;
                                }
                            }
                            stack.push(Frame::Run {
                                cont: cont_pc,
                                first,
                                p: sp,
                                flags: cur,
                            });
                            break;
                        }
                        if greedy {
                            // Greedy: consume as many as possible, recording each
                            // position, then try the continuation from longest down.
                            let base = positions.len();
                            positions.push(sp);
                            while let Some(nsp) =
                                consume_one(&ctx.prog[consume_pc], ctx.input, sp, cur, ctx.reverse)
                            {
                                sp = nsp;
                                positions.push(sp);
                                if !ctx.tick() {
                                    return false;
                                }
                            }
                            stack.push(Frame::Positions {
                                cont: cont_pc,
                                base,
                            });
                            break;
                        }
                        // Lazy: try the continuation first at each length, growing.
                        stack.push(Frame::Lazy {
                            consume: consume_pc,
                            cont: cont_pc,
                            sp,
                            flags: cur,
                        });
                        pc = cont_pc;
                        continue;
                    }

                    // An ordinary Split — an alternation, a `?`, or a `*`/`+` loop
                    // head. The first target is always the preferred branch (the
                    // compiler puts the body first for a greedy quantifier and the
                    // exit first for a lazy one), so trying `a` before `b` gives the
                    // right preference order for every construct. Zero-width loop
                    // iterations are stopped by the `Mark`/`EmptyFail` pair the
                    // compiler wraps around a nullable quantifier body, not here.
                    // Every unbounded growth of the stack passes through here (a
                    // loop iteration takes a `Split`), so the cap is checked here.
                    if stack.len() > MAX_STACK {
                        return false;
                    }
                    stack.push(Frame::Alt { pc: *b, sp });
                    pc = *a;
                }
                Inst::Save(slot) => {
                    stack.push(Frame::Save {
                        slot: *slot,
                        old: saves[*slot],
                    });
                    saves[*slot] = Some(sp);
                    pc += 1;
                }
                Inst::ClearCaptures { from, to } => {
                    let lo = (*from).min(saves.len());
                    let hi = (*to + 1).min(saves.len());
                    slots.extend_from_slice(&saves[lo..hi]);
                    stack.push(Frame::Clear { lo, n: hi - lo });
                    for s in &mut saves[lo..hi] {
                        *s = None;
                    }
                    pc += 1;
                }
                Inst::Mark => {
                    // Remember where this optional iteration starts; the paired
                    // `EmptyFail` pops it.
                    marks.push(sp);
                    stack.push(Frame::MarkPop);
                    pc += 1;
                }
                Inst::EmptyFail => {
                    // RepeatMatcher step 2.b. Pop first so the continuation (which
                    // may start the next iteration, or leave an enclosing
                    // quantifier) sees the enclosing mark on top.
                    match marks.pop() {
                        None => pc += 1,
                        Some(mark) if sp == mark => {
                            marks.push(mark);
                            break;
                        }
                        Some(mark) => {
                            stack.push(Frame::MarkPush(mark));
                            pc += 1;
                        }
                    }
                }
                Inst::Fail => break,
                Inst::BackrefMulti(groups) => {
                    let span = groups.iter().find_map(|g| {
                        match (
                            saves.get(2 * g).copied().flatten(),
                            saves.get(2 * g + 1).copied().flatten(),
                        ) {
                            (Some(s), Some(e)) => Some((s, e)),
                            _ => None,
                        }
                    });
                    match span {
                        Some((s, e)) => match match_backref(ctx, sp, s, e, cur) {
                            Some(nsp) => {
                                sp = nsp;
                                pc += 1;
                            }
                            None => break,
                        },
                        None => pc += 1,
                    }
                }
                Inst::Look { neg, prog } | Inst::LookBehind { neg, prog } => {
                    // Zero-width: run the sub-program at `sp`. A lookahead always
                    // matches forward, even inside a lookbehind; a lookbehind's body
                    // is compiled *in reverse* (ECMAScript direction `-1`), so it
                    // consumes the subject backward from `sp` and succeeds when it
                    // reaches `Match` at some start position `<= sp` (greedy
                    // quantifiers bind from the right and groups record their
                    // rightmost iteration). A *positive* assertion propagates the
                    // groups its sub-match captured to the outer match
                    // (`/(?=(\d+))/.exec("123")` captures `"123"`); a *negative* one
                    // contributes nothing. The sub-saves are seeded from the current
                    // saves so groups already bound outside (and backrefs to them)
                    // stay visible inside the assertion. The assertion is atomic:
                    // its sub-match's own choice points are not revisited.
                    let sub = Ctx {
                        prog,
                        input: ctx.input,
                        flags: cur,
                        match_end: None,
                        reverse: matches!(&ctx.prog[pc], Inst::LookBehind { .. }),
                        steps: ctx.steps,
                        budget: ctx.budget,
                    };
                    let mut sub_saves = saves.clone();
                    let mut sub_marks = Vec::new();
                    // The sub-program starts under the flags in effect at the
                    // assertion (so an enclosing `(?i:…(?=…)…)` is honored).
                    let mut sub_flags = alloc::vec![cur];
                    let matched = backtrack(
                        &sub,
                        0,
                        sp,
                        &mut sub_saves,
                        depth + 1,
                        &mut sub_marks,
                        &mut sub_flags,
                        &mut BtStack::default(),
                    );
                    if matched == *neg {
                        break;
                    }
                    if !*neg {
                        // Positive: adopt the sub-match's captures; a later failure
                        // restores the originals so backtracking past the assertion
                        // is sound.
                        slots.extend_from_slice(saves);
                        *saves = sub_saves;
                        stack.push(Frame::Saves);
                    }
                    pc += 1;
                }
                Inst::Backref(g) => {
                    match (
                        saves.get(2 * g).copied().flatten(),
                        saves.get(2 * g + 1).copied().flatten(),
                    ) {
                        // Backrefs compare raw code units (a captured span is a run
                        // of units); case folding is applied unit-by-unit.
                        (Some(s), Some(e)) => match match_backref(ctx, sp, s, e, cur) {
                            Some(nsp) => {
                                sp = nsp;
                                pc += 1;
                            }
                            None => break,
                        },
                        // An unmatched group backreference matches the empty string.
                        _ => pc += 1,
                    }
                }
                Inst::Assert(assert) => {
                    if assert_ok(assert, ctx.input, sp, cur) {
                        pc += 1;
                    } else {
                        break;
                    }
                }
                Inst::PushFlags(delta) => {
                    // Enter an inline-modifier scope; unwinding past this point
                    // pops it again.
                    cur = delta.apply(cur);
                    flags_stack.push(cur);
                    stack.push(Frame::FlagsPop);
                    pc += 1;
                }
                Inst::PopFlags => {
                    // Leave the innermost modifier scope; unwinding past this
                    // point re-enters it.
                    if let Some(f) = flags_stack.pop() {
                        stack.push(Frame::FlagsPush(f));
                    }
                    cur = *flags_stack.last().unwrap_or(&ctx.flags);
                    pc += 1;
                }
            }
        }

        // The path failed: unwind to the most recent choice point, undoing every
        // state change made since it, and resume there.
        'unwind: loop {
            let Some(frame) = stack.pop() else {
                return false;
            };
            match frame {
                Frame::Alt { pc: p, sp: s } => {
                    pc = p;
                    sp = s;
                    break 'unwind;
                }
                Frame::Save { slot, old } => saves[slot] = old,
                Frame::Clear { lo, n } => {
                    let from = slots.len() - n;
                    saves[lo..lo + n].copy_from_slice(&slots[from..]);
                    slots.truncate(from);
                }
                Frame::Saves => {
                    let from = slots.len() - saves.len();
                    saves.copy_from_slice(&slots[from..]);
                    slots.truncate(from);
                }
                Frame::MarkPop => {
                    marks.pop();
                }
                Frame::MarkPush(m) => marks.push(m),
                Frame::FlagsPop => {
                    flags_stack.pop();
                    cur = *flags_stack.last().unwrap_or(&ctx.flags);
                }
                Frame::FlagsPush(f) => {
                    flags_stack.push(f);
                    cur = f;
                }
                Frame::Run {
                    cont,
                    first,
                    mut p,
                    flags,
                } => loop {
                    // A continuation that must begin with a literal unit is only
                    // entered where that unit is.
                    let candidate = !continuation_cannot_start(ctx, cont, p, flags);
                    let at_first = p == first;
                    let here = p;
                    if !at_first {
                        p = if ctx.reverse { p + 1 } else { p - 1 };
                    }
                    if candidate {
                        if !at_first {
                            stack.push(Frame::Run {
                                cont,
                                first,
                                p,
                                flags,
                            });
                        }
                        pc = cont;
                        sp = here;
                        break 'unwind;
                    }
                    if !ctx.tick() {
                        return false;
                    }
                    if at_first {
                        break;
                    }
                },
                Frame::Positions { cont, base } => {
                    if let Some(p) = positions.pop() {
                        if positions.len() > base {
                            stack.push(Frame::Positions { cont, base });
                        }
                        pc = cont;
                        sp = p;
                        break 'unwind;
                    }
                }
                Frame::Lazy {
                    consume,
                    cont,
                    sp: at,
                    flags,
                } => {
                    if let Some(nsp) =
                        consume_one(&ctx.prog[consume], ctx.input, at, flags, ctx.reverse)
                    {
                        stack.push(Frame::Lazy {
                            consume,
                            cont,
                            sp: nsp,
                            flags,
                        });
                        pc = cont;
                        sp = nsp;
                        break 'unwind;
                    }
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

/// Whether the continuation at `pc` certainly fails at `sp` (non-`u`, forward,
/// case-sensitive): after any capture saves, which consume nothing, its first
/// instruction is a literal BMP character that is not the unit at `sp`. A cheap
/// pre-check so a greedy loop does not enter a doomed continuation at every
/// position it backs off to.
fn continuation_cannot_start(ctx: &Ctx, mut pc: usize, sp: usize, flags: Flags) -> bool {
    if ctx.reverse || flags.ignore_case || flags.unicode {
        return false;
    }
    while let Some(Inst::Save(_)) = ctx.prog.get(pc) {
        pc += 1;
    }
    match ctx.prog.get(pc) {
        Some(Inst::Char(c)) if *c <= 0xFFFF => {
            ctx.input.get(sp).is_none_or(|&u| u32::from(u) != *c)
        }
        _ => false,
    }
}

/// Whether `inst` consumes exactly one character (1 or 2 code units) on success.
fn is_single_consume(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. }
    )
}

/// Tries to consume one character with a single-consume instruction at `sp` in the
/// current direction, returning the resulting position (`sp ± len`) on success.
fn consume_one(
    inst: &Inst,
    input: &[u16],
    sp: usize,
    flags: Flags,
    reverse: bool,
) -> Option<usize> {
    let (cp, nsp) = read_dir(input, sp, flags.unicode, reverse)?;
    single_consume_matches(inst, cp, flags).then_some(nsp)
}

/// Whether a single-consume `inst` accepts code point `cp` under `flags`.
///
/// Split out of [`consume_one`] so the start-of-match filter
/// (`Regex::start_set`) can be built from the *same* predicate the matcher
/// applies, rather than a parallel reimplementation that could disagree and
/// skip an offset that would have matched.
pub(crate) fn single_consume_matches(inst: &Inst, cp: u32, flags: Flags) -> bool {
    match inst {
        Inst::Char(c) => cp_eq(cp, *c, flags),
        Inst::Any => flags.dotall || !is_line_term(cp),
        Inst::Class(class) => class_matches(class, cp, flags),
        Inst::ClassSet { neg, matcher } => matcher.matches(cp, flags) ^ *neg,
        _ => false,
    }
}

/// Matches a backreference to the captured span `[s, e)` at position `sp` in the
/// context's direction, comparing code units (case-folded under `i`). Forward:
/// consumes `input[sp..sp+len]`, returning `sp + len`. Reverse: consumes
/// `input[sp-len..sp]`, returning `sp - len`. `None` if it does not match or would
/// run off the subject.
fn match_backref(ctx: &Ctx, sp: usize, s: usize, e: usize, flags: Flags) -> Option<usize> {
    let len = e - s;
    // Under `u`/`v` a backreference matches a list of *code points*, so the span it
    // consumes may not stop in the middle of a surrogate pair: a captured lone lead
    // must not match the lead half of a real pair (`/foo(.+)bar\1/u` on
    // `"foo\uD834bar\uD834\uDC00"` is `null`).
    let cp_ok = |end: usize| {
        !(flags.unicode || flags.unicode_sets) || is_code_point_boundary(ctx.input, end)
    };
    if ctx.reverse {
        if sp >= len
            && (0..len).all(|i| unit_eq(ctx.input[sp - len + i], ctx.input[s + i], flags))
            && cp_ok(sp - len)
        {
            Some(sp - len)
        } else {
            None
        }
    } else if sp + len <= ctx.input.len()
        && (0..len).all(|i| unit_eq(ctx.input[sp + i], ctx.input[s + i], flags))
        && cp_ok(sp + len)
    {
        Some(sp + len)
    } else {
        None
    }
}

/// Whether `i` is a code-point boundary in `units` — i.e. not wedged between a
/// lead surrogate and its trail.
fn is_code_point_boundary(units: &[u16], i: usize) -> bool {
    i == 0
        || i >= units.len()
        || !((0xD800..=0xDBFF).contains(&units[i - 1]) && (0xDC00..=0xDFFF).contains(&units[i]))
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
        (Some(ca), Some(cb)) => char_fold_eq(ca, cb, flags),
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
        (Some(ca), Some(cb)) => char_fold_eq(ca, cb, flags),
        _ => false,
    }
}

fn char_fold_eq(a: char, b: char, flags: Flags) -> bool {
    // §22.2.2.9.1 `Canonicalize`. Under `u`/`v` it is Unicode simple case folding,
    // which catches pairs simple lowercasing misses (the Kelvin sign U+212A ↔ `k`,
    // long s U+017F ↔ `s`, final sigma ς ↔ σ). WITHOUT those flags it is the
    // legacy `toUppercase` rule, which deliberately does *not* fold a non-ASCII
    // character onto an ASCII one — so `/K/i` must not match `k`.
    if !(flags.unicode || flags.unicode_sets) {
        return canonicalize_legacy(a) == canonicalize_legacy(b);
    }
    #[cfg(feature = "intl")]
    {
        intl::unicode::case::case_fold(a).eq(intl::unicode::case::case_fold(b))
    }
    #[cfg(not(feature = "intl"))]
    {
        a.eq_ignore_ascii_case(&b) || a.to_lowercase().eq(b.to_lowercase())
    }
}

/// `Canonicalize(ch)` for a non-`u`/`v` case-insensitive pattern: uppercase the
/// character, keep the result only if it is a *single* UTF-16 code unit (so `ß`
/// → "SS" is rejected), and never map a non-ASCII character onto an ASCII one.
fn canonicalize_legacy(ch: char) -> char {
    let mut up = ch.to_uppercase();
    let Some(first) = up.next() else {
        return ch;
    };
    if up.next().is_some() || first.len_utf16() != 1 {
        return ch;
    }
    if ch as u32 >= 128 && (first as u32) < 128 {
        return ch;
    }
    first
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
    if c < 256 && !flags.ignore_case {
        return class.latin1[(c >> 6) as usize] & (1u64 << (c & 63)) != 0;
    }
    leaf_members_match(&class.members, c, flags) ^ class.neg
}

/// Case-insensitive class-range membership: `c` matches `[lo-hi]` under `i` when
/// some code point in the range canonicalizes to the same thing `c` does. The
/// search runs over `c`'s own case variants (the same approximation
/// `prop_matches_ci` uses) and keeps only those that really share `c`'s
/// canonical form under the active flags — so `/[a-z]/iu` matches the Kelvin
/// sign while the legacy (non-`u`) `/[a-z]/i` does not.
///
/// ASCII takes a fast path: an ASCII `c`'s canonical partners are its two ASCII
/// case variants, plus — under `u`/`v` only — the long s U+017F for `s`/`S` and
/// the Kelvin sign U+212A for `k`/`K` (`/[\u0100-\u017F]/iu` matches `s`). This
/// predicate runs per character on the matching hot path.
fn range_fold_hit(c: u32, lo: u32, hi: u32, flags: Flags) -> bool {
    let Some(ch) = char::from_u32(c) else {
        return false;
    };
    if ch.is_ascii() {
        let cl = ch.to_ascii_lowercase() as u32;
        let cu = ch.to_ascii_uppercase() as u32;
        if (cl >= lo && cl <= hi) || (cu >= lo && cu <= hi) {
            return true;
        }
        let extra = match cl {
            0x6B => 0x212A,
            0x73 => 0x017F,
            _ => return false,
        };
        return (flags.unicode || flags.unicode_sets) && extra >= lo && extra <= hi;
    }
    case_variants(ch, flags).any(|v| {
        v >= lo && v <= hi && char::from_u32(v).is_some_and(|cv| char_fold_eq(cv, ch, flags))
    })
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
///
/// The `v` flag changes the order: there the escape's set is first case-folded
/// (`MaybeSimpleCaseFolding`) and `\P{…}` is its `CharacterComplement` within
/// the fold-stable code points, so under `vi` `c` matches `\P{…}` exactly when
/// it does *not* match `\p{…}` case-insensitively (`/\P{Ll}/vi` rejects `A`).
fn prop_matches_ci(prop: &super::props::PropEscape, neg: bool, c: u32, flags: Flags) -> bool {
    if neg && flags.ignore_case && flags.unicode_sets {
        return !prop_matches_ci(prop, false, c, flags);
    }
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
    for v in case_variants(ch, flags) {
        if v != c && base(v) {
            return true;
        }
    }
    false
}

/// `scf(c)`: Unicode simple case folding, the `u`/`v`-mode `Canonicalize`.
fn simple_fold(c: u32) -> u32 {
    let tab = super::fold_data::SCF_BY_SOURCE;
    match tab.binary_search_by_key(&c, |&(s, _)| s) {
        Ok(i) => tab[i].1,
        Err(_) => c,
    }
}

/// Every code point `x` with `scf(x) == scf(c)` (including `c`): the members a
/// `u`/`v`-mode class may contain for `c` to match it case-insensitively.
fn simple_fold_class(c: u32) -> Vec<u32> {
    let t = simple_fold(c);
    let tab = super::fold_data::SCF_BY_TARGET;
    let start = tab.partition_point(|&(_, d)| d < t);
    let mut out = Vec::with_capacity(4);
    out.push(t);
    out.extend(
        tab[start..]
            .iter()
            .take_while(|&&(_, d)| d == t)
            .map(|&(s, _)| s),
    );
    out
}

/// The case variants of `ch` to test for case-insensitive set membership.
///
/// Under `u`/`v` that is exactly `ch`'s simple-case-folding equivalence class
/// (so `ß`, `İ` and `ı`, which fold only to themselves, have no variants,
/// while `ǅ` reaches both `Ǆ` and `ǆ` and `k` reaches the Kelvin sign). The
/// legacy mode's candidates are that class plus `ch`'s own upper- and
/// lowercase mappings; callers filter them by the legacy `Canonicalize`.
/// Returns an iterator of code points.
fn case_variants(ch: char, flags: Flags) -> impl Iterator<Item = u32> {
    if flags.unicode || flags.unicode_sets {
        return simple_fold_class(ch as u32).into_iter();
    }
    // The legacy `Canonicalize` (single-unit uppercase) partners of `ch` all
    // lie in its simple-fold class or among its own case mappings — e.g. `β`
    // and `ϐ` both uppercase to `Β`, which neither mapping of `β` reaches.
    let mut out = simple_fold_class(ch as u32);
    out.extend(ch.to_uppercase().map(|u| u as u32));
    out.extend(ch.to_lowercase().map(|l| l as u32));
    out.into_iter()
}

fn is_ascii_digit(c: u32) -> bool {
    (0x30..=0x39).contains(&c)
}

/// `\s` per §22.2.2.9 `CharacterClassEscape :: s` — *WhiteSpace* ∪
/// *LineTerminator*, i.e. TAB/VT/FF/ZWNBSP plus the `Zs` category (*USP*) plus
/// LF/CR/LS/PS. This is deliberately NOT Unicode `White_Space`: that property
/// includes U+0085 (NEL), which ECMAScript excludes, and omits U+FEFF, which
/// ECMAScript includes. The set is closed-form and version-stable except for
/// `Zs`, which has not changed since Unicode 6.3.
fn is_space(c: u32) -> bool {
    matches!(
        c,
        0x09..=0x0D          // TAB, LF, VT, FF, CR
            | 0x20           // SPACE (Zs)
            | 0xA0           // NBSP (Zs)
            | 0x1680         // OGHAM SPACE MARK (Zs)
            | 0x2000..=0x200A// EN QUAD..HAIR SPACE (Zs)
            | 0x2028         // LINE SEPARATOR
            | 0x2029         // PARAGRAPH SEPARATOR
            | 0x202F         // NARROW NBSP (Zs)
            | 0x205F         // MEDIUM MATHEMATICAL SPACE (Zs)
            | 0x3000         // IDEOGRAPHIC SPACE (Zs)
            | 0xFEFF         // ZWNBSP
    )
}

pub(crate) fn assert_ok(assert: &Assert, input: &[u16], sp: usize, flags: Flags) -> bool {
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
