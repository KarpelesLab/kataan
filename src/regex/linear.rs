//! A fast matcher for *linear* patterns: a straight sequence of literals,
//! single-character sets, single-character quantifiers (`\d+`, `[a-z]*?`,
//! `x{2,5}`), capture-group boundaries and zero-width assertions — no
//! quantified group, no lookaround, no backreference — optionally with one
//! alternation (`^\s+|\s+$`, `(https?|ftp):`). Most regexes in real programs
//! have this shape (`/(\w+)@([\w.]+)/`, `/page\/(\d)9\d/`, `/\d+/g`).
//!
//! It accepts exactly what the backtracking VM ([`super::vm`]) accepts, in
//! the same preference order, but runs on flat step lists with an explicit,
//! bounded choice stack instead of one recursive call per instruction:
//!
//! * a sequence has one path through it, so every capture boundary is written
//!   again on every retry of the steps after a choice point — nothing has to
//!   be restored when backtracking;
//! * each quantifier step leaves at most one choice point, so the stack depth
//!   is bounded by the number of quantifier steps (known at compile time);
//! * an alternation `P(A|B)S` whose prefix `P` makes no choice is tried as the
//!   two sequences `PAS` and `PBS`, in that order: a depth-first search
//!   exhausts every way of matching `PAS` (all of `A`'s choices and all of
//!   `S`'s) before it first tries `B`, exactly as the VM's does;
//! * outside `u`/`v` mode every consume is one code unit, so a greedy step
//!   backs off by plain index decrement and set membership is a bit test for
//!   Latin-1 units; in `u`/`v` mode a consume reads a whole code point.
//!
//! The steps are derived from the compiled VM program, and every set's
//! membership is the VM's own predicate ([`vm::single_consume_matches`]),
//! tabulated for Latin-1 and called directly above it, so the two engines
//! cannot disagree about what a character class contains.

use super::Flags;
use super::vm::{self, Assert, ClassMember, Inst};
use alloc::vec::Vec;
use core::cell::Cell;

/// Membership of the code points at or above 256 in a [`Set`].
#[derive(Clone, Copy)]
enum High {
    /// No code point >= 256 is a member.
    None,
    /// Ask the VM predicate for the instruction at this program index.
    Check(usize),
    /// Every unit >= 256 may start a match (a start filter only).
    Any,
}

/// One character set: the Latin-1 bitmap plus the rule above it.
struct Set {
    latin1: [u64; 4],
    high: High,
}

impl Set {
    /// Whether code point (or, outside `u` mode, code unit) `c` is a member.
    #[inline(always)]
    fn has(&self, c: u32, prog: &[Inst], flags: Flags) -> bool {
        if c < 256 {
            return self.latin1[(c >> 6) as usize] & (1u64 << (c & 63)) != 0;
        }
        match self.high {
            High::None => false,
            High::Check(pc) => vm::single_consume_matches(&prog[pc], c, flags),
            High::Any => true,
        }
    }

    /// Whether a match whose first character is in this set may begin with
    /// code unit `u`: exact below 256; above, a `u`-mode unit may be the lead
    /// half of a pair, so anything but an empty high range is a candidate.
    #[inline(always)]
    fn may_start(&self, u: u16, prog: &[Inst], flags: Flags) -> bool {
        if u < 256 || !flags.unicode {
            return self.has(u32::from(u), prog, flags);
        }
        !matches!(self.high, High::None)
    }
}

#[derive(Clone, Copy)]
enum Step {
    /// Record the position in capture slot `n`.
    Save(usize),
    /// One exact code unit (never a surrogate in `u` mode, so the unit is a
    /// whole character there too).
    Unit(u16),
    /// One character from a set.
    One(usize),
    /// Up to `max` characters from a set (unbounded when `None`), greedy or
    /// lazy. `next_unit` is the literal the steps after it must begin with,
    /// if any, so a greedy back-off only stops where that literal is.
    Rep {
        set: usize,
        max: Option<usize>,
        greedy: bool,
        next_unit: Option<u16>,
    },
    /// A zero-width assertion (an index into [`ASSERTS`]).
    Assert(u8),
    Match,
}

const ASSERTS: [Assert; 4] = [
    Assert::Start,
    Assert::End,
    Assert::WordBoundary,
    Assert::NotWordBoundary,
];

/// A linear program; see the module docs.
pub(crate) struct Linear {
    /// The sequences tried, in order, at each start position.
    branches: Vec<Vec<Step>>,
    sets: Vec<Set>,
    /// The most quantifier steps in a branch, which bounds the choice-stack
    /// depth.
    reps: usize,
    slots: usize,
    /// What a match must begin with, to skip offsets that cannot start one.
    first: First,
    /// `(A, B, x)` when the one branch opens with `A B* x` (captures aside)
    /// and the literal `x` is outside `B`; see [`Linear::find`].
    lead_run: Option<(usize, usize, u16)>,
}

/// The unit every match begins with.
#[derive(Clone, Copy)]
enum First {
    Unit(u16),
    Set(usize),
    /// Anything, or nothing (an empty match is possible).
    Any,
}

/// The result of one attempt.
enum Outcome {
    Match,
    NoMatch,
    /// The shared step budget ran out: the whole find gives up, as the VM does.
    Abort,
}

#[derive(Clone, Copy, Default)]
struct Choice {
    /// The `Rep` step this choice belongs to.
    step: usize,
    /// The position the steps after it were last tried at.
    cur: usize,
    /// Greedy: the lowest position it may back off to. Lazy: how many more
    /// characters it may take.
    bound: usize,
}

const STACK_CHOICES: usize = 16;

/// At most this many branches (one alternation's).
const MAX_BRANCHES: usize = 16;

/// Parses the steps of `prog[pc..end]` (to the `Match` when `end` is `None`)
/// into `out`. Stops early, returning the program index, at a `Split` that is
/// not a single-character quantifier (an alternation); returns `end` (or the
/// `Match` index) when the whole range is a sequence. `None` if anything in it
/// cannot be expressed.
fn parse_seq(
    prog: &[Inst],
    mut pc: usize,
    end: Option<usize>,
    flags: Flags,
    sets: &mut Vec<Set>,
    out: &mut Vec<Step>,
) -> Option<usize> {
    let new_set = |pc: usize, sets: &mut Vec<Set>| -> usize {
        sets.push(make_set(&prog[pc], pc, flags));
        sets.len() - 1
    };
    loop {
        if Some(pc) == end {
            return Some(pc);
        }
        match prog.get(pc)? {
            Inst::Save(s) => {
                out.push(Step::Save(*s));
                pc += 1;
            }
            Inst::Char(c)
                if !flags.ignore_case
                    && *c <= 0xFFFF
                    && !(flags.unicode && (0xD800..=0xDFFF).contains(c)) =>
            {
                out.push(Step::Unit(*c as u16));
                pc += 1;
            }
            Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. } => {
                out.push(Step::One(new_set(pc, sets)));
                pc += 1;
            }
            Inst::Assert(a) => {
                out.push(Step::Assert(match a {
                    Assert::Start => 0,
                    Assert::End => 1,
                    Assert::WordBoundary => 2,
                    Assert::NotWordBoundary => 3,
                }));
                pc += 1;
            }
            Inst::Match => {
                if end.is_some() {
                    return None;
                }
                out.push(Step::Match);
                return Some(pc);
            }
            Inst::Split(a, b) => {
                let (a, b) = (*a, *b);
                // `x*` / `x*?`: `Split; x; Jmp(split)`, continuing after it.
                if is_consume(prog.get(pc + 1))
                    && matches!(prog.get(pc + 2), Some(Inst::Jmp(t)) if *t == pc)
                    && ((a == pc + 1 && b == pc + 3) || (b == pc + 1 && a == pc + 3))
                {
                    out.push(Step::Rep {
                        set: new_set(pc + 1, sets),
                        max: None,
                        greedy: a == pc + 1,
                        next_unit: None,
                    });
                    pc += 3;
                    continue;
                }
                // `x{0,k}` (and the optional tail of `x{n,m}`): a nested
                // chain `Split(body, exit); x; Split(body, exit); x; … exit:`
                // where every split shares one exit — one choice of count.
                if let Some((count, greedy, exit)) = optional_chain(prog, pc) {
                    out.push(Step::Rep {
                        set: new_set(pc + 1, sets),
                        max: Some(count),
                        greedy,
                        next_unit: None,
                    });
                    pc = exit;
                    continue;
                }
                return Some(pc);
            }
            // Quantified groups, lookaround, backreferences, inline
            // modifiers: not a sequence.
            _ => return None,
        }
    }
}

/// Recognizes the optional chain of `x{0,count}` at `pc`, returning `(count,
/// greedy, exit)`.
fn optional_chain(prog: &[Inst], pc: usize) -> Option<(usize, bool, usize)> {
    let Some(&Inst::Split(a, b)) = prog.get(pc) else {
        return None;
    };
    let greedy = a == pc + 1;
    if !greedy && b != pc + 1 {
        return None;
    }
    let exit = if greedy { b } else { a };
    let mut q = pc;
    let mut count = 0usize;
    while q < exit {
        let Some(Inst::Split(x, y)) = prog.get(q) else {
            return None;
        };
        let (body, ex) = if greedy { (*x, *y) } else { (*y, *x) };
        if body != q + 1 || ex != exit || !is_consume(prog.get(q + 1)) {
            return None;
        }
        count += 1;
        q += 2;
    }
    (q == exit && count > 0).then_some((count, greedy, exit))
}

/// The branches `(start, stop)` and the join point of the alternation whose
/// first `Split` is at `pc`: `Split(b1, L2); b1; Jmp(end); L2: Split(b2, L3);
/// b2; Jmp(end); …; Ln: bn; end:`.
fn alternation(prog: &[Inst], pc: usize) -> Option<(Vec<(usize, usize)>, usize)> {
    let mut branches = Vec::new();
    let mut at = pc;
    let mut end = None;
    loop {
        if let Some(&Inst::Split(a, b)) = prog.get(at)
            && a == at + 1
            && b > a
            && let Some(&Inst::Jmp(e)) = prog.get(b - 1)
            && e >= b
            && end.is_none_or(|x| x == e)
        {
            end = Some(e);
            branches.push((a, b - 1));
            at = b;
            if branches.len() >= MAX_BRANCHES {
                return None;
            }
            continue;
        }
        let end = end?;
        branches.push((at, end));
        return Some((branches, end));
    }
}

impl Linear {
    /// Builds the linear form of `prog` (compiled with `flags`), or `None` when
    /// the program is not one.
    pub(crate) fn build(prog: &[Inst], flags: Flags, slots: usize) -> Option<Linear> {
        let mut sets: Vec<Set> = Vec::new();
        let mut prefix = Vec::new();
        let stop = parse_seq(prog, 0, None, flags, &mut sets, &mut prefix)?;
        let mut branches = Vec::new();
        if matches!(prog.get(stop), Some(Inst::Match)) {
            branches.push(prefix);
        } else {
            // An alternation. Splitting it off the prefix is only sound when
            // the prefix makes no choice of its own (see the module docs).
            if prefix.iter().any(|s| matches!(s, Step::Rep { .. })) {
                return None;
            }
            let (alts, end) = alternation(prog, stop)?;
            let mut suffix = Vec::new();
            let fin = parse_seq(prog, end, None, flags, &mut sets, &mut suffix)?;
            if !matches!(prog.get(fin), Some(Inst::Match)) {
                return None;
            }
            for (start, stop) in alts {
                let mut steps = prefix.clone();
                // The whole branch must be a sequence (no nested alternation).
                if parse_seq(prog, start, Some(stop), flags, &mut sets, &mut steps)? != stop {
                    return None;
                }
                steps.extend_from_slice(&suffix);
                branches.push(steps);
            }
        }
        let mut reps = 0usize;
        for steps in &mut branches {
            reps = reps.max(
                steps
                    .iter()
                    .filter(|s| matches!(s, Step::Rep { .. }))
                    .count(),
            );
            // The literal a greedy step's continuation must start with.
            for i in 0..steps.len() {
                if let Step::Rep { greedy: true, .. } = steps[i] {
                    let mut j = i + 1;
                    while let Some(Step::Save(_)) = steps.get(j) {
                        j += 1;
                    }
                    let lit = match steps.get(j) {
                        Some(Step::Unit(u)) => Some(*u),
                        _ => None,
                    };
                    if let Step::Rep { next_unit, .. } = &mut steps[i] {
                        *next_unit = lit;
                    }
                }
            }
        }
        let first = start_filter(&branches, &mut sets);
        let lead_run = match branches.as_slice() {
            [only] if !flags.unicode => {
                let consuming: Vec<&Step> = only
                    .iter()
                    .filter(|s| !matches!(s, Step::Save(_)))
                    .collect();
                match consuming.as_slice() {
                    [
                        Step::One(a),
                        Step::Rep {
                            set: b,
                            max: None,
                            greedy: true,
                            ..
                        },
                        Step::Unit(x),
                        ..,
                    ] if !sets[*b].has(u32::from(*x), prog, flags) => Some((*a, *b, *x)),
                    _ => None,
                }
            }
            _ => None,
        };
        Some(Linear {
            branches,
            sets,
            reps,
            slots,
            first,
            lead_run,
        })
    }

    /// The number of capture slots an attempt writes.
    pub(crate) fn slots(&self) -> usize {
        self.slots
    }

    /// Finds the first match starting in `start..=last`, writing the raw slot
    /// positions (`usize::MAX` for an unset slot) into `saves`, which must hold
    /// [`Linear::slots`] entries. `None` means no match, including when the
    /// shared step budget runs out (the VM gives up the same way).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn find(
        &self,
        prog: &[Inst],
        units: &[u16],
        start: usize,
        last: usize,
        flags: Flags,
        saves: &mut [usize],
        steps_used: &Cell<u64>,
        budget: u64,
    ) -> Option<()> {
        let mut small = [Choice::default(); STACK_CHOICES];
        let mut big: Vec<Choice>;
        let stack: &mut [Choice] = if self.reps <= STACK_CHOICES {
            &mut small
        } else {
            big = alloc::vec![Choice::default(); self.reps];
            &mut big
        };
        let ctx = Run {
            prog,
            units,
            flags,
            steps_used,
            budget,
        };
        if flags.unicode {
            self.find_on::<true>(&ctx, start, last, saves, stack)
        } else {
            self.find_on::<false>(&ctx, start, last, saves, stack)
        }
    }

    fn find_on<const U: bool>(
        &self,
        ctx: &Run,
        start: usize,
        last: usize,
        saves: &mut [usize],
        stack: &mut [Choice],
    ) -> Option<()> {
        let (units, prog, flags) = (ctx.units, ctx.prog, ctx.flags);
        // Only offsets in `start..=last` (and inside the subject, except for
        // the empty match at its end) are candidates.
        let end = last.saturating_add(1).min(units.len());
        if !U && let Some(lead) = self.lead_run {
            return self.find_lead_run(ctx, lead, start, last, saves, stack);
        }
        let mut s = start;
        while s <= last {
            match self.first {
                First::Unit(u) => {
                    if s >= end {
                        return None;
                    }
                    s += find_unit(&units[s..end], u)?;
                }
                First::Set(k) => {
                    if s >= end {
                        return None;
                    }
                    let set = &self.sets[k];
                    s += units[s..end]
                        .iter()
                        .position(|&x| set.may_start(x, prog, flags))?;
                }
                First::Any => {}
            }
            // In `u` mode a match can only begin on a character boundary.
            if U && !is_char_boundary(units, s) {
                s += 1;
                continue;
            }
            for steps in &self.branches {
                match self.run_on::<U>(ctx, steps, s, saves, stack) {
                    Outcome::Match => return Some(()),
                    Outcome::Abort => return None,
                    Outcome::NoMatch => {}
                }
            }
            s += if U { char_len(units, s) } else { 1 };
        }
        None
    }

    /// [`find_on`](Self::find_on) outside `u` mode for a pattern that opens
    /// with `A B* x` (`x` outside `B`): every match holds `x` right after its
    /// opening run, so the scan looks for the next `x` and only then for the
    /// run in front of it, instead of starting an attempt at every unit of `A`.
    ///
    /// With `q` the first `x` after the scan position `s`, a start `p` can
    /// reach `x` only if `units[p]` is in `A` and every unit strictly between
    /// `p` and `q` is in `B`: `B*` cannot stop early (the steps after it begin
    /// with `x`, which is not in `B`), and from any `p` whose run of `B` ends
    /// before `q` the unit there is not `x`. So the candidates are the units of
    /// `A` among `r - 1 ..= q - 1`, where `r..q` is the run of `B` before `q`,
    /// and the leftmost one is the only one worth an attempt: a failed attempt
    /// from it fails from every later start up to `q` too (they all see the
    /// same steps from `q` on). The candidates come in increasing order, so the
    /// first match found is the leftmost, as in [`find_on`](Self::find_on).
    fn find_lead_run(
        &self,
        ctx: &Run,
        (a, b, x): (usize, usize, u16),
        start: usize,
        last: usize,
        saves: &mut [usize],
        stack: &mut [Choice],
    ) -> Option<()> {
        let (units, prog, flags) = (ctx.units, ctx.prog, ctx.flags);
        let (a, b) = (&self.sets[a], &self.sets[b]);
        let len = units.len();
        let mut s = start;
        while s <= last && s < len {
            // The next `x` that a start at `s` or later could reach.
            let q = s + 1 + find_unit(units.get(s + 1..)?, x)?;
            let mut r = q;
            while r > s + 1 && b.has(u32::from(units[r - 1]), prog, flags) {
                r -= 1;
            }
            let Some(p) = (r - 1..q).find(|&p| a.has(u32::from(units[p]), prog, flags)) else {
                s = q;
                continue;
            };
            if p > last {
                return None;
            }
            match self.run_on::<false>(ctx, &self.branches[0], p, saves, stack) {
                Outcome::Match => return Some(()),
                Outcome::Abort => return None,
                Outcome::NoMatch => {}
            }
            s = q.max(p + 1);
        }
        None
    }

    /// One attempt of `steps` at `start`.
    fn run_on<const U: bool>(
        &self,
        ctx: &Run,
        steps: &[Step],
        start: usize,
        saves: &mut [usize],
        stack: &mut [Choice],
    ) -> Outcome {
        let (units, prog, flags) = (ctx.units, ctx.prog, ctx.flags);
        for s in saves.iter_mut() {
            *s = usize::MAX;
        }
        let len = units.len();
        let mut depth = 0usize;
        let mut i = 0usize;
        let mut p = start;
        // Work done since the last flush into the shared counter.
        let mut work: u64 = 0;
        'run: loop {
            // Advance through the steps until one fails.
            let failed = loop {
                match steps[i] {
                    Step::Save(s) => {
                        saves[s] = p;
                        i += 1;
                    }
                    Step::Unit(u) => {
                        if p < len && units[p] == u {
                            p += 1;
                            i += 1;
                        } else {
                            break true;
                        }
                    }
                    Step::One(k) => match read::<U>(units, p) {
                        Some((c, n)) if self.sets[k].has(c, prog, flags) => {
                            p = n;
                            i += 1;
                        }
                        _ => break true,
                    },
                    Step::Rep {
                        set,
                        max,
                        greedy,
                        next_unit,
                    } => {
                        let set = &self.sets[set];
                        let lo = p;
                        if greedy {
                            if U {
                                let mut left = max.unwrap_or(usize::MAX);
                                while left > 0
                                    && let Some((c, n)) = read::<true>(units, p)
                                    && set.has(c, prog, flags)
                                {
                                    p = n;
                                    left -= 1;
                                }
                            } else {
                                let limit = max.map_or(len, |m| len.min(p.saturating_add(m)));
                                while p < limit && set.has(u32::from(units[p]), prog, flags) {
                                    p += 1;
                                }
                            }
                            work += (p - lo) as u64;
                            // Back off to where the continuation can start.
                            if let Some(c) = next_unit
                                && units.get(p) != Some(&c)
                            {
                                match units[lo..p].iter().rposition(|&u| u == c) {
                                    Some(k) => p = lo + k,
                                    None => break true,
                                }
                            }
                            stack[depth] = Choice {
                                step: i,
                                cur: p,
                                bound: lo,
                            };
                        } else {
                            stack[depth] = Choice {
                                step: i,
                                cur: p,
                                bound: max.unwrap_or(usize::MAX),
                            };
                        }
                        depth += 1;
                        i += 1;
                    }
                    Step::Assert(a) => {
                        if vm::assert_ok(&ASSERTS[usize::from(a)], units, p, flags) {
                            i += 1;
                        } else {
                            break true;
                        }
                    }
                    Step::Match => break false,
                }
            };
            if !failed {
                if work > 0 {
                    ctx.steps_used
                        .set(ctx.steps_used.get().saturating_add(work));
                }
                return Outcome::Match;
            }
            // Backtrack into the most recent choice point that has an
            // alternative left.
            loop {
                work += 1;
                if work >= 4096 {
                    let n = ctx.steps_used.get().saturating_add(work);
                    ctx.steps_used.set(n);
                    work = 0;
                    if n > ctx.budget {
                        return Outcome::Abort;
                    }
                }
                if depth == 0 {
                    let n = ctx.steps_used.get().saturating_add(work);
                    ctx.steps_used.set(n);
                    return if n > ctx.budget {
                        Outcome::Abort
                    } else {
                        Outcome::NoMatch
                    };
                }
                let ch = &mut stack[depth - 1];
                let Step::Rep {
                    set,
                    greedy,
                    next_unit,
                    ..
                } = steps[ch.step]
                else {
                    unreachable!("choice points are only pushed by Rep steps");
                };
                if greedy {
                    if ch.cur == ch.bound {
                        depth -= 1;
                        continue;
                    }
                    let np = if let Some(c) = next_unit {
                        // `c` is never a surrogate, so each occurrence starts
                        // a character of the run.
                        match units[ch.bound..ch.cur].iter().rposition(|&u| u == c) {
                            Some(k) => ch.bound + k,
                            None => {
                                depth -= 1;
                                continue;
                            }
                        }
                    } else if U
                        && ch.cur >= ch.bound + 2
                        && (0xDC00..=0xDFFF).contains(&units[ch.cur - 1])
                        && (0xD800..=0xDBFF).contains(&units[ch.cur - 2])
                    {
                        ch.cur - 2
                    } else {
                        ch.cur - 1
                    };
                    ch.cur = np;
                } else {
                    if ch.bound == 0 {
                        depth -= 1;
                        continue;
                    }
                    match read::<U>(units, ch.cur) {
                        Some((c, n)) if self.sets[set].has(c, prog, flags) => {
                            ch.cur = n;
                            ch.bound -= 1;
                        }
                        _ => {
                            depth -= 1;
                            continue;
                        }
                    }
                }
                i = ch.step + 1;
                p = ch.cur;
                continue 'run;
            }
        }
    }
}

/// The index of the first `x` in `hay`: the same answer as
/// `hay.iter().position(|&u| u == x)`, four units per step. Each group of four
/// is packed into a word and XORed with `x` in every lane, so a matching lane
/// becomes zero; the classic "has a zero lane" test (`(w - 1s) & !w & highs`)
/// then flags the group, and the exact lane is found by plain comparison.
#[inline]
pub(super) fn find_unit(hay: &[u16], x: u16) -> Option<usize> {
    const ONES: u64 = 0x0001_0001_0001_0001;
    const HIGHS: u64 = 0x8000_8000_8000_8000;
    let pat = ONES.wrapping_mul(u64::from(x));
    let (chunks, rest) = hay.as_chunks::<4>();
    for (i, c) in chunks.iter().enumerate() {
        let w = (u64::from(c[0])
            | u64::from(c[1]) << 16
            | u64::from(c[2]) << 32
            | u64::from(c[3]) << 48)
            ^ pat;
        if w.wrapping_sub(ONES) & !w & HIGHS != 0 {
            return c.iter().position(|&u| u == x).map(|k| i * 4 + k);
        }
    }
    rest.iter()
        .position(|&u| u == x)
        .map(|k| chunks.len() * 4 + k)
}

/// The per-find context of [`Linear::run_on`].
struct Run<'a> {
    prog: &'a [Inst],
    units: &'a [u16],
    flags: Flags,
    steps_used: &'a Cell<u64>,
    budget: u64,
}

/// The character at `p` and the position after it: one unit outside `u`
/// mode; in `u` mode a well-formed surrogate pair is one code point.
#[inline(always)]
fn read<const U: bool>(units: &[u16], p: usize) -> Option<(u32, usize)> {
    let u = *units.get(p)?;
    if U && (0xD800..=0xDBFF).contains(&u)
        && let Some(&lo) = units.get(p + 1)
        && (0xDC00..=0xDFFF).contains(&lo)
    {
        let cp = 0x10000 + ((u32::from(u) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
        return Some((cp, p + 2));
    }
    Some((u32::from(u), p + 1))
}

/// The number of units the character at `s` spans in `u` mode (1 at the end).
fn char_len(units: &[u16], s: usize) -> usize {
    match read::<true>(units, s) {
        Some((_, n)) => n - s,
        None => 1,
    }
}

fn is_char_boundary(units: &[u16], s: usize) -> bool {
    s == 0
        || s >= units.len()
        || !(0xDC00..=0xDFFF).contains(&units[s])
        || !(0xD800..=0xDBFF).contains(&units[s - 1])
}

fn is_consume(inst: Option<&Inst>) -> bool {
    matches!(
        inst,
        Some(Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. })
    )
}

/// The start filter for `branches`: the one unit or set every match begins
/// with, the union over the branches.
fn start_filter(branches: &[Vec<Step>], sets: &mut Vec<Set>) -> First {
    let firsts: Vec<First> = branches
        .iter()
        .map(
            |steps| match steps.iter().find(|s| !matches!(s, Step::Save(_))) {
                Some(Step::Unit(u)) => First::Unit(*u),
                Some(Step::One(k)) => First::Set(*k),
                _ => First::Any,
            },
        )
        .collect();
    match firsts.as_slice() {
        [one] => *one,
        _ => {
            let mut union = Set {
                latin1: [0; 4],
                high: High::None,
            };
            for f in &firsts {
                match *f {
                    First::Any => return First::Any,
                    First::Unit(u) if u < 256 => {
                        union.latin1[usize::from(u >> 6)] |= 1u64 << (u & 63);
                    }
                    First::Unit(_) => union.high = High::Any,
                    First::Set(k) => {
                        for (w, x) in union.latin1.iter_mut().zip(sets[k].latin1) {
                            *w |= x;
                        }
                        if !matches!(sets[k].high, High::None) {
                            union.high = High::Any;
                        }
                    }
                }
            }
            sets.push(union);
            First::Set(sets.len() - 1)
        }
    }
}

/// The set a single-consume instruction accepts under `flags`.
fn make_set(inst: &Inst, pc: usize, flags: Flags) -> Set {
    let mut latin1 = [0u64; 4];
    for u in 0u32..256 {
        if vm::single_consume_matches(inst, u, flags) {
            latin1[(u >> 6) as usize] |= 1u64 << (u & 63);
        }
    }
    Set {
        latin1,
        high: if high_is_empty(inst, flags) {
            High::None
        } else {
            High::Check(pc)
        },
    }
}

/// Whether provably no code point >= 256 is accepted by `inst`. Outside `u`
/// mode a case-insensitive comparison never maps a non-ASCII unit onto an
/// ASCII one, so an all-ASCII class stays all-ASCII under `i`; under `u` it
/// can (`ſ` folds to `s`, the Kelvin sign to `k`), so nothing is assumed.
fn high_is_empty(inst: &Inst, flags: Flags) -> bool {
    if flags.ignore_case && flags.unicode {
        return false;
    }
    let ceiling = if flags.ignore_case { 128 } else { 256 };
    match inst {
        Inst::Char(c) => *c < ceiling,
        Inst::Class(class) => {
            !class.neg
                && class.members.iter().all(|m| match m {
                    ClassMember::Char(c) => *c < ceiling,
                    ClassMember::Range(_, hi) => *hi < ceiling,
                    ClassMember::Shorthand(s) => matches!(
                        s,
                        super::parser::Shorthand::Digit | super::parser::Shorthand::Word
                    ),
                })
        }
        _ => false,
    }
}
