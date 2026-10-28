//! A fast matcher for *linear* patterns: a straight sequence of literals,
//! single-character sets, single-character quantifiers (`\d+`, `[a-z]*?`,
//! `x{2,5}`), capture-group boundaries and zero-width assertions — no
//! alternation, no quantified group, no lookaround, no backreference. Most
//! regexes in real programs have this shape (`/(\w+)@([\w.]+)/`,
//! `/page\/(\d)9\d/`, `/\d+/g`).
//!
//! It accepts exactly what the backtracking VM ([`super::vm`]) accepts, in
//! the same preference order, but runs on a flat step list with an explicit,
//! bounded choice stack instead of one recursive call per instruction:
//!
//! * a sequence has one path through it, so every capture boundary is written
//!   again on every retry of the steps after a choice point — nothing has to
//!   be restored when backtracking;
//! * each quantifier step leaves at most one choice point, so the stack depth
//!   is bounded by the number of quantifier steps (known at compile time);
//! * every consume is exactly one code unit (only built outside `u`/`v`
//!   mode), so a greedy step backs off by plain index decrement and set
//!   membership is a bit test for Latin-1 units.
//!
//! The steps are derived from the compiled VM program, and every set's
//! membership is the VM's own predicate ([`vm::single_consume_matches`]),
//! tabulated for Latin-1 and called directly above it, so the two engines
//! cannot disagree about what a character class contains.

use super::Flags;
use super::vm::{self, Assert, ClassMember, Inst};
use alloc::vec::Vec;
use core::cell::Cell;

/// Membership of the units at or above 256 in a [`Set`].
#[derive(Clone, Copy)]
enum High {
    /// No unit >= 256 is a member.
    None,
    /// Ask the VM predicate for the instruction at this program index.
    Check(usize),
}

/// One character set: the Latin-1 bitmap plus the rule above it.
struct Set {
    latin1: [u64; 4],
    high: High,
}

impl Set {
    #[inline(always)]
    fn has(&self, unit: u16, prog: &[Inst], flags: Flags) -> bool {
        if unit < 256 {
            return self.latin1[usize::from(unit >> 6)] & (1u64 << (unit & 63)) != 0;
        }
        match self.high {
            High::None => false,
            High::Check(pc) => vm::single_consume_matches(&prog[pc], u32::from(unit), flags),
        }
    }
}

enum Step {
    /// Record the position in capture slot `n`.
    Save(usize),
    /// One exact code unit.
    Unit(u16),
    /// One unit from a set.
    One(usize),
    /// Up to `max` units from a set (unbounded when `None`), greedy or lazy.
    /// `next_unit` is the literal the steps after it must begin with, if any,
    /// so a greedy back-off only stops where that literal is.
    Rep {
        set: usize,
        max: Option<usize>,
        greedy: bool,
        next_unit: Option<u16>,
    },
    /// A zero-width assertion.
    Assert(Assert),
    Match,
}

/// A linear program; see the module docs.
pub(crate) struct Linear {
    steps: Vec<Step>,
    sets: Vec<Set>,
    /// The number of quantifier steps, which bounds the choice-stack depth.
    reps: usize,
    slots: usize,
    /// What a match must begin with, to skip offsets that cannot start one.
    first: First,
    /// `(A, B)` when the steps open with `A B* x` (captures aside) and the
    /// literal `x` is outside `B`; see [`Linear::find`].
    lead_run: Option<(usize, usize)>,
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
    /// Matched; the capture slots are in the caller's buffer.
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
    /// Greedy: the lowest position it may back off to. Lazy: the highest
    /// position it may extend to.
    bound: usize,
}

const STACK_CHOICES: usize = 16;

impl Linear {
    /// Builds the linear form of `prog` (compiled with `flags`), or `None` when
    /// the program is not a straight sequence.
    pub(crate) fn build(prog: &[Inst], flags: Flags, slots: usize) -> Option<Linear> {
        if flags.unicode || flags.unicode_sets {
            return None;
        }
        let mut steps = Vec::new();
        let mut sets: Vec<Set> = Vec::new();
        let mut reps = 0usize;
        let mut pc = 0usize;
        let new_set = |pc: usize, sets: &mut Vec<Set>| -> usize {
            sets.push(make_set(&prog[pc], pc, flags));
            sets.len() - 1
        };
        loop {
            match prog.get(pc)? {
                Inst::Save(s) => {
                    steps.push(Step::Save(*s));
                    pc += 1;
                }
                Inst::Char(c) if !flags.ignore_case => {
                    steps.push(Step::Unit(u16::try_from(*c).ok()?));
                    pc += 1;
                }
                Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. } => {
                    steps.push(Step::One(new_set(pc, &mut sets)));
                    pc += 1;
                }
                Inst::Assert(a) => {
                    steps.push(Step::Assert(match a {
                        Assert::Start => Assert::Start,
                        Assert::End => Assert::End,
                        Assert::WordBoundary => Assert::WordBoundary,
                        Assert::NotWordBoundary => Assert::NotWordBoundary,
                    }));
                    pc += 1;
                }
                Inst::Match => {
                    steps.push(Step::Match);
                    break;
                }
                Inst::Split(a, b) => {
                    let (a, b) = (*a, *b);
                    // `x*` / `x*?`: `Split; x; Jmp(split)`, continuing after it.
                    if is_consume(prog.get(pc + 1))
                        && matches!(prog.get(pc + 2), Some(Inst::Jmp(t)) if *t == pc)
                        && ((a == pc + 1 && b == pc + 3) || (b == pc + 1 && a == pc + 3))
                    {
                        steps.push(Step::Rep {
                            set: new_set(pc + 1, &mut sets),
                            max: None,
                            greedy: a == pc + 1,
                            next_unit: None,
                        });
                        reps += 1;
                        pc += 3;
                        continue;
                    }
                    // `x{0,k}` (and the optional tail of `x{n,m}`): a nested
                    // chain `Split(body, exit); x; Split(body, exit); x; … exit:`
                    // where every split shares one exit — one choice of count.
                    let greedy = a == pc + 1;
                    let exit = if greedy { b } else { a };
                    if !greedy && b != pc + 1 {
                        return None;
                    }
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
                    if q != exit || count == 0 {
                        return None;
                    }
                    steps.push(Step::Rep {
                        set: new_set(pc + 1, &mut sets),
                        max: Some(count),
                        greedy,
                        next_unit: None,
                    });
                    reps += 1;
                    pc = exit;
                }
                // Alternation, quantified groups, lookaround, backreferences,
                // inline modifiers: not a straight sequence.
                _ => return None,
            }
        }
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
        let consuming: Vec<&Step> = steps
            .iter()
            .filter(|s| !matches!(s, Step::Save(_)))
            .collect();
        let first = match consuming.first() {
            Some(Step::Unit(u)) => First::Unit(*u),
            Some(Step::One(k)) => First::Set(*k),
            _ => First::Any,
        };
        let lead_run = match consuming.as_slice() {
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
            ] if !sets[*b].has(*x, prog, flags) => Some((*a, *b)),
            _ => None,
        };
        Some(Linear {
            steps,
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
        if self.reps <= STACK_CHOICES {
            let mut stack = [Choice::default(); STACK_CHOICES];
            self.find_on(
                prog, units, start, last, flags, saves, steps_used, budget, &mut stack,
            )
        } else {
            let mut stack = alloc::vec![Choice::default(); self.reps];
            self.find_on(
                prog, units, start, last, flags, saves, steps_used, budget, &mut stack,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn find_on(
        &self,
        prog: &[Inst],
        units: &[u16],
        start: usize,
        last: usize,
        flags: Flags,
        saves: &mut [usize],
        steps_used: &Cell<u64>,
        budget: u64,
        stack: &mut [Choice],
    ) -> Option<()> {
        // Only offsets in `start..=last` (and inside the subject, except for
        // the empty match at its end) are candidates.
        let end = last.saturating_add(1).min(units.len());
        let mut s = start;
        while s <= last {
            match self.first {
                First::Unit(u) => {
                    if s >= end {
                        return None;
                    }
                    s += units[s..end].iter().position(|&x| x == u)?;
                }
                First::Set(k) => {
                    if s >= end {
                        return None;
                    }
                    let set = &self.sets[k];
                    s += units[s..end]
                        .iter()
                        .position(|&x| set.has(x, prog, flags))?;
                }
                First::Any => {}
            }
            match self.run_on(prog, units, s, flags, saves, steps_used, budget, stack) {
                Outcome::Match => return Some(()),
                Outcome::Abort => return None,
                Outcome::NoMatch => {}
            }
            // A failed attempt at the head of an `A B+ x …` run fails at every
            // later start inside the same run of `B` units too: from any of
            // them, `A` either fails or `B*` stops where the run ends (`x` is
            // outside `B`, so it can only be matched there), leaving the rest
            // of the attempt exactly as it was. Skip to the end of the run.
            if let Some((a, b)) = self.lead_run
                && units
                    .get(s)
                    .is_some_and(|&u| self.sets[a].has(u, prog, flags))
            {
                s += 1;
                let b = &self.sets[b];
                while units.get(s).is_some_and(|&u| b.has(u, prog, flags)) {
                    s += 1;
                }
                continue;
            }
            s += 1;
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn run_on(
        &self,
        prog: &[Inst],
        units: &[u16],
        start: usize,
        flags: Flags,
        saves: &mut [usize],
        steps_used: &Cell<u64>,
        budget: u64,
        stack: &mut [Choice],
    ) -> Outcome {
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
                match &self.steps[i] {
                    Step::Save(s) => {
                        saves[*s] = p;
                        i += 1;
                    }
                    Step::Unit(u) => {
                        if p < len && units[p] == *u {
                            p += 1;
                            i += 1;
                        } else {
                            break true;
                        }
                    }
                    Step::One(k) => {
                        if p < len && self.sets[*k].has(units[p], prog, flags) {
                            p += 1;
                            i += 1;
                        } else {
                            break true;
                        }
                    }
                    Step::Rep {
                        set,
                        max,
                        greedy,
                        next_unit,
                    } => {
                        let set = &self.sets[*set];
                        let limit = match max {
                            Some(m) => len.min(p.saturating_add(*m)),
                            None => len,
                        };
                        if *greedy {
                            let lo = p;
                            while p < limit && set.has(units[p], prog, flags) {
                                p += 1;
                            }
                            work += (p - lo) as u64;
                            // Back off to where the continuation can start.
                            if let Some(c) = next_unit
                                && units.get(p) != Some(c)
                            {
                                match units[lo..p].iter().rposition(|u| u == c) {
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
                                bound: limit,
                            };
                        }
                        depth += 1;
                        i += 1;
                    }
                    Step::Assert(a) => {
                        if vm::assert_ok(a, units, p, flags) {
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
                    steps_used.set(steps_used.get().saturating_add(work));
                }
                return Outcome::Match;
            }
            // Backtrack into the most recent choice point that has an
            // alternative left.
            loop {
                work += 1;
                if work >= 4096 {
                    let n = steps_used.get().saturating_add(work);
                    steps_used.set(n);
                    work = 0;
                    if n > budget {
                        return Outcome::Abort;
                    }
                }
                if depth == 0 {
                    steps_used.set(steps_used.get().saturating_add(work));
                    if steps_used.get() > budget {
                        return Outcome::Abort;
                    }
                    return Outcome::NoMatch;
                }
                let ch = &mut stack[depth - 1];
                let Step::Rep {
                    set,
                    greedy,
                    next_unit,
                    ..
                } = &self.steps[ch.step]
                else {
                    unreachable!("choice points are only pushed by Rep steps");
                };
                if *greedy {
                    if ch.cur == ch.bound {
                        depth -= 1;
                        continue;
                    }
                    let mut np = ch.cur - 1;
                    if let Some(c) = next_unit {
                        match units[ch.bound..ch.cur].iter().rposition(|u| u == c) {
                            Some(k) => np = ch.bound + k,
                            None => {
                                depth -= 1;
                                continue;
                            }
                        }
                    }
                    ch.cur = np;
                } else {
                    if ch.cur >= ch.bound || !self.sets[*set].has(units[ch.cur], prog, flags) {
                        depth -= 1;
                        continue;
                    }
                    ch.cur += 1;
                }
                i = ch.step + 1;
                p = ch.cur;
                continue 'run;
            }
        }
    }
}

fn is_consume(inst: Option<&Inst>) -> bool {
    matches!(
        inst,
        Some(Inst::Char(_) | Inst::Any | Inst::Class(_) | Inst::ClassSet { .. })
    )
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

/// Whether provably no unit >= 256 is accepted by `inst` (outside `u` mode,
/// where a case-insensitive comparison never maps a non-ASCII unit onto an
/// ASCII one, so an all-ASCII class stays all-ASCII under `i`).
fn high_is_empty(inst: &Inst, flags: Flags) -> bool {
    let ceiling = if flags.ignore_case { 128 } else { 256 };
    match inst {
        Inst::Char(c) => *c < ceiling,
        Inst::Class(class) => {
            !class.neg
                && class.members.iter().all(|m| match m {
                    ClassMember::Char(c) => *c < ceiling,
                    ClassMember::Range(_, hi) => *hi < ceiling,
                    ClassMember::Shorthand(s) => {
                        matches!(
                            s,
                            super::parser::Shorthand::Digit | super::parser::Shorthand::Word
                        )
                    }
                })
        }
        _ => false,
    }
}
