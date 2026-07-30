//! Compiles the regex AST to the backtracking VM's instruction list.

use super::parser::{ClassItem, ClassSetExpr, Node, RegexError};
use super::vm::{Assert, Class, ClassMember, FlagDelta, Inst, SetMatcher};
use alloc::vec::Vec;

/// Maximum number of instructions a compiled program may contain. Bounded-but-
/// generous quantifiers (`a{N}` / `(a{K}){K}`) expand by literal copying, so the
/// projected size is checked against this budget before emitting; an over-budget
/// expansion is a compile error rather than an OOM/hang (RE-3).
const MAX_PROG_SIZE: usize = crate::limits::DEFAULT_REGEX_MAX_PROG_SIZE;

/// Compiles `ast` to a program. The program is wrapped in `Save(0)…Save(1)` so
/// group 0 records the whole match, and ends in `Match`. Returns the program
/// and the number of capturing groups, or a `RegexError` if the program would
/// exceed `MAX_PROG_SIZE` instructions.
pub(crate) fn compile(
    ast: &Node,
    group_names: &[(usize, alloc::string::String)],
    unicode: bool,
) -> Result<(Vec<Inst>, usize), RegexError> {
    let mut c = Compiler {
        prog: Vec::new(),
        groups: 0,
        group_names,
        unicode,
        reverse: false,
    };
    c.emit(Inst::Save(0));
    c.compile(ast)?;
    c.emit(Inst::Save(1));
    c.emit(Inst::Match);
    Ok((c.prog, c.groups))
}

struct Compiler<'a> {
    prog: Vec<Inst>,
    groups: usize,
    /// `(index, name)` of each named group, for resolving `\k<name>`.
    group_names: &'a [(usize, alloc::string::String)],
    /// Whether the `u` flag is set. In non-`u` mode an astral literal compiles to
    /// its two surrogate code units (each a one-unit `Char`); in `u` mode it
    /// compiles to a single code-point `Char` that the VM matches as a pair.
    unicode: bool,
    /// Whether this (sub-)program matches **right-to-left**. Set only inside a
    /// lookbehind body (ECMAScript direction `-1`): concatenations compile in
    /// reversed order and a capturing group emits its end-marker before its
    /// start-marker, so that a VM consuming input backward records the group's
    /// forward `(start, end)` span correctly and greedy quantifiers bind from the
    /// right. A lookahead nested inside a lookbehind resets to forward; a nested
    /// lookbehind flips back to reverse.
    reverse: bool,
}

impl Compiler<'_> {
    fn emit(&mut self, inst: Inst) -> usize {
        self.prog.push(inst);
        self.prog.len() - 1
    }

    fn here(&self) -> usize {
        self.prog.len()
    }

    /// Emits, then errors if the program has grown past the instruction budget.
    /// Checked after each emit so even a runaway expansion stops promptly.
    fn check_size(&self) -> Result<(), RegexError> {
        if self.prog.len() > MAX_PROG_SIZE {
            return Err(RegexError::new("compiled pattern too large"));
        }
        Ok(())
    }

    fn compile(&mut self, node: &Node) -> Result<(), RegexError> {
        match node {
            Node::Empty => {}
            Node::Char(c) => {
                self.emit_char(*c);
            }
            Node::Any => {
                self.emit(Inst::Any);
            }
            Node::Start => {
                self.emit(Inst::Assert(Assert::Start));
            }
            Node::End => {
                self.emit(Inst::Assert(Assert::End));
            }
            Node::WordBoundary { neg } => {
                self.emit(Inst::Assert(if *neg {
                    Assert::NotWordBoundary
                } else {
                    Assert::WordBoundary
                }));
            }
            Node::Class { neg, items } => {
                let mut members = Vec::new();
                for item in items {
                    self.convert_item(item, &mut members);
                }
                self.emit(Inst::Class(Class { neg: *neg, members }));
            }
            Node::Concat(nodes) => {
                // In a lookbehind the concatenation is matched right-to-left, so
                // emit the sub-nodes in reversed source order (the VM consumes the
                // subject backward). Each sub-node is itself compiled in reverse.
                if self.reverse {
                    for n in nodes.iter().rev() {
                        self.compile(n)?;
                    }
                } else {
                    for n in nodes {
                        self.compile(n)?;
                    }
                }
            }
            Node::Group { index, inner } => {
                if let Some(idx) = index {
                    self.groups = self.groups.max(*idx);
                    // Forward: Save(start); body; Save(end). Reverse: the right edge
                    // of the group is reached first, so record the end slot first,
                    // then the (reversed) body, then the start slot — the resulting
                    // span is still forward `(start, end)`.
                    if self.reverse {
                        self.emit(Inst::Save(2 * idx + 1));
                        self.compile(inner)?;
                        self.emit(Inst::Save(2 * idx));
                    } else {
                        self.emit(Inst::Save(2 * idx));
                        self.compile(inner)?;
                        self.emit(Inst::Save(2 * idx + 1));
                    }
                } else {
                    self.compile(inner)?;
                }
            }
            Node::ClassSet { neg, set } => self.compile_class_set(*neg, set)?,
            Node::Modifier {
                ignore_case,
                multiline,
                dotall,
                inner,
            } => {
                // `(?ims-ims:…)` — wrap the body in a flag-stack push/pop so the VM
                // matches it under the adjusted `i`/`m`/`s` flags, then restores.
                self.emit(Inst::PushFlags(FlagDelta {
                    ignore_case: *ignore_case,
                    multiline: *multiline,
                    dotall: *dotall,
                }));
                self.compile(inner)?;
                self.emit(Inst::PopFlags);
            }
            Node::Alt(branches) => self.compile_alt(branches)?,
            Node::Repeat {
                inner,
                min,
                max,
                greedy,
            } => self.compile_repeat(inner, *min, *max, *greedy)?,
            // A lookahead compiles its body into a self-contained sub-program
            // (ending in `Match`) run zero-width by the VM.
            Node::Look { neg, inner } => {
                // A lookahead always matches forward (direction `+1`), even when
                // nested inside a lookbehind — reset `reverse`.
                let mut sub = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                    group_names: self.group_names,
                    unicode: self.unicode,
                    reverse: false,
                };
                sub.compile(inner)?;
                sub.emit(Inst::Match);
                self.groups = self.groups.max(sub.groups);
                self.emit(Inst::Look {
                    neg: *neg,
                    prog: sub.prog,
                });
            }
            Node::LookBehind { neg, inner } => {
                // A lookbehind matches right-to-left (direction `-1`): compile its
                // body in reverse so the VM consumes the subject backward from the
                // assertion point.
                let mut sub = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                    group_names: self.group_names,
                    unicode: self.unicode,
                    reverse: true,
                };
                sub.compile(inner)?;
                sub.emit(Inst::Match);
                self.groups = self.groups.max(sub.groups);
                self.emit(Inst::LookBehind {
                    neg: *neg,
                    prog: sub.prog,
                });
            }
            Node::Backref(n) => {
                self.groups = self.groups.max(*n);
                self.emit(Inst::Backref(*n));
            }
            Node::NamedBackref(name) => {
                // Resolve the name to its group index(es). A unique name emits a plain
                // `Backref`; a *duplicate* name (ES2025) emits a `BackrefMulti` that
                // references whichever participated at match time.
                let indices: alloc::vec::Vec<usize> = self
                    .group_names
                    .iter()
                    .filter(|(_, gn)| gn == name)
                    .map(|(idx, _)| *idx)
                    .collect();
                match indices.as_slice() {
                    [] => {
                        self.emit(Inst::Backref(0));
                    }
                    [n] => {
                        self.groups = self.groups.max(*n);
                        self.emit(Inst::Backref(*n));
                    }
                    _ => {
                        if let Some(m) = indices.iter().copied().max() {
                            self.groups = self.groups.max(m);
                        }
                        self.emit(Inst::BackrefMulti(indices));
                    }
                }
            }
        }
        self.check_size()
    }

    fn compile_alt(&mut self, branches: &[Node]) -> Result<(), RegexError> {
        // Each non-last branch: Split(this, next); branch; Jmp(end).
        let mut jmp_to_end = Vec::new();
        for (i, branch) in branches.iter().enumerate() {
            if i + 1 < branches.len() {
                let split = self.emit(Inst::Split(0, 0));
                let branch_start = self.here();
                self.compile(branch)?;
                let jmp = self.emit(Inst::Jmp(0));
                jmp_to_end.push(jmp);
                let next = self.here();
                self.prog[split] = Inst::Split(branch_start, next);
            } else {
                self.compile(branch)?;
            }
        }
        let end = self.here();
        for j in jmp_to_end {
            self.prog[j] = Inst::Jmp(end);
        }
        Ok(())
    }

    fn compile_repeat(
        &mut self,
        inner: &Node,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    ) -> Result<(), RegexError> {
        // A quantifier bound may be any decimal integer — the spec caps neither
        // bound of `{n,m}` — but a bound larger than any subject could satisfy
        // needs no expansion at all. An atom that cannot match the empty string
        // consumes at least one code unit per iteration, and a subject can never
        // hold more than `DEFAULT_MAX_STRING_LEN` of them, so no more than `cap`
        // iterations are ever possible: a larger `min` can never be reached (the
        // whole quantifier can never match) and a larger `max` is indistinguishable
        // from "unbounded". A nullable atom has no such ceiling, and an
        // unexpandable bound on one is still rejected by `MAX_PROG_SIZE` below.
        let (min, max) = if can_match_empty(inner) {
            (min, max)
        } else {
            let cap = crate::limits::DEFAULT_MAX_STRING_LEN;
            if min > cap {
                self.emit(Inst::Fail);
                return self.check_size();
            }
            (min, max.filter(|m| *m <= cap))
        };
        match max {
            // Unbounded tail (`*`, `+`, `{n,}`): the optional/star part adds only
            // a small constant (a `Split`/`Jmp` around one real copy of `inner`),
            // so there is nothing to project — and crucially we must NOT run a
            // throwaway `probe.compile(inner)` here. Probing recompiles `inner`,
            // and for a nested unbounded quantifier (`(?:…*)*`) the probe at each
            // level recursively re-probes its own inner, doubling the compile work
            // per nesting level → 2^depth blowup at *compile* time (RE-9). RE-3's
            // size cap never fires because the final program stays tiny (min=0 ⇒
            // no mandatory copies). `compile`'s per-emit `check_size` already
            // bounds any real growth, so a single real compile is all we need.
            None => {
                // `min` mandatory copies, then the `(inner)*` tail. Each real
                // `compile` self-checks `MAX_PROG_SIZE`, so a large `min` (e.g.
                // `a{1000000,}`) is still rejected promptly as it emits.
                for _ in 0..min {
                    self.emit_clear_captures(inner);
                    self.compile(inner)?;
                }
                self.compile_star(inner, greedy)?;
            }
            // Bounded `{min,max}`: project the expanded size *before* emitting any
            // copies, so a bound like `a{1000000}` (or `(a{K}){K}`) is rejected up
            // front rather than after it has already allocated a huge program
            // (RE-3). One copy of `inner` is compiled into a scratch program to
            // learn its instruction count.
            Some(max) => {
                let mut probe = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                    group_names: self.group_names,
                    unicode: self.unicode,
                    reverse: self.reverse,
                };
                probe.compile(inner)?;
                let inner_size = probe.prog.len().max(1);
                // Each mandatory copy is `inner_size`; each optional copy adds ~1
                // split around `inner_size`.
                let projected = self
                    .prog
                    .len()
                    .saturating_add(max.saturating_mul(inner_size.saturating_add(1)));
                if projected > MAX_PROG_SIZE {
                    return Err(RegexError::new("compiled pattern too large"));
                }
                // `min` mandatory copies, then `(max - min)` optional copies.
                for _ in 0..min {
                    self.emit_clear_captures(inner);
                    self.compile(inner)?;
                }
                self.compile_optional_chain(inner, max - min, greedy)?;
            }
        }
        Ok(())
    }

    /// Emits a [`Inst::ClearCaptures`] for a quantified atom's groups (a no-op when
    /// it has none), so each repetition starts with them unset (RepeatMatcher).
    fn emit_clear_captures(&mut self, inner: &Node) {
        if let Some((lo, hi)) = group_slot_range(inner) {
            self.emit(Inst::ClearCaptures { from: lo, to: hi });
        }
    }

    /// `inner*` — `L1: Split(mark, exit); mark; clear; body; empty-fail; Jmp L1; exit:`.
    ///
    /// The `Mark`/`EmptyFail` bracket implements RepeatMatcher step 2.b and is
    /// emitted only when `inner` can match the empty string; a body that always
    /// consumes keeps the tight `Split; <consume>; Jmp` shape the VM's
    /// `simple_loop` fast path recognizes.
    fn compile_star(&mut self, inner: &Node, greedy: bool) -> Result<(), RegexError> {
        let nullable = can_match_empty(inner);
        let l1 = self.emit(Inst::Split(0, 0));
        let body = self.here();
        if nullable {
            self.emit(Inst::Mark);
        }
        self.emit_clear_captures(inner);
        self.compile(inner)?;
        if nullable {
            self.emit(Inst::EmptyFail);
        }
        self.emit(Inst::Jmp(l1));
        let exit = self.here();
        self.prog[l1] = if greedy {
            Inst::Split(body, exit)
        } else {
            Inst::Split(exit, body)
        };
        Ok(())
    }

    /// The `(max - min)` optional copies of a bounded `{min,max}` quantifier, as a
    /// **nested** chain — `(?:x(?:x(?:x)?)?)?`, not `x?x?x?`.
    ///
    /// The two accept the same language, but not with the same search space. In the
    /// flat form each copy is skippable independently, so a subject that matches `k`
    /// of `n` copies and then fails later gives the backtracker `C(n, k)` distinct
    /// ways to place those `k` matches — `[A-Za-z]{0,12}/[A-Za-z]{0,12}` against
    /// `"Etc/GMT-1"` explores ~48 000 of them, exhausts the step budget, and the
    /// whole find gives up (so `/^(?:[A-Za-z]{0,12}\/[A-Za-z]{0,12}|Etc\/GMT-1)$/`
    /// never reaches its second alternative). Nesting makes a skip commit: skipping
    /// copy `k` skips every later copy too, so there is exactly one path per match
    /// count and the search is linear.
    ///
    /// Emitted iteratively (a copy's exit is the end of the *whole* chain, which is
    /// what nesting amounts to) so a large bound costs no compiler recursion.
    fn compile_optional_chain(
        &mut self,
        inner: &Node,
        count: usize,
        greedy: bool,
    ) -> Result<(), RegexError> {
        let nullable = can_match_empty(inner);
        let mut splits: Vec<(usize, usize)> = Vec::with_capacity(count);
        for _ in 0..count {
            let split = self.emit(Inst::Split(0, 0));
            let body = self.here();
            if nullable {
                self.emit(Inst::Mark);
            }
            self.emit_clear_captures(inner);
            self.compile(inner)?;
            if nullable {
                self.emit(Inst::EmptyFail);
            }
            splits.push((split, body));
        }
        let exit = self.here();
        for (split, body) in splits {
            self.prog[split] = if greedy {
                Inst::Split(body, exit)
            } else {
                Inst::Split(exit, body)
            };
        }
        Ok(())
    }

    /// Emits a literal scalar code point. In non-`u` mode an astral scalar is
    /// split into its two surrogate code units, each a one-unit `Char`, so the
    /// subject is matched code-unit by code-unit (web-compat default). In `u`
    /// mode a single `Char` carries the whole scalar; the VM reads a code point
    /// (a surrogate pair) to match it.
    fn emit_char(&mut self, c: u32) {
        if !self.unicode && c > 0xFFFF {
            let (hi, lo) = surrogate_pair(c);
            self.emit(Inst::Char(hi as u32));
            self.emit(Inst::Char(lo as u32));
        } else {
            self.emit(Inst::Char(c));
        }
    }

    /// Compiles a `v`-mode extended character class. The set expression is split
    /// into a single-code-point [`SetMatcher`] and a set of string alternatives
    /// (length ≠ 1). With no strings, this is a single `ClassSet` instruction.
    /// With strings, it compiles to an alternation that tries each string literal
    /// (longest first, so the maximal match wins) and finally the char class.
    /// A negated class may not contain strings (a Syntax Error per the spec).
    fn compile_class_set(&mut self, neg: bool, set: &ClassSetExpr) -> Result<(), RegexError> {
        let mut strings: Vec<Vec<u32>> = Vec::new();
        collect_strings(set, &mut strings);
        // Drop length-1 strings: they behave as ordinary characters and are
        // already covered by the code-point matcher.
        strings.retain(|s| s.len() != 1);
        if neg && !strings.is_empty() {
            return Err(RegexError::new(
                "negated character class may not contain strings",
            ));
        }
        let matcher = build_set_matcher(set);
        if strings.is_empty() {
            self.emit(Inst::ClassSet { neg, matcher });
            return Ok(());
        }
        // Strings present: longest first so the alternation prefers a maximal
        // match (e.g. `\q{abc}` over `\q{ab}` over the single-char class).
        strings.sort_by_key(|s| core::cmp::Reverse(s.len()));
        // Build an Alt-like structure manually: each non-final branch is
        // `Split(branch, next); branch; Jmp(end)`, the final branch is the class.
        let mut jmp_to_end = Vec::new();
        for s in &strings {
            let split = self.emit(Inst::Split(0, 0));
            let branch_start = self.here();
            for &cp in s {
                self.emit_char(cp);
            }
            let jmp = self.emit(Inst::Jmp(0));
            jmp_to_end.push(jmp);
            let next = self.here();
            self.prog[split] = Inst::Split(branch_start, next);
            self.check_size()?;
        }
        // Final branch: the single-code-point class.
        self.emit(Inst::ClassSet { neg, matcher });
        let end = self.here();
        for j in jmp_to_end {
            self.prog[j] = Inst::Jmp(end);
        }
        Ok(())
    }

    /// Lowers one class item into compiled members, splitting an astral literal
    /// member into its surrogate units in non-`u` mode so `[😀]` (no `u`) matches
    /// either surrogate half, matching JS web-compat semantics. Astral *ranges*
    /// are kept whole only in `u` mode; in non-`u` mode their bounds are above
    /// `0xFFFF` and can never match a single code unit, so they are dropped.
    fn convert_item(&self, item: &ClassItem, out: &mut Vec<ClassMember>) {
        match item {
            ClassItem::Char(c) => {
                if !self.unicode && *c > 0xFFFF {
                    let (hi, lo) = surrogate_pair(*c);
                    out.push(ClassMember::Char(hi as u32));
                    out.push(ClassMember::Char(lo as u32));
                } else {
                    out.push(ClassMember::Char(*c));
                }
            }
            ClassItem::Range(a, b) => out.push(ClassMember::Range(*a, *b)),
            ClassItem::Shorthand(s) => out.push(ClassMember::Shorthand(*s)),
        }
    }
}

/// Collects the multi-(or zero-)code-point string alternatives a `v`-mode set
/// expression can match, honoring set operations. A string of length 1 is also
/// collected here but the caller drops it (it is an ordinary character). Set
/// algebra on the string component: union concatenates; intersection keeps
/// strings present in every operand; difference keeps first-operand strings
/// absent from all later operands. Single code points never appear as strings.
fn collect_strings(expr: &ClassSetExpr, out: &mut Vec<Vec<u32>>) {
    for s in set_strings(expr) {
        out.push(s);
    }
}

/// The set of string alternatives (length ≠ 1) of a class-set expression.
fn set_strings(expr: &ClassSetExpr) -> Vec<Vec<u32>> {
    match expr {
        ClassSetExpr::Items(_) => Vec::new(),
        ClassSetExpr::Strings(list) => list.iter().filter(|s| s.len() != 1).cloned().collect(),
        ClassSetExpr::Negated(_) => Vec::new(),
        ClassSetExpr::Union(kids) => {
            let mut acc: Vec<Vec<u32>> = Vec::new();
            for k in kids {
                for s in set_strings(k) {
                    if !acc.contains(&s) {
                        acc.push(s);
                    }
                }
            }
            acc
        }
        ClassSetExpr::Intersection(kids) => {
            let mut it = kids.iter();
            let Some(first) = it.next() else {
                return Vec::new();
            };
            let mut acc = set_strings(first);
            for k in it {
                let other = set_strings(k);
                acc.retain(|s| other.contains(s));
            }
            acc
        }
        ClassSetExpr::Difference(kids) => {
            let mut it = kids.iter();
            let Some(first) = it.next() else {
                return Vec::new();
            };
            let mut acc = set_strings(first);
            for k in it {
                let other = set_strings(k);
                acc.retain(|s| !other.contains(s));
            }
            acc
        }
    }
}

/// Builds the single-code-point [`SetMatcher`] for a class-set expression. The
/// string component is handled separately ([`set_strings`]); a `\q{…}` string of
/// length 1 contributes its single code point to the char matcher here.
fn build_set_matcher(expr: &ClassSetExpr) -> SetMatcher {
    match expr {
        ClassSetExpr::Items(items) => {
            let mut members = Vec::new();
            for it in items {
                match it {
                    ClassItem::Char(c) => members.push(ClassMember::Char(*c)),
                    ClassItem::Range(a, b) => members.push(ClassMember::Range(*a, *b)),
                    ClassItem::Shorthand(s) => members.push(ClassMember::Shorthand(*s)),
                }
            }
            SetMatcher::Leaf(members)
        }
        ClassSetExpr::Strings(list) => {
            // Length-1 strings act as ordinary characters; longer/empty ones add
            // nothing to single-code-point membership.
            let members = list
                .iter()
                .filter(|s| s.len() == 1)
                .map(|s| ClassMember::Char(s[0]))
                .collect();
            SetMatcher::Leaf(members)
        }
        ClassSetExpr::Negated(inner) => {
            SetMatcher::Negated(alloc::boxed::Box::new(build_set_matcher(inner)))
        }
        ClassSetExpr::Union(kids) => {
            SetMatcher::Union(kids.iter().map(build_set_matcher).collect())
        }
        ClassSetExpr::Intersection(kids) => {
            SetMatcher::Intersection(kids.iter().map(build_set_matcher).collect())
        }
        ClassSetExpr::Difference(kids) => {
            SetMatcher::Difference(kids.iter().map(build_set_matcher).collect())
        }
    }
}

/// Whether `node` can match the empty string.
///
/// Used for two decisions, both of which want a *conservative* answer: a
/// quantifier over a nullable atom needs the `Mark`/`EmptyFail` empty-iteration
/// guard, and only a non-nullable atom lets a huge bound be answered statically.
/// Erring towards `true` costs an optimization; erring towards `false` would be
/// unsound, so every zero-width construct answers `true`.
fn can_match_empty(node: &Node) -> bool {
    match node {
        // Consume exactly one code point. A `v`-mode class is the one exception:
        // `\q{}` gives it an empty-string alternative.
        Node::Char(_) | Node::Any | Node::Class { .. } => false,
        Node::ClassSet { set, .. } => set_strings(set).iter().any(alloc::vec::Vec::is_empty),
        // Zero-width by construction.
        Node::Empty | Node::Start | Node::End | Node::WordBoundary { .. } => true,
        Node::Look { .. } | Node::LookBehind { .. } => true,
        // An unset group backreference matches the empty string, as does a
        // reference to a group that captured nothing.
        Node::Backref(_) | Node::NamedBackref(_) => true,
        Node::Group { inner, .. } | Node::Modifier { inner, .. } => can_match_empty(inner),
        Node::Concat(kids) => kids.iter().all(can_match_empty),
        Node::Alt(kids) => kids.iter().any(can_match_empty),
        Node::Repeat { inner, min, .. } => *min == 0 || can_match_empty(inner),
    }
}

/// The inclusive capture-*slot* range `(2·minIdx, 2·maxIdx+1)` of every capturing
/// group nested anywhere in `node`, or `None` if it contains no capturing group.
fn group_slot_range(node: &Node) -> Option<(usize, usize)> {
    fn walk(node: &Node, lo: &mut Option<usize>, hi: &mut Option<usize>) {
        match node {
            Node::Group { index, inner } => {
                if let Some(idx) = index {
                    *lo = Some(lo.map_or(*idx, |v: usize| v.min(*idx)));
                    *hi = Some(hi.map_or(*idx, |v: usize| v.max(*idx)));
                }
                walk(inner, lo, hi);
            }
            Node::Modifier { inner, .. }
            | Node::Look { inner, .. }
            | Node::LookBehind { inner, .. }
            | Node::Repeat { inner, .. } => walk(inner, lo, hi),
            Node::Concat(kids) | Node::Alt(kids) => {
                for k in kids {
                    walk(k, lo, hi);
                }
            }
            _ => {}
        }
    }
    let (mut lo, mut hi) = (None, None);
    walk(node, &mut lo, &mut hi);
    match (lo, hi) {
        (Some(l), Some(h)) => Some((2 * l, 2 * h + 1)),
        _ => None,
    }
}

/// Splits an astral scalar (`> 0xFFFF`) into its UTF-16 surrogate pair.
fn surrogate_pair(c: u32) -> (u16, u16) {
    let v = c - 0x10000;
    let hi = 0xD800 + (v >> 10) as u16;
    let lo = 0xDC00 + (v & 0x3FF) as u16;
    (hi, lo)
}
