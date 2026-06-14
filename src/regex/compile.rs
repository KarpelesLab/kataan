//! Compiles the regex AST to the backtracking VM's instruction list.

use super::parser::{ClassItem, Node, RegexError};
use super::vm::{Assert, Class, ClassMember, Inst};
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
                for n in nodes {
                    self.compile(n)?;
                }
            }
            Node::Group { index, inner } => {
                if let Some(idx) = index {
                    self.groups = self.groups.max(*idx);
                    self.emit(Inst::Save(2 * idx));
                    self.compile(inner)?;
                    self.emit(Inst::Save(2 * idx + 1));
                } else {
                    self.compile(inner)?;
                }
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
                let mut sub = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                    group_names: self.group_names,
                    unicode: self.unicode,
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
                let mut sub = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                    group_names: self.group_names,
                    unicode: self.unicode,
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
                // Resolve the name to its group index; an unknown name back-refers
                // group 0 (the whole match never re-binds, so it matches empty).
                let n = self
                    .group_names
                    .iter()
                    .find(|(_, gn)| gn == name)
                    .map_or(0, |(idx, _)| *idx);
                self.groups = self.groups.max(n);
                self.emit(Inst::Backref(n));
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
                    self.compile(inner)?;
                }
                for _ in min..max {
                    self.compile_optional(inner, greedy)?;
                }
            }
        }
        Ok(())
    }

    /// `inner*` — `L1: Split(body, exit); body; Jmp L1; exit:` (greedy prefers
    /// the body; lazy prefers the exit).
    fn compile_star(&mut self, inner: &Node, greedy: bool) -> Result<(), RegexError> {
        let l1 = self.emit(Inst::Split(0, 0));
        let body = self.here();
        self.compile(inner)?;
        self.emit(Inst::Jmp(l1));
        let exit = self.here();
        self.prog[l1] = if greedy {
            Inst::Split(body, exit)
        } else {
            Inst::Split(exit, body)
        };
        Ok(())
    }

    /// `inner?` — `Split(body, exit); body; exit:`.
    fn compile_optional(&mut self, inner: &Node, greedy: bool) -> Result<(), RegexError> {
        let split = self.emit(Inst::Split(0, 0));
        let body = self.here();
        self.compile(inner)?;
        let exit = self.here();
        self.prog[split] = if greedy {
            Inst::Split(body, exit)
        } else {
            Inst::Split(exit, body)
        };
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

/// Splits an astral scalar (`> 0xFFFF`) into its UTF-16 surrogate pair.
fn surrogate_pair(c: u32) -> (u16, u16) {
    let v = c - 0x10000;
    let hi = 0xD800 + (v >> 10) as u16;
    let lo = 0xDC00 + (v & 0x3FF) as u16;
    (hi, lo)
}
