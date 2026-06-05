//! Compiles the regex AST to the backtracking VM's instruction list.

use super::parser::{ClassItem, Node};
use super::vm::{Assert, Class, ClassMember, Inst};
use alloc::vec::Vec;

/// Compiles `ast` to a program. The program is wrapped in `Save(0)…Save(1)` so
/// group 0 records the whole match, and ends in `Match`. Returns the program
/// and the number of capturing groups.
pub(crate) fn compile(ast: &Node) -> (Vec<Inst>, usize) {
    let mut c = Compiler {
        prog: Vec::new(),
        groups: 0,
    };
    c.emit(Inst::Save(0));
    c.compile(ast);
    c.emit(Inst::Save(1));
    c.emit(Inst::Match);
    (c.prog, c.groups)
}

struct Compiler {
    prog: Vec<Inst>,
    groups: usize,
}

impl Compiler {
    fn emit(&mut self, inst: Inst) -> usize {
        self.prog.push(inst);
        self.prog.len() - 1
    }

    fn here(&self) -> usize {
        self.prog.len()
    }

    fn compile(&mut self, node: &Node) {
        match node {
            Node::Empty => {}
            Node::Char(c) => {
                self.emit(Inst::Char(*c));
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
                let class = Class {
                    neg: *neg,
                    members: items.iter().map(convert_item).collect(),
                };
                self.emit(Inst::Class(class));
            }
            Node::Concat(nodes) => {
                for n in nodes {
                    self.compile(n);
                }
            }
            Node::Group { index, inner } => {
                if let Some(idx) = index {
                    self.groups = self.groups.max(*idx);
                    self.emit(Inst::Save(2 * idx));
                    self.compile(inner);
                    self.emit(Inst::Save(2 * idx + 1));
                } else {
                    self.compile(inner);
                }
            }
            Node::Alt(branches) => self.compile_alt(branches),
            Node::Repeat {
                inner,
                min,
                max,
                greedy,
            } => self.compile_repeat(inner, *min, *max, *greedy),
            // A lookahead compiles its body into a self-contained sub-program
            // (ending in `Match`) run zero-width by the VM.
            Node::Look { neg, inner } => {
                let mut sub = Compiler {
                    prog: Vec::new(),
                    groups: 0,
                };
                sub.compile(inner);
                sub.emit(Inst::Match);
                self.groups = self.groups.max(sub.groups);
                self.emit(Inst::Look {
                    neg: *neg,
                    prog: sub.prog,
                });
            }
            Node::Backref(n) => {
                self.groups = self.groups.max(*n);
                self.emit(Inst::Backref(*n));
            }
        }
    }

    fn compile_alt(&mut self, branches: &[Node]) {
        // Each non-last branch: Split(this, next); branch; Jmp(end).
        let mut jmp_to_end = Vec::new();
        for (i, branch) in branches.iter().enumerate() {
            if i + 1 < branches.len() {
                let split = self.emit(Inst::Split(0, 0));
                let branch_start = self.here();
                self.compile(branch);
                let jmp = self.emit(Inst::Jmp(0));
                jmp_to_end.push(jmp);
                let next = self.here();
                self.prog[split] = Inst::Split(branch_start, next);
            } else {
                self.compile(branch);
            }
        }
        let end = self.here();
        for j in jmp_to_end {
            self.prog[j] = Inst::Jmp(end);
        }
    }

    fn compile_repeat(&mut self, inner: &Node, min: usize, max: Option<usize>, greedy: bool) {
        // `min` mandatory copies.
        for _ in 0..min {
            self.compile(inner);
        }
        match max {
            // Unbounded tail: `(inner)*` after the mandatory copies.
            None => self.compile_star(inner, greedy),
            // `{min,max}`: (max - min) optional copies.
            Some(max) => {
                for _ in min..max {
                    self.compile_optional(inner, greedy);
                }
            }
        }
    }

    /// `inner*` — `L1: Split(body, exit); body; Jmp L1; exit:` (greedy prefers
    /// the body; lazy prefers the exit).
    fn compile_star(&mut self, inner: &Node, greedy: bool) {
        let l1 = self.emit(Inst::Split(0, 0));
        let body = self.here();
        self.compile(inner);
        self.emit(Inst::Jmp(l1));
        let exit = self.here();
        self.prog[l1] = if greedy {
            Inst::Split(body, exit)
        } else {
            Inst::Split(exit, body)
        };
    }

    /// `inner?` — `Split(body, exit); body; exit:`.
    fn compile_optional(&mut self, inner: &Node, greedy: bool) {
        let split = self.emit(Inst::Split(0, 0));
        let body = self.here();
        self.compile(inner);
        let exit = self.here();
        self.prog[split] = if greedy {
            Inst::Split(body, exit)
        } else {
            Inst::Split(exit, body)
        };
    }
}

fn convert_item(item: &ClassItem) -> ClassMember {
    match item {
        ClassItem::Char(c) => ClassMember::Char(*c),
        ClassItem::Range(a, b) => ClassMember::Range(*a, *b),
        ClassItem::Shorthand(s) => ClassMember::Shorthand(*s),
    }
}
