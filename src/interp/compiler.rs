//! A small AST→bytecode compiler for the *expression* subset, plus a thin
//! statement wrapper that returns the value of the final expression.
//!
//! This is the first slice of the Phase-D pipeline: it lowers an [`Expr`] to a
//! [`Chunk`] of register bytecode that [`super::vm`] then executes, reusing the
//! interpreter's value semantics (so the bytecode path stays consistent with
//! the tree-walker). Constructs outside the supported subset return
//! [`CompileError::Unsupported`] so the caller can fall back to the tree-walker.

use crate::ast::{AssignOp, BinaryOp, BindingTarget, Expr, LogicalOp, Stmt, UnaryOp};
use crate::bytecode::{Chunk, Const, Op, Reg};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// A reason the expression could not be compiled (yet).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    /// What was unsupported.
    pub message: String,
}

impl CompileError {
    fn unsupported(what: &str) -> Self {
        Self {
            message: format!("bytecode compiler: unsupported {what}"),
        }
    }
}

/// Compiles a program (the supported statement subset) into a chunk whose
/// returned value is that of the final expression statement (REPL-style).
pub(super) fn compile_program(body: &[Stmt]) -> Result<Chunk, CompileError> {
    let mut c = Compiler {
        chunk: Chunk::new("<main>"),
        next_reg: 0,
        locals: Vec::new(),
    };
    let mut last: Option<Reg> = None;
    for stmt in body {
        if let Some(reg) = c.stmt(stmt)? {
            last = Some(reg);
        }
    }
    match last {
        Some(reg) => c.chunk.emit(Op::Return { src: reg }),
        None => c.chunk.emit(Op::ReturnUndefined),
    };
    c.chunk.register_count = c.next_reg;
    Ok(c.chunk)
}

struct Compiler {
    chunk: Chunk,
    next_reg: Reg,
    /// In-scope local variables mapped to their backing register (innermost
    /// last; shadowing resolves to the latest entry).
    locals: Vec<(String, Reg)>,
}

impl Compiler {
    /// Allocates a fresh register.
    fn reg(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    /// The register backing local `name`, if it is in scope.
    fn resolve(&self, name: &str) -> Option<Reg> {
        self.locals
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    }

    /// Compiles a statement, returning the value register for an expression
    /// statement (so the program's completion value can be tracked).
    fn stmt(&mut self, stmt: &Stmt) -> Result<Option<Reg>, CompileError> {
        match stmt {
            Stmt::Expr { expression, .. } => Ok(Some(self.expr(expression)?)),
            Stmt::Empty { .. } => Ok(None),
            Stmt::Var(decl) => {
                for d in &decl.declarations {
                    let BindingTarget::Ident(id) = &d.target else {
                        return Err(CompileError::unsupported("destructuring declaration"));
                    };
                    let value = match &d.init {
                        Some(init) => self.expr(init)?,
                        None => {
                            let r = self.reg();
                            self.chunk.emit(Op::LoadUndefined { dst: r });
                            r
                        }
                    };
                    // Give the variable its own register, copying the init value
                    // in (so later reuse of `value` cannot clobber it).
                    let slot = self.reg();
                    self.chunk.emit(Op::Move {
                        dst: slot,
                        src: value,
                    });
                    self.locals.push((id.name.clone().into_string(), slot));
                }
                Ok(None)
            }
            Stmt::Block { body, .. } => {
                let mark = self.locals.len();
                for s in body {
                    self.stmt(s)?;
                }
                self.locals.truncate(mark);
                Ok(None)
            }
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                let cond = self.expr(test)?;
                let jf = self.chunk.emit(Op::JumpIfFalse { cond, offset: 0 });
                self.stmt(consequent)?;
                if let Some(alt) = alternate {
                    let jmp = self.chunk.emit(Op::Jump { offset: 0 });
                    self.patch_to_here(jf);
                    self.stmt(alt)?;
                    self.patch_to_here(jmp);
                } else {
                    self.patch_to_here(jf);
                }
                Ok(None)
            }
            Stmt::While { test, body, .. } => {
                let top = self.chunk.code.len();
                let cond = self.expr(test)?;
                let jf = self.chunk.emit(Op::JumpIfFalse { cond, offset: 0 });
                self.stmt(body)?;
                let back = self.chunk.emit(Op::Jump { offset: 0 });
                self.patch_jump(back, top);
                self.patch_to_here(jf);
                Ok(None)
            }
            _ => Err(CompileError::unsupported("statement in bytecode mode")),
        }
    }

    /// Patches a previously-emitted jump at `idx` to target the current end of
    /// code.
    fn patch_to_here(&mut self, idx: usize) {
        let target = self.chunk.code.len();
        self.patch_jump(idx, target);
    }

    /// Patches the jump at `idx` to target instruction `target`.
    fn patch_jump(&mut self, idx: usize, target: usize) {
        let offset = (target as i64 - (idx as i64 + 1)) as i32;
        self.chunk.code[idx] = match &self.chunk.code[idx] {
            Op::Jump { .. } => Op::Jump { offset },
            Op::JumpIfFalse { cond, .. } => Op::JumpIfFalse {
                cond: *cond,
                offset,
            },
            Op::JumpIfTrue { cond, .. } => Op::JumpIfTrue {
                cond: *cond,
                offset,
            },
            other => other.clone(),
        };
    }

    /// Compiles `expr`, leaving its value in the returned register.
    fn expr(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        match expr {
            Expr::Number { value, .. } => {
                let dst = self.reg();
                // Small integers use the compact immediate form.
                if value.fract() == 0.0
                    && *value >= f64::from(i32::MIN)
                    && *value <= f64::from(i32::MAX)
                {
                    self.chunk.emit(Op::LoadInt {
                        dst,
                        value: *value as i32,
                    });
                } else {
                    let k = self.chunk.add_constant(Const::Number(*value));
                    self.chunk.emit(Op::LoadConst { dst, k });
                }
                Ok(dst)
            }
            Expr::Str { value, .. } => {
                let dst = self.reg();
                let k = self
                    .chunk
                    .add_constant(Const::Str(value.clone().into_string()));
                self.chunk.emit(Op::LoadConst { dst, k });
                Ok(dst)
            }
            Expr::Bool { value, .. } => {
                let dst = self.reg();
                self.chunk.emit(Op::LoadBool { dst, value: *value });
                Ok(dst)
            }
            Expr::Null(_) => {
                let dst = self.reg();
                self.chunk.emit(Op::LoadNull { dst });
                Ok(dst)
            }
            Expr::Ident(id) => {
                if id.name.as_ref() == "undefined" {
                    let dst = self.reg();
                    self.chunk.emit(Op::LoadUndefined { dst });
                    return Ok(dst);
                }
                // A local: copy its register so later writes can't clobber it.
                if let Some(slot) = self.resolve(&id.name) {
                    let dst = self.reg();
                    self.chunk.emit(Op::Move { dst, src: slot });
                    return Ok(dst);
                }
                let dst = self.reg();
                let name = self
                    .chunk
                    .add_constant(Const::Str(id.name.clone().into_string()));
                self.chunk.emit(Op::GetGlobal { dst, name });
                Ok(dst)
            }
            Expr::Assign {
                op, target, value, ..
            } => self.assign(*op, target, value),
            Expr::Unary { op, argument, .. } => self.unary(*op, argument),
            Expr::Binary {
                op, left, right, ..
            } => self.binary(*op, left, right),
            Expr::Logical {
                op, left, right, ..
            } => self.logical(*op, left, right),
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                if *optional {
                    return Err(CompileError::unsupported("optional chaining"));
                }
                self.member(object, property)
            }
            Expr::Call {
                callee,
                arguments,
                optional,
                ..
            } => {
                if *optional {
                    return Err(CompileError::unsupported("optional call"));
                }
                self.call(callee, arguments)
            }
            _ => Err(CompileError::unsupported("expression")),
        }
    }

    fn unary(&mut self, op: UnaryOp, argument: &Expr) -> Result<Reg, CompileError> {
        let src = self.expr(argument)?;
        let dst = self.reg();
        match op {
            UnaryOp::Minus => self.chunk.emit(Op::Neg { dst, src }),
            UnaryOp::Not => self.chunk.emit(Op::Not { dst, src }),
            _ => return Err(CompileError::unsupported("unary operator")),
        };
        Ok(dst)
    }

    /// Compiles an assignment `target OP= value`, leaving the assigned value in
    /// the returned register. Only simple identifier targets are supported.
    fn assign(&mut self, op: AssignOp, target: &Expr, value: &Expr) -> Result<Reg, CompileError> {
        let Expr::Ident(id) = target else {
            return Err(CompileError::unsupported("assignment target"));
        };
        // Compute the right-hand side (folding the binary op for `+=` etc.).
        let rhs = self.expr(value)?;
        let result = match compound_binop(op) {
            None => rhs, // plain `=`
            Some(binop) => {
                let cur = self.read_ident(&id.name)?;
                let dst = self.reg();
                self.emit_binop(binop, dst, cur, rhs)?;
                dst
            }
        };
        // Store into the local register or the global binding.
        if let Some(slot) = self.resolve(&id.name) {
            self.chunk.emit(Op::Move {
                dst: slot,
                src: result,
            });
        } else {
            let name = self
                .chunk
                .add_constant(Const::Str(id.name.clone().into_string()));
            self.chunk.emit(Op::SetGlobal { name, src: result });
        }
        Ok(result)
    }

    /// Reads `name` (local copy or global) into a fresh register.
    fn read_ident(&mut self, name: &str) -> Result<Reg, CompileError> {
        if let Some(slot) = self.resolve(name) {
            let dst = self.reg();
            self.chunk.emit(Op::Move { dst, src: slot });
            Ok(dst)
        } else {
            let dst = self.reg();
            let k = self.chunk.add_constant(Const::Str(name.into()));
            self.chunk.emit(Op::GetGlobal { dst, name: k });
            Ok(dst)
        }
    }

    fn binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let a = self.expr(left)?;
        let b = self.expr(right)?;
        let dst = self.reg();
        self.emit_binop(op, dst, a, b)?;
        Ok(dst)
    }

    /// Emits a single binary-op instruction for two existing registers.
    fn emit_binop(&mut self, op: BinaryOp, dst: Reg, a: Reg, b: Reg) -> Result<(), CompileError> {
        let inst = match op {
            BinaryOp::Add => Op::Add { dst, a, b },
            BinaryOp::Sub => Op::Sub { dst, a, b },
            BinaryOp::Mul => Op::Mul { dst, a, b },
            BinaryOp::Div => Op::Div { dst, a, b },
            BinaryOp::Mod => Op::Mod { dst, a, b },
            BinaryOp::Exp => Op::Pow { dst, a, b },
            BinaryOp::EqEq => Op::Eq { dst, a, b },
            BinaryOp::EqEqEq => Op::StrictEq { dst, a, b },
            BinaryOp::Lt => Op::Lt { dst, a, b },
            BinaryOp::LtEq => Op::Le { dst, a, b },
            BinaryOp::Gt => Op::Gt { dst, a, b },
            BinaryOp::GtEq => Op::Ge { dst, a, b },
            _ => return Err(CompileError::unsupported("binary operator")),
        };
        self.chunk.emit(inst);
        Ok(())
    }

    fn logical(&mut self, op: LogicalOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        // Evaluate `left` into `dst`; short-circuit over `right` based on the op.
        let dst = self.reg();
        let l = self.expr(left)?;
        self.chunk.emit(Op::Move { dst, src: l });
        // Reserve the conditional jump; patch its offset after compiling `right`.
        let jump_idx = match op {
            LogicalOp::And => self.chunk.emit(Op::JumpIfFalse {
                cond: dst,
                offset: 0,
            }),
            LogicalOp::Or => self.chunk.emit(Op::JumpIfTrue {
                cond: dst,
                offset: 0,
            }),
            LogicalOp::Nullish => return Err(CompileError::unsupported("`??` operator")),
        };
        let r = self.expr(right)?;
        self.chunk.emit(Op::Move { dst, src: r });
        // Offset is relative to the instruction *after* the jump.
        let target = self.chunk.code.len();
        let offset = (target as i64 - (jump_idx as i64 + 1)) as i32;
        self.chunk.code[jump_idx] = match op {
            LogicalOp::And => Op::JumpIfFalse { cond: dst, offset },
            LogicalOp::Or => Op::JumpIfTrue { cond: dst, offset },
            LogicalOp::Nullish => unreachable!(),
        };
        Ok(dst)
    }

    fn member(
        &mut self,
        object: &Expr,
        property: &crate::ast::PropertyKey,
    ) -> Result<Reg, CompileError> {
        use crate::ast::PropertyKey;
        let obj = self.expr(object)?;
        match property {
            PropertyKey::Ident(name) => {
                let dst = self.reg();
                let key = self
                    .chunk
                    .add_constant(Const::Str(name.clone().into_string()));
                self.chunk.emit(Op::GetProp { dst, obj, key });
                Ok(dst)
            }
            PropertyKey::Str(s) => {
                let dst = self.reg();
                let key = self.chunk.add_constant(Const::Str(s.clone().into_string()));
                self.chunk.emit(Op::GetProp { dst, obj, key });
                Ok(dst)
            }
            PropertyKey::Computed(expr) => {
                let index = self.expr(expr)?;
                let dst = self.reg();
                self.chunk.emit(Op::GetElem { dst, obj, index });
                Ok(dst)
            }
            _ => Err(CompileError::unsupported("member key")),
        }
    }

    fn call(
        &mut self,
        callee: &Expr,
        arguments: &[crate::ast::Argument],
    ) -> Result<Reg, CompileError> {
        use crate::ast::Argument;
        let callee_reg = self.expr(callee)?;
        // Compile each argument first (each may use scratch registers), then
        // copy the results into a fresh contiguous window the `Call` expects.
        let mut arg_regs = alloc::vec::Vec::new();
        for arg in arguments {
            match arg {
                Argument::Item(e) => arg_regs.push(self.expr(e)?),
                Argument::Spread(_) => return Err(CompileError::unsupported("spread argument")),
            }
        }
        let args_base = self.next_reg;
        for &src in &arg_regs {
            let slot = self.reg();
            self.chunk.emit(Op::Move { dst: slot, src });
        }
        let dst = self.reg();
        self.chunk.emit(Op::Call {
            dst,
            callee: callee_reg,
            args_base,
            argc: arg_regs.len() as u16,
        });
        Ok(dst)
    }
}

/// Maps a compound assignment operator to its underlying binary op; `=` returns
/// `None`. Bitwise/logical compound forms are not yet lowered.
fn compound_binop(op: AssignOp) -> Option<BinaryOp> {
    match op {
        AssignOp::Assign => None,
        AssignOp::AddAssign => Some(BinaryOp::Add),
        AssignOp::SubAssign => Some(BinaryOp::Sub),
        AssignOp::MulAssign => Some(BinaryOp::Mul),
        AssignOp::DivAssign => Some(BinaryOp::Div),
        AssignOp::ModAssign => Some(BinaryOp::Mod),
        AssignOp::ExpAssign => Some(BinaryOp::Exp),
        _ => None,
    }
}
