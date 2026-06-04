//! A small AST→bytecode compiler for the *expression* subset, plus a thin
//! statement wrapper that returns the value of the final expression.
//!
//! This is the first slice of the Phase-D pipeline: it lowers an [`Expr`] to a
//! [`Chunk`] of register bytecode that [`super::vm`] then executes, reusing the
//! interpreter's value semantics (so the bytecode path stays consistent with
//! the tree-walker). Constructs outside the supported subset return
//! [`CompileError::Unsupported`] so the caller can fall back to the tree-walker.

use crate::ast::{BinaryOp, Expr, LogicalOp, Stmt, UnaryOp};
use crate::bytecode::{Chunk, Const, Op, Reg};
use alloc::format;
use alloc::string::String;

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

/// Compiles a program whose value is its final expression statement into a
/// chunk that leaves that value in the returned register.
pub(super) fn compile_program(body: &[Stmt]) -> Result<Chunk, CompileError> {
    let mut c = Compiler {
        chunk: Chunk::new("<expr>"),
        next_reg: 0,
    };
    let mut last: Option<Reg> = None;
    for stmt in body {
        match stmt {
            Stmt::Expr { expression, .. } => {
                last = Some(c.expr(expression)?);
            }
            Stmt::Empty { .. } => {}
            _ => return Err(CompileError::unsupported("statement in bytecode mode")),
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
}

impl Compiler {
    /// Allocates a fresh register.
    fn reg(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        r
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
                let dst = self.reg();
                let name = self
                    .chunk
                    .add_constant(Const::Str(id.name.clone().into_string()));
                self.chunk.emit(Op::GetGlobal { dst, name });
                Ok(dst)
            }
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

    fn binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> Result<Reg, CompileError> {
        let a = self.expr(left)?;
        let b = self.expr(right)?;
        let dst = self.reg();
        let make = |dst, a, b| match op {
            BinaryOp::Add => Some(Op::Add { dst, a, b }),
            BinaryOp::Sub => Some(Op::Sub { dst, a, b }),
            BinaryOp::Mul => Some(Op::Mul { dst, a, b }),
            BinaryOp::Div => Some(Op::Div { dst, a, b }),
            BinaryOp::Mod => Some(Op::Mod { dst, a, b }),
            BinaryOp::Exp => Some(Op::Pow { dst, a, b }),
            BinaryOp::EqEq => Some(Op::Eq { dst, a, b }),
            BinaryOp::EqEqEq => Some(Op::StrictEq { dst, a, b }),
            BinaryOp::Lt => Some(Op::Lt { dst, a, b }),
            BinaryOp::LtEq => Some(Op::Le { dst, a, b }),
            BinaryOp::Gt => Some(Op::Gt { dst, a, b }),
            BinaryOp::GtEq => Some(Op::Ge { dst, a, b }),
            _ => None,
        };
        match make(dst, a, b) {
            Some(op) => {
                self.chunk.emit(op);
                Ok(dst)
            }
            None => Err(CompileError::unsupported("binary operator")),
        }
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
