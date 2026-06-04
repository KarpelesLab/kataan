//! A register-machine interpreter for the [`crate::bytecode`] language subset.
//!
//! It executes a [`Module`]'s chunks against the [`Interp`]'s globals, reusing
//! the tree-walker's value operations (`eval_binary`, `get_member`,
//! `call_with_this`) so the two execution paths agree on semantics. Function
//! calls recurse through `call_with_this`, so the Rust call stack mirrors the
//! JS one (and bytecode/native/tree-walker callees interoperate).

use super::Completion;
use super::eval::Interp;
use super::value::{BytecodeFn, Obj, Value};
use crate::ast::BinaryOp;
use crate::bytecode::{Chunk, Const, Module, Op};
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

impl<'a> Interp<'a> {
    /// Compiles a program's statements to bytecode and executes them, returning
    /// the value of the final expression. Returns a
    /// [`CompileError`](super::compiler::CompileError) if the program uses a
    /// construct the bytecode compiler does not yet support (the caller can
    /// then fall back to the tree-walker).
    pub fn eval_via_bytecode(
        &mut self,
        body: &[crate::ast::Stmt],
    ) -> Result<Completion<'a, Value<'a>>, super::compiler::CompileError> {
        let module = Rc::new(super::compiler::compile_program(body)?);
        Ok(self.run_chunk(&module, 0, Vec::new()))
    }

    /// Calls a bytecode function value (dispatched from `call_with_this`).
    pub(super) fn call_bytecode_fn(
        &mut self,
        func: &Rc<BytecodeFn<'a>>,
        args: Vec<Value<'a>>,
    ) -> Completion<'a, Value<'a>> {
        self.run_chunk(&func.module, func.chunk, args)
    }

    /// Executes chunk `chunk_idx` of `module` with `args` bound to its leading
    /// (parameter) registers, returning the value it returns.
    fn run_chunk(
        &mut self,
        module: &Rc<Module>,
        chunk_idx: u32,
        args: Vec<Value<'a>>,
    ) -> Completion<'a, Value<'a>> {
        let chunk: &Chunk = &module.chunks[chunk_idx as usize];
        let mut regs: Vec<Value<'a>> = alloc::vec![Value::Undefined; chunk.register_count as usize];
        // Bind arguments to the parameter registers (extras ignored, missing
        // stay `undefined`).
        for (i, arg) in args.into_iter().enumerate() {
            if i < regs.len() {
                regs[i] = arg;
            }
        }
        let mut pc = 0usize;
        loop {
            let op = &chunk.code[pc];
            pc += 1;
            match op {
                Op::LoadConst { dst, k } => {
                    regs[*dst as usize] = match &chunk.constants[*k as usize] {
                        Const::Number(n) => Value::Number(*n),
                        Const::Str(s) => Value::str(s.clone()),
                        // A function constant materializes a bytecode-function
                        // value over the same module.
                        Const::Func(idx) => Value::Object(make_bytecode_fn(module, *idx)),
                    };
                }
                Op::LoadUndefined { dst } => regs[*dst as usize] = Value::Undefined,
                Op::LoadNull { dst } => regs[*dst as usize] = Value::Null,
                Op::LoadBool { dst, value } => regs[*dst as usize] = Value::Bool(*value),
                Op::LoadInt { dst, value } => {
                    regs[*dst as usize] = Value::Number(f64::from(*value));
                }
                Op::Move { dst, src } => regs[*dst as usize] = regs[*src as usize].clone(),

                Op::Add { dst, a, b } => self.bin(&mut regs, BinaryOp::Add, *dst, *a, *b)?,
                Op::Sub { dst, a, b } => self.bin(&mut regs, BinaryOp::Sub, *dst, *a, *b)?,
                Op::Mul { dst, a, b } => self.bin(&mut regs, BinaryOp::Mul, *dst, *a, *b)?,
                Op::Div { dst, a, b } => self.bin(&mut regs, BinaryOp::Div, *dst, *a, *b)?,
                Op::Mod { dst, a, b } => self.bin(&mut regs, BinaryOp::Mod, *dst, *a, *b)?,
                Op::Pow { dst, a, b } => self.bin(&mut regs, BinaryOp::Exp, *dst, *a, *b)?,
                Op::Eq { dst, a, b } => self.bin(&mut regs, BinaryOp::EqEq, *dst, *a, *b)?,
                Op::StrictEq { dst, a, b } => {
                    self.bin(&mut regs, BinaryOp::EqEqEq, *dst, *a, *b)?;
                }
                Op::Lt { dst, a, b } => self.bin(&mut regs, BinaryOp::Lt, *dst, *a, *b)?,
                Op::Le { dst, a, b } => self.bin(&mut regs, BinaryOp::LtEq, *dst, *a, *b)?,
                Op::Gt { dst, a, b } => self.bin(&mut regs, BinaryOp::Gt, *dst, *a, *b)?,
                Op::Ge { dst, a, b } => self.bin(&mut regs, BinaryOp::GtEq, *dst, *a, *b)?,

                Op::Neg { dst, src } => {
                    regs[*dst as usize] = Value::Number(-regs[*src as usize].to_number());
                }
                Op::Not { dst, src } => {
                    regs[*dst as usize] = Value::Bool(!regs[*src as usize].to_boolean());
                }

                Op::GetGlobal { dst, name } => {
                    let key = const_str(chunk, *name);
                    let value = self.global().get(&key).ok_or_else(|| {
                        super::eval::make_error(
                            "ReferenceError",
                            alloc::format!("{key} is not defined"),
                        )
                    })?;
                    regs[*dst as usize] = value;
                }
                Op::SetGlobal { name, src } => {
                    let key = const_str(chunk, *name);
                    self.global()
                        .declare(&key, regs[*src as usize].clone(), true);
                }

                Op::GetProp { dst, obj, key } => {
                    let k = const_str(chunk, *key);
                    let obj = regs[*obj as usize].clone();
                    regs[*dst as usize] = self.get_member(&obj, &k)?;
                }
                Op::GetElem { dst, obj, index } => {
                    let obj = regs[*obj as usize].clone();
                    let key = regs[*index as usize].to_js_string();
                    regs[*dst as usize] = self.get_member(&obj, &key)?;
                }

                Op::Jump { offset } => pc = apply_offset(pc, *offset),
                Op::JumpIfFalse { cond, offset } => {
                    if !regs[*cond as usize].to_boolean() {
                        pc = apply_offset(pc, *offset);
                    }
                }
                Op::JumpIfTrue { cond, offset } => {
                    if regs[*cond as usize].to_boolean() {
                        pc = apply_offset(pc, *offset);
                    }
                }

                Op::Call {
                    dst,
                    callee,
                    args_base,
                    argc,
                } => {
                    let callee_val = regs[*callee as usize].clone();
                    let base = *args_base as usize;
                    let args: Vec<Value<'a>> = regs[base..base + *argc as usize].to_vec();
                    regs[*dst as usize] =
                        self.call_with_this(callee_val, Value::Undefined, args)?;
                }

                Op::Return { src } => return Ok(regs[*src as usize].clone()),
                Op::ReturnUndefined => return Ok(Value::Undefined),

                // Object/array construction is added with a later compiler slice.
                Op::NewObject { .. }
                | Op::NewArray { .. }
                | Op::SetProp { .. }
                | Op::SetElem { .. } => {
                    return Err(super::eval::make_error(
                        "InternalError",
                        "bytecode op not yet implemented in the VM",
                    ));
                }
            }
        }
    }

    /// Applies a binary op to two registers, writing the result into `dst`.
    fn bin(
        &mut self,
        regs: &mut [Value<'a>],
        op: BinaryOp,
        dst: u16,
        a: u16,
        b: u16,
    ) -> Completion<'a, ()> {
        let l = regs[a as usize].clone();
        let r = regs[b as usize].clone();
        regs[dst as usize] = self.eval_binary(op, l, r)?;
        Ok(())
    }
}

/// Builds a bytecode-function value (an object carrying the compiled function).
fn make_bytecode_fn<'a>(module: &Rc<Module>, chunk: u32) -> Rc<Obj<'a>> {
    let obj = Obj::object();
    obj.set_bytecode_fn(Rc::new(BytecodeFn {
        module: Rc::clone(module),
        chunk,
        captures: Vec::new(),
    }));
    obj
}

/// Reads a string constant from the pool.
fn const_str(chunk: &Chunk, idx: u32) -> String {
    match &chunk.constants[idx as usize] {
        Const::Str(s) => s.clone(),
        other => alloc::format!("{other:?}"),
    }
}

/// Applies a signed instruction offset to a program counter.
fn apply_offset(pc: usize, offset: i32) -> usize {
    (pc as i64 + i64::from(offset)) as usize
}
