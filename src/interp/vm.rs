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
use crate::bytecode::{Chunk, Const, Module, Op, Reg};
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
        Ok(self.run_chunk(&module, 0, Value::Undefined, Vec::new(), &[]))
    }

    /// Runs a program through the bytecode VM, draining the event loop
    /// afterward, and **falls back to the tree-walker** if the program uses a
    /// construct the bytecode compiler does not yet support. The result is
    /// identical either way (the two paths share value semantics); the VM is
    /// the faster path where it applies.
    pub fn run_with_vm(&mut self, program: &'a crate::ast::Program) -> Completion<'a, Value<'a>> {
        match super::compiler::compile_program(&program.body) {
            Ok(module) => {
                let result = self.run_chunk(&Rc::new(module), 0, Value::Undefined, Vec::new(), &[]);
                self.run_event_loop();
                result
            }
            // Unsupported construct → run the reference tree-walker instead.
            Err(_) => self.run(program),
        }
    }

    /// Executes a compiled module's entry chunk. Together with
    /// [`crate::bytecode::serialize`]/[`deserialize`](crate::bytecode::deserialize)
    /// this is the export/reload path: a module can be compiled once, persisted,
    /// and later reloaded and run without the source.
    pub fn run_module(&mut self, module: Rc<Module>) -> Completion<'a, Value<'a>> {
        self.run_chunk(&module, 0, Value::Undefined, Vec::new(), &[])
    }

    /// Calls a bytecode function value (dispatched from `call_with_this`). Its
    /// captured upvalues are made available to `GetUpvalue`.
    pub(super) fn call_bytecode_fn(
        &mut self,
        func: &Rc<BytecodeFn<'a>>,
        this: Value<'a>,
        args: Vec<Value<'a>>,
    ) -> Completion<'a, Value<'a>> {
        let captures = func.captures.clone();
        self.run_chunk(&func.module, func.chunk, this, args, &captures)
    }

    /// Executes chunk `chunk_idx` of `module` with `this` in register 0,
    /// `args` bound to the parameter registers (1..), and `captures` exposed as
    /// the frame's upvalues, returning its value.
    fn run_chunk(
        &mut self,
        module: &Rc<Module>,
        chunk_idx: u32,
        this: Value<'a>,
        args: Vec<Value<'a>>,
        captures: &[Value<'a>],
    ) -> Completion<'a, Value<'a>> {
        let chunk: &Chunk = &module.chunks[chunk_idx as usize];
        let mut regs: Vec<Value<'a>> = alloc::vec![Value::Undefined; chunk.register_count as usize];
        // Register 0 holds `this`; parameters occupy registers 1.. (extras
        // ignored, missing stay `undefined`).
        if !regs.is_empty() {
            regs[0] = this;
        }
        // With a rest parameter, the fixed params bind positionally and the
        // remaining arguments collect into an array in the rest register.
        let fixed = if chunk.has_rest {
            (chunk.param_count as usize).saturating_sub(1)
        } else {
            chunk.param_count as usize
        };
        let mut args_iter = args.into_iter();
        for slot in 1..=fixed {
            match args_iter.next() {
                Some(arg) if slot < regs.len() => regs[slot] = arg,
                Some(_) | None => break,
            }
        }
        if chunk.has_rest {
            let rest: Vec<Value<'a>> = args_iter.collect();
            let slot = fixed + 1;
            if slot < regs.len() {
                regs[slot] = Value::Object(Obj::array(rest));
            }
        }
        let mut pc = 0usize;
        // Installed exception handlers: `(catch_pc, error_register)`.
        let mut handlers: Vec<(usize, Reg)> = Vec::new();
        loop {
            let op = &chunk.code[pc];
            pc += 1;
            // Run one op inside a closure so a thrown `Err` (from a fallible
            // op or an explicit `Throw`) can be caught and routed to the
            // nearest installed handler. `Ok(Some(v))` is a `Return`.
            let outcome: Completion<'a, Option<Value<'a>>> = (|| {
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
                    Op::MakeClosure {
                        dst,
                        chunk: target,
                        upvals_base,
                        count,
                    } => {
                        let base = *upvals_base as usize;
                        let captured: Vec<Value<'a>> = regs[base..base + *count as usize].to_vec();
                        regs[*dst as usize] =
                            Value::Object(make_closure(module, *target, captured));
                    }
                    Op::GetUpvalue { dst, idx } => {
                        regs[*dst as usize] = captures
                            .get(*idx as usize)
                            .cloned()
                            .unwrap_or(Value::Undefined);
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
                    Op::Binary { dst, a, b, op } => {
                        self.bin(&mut regs, binop_from_code(*op), *dst, *a, *b)?;
                    }
                    Op::TypeOf { dst, src } => {
                        regs[*dst as usize] = Value::str(regs[*src as usize].type_of());
                    }
                    Op::DeleteMember { dst, obj, key } => {
                        let removed = match &regs[*obj as usize] {
                            Value::Object(o) => o.delete_key(&regs[*key as usize].to_js_string()),
                            _ => true,
                        };
                        regs[*dst as usize] = Value::Bool(removed);
                    }
                    Op::IterValues { dst, src } => {
                        let v = regs[*src as usize].clone();
                        let iterable = matches!(&v, Value::Str(_))
                            || matches!(&v, Value::Object(o) if o.is_array() || o.as_collection().is_some());
                        if !iterable {
                            return Err(super::eval::make_error(
                                "TypeError",
                                alloc::format!("{} is not iterable", v.to_js_string()),
                            ));
                        }
                        let mut items = Vec::new();
                        super::builtins::iterate_into(&v, &mut items);
                        regs[*dst as usize] = Value::Object(Obj::array(items));
                    }
                    Op::IterKeys { dst, src } => {
                        // Own enumerable keys (array indices / property names).
                        let keys: Vec<Value<'a>> = match &regs[*src as usize] {
                            Value::Object(o) => o.own_keys().into_iter().map(Value::str).collect(),
                            _ => Vec::new(),
                        };
                        regs[*dst as usize] = Value::Object(Obj::array(keys));
                    }
                    Op::TypeOfGlobal { dst, name } => {
                        let key = const_str(chunk, *name);
                        // An unbound global yields "undefined" (no throw).
                        let ty = self.global().get(&key).map_or("undefined", |v| v.type_of());
                        regs[*dst as usize] = Value::str(ty);
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

                    Op::New {
                        dst,
                        callee,
                        args_base,
                        argc,
                    } => {
                        let callee_val = regs[*callee as usize].clone();
                        let base = *args_base as usize;
                        let args: Vec<Value<'a>> = regs[base..base + *argc as usize].to_vec();
                        regs[*dst as usize] = self.construct(callee_val, args)?;
                    }

                    Op::CallMethod {
                        dst,
                        recv,
                        key,
                        args_base,
                        argc,
                    } => {
                        let receiver = regs[*recv as usize].clone();
                        let key = regs[*key as usize].to_js_string();
                        let base = *args_base as usize;
                        let args: Vec<Value<'a>> = regs[base..base + *argc as usize].to_vec();
                        regs[*dst as usize] = self.call_member(receiver, &key, args)?;
                    }

                    Op::Return { src } => return Ok(Some(regs[*src as usize].clone())),
                    Op::ReturnUndefined => return Ok(Some(Value::Undefined)),

                    Op::Throw { src } => return Err(regs[*src as usize].clone()),
                    Op::PushHandler { catch, err } => {
                        handlers.push((apply_offset(pc, *catch), *err));
                    }
                    Op::PopHandler => {
                        handlers.pop();
                    }

                    Op::NewObject { dst } => {
                        regs[*dst as usize] = Value::Object(Obj::object());
                    }
                    Op::NewArray { dst, len } => {
                        regs[*dst as usize] =
                            Value::Object(Obj::array(alloc::vec![Value::Undefined; *len as usize]));
                    }
                    Op::SetProp { obj, key, src } => {
                        let k = const_str(chunk, *key);
                        let target = regs[*obj as usize].clone();
                        let value = regs[*src as usize].clone();
                        self.set_member(&target, &k, value)?;
                    }
                    Op::SetElem { obj, index, src } => {
                        let target = regs[*obj as usize].clone();
                        let key = regs[*index as usize].to_js_string();
                        let value = regs[*src as usize].clone();
                        self.set_member(&target, &key, value)?;
                    }
                }
                Ok(None)
            })();
            match outcome {
                Ok(Some(value)) => return Ok(value),
                Ok(None) => {}
                // Unwind to the nearest installed handler; if none, propagate.
                Err(error) => match handlers.pop() {
                    Some((catch_pc, err_reg)) => {
                        regs[err_reg as usize] = error;
                        pc = catch_pc;
                    }
                    None => return Err(error),
                },
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
    make_closure(module, chunk, Vec::new())
}

/// Builds a closure value over `chunk`, carrying its captured upvalue cells.
fn make_closure<'a>(module: &Rc<Module>, chunk: u32, captures: Vec<Value<'a>>) -> Rc<Obj<'a>> {
    let obj = Obj::object();
    obj.set_bytecode_fn(Rc::new(BytecodeFn {
        module: Rc::clone(module),
        chunk,
        captures,
    }));
    obj
}

/// Maps a generic [`Op::Binary`] operator code back to its [`BinaryOp`].
fn binop_from_code(code: u8) -> BinaryOp {
    use crate::bytecode::binop;
    match code {
        binop::BIT_AND => BinaryOp::BitAnd,
        binop::BIT_OR => BinaryOp::BitOr,
        binop::BIT_XOR => BinaryOp::BitXor,
        binop::SHL => BinaryOp::Shl,
        binop::SHR => BinaryOp::Shr,
        binop::USHR => BinaryOp::Ushr,
        binop::IN => BinaryOp::In,
        _ => BinaryOp::Instanceof,
    }
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
