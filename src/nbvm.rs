//! A minimal register VM over the [`Realm`] / [`NanBox`] representation
//! (`ROADMAP.md` §3 → Phase D migration).
//!
//! [`Realm`]: crate::realm::Realm
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! This is the **proof of execution** for the performance object model: a small
//! register machine whose values are [`NanBox`]s and whose objects live in a
//! [`Realm`]'s heap under the GC. It demonstrates that the foundation actually
//! *runs* code — arithmetic on boxed numbers, control flow off `ToBoolean`,
//! and object property reads/writes through shapes — end to end, ahead of
//! migrating the full bytecode VM onto this representation.
//!
//! It also carries the **bytecode-VM fold**: an AST → bytecode `compile_and_run`
//! that lowers a growing JavaScript subset (arithmetic, control flow,
//! arrays/objects, `for`/`do-while` loops, compound assignment and `++`/`--`,
//! functions with recursion via a per-activation register window, `try`/`catch`/
//! `throw` exceptions that unwind across calls, and native `console.log`/`Math.*`
//! calls) onto these ops, with no tree-walking — and it agrees with the
//! tree-walker on output (a cross-engine parity test). Closures and first-class
//! function values arrive with later slices; the point is that every value
//! flowing through it is a single 64-bit word and every object is a GC-managed
//! heap node, exactly as the production VM will work.
//!
//! Pure, safe `alloc`-only Rust.

use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::Realm;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// A register index.
pub type Reg = u16;

/// An instruction of the minimal VM. Register operands index a flat register
/// file of [`NanBox`] values.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum Op {
    /// `dst = constant`.
    LoadConst { dst: Reg, value: NanBox },
    /// `dst = a + b` (numeric).
    Add { dst: Reg, a: Reg, b: Reg },
    /// `dst = a - b` (numeric).
    Sub { dst: Reg, a: Reg, b: Reg },
    /// `dst = a * b` (numeric).
    Mul { dst: Reg, a: Reg, b: Reg },
    /// `dst = a / b` (numeric).
    Div { dst: Reg, a: Reg, b: Reg },
    /// `dst = -a` (numeric negation).
    Neg { dst: Reg, a: Reg },
    /// `dst = !a` (ECMAScript `ToBoolean` then logical not).
    Not { dst: Reg, a: Reg },
    /// `dst = (a < b)` (numeric) as a boolean.
    Lt { dst: Reg, a: Reg, b: Reg },
    /// `dst = src` (register copy).
    Move { dst: Reg, src: Reg },
    /// Jump to `target` if `cond` is falsy (ECMAScript `ToBoolean`).
    JumpIfFalse { cond: Reg, target: usize },
    /// Unconditional jump to `target`.
    Jump { target: usize },
    /// `dst = a + b` via the realm's `+` (numeric add or string concatenation).
    AddValue { dst: Reg, a: Reg, b: Reg },
    /// `dst = (a === b)` via the realm's strict equality (strings by value).
    StrictEq { dst: Reg, a: Reg, b: Reg },
    /// `dst = a new heap string`.
    NewString { dst: Reg, value: String },
    /// `dst = a new array of `len` `undefined` elements`.
    NewArray { dst: Reg, len: usize },
    /// `dst = arr[index]` (index taken from a register, `undefined` if absent).
    GetElem { dst: Reg, arr: Reg, index: Reg },
    /// `arr[index] = src` (grows the array if needed).
    SetElem { arr: Reg, index: Reg, src: Reg },
    /// `dst = arr.length`.
    ArrayLen { dst: Reg, arr: Reg },
    /// `dst = a new empty object` (allocated in the realm's heap).
    NewObject { dst: Reg },
    /// `obj[key] = src` (own property set through the object's shape).
    SetProp { obj: Reg, key: String, src: Reg },
    /// `dst = obj[key]` (`undefined` if absent).
    GetProp { dst: Reg, obj: Reg, key: String },
    /// `dst = func(args…)` — call function `func` (an index into the program's
    /// function table) with the values in the `args` registers.
    Call { dst: Reg, func: u32, args: Vec<Reg> },
    /// `dst = native#id(args…)` — invoke a built-in (`console.log`, `Math.*`).
    CallNative {
        dst: Reg,
        native: u16,
        args: Vec<Reg>,
    },
    /// Installs an exception handler: on a throw, control jumps to `target` with
    /// the thrown value placed in register `reg`.
    PushHandler { target: usize, reg: Reg },
    /// Removes the most recently installed handler (a try block completed).
    PopHandler,
    /// Throws the value in `src` (caught by the nearest handler, else unwinds
    /// the call stack).
    Throw { src: Reg },
    /// Halt, yielding the value in `src`.
    Return { src: Reg },
}

// Native built-in ids for `Op::CallNative`.
const NB_CONSOLE_LOG: u16 = 0;
const NB_MATH_MAX: u16 = 1;
const NB_MATH_MIN: u16 = 2;
const NB_MATH_ABS: u16 = 3;

/// A compiled function: its instruction stream, register-file size, and the
/// number of leading registers that receive call arguments.
#[derive(Clone, Debug)]
pub struct FnProto {
    /// The instruction stream.
    pub ops: Vec<Op>,
    /// Total registers the body uses.
    pub n_regs: usize,
    /// Parameters, bound to registers `0..n_params` on entry.
    pub n_params: usize,
}

/// Execution context shared across activations: the heap and the captured
/// `console` output sink.
struct Ctx<'a> {
    realm: &'a mut Realm,
    output: String,
}

/// Runs function `id` of `funcs` with `args`, allocating in `realm`. Calls
/// recurse on the Rust stack — one register window per activation, exactly the
/// frame model the production VM will use.
///
/// # Errors
/// Propagates a [`VmError`] from any faulting instruction.
pub fn run_program(
    realm: &mut Realm,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
) -> Result<NanBox, VmError> {
    let mut ctx = Ctx {
        realm,
        output: String::new(),
    };
    call(&mut ctx, funcs, id, args)
}

/// Like [`run_program`], also returning the captured `console` output.
///
/// # Errors
/// Propagates a [`VmError`] from any faulting instruction.
pub fn run_program_capturing(
    realm: &mut Realm,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
) -> Result<(NanBox, String), VmError> {
    let mut ctx = Ctx {
        realm,
        output: String::new(),
    };
    let value = call(&mut ctx, funcs, id, args)?;
    Ok((value, ctx.output))
}

fn call(ctx: &mut Ctx, funcs: &[FnProto], id: usize, args: &[NanBox]) -> Result<NanBox, VmError> {
    let proto = &funcs[id];
    let mut regs: Vec<NanBox> = vec![NanBox::undefined(); proto.n_regs];
    for (i, a) in args.iter().enumerate().take(proto.n_params) {
        regs[i] = *a;
    }
    Ok(run_frame(ctx, funcs, &proto.ops, &mut regs)?.unwrap_or(NanBox::undefined()))
}

/// Why execution stopped abnormally.
#[derive(Clone, PartialEq, Debug)]
pub enum VmError {
    /// An arithmetic op saw a non-number operand (this toy VM has no coercion).
    NotANumber,
    /// A property op was used on a non-object operand.
    NotAnObject,
    /// An uncaught `throw` propagating out of the call stack (the thrown value).
    Thrown(NanBox),
}

/// Runs `program` with a register file of `register_count` slots (initialized to
/// `undefined`), allocating objects in `realm`. Returns the `Return`ed value, or
/// `undefined` if the program falls off the end. (Convenience for call-free
/// programs; `Call` ops require [`run_program`]'s function table.)
pub fn run(realm: &mut Realm, program: &[Op], register_count: usize) -> Result<NanBox, VmError> {
    let mut regs: Vec<NanBox> = vec![NanBox::undefined(); register_count];
    let mut ctx = Ctx {
        realm,
        output: String::new(),
    };
    Ok(run_frame(&mut ctx, &[], program, &mut regs)?.unwrap_or(NanBox::undefined()))
}

/// Executes one function body (`program`) against the register file `regs`.
/// Returns `Some(value)` on `Return`, `None` if control falls off the end.
/// `Call` ops dispatch into `funcs` via [`call`] (a fresh register window per
/// activation); `CallNative` dispatches to a built-in via [`call_native`].
fn run_frame(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    program: &[Op],
    regs: &mut [NanBox],
) -> Result<Option<NanBox>, VmError> {
    let mut pc = 0;
    // Active exception handlers: `(catch_pc, catch_reg)`, innermost last.
    let mut handlers: Vec<(usize, Reg)> = Vec::new();

    let num = |v: NanBox| v.as_number().ok_or(VmError::NotANumber);
    // A register holding an object: recover its heap handle from the boxed value
    // (no side table — the handle *is* the value's payload).
    let object_handle = |v: NanBox| {
        v.as_handle()
            .map(Handle::from_raw)
            .ok_or(VmError::NotAnObject)
    };

    while pc < program.len() {
        let op = &program[pc];
        pc += 1;
        match op {
            Op::LoadConst { dst, value } => regs[*dst as usize] = *value,
            Op::Add { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? + num(regs[*b as usize])?);
            }
            Op::Sub { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? - num(regs[*b as usize])?);
            }
            Op::Mul { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? * num(regs[*b as usize])?);
            }
            Op::Div { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? / num(regs[*b as usize])?);
            }
            Op::Neg { dst, a } => {
                regs[*dst as usize] = NanBox::number(-num(regs[*a as usize])?);
            }
            Op::Not { dst, a } => {
                regs[*dst as usize] = NanBox::boolean(!regs[*a as usize].to_boolean());
            }
            Op::Move { dst, src } => regs[*dst as usize] = regs[*src as usize],
            Op::Lt { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::boolean(num(regs[*a as usize])? < num(regs[*b as usize])?);
            }
            Op::AddValue { dst, a, b } => {
                regs[*dst as usize] = ctx.realm.add(regs[*a as usize], regs[*b as usize]);
            }
            Op::StrictEq { dst, a, b } => {
                regs[*dst as usize] = NanBox::boolean(
                    ctx.realm
                        .strict_equals(regs[*a as usize], regs[*b as usize]),
                );
            }
            Op::JumpIfFalse { cond, target } => {
                if !regs[*cond as usize].to_boolean() {
                    pc = *target;
                }
            }
            Op::Jump { target } => pc = *target,
            Op::NewString { dst, value } => {
                let handle = ctx.realm.new_string(value);
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::NewArray { dst, len } => {
                let handle = ctx.realm.new_array(vec![NanBox::undefined(); *len]);
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::GetElem { dst, arr, index } => {
                let handle = object_handle(regs[*arr as usize])?;
                let i = num(regs[*index as usize])? as usize;
                regs[*dst as usize] = ctx.realm.get_element(handle, i);
            }
            Op::SetElem { arr, index, src } => {
                let handle = object_handle(regs[*arr as usize])?;
                let i = num(regs[*index as usize])? as usize;
                ctx.realm.set_element(handle, i, regs[*src as usize]);
            }
            Op::ArrayLen { dst, arr } => {
                let handle = object_handle(regs[*arr as usize])?;
                let len = ctx.realm.array_length(handle).unwrap_or(0);
                regs[*dst as usize] = NanBox::number(len as f64);
            }
            Op::NewObject { dst } => {
                let handle = ctx.realm.new_object();
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::SetProp { obj, key, src } => {
                let handle = object_handle(regs[*obj as usize])?;
                ctx.realm.set_property(handle, key, regs[*src as usize]);
            }
            Op::GetProp { dst, obj, key } => {
                let handle = object_handle(regs[*obj as usize])?;
                regs[*dst as usize] = ctx
                    .realm
                    .get_property(handle, key)
                    .unwrap_or(NanBox::undefined());
            }
            Op::Call { dst, func, args } => {
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                // A throw from the callee is caught by this frame's nearest
                // handler, else it keeps unwinding.
                match call(ctx, funcs, *func as usize, &argv) {
                    Ok(ret) => regs[*dst as usize] = ret,
                    Err(VmError::Thrown(v)) => match handlers.pop() {
                        Some((target, reg)) => {
                            regs[reg as usize] = v;
                            pc = target;
                        }
                        None => return Err(VmError::Thrown(v)),
                    },
                    Err(other) => return Err(other),
                }
            }
            Op::CallNative { dst, native, args } => {
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                regs[*dst as usize] = call_native(ctx, *native, &argv);
            }
            Op::PushHandler { target, reg } => handlers.push((*target, *reg)),
            Op::PopHandler => {
                handlers.pop();
            }
            Op::Throw { src } => {
                let v = regs[*src as usize];
                match handlers.pop() {
                    Some((target, reg)) => {
                        regs[reg as usize] = v;
                        pc = target;
                    }
                    None => return Err(VmError::Thrown(v)),
                }
            }
            Op::Return { src } => return Ok(Some(regs[*src as usize])),
        }
    }
    Ok(None)
}

/// Invokes a built-in by id (`console.log` writes to `ctx.output`; `Math.*`
/// fold over the numeric arguments).
fn call_native(ctx: &mut Ctx, native: u16, args: &[NanBox]) -> NanBox {
    match native {
        NB_CONSOLE_LOG => {
            let line: Vec<String> = args
                .iter()
                .map(|a| ctx.realm.to_display_string(*a))
                .collect();
            ctx.output.push_str(&line.join(" "));
            ctx.output.push('\n');
            NanBox::undefined()
        }
        NB_MATH_MAX | NB_MATH_MIN | NB_MATH_ABS => {
            let mut nums = args.iter().filter_map(|a| a.as_number());
            let val = match native {
                NB_MATH_ABS => nums.next().map(f64::abs).unwrap_or(f64::NAN),
                NB_MATH_MAX => nums.fold(f64::NEG_INFINITY, f64::max),
                _ => nums.fold(f64::INFINITY, f64::min),
            };
            NanBox::number(val)
        }
        _ => NanBox::undefined(),
    }
}

// --- AST → bytecode compiler (the first slice of the bytecode-VM fold) ---

use crate::ast::{
    ArrayElement, BinaryOp, BindingTarget, Expr, ForInit, Ident, LogicalOp, ObjectMember, Program,
    PropertyKey, Stmt, UnaryOp,
};

/// Why compilation could not proceed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CompileError {
    /// A construct the bytecode compiler does not yet handle.
    Unsupported(&'static str),
    /// A reference to a name not declared in scope.
    Undefined(String),
}

/// Compiles a (subset of a) program to [`Op`]s over the [`Realm`]/[`NanBox`]
/// model and runs it, returning the completion value. Supports numeric/string/
/// boolean/null literals, `let` bindings and assignment, arithmetic and
/// comparison operators, `!`/unary `-`, `&&`/`||`/`?:` short-circuiting, and
/// `if`/`while`/block control flow — the first slice of the production VM's
/// migration onto the new representation.
///
/// # Errors
/// Returns [`CompileError`] for unsupported constructs and [`VmError`] (wrapped)
/// for runtime faults.
pub fn compile_and_run(realm: &mut Realm, program: &Program) -> Result<NanBox, CompileError> {
    let protos = compile_program(program)?;
    run_program(realm, &protos, 0, &[]).map_err(|_| CompileError::Unsupported("runtime fault"))
}

/// Compiles and runs `program`, returning its completion value and captured
/// `console` output — the bytecode path's analogue of the tree-walker's
/// `eval_source`.
///
/// # Errors
/// Returns [`CompileError`] for unsupported constructs / runtime faults.
pub fn compile_run_output(
    realm: &mut Realm,
    program: &Program,
) -> Result<(NanBox, String), CompileError> {
    let protos = compile_program(program)?;
    run_program_capturing(realm, &protos, 0, &[])
        .map_err(|_| CompileError::Unsupported("runtime fault"))
}

/// Compiles `program` to a function table (function 0 is the top-level body).
///
/// # Errors
/// Returns [`CompileError`] for unsupported constructs.
pub fn compile_program(program: &Program) -> Result<Vec<FnProto>, CompileError> {
    let decls: Vec<&crate::ast::Function> = program
        .body
        .iter()
        .filter_map(|s| match s {
            Stmt::Function(f) => Some(f),
            _ => None,
        })
        .collect();
    let mut fn_ids = alloc::collections::BTreeMap::new();
    for (i, f) in decls.iter().enumerate() {
        if let Some(id) = &f.id {
            fn_ids.insert(String::from(&*id.name), (i + 1) as u32);
        }
    }
    let mut protos = Vec::with_capacity(decls.len() + 1);
    protos.push(Compiler::compile_fn(&fn_ids, &[], &program.body, true)?);
    for f in &decls {
        protos.push(Compiler::compile_fn(&fn_ids, &f.params, &f.body, false)?);
    }
    Ok(protos)
}

/// Maps a built-in namespace member call (`console.log`, `Math.max`/`min`/
/// `abs`) to its native id, if the callee is such a member.
fn native_call(callee: &Expr) -> Option<u16> {
    let Expr::Member {
        object, property, ..
    } = callee
    else {
        return None;
    };
    let Expr::Ident(ns) = &**object else {
        return None;
    };
    let (PropertyKey::Ident(method) | PropertyKey::Str(method)) = property else {
        return None;
    };
    match (&*ns.name, &**method) {
        ("console", "log") => Some(NB_CONSOLE_LOG),
        ("Math", "max") => Some(NB_MATH_MAX),
        ("Math", "min") => Some(NB_MATH_MIN),
        ("Math", "abs") => Some(NB_MATH_ABS),
        _ => None,
    }
}

/// The static string key of a non-computed property key.
fn static_key(key: &PropertyKey) -> Result<String, CompileError> {
    match key {
        PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(String::from(&**s)),
        PropertyKey::Number(n) => Ok(alloc::format!("{n}")),
        _ => Err(CompileError::Unsupported("computed/private key")),
    }
}

/// A single-pass register-allocating compiler from the AST to [`Op`]s.
#[derive(Default)]
struct Compiler {
    ops: Vec<Op>,
    /// Lexical scopes mapping a name to the register holding its value.
    scopes: Vec<alloc::collections::BTreeMap<String, Reg>>,
    next_reg: Reg,
    /// Function name → table id, for resolving calls.
    fn_ids: alloc::collections::BTreeMap<String, u32>,
}

impl Compiler {
    /// Compiles one function: binds `params` to registers `0..n`, compiles
    /// `body`, and (for `is_main`) returns the last expression's value.
    fn compile_fn(
        fn_ids: &alloc::collections::BTreeMap<String, u32>,
        params: &[crate::ast::Param],
        body: &[Stmt],
        is_main: bool,
    ) -> Result<FnProto, CompileError> {
        let mut c = Compiler {
            fn_ids: fn_ids.clone(),
            ..Compiler::default()
        };
        c.scopes.push(alloc::collections::BTreeMap::new());
        for p in params {
            let BindingTarget::Ident(Ident { name, .. }) = &p.target else {
                return Err(CompileError::Unsupported("destructuring parameter"));
            };
            c.declare(name);
        }
        let mut last: Option<Reg> = None;
        for stmt in body {
            if let Some(r) = c.stmt(stmt)? {
                last = Some(r);
            }
        }
        if is_main {
            let src = match last {
                Some(r) => r,
                None => c.constant(NanBox::undefined())?,
            };
            c.ops.push(Op::Return { src });
        }
        Ok(FnProto {
            n_regs: c.next_reg as usize,
            n_params: params.len(),
            ops: c.ops,
        })
    }
}

impl Compiler {
    fn alloc(&mut self) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn declare(&mut self, name: &str) -> Reg {
        let r = self.alloc();
        self.scopes
            .last_mut()
            .expect("a scope")
            .insert(String::from(name), r);
        r
    }

    fn lookup(&self, name: &str) -> Option<Reg> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// Compiles a statement; returns the register of its value if it is an
    /// expression statement (for the program's completion value).
    fn stmt(&mut self, stmt: &Stmt) -> Result<Option<Reg>, CompileError> {
        match stmt {
            Stmt::Empty { .. } => Ok(None),
            // Function declarations are hoisted into the table; nothing to emit.
            Stmt::Function(_) => Ok(None),
            Stmt::Return { argument, .. } => {
                let src = match argument {
                    Some(e) => self.expr(e)?,
                    None => self.constant(NanBox::undefined())?,
                };
                self.ops.push(Op::Return { src });
                Ok(None)
            }
            Stmt::Throw { argument, .. } => {
                let src = self.expr(argument)?;
                self.ops.push(Op::Throw { src });
                Ok(None)
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                if finalizer.is_some() {
                    return Err(CompileError::Unsupported("finally"));
                }
                let Some(catch) = handler else {
                    return Err(CompileError::Unsupported("try without catch"));
                };
                // The register the thrown value lands in (and the catch binding,
                // if any, names it).
                let catch_reg = self.alloc();
                let push = self.ops.len();
                self.ops.push(Op::PushHandler {
                    target: 0,
                    reg: catch_reg,
                });
                // try body
                self.scopes.push(alloc::collections::BTreeMap::new());
                for s in block {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                self.ops.push(Op::PopHandler);
                let jend = self.emit_jump();
                // catch entry
                self.patch(push);
                self.scopes.push(alloc::collections::BTreeMap::new());
                if let Some(BindingTarget::Ident(Ident { name, .. })) = &catch.param {
                    self.scopes
                        .last_mut()
                        .expect("a scope")
                        .insert(String::from(&**name), catch_reg);
                }
                for s in &catch.body {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                self.patch(jend);
                Ok(None)
            }
            Stmt::Expr { expression, .. } => Ok(Some(self.expr(expression)?)),
            Stmt::Var(decl) => {
                for d in &decl.declarations {
                    let BindingTarget::Ident(Ident { name, .. }) = &d.target else {
                        return Err(CompileError::Unsupported("destructuring binding"));
                    };
                    let value = match &d.init {
                        Some(e) => self.expr(e)?,
                        None => {
                            let r = self.alloc();
                            self.ops.push(Op::LoadConst {
                                dst: r,
                                value: NanBox::undefined(),
                            });
                            r
                        }
                    };
                    let slot = self.declare(name);
                    self.ops.push(Op::Move {
                        dst: slot,
                        src: value,
                    });
                }
                Ok(None)
            }
            Stmt::Block { body, .. } => {
                self.scopes.push(alloc::collections::BTreeMap::new());
                for s in body {
                    self.stmt(s)?;
                }
                self.scopes.pop();
                Ok(None)
            }
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                let cond = self.expr(test)?;
                let jf = self.emit_jump_if_false(cond);
                self.stmt(consequent)?;
                if let Some(alt) = alternate {
                    let jend = self.emit_jump();
                    self.patch(jf);
                    self.stmt(alt)?;
                    self.patch(jend);
                } else {
                    self.patch(jf);
                }
                Ok(None)
            }
            Stmt::While { test, body, .. } => {
                let top = self.ops.len();
                let cond = self.expr(test)?;
                let jf = self.emit_jump_if_false(cond);
                self.stmt(body)?;
                self.ops.push(Op::Jump { target: top });
                self.patch(jf);
                Ok(None)
            }
            Stmt::DoWhile { body, test, .. } => {
                let top = self.ops.len();
                self.stmt(body)?;
                let cond = self.expr(test)?;
                // Loop back to the top while the condition holds: jump if the
                // *negated* condition is false (i.e. while the condition is true).
                let not = self.alloc();
                self.ops.push(Op::Not { dst: not, a: cond });
                let jf = self.emit_jump_if_false(not);
                self.patch_to(jf, top);
                Ok(None)
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => {
                self.scopes.push(alloc::collections::BTreeMap::new());
                match init {
                    Some(ForInit::Var(decl)) => {
                        self.stmt(&Stmt::Var(decl.clone()))?;
                    }
                    Some(ForInit::Expr(e)) => {
                        self.expr(e)?;
                    }
                    None => {}
                }
                let top = self.ops.len();
                let exit = match test {
                    Some(t) => {
                        let cond = self.expr(t)?;
                        Some(self.emit_jump_if_false(cond))
                    }
                    None => None,
                };
                self.stmt(body)?;
                if let Some(u) = update {
                    self.expr(u)?;
                }
                self.ops.push(Op::Jump { target: top });
                if let Some(jf) = exit {
                    self.patch(jf);
                }
                self.scopes.pop();
                Ok(None)
            }
            _ => Err(CompileError::Unsupported("statement")),
        }
    }

    fn expr(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        match expr {
            Expr::Number { value, .. } => self.constant(NanBox::number(*value)),
            Expr::Bool { value, .. } => self.constant(NanBox::boolean(*value)),
            Expr::Null(_) => self.constant(NanBox::null()),
            Expr::Str { value, .. } => {
                let r = self.alloc();
                self.ops.push(Op::NewString {
                    dst: r,
                    value: String::from(&**value),
                });
                Ok(r)
            }
            Expr::Ident(id) => self
                .lookup(&id.name)
                .ok_or_else(|| CompileError::Undefined(String::from(&*id.name))),
            Expr::Unary { op, argument, .. } => {
                let a = self.expr(argument)?;
                let dst = self.alloc();
                match op {
                    UnaryOp::Minus => self.ops.push(Op::Neg { dst, a }),
                    UnaryOp::Not => self.ops.push(Op::Not { dst, a }),
                    _ => return Err(CompileError::Unsupported("unary operator")),
                }
                Ok(dst)
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                let a = self.expr(left)?;
                let b = self.expr(right)?;
                self.emit_binop(*op, a, b)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                // Short-circuit: `dst = left`; conditionally overwrite with right.
                let l = self.expr(left)?;
                let dst = self.alloc();
                self.ops.push(Op::Move { dst, src: l });
                let guard = match op {
                    LogicalOp::And => dst,
                    LogicalOp::Or => {
                        let n = self.alloc();
                        self.ops.push(Op::Not { dst: n, a: dst });
                        n
                    }
                    LogicalOp::Nullish => return Err(CompileError::Unsupported("?? operator")),
                };
                let jf = self.emit_jump_if_false(guard);
                let r = self.expr(right)?;
                self.ops.push(Op::Move { dst, src: r });
                self.patch(jf);
                Ok(dst)
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                let cond = self.expr(test)?;
                let dst = self.alloc();
                let jf = self.emit_jump_if_false(cond);
                let c = self.expr(consequent)?;
                self.ops.push(Op::Move { dst, src: c });
                let jend = self.emit_jump();
                self.patch(jf);
                let a = self.expr(alternate)?;
                self.ops.push(Op::Move { dst, src: a });
                self.patch(jend);
                Ok(dst)
            }
            Expr::Array { elements, .. } => {
                let dst = self.alloc();
                self.ops.push(Op::NewArray {
                    dst,
                    len: elements.len(),
                });
                for (i, el) in elements.iter().enumerate() {
                    let ArrayElement::Item(e) = el else {
                        return Err(CompileError::Unsupported("array hole/spread"));
                    };
                    let v = self.expr(e)?;
                    let idx = self.constant(NanBox::number(i as f64))?;
                    self.ops.push(Op::SetElem {
                        arr: dst,
                        index: idx,
                        src: v,
                    });
                }
                Ok(dst)
            }
            Expr::Object { members, .. } => {
                let dst = self.alloc();
                self.ops.push(Op::NewObject { dst });
                for m in members {
                    let ObjectMember::Property { key, value, .. } = m else {
                        return Err(CompileError::Unsupported("object member"));
                    };
                    let key = static_key(key)?;
                    let v = self.expr(value)?;
                    self.ops.push(Op::SetProp {
                        obj: dst,
                        key,
                        src: v,
                    });
                }
                Ok(dst)
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.expr(object)?;
                self.member_read(obj, property)
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                // Evaluate the argument registers (no spreads).
                let mut args = Vec::with_capacity(arguments.len());
                for a in arguments {
                    let crate::ast::Argument::Item(e) = a else {
                        return Err(CompileError::Unsupported("spread argument"));
                    };
                    args.push(self.expr(e)?);
                }
                // A built-in namespace call (`console.log`, `Math.max`, …).
                if let Some(native) = native_call(callee) {
                    let dst = self.alloc();
                    self.ops.push(Op::CallNative { dst, native, args });
                    return Ok(dst);
                }
                // Otherwise a direct call to a hoisted function (static dispatch
                // + recursion).
                let Expr::Ident(id) = &**callee else {
                    return Err(CompileError::Unsupported("indirect call"));
                };
                let func = *self
                    .fn_ids
                    .get(&*id.name)
                    .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                let dst = self.alloc();
                self.ops.push(Op::Call { dst, func, args });
                Ok(dst)
            }
            Expr::Assign {
                op, target, value, ..
            } => {
                use crate::ast::AssignOp;
                let compound = !matches!(op, AssignOp::Assign);
                match &**target {
                    Expr::Ident(id) => {
                        let slot = self
                            .lookup(&id.name)
                            .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                        let v = self.expr(value)?;
                        let src = if compound {
                            self.emit_binop(Self::compound_binop(*op)?, slot, v)?
                        } else {
                            v
                        };
                        self.ops.push(Op::Move { dst: slot, src });
                        Ok(slot)
                    }
                    // `obj.k (op)= v` / `arr[i] (op)= v`.
                    Expr::Member {
                        object, property, ..
                    } => {
                        let obj = self.expr(object)?;
                        let v = self.expr(value)?;
                        let src = if compound {
                            let cur = self.member_read(obj, property)?;
                            self.emit_binop(Self::compound_binop(*op)?, cur, v)?
                        } else {
                            v
                        };
                        match property {
                            PropertyKey::Computed(e) => {
                                let index = self.expr(e)?;
                                self.ops.push(Op::SetElem {
                                    arr: obj,
                                    index,
                                    src,
                                });
                            }
                            _ => {
                                let key = static_key(property)?;
                                self.ops.push(Op::SetProp { obj, key, src });
                            }
                        }
                        Ok(src)
                    }
                    _ => Err(CompileError::Unsupported("assignment target")),
                }
            }
            // `x++` / `++x` / `x--` / `--x` on a local variable.
            Expr::Update {
                op,
                prefix,
                argument,
                ..
            } => {
                let Expr::Ident(id) = &**argument else {
                    return Err(CompileError::Unsupported("update target"));
                };
                let slot = self
                    .lookup(&id.name)
                    .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                let one = self.constant(NanBox::number(1.0))?;
                let bop = match op {
                    crate::ast::UpdateOp::Inc => BinaryOp::Add,
                    crate::ast::UpdateOp::Dec => BinaryOp::Sub,
                };
                // Keep the pre-update value for a postfix result.
                let old = self.alloc();
                self.ops.push(Op::Move {
                    dst: old,
                    src: slot,
                });
                let next = self.emit_binop(bop, slot, one)?;
                self.ops.push(Op::Move {
                    dst: slot,
                    src: next,
                });
                Ok(if *prefix { slot } else { old })
            }
            _ => Err(CompileError::Unsupported("expression")),
        }
    }

    fn constant(&mut self, value: NanBox) -> Result<Reg, CompileError> {
        let r = self.alloc();
        self.ops.push(Op::LoadConst { dst: r, value });
        Ok(r)
    }

    /// Emits the op(s) for `a <op> b` into a fresh register, returning it.
    fn emit_binop(&mut self, op: BinaryOp, a: Reg, b: Reg) -> Result<Reg, CompileError> {
        let dst = self.alloc();
        match op {
            BinaryOp::Add => self.ops.push(Op::AddValue { dst, a, b }),
            BinaryOp::Sub => self.ops.push(Op::Sub { dst, a, b }),
            BinaryOp::Mul => self.ops.push(Op::Mul { dst, a, b }),
            BinaryOp::Div => self.ops.push(Op::Div { dst, a, b }),
            BinaryOp::Lt => self.ops.push(Op::Lt { dst, a, b }),
            BinaryOp::Gt => self.ops.push(Op::Lt { dst, a: b, b: a }),
            BinaryOp::LtEq => {
                self.ops.push(Op::Lt { dst, a: b, b: a });
                self.ops.push(Op::Not { dst, a: dst });
            }
            BinaryOp::GtEq => {
                self.ops.push(Op::Lt { dst, a, b });
                self.ops.push(Op::Not { dst, a: dst });
            }
            BinaryOp::EqEqEq => self.ops.push(Op::StrictEq { dst, a, b }),
            BinaryOp::NotEqEq => {
                self.ops.push(Op::StrictEq { dst, a, b });
                self.ops.push(Op::Not { dst, a: dst });
            }
            _ => return Err(CompileError::Unsupported("binary operator")),
        }
        Ok(dst)
    }

    /// The arithmetic operator underlying a compound assignment (`+=` → `+`).
    fn compound_binop(op: crate::ast::AssignOp) -> Result<BinaryOp, CompileError> {
        use crate::ast::AssignOp;
        Ok(match op {
            AssignOp::AddAssign => BinaryOp::Add,
            AssignOp::SubAssign => BinaryOp::Sub,
            AssignOp::MulAssign => BinaryOp::Mul,
            AssignOp::DivAssign => BinaryOp::Div,
            _ => return Err(CompileError::Unsupported("compound assignment operator")),
        })
    }

    /// Compiles a member read `obj.key` / `obj[i]` (with `.length` mapped to the
    /// array-length op).
    fn member_read(&mut self, obj: Reg, property: &PropertyKey) -> Result<Reg, CompileError> {
        let dst = self.alloc();
        match property {
            PropertyKey::Computed(e) => {
                let index = self.expr(e)?;
                self.ops.push(Op::GetElem {
                    dst,
                    arr: obj,
                    index,
                });
            }
            _ => {
                let key = static_key(property)?;
                if key == "length" {
                    self.ops.push(Op::ArrayLen { dst, arr: obj });
                } else {
                    self.ops.push(Op::GetProp { dst, obj, key });
                }
            }
        }
        Ok(dst)
    }

    /// Emits a `JumpIfFalse` with a placeholder target; returns its index.
    fn emit_jump_if_false(&mut self, cond: Reg) -> usize {
        let i = self.ops.len();
        self.ops.push(Op::JumpIfFalse { cond, target: 0 });
        i
    }

    fn emit_jump(&mut self) -> usize {
        let i = self.ops.len();
        self.ops.push(Op::Jump { target: 0 });
        i
    }

    /// Backpatches the jump at `idx` to land at the current instruction.
    fn patch(&mut self, idx: usize) {
        let target = self.ops.len();
        self.patch_to(idx, target);
    }

    /// Backpatches the jump (or handler) at `idx` to land at `target`.
    fn patch_to(&mut self, idx: usize, target: usize) {
        match &mut self.ops[idx] {
            Op::JumpIfFalse { target: t, .. }
            | Op::Jump { target: t }
            | Op::PushHandler { target: t, .. } => *t = target,
            _ => unreachable!("patch a non-jump"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compiles `src` to bytecode and runs it over a fresh realm, returning the
    /// completion value as a display string.
    fn bc(src: &str) -> String {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut realm = Realm::new();
        let value = compile_and_run(&mut realm, &program).expect("compile+run");
        realm.to_display_string(value)
    }

    #[test]
    fn bytecode_arithmetic_and_precedence() {
        assert_eq!(bc("2 + 3 * 4"), "14");
        assert_eq!(bc("(2 + 3) * 4"), "20");
        assert_eq!(bc("10 / 4"), "2.5");
        assert_eq!(bc("-5 + 3"), "-2");
        assert_eq!(bc("'a' + 'b' + 'c'"), "abc");
        assert_eq!(bc("1 + 2; 3 + 4"), "7"); // completion is the last expression
    }

    #[test]
    fn bytecode_comparisons_and_logic() {
        assert_eq!(bc("3 < 5"), "true");
        assert_eq!(bc("5 <= 5"), "true");
        assert_eq!(bc("7 > 2"), "true");
        assert_eq!(bc("2 >= 9"), "false");
        assert_eq!(bc("1 === 1"), "true");
        assert_eq!(bc("1 !== 2"), "true");
        assert_eq!(bc("!false"), "true");
        assert_eq!(bc("true && 7"), "7");
        assert_eq!(bc("false || 'fallback'"), "fallback");
        assert_eq!(bc("0 && 9"), "0"); // short-circuit keeps the falsy left
        assert_eq!(bc("(3 > 2) ? 'yes' : 'no'"), "yes");
    }

    #[test]
    fn bytecode_variables_and_control_flow() {
        // let bindings, assignment, and a while loop, all compiled to bytecode.
        assert_eq!(
            bc("let sum = 0; let i = 1; while (i <= 5) { sum = sum + i; i = i + 1; } sum"),
            "15"
        );
        // if / else.
        assert_eq!(
            bc("let x = 7; let r = 0; if (x > 5) { r = 1; } else { r = 2; } r"),
            "1"
        );
        // Block scoping doesn't clobber the outer binding's register.
        assert_eq!(bc("let a = 1; { let a = 99; } a"), "1");
        // Fibonacci via a loop — exercises the whole pipeline.
        assert_eq!(
            bc(
                "let a = 0; let b = 1; let n = 10; while (n > 0) { let t = a + b; a = b; b = t; n = n - 1; } a"
            ),
            "55"
        );
    }

    /// Compiles + runs `src`, returning captured `console` output.
    fn bc_out(src: &str) -> String {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut realm = Realm::new();
        let (_, output) = compile_run_output(&mut realm, &program).expect("compile+run");
        output
    }

    #[test]
    fn bytecode_exceptions() {
        // throw caught by the local catch, binding the thrown value.
        assert_eq!(
            bc("let r = 'none'; try { throw 'boom'; } catch (e) { r = 'caught:' + e; } r"),
            "caught:boom"
        );
        // No throw: the catch body is skipped.
        assert_eq!(bc("let r = 0; try { r = 1; } catch (e) { r = 99; } r"), "1");
        // A throw inside a called function unwinds to the caller's catch.
        assert_eq!(
            bc(
                "function boom() { throw 'x'; } let r = 'ok'; try { boom(); r = 'no'; } catch (e) { r = 'got:' + e; } r"
            ),
            "got:x"
        );
        // catch without a binding.
        assert_eq!(
            bc("let r = 'a'; try { throw 1; } catch { r = 'b'; } r"),
            "b"
        );
        // Conditional throw in a loop; the loop continues after catching.
        assert_eq!(
            bc(
                "let s = 0; for (let i = 0; i < 5; i++) { try { if (i === 2) { throw 0; } s += i; } catch (e) { s += 100; } } s"
            ),
            "108"
        );
    }

    #[test]
    fn bytecode_console_and_math_natives() {
        assert_eq!(bc_out("console.log('hello')"), "hello\n");
        assert_eq!(bc_out("console.log(1 + 2, 'x')"), "3 x\n");
        // console.log inside a loop, driven entirely by bytecode.
        assert_eq!(
            bc_out("for (let i = 1; i <= 3; i++) { console.log(i * i); }"),
            "1\n4\n9\n"
        );
        // Math.* natives folded over the args.
        assert_eq!(bc("Math.max(3, 9, 4)"), "9");
        assert_eq!(bc("Math.min(3, -2, 8)"), "-2");
        assert_eq!(bc("Math.abs(-7)"), "7");
        // A function that logs, called from bytecode.
        assert_eq!(
            bc_out("function greet(n) { console.log('hi ' + n); } greet('ada'); greet('bob');"),
            "hi ada\nhi bob\n"
        );
    }

    #[test]
    fn bytecode_matches_tree_walker() {
        // Cross-engine parity: the bytecode VM and the tree-walker agree on the
        // captured output for the same program (the migration's correctness bar).
        let programs = [
            "let s = 0; for (let i = 1; i <= 10; i++) { s += i; } console.log(s);",
            "function fib(n) { if (n < 2) { return n; } return fib(n-1) + fib(n-2); } console.log(fib(15));",
            "let a = [5, 3, 8]; let m = a[0]; for (let i = 1; i < a.length; i++) { if (a[i] > m) { m = a[i]; } } console.log(m);",
        ];
        for src in programs {
            let program = crate::parser::Parser::parse_program(src).expect("parse");
            let mut realm = Realm::new();
            let (_, vm_out) = compile_run_output(&mut realm, &program).expect("bytecode");
            let (tw_out, _) = crate::nbexec::eval_source(src).expect("tree-walker");
            assert_eq!(vm_out, tw_out, "engines disagree on: {src}");
        }
    }

    #[test]
    fn bytecode_compound_update_and_do_while() {
        // Compound assignment on a local.
        assert_eq!(bc("let x = 10; x += 5; x"), "15");
        assert_eq!(bc("let x = 10; x -= 3; x *= 2; x"), "14");
        // Compound assignment on a member.
        assert_eq!(bc("let o = { n: 1 }; o.n += 9; o.n"), "10");
        assert_eq!(bc("let a = [1, 2, 3]; a[1] *= 10; a[1]"), "20");
        // Update operators (prefix / postfix).
        assert_eq!(bc("let i = 5; i++; i"), "6");
        assert_eq!(bc("let i = 5; let a = i++; a + ',' + i"), "5,6");
        assert_eq!(bc("let i = 5; let a = ++i; a + ',' + i"), "6,6");
        assert_eq!(bc("let i = 5; --i; i"), "4");
        // A for loop using `++` in its update — a common shape, in bytecode.
        assert_eq!(
            bc("let s = 0; for (let i = 0; i < 5; i++) { s += i; } s"),
            "10"
        );
        // do/while runs the body at least once.
        assert_eq!(
            bc("let n = 0; let s = 0; do { s += n; n++; } while (n < 4); s"),
            "6"
        );
        assert_eq!(bc("let r = 0; do { r++; } while (false); r"), "1");
    }

    #[test]
    fn bytecode_functions_and_recursion() {
        // A simple call with arguments.
        assert_eq!(bc("function add(a, b) { return a + b; } add(3, 4)"), "7");
        // Recursion: factorial.
        assert_eq!(
            bc("function fact(n) { if (n <= 1) { return 1; } return n * fact(n - 1); } fact(6)"),
            "720"
        );
        // Mutual / forward reference (isEven defined before isOdd is used).
        assert_eq!(
            bc(
                "function fib(n) { if (n < 2) { return n; } return fib(n - 1) + fib(n - 2); } fib(12)"
            ),
            "144"
        );
        // A function operating on an array argument.
        assert_eq!(
            bc(
                "function sum(a) { let s = 0; for (let i = 0; i < a.length; i = i + 1) { s = s + a[i]; } return s; } sum([5, 10, 15])"
            ),
            "30"
        );
        // Local variables don't leak between activations.
        assert_eq!(
            bc("function f(x) { let y = x * 2; return y; } f(3) + f(10)"),
            "26"
        );
    }

    #[test]
    fn bytecode_arrays_objects_and_for() {
        // Array literal, element read, and `.length`.
        assert_eq!(bc("let a = [10, 20, 30]; a[1]"), "20");
        assert_eq!(bc("[1, 2, 3, 4].length"), "4");
        // Element assignment.
        assert_eq!(bc("let a = [0, 0, 0]; a[2] = 7; a[2]"), "7");
        // Object literal + property read/write.
        assert_eq!(bc("let o = { x: 1, y: 2 }; o.x + o.y"), "3");
        assert_eq!(bc("let o = {}; o.k = 42; o.k"), "42");
        // A C-style for loop summing an array, compiled to bytecode.
        assert_eq!(
            bc(
                "let a = [3, 1, 4, 1, 5]; let s = 0; for (let i = 0; i < a.length; i = i + 1) { s = s + a[i]; } s"
            ),
            "14"
        );
        // Nested data + computed access.
        assert_eq!(
            bc("let grid = [[1, 2], [3, 4]]; grid[1][0] + grid[0][1]"),
            "5"
        );
    }

    #[test]
    fn arithmetic() {
        // (2 + 3) * 4 = 20
        let prog = [
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(2.0),
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(3.0),
            },
            Op::Add { dst: 0, a: 0, b: 1 },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(4.0),
            },
            Op::Mul { dst: 0, a: 0, b: 1 },
            Op::Return { src: 0 },
        ];
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 2).unwrap();
        assert_eq!(result.as_number(), Some(20.0));
    }

    #[test]
    fn counting_loop_sums_one_to_ten() {
        // r0 = sum, r1 = i, r2 = limit(11), r3 = step(1), r4 = cond
        // while (i < 11) { sum += i; i += 1; }  → 55
        let prog = [
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(0.0),
            }, // 0: sum = 0
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(1.0),
            }, // 1: i = 1
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(11.0),
            }, // 2: limit
            Op::LoadConst {
                dst: 3,
                value: NanBox::number(1.0),
            }, // 3: step
            // loop head (pc 4):
            Op::Lt { dst: 4, a: 1, b: 2 },          // 4: cond = i < 11
            Op::JumpIfFalse { cond: 4, target: 8 }, // 5: exit if !cond
            Op::Add { dst: 0, a: 0, b: 1 },         // 6: sum += i
            Op::Add { dst: 1, a: 1, b: 3 },         // 7: i += 1
            // (fallthrough would be 8; we need to loop back to 4)
            Op::Jump { target: 4 }, // 8 -> but we want exit at 8...
            Op::Return { src: 0 },  // 9
        ];
        // Fix the jump targets: exit should land on the Return.
        let prog = {
            let mut p = prog.to_vec();
            p[5] = Op::JumpIfFalse { cond: 4, target: 9 }; // exit → Return
            p[8] = Op::Jump { target: 4 }; // loop back to head
            p
        };
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 5).unwrap();
        assert_eq!(result.as_number(), Some(55.0));
    }

    #[test]
    fn object_property_round_trip() {
        // o = {}; o.x = 7; o.y = 8; return o.x + o.y  → 15
        let prog = [
            Op::NewObject { dst: 0 },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(7.0),
            },
            Op::SetProp {
                obj: 0,
                key: String::from("x"),
                src: 1,
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(8.0),
            },
            Op::SetProp {
                obj: 0,
                key: String::from("y"),
                src: 1,
            },
            Op::GetProp {
                dst: 2,
                obj: 0,
                key: String::from("x"),
            },
            Op::GetProp {
                dst: 3,
                obj: 0,
                key: String::from("y"),
            },
            Op::Add { dst: 2, a: 2, b: 3 },
            Op::Return { src: 2 },
        ];
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 4).unwrap();
        assert_eq!(result.as_number(), Some(15.0));
        // The object really lives in the realm's heap.
        assert_eq!(realm.object_count(), 1);
    }

    #[test]
    fn builds_and_compares_strings() {
        // greeting = "Hello, " + "world"; return (greeting === "Hello, world")
        let prog = [
            Op::NewString {
                dst: 0,
                value: String::from("Hello, "),
            },
            Op::NewString {
                dst: 1,
                value: String::from("world"),
            },
            Op::AddValue { dst: 0, a: 0, b: 1 }, // r0 = "Hello, world"
            Op::NewString {
                dst: 2,
                value: String::from("Hello, world"),
            },
            Op::StrictEq { dst: 0, a: 0, b: 2 }, // string === by value
            Op::Return { src: 0 },
        ];
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 3).unwrap();
        assert_eq!(result.as_boolean(), Some(true));
    }

    #[test]
    fn string_concat_loop() {
        // s = ""; i = 0; while (i < 5) { s = s + "x"; i = i + 1; } return s  → "xxxxx"
        let prog = vec![
            Op::NewString {
                dst: 0,
                value: String::new(),
            }, // 0: s = ""
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(0.0),
            }, // 1: i = 0
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(5.0),
            }, // 2: limit
            Op::LoadConst {
                dst: 3,
                value: NanBox::number(1.0),
            }, // 3: step
            Op::NewString {
                dst: 4,
                value: String::from("x"),
            }, // 4: "x"
            Op::Lt { dst: 5, a: 1, b: 2 }, // 5: cond = i < 5
            Op::JumpIfFalse {
                cond: 5,
                target: 10,
            }, // 6: exit
            Op::AddValue { dst: 0, a: 0, b: 4 }, // 7: s = s + "x"
            Op::Add { dst: 1, a: 1, b: 3 }, // 8: i += 1
            Op::Jump { target: 5 },        // 9: loop
            Op::Return { src: 0 },         // 10
        ];
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 6).unwrap();
        assert_eq!(realm.to_display_string(result), "xxxxx");
    }

    #[test]
    fn sums_an_array_in_a_loop() {
        // a = [10, 20, 30]; sum = 0; for (i=0; i<a.length; i++) sum += a[i]; → 60
        // regs: 0=a 1=sum 2=i 3=len 4=step 5=cond 6=elem
        let prog = vec![
            Op::NewArray { dst: 0, len: 3 },
            Op::LoadConst {
                dst: 6,
                value: NanBox::number(10.0),
            },
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(0.0),
            },
            Op::SetElem {
                arr: 0,
                index: 2,
                src: 6,
            }, // a[0] = 10
            Op::LoadConst {
                dst: 6,
                value: NanBox::number(20.0),
            },
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(1.0),
            },
            Op::SetElem {
                arr: 0,
                index: 2,
                src: 6,
            }, // a[1] = 20
            Op::LoadConst {
                dst: 6,
                value: NanBox::number(30.0),
            },
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(2.0),
            },
            Op::SetElem {
                arr: 0,
                index: 2,
                src: 6,
            }, // a[2] = 30
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(0.0),
            }, // sum = 0
            Op::LoadConst {
                dst: 2,
                value: NanBox::number(0.0),
            }, // i = 0
            Op::ArrayLen { dst: 3, arr: 0 }, // len = 3
            Op::LoadConst {
                dst: 4,
                value: NanBox::number(1.0),
            }, // step
            // loop head @ 14:
            Op::Lt { dst: 5, a: 2, b: 3 }, // 14: cond = i < len
            Op::JumpIfFalse {
                cond: 5,
                target: 19,
            }, // 15: exit
            Op::GetElem {
                dst: 6,
                arr: 0,
                index: 2,
            }, // 16: elem = a[i]
            Op::Add { dst: 1, a: 1, b: 6 }, // 17: sum += elem
            Op::Add { dst: 2, a: 2, b: 4 }, // 18: i += 1 ... then loop
            Op::Return { src: 1 },         // 19
        ];
        // patch the loop-back: after 18 we must jump to 14, and exit at 19.
        let prog = {
            let mut p = prog;
            // Insert a Jump back to head before the Return by rewriting index 18's
            // successor: we append a Jump and move Return.
            p.insert(19, Op::Jump { target: 14 });
            // Now Return is at 20; the exit target (was 19) still points at the Jump,
            // so fix JumpIfFalse to land on the Return at 20.
            p[15] = Op::JumpIfFalse {
                cond: 5,
                target: 20,
            };
            p
        };
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 7).unwrap();
        assert_eq!(result.as_number(), Some(60.0));
    }

    #[test]
    fn absent_property_reads_undefined() {
        let prog = [
            Op::NewObject { dst: 0 },
            Op::GetProp {
                dst: 1,
                obj: 0,
                key: String::from("missing"),
            },
            Op::Return { src: 1 },
        ];
        let mut realm = Realm::new();
        let result = run(&mut realm, &prog, 2).unwrap();
        assert!(result.is_undefined());
    }

    #[test]
    fn type_error_on_non_number_arithmetic() {
        let prog = [
            Op::LoadConst {
                dst: 0,
                value: NanBox::undefined(),
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(1.0),
            },
            Op::Add { dst: 0, a: 0, b: 1 },
            Op::Return { src: 0 },
        ];
        let mut realm = Realm::new();
        assert_eq!(run(&mut realm, &prog, 2), Err(VmError::NotANumber));
    }
}
