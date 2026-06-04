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
//! that lowers a broad JavaScript subset (arithmetic, control flow,
//! arrays/objects, `for`/`do-while`/`for-of`/`switch` with `break`/`continue`,
//! compound assignment and `++`/`--`, functions with recursion via a
//! per-activation register window, first-class function values and **closures
//! with mutable capture** — free variables become shared heap *cells* —
//! `try`/`catch`/`finally`/`throw` exceptions that unwind across calls, and
//! native `console.log`/`Math.*`/`String`/`Number` calls) onto these ops, with
//! no tree-walking — and it agrees with the tree-walker on output (a
//! cross-engine parity test). The point is that every value flowing through it
//! is a single 64-bit word and every object is a GC-managed heap node, exactly
//! as the production VM will work.
//!
//! Pure, safe `alloc`-only Rust.

use crate::heap::Handle;
use crate::nanbox::NanBox;
use crate::realm::Realm;
use alloc::boxed::Box;
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
    /// `dst = a % b` (numeric remainder).
    Mod { dst: Reg, a: Reg, b: Reg },
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
    /// `dst = a first-class function value` wrapping function-table index `func`.
    LoadFunc { dst: Reg, func: u32 },
    /// `dst = a closure` over `func` capturing the cells in `captures` — a heap
    /// array `[func_id, cell0, cell1, …]`. Cells are shared by handle, so a
    /// mutation through a captured variable is visible to every closure sharing
    /// it.
    MakeClosure {
        dst: Reg,
        func: u32,
        captures: Vec<Reg>,
    },
    /// `dst = callee(args…)` — an indirect call through a function value held in
    /// the `callee` register.
    CallValue {
        dst: Reg,
        callee: Reg,
        args: Vec<Reg>,
    },
    /// `dst = recv[key](args…)` — a method call: reads the closure at
    /// `recv[key]` and invokes it with `this` bound to `recv`.
    CallMethod {
        dst: Reg,
        recv: Reg,
        key: String,
        args: Vec<Reg>,
    },
    /// Runs constructor function `ctor` with `this = recv` and `args` (the
    /// return value is discarded; `new` yields the instance).
    CallCtor {
        ctor: u32,
        recv: Reg,
        args: Vec<Reg>,
    },
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
const NB_MATH_FLOOR: u16 = 4;
const NB_MATH_CEIL: u16 = 5;
const NB_MATH_ROUND: u16 = 6;
const NB_MATH_SQRT: u16 = 7;
const NB_MATH_POW: u16 = 8;
const NB_STRING: u16 = 9;
const NB_NUMBER: u16 = 10;

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
    /// Captured cells, bound to registers `n_params..n_params + n_captures` on
    /// entry (a closure passes its cells here).
    pub n_captures: usize,
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
    call_with(ctx, funcs, id, args, &[], NanBox::undefined())
}

/// Calls function `id` with `args` (registers `0..n_params`), `captures`
/// (registers `n_params..n_params + n_captures`), and `this_val` (the register
/// right after).
fn call_with(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
    captures: &[NanBox],
    this_val: NanBox,
) -> Result<NanBox, VmError> {
    let proto = &funcs[id];
    let mut regs: Vec<NanBox> = vec![NanBox::undefined(); proto.n_regs];
    for (i, a) in args.iter().enumerate().take(proto.n_params) {
        regs[i] = *a;
    }
    for (i, c) in captures.iter().enumerate().take(proto.n_captures) {
        regs[proto.n_params + i] = *c;
    }
    // The `this` slot sits right after the captures.
    if let Some(slot) = regs.get_mut(proto.n_params + proto.n_captures) {
        *slot = this_val;
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
            Op::Mod { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? % num(regs[*b as usize])?);
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
            Op::LoadFunc { dst, func } => {
                // A function value is a one-element heap array holding the
                // function-table index (as a number).
                let handle = ctx
                    .realm
                    .new_array(alloc::vec![NanBox::number(*func as f64)]);
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::MakeClosure {
                dst,
                func,
                captures,
            } => {
                // `[func_id, cell0, cell1, …]`.
                let mut elems = alloc::vec![NanBox::number(*func as f64)];
                elems.extend(captures.iter().map(|r| regs[*r as usize]));
                let handle = ctx.realm.new_array(elems);
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::CallValue { dst, callee, args } => {
                let handle = object_handle(regs[*callee as usize])?;
                let id = ctx
                    .realm
                    .get_element(handle, 0)
                    .as_number()
                    .ok_or(VmError::NotAnObject)? as usize;
                // Captured cells live in array slots `1..`.
                let n_caps = ctx
                    .realm
                    .array_length(handle)
                    .unwrap_or(1)
                    .saturating_sub(1);
                let caps: Vec<NanBox> = (0..n_caps)
                    .map(|i| ctx.realm.get_element(handle, i + 1))
                    .collect();
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                match call_with(ctx, funcs, id, &argv, &caps, NanBox::undefined()) {
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
            Op::CallMethod {
                dst,
                recv,
                key,
                args,
            } => {
                let recv_val = regs[*recv as usize];
                let robj = object_handle(recv_val)?;
                let f = ctx
                    .realm
                    .get_property(robj, key)
                    .unwrap_or(NanBox::undefined());
                let fh = object_handle(f)?;
                let id = ctx
                    .realm
                    .get_element(fh, 0)
                    .as_number()
                    .ok_or(VmError::NotAnObject)? as usize;
                let n_caps = ctx.realm.array_length(fh).unwrap_or(1).saturating_sub(1);
                let caps: Vec<NanBox> = (0..n_caps)
                    .map(|i| ctx.realm.get_element(fh, i + 1))
                    .collect();
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                // Bind `this` to the receiver.
                match call_with(ctx, funcs, id, &argv, &caps, recv_val) {
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
            Op::CallCtor { ctor, recv, args } => {
                let recv_val = regs[*recv as usize];
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                match call_with(ctx, funcs, *ctor as usize, &argv, &[], recv_val) {
                    Ok(_) => {}
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
        #[cfg(feature = "std")]
        NB_MATH_FLOOR | NB_MATH_CEIL | NB_MATH_ROUND | NB_MATH_SQRT | NB_MATH_POW => {
            let a = args.first().and_then(|v| v.as_number()).unwrap_or(f64::NAN);
            let val = match native {
                NB_MATH_FLOOR => a.floor(),
                NB_MATH_CEIL => a.ceil(),
                NB_MATH_ROUND => a.round(),
                NB_MATH_SQRT => a.sqrt(),
                _ => a.powf(args.get(1).and_then(|v| v.as_number()).unwrap_or(f64::NAN)),
            };
            NanBox::number(val)
        }
        #[cfg(not(feature = "std"))]
        NB_MATH_FLOOR | NB_MATH_CEIL | NB_MATH_ROUND | NB_MATH_SQRT | NB_MATH_POW => {
            NanBox::number(f64::NAN)
        }
        NB_STRING => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            NanBox::handle(ctx.realm.new_string(&s).to_raw())
        }
        NB_NUMBER => NanBox::number(
            ctx.realm
                .to_number(args.first().copied().unwrap_or(NanBox::undefined())),
        ),
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

/// Runs `source` on the **bytecode VM**, falling back to the tree-walker
/// ([`crate::nbexec::eval_source`]) for any construct the bytecode compiler does
/// not yet handle — the production execution model (a fast bytecode path with a
/// complete-semantics safety net). Returns the captured `console` output and the
/// completion value (as a display string).
///
/// # Errors
/// Returns a parse or execution error message.
#[cfg(feature = "std")]
pub fn execute(source: &str) -> Result<(String, String), String> {
    let program =
        crate::parser::Parser::parse_program(source).map_err(|e| alloc::format!("{e}"))?;
    // Compile to bytecode; an unsupported construct routes the whole program to
    // the tree-walker (compilation happens before execution, so no output has
    // been produced yet — the fallback is clean).
    let Ok(protos) = compile_program(&program) else {
        return crate::nbexec::eval_source(source);
    };
    let mut realm = Realm::new();
    match run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, output)) => Ok((output, realm.to_display_string(value))),
        // A runtime fault on the bytecode path (an unsupported coercion, etc.):
        // re-run on the reference tree-walker.
        Err(_) => crate::nbexec::eval_source(source),
    }
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
    let fn_ids = alloc::rc::Rc::new(fn_ids);
    // Scan top-level classes, reserving constructor/method ids after the
    // functions. Each `(id, params, body)` is compiled below.
    let mut next_id = (decls.len() + 1) as u32;
    let mut class_map = alloc::collections::BTreeMap::new();
    let mut class_jobs: Vec<ClassJob> = Vec::new();
    for s in &program.body {
        if let Stmt::Class(class) = s {
            let Some(id) = &class.id else { continue };
            let info = scan_class(class, &mut next_id, &mut class_jobs)?;
            class_map.insert(String::from(&*id.name), info);
        }
    }
    let classes = alloc::rc::Rc::new(class_map);

    let protos = alloc::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
    let placeholder = || FnProto {
        ops: Vec::new(),
        n_regs: 0,
        n_params: 0,
        n_captures: 0,
    };
    // Reserve slots: main (0), top-level functions (1..=N), then class members
    // (N+1..next_id). Nested function expressions append beyond `next_id`.
    protos
        .borrow_mut()
        .extend((0..next_id).map(|_| placeholder()));
    // Compile main (id 0), each top-level function, then each class member.
    let main = Compiler::compile_fn(&fn_ids, &classes, &protos, &[], &[], &program.body, true)?;
    protos.borrow_mut()[0] = main;
    for (i, f) in decls.iter().enumerate() {
        let proto =
            Compiler::compile_fn(&fn_ids, &classes, &protos, &f.params, &[], &f.body, false)?;
        protos.borrow_mut()[i + 1] = proto;
    }
    for job in &class_jobs {
        // A subclass constructor resolves `super(...)` to the nearest ancestor
        // constructor.
        let super_ctor = job
            .super_of
            .as_deref()
            .and_then(|name| nearest_ctor(name, &classes));
        let proto = Compiler::compile_fn_inner(
            &fn_ids,
            &classes,
            &protos,
            job.params,
            &[],
            job.body,
            false,
            super_ctor,
        )?;
        protos.borrow_mut()[job.id as usize] = proto;
    }
    Ok(alloc::rc::Rc::try_unwrap(protos)
        .expect("unique proto table")
        .into_inner())
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
        ("Math", "floor") => Some(NB_MATH_FLOOR),
        ("Math", "ceil") => Some(NB_MATH_CEIL),
        ("Math", "round") => Some(NB_MATH_ROUND),
        ("Math", "sqrt") => Some(NB_MATH_SQRT),
        ("Math", "pow") => Some(NB_MATH_POW),
        _ => None,
    }
}

/// Maps a global function call (`String(x)`, `Number(x)`) to its native id.
fn native_global(callee: &Expr) -> Option<u16> {
    let Expr::Ident(id) = callee else {
        return None;
    };
    match &*id.name {
        "String" => Some(NB_STRING),
        "Number" => Some(NB_NUMBER),
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

use alloc::collections::BTreeSet;

/// The free variables of a function: names it references that are bound neither
/// by its parameters nor its local declarations (so they come from an enclosing
/// scope — i.e. its captures).
fn free_of_function(params: &[crate::ast::Param], body: &[Stmt]) -> BTreeSet<String> {
    let bound = bound_names(params, body);
    let mut direct = BTreeSet::new();
    let mut nested = BTreeSet::new();
    for s in body {
        refs_stmt(s, &mut direct, &mut nested);
    }
    direct
        .into_iter()
        .chain(nested)
        .filter(|n| !bound.contains(n))
        .collect()
}

/// The names a function declares (parameters + local declarations), *not*
/// descending into nested functions.
fn bound_names(params: &[crate::ast::Param], body: &[Stmt]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for p in params {
        if let BindingTarget::Ident(Ident { name, .. }) = &p.target {
            out.insert(String::from(&**name));
        }
    }
    for s in body {
        declared_in_stmt(s, &mut out);
    }
    out
}

/// This function's own bound names that are captured by some nested function
/// (and so must be cells).
fn captured_names(params: &[crate::ast::Param], body: &[Stmt]) -> BTreeSet<String> {
    let bound = bound_names(params, body);
    let mut direct = BTreeSet::new();
    let mut nested = BTreeSet::new();
    for s in body {
        refs_stmt(s, &mut direct, &mut nested);
    }
    bound.intersection(&nested).cloned().collect()
}

/// Collects the names declared by `s` (let/const/var/function/catch/for-head),
/// not descending into nested functions or expressions.
fn declared_in_stmt(s: &Stmt, out: &mut BTreeSet<String>) {
    let decl_target = |t: &BindingTarget, out: &mut BTreeSet<String>| {
        if let BindingTarget::Ident(Ident { name, .. }) = t {
            out.insert(String::from(&**name));
        }
    };
    match s {
        Stmt::Var(d) => {
            for dr in &d.declarations {
                decl_target(&dr.target, out);
            }
        }
        Stmt::Function(f) => {
            if let Some(id) = &f.id {
                out.insert(String::from(&*id.name));
            }
        }
        Stmt::Block { body, .. } => {
            for s in body {
                declared_in_stmt(s, out);
            }
        }
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            declared_in_stmt(consequent, out);
            if let Some(a) = alternate {
                declared_in_stmt(a, out);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => declared_in_stmt(body, out),
        Stmt::For { init, body, .. } => {
            if let Some(ForInit::Var(d)) = init {
                for dr in &d.declarations {
                    decl_target(&dr.target, out);
                }
            }
            declared_in_stmt(body, out);
        }
        Stmt::ForOf { left, body, .. } | Stmt::ForIn { left, body, .. } => {
            if let crate::ast::ForLeft::Decl { target, .. } = left {
                decl_target(target, out);
            }
            declared_in_stmt(body, out);
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            for s in block {
                declared_in_stmt(s, out);
            }
            if let Some(h) = handler {
                if let Some(p) = &h.param {
                    decl_target(p, out);
                }
                for s in &h.body {
                    declared_in_stmt(s, out);
                }
            }
            if let Some(f) = finalizer {
                for s in f {
                    declared_in_stmt(s, out);
                }
            }
        }
        Stmt::Switch { cases, .. } => {
            for c in cases {
                for s in &c.body {
                    declared_in_stmt(s, out);
                }
            }
        }
        _ => {}
    }
}

/// Walks `s` collecting direct identifier references (`direct`) and the free
/// variables of any nested function expression (`nested`).
fn refs_stmt(s: &Stmt, direct: &mut BTreeSet<String>, nested: &mut BTreeSet<String>) {
    match s {
        Stmt::Expr { expression, .. } => refs_expr(expression, direct, nested),
        Stmt::Var(d) => {
            for dr in &d.declarations {
                if let Some(e) = &dr.init {
                    refs_expr(e, direct, nested);
                }
            }
        }
        Stmt::Return {
            argument: Some(e), ..
        }
        | Stmt::Throw { argument: e, .. } => refs_expr(e, direct, nested),
        Stmt::Block { body, .. } => body.iter().for_each(|s| refs_stmt(s, direct, nested)),
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => {
            refs_expr(test, direct, nested);
            refs_stmt(consequent, direct, nested);
            if let Some(a) = alternate {
                refs_stmt(a, direct, nested);
            }
        }
        Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
            refs_expr(test, direct, nested);
            refs_stmt(body, direct, nested);
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            match init {
                Some(ForInit::Var(d)) => {
                    for dr in &d.declarations {
                        if let Some(e) = &dr.init {
                            refs_expr(e, direct, nested);
                        }
                    }
                }
                Some(ForInit::Expr(e)) => refs_expr(e, direct, nested),
                None => {}
            }
            if let Some(t) = test {
                refs_expr(t, direct, nested);
            }
            if let Some(u) = update {
                refs_expr(u, direct, nested);
            }
            refs_stmt(body, direct, nested);
        }
        Stmt::ForOf {
            left, right, body, ..
        }
        | Stmt::ForIn {
            left, right, body, ..
        } => {
            if let crate::ast::ForLeft::Target(e) = left {
                refs_expr(e, direct, nested);
            }
            refs_expr(right, direct, nested);
            refs_stmt(body, direct, nested);
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            block.iter().for_each(|s| refs_stmt(s, direct, nested));
            if let Some(h) = handler {
                h.body.iter().for_each(|s| refs_stmt(s, direct, nested));
            }
            if let Some(f) = finalizer {
                f.iter().for_each(|s| refs_stmt(s, direct, nested));
            }
        }
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            refs_expr(discriminant, direct, nested);
            for c in cases {
                if let Some(t) = &c.test {
                    refs_expr(t, direct, nested);
                }
                c.body.iter().for_each(|s| refs_stmt(s, direct, nested));
            }
        }
        _ => {}
    }
}

/// Walks `e` collecting direct identifier references; for a nested function
/// expression, collects *its* free variables into `nested` (without descending
/// for direct refs).
fn refs_expr(e: &Expr, direct: &mut BTreeSet<String>, nested: &mut BTreeSet<String>) {
    match e {
        Expr::Ident(id) => {
            direct.insert(String::from(&*id.name));
        }
        Expr::Function(f) => nested.extend(free_of_function(&f.params, &f.body)),
        Expr::Arrow(a) => {
            let body: Vec<Stmt> = match &a.body {
                crate::ast::ArrowBody::Block(b) => b.clone(),
                crate::ast::ArrowBody::Expr(e) => alloc::vec![Stmt::Return {
                    argument: Some(Box::new((**e).clone())),
                    span: crate::common::Span::point(0),
                }],
            };
            nested.extend(free_of_function(&a.params, &body));
        }
        Expr::Unary { argument, .. } | Expr::Update { argument, .. } => {
            refs_expr(argument, direct, nested);
        }
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            refs_expr(left, direct, nested);
            refs_expr(right, direct, nested);
        }
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            refs_expr(test, direct, nested);
            refs_expr(consequent, direct, nested);
            refs_expr(alternate, direct, nested);
        }
        Expr::Assign { target, value, .. } => {
            refs_expr(target, direct, nested);
            refs_expr(value, direct, nested);
        }
        Expr::Member {
            object, property, ..
        } => {
            refs_expr(object, direct, nested);
            if let PropertyKey::Computed(e) = property {
                refs_expr(e, direct, nested);
            }
        }
        Expr::Call {
            callee, arguments, ..
        } => {
            refs_expr(callee, direct, nested);
            for a in arguments {
                if let crate::ast::Argument::Item(e) = a {
                    refs_expr(e, direct, nested);
                }
            }
        }
        Expr::Array { elements, .. } => {
            for el in elements {
                if let ArrayElement::Item(e) = el {
                    refs_expr(e, direct, nested);
                }
            }
        }
        Expr::Object { members, .. } => {
            for m in members {
                if let ObjectMember::Property { key, value, .. } = m {
                    if let PropertyKey::Computed(e) = key {
                        refs_expr(e, direct, nested);
                    }
                    refs_expr(value, direct, nested);
                }
            }
        }
        _ => {}
    }
}

/// Scans a *plain* class (no `extends`, fields, statics, or accessors — those
/// fall back to the tree-walker), reserving function ids for its constructor and
/// methods and queueing them for compilation.
fn scan_class<'a>(
    class: &'a crate::ast::Class,
    next_id: &mut u32,
    jobs: &mut Vec<ClassJob<'a>>,
) -> Result<ClassInfo, CompileError> {
    use crate::ast::{ClassMember, Expr, MethodKind};
    // `extends Identifier` is supported; a computed superclass falls back.
    let super_name = match &class.super_class {
        None => None,
        Some(e) => match &**e {
            Expr::Ident(id) => Some(String::from(&*id.name)),
            _ => return Err(CompileError::Unsupported("computed extends")),
        },
    };
    let mut info = ClassInfo {
        super_name: super_name.clone(),
        ctor: None,
        methods: Vec::new(),
    };
    for member in &class.body {
        match member {
            ClassMember::Method(m) if !m.is_static && m.kind == MethodKind::Constructor => {
                let id = *next_id;
                *next_id += 1;
                info.ctor = Some(id);
                jobs.push(ClassJob {
                    id,
                    params: &m.value.params,
                    body: &m.value.body,
                    super_of: super_name.clone(),
                });
            }
            ClassMember::Method(m) if !m.is_static && m.kind == MethodKind::Method => {
                let id = *next_id;
                *next_id += 1;
                let name = static_key(&m.key)?;
                info.methods.push((name, id));
                jobs.push(ClassJob {
                    id,
                    params: &m.value.params,
                    body: &m.value.body,
                    super_of: None,
                });
            }
            // Fields, statics, getters/setters → fall back.
            _ => return Err(CompileError::Unsupported("class member")),
        }
    }
    Ok(info)
}

/// A class member queued for compilation: its reserved function id, signature,
/// and (for a constructor of a subclass) the superclass name for `super(...)`.
struct ClassJob<'a> {
    id: u32,
    params: &'a [crate::ast::Param],
    body: &'a [Stmt],
    super_of: Option<String>,
}

/// The nearest constructor up `name`'s `extends` chain (its own, else an
/// ancestor's — JS's implicit-super forwarding).
fn nearest_ctor(
    name: &str,
    classes: &alloc::collections::BTreeMap<String, ClassInfo>,
) -> Option<u32> {
    let info = classes.get(name)?;
    if let Some(c) = info.ctor {
        return Some(c);
    }
    nearest_ctor(info.super_name.as_deref()?, classes)
}

/// A compiled class: its superclass name (for `extends`), its constructor, and
/// its instance methods as function-table ids.
#[derive(Clone)]
struct ClassInfo {
    /// The `extends` superclass name, if any.
    super_name: Option<String>,
    /// The constructor's function id, if the class declares one.
    ctor: Option<u32>,
    /// `(method_name, function_id)` for each instance method.
    methods: Vec<(String, u32)>,
}

/// A variable binding: the register holding it, and whether that register holds
/// a *cell* (a one-element heap array) rather than the value directly. Captured
/// variables are cells so closures share their mutations.
#[derive(Clone, Copy)]
struct Binding {
    reg: Reg,
    cell: bool,
}

/// A single-pass register-allocating compiler from the AST to [`Op`]s.
#[derive(Default)]
struct Compiler {
    ops: Vec<Op>,
    /// Lexical scopes mapping a name to its binding.
    scopes: Vec<alloc::collections::BTreeMap<String, Binding>>,
    next_reg: Reg,
    /// Function name → table id, for resolving calls.
    fn_ids: alloc::rc::Rc<alloc::collections::BTreeMap<String, u32>>,
    /// Class name → its constructor/method function ids, for `new C(...)`.
    classes: alloc::rc::Rc<alloc::collections::BTreeMap<String, ClassInfo>>,
    /// The shared function table; nested function expressions append to it.
    protos: alloc::rc::Rc<core::cell::RefCell<Vec<FnProto>>>,
    /// Names in *this* function that are captured by a nested function and so
    /// must be stored as cells.
    cell_names: alloc::collections::BTreeSet<String>,
    /// The register holding `this` (seeded by the caller at `n_params +
    /// n_captures`).
    this_reg: Reg,
    /// When compiling a subclass constructor, the function id `super(...)` calls.
    super_ctor: Option<u32>,
    /// Per enclosing loop/switch: `break` jump indices awaiting the exit target
    /// (loops and `switch` both push here).
    break_sites: Vec<Vec<usize>>,
    /// Per enclosing loop: `continue` jump indices awaiting the loop's continue
    /// point (`switch` does *not* push here — `continue` targets the nearest
    /// loop).
    continue_sites: Vec<Vec<usize>>,
}

impl Compiler {
    /// Compiles one function. `params` bind to registers `0..n_params`;
    /// `captures` (a closure's free variables, in sorted order) bind to the next
    /// registers as cells. For `is_main`, the last expression's value is
    /// returned.
    #[allow(clippy::too_many_arguments)]
    fn compile_fn(
        fn_ids: &alloc::rc::Rc<alloc::collections::BTreeMap<String, u32>>,
        classes: &alloc::rc::Rc<alloc::collections::BTreeMap<String, ClassInfo>>,
        protos: &alloc::rc::Rc<core::cell::RefCell<Vec<FnProto>>>,
        params: &[crate::ast::Param],
        captures: &[String],
        body: &[Stmt],
        is_main: bool,
    ) -> Result<FnProto, CompileError> {
        Self::compile_fn_inner(
            fn_ids, classes, protos, params, captures, body, is_main, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_fn_inner(
        fn_ids: &alloc::rc::Rc<alloc::collections::BTreeMap<String, u32>>,
        classes: &alloc::rc::Rc<alloc::collections::BTreeMap<String, ClassInfo>>,
        protos: &alloc::rc::Rc<core::cell::RefCell<Vec<FnProto>>>,
        params: &[crate::ast::Param],
        captures: &[String],
        body: &[Stmt],
        is_main: bool,
        super_ctor: Option<u32>,
    ) -> Result<FnProto, CompileError> {
        // Which of this function's own names are captured by nested functions →
        // must be cells.
        let cell_names = captured_names(params, body);
        let mut c = Compiler {
            fn_ids: alloc::rc::Rc::clone(fn_ids),
            classes: alloc::rc::Rc::clone(classes),
            protos: alloc::rc::Rc::clone(protos),
            cell_names,
            super_ctor,
            ..Compiler::default()
        };
        c.scopes.push(alloc::collections::BTreeMap::new());
        // The VM places arguments in registers `0..n_params`, captured cells in
        // `n_params..n_params + n_captures`, and `this` right after, so reserve
        // those slots first…
        let arg_regs: Vec<Reg> = params.iter().map(|_| c.alloc()).collect();
        let cap_regs: Vec<Reg> = captures.iter().map(|_| c.alloc()).collect();
        c.this_reg = c.alloc(); // = n_params + n_captures
        // …then bind. A captured parameter is boxed into a fresh cell (preserving
        // the incoming argument value); a captured local that's a parameter must
        // share the cell so mutations are visible.
        for (i, p) in params.iter().enumerate() {
            let BindingTarget::Ident(Ident { name, .. }) = &p.target else {
                return Err(CompileError::Unsupported("destructuring parameter"));
            };
            let b = if c.cell_names.contains(&**name) {
                let cell = c.alloc();
                c.ops.push(Op::NewArray { dst: cell, len: 1 });
                let bind = Binding {
                    reg: cell,
                    cell: true,
                };
                c.write_var(bind, arg_regs[i]);
                bind
            } else {
                Binding {
                    reg: arg_regs[i],
                    cell: false,
                }
            };
            c.scopes
                .last_mut()
                .expect("a scope")
                .insert(String::from(&**name), b);
        }
        // Captured cells arrive already boxed (the closure passes the cell).
        for (j, name) in captures.iter().enumerate() {
            c.scopes.last_mut().expect("a scope").insert(
                name.clone(),
                Binding {
                    reg: cap_regs[j],
                    cell: true,
                },
            );
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
            n_captures: captures.len(),
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

    /// Declares `name`, allocating a register (and a backing cell if the name is
    /// captured). Returns the binding.
    fn declare(&mut self, name: &str) -> Binding {
        let reg = self.alloc();
        let cell = self.cell_names.contains(name);
        if cell {
            // A fresh one-element cell to hold the value.
            self.ops.push(Op::NewArray { dst: reg, len: 1 });
        }
        let b = Binding { reg, cell };
        self.scopes
            .last_mut()
            .expect("a scope")
            .insert(String::from(name), b);
        b
    }

    fn lookup(&self, name: &str) -> Option<Binding> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// Emits a read of `name` into a register and returns it (a cell read goes
    /// through `GetElem`).
    fn read_var(&mut self, b: Binding) -> Reg {
        if b.cell {
            let dst = self.alloc();
            let idx = self.constant(NanBox::number(0.0)).expect("const");
            self.ops.push(Op::GetElem {
                dst,
                arr: b.reg,
                index: idx,
            });
            dst
        } else {
            b.reg
        }
    }

    /// Emits a write of `src` into the variable bound by `b` (a cell write goes
    /// through `SetElem`).
    fn write_var(&mut self, b: Binding, src: Reg) {
        if b.cell {
            let idx = self.constant(NanBox::number(0.0)).expect("const");
            self.ops.push(Op::SetElem {
                arr: b.reg,
                index: idx,
                src,
            });
        } else {
            self.ops.push(Op::Move { dst: b.reg, src });
        }
    }

    /// Compiles a statement; returns the register of its value if it is an
    /// expression statement (for the program's completion value).
    fn stmt(&mut self, stmt: &Stmt) -> Result<Option<Reg>, CompileError> {
        match stmt {
            Stmt::Empty { .. } => Ok(None),
            // Function and (top-level) class declarations are compiled into the
            // table up front; nothing to emit at the declaration site.
            Stmt::Function(_) | Stmt::Class(_) => Ok(None),
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
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => {
                let d = self.expr(discriminant)?;
                // Only `break` targets a switch; `continue` skips to the loop.
                self.break_sites.push(Vec::new());
                // Dispatch: jump to the first matching `case` body (else default,
                // else the end). Bodies (compiled next) fall through.
                let mut case_jumps: Vec<(usize, usize)> = Vec::new();
                for (i, case) in cases.iter().enumerate() {
                    if let Some(test) = &case.test {
                        let t = self.expr(test)?;
                        let eq = self.alloc();
                        self.ops.push(Op::StrictEq {
                            dst: eq,
                            a: d,
                            b: t,
                        });
                        let skip = self.emit_jump_if_false(eq);
                        let to_body = self.emit_jump();
                        case_jumps.push((i, to_body));
                        self.patch(skip); // not this case → next test
                    }
                }
                let exit_dispatch = self.emit_jump(); // → default body, else end
                // Bodies, in order, falling through.
                let mut entries = alloc::vec![0usize; cases.len()];
                for (i, case) in cases.iter().enumerate() {
                    entries[i] = self.ops.len();
                    self.scopes.push(alloc::collections::BTreeMap::new());
                    for s in &case.body {
                        self.stmt(s)?;
                    }
                    self.scopes.pop();
                }
                for (i, j) in case_jumps {
                    self.patch_to(j, entries[i]);
                }
                match cases.iter().position(|c| c.test.is_none()) {
                    Some(di) => self.patch_to(exit_dispatch, entries[di]),
                    None => self.patch(exit_dispatch), // no default → end
                }
                let breaks = self.break_sites.pop().unwrap_or_default();
                let end = self.ops.len();
                for b in breaks {
                    self.patch_to(b, end);
                }
                Ok(None)
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                if handler.is_none() && finalizer.is_none() {
                    return Err(CompileError::Unsupported("try without catch/finally"));
                }
                // The register the thrown value lands in (and the catch binding,
                // if any, names it).
                let catch_reg = self.alloc();
                let push = self.ops.len();
                self.ops.push(Op::PushHandler {
                    target: 0,
                    reg: catch_reg,
                });
                self.block_stmts(block)?;
                self.ops.push(Op::PopHandler);
                // Normal completion: run `finally`, then jump past the handler.
                if let Some(fin) = finalizer {
                    self.block_stmts(fin)?;
                }
                let jend = self.emit_jump();

                // Handler entry: the thrown value is in `catch_reg`.
                self.patch(push);
                if let Some(catch) = handler {
                    self.scopes.push(alloc::collections::BTreeMap::new());
                    if let Some(BindingTarget::Ident(Ident { name, .. })) = &catch.param {
                        // The thrown value is in `catch_reg`; box it into a cell
                        // if the binding is captured.
                        let b = if self.cell_names.contains(&**name) {
                            let cell = self.alloc();
                            self.ops.push(Op::NewArray { dst: cell, len: 1 });
                            let bind = Binding {
                                reg: cell,
                                cell: true,
                            };
                            self.write_var(bind, catch_reg);
                            bind
                        } else {
                            Binding {
                                reg: catch_reg,
                                cell: false,
                            }
                        };
                        self.scopes
                            .last_mut()
                            .expect("a scope")
                            .insert(String::from(&**name), b);
                    }
                    for s in &catch.body {
                        self.stmt(s)?;
                    }
                    self.scopes.pop();
                    if let Some(fin) = finalizer {
                        self.block_stmts(fin)?;
                    }
                } else {
                    // `try { } finally { }`: run `finally`, then re-raise.
                    if let Some(fin) = finalizer {
                        self.block_stmts(fin)?;
                    }
                    self.ops.push(Op::Throw { src: catch_reg });
                }
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
                    self.write_var(slot, value);
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
            Stmt::Break { label: None, .. } => {
                let j = self.emit_jump();
                self.break_sites
                    .last_mut()
                    .ok_or(CompileError::Unsupported("break outside loop/switch"))?
                    .push(j);
                Ok(None)
            }
            Stmt::Continue { label: None, .. } => {
                let j = self.emit_jump();
                self.continue_sites
                    .last_mut()
                    .ok_or(CompileError::Unsupported("continue outside loop"))?
                    .push(j);
                Ok(None)
            }
            Stmt::Break { .. } | Stmt::Continue { .. } => {
                Err(CompileError::Unsupported("labeled break/continue"))
            }
            Stmt::While { test, body, .. } => {
                let top = self.ops.len();
                let cond = self.expr(test)?;
                let jf = self.emit_jump_if_false(cond);
                self.enter_loop();
                self.stmt(body)?;
                self.ops.push(Op::Jump { target: top });
                self.patch(jf);
                self.exit_loop(top); // `continue` re-tests
                Ok(None)
            }
            // `for (const x of arr)` over an array, indexed by a hidden counter.
            Stmt::ForOf {
                left, right, body, ..
            } => {
                use crate::ast::ForLeft;
                let ForLeft::Decl {
                    target: BindingTarget::Ident(Ident { name, .. }),
                    ..
                } = left
                else {
                    return Err(CompileError::Unsupported("for-of binding"));
                };
                self.scopes.push(alloc::collections::BTreeMap::new());
                let arr = self.expr(right)?;
                let len = self.alloc();
                self.ops.push(Op::ArrayLen { dst: len, arr });
                let i = self.alloc();
                self.ops.push(Op::LoadConst {
                    dst: i,
                    value: NanBox::number(0.0),
                });
                let elem = self.declare(name); // the loop variable
                let top = self.ops.len();
                let cond = self.alloc();
                self.ops.push(Op::Lt {
                    dst: cond,
                    a: i,
                    b: len,
                });
                let jf = self.emit_jump_if_false(cond);
                let cur = self.alloc();
                self.ops.push(Op::GetElem {
                    dst: cur,
                    arr,
                    index: i,
                });
                self.write_var(elem, cur);
                self.enter_loop();
                self.stmt(body)?;
                let cont = self.ops.len(); // `continue` advances the index
                let one = self.alloc();
                self.ops.push(Op::LoadConst {
                    dst: one,
                    value: NanBox::number(1.0),
                });
                self.ops.push(Op::Add {
                    dst: i,
                    a: i,
                    b: one,
                });
                self.ops.push(Op::Jump { target: top });
                self.patch(jf);
                self.exit_loop(cont);
                self.scopes.pop();
                Ok(None)
            }
            Stmt::DoWhile { body, test, .. } => {
                let top = self.ops.len();
                self.enter_loop();
                self.stmt(body)?;
                let cont = self.ops.len(); // `continue` re-tests
                let cond = self.expr(test)?;
                // Loop back to the top while the condition holds: jump if the
                // *negated* condition is false (i.e. while the condition is true).
                let not = self.alloc();
                self.ops.push(Op::Not { dst: not, a: cond });
                let jf = self.emit_jump_if_false(not);
                self.patch_to(jf, top);
                self.exit_loop(cont);
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
                self.enter_loop();
                self.stmt(body)?;
                let cont = self.ops.len(); // `continue` runs the update
                if let Some(u) = update {
                    self.expr(u)?;
                }
                self.ops.push(Op::Jump { target: top });
                if let Some(jf) = exit {
                    self.patch(jf);
                }
                self.exit_loop(cont);
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
            Expr::Ident(id) => {
                if let Some(b) = self.lookup(&id.name) {
                    Ok(self.read_var(b))
                } else if let Some(&func) = self.fn_ids.get(&*id.name) {
                    // A function referenced as a value: materialize a closure.
                    let dst = self.alloc();
                    self.ops.push(Op::LoadFunc { dst, func });
                    Ok(dst)
                } else {
                    Err(CompileError::Undefined(String::from(&*id.name)))
                }
            }
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
                // `super(args)` — run the base constructor on the current `this`.
                if matches!(&**callee, Expr::Super(_)) {
                    let ctor = self
                        .super_ctor
                        .ok_or(CompileError::Unsupported("super outside a subclass ctor"))?;
                    let recv = self.this_reg;
                    self.ops.push(Op::CallCtor { ctor, recv, args });
                    return Ok(self.this_reg);
                }
                // A built-in call (`console.log`, `Math.max`, `String`, …).
                if let Some(native) = native_call(callee).or_else(|| native_global(callee)) {
                    let dst = self.alloc();
                    self.ops.push(Op::CallNative { dst, native, args });
                    return Ok(dst);
                }
                // A direct call to a hoisted function by name (static dispatch +
                // recursion), when the name isn't shadowed by a local.
                if let Expr::Ident(id) = &**callee
                    && self.lookup(&id.name).is_none()
                    && let Some(&func) = self.fn_ids.get(&*id.name)
                {
                    let dst = self.alloc();
                    self.ops.push(Op::Call { dst, func, args });
                    return Ok(dst);
                }
                // A method call `recv.method(args)` (named, non-computed
                // property) binds `this` to the receiver.
                if let Expr::Member {
                    object,
                    property: PropertyKey::Ident(key) | PropertyKey::Str(key),
                    ..
                } = &**callee
                {
                    let recv = self.expr(object)?;
                    let dst = self.alloc();
                    self.ops.push(Op::CallMethod {
                        dst,
                        recv,
                        key: String::from(&**key),
                        args,
                    });
                    return Ok(dst);
                }
                // Otherwise an indirect call through a function *value* (a local
                // holding a function, or any callee expression).
                let callee_reg = self.expr(callee)?;
                let dst = self.alloc();
                self.ops.push(Op::CallValue {
                    dst,
                    callee: callee_reg,
                    args,
                });
                Ok(dst)
            }
            Expr::Assign {
                op, target, value, ..
            } => {
                use crate::ast::AssignOp;
                let compound = !matches!(op, AssignOp::Assign);
                match &**target {
                    Expr::Ident(id) => {
                        let b = self
                            .lookup(&id.name)
                            .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                        let v = self.expr(value)?;
                        let src = if compound {
                            let cur = self.read_var(b);
                            self.emit_binop(Self::compound_binop(*op)?, cur, v)?
                        } else {
                            v
                        };
                        self.write_var(b, src);
                        Ok(src)
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
                let b = self
                    .lookup(&id.name)
                    .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                let one = self.constant(NanBox::number(1.0))?;
                let bop = match op {
                    crate::ast::UpdateOp::Inc => BinaryOp::Add,
                    crate::ast::UpdateOp::Dec => BinaryOp::Sub,
                };
                // Keep the pre-update value for a postfix result.
                let cur = self.read_var(b);
                let old = self.alloc();
                self.ops.push(Op::Move { dst: old, src: cur });
                let next = self.emit_binop(bop, cur, one)?;
                self.write_var(b, next);
                Ok(if *prefix { next } else { old })
            }
            Expr::This(_) => Ok(self.this_reg),
            // `new C(args)` for a known plain class: create the instance, install
            // its methods, run the constructor with `this` = instance.
            Expr::New {
                callee, arguments, ..
            } => {
                let Expr::Ident(id) = &**callee else {
                    return Err(CompileError::Unsupported("new on non-class"));
                };
                let Some(info) = self.classes.get(&*id.name).cloned() else {
                    return Err(CompileError::Unsupported("new on unknown class"));
                };
                let mut args = Vec::with_capacity(arguments.len());
                for a in arguments {
                    let crate::ast::Argument::Item(e) = a else {
                        return Err(CompileError::Unsupported("spread argument"));
                    };
                    args.push(self.expr(e)?);
                }
                let _ = info;
                let instance = self.alloc();
                self.ops.push(Op::NewObject { dst: instance });
                // Walk the `extends` chain root→derived and install each class's
                // methods, so a derived method overrides an inherited one.
                let mut chain: Vec<String> = Vec::new();
                let mut cur = Some(String::from(&*id.name));
                while let Some(name) = cur {
                    cur = self.classes.get(&name).and_then(|c| c.super_name.clone());
                    chain.push(name);
                }
                for name in chain.iter().rev() {
                    let methods = self.classes.get(name).expect("class").methods.clone();
                    for (mname, mid) in &methods {
                        let m = self.alloc();
                        self.ops.push(Op::LoadFunc { dst: m, func: *mid });
                        self.ops.push(Op::SetProp {
                            obj: instance,
                            key: mname.clone(),
                            src: m,
                        });
                    }
                }
                // Run the most-derived constructor (or the nearest ancestor's,
                // forwarding args, when the class declares none).
                if let Some(ctor) = nearest_ctor(&id.name, &self.classes) {
                    self.ops.push(Op::CallCtor {
                        ctor,
                        recv: instance,
                        args,
                    });
                }
                Ok(instance)
            }
            // A template literal: interleave cooked quasis with interpolations,
            // concatenating via the realm's `+` (ToString on each value).
            Expr::Template(t) => {
                let cooked = |q: &crate::ast::TemplateElement| -> String {
                    q.cooked.as_deref().map(String::from).unwrap_or_default()
                };
                let mut acc = self.alloc();
                self.ops.push(Op::NewString {
                    dst: acc,
                    value: t.quasis.first().map(cooked).unwrap_or_default(),
                });
                for (i, e) in t.expressions.iter().enumerate() {
                    let v = self.expr(e)?;
                    let s1 = self.alloc();
                    self.ops.push(Op::AddValue {
                        dst: s1,
                        a: acc,
                        b: v,
                    });
                    let q = self.alloc();
                    self.ops.push(Op::NewString {
                        dst: q,
                        value: t.quasis.get(i + 1).map(cooked).unwrap_or_default(),
                    });
                    acc = self.alloc();
                    self.ops.push(Op::AddValue {
                        dst: acc,
                        a: s1,
                        b: q,
                    });
                }
                Ok(acc)
            }
            // A function expression / arrow → a closure capturing its free
            // variables (as shared cells).
            Expr::Function(f) => self.make_closure(&f.params, &f.body),
            Expr::Arrow(a) => {
                let body: Vec<Stmt> = match &a.body {
                    crate::ast::ArrowBody::Block(b) => b.clone(),
                    crate::ast::ArrowBody::Expr(e) => alloc::vec![Stmt::Return {
                        argument: Some(Box::new((**e).clone())),
                        span: crate::common::Span::point(0),
                    }],
                };
                self.make_closure(&a.params, &body)
            }
            _ => Err(CompileError::Unsupported("expression")),
        }
    }

    /// Compiles a nested function into the shared table and emits the code to
    /// build a closure over its captured cells.
    fn make_closure(
        &mut self,
        params: &[crate::ast::Param],
        body: &[Stmt],
    ) -> Result<Reg, CompileError> {
        // Captures = free variables that resolve to an enclosing binding (others
        // are top-level functions / globals, reached directly).
        let free = free_of_function(params, body);
        let captures: Vec<String> = free
            .into_iter()
            .filter(|n| self.lookup(n).is_some())
            .collect();
        // Reserve the new function's table id, compile it, then store it.
        let id = {
            let mut p = self.protos.borrow_mut();
            p.push(FnProto {
                ops: Vec::new(),
                n_regs: 0,
                n_params: 0,
                n_captures: 0,
            });
            (p.len() - 1) as u32
        };
        let proto = Compiler::compile_fn(
            &self.fn_ids,
            &self.classes,
            &self.protos,
            params,
            &captures,
            body,
            false,
        )?;
        self.protos.borrow_mut()[id as usize] = proto;
        // Capture the cell registers for each free variable (in the same sorted
        // order the callee binds them).
        let capture_regs: Vec<Reg> = captures
            .iter()
            .map(|n| self.lookup(n).expect("captured binding").reg)
            .collect();
        let dst = self.alloc();
        self.ops.push(Op::MakeClosure {
            dst,
            func: id,
            captures: capture_regs,
        });
        Ok(dst)
    }

    fn constant(&mut self, value: NanBox) -> Result<Reg, CompileError> {
        let r = self.alloc();
        self.ops.push(Op::LoadConst { dst: r, value });
        Ok(r)
    }

    /// Compiles a statement list in a fresh lexical scope.
    fn block_stmts(&mut self, stmts: &'_ [Stmt]) -> Result<(), CompileError> {
        self.scopes.push(alloc::collections::BTreeMap::new());
        for s in stmts {
            self.stmt(s)?;
        }
        self.scopes.pop();
        Ok(())
    }

    /// Emits the op(s) for `a <op> b` into a fresh register, returning it.
    fn emit_binop(&mut self, op: BinaryOp, a: Reg, b: Reg) -> Result<Reg, CompileError> {
        let dst = self.alloc();
        match op {
            BinaryOp::Add => self.ops.push(Op::AddValue { dst, a, b }),
            BinaryOp::Sub => self.ops.push(Op::Sub { dst, a, b }),
            BinaryOp::Mul => self.ops.push(Op::Mul { dst, a, b }),
            BinaryOp::Div => self.ops.push(Op::Div { dst, a, b }),
            BinaryOp::Mod => self.ops.push(Op::Mod { dst, a, b }),
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

    /// Opens a loop scope for `break`/`continue` collection.
    fn enter_loop(&mut self) {
        self.break_sites.push(Vec::new());
        self.continue_sites.push(Vec::new());
    }

    /// Closes a loop scope: `break`s jump past the loop (here), `continue`s jump
    /// to `continue_target`.
    fn exit_loop(&mut self, continue_target: usize) {
        let breaks = self.break_sites.pop().unwrap_or_default();
        let continues = self.continue_sites.pop().unwrap_or_default();
        let end = self.ops.len();
        for b in breaks {
            self.patch_to(b, end);
        }
        for c in continues {
            self.patch_to(c, continue_target);
        }
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

    #[cfg(feature = "std")]
    #[test]
    fn bytecode_math_float_natives() {
        assert_eq!(bc("Math.floor(3.9)"), "3");
        assert_eq!(bc("Math.ceil(3.1)"), "4");
        assert_eq!(bc("Math.round(2.5)"), "3");
        assert_eq!(bc("Math.sqrt(81)"), "9");
        assert_eq!(bc("Math.pow(2, 10)"), "1024");
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
    fn bytecode_finally() {
        // finally runs on the normal (no-throw) path.
        assert_eq!(
            bc("let log = ''; try { log += 't'; } finally { log += 'f'; } log"),
            "tf"
        );
        // finally runs after the catch on the throwing path.
        assert_eq!(
            bc(
                "let log = ''; try { log += 't'; throw 1; } catch (e) { log += 'c'; } finally { log += 'f'; } log"
            ),
            "tcf"
        );
        // try/finally (no catch): finally runs, then the throw propagates and is
        // caught by an outer try.
        assert_eq!(
            bc("let log = '';
                try {
                  try { log += 't'; throw 'x'; } finally { log += 'f'; }
                } catch (e) { log += 'o:' + e; }
                log"),
            "tfo:x"
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
        // String / Number coercion globals.
        assert_eq!(bc("String(42) + '!'"), "42!");
        assert_eq!(bc("Number('15') + 5"), "20");
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
    fn bytecode_for_of_arrays() {
        // Sum an array with for-of.
        assert_eq!(
            bc("let s = 0; for (const x of [3, 1, 4, 1, 5]) { s += x; } s"),
            "14"
        );
        // for-of with the loop variable used in an expression.
        assert_eq!(
            bc("let p = 1; for (const n of [1, 2, 3, 4]) { p *= n; } p"),
            "24"
        );
        // break / continue inside a for-of.
        assert_eq!(
            bc(
                "let s = 0; for (const x of [1, 2, 3, 4, 5]) { if (x === 4) { break; } if (x === 2) { continue; } s += x; } s"
            ),
            "4"
        );
        // for-of over an array built from a function result.
        assert_eq!(
            bc(
                "function pair(a, b) { return [a, b]; } let s = ''; for (const v of pair('x', 'y')) { s += v; } s"
            ),
            "xy"
        );
    }

    #[test]
    fn bytecode_break_continue_switch() {
        // break exits the loop.
        assert_eq!(
            bc("let s = 0; for (let i = 0; i < 100; i++) { if (i === 5) { break; } s += i; } s"),
            "10"
        );
        // continue skips to the next iteration.
        assert_eq!(
            bc(
                "let s = 0; for (let i = 0; i < 6; i++) { if (i % 2 === 0) { continue; } s += i; } s"
            ),
            "9"
        );
        // break / continue in a while loop.
        assert_eq!(
            bc(
                "let i = 0; let s = 0; while (true) { i++; if (i > 5) { break; } if (i === 3) { continue; } s += i; } s"
            ),
            "12"
        );
        // continue in a do/while.
        assert_eq!(
            bc(
                "let i = 0; let s = 0; do { i++; if (i === 2) { continue; } s += i; } while (i < 4); s"
            ),
            "8"
        );
        // switch with fall-through and default; break ends the switch.
        assert_eq!(
            bc("function classify(n) {
                  let r = '';
                  switch (n) {
                    case 1: r = 'one'; break;
                    case 2:
                    case 3: r = 'few'; break;
                    default: r = 'many';
                  }
                  return r;
                }
                classify(1) + ',' + classify(2) + ',' + classify(3) + ',' + classify(9)"),
            "one,few,few,many"
        );
        // A continue inside a switch targets the enclosing loop.
        assert_eq!(
            bc(
                "let s = 0; for (let i = 0; i < 4; i++) { switch (i) { case 1: continue; default: s += i; } } s"
            ),
            "5"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn execute_bytecode_first_with_tree_walker_fallback() {
        // A program the bytecode VM compiles fully (closures, loops, output).
        let (out, _) = execute(
            "function makeCounter() { let c = 0; return function() { c += 1; return c; }; }
             let n = makeCounter(); console.log(n()); console.log(n());",
        )
        .expect("ok");
        assert_eq!(out, "1\n2\n");

        // Plain classes now compile to bytecode directly.
        let (out, _) = execute(
            "class Point { constructor(x, y) { this.x = x; this.y = y; }
               sum() { return this.x + this.y; } }
             console.log(new Point(3, 4).sum());",
        )
        .expect("ok");
        assert_eq!(out, "7\n");

        // A class feature the bytecode path doesn't compile (a getter) routes
        // the program to the tree-walker, which still runs it correctly.
        let (out, _) = execute(
            "class Box { constructor(v) { this._v = v; } get value() { return this._v * 2; } }
             console.log(new Box(21).value);",
        )
        .expect("ok");
        assert_eq!(out, "42\n");

        // The completion value is surfaced for an expression program.
        let (_, completion) = execute("1 + 2 * 3").expect("ok");
        assert_eq!(completion, "7");

        // Both engines agree on a shared program (sanity).
        let src = "let s = 0; for (let i = 1; i <= 5; i++) { s += i; } console.log(s);";
        let (bc, _) = execute(src).expect("ok");
        let (tw, _) = crate::nbexec::eval_source(src).expect("ok");
        assert_eq!(bc, tw);
    }

    #[test]
    fn bytecode_classes() {
        // A class with a constructor and a method using `this`.
        assert_eq!(
            bc("class Point {
                  constructor(x, y) { this.x = x; this.y = y; }
                  sum() { return this.x + this.y; }
                }
                new Point(3, 4).sum()"),
            "7"
        );
        // A mutable instance via methods.
        assert_eq!(
            bc("class Counter {
                  constructor() { this.n = 0; }
                  inc() { this.n += 1; return this.n; }
                }
                let c = new Counter(); c.inc(); c.inc(); c.inc()"),
            "3"
        );
        // A method calling another via `this`.
        assert_eq!(
            bc("class Calc {
                  constructor(v) { this.v = v; }
                  dbl() { return this.v * 2; }
                  quad() { return this.dbl() * 2; }
                }
                new Calc(5).quad()"),
            "20"
        );
        // Two instances keep independent state.
        assert_eq!(
            bc(
                "class Box { constructor(v) { this.v = v; } get() { return this.v; } }
                let a = new Box(1); let b = new Box(99);
                a.get() + ',' + b.get()"
            ),
            "1,99"
        );
    }

    #[test]
    fn bytecode_class_inheritance() {
        // A subclass inherits a base method.
        assert_eq!(
            bc(
                "class Animal { constructor(n) { this.n = n; } describe() { return this.n; } }
                class Dog extends Animal {}
                new Dog('Rex').describe()"
            ),
            "Rex"
        );
        // super(...) calls the base constructor; the derived adds state.
        assert_eq!(
            bc("class Animal { constructor(n) { this.n = n; } }
                class Dog extends Animal {
                  constructor(n, b) { super(n); this.b = b; }
                  tag() { return this.n + ':' + this.b; }
                }
                new Dog('Rex', 'Lab').tag()"),
            "Rex:Lab"
        );
        // A derived method overrides the base.
        assert_eq!(
            bc("class A { kind() { return 'A'; } }
                class B extends A { kind() { return 'B'; } }
                new B().kind() + new A().kind()"),
            "BA"
        );
        // Implicit super (subclass with no constructor) forwards args.
        assert_eq!(
            bc(
                "class Base { constructor(v) { this.v = v; } get() { return this.v; } }
                class Sub extends Base {}
                new Sub(7).get()"
            ),
            "7"
        );
        // A three-level chain accumulating fields.
        assert_eq!(
            bc("class A { constructor() { this.a = 1; } }
                class B extends A { constructor() { super(); this.b = 2; } }
                class C extends B { constructor() { super(); this.c = 3; } }
                let o = new C(); o.a + o.b + o.c"),
            "6"
        );
    }

    #[test]
    fn bytecode_this_and_methods() {
        // A method using `this` on an object literal.
        assert_eq!(
            bc("let o = { x: 10, getX: function() { return this.x; } }; o.getX()"),
            "10"
        );
        // A method mutating instance state via `this`.
        assert_eq!(
            bc(
                "let c = { n: 0, inc: function() { this.n += 1; return this.n; } };
                c.inc(); c.inc(); c.inc()"
            ),
            "3"
        );
        // A method calling another method on `this`.
        assert_eq!(
            bc("let calc = {
                  v: 5,
                  dbl: function() { return this.v * 2; },
                  quad: function() { return this.dbl() * 2; }
                };
                calc.quad()"),
            "20"
        );
        // `this` flows through nested object method calls.
        assert_eq!(
            bc(
                "let acc = { total: 0, add: function(n) { this.total += n; return this; } };
                acc.add(3); acc.add(4); acc.total"
            ),
            "7"
        );
    }

    #[test]
    fn bytecode_template_literals() {
        assert_eq!(bc("let n = 'world'; `Hello, ${n}!`"), "Hello, world!");
        assert_eq!(
            bc("let a = 2, b = 3; `${a} + ${b} = ${a + b}`"),
            "2 + 3 = 5"
        );
        assert_eq!(bc("`no interpolation`"), "no interpolation");
        // A template in a function, over a captured value.
        assert_eq!(
            bc("function greet(who) { return `hi ${who}`; } greet('ada')"),
            "hi ada"
        );
    }

    #[test]
    fn bytecode_closures_and_capture() {
        // Capture by value (read-only): currying.
        assert_eq!(
            bc("function adder(x) { return function(y) { return x + y; }; } adder(3)(4)"),
            "7"
        );
        // Arrow closures, deeper currying.
        assert_eq!(
            bc("let add = (a) => (b) => (c) => a + b + c; add(1)(2)(3)"),
            "6"
        );
        // Mutable shared capture: a counter whose closure mutates the captured
        // variable — the headline closure case.
        assert_eq!(
            bc(
                "function makeCounter() { let c = 0; return function() { c = c + 1; return c; }; }
                let n = makeCounter();
                n(); n(); n()"
            ),
            "3"
        );
        // Two counters keep independent state.
        assert_eq!(
            bc(
                "function makeCounter() { let c = 0; return function() { c += 1; return c; }; }
                let a = makeCounter(); let b = makeCounter();
                a(); a(); b();
                a() + ',' + b()"
            ),
            "3,2"
        );
        // A closure observes a mutation made after it was created (shared cell).
        assert_eq!(
            bc(
                "function f() { let v = 'before'; let read = function() { return v; }; v = 'after'; return read(); } f()"
            ),
            "after"
        );
        // The accumulator pattern.
        assert_eq!(
            bc("function makeAcc() { let total = 0; return function(n) { total += n; return total; }; }
                let acc = makeAcc(); acc(10); acc(20); acc(5)"),
            "35"
        );
    }

    #[test]
    fn bytecode_first_class_functions() {
        // A function passed by name and called indirectly (higher-order).
        assert_eq!(
            bc(
                "function apply(f, x) { return f(x); } function dbl(n) { return n * 2; } apply(dbl, 21)"
            ),
            "42"
        );
        // A function stored in a variable, then called via the variable.
        assert_eq!(
            bc("function inc(n) { return n + 1; } let g = inc; g(g(g(10)))"),
            "13"
        );
        // Selecting one of several functions at runtime.
        assert_eq!(
            bc("function add(a, b) { return a + b; }
                function mul(a, b) { return a * b; }
                function pick(cond) { if (cond) { return add; } return mul; }
                pick(true)(3, 4) + ',' + pick(false)(3, 4)"),
            "7,12"
        );
        // A function value passed through an array element.
        assert_eq!(
            bc("function sq(n) { return n * n; } let ops = [sq]; ops[0](9)"),
            "81"
        );
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
