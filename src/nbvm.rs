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
    /// A realm-backed binary op (`**`, bitwise, loose `==`/`!=`) selected by a
    /// small op code — for operators that need full coercion / `i32` semantics.
    ValueBin { dst: Reg, op: u8, a: Reg, b: Reg },
    /// `dst = (key in obj)` — own-property / array-index membership.
    HasProp { dst: Reg, key: Reg, obj: Reg },
    /// `dst = obj is a built-in of `kind`` (`0`=RegExp, `1`=Array, `2`=Map,
    /// `3`=Set) — for `instanceof RegExp`/`Array`/`Map`/`Set`.
    IsBuiltin { dst: Reg, obj: Reg, kind: u8 },
    /// `dst = delete obj[key]` — removes own property `key`, yielding `true`.
    DeleteProp { dst: Reg, obj: Reg, key: Reg },
    /// Tags the object in `obj` with `class_id` (for `instanceof`).
    SetClassTag { obj: Reg, class_id: u32 },
    /// Defines accessor `key` on `obj` with `getter`/`setter` closure registers
    /// (either may be `undefined`).
    DefineAccessor {
        obj: Reg,
        key: String,
        getter: Reg,
        setter: Reg,
    },
    /// `dst = (obj instanceof C)` — true iff `obj`'s class tag is one of `ids`
    /// (the queried class and all its subclasses, computed at compile time).
    InstanceOf {
        dst: Reg,
        obj: Reg,
        ids: alloc::rc::Rc<[u32]>,
    },
    /// `dst = typeof a` (a heap string).
    TypeOf { dst: Reg, a: Reg },
    /// `dst = ~a` (bitwise NOT, `i32` semantics).
    BitNot { dst: Reg, a: Reg },
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
    /// `new Array(arg)` with a single argument: a number is the new array's
    /// length (a `RangeError` if invalid/too large); anything else is its sole
    /// element.
    NewArrayCtor { dst: Reg, arg: Reg },
    /// `dst = arr[index]` (index taken from a register, `undefined` if absent).
    GetElem { dst: Reg, arr: Reg, index: Reg },
    /// `arr[index] = src` (grows the array if needed).
    SetElem { arr: Reg, index: Reg, src: Reg },
    /// `dst = obj[key]` — dynamic access: array element for a numeric key on an
    /// array, else a string property.
    GetKey { dst: Reg, obj: Reg, key: Reg },
    /// `obj[key] = src` — the dynamic mirror of `GetKey`.
    SetKey { obj: Reg, key: Reg, src: Reg },
    /// Copies the own properties (or array elements) of `src` into object `dst`
    /// (an object-literal `...spread`).
    ObjectSpread { dst: Reg, src: Reg },
    /// `dst = a new array of `obj`'s enumerable keys` (own property names, or
    /// array index strings) — for `for-in`.
    EnumKeys { dst: Reg, obj: Reg },
    /// `dst = arr.length` (array length or string character count).
    ArrayLen { dst: Reg, arr: Reg },
    /// `dst = coll.size` (a `Map`/`Set`'s entry count).
    CollectionSize { dst: Reg, recv: Reg },
    /// Appends the value in `src` to array `arr` (grows it).
    ArrayPush { arr: Reg, src: Reg },
    /// Appends every element of the array in `src` to `arr` (a spread).
    ArrayExtend { arr: Reg, src: Reg },
    /// `dst = src.slice(from)` — a new array of `src`'s elements from index
    /// `from` (a numeric register) onward (for a rest pattern).
    ArraySliceFrom { dst: Reg, src: Reg, from: Reg },
    /// `dst = { ...src }` minus the `exclude` keys — `src`'s own properties not
    /// already destructured (an object-rest pattern).
    ObjectRest {
        dst: Reg,
        src: Reg,
        exclude: alloc::rc::Rc<[String]>,
    },
    /// `dst = a new `Map`/`Set``, optionally seeded from the iterable array in
    /// `seed` (a `Set` from its elements, a `Map` from `[k, v]` pairs).
    NewCollection {
        dst: Reg,
        is_set: bool,
        seed: Option<Reg>,
    },
    /// `dst = /source/flags` — a new `RegExp` value.
    NewRegExp {
        dst: Reg,
        source: String,
        flags: String,
    },
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
    /// `dst = callee(args…)` with `this = recv` — an indirect call through a
    /// function value, binding an explicit receiver (used by `super.method()`).
    CallValueThis {
        dst: Reg,
        callee: Reg,
        recv: Reg,
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

// `Op::ValueBin` op codes.
const VB_POW: u8 = 0;
/// `Op::ValueBin` op code for `&` (exposed for the JIT's bitwise lowering).
pub(crate) const VB_BIT_AND: u8 = 1;
/// `Op::ValueBin` op code for `|`.
pub(crate) const VB_BIT_OR: u8 = 2;
/// `Op::ValueBin` op code for `^`.
pub(crate) const VB_BIT_XOR: u8 = 3;
/// `Op::ValueBin` op code for `<<`.
pub(crate) const VB_SHL: u8 = 4;
/// `Op::ValueBin` op code for `>>`.
pub(crate) const VB_SHR: u8 = 5;
/// `Op::ValueBin` op code for `>>>`.
pub(crate) const VB_USHR: u8 = 6;
/// `Op::ValueBin` op code for loose `==` (exposed for the JIT's integer lowering).
pub(crate) const VB_LOOSE_EQ: u8 = 7;
const VB_LOOSE_NEQ: u8 = 8;

// Native built-in ids for `Op::CallNative`.
const NB_CONSOLE_LOG: u16 = 0;
pub(crate) const NB_MATH_MAX: u16 = 1;
pub(crate) const NB_MATH_MIN: u16 = 2;
pub(crate) const NB_MATH_ABS: u16 = 3;
pub(crate) const NB_MATH_FLOOR: u16 = 4;
pub(crate) const NB_MATH_CEIL: u16 = 5;
const NB_MATH_ROUND: u16 = 6;
pub(crate) const NB_MATH_SQRT: u16 = 7;
const NB_MATH_POW: u16 = 8;
const NB_STRING: u16 = 9;
const NB_NUMBER: u16 = 10;
const NB_PARSE_INT: u16 = 11;
const NB_PARSE_FLOAT: u16 = 12;
const NB_IS_NAN: u16 = 13;
const NB_IS_FINITE: u16 = 14;
const NB_OBJECT_KEYS: u16 = 15;
const NB_OBJECT_VALUES: u16 = 16;
const NB_OBJECT_ENTRIES: u16 = 17;
const NB_OBJECT_ASSIGN: u16 = 18;
const NB_JSON_STRINGIFY: u16 = 19;
const NB_JSON_PARSE: u16 = 20;
const NB_NUMBER_IS_INTEGER: u16 = 21;
const NB_NUMBER_IS_FINITE: u16 = 22;
const NB_NUMBER_IS_NAN: u16 = 23;
const NB_NUMBER_PARSE_FLOAT: u16 = 24;
const NB_NUMBER_PARSE_INT: u16 = 25;
const NB_STRING_FROM_CHAR_CODE: u16 = 26;
const NB_ARRAY_FROM: u16 = 27;
const NB_ARRAY_IS_ARRAY: u16 = 28;
const NB_OBJECT_FROM_ENTRIES: u16 = 29;
const NB_PROMISE_RESOLVE: u16 = 30;
const NB_PROMISE_REJECT: u16 = 31;
pub(crate) const NB_MATH_TRUNC: u16 = 32;

/// Built-in globals the tree-walker provides as bare values. An unknown
/// identifier that is *not* one of these throws a `ReferenceError` at runtime
/// (correct JS); one that *is* falls back so the tree-walker resolves it.
const KNOWN_GLOBALS: &[&str] = &[
    "Math",
    "console",
    "Promise",
    "Date",
    "RegExp",
    "JSON",
    "Object",
    "Array",
    "String",
    "Number",
    "Boolean",
    "parseInt",
    "parseFloat",
    "isNaN",
    "isFinite",
    "Map",
    "Set",
    "Symbol",
    "BigInt",
    "WeakMap",
    "WeakSet",
    "WeakRef",
    "Proxy",
    "Reflect",
    "Function",
    "ArrayBuffer",
    "DataView",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "Float32Array",
    "Float64Array",
    "BigInt64Array",
    "BigUint64Array",
    "globalThis",
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "EvalError",
    "URIError",
    "AggregateError",
    "Intl",
    "WebAssembly",
    "structuredClone",
    "setTimeout",
    "clearTimeout",
    "queueMicrotask",
];

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
    /// If the last parameter is a rest (`...args`), the number of *fixed*
    /// parameters before it; the caller gathers any further arguments into an
    /// array placed in register `fixed`.
    pub rest_from: Option<usize>,
    /// An `async` function: its completion is wrapped in a settled `Promise`.
    pub is_async: bool,
    /// `Function.prototype.length`: parameters before the first one with a default
    /// value or the rest parameter.
    pub length: usize,
    /// `Function.prototype.name`: the function's own name, or a name inferred from
    /// the binding/property it was assigned to (empty for a truly anonymous one).
    pub name: alloc::string::String,
}

/// A queued promise reaction: run `handler(value)` then settle `result` with
/// the outcome (`fulfilled` indicates which settlement the value came from, for
/// pass-through reactions where `handler` is `undefined`).
struct Microtask {
    handler: NanBox,
    value: NanBox,
    result: Handle,
    fulfilled: bool,
}

/// Execution context shared across activations: the heap, the captured
/// `console` output sink, and the promise microtask queue (the event loop).
struct Ctx<'a> {
    realm: &'a mut Realm,
    output: String,
    microtasks: alloc::collections::VecDeque<Microtask>,
    /// Per-function tiering state, keyed by function id.
    tiers: alloc::collections::BTreeMap<usize, TierState>,
    /// Per-function native-code cache (Phase G JIT), keyed by function id.
    /// `None` means "tried, not JIT-eligible". Present only where the JIT exists.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    jit_cache: alloc::collections::BTreeMap<usize, Option<alloc::rc::Rc<crate::jit::JitProto>>>,
    /// Function-call nesting depth (recursion guard).
    call_depth: usize,
}

/// Maximum function-call nesting before a `RangeError` (recursion guard), sized
/// to fire before the large stack the engine entry points run on overflows.
const MAX_VM_CALL_DEPTH: usize = 3500;

/// Tier state for one function: its activation count and, once hot, its
/// optimized bytecode body.
type TierState = (u32, Option<alloc::rc::Rc<Vec<Op>>>);

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
        microtasks: alloc::collections::VecDeque::new(),
        tiers: alloc::collections::BTreeMap::new(),
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_cache: alloc::collections::BTreeMap::new(),
        call_depth: 0,
    };
    let value = call(&mut ctx, funcs, id, args)?;
    drain_microtasks(&mut ctx, funcs)?;
    Ok(value)
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
        microtasks: alloc::collections::VecDeque::new(),
        tiers: alloc::collections::BTreeMap::new(),
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_cache: alloc::collections::BTreeMap::new(),
        call_depth: 0,
    };
    let value = call(&mut ctx, funcs, id, args)?;
    // Run the promise event loop before returning (then-callbacks, async tails).
    drain_microtasks(&mut ctx, funcs)?;
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
    // Recursion guard: throw a catchable `RangeError` rather than overflowing.
    if ctx.call_depth >= MAX_VM_CALL_DEPTH {
        let e = make_error(ctx.realm, "RangeError", "Maximum call stack size exceeded");
        return Err(VmError::Thrown(e));
    }
    ctx.call_depth += 1;
    let result = call_with_inner(ctx, funcs, id, args, captures, this_val);
    ctx.call_depth -= 1;
    result
}

fn call_with_inner(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    args: &[NanBox],
    captures: &[NanBox],
    this_val: NanBox,
) -> Result<NanBox, VmError> {
    let proto = &funcs[id];
    let mut regs: Vec<NanBox> = vec![NanBox::undefined(); proto.n_regs];
    match proto.rest_from {
        // A rest parameter: fixed args fill `0..fixed`, the remainder becomes an
        // array in register `fixed`.
        Some(fixed) => {
            for (i, a) in args.iter().enumerate().take(fixed) {
                regs[i] = *a;
            }
            let rest: Vec<NanBox> = args.get(fixed..).unwrap_or(&[]).to_vec();
            let arr = ctx.realm.new_array(rest);
            if fixed < regs.len() {
                regs[fixed] = NanBox::handle(arr.to_raw());
            }
        }
        None => {
            for (i, a) in args.iter().enumerate().take(proto.n_params) {
                regs[i] = *a;
            }
        }
    }
    for (i, c) in captures.iter().enumerate().take(proto.n_captures) {
        regs[proto.n_params + i] = *c;
    }
    // The `this` slot sits right after the captures.
    if let Some(slot) = regs.get_mut(proto.n_params + proto.n_captures) {
        *slot = this_val;
    }
    // Tier-up: count this activation, optimize the body once the function gets
    // hot, and run the optimized bytecode thereafter.
    let optimized: Option<alloc::rc::Rc<Vec<Op>>> = {
        let entry = ctx.tiers.entry(id).or_default();
        entry.0 = entry.0.saturating_add(1);
        if entry.0 == TIER_UP_THRESHOLD && entry.1.is_none() {
            entry.1 = Some(alloc::rc::Rc::new(optimize_ops(&funcs[id].ops)));
        }
        entry.1.clone()
    };
    let body: &[Op] = match &optimized {
        Some(o) => o.as_slice(),
        None => proto.ops.as_slice(),
    };
    // Native fast path (Phase G JIT): once a function is hot, try compiling it to
    // machine code. Eligible functions are pure straight-line/looping integer
    // arithmetic (no side effects), so running the native code is observationally
    // equivalent to the interpreter; a non-integer/overflowing call deopts to
    // `None` and we fall through to `run_frame`.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    if optimized.is_some() && !proto.is_async && proto.rest_from.is_none() && proto.n_captures == 0
    {
        let mut stack = alloc::collections::BTreeSet::new();
        let cached = ensure_jit(&mut ctx.jit_cache, funcs, id, &mut stack);
        if let Some(jit) = cached
            && let Some(result) = jit.call_guarded(args)
        {
            return Ok(result);
        }
    }
    // An `async` function: its synchronous body runs to completion, and its
    // result (or thrown value) settles a returned `Promise`. (No `await` yet —
    // a body that awaits falls back at compile time.)
    if proto.is_async {
        let p = ctx.realm.new_promise();
        match run_frame(ctx, funcs, body, &mut regs) {
            Ok(ret) => settle(ctx, p, ret.unwrap_or(NanBox::undefined()), true),
            Err(VmError::Thrown(e)) => settle(ctx, p, e, false),
            Err(other) => return Err(other),
        }
        return Ok(NanBox::handle(p.to_raw()));
    }
    Ok(run_frame(ctx, funcs, body, &mut regs)?.unwrap_or(NanBox::undefined()))
}

/// Compiles function `id` to native code (memoized in `cache`), first compiling
/// any function it statically calls so their code addresses can be wired into a
/// native call. Returns the compiled function, or `None` if it isn't JIT-eligible.
///
/// `stack` guards against recursion: a function reached while already being
/// compiled (direct or mutual recursion) is left unregistered, so the recursive
/// call simply bails the caller's JIT — sound, just not accelerated.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
fn ensure_jit(
    cache: &mut alloc::collections::BTreeMap<usize, Option<alloc::rc::Rc<crate::jit::JitProto>>>,
    funcs: &[FnProto],
    id: usize,
    stack: &mut alloc::collections::BTreeSet<usize>,
) -> Option<alloc::rc::Rc<crate::jit::JitProto>> {
    if let Some(c) = cache.get(&id) {
        return c.clone();
    }
    if stack.contains(&id) {
        return None; // recursion: this function is mid-compilation
    }
    stack.insert(id);
    // Resolve each statically-called function to its compiled code address.
    let mut registry = alloc::collections::BTreeMap::new();
    for op in &funcs[id].ops {
        if let Op::Call { func, .. } = op {
            let fid = *func as usize;
            if fid != id
                && fid < funcs.len()
                && let Some(j) = ensure_jit(cache, funcs, fid, stack)
            {
                registry.insert(*func, j.code_ptr() as u64);
            }
        }
    }
    stack.remove(&id);
    let compiled =
        crate::jit::JitProto::compile_with_registry(&funcs[id], &registry).map(alloc::rc::Rc::new);
    cache.insert(id, compiled.clone());
    compiled
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
        microtasks: alloc::collections::VecDeque::new(),
        tiers: alloc::collections::BTreeMap::new(),
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_cache: alloc::collections::BTreeMap::new(),
        call_depth: 0,
    };
    Ok(run_frame(&mut ctx, &[], program, &mut regs)?.unwrap_or(NanBox::undefined()))
}

/// Builds an error value `{ name, message }` in the realm (for runtime throws).
/// For `==`/`!=`: when exactly one operand is a non-string object reference and
/// the other is a primitive (number/boolean/string), convert the object to its
/// ToPrimitive (toString) form so e.g. `[] == 0` compares as `"" == 0`. Two
/// objects (compared by identity) and two primitives are left untouched.
fn loose_eq_coerce(realm: &mut Realm, x: NanBox, y: NanBox) -> (NanBox, NanBox) {
    let is_obj = |realm: &Realm, v: NanBox| {
        v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| realm.string_value(h).is_none())
    };
    let is_prim = |realm: &Realm, v: NanBox| {
        v.as_number().is_some()
            || matches!(v.unpack(), crate::nanbox::Unpacked::Bool(_))
            || v.as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| realm.string_value(h).is_some())
    };
    if is_obj(realm, x) && is_prim(realm, y) {
        let s = realm.to_display_string(x);
        (NanBox::handle(realm.new_string(&s).to_raw()), y)
    } else if is_obj(realm, y) && is_prim(realm, x) {
        let s = realm.to_display_string(y);
        (x, NanBox::handle(realm.new_string(&s).to_raw()))
    } else {
        (x, y)
    }
}

/// ECMAScript ToPrimitive for the bytecode VM's operators: for a plain object
/// carrying a user-defined `valueOf`/`toString`, calls it (in the hint's order)
/// and returns the primitive result. Non-objects, strings, Dates, and objects
/// without a usable own method are returned unchanged (the realm then applies its
/// default coercion). Keeps numeric fast paths cheap: a non-handle returns at once.
fn to_primitive(ctx: &mut Ctx, funcs: &[FnProto], v: NanBox, number_hint: bool) -> NanBox {
    let Some(raw) = v.as_handle() else {
        return v;
    };
    let h = Handle::from_raw(raw);
    if ctx.realm.string_value(h).is_some() || ctx.realm.date_at(h).is_some() {
        return v;
    }
    let order: [&str; 2] = if number_hint {
        ["valueOf", "toString"]
    } else {
        ["toString", "valueOf"]
    };
    for name in order {
        if let Some(method) = ctx.realm.get_property(h, name)
            && method.as_handle().is_some()
            && let Ok(res) = call_closure(ctx, funcs, method, &[], v)
        {
            use crate::nanbox::Unpacked;
            let is_prim = res.as_number().is_some()
                || matches!(
                    res.unpack(),
                    Unpacked::Bool(_) | Unpacked::Null | Unpacked::Undefined
                )
                || res
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|rh| ctx.realm.string_value(rh).is_some());
            if is_prim {
                return res;
            }
        }
    }
    v
}

fn make_error(realm: &mut Realm, name: &str, message: &str) -> NanBox {
    let obj = realm.new_object();
    let n = NanBox::handle(realm.new_string(name).to_raw());
    realm.set_property(obj, "name", n);
    let m = NanBox::handle(realm.new_string(message).to_raw());
    realm.set_property(obj, "message", m);
    NanBox::handle(obj.to_raw())
}

const SYM_NUM_ERR: &str = "Cannot convert a Symbol value to a number";
const SYM_STR_ERR: &str = "Cannot convert a Symbol value to a string";

/// A `TypeError` value when `x` or `y` is a Symbol (which has no implicit numeric/
/// string conversion), else `None`. The caller `handle_throw!`s the result so the
/// VM's exception-handler stack (`try`/`catch`) sees it.
fn symbol_coercion_error(realm: &mut Realm, x: NanBox, y: NanBox, msg: &str) -> Option<NanBox> {
    let is_sym = |v: NanBox| {
        v.as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| realm.symbol_at(h).is_some())
    };
    if is_sym(x) || is_sym(y) {
        Some(make_error(realm, "TypeError", msg))
    } else {
        None
    }
}

/// Settles promise `p` with `value` (fulfilled or rejected), queueing its
/// pending reactions as microtasks. A no-op if already settled or not a promise.
fn settle(ctx: &mut Ctx, p: Handle, value: NanBox, fulfilled: bool) {
    use crate::cell::PromiseStatus;
    let Some(state) = ctx.realm.promise_state(p) else {
        return;
    };
    let reactions = {
        let mut s = state.borrow_mut();
        if s.status != PromiseStatus::Pending {
            return;
        }
        s.status = if fulfilled {
            PromiseStatus::Fulfilled
        } else {
            PromiseStatus::Rejected
        };
        s.value = value;
        core::mem::take(&mut s.reactions)
    };
    for r in reactions {
        let handler = if fulfilled {
            r.on_fulfilled
        } else {
            r.on_rejected
        };
        ctx.microtasks.push_back(Microtask {
            handler,
            value,
            result: r.result,
            fulfilled,
        });
    }
}

/// `promise.then(on_fulfilled, on_rejected)` — registers reactions and returns
/// the dependent promise.
fn promise_then(ctx: &mut Ctx, p: Handle, on_f: NanBox, on_r: NanBox) -> Handle {
    use crate::cell::{PromiseStatus, Reaction};
    let result = ctx.realm.new_promise();
    let Some(state) = ctx.realm.promise_state(p) else {
        return result;
    };
    let (status, value) = {
        let s = state.borrow();
        (s.status, s.value)
    };
    match status {
        PromiseStatus::Pending => state.borrow_mut().reactions.push(Reaction {
            on_fulfilled: on_f,
            on_rejected: on_r,
            result,
            finally: false,
        }),
        PromiseStatus::Fulfilled => ctx.microtasks.push_back(Microtask {
            handler: on_f,
            value,
            result,
            fulfilled: true,
        }),
        PromiseStatus::Rejected => ctx.microtasks.push_back(Microtask {
            handler: on_r,
            value,
            result,
            fulfilled: false,
        }),
    }
    result
}

/// Runs the promise microtask queue to completion (the event loop).
fn drain_microtasks(ctx: &mut Ctx, funcs: &[FnProto]) -> Result<(), VmError> {
    while let Some(task) = ctx.microtasks.pop_front() {
        if task.handler.as_handle().is_some() {
            // A handler runs and its result settles the dependent promise.
            match call_closure(ctx, funcs, task.handler, &[task.value], NanBox::undefined()) {
                Ok(ret) => settle(ctx, task.result, ret, true),
                Err(VmError::Thrown(e)) => settle(ctx, task.result, e, false),
                Err(other) => return Err(other),
            }
        } else {
            // Pass-through: settle the dependent promise with the same value.
            settle(ctx, task.result, task.value, task.fulfilled);
        }
    }
    Ok(())
}

// --- optimizing tier: a constant-folding bytecode → bytecode pass ---

/// The number of activations after which a function is "tiered up": its baseline
/// bytecode is run through [`optimize_ops`] once and the result reused.
const TIER_UP_THRESHOLD: u32 = 8;

/// A bytecode→bytecode optimizer (the second tier): folds compile-time-constant
/// arithmetic within each basic block, so an expression like `2 ** 10` or
/// `1 + 2 * 3` collapses to a single `LoadConst`. Op count is preserved (folded
/// ops are rewritten in place), so jump targets stay valid.
///
/// Soundness: a register's constant value is tracked only from a `LoadConst`,
/// and the tracking is conservatively cleared at every basic-block leader (jump
/// target) and on any op that isn't itself foldable — so a stale constant can
/// never feed a fold.
fn optimize_ops(ops: &[Op]) -> Vec<Op> {
    use alloc::collections::{BTreeMap, BTreeSet};
    // Basic-block leaders: the targets of every branch / handler install.
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    for op in ops {
        match op {
            Op::Jump { target }
            | Op::JumpIfFalse { target, .. }
            | Op::PushHandler { target, .. } => {
                leaders.insert(*target);
            }
            _ => {}
        }
    }

    let mut consts: BTreeMap<Reg, NanBox> = BTreeMap::new();
    let num = |consts: &BTreeMap<Reg, NanBox>, r: &Reg| consts.get(r).and_then(|v| v.as_number());

    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if leaders.contains(&i) {
            consts.clear();
        }
        // Try to fold; `folded` is the (possibly rewritten) op to emit.
        let folded: Op = match op {
            Op::LoadConst { dst, value } => {
                consts.insert(*dst, *value);
                op.clone()
            }
            Op::Add { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x + y)
            })
            .unwrap_or_else(|| op.clone()),
            Op::Sub { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x - y)
            })
            .unwrap_or_else(|| op.clone()),
            Op::Mul { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x * y)
            })
            .unwrap_or_else(|| op.clone()),
            Op::Div { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x / y)
            })
            .unwrap_or_else(|| op.clone()),
            Op::Mod { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x % y)
            })
            .unwrap_or_else(|| op.clone()),
            Op::Lt { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::boolean(x < y)
            })
            .unwrap_or_else(|| op.clone()),
            // `+` over two number constants is plain addition (no string can be
            // involved — only `LoadConst` primitives are tracked).
            Op::AddValue { dst, a, b } => fold2(*dst, num(&consts, a), num(&consts, b), |x, y| {
                NanBox::number(x + y)
            })
            .unwrap_or_else(|| op.clone()),
            // Loose `==`/`!=` over two number constants is numeric equality
            // (matching the realm's `loose_equals` for two numbers).
            Op::ValueBin { dst, op: vop, a, b } if matches!(*vop, VB_LOOSE_EQ | VB_LOOSE_NEQ) => {
                let eq = matches!(*vop, VB_LOOSE_EQ);
                fold2(*dst, num(&consts, a), num(&consts, b), move |x, y| {
                    NanBox::boolean((x == y) == eq)
                })
                .unwrap_or_else(|| op.clone())
            }
            Op::Neg { dst, a } => match num(&consts, a) {
                Some(x) => Op::LoadConst {
                    dst: *dst,
                    value: NanBox::number(-x),
                },
                None => op.clone(),
            },
            Op::Not { dst, a } => match consts.get(a) {
                Some(v) => Op::LoadConst {
                    dst: *dst,
                    value: NanBox::boolean(!v.to_boolean()),
                },
                None => op.clone(),
            },
            // Anything else ends the constant run for its destination (and we
            // conservatively forget all tracked constants, so nothing stale
            // survives into a fold).
            _ => {
                consts.clear();
                op.clone()
            }
        };
        // A folded arithmetic op now produces a known constant.
        if let Op::LoadConst { dst, value } = &folded {
            consts.insert(*dst, *value);
        }
        out.push(folded);
    }
    out
}

/// Folds a binary op over two known-number operands, producing a `LoadConst`.
fn fold2(dst: Reg, a: Option<f64>, b: Option<f64>, f: impl Fn(f64, f64) -> NanBox) -> Option<Op> {
    match (a, b) {
        (Some(x), Some(y)) => Some(Op::LoadConst {
            dst,
            value: f(x, y),
        }),
        _ => None,
    }
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

    // Routes a `VmError` to the innermost handler (or unwinds): on a `Thrown`,
    // bind the value in the catch register and jump; otherwise propagate.
    macro_rules! handle_throw {
        ($e:expr) => {
            match $e {
                VmError::Thrown(v) => match handlers.pop() {
                    Some((target, reg)) => {
                        regs[reg as usize] = v;
                        pc = target;
                    }
                    None => return Err(VmError::Thrown(v)),
                },
                other => return Err(other),
            }
        };
    }

    while pc < program.len() {
        let op = &program[pc];
        pc += 1;
        match op {
            Op::LoadConst { dst, value } => regs[*dst as usize] = *value,
            Op::Add { dst, a, b } => {
                regs[*dst as usize] =
                    NanBox::number(num(regs[*a as usize])? + num(regs[*b as usize])?);
            }
            // Use the realm's arithmetic (ToNumber on each operand, which applies
            // ToPrimitive to objects) so `[5] - 2` is `3` natively, without an
            // error-driven fall back to the tree-walker. A user `valueOf`/
            // `toString` is honored first via `to_primitive`.
            Op::Sub { dst, a, b } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.sub(x, y);
            }
            Op::Mul { dst, a, b } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.mul(x, y);
            }
            Op::Div { dst, a, b } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.div(x, y);
            }
            Op::Mod { dst, a, b } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.rem(x, y);
            }
            Op::HasProp { dst, key, obj } => {
                let present = match regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    Some(h) => {
                        let k = ctx.realm.to_display_string(regs[*key as usize]);
                        // Own or inherited (walk the prototype chain); arrays also
                        // report in-bounds indices.
                        let mut found = false;
                        let mut cur = Some(h);
                        while let Some(c) = cur {
                            if ctx.realm.has_own(c, &k) {
                                found = true;
                                break;
                            }
                            cur = ctx.realm.object_proto(c);
                        }
                        found
                            || ctx
                                .realm
                                .array_length(h)
                                .is_some_and(|len| k.parse::<usize>().is_ok_and(|i| i < len))
                    }
                    None => false,
                };
                regs[*dst as usize] = NanBox::boolean(present);
            }
            Op::IsBuiltin { dst, obj, kind } => {
                let yes = regs[*obj as usize]
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| match kind {
                        0 => ctx.realm.regexp_at(h).is_some(),
                        1 => ctx.realm.is_array(h),
                        2 => ctx.realm.collection_is_set(h) == Some(false),
                        3 => ctx.realm.collection_is_set(h) == Some(true),
                        _ => false,
                    });
                regs[*dst as usize] = NanBox::boolean(yes);
            }
            Op::DeleteProp { dst, obj, key } => {
                let removed = match regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    Some(h) => {
                        let k = ctx.realm.to_display_string(regs[*key as usize]);
                        ctx.realm.delete_property(h, &k)
                    }
                    None => true,
                };
                regs[*dst as usize] = NanBox::boolean(removed);
            }
            Op::SetClassTag { obj, class_id } => {
                if let Some(h) = regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    ctx.realm.set_class_tag(h, *class_id);
                }
            }
            Op::DefineAccessor {
                obj,
                key,
                getter,
                setter,
            } => {
                if let Some(h) = regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    ctx.realm.define_accessor(
                        h,
                        key,
                        regs[*getter as usize],
                        regs[*setter as usize],
                    );
                }
            }
            Op::InstanceOf { dst, obj, ids } => {
                let yes = regs[*obj as usize]
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| ctx.realm.class_tag(h))
                    .is_some_and(|t| ids.contains(&t));
                regs[*dst as usize] = NanBox::boolean(yes);
            }
            Op::TypeOf { dst, a } => {
                let t = ctx.realm.type_of_value(regs[*a as usize]);
                regs[*dst as usize] = NanBox::handle(ctx.realm.new_string(t).to_raw());
            }
            #[cfg(feature = "std")]
            Op::BitNot { dst, a } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, x, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.bit_not(x);
            }
            #[cfg(not(feature = "std"))]
            Op::BitNot { dst, .. } => regs[*dst as usize] = NanBox::number(f64::NAN),
            Op::ValueBin { dst, op, a, b } => {
                let (x, y) = (regs[*a as usize], regs[*b as usize]);
                regs[*dst as usize] = match *op {
                    VB_LOOSE_EQ | VB_LOOSE_NEQ => {
                        // `obj == primitive` converts the object with ToPrimitive
                        // (toString) before comparing, so e.g. `[] == 0` is true.
                        let (xc, yc) = loose_eq_coerce(ctx.realm, x, y);
                        let r = ctx.realm.loose_equals(xc, yc);
                        NanBox::boolean(if *op == VB_LOOSE_EQ { r } else { !r })
                    }
                    // `**`/bitwise/shifts are numeric: ToPrimitive each operand
                    // (honoring a user `valueOf`) before the realm's `std`-gated math.
                    #[cfg(feature = "std")]
                    _ => {
                        let xn = to_primitive(ctx, funcs, x, true);
                        let yn = to_primitive(ctx, funcs, y, true);
                        if let Some(e) = symbol_coercion_error(ctx.realm, xn, yn, SYM_NUM_ERR) {
                            handle_throw!(VmError::Thrown(e));
                        }
                        match *op {
                            VB_POW => ctx.realm.pow(xn, yn),
                            VB_BIT_AND => ctx.realm.bit_and(xn, yn),
                            VB_BIT_OR => ctx.realm.bit_or(xn, yn),
                            VB_BIT_XOR => ctx.realm.bit_xor(xn, yn),
                            VB_SHL => ctx.realm.shl(xn, yn),
                            VB_SHR => ctx.realm.shr(xn, yn),
                            VB_USHR => ctx.realm.ushr(xn, yn),
                            _ => NanBox::number(f64::NAN),
                        }
                    }
                    #[cfg(not(feature = "std"))]
                    _ => NanBox::number(f64::NAN),
                };
            }
            Op::Neg { dst, a } => {
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, x, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.neg(x);
            }
            Op::Not { dst, a } => {
                regs[*dst as usize] = NanBox::boolean(!ctx.realm.truthy(regs[*a as usize]));
            }
            Op::Move { dst, src } => regs[*dst as usize] = regs[*src as usize],
            Op::Lt { dst, a, b } => {
                // Use the realm's relational comparison so strings and objects
                // (ToPrimitive) work natively instead of erroring into a fallback.
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.less_than(x, y);
            }
            Op::AddValue { dst, a, b } => {
                // `+` uses ToPrimitive (default hint) on each operand — honoring a
                // user `valueOf`/`toString` — then `realm.add` picks string
                // concatenation vs numeric addition from the resulting primitives.
                let x = to_primitive(ctx, funcs, regs[*a as usize], true);
                let y = to_primitive(ctx, funcs, regs[*b as usize], true);
                if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_STR_ERR) {
                    handle_throw!(VmError::Thrown(e));
                }
                regs[*dst as usize] = ctx.realm.add(x, y);
            }
            Op::StrictEq { dst, a, b } => {
                regs[*dst as usize] = NanBox::boolean(
                    ctx.realm
                        .strict_equals(regs[*a as usize], regs[*b as usize]),
                );
            }
            Op::JumpIfFalse { cond, target } => {
                if !ctx.realm.truthy(regs[*cond as usize]) {
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
            Op::NewArrayCtor { dst, arg } => {
                let v = regs[*arg as usize];
                let handle = if let Some(n) = v.as_number() {
                    // A number is the length: must be a non-negative integer that
                    // fits uint32 (and, to avoid OOM in this dense model, a sane cap).
                    if n < 0.0 || n > f64::from(u32::MAX) || n != f64::from(n as u32) {
                        let e = make_error(ctx.realm, "RangeError", "Invalid array length");
                        return Err(VmError::Thrown(e));
                    }
                    if n > 100_000_000.0 {
                        let e = make_error(ctx.realm, "RangeError", "Array length too large");
                        return Err(VmError::Thrown(e));
                    }
                    ctx.realm.new_array(vec![NanBox::undefined(); n as usize])
                } else {
                    ctx.realm.new_array(vec![v])
                };
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
            Op::GetKey { dst, obj, key } => {
                let handle = object_handle(regs[*obj as usize])?;
                let k = regs[*key as usize];
                regs[*dst as usize] = match k.as_number() {
                    Some(n) if ctx.realm.is_array(handle) => {
                        ctx.realm.get_element(handle, n as usize)
                    }
                    _ => {
                        // ToPropertyKey: an object key uses its `toString`.
                        let pk = to_primitive(ctx, funcs, k, false);
                        let ks = ctx.realm.to_display_string(pk);
                        // A canonical numeric string key on an array (`arr["0"]`)
                        // reads the element, like `arr[0]`.
                        if ctx.realm.is_array(handle)
                            && let Ok(i) = ks.parse::<usize>()
                            && alloc::format!("{i}") == ks
                        {
                            ctx.realm.get_element(handle, i)
                        } else if ks == "length"
                            && let Some(len) = ctx.realm.array_length(handle).or_else(|| {
                                ctx.realm.string_value(handle).map(|s| s.chars().count())
                            })
                        {
                            // Computed `arr["length"]` / `str["length"]`.
                            NanBox::number(len as f64)
                        } else {
                            ctx.realm
                                .get_property(handle, &ks)
                                .unwrap_or(NanBox::undefined())
                        }
                    }
                };
            }
            Op::SetKey { obj, key, src } => {
                let handle = object_handle(regs[*obj as usize])?;
                let k = regs[*key as usize];
                match k.as_number() {
                    Some(n) if ctx.realm.is_array(handle) => {
                        ctx.realm
                            .set_element(handle, n as usize, regs[*src as usize]);
                    }
                    _ => {
                        // ToPropertyKey: an object key uses its `toString`.
                        let pk = to_primitive(ctx, funcs, k, false);
                        let ks = ctx.realm.to_display_string(pk);
                        ctx.realm.set_property(handle, &ks, regs[*src as usize]);
                    }
                }
            }
            Op::EnumKeys { dst, obj } => {
                let h = object_handle(regs[*obj as usize])?;
                let mut seen = alloc::collections::BTreeSet::new();
                let mut out = Vec::new();
                // An array leads with its integer indices (a VM closure's backing
                // cells are not enumerable).
                if !ctx.realm.is_vm_function(h)
                    && let Some(len) = ctx.realm.array_length(h)
                {
                    for i in 0..len {
                        let k = alloc::format!("{i}");
                        if seen.insert(k.clone()) {
                            out.push(NanBox::handle(ctx.realm.new_string(&k).to_raw()));
                        }
                    }
                }
                // Own enumerable keys (named props live in the cell for objects, in
                // the auxiliary object for arrays/functions), then inherited.
                let mut cur = Some(h);
                while let Some(c) = cur {
                    let named = ctx
                        .realm
                        .object_keys(c)
                        .unwrap_or_else(|| ctx.realm.aux_named_keys(c));
                    for k in named {
                        if seen.insert(k.clone()) {
                            out.push(NanBox::handle(ctx.realm.new_string(&k).to_raw()));
                        }
                    }
                    cur = ctx.realm.object_proto(c);
                }
                let keys = out;
                regs[*dst as usize] = NanBox::handle(ctx.realm.new_array(keys).to_raw());
            }
            Op::ObjectSpread { dst, src } => {
                let target = object_handle(regs[*dst as usize])?;
                if let Some(sh) = regs[*src as usize].as_handle().map(Handle::from_raw) {
                    if let Some(elems) = ctx.realm.array_elements(sh).map(<[_]>::to_vec) {
                        for (i, e) in elems.into_iter().enumerate() {
                            ctx.realm.set_property(target, &alloc::format!("{i}"), e);
                        }
                    } else {
                        for k in ctx.realm.object_keys(sh).unwrap_or_default() {
                            let v = ctx
                                .realm
                                .get_property(sh, &k)
                                .unwrap_or(NanBox::undefined());
                            ctx.realm.set_property(target, &k, v);
                        }
                        // Accessor (getter) properties are enumerable too.
                        let recv = regs[*src as usize];
                        for k in ctx.realm.object_accessor_keys(sh) {
                            if let Some((getter, _)) = ctx.realm.accessor(sh, &k)
                                && getter.as_handle().is_some()
                            {
                                let v = call_closure(ctx, funcs, getter, &[], recv)?;
                                ctx.realm.set_property(target, &k, v);
                            }
                        }
                    }
                }
            }
            Op::ArrayLen { dst, arr } => {
                let handle = object_handle(regs[*arr as usize])?;
                // A VM function (a tagged closure array) reports its parameter
                // count from the proto, not the backing array's length.
                if ctx.realm.is_vm_function(handle) {
                    let n = ctx
                        .realm
                        .get_element(handle, 0)
                        .as_number()
                        .and_then(|f| funcs.get(f as usize))
                        .map_or(0, |p| p.length);
                    regs[*dst as usize] = NanBox::number(n as f64);
                } else {
                    // `.length` on an array, or a string's character count.
                    let len = ctx
                        .realm
                        .array_length(handle)
                        .or_else(|| ctx.realm.string_value(handle).map(|s| s.chars().count()));
                    regs[*dst as usize] = match len {
                        Some(n) => NanBox::number(n as f64),
                        // Otherwise an explicit `length` data property (e.g. a regex
                        // match result, which is object-shaped here), else undefined.
                        None => ctx
                            .realm
                            .get_property(handle, "length")
                            .unwrap_or(NanBox::undefined()),
                    };
                }
            }
            Op::CollectionSize { dst, recv } => {
                let h = object_handle(regs[*recv as usize])?;
                let n = ctx
                    .realm
                    .collection_size(h)
                    .or_else(|| {
                        ctx.realm
                            .get_property(h, "size")
                            .and_then(|v| v.as_number().map(|n| n as usize))
                    })
                    .unwrap_or(0);
                regs[*dst as usize] = NanBox::number(n as f64);
            }
            Op::ArrayPush { arr, src } => {
                let handle = object_handle(regs[*arr as usize])?;
                let len = ctx.realm.array_length(handle).unwrap_or(0);
                ctx.realm.set_element(handle, len, regs[*src as usize]);
            }
            Op::ArrayExtend { arr, src } => {
                let handle = object_handle(regs[*arr as usize])?;
                let srch = object_handle(regs[*src as usize])?;
                let elems = ctx
                    .realm
                    .array_elements(srch)
                    .map(<[_]>::to_vec)
                    .ok_or(VmError::NotAnObject)?;
                let start = ctx.realm.array_length(handle).unwrap_or(0);
                for (i, e) in elems.into_iter().enumerate() {
                    ctx.realm.set_element(handle, start + i, e);
                }
            }
            Op::ObjectRest { dst, src, exclude } => {
                let srch = object_handle(regs[*src as usize])?;
                let new_obj = ctx.realm.new_object();
                for k in ctx.realm.object_keys(srch).unwrap_or_default() {
                    if !exclude.contains(&k) {
                        let v = ctx
                            .realm
                            .get_property(srch, &k)
                            .unwrap_or(NanBox::undefined());
                        ctx.realm.set_property(new_obj, &k, v);
                    }
                }
                regs[*dst as usize] = NanBox::handle(new_obj.to_raw());
            }
            Op::NewCollection { dst, is_set, seed } => {
                let coll = ctx.realm.new_collection(*is_set);
                if let Some(seed) = seed
                    && let Some(sh) = regs[*seed as usize].as_handle().map(Handle::from_raw)
                {
                    let items = ctx
                        .realm
                        .array_elements(sh)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default();
                    for item in items {
                        if *is_set {
                            ctx.realm.collection_set(coll, item, item);
                        } else if let Some(ih) = item.as_handle().map(Handle::from_raw) {
                            // A `[key, value]` pair.
                            let k = ctx.realm.get_element(ih, 0);
                            let v = ctx.realm.get_element(ih, 1);
                            ctx.realm.collection_set(coll, k, v);
                        }
                    }
                }
                regs[*dst as usize] = NanBox::handle(coll.to_raw());
            }
            Op::ArraySliceFrom { dst, src, from } => {
                let srch = object_handle(regs[*src as usize])?;
                let from = num(regs[*from as usize])? as usize;
                let rest: Vec<NanBox> = ctx
                    .realm
                    .array_elements(srch)
                    .map(|e| e.get(from..).map(<[_]>::to_vec).unwrap_or_default())
                    .unwrap_or_default();
                regs[*dst as usize] = NanBox::handle(ctx.realm.new_array(rest).to_raw());
            }
            Op::NewRegExp { dst, source, flags } => {
                let h = ctx.realm.new_regexp(source, flags);
                regs[*dst as usize] = NanBox::handle(h.to_raw());
            }
            Op::NewObject { dst } => {
                let handle = ctx.realm.new_object();
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::SetProp { obj, key, src } => {
                let handle = object_handle(regs[*obj as usize])?;
                let recv = regs[*obj as usize];
                // `regex.lastIndex = n` updates the stateful search position.
                if key.as_str() == "lastIndex" && ctx.realm.regexp_at(handle).is_some() {
                    let n = num(regs[*src as usize]).unwrap_or(0.0).max(0.0) as usize;
                    ctx.realm.set_regex_last_index(handle, n);
                    continue;
                }
                // A setter accessor takes precedence over a data slot.
                match ctx.realm.accessor(handle, key) {
                    Some((_, setter)) if setter.as_handle().is_some() => {
                        if let Err(e) =
                            call_closure(ctx, funcs, setter, &[regs[*src as usize]], recv)
                        {
                            handle_throw!(e);
                        }
                    }
                    _ => {
                        ctx.realm.set_property(handle, key, regs[*src as usize]);
                    }
                }
            }
            Op::GetProp { dst, obj, key } => {
                let recv = regs[*obj as usize];
                match recv.as_handle().map(Handle::from_raw) {
                    None => {
                        use crate::nanbox::Unpacked;
                        match recv.unpack() {
                            // Property access on null/undefined throws a TypeError.
                            Unpacked::Null | Unpacked::Undefined => {
                                let what = if matches!(recv.unpack(), Unpacked::Null) {
                                    "null"
                                } else {
                                    "undefined"
                                };
                                let e = make_error(
                                    ctx.realm,
                                    "TypeError",
                                    &alloc::format!(
                                        "Cannot read properties of {what} (reading '{key}')"
                                    ),
                                );
                                handle_throw!(VmError::Thrown(e));
                            }
                            // Other primitives: a missing property reads `undefined`.
                            _ => regs[*dst as usize] = NanBox::undefined(),
                        }
                    }
                    Some(handle) => {
                        // A VM function's `.name` comes from its proto (the closure is a
                        // tagged array whose element 0 is the function id).
                        if key.as_str() == "name"
                            && ctx.realm.is_vm_function(handle)
                            && !ctx.realm.has_own(handle, "name")
                        {
                            let nm = ctx
                                .realm
                                .get_element(handle, 0)
                                .as_number()
                                .and_then(|f| funcs.get(f as usize))
                                .map_or("", |p| p.name.as_str());
                            let s = ctx.realm.new_string(nm);
                            regs[*dst as usize] = NanBox::handle(s.to_raw());
                            continue;
                        }
                        // RegExp introspection properties.
                        let mut done = true;
                        if let Some((src, flags)) = ctx.realm.regexp_at(handle) {
                            match key.as_str() {
                                "source" => {
                                    let s = ctx.realm.new_string(&src);
                                    regs[*dst as usize] = NanBox::handle(s.to_raw());
                                }
                                "flags" => {
                                    let s = ctx.realm.new_string(&flags);
                                    regs[*dst as usize] = NanBox::handle(s.to_raw());
                                }
                                "global" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('g'))
                                }
                                "ignoreCase" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('i'));
                                }
                                "multiline" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('m'));
                                }
                                "sticky" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('y'))
                                }
                                "dotAll" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('s'))
                                }
                                "unicode" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('u'))
                                }
                                "hasIndices" => {
                                    regs[*dst as usize] = NanBox::boolean(flags.contains('d'))
                                }
                                "lastIndex" => {
                                    regs[*dst as usize] =
                                        NanBox::number(ctx.realm.regex_last_index(handle) as f64);
                                }
                                _ => done = false,
                            }
                        } else {
                            done = false;
                        }
                        if !done {
                            // A getter accessor takes precedence over a data slot.
                            match ctx.realm.accessor(handle, key) {
                                Some((getter, _)) if getter.as_handle().is_some() => {
                                    match call_closure(ctx, funcs, getter, &[], recv) {
                                        Ok(v) => regs[*dst as usize] = v,
                                        Err(e) => handle_throw!(e),
                                    }
                                }
                                _ => {
                                    regs[*dst as usize] = ctx
                                        .realm
                                        .get_property(handle, key)
                                        .unwrap_or(NanBox::undefined());
                                }
                            }
                        }
                    }
                }
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
                // function-table index (as a number), tagged `\0vmfn` so `typeof`
                // and friends see a function rather than the backing array.
                let handle = ctx
                    .realm
                    .new_array(alloc::vec![NanBox::number(*func as f64)]);
                ctx.realm
                    .set_hidden_property(handle, "\u{0}vmfn", NanBox::boolean(true));
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::MakeClosure {
                dst,
                func,
                captures,
            } => {
                // `[func_id, cell0, cell1, …]`, tagged so `typeof` and friends see
                // a function rather than the backing array.
                let mut elems = alloc::vec![NanBox::number(*func as f64)];
                elems.extend(captures.iter().map(|r| regs[*r as usize]));
                let handle = ctx.realm.new_array(elems);
                ctx.realm
                    .set_hidden_property(handle, "\u{0}vmfn", NanBox::boolean(true));
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
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                // A user method is a closure property on an object; otherwise try
                // a built-in `Array`/`String` method on the fast path.
                let user_method = recv_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| ctx.realm.get_property(h, key))
                    .filter(|p| p.as_handle().is_some());
                let outcome = match user_method {
                    Some(closure) => call_closure(ctx, funcs, closure, &argv, recv_val),
                    None => match builtin_method(ctx, funcs, recv_val, key, &argv) {
                        Some(r) => r,
                        // Unknown method → fall back to the tree-walker.
                        None => return Err(VmError::NotAnObject),
                    },
                };
                match outcome {
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
            Op::CallValueThis {
                dst,
                callee,
                recv,
                args,
            } => {
                let closure = regs[*callee as usize];
                let recv_val = regs[*recv as usize];
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                match call_closure(ctx, funcs, closure, &argv, recv_val) {
                    Ok(ret) => regs[*dst as usize] = ret,
                    Err(e) => handle_throw!(e),
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
                // `JSON.stringify` needs interpreter access (toJSON / getters / a
                // function or array replacer), so it is handled here where `funcs`
                // and the throw machinery are available rather than in `call_native`.
                if *native == NB_JSON_STRINGIFY {
                    match json_stringify(ctx, funcs, &argv) {
                        Ok(v) => regs[*dst as usize] = v,
                        Err(e) => handle_throw!(e),
                    }
                } else if *native == NB_JSON_PARSE {
                    match json_parse(ctx, funcs, &argv) {
                        Ok(v) => regs[*dst as usize] = v,
                        Err(e) => handle_throw!(e),
                    }
                } else {
                    regs[*dst as usize] = call_native(ctx, *native, &argv);
                }
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

/// Interpreter-aware `JSON.stringify(value, replacer?, space?)`: normalizes the
/// value tree (honoring `toJSON`, getters, and the replacer) then serializes it.
fn json_stringify(ctx: &mut Ctx, funcs: &[FnProto], args: &[NanBox]) -> Result<NanBox, VmError> {
    let v = args.first().copied().unwrap_or(NanBox::undefined());
    // Optional `space` (arg 2): number → spaces, string → that string (cap 10).
    let space = args.get(2).copied().unwrap_or(NanBox::undefined());
    let indent: alloc::string::String = if let Some(n) = space.as_number() {
        " ".repeat((n.max(0.0) as usize).min(10))
    } else if let Some(s) = space
        .as_handle()
        .and_then(|r| ctx.realm.string_value(Handle::from_raw(r)))
    {
        s.chars().take(10).collect()
    } else {
        alloc::string::String::new()
    };
    // Optional `replacer` (arg 1): a function transforms each value, an array
    // allowlists object keys.
    let replacer = args.get(1).copied().unwrap_or(NanBox::undefined());
    let (repl_fn, allow): (Option<NanBox>, Option<Vec<alloc::string::String>>) =
        match replacer.as_handle().map(Handle::from_raw) {
            Some(rh) if ctx.realm.is_vm_function(rh) => (Some(replacer), None),
            Some(rh) if ctx.realm.is_array(rh) => {
                let a = ctx
                    .realm
                    .array_elements(rh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default()
                    .iter()
                    .map(|e| ctx.realm.to_display_string(*e))
                    .collect();
                (None, Some(a))
            }
            _ => (None, None),
        };
    let holder = ctx.realm.new_object();
    ctx.realm.set_property(holder, "", v);
    let mut seen = Vec::new();
    let v = json_normalize(
        ctx,
        funcs,
        NanBox::handle(holder.to_raw()),
        "",
        v,
        repl_fn,
        allow.as_deref(),
        &mut seen,
    )?;
    let result = if indent.is_empty() {
        crate::json::stringify(ctx.realm, v)
    } else {
        crate::json::stringify_pretty(ctx.realm, v, &indent)
    };
    Ok(match result {
        Some(s) => NanBox::handle(ctx.realm.new_string(&s).to_raw()),
        None => NanBox::undefined(),
    })
}

/// Interpreter-aware `JSON.parse(text, reviver?)`: parses (throwing a `SyntaxError`
/// on malformed input) then, with a function reviver, walks the result bottom-up
/// transforming each value.
fn json_parse(ctx: &mut Ctx, funcs: &[FnProto], args: &[NanBox]) -> Result<NanBox, VmError> {
    let s = ctx
        .realm
        .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
    let value = match crate::json::parse(ctx.realm, &s) {
        Ok(v) => v,
        Err(msg) => return Err(VmError::Thrown(make_error(ctx.realm, "SyntaxError", &msg))),
    };
    let reviver = args.get(1).copied().unwrap_or(NanBox::undefined());
    if reviver
        .as_handle()
        .map(Handle::from_raw)
        .is_some_and(|r| ctx.realm.is_vm_function(r))
    {
        let holder = ctx.realm.new_object();
        ctx.realm.set_property(holder, "", value);
        return json_revive(ctx, funcs, holder, "", reviver);
    }
    Ok(value)
}

/// `JSON.parse` reviver walk (spec `InternalizeJSONProperty`): recurse into a value's
/// children first, then call `reviver.call(holder, key, value)`. An `undefined` result
/// deletes the (object) property.
fn json_revive(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    holder: Handle,
    key: &str,
    reviver: NanBox,
) -> Result<NanBox, VmError> {
    let value = if ctx.realm.is_array(holder)
        && let Ok(i) = key.parse::<usize>()
    {
        ctx.realm.get_element(holder, i)
    } else {
        ctx.realm
            .get_property(holder, key)
            .unwrap_or(NanBox::undefined())
    };
    if let Some(vh) = value.as_handle().map(Handle::from_raw) {
        if ctx.realm.is_array(vh) {
            let len = ctx.realm.array_length(vh).unwrap_or(0);
            for i in 0..len {
                let ks = alloc::format!("{i}");
                let nv = json_revive(ctx, funcs, vh, &ks, reviver)?;
                ctx.realm.set_element(vh, i, nv);
            }
        } else if let Some(keys) = ctx.realm.object_keys(vh) {
            for k in keys {
                let nv = json_revive(ctx, funcs, vh, &k, reviver)?;
                if matches!(nv.unpack(), crate::nanbox::Unpacked::Undefined) {
                    ctx.realm.delete_property(vh, &k);
                } else {
                    ctx.realm.set_property(vh, &k, nv);
                }
            }
        }
    }
    let kb = NanBox::handle(ctx.realm.new_string(key).to_raw());
    call_closure(
        ctx,
        funcs,
        reviver,
        &[kb, value],
        NanBox::handle(holder.to_raw()),
    )
}

/// Reads property `key` from `h`, invoking a getter accessor (a closure) so
/// `JSON.stringify` observes computed values; a data property is read directly.
fn json_read_prop(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    h: Handle,
    key: &str,
) -> Result<NanBox, VmError> {
    if let Some((getter, _)) = ctx.realm.accessor(h, key) {
        if getter
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|r| ctx.realm.is_vm_function(r))
        {
            return call_closure(ctx, funcs, getter, &[], NanBox::handle(h.to_raw()));
        }
        return Ok(NanBox::undefined());
    }
    Ok(ctx
        .realm
        .get_property(h, key)
        .unwrap_or(NanBox::undefined()))
}

/// Interpreter-aware `JSON.stringify` pre-pass: applies `toJSON`, invokes getters,
/// and runs the `replacer` (a function, or `allow` key allowlist), producing a plain
/// value tree that `crate::json::stringify` can serialize. Mirrors the tree-walker's
/// `json_to_string_seen` + `json_apply_replacer`. A cycle throws a `TypeError`.
#[allow(clippy::too_many_arguments)]
fn json_normalize(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    holder: NanBox,
    key: &str,
    value: NanBox,
    replacer: Option<NanBox>,
    allow: Option<&[alloc::string::String]>,
    seen: &mut Vec<Handle>,
) -> Result<NanBox, VmError> {
    let mut v = value;
    // `toJSON(key)` replaces the value (real objects only — not strings/bigints/Dates,
    // whose serialization the realm handles).
    if let Some(h) = v.as_handle().map(Handle::from_raw)
        && ctx.realm.string_value(h).is_none()
        && ctx.realm.bigint_at(h).is_none()
        && ctx.realm.date_at(h).is_none()
    {
        let tj = json_read_prop(ctx, funcs, h, "toJSON")?;
        if tj
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|r| ctx.realm.is_vm_function(r))
        {
            let kb = NanBox::handle(ctx.realm.new_string(key).to_raw());
            v = call_closure(ctx, funcs, tj, &[kb], v)?;
        }
    }
    // The replacer function transforms the value.
    if let Some(rf) = replacer {
        let kb = NanBox::handle(ctx.realm.new_string(key).to_raw());
        v = call_closure(ctx, funcs, rf, &[kb, v], holder)?;
    }
    // Recurse into plain arrays/objects, rebuilding a normalized copy. Closures,
    // strings, and Dates pass through to the serializer unchanged.
    if let Some(h) = v.as_handle().map(Handle::from_raw)
        && ctx.realm.string_value(h).is_none()
        && !ctx.realm.is_vm_function(h)
        && ctx.realm.date_at(h).is_none()
    {
        if let Some(elems) = ctx.realm.array_elements(h).map(<[_]>::to_vec) {
            if seen.contains(&h) {
                return Err(VmError::Thrown(make_error(
                    ctx.realm,
                    "TypeError",
                    "Converting circular structure to JSON",
                )));
            }
            seen.push(h);
            let mut out = Vec::with_capacity(elems.len());
            for (i, e) in elems.into_iter().enumerate() {
                let kk = alloc::format!("{i}");
                out.push(json_normalize(
                    ctx, funcs, v, &kk, e, replacer, allow, seen,
                )?);
            }
            seen.pop();
            return Ok(NanBox::handle(ctx.realm.new_array(out).to_raw()));
        }
        if let Some(keys) = ctx.realm.object_keys(h) {
            if seen.contains(&h) {
                return Err(VmError::Thrown(make_error(
                    ctx.realm,
                    "TypeError",
                    "Converting circular structure to JSON",
                )));
            }
            seen.push(h);
            // With an array replacer the keys come from the allowlist, in allowlist
            // order (deduplicated, own keys only); otherwise the object's own order.
            let key_list: Vec<alloc::string::String> = match allow {
                Some(a) => {
                    let mut ks: Vec<alloc::string::String> = Vec::new();
                    for k in a {
                        if keys.contains(k) && !ks.contains(k) {
                            ks.push(k.clone());
                        }
                    }
                    ks
                }
                None => keys,
            };
            let new_obj = ctx.realm.new_object();
            for k in key_list {
                let pv = json_read_prop(ctx, funcs, h, &k)?;
                let nv = json_normalize(ctx, funcs, v, &k, pv, replacer, allow, seen)?;
                ctx.realm.set_property(new_obj, &k, nv);
            }
            seen.pop();
            return Ok(NanBox::handle(new_obj.to_raw()));
        }
    }
    Ok(v)
}

/// Invokes a closure value (`[func_id, cell…]`) with `args` and `this_val`.
fn call_closure(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    closure: NanBox,
    args: &[NanBox],
    this_val: NanBox,
) -> Result<NanBox, VmError> {
    let fh = closure
        .as_handle()
        .map(Handle::from_raw)
        .ok_or(VmError::NotAnObject)?;
    let id = ctx
        .realm
        .get_element(fh, 0)
        .as_number()
        .ok_or(VmError::NotAnObject)? as usize;
    let n_caps = ctx.realm.array_length(fh).unwrap_or(1).saturating_sub(1);
    let caps: Vec<NanBox> = (0..n_caps)
        .map(|i| ctx.realm.get_element(fh, i + 1))
        .collect();
    call_with(ctx, funcs, id, args, &caps, this_val)
}

/// Dispatches a built-in `Array`/`String` instance method on the fast path.
/// Returns `None` when `key` isn't a recognized method (the caller then routes
/// the program to the tree-walker).
/// Slices `s` by *character* (scalar) indices `[st, en)` — the regex engine's
/// index space — so a multi-byte character never splits a byte boundary.
#[cfg(feature = "regex")]
fn char_substr(s: &str, st: usize, en: usize) -> String {
    s.chars().skip(st).take(en.saturating_sub(st)).collect()
}

/// Slices `s` from character index `st` to the end.
#[cfg(feature = "regex")]
fn char_substr_from(s: &str, st: usize) -> String {
    s.chars().skip(st).collect()
}

/// Builds a regex match result object `{ 0: whole, 1: g1, …, index, input,
/// groups, length }` (the shape `RegExp.exec` / `String.match` return).
#[cfg(feature = "regex")]
fn regex_match_object(
    realm: &mut Realm,
    text: &str,
    caps: &crate::regex::Captures,
    group_names: &[(usize, alloc::string::String)],
) -> NanBox {
    // A match result is a real Array: element `i` is capture group `i` (the whole
    // match at 0), so `Array.isArray`, `JSON.stringify`, `.length`, and the array
    // methods behave; `index`/`input`/`groups` are enumerable named own properties
    // (kept in the array's auxiliary object).
    let elems: Vec<NanBox> = caps
        .groups
        .iter()
        .map(|g| match g {
            Some((s, e)) => NanBox::handle(realm.new_string(&char_substr(text, *s, *e)).to_raw()),
            None => NanBox::undefined(),
        })
        .collect();
    let obj = realm.new_array(elems);
    let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
    realm.set_property(obj, "index", NanBox::number(index as f64));
    let input = NanBox::handle(realm.new_string(text).to_raw());
    realm.set_property(obj, "input", input);
    // `.groups`: an object of named captures (or `undefined` if none).
    let groups = if group_names.is_empty() {
        NanBox::undefined()
    } else {
        let g = realm.new_object();
        for (idx, name) in group_names {
            let v = match caps.groups.get(*idx).and_then(|x| *x) {
                Some((s, e)) => NanBox::handle(realm.new_string(&char_substr(text, s, e)).to_raw()),
                None => NanBox::undefined(),
            };
            realm.set_property(g, name, v);
        }
        NanBox::handle(g.to_raw())
    };
    realm.set_property(obj, "groups", groups);
    NanBox::handle(obj.to_raw())
}

/// Dispatches `RegExp` instance methods (`test`/`exec`) and the regex-backed
/// `String` methods (`match`/`replace`/`replaceAll`/`split`/`search` when given
/// a `RegExp`). Returns `None` if `key`/`recv` aren't a regex operation.
#[cfg(feature = "regex")]
fn regex_method(
    ctx: &mut Ctx,
    recv: NanBox,
    key: &str,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    use crate::regex::Regex;
    let h = recv.as_handle().map(Handle::from_raw)?;
    let arg0 = args.first().copied().unwrap_or(NanBox::undefined());

    // `re.test(s)` / `re.exec(s)`.
    if let Some((source, flags)) = ctx.realm.regexp_at(h) {
        if !matches!(key, "test" | "exec") {
            return None;
        }
        let text = ctx.realm.to_display_string(arg0);
        let Ok(re) = Regex::new(&source, &flags) else {
            return Some(Ok(NanBox::null()));
        };
        // `g`/`y` regexes resume at `lastIndex` and update it (reset to 0 on miss).
        let stateful = flags.contains('g') || flags.contains('y');
        let start = if stateful {
            ctx.realm.regex_last_index(h)
        } else {
            0
        };
        let caps = re.captures_from(&text, start);
        if stateful {
            let next = caps.as_ref().map_or(0, |c| c.whole().1);
            ctx.realm.set_regex_last_index(h, next);
        }
        return Some(Ok(match (key, caps) {
            ("test", c) => NanBox::boolean(c.is_some()),
            (_, Some(caps)) => regex_match_object(ctx.realm, &text, &caps, re.group_names()),
            (_, None) => NanBox::null(),
        }));
    }

    // `str.match/replace/replaceAll/split/search(re)` — only when the argument
    // is a RegExp (string-argument forms stay in `builtin_method`).
    let text = ctx.realm.string_value(h)?;
    if !matches!(key, "match" | "replace" | "replaceAll" | "split" | "search") {
        return None;
    }
    let (src, flags) = arg0
        .as_handle()
        .map(Handle::from_raw)
        .and_then(|rh| ctx.realm.regexp_at(rh))?;
    let Ok(re) = Regex::new(&src, &flags) else {
        return Some(Ok(NanBox::null()));
    };
    let global = flags.contains('g');
    // `replaceAll` requires a global RegExp.
    if !global && key == "replaceAll" {
        return Some(Err(VmError::Thrown(make_error(
            ctx.realm,
            "TypeError",
            "replaceAll must be called with a global RegExp",
        ))));
    }
    let result = match key {
        "search" => {
            let i = re
                .find_from(&text, 0)
                .map_or(-1.0, |(s, _)| text[..s].chars().count() as f64);
            NanBox::number(i)
        }
        "match" if !global => match re.captures_from(&text, 0) {
            Some(caps) => regex_match_object(ctx.realm, &text, &caps, re.group_names()),
            None => NanBox::null(),
        },
        "match" => {
            // Global match → an array of the whole matches.
            let mut out = Vec::new();
            let mut pos = 0;
            while let Some((s, e)) = re.find_from(&text, pos) {
                out.push(NanBox::handle(
                    ctx.realm.new_string(&char_substr(&text, s, e)).to_raw(),
                ));
                pos = if e > s { e } else { e + 1 };
            }
            if out.is_empty() {
                NanBox::null()
            } else {
                NanBox::handle(ctx.realm.new_array(out).to_raw())
            }
        }
        "replace" | "replaceAll" => {
            let repl_val = args.get(1).copied().unwrap_or(NanBox::undefined());
            // A non-string replacement (a function/closure, called per match) is
            // handled by the tree-walker; defer instead of stringifying it.
            // (nbvm closures are arrays, so this can't use `function_at`.)
            if repl_val
                .as_handle()
                .is_some_and(|raw| ctx.realm.string_value(Handle::from_raw(raw)).is_none())
            {
                return None;
            }
            let repl = ctx.realm.to_display_string(repl_val);
            NanBox::handle(ctx.realm.new_string(&re.replace(&text, &repl)).to_raw())
        }
        // "split" — splices capture groups and handles zero-width matches (kept
        // in sync with the tree-walker's `split`).
        _ => {
            // An optional limit caps the segment count (args[1]).
            let limit = match args.get(1) {
                Some(a) if !matches!(a.unpack(), crate::nanbox::Unpacked::Undefined) => {
                    let n = ctx.realm.to_number(*a);
                    if n >= 0.0 { Some(n as usize) } else { None }
                }
                _ => None,
            };
            let mut out = Vec::new();
            let mut seg_start = 0;
            let mut search = 0;
            // Match positions are `< len` (spec `q < size`); the tail is appended once.
            while search < text.len() && limit.is_none_or(|l| out.len() < l) {
                let Some(caps) = re.captures_from(&text, search) else {
                    break;
                };
                let Some((st, en)) = caps.groups[0] else {
                    break;
                };
                if en == seg_start {
                    if text.chars().nth(search).is_some() {
                        search = search.max(st) + 1;
                        continue;
                    }
                    break;
                }
                out.push(NanBox::handle(
                    ctx.realm
                        .new_string(&char_substr(&text, seg_start, st))
                        .to_raw(),
                ));
                for g in &caps.groups[1..] {
                    out.push(match g {
                        Some((gs, ge)) => NanBox::handle(
                            ctx.realm.new_string(&char_substr(&text, *gs, *ge)).to_raw(),
                        ),
                        None => NanBox::undefined(),
                    });
                }
                seg_start = en;
                search = if en > st { en } else { en + 1 };
            }
            if limit.is_none_or(|l| out.len() < l) {
                out.push(NanBox::handle(
                    ctx.realm
                        .new_string(&char_substr_from(&text, seg_start))
                        .to_raw(),
                ));
            }
            if let Some(l) = limit {
                out.truncate(l);
            }
            NanBox::handle(ctx.realm.new_array(out).to_raw())
        }
    };
    Some(Ok(result))
}

fn builtin_method(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    use crate::nanbox::Unpacked;
    let h = recv.as_handle().map(Handle::from_raw)?;
    let arg0 = || args.first().copied().unwrap_or(NanBox::undefined());

    // --- RegExp / regex-backed String methods (when the `regex` feature is on) ---
    #[cfg(feature = "regex")]
    if let Some(r) = regex_method(ctx, recv, key, args) {
        return Some(r);
    }

    // --- array methods ---
    if ctx.realm.is_array(h) {
        let elems = |ctx: &Ctx| {
            ctx.realm
                .array_elements(h)
                .map(<[_]>::to_vec)
                .unwrap_or_default()
        };
        let result = match key {
            "push" => {
                let mut len = ctx.realm.array_length(h).unwrap_or(0);
                for a in args {
                    ctx.realm.set_element(h, len, *a);
                    len += 1;
                }
                NanBox::number(len as f64)
            }
            "pop" => ctx.realm.array_pop(h),
            "join" => {
                let sep = if matches!(arg0().unpack(), Unpacked::Undefined) {
                    String::from(",")
                } else {
                    ctx.realm.to_display_string(arg0())
                };
                let parts: Vec<String> = elems(ctx)
                    .iter()
                    .map(|e| match e.unpack() {
                        Unpacked::Undefined | Unpacked::Null => String::new(),
                        // A direct self-reference renders empty (no recursion).
                        Unpacked::Handle(raw) if raw == h.to_raw() => String::new(),
                        _ => ctx.realm.to_display_string(*e),
                    })
                    .collect();
                NanBox::handle(ctx.realm.new_string(&parts.join(&sep)).to_raw())
            }
            "includes" => {
                let t = arg0();
                // SameValueZero: like `===` but `NaN` matches `NaN`.
                let t_nan = t.as_number().is_some_and(f64::is_nan);
                NanBox::boolean(elems(ctx).iter().any(|e| {
                    ctx.realm.strict_equals(*e, t)
                        || (t_nan && e.as_number().is_some_and(f64::is_nan))
                }))
            }
            "indexOf" => {
                let t = arg0();
                let i = elems(ctx)
                    .iter()
                    .position(|e| ctx.realm.strict_equals(*e, t));
                NanBox::number(i.map_or(-1.0, |i| i as f64))
            }
            "map" => {
                let f = arg0();
                let mut out = Vec::new();
                for (i, e) in elems(ctx).iter().enumerate() {
                    match call_closure(
                        ctx,
                        funcs,
                        f,
                        &[*e, NanBox::number(i as f64)],
                        NanBox::undefined(),
                    ) {
                        Ok(v) => out.push(v),
                        Err(e) => return Some(Err(e)),
                    }
                }
                NanBox::handle(ctx.realm.new_array(out).to_raw())
            }
            "filter" => {
                let f = arg0();
                let mut out = Vec::new();
                for (i, e) in elems(ctx).iter().enumerate() {
                    match call_closure(
                        ctx,
                        funcs,
                        f,
                        &[*e, NanBox::number(i as f64)],
                        NanBox::undefined(),
                    ) {
                        Ok(v) if ctx.realm.truthy(v) => out.push(*e),
                        Ok(_) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }
                NanBox::handle(ctx.realm.new_array(out).to_raw())
            }
            "forEach" => {
                let f = arg0();
                for (i, e) in elems(ctx).iter().enumerate() {
                    if let Err(e) = call_closure(
                        ctx,
                        funcs,
                        f,
                        &[*e, NanBox::number(i as f64)],
                        NanBox::undefined(),
                    ) {
                        return Some(Err(e));
                    }
                }
                NanBox::undefined()
            }
            "reduce" => {
                let f = arg0();
                let mut acc = args.get(1).copied();
                // Empty array with no seed is a TypeError.
                if acc.is_none() && elems(ctx).is_empty() {
                    let e = make_error(
                        ctx.realm,
                        "TypeError",
                        "Reduce of empty array with no initial value",
                    );
                    return Some(Err(VmError::Thrown(e)));
                }
                for (i, e) in elems(ctx).iter().enumerate() {
                    match acc {
                        None => acc = Some(*e), // first element seeds the accumulator
                        Some(a) => match call_closure(
                            ctx,
                            funcs,
                            f,
                            &[a, *e, NanBox::number(i as f64)],
                            NanBox::undefined(),
                        ) {
                            Ok(v) => acc = Some(v),
                            Err(e) => return Some(Err(e)),
                        },
                    }
                }
                acc.unwrap_or(NanBox::undefined())
            }
            "find" => {
                let f = arg0();
                let mut found = NanBox::undefined();
                for (i, e) in elems(ctx).iter().enumerate() {
                    match call_closure(
                        ctx,
                        funcs,
                        f,
                        &[*e, NanBox::number(i as f64)],
                        NanBox::undefined(),
                    ) {
                        Ok(v) if ctx.realm.truthy(v) => {
                            found = *e;
                            break;
                        }
                        Ok(_) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }
                found
            }
            "some" | "every" => {
                let want_all = key == "every";
                let f = arg0();
                let mut acc = want_all;
                for (i, e) in elems(ctx).iter().enumerate() {
                    let ok = match call_closure(
                        ctx,
                        funcs,
                        f,
                        &[*e, NanBox::number(i as f64)],
                        NanBox::undefined(),
                    ) {
                        Ok(v) => ctx.realm.truthy(v),
                        Err(e) => return Some(Err(e)),
                    };
                    if want_all && !ok {
                        acc = false;
                        break;
                    }
                    if !want_all && ok {
                        acc = true;
                        break;
                    }
                }
                NanBox::boolean(acc)
            }
            "concat" => {
                let mut out = elems(ctx);
                for a in args {
                    match a.as_handle().map(Handle::from_raw) {
                        Some(ah) if ctx.realm.is_array(ah) => out.extend(
                            ctx.realm
                                .array_elements(ah)
                                .map(<[_]>::to_vec)
                                .unwrap_or_default(),
                        ),
                        _ => out.push(*a),
                    }
                }
                NanBox::handle(ctx.realm.new_array(out).to_raw())
            }
            "reverse" => {
                let mut out = elems(ctx);
                out.reverse();
                NanBox::handle(ctx.realm.new_array(out).to_raw())
            }
            _ => return None,
        };
        return Some(Ok(result));
    }

    // --- string methods ---
    if let Some(s) = ctx.realm.string_value(h) {
        let result = match key {
            "toUpperCase" => NanBox::handle(ctx.realm.new_string(&s.to_uppercase()).to_raw()),
            "toLowerCase" => NanBox::handle(ctx.realm.new_string(&s.to_lowercase()).to_raw()),
            "trim" => NanBox::handle(ctx.realm.new_string(s.trim()).to_raw()),
            "includes" => NanBox::boolean(s.contains(&ctx.realm.to_display_string(arg0()))),
            "startsWith" => NanBox::boolean(s.starts_with(&ctx.realm.to_display_string(arg0()))),
            "endsWith" => NanBox::boolean(s.ends_with(&ctx.realm.to_display_string(arg0()))),
            "indexOf" => {
                let needle = ctx.realm.to_display_string(arg0());
                let i = s.find(&needle).map(|b| s[..b].chars().count());
                NanBox::number(i.map_or(-1.0, |i| i as f64))
            }
            "repeat" => {
                // Negative / non-finite / overflowing counts would panic on
                // `str::repeat`; clamp to an empty string instead (the spec's
                // `RangeError` can't be raised from this native return path).
                let nf = ctx.realm.to_number(arg0());
                let n = if nf.is_finite() && nf >= 0.0 {
                    nf as usize
                } else {
                    0
                };
                let repeated = match n.checked_mul(s.len()) {
                    Some(_) => s.repeat(n),
                    None => String::new(),
                };
                NanBox::handle(ctx.realm.new_string(&repeated).to_raw())
            }
            "charAt" => {
                // ToInteger: `NaN`/no-arg → 0; a negative index is out of range → "".
                let n = ctx.realm.to_number(arg0());
                let n = if n.is_nan() { 0.0 } else { n };
                let c = if n >= 0.0 {
                    s.chars()
                        .nth(n as usize)
                        .map(String::from)
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                NanBox::handle(ctx.realm.new_string(&c).to_raw())
            }
            "split" => {
                let sep = ctx.realm.to_display_string(arg0());
                let mut parts: Vec<NanBox> = if sep.is_empty() {
                    s.chars()
                        .map(|c| NanBox::handle(ctx.realm.new_string(&String::from(c)).to_raw()))
                        .collect()
                } else {
                    s.split(&sep)
                        .map(|p| NanBox::handle(ctx.realm.new_string(p).to_raw()))
                        .collect()
                };
                // An optional limit caps the number of returned segments.
                if let Some(lim) = args.get(1)
                    && !matches!(lim.unpack(), Unpacked::Undefined)
                {
                    let limit = ctx.realm.to_number(*lim);
                    if limit >= 0.0 {
                        parts.truncate(limit as usize);
                    }
                }
                NanBox::handle(ctx.realm.new_array(parts).to_raw())
            }
            _ => return None,
        };
        return Some(Ok(result));
    }

    // --- Promise methods (`then`/`catch`/`finally`) ---
    if ctx.realm.promise_state(h).is_some() {
        let result = match key {
            "then" => promise_then(
                ctx,
                h,
                arg0(),
                args.get(1).copied().unwrap_or(NanBox::undefined()),
            ),
            "catch" => promise_then(ctx, h, NanBox::undefined(), arg0()),
            // Simplified: run the callback on either settlement, value passes through.
            "finally" => promise_then(ctx, h, arg0(), arg0()),
            _ => return None,
        };
        return Some(Ok(NanBox::handle(result.to_raw())));
    }

    // --- Map / Set methods ---
    if let Some(is_set) = ctx.realm.collection_is_set(h) {
        let result = match key {
            "set" if !is_set => {
                let v = args.get(1).copied().unwrap_or(NanBox::undefined());
                ctx.realm.collection_set(h, arg0(), v);
                recv // chainable
            }
            "add" if is_set => {
                ctx.realm.collection_set(h, arg0(), arg0());
                recv
            }
            "get" => ctx
                .realm
                .collection_get(h, arg0())
                .unwrap_or(NanBox::undefined()),
            "has" => NanBox::boolean(ctx.realm.collection_has(h, arg0())),
            "delete" => NanBox::boolean(ctx.realm.collection_delete(h, arg0())),
            "forEach" => {
                let f = arg0();
                for (k, v) in ctx.realm.collection_entries(h).unwrap_or_default() {
                    // Callback receives (value, key) per the JS signature.
                    if let Err(e) = call_closure(ctx, funcs, f, &[v, k], NanBox::undefined()) {
                        return Some(Err(e));
                    }
                }
                NanBox::undefined()
            }
            "keys" | "values" | "entries" => {
                let entries = ctx.realm.collection_entries(h).unwrap_or_default();
                let items: Vec<NanBox> = entries
                    .into_iter()
                    .map(|(k, v)| match key {
                        "keys" => k,
                        "values" => v,
                        _ => NanBox::handle(ctx.realm.new_array(alloc::vec![k, v]).to_raw()),
                    })
                    .collect();
                NanBox::handle(ctx.realm.new_array(items).to_raw())
            }
            _ => return None,
        };
        return Some(Ok(result));
    }
    None
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
        NB_MATH_FLOOR | NB_MATH_CEIL | NB_MATH_ROUND | NB_MATH_SQRT | NB_MATH_POW
        | NB_MATH_TRUNC => {
            let a = args.first().and_then(|v| v.as_number()).unwrap_or(f64::NAN);
            let val = match native {
                NB_MATH_FLOOR => a.floor(),
                NB_MATH_CEIL => a.ceil(),
                NB_MATH_ROUND => crate::common::js_round(a),
                NB_MATH_SQRT => a.sqrt(),
                NB_MATH_TRUNC => a.trunc(),
                _ => a.powf(args.get(1).and_then(|v| v.as_number()).unwrap_or(f64::NAN)),
            };
            NanBox::number(val)
        }
        #[cfg(not(feature = "std"))]
        NB_MATH_FLOOR | NB_MATH_CEIL | NB_MATH_ROUND | NB_MATH_SQRT | NB_MATH_POW
        | NB_MATH_TRUNC => NanBox::number(f64::NAN),
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
        NB_PARSE_INT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            // Keep the sign: a `… as u32` cast saturates a negative radix to 0,
            // which would wrongly default to base 10. A nonzero radix outside
            // [2, 36] is invalid → NaN.
            let radix = args
                .get(1)
                .and_then(|r| r.as_number())
                .filter(|n| n.is_finite())
                .map_or(0i64, |n| n as i64);
            if radix != 0 && !(2..=36).contains(&radix) {
                NanBox::number(f64::NAN)
            } else {
                NanBox::number(parse_int(s.trim(), radix as u32))
            }
        }
        NB_PARSE_FLOAT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            NanBox::number(parse_float_prefix(s.trim()))
        }
        NB_IS_NAN => NanBox::boolean(
            ctx.realm
                .to_number(args.first().copied().unwrap_or(NanBox::undefined()))
                .is_nan(),
        ),
        NB_IS_FINITE => NanBox::boolean(
            ctx.realm
                .to_number(args.first().copied().unwrap_or(NanBox::undefined()))
                .is_finite(),
        ),
        NB_OBJECT_KEYS | NB_OBJECT_VALUES | NB_OBJECT_ENTRIES => {
            let h = args
                .first()
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw);
            // Build `(key, value)` pairs. An array's own enumerable keys are its
            // integer indices (stored as elements, ascending) before any named
            // properties — and the index values come from element access.
            let mut pairs: Vec<(String, NanBox)> = Vec::new();
            if let Some(h) = h {
                // A VM closure backs onto an array but is a function — its captured
                // cells are not enumerable keys/values.
                if !ctx.realm.is_vm_function(h)
                    && let Some(elems) = ctx.realm.array_elements(h).map(<[_]>::to_vec)
                {
                    for (i, v) in elems.into_iter().enumerate() {
                        pairs.push((alloc::format!("{i}"), v));
                    }
                }
                // Plain objects keep keys in the cell; arrays/functions keep named
                // properties in their auxiliary object.
                let named = ctx
                    .realm
                    .object_keys(h)
                    .unwrap_or_else(|| ctx.realm.aux_named_keys(h));
                for k in named {
                    let v = ctx.realm.get_property(h, &k).unwrap_or(NanBox::undefined());
                    pairs.push((k, v));
                }
            }
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs {
                let item = match native {
                    NB_OBJECT_KEYS => NanBox::handle(ctx.realm.new_string(&k).to_raw()),
                    NB_OBJECT_VALUES => v,
                    _ => {
                        let key = NanBox::handle(ctx.realm.new_string(&k).to_raw());
                        NanBox::handle(ctx.realm.new_array(alloc::vec![key, v]).to_raw())
                    }
                };
                out.push(item);
            }
            NanBox::handle(ctx.realm.new_array(out).to_raw())
        }
        NB_JSON_STRINGIFY => {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            // Optional `space` (arg 2): number → spaces, string → that string.
            let space = args.get(2).copied().unwrap_or(NanBox::undefined());
            let indent: String = if let Some(n) = space.as_number() {
                " ".repeat((n.max(0.0) as usize).min(10))
            } else if let Some(s) = space
                .as_handle()
                .and_then(|r| ctx.realm.string_value(Handle::from_raw(r)))
            {
                s.chars().take(10).collect()
            } else {
                String::new()
            };
            // Plain serialization (no interpreter features). The `Op::CallNative`
            // site intercepts `JSON.stringify` and uses the interpreter-aware
            // `json_stringify` instead; this is a pure fallback.
            let result = if indent.is_empty() {
                crate::json::stringify(ctx.realm, v)
            } else {
                crate::json::stringify_pretty(ctx.realm, v, &indent)
            };
            match result {
                Some(s) => NanBox::handle(ctx.realm.new_string(&s).to_raw()),
                None => NanBox::undefined(),
            }
        }
        NB_JSON_PARSE => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            crate::json::parse(ctx.realm, &s).unwrap_or(NanBox::undefined())
        }
        // `Number.*` predicates take a value WITHOUT coercion (unlike the globals).
        NB_NUMBER_IS_INTEGER => {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            NanBox::boolean(
                v.as_number()
                    .is_some_and(|n| n.is_finite() && n % 1.0 == 0.0),
            )
        }
        NB_NUMBER_IS_FINITE => {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            NanBox::boolean(v.as_number().is_some_and(f64::is_finite))
        }
        NB_NUMBER_IS_NAN => {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            NanBox::boolean(v.as_number().is_some_and(f64::is_nan))
        }
        NB_NUMBER_PARSE_FLOAT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            NanBox::number(parse_float_prefix(s.trim()))
        }
        NB_NUMBER_PARSE_INT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            // See NB_PARSE_INT: preserve the sign and reject an out-of-range radix.
            let radix = args
                .get(1)
                .and_then(|r| r.as_number())
                .filter(|n| n.is_finite())
                .map_or(0i64, |n| n as i64);
            if radix != 0 && !(2..=36).contains(&radix) {
                NanBox::number(f64::NAN)
            } else {
                NanBox::number(parse_int(s.trim(), radix as u32))
            }
        }
        NB_STRING_FROM_CHAR_CODE => {
            // Each argument is ToUint16'd into a UTF-16 code unit; decoding the
            // sequence combines an adjacent high/low surrogate pair into one astral
            // code point (a lone surrogate becomes U+FFFD, unrepresentable in UTF-8).
            let units: Vec<u16> = args
                .iter()
                .map(|a| {
                    let n = ctx.realm.to_number(*a);
                    if n.is_finite() {
                        (n as i64).rem_euclid(65536) as u16
                    } else {
                        0
                    }
                })
                .collect();
            let s: String = char::decode_utf16(units)
                .map(|r| r.unwrap_or('\u{FFFD}'))
                .collect();
            NanBox::handle(ctx.realm.new_string(&s).to_raw())
        }
        NB_ARRAY_IS_ARRAY => {
            let yes = args
                .first()
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw)
                .is_some_and(|h| ctx.realm.is_array(h) && !ctx.realm.is_vm_function(h));
            NanBox::boolean(yes)
        }
        NB_ARRAY_FROM => {
            let arg = args.first().copied().unwrap_or(NanBox::undefined());
            let items: Vec<NanBox> = match arg.as_handle().map(Handle::from_raw) {
                Some(h) if ctx.realm.is_array(h) => ctx
                    .realm
                    .array_elements(h)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default(),
                // A `Map`/`Set`: its entries (Set → its elements).
                Some(h) if ctx.realm.collection_is_set(h).is_some() => ctx
                    .realm
                    .collection_entries(h)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect(),
                // A string: its characters.
                Some(h) if ctx.realm.string_value(h).is_some() => ctx
                    .realm
                    .string_value(h)
                    .unwrap_or_default()
                    .chars()
                    .map(|c| NanBox::handle(ctx.realm.new_string(&String::from(c)).to_raw()))
                    .collect(),
                // An array-like object (a `length` + indexed properties). The map
                // callback form is handled only by the tree-walker (call_native
                // here can't invoke a closure).
                Some(h) => {
                    let len = ctx
                        .realm
                        .get_property(h, "length")
                        .map(|v| ctx.realm.to_number(v))
                        .unwrap_or(0.0)
                        .max(0.0) as usize;
                    (0..len)
                        .map(|i| {
                            ctx.realm
                                .get_property(h, &alloc::format!("{i}"))
                                .unwrap_or(NanBox::undefined())
                        })
                        .collect()
                }
                None => Vec::new(),
            };
            NanBox::handle(ctx.realm.new_array(items).to_raw())
        }
        NB_PROMISE_RESOLVE | NB_PROMISE_REJECT => {
            let p = ctx.realm.new_promise();
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            settle(ctx, p, v, native == NB_PROMISE_RESOLVE);
            NanBox::handle(p.to_raw())
        }
        NB_OBJECT_FROM_ENTRIES => {
            let obj = ctx.realm.new_object();
            if let Some(h) = args
                .first()
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw)
            {
                for pair in ctx
                    .realm
                    .array_elements(h)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default()
                {
                    if let Some(ph) = pair.as_handle().map(Handle::from_raw) {
                        let k = ctx.realm.to_display_string(ctx.realm.get_element(ph, 0));
                        let v = ctx.realm.get_element(ph, 1);
                        ctx.realm.set_property(obj, &k, v);
                    }
                }
            }
            NanBox::handle(obj.to_raw())
        }
        NB_OBJECT_ASSIGN => {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            if let Some(t) = target.as_handle().map(Handle::from_raw) {
                for src in args.iter().skip(1) {
                    if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                        for k in ctx.realm.object_keys(sh).unwrap_or_default() {
                            let v = ctx
                                .realm
                                .get_property(sh, &k)
                                .unwrap_or(NanBox::undefined());
                            ctx.realm.set_property(t, &k, v);
                        }
                    }
                }
            }
            target
        }
        _ => NanBox::undefined(),
    }
}

/// Parses the longest leading decimal-float prefix of `s` (à la `parseFloat`).
fn parse_float_prefix(s: &str) -> f64 {
    // A leading (optionally signed) `Infinity`.
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => (-1.0, r),
        None => (1.0, s.strip_prefix('+').unwrap_or(s)),
    };
    if rest.starts_with("Infinity") {
        return sign * f64::INFINITY;
    }
    let bytes = s.as_bytes();
    let mut end = 0;
    let (mut dot, mut e) = (false, false);
    while end < bytes.len() {
        let ch = bytes[end] as char;
        let ok = match ch {
            '0'..='9' => true,
            '+' | '-' if end == 0 || matches!(bytes[end - 1] as char, 'e' | 'E') => true,
            '.' if !dot && !e => {
                dot = true;
                true
            }
            'e' | 'E' if !e && end > 0 => {
                e = true;
                true
            }
            _ => false,
        };
        if !ok {
            break;
        }
        end += 1;
    }
    s[..end].parse::<f64>().unwrap_or(f64::NAN)
}

/// A minimal `parseInt`: leading sign, optional `0x` (radix 0 or 16), then the
/// digits valid in `radix` (default 10).
fn parse_int(s: &str, radix: u32) -> f64 {
    let mut t = s;
    let mut neg = false;
    if let Some(r) = t.strip_prefix('-') {
        neg = true;
        t = r;
    } else if let Some(r) = t.strip_prefix('+') {
        t = r;
    }
    let mut radix = radix;
    if (radix == 0 || radix == 16)
        && let Some(r) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
    {
        t = r;
        radix = 16;
    }
    if radix == 0 {
        radix = 10;
    }
    if !(2..=36).contains(&radix) {
        return f64::NAN;
    }
    let mut value = 0.0;
    let mut any = false;
    for c in t.chars() {
        match c.to_digit(radix) {
            Some(d) => {
                value = value * f64::from(radix) + f64::from(d);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return f64::NAN;
    }
    if neg { -value } else { value }
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
    let mut class_id = 0u32;
    for s in &program.body {
        // A class declaration, or a top-level `const Name = class {…}` (treated
        // as a named class so `new Name(...)` resolves).
        let named = match s {
            Stmt::Class(class) => class.id.as_ref().map(|id| (String::from(&*id.name), class)),
            Stmt::Var(decl) => match decl.declarations.as_slice() {
                [d] => match (&d.target, &d.init) {
                    (BindingTarget::Ident(id), Some(Expr::Class(class))) => {
                        Some((String::from(&*id.name), class))
                    }
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        if let Some((name, class)) = named {
            let info = scan_class(class, class_id, &mut next_id, &mut class_jobs)?;
            class_id += 1;
            class_map.insert(name, info);
        }
    }
    let classes = alloc::rc::Rc::new(class_map);

    let protos = alloc::rc::Rc::new(core::cell::RefCell::new(Vec::new()));
    let placeholder = || FnProto {
        ops: Vec::new(),
        n_regs: 0,
        n_params: 0,
        n_captures: 0,
        rest_from: None,
        is_async: false,
        length: 0,
        name: alloc::string::String::new(),
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
        let mut proto = Compiler::compile_fn_inner(
            &fn_ids,
            &classes,
            &protos,
            &f.params,
            &[],
            &f.body,
            false,
            None,
            &[],
            None,
            f.is_async,
        )?;
        // A function declaration's `name` is its declared identifier.
        if let Some(id) = &f.id {
            proto.name = alloc::string::String::from(id.name.as_ref());
        }
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
            &job.fields,
            job.super_of.clone(),
            false,
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
        ("Math", "trunc") => Some(NB_MATH_TRUNC),
        ("Math", "round") => Some(NB_MATH_ROUND),
        ("Math", "sqrt") => Some(NB_MATH_SQRT),
        ("Math", "pow") => Some(NB_MATH_POW),
        ("Object", "keys") => Some(NB_OBJECT_KEYS),
        ("Object", "values") => Some(NB_OBJECT_VALUES),
        ("Object", "entries") => Some(NB_OBJECT_ENTRIES),
        ("Object", "assign") => Some(NB_OBJECT_ASSIGN),
        ("Object", "fromEntries") => Some(NB_OBJECT_FROM_ENTRIES),
        ("JSON", "stringify") => Some(NB_JSON_STRINGIFY),
        ("JSON", "parse") => Some(NB_JSON_PARSE),
        ("Number", "isInteger") => Some(NB_NUMBER_IS_INTEGER),
        ("Number", "isFinite") => Some(NB_NUMBER_IS_FINITE),
        ("Number", "isNaN") => Some(NB_NUMBER_IS_NAN),
        ("Number", "parseFloat") => Some(NB_NUMBER_PARSE_FLOAT),
        ("Number", "parseInt") => Some(NB_NUMBER_PARSE_INT),
        ("String", "fromCharCode") => Some(NB_STRING_FROM_CHAR_CODE),
        ("Array", "from") => Some(NB_ARRAY_FROM),
        ("Array", "isArray") => Some(NB_ARRAY_IS_ARRAY),
        ("Promise", "resolve") => Some(NB_PROMISE_RESOLVE),
        ("Promise", "reject") => Some(NB_PROMISE_REJECT),
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
        "parseInt" => Some(NB_PARSE_INT),
        "parseFloat" => Some(NB_PARSE_FLOAT),
        "isNaN" => Some(NB_IS_NAN),
        "isFinite" => Some(NB_IS_FINITE),
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
/// Whether a statement list can complete abruptly via `return`/`break`/`continue`
/// that reaches out of an enclosing `try` block — so its `finally` must run on the
/// way out. Conservative: it descends into control flow (over-reporting a
/// `break`/`continue` fully contained in a nested loop is safe — it only routes the
/// program to the tree-walker), but not into nested functions/classes, whose
/// abrupt statements exit *them*, not the try.
fn block_can_exit_abruptly(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_can_exit_abruptly)
}

fn stmt_can_exit_abruptly(s: &Stmt) -> bool {
    match s {
        Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => true,
        Stmt::Block { body, .. } => block_can_exit_abruptly(body),
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            stmt_can_exit_abruptly(consequent)
                || alternate.as_deref().is_some_and(stmt_can_exit_abruptly)
        }
        Stmt::For { body, .. }
        | Stmt::ForIn { body, .. }
        | Stmt::ForOf { body, .. }
        | Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::Labeled { body, .. }
        | Stmt::With { body, .. } => stmt_can_exit_abruptly(body),
        Stmt::Switch { cases, .. } => cases.iter().any(|c| block_can_exit_abruptly(&c.body)),
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            block_can_exit_abruptly(block)
                || handler
                    .as_ref()
                    .is_some_and(|h| block_can_exit_abruptly(&h.body))
                || finalizer
                    .as_ref()
                    .is_some_and(|f| block_can_exit_abruptly(f))
        }
        _ => false, // nested functions/classes exit themselves, not the try
    }
}

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
    class_id: u32,
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
        class_id,
        super_name: super_name.clone(),
        ctor: None,
        methods: Vec::new(),
        accessors: Vec::new(),
        statics: Vec::new(),
    };
    let mut ctor_member: Option<&crate::ast::ClassMethod> = None;
    let mut fields: Vec<(String, Option<&'a Expr>)> = Vec::new();
    // Records a getter/setter id against accessor `name`.
    let add_accessor = |name: String,
                        getter: Option<u32>,
                        setter: Option<u32>,
                        acc: &mut Vec<(String, Option<u32>, Option<u32>)>| {
        if let Some(a) = acc.iter_mut().find(|(n, _, _)| *n == name) {
            a.1 = a.1.or(getter);
            a.2 = a.2.or(setter);
        } else {
            acc.push((name, getter, setter));
        }
    };
    for member in &class.body {
        match member {
            ClassMember::Method(m) if !m.is_static && m.kind == MethodKind::Constructor => {
                ctor_member = Some(m);
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
                    super_of: super_name.clone(),
                    fields: Vec::new(),
                });
            }
            ClassMember::Method(m)
                if !m.is_static && matches!(m.kind, MethodKind::Get | MethodKind::Set) =>
            {
                let id = *next_id;
                *next_id += 1;
                let name = static_key(&m.key)?;
                jobs.push(ClassJob {
                    id,
                    params: &m.value.params,
                    body: &m.value.body,
                    super_of: super_name.clone(),
                    fields: Vec::new(),
                });
                if m.kind == MethodKind::Get {
                    add_accessor(name, Some(id), None, &mut info.accessors);
                } else {
                    add_accessor(name, None, Some(id), &mut info.accessors);
                }
            }
            ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                let id = *next_id;
                *next_id += 1;
                let name = static_key(&m.key)?;
                info.statics.push((name, id));
                jobs.push(ClassJob {
                    id,
                    params: &m.value.params,
                    body: &m.value.body,
                    super_of: None,
                    fields: Vec::new(),
                });
            }
            ClassMember::Field(f) if !f.is_static => {
                fields.push((static_key(&f.key)?, f.value.as_ref()));
            }
            // Static fields are installed at the declaration site (see
            // `Stmt::Class`); static getters/setters → fall back.
            ClassMember::Field(f) if f.is_static => {}
            _ => return Err(CompileError::Unsupported("class member")),
        }
    }
    // Field initializers run in the constructor; with `extends` their ordering
    // relative to `super()` falls back for now.
    if !fields.is_empty() && super_name.is_some() {
        return Err(CompileError::Unsupported("fields with extends"));
    }
    // A constructor job (synthetic when only fields are present) runs the field
    // initializers, then the declared constructor body.
    if ctor_member.is_some() || !fields.is_empty() {
        let id = *next_id;
        *next_id += 1;
        info.ctor = Some(id);
        let (params, body): (&[crate::ast::Param], &[Stmt]) = match ctor_member {
            Some(m) => (&m.value.params, &m.value.body),
            None => (&[], &[]),
        };
        jobs.push(ClassJob {
            id,
            params,
            body,
            super_of: super_name.clone(),
            fields,
        });
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
    /// `(field_name, initializer)` to run at the start of the constructor (only
    /// the constructor job carries these).
    fields: Vec<(String, Option<&'a crate::ast::Expr>)>,
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
    /// A unique class id (the instance's `class_tag`, for `instanceof`).
    class_id: u32,
    /// The `extends` superclass name, if any.
    super_name: Option<String>,
    /// The constructor's function id, if the class declares one.
    ctor: Option<u32>,
    /// `(method_name, function_id)` for each instance method.
    methods: Vec<(String, u32)>,
    /// `(name, getter_id, setter_id)` for each accessor property.
    accessors: Vec<(String, Option<u32>, Option<u32>)>,
    /// `(method_name, function_id)` for each `static` method.
    statics: Vec<(String, u32)>,
}

/// A variable binding: the register holding it, and whether that register holds
/// a *cell* (a one-element heap array) rather than the value directly. Captured
/// variables are cells so closures share their mutations.
#[derive(Clone, Copy)]
struct Binding {
    reg: Reg,
    cell: bool,
    /// Declared `const` (reassignment is a TypeError — routed to the tree-walker).
    konst: bool,
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
    /// When compiling a subclass method/constructor, the superclass name — for
    /// resolving `super.method(...)`.
    super_class: Option<String>,
    /// Per enclosing loop/switch: `break` jump indices awaiting the exit target
    /// (loops and `switch` both push here).
    break_sites: Vec<Vec<usize>>,
    /// Per enclosing loop: `continue` jump indices awaiting the loop's continue
    /// point (`switch` does *not* push here — `continue` targets the nearest
    /// loop).
    continue_sites: Vec<Vec<usize>>,
    /// Per enclosing optional-chain (`Expr::OptChain`): the `?.`-link jump indices
    /// awaiting the chain end. A nullish `?.` base jumps here, short-circuiting the
    /// whole remaining chain to `undefined`.
    optchain_ends: Vec<Vec<usize>>,
    /// Active statement labels → the `break_sites`/`continue_sites` stack index
    /// of the loop they label (for `break label` / `continue label`).
    labels: Vec<(String, usize)>,
    /// In `main`: a top-level function declaration name → the register holding its
    /// one canonical closure (materialized once at entry). Reading the function as
    /// a *value* uses this register, so it has a stable identity (`f === f`) and
    /// holds assigned properties; *calls* still dispatch directly via `fn_ids`.
    fn_value_regs: alloc::collections::BTreeMap<String, Reg>,
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
            fn_ids,
            classes,
            protos,
            params,
            captures,
            body,
            is_main,
            None,
            &[],
            None,
            false,
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
        fields: &[(String, Option<&crate::ast::Expr>)],
        super_class: Option<String>,
        is_async: bool,
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
            super_class,
            ..Compiler::default()
        };
        c.scopes.push(alloc::collections::BTreeMap::new());
        // The VM places arguments in registers `0..n_params`, captured cells in
        // `n_params..n_params + n_captures`, and `this` right after, so reserve
        // those slots first…
        let arg_regs: Vec<Reg> = params.iter().map(|_| c.alloc()).collect();
        let cap_regs: Vec<Reg> = captures.iter().map(|_| c.alloc()).collect();
        c.this_reg = c.alloc(); // = n_params + n_captures
        // A trailing rest parameter: the caller fills its register with an array.
        let rest_from = if params.last().is_some_and(|p| p.rest) {
            Some(params.len() - 1)
        } else {
            None
        };
        // …then bind. A captured parameter is boxed into a fresh cell (preserving
        // the incoming argument value); a captured local that's a parameter must
        // share the cell so mutations are visible.
        for (i, p) in params.iter().enumerate() {
            match &p.target {
                BindingTarget::Ident(Ident { name, .. }) => {
                    let b = if c.cell_names.contains(&**name) {
                        let cell = c.alloc();
                        c.ops.push(Op::NewArray { dst: cell, len: 1 });
                        let bind = Binding {
                            reg: cell,
                            cell: true,
                            konst: false,
                        };
                        c.write_var(bind, arg_regs[i]);
                        bind
                    } else {
                        Binding {
                            reg: arg_regs[i],
                            cell: false,
                            konst: false,
                        }
                    };
                    c.scopes
                        .last_mut()
                        .expect("a scope")
                        .insert(String::from(&**name), b);
                }
                // A destructuring parameter binds from the incoming arg register.
                other => c.bind_pattern(other, arg_regs[i])?,
            }
        }
        // Captured cells arrive already boxed (the closure passes the cell).
        for (j, name) in captures.iter().enumerate() {
            c.scopes.last_mut().expect("a scope").insert(
                name.clone(),
                Binding {
                    reg: cap_regs[j],
                    cell: true,
                    konst: false,
                },
            );
        }
        // Apply `= default` to any (non-rest) parameter left `undefined` — after
        // binding, so a default may reference earlier parameters; written back
        // through the binding (honoring cells).
        for p in params {
            if let (Some(def), BindingTarget::Ident(Ident { name, .. })) = (&p.default, &p.target) {
                let b = c.lookup(name).expect("a bound param");
                let cur = c.read_var(b);
                c.apply_default(cur, Some(def))?;
                c.write_var(b, cur);
            }
        }
        // Field initializers run first (constructors only): `this.field = init`.
        for (name, init) in fields {
            let v = match init {
                Some(e) => c.expr(e)?,
                None => c.constant(NanBox::undefined())?,
            };
            let this = c.this_reg;
            c.ops.push(Op::SetProp {
                obj: this,
                key: name.clone(),
                src: v,
            });
        }
        // In `main`, materialize one canonical closure per top-level function
        // declaration so referencing it as a value has a stable identity (and can
        // hold assigned properties). Calls still dispatch directly by id.
        if is_main {
            for stmt in body {
                if let Stmt::Function(f) = stmt
                    && let Some(id) = &f.id
                    && let Some(&func) = c.fn_ids.get(&*id.name)
                {
                    let reg = c.alloc();
                    c.ops.push(Op::LoadFunc { dst: reg, func });
                    c.fn_value_regs.insert(String::from(&*id.name), reg);
                }
            }
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
        // `fn.length`: params before the first default value or the rest param.
        let length = params
            .iter()
            .take_while(|p| !p.rest && p.default.is_none())
            .count();
        Ok(FnProto {
            n_regs: c.next_reg as usize,
            n_params: params.len(),
            n_captures: captures.len(),
            rest_from,
            is_async,
            length,
            ops: c.ops,
            name: alloc::string::String::new(),
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
        let b = Binding {
            reg,
            cell,
            konst: false,
        };
        self.scopes
            .last_mut()
            .expect("a scope")
            .insert(String::from(name), b);
        b
    }

    /// Marks the just-declared local `name` as `const`.
    fn mark_const(&mut self, name: &str) {
        if let Some(b) = self.scopes.last_mut().and_then(|s| s.get_mut(name)) {
            b.konst = true;
        }
    }

    fn lookup(&self, name: &str) -> Option<Binding> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    /// Binds a (possibly destructuring) target to the value in `value_reg`,
    /// declaring the names it introduces. Array/object patterns read elements/
    /// properties via `GetElem`/`GetProp` with `=`-defaults; rest patterns fall
    /// back.
    /// Destructuring *assignment* to existing targets (`[a, b] = …`,
    /// `({ x } = …)`) — the mirror of `bind_pattern` that writes to existing
    /// variables / members rather than declaring.
    fn assign_pattern(&mut self, target: &Expr, value_reg: Reg) -> Result<(), CompileError> {
        match target {
            Expr::Ident(id) => {
                let b = self
                    .lookup(&id.name)
                    .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                self.write_var(b, value_reg);
                Ok(())
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.expr(object)?;
                self.member_write(obj, property, value_reg)
            }
            Expr::Array { elements, .. } => {
                for (i, el) in elements.iter().enumerate() {
                    match el {
                        ArrayElement::Item(e) => {
                            let idx = self.constant(NanBox::number(i as f64))?;
                            let v = self.alloc();
                            self.ops.push(Op::GetElem {
                                dst: v,
                                arr: value_reg,
                                index: idx,
                            });
                            self.assign_target_with_default(e, v)?;
                        }
                        ArrayElement::Hole => {}
                        ArrayElement::Spread(e) => {
                            let from = self.constant(NanBox::number(i as f64))?;
                            let rest = self.alloc();
                            self.ops.push(Op::ArraySliceFrom {
                                dst: rest,
                                src: value_reg,
                                from,
                            });
                            self.assign_pattern(e, rest)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Object { members, .. } => {
                let mut named: Vec<String> = Vec::new();
                for m in members {
                    match m {
                        ObjectMember::Property { key, value, .. } => {
                            let key = static_key(key)?;
                            let v = self.alloc();
                            self.ops.push(Op::GetProp {
                                dst: v,
                                obj: value_reg,
                                key: key.clone(),
                            });
                            self.assign_target_with_default(value, v)?;
                            named.push(key);
                        }
                        ObjectMember::Spread { value, .. } => {
                            let r = self.alloc();
                            self.ops.push(Op::ObjectRest {
                                dst: r,
                                src: value_reg,
                                exclude: named.clone().into(),
                            });
                            self.assign_pattern(value, r)?;
                        }
                        ObjectMember::Accessor { .. } => {
                            return Err(CompileError::Unsupported(
                                "accessor in assignment pattern",
                            ));
                        }
                    }
                }
                Ok(())
            }
            _ => Err(CompileError::Unsupported("assignment pattern target")),
        }
    }

    /// Assigns `value_reg` to `target`, applying a `=`-default when the target is
    /// `Expr::Assign { target, value: default }` (the destructuring-default form).
    fn assign_target_with_default(
        &mut self,
        target: &Expr,
        value_reg: Reg,
    ) -> Result<(), CompileError> {
        if let Expr::Assign {
            op: crate::ast::AssignOp::Assign,
            target: inner,
            value: default,
            ..
        } = target
        {
            self.apply_default(value_reg, Some(default))?;
            return self.assign_pattern(inner, value_reg);
        }
        self.assign_pattern(target, value_reg)
    }

    fn bind_pattern(&mut self, target: &BindingTarget, value_reg: Reg) -> Result<(), CompileError> {
        match target {
            BindingTarget::Ident(Ident { name, .. }) => {
                let b = self.declare(name);
                self.write_var(b, value_reg);
                Ok(())
            }
            BindingTarget::Array(pat) => {
                use crate::ast::ArrayPatternElement;
                for (i, el) in pat.elements.iter().enumerate() {
                    match el {
                        ArrayPatternElement::Hole => {}
                        ArrayPatternElement::Item {
                            target, default, ..
                        } => {
                            let idx = self.constant(NanBox::number(i as f64))?;
                            let v = self.alloc();
                            self.ops.push(Op::GetElem {
                                dst: v,
                                arr: value_reg,
                                index: idx,
                            });
                            self.apply_default(v, default.as_ref())?;
                            self.bind_pattern(target, v)?;
                        }
                        ArrayPatternElement::Rest { target, .. } => {
                            // `...rest` = the source array sliced from here.
                            let from = self.constant(NanBox::number(i as f64))?;
                            let rest = self.alloc();
                            self.ops.push(Op::ArraySliceFrom {
                                dst: rest,
                                src: value_reg,
                                from,
                            });
                            self.bind_pattern(target, rest)?;
                        }
                    }
                }
                Ok(())
            }
            BindingTarget::Object(pat) => {
                let mut named: Vec<String> = Vec::new();
                for prop in &pat.properties {
                    let key = static_key(&prop.key)?;
                    let v = self.alloc();
                    self.ops.push(Op::GetProp {
                        dst: v,
                        obj: value_reg,
                        key: key.clone(),
                    });
                    self.apply_default(v, prop.default.as_ref())?;
                    self.bind_pattern(&prop.value, v)?;
                    named.push(key);
                }
                // `...rest` = a new object of the remaining own properties.
                if let Some(rest) = &pat.rest {
                    let r = self.alloc();
                    self.ops.push(Op::ObjectRest {
                        dst: r,
                        src: value_reg,
                        exclude: named.into(),
                    });
                    self.bind_pattern(rest, r)?;
                }
                Ok(())
            }
        }
    }

    /// If `reg` holds `undefined` and a `default` exists, overwrites `reg` with
    /// the default's value.
    fn apply_default(&mut self, reg: Reg, default: Option<&Expr>) -> Result<(), CompileError> {
        let Some(e) = default else { return Ok(()) };
        let undef = self.constant(NanBox::undefined())?;
        let is_undef = self.alloc();
        self.ops.push(Op::StrictEq {
            dst: is_undef,
            a: reg,
            b: undef,
        });
        // Skip the default unless the value is `undefined`.
        let jf = self.emit_jump_if_false(is_undef);
        let d = self.expr(e)?;
        self.ops.push(Op::Move { dst: reg, src: d });
        self.patch(jf);
        Ok(())
    }

    /// Emits a read of `name` into a register and returns it (a cell read goes
    /// through `GetElem`).
    /// `CreatePerIterationEnvironment` for a `let`/`const` C-style `for` head: give
    /// each captured loop variable a *fresh* cell seeded with its current value, so a
    /// closure made in one iteration's body captures that iteration's binding (the
    /// classic `for (let i …) … () => i` semantics). Only cell (captured) bindings
    /// need it; a plain register binding is invisible to closures.
    fn refresh_loop_cells(&mut self, bindings: &[Binding]) {
        for b in bindings {
            if b.cell {
                let val = self.read_var(*b);
                self.ops.push(Op::NewArray { dst: b.reg, len: 1 });
                self.write_var(*b, val);
            }
        }
    }

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
            Stmt::Function(_) => Ok(None),
            // Methods/constructors are compiled up front; here we materialize the
            // class's *static* side as a value object bound to the class name, so
            // `ClassName.staticMethod()` / `ClassName.staticField` work.
            Stmt::Class(class) => {
                if let Some(cid) = &class.id {
                    self.materialize_class(&cid.name, class)?;
                }
                Ok(None)
            }
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
                // A `finally` must run even when the `try`/`catch` exits via
                // `return`/`break`/`continue`, but the emitter only runs it on the
                // normal/throw paths. When the body can exit abruptly, defer the
                // whole program to the tree-walker (which handles it correctly).
                if finalizer.is_some()
                    && (block_can_exit_abruptly(block)
                        || handler
                            .as_ref()
                            .is_some_and(|h| block_can_exit_abruptly(&h.body)))
                {
                    return Err(CompileError::Unsupported("try/finally with abrupt exit"));
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
                                konst: false,
                            };
                            self.write_var(bind, catch_reg);
                            bind
                        } else {
                            Binding {
                                reg: catch_reg,
                                cell: false,
                                konst: false,
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
                    // `const Name = class {…}` was registered as a named class
                    // (so `new Name()` resolves); materialize its static side.
                    if let (BindingTarget::Ident(id), Some(Expr::Class(class))) =
                        (&d.target, &d.init)
                        && self.classes.contains_key(&*id.name)
                    {
                        self.materialize_class(&id.name, class)?;
                        continue;
                    }
                    let value = match &d.init {
                        Some(e) => self.expr_named(e, &d.target)?,
                        None => self.constant(NanBox::undefined())?,
                    };
                    self.bind_pattern(&d.target, value)?;
                    if matches!(decl.kind, crate::ast::VarDeclKind::Const)
                        && let BindingTarget::Ident(id) = &d.target
                    {
                        self.mark_const(&id.name);
                    }
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
            Stmt::Break {
                label: Some(label), ..
            } => {
                let idx = self
                    .labels
                    .iter()
                    .rev()
                    .find(|(n, _)| n == &*label.name)
                    .map(|(_, i)| *i)
                    .ok_or(CompileError::Unsupported("break to unknown label"))?;
                let j = self.emit_jump();
                self.break_sites[idx].push(j);
                Ok(None)
            }
            Stmt::Continue {
                label: Some(label), ..
            } => {
                let idx = self
                    .labels
                    .iter()
                    .rev()
                    .find(|(n, _)| n == &*label.name)
                    .map(|(_, i)| *i)
                    .ok_or(CompileError::Unsupported("continue to unknown label"))?;
                let j = self.emit_jump();
                self.continue_sites[idx].push(j);
                Ok(None)
            }
            // A labeled loop: record the label against the loop's site index so
            // `break label` / `continue label` can target it.
            Stmt::Labeled { label, body, .. } => {
                let is_loop = matches!(
                    &**body,
                    Stmt::While { .. }
                        | Stmt::DoWhile { .. }
                        | Stmt::For { .. }
                        | Stmt::ForIn { .. }
                        | Stmt::ForOf { .. }
                );
                let idx = self.break_sites.len();
                self.labels.push((String::from(&*label.name), idx));
                if is_loop {
                    // The labeled loop pushes its own break/continue sites at `idx`.
                    let r = self.stmt(body);
                    self.labels.pop();
                    r
                } else {
                    // A labeled non-loop (e.g. a block): give the label its own
                    // break target so `break label` jumps past the body.
                    self.break_sites.push(Vec::new());
                    self.continue_sites.push(Vec::new());
                    let r = self.stmt(body);
                    self.labels.pop();
                    let end = self.ops.len();
                    for b in self.break_sites.pop().unwrap_or_default() {
                        self.patch_to(b, end);
                    }
                    self.continue_sites.pop();
                    r
                }
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
                let ForLeft::Decl { target, .. } = left else {
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
                // The loop variable — an identifier or a destructuring pattern.
                self.bind_pattern(target, cur)?;
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
            // `for (const k in obj)` — iterate the object's enumerable keys.
            Stmt::ForIn {
                left, right, body, ..
            } => {
                use crate::ast::ForLeft;
                let ForLeft::Decl { target, .. } = left else {
                    return Err(CompileError::Unsupported("for-in binding"));
                };
                self.scopes.push(alloc::collections::BTreeMap::new());
                let obj = self.expr(right)?;
                let arr = self.alloc();
                self.ops.push(Op::EnumKeys { dst: arr, obj });
                let len = self.alloc();
                self.ops.push(Op::ArrayLen { dst: len, arr });
                let i = self.alloc();
                self.ops.push(Op::LoadConst {
                    dst: i,
                    value: NanBox::number(0.0),
                });
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
                self.bind_pattern(target, cur)?;
                self.enter_loop();
                self.stmt(body)?;
                let cont = self.ops.len();
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
                // A `let`/`const` head gives each iteration a fresh binding for any
                // captured loop variable (so closures capture per-iteration values).
                let per_iter: Vec<Binding> = match init {
                    Some(ForInit::Var(decl)) if decl.kind != crate::ast::VarDeclKind::Var => decl
                        .declarations
                        .iter()
                        .filter_map(|d| match &d.target {
                            BindingTarget::Ident(id) => self.lookup(&id.name),
                            _ => None,
                        })
                        .filter(|b| b.cell)
                        .collect(),
                    _ => Vec::new(),
                };
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
                // `continue` runs the per-iteration copy and then the update.
                let cont = self.ops.len();
                self.refresh_loop_cells(&per_iter);
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
                } else if let Some(&reg) = self.fn_value_regs.get(&*id.name) {
                    // A top-level function used as a value: its one canonical
                    // closure (same handle each time → stable identity, holds
                    // assigned properties).
                    Ok(reg)
                } else if let Some(&func) = self.fn_ids.get(&*id.name) {
                    // A function referenced as a value before the canonical closure
                    // is available (e.g. inside a nested function): materialize one.
                    let dst = self.alloc();
                    self.ops.push(Op::LoadFunc { dst, func });
                    Ok(dst)
                } else {
                    // The global value identifiers.
                    match &*id.name {
                        "undefined" => self.constant(NanBox::undefined()),
                        "NaN" => self.constant(NanBox::number(f64::NAN)),
                        "Infinity" => self.constant(NanBox::number(f64::INFINITY)),
                        // A built-in the tree-walker provides as a bare value:
                        // fall back so it resolves correctly.
                        n if KNOWN_GLOBALS.contains(&n) => {
                            Err(CompileError::Undefined(String::from(n)))
                        }
                        // A genuinely undefined identifier throws a ReferenceError
                        // at runtime (matching JS), so a `try`/`catch` sees it.
                        _ => {
                            let msg = alloc::format!("{} is not defined", id.name);
                            Ok(self.emit_throw_error("ReferenceError", &msg))
                        }
                    }
                }
            }
            Expr::Unary { op, argument, .. } => {
                // `delete obj.k` / `delete obj[k]` removes an own property.
                if matches!(op, UnaryOp::Delete) {
                    if let Expr::Member {
                        object, property, ..
                    } = &**argument
                    {
                        let obj = self.expr(object)?;
                        let key = match property {
                            PropertyKey::Computed(e) => self.expr(e)?,
                            _ => self.constant_str(&static_key(property)?),
                        };
                        let dst = self.alloc();
                        self.ops.push(Op::DeleteProp { dst, obj, key });
                        return Ok(dst);
                    }
                    // `delete` of a non-reference is a no-op that yields `true`.
                    return self.constant(NanBox::boolean(true));
                }
                // `typeof x` must not throw for a *genuinely* undefined identifier.
                // A resolvable bare name — a global value (`NaN`/`Infinity`/
                // `undefined`) or a known builtin (`Math`, `BigInt`, …) — instead
                // goes through the normal path (the builtin bails to the
                // tree-walker), so `typeof Math` is `"object"`, not `"undefined"`.
                if matches!(op, UnaryOp::Typeof)
                    && let Expr::Ident(id) = &**argument
                    && self.lookup(&id.name).is_none()
                    && !self.fn_ids.contains_key(&*id.name)
                    && !matches!(&*id.name, "undefined" | "NaN" | "Infinity")
                    && !KNOWN_GLOBALS.contains(&&*id.name)
                {
                    return Ok(self.constant_str("undefined"));
                }
                let a = self.expr(argument)?;
                let dst = self.alloc();
                match op {
                    UnaryOp::Minus => self.ops.push(Op::Neg { dst, a }),
                    UnaryOp::Not => self.ops.push(Op::Not { dst, a }),
                    UnaryOp::Plus => {
                        // `+x` → ToNumber; `Number(x)` does the same.
                        self.ops.push(Op::CallNative {
                            dst,
                            native: NB_NUMBER,
                            args: alloc::vec![a],
                        });
                    }
                    UnaryOp::Typeof => self.ops.push(Op::TypeOf { dst, a }),
                    UnaryOp::BitNot => self.ops.push(Op::BitNot { dst, a }),
                    UnaryOp::Void => {
                        self.ops.push(Op::LoadConst {
                            dst,
                            value: NanBox::undefined(),
                        });
                    }
                    UnaryOp::Delete => return Err(CompileError::Unsupported("delete")),
                }
                Ok(dst)
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                // `x instanceof Class` for a known class: true iff `x`'s class
                // tag is the class or one of its subclasses (computed here).
                if matches!(op, BinaryOp::Instanceof)
                    && let Expr::Ident(cls) = &**right
                    && self.classes.contains_key(&*cls.name)
                {
                    let target_name = &*cls.name;
                    // Every class whose `extends` chain reaches the target.
                    let mut ids: Vec<u32> = Vec::new();
                    for (name, info) in self.classes.iter() {
                        let mut cur = Some(name.clone());
                        while let Some(n) = cur {
                            if n == target_name {
                                ids.push(info.class_id);
                                break;
                            }
                            cur = self.classes.get(&n).and_then(|c| c.super_name.clone());
                        }
                    }
                    let obj = self.expr(left)?;
                    let dst = self.alloc();
                    self.ops.push(Op::InstanceOf {
                        dst,
                        obj,
                        ids: ids.into(),
                    });
                    return Ok(dst);
                }
                // `x instanceof Error/TypeError/…` (the built-in error objects we
                // model as `{ name, message }`): compare the `name` property.
                if matches!(op, BinaryOp::Instanceof)
                    && let Expr::Ident(cls) = &**right
                    && matches!(
                        &*cls.name,
                        "Error"
                            | "TypeError"
                            | "RangeError"
                            | "SyntaxError"
                            | "ReferenceError"
                            | "EvalError"
                            | "URIError"
                    )
                {
                    let obj = self.expr(left)?;
                    let name = self
                        .member_read(obj, &PropertyKey::Ident(alloc::boxed::Box::from("name")))?;
                    let want = self.constant_str(&cls.name);
                    let dst = self.alloc();
                    // `instanceof Error` matches any of our error objects.
                    if &*cls.name == "Error" {
                        // name ends with "Error" → treat as an Error instance.
                        let suffix = self.constant_str("Error");
                        self.ops.push(Op::CallMethod {
                            dst,
                            recv: name,
                            key: String::from("endsWith"),
                            args: alloc::vec![suffix],
                        });
                    } else {
                        self.ops.push(Op::StrictEq {
                            dst,
                            a: name,
                            b: want,
                        });
                    }
                    return Ok(dst);
                }
                // `x instanceof RegExp/Array/Map/Set` (built-in heap types).
                if matches!(op, BinaryOp::Instanceof)
                    && let Expr::Ident(cls) = &**right
                    && let Some(kind) = match &*cls.name {
                        "RegExp" => Some(0u8),
                        "Array" => Some(1),
                        "Map" => Some(2),
                        "Set" => Some(3),
                        _ => None,
                    }
                    && self.classes.get(&*cls.name).is_none()
                {
                    let obj = self.expr(left)?;
                    let dst = self.alloc();
                    self.ops.push(Op::IsBuiltin { dst, obj, kind });
                    return Ok(dst);
                }
                let a = self.expr(left)?;
                let b = self.expr(right)?;
                if matches!(op, BinaryOp::In) {
                    let dst = self.alloc();
                    self.ops.push(Op::HasProp {
                        dst,
                        key: a,
                        obj: b,
                    });
                    return Ok(dst);
                }
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
                    LogicalOp::Nullish => {
                        // Take the right side when the left is nullish.
                        let nn = self.emit_not_nullish(dst)?;
                        let g = self.alloc();
                        self.ops.push(Op::Not { dst: g, a: nn });
                        g
                    }
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
                // Build incrementally (empty + push/extend) so spreads and holes
                // work uniformly.
                let dst = self.alloc();
                self.ops.push(Op::NewArray { dst, len: 0 });
                for el in elements {
                    match el {
                        ArrayElement::Item(e) => {
                            let v = self.expr(e)?;
                            self.ops.push(Op::ArrayPush { arr: dst, src: v });
                        }
                        ArrayElement::Hole => {
                            let u = self.constant(NanBox::undefined())?;
                            self.ops.push(Op::ArrayPush { arr: dst, src: u });
                        }
                        ArrayElement::Spread(e) => {
                            let s = self.expr(e)?;
                            self.ops.push(Op::ArrayExtend { arr: dst, src: s });
                        }
                    }
                }
                Ok(dst)
            }
            Expr::Object { members, .. } => {
                let dst = self.alloc();
                self.ops.push(Op::NewObject { dst });
                for m in members {
                    match m {
                        ObjectMember::Property { key, value, .. } => {
                            let v = self.expr(value)?;
                            match key {
                                PropertyKey::Computed(e) => {
                                    let k = self.expr(e)?;
                                    self.ops.push(Op::SetKey {
                                        obj: dst,
                                        key: k,
                                        src: v,
                                    });
                                }
                                _ => self.ops.push(Op::SetProp {
                                    obj: dst,
                                    key: static_key(key)?,
                                    src: v,
                                }),
                            }
                        }
                        ObjectMember::Spread { value, .. } => {
                            let src = self.expr(value)?;
                            self.ops.push(Op::ObjectSpread { dst, src });
                        }
                        ObjectMember::Accessor {
                            is_getter,
                            key,
                            value,
                            ..
                        } => {
                            let key = static_key(key)?;
                            let f = self.make_closure(&value.params, &value.body, false, "")?;
                            let undef = self.constant(NanBox::undefined())?;
                            let (getter, setter) = if *is_getter { (f, undef) } else { (undef, f) };
                            self.ops.push(Op::DefineAccessor {
                                obj: dst,
                                key,
                                getter,
                                setter,
                            });
                        }
                    }
                }
                Ok(dst)
            }
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                let obj = self.expr(object)?;
                if *optional {
                    let go = self.emit_not_nullish(obj)?;
                    let jf = self.emit_jump_if_false(go); // jump when nullish
                    if self.optchain_ends.is_empty() {
                        // Defensive (a `?.` not wrapped by the parser): per-link skip.
                        let dst = self.alloc();
                        self.ops.push(Op::LoadConst {
                            dst,
                            value: NanBox::undefined(),
                        });
                        let v = self.member_read(obj, property)?;
                        self.ops.push(Op::Move { dst, src: v });
                        self.patch(jf);
                        Ok(dst)
                    } else {
                        // A nullish base short-circuits the *whole* enclosing chain:
                        // jump to the `OptChain` end (where the result stays
                        // `undefined`), skipping the rest of the chain's links.
                        self.optchain_ends.last_mut().unwrap().push(jf);
                        self.member_read(obj, property)
                    }
                } else {
                    self.member_read(obj, property)
                }
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
                // A subclass whose base has no explicit constructor: a no-op.
                if matches!(&**callee, Expr::Super(_)) {
                    if let Some(ctor) = self.super_ctor {
                        let recv = self.this_reg;
                        self.ops.push(Op::CallCtor { ctor, recv, args });
                    } else if self.super_class.is_none() {
                        return Err(CompileError::Unsupported("super outside a subclass ctor"));
                    }
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
                // `super.method(args)` — call the superclass's method with the
                // current `this`, resolved at compile time up the `extends` chain.
                if let Expr::Member {
                    object,
                    property: PropertyKey::Ident(key) | PropertyKey::Str(key),
                    ..
                } = &**callee
                    && matches!(&**object, Expr::Super(_))
                {
                    let sup = self
                        .super_class
                        .clone()
                        .ok_or(CompileError::Unsupported("super outside a subclass"))?;
                    let func = self
                        .resolve_method(&sup, key)
                        .ok_or(CompileError::Unsupported("super method not found"))?;
                    let m = self.alloc();
                    self.ops.push(Op::LoadFunc { dst: m, func });
                    let recv = self.this_reg;
                    let dst = self.alloc();
                    self.ops.push(Op::CallValueThis {
                        dst,
                        callee: m,
                        recv,
                        args,
                    });
                    return Ok(dst);
                }
                // `ClassName.staticMethod(args)` where `ClassName` isn't a local
                // in scope (e.g. referenced from inside another method) → a
                // direct static dispatch, since the class object isn't reachable.
                if let Expr::Member {
                    object,
                    property: PropertyKey::Ident(key) | PropertyKey::Str(key),
                    ..
                } = &**callee
                    && let Expr::Ident(cn) = &**object
                    && self.lookup(&cn.name).is_none()
                    && let Some(sid) = self.classes.get(&*cn.name).and_then(|info| {
                        info.statics
                            .iter()
                            .find(|(n, _)| n == &**key)
                            .map(|(_, id)| *id)
                    })
                {
                    let dst = self.alloc();
                    self.ops.push(Op::Call {
                        dst,
                        func: sid,
                        args,
                    });
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
                // Reassigning a `const` binding is a TypeError; route the program
                // to the tree-walker, which enforces it at the right point.
                if let Expr::Ident(id) = &**target
                    && self.lookup(&id.name).is_some_and(|b| b.konst)
                {
                    return Err(CompileError::Unsupported("assignment to const"));
                }
                // Logical assignment (`&&=`/`||=`/`??=`) short-circuits.
                if matches!(
                    op,
                    AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::NullishAssign
                ) {
                    // `cond(cur)` decides whether the assignment fires.
                    let cond = |this: &mut Self, cur: Reg| -> Result<Reg, CompileError> {
                        Ok(match op {
                            AssignOp::AndAssign => cur,
                            AssignOp::OrAssign => {
                                let n = this.alloc();
                                this.ops.push(Op::Not { dst: n, a: cur });
                                n
                            }
                            _ => {
                                let nn = this.emit_not_nullish(cur)?;
                                let g = this.alloc();
                                this.ops.push(Op::Not { dst: g, a: nn });
                                g
                            }
                        })
                    };
                    match &**target {
                        Expr::Ident(id) => {
                            let b = self
                                .lookup(&id.name)
                                .ok_or_else(|| CompileError::Undefined(String::from(&*id.name)))?;
                            let cur = self.read_var(b);
                            let c = cond(self, cur)?;
                            let jf = self.emit_jump_if_false(c);
                            let v = self.expr(value)?;
                            self.write_var(b, v);
                            self.patch(jf);
                            return Ok(self.read_var(b));
                        }
                        Expr::Member {
                            object, property, ..
                        } => {
                            let obj = self.expr(object)?;
                            let cur = self.member_read(obj, property)?;
                            let c = cond(self, cur)?;
                            let jf = self.emit_jump_if_false(c);
                            let v = self.expr(value)?;
                            self.member_write(obj, property, v)?;
                            self.patch(jf);
                            return self.member_read(obj, property);
                        }
                        _ => return Err(CompileError::Unsupported("logical assign target")),
                    }
                }
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
                        self.member_write(obj, property, src)?;
                        Ok(src)
                    }
                    // Destructuring assignment (`[a, b] = …`, `({ x } = …)`).
                    Expr::Array { .. } | Expr::Object { .. } if !compound => {
                        let v = self.expr(value)?;
                        self.assign_pattern(target, v)?;
                        Ok(v)
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
                // `++`/`--` operate on `ToNumber(x)`, so coerce first: this makes
                // `"5"++` yield 5 then 6 (numeric) rather than the string concat
                // `"51"`, and the postfix result is the coerced number.
                let raw = self.read_var(b);
                let cur = self.alloc();
                self.ops.push(Op::CallNative {
                    dst: cur,
                    native: NB_NUMBER,
                    args: alloc::vec![raw],
                });
                let old = self.alloc();
                self.ops.push(Op::Move { dst: old, src: cur });
                let next = self.emit_binop(bop, cur, one)?;
                self.write_var(b, next);
                Ok(if *prefix { next } else { old })
            }
            Expr::This(_) => Ok(self.this_reg),
            // A regex literal `/source/flags`.
            Expr::Regex { pattern, flags, .. } => {
                let dst = self.alloc();
                self.ops.push(Op::NewRegExp {
                    dst,
                    source: String::from(&**pattern),
                    flags: String::from(&**flags),
                });
                Ok(dst)
            }
            // A comma sequence: evaluate all, yield the last.
            Expr::Sequence { expressions, .. } => {
                let mut last = self.constant(NanBox::undefined())?;
                for e in expressions {
                    last = self.expr(e)?;
                }
                Ok(last)
            }
            // A tagged template `tag`a${x}b`` → `tag(strings, x, …)`.
            Expr::TaggedTemplate { tag, quasi, .. } => {
                let strings = self.alloc();
                self.ops.push(Op::NewArray {
                    dst: strings,
                    len: 0,
                });
                for q in &quasi.quasis {
                    // An invalid escape yields no cooked value (`undefined`); `.raw`
                    // still preserves it (ES2018 tagged-template revision).
                    let s = match q.cooked.as_deref() {
                        Some(c) => self.constant_str(c),
                        None => self.constant(NanBox::undefined())?,
                    };
                    self.ops.push(Op::ArrayPush {
                        arr: strings,
                        src: s,
                    });
                }
                let mut args = alloc::vec![strings];
                for e in &quasi.expressions {
                    args.push(self.expr(e)?);
                }
                // Dispatch the tag like a normal call.
                let dst = self.alloc();
                if let Expr::Ident(id) = &**tag
                    && self.lookup(&id.name).is_none()
                    && let Some(&func) = self.fn_ids.get(&*id.name)
                {
                    self.ops.push(Op::Call { dst, func, args });
                } else {
                    let callee = self.expr(tag)?;
                    self.ops.push(Op::CallValue { dst, callee, args });
                }
                Ok(dst)
            }
            // `new C(args)` for a known plain class: create the instance, install
            // its methods, run the constructor with `this` = instance.
            Expr::New {
                callee, arguments, ..
            } => {
                let Expr::Ident(id) = &**callee else {
                    return Err(CompileError::Unsupported("new on non-class"));
                };
                // Built-in `new Map()` / `new Set()` (optionally seeded).
                if (&*id.name == "Map" || &*id.name == "Set")
                    && self.classes.get(&*id.name).is_none()
                {
                    let is_set = &*id.name == "Set";
                    let seed = match arguments.first() {
                        Some(crate::ast::Argument::Item(e)) => Some(self.expr(e)?),
                        _ => None,
                    };
                    let dst = self.alloc();
                    self.ops.push(Op::NewCollection { dst, is_set, seed });
                    return Ok(dst);
                }
                // Built-in `Error` family → an object with `name`/`message`.
                if self.classes.get(&*id.name).is_none()
                    && matches!(
                        &*id.name,
                        "Error"
                            | "TypeError"
                            | "RangeError"
                            | "SyntaxError"
                            | "ReferenceError"
                            | "EvalError"
                            | "URIError"
                    )
                {
                    let msg = match arguments.first() {
                        Some(crate::ast::Argument::Item(e)) => self.expr(e)?,
                        _ => self.constant_str(""),
                    };
                    let dst = self.alloc();
                    self.ops.push(Op::NewObject { dst });
                    let name = self.constant_str(&id.name);
                    self.ops.push(Op::SetProp {
                        obj: dst,
                        key: String::from("name"),
                        src: name,
                    });
                    self.ops.push(Op::SetProp {
                        obj: dst,
                        key: String::from("message"),
                        src: msg,
                    });
                    return Ok(dst);
                }
                // `new RegExp(source, flags)` (string-literal args) → a regex.
                if &*id.name == "RegExp" && self.classes.get("RegExp").is_none() {
                    let lit = |a: Option<&crate::ast::Argument>| match a {
                        Some(crate::ast::Argument::Item(Expr::Str { value, .. })) => {
                            Some(String::from(&**value))
                        }
                        None => Some(String::new()),
                        _ => None,
                    };
                    if let (Some(source), Some(flags)) =
                        (lit(arguments.first()), lit(arguments.get(1)))
                    {
                        let dst = self.alloc();
                        self.ops.push(Op::NewRegExp { dst, source, flags });
                        return Ok(dst);
                    }
                    return Err(CompileError::Unsupported("new RegExp with dynamic args"));
                }
                // `new Array(...items)` → an array of the arguments.
                if &*id.name == "Array" && self.classes.get("Array").is_none() {
                    let dst = self.alloc();
                    // `new Array(n)` with one argument: a number is the length, any
                    // other value is the sole element (decided at runtime).
                    if arguments.len() == 1
                        && let crate::ast::Argument::Item(e) = &arguments[0]
                    {
                        let arg = self.expr(e)?;
                        self.ops.push(Op::NewArrayCtor { dst, arg });
                        return Ok(dst);
                    }
                    self.ops.push(Op::NewArray { dst, len: 0 });
                    for a in arguments {
                        if let crate::ast::Argument::Item(e) = a {
                            let v = self.expr(e)?;
                            self.ops.push(Op::ArrayPush { arr: dst, src: v });
                        }
                    }
                    return Ok(dst);
                }
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
                let instance = self.alloc();
                self.ops.push(Op::NewObject { dst: instance });
                // Tag the instance with its class (for `instanceof`).
                self.ops.push(Op::SetClassTag {
                    obj: instance,
                    class_id: info.class_id,
                });
                // Walk the `extends` chain root→derived and install each class's
                // methods, so a derived method overrides an inherited one.
                let mut chain: Vec<String> = Vec::new();
                let mut cur = Some(String::from(&*id.name));
                while let Some(name) = cur {
                    cur = self.classes.get(&name).and_then(|c| c.super_name.clone());
                    chain.push(name);
                }
                for name in chain.iter().rev() {
                    // A native superclass (e.g. `extends Error`/`Array`) isn't a user
                    // class; the bytecode VM can't model the native chain (`super(...)`
                    // into a native constructor), so bail the whole program to the
                    // tree-walker, which handles it.
                    let Some(cls) = self.classes.get(name) else {
                        return Err(CompileError::Unsupported("extends a native class"));
                    };
                    let methods = cls.methods.clone();
                    let accessors = cls.accessors.clone();
                    for (mname, mid) in &methods {
                        let m = self.alloc();
                        self.ops.push(Op::LoadFunc { dst: m, func: *mid });
                        self.ops.push(Op::SetProp {
                            obj: instance,
                            key: mname.clone(),
                            src: m,
                        });
                    }
                    // Install getter/setter accessors.
                    for (aname, getter_id, setter_id) in &accessors {
                        let load = |this: &mut Self, id: Option<u32>| match id {
                            Some(fid) => {
                                let r = this.alloc();
                                this.ops.push(Op::LoadFunc { dst: r, func: fid });
                                r
                            }
                            None => this.constant(NanBox::undefined()).expect("const"),
                        };
                        let getter = load(self, *getter_id);
                        let setter = load(self, *setter_id);
                        self.ops.push(Op::DefineAccessor {
                            obj: instance,
                            key: aname.clone(),
                            getter,
                            setter,
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
                // An invalid escape is allowed only in a *tagged* template; in a plain
                // template literal it is a SyntaxError — defer to the tree-walker, which
                // raises it at evaluation.
                if t.quasis.iter().any(|q| q.cooked.is_none()) {
                    return Err(CompileError::Unsupported(
                        "invalid escape in template literal",
                    ));
                }
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
            Expr::Function(f) => {
                let nm = f.id.as_ref().map_or("", |i| i.name.as_ref());
                self.make_closure(&f.params, &f.body, f.is_async, nm)
            }
            Expr::Arrow(a) => {
                let body: Vec<Stmt> = match &a.body {
                    crate::ast::ArrowBody::Block(b) => b.clone(),
                    crate::ast::ArrowBody::Expr(e) => alloc::vec![Stmt::Return {
                        argument: Some(Box::new((**e).clone())),
                        span: crate::common::Span::point(0),
                    }],
                };
                self.make_closure(&a.params, &body, a.is_async, "")
            }
            // The optional-chain boundary. Allocate the result (defaulting to
            // `undefined`), then compile the inner chain: each `?.` link with a
            // nullish base jumps here, leaving the result `undefined`; otherwise the
            // chain's value is moved in. This makes a nullish `?.` base short-circuit
            // the *entire* remaining chain (`a?.b.c.d`).
            Expr::OptChain { expr, .. } => {
                let result = self.alloc();
                self.ops.push(Op::LoadConst {
                    dst: result,
                    value: NanBox::undefined(),
                });
                self.optchain_ends.push(Vec::new());
                let v = self.expr(expr)?;
                self.ops.push(Op::Move {
                    dst: result,
                    src: v,
                });
                let sites = self.optchain_ends.pop().unwrap_or_default();
                let end = self.ops.len();
                for s in sites {
                    self.patch_to(s, end);
                }
                Ok(result)
            }
            _ => Err(CompileError::Unsupported("expression")),
        }
    }

    /// Compiles a nested function into the shared table and emits the code to
    /// build a closure over its captured cells.
    /// Like `expr`, but when `e` is an *anonymous* function/arrow bound to a simple
    /// identifier, the binding name is inferred as the function's `.name`
    /// (NamedEvaluation: `const f = () => {}` ⇒ `f.name === "f"`).
    fn expr_named(&mut self, e: &Expr, target: &BindingTarget) -> Result<Reg, CompileError> {
        if let BindingTarget::Ident(id) = target {
            match e {
                Expr::Function(f) if f.id.is_none() => {
                    return self.make_closure(&f.params, &f.body, f.is_async, id.name.as_ref());
                }
                Expr::Arrow(a) => {
                    let body: Vec<Stmt> = match &a.body {
                        crate::ast::ArrowBody::Block(b) => b.clone(),
                        crate::ast::ArrowBody::Expr(ex) => alloc::vec![Stmt::Return {
                            argument: Some(Box::new((**ex).clone())),
                            span: crate::common::Span::point(0),
                        }],
                    };
                    return self.make_closure(&a.params, &body, a.is_async, id.name.as_ref());
                }
                _ => {}
            }
        }
        self.expr(e)
    }

    fn make_closure(
        &mut self,
        params: &[crate::ast::Param],
        body: &[Stmt],
        is_async: bool,
        name: &str,
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
                rest_from: None,
                is_async: false,
                length: 0,
                name: alloc::string::String::new(),
            });
            (p.len() - 1) as u32
        };
        let proto = Compiler::compile_fn_inner(
            &self.fn_ids,
            &self.classes,
            &self.protos,
            params,
            &captures,
            body,
            false,
            None,
            &[],
            None,
            is_async,
        )?;
        let mut proto = proto;
        proto.name = alloc::string::String::from(name);
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

    /// Emits the construction of an error object `{ name, message }` and a
    /// `Throw` of it; returns a dummy register (the throw unwinds, so the value
    /// is never read).
    fn emit_throw_error(&mut self, error_name: &str, message: &str) -> Reg {
        let obj = self.alloc();
        self.ops.push(Op::NewObject { dst: obj });
        let name = self.constant_str(error_name);
        self.ops.push(Op::SetProp {
            obj,
            key: String::from("name"),
            src: name,
        });
        let msg = self.constant_str(message);
        self.ops.push(Op::SetProp {
            obj,
            key: String::from("message"),
            src: msg,
        });
        self.ops.push(Op::Throw { src: obj });
        obj
    }

    /// A fresh register holding a new heap string.
    fn constant_str(&mut self, s: &str) -> Reg {
        let r = self.alloc();
        self.ops.push(Op::NewString {
            dst: r,
            value: String::from(s),
        });
        r
    }

    /// Emits `!(v === null || v === undefined)` into a fresh register.
    fn emit_not_nullish(&mut self, v: Reg) -> Result<Reg, CompileError> {
        let null = self.constant(NanBox::null())?;
        let undef = self.constant(NanBox::undefined())?;
        let is_null = self.alloc();
        self.ops.push(Op::StrictEq {
            dst: is_null,
            a: v,
            b: null,
        });
        let is_undef = self.alloc();
        self.ops.push(Op::StrictEq {
            dst: is_undef,
            a: v,
            b: undef,
        });
        // nullish = is_null || is_undef; go = !nullish.
        let go = self.alloc();
        // `is_null ? true : is_undef` collapses to `is_null || is_undef`; here
        // compute via Not(Not(is_null) && Not(is_undef))? Simpler: go = is_null
        // false-path. Use a small sequence.
        let not_null = self.alloc();
        self.ops.push(Op::Not {
            dst: not_null,
            a: is_null,
        });
        let not_undef = self.alloc();
        self.ops.push(Op::Not {
            dst: not_undef,
            a: is_undef,
        });
        // go = not_null && not_undef (both must hold to access).
        self.ops.push(Op::Move {
            dst: go,
            src: not_null,
        });
        let jf = self.emit_jump_if_false(go);
        self.ops.push(Op::Move {
            dst: go,
            src: not_undef,
        });
        self.patch(jf);
        Ok(go)
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
            // Realm-backed ops (`**`, bitwise, loose `==`/`!=`).
            BinaryOp::Exp => self.ops.push(Op::ValueBin {
                dst,
                op: VB_POW,
                a,
                b,
            }),
            BinaryOp::BitAnd => self.ops.push(Op::ValueBin {
                dst,
                op: VB_BIT_AND,
                a,
                b,
            }),
            BinaryOp::BitOr => self.ops.push(Op::ValueBin {
                dst,
                op: VB_BIT_OR,
                a,
                b,
            }),
            BinaryOp::BitXor => self.ops.push(Op::ValueBin {
                dst,
                op: VB_BIT_XOR,
                a,
                b,
            }),
            BinaryOp::Shl => self.ops.push(Op::ValueBin {
                dst,
                op: VB_SHL,
                a,
                b,
            }),
            BinaryOp::Shr => self.ops.push(Op::ValueBin {
                dst,
                op: VB_SHR,
                a,
                b,
            }),
            BinaryOp::Ushr => self.ops.push(Op::ValueBin {
                dst,
                op: VB_USHR,
                a,
                b,
            }),
            BinaryOp::EqEq => self.ops.push(Op::ValueBin {
                dst,
                op: VB_LOOSE_EQ,
                a,
                b,
            }),
            BinaryOp::NotEq => self.ops.push(Op::ValueBin {
                dst,
                op: VB_LOOSE_NEQ,
                a,
                b,
            }),
            BinaryOp::In | BinaryOp::Instanceof => {
                return Err(CompileError::Unsupported("in / instanceof"));
            }
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
            AssignOp::ModAssign => BinaryOp::Mod,
            AssignOp::ExpAssign => BinaryOp::Exp,
            AssignOp::ShlAssign => BinaryOp::Shl,
            AssignOp::ShrAssign => BinaryOp::Shr,
            AssignOp::UshrAssign => BinaryOp::Ushr,
            AssignOp::BitAndAssign => BinaryOp::BitAnd,
            AssignOp::BitOrAssign => BinaryOp::BitOr,
            AssignOp::BitXorAssign => BinaryOp::BitXor,
            // `&&=`/`||=`/`??=` are handled separately (short-circuit), and `=`
            // isn't compound.
            _ => return Err(CompileError::Unsupported("compound assignment operator")),
        })
    }

    /// Compiles a member read `obj.key` / `obj[i]` (with `.length` mapped to the
    /// array-length op).
    fn member_read(&mut self, obj: Reg, property: &PropertyKey) -> Result<Reg, CompileError> {
        let dst = self.alloc();
        match property {
            PropertyKey::Computed(e) => {
                let key = self.expr(e)?;
                self.ops.push(Op::GetKey { dst, obj, key });
            }
            _ => {
                let key = static_key(property)?;
                if key == "length" {
                    self.ops.push(Op::ArrayLen { dst, arr: obj });
                } else if key == "size" {
                    self.ops.push(Op::CollectionSize { dst, recv: obj });
                } else {
                    self.ops.push(Op::GetProp { dst, obj, key });
                }
            }
        }
        Ok(dst)
    }

    /// The function id of method `name` on class `class` or the nearest ancestor
    /// that declares it (for `super.method()`).
    fn resolve_method(&self, class: &str, name: &str) -> Option<u32> {
        let mut cur = Some(String::from(class));
        while let Some(cname) = cur {
            let info = self.classes.get(&cname)?;
            if let Some((_, id)) = info.methods.iter().find(|(n, _)| n == name) {
                return Some(*id);
            }
            cur = info.super_name.clone();
        }
        None
    }

    /// Materializes a class's *static* side as a value object bound to `name`
    /// (static methods + static fields), so `Name.staticMember` works. Used by
    /// both class declarations and top-level `const Name = class {…}`.
    fn materialize_class(
        &mut self,
        name: &str,
        class: &crate::ast::Class,
    ) -> Result<(), CompileError> {
        let info = match self.classes.get(name) {
            Some(i) => i.clone(),
            None => return Ok(()), // a class that fell back at scan time
        };
        let cobj = self.alloc();
        self.ops.push(Op::NewObject { dst: cobj });
        // Tag the constructor object `\0vmfn` (non-enumerable: `\0` keys are
        // filtered from enumeration) so `typeof Class === "function"`.
        let fn_tag = self.constant(NanBox::boolean(true))?;
        self.ops.push(Op::SetProp {
            obj: cobj,
            key: String::from("\u{0}vmfn"),
            src: fn_tag,
        });
        for (sname, sid) in &info.statics {
            let f = self.alloc();
            self.ops.push(Op::LoadFunc { dst: f, func: *sid });
            self.ops.push(Op::SetProp {
                obj: cobj,
                key: sname.clone(),
                src: f,
            });
        }
        for member in &class.body {
            if let crate::ast::ClassMember::Field(f) = member
                && f.is_static
            {
                let key = static_key(&f.key)?;
                let v = match &f.value {
                    Some(e) => self.expr(e)?,
                    None => self.constant(NanBox::undefined())?,
                };
                self.ops.push(Op::SetProp {
                    obj: cobj,
                    key,
                    src: v,
                });
            }
        }
        let b = self.declare(name);
        self.write_var(b, cobj);
        Ok(())
    }

    /// Writes `src` to `obj.key` / `obj[i]` (the mirror of `member_read`).
    fn member_write(
        &mut self,
        obj: Reg,
        property: &PropertyKey,
        src: Reg,
    ) -> Result<(), CompileError> {
        match property {
            PropertyKey::Computed(e) => {
                let key = self.expr(e)?;
                self.ops.push(Op::SetKey { obj, key, src });
            }
            _ => {
                let key = static_key(property)?;
                self.ops.push(Op::SetProp { obj, key, src });
            }
        }
        Ok(())
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
    fn optional_chain_short_circuits_whole_chain() {
        // A nullish `?.` base short-circuits the entire chain (not just one link).
        assert_eq!(bc("let n = null; n?.a.b.c"), "undefined");
        assert_eq!(bc("let o = {a:{b:7}}; o?.a?.b"), "7");
        assert_eq!(bc("let o = {a:{b:7}}; o?.x?.y.z"), "undefined");
        // `??` over a short-circuited chain takes the fallback.
        assert_eq!(bc("let n = null; n?.a.b ?? 42"), "42");
        // A genuinely-present intermediate that is nullish still throws on a plain
        // access — caught here to confirm it is NOT silently short-circuited.
        assert_eq!(
            bc("let o = {a:{}}; let r; try { r = o.a?.zzz.qqq; } catch (e) { r = 'threw'; } r"),
            "threw"
        );
    }

    /// A hot integer function is called enough times to tier up; with the JIT
    /// feature on Linux x86-64 it executes as native code, otherwise it stays in
    /// the interpreter — either way the result must be identical.
    #[test]
    fn hot_integer_function_matches_interpreter() {
        // f(i,2) = i*2 + i - 2 = 3i - 2; called 40 times (> TIER_UP_THRESHOLD).
        // sum_{i=0..39}(3i - 2) = 3*780 - 80 = 2260.
        assert_eq!(
            bc("function f(a,b){ return a*b + a - b; } \
                let t = 0; \
                for (let i = 0; i < 40; i = i + 1) { t = t + f(i, 2); } \
                t"),
            "2260"
        );
        // A hot function that overflows the safe-integer range must still be
        // correct (the JIT deopts to the interpreter when it would diverge).
        assert_eq!(
            bc("function sq(x){ return x*x; } \
                let r = 0; \
                for (let i = 0; i < 20; i = i + 1) { r = sq(100000000); } \
                r"),
            "10000000000000000"
        );
        // A hot function mixing integers and a non-integer argument: the call with
        // a fractional value must yield the exact float result, not a deopt error.
        assert_eq!(
            bc("function g(a,b){ return a + b; } \
                let s = 0; \
                for (let i = 0; i < 30; i = i + 1) { s = g(i, 0.5); } \
                s"),
            "29.5"
        );
    }

    /// A hot function that *calls another function* — with the JIT on Linux
    /// x86-64 the callee is compiled first and wired as a native call inside the
    /// caller's machine code; either way the result must match the interpreter.
    #[test]
    fn hot_function_with_a_call_matches_interpreter() {
        // triple(x) = x*3; f(x) = triple(x) + x = 4x. Called 40 times (> threshold).
        // sum_{i=0..39}(4i) = 4 * 780 = 3120.
        assert_eq!(
            bc("function triple(x){ return x*3; } \
                function f(x){ return triple(x) + x; } \
                let t = 0; \
                for (let i = 0; i < 40; i = i + 1) { t = t + f(i); } \
                t"),
            "3120"
        );
        // Nested calls: a(x)=x+1; b(x)=a(x)*2; c(x)=b(x)+a(x). c(x)=3(x+1)=3x+3.
        // sum_{i=0..29}(3i+3) = 3*435 + 90 = 1395.
        assert_eq!(
            bc("function a(x){ return x+1; } \
                function b(x){ return a(x)*2; } \
                function c(x){ return b(x) + a(x); } \
                let t = 0; \
                for (let i = 0; i < 30; i = i + 1) { t = t + c(i); } \
                t"),
            "1395"
        );
        // A callee that overflows the safe-integer range must still give the right
        // answer: the callee deopts → the caller's post-call guard deopts → the
        // interpreter runs it (and `1e16 + 1` rounds back to `1e16` in f64, exactly
        // as the interpreter computes — the point is JIT and interpreter agree).
        assert_eq!(
            bc("function sq(x){ return x*x; } \
                function h(x){ return sq(x) + 1; } \
                let r = 0; \
                for (let i = 0; i < 20; i = i + 1) { r = h(100000000); } \
                r"),
            "10000000000000000"
        );
    }

    /// A hot function using `<=` / `>=` / `!==` — which compile to `Lt`/`StrictEq`
    /// + `Not` — must JIT and match the interpreter (the `Not` lowers to `Eqz`).
    #[test]
    fn hot_function_with_not_matches_interpreter() {
        // count how many of 0..40 are <= 20: 21.
        assert_eq!(
            bc("function le(a,b){ return a <= b ? 1 : 0; } \
                let c = 0; \
                for (let i = 0; i < 40; i = i + 1) { c = c + le(i, 20); } \
                c"),
            "21"
        );
        // f(x) = (x >= 10) ? x : 0; sum over 0..20 → sum_{i=10..19} i = 145.
        assert_eq!(
            bc("function ge(x){ return x >= 10 ? x : 0; } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + ge(i); } \
                s"),
            "145"
        );
    }

    /// A hot function using `===` / `!==` (compiled to `StrictEq` [+ `Not`]) must
    /// JIT (lowered to `Eq` [/ `Eqz`]) and match the interpreter.
    #[test]
    fn hot_function_with_strict_eq_matches_interpreter() {
        // count i in 0..40 with i === 7 → exactly 1.
        assert_eq!(
            bc("function is7(x){ return x === 7 ? 1 : 0; } \
                let c = 0; \
                for (let i = 0; i < 40; i = i + 1) { c = c + is7(i); } \
                c"),
            "1"
        );
        // count i in 0..40 with i !== 0 → 39.
        assert_eq!(
            bc("function nz(x){ return x !== 0 ? 1 : 0; } \
                let c = 0; \
                for (let i = 0; i < 40; i = i + 1) { c = c + nz(i); } \
                c"),
            "39"
        );
        // Loose `==` between integers (lowered to `Eq`): count i == 3 over 0..40 → 1.
        assert_eq!(
            bc("function eq3(x){ return x == 3 ? 1 : 0; } \
                let c = 0; \
                for (let i = 0; i < 40; i = i + 1) { c = c + eq3(i); } \
                c"),
            "1"
        );
    }

    /// A hot function using unary minus (`-x`) must JIT (lowered to `Neg`) and
    /// match the interpreter.
    #[test]
    fn hot_function_with_neg_matches_interpreter() {
        // g(x) = -x + 100; sum over 0..20 → sum(100 - i) = 2000 - 190 = 1810.
        assert_eq!(
            bc("function g(x){ return -x + 100; } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + g(i); } \
                s"),
            "1810"
        );
    }

    /// A hot function using JS bitwise `&` / `|` / `^` must JIT (lowered to
    /// `Bit32` with ToInt32 truncation) and match the interpreter.
    #[test]
    fn hot_function_with_bitwise_matches_interpreter() {
        // mask(x) = (x & 7) | 16; sum over 0..40 of mask(i):
        // each i contributes (i%8) + 16; sum_{i=0..39}(i&7) = 5*(0+..+7)=140; +40*16=640 → 780.
        assert_eq!(
            bc("function mask(x){ return (x & 7) | 16; } \
                let s = 0; \
                for (let i = 0; i < 40; i = i + 1) { s = s + mask(i); } \
                s"),
            "780"
        );
        // xor is its own inverse: (x ^ 255) ^ 255 == x for x in 0..20 → sum = 190.
        assert_eq!(
            bc("function rt(x){ return (x ^ 255) ^ 255; } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + rt(i); } \
                s"),
            "190"
        );
    }

    /// A hot function using `%` must JIT (lowered to `Mod`) and match the
    /// interpreter — including the JS sign-follows-dividend rule.
    #[test]
    fn hot_function_with_mod_matches_interpreter() {
        // count i in 0..40 with i % 7 == 0: i = 0,7,14,21,28,35 → 6.
        assert_eq!(
            bc("function isMul7(x){ return (x % 7) == 0 ? 1 : 0; } \
                let c = 0; \
                for (let i = 0; i < 40; i = i + 1) { c = c + isMul7(i); } \
                c"),
            "6"
        );
        // sum of (i % 10) for i in 0..20 → (0+..+9) + (0+..+9) = 90.
        assert_eq!(
            bc("function r(x){ return x % 10; } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + r(i); } \
                s"),
            "90"
        );
    }

    /// A hot function using bitwise-not `~x` must JIT (lowered to `BitNot32`).
    #[test]
    fn hot_function_with_bitnot_matches_interpreter() {
        // ~x == -x - 1; f(x) = ~x + x == -1 for every x. Sum over 0..20 → -20.
        assert_eq!(
            bc("function f(x){ return (~x) + x; } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + f(i); } \
                s"),
            "-20"
        );
    }

    /// A hot function using JS shifts `<<` / `>>` / `>>>` must JIT (lowered to
    /// `Shift32`) and match the interpreter.
    #[test]
    fn hot_function_with_shifts_matches_interpreter() {
        // f(x) = (x << 2) + (x >> 1); sum over 0..20.
        // sum(4i) = 4*190 = 760; sum(floor(i/2)) for i=0..19 = 90; total 850.
        assert_eq!(
            bc("function f(x){ return (x << 2) + (x >> 1); } \
                let s = 0; \
                for (let i = 0; i < 20; i = i + 1) { s = s + f(i); } \
                s"),
            "850"
        );
    }

    /// A hot float function (division, non-integer values) runs identically
    /// whether it takes the JIT's float path (with the `jit` feature on
    /// Linux x86-64) or the interpreter.
    #[test]
    fn hot_float_function_matches_interpreter() {
        // ratio(a,b) = (a + b) / b, called 30× past the tier-up threshold.
        assert_eq!(
            bc("function ratio(a,b){ return (a + b) / b; } \
                let r = 0; \
                for (let i = 0; i < 30; i = i + 1) { r = ratio(3, 2); } \
                r"),
            "2.5"
        );
        // A division producing a non-integer across many calls.
        assert_eq!(
            bc("function half(x){ return x / 2; } \
                let s = 0; \
                for (let i = 0; i < 25; i = i + 1) { s = half(7); } \
                s"),
            "3.5"
        );
        // A hot function with a *float loop* (fractional step + f64 comparison) —
        // exercises the JIT's float branch path. sum of x for x in 0,0.5,..<3 = 7.5.
        assert_eq!(
            bc(
                "function tri(n){ let s = 0.0; for (let x = 0.0; x < n; x = x + 0.5) { s = s + x; } return s; } \
                let r = 0; \
                for (let i = 0; i < 20; i = i + 1) { r = tri(3); } \
                r"
            ),
            "7.5"
        );
        // A hot float function using unary minus on a fractional value (the float
        // path's `Neg`): g(x) = -x + 0.5; sum of g(0.5) over 25 calls = 0.
        assert_eq!(
            bc("function g(x){ return -x + 0.5; } \
                let s = 0; \
                for (let i = 0; i < 25; i = i + 1) { s = g(0.5); } \
                s"),
            "0"
        );
        // A hot float function using `<=` (Lt + Not → the float path's `Eqz`):
        // count how many of 0,0.5,..,9.5 are <= 4.5 → 10.
        assert_eq!(
            bc("function le(a,b){ return a <= b ? 1 : 0; } \
                let c = 0; \
                for (let x = 0.0; x < 10.0; x = x + 0.5) { c = c + le(x, 4.5); } \
                c"),
            "10"
        );
        // A hot float function using `===` (the float path's `Eq`): count how many
        // of 0,0.5,..,4.5 equal 2.5 → exactly 1.
        assert_eq!(
            bc("function eq(a,b){ return a === b ? 1 : 0; } \
                let c = 0; \
                for (let x = 0.0; x < 5.0; x = x + 0.5) { c = c + eq(x, 2.5); } \
                c"),
            "1"
        );
        // A hot float function using `Math.sqrt`/`Math.abs` (the float path's `Sqrt`
        // and `Abs`, lowered from the Math native calls): h(-6.25) = sqrt(6.25) = 2.5.
        assert_eq!(
            bc("function h(x){ return Math.sqrt(Math.abs(x)); } \
                let r = 0; \
                for (let i = 0; i < 40; i = i + 1) { r = h(-6.25); } \
                r"),
            "2.5"
        );
        // A hot float function using `Math.min`/`Math.max` (the float path's `Min`
        // and `Max`): clamp(x) to [0,10]. clamp(13.5)=10, then clamp(-2.5)=0.
        assert_eq!(
            bc(
                "function clamp(x){ return Math.max(0.0, Math.min(10.0, x)); } \
                let r = 0; \
                for (let i = 0; i < 40; i = i + 1) { r = clamp(13.5); } \
                let r2 = 0; \
                for (let j = 0; j < 40; j = j + 1) { r2 = clamp(-2.5); } \
                r + r2"
            ),
            "10"
        );
        // A hot float function using `Math.floor`/`Math.ceil` (the float path's
        // `Floor`/`Ceil` via SSE4.1 `roundsd`, or the interpreter fallback on older
        // CPUs — either way the result matches): floor(3.2) + ceil(3.7) = 3 + 4 = 7.
        assert_eq!(
            bc(
                "function fc(x){ return Math.floor(x) + Math.ceil(x + 0.5); } \
                let r = 0; \
                for (let i = 0; i < 40; i = i + 1) { r = fc(3.2); } \
                r"
            ),
            "7"
        );
    }

    #[test]
    fn bytecode_valueof_in_operators() {
        // User valueOf/toString honored in the bytecode VM's operators.
        assert_eq!(bc("let m={valueOf(){return 5;}}; m - 2"), "3");
        assert_eq!(bc("let m={valueOf(){return 5;}}; m + 1"), "6");
        assert_eq!(bc("let m={valueOf(){return 5;}}; ~m"), "-6");
        assert_eq!(bc("let m={valueOf(){return 5;}}; -m"), "-5");
        assert_eq!(bc("let m={valueOf(){return 5;}}; m & 3"), "1");
        assert_eq!(bc("let s={toString(){return 'x';}}; s + '!'"), "x!");
        // Increment/decrement are numeric.
        assert_eq!(bc("let x='5'; ++x"), "6");
        assert_eq!(bc("let x='3'; let y=x++; y + ',' + x"), "3,4");
    }

    #[test]
    fn bytecode_arithmetic_object_coercion() {
        assert_eq!(bc("[5] - 2"), "3");
        assert_eq!(bc("[10] / 2"), "5");
        assert_eq!(bc("[10] % 3"), "1");
        assert_eq!(bc("[6] & 3"), "2");
        assert_eq!(bc("[2] ** 3"), "8");
        assert_eq!(bc("'5' - 2"), "3");
        assert_eq!(bc("({a:1}) - 1"), "NaN");
    }

    #[test]
    fn bytecode_relational_object_coercion() {
        assert_eq!(bc("[5] < 10"), "true");
        assert_eq!(bc("[1] < [2]"), "true");
        assert_eq!(bc("[10] < [9]"), "true");
        assert_eq!(bc("({}) < 1"), "false");
    }

    #[test]
    fn bytecode_loose_eq_object_coercion() {
        assert_eq!(bc("String([] == false)"), "true");
        assert_eq!(bc("String([] == 0)"), "true");
        assert_eq!(bc("String({} == 0)"), "false");
        assert_eq!(bc("String({} == {})"), "false");
        assert_eq!(bc("String([1,2] == '1,2')"), "true");
    }

    #[test]
    fn bytecode_array_string_index() {
        // `arr["0"]` reads the element (canonical numeric string key).
        assert_eq!(bc("let a=[10,20,30]; a['1']"), "20");
        assert_eq!(bc("let a=[10,20,30]; let k='2'; a[k]"), "30");
        // Non-canonical keys are not indices.
        assert_eq!(bc("let a=[10,20,30]; String(a['00'])"), "undefined");
        assert_eq!(bc("let a=[1,2,3]; a['length']"), "3");
    }

    #[test]
    fn bytecode_regex_split_matches_tree_walker() {
        // The bytecode VM's regex split splices capture groups and keeps the
        // boundary char on zero-width (lookahead) matches, like the tree-walker.
        assert_eq!(bc("'a1b2c3'.split(/(\\d)/).join(',')"), "a,1,b,2,c,3,");
        assert_eq!(
            bc("'camelCaseWord'.split(/(?=[A-Z])/).join('|')"),
            "camel|Case|Word"
        );
        assert_eq!(bc("'aXbYc'.split(/[XY]/).join(',')"), "a,b,c");
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
    fn typeof_of_a_user_function_is_function_on_the_vm() {
        // These compile fully to bytecode (no tree-walker fallback), so they
        // exercise the `\0vmfn` tag on `LoadFunc`/`MakeClosure` closures that makes
        // `typeof` report a function rather than the backing array.
        let (out, _) = execute(
            "var f = function(){}; function g(){} var h = () => 1;
             console.log(typeof f); console.log(typeof g); console.log(typeof h);
             console.log(typeof []); console.log(typeof {});
             console.log(Object.keys(f).indexOf('\u{0}vmfn'));",
        )
        .expect("ok");
        // Functions report `function`; arrays/objects unchanged; the internal tag
        // is non-enumerable (absent from Object.keys).
        assert_eq!(out, "function\nfunction\nfunction\nobject\nobject\n-1\n");
    }

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

    #[cfg(feature = "std")]
    #[test]
    fn bytecode_async_promise_and_runtime_errors() {
        use crate::nbvm::execute;
        // async function returns a promise; .then runs in the microtask drain.
        let (out, _) = execute(
            "async function f() { return 7; } f().then((v) => { console.log('async:' + v); });",
        )
        .expect("ok");
        assert_eq!(out, "async:7\n");
        // Promise.resolve + chained then.
        let (out, _) = execute(
            "Promise.resolve(1).then((v) => v + 1).then((v) => { console.log('chain:' + v); });",
        )
        .expect("ok");
        assert_eq!(out, "chain:2\n");
        // Promise.reject + catch.
        let (out, _) =
            execute("Promise.reject('boom').catch((e) => { console.log('caught:' + e); });")
                .expect("ok");
        assert_eq!(out, "caught:boom\n");
        // Runtime ReferenceError on an undefined identifier (caught).
        assert_eq!(
            bc("let r = ''; try { undefinedXYZ; } catch (e) { r = e.name; } r"),
            "ReferenceError"
        );
        // Runtime TypeError on null property access (caught).
        assert_eq!(
            bc("let r = ''; try { null.field; } catch (e) { r = e.name; } r"),
            "TypeError"
        );
        // A known global (Math) is NOT a ReferenceError — it works.
        assert_eq!(bc("Math.max(3, 7)"), "7");
    }

    #[test]
    fn bytecode_class_expressions() {
        // Class expression assigned to a const, used with `new`.
        assert_eq!(
            bc("const Pair = class { constructor(a, b) { this.a = a; this.b = b; } sum() { return this.a + this.b; } };
                new Pair(10, 20).sum()"),
            "30"
        );
        // Class expression with a getter.
        assert_eq!(
            bc("const Box = class { constructor(v) { this._v = v; } get value() { return this._v * 2; } };
                new Box(21).value"),
            "42"
        );
        // A static method referencing the class by name from another static.
        assert_eq!(
            bc("class MathUtil { static square(x) { return x * x; } static sumOfSquares(a, b) { return MathUtil.square(a) + MathUtil.square(b); } }
                MathUtil.sumOfSquares(3, 4)"),
            "25"
        );
    }

    #[test]
    fn bytecode_class_statics() {
        // A static method.
        assert_eq!(
            bc("class M { static add(a, b) { return a + b; } } M.add(3, 4)"),
            "7"
        );
        // A static field.
        assert_eq!(bc("class C { static version = 42; } C.version"), "42");
        // A static factory returning an instance.
        assert_eq!(
            bc(
                "class P { constructor(x) { this.x = x; } static of(x) { return new P(x); } }
                P.of(9).x"
            ),
            "9"
        );
        // Static method + static field together; instance methods unaffected.
        assert_eq!(
            bc("class Counter {
                  static total = 0;
                  constructor() { this.n = 1; }
                  static describe() { return 'counter'; }
                  bump() { return this.n + 1; }
                }
                Counter.describe() + ':' + Counter.total + ':' + new Counter().bump()"),
            "counter:0:2"
        );
    }

    #[test]
    fn bytecode_class_accessors() {
        // A getter computed from instance state.
        assert_eq!(
            bc("class C { constructor(w, h) { this.w = w; this.h = h; }
                  get area() { return this.w * this.h; } }
                new C(3, 4).area"),
            "12"
        );
        // A setter mutating instance state.
        assert_eq!(
            bc("class T { constructor() { this.c = 0; }
                  get count() { return this.c; }
                  set count(v) { this.c = v * 2; } }
                let t = new T(); t.count = 5; t.count"),
            "10"
        );
        // Getter/setter pair backing a private-ish field.
        assert_eq!(
            bc("class Box { constructor(v) { this._v = v; }
                  get value() { return this._v; }
                  set value(x) { this._v = x + 1; } }
                let b = new Box(10); b.value = 20; b.value"),
            "21"
        );
        // An inherited getter.
        assert_eq!(
            bc("class A { get kind() { return 'A'; } }
                class B extends A {}
                new B().kind"),
            "A"
        );
    }

    #[test]
    fn bytecode_class_fields() {
        // A field initializer, no explicit constructor (synthetic ctor).
        assert_eq!(
            bc("class Box { value = 42; get() { return this.value; } } new Box().get()"),
            "42"
        );
        // Fields run before the constructor body.
        assert_eq!(
            bc(
                "class C { base = 10; constructor(n) { this.total = this.base + n; } }
                new C(5).total"
            ),
            "15"
        );
        // A field with no initializer is `undefined`, then set in the ctor.
        assert_eq!(
            bc("class C { x; constructor() { this.x = 7; } } new C().x"),
            "7"
        );
        // Multiple fields in declaration order.
        assert_eq!(
            bc(
                "class P { a = 1; b = 2; c = 3; sum() { return this.a + this.b + this.c; } }
                new P().sum()"
            ),
            "6"
        );
    }

    #[test]
    fn bytecode_instanceof() {
        // Direct instance.
        assert_eq!(bc("class A {} new A() instanceof A"), "true");
        // Subclass is an instance of its base and itself.
        assert_eq!(
            bc("class A {} class B extends A {}
                let b = new B(); '' + (b instanceof B) + ',' + (b instanceof A)"),
            "true,true"
        );
        // A base is not an instance of its subclass.
        assert_eq!(
            bc("class A {} class B extends A {} new A() instanceof B"),
            "false"
        );
        // Unrelated classes.
        assert_eq!(bc("class A {} class C {} new A() instanceof C"), "false");
        // Three-level chain.
        assert_eq!(
            bc("class A {} class B extends A {} class C extends B {}
                let c = new C(); '' + (c instanceof A) + (c instanceof B) + (c instanceof C)"),
            "truetruetrue"
        );
    }

    #[test]
    fn bytecode_sequence_tagged_labeled_instanceof_error() {
        // Sequence expression.
        assert_eq!(bc("let x = (1, 2, 3); x"), "3");
        assert_eq!(bc("let a = 0; let b = (a = 5, a + 1); b"), "6");
        // Tagged template.
        assert_eq!(
            bc("function tag(s, ...v) { let out = s[0]; for (let i = 0; i < v.length; i++) { out += '<' + v[i] + '>' + s[i + 1]; } return out; }
                tag`a${1}b${2}c`"),
            "a<1>b<2>c"
        );
        // Labeled loop with labeled continue/break.
        assert_eq!(
            bc(
                "let hits = ''; outer: for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) { if (j === 1) continue outer; hits += i + '' + j + ','; } } hits"
            ),
            "00,10,20,"
        );
        assert_eq!(
            bc(
                "let n = 0; search: for (let i = 0; i < 5; i++) { for (let j = 0; j < 5; j++) { n++; if (i + j === 3) break search; } } n"
            ),
            "4"
        );
        // instanceof on built-in errors.
        assert_eq!(bc("new TypeError('x') instanceof TypeError"), "true");
        assert_eq!(bc("new RangeError('x') instanceof TypeError"), "false");
        assert_eq!(bc("new TypeError('x') instanceof Error"), "true");
    }

    #[test]
    fn bytecode_default_and_rest_params() {
        // Default parameters.
        assert_eq!(bc("function f(a, b = 10) { return a + b; } f(5)"), "15");
        assert_eq!(bc("function f(a, b = 10) { return a + b; } f(5, 20)"), "25");
        assert_eq!(bc("let g = (x, y = x * 2) => x + y; g(3)"), "9");
        // Rest parameters.
        assert_eq!(
            bc("function sum(...nums) { return nums.reduce((a, b) => a + b, 0); } sum(1, 2, 3, 4)"),
            "10"
        );
        assert_eq!(
            bc(
                "function tag(first, ...rest) { return first + ':' + rest.join(','); } tag('a', 'b', 'c')"
            ),
            "a:b,c"
        );
        // Default + rest together.
        assert_eq!(
            bc("function f(a, b = 2, ...rest) { return a + b + rest.length; } f(1)"),
            "3"
        );
        assert_eq!(
            bc("function f(a, b = 2, ...rest) { return a + b + rest.length; } f(1, 5, 9, 9, 9)"),
            "9"
        );
    }

    #[test]
    fn bytecode_super_method() {
        // super.method() calls the base implementation.
        assert_eq!(
            bc("class A { greet() { return 'hi'; } }
                class B extends A { greet() { return super.greet() + '!'; } }
                new B().greet()"),
            "hi!"
        );
        // super reaches through a multi-level chain and combines with this.
        assert_eq!(
            bc("class Shape { describe() { return 'shape'; } }
                class Round extends Shape {
                  constructor(r) { super(); this.r = r; }
                  describe() { return super.describe() + ' r=' + this.r; }
                }
                new Round(5).describe()"),
            "shape r=5"
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
    fn bytecode_forin_builtins_and_destructuring_assign() {
        // for-in over object keys and array indices.
        assert_eq!(
            bc("let o = { a: 1, b: 2, c: 3 }; let s = ''; for (const k in o) { s += k; } s"),
            "abc"
        );
        assert_eq!(
            bc("let sum = 0; for (const i in [9, 8, 7]) { sum += Number(i); } sum"),
            "3"
        );
        // new Error / new Array.
        assert_eq!(bc("new Error('boom').message"), "boom");
        assert_eq!(bc("new TypeError('bad').name"), "TypeError");
        assert_eq!(bc("new Array(1, 2, 3).join(',')"), "1,2,3");
        // Destructuring assignment (existing vars): swap, rest, object.
        assert_eq!(bc("let a = 1, b = 2; [a, b] = [b, a]; a + ',' + b"), "2,1");
        assert_eq!(
            bc("let h, t; [h, ...t] = [1, 2, 3, 4]; h + '|' + t.join(',')"),
            "1|2,3,4"
        );
        assert_eq!(bc("let x, y; ({ x, y } = { x: 10, y: 20 }); x + y"), "30");
        // Object destructuring assignment into members.
        assert_eq!(
            bc("let p = {}; ({ a: p.x, b: p.y } = { a: 3, b: 4 }); p.x * p.y"),
            "12"
        );
        // Destructuring function parameters.
        assert_eq!(
            bc("function f({ a, b }) { return a + b; } f({ a: 3, b: 4 })"),
            "7"
        );
        assert_eq!(bc("let g = ([x, y]) => x * y; g([3, 4])"), "12");
        assert_eq!(
            bc("[{ n: 1 }, { n: 2 }, { n: 3 }].map(({ n }) => n * 10).join(',')"),
            "10,20,30"
        );
    }

    #[cfg(feature = "regex")]
    #[test]
    fn bytecode_regex() {
        // test / exec.
        assert_eq!(bc(r"/^\d{4}-\d{2}-\d{2}$/.test('2026-06-04')"), "true");
        assert_eq!(bc(r"/^\d+$/.test('12a')"), "false");
        assert_eq!(bc(r"/(\w+)\s+(\w+)/.exec('hello world')[2]"), "world");
        assert_eq!(bc(r"/(\w+)\s+(\w+)/.exec('hello world').index"), "0");
        // String.match (single + global).
        assert_eq!(bc(r"'a1b2c3'.match(/\d/g).join('')"), "123");
        assert_eq!(bc(r"'key=value'.match(/(\w+)=(\w+)/)[2]"), "value");
        // String.replace with captures + global.
        assert_eq!(
            bc(r"'John Smith'.replace(/(\w+)\s(\w+)/, '$2, $1')"),
            "Smith, John"
        );
        assert_eq!(bc(r"'aaa'.replace(/a/g, 'b')"), "bbb");
        // split / search.
        assert_eq!(bc(r"'1, 2,3 ,4'.split(/\s*,\s*/).join('|')"), "1|2|3|4");
        assert_eq!(bc(r"'find the needle'.search(/needle/)"), "9");
        assert_eq!(bc(r"'nope'.search(/xyz/)"), "-1");
        // new RegExp + instanceof RegExp.
        assert_eq!(bc(r#"new RegExp('\\d+', 'g').test('abc123')"#), "true");
        assert_eq!(bc(r"/x/ instanceof RegExp"), "true");
        assert_eq!(bc(r"[] instanceof Array"), "true");
        assert_eq!(bc(r"new Map() instanceof Map"), "true");
        assert_eq!(bc(r"new Set() instanceof Set"), "true");
        assert_eq!(bc(r"[] instanceof Map"), "false");
    }

    #[test]
    fn bytecode_number_string_array_object_statics() {
        // Number.* predicates (no coercion).
        assert_eq!(
            bc("'' + Number.isInteger(42) + Number.isInteger(4.2)"),
            "truefalse"
        );
        assert_eq!(
            bc("'' + Number.isFinite(1) + Number.isNaN(0 / 0)"),
            "truetrue"
        );
        // String.fromCharCode.
        assert_eq!(bc("String.fromCharCode(75, 97)"), "Ka");
        // Array.from (Set, array, string) and Array.isArray.
        assert_eq!(bc("Array.from(new Set([1, 1, 2])).length"), "2");
        assert_eq!(bc("Array.from('abc').join('-')"), "a-b-c");
        assert_eq!(
            bc("'' + Array.isArray([1]) + Array.isArray(5)"),
            "truefalse"
        );
        // Object.fromEntries (round-trip with entries).
        assert_eq!(bc("Object.fromEntries([['k', 'v'], ['n', 1]]).k"), "v");
        assert_eq!(
            bc("let o = Object.fromEntries(Object.entries({ a: 1, b: 2 })); o.a + o.b"),
            "3"
        );
    }

    #[test]
    fn bytecode_json_and_object_features() {
        // JSON round-trip.
        assert_eq!(
            bc("JSON.stringify({ a: 1, b: [2, 3], c: 'x' })"),
            "{\"a\":1,\"b\":[2,3],\"c\":\"x\"}"
        );
        assert_eq!(bc("JSON.parse('{\"x\": 42}').x"), "42");
        assert_eq!(bc("JSON.parse('[1, 2, 3]')[1]"), "2");
        assert_eq!(
            bc("let o = JSON.parse(JSON.stringify({ n: 7, s: 'hi' })); o.n + o.s"),
            "7hi"
        );
        // Object spread (merge + override).
        assert_eq!(
            bc("let a = { x: 1, y: 2 }; let b = { ...a, y: 9, z: 3 }; b.x + ',' + b.y + ',' + b.z"),
            "1,9,3"
        );
        // Computed object keys.
        assert_eq!(
            bc("let k = 'dyn'; let o = { [k]: 5, ['a' + 'b']: 6 }; o.dyn + o.ab"),
            "11"
        );
        // Object-literal getter.
        assert_eq!(
            bc("let o = { _v: 10, get v() { return this._v * 2; } }; o.v"),
            "20"
        );
        // Dynamic string-keyed access.
        assert_eq!(
            bc("let o = { foo: 1, bar: 2 }; let key = 'bar'; o[key] = 9; o['foo'] + o.bar"),
            "10"
        );
    }

    #[test]
    fn bytecode_map_and_set() {
        // Map: set/get/has/size, chaining, delete.
        assert_eq!(
            bc("let m = new Map(); m.set('a', 1).set('b', 2); m.get('a') + m.get('b')"),
            "3"
        );
        assert_eq!(
            bc("let m = new Map(); m.set('k', 9); '' + m.has('k') + ',' + m.size"),
            "true,1"
        );
        assert_eq!(
            bc("let m = new Map(); m.set('x', 1); m.delete('x'); m.size"),
            "0"
        );
        // Map seeded from pairs; iterate entries.
        assert_eq!(
            bc(
                "let m = new Map([['a', 1], ['b', 2]]); let s = 0; m.forEach((v) => { s += v; }); s"
            ),
            "3"
        );
        assert_eq!(
            bc("let m = new Map([['a', 1], ['b', 2]]); m.keys().join(',')"),
            "a,b"
        );
        // Set: add/has/size, dedup, seeded.
        assert_eq!(bc("let s = new Set(); s.add(1).add(2).add(1); s.size"), "2");
        assert_eq!(
            bc("let s = new Set([1, 2, 3, 2, 1]); s.size + ',' + s.has(3)"),
            "3,true"
        );
        // String .length on the bytecode path.
        assert_eq!(bc("'hello'.length"), "5");
    }

    #[test]
    fn bytecode_delete_and_member_logical_assign() {
        // delete on object/array members.
        assert_eq!(
            bc("let o = { a: 1, b: 2 }; delete o.a; '' + ('a' in o) + ',' + o.b"),
            "false,2"
        );
        assert_eq!(
            bc("let o = { x: 1 }; let r = delete o.x; '' + r + ',' + ('x' in o)"),
            "true,false"
        );
        // computed delete.
        assert_eq!(
            bc("let o = { k: 5 }; let key = 'k'; delete o[key]; 'k' in o"),
            "false"
        );
        // Logical assignment to a member.
        assert_eq!(bc("let o = { a: 0 }; o.a ||= 7; o.a"), "7");
        assert_eq!(bc("let o = { a: 3 }; o.a &&= 9; o.a"), "9");
        assert_eq!(
            bc("let o = {}; o.cache ??= 'computed'; o.cache"),
            "computed"
        );
        // ??= only assigns when nullish (memoization pattern).
        assert_eq!(bc("let c = { v: 5 }; c.v ??= 99; c.v"), "5");
        // for-of with array destructuring.
        assert_eq!(
            bc("let s = 0; for (const [a, b] of [[1, 2], [3, 4]]) { s += a * b; } s"),
            "14"
        );
        // for-of with object destructuring.
        assert_eq!(
            bc(
                "let names = ''; for (const { name } of [{ name: 'a' }, { name: 'b' }]) { names += name; } names"
            ),
            "ab"
        );
    }

    #[test]
    fn bytecode_object_namespace_and_in() {
        // Object.keys / values / entries.
        assert_eq!(bc("Object.keys({ a: 1, b: 2 }).join(',')"), "a,b");
        assert_eq!(bc("Object.values({ a: 1, b: 2 }).join(',')"), "1,2");
        assert_eq!(
            bc("Object.entries({ a: 1, b: 2 }).map((e) => e[0] + '=' + e[1]).join(',')"),
            "a=1,b=2"
        );
        // Object.assign copies and returns the target.
        assert_eq!(
            bc("let t = Object.assign({}, { a: 1 }, { b: 2 }); t.a + t.b"),
            "3"
        );
        // `in` operator on objects and arrays.
        assert_eq!(bc("'x' in { x: 1 }"), "true");
        assert_eq!(bc("'y' in { x: 1 }"), "false");
        assert_eq!(bc("0 in [10, 20]"), "true");
        assert_eq!(bc("5 in [10, 20]"), "false");
        // `in` guarding access (memoization-style).
        assert_eq!(
            bc("let c = {}; c.k = 7; let r = ('k' in c) ? c.k : -1; r"),
            "7"
        );
    }

    #[test]
    fn optimizer_folds_constant_arithmetic() {
        // 6 * 7 over two constants folds to a single LoadConst 42.
        let ops = alloc::vec![
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(6.0)
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(7.0)
            },
            Op::Mul { dst: 2, a: 0, b: 1 },
            Op::Return { src: 2 },
        ];
        let opt = optimize_ops(&ops);
        match &opt[2] {
            Op::LoadConst { dst, value } => {
                assert_eq!(*dst, 2);
                assert_eq!(value.as_number(), Some(42.0));
            }
            other => panic!("expected a folded LoadConst, got {other:?}"),
        }
    }

    #[test]
    fn optimizer_folds_addvalue_and_loose_eq() {
        // `+` over two number constants folds to addition.
        let ops = alloc::vec![
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(40.0)
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(2.0)
            },
            Op::AddValue { dst: 2, a: 0, b: 1 },
            Op::Return { src: 2 },
        ];
        let opt = optimize_ops(&ops);
        assert!(matches!(&opt[2], Op::LoadConst { value, .. } if value.as_number() == Some(42.0)));

        // Loose `==` over two number constants folds to a boolean.
        let ops = alloc::vec![
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(3.0)
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(3.0)
            },
            Op::ValueBin {
                dst: 2,
                op: VB_LOOSE_EQ,
                a: 0,
                b: 1
            },
            Op::Return { src: 2 },
        ];
        let opt = optimize_ops(&ops);
        assert!(matches!(&opt[2], Op::LoadConst { value, .. } if value.to_boolean()));
    }

    #[test]
    fn optimizer_propagates_folded_constants() {
        // (10 - 4) then * 7 — the Sub's folded constant feeds the Mul.
        let ops = alloc::vec![
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(10.0)
            },
            Op::LoadConst {
                dst: 1,
                value: NanBox::number(4.0)
            },
            Op::Sub { dst: 2, a: 0, b: 1 }, // -> 6
            Op::LoadConst {
                dst: 3,
                value: NanBox::number(7.0)
            },
            Op::Mul { dst: 4, a: 2, b: 3 }, // -> 42
            Op::Return { src: 4 },
        ];
        let opt = optimize_ops(&ops);
        assert!(matches!(&opt[2], Op::LoadConst { value, .. } if value.as_number() == Some(6.0)));
        assert!(matches!(&opt[4], Op::LoadConst { value, .. } if value.as_number() == Some(42.0)));
    }

    #[test]
    fn optimizer_does_not_fold_across_basic_blocks() {
        // A constant set before a jump target must NOT be assumed at the target
        // (control could arrive without having run the LoadConst).
        let ops = alloc::vec![
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(2.0)
            },
            Op::Jump { target: 3 },
            Op::LoadConst {
                dst: 0,
                value: NanBox::number(99.0)
            }, // skipped path
            // index 3 is a jump target → constants cleared here.
            Op::Mul { dst: 1, a: 0, b: 0 },
            Op::Return { src: 1 },
        ];
        let opt = optimize_ops(&ops);
        // The Mul at the leader stays a Mul (not folded on a stale constant).
        assert!(matches!(&opt[3], Op::Mul { .. }));
    }

    #[test]
    fn tier_up_preserves_results_over_many_calls() {
        // A function called past the tier-up threshold must keep returning the
        // right values (the optimized body is semantically identical).
        assert_eq!(
            bc("function sq(x) { return x * x; }
                let s = 0;
                for (let i = 0; i < 12; i++) { s += sq(i); }
                s"),
            "506" // sum of squares 0..=11
        );
        // A const-folding-heavy body run hot still computes correctly.
        assert_eq!(
            bc("function f() { return 6 * 7 - 2; }
                let t = 0;
                for (let i = 0; i < 20; i++) { t += f(); }
                t"),
            "800" // 40 * 20
        );
    }

    #[test]
    fn bytecode_more_operators() {
        // `**`, `%`, bitwise, shifts.
        assert_eq!(bc("2 ** 10"), "1024");
        assert_eq!(bc("17 % 5"), "2");
        assert_eq!(bc("6 & 3"), "2");
        assert_eq!(bc("5 | 2"), "7");
        assert_eq!(bc("5 ^ 1"), "4");
        assert_eq!(bc("1 << 4"), "16");
        assert_eq!(bc("256 >> 2"), "64");
        // Loose equality.
        assert_eq!(bc("'' + (1 == 1) + ',' + (1 != 2)"), "true,true");
        // typeof / unary +.
        assert_eq!(bc("typeof 42"), "number");
        assert_eq!(bc("typeof 'hi'"), "string");
        assert_eq!(bc("typeof undefinedVar"), "undefined");
        assert_eq!(bc("+'15' + 5"), "20");
        assert_eq!(bc("~5"), "-6");
        // Global value identifiers.
        assert_eq!(bc("undefined === undefined"), "true");
        assert_eq!(bc("'' + Infinity"), "Infinity");
        // Nullish coalescing.
        assert_eq!(bc("let x = null; x ?? 'fallback'"), "fallback");
        assert_eq!(bc("let x = 0; x ?? 'fallback'"), "0"); // 0 isn't nullish
        // Compound bitwise / logical assignment.
        assert_eq!(bc("let n = 12; n &= 10; n"), "8");
        assert_eq!(bc("let n = 8; n |= 1; n"), "9");
        assert_eq!(bc("let n = 1; n <<= 3; n"), "8");
        assert_eq!(bc("let x = 0; x ||= 5; x"), "5");
        assert_eq!(bc("let x = 1; x &&= 9; x"), "9");
        assert_eq!(bc("let x = null; x ??= 7; x"), "7");
        // parseInt / parseFloat / isNaN / isFinite globals.
        assert_eq!(bc("parseInt('42px')"), "42");
        assert_eq!(bc("parseInt('ff', 16)"), "255");
        assert_eq!(bc("parseFloat('3.14xyz')"), "3.14");
        assert_eq!(bc("'' + isNaN(0 / 0) + ',' + isFinite(1)"), "true,true");
    }

    #[test]
    fn bytecode_array_spread_and_optional_chaining() {
        // Array spread.
        assert_eq!(bc("let a = [1, 2, 3]; [...a].join(',')"), "1,2,3");
        assert_eq!(bc("let a = [2, 3]; [1, ...a, 4].join(',')"), "1,2,3,4");
        assert_eq!(
            bc("let a = [1]; let b = [2, 3]; [...a, ...b].join(',')"),
            "1,2,3"
        );
        assert_eq!(bc("[...[1, 2], ...[3, 4], 5].length"), "5");
        // Optional chaining on a member.
        assert_eq!(bc("let o = { x: { y: 7 } }; o?.x?.y"), "7");
        assert_eq!(bc("let o = { x: null }; '' + (o?.x?.y)"), "undefined");
        assert_eq!(bc("let o = null; '' + (o?.x)"), "undefined");
        // Optional chaining with a fallback via ??-style (|| here).
        assert_eq!(bc("let o = {}; o?.missing || 'default'"), "default");
        // Spread of a function-returned array.
        assert_eq!(
            bc("function nums() { return [10, 20]; } [...nums(), 30].join(',')"),
            "10,20,30"
        );
    }

    #[test]
    fn bytecode_destructuring() {
        // Array destructuring with defaults, holes, and nesting.
        assert_eq!(bc("let [a, b] = [1, 2]; a + b"), "3");
        assert_eq!(bc("let [a, , c] = [1, 2, 3]; a + c"), "4");
        assert_eq!(bc("let [a, b = 9] = [1]; a + b"), "10");
        assert_eq!(bc("let [[a], [b]] = [[1], [2]]; a + b"), "3");
        // Array rest pattern.
        assert_eq!(
            bc("let [h, ...t] = [1, 2, 3, 4]; h + '|' + t.join(',')"),
            "1|2,3,4"
        );
        assert_eq!(
            bc("let [, , ...rest] = [1, 2, 3, 4, 5]; rest.join(',')"),
            "3,4,5"
        );
        // Object rest pattern.
        assert_eq!(
            bc("let { a, ...rest } = { a: 1, b: 2, c: 3 }; a + '|' + Object.keys(rest).join(',')"),
            "1|b,c"
        );
        assert_eq!(
            bc("let { a, b, ...rest } = { a: 1, b: 2, c: 3, d: 4 }; rest.c + rest.d"),
            "7"
        );
        // Object destructuring with shorthand, rename, and default.
        assert_eq!(bc("let { x, y } = { x: 1, y: 2 }; x + y"), "3");
        assert_eq!(bc("let { a: p, b: q } = { a: 10, b: 20 }; p + q"), "30");
        assert_eq!(bc("let { m = 7 } = {}; m"), "7");
        assert_eq!(bc("let { p: { q } } = { p: { q: 42 } }; q"), "42");
        // Destructuring a function result / swap-via-array.
        assert_eq!(
            bc("function pair() { return [3, 4]; } let [a, b] = pair(); a * b"),
            "12"
        );
        // Destructuring inside a loop body.
        assert_eq!(
            bc("let s = 0; for (const p of [[1, 2], [3, 4]]) { let [a, b] = p; s += a * b; } s"),
            "14"
        );
    }

    #[test]
    fn bytecode_array_and_string_methods() {
        // Higher-order array methods with closures, all on the bytecode path.
        assert_eq!(bc("[1, 2, 3, 4].map((x) => x * 2).join(',')"), "2,4,6,8");
        assert_eq!(
            bc("[1, 2, 3, 4, 5].filter((x) => x % 2 === 0).join(',')"),
            "2,4"
        );
        assert_eq!(bc("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
        assert_eq!(bc("[1, 2, 3].reduce((a, b) => a + b)"), "6"); // no seed
        assert_eq!(
            bc("let s = 0; [10, 20, 30].forEach((x) => { s += x; }); s"),
            "60"
        );
        assert_eq!(bc("[5, 8, 2].find((x) => x > 6)"), "8");
        assert_eq!(
            bc("'' + [1, 2, 3].some((x) => x > 2) + ',' + [1, 2, 3].every((x) => x > 0)"),
            "true,true"
        );
        // Mutation + non-callback methods.
        assert_eq!(
            bc("let a = [1, 2]; a.push(3); a.push(4); a.join('')"),
            "1234"
        );
        assert_eq!(bc("let a = [1, 2, 3]; a.pop(); a.join('')"), "12");
        assert_eq!(
            bc("[1, 2, 3].includes(2) + ',' + [1, 2, 3].indexOf(3)"),
            "true,2"
        );
        assert_eq!(bc("[1, 2].concat([3, 4], 5).join('')"), "12345");
        assert_eq!(bc("[1, 2, 3].reverse().join('')"), "321");
        // A map → filter → reduce pipeline in bytecode.
        assert_eq!(
            bc("[1, 2, 3, 4, 5].map((x) => x * x).filter((x) => x > 4).reduce((a, b) => a + b, 0)"),
            "50"
        );
        // String methods.
        assert_eq!(bc("'Hello'.toUpperCase()"), "HELLO");
        assert_eq!(bc("'WORLD'.toLowerCase()"), "world");
        assert_eq!(bc("'  hi  '.trim()"), "hi");
        assert_eq!(bc("'a,b,c'.split(',').join('|')"), "a|b|c");
        assert_eq!(bc("'abcabc'.indexOf('c')"), "2");
        assert_eq!(bc("'ab'.repeat(3)"), "ababab");
        assert_eq!(bc("'hello world'.includes('world')"), "true");
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
