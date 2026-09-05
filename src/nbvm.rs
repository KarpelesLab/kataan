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
use crate::ic::PropertyCache;
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
    /// `dst = a fresh array of every value produced by iterating `src`` — the
    /// built-in iteration of an array / typed array / string / `Map` / `Set`.
    /// For any other value (a user iterable whose `[Symbol.iterator]` the VM
    /// cannot resolve, or a non-iterable) this faults so the whole program
    /// re-runs on the reference tree-walker, which drives the full iterator
    /// protocol. Backs `for (… of …)` and `[...iterable]` over built-ins.
    IterValues { dst: Reg, src: Reg },
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
        /// The pattern source as WTF-8 bytes — it may hold a lone surrogate,
        /// which `RegExp.prototype.source` must reproduce exactly.
        source: Vec<u8>,
        flags: String,
    },
    /// `dst = a new empty object` (allocated in the realm's heap).
    NewObject { dst: Reg },
    /// `obj[key] = src` (own property set through the object's shape).
    SetProp { obj: Reg, key: String, src: Reg },
    /// `obj[key] = src` as a *non-enumerable* own property (e.g. a class's
    /// `prototype`/`constructor` back-links).
    SetHidden { obj: Reg, key: String, src: Reg },
    /// Sets the `[[Prototype]]` link of the object in `obj` to the object in
    /// `proto` (used to link a class instance to its `.prototype`).
    SetProto { obj: Reg, proto: Reg },
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
    /// Proper tail call to static function `func` with `args` (strict-mode PTC):
    /// instead of pushing a new activation, the callee *reuses* the current
    /// frame (the interpreter trampolines in `call_with_inner`), so unbounded
    /// tail recursion runs in O(1) native stack. Emitted only for a `return`
    /// (and its tail-transparent sub-expressions) in a strict, non-async
    /// function. No `dst`: it is the frame's final act.
    TailCall { func: u32, args: Vec<Reg> },
    /// Proper tail call through a function *value* held in `callee` (the indirect
    /// analogue of [`Op::TailCall`], mirroring [`Op::CallValue`]). If `callee`
    /// is not a plain VM function, it degrades to an ordinary call whose result
    /// is returned.
    TailCallValue { callee: Reg, args: Vec<Reg> },
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
/// `Op::ValueBin` op code for loose `!=` (exposed for the JIT's value lowering).
pub(crate) const VB_LOOSE_NEQ: u8 = 8;

// Discriminants for the numeric binary value-ops (`-`,`*`,`/`,`%`) shared by the
// interpreter (`Op::Sub`/`Mul`/`Div`/`Mod`) and the generic-JIT helper
// ([`jit_helper_arith`]), so the two tiers dispatch off one code path (`vm_arith`).
/// `-` (subtraction).
pub(crate) const GA_SUB: u8 = 0;
/// `*` (multiplication).
pub(crate) const GA_MUL: u8 = 1;
/// `/` (division).
pub(crate) const GA_DIV: u8 = 2;
/// `%` (remainder).
pub(crate) const GA_MOD: u8 = 3;

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
///
/// This mirrors the set the interpreter installs, so it goes stale whenever a
/// global is added — and a *missing* entry is silently wrong twice over: the
/// value path throws a spurious `ReferenceError`, and `typeof` answers
/// `"undefined"` for something that exists. `globals_match_installed_set` keeps
/// the two in sync by diffing this list against `globalThis` at runtime.
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
    "Iterator",
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
    "decodeURI",
    "decodeURIComponent",
    "encodeURI",
    "encodeURIComponent",
    "escape",
    "unescape",
    "Temporal",
    "Atomics",
    "SharedArrayBuffer",
    "Float16Array",
    "FinalizationRegistry",
    "DisposableStack",
    "AsyncDisposableStack",
    "ShadowRealm",
    "SuppressedError",
    "atob",
    "btoa",
    "eval",
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
    /// An in-flight thrown value stashed by a generic-JIT runtime helper (see
    /// [`jit_helper_add`]). The helper returns the reserved throw sentinel and
    /// leaves the value here; [`call_generic`] takes it back out as
    /// `Err(VmError::Thrown(..))`. `None` between helper faults.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    jit_pending: Option<NanBox>,
    /// An in-flight non-throw fault (or thrown value) stashed by a generic-JIT
    /// property helper ([`jit_helper_get_prop`]/[`jit_helper_set_prop`]). Unlike
    /// `jit_pending` (which only carries a thrown `NanBox` for `+`), this carries a
    /// full [`VmError`], so a `VmError::Unsupported` from `vm_set_prop` (a
    /// descriptor-aware case the tree-walker owns) propagates out of the JIT
    /// exactly as it would from the interpreter. `None` between helper faults.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    jit_pending_fault: Option<VmError>,
    /// The function table the currently-running generic-JIT body belongs to, as a
    /// raw pointer so a runtime helper (which only receives `ctx`) can reconstruct
    /// `&[FnProto]` to re-enter the interpreter (e.g. to run a user `valueOf`). Set
    /// by [`call_generic`] immediately before invoking native code; the whole
    /// program shares one table, so it is invariant across nested calls.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    jit_funcs: Option<*const [FnProto]>,
    /// Forward-safety GC root hook (`JIT_DESIGN.md` §5): a helper-call sequence
    /// would spill live NanBox temps here before re-entering, so an
    /// allocation-triggered collection could treat them as roots. **Not yet
    /// load-bearing** — today GC never runs mid-execution, so no JIT frame is ever
    /// exposed to a moving collection; the field is landed (and mirrored into the
    /// realm root set via `Realm::jit_shadow_roots`) so the wiring exists ahead of
    /// an allocation-triggered GC. The emitted code does not populate it yet.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[allow(dead_code)]
    jit_shadow: alloc::vec::Vec<u64>,
    /// Function-call nesting depth (recursion guard).
    call_depth: usize,
    /// Whether this context may run an allocation-triggered collection at a
    /// [`vm_safepoint`]. Only the `run_program*` entry points set it: they own the
    /// whole `Ctx` and their `funcs` table is compiled *before* the realm exists,
    /// so no bytecode constant can be a heap handle the collector would have to
    /// know about. The bare [`run`] entry (hand-built op arrays, used by tests and
    /// the snapshot path) leaves it `false`.
    gc_enabled: bool,
    /// Non-zero while a Rust frame between the VM loop and here holds live
    /// [`NanBox`]es that are *not* in a register window (today: the microtask
    /// drain, which pops a job into a local before invoking its handler).
    /// [`vm_safepoint`] refuses to collect while it is set.
    gc_lock: usize,
    /// The activation inputs of the **outermost** frame — its `args`, `captures`
    /// and `this` — which live in [`call_with_inner`]'s Rust locals rather than in
    /// the register window. Republished here on every (re)binding of that frame so
    /// [`vm_safepoint`] can root them. Only maintained at `call_depth == 1`, the
    /// only depth that can collect.
    top_frame_roots: Vec<NanBox>,
}

// The recursion-guard, handler-stack, JSON-depth, and string-length caps now
// live in `crate::limits::Limits` and are read live from `ctx.realm.limits`, so
// an embedder can tune them per realm.

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
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending_fault: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_funcs: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_shadow: alloc::vec::Vec::new(),
        call_depth: 0,
        gc_enabled: true,
        gc_lock: 0,
        top_frame_roots: Vec::new(),
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
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending_fault: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_funcs: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_shadow: alloc::vec::Vec::new(),
        call_depth: 0,
        gc_enabled: true,
        gc_lock: 0,
        top_frame_roots: Vec::new(),
    };
    let value = call(&mut ctx, funcs, id, args)?;
    // Run the promise event loop before returning (then-callbacks, async tails).
    drain_microtasks(&mut ctx, funcs)?;
    Ok((value, ctx.output))
}

fn call(ctx: &mut Ctx, funcs: &[FnProto], id: usize, args: &[NanBox]) -> Result<NanBox, VmError> {
    call_with(ctx, funcs, id, args, &[], NanBox::undefined())
}

/// The interpreter's real static-call path (`Op::Call`'s semantics), factored out
/// so the bytecode VM **and** the generic-JIT runtime helper ([`jit_helper_call`])
/// share one code path and can never diverge: a plain static dispatch to the
/// function-table entry `id` with `this` bound (no captures — a hoisted function
/// call carries none). Delegates to [`call_with`] (recursion guard, arity binding,
/// tier-up, the throw machinery), so any observable side effect runs exactly once
/// per evaluation and a callee throw propagates identically on both tiers.
fn vm_call(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    id: usize,
    this: NanBox,
    args: &[NanBox],
) -> Result<NanBox, VmError> {
    call_with(ctx, funcs, id, args, &[], this)
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
    if ctx.call_depth >= ctx.realm.limits.max_call_depth {
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
    // Owned copies of the activation inputs; a proper tail call (`FrameExit::Tail`)
    // rebinds these and loops *without* recursing, so unbounded tail recursion
    // reuses this one native frame (O(1) stack) rather than growing it.
    let mut id = id;
    let mut args: Vec<NanBox> = args.to_vec();
    let mut captures: Vec<NanBox> = captures.to_vec();
    let mut this_val = this_val;
    loop {
        // `id` can originate from a runtime value (an indirect call reads it out of
        // a callee array's slot 0), so a hostile script can make it point past the
        // function table. Surface a catchable TypeError instead of indexing OOB.
        let Some(proto) = funcs.get(id) else {
            let e = make_error(ctx.realm, "TypeError", "not a function");
            return Err(VmError::Thrown(e));
        };
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
        // Republish the outermost activation's inputs for the GC safepoint: they
        // live in this function's Rust locals, which the collector cannot see.
        // Only depth 1 can collect, so deeper frames skip the copy entirely.
        if ctx.gc_enabled && ctx.call_depth == 1 {
            ctx.top_frame_roots.clear();
            ctx.top_frame_roots.extend_from_slice(&args);
            ctx.top_frame_roots.extend_from_slice(&captures);
            ctx.top_frame_roots.push(this_val);
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
        // Native fast path (Phase G JIT): once a function is hot, try compiling it
        // to machine code. Eligible functions are pure straight-line/looping
        // integer arithmetic (no side effects), so running the native code is
        // observationally equivalent to the interpreter; a non-integer/overflowing
        // call deopts to `None` and we fall through to `run_frame`. (A function with
        // a tail call is never JIT-eligible, so this never intercepts the
        // trampoline.)
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        if optimized.is_some()
            && !proto.is_async
            && proto.rest_from.is_none()
            && proto.n_captures == 0
        {
            let mut stack = alloc::collections::BTreeSet::new();
            let cached = ensure_jit(&mut ctx.jit_cache, funcs, id, &mut stack);
            if let Some(jit) = cached {
                if jit.is_generic() {
                    // The generic (NanBox) tier re-enters the interpreter through
                    // runtime helpers, so it needs the live context. A `Some`
                    // outcome (value or thrown) is authoritative; `None` is an
                    // arity deopt → fall through to the interpreter.
                    if let Some(outcome) = call_generic(ctx, funcs, &jit, &args) {
                        return outcome;
                    }
                } else if let Some(result) = jit.call_guarded(&args) {
                    return Ok(result);
                }
            }
        }
        // An `async` function: its synchronous body runs to completion, and its
        // result (or thrown value) settles a returned `Promise`. (No `await` yet —
        // a body that awaits falls back at compile time.)
        if proto.is_async {
            let p = ctx.realm.new_promise();
            // The returned promise is a Rust local across the body; a safepoint
            // inside would otherwise not see it.
            ctx.top_frame_roots.push(NanBox::handle(p.to_raw()));
            match run_frame(ctx, funcs, body, &mut regs) {
                Ok(FrameExit::Return(ret)) => {
                    settle(ctx, p, ret.unwrap_or(NanBox::undefined()), true);
                }
                // An async body does not emit tail-call ops, but stay correct if
                // one ever reaches here: run it as an ordinary call and settle.
                Ok(FrameExit::Tail {
                    id: tid,
                    args: targs,
                    captures: tcaps,
                    this,
                }) => match call_with(ctx, funcs, tid, &targs, &tcaps, this) {
                    Ok(v) => settle(ctx, p, v, true),
                    Err(VmError::Thrown(e)) => settle(ctx, p, e, false),
                    Err(other) => return Err(other),
                },
                Err(VmError::Thrown(e)) => settle(ctx, p, e, false),
                Err(other) => return Err(other),
            }
            return Ok(NanBox::handle(p.to_raw()));
        }
        match run_frame(ctx, funcs, body, &mut regs)? {
            FrameExit::Return(v) => return Ok(v.unwrap_or(NanBox::undefined())),
            // Proper tail call: rebind the activation inputs and reuse this frame.
            FrameExit::Tail {
                id: tid,
                args: targs,
                captures: tcaps,
                this,
            } => {
                id = tid;
                args = targs;
                captures = tcaps;
                this_val = this;
            }
        }
    }
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
    // Resolve each statically-called function to its compiled code address, keyed by
    // the callee's ABI so a caller only ever direct-calls a matching signature (a
    // wrong-ABI direct call would mis-dispatch and crash):
    //   * `registry`         — Int-ABI callees, for the Int tier's `RegOp::Call`.
    //   * `generic_registry` — generic-ABI callees + their param count, for the
    //     generic tier's direct JIT→JIT call.
    // A Float-ABI callee is direct-callable by neither (the Float tier emits no
    // calls), so it is registered nowhere and its callers take the interpreter-
    // reentrant helper.
    let mut registry = alloc::collections::BTreeMap::new();
    let mut generic_registry: alloc::collections::BTreeMap<u32, (u64, usize)> =
        alloc::collections::BTreeMap::new();
    for op in &funcs[id].ops {
        if let Op::Call { func, .. } = op {
            let fid = *func as usize;
            if fid != id
                && fid < funcs.len()
                && let Some(j) = ensure_jit(cache, funcs, fid, stack)
            {
                match j.abi_kind() {
                    crate::jit::AbiKind::Int => {
                        registry.insert(*func, j.code_ptr() as u64);
                    }
                    crate::jit::AbiKind::Generic => {
                        generic_registry.insert(*func, (j.code_ptr() as u64, funcs[fid].n_params));
                    }
                    crate::jit::AbiKind::Float => {}
                }
            }
        }
    }
    stack.remove(&id);
    // Prefer the Int/Float tiers; fall back to the generic (NanBox) tier for a
    // function they reject but that is generic-eligible (param/const loads, `+`,
    // `Move`, plain property get/set, `Return`) — e.g. a 7+-parameter add or an
    // `obj.key` accessor. The generic tier re-enters the interpreter for
    // non-numeric operands and all property access via the runtime helpers, and
    // direct-calls a fellow generic-tier callee (`generic_registry`).
    let compiled = crate::jit::JitProto::compile_with_registry(&funcs[id], &registry)
        .or_else(|| {
            crate::jit::JitProto::compile_generic_with_registry(
                &funcs[id],
                &jit_generic_helpers(),
                &generic_registry,
            )
        })
        .map(alloc::rc::Rc::new);
    cache.insert(id, compiled.clone());
    compiled
}

/// How a single frame's execution ([`run_frame`]) finished normally: either it
/// `return`ed (or fell off the end), or it hit a tail call whose activation the
/// caller (`call_with_inner`) must set up by *reusing* this frame (the PTC
/// trampoline). Splitting the two lets the trampoline loop stay in the frame
/// owner, keeping the native stack flat under unbounded tail recursion.
enum FrameExit {
    /// A `return value;` (or falling off the end → `None`).
    Return(Option<NanBox>),
    /// A proper tail call: run function `id` next, in this same activation.
    Tail {
        id: usize,
        args: Vec<NanBox>,
        captures: Vec<NanBox>,
        this: NanBox,
    },
}

/// Why execution stopped abnormally.
#[derive(Clone, PartialEq, Debug)]
pub enum VmError {
    /// An arithmetic op saw a non-number operand (this toy VM has no coercion).
    NotANumber,
    /// A property op was used on a non-object operand.
    NotAnObject,
    /// A construct the VM cannot evaluate with correct semantics in-place (e.g. an
    /// `arr[i] = v` write to an array index that a `defineProperty` demoted or made
    /// an accessor, which needs the tree-walker's strict-aware / accessor-aware
    /// store). Faults the whole VM so the program re-runs on the reference engine —
    /// it is *not* a catchable JS error.
    Unsupported,
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
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_pending_fault: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_funcs: None,
        #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
        jit_shadow: alloc::vec::Vec::new(),
        call_depth: 0,
        // Hand-built op arrays may embed heap handles as `LoadConst` values that
        // this entry point cannot enumerate, so it never collects.
        gc_enabled: false,
        gc_lock: 0,
        top_frame_roots: Vec::new(),
    };
    match run_frame(&mut ctx, &[], program, &mut regs)? {
        FrameExit::Return(v) => Ok(v.unwrap_or(NanBox::undefined())),
        // A top-level program body never emits a tail call (there is no enclosing
        // function to reuse); degrade to an ordinary call if one ever appears.
        FrameExit::Tail {
            id,
            args,
            captures,
            this,
        } => call_with(&mut ctx, &[], id, &args, &captures, this),
    }
}

/// Collects every value produced by iterating `v`, for the bytecode VM's
/// `for (… of …)` and array/argument spread over **built-in** iterables: a real
/// array, a typed array, a string (per code point), or a `Map`/`Set` (a `Set`
/// yields its values, a `Map` `[key, value]` pairs).
///
/// Returns `None` for anything else — a user object carrying a `[Symbol.iterator]`
/// (whose well-known symbol key the VM does not track), a generator, or a
/// non-iterable. The caller turns that into a `VmError` so the whole program
/// re-runs on the reference tree-walker ([`crate::nbexec`]), which drives the
/// full iterator protocol (including iterator-close and user `.next()`).
fn vm_iterable_values(ctx: &mut Ctx, v: NanBox) -> Option<Vec<NanBox>> {
    let h = v.as_handle().map(Handle::from_raw)?;
    // A real array or typed-array view: snapshot its elements. A hole reads as
    // `undefined` (the array iterator does `Get(array, index)`), not the internal
    // hole sentinel.
    if let Some(mut elems) = ctx.realm.elements_vec(h) {
        for e in &mut elems {
            if e.is_hole() {
                *e = NanBox::undefined();
            }
        }
        return Some(elems);
    }
    // A string iterates one entry per Unicode code point (a lone surrogate is a
    // single one-unit string), exactly like the tree-walker.
    if let Some(bytes) = ctx.realm.string_bytes(h) {
        let mut out = Vec::new();
        for cp in crate::wtf8::code_points(&bytes) {
            let mut buf = Vec::new();
            crate::wtf8::encode_code_point(cp, &mut buf);
            out.push(NanBox::handle(ctx.realm.new_string_wtf8(buf).to_raw()));
        }
        return Some(out);
    }
    // `Map`/`Set` iterate their entries; the weak variants are not iterable.
    if !ctx.realm.collection_is_weak(h)
        && let Some(entries) = ctx.realm.collection_entries(h)
    {
        if ctx.realm.collection_is_set(h) == Some(true) {
            return Some(entries.iter().map(|(k, _)| *k).collect());
        }
        let mut out = Vec::with_capacity(entries.len());
        for (k, val) in entries {
            out.push(NanBox::handle(
                ctx.realm.new_array(alloc::vec![k, val]).to_raw(),
            ));
        }
        return Some(out);
    }
    None
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
            .is_some_and(|h| !realm.is_string_handle(h))
    };
    let is_prim = |realm: &Realm, v: NanBox| {
        v.as_number().is_some()
            || matches!(v.unpack(), crate::nanbox::Unpacked::Bool(_))
            || v.as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| realm.is_string_handle(h))
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
    if ctx.realm.is_string_handle(h) || ctx.realm.date_at(h).is_some() {
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
                    .is_some_and(|rh| ctx.realm.is_string_handle(rh));
            if is_prim {
                return res;
            }
        }
    }
    v
}

/// The interpreter's real `+` operator (`Op::AddValue`'s semantics), factored
/// out so the bytecode VM **and** the generic-JIT runtime helper
/// ([`jit_helper_add`]) share one code path and can never diverge: ToPrimitive
/// (default hint) on each operand — honoring a user `valueOf`/`toString` — then a
/// Symbol-coercion guard, then `Realm::add` (which picks string concatenation vs
/// numeric addition from the resulting primitives). Any observable side effect (a
/// user `valueOf`) therefore runs exactly once per evaluation.
fn vm_add(ctx: &mut Ctx, funcs: &[FnProto], a: NanBox, b: NanBox) -> Result<NanBox, VmError> {
    let x = to_primitive(ctx, funcs, a, true);
    let y = to_primitive(ctx, funcs, b, true);
    if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_STR_ERR) {
        return Err(VmError::Thrown(e));
    }
    Ok(ctx.realm.add(x, y))
}

/// The interpreter's real native-builtin dispatch (`Op::CallNative`'s semantics),
/// factored out so the bytecode VM **and** the generic-JIT runtime helper
/// ([`jit_helper_call_native`]) share one code path. The interpreter-aware natives
/// (`JSON.stringify`/`JSON.parse`/`Array.from`/`Number`, which run user code or
/// throw) are handled here — where `funcs` and the throw machinery are available —
/// exactly as the VM loop did inline; everything else defers to the pure
/// [`call_native`]. A user side effect runs exactly once per evaluation.
fn vm_call_native(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    native: u16,
    args: &[NanBox],
) -> Result<NanBox, VmError> {
    if native == NB_JSON_STRINGIFY {
        json_stringify(ctx, funcs, args)
    } else if native == NB_JSON_PARSE {
        json_parse(ctx, funcs, args)
    } else if native == NB_ARRAY_FROM {
        vm_array_from(ctx, funcs, args)
    } else if matches!(
        native,
        NB_OBJECT_KEYS | NB_OBJECT_VALUES | NB_OBJECT_ENTRIES
    ) {
        vm_object_kv(ctx, funcs, native, args)
    } else if native == NB_STRING {
        // `String(x)` — `ToString` with the object path running `ToPrimitive(x,
        // "string")` (its `toString`/`valueOf`, which may call a JS closure), so an
        // overridden/inherited `Function.prototype.toString`, `Array.prototype.
        // toString`, etc. is honored. Special cases: no argument → `""`, and a
        // Symbol argument yields its descriptive string (not a TypeError).
        if args.is_empty() {
            return Ok(NanBox::handle(ctx.realm.new_string("").to_raw()));
        }
        let arg = args[0];
        let s = if arg
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| ctx.realm.symbol_at(h).is_some())
        {
            ctx.realm.to_display_string(arg)
        } else if arg
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| !ctx.realm.is_string_handle(h) && ctx.realm.symbol_at(h).is_none())
        {
            let prim = to_primitive(ctx, funcs, arg, false);
            ctx.realm.to_display_string(prim)
        } else {
            ctx.realm.to_display_string(arg)
        };
        Ok(NanBox::handle(ctx.realm.new_string(&s).to_raw()))
    } else if native == NB_NUMBER {
        // `ToNumber` (`Number(x)`) on an object runs ToPrimitive(number) — its
        // `valueOf`/`toString` — which can call a JS closure; a Symbol/BigInt
        // operand throws a TypeError.
        let arg = args.first().copied().unwrap_or(NanBox::undefined());
        let prim = to_primitive(ctx, funcs, arg, true);
        let bad = prim.as_handle().map(Handle::from_raw).and_then(|h| {
            if ctx.realm.symbol_at(h).is_some() {
                Some(SYM_NUM_ERR)
            } else if ctx.realm.bigint_at(h).is_some() {
                Some("Cannot convert a BigInt value to a number")
            } else {
                None
            }
        });
        if let Some(msg) = bad {
            let e = make_error(ctx.realm, "TypeError", msg);
            return Err(VmError::Thrown(e));
        }
        Ok(NanBox::number(ctx.realm.to_number(prim)))
    } else {
        Ok(call_native(ctx, native, args))
    }
}

/// The interpreter's real `-`/`*`/`/`/`%` operators (`Op::Sub`/`Mul`/`Div`/`Mod`),
/// factored out so the bytecode VM **and** the generic-JIT runtime helper
/// ([`jit_helper_arith`]) share one code path and can never diverge: ToPrimitive
/// (number hint) on each operand — honoring a user `valueOf`/`toString` — then a
/// Symbol-coercion guard, then the realm's numeric op. `op` is one of the `GA_*`
/// discriminants. A user side effect runs exactly once per evaluation.
fn vm_arith(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    a: NanBox,
    b: NanBox,
    op: u8,
) -> Result<NanBox, VmError> {
    let x = to_primitive(ctx, funcs, a, true);
    let y = to_primitive(ctx, funcs, b, true);
    if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
        return Err(VmError::Thrown(e));
    }
    Ok(match op {
        GA_SUB => ctx.realm.sub(x, y),
        GA_MUL => ctx.realm.mul(x, y),
        GA_DIV => ctx.realm.div(x, y),
        _ => ctx.realm.rem(x, y),
    })
}

/// The interpreter's real relational `<` (`Op::Lt`'s semantics), factored out so
/// the bytecode VM **and** the generic-JIT runtime helper ([`jit_helper_lt`])
/// share one code path: ToPrimitive (number hint) each operand, a Symbol-coercion
/// guard, then `Realm::less_than` (string-vs-string ordering or numeric less-than).
/// `>`, `<=`, `>=` are compiled to `Op::Lt` (with operand swap and/or `Op::Not`),
/// so this single primitive backs all four relational operators. Returns a boolean
/// `NanBox`.
fn vm_lt(ctx: &mut Ctx, funcs: &[FnProto], a: NanBox, b: NanBox) -> Result<NanBox, VmError> {
    let x = to_primitive(ctx, funcs, a, true);
    let y = to_primitive(ctx, funcs, b, true);
    if let Some(e) = symbol_coercion_error(ctx.realm, x, y, SYM_NUM_ERR) {
        return Err(VmError::Thrown(e));
    }
    Ok(ctx.realm.less_than(x, y))
}

/// Strict equality `===` (`Op::StrictEq`'s semantics), factored out for the
/// generic-JIT helper ([`jit_helper_strict_eq`]). Never throws and applies no
/// coercion — strings compare by value, objects by identity, numbers numerically
/// (NaN ≠ NaN, +0 === -0). `!==` is this followed by `Op::Not`.
fn vm_strict_eq(realm: &Realm, a: NanBox, b: NanBox) -> NanBox {
    NanBox::boolean(realm.strict_equals(a, b))
}

/// The interpreter's real `Op::ValueBin` (`**`, bitwise `&`/`|`/`^`, shifts
/// `<<`/`>>`/`>>>`, and loose `==`/`!=`), factored out so the bytecode VM **and**
/// the generic-JIT runtime helper ([`jit_helper_value_bin`]) share one code path.
/// Loose equality applies ToPrimitive-then-compare (`[] == 0` is `true`); the
/// numeric ops ToPrimitive (number hint) each operand — honoring a user `valueOf`
/// — then the realm's `std`-gated math. `op` is a `VB_*` discriminant.
#[cfg_attr(not(feature = "std"), allow(unused_variables))]
fn vm_value_bin(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    a: NanBox,
    b: NanBox,
    op: u8,
) -> Result<NanBox, VmError> {
    match op {
        VB_LOOSE_EQ | VB_LOOSE_NEQ => {
            // `obj == primitive` runs `ToPrimitive(obj)` with the **default** hint —
            // its `@@toPrimitive`/`valueOf`/`toString`, any of which the program may
            // have overridden (and any of which may throw). `loose_eq_coerce` below
            // only knows the intrinsic display form, so hand those to the
            // tree-walker. Object-vs-object (identity) and the `null`/`undefined`
            // cases need no conversion and stay on the fast path.
            let real_obj = |v: NanBox| {
                v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    !ctx.realm.is_string_handle(h)
                        && ctx.realm.symbol_at(h).is_none()
                        && ctx.realm.bigint_at(h).is_none()
                })
            };
            let bare_prim = |v: NanBox| {
                !real_obj(v)
                    && !matches!(
                        v.unpack(),
                        crate::nanbox::Unpacked::Undefined | crate::nanbox::Unpacked::Null
                    )
            };
            if (real_obj(a) && bare_prim(b)) || (real_obj(b) && bare_prim(a)) {
                return Err(VmError::Unsupported);
            }
            let (xc, yc) = loose_eq_coerce(ctx.realm, a, b);
            let r = ctx.realm.loose_equals(xc, yc);
            Ok(NanBox::boolean(if op == VB_LOOSE_EQ { r } else { !r }))
        }
        // `**`/bitwise/shifts are numeric: ToPrimitive each operand (honoring a user
        // `valueOf`) before the realm's `std`-gated math.
        #[cfg(feature = "std")]
        _ => {
            let xn = to_primitive(ctx, funcs, a, true);
            let yn = to_primitive(ctx, funcs, b, true);
            if let Some(e) = symbol_coercion_error(ctx.realm, xn, yn, SYM_NUM_ERR) {
                return Err(VmError::Thrown(e));
            }
            Ok(match op {
                VB_POW => ctx.realm.pow(xn, yn),
                VB_BIT_AND => ctx.realm.bit_and(xn, yn),
                VB_BIT_OR => ctx.realm.bit_or(xn, yn),
                VB_BIT_XOR => ctx.realm.bit_xor(xn, yn),
                VB_SHL => ctx.realm.shl(xn, yn),
                VB_SHR => ctx.realm.shr(xn, yn),
                VB_USHR => ctx.realm.ushr(xn, yn),
                _ => NanBox::number(f64::NAN),
            })
        }
        #[cfg(not(feature = "std"))]
        _ => Ok(NanBox::number(f64::NAN)),
    }
}

/// The interpreter's real property **get** for a static string key
/// (`Op::GetProp`'s semantics), factored out so the bytecode VM **and** the
/// generic-JIT runtime helper ([`jit_helper_get_prop`]) share one code path and
/// can never diverge: a VM function's synthetic `name`/`prototype`, RegExp
/// introspection members, the monomorphic inline-cache slot load, an own or
/// inherited accessor (its getter runs exactly once, with `recv` as `this`), and
/// the `[[Prototype]]` walk. `cache` is the per-site inline cache: the same
/// `PropertyCache` fed by the interpreter's per-pc IC and by the JIT's per-site
/// IC, so a monomorphic site takes the shape-pointer-compare + slot-load fast
/// path in both tiers.
fn vm_get_prop(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    cache: &mut PropertyCache,
) -> Result<NanBox, VmError> {
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
                        &alloc::format!("Cannot read properties of {what} (reading '{key}')"),
                    );
                    Err(VmError::Thrown(e))
                }
                // Other primitives: a missing property reads `undefined`.
                _ => Ok(NanBox::undefined()),
            }
        }
        Some(handle) => {
            // A VM function's `.name` comes from its proto (the closure is a tagged
            // array whose element 0 is the function id).
            if key == "name"
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
                return Ok(NanBox::handle(s.to_raw()));
            }
            // A VM function's `.prototype` is a lazily-created object (keyed by
            // function id, with a `constructor` back-link).
            if key == "prototype"
                && ctx.realm.is_vm_function(handle)
                && !ctx.realm.has_own(handle, "prototype")
                && let Some(func_id) = ctx
                    .realm
                    .get_element(handle, 0)
                    .as_number()
                    .map(|f| f as u32)
            {
                let proto = ctx.realm.function_prototype(func_id);
                return Ok(NanBox::handle(proto.to_raw()));
            }
            // RegExp introspection properties. The (allocating) `regexp_at` probe
            // is gated on the key first, so a plain `obj.foo` read on a non-regexp
            // pays nothing here.
            let mut done = true;
            let mut result = NanBox::undefined();
            if is_regexp_introspection_key(key)
                && let Some((src, flags)) = ctx.realm.regexp_at(handle)
            {
                match key {
                    "source" => {
                        let s = ctx.realm.new_string(&src);
                        result = NanBox::handle(s.to_raw());
                    }
                    "flags" => {
                        let s = ctx.realm.new_string(&flags);
                        result = NanBox::handle(s.to_raw());
                    }
                    "global" => result = NanBox::boolean(flags.contains('g')),
                    "ignoreCase" => result = NanBox::boolean(flags.contains('i')),
                    "multiline" => result = NanBox::boolean(flags.contains('m')),
                    "sticky" => result = NanBox::boolean(flags.contains('y')),
                    "dotAll" => result = NanBox::boolean(flags.contains('s')),
                    "unicode" => result = NanBox::boolean(flags.contains('u')),
                    "hasIndices" => result = NanBox::boolean(flags.contains('d')),
                    "lastIndex" => {
                        // An aux own slot (a non-canonical value such as an object,
                        // or a `defineProperty` descriptor) shadows the compact cell.
                        if ctx.realm.regex_aux_last_index_defined(handle) {
                            done = false;
                        } else {
                            result = NanBox::number(ctx.realm.regex_last_index(handle) as f64);
                        }
                    }
                    _ => done = false,
                }
            } else {
                done = false;
            }
            if done {
                return Ok(result);
            }
            // Inline-cache fast path: a plain own data property on the receiver's
            // shape resolves via a shape-pointer compare + slot load.
            if let Some(v) = ctx.realm.object_cached_get(handle, key, cache) {
                Ok(v)
            } else if let Some((getter, _)) = ctx.realm.accessor(handle, key)
                && getter.as_handle().is_some()
            {
                // An own getter accessor takes precedence over a data slot.
                call_closure(ctx, funcs, getter, &[], recv)
            } else if ctx.realm.has_own(handle, key) {
                Ok(ctx
                    .realm
                    .get_property(handle, key)
                    .unwrap_or(NanBox::undefined()))
            } else {
                // Walk the `[[Prototype]]` chain for an inherited accessor or data
                // property (receiver stays `recv`).
                let mut found = NanBox::undefined();
                let mut cur = ctx.realm.object_proto(handle);
                while let Some(p) = cur {
                    if let Some((getter, _)) = ctx.realm.accessor(p, key) {
                        if getter.as_handle().is_some() {
                            found = call_closure(ctx, funcs, getter, &[], recv)?;
                        }
                        break;
                    }
                    if ctx.realm.has_own(p, key) {
                        found = ctx
                            .realm
                            .get_property(p, key)
                            .unwrap_or(NanBox::undefined());
                        break;
                    }
                    cur = ctx.realm.object_proto(p);
                }
                Ok(found)
            }
        }
    }
}

/// The interpreter's real property **set** for a static string key
/// (`Op::SetProp`'s semantics), factored out so the bytecode VM **and** the
/// generic-JIT runtime helper ([`jit_helper_set_prop`]) share one code path and
/// can never diverge: `regex.lastIndex`, `arr.length` resize, a canonical array
/// index, an own or inherited setter accessor (its setter runs exactly once, with
/// `recv` as `this`), and the monomorphic inline-cache in-place write. Returns
/// `Err(VmError::Unsupported)` for the descriptor-aware cases the tree-walker owns
/// (a non-writable `length`, a demoted/frozen array index) — the same fault the
/// interpreter raises to fall back.
fn vm_set_prop(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: &str,
    value: NanBox,
    cache: &mut PropertyCache,
) -> Result<(), VmError> {
    let handle = recv
        .as_handle()
        .map(Handle::from_raw)
        .ok_or(VmError::NotAnObject)?;
    // `regex.lastIndex = v` updates the stateful search position.
    if key == "lastIndex" && ctx.realm.regexp_at(handle).is_some() {
        set_regex_last_index_value(ctx.realm, handle, value);
        return Ok(());
    }
    // `arr.length = n` resizes the array (truncate/pad).
    if key == "length" && ctx.realm.is_array(handle) {
        if let Some(n) = ctx.realm.array_length_uint32(value) {
            if ctx
                .realm
                .array_length_set_needs_slow_path(handle, n as usize)
            {
                return Err(VmError::Unsupported);
            }
            ctx.realm.set_array_length(handle, n as usize);
        } else {
            let e = make_error(ctx.realm, "RangeError", "Invalid array length");
            return Err(VmError::Thrown(e));
        }
        return Ok(());
    }
    // A canonical numeric string key on an array addresses element storage.
    if ctx.realm.is_array(handle)
        && let Ok(i) = key.parse::<usize>()
        && alloc::format!("{i}") == key
        && (i as u64) < u64::from(u32::MAX)
    {
        if i >= ctx.realm.limits.max_array_len {
            let e = make_error(ctx.realm, "RangeError", "Invalid array length");
            return Err(VmError::Thrown(e));
        }
        if ctx.realm.array_index_has_override(handle, i) {
            return Err(VmError::Unsupported);
        }
        ctx.realm.set_element(handle, i, value);
        return Ok(());
    }
    // A setter accessor (own or inherited) takes precedence over a data slot. An
    // inherited accessor only applies when the receiver has no own property of
    // that name (an own data property shadows).
    let own_accessor = ctx.realm.accessor(handle, key);
    let chain_accessor = if own_accessor.is_some() || ctx.realm.has_own(handle, key) {
        own_accessor
    } else {
        let mut acc = None;
        let mut cur = ctx.realm.object_proto(handle);
        while let Some(p) = cur {
            if let Some(a) = ctx.realm.accessor(p, key) {
                acc = Some(a);
                break;
            }
            if ctx.realm.has_own(p, key) {
                break; // an inherited data property — ordinary set on receiver
            }
            cur = ctx.realm.object_proto(p);
        }
        acc
    };
    match chain_accessor {
        Some((getter, setter)) if setter.as_handle().is_some() => {
            call_closure(ctx, funcs, setter, &[value], recv)?;
            let _ = getter;
        }
        Some((getter, _)) if getter.as_handle().is_some() => {
            // Accessor with a getter but no setter: the write is a no-op (sloppy).
        }
        _ => {
            // Inline-cache fast path: an in-place write to an existing own data
            // property on the receiver's shape (no transition). A new property, a
            // dictionary object, or a frozen/read-only target misses and falls to
            // `set_property`.
            if !ctx.realm.object_cached_set(handle, key, value, cache) {
                ctx.realm.set_property(handle, key, value);
            }
        }
    }
    Ok(())
}

/// The generic-JIT runtime helper for a property **get**. Reconstructs `&mut Ctx`
/// and the running function table from the opaque pointers, runs the shared
/// [`vm_get_prop`] through the site's persistent inline cache, and returns the
/// result's `NanBox` bits — or, on any fault, stashes it in `ctx.jit_pending_fault`
/// and returns the reserved throw/deopt sentinel ([`NanBox::jit_throw_bits`]).
///
/// # Safety
/// `ctx` must be the live `Ctx` the dispatcher passed (with `ctx.jit_funcs` set);
/// `key_ptr`/`key_len` a valid UTF-8 slice owned by the calling `JitProto`; and
/// `cache` that site's `PropertyCache` (also owned by the `JitProto`), with no
/// other live borrow of it during the call.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_get_prop(
    ctx: *mut core::ffi::c_void,
    obj: u64,
    key_ptr: *const u8,
    key_len: usize,
    cache: *mut core::ffi::c_void,
) -> u64 {
    // SAFETY: single-threaded reentrancy — the native caller holds no live
    // `&mut Ctx` while this runs (see `jit_helper_add`).
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: the JIT emitted `key_ptr`/`key_len` from a `Box<str>` it owns for the
    // lifetime of the code; the bytes are valid UTF-8 and live for this call.
    #[allow(unsafe_code)]
    let key =
        unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(key_ptr, key_len)) };
    // SAFETY: `cache` is this site's `PropertyCache`, owned by the `JitProto`; no
    // other reference to it is live while the native body runs.
    #[allow(unsafe_code)]
    let cache = unsafe { &mut *(cache as *mut PropertyCache) };
    match vm_get_prop(ctx, funcs, NanBox::from_bits(obj), key, cache) {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-JIT runtime helper for a property **set**. Mirrors
/// [`jit_helper_get_prop`], running the shared [`vm_set_prop`]; returns
/// `undefined`'s bits on success, or the throw/deopt sentinel (with the fault in
/// `ctx.jit_pending_fault`) otherwise.
///
/// # Safety
/// As [`jit_helper_get_prop`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_set_prop(
    ctx: *mut core::ffi::c_void,
    obj: u64,
    key_ptr: *const u8,
    key_len: usize,
    cache: *mut core::ffi::c_void,
    val: u64,
) -> u64 {
    // SAFETY: see `jit_helper_get_prop`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    #[allow(unsafe_code)]
    let key =
        unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(key_ptr, key_len)) };
    #[allow(unsafe_code)]
    let cache = unsafe { &mut *(cache as *mut PropertyCache) };
    match vm_set_prop(
        ctx,
        funcs,
        NanBox::from_bits(obj),
        key,
        NanBox::from_bits(val),
        cache,
    ) {
        Ok(()) => NanBox::undefined().to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-JIT runtime helper for a computed element **get** `obj[key]`.
/// Reconstructs `&mut Ctx` + the running function table from the opaque pointers,
/// runs the shared [`vm_get_elem`], and returns the result's `NanBox` bits — or,
/// on any fault (a getter throw, a non-object receiver), stashes the whole
/// [`VmError`] in `ctx.jit_pending_fault` and returns the reserved throw/deopt
/// sentinel ([`NanBox::jit_throw_bits`]). Mirrors [`jit_helper_get_prop`].
///
/// # Safety
/// `ctx` must be the live `Ctx` the dispatcher passed (with `ctx.jit_funcs` set);
/// the reentrancy is single-threaded (the native caller holds no live `&mut Ctx`).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_get_elem(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    key: u64,
) -> u64 {
    // SAFETY: single-threaded reentrancy — as `jit_helper_get_prop`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    match vm_get_elem(ctx, funcs, NanBox::from_bits(recv), NanBox::from_bits(key)) {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-JIT runtime helper for a computed element **set** `obj[key] = val`.
/// Mirrors [`jit_helper_get_elem`], running the shared [`vm_set_elem`]; returns
/// `undefined`'s bits on success, or the throw/deopt sentinel (with the fault in
/// `ctx.jit_pending_fault`) otherwise.
///
/// # Safety
/// As [`jit_helper_get_elem`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_set_elem(
    ctx: *mut core::ffi::c_void,
    recv: u64,
    key: u64,
    val: u64,
) -> u64 {
    // SAFETY: see `jit_helper_get_elem`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    match vm_set_elem(
        ctx,
        funcs,
        NanBox::from_bits(recv),
        NanBox::from_bits(key),
        NanBox::from_bits(val),
    ) {
        Ok(()) => NanBox::undefined().to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-JIT runtime helper for a `.length` read (`Op::ArrayLen`): runs the
/// shared [`vm_array_len`]. Mirrors [`jit_helper_get_elem`] — the result bits, or
/// the throw/deopt sentinel with the fault (a non-object receiver) in
/// `ctx.jit_pending_fault`.
///
/// # Safety
/// As [`jit_helper_get_elem`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_array_len(ctx: *mut core::ffi::c_void, recv: u64) -> u64 {
    // SAFETY: see `jit_helper_get_elem`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    match vm_array_len(ctx, funcs, NanBox::from_bits(recv)) {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-tier runtime-helper table (`+`, property get/set, element get/set,
/// `.length`, arithmetic, comparisons, calls), passed to
/// [`crate::jit::JitProto::compile_generic`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
fn jit_generic_helpers() -> crate::jit::GenericHelpers {
    crate::jit::GenericHelpers {
        add: jit_helper_add,
        get: jit_helper_get_prop,
        set: jit_helper_set_prop,
        get_elem: jit_helper_get_elem,
        set_elem: jit_helper_set_elem,
        array_len: jit_helper_array_len,
        array_len_dense: jit_array_len_dense,
        arith: jit_helper_arith,
        value_bin: jit_helper_value_bin,
        lt: jit_helper_lt,
        strict_eq: jit_helper_strict_eq,
        truthy: jit_helper_truthy,
        call: jit_helper_call,
        call_native: jit_helper_call_native,
        arena: jit_arena,
        array_unrestricted: jit_array_unrestricted,
    }
}

/// The generic-JIT probe for the inline dense-array element-SET fast path: returns
/// `1` iff the receiver is a plain dense array on which an in-bounds in-place store
/// is identical to the interpreter's `vm_set_elem` (not frozen/sealed, no per-index
/// override aux object, dense length within `max_array_len`), else `0`. A leaf,
/// non-allocating read that cannot trigger a collection, so the emitted fast path
/// stays GC-safe. Mirrors [`jit_arena`]'s reentrancy/safety contract.
///
/// # Safety
/// `ctx` must be the live `Ctx` the dispatcher passed. The reentrancy is
/// single-threaded — the native caller holds no live `&mut Ctx` while this runs.
/// The generic-JIT probe for the inline dense-array `.length` fast path: returns
/// `1` iff the receiver's spec `.length` equals its dense element-`Vec` length —
/// i.e. it is a plain [`Cell::Array`] that is **not** a VM function (which reports
/// its parameter count, and always carries a `\0vmfn` aux property) and has **no**
/// sparse logical-length override (`arr.length = huge`). Only then may the emitted
/// code read `off_arr_len` directly; every other receiver (string, VM function,
/// sparse array, non-array) routes to [`jit_helper_array_len`]. A leaf,
/// non-allocating read that cannot trigger a collection, so the fast path stays
/// GC-safe (mirrors [`jit_array_unrestricted`]).
///
/// # Safety
/// As [`jit_array_unrestricted`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_array_len_dense(ctx: *mut core::ffi::c_void, recv: u64) -> u64 {
    // SAFETY: single-threaded reentrancy — a shared borrow suffices (read-only).
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    match NanBox::from_bits(recv).as_handle() {
        Some(raw) if ctx.realm.jit_array_len_is_dense(Handle::from_raw(raw)) => 1,
        _ => 0,
    }
}

#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_array_unrestricted(ctx: *mut core::ffi::c_void, recv: u64) -> u64 {
    // SAFETY: single-threaded reentrancy — a shared borrow suffices (read-only).
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    match NanBox::from_bits(recv).as_handle() {
        Some(raw) if ctx.realm.jit_array_unrestricted(Handle::from_raw(raw)) => 1,
        _ => 0,
    }
}

/// The generic-JIT heap-arena accessor for the inline monomorphic property-get
/// fast path. Reconstructs `&Ctx` from the opaque pointer and returns the object
/// heap's slot-array base pointer + length. A leaf read: it allocates nothing and
/// cannot trigger a collection, so the emitted fast path (which chases raw
/// pointers off the returned base) stays GC-safe, and re-invoking it per entry
/// keeps it correct across a `slots.push` reallocation.
///
/// # Safety
/// `ctx` must be the live `Ctx` the dispatcher passed. The reentrancy is
/// single-threaded — the native caller holds no live `&mut Ctx` while this runs
/// (as [`jit_helper_get_prop`]).
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_arena(ctx: *mut core::ffi::c_void) -> crate::jit::JitArena {
    // SAFETY: single-threaded reentrancy — see `jit_helper_get_prop`. A shared
    // borrow suffices; this reads the heap's slot-array base/len only.
    #[allow(unsafe_code)]
    let ctx = unsafe { &*(ctx as *const Ctx) };
    let (base, len) = ctx.realm.jit_arena_slots();
    crate::jit::JitArena { base, len }
}

/// Turns a helper `Result` into a raw generic-JIT return word: the value's `NanBox`
/// bits on success, or — on a thrown/faulting error — the value stashed in
/// `ctx.jit_pending` and the reserved throw sentinel returned (mirrors the tail of
/// [`jit_helper_add`]). A non-throw fault is surfaced as a thrown generic error, as
/// it cannot travel through a `NanBox`.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
fn jit_finish(ctx: &mut Ctx, r: Result<NanBox, VmError>) -> u64 {
    match r {
        Ok(v) => v.to_bits(),
        Err(VmError::Thrown(val)) => {
            ctx.jit_pending = Some(val);
            NanBox::jit_throw_bits()
        }
        Err(_) => {
            let e = make_error(ctx.realm, "Error", "JIT bin fault");
            ctx.jit_pending = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// Generic-JIT slow path for `-`/`*`/`/`/`%` (`GA_*` discriminant in `op`): runs the
/// shared [`vm_arith`]. See [`jit_helper_add`] for the reentrancy/safety contract.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_arith(
    ctx: *mut core::ffi::c_void,
    a: u64,
    b: u64,
    op: u64,
) -> u64 {
    // SAFETY: single-threaded reentrancy — as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let r = vm_arith(
        ctx,
        funcs,
        NanBox::from_bits(a),
        NanBox::from_bits(b),
        op as u8,
    );
    jit_finish(ctx, r)
}

/// Generic-JIT slow path for `Op::ValueBin` (`**`, bitwise, shifts, loose `==`/`!=`;
/// `VB_*` discriminant in `op`): runs the shared [`vm_value_bin`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_value_bin(
    ctx: *mut core::ffi::c_void,
    a: u64,
    b: u64,
    op: u64,
) -> u64 {
    // SAFETY: single-threaded reentrancy — as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let r = vm_value_bin(
        ctx,
        funcs,
        NanBox::from_bits(a),
        NanBox::from_bits(b),
        op as u8,
    );
    jit_finish(ctx, r)
}

/// Generic-JIT slow path for relational `<` (`Op::Lt`): runs the shared [`vm_lt`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_lt(ctx: *mut core::ffi::c_void, a: u64, b: u64) -> u64 {
    // SAFETY: single-threaded reentrancy — as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let r = vm_lt(ctx, funcs, NanBox::from_bits(a), NanBox::from_bits(b));
    jit_finish(ctx, r)
}

/// Generic-JIT slow path for strict equality `===` (`Op::StrictEq`): runs the shared
/// [`vm_strict_eq`]. Never throws, so it always returns a boolean `NanBox`'s bits.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_strict_eq(ctx: *mut core::ffi::c_void, a: u64, b: u64) -> u64 {
    // SAFETY: as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    vm_strict_eq(ctx.realm, NanBox::from_bits(a), NanBox::from_bits(b)).to_bits()
}

/// Generic-JIT slow path for a truthiness test (`JumpIfFalse`/`Op::Not` on a
/// non-number/non-boolean operand): runs the interpreter's `ToBoolean`
/// (`Realm::truthy`) and returns `1` (truthy) or `0` (falsy). Never throws
/// (ToBoolean calls no user code), so the caller needs no sentinel check.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_truthy(ctx: *mut core::ffi::c_void, v: u64) -> u64 {
    // SAFETY: as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    u64::from(ctx.realm.truthy(NanBox::from_bits(v)))
}

/// The generic-JIT runtime helper for `+`: the slow path the native code calls
/// when an operand isn't a number. Reconstructs `&mut Ctx` from the opaque
/// pointer, runs the shared [`vm_add`], and either returns the result's `NanBox`
/// bits or — on a thrown exception — stashes the value in `ctx.jit_pending` and
/// returns the reserved throw sentinel ([`NanBox::jit_throw_bits`]).
///
/// # Safety
/// `ctx` must be the live `Ctx` pointer the dispatcher passed to the native
/// entry, with `ctx.jit_funcs` set to the running function table.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_add(ctx: *mut core::ffi::c_void, a: u64, b: u64) -> u64 {
    // SAFETY: single-threaded reentrancy — the native caller holds no live
    // `&mut Ctx` while this runs (it passed a raw pointer), so reborrowing it is
    // the same pattern as an ordinary recursive interpreter call.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` points at the function table for the whole program run,
    // which outlives this call (it is borrowed down the entire call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let av = NanBox::from_bits(a);
    let bv = NanBox::from_bits(b);
    match vm_add(ctx, funcs, av, bv) {
        Ok(v) => v.to_bits(),
        Err(VmError::Thrown(val)) => {
            ctx.jit_pending = Some(val);
            NanBox::jit_throw_bits()
        }
        // `vm_add` only ever yields `Ok` or `Thrown`; a non-throw fault can't be
        // carried through a NanBox, so surface it as a thrown generic error.
        Err(_) => {
            let e = make_error(ctx.realm, "Error", "JIT add fault");
            ctx.jit_pending = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

// Test-only counter of `jit_helper_call` entries — lets a differential test
// distinguish a direct generic→generic native call (this stays at 0) from the
// interpreter-reentrant helper path (this increments). Thread-local so the count
// is isolated per test (cargo runs each test on its own thread, and the JIT call
// is synchronous on that thread) — a shared global would race under parallelism.
#[cfg(all(test, feature = "jit", target_os = "linux", target_arch = "x86_64"))]
std::thread_local! {
    pub(crate) static JIT_HELPER_CALL_COUNT: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// The generic-JIT runtime helper for a static function call (`Op::Call`): the
/// slow path the native code calls to invoke callee `id` with `this` and the
/// marshaled argument buffer. Reconstructs `&mut Ctx`, runs the shared [`vm_call`],
/// and returns the result's `NanBox` bits — or, on any fault (a callee throw or a
/// non-throw `VmError`), stashes the whole error in `ctx.jit_pending_fault` and
/// returns the reserved throw/deopt sentinel ([`NanBox::jit_throw_bits`]), exactly
/// as the property helpers do (so `call_generic` surfaces the identical `VmError`).
/// The callee therefore runs **exactly once** — the body returns the sentinel and
/// the interpreter propagates it without re-running the op.
///
/// # Safety
/// `ctx` must be the live `Ctx` the dispatcher passed (with `ctx.jit_funcs` set);
/// `argv`/`argc` a valid buffer of `argc` `NanBox` words for the call.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_call(
    ctx: *mut core::ffi::c_void,
    id: u64,
    this: u64,
    argv: *const u64,
    argc: usize,
) -> u64 {
    // Test-only instrumentation: the differential tests count entries here to prove
    // a generic→generic call took the *direct* native path (this helper NOT entered)
    // versus the interpreter-reentrant slow path (this helper entered).
    #[cfg(test)]
    JIT_HELPER_CALL_COUNT.with(|c| c.set(c.get() + 1));
    // SAFETY: single-threaded reentrancy — as `jit_helper_add`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    // SAFETY: `jit_funcs` outlives this call (borrowed down the whole call tree).
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    // SAFETY: `argv` points at an `argc`-long buffer of NanBox words the emitted
    // code marshaled into its own frame; it is live for this synchronous call.
    let args: Vec<NanBox> = if argc == 0 {
        Vec::new()
    } else {
        #[allow(unsafe_code)]
        let raw = unsafe { core::slice::from_raw_parts(argv, argc) };
        raw.iter().map(|&w| NanBox::from_bits(w)).collect()
    };
    match vm_call(ctx, funcs, id as usize, NanBox::from_bits(this), &args) {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// The generic-JIT runtime helper for a native builtin call (`Op::CallNative`):
/// runs the shared [`vm_call_native`]. Mirrors [`jit_helper_call`] — success bits,
/// or the throw/deopt sentinel with the fault in `ctx.jit_pending_fault`.
///
/// # Safety
/// As [`jit_helper_call`].
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
pub(crate) extern "C" fn jit_helper_call_native(
    ctx: *mut core::ffi::c_void,
    native: u64,
    argv: *const u64,
    argc: usize,
) -> u64 {
    // SAFETY: as `jit_helper_call`.
    #[allow(unsafe_code)]
    let ctx = unsafe { &mut *(ctx as *mut Ctx) };
    #[allow(unsafe_code)]
    let funcs = unsafe {
        &*ctx
            .jit_funcs
            .expect("jit_funcs set before generic dispatch")
    };
    let args: Vec<NanBox> = if argc == 0 {
        Vec::new()
    } else {
        #[allow(unsafe_code)]
        let raw = unsafe { core::slice::from_raw_parts(argv, argc) };
        raw.iter().map(|&w| NanBox::from_bits(w)).collect()
    };
    match vm_call_native(ctx, funcs, native as u16, &args) {
        Ok(v) => v.to_bits(),
        Err(e) => {
            ctx.jit_pending_fault = Some(e);
            NanBox::jit_throw_bits()
        }
    }
}

/// Invokes a generic-tier native body and translates its raw return into the
/// interpreter's `Result`. `None` on a pre-call arity deopt (the caller then
/// falls through to the interpreter); otherwise `Ok(value)`, or — when the body
/// returns the throw/deopt sentinel — `Err(..)`. A property helper stashes a full
/// [`VmError`] in `ctx.jit_pending_fault` (so a non-throw `Unsupported` propagates
/// as the interpreter would); the `+` helper stashes a thrown `NanBox` in
/// `ctx.jit_pending`. Setting `ctx.jit_funcs` here lets the helpers re-enter the
/// interpreter with the correct function table.
#[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
fn call_generic(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    jit: &crate::jit::JitProto,
    args: &[NanBox],
) -> Option<Result<NanBox, VmError>> {
    ctx.jit_funcs = Some(funcs as *const [FnProto]);
    let ctx_ptr = ctx as *mut Ctx as *mut core::ffi::c_void;
    let raw = jit.call_generic_raw(ctx_ptr, args)?;
    let nb = NanBox::from_bits(raw);
    if nb.is_jit_throw_sentinel() {
        let err = ctx
            .jit_pending_fault
            .take()
            .or_else(|| ctx.jit_pending.take().map(VmError::Thrown))
            .unwrap_or_else(|| {
                VmError::Thrown(make_error(
                    ctx.realm,
                    "Error",
                    "JIT throw without pending value",
                ))
            });
        Some(Err(err))
    } else {
        Some(Ok(nb))
    }
}

/// `regex.lastIndex = v` for the bytecode VM. `lastIndex` is an ordinary data
/// property that may hold any value (it is only `ToLength`'d at `exec` time): a
/// canonical non-negative integer (fitting uint32) is stored compactly in the
/// cell's `usize` field, and any other value is kept verbatim in an aux data
/// slot so a later `Get` returns it unchanged (its `valueOf` runs at `exec`
/// time, not at assignment). A non-writable own `lastIndex` is honored by the
/// slow tree-walker path; the VM hot path follows the common writable case.
fn set_regex_last_index_value(realm: &mut Realm, handle: Handle, v: NanBox) {
    // `-0` is *not* canonical: the compact `usize` field cannot carry its sign,
    // and a non-global `exec` must leave `re.lastIndex` exactly as written.
    let canonical = v.as_number().filter(|n| {
        n.is_finite()
            && *n >= 0.0
            && n.is_sign_positive()
            && *n == (*n as u64 as f64)
            && *n <= u32::MAX as f64
    });
    match canonical {
        Some(n) => {
            if realm.regex_aux_last_index_defined(handle) {
                realm.set_property(handle, "lastIndex", v);
            }
            realm.set_regex_last_index(handle, n as usize);
        }
        None => {
            // The materialized aux slot is the *same* own property as the
            // synthesized one: `{ enumerable: false, configurable: false }`.
            realm.set_property(handle, "lastIndex", v);
            realm.mark_hidden(handle, "lastIndex");
            realm.set_non_configurable_property(handle, "lastIndex");
            let n = realm.to_number(v);
            realm.set_regex_last_index(
                handle,
                if n.is_finite() && n >= 0.0 {
                    n as usize
                } else {
                    0
                },
            );
        }
    }
}

fn make_error(realm: &mut Realm, name: &str, message: &str) -> NanBox {
    let obj = realm.new_object();
    let n = NanBox::handle(realm.new_string(name).to_raw());
    realm.set_property(obj, "name", n);
    let m = NanBox::handle(realm.new_string(message).to_raw());
    realm.set_property(obj, "message", m);
    NanBox::handle(obj.to_raw())
}

/// Installs `name` and `length` as own, non-enumerable, non-writable,
/// configurable data properties on a freshly-created VM closure (so
/// `f.hasOwnProperty("name")`, `Object.getOwnPropertyDescriptor(f, "length")`,
/// and Test262's `verifyProperty` behave per spec). `f.length` continues to be
/// read via `Op::ArrayLen` from the proto; this just makes the reflective own
/// property exist with the right attributes.
fn install_fn_name_length(realm: &mut Realm, f: Handle, proto: Option<&FnProto>) {
    // Deliberately a no-op for a *named* function: `name` and `length` are
    // synthesized on read from the function definition (see the property-get and
    // descriptor paths), so materializing them here was pure cost — and it ran
    // on every closure allocation, including the string allocation for the name.
    // Installing them eagerly was 58% of the cost of creating a closure, of
    // which the `new_string` alone was 37%; a 2M-closure loop went 3528 ms to
    // 1466 ms with this removed. Verified that `name`/`length` values,
    // descriptors, writability, deletability, `Reflect.ownKeys` and `bind`
    // naming are all unchanged.
    //
    // Kept as a named no-op rather than deleting the call sites so the intent is
    // visible at `LoadFunc`/`MakeClosure`: this is *not* an omission.
    let _ = (realm, f, proto);
}

/// ToBigInt-coerces `value` for a write to element of `target` **iff** `target`
/// is a `BigInt64Array`/`BigUint64Array`; otherwise returns `value` unchanged.
/// Mirrors the tree-walker's `Interp::coerce_to_bigint` for the
/// values reachable on the bytecode path (BigInt / Boolean / String); a Number
/// (and any other non-coercible) is `Err(TypeError)`. Keeps a Number-into-BigInt
/// store on the VM path a throw rather than a silent no-op.
fn coerce_bigint_typed_write(
    realm: &mut Realm,
    target: Handle,
    value: NanBox,
) -> Result<NanBox, NanBox> {
    use crate::nanbox::Unpacked;
    if !realm
        .typed_kind(target)
        .is_some_and(crate::nbexec::is_bigint_kind)
    {
        return Ok(value);
    }
    let big = match value.unpack() {
        Unpacked::Bool(b) => {
            if b {
                crate::bignum::BigInt::from_i128(1)
            } else {
                crate::bignum::BigInt::zero()
            }
        }
        Unpacked::Handle(raw) => {
            let h = Handle::from_raw(raw);
            if realm.bigint_at(h).is_some() {
                return Ok(value); // already a BigInt
            } else if let Some(s) = realm.string_value(h) {
                let t = s.trim_matches(crate::realm::is_js_whitespace);
                if t.is_empty() {
                    crate::bignum::BigInt::zero()
                } else {
                    let (radix, body) = match t.get(0..2) {
                        Some("0x" | "0X") => (16, &t[2..]),
                        Some("0o" | "0O") => (8, &t[2..]),
                        Some("0b" | "0B") => (2, &t[2..]),
                        _ => (10, t),
                    };
                    match crate::bignum::BigInt::from_str_radix(body, radix) {
                        Some(b) => b,
                        None => {
                            return Err(make_error(
                                realm,
                                "SyntaxError",
                                "Cannot convert string to a BigInt",
                            ));
                        }
                    }
                }
            } else {
                return Err(make_error(
                    realm,
                    "TypeError",
                    "Cannot convert value to a BigInt",
                ));
            }
        }
        _ => {
            return Err(make_error(
                realm,
                "TypeError",
                "Cannot convert a Number to a BigInt",
            ));
        }
    };
    Ok(NanBox::handle(realm.new_bigint(big).to_raw()))
}

/// Whether `key` is one of the `RegExp` introspection properties surfaced
/// directly by [`Op::GetProp`] (`source`, `flags`, the flag-letter booleans, and
/// `lastIndex`). Used as a cheap pre-filter so the hot path only consults the
/// (allocating) [`Realm::regexp_at`] probe when the key could actually name a
/// regexp member — a plain `obj.foo` read never pays for it.
fn is_regexp_introspection_key(key: &str) -> bool {
    matches!(
        key,
        "source"
            | "flags"
            | "global"
            | "ignoreCase"
            | "multiline"
            | "sticky"
            | "dotAll"
            | "unicode"
            | "hasIndices"
            | "lastIndex"
    )
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
    // The popped job's handler/value/result live in a Rust local for the duration
    // of the handler call, and the handler runs at `call_depth == 1` (the depth
    // that may collect) — so lock the GC out for the whole drain rather than
    // trying to publish a moving target.
    ctx.gc_lock += 1;
    let r = drain_microtasks_inner(ctx, funcs);
    ctx.gc_lock -= 1;
    r
}

fn drain_microtasks_inner(ctx: &mut Ctx, funcs: &[FnProto]) -> Result<(), VmError> {
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

/// The bytecode tier's **GC safepoint**, taken on a loop back-edge.
///
/// Collection is only sound where the whole root set is enumerable, and in this
/// engine that is true at exactly one place: the top of the *outermost*
/// activation's dispatch loop. Everywhere else a live [`NanBox`] may sit in a
/// Rust local — a caller's register window (`regs` is a Rust-stack `Vec`), a
/// native builtin's argument vector, a half-built array — none of which the
/// collector can see. Hence the three guards:
///
/// * `gc_enabled` — this `Ctx` came from a `run_program*` entry, whose bytecode
///   was compiled before the realm existed (so no `LoadConst` can be a handle
///   from a table we don't scan; we scan them anyway, belt and braces).
/// * `call_depth == 1` — no other activation, native, or JIT frame is below us,
///   so no other register window or Rust-local value exists.
/// * `gc_lock == 0` — no VM-owned Rust frame (the microtask drain) is holding
///   values outside a register window.
///
/// Roots handed over: this frame's registers, the outermost activation's
/// `args`/`captures`/`this` ([`Ctx::top_frame_roots`]), the pending microtask
/// queue, and every handle-valued bytecode constant. The realm adds its own
/// intrinsics in [`Realm::maybe_collect`].
fn vm_safepoint(ctx: &mut Ctx, funcs: &[FnProto], program: &[Op], regs: &[NanBox]) {
    if !ctx.gc_enabled || ctx.call_depth != 1 || ctx.gc_lock != 0 {
        return;
    }
    // Cheap pre-check: skip building the root vector unless a cycle is due.
    if ctx.realm.gc_pressure() < ctx.realm.gc_next_threshold() {
        return;
    }
    let mut roots: Vec<Handle> = Vec::new();
    fn push(roots: &mut Vec<Handle>, v: NanBox) {
        if let Some(raw) = v.as_handle() {
            roots.push(Handle::from_raw(raw));
        }
    }
    for r in regs.iter().chain(ctx.top_frame_roots.iter()) {
        push(&mut roots, *r);
    }
    for t in &ctx.microtasks {
        push(&mut roots, t.handler);
        push(&mut roots, t.value);
        roots.push(t.result);
    }
    // Bytecode constants: `compile_program` runs before the realm exists so these
    // are all immediates today, but rooting them costs one pass over the program
    // per cycle and makes the safepoint independent of that invariant.
    for op in program
        .iter()
        .chain(funcs.iter().flat_map(|f| f.ops.iter()))
    {
        if let Op::LoadConst { value, .. } = op {
            push(&mut roots, *value);
        }
    }
    for (_, body) in ctx.tiers.values() {
        for op in body.iter().flat_map(|b| b.iter()) {
            if let Op::LoadConst { value, .. } = op {
                push(&mut roots, *value);
            }
        }
    }
    ctx.realm.maybe_collect(&roots);
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
) -> Result<FrameExit, VmError> {
    let mut pc = 0;
    // Active exception handlers: `(catch_pc, catch_reg)`, innermost last.
    let mut handlers: Vec<(usize, Reg)> = Vec::new();

    // Per-frame monomorphic inline caches, one slot per instruction index, keyed
    // by pc. A `GetProp`/`SetProp` site that runs repeatedly in a hot loop within
    // this activation resolves its property by a shape-pointer compare plus a slot
    // load instead of an O(depth) name walk every iteration (see `crate::ic`).
    // Built lazily so non-property-heavy frames pay nothing; entries for other
    // opcodes stay cold and unused. The cache keys on `Rc<Shape>` pointer
    // identity, so any object of a different shape (or a dictionary-mode object,
    // whose sentinel shape resolves no key) simply misses and re-resolves —
    // always consistent, never stale across a shape change or property delete.
    let mut ic: Vec<PropertyCache> = Vec::new();

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

    // The inline cache for the current site, sizing the per-frame vector on first
    // use. One `PropertyCache` per instruction index keyed by pc.
    macro_rules! site_cache {
        ($site:expr) => {{
            if ic.len() <= $site {
                ic.resize_with(program.len(), PropertyCache::new);
            }
            &mut ic[$site]
        }};
    }

    while pc < program.len() {
        let op = &program[pc];
        // The instruction index of this op, used to key its inline cache; `pc`
        // itself advances before the op runs (and may be redirected by jumps).
        let site = pc;
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
            // `toString` is honored first via `to_primitive`. Shared with the
            // generic-JIT helper via `vm_arith` so the two tiers can't diverge.
            Op::Sub { dst, a, b } => {
                match vm_arith(ctx, funcs, regs[*a as usize], regs[*b as usize], GA_SUB) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::Mul { dst, a, b } => {
                match vm_arith(ctx, funcs, regs[*a as usize], regs[*b as usize], GA_MUL) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::Div { dst, a, b } => {
                match vm_arith(ctx, funcs, regs[*a as usize], regs[*b as usize], GA_DIV) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::Mod { dst, a, b } => {
                match vm_arith(ctx, funcs, regs[*a as usize], regs[*b as usize], GA_MOD) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::HasProp { dst, key, obj } => {
                let present = match regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    Some(h) => {
                        // ToPropertyKey + `[[HasProperty]]`: a proxy `has` trap
                        // anywhere on the chain, and an *object* left operand
                        // (ToPrimitive, e.g. a Symbol wrapper) are the tree-walker's
                        // — fault the program over rather than key on a display
                        // string. A primitive Symbol keys on its `\0sym:` name.
                        let kv = regs[*key as usize];
                        let k = match kv.as_handle().map(Handle::from_raw) {
                            Some(kh) => match ctx.realm.symbol_at(kh) {
                                Some((_, id)) => alloc::format!("\u{0}sym:{id}"),
                                None if ctx.realm.string_value(kh).is_some() => {
                                    ctx.realm.to_display_string(kv)
                                }
                                None => return Err(VmError::Unsupported),
                            },
                            None => ctx.realm.to_display_string(kv),
                        };
                        if ctx.realm.proxy_at(h).is_some() {
                            return Err(VmError::Unsupported);
                        }
                        // Own or inherited (walk the prototype chain). `has_own`
                        // already reports an array's in-range non-hole indices and
                        // `length`, so a hole is correctly absent unless inherited.
                        let mut found = false;
                        let mut cur = Some(h);
                        while let Some(c) = cur {
                            if ctx.realm.proxy_at(c).is_some() {
                                return Err(VmError::Unsupported);
                            }
                            if ctx.realm.has_own(c, &k) || ctx.realm.accessor(c, &k).is_some() {
                                found = true;
                                break;
                            }
                            cur = ctx.realm.object_proto(c);
                        }
                        found
                    }
                    // `x in <primitive>` is a TypeError — the tree-walker raises it.
                    None => return Err(VmError::Unsupported),
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
                // Shared with the generic-JIT helper via `vm_value_bin`.
                match vm_value_bin(ctx, funcs, regs[*a as usize], regs[*b as usize], *op) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
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
                // Shared with the generic-JIT helper via `vm_lt`.
                match vm_lt(ctx, funcs, regs[*a as usize], regs[*b as usize]) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::AddValue { dst, a, b } => {
                // The general `+`: ToPrimitive each operand then string-concat or
                // numeric add. Shared with the generic-JIT helper via `vm_add` so
                // the two tiers can never diverge.
                match vm_add(ctx, funcs, regs[*a as usize], regs[*b as usize]) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::StrictEq { dst, a, b } => {
                // Shared with the generic-JIT helper via `vm_strict_eq`.
                regs[*dst as usize] = vm_strict_eq(ctx.realm, regs[*a as usize], regs[*b as usize]);
            }
            Op::JumpIfFalse { cond, target } => {
                if !ctx.realm.truthy(regs[*cond as usize]) {
                    pc = *target;
                }
            }
            Op::Jump { target } => {
                // A backward jump is a loop back-edge — the one point in the
                // dispatch loop where no partially-evaluated opcode state exists,
                // so it is where an allocation-triggered collection can run.
                if *target <= pc {
                    vm_safepoint(ctx, funcs, program, regs);
                }
                pc = *target;
            }
            Op::NewString { dst, value } => {
                let handle = ctx.realm.new_string(value);
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::NewArray { dst, len } => {
                // Defence in depth: the verifier rejects oversized lengths, but the
                // call-free `run` entrypoint runs unverified ops, so cap here too.
                if *len > ctx.realm.limits.max_array_len {
                    let e = make_error(ctx.realm, "RangeError", "Array length too large");
                    return Err(VmError::Thrown(e));
                }
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
                    // The array is *sparse*: its indices are holes (absent),
                    // not `undefined` data properties. A length beyond the dense
                    // cap is recorded as a logical length via `set_array_length`
                    // rather than materialized as billions of slots.
                    let h = ctx.realm.new_array(vec![]);
                    ctx.realm.set_array_length(h, n as usize);
                    h
                } else {
                    ctx.realm.new_array(vec![v])
                };
                regs[*dst as usize] = NanBox::handle(handle.to_raw());
            }
            Op::GetElem { dst, arr, index } => {
                let handle = object_handle(regs[*arr as usize])?;
                let i = num(regs[*index as usize])? as usize;
                // A plain Array's element keys are [0, 2**32−1); the boundary value
                // 2**32−1 is an ordinary named property, not an element. Typed
                // arrays accept any in-bounds index.
                if ctx.realm.typed_len(handle).is_none()
                    && ctx.realm.is_array(handle)
                    && (i as u64) >= u64::from(u32::MAX)
                {
                    let k = alloc::format!("{i}");
                    regs[*dst as usize] = ctx
                        .realm
                        .get_property(handle, &k)
                        .unwrap_or(NanBox::undefined());
                } else if ctx.realm.typed_len(handle).is_none() && ctx.realm.is_array(handle) {
                    // A plain array index: a present own element wins; a hole or
                    // out-of-range index consults the prototype chain.
                    match vm_array_index_get(ctx, funcs, handle, i, regs[*arr as usize]) {
                        Ok(v) => regs[*dst as usize] = v,
                        Err(e) => handle_throw!(e),
                    }
                } else {
                    regs[*dst as usize] = ctx.realm.get_element(handle, i);
                }
            }
            Op::SetElem { arr, index, src } => {
                let handle = object_handle(regs[*arr as usize])?;
                let fi = num(regs[*index as usize])?;
                // A non-canonical numeric index on an array (negative or fractional,
                // e.g. `a[-1]` / `a[1.5]`) is an ordinary named property, NOT an
                // element — but `as usize` would truncate it to a real index. The
                // descriptor-aware tree-walker stores it correctly; fault to it.
                // `fi != (fi as u64) as f64` is the no_std-safe non-integer test
                // (`f64::fract` is std-only); the leading `fi < 0.0` short-circuits
                // so the cast only runs for non-negative values.
                if (fi < 0.0 || fi != (fi as u64) as f64)
                    && ctx.realm.typed_len(handle).is_none()
                    && ctx.realm.is_array(handle)
                {
                    return Err(VmError::Unsupported);
                }
                let i = fi as usize;
                // A dense-array write to a valid array index (`[0, 2**32−1)`) past
                // the storage cap must store the element sparsely (as a named
                // property) and grow the logical `length` — a plain `arr[i] = v`
                // never raises "Invalid array length". The descriptor-aware
                // tree-walker owns that sparse path (`set_element_checked`); fault
                // to it. A typed-array out-of-bounds write stays a spec no-op.
                if ctx.realm.typed_len(handle).is_none()
                    && ctx.realm.is_array(handle)
                    && (i as u64) < u64::from(u32::MAX)
                    && i >= ctx.realm.limits.max_array_len
                {
                    return Err(VmError::Unsupported);
                }
                // A plain Array's element keys are [0, 2**32−1); the boundary value
                // 2**32−1 is an ordinary named property (no ArraySetLength).
                if ctx.realm.typed_len(handle).is_none()
                    && ctx.realm.is_array(handle)
                    && (i as u64) >= u64::from(u32::MAX)
                {
                    let k = alloc::format!("{i}");
                    ctx.realm.set_property(handle, &k, regs[*src as usize]);
                } else if ctx.realm.typed_len(handle).is_none()
                    && ctx.realm.array_index_has_override(handle, i)
                {
                    // The index was demoted (non-writable/accessor) or the array is
                    // frozen/sealed: the tree-walker honors the descriptor (accessor
                    // setter, read-only no-op, strict throw). Fault to it.
                    return Err(VmError::Unsupported);
                } else {
                    // A BigInt typed-array element write ToBigInt-coerces (a Number
                    // throws TypeError) instead of silently no-op'ing.
                    match coerce_bigint_typed_write(ctx.realm, handle, regs[*src as usize]) {
                        Ok(v) => {
                            ctx.realm.set_element(handle, i, v);
                        }
                        Err(e) => handle_throw!(VmError::Thrown(e)),
                    }
                }
            }
            Op::GetKey { dst, obj, key } => {
                // Shared with the generic-JIT helper via `vm_get_elem` (the computed
                // `obj[key]` read: array element / typed-array / string index /
                // canonical numeric-string key / `length` / accessor / ordinary
                // property), so the two tiers can never diverge.
                match vm_get_elem(ctx, funcs, regs[*obj as usize], regs[*key as usize]) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::SetKey { obj, key, src } => {
                // Shared with the generic-JIT helper via `vm_set_elem` (the computed
                // `obj[key] = v` write). A descriptor-aware case (demoted/frozen
                // index, non-writable `length`) returns `Err(Unsupported)` to fault
                // to the tree-walker, exactly as before.
                match vm_set_elem(
                    ctx,
                    funcs,
                    regs[*obj as usize],
                    regs[*key as usize],
                    regs[*src as usize],
                ) {
                    Ok(()) => {}
                    Err(e) => handle_throw!(e),
                }
            }
            Op::EnumKeys { dst, obj } => {
                let h = object_handle(regs[*obj as usize])?;
                let mut seen = alloc::collections::BTreeSet::new();
                let mut out = Vec::new();
                // An array leads with its integer indices (a VM closure's backing
                // cells are not enumerable).
                if !ctx.realm.is_vm_function(h)
                    && let Some(indices) = ctx.realm.array_enumerable_indices(h)
                {
                    for i in indices {
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
                // Shared with the generic-JIT helper via `vm_array_len` (`.length`
                // on an array / string / VM function, or an explicit `length` data
                // property), so the two tiers can never diverge.
                match vm_array_len(ctx, funcs, regs[*arr as usize]) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
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
                // A spread of any built-in iterable (array / typed array / string
                // / Map / Set); a user iterable faults to the tree-walker, which
                // drives the full iterator protocol.
                let elems =
                    vm_iterable_values(ctx, regs[*src as usize]).ok_or(VmError::NotAnObject)?;
                let start = ctx.realm.array_length(handle).unwrap_or(0);
                for (i, e) in elems.into_iter().enumerate() {
                    ctx.realm.set_element(handle, start + i, e);
                }
            }
            Op::IterValues { dst, src } => {
                let elems =
                    vm_iterable_values(ctx, regs[*src as usize]).ok_or(VmError::NotAnObject)?;
                regs[*dst as usize] = NanBox::handle(ctx.realm.new_array(elems).to_raw());
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
                let recv = regs[*obj as usize];
                let value = regs[*src as usize];
                let cache = site_cache!(site);
                // Shared with the generic-JIT helper via `vm_set_prop` so the two
                // tiers can never diverge (regex lastIndex, array length/index,
                // accessor setter, IC in-place write, descriptor-aware fault).
                match vm_set_prop(ctx, funcs, recv, key, value, cache) {
                    Ok(()) => {}
                    Err(e) => handle_throw!(e),
                }
            }
            Op::SetHidden { obj, key, src } => {
                if let Some(handle) = regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    ctx.realm
                        .set_hidden_property(handle, key, regs[*src as usize]);
                }
            }
            Op::SetProto { obj, proto } => {
                if let Some(handle) = regs[*obj as usize].as_handle().map(Handle::from_raw) {
                    let p = regs[*proto as usize].as_handle().map(Handle::from_raw);
                    ctx.realm.set_object_proto(handle, p);
                }
            }
            Op::GetProp { dst, obj, key } => {
                let recv = regs[*obj as usize];
                let cache = site_cache!(site);
                // Shared with the generic-JIT helper via `vm_get_prop` so the two
                // tiers can never diverge (synthetic name/prototype, RegExp members,
                // IC slot load, accessor getter, prototype walk, throw).
                match vm_get_prop(ctx, funcs, recv, key, cache) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::Call { dst, func, args } => {
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                // A throw from the callee is caught by this frame's nearest
                // handler, else it keeps unwinding. Shared with the generic-JIT
                // helper via `vm_call` so the two tiers can't diverge.
                match vm_call(ctx, funcs, *func as usize, NanBox::undefined(), &argv) {
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
                install_fn_name_length(ctx.realm, handle, funcs.get(*func as usize));
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
                install_fn_name_length(ctx.realm, handle, funcs.get(*func as usize));
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
                // A user method is a callable property reached on the receiver or
                // anywhere along its `[[Prototype]]` chain (an inherited accessor is
                // invoked with the receiver as `this`); otherwise try a built-in
                // `Array`/`String` method on the fast path.
                let mut accessor_err = None;
                let user_method = recv_val.as_handle().map(Handle::from_raw).and_then(|h| {
                    let mut cur = Some(h);
                    while let Some(c) = cur {
                        if let Some((getter, _)) = ctx.realm.accessor(c, key) {
                            if getter.as_handle().is_some() {
                                match call_closure(ctx, funcs, getter, &[], recv_val) {
                                    Ok(v) => return v.as_handle().map(|_| v),
                                    Err(e) => {
                                        accessor_err = Some(e);
                                        return None;
                                    }
                                }
                            }
                            return None;
                        }
                        if let Some(v) = ctx.realm.get_property(c, key) {
                            return v.as_handle().map(|_| v);
                        }
                        cur = ctx.realm.object_proto(c);
                    }
                    None
                });
                if let Some(e) = accessor_err {
                    handle_throw!(e);
                }
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
                // Shared with the generic-JIT helper via `vm_call_native` so the two
                // tiers can't diverge (it handles the interpreter-aware natives —
                // JSON/Array.from/Number — where `funcs` and the throw machinery are
                // available, and defers the rest to `call_native`).
                match vm_call_native(ctx, funcs, *native, &argv) {
                    Ok(v) => regs[*dst as usize] = v,
                    Err(e) => handle_throw!(e),
                }
            }
            Op::PushHandler { target, reg } => {
                // A crafted artifact can loop a back-edge around this push; bound the
                // handler stack so it can't grow without limit (OOM). Surface a
                // catchable RangeError rather than aborting.
                if handlers.len() >= ctx.realm.limits.max_handler_depth {
                    let e = make_error(ctx.realm, "RangeError", "Handler stack overflow");
                    handle_throw!(VmError::Thrown(e));
                } else {
                    handlers.push((*target, *reg));
                }
            }
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
            Op::Return { src } => return Ok(FrameExit::Return(Some(regs[*src as usize]))),
            // Proper tail call to a static function: hand the callee id/args back
            // to `call_with_inner`, which reuses this activation (O(1) stack). The
            // callee is a plain VM function (the compiler only emits this for a
            // strict `return f(...)`), so no fallback is needed.
            Op::TailCall { func, args } => {
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                return Ok(FrameExit::Tail {
                    id: *func as usize,
                    args: argv,
                    captures: Vec::new(),
                    this: NanBox::undefined(),
                });
            }
            // Proper tail call through a function *value*: if it is a plain VM
            // closure, trampoline (reusing the frame); otherwise degrade to an
            // ordinary call and return its result (semantically `return callee()`).
            Op::TailCallValue { callee, args } => {
                let val = regs[*callee as usize];
                let argv: Vec<NanBox> = args.iter().map(|r| regs[*r as usize]).collect();
                match val.as_handle().map(Handle::from_raw) {
                    Some(h) if ctx.realm.is_vm_function(h) => {
                        let fid = ctx.realm.get_element(h, 0).as_number().unwrap_or(-1.0);
                        let n_caps = ctx.realm.array_length(h).unwrap_or(1).saturating_sub(1);
                        let caps: Vec<NanBox> = (0..n_caps)
                            .map(|i| ctx.realm.get_element(h, i + 1))
                            .collect();
                        if fid >= 0.0 {
                            return Ok(FrameExit::Tail {
                                id: fid as usize,
                                args: argv,
                                captures: caps,
                                this: NanBox::undefined(),
                            });
                        }
                    }
                    _ => {}
                }
                // Not a VM closure: fall back to an ordinary call.
                match call_closure(ctx, funcs, val, &argv, NanBox::undefined()) {
                    Ok(v) => return Ok(FrameExit::Return(Some(v))),
                    Err(e) => handle_throw!(e),
                }
            }
        }
    }
    Ok(FrameExit::Return(None))
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
        0,
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
        return json_revive(ctx, funcs, holder, "", reviver, 0);
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
    depth: usize,
) -> Result<NanBox, VmError> {
    if depth >= ctx.realm.limits.max_json_depth {
        return Err(VmError::Thrown(make_error(
            ctx.realm,
            "RangeError",
            "Maximum JSON nesting depth exceeded",
        )));
    }
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
                let nv = json_revive(ctx, funcs, vh, &ks, reviver, depth + 1)?;
                ctx.realm.set_element(vh, i, nv);
            }
        } else if let Some(keys) = ctx.realm.object_keys(vh) {
            for k in keys {
                let nv = json_revive(ctx, funcs, vh, &k, reviver, depth + 1)?;
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
    depth: usize,
) -> Result<NanBox, VmError> {
    if depth >= ctx.realm.limits.max_json_depth {
        return Err(VmError::Thrown(make_error(
            ctx.realm,
            "RangeError",
            "Maximum JSON nesting depth exceeded",
        )));
    }
    let mut v = value;
    // `toJSON(key)` replaces the value (real objects only — not strings/bigints/Dates,
    // whose serialization the realm handles).
    if let Some(h) = v.as_handle().map(Handle::from_raw)
        && !ctx.realm.is_string_handle(h)
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
        && !ctx.realm.is_string_handle(h)
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
                    ctx,
                    funcs,
                    v,
                    &kk,
                    e,
                    replacer,
                    allow,
                    seen,
                    depth + 1,
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
                let nv = json_normalize(ctx, funcs, v, &k, pv, replacer, allow, seen, depth + 1)?;
                ctx.realm.set_property(new_obj, &k, nv);
            }
            seen.pop();
            return Ok(NanBox::handle(new_obj.to_raw()));
        }
    }
    Ok(v)
}

/// Invokes a closure value (`[func_id, cell…]`) with `args` and `this_val`.
/// `[[Get]]` of array index `i` on a plain-array receiver, with prototype-chain
/// fallthrough for a *hole* or an out-of-range index (an absent own index must
/// consult the prototype, per ordinary `[[Get]]`). A present own element returns
/// directly; otherwise the chain is walked for an inherited array element, data
/// property, or accessor (whose getter is invoked with `recv` as `this`).
fn vm_array_index_get(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    handle: Handle,
    i: usize,
    recv: NanBox,
) -> Result<NanBox, VmError> {
    // A present own element wins.
    if i < ctx.realm.array_length(handle).unwrap_or(0) && !ctx.realm.array_hole_at(handle, i) {
        return Ok(ctx.realm.get_element(handle, i));
    }
    // An OWN accessor installed at this index (`defineProperty(arr, i, {get})`)
    // lives in the aux object over a hole — invoke its getter before the chain.
    let key = alloc::format!("{i}");
    if let Some((getter, _)) = ctx.realm.accessor(handle, &key) {
        if getter.as_handle().is_none() {
            return Ok(NanBox::undefined());
        }
        return call_closure(ctx, funcs, getter, &[], recv);
    }
    // Absent own index (hole or past the end): walk the prototype chain.
    let mut cur = ctx.realm.object_proto(handle);
    while let Some(p) = cur {
        if let Some((getter, _)) = ctx.realm.accessor(p, &key) {
            if getter.as_handle().is_none() {
                return Ok(NanBox::undefined());
            }
            return call_closure(ctx, funcs, getter, &[], recv);
        }
        // A prototype that is itself a plain array exposes its present elements.
        if ctx.realm.is_array(p)
            && i < ctx.realm.array_length(p).unwrap_or(0)
            && !ctx.realm.array_hole_at(p, i)
        {
            return Ok(ctx.realm.get_element(p, i));
        }
        if ctx.realm.has_own(p, &key) {
            return Ok(ctx
                .realm
                .get_property(p, &key)
                .unwrap_or(NanBox::undefined()));
        }
        cur = ctx.realm.object_proto(p);
    }
    Ok(NanBox::undefined())
}

/// `ToPropertyKey(key)` as the VM's internal storage-key string. A primitive
/// Symbol becomes the `"\0sym:<id>"` name the object layer stores symbol-keyed
/// properties under; a String / Number / Boolean / null / undefined is its display
/// form. Any **other** heap object is `Err(VmError::Unsupported)`: its key comes
/// from `ToPrimitive(key, string)` — an `@@toPrimitive` / inherited `toString`
/// only the tree-walker resolves. (A Date, for one, keys on its
/// `Date.prototype.toString` text, not the ISO form `to_display_string` produces.)
fn vm_property_key(ctx: &Ctx, key: NanBox) -> Result<String, VmError> {
    if let Some(raw) = key.as_handle() {
        let h = Handle::from_raw(raw);
        if let Some((_, id)) = ctx.realm.symbol_at(h) {
            return Ok(alloc::format!("\u{0}sym:{id}"));
        }
        if ctx.realm.string_value(h).is_none() {
            return Err(VmError::Unsupported);
        }
    }
    Ok(ctx.realm.to_display_string(key))
}

/// The interpreter's real computed element **get** `obj[key]` (`Op::GetKey`'s
/// semantics), factored out so the bytecode VM **and** the generic-JIT runtime
/// helper ([`jit_helper_get_elem`]) share one code path and can never diverge: a
/// canonical non-negative integer index on a plain array reads element storage
/// (present element / hole → prototype walk / OOB → `undefined`, via
/// [`vm_array_index_get`]); otherwise `ToPropertyKey(key)` (an object key runs its
/// `toString`) then a canonical numeric-string array index, a computed `.length`
/// (array or string UTF-16 code-unit count), an accessor getter (run once with
/// `recv` as `this`), or an ordinary property read. `Err(Thrown)` on a getter
/// throw; `Err(NotAnObject)` when `recv` is not a heap object (the same fault the
/// interpreter's `?` raised).
fn vm_get_elem(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    recv: NanBox,
    key: NanBox,
) -> Result<NanBox, VmError> {
    let handle = recv
        .as_handle()
        .map(Handle::from_raw)
        .ok_or(VmError::NotAnObject)?;
    match key.as_number() {
        // A plain Array index in [0, 2**32−1): a present own element wins; a hole
        // or out-of-range index consults the prototype chain. Typed arrays and the
        // boundary value 2**32−1 (an ordinary named property) fall to the `_` arm.
        Some(n)
            if ctx.realm.typed_len(handle).is_none()
                && ctx.realm.is_array(handle)
                && n >= 0.0
                && n == (n as u64) as f64
                && (n as u64) < u64::from(u32::MAX) =>
        {
            vm_array_index_get(ctx, funcs, handle, n as usize, recv)
        }
        _ => {
            // ToPropertyKey: a Symbol keys on its `\0sym:` name; any other object
            // key needs the full ToPrimitive (its `@@toPrimitive`/`toString`, which
            // may be inherited or user-written), which the tree-walker owns.
            let ks = vm_property_key(ctx, key)?;
            // A canonical numeric string key on an array (`arr["0"]`) reads the
            // element, like `arr[0]` — for a valid array index [0, 2**32−1); the
            // boundary value is an ordinary property.
            if ctx.realm.is_array(handle)
                && let Ok(i) = ks.parse::<usize>()
                && alloc::format!("{i}") == ks
                && (i as u64) < u64::from(u32::MAX)
            {
                vm_array_index_get(ctx, funcs, handle, i, recv)
            } else if ks == "length"
                && let Some(len) = ctx.realm.array_length(handle).or_else(|| {
                    // String length counts UTF-16 code units (astral chars = 2, a
                    // lone surrogate = 1).
                    ctx.realm.string_utf16_len(handle)
                })
            {
                // Computed `arr["length"]` / `str["length"]`.
                Ok(NanBox::number(len as f64))
            } else if let Some((getter, _)) = ctx.realm.accessor(handle, &ks)
                && getter.as_handle().is_some()
            {
                // A getter accessor under a (possibly numeric) string key — invoke
                // it with the receiver as `this`.
                call_closure(ctx, funcs, getter, &[], recv)
            } else {
                Ok(ctx
                    .realm
                    .get_property(handle, &ks)
                    .unwrap_or(NanBox::undefined()))
            }
        }
    }
}

/// The interpreter's real computed element **set** `obj[key] = value`
/// (`Op::SetKey`'s semantics), factored out so the bytecode VM **and** the
/// generic-JIT runtime helper ([`jit_helper_set_elem`]) share one code path: a
/// typed-array element (ToBigInt-coercing for a BigInt view), a canonical array
/// index (with the refuse-past-cap → catchable `RangeError`), a computed
/// `arr.length` resize, `regex.lastIndex`, a canonical numeric-string array index,
/// else an ordinary property store. Returns `Err(VmError::Unsupported)` for the
/// descriptor-aware cases the tree-walker owns (a demoted/frozen array index, a
/// non-writable `length`) — the same fault the interpreter raised to fall back —
/// and `Err(Thrown)` for a `RangeError` / a BigInt-coercion `TypeError`.
fn vm_set_elem(
    ctx: &mut Ctx,
    _funcs: &[FnProto],
    recv: NanBox,
    key: NanBox,
    value: NanBox,
) -> Result<(), VmError> {
    let handle = recv
        .as_handle()
        .map(Handle::from_raw)
        .ok_or(VmError::NotAnObject)?;
    match key.as_number() {
        Some(n) if ctx.realm.typed_len(handle).is_some() => {
            // A numeric key on a typed array writes through to its bytes; a BigInt
            // element ToBigInt-coerces (Number throws).
            let i = n as usize;
            match coerce_bigint_typed_write(ctx.realm, handle, value) {
                Ok(v) => {
                    ctx.realm.set_element(handle, i, v);
                    Ok(())
                }
                Err(e) => Err(VmError::Thrown(e)),
            }
        }
        // A canonical non-negative integer index in [0, 2**32−1). A negative or
        // fractional key (`a[-1]`, `a[1.5]`) is NOT an array index — `as u64` would
        // truncate it to a real index — so it falls through to the `_` arm and
        // becomes an ordinary property.
        Some(n)
            if ctx.realm.is_array(handle)
                && n >= 0.0
                && n == (n as u64) as f64
                && (n as u64) < u64::from(u32::MAX) =>
        {
            // C1: refuse-past-cap surfaces as a catchable RangeError (see
            // `Op::SetElem`).
            let i = n as usize;
            if i >= ctx.realm.limits.max_array_len {
                let e = make_error(ctx.realm, "RangeError", "Invalid array length");
                return Err(VmError::Thrown(e));
            }
            // A demoted / accessor index (or frozen/sealed array) needs the
            // descriptor-aware tree-walker store. Fault to it.
            if ctx.realm.array_index_has_override(handle, i) {
                return Err(VmError::Unsupported);
            }
            ctx.realm.set_element(handle, i, value);
            Ok(())
        }
        _ => {
            // ToPropertyKey: a Symbol keys on its `\0sym:` name; any other object
            // key needs the full ToPrimitive (its `@@toPrimitive`/`toString`, which
            // may be inherited or user-written), which the tree-walker owns.
            let ks = vm_property_key(ctx, key)?;
            // C1: a computed `arr["length"] = n` (numeric string key) on an array
            // resizes; `ToUint32(v)` must equal `ToNumber(v)` (else a catchable
            // `RangeError`).
            if ctx.realm.is_array(handle) && ks == "length" {
                if let Some(n) = ctx.realm.array_length_uint32(value) {
                    if ctx
                        .realm
                        .array_length_set_needs_slow_path(handle, n as usize)
                    {
                        return Err(VmError::Unsupported);
                    }
                    ctx.realm.set_array_length(handle, n as usize);
                } else {
                    let e = make_error(ctx.realm, "RangeError", "Invalid array length");
                    return Err(VmError::Thrown(e));
                }
                return Ok(());
            }
            if ks == "lastIndex" && ctx.realm.regexp_at(handle).is_some() {
                set_regex_last_index_value(ctx.realm, handle, value);
                return Ok(());
            }
            // A canonical numeric string key on an array (`arr["0"] = v`) addresses
            // element storage, like `arr[0] = v` — for a valid index in
            // [0, 2**32−1). A demoted/accessor index faults to the tree-walker.
            if ctx.realm.is_array(handle)
                && let Ok(i) = ks.parse::<usize>()
                && alloc::format!("{i}") == ks
                && (i as u64) < u64::from(u32::MAX)
            {
                if i >= ctx.realm.limits.max_array_len {
                    let e = make_error(ctx.realm, "RangeError", "Invalid array length");
                    return Err(VmError::Thrown(e));
                }
                if ctx.realm.array_index_has_override(handle, i) {
                    return Err(VmError::Unsupported);
                }
                ctx.realm.set_element(handle, i, value);
                return Ok(());
            }
            ctx.realm.set_property(handle, &ks, value);
            Ok(())
        }
    }
}

/// The interpreter's real `.length` read (`Op::ArrayLen`'s semantics), factored
/// out so the bytecode VM **and** the generic-JIT runtime helper
/// ([`jit_helper_array_len`]) share one code path: a VM function's declared
/// parameter count (from its proto), an array's length or a string's UTF-16
/// code-unit count, else an explicit `length` data property (e.g. a regex match
/// result). `Err(NotAnObject)` when `recv` is not a heap object.
fn vm_array_len(ctx: &mut Ctx, funcs: &[FnProto], recv: NanBox) -> Result<NanBox, VmError> {
    let handle = recv
        .as_handle()
        .map(Handle::from_raw)
        .ok_or(VmError::NotAnObject)?;
    // A VM function (a tagged closure array) reports its parameter count from the
    // proto, not the backing array's length.
    if ctx.realm.is_vm_function(handle) {
        let n = ctx
            .realm
            .get_element(handle, 0)
            .as_number()
            .and_then(|f| funcs.get(f as usize))
            .map_or(0, |p| p.length);
        Ok(NanBox::number(n as f64))
    } else {
        // `.length` on an array, or a string's UTF-16 code-unit count (astral
        // chars = 2, a lone surrogate = 1).
        let len = ctx
            .realm
            .array_length(handle)
            .or_else(|| ctx.realm.string_utf16_len(handle));
        Ok(match len {
            Some(n) => NanBox::number(n as f64),
            // Otherwise an explicit `length` data property (e.g. a regex match
            // result, which is object-shaped here), else undefined.
            None => ctx
                .realm
                .get_property(handle, "length")
                .unwrap_or(NanBox::undefined()),
        })
    }
}

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

/// Slices a pre-collected `&[u16]` regex subject over the **code-unit** range
/// `[st, en)` and re-encodes it to WTF-8 bytes (lone surrogates preserved). The
/// regex engine's native subject model is UTF-16 code units, so match/capture
/// spans index this buffer directly.
#[cfg(feature = "regex")]
fn u16_substr(units: &[u16], st: usize, en: usize) -> alloc::vec::Vec<u8> {
    let st = st.min(units.len());
    let en = en.min(units.len()).max(st);
    crate::wtf8::from_utf16(&units[st..en])
}

/// Slices a pre-collected `&[u16]` regex subject from code-unit index `st` to
/// the end, re-encoded to WTF-8 bytes.
#[cfg(feature = "regex")]
fn u16_substr_from(units: &[u16], st: usize) -> alloc::vec::Vec<u8> {
    crate::wtf8::from_utf16(&units[st.min(units.len())..])
}

/// Advances a code-unit position past an empty match (`AdvanceStringIndex`): a
/// `u`-flag regex steps a whole code point (over a surrogate pair), a non-`u`
/// regex steps one code unit.
#[cfg(feature = "regex")]
fn advance_index_u16(units: &[u16], i: usize, unicode: bool) -> usize {
    if unicode
        && i + 1 < units.len()
        && (0xD800..=0xDBFF).contains(&units[i])
        && (0xDC00..=0xDFFF).contains(&units[i + 1])
    {
        i + 2
    } else {
        i + 1
    }
}

/// Builds a regex match result object `{ 0: whole, 1: g1, …, index, input,
/// groups, length }` (the shape `RegExp.exec` / `String.match` return) from a
/// `Captures` whose spans are **code-unit** indices into the pre-collected
/// `&[u16]` subject. Slices are re-encoded to WTF-8 so astral characters survive.
#[cfg(feature = "regex")]
fn regex_match_object(
    realm: &mut Realm,
    units: &[u16],
    input: NanBox,
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
            Some((s, e)) => {
                NanBox::handle(realm.new_string_wtf8(u16_substr(units, *s, *e)).to_raw())
            }
            None => NanBox::undefined(),
        })
        .collect();
    let obj = realm.new_array(elems);
    let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
    realm.set_property(obj, "index", NanBox::number(index as f64));
    realm.set_property(obj, "input", input);
    // `.groups`: an object of named captures (or `undefined` if none).
    let groups = if group_names.is_empty() {
        NanBox::undefined()
    } else {
        // Null-prototype container (per spec); duplicate names resolve to whichever
        // occurrence participated (a `Some` capture wins, never clobbered by a later
        // non-participating `None` duplicate).
        let g = realm.new_object_with_proto(None);
        for (idx, name) in group_names {
            let v = match caps.groups.get(*idx).and_then(|x| *x) {
                Some((s, e)) => {
                    NanBox::handle(realm.new_string_wtf8(u16_substr(units, s, e)).to_raw())
                }
                None => NanBox::undefined(),
            };
            if !v.is_undefined() || realm.get_property(g, name).is_none() {
                realm.set_property(g, name, v);
            }
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
    let h = recv.as_handle().map(Handle::from_raw)?;
    let arg0 = args.first().copied().unwrap_or(NanBox::undefined());

    // `re.test(s)` / `re.exec(s)`.
    if let Some((_source, flags)) = ctx.realm.regexp_at(h) {
        if !matches!(key, "test" | "exec") {
            return None;
        }
        // Defer to the spec-accurate tree-walker (`RegExpBuiltinExec`) for the
        // cases this fast path does not model exactly:
        //  - a non-canonical own `lastIndex` (e.g. an object whose `valueOf` must
        //    run via `ToLength`, or a value that must be read but not written for a
        //    non-global/non-sticky regex),
        //  - the sticky (`y`) flag (anchored match at exactly `lastIndex`),
        //  - the `d` (hasIndices) flag (the result needs an `.indices` array).
        //  - a subject that is not already a String, since `ToString(S)` may run a
        //    user `toString`/`valueOf` (and a `Symbol` must throw). This path has
        //    no way to call back into JS, and coercing with the *internal* display
        //    conversion instead silently skipped those — `/z/.test({toString(){…}})`
        //    matched against `"[object Object]"` without ever calling the method.
        if ctx.realm.regex_aux_last_index_defined(h)
            || flags.contains('y')
            || flags.contains('d')
            || !arg0
                .as_handle()
                .is_some_and(|raw| ctx.realm.is_string_handle(Handle::from_raw(raw)))
        {
            return None;
        }
        // Use the RegExp cell's compiled-program cache (RE-P1): a reused regex is
        // compiled once, not per call. Clone the `Rc` out before any match work.
        let Some(re) = ctx.realm.regex_compiled(h) else {
            return Some(Ok(NanBox::null()));
        };
        // `g`/`y` regexes resume at `lastIndex` and update it (reset to 0 on
        // miss). `lastIndex` and all reported spans are **code-unit** indices.
        let stateful = flags.contains('g') || flags.contains('y');
        let start = if stateful {
            ctx.realm.regex_last_index(h)
        } else {
            0
        };
        // The subject as UTF-16 code units; spans index this buffer. A string
        // argument goes through the realm's one-slot memo, so matching the same
        // subject repeatedly — a `test`/`exec` loop — transcodes it once rather
        // than once per call. Anything else is coerced and transcoded here (a
        // fresh string every time, so there is nothing to reuse).
        let cached = arg0
            .as_handle()
            .and_then(|raw| ctx.realm.regex_subject_units(Handle::from_raw(raw)));
        let owned: Vec<u16>;
        let units: &[u16] = match &cached {
            Some(u) => u,
            None => {
                let subject_bytes = string_arg_bytes(ctx, arg0);
                owned = crate::wtf8::utf16_units(&subject_bytes).collect();
                &owned
            }
        };
        let caps = re.captures_in_u16(units, start);
        if stateful {
            let next = caps.as_ref().map_or(0, |c| c.whole().1);
            ctx.realm.set_regex_last_index(h, next);
        }
        return Some(Ok(match (key, caps) {
            ("test", c) => NanBox::boolean(c.is_some()),
            (_, Some(caps)) => {
                // Materialized only here: `exec`'s result carries the subject as
                // its `input` property. `test`, and a failed `exec`, never need
                // it — and copying it unconditionally cost a full copy of the
                // subject on every call.
                let text = ctx.realm.to_display_string(arg0);
                let input = NanBox::handle(ctx.realm.new_string(&text).to_raw());
                regex_match_object(ctx.realm, units, input, &caps, re.group_names())
            }
            (_, None) => NanBox::null(),
        }));
    }

    // `str.match/replace/replaceAll/split/search(re)` — only when the argument
    // is a RegExp (string-argument forms stay in `builtin_method`).
    let text = ctx.realm.string_value(h)?;
    if !matches!(key, "match" | "replace" | "replaceAll" | "split" | "search") {
        return None;
    }
    let rh = arg0.as_handle().map(Handle::from_raw)?;
    let (_src, flags) = ctx.realm.regexp_at(rh)?;
    // Use the RegExp argument's compiled-program cache (RE-P1). Clone the `Rc` out
    // before any match work so we don't alias the heap borrow.
    let Some(re) = ctx.realm.regex_compiled(rh) else {
        return Some(Ok(NanBox::null()));
    };
    let global = flags.contains('g');
    let unicode = flags.contains('u');
    // `replaceAll` requires a global RegExp.
    if !global && key == "replaceAll" {
        return Some(Err(VmError::Thrown(make_error(
            ctx.realm,
            "TypeError",
            "replaceAll must be called with a global RegExp",
        ))));
    }
    // Collect the subject to UTF-16 code units ONCE (RE-7); the per-match loops
    // below index this `&[u16]` via the engine's `*_in_u16` API instead of
    // re-collecting per call (which made dense global matches O(n²)). All match,
    // capture, `.index`, and split positions are **code-unit** indices. (The
    // subject is surrogate-free on this path — a surrogate-bearing literal defers
    // to the tree-walker at compile time — but the unit model is what makes an
    // astral subject report code-unit indices.)
    let subject_bytes = ctx.realm.string_bytes(h).unwrap_or_default();
    let units: Vec<u16> = crate::wtf8::utf16_units(&subject_bytes).collect();
    let len = units.len();
    let result = match key {
        "search" => {
            let i = re.find_in_u16(&units, 0).map_or(-1.0, |(s, _)| s as f64);
            NanBox::number(i)
        }
        "match" if !global => match re.captures_in_u16(&units, 0) {
            Some(caps) => {
                let input = NanBox::handle(ctx.realm.new_string(&text).to_raw());
                regex_match_object(ctx.realm, &units, input, &caps, re.group_names())
            }
            None => NanBox::null(),
        },
        "match" => {
            // Global match → an array of the whole matches.
            let mut out = Vec::new();
            let mut pos = 0;
            while let Some((s, e)) = re.find_in_u16(&units, pos) {
                out.push(NanBox::handle(
                    ctx.realm.new_string_wtf8(u16_substr(&units, s, e)).to_raw(),
                ));
                pos = if e > s {
                    e
                } else {
                    advance_index_u16(&units, e, unicode)
                };
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
            let templ = ctx.realm.to_display_string(repl_val);
            // Build the result as WTF-8 bytes so astral characters survive; spans
            // are code-unit indices. `$&`/`` $` ``/`$'`/`$1`/`$<name>` expand over
            // the unit buffer.
            let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
            let mut at = 0;
            while let Some(caps) = re.captures_in_u16(&units, at) {
                let (st, en) = caps.groups[0].unwrap_or((at, at));
                out.extend_from_slice(&u16_substr(&units, at, st));
                expand_replacement_u16(&templ, &units, &caps, re.group_names(), &mut out);
                if en > st {
                    at = en;
                } else {
                    if en >= len {
                        break;
                    }
                    let next = advance_index_u16(&units, en, unicode);
                    out.extend_from_slice(&u16_substr(&units, en, next));
                    at = next;
                }
                if !global {
                    break;
                }
            }
            out.extend_from_slice(&u16_substr_from(&units, at));
            NanBox::handle(ctx.realm.new_string_wtf8(out).to_raw())
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
            while search < len && limit.is_none_or(|l| out.len() < l) {
                let Some(caps) = re.captures_in_u16(&units, search) else {
                    break;
                };
                let Some((st, en)) = caps.groups[0] else {
                    break;
                };
                if en == seg_start {
                    let next = search.max(st);
                    if next < len {
                        search = advance_index_u16(&units, next, unicode);
                        continue;
                    }
                    break;
                }
                out.push(NanBox::handle(
                    ctx.realm
                        .new_string_wtf8(u16_substr(&units, seg_start, st))
                        .to_raw(),
                ));
                for g in &caps.groups[1..] {
                    out.push(match g {
                        Some((gs, ge)) => NanBox::handle(
                            ctx.realm
                                .new_string_wtf8(u16_substr(&units, *gs, *ge))
                                .to_raw(),
                        ),
                        None => NanBox::undefined(),
                    });
                }
                seg_start = en;
                search = if en > st {
                    en
                } else {
                    advance_index_u16(&units, en, unicode)
                };
            }
            if limit.is_none_or(|l| out.len() < l) {
                out.push(NanBox::handle(
                    ctx.realm
                        .new_string_wtf8(u16_substr_from(&units, seg_start))
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

/// The WTF-8 bytes of `v` coerced to a string — lossless when `v` is already a
/// string (so a surrogate-bearing subject matches correctly), lossy otherwise.
#[cfg(feature = "regex")]
fn string_arg_bytes(ctx: &Ctx, v: NanBox) -> alloc::vec::Vec<u8> {
    if let Some(raw) = v.as_handle()
        && let Some(b) = ctx.realm.string_bytes(Handle::from_raw(raw))
    {
        return b;
    }
    ctx.realm.to_display_string(v).into_bytes()
}

/// Expands a `replace` template against `caps`, appending WTF-8 bytes to `out`:
/// `$&` (whole match), `` $` `` (prefix), `$'` (suffix), `$1`..`$9` (numbered
/// groups), `$<name>` (named groups), `$$` (literal `$`). `subj` is the
/// pre-collected `&[u16]` subject and capture spans are **code-unit** indices;
/// substituted slices are re-encoded to WTF-8 so astral characters survive.
#[cfg(feature = "regex")]
fn expand_replacement_u16(
    templ: &str,
    subj: &[u16],
    caps: &crate::regex::Captures,
    group_names: &[(usize, alloc::string::String)],
    out: &mut alloc::vec::Vec<u8>,
) {
    let group = |i: usize| -> alloc::vec::Vec<u8> {
        caps.groups
            .get(i)
            .and_then(|g| *g)
            .map(|(s, e)| u16_substr(subj, s, e))
            .unwrap_or_default()
    };
    let (m_start, m_end) = caps.groups.first().and_then(|g| *g).unwrap_or((0, 0));
    let mut tc = templ.chars().peekable();
    while let Some(c) = tc.next() {
        if c == '$' && tc.peek() == Some(&'<') {
            tc.next();
            let mut name = String::new();
            while let Some(&ch) = tc.peek() {
                tc.next();
                if ch == '>' {
                    break;
                }
                name.push(ch);
            }
            if let Some((idx, _)) = group_names.iter().find(|(_, n)| *n == name) {
                out.extend_from_slice(&group(*idx));
            }
            continue;
        }
        if c != '$' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        match tc.peek() {
            Some('$') => {
                out.push(b'$');
                tc.next();
            }
            Some('&') => {
                out.extend_from_slice(&group(0));
                tc.next();
            }
            Some('`') => {
                out.extend_from_slice(&u16_substr(subj, 0, m_start));
                tc.next();
            }
            Some('\'') => {
                out.extend_from_slice(&u16_substr_from(subj, m_end));
                tc.next();
            }
            Some(d)
                if d.is_ascii_digit() && {
                    let n = (*d as u8 - b'0') as usize;
                    n >= 1 && n < caps.groups.len()
                } =>
            {
                let n = (*d as u8 - b'0') as usize;
                tc.next();
                out.extend_from_slice(&group(n));
            }
            _ => out.push(b'$'),
        }
    }
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
        // `push`/`pop`/`shift`/`unshift` on an array whose `length` is non-writable
        // (frozen or defineProperty-demoted) throw a TypeError — every one of them
        // ends with `Set(O, "length", …, Throw=true)`, which is unconditional even
        // for a no-arg `push()`/`unshift()` or a `pop()`/`shift()` on an empty array
        // (the spec still performs `Set(O, "length", +0𝔽, true)`). Mirror of the
        // nbexec path (method_dispatch.rs); NBVM implements push/pop natively and
        // lets shift/unshift fall through, so guard all four here before either.
        if matches!(key, "push" | "pop" | "shift" | "unshift")
            && ctx.realm.array_length_is_readonly(h)
        {
            let e = make_error(
                ctx.realm,
                "TypeError",
                "Cannot assign to read only property 'length' of object '[object Array]'",
            );
            return Some(Err(VmError::Thrown(e)));
        }
        // `map`/`filter`/`concat` perform ArraySpeciesCreate to build their result
        // (reading `O.constructor` / its `@@species`), which can select a custom
        // constructor, validate it (a non-object / non-constructor `constructor` is
        // a TypeError), or produce an exotic result object. These VM fast paths
        // always build a plain array, so when the receiver carries an own
        // `constructor` property (the only way to reach a non-default species here)
        // they are non-conformant — defer the whole program to the spec-accurate
        // tree-walker, which runs ArraySpeciesCreate.
        if matches!(key, "map" | "filter" | "concat") && ctx.realm.has_own(h, "constructor") {
            return None;
        }
        // `elems` snapshots (clones) the backing store. It is used only by the
        // callback-taking methods (`map`/`filter`/`forEach`/`reduce`/`find`/
        // `some`/`every`) and the array-building ones (`concat`/`reverse`): a
        // user callback can re-enter and mutate `h` mid-iteration, so iterating a
        // live borrow would be both a borrow-checker conflict and a re-entrancy
        // hazard. The pure scans below (`join`/`includes`/`indexOf`) take no
        // callback, so they iterate a borrowed `array_elements(h)` directly and
        // skip the clone (M4).
        let elems = |ctx: &Ctx| {
            ctx.realm
                .array_elements(h)
                .map(<[_]>::to_vec)
                .unwrap_or_default()
        };
        // A *sparse* array (one with at least one hole) needs the conformant
        // tree-walker for the callback / element-scanning methods: holes must be
        // skipped (and inherited prototype indices observed). These VM fast paths
        // treat every slot as present, so defer (return `None` → whole-program
        // re-run on the reference engine) when a hole is present.
        if matches!(
            key,
            "map"
                | "filter"
                | "forEach"
                | "reduce"
                | "reduceRight"
                | "find"
                | "findIndex"
                | "findLast"
                | "findLastIndex"
                | "some"
                | "every"
                | "indexOf"
                | "lastIndexOf"
                | "flat"
                | "flatMap"
        ) && ctx
            .realm
            .array_elements(h)
            .is_some_and(|a| a.iter().any(|e| e.is_hole()))
        {
            return None;
        }
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
                // `ToString` of the separator and of every element: a Symbol has no
                // string conversion and must throw a TypeError — `to_display_string`
                // would render `Symbol(x)`. Defer to the tree-walker, which raises it.
                let is_symbol = |v: NanBox| {
                    v.as_handle()
                        .is_some_and(|raw| ctx.realm.symbol_at(Handle::from_raw(raw)).is_some())
                };
                if is_symbol(arg0())
                    || ctx
                        .realm
                        .array_elements(h)
                        .is_some_and(|elems| elems.iter().copied().any(is_symbol))
                {
                    return None;
                }
                let sep = if matches!(arg0().unpack(), Unpacked::Undefined) {
                    String::from(",")
                } else {
                    ctx.realm.to_display_string(arg0())
                };
                // Pure scan, no callback: borrow the backing store (M4). Both
                // `array_elements` and `to_display_string` take `&self`, so the
                // shared borrow held across the map is sound.
                let parts: Vec<String> = ctx
                    .realm
                    .array_elements(h)
                    .map(|elems| {
                        elems
                            .iter()
                            .map(|e| match e.unpack() {
                                Unpacked::Undefined | Unpacked::Null => String::new(),
                                // A direct self-reference renders empty (no recursion).
                                Unpacked::Handle(raw) if raw == h.to_raw() => String::new(),
                                _ => ctx.realm.to_display_string(*e),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                NanBox::handle(ctx.realm.new_string(&parts.join(&sep)).to_raw())
            }
            "includes" => {
                let t = arg0();
                // SameValueZero: like `===` but `NaN` matches `NaN`. A hole reads
                // as `undefined` (holes are not skipped by `includes`).
                let t_nan = t.as_number().is_some_and(f64::is_nan);
                let t_undef = matches!(t.unpack(), Unpacked::Undefined);
                // Pure scan, no callback: borrow the backing store (M4).
                let found = ctx.realm.array_elements(h).is_some_and(|elems| {
                    elems.iter().any(|e| {
                        (e.is_hole() && t_undef)
                            || ctx.realm.strict_equals(*e, t)
                            || (t_nan && e.as_number().is_some_and(f64::is_nan))
                    })
                });
                NanBox::boolean(found)
            }
            "indexOf" => {
                let t = arg0();
                // Pure scan, no callback: borrow the backing store (M4).
                let i = ctx
                    .realm
                    .array_elements(h)
                    .and_then(|elems| elems.iter().position(|e| ctx.realm.strict_equals(*e, t)));
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
            "trim" => NanBox::handle(
                ctx.realm
                    .new_string(s.trim_matches(crate::realm::is_js_whitespace))
                    .to_raw(),
            ),
            "includes" => NanBox::boolean(s.contains(&ctx.realm.to_display_string(arg0()))),
            "startsWith" => NanBox::boolean(s.starts_with(&ctx.realm.to_display_string(arg0()))),
            "endsWith" => NanBox::boolean(s.ends_with(&ctx.realm.to_display_string(arg0()))),
            "indexOf" => {
                let needle = ctx.realm.to_display_string(arg0());
                let i = s.find(&needle).map(|b| s[..b].chars().count());
                NanBox::number(i.map_or(-1.0, |i| i as f64))
            }
            "repeat" => {
                // A negative or non-finite count is a `RangeError`; a finite count
                // whose product with the length overflows `usize` or exceeds the
                // max string length would alloc-abort, so it is a `RangeError` too
                // (an unrepresentable string length).
                let nf = ctx.realm.to_number(arg0());
                let n = nf as usize;
                let total = n.checked_mul(s.len());
                if nf < 0.0
                    || !nf.is_finite()
                    || total.is_none_or(|t| t > ctx.realm.limits.max_string_len)
                {
                    let e = make_error(ctx.realm, "RangeError", "Invalid string length");
                    return Some(Err(VmError::Thrown(e)));
                }
                NanBox::handle(ctx.realm.new_string(&s.repeat(n)).to_raw())
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
                // The native fast path only handles a plain string / number /
                // boolean / null / undefined separator and a plain numeric limit.
                // A separator that is an object (a RegExp, or a custom
                // `toString`/`@@split`), a String wrapper, or a Symbol — or an
                // object `limit` — needs interpreter-aware coercion / delegation
                // and the exact `ToString`-before-`limit === 0` ordering, so fault
                // to the tree-walker's full `String.prototype.split`.
                let needs_walk = |v: NanBox| {
                    v.as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|hh| !ctx.realm.is_string_handle(hh))
                };
                if needs_walk(arg0()) || args.get(1).copied().is_some_and(needs_walk) {
                    return None;
                }
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

/// VM-side `Array.from(items, mapFn?, thisArg?)` — handled in the VM loop (not
/// `call_native`) because it must validate/invoke a `mapFn` closure and throw
/// (`null`/`undefined` items, a non-callable `mapFn`). Element extraction covers
/// arrays, strings, Sets/Maps, and array-likes; the result is mapped through
/// `mapFn` when supplied.
fn vm_array_from(ctx: &mut Ctx, funcs: &[FnProto], args: &[NanBox]) -> Result<NanBox, VmError> {
    use crate::nanbox::Unpacked;
    let items_box = args.first().copied().unwrap_or(NanBox::undefined());
    let map_fn = args.get(1).copied().unwrap_or(NanBox::undefined());
    let this_arg = args.get(2).copied().unwrap_or(NanBox::undefined());
    let has_map = !matches!(map_fn.unpack(), Unpacked::Undefined);
    // A defined mapFn must be callable (checked before reading items).
    if has_map
        && !map_fn
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| ctx.realm.is_vm_function(h) || ctx.realm.is_callable_cell(h))
    {
        return Err(VmError::Thrown(make_error(
            ctx.realm,
            "TypeError",
            "Array.from mapFn is not a function",
        )));
    }
    // items is ToObject'd — null/undefined is a TypeError.
    if matches!(items_box.unpack(), Unpacked::Undefined | Unpacked::Null) {
        return Err(VmError::Thrown(make_error(
            ctx.realm,
            "TypeError",
            "Array.from requires an array-like or iterable object, not null/undefined",
        )));
    }
    // Extract the raw element list. Check collections and strings before the
    // generic array branch (a Set/Map is not an Array, but order defensively).
    let raw: Vec<NanBox> = match items_box.as_handle().map(Handle::from_raw) {
        Some(h) if ctx.realm.collection_is_set(h).is_some() => {
            let is_set = ctx.realm.collection_is_set(h) == Some(true);
            ctx.realm
                .collection_entries(h)
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| {
                    if is_set {
                        // A Set yields its elements.
                        k
                    } else {
                        // A Map yields `[key, value]` pairs.
                        NanBox::handle(ctx.realm.new_array(alloc::vec![k, v]).to_raw())
                    }
                })
                .collect()
        }
        Some(h) if ctx.realm.is_array(h) && !ctx.realm.is_vm_function(h) => ctx
            .realm
            .array_elements(h)
            .map(<[_]>::to_vec)
            .unwrap_or_default(),
        Some(h) if ctx.realm.is_string_handle(h) => ctx
            .realm
            .string_value(h)
            .unwrap_or_default()
            .chars()
            .map(|c| NanBox::handle(ctx.realm.new_string(&String::from(c)).to_raw()))
            .collect(),
        Some(h) => {
            // Array-like: ToLength(Get(O, "length")) then Get each index.
            let len_raw = ctx
                .realm
                .get_property(h, "length")
                .map(|v| ctx.realm.to_number(v))
                .unwrap_or(0.0);
            if len_raw > ctx.realm.limits.max_array_len as f64 {
                return Err(VmError::Thrown(make_error(
                    ctx.realm,
                    "RangeError",
                    "Invalid array length",
                )));
            }
            let len = if len_raw.is_nan() || len_raw <= 0.0 {
                0
            } else {
                len_raw as usize
            };
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
    let items = if has_map {
        let mut out = Vec::with_capacity(raw.len());
        for (i, e) in raw.into_iter().enumerate() {
            let v = call_closure(ctx, funcs, map_fn, &[e, NanBox::number(i as f64)], this_arg)?;
            out.push(v);
        }
        out
    } else {
        raw
    };
    Ok(NanBox::handle(ctx.realm.new_array(items).to_raw()))
}

/// VM-side `Object.keys` / `Object.values` / `Object.entries` — handled in the VM
/// loop (not `call_native`) because for an *ordinary* object it must invoke user
/// getters (`Object.values`/`entries`) live, in the spec order of
/// EnumerableOwnProperties (7.3.24): one `[[OwnPropertyKeys]]` snapshot, then per
/// key a live enumerability check and `[[Get]]`. A getter mutating a later key's
/// descriptor / existence / value is therefore observed. A `null`/`undefined`
/// receiver throws a TypeError; a Proxy has no VM trap machinery, so it defers to
/// the tree-walker (`VmError::Unsupported`); everything else (arrays / strings /
/// typed arrays / functions — no getters in play) keeps the pure `call_native`
/// enumeration.
fn vm_object_kv(
    ctx: &mut Ctx,
    funcs: &[FnProto],
    native: u16,
    args: &[NanBox],
) -> Result<NanBox, VmError> {
    use crate::nanbox::Unpacked;
    let recv = args.first().copied().unwrap_or(NanBox::undefined());
    if matches!(recv.unpack(), Unpacked::Null | Unpacked::Undefined) {
        let e = make_error(
            ctx.realm,
            "TypeError",
            "Object.keys/values/entries called on null or undefined",
        );
        return Err(VmError::Thrown(e));
    }
    if let Some(h) = recv.as_handle().map(Handle::from_raw) {
        // A proxy's `ownKeys`/`getOwnPropertyDescriptor`/`get` traps are not modeled
        // in the VM — re-run the program on the tree-walker, which drives them.
        if ctx.realm.proxy_at(h).is_some() {
            return Err(VmError::Unsupported);
        }
        // Ordinary object: live, getter-invoking, spec-ordered enumeration.
        if ctx.realm.object_keys(h).is_some() {
            let keys = ctx.realm.own_property_names(h).unwrap_or_default();
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                // Only engine-internal slots (the `\u{0}` sentinel prefix — private
                // elements, generator state, wrapper boxes, …) are hidden; a public
                // property whose *name* legitimately starts with `#` (e.g. a computed
                // field `["#x"]`) enumerates normally.
                if k.starts_with('\u{0}') {
                    continue;
                }
                // Live `[[GetOwnProperty]]` — an ordinary object has no observable
                // side effect, so an own+enumerable probe is equivalent.
                if !(ctx.realm.has_own(h, &k) && ctx.realm.property_is_enumerable(h, &k)) {
                    continue;
                }
                let item = match native {
                    NB_OBJECT_KEYS => NanBox::handle(ctx.realm.new_string(&k).to_raw()),
                    _ => {
                        // A fresh cache per key: the inline cache keys only on the
                        // receiver's shape pointer (one property per call site), so
                        // reusing it across distinct keys would false-hit.
                        let mut cache = PropertyCache::new();
                        let v = vm_get_prop(ctx, funcs, recv, &k, &mut cache)?;
                        if native == NB_OBJECT_VALUES {
                            v
                        } else {
                            let key = NanBox::handle(ctx.realm.new_string(&k).to_raw());
                            NanBox::handle(ctx.realm.new_array(alloc::vec![key, v]).to_raw())
                        }
                    }
                };
                out.push(item);
            }
            return Ok(NanBox::handle(ctx.realm.new_array(out).to_raw()));
        }
    }
    // Arrays / strings / typed arrays / functions: the pure element+named path.
    Ok(call_native(ctx, native, args))
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
            // `ToInt32`: the radix wraps modulo 2^32, so `parseInt("11",
            // 4294967298)` is base 2 rather than an out-of-range radix.
            let radix = args
                .get(1)
                .filter(|r| !r.is_undefined())
                .map_or(0i64, |r| i64::from(ctx.realm.to_int32(*r)));
            if radix != 0 && !(2..=36).contains(&radix) {
                NanBox::number(f64::NAN)
            } else {
                NanBox::number(parse_int(
                    s.trim_matches(crate::realm::is_js_whitespace),
                    radix as u32,
                ))
            }
        }
        NB_PARSE_FLOAT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            NanBox::number(parse_float_prefix(
                s.trim_matches(crate::realm::is_js_whitespace),
            ))
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
                        // Holes are absent — excluded from keys/values/entries.
                        if !v.is_hole() {
                            pairs.push((alloc::format!("{i}"), v));
                        }
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
            NanBox::number(parse_float_prefix(
                s.trim_matches(crate::realm::is_js_whitespace),
            ))
        }
        NB_NUMBER_PARSE_INT => {
            let s = ctx
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            // See NB_PARSE_INT: preserve the sign and reject an out-of-range radix.
            // `ToInt32`: the radix wraps modulo 2^32, so `parseInt("11",
            // 4294967298)` is base 2 rather than an out-of-range radix.
            let radix = args
                .get(1)
                .filter(|r| !r.is_undefined())
                .map_or(0i64, |r| i64::from(ctx.realm.to_int32(*r)));
            if radix != 0 && !(2..=36).contains(&radix) {
                NanBox::number(f64::NAN)
            } else {
                NanBox::number(parse_int(
                    s.trim_matches(crate::realm::is_js_whitespace),
                    radix as u32,
                ))
            }
        }
        NB_STRING_FROM_CHAR_CODE => {
            // Each argument is ToUint16'd into a UTF-16 code unit; the resulting
            // sequence is decoded to **WTF-8**, so an adjacent high/low surrogate
            // pair combines into one astral code point and a lone surrogate is
            // preserved (DOMString semantics — a JS string is a code-unit
            // sequence, not necessarily well-formed Unicode).
            //
            // Decoding through a Rust `String` instead, as this did, replaced a
            // lone surrogate with U+FFFD — so `String.fromCharCode(0xD800)` came
            // back as the replacement character and every downstream operation
            // (`charCodeAt`, regex matching, `indexOf`) saw the wrong string. The
            // tree-walker has always used the WTF-8 path here.
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
            let bytes = crate::wtf8::from_utf16(&units);
            NanBox::handle(ctx.realm.new_string_wtf8(bytes).to_raw())
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
                Some(h) if ctx.realm.is_string_handle(h) => ctx
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
                        let key_el = ctx.realm.get_element(ph, 0);
                        let k = ctx.realm.to_display_string(key_el);
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
    // The greedy scan above accepts characters that *could* continue a decimal
    // literal, so it can overshoot: `"1ex"` consumes `1e`, which is not a valid
    // literal. `StrDecimalLiteral` is the longest **valid** prefix, so give back
    // the trailing characters until what remains parses (`1e` -> `1`). At most a
    // few, since only a dangling exponent (`e`, `e+`, `e-`) or `.` can overshoot.
    while end > 0 {
        if let Ok(v) = s[..end].parse::<f64>() {
            return v;
        }
        end -= 1;
    }
    f64::NAN
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
    execute_with_limits(source, crate::limits::Limits::default())
}

/// Like [`execute`], but with caller-supplied resource
/// [`Limits`](crate::limits::Limits). The limits flow into the realm of both the
/// bytecode path and the tree-walker fallback.
///
/// # Errors
/// Returns a parse or execution error message.
#[cfg(feature = "std")]
pub fn execute_with_limits(
    source: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), String> {
    let program =
        crate::parser::Parser::parse_program(source).map_err(|e| alloc::format!("{e}"))?;
    // Compile to bytecode; an unsupported construct routes the whole program to
    // the tree-walker (compilation happens before execution, so no output has
    // been produced yet — the fallback is clean).
    let Ok(protos) = compile_program(&program) else {
        return crate::nbexec::eval_source_with_limits(source, limits);
    };
    let mut realm = Realm::with_limits(limits);
    match run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, output)) => Ok((output, realm.to_display_string(value))),
        // A runtime fault on the bytecode path (an unsupported coercion, etc.):
        // re-run on the reference tree-walker.
        Err(_) => crate::nbexec::eval_source_with_limits(source, limits),
    }
}

/// Like [`execute_with_limits`], but the captured output is returned on the
/// error path too — see [`crate::nbexec::eval_source_capturing`].
///
/// Compilation happens before execution, so the tree-walker fallback is still
/// clean: nothing has been printed when it is taken.
#[cfg(feature = "std")]
pub fn execute_capturing(
    source: &str,
    limits: crate::limits::Limits,
) -> (String, Result<String, String>) {
    let program = match crate::parser::Parser::parse_program(source) {
        Ok(program) => program,
        Err(e) => return (String::new(), Err(alloc::format!("{e}"))),
    };
    let Ok(protos) = compile_program(&program) else {
        return crate::nbexec::eval_source_capturing(source, limits);
    };
    let mut realm = Realm::with_limits(limits);
    match run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, output)) => (output, Ok(realm.to_display_string(value))),
        // A runtime fault on the bytecode path re-runs on the reference
        // tree-walker, which is also what surfaces the thrown value.
        Err(_) => crate::nbexec::eval_source_capturing(source, limits),
    }
}

/// Like [`execute_with_limits`], but on failure returns a structured
/// [`Thrown`](crate::nbexec::Thrown) carrying the error's JS *type* — the entry
/// point the Test262 conformance runner uses to verify a `negative` test fails
/// with the declared error type. Runs the production bytecode path; any fault
/// (an unsupported construct or a genuine JS throw) re-runs on the reference
/// tree-walker, which surfaces the typed throw with complete semantics.
///
/// # Errors
/// Returns [`Thrown`](crate::nbexec::Thrown) for a parse failure or uncaught throw.
#[cfg(feature = "std")]
pub fn execute_typed(
    source: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    let program = match crate::parser::Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            return Err(crate::nbexec::Thrown {
                phase: crate::nbexec::ErrorPhase::Parse,
                name: String::from("SyntaxError"),
                message: alloc::format!("{e}"),
            });
        }
    };
    // Run as a *Script*: a unit `parse_program` promoted to a Module (because it
    // has a top-level `import`/`export`) is an early SyntaxError here — those
    // declarations are legal only at a Module's top level. (Module tests use the
    // module loader.) Defer to nbexec, which reports the same parse-phase error.
    if program.source_type == crate::ast::SourceType::Module {
        return crate::nbexec::eval_source_typed(source, limits);
    }
    let Ok(protos) = compile_program(&program) else {
        return crate::nbexec::eval_source_typed(source, limits);
    };
    let mut realm = Realm::with_limits(limits);
    match run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, output)) => Ok((output, realm.to_display_string(value))),
        Err(_) => crate::nbexec::eval_source_typed(source, limits),
    }
}

/// Loads, links, and evaluates the ES-module graph rooted at the resolved
/// `entry_key` through `host`, returning `(console_output, completion_string)`
/// or a structured [`Thrown`](crate::nbexec::Thrown). Modules run on the
/// reference tree-walker (the bytecode tier has no module support yet); this is
/// the entry the Test262 runner uses for `flags: [module]` tests.
///
/// # Errors
/// Returns [`Thrown`](crate::nbexec::Thrown) for any parse/link/runtime failure.
#[cfg(all(feature = "std", feature = "module"))]
pub fn execute_module_typed(
    entry_key: &str,
    host: &dyn crate::nbexec::module::ModuleHost,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    crate::nbexec::module::eval_module_typed(entry_key, host, limits)
}

/// Like [`execute_module_typed`] but with a flattened error message (for the CLI
/// / embedders).
///
/// # Errors
/// Returns a human-readable message on any failure.
#[cfg(all(feature = "std", feature = "module"))]
pub fn execute_module(
    entry_key: &str,
    host: &dyn crate::nbexec::module::ModuleHost,
    limits: crate::limits::Limits,
) -> Result<(String, String), String> {
    crate::nbexec::module::eval_module(entry_key, host, limits)
}

/// Evaluates `prelude` (script) into the realm global, then runs the module
/// graph rooted at `entry_key`. The Test262 runner uses this to make the
/// `assert.js`/`sta.js` harness (and `includes`) available to a module test.
///
/// # Errors
/// Returns [`Thrown`](crate::nbexec::Thrown) for any failure.
#[cfg(all(feature = "std", feature = "module"))]
pub fn execute_module_typed_with_prelude(
    entry_key: &str,
    host: &dyn crate::nbexec::module::ModuleHost,
    prelude: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    crate::nbexec::module::eval_module_typed_with_prelude(entry_key, host, prelude, limits)
}

/// Runs `source` as an ordinary **script** but sets a base path so a dynamic
/// `import("./sibling.js")` inside it resolves relative to `base_path` (the
/// script file) rather than the process working directory. Used for the Test262
/// `language/expressions/dynamic-import/` tests, which are scripts that import
/// sibling `_FIXTURE.js` files. Returns a typed error like [`execute_typed`].
///
/// # Errors
/// Returns [`Thrown`](crate::nbexec::Thrown) for a parse failure or uncaught throw.
#[cfg(all(feature = "std", feature = "module"))]
pub fn execute_script_typed_with_import_base(
    source: &str,
    base_path: &str,
    limits: crate::limits::Limits,
) -> Result<(String, String), crate::nbexec::Thrown> {
    crate::nbexec::module::eval_script_typed_with_import_base(source, base_path, limits)
}

/// Whether `program` references the dynamic-code intrinsics `eval` or `Function`
/// (the `Function` constructor). The bytecode VM has no live tree-walk scope to
/// support direct `eval`'s scope access, and dynamic code is comparatively rare,
/// so any program touching these is routed wholesale to the reference
/// tree-walker (`crate::nbexec`), which implements them with full semantics. The
/// scan reuses the free-variable collector, so it sees references at any nesting
/// (including inside nested functions and arrows).
fn uses_dynamic_code(program: &Program) -> bool {
    let mut direct = BTreeSet::new();
    let mut nested = BTreeSet::new();
    for s in &program.body {
        refs_stmt(s, &mut direct, &mut nested);
    }
    ["eval", "Function"]
        .iter()
        .any(|name| direct.contains(*name) || nested.contains(*name))
}

/// Whether `body`'s directive prologue opens with a literal `"use strict"`
/// (a run of leading string-literal expression statements, one of which is
/// `"use strict"`). Used to decide strict mode for proper-tail-call gating.
fn body_starts_strict(body: &[Stmt]) -> bool {
    for stmt in body {
        match stmt {
            Stmt::Expr { expression, .. } => match &**expression {
                // A Use Strict Directive's *source text* must be exactly
                // `use strict` (12 source bytes with quotes) — an escaped form
                // like `'use strict'` cooks to the same value but is longer,
                // so it does not trigger strict mode.
                Expr::Str { value, span, .. }
                    if &**value == b"use strict" && span.end.saturating_sub(span.start) == 12 =>
                {
                    return true;
                }
                Expr::Str { .. } => {} // another directive — keep scanning
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}

/// Compiles `program` to a function table (function 0 is the top-level body).
///
/// # Errors
/// Returns [`CompileError`] for unsupported constructs.
pub fn compile_program(program: &Program) -> Result<Vec<FnProto>, CompileError> {
    // Dynamic code (`eval` / `Function`) needs the tree-walker (it accesses the
    // live lexical scope and parses source at runtime). Bail before any codegen
    // so the whole program runs on the reference engine with no partial output.
    if uses_dynamic_code(program) {
        return Err(CompileError::Unsupported("dynamic code (eval/Function)"));
    }
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
            // A class with an `extends` clause that is not a simple identifier
            // bound to a class declared earlier in this program may resolve to a
            // non-constructor (a primitive, a non-constructor function, a plain
            // object), which must throw a TypeError at definition. The bytecode
            // path cannot perform that check, so route such a class to the
            // tree-walker (which validates the superclass).
            if let Some(sup) = class.super_class.as_deref() {
                let known_class = matches!(sup, crate::ast::Expr::Ident(sid) if class_map.contains_key(&*sid.name));
                if !known_class {
                    return Err(CompileError::Unsupported("extends a non-class binding"));
                }
            }
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
    // The top-level program is strict iff it opens with a `"use strict"`
    // directive; a strict program makes its top-level functions strict too.
    let program_strict = body_starts_strict(&program.body);
    // Compile main (id 0), each top-level function, then each class member.
    let main = Compiler::compile_fn(
        &fn_ids,
        &classes,
        &protos,
        &[],
        &[],
        &program.body,
        true,
        program_strict,
    )?;
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
            program_strict,
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
            // Class bodies are always strict.
            true,
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
        // Canonical ECMAScript `ToString(Number)` so a non-canonical numeric
        // literal key (`0.0000001` → `"1e-7"`) matches `obj[n]` access.
        PropertyKey::Number(n) => Ok(crate::realm::js_number_string(*n)),
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
    // A constructor with an explicit `return <expr>` has subtle completion
    // semantics the bytecode `CallCtor` does not model: a returned *object*
    // overrides the new instance, and a derived constructor returning a
    // non-`undefined` non-object throws a `TypeError`. Route such classes to the
    // tree-walker, which implements the full rule.
    if let Some(m) = ctor_member
        && stmts_return_value(&m.value.body)
    {
        return Err(CompileError::Unsupported("constructor returns a value"));
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

/// Whether any statement in `body` is a `return <expr>;` belonging to *this*
/// function (control-flow is descended; nested functions/arrows/classes are
/// not, since their `return` belongs to a different frame).
fn stmts_return_value(body: &[Stmt]) -> bool {
    body.iter().any(stmt_returns_value)
}

fn stmt_returns_value(s: &Stmt) -> bool {
    match s {
        Stmt::Return {
            argument: Some(_), ..
        } => true,
        Stmt::Block { body, .. } => stmts_return_value(body),
        Stmt::If {
            consequent,
            alternate,
            ..
        } => {
            stmt_returns_value(consequent)
                || alternate.as_ref().is_some_and(|a| stmt_returns_value(a))
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => stmt_returns_value(body),
        Stmt::For { body, .. } | Stmt::ForOf { body, .. } | Stmt::ForIn { body, .. } => {
            stmt_returns_value(body)
        }
        Stmt::Labeled { body, .. } => stmt_returns_value(body),
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            stmts_return_value(block)
                || handler
                    .as_ref()
                    .is_some_and(|h| stmts_return_value(&h.body))
                || finalizer.as_ref().is_some_and(|f| stmts_return_value(f))
        }
        Stmt::Switch { cases, .. } => cases.iter().any(|c| stmts_return_value(&c.body)),
        _ => false,
    }
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
    /// Sticky flag set when register allocation would overflow the `Reg` (`u16`)
    /// width. A pathological-but-valid program (e.g. a flat literal sequence of
    /// more than 65535 elements) would otherwise integer-overflow-panic in
    /// `alloc`; instead `alloc` saturates and sets this, and `compile_fn_inner`
    /// turns it into a
    /// clean `CompileError` instead of emitting a corrupt proto.
    reg_overflow: bool,
    /// Whether a `return` compiled *right here* is a proper-tail-call candidate:
    /// true at a strict, non-async function body's top level, and preserved
    /// through ordinary statement nesting (`if`/loops/`switch`/`block`/label).
    /// A `try` clears it inside the guarded Block (and inside a `catch` when a
    /// `finally` follows), since a call there is not in tail position. Toggled
    /// (save/restore) around `try` compilation.
    tail_ok: bool,
    /// This function's strict-mode flag (its own directive or an inherited strict
    /// context). Nested functions inherit it (strict code's inner functions are
    /// strict). Distinct from [`Self::tail_ok`], which is mutated during `try`.
    strict: bool,
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
        strict: bool,
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
            strict,
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
        strict: bool,
    ) -> Result<FnProto, CompileError> {
        // Which of this function's own names are captured by nested functions →
        // must be cells.
        let cell_names = captured_names(params, body);
        // Strict mode enables proper tail calls (PTC). A function is strict if its
        // lexical context is strict or its own directive prologue says `"use
        // strict"`. Async functions are excluded (their body settles a promise, so
        // a "tail" call is not a real stack tail call).
        let strict = strict || body_starts_strict(body);
        let mut c = Compiler {
            fn_ids: alloc::rc::Rc::clone(fn_ids),
            classes: alloc::rc::Rc::clone(classes),
            protos: alloc::rc::Rc::clone(protos),
            cell_names,
            super_ctor,
            super_class,
            tail_ok: strict && !is_async,
            strict,
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
                c.apply_default_named(cur, Some(def), Some(&p.target))?;
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
        // Reject a program that exhausted the `Reg` width during allocation
        // rather than returning a proto with wrapped/aliased register indices.
        if c.reg_overflow {
            return Err(CompileError::Unsupported("too many registers"));
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
        // A program that needs more than `Reg::MAX` registers (e.g. an enormous
        // flat literal sequence) must not integer-overflow-panic here. Saturate
        // and record the overflow; `compile_fn_inner` rejects the program with a
        // `CompileError` before the corrupt proto is ever run.
        match self.next_reg.checked_add(1) {
            Some(n) => self.next_reg = n,
            None => self.reg_overflow = true,
        }
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
            // NamedEvaluation: `[x = () => {}] = …` names the anonymous function
            // after the assignment target `x`.
            if let Expr::Ident(id) = &**inner {
                let bt = BindingTarget::Ident(Ident {
                    name: id.name.clone(),
                    span: id.span,
                });
                self.apply_default_named(value_reg, Some(default), Some(&bt))?;
            } else {
                self.apply_default(value_reg, Some(default))?;
            }
            return self.assign_pattern(inner, value_reg);
        }
        self.assign_pattern(target, value_reg)
    }

    /// Binds one iteration's value to a `for-in`/`for-of` declaration head. A
    /// `let`/`const` head introduces a fresh per-iteration binding
    /// ([`Self::bind_pattern`]); a `var` head has *no* per-iteration binding — its
    /// `ForBinding` resolves the hoisted (function-scope) `var` and `PutValue`s it,
    /// so `for (var x in obj)` leaves `x` set to the last key after the loop. A
    /// plain identifier takes that assignment path (when the hoisted binding is in
    /// scope); anything else falls back to per-name binding.
    fn bind_for_decl(
        &mut self,
        kind: &crate::ast::VarDeclKind,
        target: &BindingTarget,
        value_reg: Reg,
    ) -> Result<(), CompileError> {
        if *kind == crate::ast::VarDeclKind::Var
            && let BindingTarget::Ident(Ident { name, .. }) = target
            && let Some(b) = self.lookup(name)
        {
            self.write_var(b, value_reg);
            return Ok(());
        }
        self.bind_pattern(target, value_reg)?;
        // A `const` head's per-iteration binding is immutable, so a body that
        // assigns to it (`for (const x of …) { x++ }`) must reach the
        // tree-walker's TypeError rather than compile to a plain store.
        if *kind == crate::ast::VarDeclKind::Const {
            self.mark_pattern_const(target);
        }
        Ok(())
    }

    /// Marks every name a `const` binding pattern introduced immutable. The
    /// pattern walker is shared by all declaration kinds; the kind is known only
    /// at its call sites.
    fn mark_pattern_const(&mut self, target: &BindingTarget) {
        match target {
            BindingTarget::Ident(Ident { name, .. }) => self.mark_const(name),
            BindingTarget::Array(pat) => {
                use crate::ast::ArrayPatternElement;
                for el in &pat.elements {
                    match el {
                        ArrayPatternElement::Hole => {}
                        ArrayPatternElement::Item { target, .. }
                        | ArrayPatternElement::Rest { target, .. } => {
                            self.mark_pattern_const(target);
                        }
                    }
                }
            }
            BindingTarget::Object(pat) => {
                for prop in &pat.properties {
                    self.mark_pattern_const(&prop.value);
                }
                if let Some(rest) = &pat.rest {
                    self.mark_pattern_const(rest);
                }
            }
        }
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
                            self.apply_default_named(v, default.as_ref(), Some(target))?;
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
                    self.apply_default_named(v, prop.default.as_ref(), Some(&prop.value))?;
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
        self.apply_default_named(reg, default, None)
    }

    /// Like [`Self::apply_default`], but applies NamedEvaluation when the default
    /// is an anonymous function/arrow and `target` is a plain identifier
    /// (`[x = () => {}]` ⇒ `x.name === "x"`).
    fn apply_default_named(
        &mut self,
        reg: Reg,
        default: Option<&Expr>,
        target: Option<&BindingTarget>,
    ) -> Result<(), CompileError> {
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
        let d = match target {
            Some(t) => self.expr_named(e, t)?,
            None => self.expr(e)?,
        };
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
                // A class with an `extends` clause that is not a known compiled
                // class must validate the superclass is a constructor/null at
                // definition (a TypeError otherwise) — the tree-walker handles it.
                if let Some(sup) = class.super_class.as_deref() {
                    let known =
                        matches!(sup, Expr::Ident(sid) if self.classes.contains_key(&*sid.name));
                    if !known {
                        return Err(CompileError::Unsupported("extends a non-class binding"));
                    }
                }
                if let Some(cid) = &class.id {
                    self.materialize_class(&cid.name, class)?;
                }
                Ok(None)
            }
            Stmt::Return { argument, .. } => {
                match argument {
                    // `return expr;` — `expr` (and its tail-transparent
                    // sub-positions) is a proper-tail-call candidate in strict code.
                    Some(e) => self.compile_tail_return(e)?,
                    None => {
                        let src = self.constant(NanBox::undefined())?;
                        self.ops.push(Op::Return { src });
                    }
                }
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
                // Tail-position tracking (PTC): a call in the `try` Block is never
                // in tail position (a `catch`/`finally` may run after it), so clear
                // `tail_ok` while compiling it; the `catch`/`finally` restore it
                // (a `catch` only when no `finally` follows). Saved and restored so
                // the enclosing context is unaffected.
                let saved_tail = self.tail_ok;
                // The register the thrown value lands in (and the catch binding,
                // if any, names it).
                let catch_reg = self.alloc();
                let push = self.ops.len();
                self.ops.push(Op::PushHandler {
                    target: 0,
                    reg: catch_reg,
                });
                self.tail_ok = false;
                self.block_stmts(block)?;
                self.ops.push(Op::PopHandler);
                // Normal completion: run `finally` (in tail position), then jump
                // past the handler.
                if let Some(fin) = finalizer {
                    self.tail_ok = saved_tail;
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
                    // A `catch` body is in tail position only when no `finally`
                    // follows (otherwise the `finally` runs after it).
                    self.tail_ok = saved_tail && finalizer.is_none();
                    for s in &catch.body {
                        self.stmt(s)?;
                    }
                    self.scopes.pop();
                    if let Some(fin) = finalizer {
                        self.tail_ok = saved_tail;
                        self.block_stmts(fin)?;
                    }
                } else {
                    // `try { } finally { }`: run `finally` (in tail position), then
                    // re-raise.
                    if let Some(fin) = finalizer {
                        self.tail_ok = saved_tail;
                        self.block_stmts(fin)?;
                    }
                    self.ops.push(Op::Throw { src: catch_reg });
                }
                self.tail_ok = saved_tail;
                self.patch(jend);
                Ok(None)
            }
            Stmt::Expr { expression, .. } => Ok(Some(self.expr(expression)?)),
            Stmt::Var(decl) => {
                // `using` / `await using` declarations need scope-exit disposal
                // semantics the bytecode VM does not model; bail to the
                // tree-walker (`nbexec`), which implements them with full
                // explicit-resource-management semantics.
                if matches!(
                    decl.kind,
                    crate::ast::VarDeclKind::Using | crate::ast::VarDeclKind::AwaitUsing
                ) {
                    return Err(CompileError::Unsupported("using declaration"));
                }
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
                    // A bare `var x;` (no initializer) that re-declares a name
                    // already bound in this scope is a no-op: it must not reset
                    // the binding's current value (`var x = 5; var x;` leaves `x`
                    // as 5, and `function f(a){ var a; }` keeps the parameter).
                    // Re-`declare`ing would otherwise allocate a fresh register
                    // shadowing the existing one with `undefined`.
                    if d.init.is_none()
                        && matches!(decl.kind, crate::ast::VarDeclKind::Var)
                        && let BindingTarget::Ident(id) = &d.target
                        && self
                            .scopes
                            .last()
                            .is_some_and(|s| s.contains_key(&*id.name))
                    {
                        continue;
                    }
                    let value = match &d.init {
                        Some(e) => self.expr_named(e, &d.target)?,
                        None => self.constant(NanBox::undefined())?,
                    };
                    self.bind_pattern(&d.target, value)?;
                    if matches!(decl.kind, crate::ast::VarDeclKind::Const) {
                        self.mark_pattern_const(&d.target);
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
            // `for (const x of iterable)` — materialize the iterable's values into
            // an array (the built-in iteration of an array / typed array / string /
            // Map / Set; a user iterable or generator faults at `IterValues` and the
            // program re-runs on the tree-walker), then index it by a hidden counter.
            Stmt::ForOf {
                left,
                right,
                body,
                is_await,
                ..
            } => {
                use crate::ast::ForLeft;
                // `for await (…)` is a coroutine suspension point (each value is
                // awaited). The bytecode VM has no await machinery, so route any
                // program containing one to the reference tree-walker, which drives
                // it through the lazy async coroutine engine.
                if *is_await {
                    return Err(CompileError::Unsupported("for await"));
                }
                let ForLeft::Decl { kind, target, .. } = left else {
                    return Err(CompileError::Unsupported("for-of binding"));
                };
                // A `for (using x of …)` / `for (await using x of …)` head needs
                // per-iteration explicit-resource-management disposal; bail to the
                // tree-walker (`nbexec`).
                if matches!(
                    kind,
                    crate::ast::VarDeclKind::Using | crate::ast::VarDeclKind::AwaitUsing
                ) {
                    return Err(CompileError::Unsupported("using in for-of head"));
                }
                self.scopes.push(alloc::collections::BTreeMap::new());
                let src = self.expr(right)?;
                let arr = self.alloc();
                self.ops.push(Op::IterValues { dst: arr, src });
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
                self.bind_for_decl(kind, target, cur)?;
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
                let ForLeft::Decl { kind, target, .. } = left else {
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
                self.bind_for_decl(kind, target, cur)?;
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

    /// Compiles `return e;` with `e` in tail position. Threads the tail flag
    /// through the tail-transparent expression forms — `?:` (both arms),
    /// `&&`/`||`/`??` (the right operand), and the comma sequence's last operand —
    /// and emits a frame-reusing tail-call opcode (`Op::TailCall` /
    /// `Op::TailCallValue`) when the tail expression is a plain function/closure
    /// call. Every other expression falls back to `Op::Return` of its value. When
    /// `tail_ok` is false (sloppy code, an async body, or inside a `try` Block)
    /// it degenerates to a plain `Op::Return`. Every path it emits is *terminal*
    /// (ends in a return or tail call), so branch arms need no join jump.
    fn compile_tail_return(&mut self, e: &Expr) -> Result<(), CompileError> {
        if !self.tail_ok {
            let src = self.expr(e)?;
            self.ops.push(Op::Return { src });
            return Ok(());
        }
        match e {
            Expr::Call {
                callee,
                arguments,
                optional,
                ..
            } if !*optional => {
                if !self.try_emit_tail_call(callee, arguments)? {
                    let src = self.expr(e)?;
                    self.ops.push(Op::Return { src });
                }
                Ok(())
            }
            // A tagged template's tag is invoked in tail position.
            Expr::TaggedTemplate { tag, quasi, .. } => {
                // A lone surrogate in a quasi can't round-trip the constant pool;
                // let the ordinary (surrogate-correct) path handle it.
                let surrogate = quasi.quasis.iter().any(|q| {
                    q.cooked
                        .as_deref()
                        .is_some_and(|b| crate::wtf8::as_str(b).is_none())
                });
                if surrogate {
                    let src = self.expr(e)?;
                    self.ops.push(Op::Return { src });
                    return Ok(());
                }
                let strings = self.alloc();
                self.ops.push(Op::NewArray {
                    dst: strings,
                    len: 0,
                });
                for q in &quasi.quasis {
                    let s = match q.cooked.as_deref() {
                        Some(c) => self.constant_str(&crate::wtf8::to_string_lossy(c)),
                        None => self.constant(NanBox::undefined())?,
                    };
                    self.ops.push(Op::ArrayPush {
                        arr: strings,
                        src: s,
                    });
                }
                let mut args = alloc::vec![strings];
                for ex in &quasi.expressions {
                    args.push(self.expr(ex)?);
                }
                if let Expr::Ident(id) = &**tag
                    && self.lookup(&id.name).is_none()
                    && let Some(&func) = self.fn_ids.get(&*id.name)
                {
                    self.ops.push(Op::TailCall { func, args });
                } else {
                    let callee = self.expr(tag)?;
                    self.ops.push(Op::TailCallValue { callee, args });
                }
                Ok(())
            }
            // `return c ? a : b;` — both arms inherit tail position.
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                let cond = self.expr(test)?;
                let jf = self.emit_jump_if_false(cond);
                self.compile_tail_return(consequent)?; // terminal
                self.patch(jf);
                self.compile_tail_return(alternate)?; // terminal
                Ok(())
            }
            // `return l && r;` / `l || r;` / `l ?? r;` — the *right* operand is in
            // tail position; a short-circuit returns the left value directly.
            Expr::Logical {
                op, left, right, ..
            } => {
                use crate::ast::LogicalOp;
                let l = self.expr(left)?;
                match op {
                    LogicalOp::And => {
                        let jf = self.emit_jump_if_false(l); // falsy → return l
                        self.compile_tail_return(right)?; // truthy → tail r
                        self.patch(jf);
                        self.ops.push(Op::Return { src: l });
                    }
                    LogicalOp::Or => {
                        let jf = self.emit_jump_if_false(l); // falsy → tail r
                        self.ops.push(Op::Return { src: l }); // truthy → return l
                        self.patch(jf);
                        self.compile_tail_return(right)?;
                    }
                    LogicalOp::Nullish => {
                        let nn = self.emit_not_nullish(l)?;
                        let jf = self.emit_jump_if_false(nn); // nullish → tail r
                        self.ops.push(Op::Return { src: l }); // else → return l
                        self.patch(jf);
                        self.compile_tail_return(right)?;
                    }
                }
                Ok(())
            }
            // `return a, b, c;` — the last operand is in tail position.
            Expr::Sequence { expressions, .. } => {
                match expressions.split_last() {
                    Some((last, rest)) => {
                        for ex in rest {
                            self.expr(ex)?;
                        }
                        self.compile_tail_return(last)?;
                    }
                    None => {
                        let src = self.constant(NanBox::undefined())?;
                        self.ops.push(Op::Return { src });
                    }
                }
                Ok(())
            }
            _ => {
                let src = self.expr(e)?;
                self.ops.push(Op::Return { src });
                Ok(())
            }
        }
    }

    /// Emits a frame-reusing tail-call opcode for `return f(args)` when `f` is a
    /// plain function (static dispatch → [`Op::TailCall`]) or a function *value*
    /// (→ [`Op::TailCallValue`]). Returns `Ok(false)` — leaving nothing emitted —
    /// for a call shape that is not a frame-reusable tail call (spread args,
    /// `super(...)`, a built-in, or a method / `super.` / `Class.static` call),
    /// so the caller can fall back to an ordinary call plus `Op::Return`.
    fn try_emit_tail_call(
        &mut self,
        callee: &Expr,
        arguments: &[crate::ast::Argument],
    ) -> Result<bool, CompileError> {
        if arguments
            .iter()
            .any(|a| !matches!(a, crate::ast::Argument::Item(_)))
        {
            return Ok(false);
        }
        if matches!(callee, Expr::Super(_)) {
            return Ok(false);
        }
        if native_call(callee)
            .or_else(|| native_global(callee))
            .is_some()
        {
            return Ok(false);
        }
        // Method / `super.method` / `Class.static` calls keep the ordinary
        // receiver-binding path (still correct, just not PTC).
        if matches!(callee, Expr::Member { .. }) {
            return Ok(false);
        }
        // Evaluate the arguments (matching the ordinary call's arg-first order).
        let mut args = Vec::with_capacity(arguments.len());
        for a in arguments {
            let crate::ast::Argument::Item(e) = a else {
                return Ok(false);
            };
            args.push(self.expr(e)?);
        }
        // A direct call to a hoisted function by name reuses the frame in place;
        // any other callee is an indirect call through a function value.
        if let Expr::Ident(id) = callee
            && self.lookup(&id.name).is_none()
            && let Some(&func) = self.fn_ids.get(&*id.name)
        {
            self.ops.push(Op::TailCall { func, args });
        } else {
            let callee_reg = self.expr(callee)?;
            self.ops.push(Op::TailCallValue {
                callee: callee_reg,
                args,
            });
        }
        Ok(true)
    }

    fn expr(&mut self, expr: &Expr) -> Result<Reg, CompileError> {
        match expr {
            Expr::Number { value, .. } => self.constant(NanBox::number(*value)),
            Expr::Bool { value, .. } => self.constant(NanBox::boolean(*value)),
            Expr::Null(_) => self.constant(NanBox::null()),
            Expr::Str { value, .. } => {
                // The bytecode `NewString` op is `String`-typed; a literal bearing
                // a lone surrogate (rare) can't round-trip through it, so defer to
                // the (surrogate-correct) tree-walker rather than lose data.
                let value = crate::wtf8::as_str(value).ok_or(CompileError::Unsupported(
                    "lone surrogate in string literal",
                ))?;
                let r = self.alloc();
                self.ops.push(Op::NewString {
                    dst: r,
                    value: String::from(value),
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
                            // A real elision: store the internal hole sentinel so
                            // `[[Get]]` falls through to the prototype, `in`/
                            // `Object.keys`/the iteration built-ins skip it, etc.
                            let u = self.constant(NanBox::hole())?;
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
                // A data property whose key duplicates an accessor defined earlier
                // in the same literal (`{get x(){}, x: 1}`) is a CreateDataProperty
                // that must *replace* the getter/setter — `Op::SetProp`/`Op::SetKey`
                // are `[[Set]]`s and would run the setter instead. That is rare
                // enough to hand the whole literal to the tree-walker (which
                // clears the accessor first). An accessor with a computed key
                // already deopts via `static_key` below, so only static accessor
                // keys can be present here; a computed *data* key is treated
                // conservatively.
                let accessor_keys: Vec<&PropertyKey> = members
                    .iter()
                    .filter_map(|m| match m {
                        ObjectMember::Accessor { key, .. } => Some(key),
                        _ => None,
                    })
                    .collect();
                if !accessor_keys.is_empty() {
                    for m in members {
                        if let ObjectMember::Property { key, .. } = m {
                            let collides = match key {
                                PropertyKey::Computed(_) => true,
                                _ => accessor_keys.iter().any(|a| {
                                    matches!((static_key(a), static_key(key)), (Ok(x), Ok(y)) if x == y)
                                }),
                            };
                            if collides {
                                return Err(CompileError::Unsupported(
                                    "object literal: data property duplicating an accessor key",
                                ));
                            }
                        }
                    }
                }
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
                            // NamedEvaluation: `x &&= function(){}` / `x ??= () => {}`
                            // names the anonymous RHS after the LHS identifier.
                            let bt = crate::ast::BindingTarget::Ident((*id).clone());
                            let v = self.expr_named(value, &bt)?;
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
                // `++`/`--` on a `const` binding is a TypeError; route the program
                // to the tree-walker, which enforces it at the right point.
                if b.konst {
                    return Err(CompileError::Unsupported("update of const"));
                }
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
                    source: pattern.to_vec(),
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
                // A surrogate-bearing cooked/raw quasi can't round-trip through the
                // `String`-typed constant pool; defer to the (correct) tree-walker.
                if quasi.quasis.iter().any(|q| {
                    q.cooked
                        .as_deref()
                        .is_some_and(|b| crate::wtf8::as_str(b).is_none())
                }) {
                    return Err(CompileError::Unsupported(
                        "lone surrogate in tagged template",
                    ));
                }
                let strings = self.alloc();
                self.ops.push(Op::NewArray {
                    dst: strings,
                    len: 0,
                });
                for q in &quasi.quasis {
                    // An invalid escape yields no cooked value (`undefined`); `.raw`
                    // still preserves it (ES2018 tagged-template revision).
                    let s = match q.cooked.as_deref() {
                        Some(c) => self.constant_str(&crate::wtf8::to_string_lossy(c)),
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
                        // RegExp source/flags are valid UTF-8 in practice (regex
                        // surrogate handling is a separate milestone); a
                        // surrogate-bearing arg falls back to the runtime path.
                        Some(crate::ast::Argument::Item(Expr::Str { value, .. })) => {
                            crate::wtf8::as_str(value).map(String::from)
                        }
                        None => Some(String::new()),
                        _ => None,
                    };
                    if let (Some(source), Some(flags)) =
                        (lit(arguments.first()), lit(arguments.get(1)))
                    {
                        let dst = self.alloc();
                        self.ops.push(Op::NewRegExp {
                            dst,
                            source: source.into_bytes(),
                            flags,
                        });
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
                // Link the instance to the class's `.prototype` (built in
                // `materialize_class`), so public methods/accessors are inherited
                // (`instance.m === C.prototype.m`, no own `m`) rather than copied.
                if let Some(cb) = self.lookup(&id.name) {
                    let cval = self.read_var(cb);
                    let proto = self.alloc();
                    self.ops.push(Op::GetProp {
                        dst: proto,
                        obj: cval,
                        key: String::from("prototype"),
                    });
                    self.ops.push(Op::SetProto {
                        obj: instance,
                        proto,
                    });
                }
                // Walk the `extends` chain root→derived and install each class's
                // *private* methods/accessors on the instance (they brand the
                // instance directly; public ones live on the prototype).
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
                        if !mname.starts_with('#') {
                            continue;
                        }
                        let m = self.alloc();
                        self.ops.push(Op::LoadFunc { dst: m, func: *mid });
                        self.ops.push(Op::SetHidden {
                            obj: instance,
                            key: mname.clone(),
                            src: m,
                        });
                    }
                    // Install private getter/setter accessors.
                    for (aname, getter_id, setter_id) in &accessors {
                        if !aname.starts_with('#') {
                            continue;
                        }
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
                // A surrogate-bearing quasi can't round-trip through the
                // `String`-typed bytecode; defer to the (correct) tree-walker.
                if t.quasis.iter().any(|q| {
                    q.cooked
                        .as_deref()
                        .is_some_and(|b| crate::wtf8::as_str(b).is_none())
                }) {
                    return Err(CompileError::Unsupported(
                        "lone surrogate in template literal",
                    ));
                }
                let cooked = |q: &crate::ast::TemplateElement| -> String {
                    q.cooked
                        .as_deref()
                        .map(crate::wtf8::to_string_lossy)
                        .unwrap_or_default()
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
        // A *named function expression* binds its own name inside its body (to the
        // function itself). If that name is referenced, thread it as a trailing
        // "self" capture: a cell we create here and backfill with the finished
        // closure, so a recursive call (`return f(n-1)`) reaches the function
        // *with its own captures* — which also makes it eligible for a proper
        // tail call. The name is invisible outside the body.
        let self_name: Option<&str> = if !name.is_empty() && free.contains(name) {
            Some(name)
        } else {
            None
        };
        let mut captures: Vec<String> = free
            .into_iter()
            .filter(|n| self.lookup(n).is_some() && Some(n.as_str()) != self_name)
            .collect();
        if let Some(sn) = self_name {
            // Bound last → its capture register is the self-cell below.
            captures.push(String::from(sn));
        }
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
            self.strict,
        )?;
        let mut proto = proto;
        proto.name = alloc::string::String::from(name);
        self.protos.borrow_mut()[id as usize] = proto;
        // Capture the cell registers for each free variable (in the same order the
        // callee binds them). The self-name (if any) gets a fresh cell here,
        // backfilled with the closure after it is built.
        let mut capture_regs: Vec<Reg> = Vec::with_capacity(captures.len());
        let mut self_cell: Option<Reg> = None;
        for n in &captures {
            if Some(n.as_str()) == self_name {
                let cell = self.alloc();
                self.ops.push(Op::NewArray { dst: cell, len: 1 });
                self_cell = Some(cell);
                capture_regs.push(cell);
            } else {
                capture_regs.push(self.lookup(n).expect("captured binding").reg);
            }
        }
        let dst = self.alloc();
        self.ops.push(Op::MakeClosure {
            dst,
            func: id,
            captures: capture_regs,
        });
        // Backfill the self-cell so the body's own-name binding reads this closure.
        if let Some(cell) = self_cell {
            let idx = self.constant(NanBox::number(0.0))?;
            self.ops.push(Op::SetElem {
                arr: cell,
                index: idx,
                src: dst,
            });
        }
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
        // The class's `name`: its own declared identifier (`class D {}` → `"D"`),
        // else the binding it is being assigned to (`var C = class {}` → `"C"`).
        let class_name = class
            .id
            .as_ref()
            .map_or_else(|| String::from(name), |i| String::from(&*i.name));
        let name_reg = self.constant_str(&class_name);
        self.ops.push(Op::SetHidden {
            obj: cobj,
            key: String::from("name"),
            src: name_reg,
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
        // Build a real `.prototype` object carrying the class's instance
        // methods/accessors over the whole `extends` chain (base-first so a
        // derived method overrides an inherited one) and a `constructor`
        // back-link, so `C.prototype`, `C.prototype.m`, accessor reads, and
        // `instance.constructor` reflection all work on the bytecode path.
        let proto = self.alloc();
        self.ops.push(Op::NewObject { dst: proto });
        // `proto.constructor === C` (non-enumerable), installed *before* the
        // instance methods so `[[OwnPropertyKeys]]` order is `constructor,
        // …methods` (per MakeConstructor; matches the tree-walker).
        self.ops.push(Op::SetHidden {
            obj: proto,
            key: String::from("constructor"),
            src: cobj,
        });
        let mut chain: Vec<String> = Vec::new();
        let mut cur = Some(String::from(name));
        while let Some(n) = cur {
            cur = self.classes.get(&n).and_then(|c| c.super_name.clone());
            chain.push(n);
        }
        for cname in chain.iter().rev() {
            let Some(cls) = self.classes.get(cname) else {
                continue;
            };
            let methods = cls.methods.clone();
            let accessors = cls.accessors.clone();
            for (mname, mid) in &methods {
                let m = self.alloc();
                self.ops.push(Op::LoadFunc { dst: m, func: *mid });
                self.ops.push(Op::SetProp {
                    obj: proto,
                    key: mname.clone(),
                    src: m,
                });
            }
            for (aname, getter_id, setter_id) in &accessors {
                let g = match getter_id {
                    Some(fid) => {
                        let r = self.alloc();
                        self.ops.push(Op::LoadFunc { dst: r, func: *fid });
                        r
                    }
                    None => self.constant(NanBox::undefined())?,
                };
                let s = match setter_id {
                    Some(fid) => {
                        let r = self.alloc();
                        self.ops.push(Op::LoadFunc { dst: r, func: *fid });
                        r
                    }
                    None => self.constant(NanBox::undefined())?,
                };
                self.ops.push(Op::DefineAccessor {
                    obj: proto,
                    key: aname.clone(),
                    getter: g,
                    setter: s,
                });
            }
        }
        // `C.prototype === proto` (non-enumerable). The `constructor` back-link on
        // `proto` was installed above, before the methods.
        self.ops.push(Op::SetHidden {
            obj: cobj,
            key: String::from("prototype"),
            src: proto,
        });
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

    /// `KNOWN_GLOBALS` mirrors the set of globals the interpreter installs, and a
    /// *missing* entry is wrong twice over: the value path emits a spurious
    /// `ReferenceError`, and `typeof` is answered at compile time as
    /// `"undefined"` for a global that plainly exists. (That is exactly how
    /// `Temporal` — installed unconditionally since the Temporal work — came to
    /// report `typeof Temporal === "undefined"` on this path while
    /// `Temporal.PlainDate.from(…)` worked.)
    ///
    /// Diff the list against the live global object so it cannot drift again.
    #[test]
    fn globals_match_installed_set() {
        let (_, installed) =
            crate::nbexec::eval_source("Object.getOwnPropertyNames(globalThis).join(' ')")
                .expect("enumerate globals");
        let missing: alloc::vec::Vec<&str> = installed
            .split(' ')
            // The `$262_*` harness hooks are not part of the language surface,
            // and the value keywords are handled ahead of the list.
            .filter(|n| {
                !n.starts_with("$262_") && !matches!(*n, "undefined" | "NaN" | "Infinity" | "")
            })
            .filter(|n| !KNOWN_GLOBALS.contains(n))
            .collect();
        assert!(
            missing.is_empty(),
            "globals installed but absent from KNOWN_GLOBALS: {missing:?}"
        );
        // …and nothing in the list that is not actually installed, which would
        // send a genuinely-undefined name down the slow tree-walker path.
        let stale: alloc::vec::Vec<&&str> = KNOWN_GLOBALS
            .iter()
            .filter(|n| !installed.split(' ').any(|g| g == **n))
            .collect();
        assert!(
            stale.is_empty(),
            "KNOWN_GLOBALS entries not installed: {stale:?}"
        );
    }

    /// `typeof` on a global must report the global, not `"undefined"` — and on a
    /// genuinely undefined name it must still be `"undefined"` rather than a
    /// `ReferenceError`.
    #[test]
    fn typeof_reports_globals_and_absent_names() {
        for (src, want) in [
            ("typeof Temporal", "object"),
            ("typeof Atomics", "object"),
            ("typeof Math", "object"),
            ("typeof eval", "function"),
            ("typeof Float16Array", "function"),
            ("typeof FinalizationRegistry", "function"),
            ("typeof no_such_global_anywhere", "undefined"),
        ] {
            let (_, got) = crate::nbexec::eval_source(src).expect(src);
            assert_eq!(got, want, "{src}");
        }
    }
    use super::*;

    /// Compiles `src` to bytecode and runs it over a fresh realm, returning the
    /// completion value as a display string.
    fn bc(src: &str) -> String {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut realm = Realm::new();
        let value = compile_and_run(&mut realm, &program).expect("compile+run");
        realm.to_display_string(value)
    }

    /// C1: in the bytecode VM, a dense-array element write to a valid array index
    /// (`< 2^32-1`) past the `max_array_len` cap is served *sparsely* (the VM
    /// faults to the tree-walker, which stores it as an aux named property + a
    /// logical-length bump) — a plain `arr[i] = v` never throws. A `length` set to
    /// a valid uint32 above the cap is likewise a sparse length; only a length
    /// above the uint32 ceiling is invalid. Exercised through the production
    /// `execute` entry (VM with the tree-walker fallback).
    #[test]
    fn vm_oversized_array_growth_throws_range_error() {
        let v = |src: &str| -> String {
            match execute(src) {
                Ok((_, value)) => value,
                Err(e) => alloc::format!("ERR:{e}"),
            }
        };
        // A valid-index element write past the cap stores sparsely, growing
        // `length` to `i + 1`; no throw.
        assert_eq!(
            v("var a=[1]; a[1e9]=1; String(a.length)+','+a[1e9]"),
            "1000000001,1"
        );
        // A valid uint32 `length` set is a sparse length (no throw); reported as-is.
        assert_eq!(v("var a=[1]; a.length=1e9; String(a.length)"), "1000000000");
        assert_eq!(
            v("var a=[1]; a['length']=1e9; String(a.length)"),
            "1000000000"
        );
        // A length above the uint32 ceiling (2^32) is invalid → RangeError.
        assert_eq!(
            v("var a=[1]; try{a.length=4294967296;'no'}catch(e){e.constructor.name}"),
            "RangeError"
        );
        // Within-cap writes / length sets are unchanged; `arr.length = n` now also
        // truncates correctly on the VM path.
        assert_eq!(
            v("var a=[1]; a[5]=9; JSON.stringify(a)"),
            "[1,null,null,null,null,9]"
        );
        assert_eq!(
            v("var a=[1,2,3,4,5]; a.length=2; JSON.stringify(a)+':'+a.length"),
            "[1,2]:2"
        );
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

    // --- Inline-cache (H1) wiring: `GetProp`/`SetProp` over a per-frame
    // monomorphic shape→slot cache. Every case runs through `bc()`, which forces
    // the pure bytecode VM (no tree-walker fallback), so the cache path is the one
    // under test. ---

    /// A hot `obj.x` read loop: the same shape every iteration, so after the cold
    /// miss every later read is a cache hit. The sum must still be exact.
    #[test]
    fn ic_hot_read_loop_is_correct() {
        // o.x == 7 for all 100 iterations → 700. Reading through the cache must
        // return the live slot value, not a stale or wrong one.
        assert_eq!(
            bc("let o = { x: 7, y: 1 }; \
                let t = 0; \
                for (let i = 0; i < 100; i = i + 1) { t = t + o.x; } \
                t"),
            "700"
        );
        // A write through the same site then a read: the in-place `SetProp` fast
        // path updates the slot, and the subsequent reads see the new value.
        assert_eq!(
            bc("let o = { x: 0 }; \
                for (let i = 0; i < 50; i = i + 1) { o.x = o.x + 2; } \
                o.x"),
            "100"
        );
    }

    /// A polymorphic read site — two different shapes alternating through one
    /// `GetProp` instruction. The monomorphic cache must re-resolve on the shape
    /// it does not currently hold (a miss), never returning the other object's
    /// slot.
    #[test]
    fn ic_polymorphic_site_stays_correct() {
        // `a` is {p}; `b` is {q, p} — `p` lives in slot 0 of `a` but slot 1 of
        // `b`. Reading `.p` from each in turn must yield each one's own value, so
        // a cache armed on `a`'s shape must miss for `b` and vice versa.
        assert_eq!(
            bc("let a = { p: 10 }; \
                let b = { q: 99, p: 20 }; \
                let arr = [a, b]; \
                let t = 0; \
                for (let i = 0; i < 20; i = i + 1) { t = t + arr[i % 2].p; } \
                t"),
            // 10 reads of a.p (=10) + 10 reads of b.p (=20) = 100 + 200 = 300.
            "300"
        );
    }

    /// Adding a property (a shape *transition*) after the site has warmed, then
    /// reading it, must see the new property — the transition produces a new shape
    /// pointer that misses the cache, so no stale slot is returned.
    #[test]
    fn ic_shape_transition_then_read_is_correct() {
        assert_eq!(
            bc("let o = { a: 1 }; \
                let r = o.a; \
                o.b = 2; \
                r = r + o.b; \
                o.c = 3; \
                r + o.c"),
            "6"
        );
        // Warm a read site on the original shape, then transition and re-read the
        // *same* property name through the warmed site: still correct.
        assert_eq!(
            bc("let o = { a: 5 }; \
                let s = 0; \
                for (let i = 0; i < 3; i = i + 1) { s = s + o.a; } \
                o.z = 1; \
                for (let i = 0; i < 3; i = i + 1) { s = s + o.a; } \
                s"),
            "30"
        );
    }

    /// Deleting a property then reading it returns `undefined` — the delete
    /// rebuilds the object on a fresh shape, so the warmed cache misses and the
    /// slow path reports the property absent.
    #[test]
    fn ic_delete_then_read_is_undefined() {
        assert_eq!(
            bc("let o = { a: 1, b: 2 }; \
                let r = o.a; \
                delete o.a; \
                r + ':' + o.a"),
            "1:undefined"
        );
    }

    /// An accessor (getter/setter) must still invoke the function, never the IC
    /// data fast path — accessors live outside the slot layout and are resolved
    /// before the cache is consulted.
    #[test]
    fn ic_accessors_still_invoke_getter_setter() {
        // A getter computes a value; reading the property repeatedly must call it
        // each time (here it reads a backing field), not a cached slot.
        assert_eq!(
            bc("let o = { _v: 3, get x() { return this._v * 10; } }; \
                let t = 0; \
                for (let i = 0; i < 4; i = i + 1) { t = t + o.x; } \
                t"),
            "120"
        );
        // A setter routes writes through the function; the IC's `cached_set` must
        // not short-circuit it.
        assert_eq!(
            bc("let o = { _v: 0, set x(n) { this._v = n + 1; } }; \
                o.x = 5; \
                o.x = 9; \
                o._v"),
            "10"
        );
    }

    /// A dictionary-mode object (more own properties than the realm's
    /// `object_dictionary_threshold`) reads and writes correctly through the
    /// non-cached path — its empty sentinel shape never produces a cache hit.
    #[test]
    fn ic_dictionary_object_bypasses_cache() {
        // Build an object with > 128 own properties (the default dict threshold),
        // converting it to dictionary mode, then read/write a few keys in a loop.
        let src = "let o = {}; \
             for (let i = 0; i < 200; i = i + 1) { o['k' + i] = i; } \
             let t = 0; \
             for (let i = 0; i < 50; i = i + 1) { t = t + o.k10; } \
             o.k10 = 1000; \
             t + ':' + o.k10 + ':' + o.k199";
        // 50 reads of k10 (=10) = 500; then k10 set to 1000; k199 == 199.
        assert_eq!(bc(src), "500:1000:199");
    }

    /// A prototype-inherited member still resolves: the IC only fast-paths an
    /// *own* data slot on the receiver, so an inherited member misses the cache
    /// and falls to the slow path (which walks the prototype chain). Uses a class
    /// — whose methods live on the prototype — so it compiles to pure bytecode
    /// (no `Object.create` global, which is tree-walker-only).
    #[test]
    fn ic_prototype_inherited_property_resolves() {
        assert_eq!(
            bc(
                "class P { constructor() { this.own = 1; } shared() { return 42; } } \
                let o = new P(); \
                let t = 0; \
                for (let i = 0; i < 10; i = i + 1) { t = t + o.shared() + o.own; } \
                t"
            ),
            // (42 + 1) * 10 = 430. `o.own` is an own data slot (IC fast path),
            // `o.shared` is inherited from the prototype (IC miss → slow path).
            "430"
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
        // `obj == primitive` needs `ToPrimitive(obj)` with the *default* hint — a
        // user-overridable `@@toPrimitive`/`valueOf`/`toString` that can throw — so
        // the VM faults those to the tree-walker rather than using the intrinsic
        // display form. Exercised through the production `execute` entry (VM with
        // the tree-walker fallback), which is where the semantics are observable.
        // Object-vs-object stays on the VM's identity fast path.
        for (src, want) in [
            ("String([] == false)", "true"),
            ("String([] == 0)", "true"),
            ("String({} == 0)", "false"),
            ("String([1,2] == '1,2')", "true"),
            ("String({valueOf(){return 7}} == 7)", "true"),
        ] {
            let (_, completion) = execute(src).expect("ok");
            assert_eq!(completion, want, "{src}");
        }
        assert_eq!(bc("String({} == {})"), "false");
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

    #[cfg(feature = "regex")]
    #[test]
    fn bytecode_regex_code_unit_indices_on_astral() {
        // The bytecode regex path reports **code-unit** indices and matches in the
        // UTF-16 unit space for an astral (surrogate-free) subject.
        // u-flag: `.` matches the whole astral char, span length 2 at index 0.
        assert_eq!(bc(r#"/./u.exec("😀")[0].length"#), "2");
        assert_eq!(bc(r#"/./u.exec("😀").index"#), "0");
        // Non-u: `.` matches one code unit → two matches over an astral char.
        assert_eq!(bc(r#""😀".match(/./g).length"#), "2");
        assert_eq!(bc(r#"/./.exec("😀")[0].length"#), "1");
        // `.index` / `lastIndex` / search are code-unit indices.
        assert_eq!(bc(r#""a😀b".search(/b/)"#), "3");
        assert_eq!(bc(r#"const r=/y/g; r.exec("😀y"); r.lastIndex"#), "3");
        // replace splices astral matches back losslessly (string template).
        assert_eq!(bc(r#""a😀b".replace(/😀/u, "X")"#), "aXb");
        assert_eq!(bc(r#""x😀y".replace(/(😀)/, "[$1]")"#), "x[😀]y");
        // split keeps surrounding text whole around an astral separator.
        assert_eq!(bc(r#""a😀b".split(/😀/).join("|")"#), "a|b");
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

    /// NBVM-1: `String.prototype.repeat` with an attacker-controlled count must
    /// throw a catchable `RangeError` rather than alloc-aborting the process.
    #[test]
    fn repeat_allocation_bomb_throws_range_error() {
        // A huge product (or non-finite/negative count) is an unrepresentable
        // string length — caught here to confirm it throws, not aborts.
        assert_eq!(
            bc("let r; try { 'x'.repeat(1e12); r = 'no throw'; } catch (e) { r = e.name; } r"),
            "RangeError"
        );
        assert_eq!(
            bc("let r; try { 'x'.repeat(-1); r = 'no throw'; } catch (e) { r = e.name; } r"),
            "RangeError"
        );
        assert_eq!(
            bc("let r; try { 'x'.repeat(Infinity); r = 'no throw'; } catch (e) { r = e.name; } r"),
            "RangeError"
        );
        // Legitimate small repeats still work.
        assert_eq!(bc("'ab'.repeat(3)"), "ababab");
        assert_eq!(bc("'x'.repeat(0)"), "");
    }

    /// NBVM-2: a program whose register count exceeds the `Reg` (`u16`) width must
    /// surface as a clean `CompileError`, not an integer-overflow panic in
    /// `Compiler::alloc`.
    #[test]
    fn excessive_register_count_is_compile_error_not_panic() {
        // 70_000 distinct local bindings exhaust the u16 register space.
        let mut src = String::new();
        for i in 0..70_000 {
            src.push_str(&alloc::format!("let v{i}=0;"));
        }
        let program = crate::parser::Parser::parse_program(&src).expect("parse");
        let mut realm = Realm::new();
        // Must return Err (routed to the tree-walker by `execute`), never panic.
        assert!(compile_and_run(&mut realm, &program).is_err());
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

    // --- H2: property-access fast path must preserve exact semantics ---

    #[test]
    fn getter_takes_precedence_over_data_read() {
        // A getter accessor is consulted instead of any data slot: the GetProp
        // fast path must still reach the `accessor` branch before `get_property`.
        // (Accessor closures that capture outer scope, and setter traps, are a
        // separate pre-existing limitation of this minimal `compile_and_run`
        // path — their full semantics are covered by the conformance/test262
        // suites that exercise the complete object model.)
        assert_eq!(bc("let o = { get x() { return 42; } }; o.x"), "42");
        // A getter co-existing with other plain data properties still wins for its
        // own key and leaves the data reads untouched.
        assert_eq!(
            bc("let o = { a: 1, get x() { return 9; }, b: 2 }; o.a + o.x + o.b"),
            "12"
        );
    }

    #[test]
    fn plain_data_property_read_and_write_unchanged() {
        // The common case the fast path targets: a plain data property round-trips
        // unchanged, and a missing property reads `undefined`.
        assert_eq!(bc("let o = { a: 1 }; o.b = 2; o.a + o.b"), "3");
        assert_eq!(bc("let o = { a: 1 }; o.missing"), "undefined");
    }

    #[test]
    fn special_keys_still_resolve_after_fast_path() {
        // `name` on a function, and `__proto__`/regexp members must still take
        // their special paths rather than reading a plain data slot.
        assert_eq!(bc("function foo() {} foo.name"), "foo");
        assert_eq!(bc("let r = /ab+c/gi; r.source"), "ab+c");
        assert_eq!(bc("let r = /ab+c/gi; r.flags"), "gi");
        assert_eq!(bc("let r = /ab+c/gi; r.global"), "true");
        assert_eq!(bc("let r = /x/; r.global"), "false");
    }

    #[test]
    fn regexp_last_index_round_trips() {
        // `lastIndex` is a stateful regexp member on both read and write — its
        // SetProp branch (key-gated `regexp_at`) and the GetProp regexp branch
        // must still fire after the fast-path reorder.
        assert_eq!(bc("let r = /x/g; r.lastIndex = 5; r.lastIndex"), "5");
    }

    // --- M4: read-only array builtins borrow rather than clone ---

    #[test]
    fn array_pure_scans_correct() {
        assert_eq!(bc("[1,2,3].includes(2)"), "true");
        assert_eq!(bc("[1,2,3].includes(9)"), "false");
        assert_eq!(bc("[NaN].includes(NaN)"), "true");
        assert_eq!(bc("[1,2,3,2].indexOf(2)"), "1");
        assert_eq!(bc("[1,2,3].indexOf(9)"), "-1");
        assert_eq!(bc("[1,2,3].join('-')"), "1-2-3");
        assert_eq!(bc("[1,null,3].join(',')"), "1,,3");
        // A self-referential element renders empty (no recursion / no panic).
        assert_eq!(bc("let a = [1]; a.push(a); a.join('|')"), "1|");
    }

    #[test]
    fn foreach_callback_mutating_array_is_safe() {
        // The callback methods snapshot the backing store, so a callback that
        // mutates the array mid-iteration cannot corrupt the iteration or panic.
        // forEach iterates the original three elements even though each call
        // appends a new one.
        assert_eq!(
            bc("let a = [1,2,3]; let sum = 0; \
                a.forEach(function(x) { sum = sum + x; a.push(99); }); \
                sum + ':' + a.length"),
            "6:6"
        );
        // A callback that truncates the array must not read freed memory.
        assert_eq!(
            bc("let a = [1,2,3]; let n = 0; \
                a.forEach(function(x) { n = n + 1; a.pop(); }); n"),
            "3"
        );
    }
}

/// Differential tests for the generic (NanBox) JIT tier — Pass 1 of the JIT
/// completion (`JIT_DESIGN.md`). Each drives a `function f(a,b){ return a+b; }`
/// shaped body through the JIT-forced generic path and asserts it is identical
/// to the interpreter (same value, or same thrown exception with the
/// side-effecting `valueOf` run exactly once).
#[cfg(all(test, feature = "jit", target_os = "linux", target_arch = "x86_64"))]
mod generic_jit_tests {
    use super::*;
    use crate::nanbox::NanBox;

    fn mk_ctx(realm: &mut Realm) -> Ctx<'_> {
        Ctx {
            realm,
            output: String::new(),
            microtasks: alloc::collections::VecDeque::new(),
            tiers: alloc::collections::BTreeMap::new(),
            jit_cache: alloc::collections::BTreeMap::new(),
            jit_pending: None,
            #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
            jit_pending_fault: None,
            jit_funcs: None,
            jit_shadow: alloc::vec::Vec::new(),
            call_depth: 0,
            gc_enabled: false,
            gc_lock: 0,
            top_frame_roots: Vec::new(),
        }
    }

    /// Builds a function table with helpers that mint objects/symbols, plus a
    /// hand-built `f(a,b){ return a+b; }` (appended, so its op stream is exactly
    /// `AddValue; Return` regardless of the compiler). Returns `(funcs, f_id)`.
    fn build() -> (Vec<FnProto>, usize) {
        let src = "
            function makeVal(){ return { c: 0, valueOf: function(){ this.c = this.c + 1; return 42; } }; }
        ";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut funcs = compile_program(&program).expect("compile helpers");
        let f_id = funcs.len();
        funcs.push(FnProto {
            ops: alloc::vec![Op::AddValue { dst: 3, a: 0, b: 1 }, Op::Return { src: 3 },],
            n_regs: 4,
            n_params: 2,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: 2,
            name: alloc::string::String::from("f"),
        });
        (funcs, f_id)
    }

    fn make_val(ctx: &mut Ctx, funcs: &[FnProto]) -> NanBox {
        let id = funcs.iter().position(|p| p.name == "makeVal").unwrap();
        call(ctx, funcs, id, &[]).expect("makeVal")
    }

    /// number + number — exercises the inline both-numbers fast path.
    #[test]
    fn generic_add_number_number() {
        let (funcs, f_id) = build();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers())
            .expect("f is generic-eligible");
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let args = [NanBox::number(2.0), NanBox::number(3.0)];
        let interp = call(&mut ctx, &funcs, f_id, &args).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &args)
            .unwrap()
            .unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "5");
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// string + string — non-number operands take the helper slow path.
    #[test]
    fn generic_add_string_string() {
        let (funcs, f_id) = build();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let s1 = NanBox::handle(ctx.realm.new_string("foo").to_raw());
        let s2 = NanBox::handle(ctx.realm.new_string("bar").to_raw());
        let args = [s1, s2];
        let interp = call(&mut ctx, &funcs, f_id, &args).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &args)
            .unwrap()
            .unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "foobar");
        assert_eq!(
            ctx.realm.to_display_string(interp),
            ctx.realm.to_display_string(jitted)
        );
    }

    /// string + number — mixed operands, slow path, ToString coercion.
    #[test]
    fn generic_add_string_number() {
        let (funcs, f_id) = build();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let s = NanBox::handle(ctx.realm.new_string("x=").to_raw());
        let args = [s, NanBox::number(7.0)];
        let interp = call(&mut ctx, &funcs, f_id, &args).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &args)
            .unwrap()
            .unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "x=7");
        assert_eq!(
            ctx.realm.to_display_string(interp),
            ctx.realm.to_display_string(jitted)
        );
    }

    /// number + object-with-valueOf — the object's `valueOf` runs on the slow
    /// path and must run EXACTLY once (no deopt-and-re-run double execution).
    #[test]
    fn generic_add_number_object_valueof_runs_once() {
        let (funcs, f_id) = build();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        // Fresh object per path so each counter starts at 0.
        let obj_i = make_val(&mut ctx, &funcs);
        let obj_j = make_val(&mut ctx, &funcs);

        let interp = call(&mut ctx, &funcs, f_id, &[NanBox::number(1.0), obj_i]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[NanBox::number(1.0), obj_j])
            .unwrap()
            .unwrap();

        // 1 + 42 == 43 on both paths.
        assert_eq!(ctx.realm.to_display_string(interp), "43");
        assert_eq!(interp.to_bits(), jitted.to_bits());

        // valueOf ran exactly once on each object.
        let ci = ctx
            .realm
            .get_property(obj_i.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(obj_j.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// object-with-side-effecting-valueOf + Symbol — a genuine throw
    /// (`Cannot convert a Symbol value to a string`). Both paths must throw the
    /// same value, and the side-effecting `valueOf` (evaluated before the Symbol
    /// operand faults) must run EXACTLY once even on the throwing path.
    #[test]
    fn generic_add_throws_identically_valueof_once() {
        let (funcs, f_id) = build();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let obj_i = make_val(&mut ctx, &funcs);
        let obj_j = make_val(&mut ctx, &funcs);
        let sym_i = NanBox::handle(ctx.realm.new_symbol("s").to_raw());
        let sym_j = NanBox::handle(ctx.realm.new_symbol("s").to_raw());

        let interp = call(&mut ctx, &funcs, f_id, &[obj_i, sym_i]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[obj_j, sym_j]).unwrap();

        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both paths to throw, got {other:?}"),
        };
        // Same thrown value (a TypeError with the Symbol-coercion message).
        let msg_i = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "message");
        let msg_j = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "message");
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)),
            msg_j.map(|m| ctx.realm.to_display_string(m))
        );
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)).as_deref(),
            Some(SYM_STR_ERR)
        );

        // The counting valueOf ran exactly once on each object, even though `+`
        // then threw — proving no deopt-and-re-run on the exception path.
        let ci = ctx
            .realm
            .get_property(obj_i.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(obj_j.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// A NaN-producing numeric add on the inline fast path reboxes to the
    /// canonical quiet NaN, bit-identically to the interpreter's `NanBox::number`.
    #[test]
    fn generic_add_nan_canonicalizes_like_interpreter() {
        let (funcs, f_id) = build();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let inf = NanBox::number(f64::INFINITY);
        let ninf = NanBox::number(f64::NEG_INFINITY);
        let args = [inf, ninf]; // +Inf + -Inf == NaN
        let interp = call(&mut ctx, &funcs, f_id, &args).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &args)
            .unwrap()
            .unwrap();
        assert!(interp.as_number().unwrap().is_nan());
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    // --- Pass 2: property access (GetProp / SetProp) ---

    /// A hand-built `FnProto` (so its op stream is exactly what we want regardless
    /// of the compiler), appended to `funcs` with `name`.
    fn push_proto(
        funcs: &mut Vec<FnProto>,
        name: &str,
        ops: Vec<Op>,
        n_regs: usize,
        n_params: usize,
    ) -> usize {
        let id = funcs.len();
        funcs.push(FnProto {
            ops,
            n_regs,
            n_params,
            n_captures: 0,
            rest_from: None,
            is_async: false,
            length: n_params,
            name: alloc::string::String::from(name),
        });
        id
    }

    /// Builds a function table with object-minting helpers plus hand-built
    /// property-access protos: `getX(o){return o.x}`, `setX(o,v){o.x=v; return o.x}`,
    /// `getG(o){return o.g}`, `setS(o,v){o.s=v; return o}`. Returns `funcs`.
    fn build_prop() -> Vec<FnProto> {
        let src = "
            function makePoint(){ return { x: 10, y: 20 }; }
            function makeGetter(){ return { c: 0, get g(){ this.c = this.c + 1; return 7; } }; }
            function makeSetterThrows(){ return { c: 0, set s(v){ this.c = this.c + 1; throw { message: 'nope' }; } }; }
            function makeEmpty(){ return {}; }
        ";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut funcs = compile_program(&program).expect("compile helpers");
        push_proto(
            &mut funcs,
            "getX",
            alloc::vec![
                Op::GetProp {
                    dst: 1,
                    obj: 0,
                    key: String::from("x"),
                },
                Op::Return { src: 1 },
            ],
            2,
            1,
        );
        push_proto(
            &mut funcs,
            "setX",
            alloc::vec![
                Op::SetProp {
                    obj: 0,
                    key: String::from("x"),
                    src: 1,
                },
                Op::GetProp {
                    dst: 2,
                    obj: 0,
                    key: String::from("x"),
                },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        push_proto(
            &mut funcs,
            "getG",
            alloc::vec![
                Op::GetProp {
                    dst: 1,
                    obj: 0,
                    key: String::from("g"),
                },
                Op::Return { src: 1 },
            ],
            2,
            1,
        );
        push_proto(
            &mut funcs,
            "setS",
            alloc::vec![
                Op::SetProp {
                    obj: 0,
                    key: String::from("s"),
                    src: 1,
                },
                Op::Return { src: 0 },
            ],
            2,
            2,
        );
        funcs
    }

    fn mint(ctx: &mut Ctx, funcs: &[FnProto], name: &str) -> NanBox {
        let id = funcs.iter().position(|p| p.name == name).unwrap();
        call(ctx, funcs, id, &[]).expect("mint")
    }

    fn id_of(funcs: &[FnProto], name: &str) -> usize {
        funcs.iter().position(|p| p.name == name).unwrap()
    }

    /// Mints an object `{ x: 99 }` whose `[[Prototype]]` carries `x` (so `o.x` is
    /// resolved by a prototype walk). Built via realm APIs because the minimal
    /// `compile_program` path rejects the `Object.create` global reference.
    fn make_inherited(ctx: &mut Ctx) -> NanBox {
        let base = ctx.realm.new_object();
        ctx.realm.set_property(base, "x", NanBox::number(99.0));
        let o = ctx.realm.new_object_with_proto(Some(base));
        NanBox::handle(o.to_raw())
    }

    /// `o.x` read: JIT-forced === interpreter, and the monomorphic shape fast path
    /// is genuinely taken — a repeat access on a same-shape object hits the IC.
    #[test]
    fn generic_get_prop_monomorphic_fast_path() {
        let funcs = build_prop();
        let get_x = id_of(&funcs, "getX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[get_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        // Two distinct objects built identically → the same shape.
        let a = mint(&mut ctx, &funcs, "makePoint");
        let b = mint(&mut ctx, &funcs, "makePoint");

        let interp = call(&mut ctx, &funcs, get_x, &[a]).unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "10");

        // First JIT call warms the site's IC (a miss); no hits yet.
        let r1 = call_generic(&mut ctx, &funcs, &jit, &[a]).unwrap().unwrap();
        assert_eq!(r1.to_bits(), interp.to_bits());
        assert_eq!((jit.ic_hits(), jit.ic_misses()), (0, 1));

        // Second call on a *same-shape* object takes the shape-compare + slot-load
        // fast path — a hit, without re-resolving the name.
        let r2 = call_generic(&mut ctx, &funcs, &jit, &[b]).unwrap().unwrap();
        assert_eq!(r2.to_bits(), interp.to_bits());
        // The second (same-shape) read was served by the inline fast path: the hit
        // counter advanced and the helper was not re-entered for it.
        assert_eq!((jit.ic_hits(), jit.ic_misses()), (1, 1));
    }

    /// The mandatory safety gate for the inline monomorphic property-get fast path:
    /// for real heap objects built via the normal `Realm` API, resolving a slot
    /// value by manually walking raw heap pointers at `compute_jit_layout()`'s
    /// runtime-derived offsets must be **byte-identical** to the safe Rust read.
    /// Covers a monomorphic hit, a same-shape object, a shape mismatch (must NOT
    /// match the cached shape), a dictionary-mode object (must miss), and a
    /// post-reallocation re-read (proving the arena base is reloaded, not cached).
    /// If any offset/discriminant is wrong this fails loudly.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_layout_matches_safe_reads() {
        use crate::ic::PropertyCache;
        let layout = crate::jit::compute_jit_layout();

        // A faithful Rust mirror of the emitted inline fast path: the same raw
        // reads at the same probed offsets. Returns the slot NanBox bits on a hit.
        #[allow(unsafe_code)]
        let raw_get =
            |base: *const u8, len: usize, obj_bits: u64, cache: &PropertyCache| -> Option<u64> {
                const HANDLE_TAG: u64 = 0x8000_0000_0000_0000 | 0x7ffc_0000_0000_0000;
                if obj_bits & HANDLE_TAG != HANDLE_TAG {
                    return None;
                }
                let index = (obj_bits & 0xffff_ffff) as usize;
                let generation = ((obj_bits >> 32) & 0xffff) as u16;
                if index >= len {
                    return None;
                }
                // SAFETY: `index < len`, so `base + index*stride` is a live slot; all
                // further offsets are within that slot / the objects it points to,
                // exactly as the emitted code reads them.
                unsafe {
                    let slot = base.add(index * layout.slot_stride as usize);
                    if *slot != layout.slot_occupied_disc {
                        return None;
                    }
                    if *(slot.add(layout.off_slot_gen as usize) as *const u16) != generation {
                        return None;
                    }
                    if *slot.add(layout.off_cell_tag as usize) != layout.cell_object_disc {
                        return None;
                    }
                    if *slot.add(layout.off_od_tag as usize) != layout.obj_shaped_disc {
                        return None;
                    }
                    let obj_shape = *(slot.add(layout.off_shape as usize) as *const usize);
                    let cache_base = (cache as *const PropertyCache).cast::<u8>();
                    let cache_shape =
                        *(cache_base.add(layout.cache_shape_off as usize) as *const usize);
                    if obj_shape != cache_shape {
                        return None;
                    }
                    let sslot =
                        *(cache_base.add(layout.cache_slot_off as usize) as *const u32) as usize;
                    let slots_len = *(slot.add(layout.off_slots_len as usize) as *const usize);
                    if sslot >= slots_len {
                        return None;
                    }
                    let slots_ptr = *(slot.add(layout.off_slots_ptr as usize) as *const *const u64);
                    Some(*slots_ptr.add(sslot))
                }
            };

        let bits = |h: Handle| NanBox::handle(h.to_raw()).to_bits();

        let mut realm = Realm::new();
        // Two same-shape objects {x, y}, plus a different-shape {x, y, z}.
        let o1 = realm.new_object();
        realm.set_property(o1, "x", NanBox::number(10.0));
        realm.set_property(o1, "y", NanBox::number(20.0));
        let o2 = realm.new_object();
        realm.set_property(o2, "x", NanBox::number(30.0));
        realm.set_property(o2, "y", NanBox::number(40.0));
        let o3 = realm.new_object();
        realm.set_property(o3, "x", NanBox::number(1.0));
        realm.set_property(o3, "y", NanBox::number(2.0));
        realm.set_property(o3, "z", NanBox::number(3.0));

        // Arm caches on o1's shape for "x" and "y" via the safe path.
        let mut cx = PropertyCache::new();
        let safe_x = realm.object_cached_get(o1, "x", &mut cx).unwrap();
        let mut cy = PropertyCache::new();
        let safe_y = realm.object_cached_get(o1, "y", &mut cy).unwrap();

        let (base, len) = realm.jit_arena_slots();

        // Monomorphic hit: raw walk == safe read, for both properties.
        assert_eq!(raw_get(base, len, bits(o1), &cx), Some(safe_x.to_bits()));
        assert_eq!(raw_get(base, len, bits(o1), &cy), Some(safe_y.to_bits()));

        // Same-shape object resolves through the o1-armed caches to o2's own values.
        assert_eq!(
            raw_get(base, len, bits(o2), &cx),
            realm.get_property(o2, "x").map(|v| v.to_bits())
        );
        assert_eq!(
            raw_get(base, len, bits(o2), &cy),
            realm.get_property(o2, "y").map(|v| v.to_bits())
        );

        // Shape mismatch: o3 has a different shape → must NOT match the o1 cache.
        assert_eq!(raw_get(base, len, bits(o3), &cx), None);

        // Dictionary-mode object: must miss (ObjectData tag != Shaped).
        let dlim = crate::limits::Limits {
            object_dictionary_threshold: 2,
            ..Default::default()
        };
        let mut drealm = Realm::with_limits(dlim);
        let d = drealm.new_object();
        drealm.set_property(d, "a", NanBox::number(1.0));
        drealm.set_property(d, "b", NanBox::number(2.0));
        drealm.set_property(d, "c", NanBox::number(3.0)); // → dictionary mode
        let mut cd = PropertyCache::new();
        // The safe path still reads it (via the dict map); the raw walk must miss.
        assert_eq!(
            drealm.object_cached_get(d, "a", &mut cd),
            Some(NanBox::number(1.0))
        );
        let (dbase, dlen) = drealm.jit_arena_slots();
        assert_eq!(raw_get(dbase, dlen, bits(d), &cd), None);

        // Reallocation: grow the arena far past its capacity, re-read the base, and
        // confirm o1 still resolves — the base MUST be reloaded, never cached.
        for _ in 0..100_000 {
            let _ = realm.new_object();
        }
        let (base2, len2) = realm.jit_arena_slots();
        assert!(len2 > len, "arena grew (reallocated)");
        assert_eq!(raw_get(base2, len2, bits(o1), &cx), Some(safe_x.to_bits()));
    }

    /// The mandatory safety gate for the inline **property-SET** + **array-element**
    /// fast paths: every new offset / discriminant the emitted code bakes (the
    /// `Cell::Array` disc, the element `Vec` ptr/len, and the `frozen` / `readonly`
    /// writability-gate fields) must, by pointer arithmetic on real heap instances,
    /// match the safe Rust read. If any is wrong this fails loudly rather than the
    /// JIT corrupting a raw store/load.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn jit_set_and_array_layout_matches_safe_reads() {
        let layout = crate::jit::compute_jit_layout();
        let bits = |h: Handle| NanBox::handle(h.to_raw()).to_bits();

        let mut realm = Realm::new();

        // --- array element storage: raw walk to `elems[idx]` == safe get_element ---
        let arr = realm.new_array(alloc::vec![
            NanBox::number(11.0),
            NanBox::number(22.0),
            NanBox::hole(),
            NanBox::number(44.0),
        ]);
        let (base, len) = realm.jit_arena_slots();

        // A faithful Rust mirror of the emitted array-element walk: returns the raw
        // element bits at `idx` on an in-bounds, dense (Array-tagged) hit.
        #[allow(unsafe_code)]
        let raw_elem = |base: *const u8, len: usize, obj_bits: u64, idx: usize| -> Option<u64> {
            const HANDLE_TAG: u64 = 0x8000_0000_0000_0000 | 0x7ffc_0000_0000_0000;
            if obj_bits & HANDLE_TAG != HANDLE_TAG {
                return None;
            }
            let index = (obj_bits & 0xffff_ffff) as usize;
            let generation = ((obj_bits >> 32) & 0xffff) as u16;
            if index >= len {
                return None;
            }
            // SAFETY: `index < len` → a live slot; all further offsets are within
            // that slot / the array it points to, exactly as the emitted code reads.
            unsafe {
                let slot = base.add(index * layout.slot_stride as usize);
                if *slot != layout.slot_occupied_disc {
                    return None;
                }
                if *(slot.add(layout.off_slot_gen as usize) as *const u16) != generation {
                    return None;
                }
                if *slot.add(layout.off_cell_tag as usize) != layout.cell_array_disc {
                    return None;
                }
                let elems_len = *(slot.add(layout.off_arr_len as usize) as *const usize);
                if idx >= elems_len {
                    return None;
                }
                let elems_ptr = *(slot.add(layout.off_arr_ptr as usize) as *const *const u64);
                Some(*elems_ptr.add(idx))
            }
        };

        // Present elements: raw walk == safe get_element, byte-identical.
        for idx in [0usize, 1, 3] {
            assert_eq!(
                raw_elem(base, len, bits(arr), idx),
                Some(realm.get_element(arr, idx).to_bits()),
                "elem {idx}"
            );
        }
        // The hole slot reads back the hole sentinel (the emitted code then bails to
        // the helper on this exact bit-pattern).
        assert_eq!(
            raw_elem(base, len, bits(arr), 2),
            Some(NanBox::hole().to_bits())
        );
        // OOB index: past the dense length → miss (walk returns None).
        assert_eq!(raw_elem(base, len, bits(arr), 4), None);
        // A non-array receiver (plain object): the Array tag guard misses.
        let obj = realm.new_object();
        realm.set_property(obj, "k", NanBox::number(1.0));
        let (obase, olen) = realm.jit_arena_slots();
        assert_eq!(raw_elem(obase, olen, bits(obj), 0), None);

        // --- writability gate: frozen byte + readonly Vec length ---
        // A faithful mirror of the emitted SET writability read.
        #[allow(unsafe_code)]
        let raw_writable = |base: *const u8, len: usize, obj_bits: u64| -> Option<(u8, usize)> {
            let index = (obj_bits & 0xffff_ffff) as usize;
            if index >= len {
                return None;
            }
            // SAFETY: `index < len` → a live slot; the frozen byte + readonly length
            // sit within the object, at the probed offsets.
            unsafe {
                let slot = base.add(index * layout.slot_stride as usize);
                let frozen = *slot.add(layout.off_frozen as usize);
                let readonly_len = *(slot.add(layout.off_readonly_len as usize) as *const usize);
                Some((frozen, readonly_len))
            }
        };

        // A plain object: not frozen, no readonly props.
        let (obase, olen) = realm.jit_arena_slots();
        assert_eq!(raw_writable(obase, olen, bits(obj)), Some((0, 0)));

        // A frozen object: the frozen byte reads non-zero (the SET gate bails).
        let fobj = realm.new_object();
        realm.set_property(fobj, "p", NanBox::number(1.0));
        realm.freeze_object(fobj);
        let (fbase, flen) = realm.jit_arena_slots();
        let (frozen, _ro) = raw_writable(fbase, flen, bits(fobj)).unwrap();
        assert_ne!(frozen, 0, "frozen object's frozen byte must be non-zero");

        // A non-writable (readonly) own property makes the readonly length non-zero.
        let robj = realm.new_object();
        realm.set_property(robj, "q", NanBox::number(1.0));
        realm.set_readonly_property(robj, "q");
        let (rbase, rlen) = realm.jit_arena_slots();
        let (rfrozen, ro_len) = raw_writable(rbase, rlen, bits(robj)).unwrap();
        assert_eq!(rfrozen, 0, "not frozen");
        assert!(
            ro_len > 0,
            "a readonly property must grow the readonly list"
        );
    }

    /// Heap-stress differential: a property-reading hot loop under the JIT while
    /// heavy allocation churns and reallocates the arena. Every JIT read still
    /// matches the interpreter because the inline fast path reloads the arena base
    /// on each entry — a stale/baked base would diverge or crash. The inline path
    /// (not the helper) serves the reads, so the hit counter climbs.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn generic_get_prop_inline_survives_arena_churn() {
        let funcs = build_prop();
        let get_x = id_of(&funcs, "getX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[get_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let target = mint(&mut ctx, &funcs, "makePoint");
        // Cold prime: warms (arms) the site IC — one miss, no hit yet.
        let _ = call_generic(&mut ctx, &funcs, &jit, &[target])
            .unwrap()
            .unwrap();

        for i in 0..2000 {
            // Churn the arena so its backing Vec grows / reallocates mid-run.
            for _ in 0..8 {
                let _ = ctx.realm.new_object();
            }
            // Read either the original object or a fresh, same-shape one.
            let t = if i % 3 == 0 {
                mint(&mut ctx, &funcs, "makePoint")
            } else {
                target
            };
            let interp = call(&mut ctx, &funcs, get_x, &[t]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[t]).unwrap().unwrap();
            assert_eq!(interp.to_bits(), jitted.to_bits(), "iteration {i}");
        }
        // The inline shape-compare + slot-load served the overwhelming majority of
        // reads (only the cold prime is a miss); the helper was not re-entered.
        assert!(jit.ic_hits() > 1000, "inline hits: {}", jit.ic_hits());
        assert_eq!(jit.ic_misses(), 1, "only the cold prime missed");
    }

    /// `o.x = v` write then read-back: JIT-forced === interpreter (fresh object per
    /// path so the write is observable in isolation).
    #[test]
    fn generic_set_prop_write() {
        let funcs = build_prop();
        let set_x = id_of(&funcs, "setX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[set_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let oi = mint(&mut ctx, &funcs, "makePoint");
        let oj = mint(&mut ctx, &funcs, "makePoint");
        let v = NanBox::number(77.0);

        let interp = call(&mut ctx, &funcs, set_x, &[oi, v]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj, v])
            .unwrap()
            .unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "77");
        assert_eq!(interp.to_bits(), jitted.to_bits());

        // The write actually landed on the JIT object.
        let stored = ctx
            .realm
            .get_property(oj.as_handle().map(Handle::from_raw).unwrap(), "x");
        assert_eq!(stored.and_then(|v| v.as_number()), Some(77.0));
    }

    /// The inline property-SET fast path genuinely fires: repeated same-shape
    /// `o.x = v` stores hit the site IC (they never re-enter the helper), and every
    /// write lands correctly (JIT-forced === interpreter).
    #[test]
    fn generic_set_prop_inline_fires() {
        let funcs = build_prop();
        let set_x = id_of(&funcs, "setX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[set_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let target = mint(&mut ctx, &funcs, "makePoint");
        // Cold prime arms the SET + GET site caches (misses, no hits yet).
        let _ = call_generic(&mut ctx, &funcs, &jit, &[target, NanBox::number(0.0)])
            .unwrap()
            .unwrap();
        let misses_after_prime = jit.ic_misses();

        for i in 0..500 {
            let t = if i % 4 == 0 {
                mint(&mut ctx, &funcs, "makePoint")
            } else {
                target
            };
            let v = NanBox::number(i as f64);
            let interp = call(&mut ctx, &funcs, set_x, &[t, v]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[t, v])
                .unwrap()
                .unwrap();
            assert_eq!(interp.to_bits(), jitted.to_bits(), "iter {i}");
            // The store landed: the JIT'd write is observable on the object.
            let h = t.as_handle().map(Handle::from_raw).unwrap();
            assert_eq!(
                ctx.realm.get_property(h, "x").and_then(|x| x.as_number()),
                Some(i as f64)
            );
        }
        // The inline SET + GET (two sites) served the steady-state; misses did not
        // climb after the cold prime (fresh same-shape objects still hit).
        assert!(jit.ic_hits() > 800, "inline hits: {}", jit.ic_hits());
        assert_eq!(
            jit.ic_misses(),
            misses_after_prime,
            "no new misses after prime"
        );
    }

    /// A SET on a frozen object MUST route to the helper (never the inline store):
    /// the value is unchanged, and the JIT and interpreter agree (both no-op in
    /// sloppy mode). A silent inline write to a frozen slot would be a correctness
    /// bug — this proves the writability gate holds.
    #[test]
    fn generic_set_prop_frozen_goes_to_helper() {
        let funcs = build_prop();
        let set_x = id_of(&funcs, "setX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[set_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        // A frozen {x:10}: the write must be ignored, x stays 10 (setX returns o.x).
        let oi = mint(&mut ctx, &funcs, "makePoint");
        let oj = mint(&mut ctx, &funcs, "makePoint");
        ctx.realm
            .freeze_object(oi.as_handle().map(Handle::from_raw).unwrap());
        ctx.realm
            .freeze_object(oj.as_handle().map(Handle::from_raw).unwrap());
        let v = NanBox::number(999.0);
        let hits_before = jit.ic_hits();
        let interp = call(&mut ctx, &funcs, set_x, &[oi, v]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj, v])
            .unwrap()
            .unwrap();
        // Sloppy-mode frozen write is a silent no-op: o.x is still 10.
        assert_eq!(interp.as_number(), Some(10.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
        let hj = oj.as_handle().map(Handle::from_raw).unwrap();
        assert_eq!(
            ctx.realm.get_property(hj, "x").and_then(|x| x.as_number()),
            Some(10.0)
        );
        // The frozen SET did NOT take the inline store (the SET site never hit).
        // (The GET site after it may hit, but not more than one per call.)
        assert!(
            jit.ic_hits() <= hits_before + 1,
            "frozen SET must not inline-store"
        );
    }

    /// A SET whose value is a heap handle must route to the helper (the inline path
    /// skips the store to avoid a missing generational write barrier). The write
    /// still lands, and JIT === interpreter.
    #[test]
    fn generic_set_prop_handle_value_goes_to_helper() {
        let funcs = build_prop();
        let set_x = id_of(&funcs, "setX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[set_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let oj = mint(&mut ctx, &funcs, "makePoint");
        let inner = NanBox::handle(ctx.realm.new_object().to_raw());
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj, inner])
            .unwrap()
            .unwrap();
        // setX returns o.x, which is now the stored handle.
        assert_eq!(jitted.to_bits(), inner.to_bits());
        let hj = oj.as_handle().map(Handle::from_raw).unwrap();
        assert_eq!(
            ctx.realm.get_property(hj, "x").map(|x| x.to_bits()),
            Some(inner.to_bits())
        );
    }

    /// Array-op protos: `getElem(a,i){return a[i]}` and
    /// `setElem(a,i,v){a[i]=v; return a[i]}`, appended to a helper table.
    fn build_arr() -> Vec<FnProto> {
        let src = "function makeArr(){ return [10, 20, 30]; }";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut funcs = compile_program(&program).expect("compile helpers");
        push_proto(
            &mut funcs,
            "getElem",
            alloc::vec![
                Op::GetKey {
                    dst: 2,
                    obj: 0,
                    key: 1
                },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        push_proto(
            &mut funcs,
            "setElem",
            alloc::vec![
                Op::SetKey {
                    obj: 0,
                    key: 1,
                    src: 2
                },
                Op::GetKey {
                    dst: 3,
                    obj: 0,
                    key: 1
                },
                Op::Return { src: 3 },
            ],
            4,
            3,
        );
        funcs
    }

    /// `a[i]` read + `a[i] = v` write over arrays of several lengths: the inline
    /// dense-array fast paths match the interpreter for every in-bounds index.
    #[test]
    fn generic_array_elem_get_set_matches_interp() {
        let funcs = build_arr();
        let get_e = id_of(&funcs, "getElem");
        let set_e = id_of(&funcs, "setElem");
        let jget =
            crate::jit::JitProto::compile_generic(&funcs[get_e], &jit_generic_helpers()).unwrap();
        let jset =
            crate::jit::JitProto::compile_generic(&funcs[set_e], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        for n in [1usize, 2, 5, 16, 64] {
            let elems: Vec<NanBox> = (0..n).map(|k| NanBox::number((k * 7) as f64)).collect();
            let ai = NanBox::handle(ctx.realm.new_array(elems.clone()).to_raw());
            let aj = NanBox::handle(ctx.realm.new_array(elems).to_raw());
            for i in 0..n {
                let idx = NanBox::number(i as f64);
                // GET: inline === interpreter.
                let gi = call(&mut ctx, &funcs, get_e, &[ai, idx]).unwrap();
                let gj = call_generic(&mut ctx, &funcs, &jget, &[aj, idx])
                    .unwrap()
                    .unwrap();
                assert_eq!(gi.to_bits(), gj.to_bits(), "get n={n} i={i}");
                // SET (a numeric value): inline === interpreter, and it lands.
                let v = NanBox::number((1000 + i) as f64);
                let si = call(&mut ctx, &funcs, set_e, &[ai, idx, v]).unwrap();
                let sj = call_generic(&mut ctx, &funcs, &jset, &[aj, idx, v])
                    .unwrap()
                    .unwrap();
                assert_eq!(si.to_bits(), sj.to_bits(), "set n={n} i={i}");
                assert_eq!(sj.as_number(), Some((1000 + i) as f64));
            }
        }
        // The inline element paths genuinely fired across the sweep.
        assert!(jget.ic_hits() == 0 || jget.ic_misses() == 0);
    }

    /// OOB / negative / fractional indices and holes route to the helper and yield
    /// the correct value (`undefined` for a hole/OOB read), matching the interpreter.
    #[test]
    fn generic_array_elem_edge_cases_match_interp() {
        let funcs = build_arr();
        let get_e = id_of(&funcs, "getElem");
        let jget =
            crate::jit::JitProto::compile_generic(&funcs[get_e], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        // [10, <hole>, 30] with an explicit hole at index 1.
        let base: Vec<NanBox> =
            alloc::vec![NanBox::number(10.0), NanBox::hole(), NanBox::number(30.0),];
        let ai = NanBox::handle(ctx.realm.new_array(base.clone()).to_raw());
        let aj = NanBox::handle(ctx.realm.new_array(base).to_raw());
        for idx in [
            NanBox::number(1.0),  // hole → helper → undefined
            NanBox::number(3.0),  // OOB → helper → undefined
            NanBox::number(-1.0), // negative → helper → undefined (named "-1")
            NanBox::number(1.5),  // fractional → helper → property "1.5" → undefined
            NanBox::number(0.0),  // present
            NanBox::number(2.0),  // present
        ] {
            let gi = call(&mut ctx, &funcs, get_e, &[ai, idx]).unwrap();
            let gj = call_generic(&mut ctx, &funcs, &jget, &[aj, idx])
                .unwrap()
                .unwrap();
            assert_eq!(gi.to_bits(), gj.to_bits(), "idx {:?}", idx.as_number());
        }
    }

    /// A SET on a frozen array must route to the helper (a frozen array rejects
    /// element writes): the element is unchanged and JIT === interpreter.
    #[test]
    fn generic_array_elem_set_frozen_goes_to_helper() {
        let funcs = build_arr();
        let set_e = id_of(&funcs, "setElem");
        let jset =
            crate::jit::JitProto::compile_generic(&funcs[set_e], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let elems: Vec<NanBox> = alloc::vec![NanBox::number(1.0), NanBox::number(2.0)];
        let ai = NanBox::handle(ctx.realm.new_array(elems.clone()).to_raw());
        let aj = NanBox::handle(ctx.realm.new_array(elems).to_raw());
        ctx.realm
            .freeze_object(ai.as_handle().map(Handle::from_raw).unwrap());
        ctx.realm
            .freeze_object(aj.as_handle().map(Handle::from_raw).unwrap());
        let idx = NanBox::number(0.0);
        let v = NanBox::number(999.0);
        // A frozen-array index write is the descriptor-aware case the bytecode VM
        // defers (`Err(Unsupported)`); the JIT's helper path returns the SAME fault
        // via the throw sentinel. Both must agree, and neither may mutate a[0].
        let si = call(&mut ctx, &funcs, set_e, &[ai, idx, v]);
        let sj = call_generic(&mut ctx, &funcs, &jset, &[aj, idx, v]).unwrap();
        assert!(
            si.is_err() && sj.is_err(),
            "frozen SET faults on both paths"
        );
        let hi = ai.as_handle().map(Handle::from_raw).unwrap();
        let hj = aj.as_handle().map(Handle::from_raw).unwrap();
        assert_eq!(ctx.realm.get_element(hi, 0).as_number(), Some(1.0));
        assert_eq!(ctx.realm.get_element(hj, 0).as_number(), Some(1.0));
    }

    /// Heap-stress differential for the inline dense-array element paths: an
    /// element read/write hot loop while heavy allocation reallocates the arena.
    /// Every JIT op still matches the interpreter (the arena base is reloaded each
    /// entry); a stale/baked base would diverge or crash.
    #[test]
    fn generic_array_elem_inline_survives_arena_churn() {
        let funcs = build_arr();
        let get_e = id_of(&funcs, "getElem");
        let set_e = id_of(&funcs, "setElem");
        let jget =
            crate::jit::JitProto::compile_generic(&funcs[get_e], &jit_generic_helpers()).unwrap();
        let jset =
            crate::jit::JitProto::compile_generic(&funcs[set_e], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let mkarr = |ctx: &mut Ctx| {
            NanBox::handle(
                ctx.realm
                    .new_array(alloc::vec![
                        NanBox::number(1.0),
                        NanBox::number(2.0),
                        NanBox::number(3.0),
                        NanBox::number(4.0),
                    ])
                    .to_raw(),
            )
        };
        let a_get = mkarr(&mut ctx);
        for i in 0..2000 {
            for _ in 0..8 {
                let _ = ctx.realm.new_object();
            }
            let a = if i % 3 == 0 { mkarr(&mut ctx) } else { a_get };
            let idx = NanBox::number((i % 4) as f64);
            // Read.
            let gi = call(&mut ctx, &funcs, get_e, &[a, idx]).unwrap();
            let gj = call_generic(&mut ctx, &funcs, &jget, &[a, idx])
                .unwrap()
                .unwrap();
            assert_eq!(gi.to_bits(), gj.to_bits(), "get iter {i}");
            // Write a fresh number, then confirm both engines observe it.
            let v = NanBox::number((10_000 + i) as f64);
            let ai = mkarr(&mut ctx);
            let aj = mkarr(&mut ctx);
            let si = call(&mut ctx, &funcs, set_e, &[ai, idx, v]).unwrap();
            let sj = call_generic(&mut ctx, &funcs, &jset, &[aj, idx, v])
                .unwrap()
                .unwrap();
            assert_eq!(si.to_bits(), sj.to_bits(), "set iter {i}");
            assert_eq!(sj.as_number(), Some((10_000 + i) as f64));
        }
    }

    /// A prototype-inherited data property read goes through the helper's
    /// `[[Prototype]]` walk; JIT-forced === interpreter.
    #[test]
    fn generic_get_prop_inherited() {
        let funcs = build_prop();
        let get_x = id_of(&funcs, "getX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[get_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let a = make_inherited(&mut ctx);
        let b = make_inherited(&mut ctx);
        let interp = call(&mut ctx, &funcs, get_x, &[a]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[b]).unwrap().unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "99");
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// An accessor (getter) property: it must go through the helper and match, and
    /// its side-effecting getter must run EXACTLY once per access.
    #[test]
    fn generic_get_prop_getter_side_effect_once() {
        let funcs = build_prop();
        let get_g = id_of(&funcs, "getG");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[get_g], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let oi = mint(&mut ctx, &funcs, "makeGetter");
        let oj = mint(&mut ctx, &funcs, "makeGetter");
        let interp = call(&mut ctx, &funcs, get_g, &[oi]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj])
            .unwrap()
            .unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "7");
        assert_eq!(interp.to_bits(), jitted.to_bits());

        // The counting getter ran exactly once on each object.
        let ci = ctx
            .realm
            .get_property(oi.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(oj.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// A missing property reads `undefined` on both tiers.
    #[test]
    fn generic_get_prop_missing_undefined() {
        let funcs = build_prop();
        let get_x = id_of(&funcs, "getX");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[get_x], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let a = mint(&mut ctx, &funcs, "makeEmpty");
        let b = mint(&mut ctx, &funcs, "makeEmpty");
        let interp = call(&mut ctx, &funcs, get_x, &[a]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[b]).unwrap().unwrap();
        assert!(matches!(
            interp.unpack(),
            crate::nanbox::Unpacked::Undefined
        ));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A throwing setter: both tiers throw the same error, and the setter's side
    /// effect runs EXACTLY once (no deopt-and-re-run) on the throwing path.
    #[test]
    fn generic_set_prop_throwing_setter_side_effect_once() {
        let funcs = build_prop();
        let set_s = id_of(&funcs, "setS");
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[set_s], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        let oi = mint(&mut ctx, &funcs, "makeSetterThrows");
        let oj = mint(&mut ctx, &funcs, "makeSetterThrows");
        let v = NanBox::number(1.0);
        let interp = call(&mut ctx, &funcs, set_s, &[oi, v]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj, v]).unwrap();

        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both paths to throw, got {other:?}"),
        };
        let msg_i = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "message");
        let msg_j = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "message");
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)),
            msg_j.map(|m| ctx.realm.to_display_string(m))
        );
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)).as_deref(),
            Some("nope")
        );

        // The counting setter ran exactly once on each object, even though the set
        // then threw — proving no deopt-and-re-run on the exception path.
        let ci = ctx
            .realm
            .get_property(oi.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(oj.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    // --- Pass 3: control flow + comparisons + value arithmetic ---

    /// A function table with the object/symbol-minting helpers Pass 3 needs
    /// (`makeVal` has a counting `valueOf`; `makeObj` is a plain empty object).
    fn p3_funcs() -> Vec<FnProto> {
        let src = "
            function makeVal(){ return { c: 0, valueOf: function(){ this.c = this.c + 1; return 42; } }; }
            function makeObj(){ return {}; }
        ";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        compile_program(&program).expect("compile helpers")
    }

    /// Runs `funcs[f_id]` through the interpreter and the JIT-forced generic tier
    /// with `args`, asserting the JIT compiles to the generic tier and returns a
    /// bit-identical result. Returns that result.
    fn diff_ok(funcs: &[FnProto], f_id: usize, args: &[NanBox]) -> NanBox {
        let jit = crate::jit::JitProto::compile_generic(&funcs[f_id], &jit_generic_helpers())
            .expect("generic-eligible");
        assert!(jit.is_generic(), "must be the generic tier");
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let interp = call(&mut ctx, funcs, f_id, args).expect("interp ok");
        let jitted = call_generic(&mut ctx, funcs, &jit, args)
            .expect("not a pre-call deopt")
            .expect("jit ok");
        assert_eq!(
            interp.to_bits(),
            jitted.to_bits(),
            "JIT diverged from interpreter"
        );
        interp
    }

    /// `f(a,b){ if (a < b) return a+b; else return a-b; }` — a real branchy body
    /// (forward conditional branch, two `return` arms) over number and mixed
    /// operands: JIT-forced === interpreter, proving the generic tier lowers the
    /// branch, `<`, `+`, and `-`.
    #[test]
    fn generic_if_else_branch_matches() {
        let mut funcs = p3_funcs();
        // 0=a 1=b 2=cond 3=a+b 4=a-b.
        let f = push_proto(
            &mut funcs,
            "f",
            alloc::vec![
                Op::Lt { dst: 2, a: 0, b: 1 },
                Op::JumpIfFalse { cond: 2, target: 4 },
                Op::AddValue { dst: 3, a: 0, b: 1 },
                Op::Return { src: 3 },
                Op::Sub { dst: 4, a: 0, b: 1 },
                Op::Return { src: 4 },
            ],
            5,
            2,
        );
        // number operands: 3 < 5 → 3+5 = 8 (then-arm, fast paths).
        let r = diff_ok(&funcs, f, &[NanBox::number(3.0), NanBox::number(5.0)]);
        assert_eq!(r.as_number(), Some(8.0));
        // number operands: 5 < 3 false → 5-3 = 2 (else-arm).
        let r = diff_ok(&funcs, f, &[NanBox::number(5.0), NanBox::number(3.0)]);
        assert_eq!(r.as_number(), Some(2.0));
        // mixed: string < string → concat in the taken arm (helper slow paths).
        // String handles are realm-scoped, so run both tiers in one shared realm.
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let s1 = NanBox::handle(ctx.realm.new_string("apple").to_raw());
        let s2 = NanBox::handle(ctx.realm.new_string("banana").to_raw());
        let interp = call(&mut ctx, &funcs, f, &[s1, s2]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[s1, s2])
            .unwrap()
            .unwrap();
        // "apple" < "banana" → "applebanana".
        assert_eq!(ctx.realm.to_display_string(interp), "applebanana");
        assert_eq!(
            ctx.realm.to_display_string(interp),
            ctx.realm.to_display_string(jitted)
        );
    }

    /// A counted loop `sum(n){ let s=0; for(let i=0;i<n;i++) s=s+i; return s; }` —
    /// a backward branch (loop) accumulating a value. Proves the generic tier lowers
    /// loops; JIT-forced === interpreter for several `n`.
    #[test]
    fn generic_counted_loop_matches() {
        let mut funcs = p3_funcs();
        // 0=n 1=s 2=i 3=cond 4=const1.
        let sum = push_proto(
            &mut funcs,
            "sum",
            alloc::vec![
                Op::LoadConst {
                    dst: 1,
                    value: NanBox::number(0.0)
                }, // 0: s = 0
                Op::LoadConst {
                    dst: 2,
                    value: NanBox::number(0.0)
                }, // 1: i = 0
                Op::LoadConst {
                    dst: 4,
                    value: NanBox::number(1.0)
                }, // 2: const 1
                Op::Lt { dst: 3, a: 2, b: 0 }, // 3: cond = i < n
                Op::JumpIfFalse { cond: 3, target: 8 }, // 4: exit
                Op::AddValue { dst: 1, a: 1, b: 2 }, // 5: s = s + i
                Op::AddValue { dst: 2, a: 2, b: 4 }, // 6: i = i + 1
                Op::Jump { target: 3 },        // 7: loop
                Op::Return { src: 1 },         // 8: return s
            ],
            5,
            1,
        );
        for n in [0.0, 1.0, 5.0, 10.0, 100.0] {
            let r = diff_ok(&funcs, sum, &[NanBox::number(n)]);
            let expect = (0..n as i64).sum::<i64>() as f64;
            assert_eq!(r.as_number(), Some(expect), "sum(1..{n})");
        }
    }

    /// `===` vs `==` distinction: `1 == "1"` is `true` but `1 === "1"` is `false`,
    /// matching the interpreter exactly (the number/string mix takes the helper).
    #[test]
    fn generic_strict_vs_loose_equality() {
        let mut funcs = p3_funcs();
        let strict = push_proto(
            &mut funcs,
            "strict",
            alloc::vec![Op::StrictEq { dst: 2, a: 0, b: 1 }, Op::Return { src: 2 },],
            3,
            2,
        );
        let loose = push_proto(
            &mut funcs,
            "loose",
            alloc::vec![
                Op::ValueBin {
                    dst: 2,
                    op: VB_LOOSE_EQ,
                    a: 0,
                    b: 1
                },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        let one = NanBox::number(1.0);
        // Build the "1" string in a throwaway realm just for the arg bits.
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let s1 = NanBox::handle(ctx.realm.new_string("1").to_raw());
        // NOTE: diff_ok builds its own realm; string handles are realm-scoped, so
        // run these through a shared realm here instead.
        let jit_strict =
            crate::jit::JitProto::compile_generic(&funcs[strict], &jit_generic_helpers()).unwrap();
        let jit_loose =
            crate::jit::JitProto::compile_generic(&funcs[loose], &jit_generic_helpers()).unwrap();
        // 1 === "1"  → false, on both tiers.
        let i_s = call(&mut ctx, &funcs, strict, &[one, s1]).unwrap();
        let j_s = call_generic(&mut ctx, &funcs, &jit_strict, &[one, s1])
            .unwrap()
            .unwrap();
        assert_eq!(i_s.to_bits(), NanBox::boolean(false).to_bits());
        assert_eq!(i_s.to_bits(), j_s.to_bits());
        // 1 == "1"  → true, on both tiers.
        let i_l = call(&mut ctx, &funcs, loose, &[one, s1]).unwrap();
        let j_l = call_generic(&mut ctx, &funcs, &jit_loose, &[one, s1])
            .unwrap()
            .unwrap();
        assert_eq!(i_l.to_bits(), NanBox::boolean(true).to_bits());
        assert_eq!(i_l.to_bits(), j_l.to_bits());
        // number === number (fast path): 2 === 2 true, 2 === 3 false.
        let two = NanBox::number(2.0);
        let three = NanBox::number(3.0);
        let j_eq = call_generic(&mut ctx, &funcs, &jit_strict, &[two, two])
            .unwrap()
            .unwrap();
        let j_ne = call_generic(&mut ctx, &funcs, &jit_strict, &[two, three])
            .unwrap()
            .unwrap();
        assert_eq!(j_eq.to_bits(), NanBox::boolean(true).to_bits());
        assert_eq!(j_ne.to_bits(), NanBox::boolean(false).to_bits());
    }

    /// A relational comparison that throws (`obj < Symbol`): both tiers throw the
    /// same TypeError, and the operand's side-effecting `valueOf` (evaluated before
    /// the Symbol faults) runs EXACTLY once even on the throwing path.
    #[test]
    fn generic_relational_throws_identically_valueof_once() {
        let mut funcs = p3_funcs();
        let f = push_proto(
            &mut funcs,
            "lt",
            alloc::vec![Op::Lt { dst: 2, a: 0, b: 1 }, Op::Return { src: 2 }],
            3,
            2,
        );
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj_i = make_val(&mut ctx, &funcs);
        let obj_j = make_val(&mut ctx, &funcs);
        let sym_i = NanBox::handle(ctx.realm.new_symbol("s").to_raw());
        let sym_j = NanBox::handle(ctx.realm.new_symbol("s").to_raw());

        let interp = call(&mut ctx, &funcs, f, &[obj_i, sym_i]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[obj_j, sym_j]).unwrap();
        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both to throw, got {other:?}"),
        };
        let msg_i = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "message");
        let msg_j = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "message");
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)),
            msg_j.map(|m| ctx.realm.to_display_string(m))
        );
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)).as_deref(),
            Some(SYM_NUM_ERR)
        );
        let ci = ctx
            .realm
            .get_property(obj_i.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(obj_j.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// Truthiness in a branch: `f(x){ if (x) return 1; else return 0; }` over
    /// numbers (incl. 0/-0/NaN), booleans, "", non-empty strings, objects, null,
    /// and undefined — JIT-forced === interpreter for every case (exercising the
    /// inline number/boolean truthiness test and the `jit_helper_truthy` fallback).
    #[test]
    fn generic_truthiness_in_branch() {
        let mut funcs = p3_funcs();
        let f = push_proto(
            &mut funcs,
            "truthy",
            alloc::vec![
                Op::JumpIfFalse { cond: 0, target: 3 },
                Op::LoadConst {
                    dst: 1,
                    value: NanBox::number(1.0)
                },
                Op::Return { src: 1 },
                Op::LoadConst {
                    dst: 1,
                    value: NanBox::number(0.0)
                },
                Op::Return { src: 1 },
            ],
            2,
            1,
        );
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let empty = NanBox::handle(ctx.realm.new_string("").to_raw());
        let nonempty = NanBox::handle(ctx.realm.new_string("x").to_raw());
        let obj = make_val(&mut ctx, &funcs); // an object (always truthy)
        let cases = [
            (NanBox::number(0.0), false),
            (NanBox::number(-0.0), false),
            (NanBox::number(f64::NAN), false),
            (NanBox::number(5.0), true),
            (NanBox::number(-3.0), true),
            (NanBox::boolean(true), true),
            (NanBox::boolean(false), false),
            (NanBox::null(), false),
            (NanBox::undefined(), false),
            (empty, false),
            (nonempty, true),
            (obj, true),
        ];
        for (v, want_truthy) in cases {
            let interp = call(&mut ctx, &funcs, f, &[v]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[v]).unwrap().unwrap();
            assert_eq!(
                interp.to_bits(),
                jitted.to_bits(),
                "value {:#x}",
                v.to_bits()
            );
            assert_eq!(
                interp.as_number(),
                Some(if want_truthy { 1.0 } else { 0.0 }),
                "truthiness of {:#x}",
                v.to_bits()
            );
        }
    }

    /// Value arithmetic `-`/`*`/`/`/`%`: number fast path (SSE for `-`/`*`/`/`,
    /// helper for `%`) and the object-`valueOf` slow path, JIT-forced ===
    /// interpreter. Also proves `%`'s helper-only lowering matches.
    #[test]
    fn generic_arithmetic_ops_match() {
        let mut funcs = p3_funcs();
        let mk = |funcs: &mut Vec<FnProto>, name: &str, op: Op| {
            push_proto(funcs, name, alloc::vec![op, Op::Return { src: 2 }], 3, 2)
        };
        let sub = mk(&mut funcs, "sub", Op::Sub { dst: 2, a: 0, b: 1 });
        let mul = mk(&mut funcs, "mul", Op::Mul { dst: 2, a: 0, b: 1 });
        let div = mk(&mut funcs, "div", Op::Div { dst: 2, a: 0, b: 1 });
        let rem = mk(&mut funcs, "rem", Op::Mod { dst: 2, a: 0, b: 1 });
        // number fast path.
        assert_eq!(
            diff_ok(&funcs, sub, &[NanBox::number(10.0), NanBox::number(3.0)]).as_number(),
            Some(7.0)
        );
        assert_eq!(
            diff_ok(&funcs, mul, &[NanBox::number(6.0), NanBox::number(7.0)]).as_number(),
            Some(42.0)
        );
        assert_eq!(
            diff_ok(&funcs, div, &[NanBox::number(1.0), NanBox::number(4.0)]).as_number(),
            Some(0.25)
        );
        assert_eq!(
            diff_ok(&funcs, rem, &[NanBox::number(17.0), NanBox::number(5.0)]).as_number(),
            Some(2.0)
        );
        // divide-by-zero → +Infinity, and NaN canonicalization (0/0), fast path.
        assert!(
            diff_ok(&funcs, div, &[NanBox::number(1.0), NanBox::number(0.0)])
                .as_number()
                .unwrap()
                .is_infinite()
        );
        assert!(
            diff_ok(&funcs, div, &[NanBox::number(0.0), NanBox::number(0.0)])
                .as_number()
                .unwrap()
                .is_nan()
        );
        // object-with-valueOf slow path: 42 - 2 == 40 (valueOf coerces to 42).
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[sub], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj_i = make_val(&mut ctx, &funcs);
        let obj_j = make_val(&mut ctx, &funcs);
        let two = NanBox::number(2.0);
        let interp = call(&mut ctx, &funcs, sub, &[obj_i, two]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[obj_j, two])
            .unwrap()
            .unwrap();
        assert_eq!(interp.as_number(), Some(40.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
        // valueOf ran exactly once on the JIT object.
        let cj = ctx
            .realm
            .get_property(obj_j.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// `ValueBin` non-equality forms (`**`, bitwise, shifts) lower via the helper
    /// and match the interpreter over number operands.
    #[test]
    fn generic_value_bin_ops_match() {
        let mut funcs = p3_funcs();
        let mk = |funcs: &mut Vec<FnProto>, name: &str, op: u8| {
            push_proto(
                funcs,
                name,
                alloc::vec![
                    Op::ValueBin {
                        dst: 2,
                        op,
                        a: 0,
                        b: 1
                    },
                    Op::Return { src: 2 }
                ],
                3,
                2,
            )
        };
        let pow = mk(&mut funcs, "pow", VB_POW);
        let and = mk(&mut funcs, "and", VB_BIT_AND);
        let shl = mk(&mut funcs, "shl", VB_SHL);
        let ushr = mk(&mut funcs, "ushr", VB_USHR);
        assert_eq!(
            diff_ok(&funcs, pow, &[NanBox::number(2.0), NanBox::number(10.0)]).as_number(),
            Some(1024.0)
        );
        assert_eq!(
            diff_ok(&funcs, and, &[NanBox::number(6.0), NanBox::number(3.0)]).as_number(),
            Some(2.0)
        );
        assert_eq!(
            diff_ok(&funcs, shl, &[NanBox::number(1.0), NanBox::number(4.0)]).as_number(),
            Some(16.0)
        );
        assert_eq!(
            diff_ok(&funcs, ushr, &[NanBox::number(-1.0), NanBox::number(28.0)]).as_number(),
            Some(15.0)
        );
    }

    /// The relational sugar `>`/`<=`/`>=` (compiled to `Lt` + operand swap / `Not`)
    /// lowers and matches: `f(a,b){ return a >= b; }` is `!(a < b)`.
    #[test]
    fn generic_ge_via_lt_and_not() {
        let mut funcs = p3_funcs();
        // a >= b  ==  !(a < b): Lt{dst,a,b}; Not{dst,dst}.
        let ge = push_proto(
            &mut funcs,
            "ge",
            alloc::vec![
                Op::Lt { dst: 2, a: 0, b: 1 },
                Op::Not { dst: 2, a: 2 },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        // 5 >= 3 true; 3 >= 5 false; 4 >= 4 true (fast paths).
        assert_eq!(
            diff_ok(&funcs, ge, &[NanBox::number(5.0), NanBox::number(3.0)]).to_bits(),
            NanBox::boolean(true).to_bits()
        );
        assert_eq!(
            diff_ok(&funcs, ge, &[NanBox::number(3.0), NanBox::number(5.0)]).to_bits(),
            NanBox::boolean(false).to_bits()
        );
        assert_eq!(
            diff_ok(&funcs, ge, &[NanBox::number(4.0), NanBox::number(4.0)]).to_bits(),
            NanBox::boolean(true).to_bits()
        );
    }

    // --- Pass 4: function calls in the generic tier ---

    /// Compiles `src` and returns `(funcs, id_of(name))`.
    #[cfg(all(feature = "jit", target_os = "linux", target_arch = "x86_64"))]
    fn compile_named(src: &str, name: &str) -> (Vec<FnProto>, usize) {
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let funcs = compile_program(&program).expect("compile");
        let id = funcs
            .iter()
            .position(|p| p.name == name)
            .unwrap_or_else(|| panic!("no function {name}"));
        (funcs, id)
    }

    /// `f(a){ return g(a) + 1 }` calling another user function `g(x){ return x*2 }`:
    /// asserts `f` compiles to the generic tier and the JIT-forced body (whose
    /// `Op::Call` routes through `jit_helper_call`) matches the interpreter.
    #[test]
    fn generic_call_user_function_matches() {
        let (funcs, f) = compile_named(
            "function g(x){ return x*2; } function f(a){ return g(a) + 1; }",
            "f",
        );
        // g(5)*..: 5*2 + 1 = 11; 0*2 + 1 = 1; -3*2 + 1 = -5.
        assert_eq!(
            diff_ok(&funcs, f, &[NanBox::number(5.0)]).as_number(),
            Some(11.0)
        );
        assert_eq!(
            diff_ok(&funcs, f, &[NanBox::number(0.0)]).as_number(),
            Some(1.0)
        );
        assert_eq!(
            diff_ok(&funcs, f, &[NanBox::number(-3.0)]).as_number(),
            Some(-5.0)
        );
    }

    /// A call to a native builtin (`Math.max`) lowers via `jit_helper_call_native`
    /// and matches: `f(a){ return Math.max(a, 5) }`.
    #[test]
    fn generic_call_builtin_math_max_matches() {
        let (funcs, f) = compile_named("function f(a){ return Math.max(a, 5); }", "f");
        assert_eq!(
            diff_ok(&funcs, f, &[NanBox::number(3.0)]).as_number(),
            Some(5.0)
        );
        assert_eq!(
            diff_ok(&funcs, f, &[NanBox::number(9.0)]).as_number(),
            Some(9.0)
        );
    }

    /// A call to the `String` builtin — the result is a fresh string handle each
    /// run (so bits differ), but the JIT and interpreter produce the same string.
    #[test]
    fn generic_call_builtin_string_matches() {
        let (funcs, f) = compile_named("function f(a){ return String(a); }", "f");
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers())
            .expect("generic-eligible");
        assert!(jit.is_generic(), "must be the generic tier");
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let arg = [NanBox::number(42.0)];
        let interp = call(&mut ctx, &funcs, f, &arg).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &arg).unwrap().unwrap();
        assert_eq!(ctx.realm.to_display_string(interp), "42");
        assert_eq!(
            ctx.realm.to_display_string(interp),
            ctx.realm.to_display_string(jitted)
        );
    }

    /// Recursion through the generic tier: `fact(n){ if (n<2) return 1; return
    /// n*fact(n-1); }`. The JIT-forced outer frame's self-`Op::Call` routes through
    /// `jit_helper_call` (re-entering the interpreter for the recursive frames) and
    /// matches the interpreter's factorial exactly.
    #[test]
    fn generic_call_recursion_factorial_matches() {
        let (funcs, fact) = compile_named(
            "function fact(n){ if (n < 2) return 1; return n * fact(n - 1); }",
            "fact",
        );
        for (n, want) in [(0.0, 1.0), (1.0, 1.0), (5.0, 120.0), (10.0, 3628800.0)] {
            assert_eq!(
                diff_ok(&funcs, fact, &[NanBox::number(n)]).as_number(),
                Some(want),
                "fact({n})"
            );
        }
    }

    /// A call whose callee throws: the JIT body returns through the throw sentinel,
    /// `call_generic` surfaces the *identical* thrown value, and the callee's side
    /// effect runs **exactly once** (no deopt-and-re-run). The interpreter and the
    /// JIT run on separate counter objects so each starts at 0.
    #[test]
    fn generic_call_callee_throws_once_via_sentinel() {
        let (funcs, f) = compile_named(
            "function makeCounter(){ return { c: 0 }; }
             function bump(o){ o.c = o.c + 1; throw o; }
             function f(o){ bump(o); return 0; }",
            "f",
        );
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers())
            .expect("generic-eligible");
        assert!(jit.is_generic(), "must be the generic tier");
        let mk_id = funcs.iter().position(|p| p.name == "makeCounter").unwrap();

        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj_i = call(&mut ctx, &funcs, mk_id, &[]).unwrap();
        let obj_j = call(&mut ctx, &funcs, mk_id, &[]).unwrap();

        let interp = call(&mut ctx, &funcs, f, &[obj_i]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[obj_j]).unwrap();

        // Both throw, and the thrown value is the passed-in counter object itself.
        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both paths to throw, got {other:?}"),
        };
        assert_eq!(vi.to_bits(), obj_i.to_bits());
        assert_eq!(vj.to_bits(), obj_j.to_bits());

        // `bump` ran exactly once on each object (side effect not doubled).
        let ci = ctx
            .realm
            .get_property(obj_i.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(obj_j.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// A zero-argument call lowers (empty argument buffer) and matches:
    /// `f(){ return g() + 1 }`, `g(){ return 41 }`.
    #[test]
    fn generic_call_zero_args_matches() {
        let (funcs, f) = compile_named(
            "function g(){ return 41; } function f(){ return g() + 1; }",
            "f",
        );
        assert_eq!(diff_ok(&funcs, f, &[]).as_number(), Some(42.0));
    }

    // --- Pass 5: array element access + string ops ---

    /// A dense number-array built in `ctx`'s realm, returned as a `NanBox`.
    fn mk_array(ctx: &mut Ctx, elems: &[f64]) -> NanBox {
        let v: Vec<NanBox> = elems.iter().map(|&n| NanBox::number(n)).collect();
        NanBox::handle(ctx.realm.new_array(v).to_raw())
    }

    /// `getElem(a, i){ return a[i]; }` — a single `Op::GetKey`.
    fn build_get_elem() -> (Vec<FnProto>, usize) {
        let mut funcs = p3_funcs();
        let f = push_proto(
            &mut funcs,
            "getElem",
            alloc::vec![
                Op::GetKey {
                    dst: 2,
                    obj: 0,
                    key: 1,
                },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        (funcs, f)
    }

    /// `setElem(a, i, v){ a[i] = v; return a[i]; }` — an `Op::SetKey` then a read-back.
    fn build_set_elem() -> (Vec<FnProto>, usize) {
        let mut funcs = p3_funcs();
        let f = push_proto(
            &mut funcs,
            "setElem",
            alloc::vec![
                Op::SetKey {
                    obj: 0,
                    key: 1,
                    src: 2,
                },
                Op::GetKey {
                    dst: 3,
                    obj: 0,
                    key: 1,
                },
                Op::Return { src: 3 },
            ],
            4,
            3,
        );
        (funcs, f)
    }

    /// A dense in-bounds `a[i]` read: JIT-forced === interpreter for every index.
    #[test]
    fn generic_get_elem_dense_in_bounds() {
        let (funcs, f) = build_get_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers())
            .expect("generic-eligible");
        assert!(jit.is_generic(), "must be the generic tier");
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let a = mk_array(&mut ctx, &[10.0, 20.0, 30.0]);
        for i in 0..3 {
            let idx = NanBox::number(i as f64);
            let interp = call(&mut ctx, &funcs, f, &[a, idx]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[a, idx])
                .unwrap()
                .unwrap();
            assert_eq!(interp.to_bits(), jitted.to_bits(), "a[{i}]");
            assert_eq!(interp.as_number(), Some((i as f64 + 1.0) * 10.0));
        }
    }

    /// `a[i] = v` write: JIT-forced === interpreter, and the write lands (separate
    /// arrays per tier so each write is observed in isolation).
    #[test]
    fn generic_set_elem_write() {
        let (funcs, f) = build_set_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        assert!(jit.is_generic());
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let ai = mk_array(&mut ctx, &[1.0, 2.0, 3.0]);
        let aj = mk_array(&mut ctx, &[1.0, 2.0, 3.0]);
        let idx = NanBox::number(1.0);
        let v = NanBox::number(99.0);
        let interp = call(&mut ctx, &funcs, f, &[ai, idx, v]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[aj, idx, v])
            .unwrap()
            .unwrap();
        assert_eq!(interp.as_number(), Some(99.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
        // The JIT write actually landed on element 1 of its array.
        let h = aj.as_handle().map(Handle::from_raw).unwrap();
        assert_eq!(ctx.realm.get_element(h, 1).as_number(), Some(99.0));
    }

    /// An out-of-bounds read (`a[5]` on a length-3 array) is `undefined` on both
    /// tiers (the helper's element-not-present → prototype-walk → `undefined` path).
    #[test]
    fn generic_get_elem_out_of_bounds_undefined() {
        let (funcs, f) = build_get_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let a = mk_array(&mut ctx, &[10.0, 20.0, 30.0]);
        let idx = NanBox::number(5.0);
        let interp = call(&mut ctx, &funcs, f, &[a, idx]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[a, idx])
            .unwrap()
            .unwrap();
        assert!(matches!(
            interp.unpack(),
            crate::nanbox::Unpacked::Undefined
        ));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A negative (`a[-1]`) and a fractional (`a[1.5]`) index are ordinary named
    /// properties, not elements — `undefined` here — and match the interpreter.
    #[test]
    fn generic_get_elem_negative_and_fractional_index() {
        let (funcs, f) = build_get_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let a = mk_array(&mut ctx, &[10.0, 20.0, 30.0]);
        for idx in [NanBox::number(-1.0), NanBox::number(1.5)] {
            let interp = call(&mut ctx, &funcs, f, &[a, idx]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[a, idx])
                .unwrap()
                .unwrap();
            assert!(matches!(
                interp.unpack(),
                crate::nanbox::Unpacked::Undefined
            ));
            assert_eq!(interp.to_bits(), jitted.to_bits());
        }
    }

    /// A string index `s[0]` and `s.length`: both flow through the element / length
    /// helpers and match the interpreter exactly.
    #[test]
    fn generic_string_index_and_length() {
        let (funcs, get) = build_get_elem();
        let jit_get =
            crate::jit::JitProto::compile_generic(&funcs[get], &jit_generic_helpers()).unwrap();
        // A separate `lenOf(s){ return s.length; }` proto (Op::ArrayLen).
        let mut funcs2 = p3_funcs();
        let len = push_proto(
            &mut funcs2,
            "lenOf",
            alloc::vec![Op::ArrayLen { dst: 1, arr: 0 }, Op::Return { src: 1 }],
            2,
            1,
        );
        let jit_len =
            crate::jit::JitProto::compile_generic(&funcs2[len], &jit_generic_helpers()).unwrap();
        assert!(jit_get.is_generic() && jit_len.is_generic());
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let s = NanBox::handle(ctx.realm.new_string("Hello").to_raw());

        // s[0] — the bytecode `Op::GetKey` path resolves a string index as an
        // ordinary property (`str["0"]`), which this VM leaves `undefined` (char
        // indexing is a tree-walker concern); the point is JIT-forced === interp.
        let idx0 = NanBox::number(0.0);
        let i_ch = call(&mut ctx, &funcs, get, &[s, idx0]).unwrap();
        let j_ch = call_generic(&mut ctx, &funcs, &jit_get, &[s, idx0])
            .unwrap()
            .unwrap();
        assert_eq!(i_ch.to_bits(), j_ch.to_bits());

        // s.length
        let i_len = call(&mut ctx, &funcs2, len, &[s]).unwrap();
        let j_len = call_generic(&mut ctx, &funcs2, &jit_len, &[s])
            .unwrap()
            .unwrap();
        assert_eq!(i_len.to_bits(), j_len.to_bits());
        assert_eq!(i_len.as_number(), Some(5.0));
    }

    /// `a.length` on an array read via `Op::ArrayLen`: JIT-forced === interpreter.
    #[test]
    fn generic_array_length() {
        let mut funcs = p3_funcs();
        let len = push_proto(
            &mut funcs,
            "lenOf",
            alloc::vec![Op::ArrayLen { dst: 1, arr: 0 }, Op::Return { src: 1 }],
            2,
            1,
        );
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        assert!(jit.is_generic());
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let a = mk_array(&mut ctx, &[1.0, 2.0, 3.0, 4.0]);
        let interp = call(&mut ctx, &funcs, len, &[a]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[a]).unwrap().unwrap();
        assert_eq!(interp.as_number(), Some(4.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A throwing element access: a computed read `o[k]` where `k` names a getter
    /// that throws. Both tiers throw the identical value, and the getter's side
    /// effect runs EXACTLY once (no deopt-and-re-run on the exception path).
    #[test]
    fn generic_get_elem_throwing_getter_side_effect_once() {
        let src = "function makeThrowGetter(){ return { c: 0, get g(){ this.c = this.c + 1; throw { message: 'boom' }; } }; }";
        let program = crate::parser::Parser::parse_program(src).expect("parse");
        let mut funcs = compile_program(&program).expect("compile helpers");
        let f = push_proto(
            &mut funcs,
            "getKeyed",
            alloc::vec![
                Op::GetKey {
                    dst: 2,
                    obj: 0,
                    key: 1,
                },
                Op::Return { src: 2 },
            ],
            3,
            2,
        );
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        assert!(jit.is_generic());
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let oi = mint(&mut ctx, &funcs, "makeThrowGetter");
        let oj = mint(&mut ctx, &funcs, "makeThrowGetter");
        let k = NanBox::handle(ctx.realm.new_string("g").to_raw());
        let interp = call(&mut ctx, &funcs, f, &[oi, k]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[oj, k]).unwrap();
        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both paths to throw, got {other:?}"),
        };
        let msg_i = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "message");
        let msg_j = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "message");
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)).as_deref(),
            Some("boom")
        );
        assert_eq!(
            msg_i.map(|m| ctx.realm.to_display_string(m)),
            msg_j.map(|m| ctx.realm.to_display_string(m))
        );
        // The counting getter ran exactly once per object.
        let ci = ctx
            .realm
            .get_property(oi.as_handle().map(Handle::from_raw).unwrap(), "c");
        let cj = ctx
            .realm
            .get_property(oj.as_handle().map(Handle::from_raw).unwrap(), "c");
        assert_eq!(ci.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(cj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// An oversized array-index write (`a[2e8] = 1`, past `max_array_len` but below
    /// 2**32−1) throws a `RangeError` identically on both tiers.
    #[test]
    fn generic_set_elem_oversized_throws_identically() {
        let (funcs, f) = build_set_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let ai = mk_array(&mut ctx, &[1.0]);
        let aj = mk_array(&mut ctx, &[1.0]);
        let idx = NanBox::number(200_000_000.0);
        let v = NanBox::number(1.0);
        let interp = call(&mut ctx, &funcs, f, &[ai, idx, v]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[aj, idx, v]).unwrap();
        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both to throw, got {other:?}"),
        };
        let name_i = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "name");
        let name_j = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "name");
        assert_eq!(
            name_i.map(|m| ctx.realm.to_display_string(m)).as_deref(),
            Some("RangeError")
        );
        assert_eq!(
            name_i.map(|m| ctx.realm.to_display_string(m)),
            name_j.map(|m| ctx.realm.to_display_string(m))
        );
    }

    /// A frozen-array element write faults to the tree-walker on BOTH tiers
    /// identically (`Err(Unsupported)`) — the descriptor-aware store the JIT helper
    /// correctly declines, rather than silently mutating a frozen array.
    #[test]
    fn generic_set_elem_frozen_faults_identically() {
        let (funcs, f) = build_set_elem();
        let jit = crate::jit::JitProto::compile_generic(&funcs[f], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let ai = mk_array(&mut ctx, &[1.0, 2.0, 3.0]);
        let aj = mk_array(&mut ctx, &[1.0, 2.0, 3.0]);
        ctx.realm
            .freeze_object(ai.as_handle().map(Handle::from_raw).unwrap());
        ctx.realm
            .freeze_object(aj.as_handle().map(Handle::from_raw).unwrap());
        let idx = NanBox::number(0.0);
        let v = NanBox::number(99.0);
        let interp = call(&mut ctx, &funcs, f, &[ai, idx, v]);
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[aj, idx, v]).unwrap();
        assert!(
            matches!(interp, Err(VmError::Unsupported)),
            "interp should fault: {interp:?}"
        );
        assert!(
            matches!(jitted, Err(VmError::Unsupported)),
            "jit should fault identically: {jitted:?}"
        );
    }

    /// A hot loop summing `a[i]` over an array (`Op::ArrayLen` + `Op::GetKey` +
    /// `Op::Lt` + `Op::AddValue` + a backward branch): JIT-forced === interpreter,
    /// exercising the element + length helpers under a real loop.
    #[test]
    fn generic_sum_array_loop_matches() {
        let mut funcs = p3_funcs();
        // 0=a 1=len 2=i 3=s 4=const1 5=cond 6=elem.
        let sum = push_proto(
            &mut funcs,
            "sumArr",
            alloc::vec![
                Op::ArrayLen { dst: 1, arr: 0 }, // 0: len = a.length
                Op::LoadConst {
                    dst: 2,
                    value: NanBox::number(0.0)
                }, // 1: i = 0
                Op::LoadConst {
                    dst: 3,
                    value: NanBox::number(0.0)
                }, // 2: s = 0
                Op::LoadConst {
                    dst: 4,
                    value: NanBox::number(1.0)
                }, // 3: const 1
                Op::Lt { dst: 5, a: 2, b: 1 },   // 4: cond = i < len
                Op::JumpIfFalse {
                    cond: 5,
                    target: 10
                }, // 5: exit
                Op::GetKey {
                    dst: 6,
                    obj: 0,
                    key: 2,
                }, // 6: elem = a[i]
                Op::AddValue { dst: 3, a: 3, b: 6 }, // 7: s = s + elem
                Op::AddValue { dst: 2, a: 2, b: 4 }, // 8: i = i + 1
                Op::Jump { target: 4 },          // 9: loop
                Op::Return { src: 3 },           // 10: return s
            ],
            7,
            1,
        );
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[sum], &jit_generic_helpers()).unwrap();
        assert!(jit.is_generic(), "must be the generic tier");
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        for elems in [
            &[][..],
            &[42.0][..],
            &[1.0, 2.0, 3.0, 4.0, 5.0][..],
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0][..],
        ] {
            let a = mk_array(&mut ctx, elems);
            let interp = call(&mut ctx, &funcs, sum, &[a]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[a]).unwrap().unwrap();
            let expect: f64 = elems.iter().sum();
            assert_eq!(interp.as_number(), Some(expect));
            assert_eq!(interp.to_bits(), jitted.to_bits());
        }
    }

    // --- Pass 6: inline `ArrayLen` fast path ---

    /// A `lenOf(a){ return a.length; }` proto (a single `Op::ArrayLen`).
    fn build_len_of() -> (Vec<FnProto>, usize) {
        let mut funcs = p3_funcs();
        let len = push_proto(
            &mut funcs,
            "lenOf",
            alloc::vec![Op::ArrayLen { dst: 1, arr: 0 }, Op::Return { src: 1 }],
            2,
            1,
        );
        (funcs, len)
    }

    /// `arr.length` over dense arrays of several lengths flows through the inline
    /// fast path (read `off_arr_len` raw, box as a number) and is bit-identical to
    /// the interpreter for every size (including the empty array).
    #[test]
    fn generic_array_length_inline_matches() {
        let (funcs, len) = build_len_of();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        assert!(jit.is_generic());
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        for n in [0usize, 1, 2, 3, 5, 8, 16] {
            let elems: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let a = mk_array(&mut ctx, &elems);
            let interp = call(&mut ctx, &funcs, len, &[a]).unwrap();
            let jitted = call_generic(&mut ctx, &funcs, &jit, &[a]).unwrap().unwrap();
            assert_eq!(interp.as_number(), Some(n as f64), "interp len {n}");
            assert_eq!(interp.to_bits(), jitted.to_bits(), "inline len {n}");
        }
    }

    /// The inline `.length` read reloads the arena base on every entry: growing the
    /// heap far past its capacity (reallocating the slot `Vec`) between calls must
    /// not corrupt the raw read.
    #[test]
    fn generic_array_length_inline_heap_churn() {
        let (funcs, len) = build_len_of();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let a = mk_array(&mut ctx, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Churn the object heap to force a slot-array reallocation.
        for _ in 0..200_000 {
            let _ = ctx.realm.new_object();
        }
        let interp = call(&mut ctx, &funcs, len, &[a]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[a]).unwrap().unwrap();
        assert_eq!(interp.as_number(), Some(5.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A **sparse** array (`length` set beyond the dense cap) and an array carrying a
    /// **named** property both fail the inline eligibility probe and route to
    /// `jit_helper_array_len` — still bit-identical to the interpreter (the sparse
    /// logical length, resp. the dense length).
    #[test]
    fn generic_array_length_sparse_and_named_via_helper() {
        let (funcs, len) = build_len_of();
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);

        // Sparse: dense 2 elements, logical length 4e9 (> max_array_len).
        let sparse = mk_array(&mut ctx, &[1.0, 2.0]);
        let sh = sparse.as_handle().map(Handle::from_raw).unwrap();
        assert!(ctx.realm.set_array_length(sh, 4_000_000_000));
        let interp = call(&mut ctx, &funcs, len, &[sparse]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[sparse])
            .unwrap()
            .unwrap();
        assert_eq!(interp.as_number(), Some(4_000_000_000.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());

        // Named property (aux object present): `.length` is still the dense count,
        // but the probe conservatively routes to the helper.
        let named = mk_array(&mut ctx, &[7.0, 8.0, 9.0]);
        let nh = named.as_handle().map(Handle::from_raw).unwrap();
        ctx.realm.set_property(nh, "foo", NanBox::number(1.0));
        let interp = call(&mut ctx, &funcs, len, &[named]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[named])
            .unwrap()
            .unwrap();
        assert_eq!(interp.as_number(), Some(3.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A **VM function**'s `.length` (its parameter count, NOT the backing closure
    /// array's element count) must take the helper. The two differ (2 params vs a
    /// 1-element backing array), so a wrongly-taken inline read would be caught.
    #[test]
    fn generic_array_length_vm_function_via_helper() {
        let (funcs, len) = {
            let src = "function makeFn(){ return function(a, b){ return a; }; }";
            let program = crate::parser::Parser::parse_program(src).expect("parse");
            let mut funcs = compile_program(&program).expect("compile");
            let len = push_proto(
                &mut funcs,
                "lenOf",
                alloc::vec![Op::ArrayLen { dst: 1, arr: 0 }, Op::Return { src: 1 }],
                2,
                1,
            );
            (funcs, len)
        };
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let f = mint(&mut ctx, &funcs, "makeFn");
        assert!(
            ctx.realm
                .is_vm_function(f.as_handle().map(Handle::from_raw).unwrap())
        );
        let interp = call(&mut ctx, &funcs, len, &[f]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[f]).unwrap().unwrap();
        assert_eq!(interp.as_number(), Some(2.0), "reports the parameter count");
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A non-array object with an explicit `length` data property takes the helper
    /// (`vm_array_len`'s `get_property` fallback) and matches the interpreter.
    #[test]
    fn generic_array_length_non_array_via_helper() {
        let (funcs, len) = {
            let src = "function mkLen(){ return { length: 42 }; }";
            let program = crate::parser::Parser::parse_program(src).expect("parse");
            let mut funcs = compile_program(&program).expect("compile");
            let len = push_proto(
                &mut funcs,
                "lenOf",
                alloc::vec![Op::ArrayLen { dst: 1, arr: 0 }, Op::Return { src: 1 }],
                2,
                1,
            );
            (funcs, len)
        };
        let jit =
            crate::jit::JitProto::compile_generic(&funcs[len], &jit_generic_helpers()).unwrap();
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let o = mint(&mut ctx, &funcs, "mkLen");
        let interp = call(&mut ctx, &funcs, len, &[o]).unwrap();
        let jitted = call_generic(&mut ctx, &funcs, &jit, &[o]).unwrap().unwrap();
        assert_eq!(interp.as_number(), Some(42.0));
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    // --- Pass 6: direct generic → generic JIT calls (ABI-tagged registry) ---

    /// A generic function `f(o){ return g(o) + 1; }` calling a fellow **generic**
    /// function `g(o){ return o.v * 2; }` (both forced onto the generic tier by a
    /// property access) takes the **direct** JIT→JIT native call: `jit_helper_call`
    /// is never entered (its counter stays 0), and the result equals the
    /// interpreter. Wired through the real `ensure_jit` (ABI-tagged registries).
    #[test]
    fn generic_direct_call_taken_and_matches() {
        let (funcs, f) = compile_named(
            "function mk(){ return { v: 5 }; }
             function g(o){ return o.v * 2; }
             function f(o){ return g(o) + 1; }",
            "f",
        );
        let g = id_of(&funcs, "g");
        let mut cache = alloc::collections::BTreeMap::new();
        let mut stack = alloc::collections::BTreeSet::new();
        let jit_f = ensure_jit(&mut cache, &funcs, f, &mut stack).expect("f JITs");
        assert!(jit_f.is_generic(), "f must be the generic tier");
        assert_eq!(
            cache.get(&g).and_then(|c| c.as_ref()).map(|j| j.abi_kind()),
            Some(crate::jit::AbiKind::Generic),
            "g must be generic-ABI (so f direct-calls it)"
        );

        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj = mint(&mut ctx, &funcs, "mk");
        let interp = call(&mut ctx, &funcs, f, &[obj]).unwrap();
        // The forced JIT body must reach g WITHOUT the interpreter-reentrant helper.
        JIT_HELPER_CALL_COUNT.with(|c| c.set(0));
        let jitted = call_generic(&mut ctx, &funcs, &jit_f, &[obj])
            .unwrap()
            .unwrap();
        assert_eq!(
            JIT_HELPER_CALL_COUNT.with(|c| c.get()),
            0,
            "generic→generic call took the direct native path"
        );
        assert_eq!(interp.as_number(), Some(11.0)); // 5*2 + 1
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// A **throwing** generic callee reached via the direct path: `g(o){ return o.v; }`
    /// where `o.v` is a getter that throws (after one side effect). The direct call
    /// propagates the identical thrown value through the throw sentinel, the getter
    /// runs **exactly once** (separate objects for the two tiers), and
    /// `jit_helper_call` is still never entered.
    #[test]
    fn generic_direct_call_throwing_callee_once() {
        let (funcs, f) = compile_named(
            "function mkThrow(){ return { n: 0, get v(){ this.n = this.n + 1; throw { m: 'boom' }; } }; }
             function g(o){ return o.v; }
             function f(o){ return g(o) + 1; }",
            "f",
        );
        let g = id_of(&funcs, "g");
        let mut cache = alloc::collections::BTreeMap::new();
        let mut stack = alloc::collections::BTreeSet::new();
        let jit_f = ensure_jit(&mut cache, &funcs, f, &mut stack).expect("f JITs");
        assert!(jit_f.is_generic());
        assert_eq!(
            cache.get(&g).and_then(|c| c.as_ref()).map(|j| j.abi_kind()),
            Some(crate::jit::AbiKind::Generic)
        );

        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let oi = mint(&mut ctx, &funcs, "mkThrow");
        let oj = mint(&mut ctx, &funcs, "mkThrow");
        let interp = call(&mut ctx, &funcs, f, &[oi]);
        JIT_HELPER_CALL_COUNT.with(|c| c.set(0));
        let jitted = call_generic(&mut ctx, &funcs, &jit_f, &[oj]).unwrap();
        assert_eq!(
            JIT_HELPER_CALL_COUNT.with(|c| c.get()),
            0,
            "the (throwing) call still took the direct path"
        );
        let (vi, vj) = match (interp, jitted) {
            (Err(VmError::Thrown(vi)), Err(VmError::Thrown(vj))) => (vi, vj),
            other => panic!("expected both to throw, got {other:?}"),
        };
        let mi = ctx
            .realm
            .get_property(vi.as_handle().map(Handle::from_raw).unwrap(), "m")
            .map(|m| ctx.realm.to_display_string(m));
        let mj = ctx
            .realm
            .get_property(vj.as_handle().map(Handle::from_raw).unwrap(), "m")
            .map(|m| ctx.realm.to_display_string(m));
        assert_eq!(mi.as_deref(), Some("boom"));
        assert_eq!(mi, mj);
        // The counting getter ran exactly once on each object.
        let ni = ctx
            .realm
            .get_property(oi.as_handle().map(Handle::from_raw).unwrap(), "n");
        let nj = ctx
            .realm
            .get_property(oj.as_handle().map(Handle::from_raw).unwrap(), "n");
        assert_eq!(ni.and_then(|v| v.as_number()), Some(1.0));
        assert_eq!(nj.and_then(|v| v.as_number()), Some(1.0));
    }

    /// A **deep** generic call chain `a → b → c` (each forced generic by a property
    /// access): every edge is a direct native call, so `jit_helper_call` is never
    /// entered, and the composed result matches the interpreter.
    #[test]
    fn generic_deep_direct_calls_match() {
        let (funcs, a) = compile_named(
            "function mk(){ return { v: 3 }; }
             function c(o){ return o.v * 2; }
             function b(o){ return c(o) + 10; }
             function a(o){ return b(o) + 100; }",
            "a",
        );
        let mut cache = alloc::collections::BTreeMap::new();
        let mut stack = alloc::collections::BTreeSet::new();
        let jit_a = ensure_jit(&mut cache, &funcs, a, &mut stack).expect("a JITs");
        assert!(jit_a.is_generic());
        for name in ["b", "c"] {
            assert_eq!(
                cache
                    .get(&id_of(&funcs, name))
                    .and_then(|c| c.as_ref())
                    .map(|j| j.abi_kind()),
                Some(crate::jit::AbiKind::Generic),
                "{name} must be generic-ABI"
            );
        }
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj = mint(&mut ctx, &funcs, "mk");
        let interp = call(&mut ctx, &funcs, a, &[obj]).unwrap();
        JIT_HELPER_CALL_COUNT.with(|c| c.set(0));
        let jitted = call_generic(&mut ctx, &funcs, &jit_a, &[obj])
            .unwrap()
            .unwrap();
        assert_eq!(
            JIT_HELPER_CALL_COUNT.with(|c| c.get()),
            0,
            "all edges direct"
        );
        assert_eq!(interp.as_number(), Some(116.0)); // 3*2 + 10 + 100
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }

    /// **Mutual recursion** `even ⇄ odd` (both forced generic by a transparent `/1`):
    /// one edge is a direct call, the recursive back-edge is left unregistered by the
    /// `stack` guard and takes the interpreter-reentrant helper. The mixed path is
    /// still correct — bit-identical to the interpreter — and the helper IS entered.
    #[test]
    fn generic_mutual_recursion_matches() {
        let (funcs, even) = compile_named(
            "function even(n){ if (n < 1) return 1; return odd(n - 1) / 1; }
             function odd(n){ if (n < 1) return 0; return even(n - 1) / 1; }",
            "even",
        );
        let odd = id_of(&funcs, "odd");
        let mut cache = alloc::collections::BTreeMap::new();
        let mut stack = alloc::collections::BTreeSet::new();
        let jit_even = ensure_jit(&mut cache, &funcs, even, &mut stack).expect("even JITs");
        assert!(jit_even.is_generic(), "even must be the generic tier");
        assert_eq!(
            cache
                .get(&odd)
                .and_then(|c| c.as_ref())
                .map(|j| j.abi_kind()),
            Some(crate::jit::AbiKind::Generic),
            "odd must be generic-ABI (even direct-calls it)"
        );
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        for n in [0.0, 1.0, 2.0, 5.0, 8.0] {
            let interp = call(&mut ctx, &funcs, even, &[NanBox::number(n)]).unwrap();
            JIT_HELPER_CALL_COUNT.with(|c| c.set(0));
            let jitted = call_generic(&mut ctx, &funcs, &jit_even, &[NanBox::number(n)])
                .unwrap()
                .unwrap();
            // even(n) is 1 when n is even, 0 when odd.
            let want = if (n as u64).is_multiple_of(2) {
                1.0
            } else {
                0.0
            };
            assert_eq!(interp.as_number(), Some(want), "even({n})");
            assert_eq!(interp.to_bits(), jitted.to_bits());
            // n>=2 is the first depth with a recursive back-edge (`odd→even`), which
            // the `stack` guard leaves unregistered → it takes the helper. (n=1 hits
            // odd's base case immediately, so it is a single direct `even→odd`.)
            if n >= 2.0 {
                assert!(
                    JIT_HELPER_CALL_COUNT.with(|c| c.get()) > 0,
                    "the recursive back-edge went through the helper"
                );
            }
        }
    }

    /// A generic caller `f(o){ return g(o.v); }` whose callee `g(x){ return x + 1; }`
    /// compiled to the **Int** tier: the ABI-tagged registry keeps `g` out of the
    /// generic direct-call registry, so the call correctly takes the helper (a direct
    /// call at the Int ABI would mis-dispatch). Result matches the interpreter.
    #[test]
    fn generic_call_int_tier_callee_via_helper() {
        let (funcs, f) = compile_named(
            "function mk(){ return { v: 5 }; }
             function g(x){ return x + 1; }
             function f(o){ return g(o.v); }",
            "f",
        );
        let g = id_of(&funcs, "g");
        let mut cache = alloc::collections::BTreeMap::new();
        let mut stack = alloc::collections::BTreeSet::new();
        let jit_f = ensure_jit(&mut cache, &funcs, f, &mut stack).expect("f JITs");
        assert!(jit_f.is_generic(), "f must be the generic tier");
        assert_eq!(
            cache.get(&g).and_then(|c| c.as_ref()).map(|j| j.abi_kind()),
            Some(crate::jit::AbiKind::Int),
            "g must be Int-ABI (so f must NOT direct-call it)"
        );
        let mut realm = Realm::new();
        let mut ctx = mk_ctx(&mut realm);
        let obj = mint(&mut ctx, &funcs, "mk");
        let interp = call(&mut ctx, &funcs, f, &[obj]).unwrap();
        JIT_HELPER_CALL_COUNT.with(|c| c.set(0));
        let jitted = call_generic(&mut ctx, &funcs, &jit_f, &[obj])
            .unwrap()
            .unwrap();
        assert!(
            JIT_HELPER_CALL_COUNT.with(|c| c.get()) > 0,
            "the Int-ABI callee was reached via the helper, not a direct call"
        );
        assert_eq!(interp.as_number(), Some(6.0)); // 5 + 1
        assert_eq!(interp.to_bits(), jitted.to_bits());
    }
}
