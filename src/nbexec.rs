//! Executing real **statements and functions** over the [`Realm`]/[`NanBox`]
//! model (`ROADMAP.md` §3 → Phase D migration).
//!
//! [`Realm`]: crate::realm::Realm
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! A small tree-walking interpreter whose values are NaN-boxed and whose
//! objects/strings/arrays/functions live in the realm's GC heap — the
//! imperative *and procedural* core of the language on the performance
//! representation. It has:
//! - lexical variable scope ([`Scope`](crate::env::Scope) chains), assignment
//!   (incl. compound and member targets), block scoping, and control flow
//!   (`if`/`while`/`for`, `return`/`break`/`continue`);
//! - **functions and closures**: declarations (hoisted), expressions, and arrows
//!   become heap closures capturing their defining scope, so a returned inner
//!   function still sees its enclosing variables — and calls bind arguments in a
//!   fresh child scope;
//! - **exceptions** (`try`/`catch`/`finally`/`throw`); and
//! - a **starter stdlib**: native globals (`Math`, `String`/`Number`/`parseInt`)
//!   and built-in String/Array methods, including the higher-order
//!   `map`/`filter`/`reduce`/`forEach` that call back into closures.
//!
//! The *full* stdlib port and folding back into the bytecode VM are the
//! remaining migration work. Pure, safe `alloc`-only Rust.

use crate::ast::{
    Argument, ArrayElement, ArrayPatternElement, Arrow, ArrowBody, AssignOp, BinaryOp,
    BindingTarget, Class, ClassMember, Expr, ForInit, Function, Ident, LogicalOp, MethodKind,
    ObjectMember, Param, Program, PropertyKey, Stmt, UnaryOp, VarDecl,
};
use crate::env::Scope;
use crate::heap::Handle;
use crate::nanbox::{NanBox, Unpacked};
use crate::realm::Realm;
use alloc::string::String;
use alloc::vec::Vec;

/// Why execution stopped.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExecError {
    /// A construct outside the supported subset (generators, classes, …).
    Unsupported(&'static str),
    /// A reference to an undeclared variable.
    NotDefined(String),
    /// A call of a non-function value.
    NotCallable,
    /// A thrown JS value, propagating until a `catch` handles it.
    Throw(NanBox),
    /// An optional-chain short-circuit: a `?.` link found a nullish base. It
    /// propagates (past intervening non-optional links) up to the enclosing
    /// `Expr::OptChain` boundary, which turns it into `undefined`. It is *not* a
    /// throw, so `try`/`catch` never sees it.
    OptShortCircuit,
}

/// The control-flow outcome of a statement.
enum Flow {
    /// Fell through normally, carrying the last expression value (for `run`).
    Normal(NanBox),
    /// A `return` (value).
    Return(NanBox),
    /// A `break`, optionally targeting a label.
    Break(Option<String>),
    /// A `continue`, optionally targeting a label.
    Continue(Option<String>),
}

/// What a loop should do with a `Flow` produced by its body, given the loop's
/// own label (if any).
enum LoopAction {
    /// Proceed to the next iteration (fall through to the update).
    Next,
    /// Stop this loop.
    Stop,
    /// Not for this loop — bubble it up to an enclosing loop / labeled block.
    Propagate(Flow),
}

/// Classifies a body `Flow` for a loop carrying `label`.
fn loop_action(flow: Flow, label: &Option<String>) -> LoopAction {
    match flow {
        Flow::Normal(_) => LoopAction::Next,
        Flow::Continue(None) => LoopAction::Next,
        Flow::Continue(Some(l)) if Some(&l) == label.as_ref() => LoopAction::Next,
        Flow::Break(None) => LoopAction::Stop,
        Flow::Break(Some(l)) if Some(&l) == label.as_ref() => LoopAction::Stop,
        other => LoopAction::Propagate(other),
    }
}

/// The body of a registered function: a block, or a concise arrow expression.
#[derive(Clone, Copy)]
enum Body<'a> {
    Block(&'a [Stmt]),
    Expr(&'a Expr),
}

/// A registered function definition (its AST, held by the interpreter; the heap
/// closure stores only an index into the table plus the captured scope).
#[derive(Clone, Copy)]
struct FnDef<'a> {
    params: &'a [Param],
    body: Body<'a>,
    is_async: bool,
    /// Whether this is a generator (`function*`) — run eagerly into an iterator.
    is_generator: bool,
    /// Whether this is an arrow function (no own `arguments` binding).
    is_arrow: bool,
    /// Whether the function is strict (its own `"use strict"`, or defined inside
    /// strict code) — strict functions keep an `undefined`/`null` `this`.
    is_strict: bool,
    /// The function's name (`fn.name`); empty for anonymous functions.
    name: &'a str,
    /// The class this is a method of (for `super.method()`), if any.
    home_class: Option<u32>,
    /// Whether the home is entered as a *static* method, so `super.x` resolves
    /// against the superclass's static members rather than its prototype's.
    home_static: bool,
}

/// A tree-walking interpreter over the performance object model.
pub struct Interp<'a> {
    realm: Realm,
    /// The current lexical scope (innermost).
    current: Scope,
    /// Function-AST table; a closure cell holds an index into this.
    functions: Vec<FnDef<'a>>,
    /// Class-AST table; a class cell holds an index into this.
    classes: Vec<&'a Class>,
    /// Per-class static members (`Class.foo`), parallel to `classes`.
    class_statics: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class static *field* names in declaration order — the enumerable own
    /// keys of the constructor (static methods are non-enumerable), for
    /// `Object.keys`/`values`/`entries` of a class.
    class_static_fields: Vec<Vec<String>>,
    /// Per-class static getter functions (`static get x() {}`), called on read.
    class_static_get: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class static setter functions (`static set x(v) {}`), called on write.
    class_static_set: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class captured definition scope, parallel to `classes`.
    class_envs: Vec<Scope>,
    /// Per-class native-constructor superclass id (`class X extends Error`),
    /// parallel to `classes`; `None` when the parent is a class or absent.
    class_native_super: Vec<Option<u16>>,
    /// Current function-call nesting depth (recursion guard).
    call_depth: usize,
    /// xorshift PRNG state backing `Math.random` (pure Rust, no foreign code).
    rng_state: u64,
    /// The current `this` binding (method/constructor receiver).
    this_val: NanBox,
    /// `new.target` for the current invocation (the constructor when reached via
    /// `new`, else `undefined`; arrows inherit the enclosing value).
    new_target: NanBox,
    /// One-shot: a pending `new.target` set by `construct`, consumed by the next
    /// non-arrow invocation (so `new.target` is the constructor inside it).
    pending_new_target: Option<NanBox>,
    /// One-shot `newTarget` override for the next `construct` (set by
    /// `Reflect.construct(target, args, newTarget)`); else `new.target` is the callee.
    reflect_new_target: Option<NanBox>,
    /// Persistent mutable state (memory/globals) of each live WASM instance, keyed
    /// by an instance id stored on the instance's export wrappers — so a
    /// `WebAssembly.Instance`'s memory and globals survive across export calls.
    wasm_states: alloc::collections::BTreeMap<u32, crate::wasm_rt::InstanceState>,
    /// Next WASM-instance id to hand out.
    wasm_next_id: u32,
    /// When running a generator body eagerly, the buffer `yield` appends to.
    gen_sink: Option<Vec<NanBox>>,
    /// The `Symbol.for` global registry: shared symbols keyed by string.
    symbol_registry: alloc::collections::BTreeMap<String, NanBox>,
    /// Cached well-known symbols (e.g. `Symbol.iterator`), created on first use.
    well_known_symbols: alloc::collections::BTreeMap<&'static str, NanBox>,
    /// The frozen template-strings object for each tagged-template site (keyed by the
    /// AST node's address), so the same array is passed to the tag on every evaluation.
    tagged_template_cache: alloc::collections::BTreeMap<usize, NanBox>,
    /// The superclass to invoke for `super(...)` inside the running constructor.
    pending_super: Option<(u32, Scope)>,
    /// The native-constructor superclass for `super(...)` (e.g. extending Error).
    pending_super_native: Option<u16>,
    /// The class of the currently-running method (for `super.method()`).
    current_home: Option<u32>,
    /// The `[[HomeObject]]` of the currently-running object-literal method — the
    /// object the method was defined on — so its `super.x` resolves through that
    /// object's prototype (when there is no enclosing class home).
    current_home_object: Option<Handle>,
    /// Whether the currently-running method was entered as a static method, so
    /// `super.x` resolves against the superclass's static members.
    current_home_static: bool,
    /// A label attached to the next loop (for `break`/`continue label`).
    pending_label: Option<String>,
    /// The promise-reaction microtask queue, drained after the script.
    microtasks: Vec<Job>,
    /// Pending `setTimeout` callbacks (macrotasks), run after the microtask queue
    /// drains — ordered by `delay`, then insertion (`seq`). No real clock: delays
    /// only order callbacks relative to each other.
    macrotasks: Vec<Timer>,
    /// Monotonic id handed out by `setTimeout` (for `clearTimeout`).
    timer_next_id: u64,
    /// Monotonic insertion counter breaking equal-`delay` ties.
    timer_seq: u64,
    /// Whether the currently-executing code is in strict mode (`"use strict"`),
    /// which propagates into nested functions.
    strict: bool,
    /// The global object (`globalThis`) — substituted for an `undefined`/`null`
    /// `this` when calling a non-strict function.
    global_this: NanBox,
    /// Captured `console.log` output (a line per call).
    output: String,
}

/// A queued promise reaction: run `handler` with `value`, then settle `result`
/// with the outcome (or pass `value` through with the source status when
/// `handler` is `undefined`).
struct Job {
    handler: NanBox,
    value: NanBox,
    result: Handle,
    fulfilled: bool,
    /// A `finally` job: run the handler for side effects, then pass the original
    /// value/rejection through to `result`.
    finally: bool,
}

/// A pending `setTimeout` callback.
struct Timer {
    id: u64,
    delay: f64,
    seq: u64,
    callback: NanBox,
    args: Vec<NanBox>,
}

impl Default for Interp<'_> {
    fn default() -> Self {
        Self::new()
    }
}

// Built-in (native) function ids.
const N_MATH_MAX: u16 = 0;
const N_MATH_MIN: u16 = 1;
const N_MATH_ABS: u16 = 2;
const N_STRING: u16 = 3;
const N_NUMBER: u16 = 4;
const N_BOOLEAN: u16 = 5;
const N_PARSE_INT: u16 = 6;
const N_CONSOLE_LOG: u16 = 7;
const N_JSON_STRINGIFY: u16 = 8;
const N_JSON_PARSE: u16 = 27;
const N_OBJECT_KEYS: u16 = 9;
const N_OBJECT_VALUES: u16 = 10;
const N_ARRAY_IS_ARRAY: u16 = 11;
const N_MATH_FLOOR: u16 = 12;
const N_MATH_CEIL: u16 = 13;
const N_MATH_ROUND: u16 = 14;
const N_MATH_SQRT: u16 = 15;
const N_MAP: u16 = 16;
const N_SET: u16 = 17;
const N_OBJECT_ASSIGN: u16 = 18;
const N_OBJECT_ENTRIES: u16 = 19;
const N_ARRAY_FROM: u16 = 20;
const N_ARRAY_OF: u16 = 21;
const N_PROMISE: u16 = 22;
const N_DATE: u16 = 25;
const N_REGEXP: u16 = 26;
const N_MATH_POW: u16 = 28;
const N_MATH_SIGN: u16 = 29;
const N_MATH_TRUNC: u16 = 30;
const N_OBJECT_FROM_ENTRIES: u16 = 34;
const N_OBJECT_FREEZE: u16 = 35;
const N_OBJECT_IS_FROZEN: u16 = 36;
const N_OBJECT_SEAL: u16 = 125;
const N_OBJECT_IS_SEALED: u16 = 126;
const N_OBJECT_PREVENT_EXT: u16 = 127;
const N_OBJECT_IS_EXTENSIBLE: u16 = 128;
const N_OBJECT_GET_OWN_NAMES: u16 = 37;
const N_OBJECT_GET_OWN_SYMBOLS: u16 = 158;
const N_ENCODE_URI_COMPONENT: u16 = 159;
const N_DECODE_URI_COMPONENT: u16 = 160;
const N_ENCODE_URI: u16 = 161;
const N_DECODE_URI: u16 = 162;
const N_STRUCTURED_CLONE: u16 = 163;
const N_BTOA: u16 = 164;
const N_ATOB: u16 = 165;
const N_INTL_NUMBER_FORMAT: u16 = 166;
const N_INTL_DATETIME_FORMAT: u16 = 167;
const N_SET_TIMEOUT: u16 = 211;
const N_CLEAR_TIMEOUT: u16 = 212;
const N_QUEUE_MICROTASK: u16 = 213;
const N_ARRAY_BUFFER_IS_VIEW: u16 = 214;
const N_INTL_FORMAT: u16 = 215;
const N_INTL_COLLATOR: u16 = 207;
const N_INTL_PLURAL_RULES: u16 = 208;
/// `Intl.Collator.prototype.compare` (a bound function value).
const N_INTL_COMPARE: u16 = 209;
/// `Intl.PluralRules.prototype.select`.
const N_INTL_PLURAL_SELECT: u16 = 210;
/// The typed-array constructors occupy `[BASE, BASE + KINDS.len())`; the id minus
/// the base indexes [`TYPED_ARRAY_KINDS`].
const N_TYPED_ARRAY_BASE: u16 = 168;
/// `(name, bytes-per-element)` for each typed-array kind, in id order.
const TYPED_ARRAY_KINDS: [(&str, u8); 9] = [
    ("Int8Array", 1),
    ("Uint8Array", 1),
    ("Uint8ClampedArray", 1),
    ("Int16Array", 2),
    ("Uint16Array", 2),
    ("Int32Array", 4),
    ("Uint32Array", 4),
    ("Float32Array", 4),
    ("Float64Array", 8),
];
const N_ARRAY_BUFFER: u16 = 177;
const N_DATA_VIEW: u16 = 178;
// `WebAssembly.validate` — decode a module and report well-formedness.
const N_WASM_VALIDATE: u16 = 184;
// `WebAssembly.instantiate` — build an instance object with callable exports.
const N_WASM_INSTANTIATE: u16 = 185;
// A WASM export wrapper (a bound native whose target carries the module bytes +
// the export name).
const N_WASM_CALL: u16 = 186;
/// The `Function` global — supports `typeof`/`instanceof` (any callable); the
/// dynamic `Function(...)` constructor (runtime code compilation) is unsupported.
const N_FUNCTION: u16 = 187;
/// `new WebAssembly.Module(bytes)` — a decoded/validated module object.
const N_WASM_MODULE: u16 = 188;
/// `new WebAssembly.Instance(module, imports?)` — an instance with `.exports`.
const N_WASM_INSTANCE: u16 = 189;
/// `WebAssembly.compile(bytes)` — async compile → `Promise<Module>`.
const N_WASM_COMPILE: u16 = 190;
/// `new WebAssembly.Global({value, mutable}, init)` — a typed value cell.
const N_WASM_GLOBAL: u16 = 191;
/// The `.value` getter / setter of a `WebAssembly.Global` (bound to the global).
const N_WASM_GLOBAL_GET: u16 = 192;
const N_WASM_GLOBAL_SET: u16 = 193;
/// `new WebAssembly.Memory({initial, maximum?})` and its `.buffer` getter / `grow`.
const N_WASM_MEMORY: u16 = 194;
const N_WASM_MEM_BUFFER_GET: u16 = 195;
const N_WASM_MEM_GROW: u16 = 196;
/// A WASM linear-memory page is 64 KiB.
const WASM_PAGE: usize = 65536;
/// `new WebAssembly.Table({element, initial, maximum?})` and its `.length` getter
/// plus `get`/`set`/`grow` methods.
const N_WASM_TABLE: u16 = 197;
const N_WASM_TABLE_LEN: u16 = 198;
const N_WASM_TABLE_GET: u16 = 199;
const N_WASM_TABLE_SET: u16 = 200;
const N_WASM_TABLE_GROW: u16 = 201;
/// Static `WebAssembly.Module.exports(module)` / `.imports(module)` introspection.
const N_WASM_MODULE_EXPORTS: u16 = 202;
const N_WASM_MODULE_IMPORTS: u16 = 203;
// Hidden slots on a WASM export wrapper's data object.
const WASM_BYTES: &str = "\u{0}wbytes";
const WASM_EXPORT: &str = "\u{0}wexport";
const WASM_IMPORTS: &str = "\u{0}wimports";
/// Marks an object built by `new WebAssembly.Module(...)`.
const WASM_IS_MODULE: &str = "\u{0}wmodule";
/// The instance id on a WASM export wrapper's data object (keys `wasm_states`, so
/// memory/globals persist across calls of the same instance).
const WASM_INSTANCE_ID: &str = "\u{0}winst";
/// Hidden slots on a `WebAssembly.Global`: its current value, value type, and
/// mutability.
const WASM_GLOBAL_VALUE: &str = "\u{0}gval";
const WASM_GLOBAL_TYPE: &str = "\u{0}gtype";
const WASM_GLOBAL_MUTABLE: &str = "\u{0}gmut";
/// Hidden slots on a `WebAssembly.Memory`: its `ArrayBuffer`, page count, and max.
const WASM_MEM_BUFFER: &str = "\u{0}mbuf";
const WASM_MEM_PAGES: &str = "\u{0}mpages";
const WASM_MEM_MAX: &str = "\u{0}mmax";
/// Hidden slots on a `WebAssembly.Table`: its element (function-ref) array and max.
const WASM_TABLE_ELEMS: &str = "\u{0}telems";
const WASM_TABLE_MAX: &str = "\u{0}tmax";
// `Object.prototype.*` methods (the receiver arrives as `this`).
const N_OBJ_PROTO_TOSTRING: u16 = 179;
const N_OBJ_PROTO_VALUEOF: u16 = 180;
const N_OBJ_PROTO_HASOWN: u16 = 181;
const N_OBJ_PROTO_ISPROTOTYPEOF: u16 = 182;
const N_OBJ_PROTO_PROPISENUM: u16 = 183;
const N_OBJECT_CREATE: u16 = 107;
const N_OBJECT_GET_PROTO: u16 = 108;
const N_OBJECT_SET_PROTO: u16 = 109;
const N_OBJECT_DEFINE_PROP: u16 = 110;
const N_OBJECT_GET_OWN_DESC: u16 = 111;
const N_WEAKMAP: u16 = 112;
const N_OBJECT_IS: u16 = 123;
const N_OBJECT_HAS_OWN: u16 = 129;
const N_OBJECT_GROUP_BY: u16 = 138;
const N_OBJECT_GET_OWN_DESCS: u16 = 130;
const N_WEAKREF: u16 = 131;
const N_FINALIZATION_REGISTRY: u16 = 132;
const N_OBJECT_DEFINE_PROPS: u16 = 124;
const N_WEAKSET: u16 = 113;
const N_REFLECT_GET: u16 = 114;
const N_REFLECT_SET: u16 = 115;
const N_REFLECT_HAS: u16 = 116;
const N_REFLECT_OWN_KEYS: u16 = 117;
const N_REFLECT_DELETE: u16 = 118;
const N_REFLECT_APPLY: u16 = 119;
const N_REFLECT_CONSTRUCT: u16 = 120;
const N_REFLECT_DEFINE_PROP: u16 = 135;
const N_REFLECT_GET_OWN_DESC: u16 = 136;
const N_REFLECT_GET_PROTO: u16 = 137;
/// `Reflect.setPrototypeOf` / `preventExtensions` — like the `Object.*` forms but
/// returning a boolean success flag.
const N_REFLECT_SET_PROTO: u16 = 204;
const N_REFLECT_PREVENT_EXT: u16 = 205;
/// A first-class `Array.prototype.<method>` value: a bound native carrying the
/// method name; calling it (via `.call`/`.apply`) dispatches that array method on
/// the supplied `this` (so `Array.prototype.slice.call(arguments)` works).
const N_ARRAY_PROTO_FN: u16 = 206;
/// The `Array.prototype` methods exposed as first-class values (each re-dispatched
/// through `call_method`).
const ARRAY_PROTO_METHODS: &[&str] = &[
    "slice",
    "splice",
    "map",
    "filter",
    "forEach",
    "reduce",
    "reduceRight",
    "indexOf",
    "lastIndexOf",
    "includes",
    "find",
    "findIndex",
    "findLast",
    "findLastIndex",
    "some",
    "every",
    "join",
    "concat",
    "reverse",
    "sort",
    "fill",
    "copyWithin",
    "flat",
    "flatMap",
    "at",
    "push",
    "pop",
    "shift",
    "unshift",
    "keys",
    "values",
    "entries",
    "toString",
    "with",
    "toReversed",
    "toSorted",
    "toSpliced",
];
/// `String.prototype` methods exposed as first-class values.
const STRING_PROTO_METHODS: &[&str] = &[
    "slice",
    "substring",
    "substr",
    "charAt",
    "charCodeAt",
    "codePointAt",
    "indexOf",
    "lastIndexOf",
    "includes",
    "startsWith",
    "endsWith",
    "split",
    "replace",
    "replaceAll",
    "match",
    "matchAll",
    "search",
    "toUpperCase",
    "toLowerCase",
    "trim",
    "trimStart",
    "trimEnd",
    "padStart",
    "padEnd",
    "repeat",
    "concat",
    "at",
    "normalize",
    "localeCompare",
    "toString",
    "valueOf",
    "isWellFormed",
    "toWellFormed",
];
/// `Number.prototype` methods exposed as first-class values.
const NUMBER_PROTO_METHODS: &[&str] = &[
    "toFixed",
    "toPrecision",
    "toExponential",
    "toString",
    "valueOf",
    "toLocaleString",
];
/// `Boolean.prototype` methods exposed as first-class values.
const BOOLEAN_PROTO_METHODS: &[&str] = &["toString", "valueOf"];
/// `Set.prototype` methods exposed as first-class values.
const SET_PROTO_METHODS: &[&str] = &[
    "add",
    "has",
    "delete",
    "clear",
    "forEach",
    "keys",
    "values",
    "entries",
    "union",
    "intersection",
    "difference",
    "symmetricDifference",
    "isSubsetOf",
    "isSupersetOf",
    "isDisjointFrom",
];
/// `Map.prototype` methods exposed as first-class values.
const MAP_PROTO_METHODS: &[&str] = &[
    "set", "get", "has", "delete", "clear", "forEach", "keys", "values", "entries",
];
/// `Function.prototype` methods exposed as first-class values.
const FUNCTION_PROTO_METHODS: &[&str] = &["call", "apply", "bind", "toString"];
/// Bound native: the `revoke` function from `Proxy.revocable` (carries the proxy).
const N_PROXY_REVOKE: u16 = 122;
const N_SYMBOL: u16 = 38;
const N_BIGINT: u16 = 39;
const N_PROXY: u16 = 106;
const N_PARSE_FLOAT: u16 = 31;
const N_IS_NAN: u16 = 32;
const N_IS_FINITE: u16 = 33;
// Error constructors (id − N_ERROR_BASE indexes ERROR_NAMES).
const N_ERROR_BASE: u16 = 40;
/// Abbreviated weekday names (index 0 = Sunday), for `Date` string methods.
const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
/// Abbreviated month names (index 0 = January).
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const ERROR_NAMES: [&str; 11] = [
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
    "AggregateError",
    // The `WebAssembly.*` error subclasses (exposed under the WebAssembly
    // namespace, not as globals — see the registration below).
    "CompileError",
    "LinkError",
    "RuntimeError",
    // Standard global error subclasses (registered as globals below).
    "URIError",
    "EvalError",
];
/// Count of `Error` subclasses exposed as JS globals (`Error`…`AggregateError`);
/// the `WebAssembly.*` ones are namespaced, and `URIError`/`EvalError` are
/// registered separately as globals.
const N_GLOBAL_ERROR_COUNT: usize = 6;
const N_TYPE_ERROR: u16 = N_ERROR_BASE + 1;
const N_RANGE_ERROR: u16 = N_ERROR_BASE + 2;
const N_REFERENCE_ERROR: u16 = N_ERROR_BASE + 4;
const N_WASM_COMPILE_ERROR: u16 = N_ERROR_BASE + 6;
const N_WASM_LINK_ERROR: u16 = N_ERROR_BASE + 7;
const N_WASM_RUNTIME_ERROR: u16 = N_ERROR_BASE + 8;
const N_URI_ERROR: u16 = N_ERROR_BASE + 9;
const N_EVAL_ERROR: u16 = N_ERROR_BASE + 10;
/// A reserved, non-identifier key under which a `new fn()` instance records its
/// constructor function (a hidden, GC-traced slot) so `instanceof` can match it.
const CTOR_KEY: &str = "\u{0}ctor";
/// Hidden slot on an object-literal concise method recording its `[[HomeObject]]`
/// (the object it was defined on), for `super` resolution.
const HOME_OBJECT: &str = "\u{0}home";
/// Reserved hidden keys for an eager generator's result object: the buffer of
/// yielded values and the current `next()` cursor.
/// Sentinel description for a `Symbol()` created with no argument (so its
/// `.description` is `undefined`, distinct from `Symbol("")`).
/// Maximum function-call nesting before a `RangeError` (recursion guard). Sized
/// to fire before the large stack the engine entry points run on overflows (the
/// tree-walker uses several tens of KB of host stack per JS call).
const MAX_CALL_DEPTH: usize = 3500;
const SYMBOL_NO_DESC: &str = "\u{0}nodesc";
/// Hidden key holding a `WeakRef`'s target (returned by `deref`).
const WEAKREF_TARGET: &str = "\u{0}wrtarget";
/// Hidden marker tagging a `FinalizationRegistry` instance.
const FINREG_TAG: &str = "\u{0}finreg";
const GEN_BUF: &str = "\u{0}gbuf";
const GEN_IDX: &str = "\u{0}gidx";
/// A generator's `return` value, surfaced once after its yields are exhausted.
const GEN_RET: &str = "\u{0}gret";
/// Reserved hidden keys for a bound function (`Function.prototype.bind`).
/// Hidden slot holding a primitive-wrapper object's boxed value, and its
/// constructor id (for `instanceof`).
const PRIM_WRAP: &str = "\u{0}prim";
const PRIM_WRAP_TYPE: &str = "\u{0}primtype";
/// Hidden slot on a typed array recording its element-kind index.
const TYPED_ARRAY_KIND: &str = "\u{0}takind";
/// `ArrayBuffer` byte store (an array of 0–255 numbers) and `DataView` linkage.
const ARRAY_BUFFER_BYTES: &str = "\u{0}abytes";
const DATA_VIEW_BUF: &str = "\u{0}dvbuf";
const DATA_VIEW_OFF: &str = "\u{0}dvoff";
/// An explicit `DataView` byteLength (the 3rd constructor arg); absent → the rest
/// of the buffer from the offset.
const DATA_VIEW_LEN: &str = "\u{0}dvlen";
const BOUND_TARGET: &str = "\u{0}bnd_t";
const BOUND_THIS: &str = "\u{0}bnd_this";
const BOUND_ARGS: &str = "\u{0}bnd_args";
/// A safety cap on eagerly-collected `yield`s (an infinite generator would
/// otherwise hang); exceeding it throws instead.
const GEN_CAP: usize = 1_000_000;
// Bound natives (carry a target promise handle):
const N_RESOLVE: u16 = 100;
const N_REJECT: u16 = 101;
const N_MATH_HYPOT: u16 = 102;
const N_MATH_CBRT: u16 = 103;
const N_MATH_LOG2: u16 = 104;
const N_MATH_LOG10: u16 = 105;
const N_MATH_EXP: u16 = 133;
const N_MATH_LOG: u16 = 134;
const N_MATH_RANDOM: u16 = 139;
// Trig / hyperbolic / extra Math functions (140..=157).
const N_MATH_SIN: u16 = 140;
const N_MATH_COS: u16 = 141;
const N_MATH_TAN: u16 = 142;
const N_MATH_ASIN: u16 = 143;
const N_MATH_ACOS: u16 = 144;
const N_MATH_ATAN: u16 = 145;
const N_MATH_ATAN2: u16 = 146;
const N_MATH_SINH: u16 = 147;
const N_MATH_COSH: u16 = 148;
const N_MATH_TANH: u16 = 149;
const N_MATH_ASINH: u16 = 150;
const N_MATH_ACOSH: u16 = 151;
const N_MATH_ATANH: u16 = 152;
const N_MATH_EXPM1: u16 = 153;
const N_MATH_LOG1P: u16 = 154;
const N_MATH_FROUND: u16 = 155;
const N_MATH_CLZ32: u16 = 156;
const N_MATH_IMUL: u16 = 157;

impl<'a> Interp<'a> {
    /// A fresh interpreter with a single (global) scope and a starter stdlib.
    #[must_use]
    pub fn new() -> Self {
        let mut interp = Self {
            realm: Realm::new(),
            current: Scope::root(),
            functions: Vec::new(),
            classes: Vec::new(),
            class_statics: Vec::new(),
            class_static_fields: Vec::new(),
            class_static_get: Vec::new(),
            class_static_set: Vec::new(),
            class_envs: Vec::new(),
            class_native_super: Vec::new(),
            call_depth: 0,
            // A fixed non-zero seed (deterministic, but advances per call).
            rng_state: 0x9E37_79B9_7F4A_7C15,
            this_val: NanBox::undefined(),
            new_target: NanBox::undefined(),
            pending_new_target: None,
            reflect_new_target: None,
            wasm_states: alloc::collections::BTreeMap::new(),
            wasm_next_id: 0,
            gen_sink: None,
            symbol_registry: alloc::collections::BTreeMap::new(),
            well_known_symbols: alloc::collections::BTreeMap::new(),
            tagged_template_cache: alloc::collections::BTreeMap::new(),
            pending_super: None,
            pending_super_native: None,
            current_home: None,
            current_home_object: None,
            current_home_static: false,
            pending_label: None,
            microtasks: Vec::new(),
            macrotasks: Vec::new(),
            timer_next_id: 1,
            timer_seq: 0,
            strict: false,
            global_this: NanBox::undefined(),
            output: String::new(),
        };
        interp.install_globals();
        interp
    }

    /// The accumulated `console.log` output.
    #[must_use]
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Renders a result value as a display string (for surfacing a completion
    /// value to a caller / REPL).
    #[must_use]
    pub fn display(&self, value: NanBox) -> String {
        self.realm.to_display_string(value)
    }

    /// Creates a native function carrying its own (non-enumerable) `name`, per the
    /// spec's named built-ins (`Math.max.name === "max"`).
    fn new_named_native(&mut self, name: &str, id: u16) -> Handle {
        let f = self.realm.new_native(id);
        let name_v = self.new_str(name);
        self.realm.set_property(f, "name", name_v);
        self.realm.mark_hidden(f, "name");
        f
    }

    /// Builds `<ctor>.prototype` as a real object whose `methods` are first-class
    /// values — each a bound native re-dispatching that method on the call's `this`
    /// — so `Ctor.prototype.method.call(thisArg, …)` works. Methods are
    /// non-enumerable; `proto.constructor` links back to the constructor.
    fn setup_first_class_prototype(&mut self, ctor_name: &str, methods: &[&str]) {
        let Some(ns) = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        else {
            return;
        };
        let proto = self.realm.new_object();
        for &name in methods {
            let name_h = self.realm.new_string(name);
            let f = self.realm.new_bound_native(N_ARRAY_PROTO_FN, name_h);
            self.realm
                .set_property(proto, name, NanBox::handle(f.to_raw()));
            self.realm.mark_hidden(proto, name);
        }
        self.realm
            .set_property(ns, "prototype", NanBox::handle(proto.to_raw()));
        self.realm
            .set_hidden_property(proto, "constructor", NanBox::handle(ns.to_raw()));
    }

    /// Installs a small built-in library: the `Math` object and the global
    /// coercion/parse functions. (A token stdlib to prove the native-call path;
    /// the full port is the remaining migration work.)
    fn install_globals(&mut self) {
        // An object whose properties are native methods, bound to `global_name`.
        let install_namespace = |this: &mut Self, global_name: &str, methods: &[(&str, u16)]| {
            let obj = this.realm.new_object();
            for (name, id) in methods {
                let f = this.new_named_native(name, *id);
                this.realm
                    .set_property(obj, name, NanBox::handle(f.to_raw()));
            }
            this.current
                .declare(global_name, NanBox::handle(obj.to_raw()));
        };
        install_namespace(
            self,
            "Math",
            &[
                ("max", N_MATH_MAX),
                ("min", N_MATH_MIN),
                ("abs", N_MATH_ABS),
                ("floor", N_MATH_FLOOR),
                ("ceil", N_MATH_CEIL),
                ("round", N_MATH_ROUND),
                ("sqrt", N_MATH_SQRT),
                ("pow", N_MATH_POW),
                ("sign", N_MATH_SIGN),
                ("hypot", N_MATH_HYPOT),
                ("cbrt", N_MATH_CBRT),
                ("log2", N_MATH_LOG2),
                ("log10", N_MATH_LOG10),
                ("exp", N_MATH_EXP),
                ("log", N_MATH_LOG),
                ("random", N_MATH_RANDOM),
                ("trunc", N_MATH_TRUNC),
                ("sin", N_MATH_SIN),
                ("cos", N_MATH_COS),
                ("tan", N_MATH_TAN),
                ("asin", N_MATH_ASIN),
                ("acos", N_MATH_ACOS),
                ("atan", N_MATH_ATAN),
                ("atan2", N_MATH_ATAN2),
                ("sinh", N_MATH_SINH),
                ("cosh", N_MATH_COSH),
                ("tanh", N_MATH_TANH),
                ("asinh", N_MATH_ASINH),
                ("acosh", N_MATH_ACOSH),
                ("atanh", N_MATH_ATANH),
                ("expm1", N_MATH_EXPM1),
                ("log1p", N_MATH_LOG1P),
                ("fround", N_MATH_FROUND),
                ("clz32", N_MATH_CLZ32),
                ("imul", N_MATH_IMUL),
            ],
        );
        // The `Math` numeric constants.
        if let Some(mh) = self.current.get("Math").and_then(NanBox::as_handle) {
            let math = Handle::from_raw(mh);
            for (name, value) in [
                ("PI", core::f64::consts::PI),
                ("E", core::f64::consts::E),
                ("LN2", core::f64::consts::LN_2),
                ("LN10", core::f64::consts::LN_10),
                ("LOG2E", core::f64::consts::LOG2_E),
                ("LOG10E", core::f64::consts::LOG10_E),
                ("SQRT2", core::f64::consts::SQRT_2),
                ("SQRT1_2", core::f64::consts::FRAC_1_SQRT_2),
            ] {
                self.realm.set_property(math, name, NanBox::number(value));
            }
        }
        install_namespace(self, "console", &[("log", N_CONSOLE_LOG)]);
        // `Promise` is a native constructor (`new Promise(executor)`); its
        // `.resolve`/`.reject` statics are dispatched in `call_method`.
        let promise_ctor = self.new_named_native("Promise", N_PROMISE);
        self.current
            .declare("Promise", NanBox::handle(promise_ctor.to_raw()));
        // `Date` is a native constructor; `Date.now()` is a static.
        let date_ctor = self.new_named_native("Date", N_DATE);
        self.current
            .declare("Date", NanBox::handle(date_ctor.to_raw()));
        // `RegExp` is a native constructor.
        let regexp_ctor = self.new_named_native("RegExp", N_REGEXP);
        self.current
            .declare("RegExp", NanBox::handle(regexp_ctor.to_raw()));
        // The `Error` family — native constructors producing `{ name, message }`.
        // Only the standard errors are globals; the `WebAssembly.*` error
        // subclasses are installed under the WebAssembly namespace below.
        for (i, name) in ERROR_NAMES.iter().enumerate().take(N_GLOBAL_ERROR_COUNT) {
            let ctor = self.new_named_native(name, N_ERROR_BASE + i as u16);
            self.current.declare(name, NanBox::handle(ctor.to_raw()));
        }
        install_namespace(
            self,
            "JSON",
            &[("stringify", N_JSON_STRINGIFY), ("parse", N_JSON_PARSE)],
        );
        install_namespace(
            self,
            "Object",
            &[
                ("keys", N_OBJECT_KEYS),
                ("values", N_OBJECT_VALUES),
                ("assign", N_OBJECT_ASSIGN),
                ("entries", N_OBJECT_ENTRIES),
                ("fromEntries", N_OBJECT_FROM_ENTRIES),
                ("freeze", N_OBJECT_FREEZE),
                ("isFrozen", N_OBJECT_IS_FROZEN),
                ("seal", N_OBJECT_SEAL),
                ("isSealed", N_OBJECT_IS_SEALED),
                ("preventExtensions", N_OBJECT_PREVENT_EXT),
                ("isExtensible", N_OBJECT_IS_EXTENSIBLE),
                ("getOwnPropertyNames", N_OBJECT_GET_OWN_NAMES),
                ("getOwnPropertySymbols", N_OBJECT_GET_OWN_SYMBOLS),
                ("create", N_OBJECT_CREATE),
                ("getPrototypeOf", N_OBJECT_GET_PROTO),
                ("setPrototypeOf", N_OBJECT_SET_PROTO),
                ("defineProperty", N_OBJECT_DEFINE_PROP),
                ("defineProperties", N_OBJECT_DEFINE_PROPS),
                ("getOwnPropertyDescriptor", N_OBJECT_GET_OWN_DESC),
                ("getOwnPropertyDescriptors", N_OBJECT_GET_OWN_DESCS),
                ("is", N_OBJECT_IS),
                ("hasOwn", N_OBJECT_HAS_OWN),
                ("groupBy", N_OBJECT_GROUP_BY),
            ],
        );
        install_namespace(
            self,
            "Array",
            &[
                ("isArray", N_ARRAY_IS_ARRAY),
                ("from", N_ARRAY_FROM),
                ("of", N_ARRAY_OF),
            ],
        );
        install_namespace(
            self,
            "Reflect",
            &[
                ("get", N_REFLECT_GET),
                ("set", N_REFLECT_SET),
                ("has", N_REFLECT_HAS),
                ("ownKeys", N_REFLECT_OWN_KEYS),
                ("defineProperty", N_REFLECT_DEFINE_PROP),
                ("getOwnPropertyDescriptor", N_REFLECT_GET_OWN_DESC),
                ("getPrototypeOf", N_REFLECT_GET_PROTO),
                ("setPrototypeOf", N_REFLECT_SET_PROTO),
                ("deleteProperty", N_REFLECT_DELETE),
                ("apply", N_REFLECT_APPLY),
                ("construct", N_REFLECT_CONSTRUCT),
                ("isExtensible", N_OBJECT_IS_EXTENSIBLE),
                ("preventExtensions", N_REFLECT_PREVENT_EXT),
            ],
        );
        for (name, id) in [
            ("String", N_STRING),
            ("Number", N_NUMBER),
            ("Boolean", N_BOOLEAN),
            ("parseInt", N_PARSE_INT),
            ("parseFloat", N_PARSE_FLOAT),
            ("isNaN", N_IS_NAN),
            ("isFinite", N_IS_FINITE),
            ("Map", N_MAP),
            ("Set", N_SET),
            ("Symbol", N_SYMBOL),
            ("BigInt", N_BIGINT),
            ("Function", N_FUNCTION),
            ("Proxy", N_PROXY),
            ("WeakMap", N_WEAKMAP),
            ("WeakSet", N_WEAKSET),
            ("WeakRef", N_WEAKREF),
            ("FinalizationRegistry", N_FINALIZATION_REGISTRY),
            ("encodeURIComponent", N_ENCODE_URI_COMPONENT),
            ("decodeURIComponent", N_DECODE_URI_COMPONENT),
            ("encodeURI", N_ENCODE_URI),
            ("decodeURI", N_DECODE_URI),
            ("structuredClone", N_STRUCTURED_CLONE),
            ("setTimeout", N_SET_TIMEOUT),
            ("clearTimeout", N_CLEAR_TIMEOUT),
            ("queueMicrotask", N_QUEUE_MICROTASK),
            ("btoa", N_BTOA),
            ("atob", N_ATOB),
            ("URIError", N_URI_ERROR),
            ("EvalError", N_EVAL_ERROR),
        ] {
            let f = self.new_named_native(name, id);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        // The `Intl` namespace with its format constructors.
        let intl = self.realm.new_object();
        for (name, id) in [
            ("NumberFormat", N_INTL_NUMBER_FORMAT),
            ("DateTimeFormat", N_INTL_DATETIME_FORMAT),
            ("Collator", N_INTL_COLLATOR),
            ("PluralRules", N_INTL_PLURAL_RULES),
        ] {
            let f = self.new_named_native(name, id);
            self.realm
                .set_property(intl, name, NanBox::handle(f.to_raw()));
        }
        self.current.declare("Intl", NanBox::handle(intl.to_raw()));
        // The typed-array constructors.
        for (i, (name, _)) in TYPED_ARRAY_KINDS.iter().enumerate() {
            let f = self.new_named_native(name, N_TYPED_ARRAY_BASE + i as u16);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        for (name, id) in [("ArrayBuffer", N_ARRAY_BUFFER), ("DataView", N_DATA_VIEW)] {
            let f = self.realm.new_native(id);
            // `ArrayBuffer.isView(x)` — true for a typed array or a DataView.
            if id == N_ARRAY_BUFFER {
                let isview = self.realm.new_native(N_ARRAY_BUFFER_IS_VIEW);
                self.realm
                    .set_hidden_property(f, "isView", NanBox::handle(isview.to_raw()));
            }
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
        // The `WebAssembly` namespace, backed by the in-house WASM engine
        // (`wasm_rt`). `validate(bytes)` decodes a module and reports whether it
        // is well-formed.
        install_namespace(
            self,
            "WebAssembly",
            &[
                ("validate", N_WASM_VALIDATE),
                ("instantiate", N_WASM_INSTANTIATE),
                ("Module", N_WASM_MODULE),
                ("Instance", N_WASM_INSTANCE),
                ("compile", N_WASM_COMPILE),
                ("Global", N_WASM_GLOBAL),
                ("Memory", N_WASM_MEMORY),
                ("Table", N_WASM_TABLE),
                ("CompileError", N_WASM_COMPILE_ERROR),
                ("LinkError", N_WASM_LINK_ERROR),
                ("RuntimeError", N_WASM_RUNTIME_ERROR),
            ],
        );
        // Static introspection methods on `WebAssembly.Module`.
        if let Some(module_ctor) = self
            .current
            .get("WebAssembly")
            .and_then(|ns| ns.as_handle())
            .map(Handle::from_raw)
            .and_then(|ns| self.realm.get_property(ns, "Module"))
            .and_then(|m| m.as_handle())
            .map(Handle::from_raw)
        {
            for (name, id) in [
                ("exports", N_WASM_MODULE_EXPORTS),
                ("imports", N_WASM_MODULE_IMPORTS),
            ] {
                let f = self.realm.new_native(id);
                self.realm
                    .set_property(module_ctor, name, NanBox::handle(f.to_raw()));
            }
        }
        // A minimal `Object.prototype` carrying the methods commonly invoked via
        // `Object.prototype.<m>.call(x)`. The receiver arrives as `this`.
        let obj_proto = self.realm.new_object();
        for (name, id) in [
            ("toString", N_OBJ_PROTO_TOSTRING),
            ("toLocaleString", N_OBJ_PROTO_TOSTRING),
            ("valueOf", N_OBJ_PROTO_VALUEOF),
            ("hasOwnProperty", N_OBJ_PROTO_HASOWN),
            ("isPrototypeOf", N_OBJ_PROTO_ISPROTOTYPEOF),
            ("propertyIsEnumerable", N_OBJ_PROTO_PROPISENUM),
        ] {
            let f = self.realm.new_native(id);
            self.realm
                .set_property(obj_proto, name, NanBox::handle(f.to_raw()));
            // Non-enumerable, so inheriting objects don't surface them in for-in /
            // Object.keys.
            self.realm.mark_hidden(obj_proto, name);
        }
        if let Some(obj_ns) = self
            .current
            .get("Object")
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            self.realm
                .set_property(obj_ns, "prototype", NanBox::handle(obj_proto.to_raw()));
            // `({}).constructor === Object` (non-enumerable, inherited via the
            // default object prototype), and `Object.name === "Object"`.
            self.realm.set_hidden_property(
                obj_proto,
                "constructor",
                NanBox::handle(obj_ns.to_raw()),
            );
            let name = self.new_str("Object");
            self.realm.set_hidden_property(obj_ns, "name", name);
        }
        // `<Ctor>.prototype` as a real object whose methods are first-class values
        // that dispatch on their `this`, so the classic `Array.prototype.slice.call`
        // / `String.prototype.X.call` / `Function.prototype.bind.call` idioms work.
        self.setup_first_class_prototype("Array", ARRAY_PROTO_METHODS);
        self.setup_first_class_prototype("String", STRING_PROTO_METHODS);
        self.setup_first_class_prototype("Number", NUMBER_PROTO_METHODS);
        self.setup_first_class_prototype("Boolean", BOOLEAN_PROTO_METHODS);
        self.setup_first_class_prototype("Function", FUNCTION_PROTO_METHODS);
        self.setup_first_class_prototype("Set", SET_PROTO_METHODS);
        self.setup_first_class_prototype("Map", MAP_PROTO_METHODS);
        // Newly-created plain objects now inherit from `Object.prototype`.
        self.realm.set_default_object_proto(obj_proto);
        // `globalThis`: an object mirroring the global bindings, referencing
        // itself. Reads like `globalThis.Math` and `globalThis.globalThis` work.
        let global = self.realm.new_object();
        for n in [
            "Math",
            "JSON",
            "Object",
            "Array",
            "Reflect",
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
            "Proxy",
            "WeakMap",
            "WeakSet",
            "WeakRef",
            "FinalizationRegistry",
            "Promise",
            "Date",
            "console",
            "Error",
            "TypeError",
            "RangeError",
            "SyntaxError",
            "ReferenceError",
            "AggregateError",
            "encodeURIComponent",
            "decodeURIComponent",
            "encodeURI",
            "decodeURI",
            "structuredClone",
            "btoa",
            "atob",
            "Intl",
        ] {
            if let Some(v) = self.current.get(n) {
                self.realm.set_property(global, n, v);
            }
        }
        for (name, _) in TYPED_ARRAY_KINDS {
            if let Some(v) = self.current.get(name) {
                self.realm.set_property(global, name, v);
            }
        }
        self.realm
            .set_property(global, "NaN", NanBox::number(f64::NAN));
        self.realm
            .set_property(global, "Infinity", NanBox::number(f64::INFINITY));
        self.realm
            .set_property(global, "undefined", NanBox::undefined());
        let gbox = NanBox::handle(global.to_raw());
        self.realm.set_property(global, "globalThis", gbox);
        self.current.declare("globalThis", gbox);
        self.global_this = gbox;
    }

    /// Invokes a built-in by id.
    fn call_native(&mut self, id: u16, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());
        Ok(match id {
            N_MATH_MAX => {
                let mut m = f64::NEG_INFINITY;
                for a in args {
                    let n = self.realm.to_number(*a);
                    if n.is_nan() {
                        return Ok(NanBox::number(f64::NAN));
                    }
                    // `+0` is treated as greater than `-0`.
                    if n > m || (n == 0.0 && m == 0.0 && n.is_sign_positive()) {
                        m = n;
                    }
                }
                NanBox::number(m)
            }
            N_MATH_MIN => {
                let mut m = f64::INFINITY;
                for a in args {
                    let n = self.realm.to_number(*a);
                    if n.is_nan() {
                        return Ok(NanBox::number(f64::NAN));
                    }
                    // `-0` is treated as less than `+0`.
                    if n < m || (n == 0.0 && m == 0.0 && n.is_sign_negative()) {
                        m = n;
                    }
                }
                NanBox::number(m)
            }
            N_MATH_ABS => NanBox::number(self.realm.to_number(arg(0)).abs()),
            N_STRING => {
                // `String(obj)` runs the object through ToString (string hint),
                // honoring a custom `toString`.
                let p = self.coerce_object(arg(0), "string")?;
                let s = self.realm.to_display_string(p);
                NanBox::handle(self.realm.new_string(&s).to_raw())
            }
            N_NUMBER => {
                // `Number(bigint)` converts to the nearest double.
                if let Some(big) = arg(0)
                    .as_handle()
                    .and_then(|r| self.realm.bigint_at(Handle::from_raw(r)))
                {
                    NanBox::number(big.to_f64())
                } else {
                    // `Number(obj)` runs the object through ToNumber (number
                    // hint), honoring a custom `valueOf`.
                    let p = self.coerce_object(arg(0), "number")?;
                    NanBox::number(self.realm.to_number(p))
                }
            }
            N_BOOLEAN => NanBox::boolean(self.realm.truthy(arg(0))),
            N_SYMBOL => {
                // A no-argument `Symbol()` has an `undefined` description, marked
                // with a reserved sentinel (distinct from `Symbol("")`).
                let desc = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                    String::from(SYMBOL_NO_DESC)
                } else {
                    self.realm.to_display_string(arg(0))
                };
                NanBox::handle(self.realm.new_symbol(&desc).to_raw())
            }
            N_BIGINT => {
                // From a number (truncated) or a numeric string.
                let v = arg(0);
                let n = if let Some(raw) = v.as_handle() {
                    let h = Handle::from_raw(raw);
                    if let Some(s) = self.realm.string_value(h) {
                        parse_bigint(s.trim())
                    } else {
                        self.realm.bigint_at(h).unwrap_or_default()
                    }
                } else {
                    // From a number: only an exact integer converts; a fractional
                    // or non-finite value is a `RangeError` (`BigInt(1.5)` throws).
                    let num = self.realm.to_number(v);
                    if !num.is_finite() || num != (num as i128) as f64 {
                        let m = self.new_str("The number is not a safe integer");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    crate::bignum::BigInt::from_i128(num as i128)
                };
                NanBox::handle(self.realm.new_bigint(n).to_raw())
            }
            N_FUNCTION => {
                // The dynamic `Function(...)` / `new Function(...)` constructor
                // compiles a string of source at runtime — unsupported here.
                let m = self.new_str("Function constructor (dynamic code) is not supported");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            N_PARSE_INT => {
                let s = self.realm.to_display_string(arg(0));
                let radix = match args.get(1) {
                    Some(r) if !matches!(r.unpack(), Unpacked::Undefined) => {
                        let n = self.realm.to_number(*r);
                        // Keep the sign (a `… as u32` cast saturates a negative
                        // radix to 0, which would wrongly default to base 10).
                        if n.is_finite() { n as i64 } else { 0 }
                    }
                    _ => 0,
                };
                // A nonzero radix outside [2, 36] is invalid → NaN; 0 means "infer".
                if radix != 0 && !(2..=36).contains(&radix) {
                    NanBox::number(f64::NAN)
                } else {
                    NanBox::number(parse_int(&s, radix as u32))
                }
            }
            N_CONSOLE_LOG => {
                let line: Vec<String> = args
                    .iter()
                    .map(|a| self.realm.to_display_string(*a))
                    .collect();
                self.output.push_str(&line.join(" "));
                self.output.push('\n');
                NanBox::undefined()
            }
            N_JSON_STRINGIFY => {
                // Optional `replacer` (arg 1): a function transforms each value,
                // an array allowlists object keys.
                let mut value = arg(0);
                if let Some(rh) = arg(1).as_handle().map(Handle::from_raw) {
                    if self.is_callable(rh) {
                        let holder = self.realm.new_object();
                        self.realm.set_property(holder, "", value);
                        value = self.json_apply_replacer(holder, "", value, arg(1))?;
                    } else if self.realm.is_array(rh) {
                        let allow: Vec<String> = self
                            .realm
                            .array_elements(rh)
                            .map(<[_]>::to_vec)
                            .unwrap_or_default()
                            .iter()
                            .map(|e| self.realm.to_display_string(*e))
                            .collect();
                        value = self.json_filter_keys(value, &allow);
                    }
                }
                // Optional `space` (arg 2): a number → that many spaces, a string
                // → that string (both capped at 10), else compact output.
                let space = arg(2);
                let indent = if let Some(n) = space.as_number() {
                    " ".repeat((n.max(0.0) as usize).min(10))
                } else if let Some(s) = space
                    .as_handle()
                    .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
                {
                    s.chars().take(10).collect()
                } else {
                    String::new()
                };
                let result = if indent.is_empty() {
                    // Interpreter-aware: honors `toJSON` and invokes getters.
                    self.json_to_string(value)?
                } else {
                    match crate::json::try_stringify_pretty(&self.realm, value, &indent) {
                        Ok(r) => r,
                        Err(crate::json::Circular) => {
                            let m = self.new_str("Converting circular structure to JSON");
                            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                        }
                    }
                };
                match result {
                    Some(s) => NanBox::handle(self.realm.new_string(&s).to_raw()),
                    None => NanBox::undefined(),
                }
            }
            N_JSON_PARSE => {
                let text = self.realm.to_display_string(arg(0));
                let chars: Vec<char> = text.chars().collect();
                let mut pos = 0;
                let value = self.json_parse(&chars, &mut pos)?;
                skip_ws(&chars, &mut pos);
                if pos != chars.len() {
                    return Err(self.json_error("Unexpected token in JSON"));
                }
                // An optional `reviver` transforms each value bottom-up.
                let reviver = arg(1);
                if reviver
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    let holder = self.realm.new_object();
                    self.realm.set_property(holder, "", value);
                    return self.json_revive(holder, "", reviver);
                }
                value
            }
            N_OBJECT_KEYS => {
                // A proxy with an `ownKeys` trap drives `Object.keys` itself.
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let boxed: Vec<NanBox> = keys.iter().map(|k| self.new_str(k)).collect();
                    return Ok(NanBox::handle(self.realm.new_array(boxed).to_raw()));
                }
                let target = arg(0)
                    .as_handle()
                    .map(|raw| self.proxy_key_target(Handle::from_raw(raw)));
                let mut keys: Vec<alloc::string::String> = Vec::new();
                if let Some(h) = target {
                    // An array's own enumerable keys are its integer indices (in
                    // ascending order) — stored as elements, not named properties.
                    // A VM closure backs onto an array but is a function, so its
                    // "indices" (captured cells) are not enumerable keys.
                    if let Some(len) = self.realm.array_length(h)
                        && !self.realm.is_vm_function(h)
                    {
                        for i in 0..len {
                            keys.push(alloc::format!("{i}"));
                        }
                    }
                    if let Some(named) = self.realm.object_keys(h) {
                        keys.extend(named);
                    } else {
                        // An array/function/native keeps named properties in its
                        // auxiliary object (e.g. `arr.custom`, a match result's
                        // `index`/`input`).
                        keys.extend(self.realm.aux_named_keys(h));
                    }
                    // A class constructor's enumerable own keys are its static
                    // fields (static methods/accessors are non-enumerable).
                    if let Some((cid, _)) = self.realm.class_at(h) {
                        keys.extend(self.class_static_fields[cid as usize].iter().cloned());
                    }
                }
                let boxed: Vec<NanBox> = keys.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            N_OBJECT_FREEZE => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.freeze_object(Handle::from_raw(raw));
                }
                arg(0) // returns the (now frozen) object
            }
            N_OBJECT_SEAL => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.seal_object(Handle::from_raw(raw));
                }
                arg(0)
            }
            N_OBJECT_PREVENT_EXT => {
                if let Some(raw) = arg(0).as_handle() {
                    self.realm.prevent_extensions(Handle::from_raw(raw));
                }
                arg(0)
            }
            N_OBJECT_IS_SEALED => {
                // A non-object argument (a primitive) is reported as sealed.
                let v = arg(0);
                let sealed = !self.is_object_value(v)
                    || v.as_handle()
                        .is_some_and(|raw| self.realm.is_sealed(Handle::from_raw(raw)));
                NanBox::boolean(sealed)
            }
            N_OBJECT_IS_EXTENSIBLE => {
                if let Some(obj) = arg(0).as_handle().map(Handle::from_raw) {
                    self.is_extensible_of(obj)?
                } else {
                    NanBox::boolean(false)
                }
            }
            // `Object.create(proto)` — a new object with the given prototype
            // (`null` → no prototype).
            N_OBJECT_CREATE => {
                let proto = arg(0).as_handle().map(Handle::from_raw);
                let obj = self.realm.new_object_with_proto(proto);
                // Optional second argument: a property-descriptors map.
                if let Some(descs) = arg(1).as_handle().map(Handle::from_raw) {
                    for key in self.realm.object_keys(descs).unwrap_or_default() {
                        if let Some(d) = self
                            .realm
                            .get_property(descs, &key)
                            .and_then(NanBox::as_handle)
                        {
                            self.apply_descriptor(obj, &key, Handle::from_raw(d))?;
                        }
                    }
                }
                NanBox::handle(obj.to_raw())
            }
            N_OBJECT_GET_PROTO => match arg(0).as_handle() {
                Some(raw) => self.get_proto_of(Handle::from_raw(raw))?,
                None => NanBox::null(),
            },
            N_OBJECT_SET_PROTO => {
                if let Some(raw) = arg(0).as_handle() {
                    let proto = arg(1).as_handle().map(Handle::from_raw);
                    self.set_proto_of(Handle::from_raw(raw), proto)?;
                }
                arg(0)
            }
            // `Object.defineProperty(obj, key, descriptor)` — a `value`
            // descriptor sets the property; a `get`/`set` descriptor defines an
            // accessor.
            N_OBJECT_DEFINE_PROP => {
                if let Some(oraw) = arg(0).as_handle()
                    && let Some(draw) = arg(2).as_handle()
                {
                    let obj = Handle::from_raw(oraw);
                    let key = self.member_key(arg(1));
                    self.apply_descriptor(obj, &key, Handle::from_raw(draw))?;
                }
                arg(0)
            }
            // `Object.defineProperties(obj, { k: descriptor, … })`.
            N_OBJECT_DEFINE_PROPS => {
                if let Some(oraw) = arg(0).as_handle()
                    && let Some(draw) = arg(1).as_handle()
                {
                    let obj = Handle::from_raw(oraw);
                    let descs = Handle::from_raw(draw);
                    for key in self.realm.object_keys(descs).unwrap_or_default() {
                        if let Some(d) = self
                            .realm
                            .get_property(descs, &key)
                            .and_then(NanBox::as_handle)
                        {
                            self.apply_descriptor(obj, &key, Handle::from_raw(d))?;
                        }
                    }
                }
                arg(0)
            }
            // `Object.is(a, b)` — SameValue: like `===` but `NaN` is equal to
            // itself and `+0`/`-0` differ.
            N_OBJECT_IS => {
                let (a, b) = (arg(0), arg(1));
                let same = match (a.as_number(), b.as_number()) {
                    (Some(x), Some(y)) => {
                        (x == y && (x != 0.0 || x.is_sign_positive() == y.is_sign_positive()))
                            || (x.is_nan() && y.is_nan())
                    }
                    _ => self.realm.strict_equals(a, b),
                };
                NanBox::boolean(same)
            }
            // `Object.hasOwn(obj, key)` — own-property check (incl. array index).
            N_OBJECT_HAS_OWN => {
                let key = self.realm.to_display_string(arg(1));
                let owned = arg(0).as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.has_own(h, &key)
                        || self
                            .realm
                            .array_length(h)
                            .is_some_and(|len| key.parse::<usize>().is_ok_and(|i| i < len))
                });
                NanBox::boolean(owned)
            }
            // `Object.groupBy(items, cb)` — groups each item by `cb(item, i)` into
            // an object of arrays keyed by the (stringified) group.
            N_OBJECT_GROUP_BY => {
                let items = self.iterate_values(arg(0))?;
                let cb = arg(1);
                let out = self.realm.new_object();
                for (i, item) in items.iter().enumerate() {
                    let key = self.call(cb, &[*item, NanBox::number(i as f64)])?;
                    let k = self.realm.to_display_string(key);
                    let bucket = match self
                        .realm
                        .get_property(out, &k)
                        .and_then(NanBox::as_handle)
                        .map(Handle::from_raw)
                    {
                        Some(h) => h,
                        None => {
                            let arr = self.realm.new_array(Vec::new());
                            self.realm
                                .set_property(out, &k, NanBox::handle(arr.to_raw()));
                            arr
                        }
                    };
                    self.realm.array_push(bucket, *item);
                }
                NanBox::handle(out.to_raw())
            }
            // --- Reflect.* ---
            N_REFLECT_GET => {
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    let key = self.member_key(arg(1));
                    // With an explicit `receiver` (3rd arg), a getter found on the
                    // prototype chain runs with `receiver` as its `this` (a data
                    // property ignores the receiver — handled by `read_member`).
                    if args.len() > 2 {
                        let receiver = arg(2);
                        let mut cur = Some(h);
                        while let Some(c) = cur {
                            if let Some((getter, _)) = self.realm.accessor(c, &key) {
                                if matches!(getter.unpack(), Unpacked::Undefined) {
                                    return Ok(NanBox::undefined());
                                }
                                return self.call_with_this(getter, receiver, &[]);
                            }
                            if self.realm.has_own(c, &key) {
                                break;
                            }
                            cur = self.realm.object_proto(c);
                        }
                    }
                    return self.read_member(h, &key);
                }
                NanBox::undefined()
            }
            N_REFLECT_SET => {
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    let key = self.member_key(arg(1));
                    let value = arg(2);
                    // The receiver defaults to the target; an explicit one (4th arg)
                    // receives the write / is the setter's `this`.
                    let receiver = if args.len() > 3 {
                        arg(3)
                    } else {
                        NanBox::handle(h.to_raw())
                    };
                    // A setter accessor found on the chain runs with `receiver` as
                    // `this` (an accessor with no setter fails).
                    let mut cur = Some(h);
                    while let Some(c) = cur {
                        if let Some((_, setter)) = self.realm.accessor(c, &key) {
                            if matches!(setter.unpack(), Unpacked::Undefined) {
                                return Ok(NanBox::boolean(false));
                            }
                            self.call_with_this(setter, receiver, &[value])?;
                            return Ok(NanBox::boolean(true));
                        }
                        if self.realm.has_own(c, &key) {
                            break;
                        }
                        cur = self.realm.object_proto(c);
                    }
                    // No setter: write the data property on the receiver (via
                    // `assign_member_value`, so array indices/`length` behave right).
                    let Some(rh) = receiver.as_handle() else {
                        return Ok(NanBox::boolean(false));
                    };
                    self.assign_member_value(Handle::from_raw(rh), arg(1), value)?;
                }
                NanBox::boolean(true)
            }
            N_REFLECT_HAS => {
                // Like the `in` operator: own property or anywhere on the
                // prototype chain (array indices bounds-checked).
                let key = self.member_key(arg(1));
                let mut present = false;
                let mut cur = arg(0).as_handle().map(Handle::from_raw);
                while let Some(c) = cur {
                    let here = if let Some(len) = self.realm.array_length(c) {
                        key == "length"
                            || key.parse::<usize>().is_ok_and(|i| i < len)
                            || self.realm.has_own(c, &key)
                    } else {
                        self.realm.has_own(c, &key)
                    };
                    if here {
                        present = true;
                        break;
                    }
                    cur = self.realm.object_proto(c);
                }
                NanBox::boolean(present)
            }
            N_REFLECT_DELETE => {
                if let Some(raw) = arg(0).as_handle() {
                    let key = self.member_key(arg(1));
                    self.realm.delete_property(Handle::from_raw(raw), &key);
                }
                NanBox::boolean(true)
            }
            N_REFLECT_OWN_KEYS => {
                // String keys (integer-indexed then insertion order), then own
                // symbol keys — matching `[[OwnPropertyKeys]]`.
                let mut boxed = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    for k in self.realm.own_property_names(h).unwrap_or_default() {
                        boxed.push(self.new_str(&k));
                    }
                    for k in self.realm.object_keys_with_symbols(h) {
                        if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                            && let Ok(id) = idstr.parse::<u64>()
                            && let Some(sh) = self.realm.symbol_for_id(id)
                        {
                            boxed.push(NanBox::handle(sh.to_raw()));
                        }
                    }
                }
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            // `Reflect.defineProperty(obj, key, desc)` → bool.
            N_REFLECT_DEFINE_PROP => {
                let done = if let (Some(obj), Some(desc)) = (
                    arg(0).as_handle().map(Handle::from_raw),
                    arg(2).as_handle().map(Handle::from_raw),
                ) {
                    let key = self.member_key(arg(1));
                    self.apply_descriptor(obj, &key, desc)?;
                    true
                } else {
                    false
                };
                NanBox::boolean(done)
            }
            // `Reflect.getOwnPropertyDescriptor(obj, key)`.
            N_REFLECT_GET_OWN_DESC => match arg(0).as_handle().map(Handle::from_raw) {
                Some(obj) => {
                    let key = self.member_key(arg(1));
                    self.descriptor_of(obj, &key)?
                }
                None => NanBox::undefined(),
            },
            // `Reflect.getPrototypeOf(obj)` (honors a proxy `getPrototypeOf` trap).
            N_REFLECT_GET_PROTO => match arg(0).as_handle() {
                Some(raw) => self.get_proto_of(Handle::from_raw(raw))?,
                None => NanBox::null(),
            },
            // `Reflect.setPrototypeOf(target, proto)` → boolean success.
            N_REFLECT_SET_PROTO => {
                if let Some(raw) = arg(0).as_handle() {
                    let proto = arg(1).as_handle().map(Handle::from_raw);
                    self.set_proto_of(Handle::from_raw(raw), proto)?;
                    NanBox::boolean(true)
                } else {
                    NanBox::boolean(false)
                }
            }
            // `Reflect.preventExtensions(target)` → boolean success.
            N_REFLECT_PREVENT_EXT => {
                let ok = arg(0).as_handle().is_some_and(|raw| {
                    self.realm.prevent_extensions(Handle::from_raw(raw));
                    true
                });
                NanBox::boolean(ok)
            }
            N_REFLECT_APPLY => {
                let list = match arg(2).as_handle().map(Handle::from_raw) {
                    Some(h) => self
                        .realm
                        .array_elements(h)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                return self.call_with_this(arg(0), arg(1), &list);
            }
            N_REFLECT_CONSTRUCT => {
                let list = match arg(1).as_handle().map(Handle::from_raw) {
                    Some(h) => self
                        .realm
                        .array_elements(h)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                // An explicit `newTarget` (3rd arg) becomes `new.target` inside the
                // constructor (else it is the target itself).
                if args.len() > 2 && !matches!(arg(2).unpack(), Unpacked::Undefined) {
                    self.reflect_new_target = Some(arg(2));
                }
                return self.construct(arg(0), &list);
            }
            N_OBJECT_GET_OWN_DESC => match arg(0).as_handle().map(Handle::from_raw) {
                Some(obj) => {
                    let key = self.member_key(arg(1));
                    self.descriptor_of(obj, &key)?
                }
                None => NanBox::undefined(),
            },
            // `Object.getOwnPropertyDescriptors(obj)` → a map of all descriptors.
            N_OBJECT_GET_OWN_DESCS => {
                let out = self.realm.new_object();
                if let Some(obj) = arg(0).as_handle().map(Handle::from_raw) {
                    let mut keys = self.realm.own_property_names(obj).unwrap_or_default();
                    keys.extend(self.realm.object_accessor_keys(obj));
                    for k in keys {
                        if let Some(d) = self.build_descriptor(obj, &k) {
                            self.realm.set_property(out, &k, d);
                        }
                    }
                }
                NanBox::handle(out.to_raw())
            }
            N_OBJECT_IS_FROZEN => {
                // A non-object argument (a primitive) is reported as frozen.
                let v = arg(0);
                let frozen = !self.is_object_value(v)
                    || v.as_handle()
                        .is_some_and(|raw| self.realm.is_frozen(Handle::from_raw(raw)));
                NanBox::boolean(frozen)
            }
            N_OBJECT_GET_OWN_NAMES => {
                let names = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.own_property_names(Handle::from_raw(raw)))
                    .unwrap_or_default();
                let boxed: Vec<NanBox> = names.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            // `Object.getOwnPropertySymbols(obj)` — the own symbol-keyed
            // properties (recovered from their `\0sym:{id}` internal names).
            N_OBJECT_GET_OWN_SYMBOLS => {
                let mut syms = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    // All own symbol keys, including non-enumerable ones (e.g. a
                    // symbol defined via `Object.defineProperty`).
                    for k in self.realm.object_all_keys(h) {
                        if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                            && let Ok(id) = idstr.parse::<u64>()
                            && let Some(sh) = self.realm.symbol_for_id(id)
                        {
                            syms.push(NanBox::handle(sh.to_raw()));
                        }
                    }
                }
                NanBox::handle(self.realm.new_array(syms).to_raw())
            }
            N_OBJECT_VALUES => {
                // A proxy with an `ownKeys` trap: its enumerable keys, each value
                // read through the proxy (so a `get` trap fires).
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let ph = Handle::from_raw(raw);
                    let mut vals = Vec::with_capacity(keys.len());
                    for k in keys {
                        vals.push(self.read_member(ph, &k)?);
                    }
                    return Ok(NanBox::handle(self.realm.new_array(vals).to_raw()));
                }
                let mut vals = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = self.proxy_key_target(Handle::from_raw(raw));
                    // Array index values come from element access (ascending) first
                    // — but a VM closure's backing cells are not enumerable values.
                    if !self.realm.is_vm_function(h)
                        && let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec)
                    {
                        vals.extend(elems);
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        vals.push(
                            self.realm
                                .get_property(h, &k)
                                .unwrap_or(NanBox::undefined()),
                        );
                    }
                    // A class constructor's static fields are its enumerable values.
                    if let Some((cid, _)) = self.realm.class_at(h) {
                        let fields = self.class_static_fields[cid as usize].clone();
                        for k in fields {
                            vals.push(self.read_member(h, &k)?);
                        }
                    }
                }
                NanBox::handle(self.realm.new_array(vals).to_raw())
            }
            N_ARRAY_IS_ARRAY => NanBox::boolean(arg(0).as_handle().is_some_and(|raw| {
                let h = Handle::from_raw(raw);
                self.realm.is_array(h) && !self.realm.is_vm_function(h)
            })),
            // `ArrayBuffer.isView(x)` — true iff `x` is a typed array or a DataView
            // (anything with a `[[ViewedArrayBuffer]]`).
            N_ARRAY_BUFFER_IS_VIEW => NanBox::boolean(arg(0).as_handle().is_some_and(|raw| {
                let h = Handle::from_raw(raw);
                self.realm.get_property(h, TYPED_ARRAY_KIND).is_some()
                    || self.realm.get_property(h, DATA_VIEW_BUF).is_some()
            })),
            N_OBJECT_ASSIGN => {
                let target = arg(0);
                if let Some(t) = target.as_handle().map(Handle::from_raw) {
                    for src in &args[1.min(args.len())..] {
                        // A primitive string source contributes its character indices.
                        if let Some(s) = src
                            .as_handle()
                            .and_then(|r| self.realm.string_value(Handle::from_raw(r)))
                            && !self
                                .realm
                                .is_array(Handle::from_raw(src.as_handle().unwrap()))
                        {
                            for (i, ch) in s.chars().enumerate() {
                                let cv = self.new_str(&alloc::format!("{ch}"));
                                let kb = self.new_str(&alloc::format!("{i}"));
                                self.assign_member_value(t, kb, cv)?;
                            }
                            continue;
                        }
                        if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                            // An array source contributes its indexed elements.
                            if let Some(elems) = self.realm.array_elements(sh).map(<[_]>::to_vec) {
                                for (i, e) in elems.iter().enumerate() {
                                    let kb = self.new_str(&alloc::format!("{i}"));
                                    self.assign_member_value(t, kb, *e)?;
                                }
                                continue;
                            }
                            // Own enumerable string *and* symbol keys; values read via
                            // `read_member` (so getters fire) and written via
                            // `assign_member_value` ([[Set]], so the target's setters
                            // run and a frozen/read-only property is honored).
                            let keys = self.realm.object_keys_with_symbols(sh);
                            for k in keys {
                                let v = self.read_member(sh, &k)?;
                                let kb = if let Some(idstr) = k.strip_prefix("\u{0}sym:")
                                    && let Ok(id) = idstr.parse::<u64>()
                                    && let Some(sym) = self.realm.symbol_for_id(id)
                                {
                                    NanBox::handle(sym.to_raw())
                                } else {
                                    self.new_str(&k)
                                };
                                self.assign_member_value(t, kb, v)?;
                            }
                        }
                    }
                }
                target
            }
            N_OBJECT_ENTRIES => {
                // A proxy with an `ownKeys` trap drives the entry list (values read
                // through the proxy so a `get` trap fires).
                if let Some(raw) = arg(0).as_handle()
                    && let Some(keys) = self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    let ph = Handle::from_raw(raw);
                    let mut pairs = Vec::with_capacity(keys.len());
                    for k in keys {
                        let v = self.read_member(ph, &k)?;
                        let key = self.new_str(&k);
                        pairs.push(NanBox::handle(
                            self.realm.new_array(alloc::vec![key, v]).to_raw(),
                        ));
                    }
                    return Ok(NanBox::handle(self.realm.new_array(pairs).to_raw()));
                }
                let mut entries: Vec<(alloc::string::String, NanBox)> = Vec::new();
                if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                    let h = self.proxy_key_target(h);
                    // Array index entries (ascending) before named ones — but a VM
                    // closure's backing cells are not enumerable entries.
                    if !self.realm.is_vm_function(h)
                        && let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec)
                    {
                        for (i, v) in elems.into_iter().enumerate() {
                            entries.push((alloc::format!("{i}"), v));
                        }
                    }
                    let named = self
                        .realm
                        .object_keys(h)
                        .unwrap_or_else(|| self.realm.aux_named_keys(h));
                    for k in named {
                        let v = self
                            .realm
                            .get_property(h, &k)
                            .unwrap_or(NanBox::undefined());
                        entries.push((k, v));
                    }
                    // A class constructor's static fields are its enumerable entries.
                    if let Some((cid, _)) = self.realm.class_at(h) {
                        let fields = self.class_static_fields[cid as usize].clone();
                        for k in fields {
                            let v = self.read_member(h, &k)?;
                            entries.push((k, v));
                        }
                    }
                }
                let pairs: Vec<NanBox> = entries
                    .into_iter()
                    .map(|(k, v)| {
                        let key = self.new_str(&k);
                        NanBox::handle(self.realm.new_array(alloc::vec![key, v]).to_raw())
                    })
                    .collect();
                NanBox::handle(self.realm.new_array(pairs).to_raw())
            }
            N_ARRAY_FROM => {
                // Iterable → array (arrays, strings, Sets, Maps), with an
                // optional map callback applied to each element. A non-iterable
                // array-like (an object with a `length`) is read by index.
                let items = match self.iterate_values(arg(0)) {
                    Ok(v) => v,
                    Err(_) => {
                        let mut out = Vec::new();
                        if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                            let len = self
                                .realm
                                .get_property(h, "length")
                                .map(|v| self.realm.to_number(v))
                                .unwrap_or(0.0)
                                .max(0.0) as usize;
                            for i in 0..len {
                                let k = alloc::format!("{i}");
                                out.push(
                                    self.realm
                                        .get_property(h, &k)
                                        .unwrap_or(NanBox::undefined()),
                                );
                            }
                        }
                        out
                    }
                };
                let items = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                    items
                } else {
                    let f = arg(1);
                    let this_arg = arg(2); // `Array.from(items, mapFn, thisArg)`
                    let mut out = Vec::with_capacity(items.len());
                    for (i, e) in items.iter().enumerate() {
                        out.push(self.call_with_this(
                            f,
                            this_arg,
                            &[*e, NanBox::number(i as f64)],
                        )?);
                    }
                    out
                };
                NanBox::handle(self.realm.new_array(items).to_raw())
            }
            N_ARRAY_OF => NanBox::handle(self.realm.new_array(args.to_vec()).to_raw()),
            N_OBJECT_FROM_ENTRIES => {
                let obj = self.realm.new_object();
                // Accepts any iterable of `[key, value]` pairs (arrays, a Map, …).
                let pairs = self.iterate_values(arg(0)).unwrap_or_default();
                for pair in pairs {
                    if let Some(kv) = pair
                        .as_handle()
                        .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                        .map(<[_]>::to_vec)
                    {
                        let k = self
                            .realm
                            .to_display_string(kv.first().copied().unwrap_or(NanBox::undefined()));
                        let v = kv.get(1).copied().unwrap_or(NanBox::undefined());
                        self.realm.set_property(obj, &k, v);
                    }
                }
                NanBox::handle(obj.to_raw())
            }
            #[cfg(feature = "std")]
            N_MATH_FLOOR => NanBox::number(self.realm.to_number(arg(0)).floor()),
            #[cfg(feature = "std")]
            N_MATH_CEIL => NanBox::number(self.realm.to_number(arg(0)).ceil()),
            #[cfg(feature = "std")]
            // JS `Math.round` rounds half toward +Infinity (`floor(x + 0.5)`),
            // unlike Rust's round-half-away-from-zero.
            N_MATH_ROUND => {
                let n = self.realm.to_number(arg(0));
                NanBox::number(crate::common::js_round(n))
            }
            #[cfg(feature = "std")]
            N_MATH_SQRT => NanBox::number(self.realm.to_number(arg(0)).sqrt()),
            #[cfg(not(feature = "std"))]
            N_MATH_FLOOR | N_MATH_CEIL | N_MATH_ROUND | N_MATH_SQRT => {
                return Err(ExecError::Unsupported("Math float ops need std"));
            }
            #[cfg(feature = "std")]
            N_MATH_POW => self.realm.pow(arg(0), arg(1)),
            #[cfg(not(feature = "std"))]
            N_MATH_POW => return Err(ExecError::Unsupported("Math.pow needs std")),
            N_MATH_SIGN => {
                let n = self.realm.to_number(arg(0));
                NanBox::number(if n.is_nan() {
                    f64::NAN
                } else if n > 0.0 {
                    1.0
                } else if n < 0.0 {
                    -1.0
                } else {
                    n // ±0
                })
            }
            #[cfg(feature = "std")]
            N_MATH_TRUNC => NanBox::number(self.realm.to_number(arg(0)).trunc()),
            #[cfg(not(feature = "std"))]
            N_MATH_TRUNC => return Err(ExecError::Unsupported("Math.trunc needs std")),
            #[cfg(feature = "std")]
            N_MATH_HYPOT => {
                // If any argument is ±Infinity the result is +Infinity, even when
                // another argument is NaN (NaN only wins if no argument is infinite).
                let mut any_inf = false;
                let mut any_nan = false;
                let mut sum = 0.0;
                for a in args {
                    let n = self.realm.to_number(*a);
                    if n.is_infinite() {
                        any_inf = true;
                    } else if n.is_nan() {
                        any_nan = true;
                    } else {
                        sum += n * n;
                    }
                }
                NanBox::number(if any_inf {
                    f64::INFINITY
                } else if any_nan {
                    f64::NAN
                } else {
                    sum.sqrt()
                })
            }
            #[cfg(feature = "std")]
            N_MATH_CBRT => NanBox::number(self.realm.to_number(arg(0)).cbrt()),
            #[cfg(feature = "std")]
            N_MATH_LOG2 => NanBox::number(self.realm.to_number(arg(0)).log2()),
            #[cfg(feature = "std")]
            N_MATH_LOG10 => NanBox::number(self.realm.to_number(arg(0)).log10()),
            #[cfg(feature = "std")]
            N_MATH_EXP => NanBox::number(self.realm.to_number(arg(0)).exp()),
            #[cfg(feature = "std")]
            N_MATH_LOG => NanBox::number(self.realm.to_number(arg(0)).ln()),
            // `Math.random()` ∈ [0, 1) — a pure-Rust xorshift64 generator; the
            // top 53 bits form the mantissa.
            N_MATH_RANDOM => {
                let mut x = self.rng_state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.rng_state = x;
                NanBox::number((x >> 11) as f64 / (1u64 << 53) as f64)
            }
            // Trig / hyperbolic / inverse — single-argument f64 functions.
            #[cfg(feature = "std")]
            N_MATH_SIN..=N_MATH_LOG1P => {
                let n = self.realm.to_number(arg(0));
                let r = match id {
                    N_MATH_SIN => n.sin(),
                    N_MATH_COS => n.cos(),
                    N_MATH_TAN => n.tan(),
                    N_MATH_ASIN => n.asin(),
                    N_MATH_ACOS => n.acos(),
                    N_MATH_ATAN => n.atan(),
                    N_MATH_ATAN2 => n.atan2(self.realm.to_number(arg(1))),
                    N_MATH_SINH => n.sinh(),
                    N_MATH_COSH => n.cosh(),
                    N_MATH_TANH => n.tanh(),
                    N_MATH_ASINH => n.asinh(),
                    N_MATH_ACOSH => n.acosh(),
                    N_MATH_ATANH => n.atanh(),
                    N_MATH_EXPM1 => n.exp_m1(),
                    _ => n.ln_1p(), // N_MATH_LOG1P
                };
                NanBox::number(r)
            }
            #[cfg(not(feature = "std"))]
            N_MATH_SIN..=N_MATH_LOG1P => {
                return Err(ExecError::Unsupported("Math fns need std"));
            }
            // `Math.fround(x)` — round to the nearest single-precision float.
            N_MATH_FROUND => NanBox::number(self.realm.to_number(arg(0)) as f32 as f64),
            // `Math.clz32(x)` — count leading zeros of the ToUint32 value.
            N_MATH_CLZ32 => {
                let u = self.realm.to_number(arg(0)) as i64 as u32;
                NanBox::number(u.leading_zeros() as f64)
            }
            // `Math.imul(a, b)` — 32-bit integer multiplication.
            N_MATH_IMUL => {
                let a = self.realm.to_number(arg(0)) as i64 as i32;
                let b = self.realm.to_number(arg(1)) as i64 as i32;
                NanBox::number(a.wrapping_mul(b) as f64)
            }
            #[cfg(not(feature = "std"))]
            N_MATH_HYPOT | N_MATH_CBRT | N_MATH_LOG2 | N_MATH_LOG10 | N_MATH_EXP | N_MATH_LOG => {
                return Err(ExecError::Unsupported("Math fns need std"));
            }
            N_PARSE_FLOAT => {
                let s = self.realm.to_display_string(arg(0));
                NanBox::number(parse_float_prefix(s.trim()))
            }
            // URI encoding/decoding. `encodeURI` preserves the URI reserved set
            // on top of the unreserved set that `encodeURIComponent` keeps.
            N_ENCODE_URI_COMPONENT | N_ENCODE_URI => {
                let s = self.realm.to_display_string(arg(0));
                let extra = if id == N_ENCODE_URI {
                    ";,/?:@&=+$#"
                } else {
                    ""
                };
                let out = uri_encode(&s, extra);
                let h = self.realm.new_string(&out);
                NanBox::handle(h.to_raw())
            }
            N_DECODE_URI_COMPONENT | N_DECODE_URI => {
                let s = self.realm.to_display_string(arg(0));
                match uri_decode(&s) {
                    Some(out) => {
                        let h = self.realm.new_string(&out);
                        NanBox::handle(h.to_raw())
                    }
                    None => {
                        let m = self.new_str("URI malformed");
                        return Err(ExecError::Throw(self.make_error(N_URI_ERROR, Some(m))));
                    }
                }
            }
            N_STRUCTURED_CLONE => {
                let mut seen: Vec<(u64, NanBox)> = Vec::new();
                self.structured_clone(arg(0), &mut seen)?
            }
            // `Intl.NumberFormat(...)` / `Intl.DateTimeFormat(...)` called without
            // `new` build the same formatter object.
            N_INTL_NUMBER_FORMAT | N_INTL_DATETIME_FORMAT => self.make_intl_formatter(id, args),
            // `Intl.Collator(...)` / `Intl.PluralRules(...)` without `new`.
            N_INTL_COLLATOR => {
                let obj = self.realm.new_object();
                let cmp = self.new_named_native("compare", N_INTL_COMPARE);
                self.realm
                    .set_property(obj, "compare", NanBox::handle(cmp.to_raw()));
                NanBox::handle(obj.to_raw())
            }
            N_INTL_PLURAL_RULES => {
                let obj = self.realm.new_object();
                let sel = self.new_named_native("select", N_INTL_PLURAL_SELECT);
                self.realm
                    .set_property(obj, "select", NanBox::handle(sel.to_raw()));
                NanBox::handle(obj.to_raw())
            }
            // `Intl.Collator.prototype.compare(a, b)` — code-point order (no locale
            // tailoring), so a negative/zero/positive result orders `a` vs `b`.
            N_INTL_COMPARE => {
                let a = self.realm.to_display_string(arg(0));
                let b = self.realm.to_display_string(arg(1));
                NanBox::number(match a.cmp(&b) {
                    core::cmp::Ordering::Less => -1.0,
                    core::cmp::Ordering::Equal => 0.0,
                    core::cmp::Ordering::Greater => 1.0,
                })
            }
            // `Intl.PluralRules.prototype.select(n)` — the English plural category:
            // `1` is "one", everything else "other".
            N_INTL_PLURAL_SELECT => {
                let n = self.realm.to_number(arg(0));
                let cat = if n == 1.0 { "one" } else { "other" };
                self.new_str(cat)
            }
            // `nf.format(x)` read as a value then called: format against the `this`
            // formatter (a detached call with no formatter falls back to ToString).
            N_INTL_FORMAT => {
                if let Some(h) = self.this_val.as_handle().map(Handle::from_raw)
                    && self.realm.get_property(h, "\u{0}intl").is_some()
                {
                    let s = self.intl_format_value(h, arg(0));
                    self.new_str(&s)
                } else {
                    let s = self.realm.to_display_string(arg(0));
                    self.new_str(&s)
                }
            }
            // `setTimeout(cb, delay?, ...args)` — queues `cb(...args)` as a macrotask
            // and returns a numeric timer id (usable with `clearTimeout`).
            N_SET_TIMEOUT => {
                let callback = arg(0);
                let delay = self.realm.to_number(arg(1)).max(0.0);
                let extra: Vec<NanBox> = args.iter().skip(2).copied().collect();
                let id = self.timer_next_id;
                self.timer_next_id += 1;
                let seq = self.timer_seq;
                self.timer_seq += 1;
                self.macrotasks.push(Timer {
                    id,
                    delay: if delay.is_finite() { delay } else { 0.0 },
                    seq,
                    callback,
                    args: extra,
                });
                NanBox::number(id as f64)
            }
            // `clearTimeout(id)` — cancels a pending `setTimeout`.
            N_CLEAR_TIMEOUT => {
                if let Some(id) = arg(0).as_number() {
                    self.macrotasks.retain(|t| (t.id as f64) != id);
                }
                NanBox::undefined()
            }
            // `queueMicrotask(cb)` — schedules `cb()` on the microtask queue.
            N_QUEUE_MICROTASK => {
                let callback = arg(0);
                let result = self.realm.new_promise();
                self.microtasks.push(Job {
                    handler: callback,
                    value: NanBox::undefined(),
                    result,
                    fulfilled: true,
                    finally: false,
                });
                NanBox::undefined()
            }
            // `WebAssembly.validate(bytes)` — true iff `bytes` decodes to a
            // well-formed module. Accepts an `ArrayBuffer` or a byte array.
            N_WASM_VALIDATE => {
                let ok = self
                    .wasm_bytes(arg(0))
                    .is_some_and(|b| crate::wasm_rt::Module::decode(&b).is_ok());
                NanBox::boolean(ok)
            }
            // `WebAssembly.Module.exports(module)` / `.imports(module)` — arrays of
            // `{ name, kind }` / `{ module, name, kind }` descriptors.
            N_WASM_MODULE_EXPORTS | N_WASM_MODULE_IMPORTS => {
                let bytes = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.get_property(h, WASM_BYTES))
                    .and_then(|v| self.wasm_bytes(v))
                    .ok_or_else(|| self.wasm_type_error("expected a WebAssembly.Module"))?;
                let module = crate::wasm_rt::Module::decode(&bytes)
                    .map_err(|e| self.wasm_compile_error(e.0))?;
                let mut out = Vec::new();
                if id == N_WASM_MODULE_EXPORTS {
                    let descs: Vec<(String, u8)> = module
                        .export_descriptors()
                        .iter()
                        .map(|(n, k)| ((*n).into(), *k))
                        .collect();
                    for (name, kind) in descs {
                        let obj = self.realm.new_object();
                        let nv = self.new_str(&name);
                        self.realm.set_property(obj, "name", nv);
                        let kv = self.new_str(wasm_extern_kind(kind));
                        self.realm.set_property(obj, "kind", kv);
                        out.push(NanBox::handle(obj.to_raw()));
                    }
                } else {
                    let descs: Vec<(String, String, u8)> = module
                        .import_descriptors()
                        .iter()
                        .map(|(m, f, k)| ((*m).into(), (*f).into(), *k))
                        .collect();
                    for (m, f, kind) in descs {
                        let obj = self.realm.new_object();
                        let mv = self.new_str(&m);
                        self.realm.set_property(obj, "module", mv);
                        let nv = self.new_str(&f);
                        self.realm.set_property(obj, "name", nv);
                        let kv = self.new_str(wasm_extern_kind(kind));
                        self.realm.set_property(obj, "kind", kv);
                        out.push(NanBox::handle(obj.to_raw()));
                    }
                }
                NanBox::handle(self.realm.new_array(out).to_raw())
            }
            // `WebAssembly.instantiate(x)` → a `Promise`: given source bytes it
            // resolves to `{ module, instance }`; given a `Module` it resolves to the
            // `Instance` alone. Each export is a callable wrapper. (A stateful module
            // re-instantiates per call.)
            N_WASM_INSTANTIATE => {
                let module_handle = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .filter(|h| self.realm.get_property(*h, WASM_IS_MODULE).is_some());
                let given_module = module_handle.is_some();
                // `build_wasm_instance` consumes source bytes; a `Module` argument
                // carries them under `WASM_BYTES`.
                let source =
                    match module_handle.and_then(|h| self.realm.get_property(h, WASM_BYTES)) {
                        Some(bytes) => bytes,
                        None => arg(0),
                    };
                let p = self.realm.new_promise();
                match self.build_wasm_instance(source, arg(1)) {
                    Ok(instance) => {
                        let resolved = if given_module {
                            instance
                        } else {
                            let result = self.realm.new_object();
                            self.realm.set_property(result, "instance", instance);
                            let module = self.realm.new_object();
                            self.realm.set_property(module, WASM_BYTES, arg(0));
                            self.realm.mark_hidden(module, WASM_BYTES);
                            self.realm.set_hidden_property(
                                module,
                                WASM_IS_MODULE,
                                NanBox::boolean(true),
                            );
                            self.realm.set_property(
                                result,
                                "module",
                                NanBox::handle(module.to_raw()),
                            );
                            NanBox::handle(result.to_raw())
                        };
                        self.settle(p, resolved, true);
                    }
                    Err(ExecError::Throw(err)) => self.settle(p, err, false),
                    Err(other) => return Err(other),
                }
                NanBox::handle(p.to_raw())
            }
            // `WebAssembly.compile(bytes)` → `Promise<Module>` (rejected, not thrown,
            // on a bad module).
            N_WASM_COMPILE => {
                let p = self.realm.new_promise();
                match self.make_wasm_module(arg(0)) {
                    Ok(module) => self.settle(p, module, true),
                    Err(ExecError::Throw(err)) => self.settle(p, err, false),
                    Err(other) => return Err(other),
                }
                NanBox::handle(p.to_raw())
            }
            // `Object.prototype.*` methods — the receiver is `self.this_val`.
            N_OBJ_PROTO_TOSTRING => {
                let this = self.this_val;
                let s = match this.unpack() {
                    Unpacked::Undefined => String::from("[object Undefined]"),
                    Unpacked::Null => String::from("[object Null]"),
                    // A primitive number/boolean (an immediate) reports its class
                    // (ToObject would box it to a Number/Boolean wrapper).
                    Unpacked::Number(_) => String::from("[object Number]"),
                    Unpacked::Bool(_) => String::from("[object Boolean]"),
                    _ => match this.as_handle().map(Handle::from_raw) {
                        Some(h) => alloc::format!("[object {}]", self.object_string_tag(h)?),
                        None => String::from("[object Object]"),
                    },
                };
                self.new_str(&s)
            }
            N_OBJ_PROTO_VALUEOF => self.this_val,
            N_OBJ_PROTO_HASOWN => {
                let key = self.member_key(arg(0));
                match self.this_val.as_handle().map(Handle::from_raw) {
                    Some(h) => NanBox::boolean(self.realm.has_own(h, &key)),
                    None => NanBox::boolean(false),
                }
            }
            N_OBJ_PROTO_PROPISENUM => {
                let key = self.member_key(arg(0));
                let enumerable = self
                    .this_val
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| {
                        self.realm.has_own(h, &key)
                            && self
                                .realm
                                .object_keys(h)
                                .is_some_and(|ks| ks.contains(&key))
                    });
                NanBox::boolean(enumerable)
            }
            N_OBJ_PROTO_ISPROTOTYPEOF => {
                // True if `this` appears in arg(0)'s prototype chain.
                let target = self.this_val.as_handle().map(Handle::from_raw);
                let mut cur = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.object_proto(h));
                let mut found = false;
                while let Some(p) = cur {
                    if Some(p) == target {
                        found = true;
                        break;
                    }
                    cur = self.realm.object_proto(p);
                }
                NanBox::boolean(found)
            }
            // `btoa(s)`: each code unit must be a byte (0–255) → base64.
            N_BTOA => {
                let s = self.realm.to_display_string(arg(0));
                let mut bytes = Vec::with_capacity(s.chars().count());
                for ch in s.chars() {
                    if (ch as u32) > 0xff {
                        let m = self.new_str("string contains a non-Latin1 character");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    bytes.push(ch as u8);
                }
                let h = self.realm.new_string(&base64_encode(&bytes));
                NanBox::handle(h.to_raw())
            }
            // `atob(s)`: base64 → a string of bytes (each a code unit 0–255).
            N_ATOB => {
                let s = self.realm.to_display_string(arg(0));
                match base64_decode(&s) {
                    Some(bytes) => {
                        let decoded: String = bytes.iter().map(|b| *b as char).collect();
                        let h = self.realm.new_string(&decoded);
                        NanBox::handle(h.to_raw())
                    }
                    None => {
                        let m = self.new_str("invalid base64");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                }
            }
            N_IS_NAN => NanBox::boolean(self.realm.to_number(arg(0)).is_nan()),
            N_IS_FINITE => NanBox::boolean(self.realm.to_number(arg(0)).is_finite()),
            // `Error(msg)` / `new Error(msg, { cause })` (the ES2022 cause option).
            id if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) => {
                let err = self.make_error(id, args.first().copied());
                if let Some(opts) = args
                    .get(1)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    && let Some(cause) = self.realm.get_property(opts, "cause")
                    && let Some(eh) = err.as_handle()
                {
                    self.realm
                        .set_property(Handle::from_raw(eh), "cause", cause);
                }
                err
            }
            _ => return Err(ExecError::NotCallable),
        })
    }

    /// Serializes a value to JSON (`None` when the value is `undefined` or a
    /// function — which `JSON.stringify` omits / drops).
    /// Recursive-descent `JSON.parse` over a char slice, advancing `pos`.
    /// `JSON.parse` reviver: transforms `holder[key]` bottom-up — children first,
    /// then `reviver.call(holder, key, value)` (a `undefined` result deletes the
    /// member). Mirrors `InternalizeJSONProperty`.
    fn json_revive(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        reviver: NanBox,
    ) -> Result<NanBox, ExecError> {
        let value = if self.realm.is_array(holder)
            && let Ok(i) = key.parse::<usize>()
        {
            self.realm.get_element(holder, i)
        } else {
            self.realm
                .get_property(holder, key)
                .unwrap_or(NanBox::undefined())
        };
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let len = self.realm.array_length(vh).unwrap_or(0);
                for i in 0..len {
                    let ks = alloc::format!("{i}");
                    let nv = self.json_revive(vh, &ks, reviver)?;
                    self.realm.set_element(vh, i, nv);
                }
            } else if let Some(keys) = self.realm.object_keys(vh) {
                for k in keys {
                    let nv = self.json_revive(vh, &k, reviver)?;
                    if matches!(nv.unpack(), Unpacked::Undefined) {
                        self.realm.delete_property(vh, &k);
                    } else {
                        self.realm.set_property(vh, &k, nv);
                    }
                }
            }
        }
        let kb = self.new_str(key);
        self.call_with_this(reviver, NanBox::handle(holder.to_raw()), &[kb, value])
    }

    /// `JSON.stringify` function replacer: returns a fresh value tree where each
    /// node is `replacer.call(holder, key, value)`, recursing into the result's
    /// own properties/elements (does not mutate the input).
    fn json_apply_replacer(
        &mut self,
        holder: crate::heap::Handle,
        key: &str,
        value: NanBox,
        replacer: NanBox,
    ) -> Result<NanBox, ExecError> {
        let kb = self.new_str(key);
        let v = self.call_with_this(replacer, NanBox::handle(holder.to_raw()), &[kb, value])?;
        if let Some(vh) = v.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let elems = self
                    .realm
                    .array_elements(vh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let mut out = Vec::with_capacity(elems.len());
                for (i, e) in elems.iter().enumerate() {
                    let ks = alloc::format!("{i}");
                    out.push(self.json_apply_replacer(vh, &ks, *e, replacer)?);
                }
                return Ok(NanBox::handle(self.realm.new_array(out).to_raw()));
            }
            if self.realm.string_value(vh).is_none()
                && let Some(keys) = self.realm.object_keys(vh)
            {
                let no = self.realm.new_object();
                for k in keys {
                    let pv = self
                        .realm
                        .get_property(vh, &k)
                        .unwrap_or(NanBox::undefined());
                    let nv = self.json_apply_replacer(vh, &k, pv, replacer)?;
                    if !matches!(nv.unpack(), Unpacked::Undefined) {
                        self.realm.set_property(no, &k, nv);
                    }
                }
                return Ok(NanBox::handle(no.to_raw()));
            }
        }
        Ok(v)
    }

    /// `JSON.stringify` array replacer: a fresh value tree keeping only object
    /// properties whose key is in `allow` (array elements are always kept).
    fn json_filter_keys(&mut self, value: NanBox, allow: &[String]) -> NanBox {
        if let Some(vh) = value.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(vh) {
                let elems = self
                    .realm
                    .array_elements(vh)
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
                let out: Vec<NanBox> = elems
                    .iter()
                    .map(|e| self.json_filter_keys(*e, allow))
                    .collect();
                return NanBox::handle(self.realm.new_array(out).to_raw());
            }
            if self.realm.string_value(vh).is_none()
                && let Some(keys) = self.realm.object_keys(vh)
            {
                let no = self.realm.new_object();
                // Keys are emitted in allowlist order (deduplicated, own keys only).
                let mut emitted: Vec<&String> = Vec::new();
                for k in allow {
                    if keys.contains(k) && !emitted.contains(&k) {
                        emitted.push(k);
                        let pv = self
                            .realm
                            .get_property(vh, k)
                            .unwrap_or(NanBox::undefined());
                        let nv = self.json_filter_keys(pv, allow);
                        self.realm.set_property(no, k, nv);
                    }
                }
                return NanBox::handle(no.to_raw());
            }
        }
        value
    }

    /// A `SyntaxError` for a malformed `JSON.parse` input (the spec error type, so
    /// `catch (e) { e instanceof SyntaxError }` works).
    fn json_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_ERROR_BASE + 3, Some(m)))
    }

    fn json_parse(&mut self, c: &[char], pos: &mut usize) -> Result<NanBox, ExecError> {
        skip_ws(c, pos);
        let err = |s: &mut Self| s.json_error("Unexpected end of JSON input");
        let Some(&ch) = c.get(*pos) else {
            return Err(err(self));
        };
        match ch {
            'n' => self.json_lit(c, pos, "null", NanBox::null()),
            't' => self.json_lit(c, pos, "true", NanBox::boolean(true)),
            'f' => self.json_lit(c, pos, "false", NanBox::boolean(false)),
            '"' => {
                let s = self.json_string(c, pos)?;
                Ok(self.new_str(&s))
            }
            '[' => {
                *pos += 1;
                let mut elems = Vec::new();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&']') {
                    *pos += 1;
                    return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
                }
                loop {
                    let v = self.json_parse(c, pos)?;
                    elems.push(v);
                    skip_ws(c, pos);
                    match c.get(*pos) {
                        Some(',') => *pos += 1,
                        Some(']') => {
                            *pos += 1;
                            break;
                        }
                        _ => return Err(self.json_error("Expected ',' or ']'")),
                    }
                }
                Ok(NanBox::handle(self.realm.new_array(elems).to_raw()))
            }
            '{' => {
                *pos += 1;
                let obj = self.realm.new_object();
                skip_ws(c, pos);
                if c.get(*pos) == Some(&'}') {
                    *pos += 1;
                    return Ok(NanBox::handle(obj.to_raw()));
                }
                loop {
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&'"') {
                        return Err(self.json_error("Expected property name"));
                    }
                    let key = self.json_string(c, pos)?;
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&':') {
                        return Err(self.json_error("Expected ':'"));
                    }
                    *pos += 1;
                    let v = self.json_parse(c, pos)?;
                    self.realm.set_property(obj, &key, v);
                    skip_ws(c, pos);
                    match c.get(*pos) {
                        Some(',') => *pos += 1,
                        Some('}') => {
                            *pos += 1;
                            break;
                        }
                        _ => return Err(self.json_error("Expected ',' or '}'")),
                    }
                }
                Ok(NanBox::handle(obj.to_raw()))
            }
            '-' | '0'..='9' => {
                let start = *pos;
                if c.get(*pos) == Some(&'-') {
                    *pos += 1;
                }
                while c
                    .get(*pos)
                    .is_some_and(|d| d.is_ascii_digit() || matches!(d, '.' | 'e' | 'E' | '+' | '-'))
                {
                    *pos += 1;
                }
                let text: String = c[start..*pos].iter().collect();
                text.parse::<f64>()
                    .map(NanBox::number)
                    .map_err(|_| self.json_error("Invalid number in JSON"))
            }
            _ => Err(self.json_error("Unexpected token in JSON")),
        }
    }

    fn json_lit(
        &mut self,
        c: &[char],
        pos: &mut usize,
        word: &str,
        value: NanBox,
    ) -> Result<NanBox, ExecError> {
        if c[*pos..].iter().take(word.len()).copied().eq(word.chars()) {
            *pos += word.len();
            Ok(value)
        } else {
            Err(self.json_error("Unexpected token in JSON"))
        }
    }

    /// Parses a JSON string literal (the opening `"` is at `pos`), handling the
    /// standard escapes.
    fn json_string(&mut self, c: &[char], pos: &mut usize) -> Result<String, ExecError> {
        *pos += 1; // opening quote
        let mut out = String::new();
        loop {
            match c.get(*pos) {
                None => {
                    return Err(self.json_error("Unterminated string in JSON"));
                }
                Some('"') => {
                    *pos += 1;
                    return Ok(out);
                }
                Some('\\') => {
                    *pos += 1;
                    match c.get(*pos) {
                        Some('"') => out.push('"'),
                        Some('\\') => out.push('\\'),
                        Some('/') => out.push('/'),
                        Some('n') => out.push('\n'),
                        Some('t') => out.push('\t'),
                        Some('r') => out.push('\r'),
                        Some('b') => out.push('\u{8}'),
                        Some('f') => out.push('\u{c}'),
                        Some('u') => {
                            let hex: String =
                                c.get(*pos + 1..*pos + 5).unwrap_or(&[]).iter().collect();
                            let code = u32::from_str_radix(&hex, 16)
                                .ok()
                                .and_then(char::from_u32)
                                .ok_or_else(|| self.json_error("Invalid \\u escape in JSON"))?;
                            out.push(code);
                            *pos += 4;
                        }
                        _ => return Err(self.json_error("Invalid escape in JSON")),
                    }
                    *pos += 1;
                }
                Some(&ch) => {
                    out.push(ch);
                    *pos += 1;
                }
            }
        }
    }

    /// Interpreter-aware `JSON.stringify` (compact): honors a `toJSON` method and
    /// invokes getters, unlike the realm-only `json_stringify`.
    fn json_to_string(&mut self, v: NanBox) -> Result<Option<String>, ExecError> {
        self.json_to_string_seen(v, "", &mut Vec::new())
    }

    /// `JSON.stringify` serialization tracking the ancestor handles in `seen`, so a
    /// circular structure throws a `TypeError` rather than overflowing the stack.
    /// `key` is the property name under which `v` appears in its parent (`""` at the
    /// top level), passed to a `toJSON(key)` method.
    fn json_to_string_seen(
        &mut self,
        v: NanBox,
        key: &str,
        seen: &mut Vec<Handle>,
    ) -> Result<Option<String>, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            // A primitive-wrapper object (`new Number`/`String`/`Boolean`) serializes
            // as its boxed primitive.
            if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
                return self.json_to_string_seen(prim, key, seen);
            }
            // A `Date` serializes as its ISO string (its built-in `toJSON`).
            if let Some(ms) = self.realm.date_at(h) {
                return Ok(Some(if ms.is_finite() {
                    json_quote(&crate::realm::date_to_iso(ms))
                } else {
                    String::from("null")
                }));
            }
            // A `BigInt` cannot be serialized to JSON.
            if self.realm.bigint_at(h).is_some() {
                let m = self.new_str("Do not know how to serialize a BigInt");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        // A `toJSON` method replaces the value before serialization.
        if let Some(h) = v.as_handle().map(Handle::from_raw)
            && self.realm.string_value(h).is_none()
            && !self.realm.is_array(h)
            && self.realm.object_keys(h).is_some()
        {
            let tj = self.read_member(h, "toJSON")?;
            if tj
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let key_box = self.new_str(key);
                let r = self.call_with_this(tj, v, &[key_box])?;
                return self.json_to_string_seen(r, key, seen);
            }
        }
        match v.unpack() {
            Unpacked::Undefined => Ok(None),
            Unpacked::Null => Ok(Some(String::from("null"))),
            Unpacked::Bool(b) => Ok(Some(String::from(if b { "true" } else { "false" }))),
            // Use the spec ToString (`0` for `-0`, exponential for ≥ 1e21, …);
            // non-finite numbers serialize as `null`.
            Unpacked::Number(n) => Ok(Some(if n.is_finite() {
                self.realm.to_display_string(v)
            } else {
                String::from("null")
            })),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(s) = self.realm.string_value(h) {
                    return Ok(Some(json_quote(&s)));
                }
                // A container that is already an ancestor is a cycle → TypeError.
                if (self.realm.array_elements(h).is_some() || self.realm.object_keys(h).is_some())
                    && seen.contains(&h)
                {
                    let m = self.new_str("Converting circular structure to JSON");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                    seen.push(h);
                    let mut parts = Vec::with_capacity(elems.len());
                    for (i, e) in elems.into_iter().enumerate() {
                        parts.push(
                            self.json_to_string_seen(e, &alloc::format!("{i}"), seen)?
                                .unwrap_or_else(|| String::from("null")),
                        );
                    }
                    seen.pop();
                    return Ok(Some(alloc::format!("[{}]", parts.join(","))));
                }
                if self.realm.object_keys(h).is_some() {
                    // Enumerable keys (incl. accessors), read via read_member so
                    // getters are invoked.
                    let keys = self.realm.object_keys(h).unwrap_or_default();
                    seen.push(h);
                    let mut parts = Vec::new();
                    for k in keys {
                        let val = self.read_member(h, &k)?;
                        if let Some(s) = self.json_to_string_seen(val, &k, seen)? {
                            parts.push(alloc::format!("{}:{}", json_quote(&k), s));
                        }
                    }
                    seen.pop();
                    return Ok(Some(alloc::format!("{{{}}}", parts.join(","))));
                }
                Ok(None) // a function
            }
        }
    }

    /// The underlying realm (e.g. to render a result with `to_display_string`).
    #[must_use]
    pub fn realm(&self) -> &Realm {
        &self.realm
    }

    /// Captures the object graph reachable from `roots` (the heap objects among
    /// them — primitives are skipped) and serializes it to portable bytes: a D′
    /// snapshot of live values that can later be reloaded into a fresh interpreter
    /// holding the same code (see [`restore_snapshot`](Self::restore_snapshot)).
    #[must_use]
    pub fn snapshot(&self, roots: &[NanBox]) -> Vec<u8> {
        let handles: Vec<Handle> = roots
            .iter()
            .filter_map(|v| v.as_handle().map(Handle::from_raw))
            .collect();
        crate::snapshot::serialize(&crate::snapshot::capture(&self.realm, &handles))
    }

    /// Reloads a snapshot produced by [`snapshot`](Self::snapshot) into this
    /// interpreter, returning the restored root values in the order their (heap)
    /// roots were captured. The restored objects are live — a restored closure runs
    /// and carries its snapshotted captured state.
    ///
    /// # Errors
    /// [`SnapError`](crate::snapshot::SnapError) if `bytes` is not a valid snapshot.
    pub fn restore_snapshot(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<NanBox>, crate::snapshot::SnapError> {
        let snap = crate::snapshot::deserialize(bytes)?;
        let handles = crate::snapshot::restore(&mut self.realm, &snap);
        Ok(handles
            .into_iter()
            .map(|h| NanBox::handle(h.to_raw()))
            .collect())
    }

    /// Runs a whole program, returning the value of its last expression
    /// statement (or `undefined`).
    pub fn run(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        self.strict = self.strict || has_use_strict(&program.body);
        self.hoist_with(&program.body, true)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = v,
                Flow::Return(v) => {
                    self.run_event_loop()?;
                    return Ok(v);
                }
                Flow::Break(_) | Flow::Continue(_) => {}
            }
        }
        // Run the event loop (microtasks + `setTimeout`) before returning.
        self.run_event_loop()?;
        Ok(last)
    }

    // --- functions ---

    /// Pre-declares hoisted `function` declarations in the current scope, so a
    /// declaration is callable before its textual position (and mutual
    /// recursion works).
    /// Hoists a statement sequence. Function declarations are always hoisted;
    /// `var` names hoist only at a function/program boundary (`hoist_vars`), not
    /// per-block, since `var` is function-scoped.
    fn hoist_with(&mut self, stmts: &'a [Stmt], hoist_vars: bool) -> Result<(), ExecError> {
        // `var` names hoist to the function/program scope as `undefined` (so a
        // read before the declaration yields `undefined`, not a ReferenceError).
        // Done first; a same-named function declaration then overwrites it.
        if hoist_vars {
            let mut var_names: Vec<&str> = Vec::new();
            collect_var_names(stmts, &mut var_names);
            // Annex B: a function declared inside a block also var-hoists its name
            // to the enclosing function scope (initially `undefined`).
            collect_block_function_names(stmts, &mut var_names);
            for name in var_names {
                if !self.current.has_local(name) {
                    self.current.declare(name, NanBox::undefined());
                }
            }
        }
        for stmt in stmts {
            if let Stmt::Function(func) = stmt
                && let Some(id) = &func.id
            {
                let value = self.make_function(
                    &func.params,
                    Body::Block(&func.body),
                    func.is_async,
                    func.is_generator,
                );
                self.set_fn_name(value, &id.name);
                if hoist_vars {
                    // A function/program top-level declaration binds here.
                    self.current.declare(&id.name, value);
                } else {
                    // A block-level declaration (Annex B) assigns the function-scope
                    // `var` binding hoisted above; if none exists, bind locally.
                    if !self.current.set(&id.name, value) {
                        self.current.declare(&id.name, value);
                    }
                }
            }
        }
        Ok(())
    }

    /// Block-level hoisting: function declarations only (`var` is function-scoped
    /// and hoisted at the function/program boundary instead).
    fn hoist(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        self.hoist_with(stmts, false)
    }

    /// Registers a function definition and allocates a closure capturing the
    /// current scope.
    fn make_function(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        is_generator: bool,
    ) -> NanBox {
        self.make_method(params, body, is_async, is_generator, None, false)
    }

    fn make_method(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        is_generator: bool,
        home_class: Option<u32>,
        home_static: bool,
    ) -> NanBox {
        // Strict mode is lexical: inherited from the defining context, or set by
        // the function body's own `"use strict"` directive prologue.
        let is_strict = self.strict || matches!(body, Body::Block(stmts) if has_use_strict(stmts));
        let func_id = self.functions.len() as u32;
        self.functions.push(FnDef {
            params,
            body,
            is_async,
            is_generator,
            is_arrow: false,
            is_strict,
            name: "",
            home_class,
            home_static,
        });
        let handle = self.realm.new_function(func_id, self.current.clone());
        NanBox::handle(handle.to_raw())
    }

    /// Calls `callee` with `args`.
    fn call(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        self.call_with_this(callee, NanBox::undefined(), args)
    }

    /// Builds an iterator object over a generator's eagerly-collected `values`:
    /// a hidden buffer array plus a `next()` cursor, recognized by `for-of`,
    /// spread, and a `next()` method.
    fn make_generator(&mut self, values: Vec<NanBox>) -> NanBox {
        self.make_generator_with_return(values, NanBox::undefined())
    }

    /// Like [`make_generator`], but with the generator's `return` value (surfaced
    /// once, with `done: true`, after the yields are exhausted).
    fn make_generator_with_return(&mut self, values: Vec<NanBox>, ret: NanBox) -> NanBox {
        let obj = self.realm.new_object();
        let buf = self.realm.new_array(values);
        self.realm
            .set_hidden_property(obj, GEN_BUF, NanBox::handle(buf.to_raw()));
        self.realm
            .set_hidden_property(obj, GEN_IDX, NanBox::number(0.0));
        self.realm.set_hidden_property(obj, GEN_RET, ret);
        NanBox::handle(obj.to_raw())
    }

    // --- promises ---

    /// Settles the promise at `handle` (no-op if already settled), queuing its
    /// reactions as microtasks.
    fn settle(&mut self, handle: Handle, value: NanBox, fulfilled: bool) {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = self.realm.promise_state(handle) else {
            return;
        };
        let reactions = {
            let mut s = state.borrow_mut();
            if s.status != Pending {
                return;
            }
            s.status = if fulfilled { Fulfilled } else { Rejected };
            s.value = value;
            core::mem::take(&mut s.reactions)
        };
        for r in reactions {
            let handler = if fulfilled {
                r.on_fulfilled
            } else {
                r.on_rejected
            };
            self.microtasks.push(Job {
                handler,
                value,
                result: r.result,
                fulfilled,
                finally: r.finally,
            });
        }
    }

    /// Resolves `handle` with `value`, adopting it if `value` is itself a
    /// promise (chain on its settlement).
    fn resolve_with(&mut self, handle: Handle, value: NanBox) {
        let inner = value
            .as_handle()
            .map(Handle::from_raw)
            .filter(|h| self.realm.promise_state(*h).is_some());
        if let Some(inner) = inner {
            // Adopt: when `inner` settles, settle `handle` the same way.
            let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
            let on_r = self.realm.new_bound_native(N_REJECT, handle);
            self.register_then(
                inner,
                NanBox::handle(on_f.to_raw()),
                NanBox::handle(on_r.to_raw()),
                false,
            );
            return;
        }
        // A thenable (a non-promise object with a callable `then`) is adopted by
        // calling `then(resolve, reject)`.
        if let Some(vh) = value.as_handle().map(Handle::from_raw)
            && let Some(then) = self.realm.get_property(vh, "then")
            && then
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let on_f = self.realm.new_bound_native(N_RESOLVE, handle);
            let on_r = self.realm.new_bound_native(N_REJECT, handle);
            let args = [NanBox::handle(on_f.to_raw()), NanBox::handle(on_r.to_raw())];
            // A throw from `then` rejects the promise.
            if let Err(ExecError::Throw(e)) = self.call_with_this(then, value, &args) {
                self.settle(handle, e, false);
            }
            return;
        }
        self.settle(handle, value, true);
    }

    /// Registers `then` reactions on `handle`, returning a new dependent promise.
    fn promise_then(&mut self, handle: Handle, on_f: NanBox, on_r: NanBox) -> NanBox {
        let result = self.register_then(handle, on_f, on_r, false);
        NanBox::handle(result.to_raw())
    }

    fn register_then(
        &mut self,
        handle: Handle,
        on_f: NanBox,
        on_r: NanBox,
        finally: bool,
    ) -> Handle {
        use crate::cell::PromiseStatus::{Fulfilled, Pending};
        let result = self.realm.new_promise();
        let state = self.realm.promise_state(handle).expect("a promise");
        let settled = {
            let s = state.borrow();
            match s.status {
                Pending => None,
                status => Some((status == Fulfilled, s.value)),
            }
        };
        match settled {
            None => state.borrow_mut().reactions.push(crate::cell::Reaction {
                on_fulfilled: on_f,
                on_rejected: on_r,
                result,
                finally,
            }),
            Some((fulfilled, value)) => {
                let handler = if fulfilled { on_f } else { on_r };
                self.microtasks.push(Job {
                    handler,
                    value,
                    result,
                    fulfilled,
                    finally,
                });
            }
        }
        result
    }

    /// Drains the microtask queue (the event loop), running each promise
    /// reaction to completion.
    fn drain_microtasks(&mut self) -> Result<(), ExecError> {
        while !self.microtasks.is_empty() {
            self.run_one_microtask()?;
        }
        Ok(())
    }

    /// Runs the earliest-due `setTimeout` macrotask (least `delay`, ties by
    /// insertion order). A no-op when none are pending.
    fn run_one_macrotask(&mut self) -> Result<(), ExecError> {
        let Some(idx) = self
            .macrotasks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.delay.total_cmp(&b.delay).then(a.seq.cmp(&b.seq)))
            .map(|(i, _)| i)
        else {
            return Ok(());
        };
        let t = self.macrotasks.remove(idx);
        self.call(t.callback, &t.args)?;
        Ok(())
    }

    /// Runs the event loop to quiescence: drain all microtasks, then run the
    /// earliest-due `setTimeout` macrotask (draining microtasks after each), until
    /// both queues are empty.
    fn run_event_loop(&mut self) -> Result<(), ExecError> {
        self.drain_microtasks()?;
        while !self.macrotasks.is_empty() {
            self.run_one_macrotask()?;
            self.drain_microtasks()?;
        }
        Ok(())
    }

    /// Runs the next queued promise reaction.
    fn run_one_microtask(&mut self) -> Result<(), ExecError> {
        let job = self.microtasks.remove(0);
        if job.finally
            && job
                .handler
                .as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.is_callable(h))
        {
            // `finally`: run the callback (no args), then pass the original
            // value/rejection through (a throw from the callback overrides it).
            match self.call(job.handler, &[]) {
                Ok(_) => {
                    if job.fulfilled {
                        self.resolve_with(job.result, job.value);
                    } else {
                        self.settle(job.result, job.value, false);
                    }
                }
                Err(ExecError::Throw(e)) => self.settle(job.result, e, false),
                Err(other) => return Err(other),
            }
        } else if job
            .handler
            .as_handle()
            .map(Handle::from_raw)
            .is_some_and(|h| self.is_callable(h))
        {
            match self.call(job.handler, &[job.value]) {
                Ok(v) => self.resolve_with(job.result, v),
                Err(ExecError::Throw(e)) => self.settle(job.result, e, false),
                Err(other) => return Err(other),
            }
        } else if job.fulfilled {
            // Passthrough: settle with the same status/value.
            self.resolve_with(job.result, job.value);
        } else {
            self.settle(job.result, job.value, false);
        }
        Ok(())
    }

    /// `await value` — for a promise, drains microtasks until it settles (this
    /// model has no timers, so all promises settle via the queue), then yields
    /// its value or throws its rejection. A non-promise passes through.
    /// The current settled state of `value`: `Some(Ok(v))` if fulfilled (a
    /// non-promise counts as fulfilled with itself), `Some(Err(e))` if rejected,
    /// `None` if it is a still-pending promise.
    fn settled_state(&self, value: NanBox) -> Option<Result<NanBox, NanBox>> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        else {
            return Some(Ok(value));
        };
        let s = state.borrow();
        match s.status {
            Fulfilled => Some(Ok(s.value)),
            Rejected => Some(Err(s.value)),
            Pending => None,
        }
    }

    fn await_value(&mut self, value: NanBox) -> Result<NanBox, ExecError> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        else {
            return Ok(value); // not a promise
        };
        // Make progress on the event loop until the promise settles: drain
        // microtasks first, then run a `setTimeout` macrotask if still pending (so an
        // `await` / `Promise.all` on a timer-backed promise observes its value).
        while state.borrow().status == Pending
            && (!self.microtasks.is_empty() || !self.macrotasks.is_empty())
        {
            if self.microtasks.is_empty() {
                self.run_one_macrotask()?;
            } else {
                self.run_one_microtask()?;
            }
        }
        let s = state.borrow();
        match s.status {
            Fulfilled => Ok(s.value),
            Rejected => Err(ExecError::Throw(s.value)),
            Pending => Ok(NanBox::undefined()), // never settles
        }
    }

    /// Throws a `TypeError` if `handle` is a revoked proxy (used to guard every
    /// proxy operation).
    fn guard_revoked(&mut self, handle: Handle) -> Result<(), ExecError> {
        if self.realm.proxy_revoked(handle) {
            let m = self.new_str("Cannot perform operation on a revoked proxy");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(())
    }

    /// Applies a property descriptor object (`{ value }` or `{ get, set }`) to
    /// `obj[key]` — shared by `Object.defineProperty`/`defineProperties`.
    /// Builds the property descriptor object for own property `key` of `obj`
    /// (accessor or data), or `None` if `key` is not an own property.
    fn build_descriptor(&mut self, obj: Handle, key: &str) -> Option<NanBox> {
        let t = NanBox::boolean(true);
        // An array index / `length` is a data property (not stored as a named slot):
        // an in-range index is writable, enumerable, configurable; `length` is
        // writable but non-enumerable and non-configurable.
        if let Some(len) = self.realm.array_length(obj) {
            let (value, enumerable, configurable) = if key == "length" {
                (Some(NanBox::number(len as f64)), false, false)
            } else if let Ok(i) = key.parse::<usize>() {
                if i < len && alloc::format!("{i}") == key {
                    (Some(self.realm.get_element(obj, i)), true, true)
                } else {
                    (None, false, false)
                }
            } else {
                (None, false, false)
            };
            if let Some(v) = value {
                let d = self.realm.new_object();
                self.realm.set_property(d, "value", v);
                self.realm.set_property(d, "writable", t);
                self.realm
                    .set_property(d, "enumerable", NanBox::boolean(enumerable));
                self.realm
                    .set_property(d, "configurable", NanBox::boolean(configurable));
                return Some(NanBox::handle(d.to_raw()));
            }
        }
        let configurable = NanBox::boolean(!self.realm.property_is_non_configurable(obj, key));
        if let Some((g, s)) = self.realm.accessor(obj, key) {
            let d = self.realm.new_object();
            self.realm.set_property(d, "get", g);
            self.realm.set_property(d, "set", s);
            self.realm.set_property(d, "enumerable", t);
            self.realm.set_property(d, "configurable", configurable);
            Some(NanBox::handle(d.to_raw()))
        } else if self.realm.has_own(obj, key) {
            let v = self
                .realm
                .get_property(obj, key)
                .unwrap_or(NanBox::undefined());
            let writable = !self.realm.property_is_readonly(obj, key);
            let enumerable = self.realm.property_is_enumerable(obj, key);
            let d = self.realm.new_object();
            self.realm.set_property(d, "value", v);
            self.realm
                .set_property(d, "writable", NanBox::boolean(writable));
            self.realm
                .set_property(d, "enumerable", NanBox::boolean(enumerable));
            self.realm.set_property(d, "configurable", configurable);
            Some(NanBox::handle(d.to_raw()))
        } else {
            None
        }
    }

    /// `Object/Reflect.getOwnPropertyDescriptor(obj, key)` — routing a proxy
    /// through its `getOwnPropertyDescriptor` trap (or forwarding to the target),
    /// else building the descriptor from the own property.
    fn descriptor_of(&mut self, obj: Handle, key: &str) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            let trap = self
                .realm
                .get_property(handler, "getOwnPropertyDescriptor")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let key_v = self.new_str(key);
                return self.call(trap, &[NanBox::handle(target.to_raw()), key_v]);
            }
            return self.descriptor_of(target, key);
        }
        Ok(self
            .build_descriptor(obj, key)
            .unwrap_or(NanBox::undefined()))
    }

    /// `Object/Reflect.isExtensible(obj)` — routing a proxy through its
    /// `isExtensible` trap (or forwarding to the target).
    fn is_extensible_of(&mut self, obj: Handle) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            let trap = self
                .realm
                .get_property(handler, "isExtensible")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let r = self.call(trap, &[NanBox::handle(target.to_raw())])?;
                return Ok(NanBox::boolean(self.realm.truthy(r)));
            }
            return Ok(NanBox::boolean(self.realm.is_extensible(target)));
        }
        Ok(NanBox::boolean(self.realm.is_extensible(obj)))
    }

    /// `Object/Reflect.setPrototypeOf(obj, proto)` — routing a proxy through its
    /// `setPrototypeOf` trap (or forwarding to the target).
    /// `Object.getPrototypeOf` / `Reflect.getPrototypeOf`, honoring a proxy's
    /// `getPrototypeOf` trap (else forwarding to the target / reading the link).
    fn get_proto_of(&mut self, obj: Handle) -> Result<NanBox, ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            let trap = self
                .realm
                .get_property(handler, "getPrototypeOf")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                return self.call(trap, &[NanBox::handle(target.to_raw())]);
            }
            return Ok(self
                .realm
                .object_proto(target)
                .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw())));
        }
        Ok(self
            .realm
            .object_proto(obj)
            .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw())))
    }

    fn set_proto_of(&mut self, obj: Handle, proto: Option<Handle>) -> Result<(), ExecError> {
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            let trap = self
                .realm
                .get_property(handler, "setPrototypeOf")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let proto_box = proto.map_or(NanBox::null(), |p| NanBox::handle(p.to_raw()));
                self.call(trap, &[NanBox::handle(target.to_raw()), proto_box])?;
                return Ok(());
            }
            self.realm.set_object_proto(target, proto);
            return Ok(());
        }
        self.realm.set_object_proto(obj, proto);
        Ok(())
    }

    /// Whether redefining own property `key` with descriptor `desc` would change
    /// nothing — every field the descriptor *specifies* already matches the current
    /// property. Such a no-op redefine is permitted even on a non-configurable property.
    fn redefine_is_noop(
        &mut self,
        obj: Handle,
        key: &str,
        desc: Handle,
        is_accessor: bool,
        writable: bool,
    ) -> Result<bool, ExecError> {
        let truthy_field = |this: &mut Self, name: &str| {
            this.realm
                .get_property(desc, name)
                .is_some_and(|v| this.realm.truthy(v))
        };
        // Switching kind (data <-> accessor) is a change.
        let wants_accessor = self.realm.has_own(desc, "get") || self.realm.has_own(desc, "set");
        if wants_accessor != is_accessor {
            return Ok(false);
        }
        // Making a non-configurable property configurable is a change.
        if self.realm.has_own(desc, "configurable") && truthy_field(self, "configurable") {
            return Ok(false);
        }
        if self.realm.has_own(desc, "enumerable")
            && truthy_field(self, "enumerable") != self.realm.property_is_enumerable(obj, key)
        {
            return Ok(false);
        }
        if is_accessor {
            let (cur_get, cur_set) = self
                .realm
                .accessor(obj, key)
                .unwrap_or((NanBox::undefined(), NanBox::undefined()));
            if self.realm.has_own(desc, "get") {
                let g = self
                    .realm
                    .get_property(desc, "get")
                    .unwrap_or(NanBox::undefined());
                if !self.realm.strict_equals(g, cur_get) {
                    return Ok(false);
                }
            }
            if self.realm.has_own(desc, "set") {
                let s = self
                    .realm
                    .get_property(desc, "set")
                    .unwrap_or(NanBox::undefined());
                if !self.realm.strict_equals(s, cur_set) {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        // A data property: a writable flip, or (for a non-writable one) a value change.
        if self.realm.has_own(desc, "writable") && truthy_field(self, "writable") != writable {
            return Ok(false);
        }
        if !writable && self.realm.has_own(desc, "value") {
            let new_val = self
                .realm
                .get_property(desc, "value")
                .unwrap_or(NanBox::undefined());
            let cur_val = self
                .realm
                .get_property(obj, key)
                .unwrap_or(NanBox::undefined());
            // SameValue (distinguishes NaN and ±0 from `===`).
            let same = match (new_val.as_number(), cur_val.as_number()) {
                (Some(x), Some(y)) => {
                    (x == y && (x != 0.0 || x.is_sign_positive() == y.is_sign_positive()))
                        || (x.is_nan() && y.is_nan())
                }
                _ => self.realm.strict_equals(new_val, cur_val),
            };
            if !same {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn apply_descriptor(&mut self, obj: Handle, key: &str, desc: Handle) -> Result<(), ExecError> {
        // A proxy routes `Object.defineProperty` through its `defineProperty` trap
        // (called `trap(target, key, descriptor)`), or forwards to the target.
        if let Some((target, handler)) = self.realm.proxy_at(obj) {
            self.guard_revoked(obj)?;
            let trap = self
                .realm
                .get_property(handler, "defineProperty")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let key_v = self.new_str(key);
                self.call(
                    trap,
                    &[
                        NanBox::handle(target.to_raw()),
                        key_v,
                        NanBox::handle(desc.to_raw()),
                    ],
                )?;
                return Ok(());
            }
            return self.apply_descriptor(target, key, desc);
        }
        // A descriptor may not mix accessor fields (`get`/`set`) with data fields
        // (`value`/`writable`) — that is an invalid descriptor (ToPropertyDescriptor).
        let has_accessor_field = self.realm.has_own(desc, "get") || self.realm.has_own(desc, "set");
        let has_data_field =
            self.realm.has_own(desc, "value") || self.realm.has_own(desc, "writable");
        if has_accessor_field && has_data_field {
            let m = self.new_str(
                "Invalid property descriptor. Cannot both specify accessors and a value or writable attribute",
            );
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        let is_own = self.realm.has_own(obj, key) || self.realm.accessor(obj, key).is_some();
        // Adding a *new* property to a non-extensible object is a TypeError.
        if !is_own && !self.realm.is_extensible(obj) {
            let m = self.new_str("Cannot define property: object is not extensible");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // Redefining a non-configurable property is a TypeError — except a
        // non-configurable *writable data* property, whose value may still change.
        if is_own && self.realm.property_is_non_configurable(obj, key) {
            let is_accessor = self.realm.accessor(obj, key).is_some();
            let writable = !self.realm.property_is_readonly(obj, key);
            if (is_accessor || !writable)
                && !self.redefine_is_noop(obj, key, desc, is_accessor, writable)?
            {
                let m = self.new_str("Cannot redefine non-configurable property");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        let getter = self.realm.get_property(desc, "get");
        let setter = self.realm.get_property(desc, "set");
        if getter.is_some() || setter.is_some() {
            self.realm.define_accessor(
                obj,
                key,
                getter.unwrap_or(NanBox::undefined()),
                setter.unwrap_or(NanBox::undefined()),
            );
            // An accessor descriptor is non-enumerable unless `enumerable: true`.
            let enumerable = self
                .realm
                .get_property(desc, "enumerable")
                .is_some_and(|v| self.realm.truthy(v));
            if !enumerable {
                self.realm.mark_hidden(obj, key);
            }
        } else {
            // Redefining as a data property removes any prior accessor.
            self.realm.clear_accessor(obj, key);
            let value = self
                .realm
                .get_property(desc, "value")
                .unwrap_or(NanBox::undefined());
            // A `defineProperty` redefines attributes from scratch: drop any prior
            // non-writable mark so the new value takes effect, then set it.
            self.realm.clear_readonly_property(obj, key);
            self.realm.set_property(obj, key, value);
            // A data descriptor defaults to non-writable unless `writable: true`.
            let writable = self
                .realm
                .get_property(desc, "writable")
                .is_some_and(|v| self.realm.truthy(v));
            if !writable {
                self.realm.set_readonly_property(obj, key);
            }
            // A descriptor defaults to non-enumerable unless `enumerable: true`.
            let enumerable = self
                .realm
                .get_property(desc, "enumerable")
                .is_some_and(|v| self.realm.truthy(v));
            if !enumerable {
                self.realm.mark_hidden(obj, key);
            }
        }
        // A descriptor defaults to non-configurable unless `configurable: true`
        // (so the property cannot be deleted).
        let configurable = self
            .realm
            .get_property(desc, "configurable")
            .is_some_and(|v| self.realm.truthy(v));
        if !configurable {
            self.realm.set_non_configurable_property(obj, key);
        }
        Ok(())
    }

    fn is_callable(&self, handle: Handle) -> bool {
        self.realm.native_at(handle).is_some()
            || self.realm.function_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
            // A bound function (`fn.bind(...)`) is callable.
            || self.realm.get_property(handle, BOUND_TARGET).is_some()
            // A proxy is callable iff its target is.
            || self
                .realm
                .proxy_at(handle)
                .is_some_and(|(t, _)| self.is_callable(t))
    }

    /// Builds a bound function (`Function.prototype.bind`): an object recording
    /// the target, the bound `this`, and the leading bound arguments under
    /// reserved hidden keys. Calling it forwards to the target.
    fn make_bound_function(
        &mut self,
        target: NanBox,
        this_val: NanBox,
        bound: Vec<NanBox>,
    ) -> NanBox {
        let obj = self.realm.new_object();
        self.realm.set_hidden_property(obj, BOUND_TARGET, target);
        self.realm.set_hidden_property(obj, BOUND_THIS, this_val);
        let arr = self.realm.new_array(bound);
        self.realm
            .set_hidden_property(obj, BOUND_ARGS, NanBox::handle(arr.to_raw()));
        NanBox::handle(obj.to_raw())
    }

    /// Calls `callee` with an explicit `this` (a method receiver, or `undefined`
    /// for a plain call).
    fn call_with_this(
        &mut self,
        callee: NanBox,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = callee.as_handle() else {
            return Err(ExecError::NotCallable);
        };
        let handle = Handle::from_raw(raw);
        // `Array(...)` without `new` behaves like `new Array(...)`.
        if self.current.get("Array").and_then(|v| v.as_handle()) == callee.as_handle() {
            return self.construct(callee, args);
        }
        // `Object(value)` (ToObject): `null`/`undefined` → a new object; an object is
        // returned as-is; a primitive is boxed in its wrapper.
        if self.current.get("Object").and_then(|v| v.as_handle()) == callee.as_handle() {
            let v = args.first().copied().unwrap_or(NanBox::undefined());
            return Ok(self.coerce_to_object(v));
        }
        // A bound function: prepend the bound `this`/args and forward.
        if let Some(target) = self.realm.get_property(handle, BOUND_TARGET) {
            let bthis = self
                .realm
                .get_property(handle, BOUND_THIS)
                .unwrap_or(NanBox::undefined());
            let mut all = self
                .realm
                .get_property(handle, BOUND_ARGS)
                .and_then(|a| a.as_handle())
                .map(Handle::from_raw)
                .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                .unwrap_or_default();
            all.extend_from_slice(args);
            return self.call_with_this(target, bthis, &all);
        }
        // A callable proxy: route through its `apply` trap, or call the target.
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "apply")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let arr = self.realm.new_array(args.to_vec());
                return self.call(
                    trap,
                    &[
                        NanBox::handle(target.to_raw()),
                        this_val,
                        NanBox::handle(arr.to_raw()),
                    ],
                );
            }
            return self.call_with_this(NanBox::handle(target.to_raw()), this_val, args);
        }
        // A built-in function dispatches directly, with the receiver available as
        // `this` (for the `Object.prototype.*` methods called via `.call`).
        if let Some(id) = self.realm.native_at(handle) {
            let saved = core::mem::replace(&mut self.this_val, this_val);
            let r = self.call_native(id, args);
            self.this_val = saved;
            return r;
        }
        // A bound native (promise resolve/reject) carries its target.
        if let Some((id, target)) = self.realm.bound_native_at(handle) {
            // A first-class `Array.prototype.<method>`: run that array method on the
            // call's `this` (e.g. `Array.prototype.slice.call(arguments)`).
            if id == N_ARRAY_PROTO_FN {
                let name = self.realm.string_value(target).unwrap_or_default();
                return Ok(self
                    .call_method(this_val, &name, args)?
                    .unwrap_or(NanBox::undefined()));
            }
            // A WASM export wrapper: decode the carried module, instantiate, and
            // invoke the named export through the JS-value boundary.
            if id == N_WASM_CALL {
                return self.call_wasm_export(target, args);
            }
            // `WebAssembly.Global` `.value` getter / setter (bound to the global).
            if id == N_WASM_GLOBAL_GET {
                return Ok(self
                    .realm
                    .get_property(target, WASM_GLOBAL_VALUE)
                    .unwrap_or(NanBox::undefined()));
            }
            if id == N_WASM_GLOBAL_SET {
                if !self
                    .realm
                    .get_property(target, WASM_GLOBAL_MUTABLE)
                    .is_some_and(|v| self.realm.truthy(v))
                {
                    return Err(self.wasm_type_error("WebAssembly.Global is immutable"));
                }
                let new_val = args.first().copied().unwrap_or(NanBox::undefined());
                let ty = self
                    .realm
                    .get_property(target, WASM_GLOBAL_TYPE)
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let coerced = self.wasm_coerce_global(&ty, new_val);
                self.realm
                    .set_hidden_property(target, WASM_GLOBAL_VALUE, coerced);
                return Ok(NanBox::undefined());
            }
            // `WebAssembly.Memory.prototype.buffer` getter.
            if id == N_WASM_MEM_BUFFER_GET {
                return Ok(self
                    .realm
                    .get_property(target, WASM_MEM_BUFFER)
                    .unwrap_or(NanBox::undefined()));
            }
            // `WebAssembly.Memory.prototype.grow(delta)` → old page count (a new,
            // larger `ArrayBuffer` replaces `.buffer`, old contents copied).
            if id == N_WASM_MEM_GROW {
                let delta = args
                    .first()
                    .map_or(0.0, |v| self.realm.to_number(*v))
                    .max(0.0) as usize;
                let old_pages = self
                    .realm
                    .get_property(target, WASM_MEM_PAGES)
                    .and_then(|v| v.as_number())
                    .unwrap_or(0.0) as usize;
                let new_pages = old_pages + delta;
                if let Some(max) = self
                    .realm
                    .get_property(target, WASM_MEM_MAX)
                    .and_then(|v| v.as_number())
                    && new_pages as f64 > max
                {
                    let m = self.new_str("memory.grow exceeds the declared maximum");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                let new_buf = self.make_array_buffer(new_pages * WASM_PAGE);
                // Copy the existing bytes into the new (zero-extended) buffer.
                if let Some(old_bytes) = self
                    .realm
                    .get_property(target, WASM_MEM_BUFFER)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|ob| self.realm.get_property(ob, ARRAY_BUFFER_BYTES))
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                    && let Some(nb) = self
                        .realm
                        .get_property(new_buf, ARRAY_BUFFER_BYTES)
                        .and_then(|v| v.as_handle())
                        .map(Handle::from_raw)
                {
                    for (i, b) in old_bytes.into_iter().enumerate() {
                        self.realm.set_element(nb, i, b);
                    }
                }
                self.realm.set_hidden_property(
                    target,
                    WASM_MEM_BUFFER,
                    NanBox::handle(new_buf.to_raw()),
                );
                self.realm.set_hidden_property(
                    target,
                    WASM_MEM_PAGES,
                    NanBox::number(new_pages as f64),
                );
                return Ok(NanBox::number(old_pages as f64));
            }
            // `WebAssembly.Table` `.length` getter and `get`/`set`/`grow` methods
            // (bound to the table `target`), over its function-ref element array.
            if matches!(
                id,
                N_WASM_TABLE_LEN | N_WASM_TABLE_GET | N_WASM_TABLE_SET | N_WASM_TABLE_GROW
            ) {
                let Some(elems) = self
                    .realm
                    .get_property(target, WASM_TABLE_ELEMS)
                    .and_then(|v| v.as_handle())
                    .map(Handle::from_raw)
                else {
                    return Ok(NanBox::undefined());
                };
                let len = self.realm.array_length(elems).unwrap_or(0);
                let idx = self
                    .realm
                    .to_number(args.first().copied().unwrap_or(NanBox::undefined()));
                match id {
                    N_WASM_TABLE_LEN => return Ok(NanBox::number(len as f64)),
                    N_WASM_TABLE_GET | N_WASM_TABLE_SET => {
                        if idx < 0.0 || idx as usize >= len {
                            let m = self.new_str("WebAssembly.Table index out of bounds");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        let i = idx as usize;
                        if id == N_WASM_TABLE_GET {
                            return Ok(self.realm.get_element(elems, i));
                        }
                        let v = args.get(1).copied().unwrap_or(NanBox::null());
                        self.realm.set_element(elems, i, v);
                        return Ok(NanBox::undefined());
                    }
                    _ => {
                        // grow(delta, init?) → prior length.
                        let new_len = len + idx.max(0.0) as usize;
                        if let Some(max) = self
                            .realm
                            .get_property(target, WASM_TABLE_MAX)
                            .and_then(|v| v.as_number())
                            && new_len as f64 > max
                        {
                            let m =
                                self.new_str("WebAssembly.Table.grow exceeds the declared maximum");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        let init = args.get(1).copied().unwrap_or(NanBox::null());
                        for i in len..new_len {
                            self.realm.set_element(elems, i, init);
                        }
                        return Ok(NanBox::number(len as f64));
                    }
                }
            }
            let arg0 = args.first().copied().unwrap_or(NanBox::undefined());
            match id {
                N_RESOLVE => self.resolve_with(target, arg0),
                N_REJECT => self.settle(target, arg0, false),
                // The `revoke` function from `Proxy.revocable`.
                N_PROXY_REVOKE => self.realm.revoke_proxy(target),
                _ => {}
            }
            return Ok(NanBox::undefined());
        }
        let Some((func_id, captured)) = self.realm.function_at(handle) else {
            return Err(ExecError::NotCallable);
        };
        let def = self.functions[func_id as usize];
        // An object-literal concise method carries its `[[HomeObject]]`; bind it for
        // the duration of the call so `super.x` in the body resolves through it.
        let home_obj = self
            .realm
            .get_property(handle, HOME_OBJECT)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw);
        let saved_home_obj = core::mem::replace(&mut self.current_home_object, home_obj);
        let r = self.invoke(def, captured, this_val, args);
        self.current_home_object = saved_home_obj;
        r
    }

    /// Runs a function body with `this` and the parameters bound in a fresh
    /// child of `captured`.
    /// Invokes a function, guarding against unbounded recursion: beyond
    /// `MAX_CALL_DEPTH` nested calls it throws a `RangeError` instead of letting
    /// the host stack overflow.
    fn invoke(
        &mut self,
        def: FnDef<'a>,
        captured: Scope,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        if self.call_depth >= MAX_CALL_DEPTH {
            let msg = self.new_str("Maximum call stack size exceeded");
            // A proper `RangeError` object (id 2 in `ERROR_NAMES`) so `instanceof
            // RangeError`/`Error` and `.name` work on the caught value.
            let err = self.make_error(N_ERROR_BASE + 2, Some(msg));
            return Err(ExecError::Throw(err));
        }
        self.call_depth += 1;
        let r = self.invoke_inner(def, captured, this_val, args);
        self.call_depth -= 1;
        r
    }

    fn invoke_inner(
        &mut self,
        def: FnDef<'a>,
        captured: Scope,
        this_val: NanBox,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let call_scope = captured.child();
        let saved = core::mem::replace(&mut self.current, call_scope);
        // An arrow has no own `this` — it inherits the enclosing one lexically,
        // so leave `self.this_val` unchanged.
        let saved_this = if def.is_arrow {
            self.this_val
        } else {
            // Sloppy-mode `this` coercion: an `undefined`/`null` receiver becomes
            // the global object. Strict functions keep it as-is.
            let bound = if !def.is_strict
                && matches!(this_val.unpack(), Unpacked::Undefined | Unpacked::Null)
            {
                self.global_this
            } else {
                this_val
            };
            core::mem::replace(&mut self.this_val, bound)
        };
        let saved_home = core::mem::replace(&mut self.current_home, def.home_class);
        let saved_home_static = core::mem::replace(&mut self.current_home_static, def.home_static);
        // A non-arrow invocation establishes its own `new.target`: the constructor
        // when reached via `new` (passed through the one-shot `pending_new_target`),
        // else `undefined`. An arrow inherits the enclosing `new.target`.
        let saved_target = if def.is_arrow {
            self.new_target
        } else {
            let nt = self
                .pending_new_target
                .take()
                .unwrap_or(NanBox::undefined());
            core::mem::replace(&mut self.new_target, nt)
        };
        // A generator body runs eagerly into a fresh yield buffer.
        let saved_sink = if def.is_generator {
            Some(self.gen_sink.replace(Vec::new()))
        } else {
            None
        };
        let result = (|| {
            for (i, param) in def.params.iter().enumerate() {
                let value = if param.rest {
                    let rest = args[i.min(args.len())..].to_vec();
                    NanBox::handle(self.realm.new_array(rest).to_raw())
                } else {
                    let mut v = args.get(i).copied().unwrap_or(NanBox::undefined());
                    if matches!(v.unpack(), Unpacked::Undefined)
                        && let Some(d) = &param.default
                    {
                        v = self.eval(d)?;
                    }
                    v
                };
                self.bind_pattern(&param.target, value)?;
            }
            // A non-arrow function gets an `arguments` array-like of its call
            // arguments. (Arrows inherit the enclosing `arguments`.)
            if !def.is_arrow {
                let arr = self.realm.new_array(args.to_vec());
                self.current
                    .declare("arguments", NanBox::handle(arr.to_raw()));
            }
            self.run_body(def.body)
        })();
        self.current = saved;
        self.this_val = saved_this;
        self.current_home = saved_home;
        self.current_home_static = saved_home_static;
        self.new_target = saved_target;
        // A generator call returns an iterator over the values it yielded.
        if def.is_generator {
            let collected = self.gen_sink.take().unwrap_or_default();
            self.gen_sink = saved_sink.flatten();
            let ret = result?; // a throw during collection propagates at call time
            return Ok(self.make_generator_with_return(collected, ret));
        }
        // An `async` function returns a promise of its result (rejected on throw).
        if def.is_async {
            let promise = self.realm.new_promise();
            match result {
                Ok(v) => self.resolve_with(promise, v),
                Err(ExecError::Throw(e)) => self.settle(promise, e, false),
                Err(other) => return Err(other),
            }
            return Ok(NanBox::handle(promise.to_raw()));
        }
        result
    }

    /// `new Callee(args)` — supports the built-in `Map`/`Set` constructors
    /// (optionally seeded from an iterable argument).
    fn construct(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        let Some(raw) = callee.as_handle() else {
            return Err(ExecError::NotCallable);
        };
        let handle = Handle::from_raw(raw);
        // `new someProxy(...)`: route through the `construct` trap, or construct
        // the target.
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "construct")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let arr = self.realm.new_array(args.to_vec());
                let target_box = NanBox::handle(target.to_raw());
                return self.call(trap, &[target_box, NanBox::handle(arr.to_raw()), callee]);
            }
            return self.construct(NanBox::handle(target.to_raw()), args);
        }
        // `new boundFn(...)`: construct the bound target with the bound arguments
        // prepended (the bound `this` is ignored when constructing).
        if let Some(target) = self.realm.get_property(handle, BOUND_TARGET) {
            let mut all = Vec::new();
            if let Some(ba) = self.realm.get_property(handle, BOUND_ARGS)
                && let Some(bh) = ba.as_handle().map(Handle::from_raw)
                && let Some(elems) = self.realm.array_elements(bh)
            {
                all.extend_from_slice(elems);
            }
            all.extend_from_slice(args);
            return self.construct(target, &all);
        }
        // `new UserClass(...)`.
        if let Some((class_id, env)) = self.realm.class_at(handle) {
            // `new.target` inside the class constructor is the class itself.
            self.pending_new_target = Some(self.reflect_new_target.take().unwrap_or(callee));
            let inst = self.instantiate(class_id, &env, args)?;
            // `instance.constructor === TheClass` (non-enumerable back-reference).
            if let Some(ih) = inst.as_handle().map(Handle::from_raw) {
                self.realm.set_hidden_property(ih, "constructor", callee);
            }
            return Ok(inst);
        }
        // `new constructorFunction(...)`: bind a fresh object as `this`, run the
        // body, and return it — unless the function explicitly returned an object
        // (the spec's constructor return rule).
        if let Some((func_id, _)) = self.realm.function_at(handle) {
            // Arrow, generator, and async functions are not constructors.
            let def = self.functions[func_id as usize];
            if def.is_arrow || def.is_generator || def.is_async {
                let m = self.new_str("is not a constructor");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            // The instance's `[[Prototype]]` is the *newTarget*'s `.prototype`
            // (the callee's, except under `Reflect.construct(target, args, newTarget)`
            // with a function newTarget), so inherited methods resolve correctly.
            let proto = match self.reflect_new_target {
                Some(nt)
                    if nt
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.function_at(h))
                        .is_some() =>
                {
                    let nt_fid = self
                        .realm
                        .function_at(Handle::from_raw(nt.as_handle().unwrap()))
                        .unwrap()
                        .0;
                    self.realm.function_prototype(nt_fid)
                }
                _ => self.realm.function_prototype(func_id),
            };
            let instance = self.realm.new_object_with_proto(Some(proto));
            let this = NanBox::handle(instance.to_raw());
            // Record the constructor for `instanceof` (hidden, GC-traced slot).
            self.realm.set_hidden_property(instance, CTOR_KEY, callee);
            // `new.target` inside the constructor body is the constructor itself.
            self.pending_new_target = Some(self.reflect_new_target.take().unwrap_or(callee));
            let ret = self.call_with_this(callee, this, args)?;
            if let Some(rh) = ret.as_handle().map(Handle::from_raw)
                && (self.realm.is_array(rh) || self.realm.object_keys(rh).is_some())
            {
                return Ok(ret);
            }
            return Ok(this);
        }
        // `new Array(...)` — Array is a namespace object, matched by identity.
        // A single number argument is the length; otherwise the elements.
        if self.current.get("Array").and_then(|v| v.as_handle()) == callee.as_handle() {
            let elems = if args.len() == 1
                && let Some(n) = args[0].as_number()
            {
                // A single number is the length: a non-negative integer fitting
                // uint32 (capped here to avoid OOM in this dense model). Otherwise a
                // `RangeError`.
                if n < 0.0
                    || n > f64::from(u32::MAX)
                    || n > 100_000_000.0
                    || n != f64::from(n as u32)
                {
                    let m = self.new_str("Invalid array length");
                    return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                }
                alloc::vec![NanBox::undefined(); n as usize]
            } else {
                args.to_vec()
            };
            return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
        }
        let id = self
            .realm
            .native_at(handle)
            .ok_or(ExecError::Unsupported("new on this value"))?;
        // `new WebAssembly.Module(bytes)` — decode/validate, keep the bytes so a
        // later `new WebAssembly.Instance(module)` can instantiate it.
        if id == N_WASM_MODULE {
            return self.make_wasm_module(args.first().copied().unwrap_or(NanBox::undefined()));
        }
        // `new WebAssembly.Instance(module, importObject?)` → `{ exports: {…} }`.
        if id == N_WASM_INSTANCE {
            let module = args
                .first()
                .copied()
                .and_then(|m| m.as_handle())
                .map(Handle::from_raw)
                .filter(|m| self.realm.get_property(*m, WASM_IS_MODULE).is_some())
                .ok_or_else(|| {
                    self.wasm_type_error(
                        "WebAssembly.Instance argument must be a WebAssembly.Module",
                    )
                })?;
            let bytes_arr = self
                .realm
                .get_property(module, WASM_BYTES)
                .unwrap_or(NanBox::undefined());
            let imports = args.get(1).copied().unwrap_or(NanBox::undefined());
            let instance = self.build_wasm_instance(bytes_arr, imports)?;
            return Ok(instance);
        }
        // `new WebAssembly.Global({ value: "i32"|…, mutable }, init)` — a typed
        // value cell exposing a `.value` accessor (settable only if mutable).
        if id == N_WASM_GLOBAL {
            let desc = args.first().copied().unwrap_or(NanBox::undefined());
            let dh = desc.as_handle().map(Handle::from_raw);
            let ty = dh
                .and_then(|h| self.realm.get_property(h, "value"))
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_else(|| String::from("i32"));
            let mutable = dh
                .and_then(|h| self.realm.get_property(h, "mutable"))
                .is_some_and(|v| self.realm.truthy(v));
            let init = args.get(1).copied().unwrap_or(NanBox::undefined());
            let value = self.wasm_coerce_global(&ty, init);
            return Ok(self.make_wasm_global(value, &ty, mutable));
        }
        // `new Proxy(target, handler)`.
        if id == N_PROXY {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            let h = args.get(1).copied().unwrap_or(NanBox::undefined());
            let (Some(tr), Some(hr)) = (target.as_handle(), h.as_handle()) else {
                let msg = self.new_str("Cannot create proxy with a non-object target or handler");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
            };
            let p = self
                .realm
                .new_proxy(Handle::from_raw(tr), Handle::from_raw(hr));
            return Ok(NanBox::handle(p.to_raw()));
        }
        // `new Intl.NumberFormat(locales, options)` / `Intl.DateTimeFormat(...)`.
        if id == N_INTL_NUMBER_FORMAT || id == N_INTL_DATETIME_FORMAT {
            return Ok(self.make_intl_formatter(id, args));
        }
        // `new Intl.Collator(...)` → an object whose `compare` is a bound function
        // (so `arr.sort(new Intl.Collator().compare)` works); code-point order, no
        // locale tailoring (matching `localeCompare`).
        if id == N_INTL_COLLATOR {
            let obj = self.realm.new_object();
            let cmp = self.new_named_native("compare", N_INTL_COMPARE);
            self.realm
                .set_property(obj, "compare", NanBox::handle(cmp.to_raw()));
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new Intl.PluralRules(...)` → an object with a `select(n)` method.
        if id == N_INTL_PLURAL_RULES {
            let obj = self.realm.new_object();
            let sel = self.new_named_native("select", N_INTL_PLURAL_SELECT);
            self.realm
                .set_property(obj, "select", NanBox::handle(sel.to_raw()));
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new Promise(executor)`: run executor(resolve, reject).
        if id == N_PROMISE {
            let promise = self.realm.new_promise();
            let resolve = self.realm.new_bound_native(N_RESOLVE, promise);
            let reject = self.realm.new_bound_native(N_REJECT, promise);
            let executor = args.first().copied().unwrap_or(NanBox::undefined());
            let r = self.call(
                executor,
                &[
                    NanBox::handle(resolve.to_raw()),
                    NanBox::handle(reject.to_raw()),
                ],
            );
            if let Err(ExecError::Throw(e)) = r {
                self.settle(promise, e, false);
            } else {
                r?;
            }
            return Ok(NanBox::handle(promise.to_raw()));
        }
        // `new Date(ms)` (or `new Date()` for "now").
        if id == N_DATE {
            let ms = if args.len() >= 2 {
                // `new Date(year, month, day?, h?, m?, s?, ms?)` (local ≈ UTC here).
                let num =
                    |i: usize, dflt: f64| args.get(i).map_or(dflt, |a| self.realm.to_number(*a));
                let year = num(0, 1970.0) as i64;
                let month = num(1, 0.0) as i64; // 0-indexed, may overflow
                let day = num(2, 1.0) as i64;
                let hours = num(3, 0.0) as i64;
                let mins = num(4, 0.0) as i64;
                let secs = num(5, 0.0) as i64;
                let millis = num(6, 0.0) as i64;
                // Normalize the (possibly out-of-range) month into the year.
                let total_months = year * 12 + month;
                let y = total_months.div_euclid(12);
                let mo = total_months.rem_euclid(12) as u32 + 1; // 1..=12
                let days = crate::realm::days_from_civil(y, mo, day as u32);
                (days * 86_400_000 + hours * 3_600_000 + mins * 60_000 + secs * 1_000 + millis)
                    as f64
            } else {
                match args.first() {
                    // A string argument is parsed as an ISO date; otherwise ToNumber.
                    Some(a) => {
                        if let Some(h) = a.as_handle().map(Handle::from_raw)
                            && let Some(s) = self.realm.string_value(h)
                        {
                            crate::realm::parse_iso_date(&s).unwrap_or(f64::NAN)
                        } else {
                            self.realm.to_number(*a)
                        }
                    }
                    None => now_ms(),
                }
            };
            let d = self.realm.new_date(ms);
            return Ok(NanBox::handle(d.to_raw()));
        }
        // `new RegExp(pattern, flags)`.
        if id == N_REGEXP {
            let pat = self
                .realm
                .to_display_string(args.first().copied().unwrap_or(NanBox::undefined()));
            let flags = match args.get(1) {
                Some(f) => self.realm.to_display_string(*f),
                None => String::new(),
            };
            // Validate the pattern/flags up front: an invalid regular expression is
            // a `SyntaxError` at construction, not a silent broken object.
            #[cfg(feature = "regex")]
            if crate::regex::Regex::new(&pat, &flags).is_err() {
                let m = self.new_str(&alloc::format!(
                    "Invalid regular expression: /{pat}/{flags}"
                ));
                return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 3, Some(m))));
            }
            let r = self.realm.new_regexp(&pat, &flags);
            return Ok(NanBox::handle(r.to_raw()));
        }
        // `new Error(message, { cause })` and friends → `{ name, message }` plus
        // the ES2022 `cause` option. `AggregateError(errors, message, { cause })`
        // takes its message second and exposes `.errors`.
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
            let is_aggregate = id == N_ERROR_BASE + 5;
            let (msg_arg, opts_arg) = if is_aggregate {
                (args.get(1).copied(), args.get(2))
            } else {
                (args.first().copied(), args.get(1))
            };
            let err = self.make_error(id, msg_arg);
            if is_aggregate && let Some(eh) = err.as_handle() {
                let errors = args.first().copied().unwrap_or(NanBox::undefined());
                let list = self.iterate_values(errors).unwrap_or_default();
                let arr = self.realm.new_array(list);
                self.realm.set_property(
                    Handle::from_raw(eh),
                    "errors",
                    NanBox::handle(arr.to_raw()),
                );
            }
            if let Some(opts) = opts_arg.and_then(|v| v.as_handle()).map(Handle::from_raw)
                && let Some(cause) = self.realm.get_property(opts, "cause")
                && let Some(eh) = err.as_handle()
            {
                self.realm
                    .set_property(Handle::from_raw(eh), "cause", cause);
            }
            return Ok(err);
        }
        // `new WeakRef(target)` — holds the target. `deref()` always returns it
        // (sound because GC is never driven mid-execution).
        if id == N_WEAKREF {
            let target = args.first().copied().unwrap_or(NanBox::undefined());
            let obj = self.realm.new_object();
            self.realm.set_hidden_property(obj, WEAKREF_TARGET, target);
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new FinalizationRegistry(cb)` — bounded: with no mid-execution GC the
        // cleanup callback never fires, so `register`/`unregister` are inert.
        if id == N_FINALIZATION_REGISTRY {
            let obj = self.realm.new_object();
            self.realm
                .set_hidden_property(obj, FINREG_TAG, NanBox::boolean(true));
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new ArrayBuffer(n)` — a zeroed byte store of length `n`.
        if id == N_ARRAY_BUFFER {
            let n = args
                .first()
                .map_or(0.0, |v| self.realm.to_number(*v))
                .max(0.0) as usize;
            return Ok(NanBox::handle(self.make_array_buffer(n).to_raw()));
        }
        // `new WebAssembly.Memory({ initial, maximum? })` — linear memory backed by
        // an `ArrayBuffer` of `initial` 64 KiB pages, exposing `.buffer` + `grow()`.
        if id == N_WASM_MEMORY {
            let dh = args
                .first()
                .copied()
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            let initial = dh
                .and_then(|h| self.realm.get_property(h, "initial"))
                .map_or(0.0, |v| self.realm.to_number(v))
                .max(0.0) as usize;
            let maximum = dh
                .and_then(|h| self.realm.get_property(h, "maximum"))
                .map(|v| self.realm.to_number(v).max(0.0) as usize);
            let buf = self.make_array_buffer(initial * WASM_PAGE);
            let mem = self.realm.new_object();
            self.realm
                .set_hidden_property(mem, WASM_MEM_BUFFER, NanBox::handle(buf.to_raw()));
            self.realm
                .set_hidden_property(mem, WASM_MEM_PAGES, NanBox::number(initial as f64));
            self.realm.set_hidden_property(
                mem,
                WASM_MEM_MAX,
                maximum.map_or(NanBox::undefined(), |m| NanBox::number(m as f64)),
            );
            let getter = self.realm.new_bound_native(N_WASM_MEM_BUFFER_GET, mem);
            self.realm.define_accessor(
                mem,
                "buffer",
                NanBox::handle(getter.to_raw()),
                NanBox::undefined(),
            );
            let grow = self.realm.new_bound_native(N_WASM_MEM_GROW, mem);
            self.realm
                .set_property(mem, "grow", NanBox::handle(grow.to_raw()));
            return Ok(NanBox::handle(mem.to_raw()));
        }
        // `new WebAssembly.Table({ element, initial, maximum? }, init?)` — a fixed
        // table of function references, exposing `.length` + `get`/`set`/`grow`.
        if id == N_WASM_TABLE {
            let dh = args
                .first()
                .copied()
                .and_then(|v| v.as_handle())
                .map(Handle::from_raw);
            let initial = dh
                .and_then(|h| self.realm.get_property(h, "initial"))
                .map_or(0.0, |v| self.realm.to_number(v))
                .max(0.0) as usize;
            let maximum = dh
                .and_then(|h| self.realm.get_property(h, "maximum"))
                .map(|v| self.realm.to_number(v).max(0.0) as usize);
            // Slots start at the init value (a function) or null.
            let init = args.get(1).copied().unwrap_or(NanBox::null());
            let elems = self.realm.new_array(alloc::vec![init; initial]);
            let table = self.realm.new_object();
            self.realm
                .set_hidden_property(table, WASM_TABLE_ELEMS, NanBox::handle(elems.to_raw()));
            self.realm.set_hidden_property(
                table,
                WASM_TABLE_MAX,
                maximum.map_or(NanBox::undefined(), |m| NanBox::number(m as f64)),
            );
            let len_get = self.realm.new_bound_native(N_WASM_TABLE_LEN, table);
            self.realm.define_accessor(
                table,
                "length",
                NanBox::handle(len_get.to_raw()),
                NanBox::undefined(),
            );
            for (name, nid) in [
                ("get", N_WASM_TABLE_GET),
                ("set", N_WASM_TABLE_SET),
                ("grow", N_WASM_TABLE_GROW),
            ] {
                let f = self.realm.new_bound_native(nid, table);
                self.realm
                    .set_property(table, name, NanBox::handle(f.to_raw()));
            }
            return Ok(NanBox::handle(table.to_raw()));
        }
        // `new DataView(buffer, byteOffset?)` — a view onto an ArrayBuffer.
        if id == N_DATA_VIEW {
            let obj = self.realm.new_object();
            let buf = args.first().copied().unwrap_or(NanBox::undefined());
            let off = args.get(1).map_or(0.0, |v| self.realm.to_number(*v));
            self.realm.set_hidden_property(obj, DATA_VIEW_BUF, buf);
            self.realm
                .set_hidden_property(obj, DATA_VIEW_OFF, NanBox::number(off));
            // An explicit byteLength (3rd arg) is honored; otherwise the view spans
            // the rest of the buffer from `byteOffset`.
            if let Some(len) = args.get(2)
                && !matches!(len.unpack(), Unpacked::Undefined)
            {
                let n = self.realm.to_number(*len);
                self.realm
                    .set_hidden_property(obj, DATA_VIEW_LEN, NanBox::number(n));
            }
            return Ok(NanBox::handle(obj.to_raw()));
        }
        // `new Int8Array(n)` / `new Uint8Array([…])` / … — a typed array backed by
        // a plain array whose element writes coerce to the element kind.
        if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16).contains(&id) {
            let kind = id - N_TYPED_ARRAY_BASE;
            let elems: Vec<NanBox> = match args.first().copied() {
                // `new T(arrayLike)` copies and coerces the source's elements.
                Some(v)
                    if v.as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                        .is_some() =>
                {
                    let src = v
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                        .unwrap();
                    src.iter()
                        .map(|e| NanBox::number(coerce_typed(kind, self.realm.to_number(*e))))
                        .collect()
                }
                // `new T(length)` allocates a zeroed array.
                Some(v) => {
                    let n = self.realm.to_number(v).max(0.0) as usize;
                    alloc::vec![NanBox::number(0.0); n]
                }
                None => Vec::new(),
            };
            let arr = self.realm.new_array(elems);
            self.realm
                .set_property(arr, TYPED_ARRAY_KIND, NanBox::number(f64::from(kind)));
            return Ok(NanBox::handle(arr.to_raw()));
        }
        // `new Number(x)` / `new String(x)` / `new Boolean(x)`: a primitive
        // wrapper object boxing the coerced primitive (`valueOf` recovers it).
        if matches!(id, N_NUMBER | N_STRING | N_BOOLEAN) {
            let prim = match id {
                N_NUMBER => {
                    let n = args.first().map_or(0.0, |v| self.realm.to_number(*v));
                    NanBox::number(n)
                }
                N_STRING => {
                    let s = args
                        .first()
                        .map_or_else(String::new, |v| self.realm.to_display_string(*v));
                    self.new_str(&s)
                }
                _ => NanBox::boolean(
                    self.realm
                        .truthy(args.first().copied().unwrap_or(NanBox::undefined())),
                ),
            };
            return Ok(self.make_primitive_wrapper(prim, id));
        }
        // `WeakMap`/`WeakSet` reuse the collection cell (no true weak refs here).
        let is_set = match id {
            N_SET | N_WEAKSET => true,
            N_MAP | N_WEAKMAP => false,
            _ => return Err(ExecError::Unsupported("new on this constructor")),
        };
        let handle = self.realm.new_collection(is_set);
        // A weak collection rejects primitive keys (its keys must be objects/symbols).
        if matches!(id, N_WEAKMAP | N_WEAKSET) {
            self.realm.set_collection_weak(handle);
        }
        // Seed from an iterable: a `Set` from array elements, a `Map` from
        // `[key, value]` pairs.
        // Seed from any iterable (array, string, Set, Map, …): a `Set` from each
        // value, a `Map` from each `[key, value]` pair.
        let first = args.first().copied().unwrap_or(NanBox::undefined());
        if !matches!(first.unpack(), Unpacked::Undefined | Unpacked::Null) {
            for item in self.iterate_values(first)? {
                if is_set {
                    self.guard_weak_key(handle, item)?;
                    self.realm.collection_set(handle, item, item);
                } else if let Some(pr) = item
                    .as_handle()
                    .and_then(|r| self.realm.array_elements(Handle::from_raw(r)))
                    .map(<[_]>::to_vec)
                {
                    let k = pr.first().copied().unwrap_or(NanBox::undefined());
                    let v = pr.get(1).copied().unwrap_or(NanBox::undefined());
                    self.guard_weak_key(handle, k)?;
                    self.realm.collection_set(handle, k, v);
                }
            }
        }
        Ok(NanBox::handle(handle.to_raw()))
    }

    /// Builds a match-result object (`[0..n]` groups, plus `index`, `input`,
    /// `length`) from regex captures over `text`.
    #[cfg(feature = "regex")]
    fn regex_match_object(
        &mut self,
        text: &str,
        caps: &crate::regex::Captures,
        group_names: &[(usize, String)],
    ) -> NanBox {
        // A match result is a real Array: element `i` is capture group `i` (the
        // whole match at 0), so `Array.isArray`, `JSON.stringify`, `.length`, and
        // the array methods all behave. `index`/`input`/`groups` are enumerable
        // named own properties (kept in the array's auxiliary object).
        let elems: Vec<NanBox> = caps
            .groups
            .iter()
            .map(|g| match g {
                Some((s, e)) => self.new_str(&char_substr(text, *s, *e)),
                None => NanBox::undefined(),
            })
            .collect();
        let obj = self.realm.new_array(elems);
        let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
        self.realm
            .set_property(obj, "index", NanBox::number(index as f64));
        let input = self.new_str(text);
        self.realm.set_property(obj, "input", input);
        // `.groups`: an object of named captures (or `undefined` if none).
        let groups = if group_names.is_empty() {
            NanBox::undefined()
        } else {
            let g = self.realm.new_object();
            for (idx, name) in group_names {
                let v = match caps.groups.get(*idx).and_then(|x| *x) {
                    Some((s, e)) => self.new_str(&char_substr(text, s, e)),
                    None => NanBox::undefined(),
                };
                self.realm.set_property(g, name, v);
            }
            NanBox::handle(g.to_raw())
        };
        self.realm.set_property(obj, "groups", groups);
        NanBox::handle(obj.to_raw())
    }

    /// Builds an error object `{ name, message }` for the constructor `id`.
    /// Applies a native superclass constructor's effect to `instance` for
    /// `super(...)` in a class that `extends` a native (e.g. `extends Error`).
    fn apply_native_super(&mut self, native_id: u16, instance: Handle, args: &[NanBox]) {
        // Error family: set `message` and the default `name` (a `this.name = …`
        // after `super()` may override it).
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&native_id) {
            let name = ERROR_NAMES[(native_id - N_ERROR_BASE) as usize];
            let name_v = self.new_str(name);
            self.realm.set_property(instance, "name", name_v);
            let msg = match args.first() {
                Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                    let s = self.realm.to_display_string(*m);
                    self.new_str(&s)
                }
                _ => self.new_str(""),
            };
            self.realm.set_property(instance, "message", msg);
            // `name`/`message` are non-enumerable (out of `Object.keys`/JSON).
            self.realm.mark_hidden(instance, "name");
            self.realm.mark_hidden(instance, "message");
        }
    }

    /// `structuredClone(v)`: a deep copy. Primitives and immutable heap values
    /// (strings, BigInts) are shared; Dates, Maps, Sets, arrays, and plain
    /// objects are recursively cloned. `seen` maps each visited source handle to
    /// its clone so cyclic and shared references are preserved. Functions and
    /// symbols are not cloneable (a TypeError, like `DataCloneError`).
    fn structured_clone(
        &mut self,
        v: NanBox,
        seen: &mut Vec<(u64, NanBox)>,
    ) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v); // a primitive
        };
        let h = Handle::from_raw(raw);
        // Immutable heap values are shared, not copied.
        if self.realm.string_value(h).is_some() || self.realm.bigint_at(h).is_some() {
            return Ok(v);
        }
        if self.is_callable(h) || self.realm.symbol_at(h).is_some() {
            let m = self.new_str("value could not be cloned");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // A previously-cloned handle (cycle or shared reference).
        if let Some((_, c)) = seen.iter().find(|(r, _)| *r == raw) {
            return Ok(*c);
        }
        if let Some(ms) = self.realm.date_at(h) {
            return Ok(NanBox::handle(self.realm.new_date(ms).to_raw()));
        }
        if let Some(is_set) = self.realm.collection_is_set(h) {
            let coll = self.realm.new_collection(is_set);
            let cbox = NanBox::handle(coll.to_raw());
            seen.push((raw, cbox));
            for (k, val) in self.realm.collection_entries(h).unwrap_or_default() {
                let ck = self.structured_clone(k, seen)?;
                let cv = self.structured_clone(val, seen)?;
                self.realm.collection_set(coll, ck, cv);
            }
            return Ok(cbox);
        }
        if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
            let arr = self.realm.new_array(Vec::new());
            let abox = NanBox::handle(arr.to_raw());
            seen.push((raw, abox));
            for e in elems {
                let c = self.structured_clone(e, seen)?;
                self.realm.array_push(arr, c);
            }
            return Ok(abox);
        }
        // A plain object: clone own enumerable string-keyed properties.
        let obj = self.realm.new_object();
        let obox = NanBox::handle(obj.to_raw());
        seen.push((raw, obox));
        for k in self.realm.object_keys(h).unwrap_or_default() {
            if let Some(pv) = self.realm.get_property(h, &k) {
                let c = self.structured_clone(pv, seen)?;
                self.realm.set_property(obj, &k, c);
            }
        }
        Ok(obox)
    }

    /// Builds an `Intl.NumberFormat`/`DateTimeFormat` instance — an object that
    /// captures the relevant options behind a `\0intl` kind marker. Used for both
    /// `new Intl.X(...)` and the callable-without-`new` form.
    fn make_intl_formatter(&mut self, id: u16, args: &[NanBox]) -> NanBox {
        let obj = self.realm.new_object();
        let kind = if id == N_INTL_NUMBER_FORMAT {
            "number"
        } else {
            "datetime"
        };
        let marker = self.new_str(kind);
        self.realm.set_hidden_property(obj, "\u{0}intl", marker);
        // `.format` is a readable function (so `typeof nf.format === "function"` and a
        // member call `nf.format(x)` works); it formats against its `this` formatter.
        let fmt = self.new_named_native("format", N_INTL_FORMAT);
        self.realm
            .set_property(obj, "format", NanBox::handle(fmt.to_raw()));
        if let Some(opts) = args
            .get(1)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
        {
            for key in [
                "style",
                "currency",
                "minimumFractionDigits",
                "maximumFractionDigits",
                "useGrouping",
            ] {
                if let Some(v) = self.realm.get_property(opts, key) {
                    self.realm.set_hidden_property(obj, key, v);
                }
            }
        }
        NanBox::handle(obj.to_raw())
    }

    /// Formats `value` per the `Intl.NumberFormat`/`DateTimeFormat` instance `handle`
    /// (a `\0intl`-marked object). Shared by `nf.format(x)` and the bound `nf.format`.
    fn intl_format_value(&mut self, handle: Handle, value: NanBox) -> String {
        let kind = self
            .realm
            .get_property(handle, "\u{0}intl")
            .map(|k| self.realm.to_display_string(k))
            .unwrap_or_default();
        if kind == "datetime" {
            let ms = match value.as_handle().map(Handle::from_raw) {
                Some(h) if self.realm.date_at(h).is_some() => self.realm.date_at(h).unwrap(),
                _ => self.realm.to_number(value),
            };
            let day = (ms as i64).div_euclid(86_400_000);
            let (y, mo, d) = crate::realm::civil_from_days(day);
            alloc::format!("{mo}/{d}/{y}")
        } else {
            let n = self.realm.to_number(value);
            self.intl_format_number(handle, n)
        }
    }

    /// Formats `n` per an `Intl.NumberFormat` instance's captured options
    /// (`style`, `currency`, min/max `FractionDigits`, `useGrouping`).
    fn intl_format_number(&mut self, handle: Handle, n: f64) -> String {
        let opt_str = |this: &mut Self, k: &str| -> Option<String> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_display_string(v))
        };
        let opt_num = |this: &mut Self, k: &str| -> Option<i32> {
            this.realm
                .get_property(handle, k)
                .filter(|v| !matches!(v.unpack(), Unpacked::Undefined))
                .map(|v| this.realm.to_number(v) as i32)
        };
        let style = opt_str(self, "style").unwrap_or_else(|| String::from("decimal"));
        let currency = opt_str(self, "currency");
        let use_grouping = !matches!(
            self.realm.get_property(handle, "useGrouping"),
            Some(v) if matches!(v.unpack(), Unpacked::Bool(false))
        );
        // Default fraction digits: currency = 2 (0 for JPY), else 0..=3.
        let is_jpy = currency.as_deref() == Some("JPY");
        let (def_min, def_max) = match style.as_str() {
            "currency" if is_jpy => (0, 0),
            "currency" => (2, 2),
            "percent" => (0, 0),
            _ => (0, 3),
        };
        let min = opt_num(self, "minimumFractionDigits")
            .unwrap_or(def_min)
            .clamp(0, 20);
        let max = opt_num(self, "maximumFractionDigits")
            .unwrap_or(def_max.max(min))
            .clamp(min, 20);
        let value = if style == "percent" { n * 100.0 } else { n };
        // Round to `max` digits, then trim trailing zeros down to `min`.
        let mut s = alloc::format!("{:.*}", max as usize, value);
        if max > min && s.contains('.') {
            while s.ends_with('0') && {
                let frac = s.split_once('.').map_or(0, |(_, f)| f.len());
                frac > min as usize
            } {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        // Group the integer part.
        let grouped = if use_grouping {
            let neg = s.starts_with('-');
            let body = s.trim_start_matches('-');
            let (ip, fp) = body
                .split_once('.')
                .map_or((body, None), |(i, f)| (i, Some(f)));
            let mut g = String::new();
            let len = ip.len();
            for (i, b) in ip.bytes().enumerate() {
                if i > 0 && (len - i) % 3 == 0 {
                    g.push(',');
                }
                g.push(b as char);
            }
            if let Some(f) = fp {
                g.push('.');
                g.push_str(f);
            }
            if neg { alloc::format!("-{g}") } else { g }
        } else {
            s
        };
        match style.as_str() {
            "percent" => alloc::format!("{grouped}%"),
            "currency" => {
                let sym = match currency.as_deref() {
                    Some("USD") => "$",
                    Some("EUR") => "€",
                    Some("GBP") => "£",
                    Some("JPY" | "CNY") => "¥",
                    Some(other) => return alloc::format!("{other}\u{a0}{grouped}"),
                    None => "$",
                };
                alloc::format!("{sym}{grouped}")
            }
            _ => grouped,
        }
    }

    /// Writes `value` to array index `i`, coercing it to the element kind first
    /// if `handle` is a typed array (`Uint8Array`, …).
    fn set_element_coerced(&mut self, handle: crate::heap::Handle, i: usize, value: NanBox) {
        let v = match self.realm.get_property(handle, TYPED_ARRAY_KIND) {
            Some(k) => {
                let n = self.realm.to_number(value);
                NanBox::number(coerce_typed(k.as_number().unwrap_or(0.0) as u16, n))
            }
            None => value,
        };
        self.realm.set_element(handle, i, v);
    }

    /// Builds a primitive wrapper object (`new Number`/`String`/`Boolean`,
    /// `Object(primitive)`): an object boxing `prim` behind a `\0prim` slot, with
    /// `\0wraptype` recording the constructor id (for `instanceof`).
    /// For a weak collection (`WeakMap`/`WeakSet`), throws a `TypeError` when `key`
    /// is a primitive — weak keys must be objects or symbols. A no-op for a
    /// non-weak (`Map`/`Set`) collection.
    fn guard_weak_key(&mut self, coll: Handle, key: NanBox) -> Result<(), ExecError> {
        if !self.realm.collection_is_weak(coll) {
            return Ok(());
        }
        let valid = key.as_handle().map(Handle::from_raw).is_some_and(|h| {
            self.realm.string_value(h).is_none() && self.realm.bigint_at(h).is_none()
        });
        if valid {
            return Ok(());
        }
        let m = self.new_str("Invalid value used as weak collection key");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// Builds the result array for an Array method invoked on `recv`. If `recv` is a
    /// typed array, the result is a same-kind typed array with its elements coerced
    /// to that element type; otherwise an ordinary array.
    fn typed_like(&mut self, recv: Handle, elems: Vec<NanBox>) -> NanBox {
        if let Some(k) = self.realm.get_property(recv, TYPED_ARRAY_KIND) {
            let kind = k.as_number().unwrap_or(0.0) as u16;
            let mut coerced = Vec::with_capacity(elems.len());
            for e in elems {
                let n = self.realm.to_number(e);
                coerced.push(NanBox::number(coerce_typed(kind, n)));
            }
            let arr = self.realm.new_array(coerced);
            self.realm
                .set_property(arr, TYPED_ARRAY_KIND, NanBox::number(f64::from(kind)));
            NanBox::handle(arr.to_raw())
        } else {
            NanBox::handle(self.realm.new_array(elems).to_raw())
        }
    }

    fn make_primitive_wrapper(&mut self, prim: NanBox, ctor_id: u16) -> NanBox {
        let obj = self.realm.new_object();
        self.realm.set_hidden_property(obj, PRIM_WRAP, prim);
        self.realm
            .set_hidden_property(obj, PRIM_WRAP_TYPE, NanBox::number(f64::from(ctor_id)));
        NanBox::handle(obj.to_raw())
    }

    /// `ToObject(v)` for `Object(v)`: `null`/`undefined` yield a fresh object; an
    /// existing object/array/function is returned unchanged; a primitive is boxed in
    /// its wrapper (so `Object(42).valueOf()` is `42`).
    fn coerce_to_object(&mut self, v: NanBox) -> NanBox {
        match v.unpack() {
            Unpacked::Undefined | Unpacked::Null => {
                NanBox::handle(self.realm.new_object().to_raw())
            }
            Unpacked::Number(_) => self.make_primitive_wrapper(v, N_NUMBER),
            Unpacked::Bool(_) => self.make_primitive_wrapper(v, N_BOOLEAN),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if self.realm.string_value(h).is_some() {
                    self.make_primitive_wrapper(v, N_STRING)
                } else {
                    // An already-object value (object/array/function/symbol/bigint).
                    v
                }
            }
        }
    }

    /// Resolves a (trap-less) proxy to its target for key enumeration, so
    /// `Object.keys`/`values`/`entries` on a pass-through proxy see the target's
    /// own keys. A non-proxy is returned unchanged. (The `ownKeys` trap is not
    /// invoked here.)
    fn proxy_key_target(&self, mut h: crate::heap::Handle) -> crate::heap::Handle {
        while let Some((target, _)) = self.realm.proxy_at(h) {
            h = target;
        }
        h
    }

    /// `Object.keys` for a proxy that defines an `ownKeys` trap: invoke the trap,
    /// then keep each string key whose property is enumerable — via the
    /// `getOwnPropertyDescriptor` trap if present, else the target. Returns `None`
    /// when there is no `ownKeys` trap (so the caller uses the target's keys).
    fn proxy_own_enumerable_keys(
        &mut self,
        proxy: Handle,
    ) -> Result<Option<Vec<String>>, ExecError> {
        let Some((target, handler)) = self.realm.proxy_at(proxy) else {
            return Ok(None);
        };
        let own_trap = self
            .realm
            .get_property(handler, "ownKeys")
            .unwrap_or(NanBox::undefined());
        if !own_trap
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Ok(None);
        }
        let target_box = NanBox::handle(target.to_raw());
        let keys = self.call(own_trap, &[target_box])?;
        let keys = self.iterate_values(keys)?;
        let gopd = self
            .realm
            .get_property(handler, "getOwnPropertyDescriptor")
            .unwrap_or(NanBox::undefined());
        let gopd_callable = gopd
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
        let mut out = Vec::new();
        for k in keys {
            // Only string keys participate in `Object.keys` (symbols are skipped).
            let Some(name) = k
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.string_value(h))
            else {
                continue;
            };
            let enumerable = if gopd_callable {
                let kbox = self.new_str(&name);
                let desc = self.call(gopd, &[target_box, kbox])?;
                desc.as_handle()
                    .map(Handle::from_raw)
                    .and_then(|dh| self.realm.get_property(dh, "enumerable"))
                    .is_some_and(|v| self.realm.truthy(v))
            } else {
                // No descriptor trap: forward to the target — an own, enumerable
                // property only (a key the target lacks is not enumerable).
                self.realm.has_own(target, &name)
                    && self.realm.property_is_enumerable(target, &name)
            };
            if enumerable {
                out.push(name);
            }
        }
        Ok(Some(out))
    }

    fn make_error(&mut self, id: u16, message: Option<NanBox>) -> NanBox {
        let name = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
        let obj = self.realm.new_object();
        let name_v = self.new_str(name);
        self.realm.set_property(obj, "name", name_v);
        let msg_str = match message {
            Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                self.realm.to_display_string(m)
            }
            _ => String::new(),
        };
        let msg = self.new_str(&msg_str);
        self.realm.set_property(obj, "message", msg);
        // `name`/`message` are non-enumerable (so `Object.keys(err)` is empty).
        self.realm.mark_hidden(obj, "name");
        self.realm.mark_hidden(obj, "message");
        // A minimal `stack` (the `name: message` header; no real frame capture),
        // non-enumerable like the real property.
        let head = if msg_str.is_empty() {
            String::from(name)
        } else {
            alloc::format!("{name}: {msg_str}")
        };
        let stack = self.new_str(&alloc::format!("{head}\n    at <anonymous>"));
        self.realm.set_property(obj, "stack", stack);
        self.realm.mark_hidden(obj, "stack");
        NanBox::handle(obj.to_raw())
    }

    /// Registers a class and allocates a class value capturing the current scope.
    fn make_class(&mut self, class: &'a Class) -> Result<NanBox, ExecError> {
        let class_id = self.classes.len() as u32;
        self.classes.push(class);
        // Build the static members (`static foo() {}` / `static x = …`).
        let mut statics = alloc::collections::BTreeMap::new();
        let mut static_fields = Vec::new();
        let mut static_getters = alloc::collections::BTreeMap::new();
        let mut static_setters = alloc::collections::BTreeMap::new();
        for member in &class.body {
            match member {
                ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                    if let Ok(key) = self.eval_prop_key(&m.key) {
                        // A static method's home is this class, entered statically, so
                        // `super.x` resolves against the superclass's static members.
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            m.value.is_generator,
                            Some(class_id),
                            true,
                        );
                        statics.insert(key, f);
                    }
                }
                ClassMember::Field(field) if field.is_static => {
                    if let Ok(key) = self.eval_prop_key(&field.key) {
                        let v = match &field.value {
                            Some(e) => self.eval(e).unwrap_or(NanBox::undefined()),
                            None => NanBox::undefined(),
                        };
                        // Static fields are enumerable own keys of the constructor.
                        if !static_fields.contains(&key) {
                            static_fields.push(key.clone());
                        }
                        statics.insert(key, v);
                    }
                }
                // `static get x() {}` / `static set x(v) {}` — accessors.
                ClassMember::Method(m)
                    if m.is_static && matches!(m.kind, MethodKind::Get | MethodKind::Set) =>
                {
                    if let Ok(key) = self.eval_prop_key(&m.key) {
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            false,
                            Some(class_id),
                            true,
                        );
                        if m.kind == MethodKind::Get {
                            static_getters.insert(key, f);
                        } else {
                            static_setters.insert(key, f);
                        }
                    }
                }
                _ => {}
            }
        }
        self.class_statics.push(statics);
        self.class_static_fields.push(static_fields);
        self.class_static_get.push(static_getters);
        self.class_static_set.push(static_setters);
        // The methods' captured scope; a named class binds its own name here (so
        // `class C { m() { return C; } }` can self-reference), filled in below.
        let class_env = self.current.child();
        self.class_envs.push(class_env.clone());
        // Record a native-constructor superclass (`extends Error`), if any, so
        // construction and `instanceof` can reach it (it has no class id).
        let native_super = if let Some(expr) = &class.super_class {
            self.eval(expr).ok().and_then(|v| {
                let h = Handle::from_raw(v.as_handle()?);
                if self.realm.class_at(h).is_some() {
                    None
                } else {
                    self.realm.native_at(h)
                }
            })
        } else {
            None
        };
        self.class_native_super.push(native_super);
        let handle = self.realm.new_class(class_id, class_env.clone());
        let class_val = NanBox::handle(handle.to_raw());
        // Bind the class's own name in its methods' scope (a named class
        // expression sees itself; the binding is read-only in spec but not
        // enforced here).
        if let Some(id) = &class.id {
            class_env.declare(&id.name, class_val);
        }
        // Run `static { … }` initialization blocks with `this` = the class and the
        // class name bound (so the block can reference the class and its statics).
        if class
            .body
            .iter()
            .any(|m| matches!(m, ClassMember::StaticBlock { .. }))
        {
            let scope = self.current.child();
            if let Some(id) = &class.id {
                scope.declare(&id.name, class_val);
            }
            let saved = core::mem::replace(&mut self.current, scope);
            let saved_this = core::mem::replace(&mut self.this_val, class_val);
            let r = (|| {
                for member in &class.body {
                    if let ClassMember::StaticBlock { body, .. } = member {
                        for stmt in body {
                            self.exec(stmt)?;
                        }
                    }
                }
                Ok(())
            })();
            self.current = saved;
            self.this_val = saved_this;
            r?;
        }
        Ok(class_val)
    }

    /// Resolves a class's `extends` superclass to `(class_id, env)`, if any.
    fn resolve_super(
        &mut self,
        class: &'a Class,
        env: &Scope,
    ) -> Result<Option<(u32, Scope)>, ExecError> {
        let Some(expr) = &class.super_class else {
            return Ok(None);
        };
        let saved = core::mem::replace(&mut self.current, env.clone());
        let value = self.eval(expr);
        self.current = saved;
        let raw = value?
            .as_handle()
            .ok_or(ExecError::Unsupported("extends a non-class"))?;
        let h = Handle::from_raw(raw);
        if let Some(parent) = self.realm.class_at(h) {
            Ok(Some(parent))
        } else if self.realm.native_at(h).is_some() {
            // A native superclass (e.g. `extends Error`) has no class chain;
            // it is tracked separately in `class_native_super`.
            Ok(None)
        } else {
            Err(ExecError::Unsupported("extends a non-class"))
        }
    }

    /// Instantiates `new Class(args)`: creates the object, installs the methods
    /// of the whole `extends` chain (derived overriding base), then runs the
    /// constructor (with `super(...)` reaching the base).
    fn instantiate(
        &mut self,
        class_id: u32,
        env: &Scope,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let instance = self.realm.new_object();
        let this_val = NanBox::handle(instance.to_raw());

        // Walk the chain derived→base, then install methods base-first so a
        // derived method overrides an inherited one.
        let mut chain: Vec<(u32, Scope)> = Vec::new();
        let mut cur = Some((class_id, env.clone()));
        while let Some((cid, cenv)) = cur {
            chain.push((cid, cenv.clone()));
            cur = self.resolve_super(self.classes[cid as usize], &cenv)?;
        }
        for (cid, cenv) in chain.iter().rev() {
            let class = self.classes[*cid as usize];
            for member in &class.body {
                let ClassMember::Method(m) = member else {
                    continue;
                };
                if m.is_static {
                    continue;
                }
                let saved = core::mem::replace(&mut self.current, cenv.clone());
                // Resolve the key in the class scope (so `[computed]` names see
                // the enclosing bindings).
                let key = self.eval_prop_key(&m.key)?;
                let f = self.make_method(
                    &m.value.params,
                    Body::Block(&m.value.body),
                    false,
                    m.value.is_generator,
                    Some(*cid),
                    false,
                );
                self.current = saved;
                match m.kind {
                    MethodKind::Method => {
                        // Methods are callable but non-enumerable.
                        self.realm.set_hidden_property(instance, &key, f);
                    }
                    MethodKind::Get => {
                        self.realm
                            .define_accessor(instance, &key, f, NanBox::undefined());
                        self.realm.mark_hidden(instance, &key); // class accessors are non-enumerable
                    }
                    MethodKind::Set => {
                        self.realm
                            .define_accessor(instance, &key, NanBox::undefined(), f);
                        self.realm.mark_hidden(instance, &key);
                    }
                    MethodKind::Constructor => {}
                }
            }
        }

        self.realm.set_class_tag(instance, class_id);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        // `new.target` (the class reached via `new`, passed through the one-shot)
        // holds for the whole constructor, incl. a base reached via `super(...)`.
        let nt = self
            .pending_new_target
            .take()
            .unwrap_or(NanBox::undefined());
        let saved_target = core::mem::replace(&mut self.new_target, nt);
        let result = self.run_constructor(class_id, env, instance, args);
        self.this_val = saved_this;
        self.new_target = saved_target;
        let ret = result?;
        // A constructor that `return`s an *object* makes `new` yield that object
        // instead of the freshly-built instance; a primitive return is ignored.
        match ret {
            Some(v)
                if v.as_handle().map(Handle::from_raw).is_some_and(|h| {
                    self.realm.string_value(h).is_none()
                        && self.realm.bigint_at(h).is_none()
                        && self.realm.symbol_at(h).is_none()
                }) =>
            {
                Ok(v)
            }
            _ => Ok(this_val),
        }
    }

    /// Runs one class's field initializers and constructor on `instance` (with
    /// `this` already bound). `super(args)` reaches the base via `pending_super`.
    /// Applies a class's own (non-static) instance field initializers to
    /// `instance`. Run before the constructor body (base class) / after the
    /// implicit super for a constructor-less derived class.
    fn init_instance_fields(&mut self, class_id: u32, instance: Handle) -> Result<(), ExecError> {
        let class = self.classes[class_id as usize];
        for member in &class.body {
            if let ClassMember::Field(field) = member
                && !field.is_static
            {
                // A computed field name (`[expr] = v`) is evaluated here.
                let key = match &field.key {
                    PropertyKey::Computed(e) => {
                        let k = self.eval(e)?;
                        self.member_key(k)
                    }
                    other => static_key(other)?,
                };
                let v = match &field.value {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                self.realm.set_property(instance, &key, v);
            }
        }
        Ok(())
    }

    fn run_constructor(
        &mut self,
        class_id: u32,
        env: &Scope,
        instance: Handle,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let class = self.classes[class_id as usize];
        let parent = self.resolve_super(class, env)?;
        let saved_super = core::mem::replace(&mut self.pending_super, parent.clone());
        let native_parent = self.class_native_super[class_id as usize];
        let saved_super_native = core::mem::replace(&mut self.pending_super_native, native_parent);
        let saved_scope = core::mem::replace(&mut self.current, env.child());
        let result = (|| {
            let ctor = class.body.iter().find_map(|m| match m {
                ClassMember::Method(m) if m.kind == MethodKind::Constructor => Some(m),
                _ => None,
            });
            match (ctor, &parent) {
                (Some(ctor), _) => {
                    // Own fields initialize before the constructor body, so a
                    // constructor write isn't clobbered by a later field decl.
                    self.init_instance_fields(class_id, instance)?;
                    let scope = self.current.child();
                    let saved = core::mem::replace(&mut self.current, scope);
                    let r: Result<Option<NanBox>, ExecError> = (|| {
                        // Bind parameters (rest/default/destructuring supported).
                        for (i, param) in ctor.value.params.iter().enumerate() {
                            let value = if param.rest {
                                let rest = args[i.min(args.len())..].to_vec();
                                NanBox::handle(self.realm.new_array(rest).to_raw())
                            } else {
                                let mut v = args.get(i).copied().unwrap_or(NanBox::undefined());
                                if matches!(v.unpack(), Unpacked::Undefined)
                                    && let Some(d) = &param.default
                                {
                                    v = self.eval(d)?;
                                }
                                v
                            };
                            self.bind_pattern(&param.target, value)?;
                        }
                        // The constructor's `return value` (if an object) overrides
                        // the new instance; captured here.
                        let mut returned = None;
                        for stmt in &ctor.value.body {
                            if let Flow::Return(v) = self.exec(stmt)? {
                                returned = Some(v);
                                break;
                            }
                        }
                        Ok(returned)
                    })();
                    self.current = saved;
                    r
                }
                // No own constructor but a base: implicit `super(args)`, then
                // this class's own field initializers.
                (None, Some((pid, penv))) => {
                    let ret = self.run_constructor(*pid, &penv.clone(), instance, args)?;
                    self.init_instance_fields(class_id, instance)?;
                    Ok(ret)
                }
                (None, None) => {
                    // A constructor-less class extending a *native* superclass
                    // (`class X extends Error {}`) performs the implicit
                    // `super(...args)` into the native constructor, so e.g. the
                    // error message is forwarded.
                    if let Some(nid) = native_parent {
                        self.apply_native_super(nid, instance, args);
                    }
                    self.init_instance_fields(class_id, instance)?;
                    Ok(None)
                }
            }
        })();
        self.current = saved_scope;
        self.pending_super = saved_super;
        self.pending_super_native = saved_super_native;
        result
    }

    /// Evaluates a call's arguments.
    fn eval_args(&mut self, arguments: &'a [Argument]) -> Result<Vec<NanBox>, ExecError> {
        let mut args = Vec::with_capacity(arguments.len());
        for a in arguments {
            match a {
                Argument::Item(e) => args.push(self.eval(e)?),
                Argument::Spread(e) => {
                    let v = self.eval(e)?;
                    args.extend(self.iterate_values(v)?);
                }
            }
        }
        Ok(args)
    }

    /// Dispatches a built-in method on a string/array receiver. Returns
    /// `Ok(None)` if `method` is not a recognized built-in (the caller then
    /// treats it as an ordinary property-valued function).
    fn call_method(
        &mut self,
        recv: NanBox,
        method: &str,
        args: &[NanBox],
    ) -> Result<Option<NanBox>, ExecError> {
        let arg = |i: usize| args.get(i).copied().unwrap_or(NanBox::undefined());

        // A primitive wrapper object (`new Number`/`String`/`Boolean`): `valueOf`
        // recovers the boxed primitive; every other method delegates to it.
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
        {
            return match method {
                "valueOf" => Ok(Some(prim)),
                _ => self.call_method(prim, method, args),
            };
        }

        // --- boolean methods (the receiver is an immediate) ---
        if let Unpacked::Bool(b) = recv.unpack() {
            return Ok(match method {
                "toString" => Some(self.new_str(if b { "true" } else { "false" })),
                "valueOf" => Some(recv),
                _ => None,
            });
        }
        // --- number methods (the receiver is an immediate, not a handle) ---
        if let Some(n) = recv.as_number() {
            return Ok(match method {
                "toString" => {
                    // An optional radix (2–36) for integers, else base 10.
                    let radix = match args.first() {
                        Some(a) => self.realm.to_number(*a) as u32,
                        None => 10,
                    };
                    if radix == 10 || !(2..=36).contains(&radix) {
                        Some(self.new_str(&self.realm.to_display_string(recv)))
                    } else {
                        Some(self.new_str(&int_to_radix(n, radix)))
                    }
                }
                "valueOf" => Some(recv),
                // `toLocaleString()` — a minimal grouping format (thousands
                // separators with `,`), since no locale data is available.
                "toLocaleString" => Some(self.new_str(&group_thousands(n))),
                #[cfg(feature = "std")]
                "toFixed" => {
                    // `fractionDigits` is ToIntegerOrInfinity'd (undefined/NaN → 0)
                    // and must be in [0, 100], else a RangeError.
                    let d = self.realm.to_number(arg(0));
                    let f = if d.is_nan() { 0 } else { d as i64 };
                    if !(0..=100).contains(&f) {
                        let m = self.new_str("toFixed() digits argument must be between 0 and 100");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    let digits = f as usize;
                    let s = if !n.is_finite() {
                        // `Infinity`/`-Infinity`/`NaN` use the spec ToString.
                        self.realm.to_display_string(NanBox::number(n))
                    } else if n.abs() >= 1e21 {
                        // Spec: a magnitude ≥ 1e21 uses the regular `ToString`
                        // (exponential), not a full decimal expansion.
                        self.realm.to_display_string(NanBox::number(n))
                    } else {
                        // Round the *exact* f64 to `digits` places. Rust's formatter is
                        // correctly rounded but ties-to-even; JS ties away from zero.
                        // Only an exact half (the dropped tail is precisely "5" then
                        // zeros) differs — detect that from the value's decimal
                        // expansion and round its magnitude up; everything else takes
                        // Rust's already-correct rounding (so e.g. `(2.355).toFixed(2)`
                        // is "2.35", since the double is 2.35499…, not "2.36").
                        let expanded = alloc::format!("{:.*}", digits + 25, n.abs());
                        let dot = expanded.find('.').unwrap_or(expanded.len());
                        let tail = &expanded[(dot + 1 + digits).min(expanded.len())..];
                        let exact_half = tail.starts_with('5')
                            && tail.as_bytes()[1..].iter().all(|&b| b == b'0');
                        if exact_half {
                            let kept: String = expanded[..dot]
                                .chars()
                                .chain(expanded[dot + 1..dot + 1 + digits].chars())
                                .collect();
                            let m = kept.parse::<u128>().unwrap_or(0) + 1;
                            let mut s = alloc::format!("{m}");
                            if digits > 0 {
                                while s.len() <= digits {
                                    s.insert(0, '0');
                                }
                                s.insert(s.len() - digits, '.');
                            }
                            if n < 0.0 {
                                s.insert(0, '-');
                            }
                            s
                        } else {
                            let mut s = alloc::format!("{n:.digits$}");
                            // A zero result never carries a sign (`(-0).toFixed(2)`).
                            if s.starts_with('-')
                                && s.bytes().all(|b| matches!(b, b'-' | b'0' | b'.'))
                            {
                                s.remove(0);
                            }
                            s
                        }
                    };
                    Some(self.new_str(&s))
                }
                // `toExponential(d)` — exponential notation with `d` fractional
                // digits and a signed exponent (`1.23e+3`).
                "toExponential" => {
                    if !n.is_finite() {
                        // `Infinity`/`-Infinity`/`NaN` use the spec ToString.
                        Some(self.new_str(&self.realm.to_display_string(NanBox::number(n))))
                    } else {
                        let raw = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                            alloc::format!("{n:e}")
                        } else {
                            let d = self.realm.to_number(arg(0)) as usize;
                            alloc::format!("{n:.d$e}")
                        };
                        // Rust prints `1.23e3`; JS wants `1.23e+3`.
                        let fixed = match raw.find('e') {
                            Some(i) if !raw[i + 1..].starts_with('-') => {
                                alloc::format!("{}e+{}", &raw[..i], &raw[i + 1..])
                            }
                            _ => raw,
                        };
                        Some(self.new_str(&fixed))
                    }
                }
                // `toPrecision(p)` — p significant digits (no arg → default
                // string form).
                "toPrecision" => {
                    if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        Some(self.new_str(&self.realm.to_display_string(recv)))
                    } else {
                        let p = (self.realm.to_number(arg(0)) as usize).max(1);
                        Some(self.new_str(&format_precision(n, p)))
                    }
                }
                _ => None,
            });
        }

        let Some(raw) = recv.as_handle() else {
            return Ok(None);
        };
        let handle = Handle::from_raw(raw);

        // --- WeakRef / FinalizationRegistry (bounded: no mid-execution GC) ---
        if method == "deref"
            && let Some(target) = self.realm.get_property(handle, WEAKREF_TARGET)
        {
            return Ok(Some(target));
        }
        if self.realm.get_property(handle, FINREG_TAG).is_some() {
            match method {
                // `register(target, heldValue, unregisterToken?)` — inert.
                "register" => return Ok(Some(NanBox::undefined())),
                // `unregister(token)` — nothing was ever registered.
                "unregister" => return Ok(Some(NanBox::boolean(false))),
                _ => {}
            }
        }

        // --- universal `Object.prototype` methods (own/inherited reflection) ---
        match method {
            "hasOwnProperty" => {
                // `member_key` maps a symbol to its internal slot name (a string key
                // passes through), so a symbol-keyed property is found.
                let key = self.member_key(arg(0));
                return Ok(Some(NanBox::boolean(self.realm.has_own(handle, &key))));
            }
            "isPrototypeOf" => {
                let mut cur = arg(0).as_handle().map(Handle::from_raw);
                while let Some(p) = cur.and_then(|h| self.realm.object_proto(h)) {
                    if p == handle {
                        return Ok(Some(NanBox::boolean(true)));
                    }
                    cur = Some(p);
                }
                return Ok(Some(NanBox::boolean(false)));
            }
            "propertyIsEnumerable" => {
                // True only for an *own* *enumerable* property (a non-enumerable one,
                // or an inherited one, is false). `member_key` resolves symbol keys.
                let key = self.member_key(arg(0));
                let r = self.realm.has_own(handle, &key)
                    && self.realm.property_is_enumerable(handle, &key);
                return Ok(Some(NanBox::boolean(r)));
            }
            // Legacy (Annex B) accessor helpers on Object.prototype.
            "__defineGetter__" => {
                let key = self.realm.to_display_string(arg(0));
                let setter = self
                    .realm
                    .accessor(handle, &key)
                    .map_or(NanBox::undefined(), |(_, s)| s);
                self.realm.define_accessor(handle, &key, arg(1), setter);
                return Ok(Some(NanBox::undefined()));
            }
            "__defineSetter__" => {
                let key = self.realm.to_display_string(arg(0));
                let getter = self
                    .realm
                    .accessor(handle, &key)
                    .map_or(NanBox::undefined(), |(g, _)| g);
                self.realm.define_accessor(handle, &key, getter, arg(1));
                return Ok(Some(NanBox::undefined()));
            }
            "__lookupGetter__" | "__lookupSetter__" => {
                let want_getter = method == "__lookupGetter__";
                let key = self.realm.to_display_string(arg(0));
                let mut cur = Some(handle);
                while let Some(c) = cur {
                    if let Some((g, s)) = self.realm.accessor(c, &key) {
                        return Ok(Some(if want_getter { g } else { s }));
                    }
                    // An own data property shadows an inherited accessor.
                    if self.realm.has_own(c, &key) {
                        break;
                    }
                    cur = self.realm.object_proto(c);
                }
                return Ok(Some(NanBox::undefined()));
            }
            // An error object (`name` + `message`, no own `toString`) renders as
            // `"Name: message"` (or just `"Name"` when the message is empty).
            "toString"
                if self.realm.has_own(handle, "name")
                    && self.realm.has_own(handle, "message")
                    && !self.realm.has_own(handle, "toString") =>
            {
                let name = self
                    .realm
                    .get_property(handle, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                let msg = self
                    .realm
                    .get_property(handle, "message")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                // `Error.prototype.toString`: an empty name yields just the message;
                // an empty message yields just the name; else `"name: message"`.
                let s = if name.is_empty() {
                    msg
                } else if msg.is_empty() {
                    name
                } else {
                    alloc::format!("{name}: {msg}")
                };
                return Ok(Some(self.new_str(&s)));
            }
            _ => {}
        }

        // --- `Function.prototype.call`/`apply`/`bind` on a callable receiver ---
        // `call`/`apply`/`bind` work on any constructor, including a class.
        if self.is_callable(handle) || self.realm.class_at(handle).is_some() {
            match method {
                "call" => {
                    let this = arg(0);
                    let rest: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return self.call_with_this(recv, this, &rest).map(Some);
                }
                "apply" => {
                    let this = arg(0);
                    let list = if let Some(h) = arg(1).as_handle().map(Handle::from_raw) {
                        if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                            elems
                        } else {
                            // An array-like: its `length` and indexed properties.
                            let len = self
                                .realm
                                .get_property(h, "length")
                                .map_or(0, |v| self.realm.to_number(v).max(0.0) as usize);
                            let mut v = Vec::with_capacity(len);
                            for i in 0..len {
                                v.push(self.read_member(h, &alloc::format!("{i}"))?);
                            }
                            v
                        }
                    } else {
                        Vec::new()
                    };
                    return self.call_with_this(recv, this, &list).map(Some);
                }
                "bind" => {
                    let this = arg(0);
                    let bound: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return Ok(Some(self.make_bound_function(recv, this, bound)));
                }
                // A textual representation (the engine does not retain source).
                "toString" | "toLocaleString" => {
                    let nm = self.read_member(handle, "name")?;
                    let nm = self.realm.to_display_string(nm);
                    let s = if self.realm.class_at(handle).is_some() {
                        alloc::format!("class {nm} {{ }}")
                    } else {
                        alloc::format!("function {nm}() {{ [native code] }}")
                    };
                    return Ok(Some(self.new_str(&s)));
                }
                _ => {}
            }
        }

        // --- generator iterator protocol (`next`/`return`) ---
        if let Some(buf) = self
            .realm
            .get_property(handle, GEN_BUF)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
        {
            match method {
                "next" => {
                    let idx = self
                        .realm
                        .get_property(handle, GEN_IDX)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    let elems = self.realm.array_elements(buf).map(<[_]>::to_vec);
                    let len = elems.as_ref().map_or(0, Vec::len);
                    let (value, done) = match elems.as_ref().and_then(|e| e.get(idx)) {
                        Some(v) => {
                            self.realm.set_hidden_property(
                                handle,
                                GEN_IDX,
                                NanBox::number((idx + 1) as f64),
                            );
                            (*v, false)
                        }
                        // The first call past the yields surfaces the `return`
                        // value (with `done: true`); later calls yield undefined.
                        None => {
                            let v = if idx == len {
                                self.realm.set_hidden_property(
                                    handle,
                                    GEN_IDX,
                                    NanBox::number((idx + 1) as f64),
                                );
                                self.realm
                                    .get_property(handle, GEN_RET)
                                    .unwrap_or(NanBox::undefined())
                            } else {
                                NanBox::undefined()
                            };
                            (v, true)
                        }
                    };
                    let res = self.realm.new_object();
                    self.realm.set_property(res, "value", value);
                    self.realm.set_property(res, "done", NanBox::boolean(done));
                    return Ok(Some(NanBox::handle(res.to_raw())));
                }
                // `return()` ends the generator early.
                "return" => {
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    let res = self.realm.new_object();
                    self.realm.set_property(res, "value", arg(0));
                    self.realm.set_property(res, "done", NanBox::boolean(true));
                    return Ok(Some(NanBox::handle(res.to_raw())));
                }
                "throw" => {
                    // Eager-generator model: the body has already run, so the thrown
                    // value can't be re-injected at the suspended `yield` (a
                    // `try`/`catch` *around* that yield won't observe it). Mark the
                    // generator done and propagate the value — correct when the
                    // generator does not catch at the yield (the common case) and for
                    // an already-exhausted generator.
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    return Err(ExecError::Throw(arg(0)));
                }
                // ES2025 iterator helpers — they consume the remaining yields.
                "map" | "filter" | "take" | "drop" | "toArray" | "forEach" | "reduce" | "some"
                | "every" | "find" | "flatMap" => {
                    let idx = self
                        .realm
                        .get_property(handle, GEN_IDX)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    let rest: Vec<NanBox> = self
                        .realm
                        .array_elements(buf)
                        .map(|e| e.get(idx..).unwrap_or(&[]).to_vec())
                        .unwrap_or_default();
                    // The source iterator is now exhausted.
                    let len = self.realm.array_elements(buf).map_or(0, <[_]>::len);
                    self.realm
                        .set_hidden_property(handle, GEN_IDX, NanBox::number(len as f64));
                    let f = arg(0);
                    return Ok(Some(match method {
                        "toArray" => NanBox::handle(self.realm.new_array(rest).to_raw()),
                        "map" => {
                            let mut out = Vec::with_capacity(rest.len());
                            for v in rest {
                                out.push(self.call(f, &[v])?);
                            }
                            self.make_generator(out)
                        }
                        "flatMap" => {
                            let mut out = Vec::new();
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                out.extend(
                                    self.iterate_values(r).unwrap_or_else(|_| alloc::vec![r]),
                                );
                            }
                            self.make_generator(out)
                        }
                        "filter" => {
                            let mut out = Vec::new();
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                if self.realm.truthy(r) {
                                    out.push(v);
                                }
                            }
                            self.make_generator(out)
                        }
                        "take" => {
                            let n = self.realm.to_number(f).max(0.0) as usize;
                            self.make_generator(rest.into_iter().take(n).collect())
                        }
                        "drop" => {
                            let n = self.realm.to_number(f).max(0.0) as usize;
                            self.make_generator(rest.into_iter().skip(n).collect())
                        }
                        "forEach" => {
                            for v in rest {
                                self.call(f, &[v])?;
                            }
                            NanBox::undefined()
                        }
                        "some" | "every" | "find" => {
                            let mut found = NanBox::undefined();
                            let mut hit = false;
                            for v in rest {
                                let r = self.call(f, &[v])?;
                                let t = self.realm.truthy(r);
                                if method == "every" && !t {
                                    return Ok(Some(NanBox::boolean(false)));
                                }
                                if method != "every" && t {
                                    found = v;
                                    hit = true;
                                    break;
                                }
                            }
                            match method {
                                "some" => NanBox::boolean(hit),
                                "every" => NanBox::boolean(true),
                                _ => found, // find
                            }
                        }
                        // reduce
                        _ => {
                            let mut it = rest.into_iter();
                            let mut acc = if args.len() >= 2 {
                                arg(1)
                            } else {
                                match it.next() {
                                    Some(v) => v,
                                    None => {
                                        let m = self.new_str(
                                            "Reduce of empty iterator with no initial value",
                                        );
                                        return Err(ExecError::Throw(
                                            self.make_error(N_TYPE_ERROR, Some(m)),
                                        ));
                                    }
                                }
                            };
                            for v in it {
                                acc = self.call(f, &[acc, v])?;
                            }
                            acc
                        }
                    }));
                }
                _ => {}
            }
        }

        // --- `Date.now()` static ---
        // `BigInt.asUintN(bits, x)` / `BigInt.asIntN(bits, x)` — wrap a BigInt to
        // the low `bits` bits, unsigned or signed (two's complement).
        if self.realm.native_at(handle) == Some(N_BIGINT) && matches!(method, "asUintN" | "asIntN")
        {
            use crate::bignum::BigInt;
            let bits = self.realm.to_number(arg(0)).max(0.0) as u64;
            let x = arg(1)
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.bigint_at(h))
                .unwrap_or_else(BigInt::zero);
            let modulus = BigInt::from_i128(2).pow(bits);
            // Non-negative remainder modulo 2^bits.
            let mut u = x.divmod(&modulus).map_or_else(BigInt::zero, |(_, r)| r);
            if u.is_negative() {
                u = u.add(&modulus);
            }
            if method == "asIntN" && bits >= 1 {
                // If the top bit is set, the signed value is `u - 2^bits`.
                let half = BigInt::from_i128(2).pow(bits - 1);
                if !u.sub(&half).is_negative() {
                    u = u.sub(&modulus);
                }
            }
            return Ok(Some(NanBox::handle(self.realm.new_bigint(u).to_raw())));
        }
        if self.realm.native_at(handle) == Some(N_DATE) && method == "now" {
            return Ok(Some(NanBox::number(now_ms())));
        }
        // `Uint8Array.of(...items)` / `Uint8Array.from(iterable|arrayLike, mapFn?)`
        // — the typed-array statics, producing a typed array of the constructor's
        // kind (each value coerced to the element type).
        if let Some(id) = self.realm.native_at(handle)
            && (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
            && matches!(method, "of" | "from")
        {
            let kind = id - N_TYPED_ARRAY_BASE;
            let mut items: Vec<NanBox> = if method == "of" {
                args.to_vec()
            } else {
                self.iterate_values(arg(0)).unwrap_or_default()
            };
            // `from`'s optional map callback `(value, index)`.
            if method == "from"
                && let Some(mapfn) = args.get(1).copied()
                && mapfn
                    .as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                for (i, v) in items.iter_mut().enumerate() {
                    *v = self.call(mapfn, &[*v, NanBox::number(i as f64)])?;
                }
            }
            let elems: Vec<NanBox> = items
                .iter()
                .map(|v| NanBox::number(coerce_typed(kind, self.realm.to_number(*v))))
                .collect();
            let arr = self.realm.new_array(elems);
            self.realm
                .set_property(arr, TYPED_ARRAY_KIND, NanBox::number(f64::from(kind)));
            return Ok(Some(NanBox::handle(arr.to_raw())));
        }
        // `Date.parse(str)` → epoch ms (or NaN) by ISO parsing.
        if self.realm.native_at(handle) == Some(N_DATE) && method == "parse" {
            let s = self.realm.to_display_string(arg(0));
            return Ok(Some(NanBox::number(
                crate::realm::parse_iso_date(&s).unwrap_or(f64::NAN),
            )));
        }
        // --- `Date.UTC(year, month, day?, h?, m?, s?, ms?)` → epoch ms ---
        if self.realm.native_at(handle) == Some(N_DATE) && method == "UTC" {
            let num = |i: usize, dflt: f64| args.get(i).map_or(dflt, |a| self.realm.to_number(*a));
            let year = num(0, 1970.0) as i64;
            let month = num(1, 0.0) as i64;
            let day = num(2, 1.0) as i64;
            let total_months = year * 12 + month;
            let y = total_months.div_euclid(12);
            let mo = total_months.rem_euclid(12) as u32 + 1;
            let days = crate::realm::days_from_civil(y, mo, day as u32);
            let ms = (days * 86_400_000
                + (num(3, 0.0) as i64) * 3_600_000
                + (num(4, 0.0) as i64) * 60_000
                + (num(5, 0.0) as i64) * 1_000
                + num(6, 0.0) as i64) as f64;
            return Ok(Some(NanBox::number(ms)));
        }
        // --- `Proxy.revocable(target, handler)` → `{ proxy, revoke }` ---
        if self.realm.native_at(handle) == Some(N_PROXY) && method == "revocable" {
            let (Some(tr), Some(hr)) = (arg(0).as_handle(), arg(1).as_handle()) else {
                let m = self.new_str("Cannot create proxy with a non-object target or handler");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let proxy = self
                .realm
                .new_proxy(Handle::from_raw(tr), Handle::from_raw(hr));
            let revoke = self.realm.new_bound_native(N_PROXY_REVOKE, proxy);
            let result = self.realm.new_object();
            self.realm
                .set_property(result, "proxy", NanBox::handle(proxy.to_raw()));
            self.realm
                .set_property(result, "revoke", NanBox::handle(revoke.to_raw()));
            return Ok(Some(NanBox::handle(result.to_raw())));
        }
        // --- `Symbol.for` / `Symbol.keyFor` (the global symbol registry) ---
        if self.realm.native_at(handle) == Some(N_SYMBOL) {
            match method {
                "for" => {
                    let key = self.realm.to_display_string(arg(0));
                    if let Some(s) = self.symbol_registry.get(&key) {
                        return Ok(Some(*s));
                    }
                    let sym = NanBox::handle(self.realm.new_symbol(&key).to_raw());
                    self.symbol_registry.insert(key, sym);
                    return Ok(Some(sym));
                }
                "keyFor" => {
                    let target = arg(0);
                    let found = self
                        .symbol_registry
                        .iter()
                        .find(|(_, v)| self.realm.strict_equals(**v, target))
                        .map(|(k, _)| k.clone());
                    return Ok(Some(match found {
                        Some(k) => self.new_str(&k),
                        None => NanBox::undefined(),
                    }));
                }
                _ => {}
            }
        }
        // --- symbol instance: `sym.toString()` ---
        if let Some((desc, _)) = self.realm.symbol_at(handle)
            && method == "toString"
        {
            // A no-argument `Symbol()` has an empty (undefined) description.
            let shown = if desc.starts_with('\u{0}') { "" } else { &desc };
            return Ok(Some(self.new_str(&alloc::format!("Symbol({shown})"))));
        }
        // --- BigInt instance: `toString(radix)` / `valueOf` ---
        if let Some(big) = self.realm.bigint_at(handle) {
            match method {
                "toString" => {
                    let radix = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        10
                    } else {
                        self.realm.to_number(arg(0)) as u32
                    };
                    return Ok(Some(self.new_str(&bigint_to_radix(&big, radix))));
                }
                "valueOf" => return Ok(Some(NanBox::handle(self.realm.new_bigint(big).to_raw()))),
                _ => {}
            }
        }
        // --- `Number.*` / `String.*` statics (on the constructor) ---
        match self.realm.native_at(handle) {
            Some(N_NUMBER) => {
                match method {
                    "isInteger" => {
                        let is_int = arg(0)
                            .as_number()
                            .is_some_and(|n| n.is_finite() && (n as i64) as f64 == n);
                        return Ok(Some(NanBox::boolean(is_int)));
                    }
                    "isSafeInteger" => {
                        let safe = arg(0).as_number().is_some_and(|n| {
                            n.is_finite()
                                && (n as i64) as f64 == n
                                && n.abs() <= 9_007_199_254_740_991.0
                        });
                        return Ok(Some(NanBox::boolean(safe)));
                    }
                    "isFinite" => {
                        return Ok(Some(NanBox::boolean(
                            arg(0).as_number().is_some_and(f64::is_finite),
                        )));
                    }
                    "isNaN" => {
                        return Ok(Some(NanBox::boolean(
                            arg(0).as_number().is_some_and(f64::is_nan),
                        )));
                    }
                    "parseFloat" => return Ok(Some(self.call_native(N_PARSE_FLOAT, args)?)),
                    "parseInt" => return Ok(Some(self.call_native(N_PARSE_INT, args)?)),
                    _ => {}
                };
            }
            Some(N_STRING) if method == "fromCharCode" => {
                // Each argument is ToUint16'd into a UTF-16 code unit; the resulting
                // sequence is then decoded, so an adjacent high/low surrogate pair
                // combines into one astral code point (a lone surrogate, which UTF-8
                // can't store, becomes U+FFFD).
                let units: Vec<u16> = args
                    .iter()
                    .map(|a| {
                        let n = self.realm.to_number(*a);
                        if n.is_finite() {
                            // ToUint16: truncate toward zero, then take mod 2^16.
                            (n as i64).rem_euclid(65536) as u16
                        } else {
                            0
                        }
                    })
                    .collect();
                let s: String = char::decode_utf16(units)
                    .map(|r| r.unwrap_or('\u{FFFD}'))
                    .collect();
                return Ok(Some(self.new_str(&s)));
            }
            // `String.fromCodePoint(...cps)` — each argument is a full Unicode
            // code point (may be astral).
            Some(N_STRING) if method == "fromCodePoint" => {
                let s: String = args
                    .iter()
                    .filter_map(|a| char::from_u32(self.realm.to_number(*a) as u32))
                    .collect();
                return Ok(Some(self.new_str(&s)));
            }
            // `String.raw(strings, ...subs)` — interleave `strings.raw[i]` with
            // each substitution (the cooked-escape-free template form).
            Some(N_STRING) if method == "raw" => {
                let raw = arg(0)
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.get_property(h, "raw"))
                    .and_then(|r| r.as_handle())
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
                    .unwrap_or_default();
                let subs = &args[1.min(args.len())..];
                let mut out = String::new();
                for (i, piece) in raw.iter().enumerate() {
                    out.push_str(&self.realm.to_display_string(*piece));
                    if let Some(s) = subs.get(i) {
                        out.push_str(&self.realm.to_display_string(*s));
                    }
                }
                return Ok(Some(self.new_str(&out)));
            }
            _ => {}
        }
        // --- ArrayBuffer.prototype.slice(begin?, end?) → a new ArrayBuffer copy ---
        if method == "slice"
            && let Some(bytesv) = self.realm.get_property(handle, ARRAY_BUFFER_BYTES)
            && let Some(bh) = bytesv.as_handle().map(Handle::from_raw)
        {
            let elems = self
                .realm
                .array_elements(bh)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            let len = elems.len() as i64;
            let norm = |this: &mut Self, v: NanBox, default: i64| -> usize {
                if matches!(v.unpack(), Unpacked::Undefined) {
                    return default as usize;
                }
                let n = this.realm.to_number(v) as i64;
                usize::try_from(if n < 0 { (len + n).max(0) } else { n.min(len) }).unwrap_or(0)
            };
            let begin = norm(self, arg(0), 0);
            let end = norm(self, arg(1), len);
            let sub = elems.get(begin..end.max(begin)).unwrap_or(&[]).to_vec();
            let nb = self.realm.new_object();
            let arr = self.realm.new_array(sub);
            self.realm
                .set_hidden_property(nb, ARRAY_BUFFER_BYTES, NanBox::handle(arr.to_raw()));
            return Ok(Some(NanBox::handle(nb.to_raw())));
        }
        // --- DataView get*/set* ---
        if let Some(bufv) = self.realm.get_property(handle, DATA_VIEW_BUF)
            && let Some((is_set, size, signed, is_float, is_bigint)) = dataview_method(method)
        {
            let bytes_h = bufv
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, ARRAY_BUFFER_BYTES))
                .and_then(|b| b.as_handle())
                .map(Handle::from_raw);
            let Some(bh) = bytes_h else {
                let m = self.new_str("DataView has no buffer");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            };
            let base = self
                .realm
                .get_property(handle, DATA_VIEW_OFF)
                .and_then(|n| n.as_number())
                .unwrap_or(0.0) as usize;
            // Bounds-check the access against the view's byte length (an explicit
            // `DATA_VIEW_LEN`, else the rest of the buffer). A negative/too-large offset
            // or an access running past the end is a RangeError — never an out-of-bounds
            // read (returning 0) or a write that silently grows the buffer.
            let total = self.realm.array_length(bh).unwrap_or(0);
            let view_len = self
                .realm
                .get_property(handle, DATA_VIEW_LEN)
                .and_then(|n| n.as_number())
                .map_or(total.saturating_sub(base), |n| n as usize);
            let requested = self.realm.to_number(arg(0)) as i64; // ToIndex: truncates; NaN -> 0
            if requested < 0 || requested as usize + size > view_len {
                let m = self.new_str("Offset is outside the bounds of the DataView");
                return Err(ExecError::Throw(self.make_error(N_RANGE_ERROR, Some(m))));
            }
            let abs = base + requested as usize;
            if is_set {
                let le = self.realm.truthy(arg(2));
                let bits = if is_bigint {
                    // `setBigInt64`/`setBigUint64`: the value is a BigInt; its low
                    // 64 bits are stored.
                    let big = arg(1)
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.bigint_at(h));
                    big.and_then(|b| b.to_i128()).unwrap_or(0) as u64
                } else if is_float {
                    let value = self.realm.to_number(arg(1));
                    if size == 4 {
                        u64::from((value as f32).to_bits())
                    } else {
                        value.to_bits()
                    }
                } else {
                    self.realm.to_number(arg(1)) as i64 as u64
                };
                for i in 0..size {
                    let shift = if le { i } else { size - 1 - i };
                    let byte = (bits >> (8 * shift)) & 0xff;
                    self.realm
                        .set_element(bh, abs + i, NanBox::number(byte as f64));
                }
                return Ok(Some(NanBox::undefined()));
            }
            let le = self.realm.truthy(arg(1));
            let mut bits: u64 = 0;
            for i in 0..size {
                let b = self
                    .realm
                    .array_elements(bh)
                    .and_then(|e| e.get(abs + i).copied())
                    .and_then(|v| v.as_number())
                    .unwrap_or(0.0) as u64
                    & 0xff;
                let shift = if le { i } else { size - 1 - i };
                bits |= b << (8 * shift);
            }
            if is_bigint {
                // `getBigInt64` reinterprets the 64 bits as a signed i64; `getBigUint64`
                // as an unsigned u64 — both returned as a BigInt.
                let big = if signed {
                    crate::bignum::BigInt::from_i128(i128::from(bits as i64))
                } else {
                    crate::bignum::BigInt::from_i128(i128::from(bits))
                };
                return Ok(Some(NanBox::handle(self.realm.new_bigint(big).to_raw())));
            }
            let value = if is_float {
                if size == 4 {
                    f64::from(f32::from_bits(bits as u32))
                } else {
                    f64::from_bits(bits)
                }
            } else if signed && size < 8 && bits & (1 << (8 * size - 1)) != 0 {
                (bits as i64 - (1i64 << (8 * size))) as f64
            } else {
                bits as f64
            };
            return Ok(Some(NanBox::number(value)));
        }

        // --- Intl.NumberFormat / Intl.DateTimeFormat instance methods ---
        if self.realm.get_property(handle, "\u{0}intl").is_some() && method == "format" {
            let s = self.intl_format_value(handle, arg(0));
            return Ok(Some(self.new_str(&s)));
        }
        // --- Date instance methods ---
        if let Some(ms) = self.realm.date_at(handle) {
            let t = ms as i64;
            let day = t.div_euclid(86_400_000);
            let tod = t.rem_euclid(86_400_000);
            let (y, mo, d) = crate::realm::civil_from_days(day);
            return Ok(Some(match method {
                // The engine models all dates in UTC, so `getUTC*` aliases `get*`.
                "getTime" | "valueOf" => NanBox::number(ms),
                "getFullYear" | "getUTCFullYear" => NanBox::number(y as f64),
                "getMonth" | "getUTCMonth" => NanBox::number((mo - 1) as f64), // 0-based
                "getDate" | "getUTCDate" => NanBox::number(d as f64),
                "getDay" | "getUTCDay" => {
                    NanBox::number((day.rem_euclid(7) + 4).rem_euclid(7) as f64)
                }
                "getHours" | "getUTCHours" => NanBox::number((tod / 3_600_000) as f64),
                "getMinutes" | "getUTCMinutes" => NanBox::number((tod / 60_000 % 60) as f64),
                "getSeconds" | "getUTCSeconds" => NanBox::number((tod / 1000 % 60) as f64),
                "getMilliseconds" | "getUTCMilliseconds" => NanBox::number((tod % 1000) as f64),
                // The engine models all dates in UTC, so the local offset is 0.
                "getTimezoneOffset" => NanBox::number(0.0),
                // `toISOString` throws on an invalid date; `toJSON` returns null.
                "toISOString" => {
                    if !ms.is_finite() {
                        let m = self.new_str("Invalid time value");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    self.new_str(&crate::realm::date_to_iso(ms))
                }
                "toJSON" => {
                    if ms.is_finite() {
                        self.new_str(&crate::realm::date_to_iso(ms))
                    } else {
                        NanBox::null()
                    }
                }
                // Human-readable forms (the engine is UTC, so `GMT+0000`).
                "toDateString" | "toTimeString" | "toString" | "toUTCString"
                | "toLocaleDateString" | "toLocaleTimeString" | "toLocaleString" => {
                    // An invalid date (NaN timestamp) stringifies as "Invalid Date".
                    if !ms.is_finite() {
                        return Ok(Some(self.new_str("Invalid Date")));
                    }
                    let wd = WEEKDAYS[((day.rem_euclid(7) + 4).rem_euclid(7)) as usize];
                    let mn = MONTHS[(mo - 1) as usize];
                    let (hh, mi, ss) = (tod / 3_600_000, tod / 60_000 % 60, tod / 1000 % 60);
                    let date_str = alloc::format!("{wd} {mn} {d:02} {y}");
                    let time_str = alloc::format!(
                        "{hh:02}:{mi:02}:{ss:02} GMT+0000 (Coordinated Universal Time)"
                    );
                    let s = match method {
                        "toDateString" => date_str,
                        "toTimeString" => time_str,
                        "toUTCString" => {
                            alloc::format!("{wd}, {d:02} {mn} {y} {hh:02}:{mi:02}:{ss:02} GMT")
                        }
                        "toLocaleDateString" => alloc::format!("{mo}/{d}/{y}"),
                        "toLocaleTimeString" => alloc::format!("{hh:02}:{mi:02}:{ss:02}"),
                        "toLocaleString" => {
                            alloc::format!("{mo}/{d}/{y}, {hh:02}:{mi:02}:{ss:02}")
                        }
                        // `toString`
                        _ => alloc::format!("{date_str} {time_str}"),
                    };
                    self.new_str(&s)
                }
                // --- `set*` mutators (all UTC; a setter returns the new time) ---
                "setTime" => {
                    let nms = self.realm.to_number(arg(0));
                    self.realm.set_date_ms(handle, nms);
                    NanBox::number(nms)
                }
                "setFullYear" | "setUTCFullYear" | "setMonth" | "setUTCMonth" | "setDate"
                | "setUTCDate" | "setHours" | "setUTCHours" | "setMinutes" | "setUTCMinutes"
                | "setSeconds" | "setUTCSeconds" | "setMilliseconds" | "setUTCMilliseconds" => {
                    // Decompose the current time, replace one field, recompose.
                    let (mut yy, mut mo0, mut dd) = (y, (mo as i64) - 1, d as i64);
                    let mut hh = tod / 3_600_000;
                    let mut mi = tod / 60_000 % 60;
                    let mut ss = tod / 1000 % 60;
                    let mut mss = tod % 1000;
                    let n = self.realm.to_number(arg(0)) as i64;
                    match method {
                        "setFullYear" | "setUTCFullYear" => yy = n,
                        "setMonth" | "setUTCMonth" => mo0 = n,
                        "setDate" | "setUTCDate" => dd = n,
                        "setHours" | "setUTCHours" => hh = n,
                        "setMinutes" | "setUTCMinutes" => mi = n,
                        "setSeconds" | "setUTCSeconds" => ss = n,
                        _ => mss = n, // setMilliseconds
                    }
                    // Normalize a possibly out-of-range month into the year, then
                    // measure the day as an offset from the 1st (so out-of-range
                    // day/hour/… values roll over via plain integer arithmetic).
                    let yy2 = yy + mo0.div_euclid(12);
                    let mo1 = (mo0.rem_euclid(12) + 1) as u32;
                    let base_days = crate::realm::days_from_civil(yy2, mo1, 1) + (dd - 1);
                    let nms =
                        (base_days * 86_400_000 + hh * 3_600_000 + mi * 60_000 + ss * 1000 + mss)
                            as f64;
                    self.realm.set_date_ms(handle, nms);
                    NanBox::number(nms)
                }
                _ => return Ok(None),
            }));
        }
        // --- RegExp instance methods (`test`/`exec`) ---
        if let Some((source, flags)) = self.realm.regexp_at(handle) {
            let _ = (&source, &flags);
            #[cfg(feature = "regex")]
            {
                let text = self.realm.to_display_string(arg(0));
                let re = crate::regex::Regex::new(&source, &flags);
                if matches!(method, "test" | "exec")
                    && let Ok(re) = re
                {
                    // `g`/`y` regexes are stateful: search resumes at `lastIndex`
                    // and is updated to the match end (or reset to 0 on no match).
                    let stateful = flags.contains('g') || flags.contains('y');
                    let start = if stateful {
                        self.realm.regex_last_index(handle)
                    } else {
                        0
                    };
                    let caps = re.captures_from(&text, start);
                    if stateful {
                        let next = caps.as_ref().map_or(0, |c| c.whole().1);
                        self.realm.set_regex_last_index(handle, next);
                    }
                    return Ok(Some(match (method, caps) {
                        ("test", c) => NanBox::boolean(c.is_some()),
                        (_, Some(c)) => self.regex_match_object(&text, &c, re.group_names()),
                        (_, None) => NanBox::null(),
                    }));
                }
            }
            #[cfg(not(feature = "regex"))]
            if matches!(method, "test" | "exec") {
                return Err(ExecError::Unsupported("RegExp needs the regex feature"));
            }
        }
        // `Map.groupBy(items, cb)` — like `Object.groupBy` but a Map (keys are
        // the callback's return value as-is, so objects work as group keys).
        if self.realm.native_at(handle) == Some(N_MAP) && method == "groupBy" {
            let items = self.iterate_values(arg(0))?;
            let cb = arg(1);
            let map = self.realm.new_collection(false);
            for (i, item) in items.iter().enumerate() {
                let key = self.call(cb, &[*item, NanBox::number(i as f64)])?;
                let bucket = match self
                    .realm
                    .collection_get(map, key)
                    .and_then(NanBox::as_handle)
                    .map(Handle::from_raw)
                {
                    Some(h) => h,
                    None => {
                        let arr = self.realm.new_array(Vec::new());
                        self.realm
                            .collection_set(map, key, NanBox::handle(arr.to_raw()));
                        arr
                    }
                };
                self.realm.array_push(bucket, *item);
            }
            return Ok(Some(NanBox::handle(map.to_raw())));
        }
        // --- `Promise.resolve` / `Promise.reject` statics (on the constructor) ---
        if self.realm.native_at(handle) == Some(N_PROMISE) {
            match method {
                "resolve" => {
                    // `Promise.resolve(x)` is idempotent on a promise: if `x` is
                    // already a promise, return it unchanged (same identity).
                    if let Some(raw) = arg(0).as_handle()
                        && self.realm.promise_state(Handle::from_raw(raw)).is_some()
                    {
                        return Ok(Some(arg(0)));
                    }
                    let p = self.realm.new_promise();
                    self.resolve_with(p, arg(0));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                "reject" => {
                    let p = self.realm.new_promise();
                    self.settle(p, arg(0), false);
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.withResolvers()` → `{ promise, resolve, reject }`.
                "withResolvers" => {
                    let p = self.realm.new_promise();
                    let resolve = self.realm.new_bound_native(N_RESOLVE, p);
                    let reject = self.realm.new_bound_native(N_REJECT, p);
                    let obj = self.realm.new_object();
                    self.realm
                        .set_property(obj, "promise", NanBox::handle(p.to_raw()));
                    self.realm
                        .set_property(obj, "resolve", NanBox::handle(resolve.to_raw()));
                    self.realm
                        .set_property(obj, "reject", NanBox::handle(reject.to_raw()));
                    return Ok(Some(NanBox::handle(obj.to_raw())));
                }
                // `Promise.all(iterable)`: resolve with the array of awaited
                // values, or reject with the first rejection (eager model).
                "all" => {
                    let items = self.iterate_values(arg(0))?;
                    let p = self.realm.new_promise();
                    let mut results = Vec::with_capacity(items.len());
                    for item in items {
                        match self.await_value(item) {
                            Ok(v) => results.push(v),
                            Err(ExecError::Throw(e)) => {
                                self.settle(p, e, false);
                                return Ok(Some(NanBox::handle(p.to_raw())));
                            }
                            Err(other) => return Err(other),
                        }
                    }
                    let arr = self.realm.new_array(results);
                    self.resolve_with(p, NanBox::handle(arr.to_raw()));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.race(iterable)`: settle with the first input to *settle*.
                // Steps the event loop, checking the inputs after each task, so a
                // timer-backed promise that settles first wins (ties in a single step
                // broken by list order).
                "race" => {
                    let items = self.iterate_values(arg(0))?;
                    let p = self.realm.new_promise();
                    'race: loop {
                        for item in &items {
                            match self.settled_state(*item) {
                                Some(Ok(v)) => {
                                    self.resolve_with(p, v);
                                    break 'race;
                                }
                                Some(Err(e)) => {
                                    self.settle(p, e, false);
                                    break 'race;
                                }
                                None => {}
                            }
                        }
                        // None settled yet: advance the loop, or stop if it is idle
                        // (the race promise then stays pending, as the spec requires).
                        if self.microtasks.is_empty() && self.macrotasks.is_empty() {
                            break;
                        }
                        if self.microtasks.is_empty() {
                            self.run_one_macrotask()?;
                        } else {
                            self.run_one_microtask()?;
                        }
                    }
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.allSettled(iterable)`: never rejects; each entry is
                // `{status, value}` or `{status, reason}`.
                "allSettled" => {
                    let items = self.iterate_values(arg(0))?;
                    let mut results = Vec::with_capacity(items.len());
                    for item in items {
                        let obj = self.realm.new_object();
                        match self.await_value(item) {
                            Ok(v) => {
                                let s = self.new_str("fulfilled");
                                self.realm.set_property(obj, "status", s);
                                self.realm.set_property(obj, "value", v);
                            }
                            Err(ExecError::Throw(e)) => {
                                let s = self.new_str("rejected");
                                self.realm.set_property(obj, "status", s);
                                self.realm.set_property(obj, "reason", e);
                            }
                            Err(other) => return Err(other),
                        }
                        results.push(NanBox::handle(obj.to_raw()));
                    }
                    let p = self.realm.new_promise();
                    let arr = self.realm.new_array(results);
                    self.resolve_with(p, NanBox::handle(arr.to_raw()));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                // `Promise.any(iterable)`: fulfills with the first input to
                // fulfill; rejects with an `AggregateError` if all reject.
                "any" => {
                    let items = self.iterate_values(arg(0))?;
                    let p = self.realm.new_promise();
                    let mut errors = Vec::new();
                    for item in items {
                        match self.await_value(item) {
                            Ok(v) => {
                                self.resolve_with(p, v);
                                return Ok(Some(NanBox::handle(p.to_raw())));
                            }
                            Err(ExecError::Throw(e)) => errors.push(e),
                            Err(other) => return Err(other),
                        }
                    }
                    // None fulfilled: reject with an AggregateError holding them.
                    let agg = self.realm.new_object();
                    let name = self.new_str("AggregateError");
                    self.realm.set_property(agg, "name", name);
                    let msg = self.new_str("All promises were rejected");
                    self.realm.set_property(agg, "message", msg);
                    let errs = self.realm.new_array(errors);
                    self.realm
                        .set_property(agg, "errors", NanBox::handle(errs.to_raw()));
                    self.settle(p, NanBox::handle(agg.to_raw()), false);
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                _ => {}
            }
        }
        // --- promise instance methods (`then`/`catch`/`finally`) ---
        if self.realm.promise_state(handle).is_some() {
            match method {
                "then" => return Ok(Some(self.promise_then(handle, arg(0), arg(1)))),
                "catch" => {
                    return Ok(Some(self.promise_then(handle, NanBox::undefined(), arg(0))));
                }
                "finally" => {
                    // The callback runs on either settlement for side effects; the
                    // original value/rejection passes through to the new promise.
                    let cb = arg(0);
                    let result = self.register_then(handle, cb, cb, true);
                    return Ok(Some(NanBox::handle(result.to_raw())));
                }
                _ => {}
            }
        }

        // A custom matcher/replacer: when the argument defines the matching
        // well-known symbol method (`Symbol.match`/`replace`/`search`/`split`/
        // `matchAll`), `str.method(obj)` delegates to `obj[@@method](str, …rest)`.
        if let Some(s) = self.realm.string_value(handle)
            && let Some(sym_name) = match method {
                "match" => Some("match"),
                "matchAll" => Some("matchAll"),
                "search" => Some("search"),
                "replace" | "replaceAll" => Some("replace"),
                "split" => Some("split"),
                _ => None,
            }
            && let Some(argh) = arg(0).as_handle().map(Handle::from_raw)
        {
            let sym = self.well_known_symbol(sym_name);
            let key = self.member_key(sym);
            let m = self.read_member(argh, &key)?;
            if m.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let this_str = self.new_str(&s);
                let mut call_args = alloc::vec![this_str];
                call_args.extend_from_slice(&args[1.min(args.len())..]);
                return Ok(Some(self.call_with_this(m, arg(0), &call_args)?));
            }
        }

        // --- regex-backed String methods (when the argument is a RegExp) ---
        #[cfg(feature = "regex")]
        if let Some(s) = self.realm.string_value(handle)
            && matches!(
                method,
                "match" | "matchAll" | "search" | "replace" | "replaceAll" | "split"
            )
            && let Some((src, flags)) = arg(0)
                .as_handle()
                .and_then(|raw| self.realm.regexp_at(Handle::from_raw(raw)))
            && let Ok(re) = crate::regex::Regex::new(&src, &flags)
        {
            let global = flags.contains('g');
            // `matchAll`/`replaceAll` require a global RegExp.
            if !global && matches!(method, "matchAll" | "replaceAll") {
                let m = self.new_str(&alloc::format!(
                    "{method} must be called with a global RegExp"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            match method {
                "search" => {
                    let idx = re.find_from(&s, 0).map_or(-1.0, |(st, _)| st as f64);
                    return Ok(Some(NanBox::number(idx)));
                }
                "match" if !global => {
                    return Ok(Some(match re.captures_from(&s, 0) {
                        Some(caps) => self.regex_match_object(&s, &caps, re.group_names()),
                        None => NanBox::null(),
                    }));
                }
                "match" => {
                    // Global: an array of all whole matches (or null).
                    let mut out = Vec::new();
                    let mut at = 0;
                    while let Some((st, en)) = re.find_from(&s, at) {
                        out.push(self.new_str(&char_substr(&s, st, en)));
                        at = if en > st { en } else { en + 1 };
                    }
                    return Ok(Some(if out.is_empty() {
                        NanBox::null()
                    } else {
                        NanBox::handle(self.realm.new_array(out).to_raw())
                    }));
                }
                // `matchAll` — an iterator of match-result arrays `[full, g1, …]`.
                "matchAll" => {
                    let mut out = Vec::new();
                    let mut at = 0;
                    while let Some(caps) = re.captures_from(&s, at) {
                        let Some((st, en)) = caps.groups[0] else {
                            break;
                        };
                        // A full match-result object (indexed groups + `.groups`
                        // named captures + `.index`/`.input`).
                        out.push(self.regex_match_object(&s, &caps, re.group_names()));
                        at = if en > st { en } else { en + 1 };
                        if at > s.len() {
                            break;
                        }
                    }
                    return Ok(Some(self.make_generator(out)));
                }
                "split" => {
                    // An optional limit caps the number of result segments.
                    let limit = match args.get(1) {
                        Some(a) if !matches!(a.unpack(), Unpacked::Undefined) => {
                            let n = self.realm.to_number(*a);
                            if n >= 0.0 { Some(n as usize) } else { None }
                        }
                        _ => None,
                    };
                    let mut parts = Vec::new();
                    // `seg_start` begins the current segment; `search` is where the
                    // next match is sought (advanced past zero-width matches so a
                    // lookahead split doesn't drop a character).
                    let mut seg_start = 0;
                    let mut search = 0;
                    // Match positions are `< len` (the spec's `q < size`); the tail
                    // after the last split is appended once, below.
                    while search < s.len() && limit.is_none_or(|l| parts.len() < l) {
                        let Some(caps) = re.captures_from(&s, search) else {
                            break;
                        };
                        let Some((st, en)) = caps.groups[0] else {
                            break;
                        };
                        // A zero-width match at the segment start: not a split;
                        // step the search past one character and retry.
                        if en == seg_start {
                            match s[search..].chars().next() {
                                Some(c) => {
                                    search = search.max(st) + c.len_utf8();
                                    continue;
                                }
                                None => break,
                            }
                        }
                        parts.push(self.new_str(&char_substr(&s, seg_start, st)));
                        // The separator's capture groups are spliced into the
                        // result (`"a1b".split(/(\d)/)` → `["a","1","b"]`).
                        for g in &caps.groups[1..] {
                            match g {
                                Some((gs, ge)) => {
                                    parts.push(self.new_str(&char_substr(&s, *gs, *ge)))
                                }
                                None => parts.push(NanBox::undefined()),
                            }
                        }
                        seg_start = en;
                        search = if en > st { en } else { en + 1 };
                    }
                    if limit.is_none_or(|l| parts.len() < l) {
                        parts.push(self.new_str(&char_substr_from(&s, seg_start)));
                    }
                    if let Some(l) = limit {
                        parts.truncate(l);
                    }
                    return Ok(Some(NanBox::handle(self.realm.new_array(parts).to_raw())));
                }
                // replace / replaceAll. The replacement is either a function
                // (called with `match, g1.., offset, whole`) or a template string
                // (`$1`..`$9` / `$&`).
                _ => {
                    let replacer = arg(1);
                    let is_fn = replacer
                        .as_handle()
                        .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)));
                    let templ = if is_fn {
                        String::new()
                    } else {
                        self.realm.to_display_string(replacer)
                    };
                    let mut out = String::new();
                    let mut at = 0;
                    while let Some(caps) = re.captures_from(&s, at) {
                        let (st, en) = caps.groups[0].unwrap_or((at, at));
                        out.push_str(&char_substr(&s, at, st));
                        if is_fn {
                            let mut call_args = alloc::vec![self.new_str(&char_substr(&s, st, en))];
                            for g in caps.groups.iter().skip(1) {
                                call_args.push(match g {
                                    Some((gs, ge)) => self.new_str(&char_substr(&s, *gs, *ge)),
                                    None => NanBox::undefined(),
                                });
                            }
                            call_args.push(NanBox::number(st as f64));
                            call_args.push(self.new_str(&s));
                            // With named groups, the final argument is a `groups`
                            // object mapping each name to its captured substring.
                            let group_names = re.group_names();
                            if !group_names.is_empty() {
                                let go = self.realm.new_object();
                                for (idx, name) in group_names {
                                    let v = match caps.groups.get(*idx).copied().flatten() {
                                        Some((gs, ge)) => self.new_str(&char_substr(&s, gs, ge)),
                                        None => NanBox::undefined(),
                                    };
                                    self.realm.set_property(go, name, v);
                                }
                                call_args.push(NanBox::handle(go.to_raw()));
                            }
                            let r = self.call(replacer, &call_args)?;
                            let rep = self.realm.to_display_string(r);
                            out.push_str(&rep);
                        } else {
                            out.push_str(&expand_replacement(&templ, &s, &caps, re.group_names()));
                        }
                        if en > st {
                            at = en;
                        } else {
                            // Empty match: keep the char at `st` and step past it.
                            match s[en..].chars().next() {
                                Some(c) => {
                                    out.push(c);
                                    at = en + c.len_utf8();
                                }
                                None => break,
                            }
                        }
                        if !global {
                            break;
                        }
                    }
                    out.push_str(&char_substr_from(&s, at));
                    return Ok(Some(self.new_str(&out)));
                }
            }
        }

        // --- string methods ---
        if let Some(s) = self.realm.string_value(handle) {
            let out = match method {
                "toUpperCase" => Some(self.new_str(&s.to_uppercase())),
                "toLowerCase" => Some(self.new_str(&s.to_lowercase())),
                "trim" => Some(self.new_str(s.trim())),
                "charAt" => {
                    // UTF-16-indexed: the unit at `i` as a one-unit string (a
                    // lone surrogate renders as U+FFFD via lossy decoding). A
                    // negative index is out of range (`NaN`/no-arg → 0).
                    let out = match str_char_index(self.realm.to_number(arg(0))) {
                        Some(i) => s
                            .encode_utf16()
                            .nth(i)
                            .map(|u| String::from_utf16_lossy(&[u]))
                            .unwrap_or_default(),
                        None => String::new(),
                    };
                    Some(self.new_str(&out))
                }
                "includes" => {
                    let needle = self.realm.to_display_string(arg(0));
                    let chars: Vec<char> = s.chars().collect();
                    let pos = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(chars.len())
                    };
                    let sub: String = chars[pos..].iter().collect();
                    Some(NanBox::boolean(sub.contains(&needle)))
                }
                "indexOf" => {
                    let needle = self.realm.to_display_string(arg(0));
                    // An optional `fromIndex` (char offset) starts the search.
                    let chars: Vec<char> = s.chars().collect();
                    let from = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(chars.len())
                    };
                    let byte_off: usize = chars[..from].iter().map(|c| c.len_utf8()).sum();
                    let idx = s[byte_off..]
                        .find(&needle)
                        .map_or(-1.0, |b| s[..byte_off + b].chars().count() as f64);
                    Some(NanBox::number(idx))
                }
                "repeat" => {
                    // A negative or `+Infinity` count is a `RangeError`; a finite
                    // count whose product with the length overflows would panic, so
                    // it is a `RangeError` too (an unrepresentable string length).
                    let nf = self.realm.to_number(arg(0));
                    let n = nf as usize;
                    if nf < 0.0 || nf.is_infinite() || n.checked_mul(s.len()).is_none() {
                        let m = self.new_str("Invalid count value");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    Some(self.new_str(&s.repeat(n)))
                }
                "startsWith" => {
                    let needle = self.realm.to_display_string(arg(0));
                    let chars: Vec<char> = s.chars().collect();
                    let pos = (self.realm.to_number(arg(1)).max(0.0) as usize).min(chars.len());
                    let sub: String = chars[pos..].iter().collect();
                    Some(NanBox::boolean(sub.starts_with(&needle)))
                }
                "endsWith" => {
                    let needle = self.realm.to_display_string(arg(0));
                    let chars: Vec<char> = s.chars().collect();
                    // `endPosition` defaults to the full length.
                    let end = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        chars.len()
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(chars.len())
                    };
                    let sub: String = chars[..end].iter().collect();
                    Some(NanBox::boolean(sub.ends_with(&needle)))
                }
                "slice" => {
                    let chars: Vec<char> = s.chars().collect();
                    let (a, b) = slice_bounds(
                        self.realm.to_number(arg(0)),
                        arg(1),
                        &self.realm,
                        chars.len(),
                    );
                    Some(self.new_str(&chars[a..b].iter().collect::<String>()))
                }
                "split" => {
                    let sep = self.realm.to_display_string(arg(0));
                    let mut parts: Vec<NanBox> = if sep.is_empty() {
                        let chars: Vec<char> = s.chars().collect();
                        chars
                            .iter()
                            .map(|c| self.new_str(&String::from(*c)))
                            .collect()
                    } else {
                        s.split(&sep).map(|p| self.new_str(p)).collect()
                    };
                    // An optional limit caps the number of returned segments.
                    if !matches!(arg(1).unpack(), Unpacked::Undefined) {
                        let limit = self.realm.to_number(arg(1));
                        if limit >= 0.0 {
                            parts.truncate(limit as usize);
                        }
                    }
                    Some(NanBox::handle(self.realm.new_array(parts).to_raw()))
                }
                "replace" => {
                    let from = self.realm.to_display_string(arg(0));
                    let repl = arg(1);
                    let is_fn = repl
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                    if is_fn {
                        match s.find(&from) {
                            Some(pos) => {
                                let m = self.new_str(&from);
                                let off = NanBox::number(s[..pos].chars().count() as f64);
                                let whole = self.new_str(&s);
                                let r = self.call(repl, &[m, off, whole])?;
                                let rs = self.realm.to_display_string(r);
                                let out =
                                    alloc::format!("{}{}{}", &s[..pos], rs, &s[pos + from.len()..]);
                                Some(self.new_str(&out))
                            }
                            None => Some(self.new_str(&s)),
                        }
                    } else {
                        let to = self.realm.to_display_string(repl);
                        match s.find(&from) {
                            Some(pos) => {
                                let before = &s[..pos];
                                let after = &s[pos + from.len()..];
                                let mid = expand_dollar(&to, &from, before, after);
                                Some(self.new_str(&alloc::format!("{before}{mid}{after}")))
                            }
                            None => Some(self.new_str(&s)),
                        }
                    }
                }
                "replaceAll" => {
                    let from = self.realm.to_display_string(arg(0));
                    let repl = arg(1);
                    let is_fn = repl
                        .as_handle()
                        .is_some_and(|r| self.is_callable(Handle::from_raw(r)));
                    if is_fn && !from.is_empty() {
                        let mut out = String::new();
                        let mut last = 0;
                        while let Some(rel) = s[last..].find(&from) {
                            let abs = last + rel;
                            out.push_str(&s[last..abs]);
                            let m = self.new_str(&from);
                            let off = NanBox::number(s[..abs].chars().count() as f64);
                            let whole = self.new_str(&s);
                            let r = self.call(repl, &[m, off, whole])?;
                            out.push_str(&self.realm.to_display_string(r));
                            last = abs + from.len();
                        }
                        out.push_str(&s[last..]);
                        Some(self.new_str(&out))
                    } else if from.is_empty() {
                        let to = self.realm.to_display_string(repl);
                        Some(self.new_str(&s.replace(&from, &to)))
                    } else {
                        let to = self.realm.to_display_string(repl);
                        let mut out = String::new();
                        let mut last = 0;
                        while let Some(rel) = s[last..].find(&from) {
                            let abs = last + rel;
                            out.push_str(&s[last..abs]);
                            let after = &s[abs + from.len()..];
                            out.push_str(&expand_dollar(&to, &from, &s[..abs], after));
                            last = abs + from.len();
                        }
                        out.push_str(&s[last..]);
                        Some(self.new_str(&out))
                    }
                }
                "at" => {
                    let i = self.realm.to_number(arg(0));
                    // UTF-16-indexed with negative-from-end support.
                    let units: Vec<u16> = s.encode_utf16().collect();
                    let idx = if i < 0.0 { units.len() as f64 + i } else { i };
                    Some(match as_index(idx).and_then(|u| units.get(u)) {
                        Some(&u) => self.new_str(&String::from_utf16_lossy(&[u])),
                        None => NanBox::undefined(),
                    })
                }
                "substring" => {
                    let chars: Vec<char> = s.chars().collect();
                    let len = chars.len();
                    let clamp = |n: f64| (n.max(0.0) as usize).min(len);
                    let mut a = clamp(self.realm.to_number(arg(0)));
                    let mut b = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        clamp(self.realm.to_number(arg(1)))
                    };
                    if a > b {
                        core::mem::swap(&mut a, &mut b);
                    }
                    Some(self.new_str(&chars[a..b].iter().collect::<String>()))
                }
                "substr" => {
                    let chars: Vec<char> = s.chars().collect();
                    let len = chars.len() as f64;
                    let start = self.realm.to_number(arg(0));
                    let start = if start < 0.0 {
                        (len + start).max(0.0)
                    } else {
                        start.min(len)
                    } as usize;
                    let count = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        chars.len() - start
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(chars.len() - start)
                    };
                    Some(self.new_str(&chars[start..start + count].iter().collect::<String>()))
                }
                "trimStart" => Some(self.new_str(s.trim_start())),
                "trimEnd" => Some(self.new_str(s.trim_end())),
                // A string's `toString`/`valueOf` is the string itself.
                "toString" | "valueOf" => Some(recv),
                // Strings are stored as well-formed UTF-8 (lone surrogates cannot
                // be represented), so these are always well-formed already.
                "isWellFormed" => Some(NanBox::boolean(true)),
                "toWellFormed" => Some(self.new_str(&s)),
                // `charCodeAt(i)` is the UTF-16 code unit at index `i` (NaN if
                // out of range); a surrogate half reads as that 16-bit value.
                "charCodeAt" => {
                    // A negative or out-of-range index is `NaN` (`NaN`/no-arg → 0).
                    let unit = str_char_index(self.realm.to_number(arg(0)))
                        .and_then(|i| s.encode_utf16().nth(i));
                    Some(unit.map_or(NanBox::number(f64::NAN), |u| NanBox::number(f64::from(u))))
                }
                // `codePointAt(i)` combines a surrogate pair at UTF-16 index `i`.
                "codePointAt" => {
                    let Some(i) = str_char_index(self.realm.to_number(arg(0))) else {
                        return Ok(Some(NanBox::undefined()));
                    };
                    let units: Vec<u16> = s.encode_utf16().collect();
                    Some(match units.get(i).copied() {
                        Some(u) if (0xD800..0xDC00).contains(&u) => {
                            match units.get(i + 1).copied() {
                                Some(low) if (0xDC00..0xE000).contains(&low) => {
                                    let cp = 0x1_0000
                                        + ((u32::from(u) - 0xD800) << 10)
                                        + (u32::from(low) - 0xDC00);
                                    NanBox::number(f64::from(cp))
                                }
                                _ => NanBox::number(f64::from(u)),
                            }
                        }
                        Some(u) => NanBox::number(f64::from(u)),
                        None => NanBox::undefined(),
                    })
                }
                "padStart" => {
                    let target = self.realm.to_number(arg(0)) as usize;
                    let pad = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        String::from(" ")
                    } else {
                        self.realm.to_display_string(arg(1))
                    };
                    Some(self.new_str(&pad_start(&s, target, &pad)))
                }
                "padEnd" => {
                    let target = self.realm.to_number(arg(0)) as usize;
                    let pad = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        String::from(" ")
                    } else {
                        self.realm.to_display_string(arg(1))
                    };
                    Some(self.new_str(&pad_end(&s, target, &pad)))
                }
                "lastIndexOf" => {
                    let needle = self.realm.to_display_string(arg(0));
                    // `fromIndex` (a char index): the match may *start* at or before it;
                    // `undefined`/`NaN` mean +Infinity (search the whole string).
                    let n = self.realm.to_number(arg(1));
                    let from = if n.is_nan() {
                        usize::MAX
                    } else {
                        n.max(0.0).min(usize::MAX as f64) as usize
                    };
                    let chars: Vec<char> = s.chars().collect();
                    let needle_chars: Vec<char> = needle.chars().collect();
                    let (len, nlen) = (chars.len(), needle_chars.len());
                    let idx = if nlen == 0 {
                        from.min(len) as f64
                    } else if nlen <= len {
                        let upper = from.min(len - nlen);
                        (0..=upper)
                            .rev()
                            .find(|&k| chars[k..k + nlen] == needle_chars[..])
                            .map_or(-1.0, |k| k as f64)
                    } else {
                        -1.0
                    };
                    Some(NanBox::number(idx))
                }
                // `concat` appends each argument's string form.
                "concat" => {
                    let mut out = s.clone();
                    for a in args {
                        // ToString each argument, honoring a user `toString`.
                        let p = self.coerce_object(*a, "string")?;
                        out.push_str(&self.realm.to_display_string(p));
                    }
                    Some(self.new_str(&out))
                }
                // `search(str)` — index of the first match (string needle).
                "search" => {
                    let needle = self.realm.to_display_string(arg(0));
                    let idx = s
                        .find(&needle)
                        .map_or(-1.0, |b| s[..b].chars().count() as f64);
                    Some(NanBox::number(idx))
                }
                // `normalize()` — Unicode normalization; a no-op here (the engine
                // stores strings as-is), sufficient for already-normal input.
                "normalize" => {
                    let form = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        String::from("NFC")
                    } else {
                        self.realm.to_display_string(arg(0))
                    };
                    #[cfg(feature = "intl")]
                    {
                        use intl::unicode::normalize;
                        let out: String = match form.as_str() {
                            "NFC" => normalize::nfc(s.chars()).collect(),
                            "NFD" => normalize::nfd(s.chars()).collect(),
                            "NFKC" => normalize::nfkc(s.chars()).collect(),
                            "NFKD" => normalize::nfkd(s.chars()).collect(),
                            _ => {
                                // An unsupported form is a RangeError *object*.
                                let m = self.new_str(&alloc::format!(
                                    "The normalization form should be one of NFC, NFD, NFKC, NFKD. Got {form}."
                                ));
                                return Err(ExecError::Throw(
                                    self.make_error(N_ERROR_BASE + 2, Some(m)),
                                ));
                            }
                        };
                        Some(self.new_str(&out))
                    }
                    #[cfg(not(feature = "intl"))]
                    {
                        let _ = &form;
                        Some(self.new_str(&s))
                    }
                }
                // `localeCompare(other)` — ordering sign (code-point order; no
                // locale tailoring).
                "localeCompare" => {
                    let other = self.realm.to_display_string(arg(0));
                    let cmp = match s.as_str().cmp(other.as_str()) {
                        core::cmp::Ordering::Less => -1.0,
                        core::cmp::Ordering::Equal => 0.0,
                        core::cmp::Ordering::Greater => 1.0,
                    };
                    Some(NanBox::number(cmp))
                }
                _ => None,
            };
            if out.is_some() {
                return Ok(out);
            }
        }

        // --- array methods ---
        if let Some(elems) = self.realm.array_elements(handle).map(<[_]>::to_vec) {
            match method {
                "push" => {
                    let mut len = elems.len();
                    // A frozen array rejects new elements (non-strict: silent).
                    if !self.realm.is_frozen(handle) {
                        for a in args {
                            len = self.realm.array_push(handle, *a).unwrap_or(len);
                        }
                    }
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "pop" => return Ok(Some(self.realm.array_pop(handle))),
                // `splice(start, deleteCount?, ...items)` — mutate in place,
                // return the removed elements as a new array.
                "shift" => {
                    if elems.is_empty() {
                        return Ok(Some(NanBox::undefined()));
                    }
                    let first = elems[0];
                    self.realm.array_set_all(handle, elems[1..].to_vec());
                    return Ok(Some(first));
                }
                "unshift" => {
                    let mut next: Vec<NanBox> = args.to_vec();
                    next.extend_from_slice(&elems);
                    let len = next.len();
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "splice" => {
                    let len = elems.len();
                    let start = {
                        let s = self.realm.to_number(arg(0));
                        if s < 0.0 {
                            (len as f64 + s).max(0.0) as usize
                        } else {
                            (s as usize).min(len)
                        }
                    };
                    let delete = if args.len() < 2 {
                        len - start
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(len - start)
                    };
                    let removed: Vec<NanBox> = elems[start..start + delete].to_vec();
                    let mut next: Vec<NanBox> = elems[..start].to_vec();
                    next.extend_from_slice(&args[2.min(args.len())..]);
                    next.extend_from_slice(&elems[start + delete..]);
                    self.realm.array_set_all(handle, next);
                    return Ok(Some(NanBox::handle(self.realm.new_array(removed).to_raw())));
                }
                // `arr.toString()` joins with a comma (like `join()`).
                "join" | "toString" => {
                    let sep =
                        if method == "toString" || matches!(arg(0).unpack(), Unpacked::Undefined) {
                            String::from(",")
                        } else {
                            self.realm.to_display_string(arg(0))
                        };
                    // `null`/`undefined` render empty; an object element is run
                    // through ToString (so a custom `toString` is honored). The
                    // receiver array seeds the cycle set, so a self-reference (or a
                    // mutual cycle back to it) renders empty rather than recursing.
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for e in &elems {
                        let s = match e.unpack() {
                            Unpacked::Null | Unpacked::Undefined => String::new(),
                            // A direct self-reference back to the receiver renders
                            // empty (per `Array.prototype.join`), without recursing.
                            Unpacked::Handle(raw) if raw == handle.to_raw() => String::new(),
                            _ => {
                                let p = self.coerce_object(*e, "string")?;
                                self.realm.to_display_string(p)
                            }
                        };
                        parts.push(s);
                    }
                    return Ok(Some(self.new_str(&parts.join(&sep))));
                }
                "includes" => {
                    let target = arg(0);
                    let from = array_from_index(&self.realm, arg(1), elems.len());
                    // SameValueZero: like `===` but `NaN` matches `NaN`.
                    let t_nan = target.as_number().is_some_and(f64::is_nan);
                    let found = elems[from..].iter().any(|e| {
                        self.realm.strict_equals(*e, target)
                            || (t_nan && e.as_number().is_some_and(f64::is_nan))
                    });
                    return Ok(Some(NanBox::boolean(found)));
                }
                // `toSorted`/`toReversed`/`with` — non-mutating array methods.
                "toReversed" => {
                    let mut out = elems.clone();
                    out.reverse();
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "with" => {
                    let len = elems.len() as i64;
                    let i = self.realm.to_number(arg(0)) as i64;
                    let idx = if i < 0 { len + i } else { i };
                    // An out-of-range index is a RangeError.
                    if idx < 0 || idx >= len {
                        let m = self.new_str("Invalid index");
                        return Err(ExecError::Throw(self.make_error(N_ERROR_BASE + 2, Some(m))));
                    }
                    let mut out = elems.clone();
                    out[idx as usize] = arg(1);
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "toSorted" => {
                    let numeric = self.realm.get_property(handle, TYPED_ARRAY_KIND).is_some();
                    let sorted = self.sort_array(elems.clone(), arg(0), numeric)?;
                    return Ok(Some(self.typed_like(handle, sorted)));
                }
                "indexOf" => {
                    let target = arg(0);
                    let from = array_from_index(&self.realm, arg(1), elems.len());
                    let idx = elems[from..]
                        .iter()
                        .position(|e| self.realm.strict_equals(*e, target))
                        .map_or(-1.0, |i| (i + from) as f64);
                    return Ok(Some(NanBox::number(idx)));
                }
                "map" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = NanBox::handle(handle.to_raw());
                    let mut out = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        out.push(self.call_with_this(f, this_arg, &cb_args)?);
                    }
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "filter" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = NanBox::handle(handle.to_raw());
                    let mut out = Vec::new();
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        let r = self.call_with_this(f, this_arg, &cb_args)?;
                        if self.realm.truthy(r) {
                            out.push(*e);
                        }
                    }
                    return Ok(Some(self.typed_like(handle, out)));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let arr = NanBox::handle(handle.to_raw());
                    for (i, e) in elems.iter().enumerate() {
                        let cb_args = [*e, NanBox::number(i as f64), arr];
                        self.call_with_this(f, this_arg, &cb_args)?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "reduce" => {
                    let f = arg(0);
                    let mut acc;
                    let mut start = 0;
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else if elems.is_empty() {
                        let m = self.new_str("Reduce of empty array with no initial value");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    } else {
                        acc = elems[0];
                        start = 1;
                    }
                    let arr = NanBox::handle(handle.to_raw());
                    for (i, e) in elems.iter().enumerate().skip(start) {
                        acc = self.call(f, &[acc, *e, NanBox::number(i as f64), arr])?;
                    }
                    return Ok(Some(acc));
                }
                // `reduceRight` — like `reduce` but right-to-left.
                "reduceRight" => {
                    let f = arg(0);
                    let mut acc;
                    let mut idx = elems.len();
                    if args.len() >= 2 {
                        acc = arg(1);
                    } else if elems.is_empty() {
                        let m = self.new_str("Reduce of empty array with no initial value");
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    } else {
                        idx -= 1;
                        acc = elems[idx];
                    }
                    let arr = NanBox::handle(handle.to_raw());
                    while idx > 0 {
                        idx -= 1;
                        acc = self.call(f, &[acc, elems[idx], NanBox::number(idx as f64), arr])?;
                    }
                    return Ok(Some(acc));
                }
                "slice" => {
                    let (a, b) = slice_bounds(
                        self.realm.to_number(arg(0)),
                        arg(1),
                        &self.realm,
                        elems.len(),
                    );
                    let sub = elems[a..b].to_vec();
                    return Ok(Some(self.typed_like(handle, sub)));
                }
                // Iterators: `keys()` over indices, `values()` over elements,
                // `entries()` over `[index, element]` pairs (eager generators).
                "keys" => {
                    let ks: Vec<NanBox> =
                        (0..elems.len()).map(|i| NanBox::number(i as f64)).collect();
                    return Ok(Some(self.make_generator(ks)));
                }
                "values" => {
                    return Ok(Some(self.make_generator(elems.clone())));
                }
                "entries" => {
                    let mut pairs = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let pair = self
                            .realm
                            .new_array(alloc::vec![NanBox::number(i as f64), *e]);
                        pairs.push(NanBox::handle(pair.to_raw()));
                    }
                    return Ok(Some(self.make_generator(pairs)));
                }
                "concat" => {
                    let mut out = elems.clone();
                    // An argument is spread iff it is concat-spreadable: its
                    // `[Symbol.isConcatSpreadable]` (if defined) decides, else it is
                    // spread exactly when it is an array.
                    let sym = self.well_known_symbol("isConcatSpreadable");
                    let spread_key = self.member_key(sym);
                    for a in args {
                        let ah = a.as_handle().map(Handle::from_raw);
                        let spread = match ah {
                            Some(h) => match self.realm.get_property(h, &spread_key) {
                                Some(v) if !matches!(v.unpack(), Unpacked::Undefined) => {
                                    self.realm.truthy(v)
                                }
                                _ => self.realm.is_array(h),
                            },
                            None => false,
                        };
                        match (spread, ah) {
                            (true, Some(h)) => {
                                if let Some(other) = self.realm.array_elements(h).map(<[_]>::to_vec)
                                {
                                    out.extend(other);
                                } else {
                                    // A spreadable array-like: read length + indices.
                                    let len = self
                                        .realm
                                        .get_property(h, "length")
                                        .map(|v| self.realm.to_number(v))
                                        .unwrap_or(0.0)
                                        .max(0.0)
                                        as usize;
                                    for i in 0..len {
                                        let k = alloc::format!("{i}");
                                        out.push(
                                            self.realm
                                                .get_property(h, &k)
                                                .unwrap_or(NanBox::undefined()),
                                        );
                                    }
                                }
                            }
                            _ => out.push(*a),
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                "reverse" => {
                    // Reverses in place and returns the same array.
                    let mut out = elems.clone();
                    out.reverse();
                    self.realm.array_set_all(handle, out);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `fill(value, start?, end?)` — mutate in place, return the array.
                // `start`/`end` default to `0`/`len`; negatives count from the end.
                // `TypedArray.prototype.set(source, offset?)`: copy a source array's
                // elements into this typed array starting at `offset`, coercing each
                // to the element type. (Only typed arrays have `set`.)
                "set" if self.realm.get_property(handle, TYPED_ARRAY_KIND).is_some() => {
                    let offset = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        self.realm.to_number(arg(1)).max(0.0) as usize
                    };
                    if let Some(src) = arg(0).as_handle().map(Handle::from_raw)
                        && let Some(src_elems) = self.realm.array_elements(src).map(<[_]>::to_vec)
                    {
                        // Out-of-range writes are a RangeError, per spec.
                        if offset + src_elems.len() > elems.len() {
                            let m = self.new_str("offset is out of bounds");
                            return Err(ExecError::Throw(
                                self.make_error(N_ERROR_BASE + 2, Some(m)),
                            ));
                        }
                        for (j, v) in src_elems.into_iter().enumerate() {
                            self.set_element_coerced(handle, offset + j, v);
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                // `subarray(begin, end)` — a new typed array of the same kind over
                // the `[begin, end)` slice. (A copy here, not a live buffer view,
                // since typed arrays are standalone — see the ArrayBuffer-backing
                // limitation.)
                "subarray" if self.realm.get_property(handle, TYPED_ARRAY_KIND).is_some() => {
                    let len = elems.len() as i64;
                    let norm = |v: NanBox, default: i64, this: &mut Self| -> usize {
                        if matches!(v.unpack(), Unpacked::Undefined) {
                            return default as usize;
                        }
                        let n = this.realm.to_number(v) as i64;
                        usize::try_from(if n < 0 { (len + n).max(0) } else { n.min(len) })
                            .unwrap_or(0)
                    };
                    let start = norm(arg(0), 0, self);
                    let end = norm(arg(1), len, self);
                    let sub: Vec<NanBox> = elems.get(start..end.max(start)).unwrap_or(&[]).to_vec();
                    let kind = self
                        .realm
                        .get_property(handle, TYPED_ARRAY_KIND)
                        .unwrap_or(NanBox::number(0.0));
                    let arr = self.realm.new_array(sub);
                    self.realm.set_property(arr, TYPED_ARRAY_KIND, kind);
                    return Ok(Some(NanBox::handle(arr.to_raw())));
                }
                "fill" => {
                    let len = elems.len();
                    let value = arg(0);
                    let start = if matches!(arg(1).unpack(), Unpacked::Undefined) {
                        0
                    } else {
                        let n = self.realm.to_number(arg(1));
                        if n < 0.0 {
                            (len as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(len)
                        }
                    };
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        let n = self.realm.to_number(arg(2));
                        if n < 0.0 {
                            (len as f64 + n).max(0.0) as usize
                        } else {
                            (n as usize).min(len)
                        }
                    };
                    for i in start..end {
                        self.realm.set_element(handle, i, value);
                    }
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `flat(depth = 1)` — recursively flatten nested arrays.
                "flat" => {
                    let depth = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        1
                    } else {
                        self.realm.to_number(arg(0)) as i32
                    };
                    let out = self.flatten(&elems, depth);
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                // `copyWithin(target, start, end?)` — copy a slice within the
                // array in place; negatives count from the end.
                "copyWithin" => {
                    let len = elems.len() as i64;
                    let norm = |v: f64| -> i64 {
                        let i = v as i64;
                        if i < 0 { (len + i).max(0) } else { i.min(len) }
                    };
                    let target = norm(self.realm.to_number(arg(0)));
                    let start = norm(self.realm.to_number(arg(1)));
                    let end = if matches!(arg(2).unpack(), Unpacked::Undefined) {
                        len
                    } else {
                        norm(self.realm.to_number(arg(2)))
                    };
                    let slice: Vec<NanBox> =
                        elems[start as usize..end.max(start) as usize].to_vec();
                    for (k, v) in slice.into_iter().enumerate() {
                        let dst = target as usize + k;
                        if dst >= elems.len() {
                            break;
                        }
                        self.realm.set_element(handle, dst, v);
                    }
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `map` then flatten one level.
                "flatMap" => {
                    let f = arg(0);
                    let mut out = Vec::new();
                    for (i, e) in elems.iter().enumerate() {
                        let r = self.call(f, &[*e, NanBox::number(i as f64)])?;
                        match r
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            Some(inner) => out.extend(inner),
                            None => out.push(r),
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                // `at` with negative-from-end indexing.
                "at" => {
                    let i = self.realm.to_number(arg(0));
                    let idx = if i < 0.0 { elems.len() as f64 + i } else { i };
                    return Ok(Some(
                        as_index(idx)
                            .and_then(|u| elems.get(u))
                            .copied()
                            .unwrap_or(NanBox::undefined()),
                    ));
                }
                "lastIndexOf" => {
                    let target = arg(0);
                    let len = elems.len();
                    if len == 0 {
                        return Ok(Some(NanBox::number(-1.0)));
                    }
                    // Optional `fromIndex` (default last; negative counts back).
                    let from = if args.len() >= 2 {
                        let n = self.realm.to_number(arg(1));
                        let n = if n < 0.0 { len as f64 + n } else { n };
                        if n < 0.0 {
                            return Ok(Some(NanBox::number(-1.0)));
                        }
                        (n as usize).min(len - 1)
                    } else {
                        len - 1
                    };
                    let found = elems[..=from]
                        .iter()
                        .rposition(|e| self.realm.strict_equals(*e, target));
                    return Ok(Some(NanBox::number(found.map_or(-1.0, |i| i as f64))));
                }
                "find" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(*e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findIndex" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                // `findLast`/`findLastIndex` — scan right-to-left.
                "findLast" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate().rev() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(*e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findLastIndex" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate().rev() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                "some" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(NanBox::boolean(true)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(false)));
                }
                "every" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if !self.call_truthy_this(
                            f,
                            arg(1),
                            &[
                                *e,
                                NanBox::number(i as f64),
                                NanBox::handle(handle.to_raw()),
                            ],
                        )? {
                            return Ok(Some(NanBox::boolean(false)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(true)));
                }
                "sort" => {
                    // Sorts in place and returns the same array. A typed array sorts
                    // numerically by default (a plain array lexicographically).
                    let numeric = self.realm.get_property(handle, TYPED_ARRAY_KIND).is_some();
                    let sorted = self.sort_array(elems, arg(0), numeric)?;
                    self.realm.array_set_all(handle, sorted);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                // `toSpliced(start, deleteCount, ...items)` — a spliced copy
                // (the ES2023 immutable counterpart of `splice`).
                "toSpliced" => {
                    let len = elems.len() as i64;
                    let start = {
                        let s = self.realm.to_number(arg(0)) as i64;
                        if s < 0 { (len + s).max(0) } else { s.min(len) }
                    } as usize;
                    let del = if args.len() < 2 {
                        elems.len() - start
                    } else {
                        (self.realm.to_number(arg(1)).max(0.0) as usize).min(elems.len() - start)
                    };
                    let mut out: Vec<NanBox> = elems[..start].to_vec();
                    out.extend_from_slice(&args[2.min(args.len())..]);
                    out.extend_from_slice(&elems[start + del..]);
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                _ => {}
            }
        }

        // --- Map / Set methods ---
        if let Some(size) = self.realm.collection_size(handle) {
            match method {
                "set" => {
                    self.guard_weak_key(handle, arg(0))?;
                    self.realm.collection_set(handle, arg(0), arg(1));
                    return Ok(Some(recv)); // Map.set returns the map (chainable)
                }
                "add" => {
                    self.guard_weak_key(handle, arg(0))?;
                    self.realm.collection_set(handle, arg(0), arg(0));
                    return Ok(Some(recv)); // Set.add returns the set
                }
                "get" => {
                    return Ok(Some(
                        self.realm
                            .collection_get(handle, arg(0))
                            .unwrap_or(NanBox::undefined()),
                    ));
                }
                "has" => {
                    return Ok(Some(NanBox::boolean(
                        self.realm.collection_has(handle, arg(0)),
                    )));
                }
                "delete" => {
                    return Ok(Some(NanBox::boolean(
                        self.realm.collection_delete(handle, arg(0)),
                    )));
                }
                "clear" => {
                    self.realm.collection_clear(handle);
                    return Ok(Some(NanBox::undefined()));
                }
                "forEach" => {
                    let f = arg(0);
                    let this_arg = arg(1);
                    let coll = NanBox::handle(handle.to_raw());
                    for (k, v) in self.realm.collection_entries(handle).unwrap_or_default() {
                        // The callback gets `(value, key, collection)` with `thisArg`.
                        self.call_with_this(f, this_arg, &[v, k, coll])?;
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "keys" => {
                    let keys: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect();
                    return Ok(Some(NanBox::handle(self.realm.new_array(keys).to_raw())));
                }
                "values" => {
                    // A Set yields its elements; a Map yields its values.
                    let is_set = self.realm.collection_is_set(handle) == Some(true);
                    let vals: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, v)| if is_set { k } else { v })
                        .collect();
                    return Ok(Some(NanBox::handle(self.realm.new_array(vals).to_raw())));
                }
                "entries" => {
                    let pairs = self.realm.collection_entries(handle).unwrap_or_default();
                    let arr: Vec<NanBox> = pairs
                        .into_iter()
                        .map(|(k, v)| {
                            NanBox::handle(self.realm.new_array(alloc::vec![k, v]).to_raw())
                        })
                        .collect();
                    return Ok(Some(NanBox::handle(self.realm.new_array(arr).to_raw())));
                }
                // ES2025 Set composition. The argument is treated as a set-like
                // (any iterable supplies its elements).
                "union"
                | "intersection"
                | "difference"
                | "symmetricDifference"
                | "isSubsetOf"
                | "isSupersetOf"
                | "isDisjointFrom"
                    if self.realm.collection_is_set(handle) == Some(true) =>
                {
                    let mine: Vec<NanBox> = self
                        .realm
                        .collection_entries(handle)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect();
                    let other = self.iterate_values(arg(0))?;
                    let in_other = |this: &Self, v: NanBox| {
                        other.iter().any(|o| this.realm.same_value_zero(*o, v))
                    };
                    let in_mine = |this: &Self, v: NanBox| {
                        mine.iter().any(|m| this.realm.same_value_zero(*m, v))
                    };
                    // Predicate methods return a boolean.
                    match method {
                        "isSubsetOf" => {
                            return Ok(Some(NanBox::boolean(
                                mine.iter().all(|m| in_other(self, *m)),
                            )));
                        }
                        "isSupersetOf" => {
                            return Ok(Some(NanBox::boolean(
                                other.iter().all(|o| in_mine(self, *o)),
                            )));
                        }
                        "isDisjointFrom" => {
                            return Ok(Some(NanBox::boolean(
                                !mine.iter().any(|m| in_other(self, *m)),
                            )));
                        }
                        _ => {}
                    }
                    // The rest build a new Set.
                    let result = self.realm.new_collection(true);
                    match method {
                        "union" => {
                            for e in mine.iter().chain(other.iter()) {
                                self.realm.collection_set(result, *e, *e);
                            }
                        }
                        "intersection" => {
                            for e in &mine {
                                if in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                        "difference" => {
                            for e in &mine {
                                if !in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                        // symmetricDifference: in exactly one of the two.
                        _ => {
                            for e in &mine {
                                if !in_other(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                            for e in &other {
                                if !in_mine(self, *e) {
                                    self.realm.collection_set(result, *e, *e);
                                }
                            }
                        }
                    }
                    return Ok(Some(NanBox::handle(result.to_raw())));
                }
                _ => {
                    let _ = size;
                }
            }
        }

        // Default `Object.prototype` methods for an object receiver that did not
        // match a more specific built-in and has no own/inherited method of its
        // own (e.g. a plain object's `toString`/`valueOf`).
        if let Some(h) = recv.as_handle().map(Handle::from_raw)
            && matches!(method, "toString" | "valueOf" | "toLocaleString")
        {
            // A user-defined (own or inherited) method takes precedence.
            let own = self.read_member(h, method)?;
            if own
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                return Ok(None);
            }
            // String values/objects: `toString`/`valueOf` yield the string.
            if let Some(s) = self.realm.string_value(h) {
                return Ok(Some(self.new_str(&s)));
            }
            if method == "valueOf" {
                return Ok(Some(recv));
            }
            let tag = self.object_string_tag(h)?;
            return Ok(Some(self.new_str(&alloc::format!("[object {tag}]"))));
        }
        Ok(None)
    }

    /// The tag used by `Object.prototype.toString` (`"[object <tag>]"`): a
    /// `Symbol.toStringTag` string property if present, else the built-in tag.
    /// Invokes a WASM export wrapper: `data` carries the module bytes and the
    /// export name; decode, instantiate, marshal `args` across the JS↔WASM
    /// boundary, and return the (first) result as a JS value.
    fn call_wasm_export(
        &mut self,
        data: crate::heap::Handle,
        args: &[NanBox],
    ) -> Result<NanBox, ExecError> {
        let bytes = self
            .realm
            .get_property(data, WASM_BYTES)
            .and_then(|v| self.wasm_bytes(v))
            .ok_or_else(|| self.wasm_compile_error("missing module bytes"))?;
        let name = self
            .realm
            .get_property(data, WASM_EXPORT)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|h| self.realm.string_value(h))
            .ok_or_else(|| self.wasm_compile_error("missing export name"))?;
        let imports_obj = self
            .realm
            .get_property(data, WASM_IMPORTS)
            .unwrap_or(NanBox::undefined());
        let module = crate::wasm_rt::Module::decode(&bytes)
            .map_err(|_| self.wasm_compile_error("invalid module"))?;

        // Resolve each function import to a JS function: importObject[mod][field].
        let import_names: Vec<(String, String)> = module
            .import_names()
            .iter()
            .map(|(m, f)| ((*m).into(), (*f).into()))
            .collect();
        let mut import_fns: Vec<NanBox> = Vec::with_capacity(import_names.len());
        for (m, f) in &import_names {
            let ns = imports_obj
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, m))
                .unwrap_or(NanBox::undefined());
            let func = ns
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, f))
                .unwrap_or(NanBox::undefined());
            import_fns.push(func);
        }
        // The result type of each import (to marshal the JS return back to a Val).
        let import_results: Vec<Vec<crate::wasm_rt::ValType>> = (0..import_names.len())
            .map(|i| {
                module
                    .func_type(i as u32)
                    .map(|t| t.results.clone())
                    .unwrap_or_default()
            })
            .collect();

        // Marshal the export's arguments per its parameter types.
        let export_idx = module
            .export(&name)
            .ok_or_else(|| self.wasm_compile_error("no such export"))?;
        let params = module
            .func_type(export_idx)
            .map(|t| t.params.clone())
            .unwrap_or_default();
        if args.len() != params.len() {
            return Err(self.wasm_compile_error("argument count mismatch"));
        }
        let val_args: Vec<crate::wasm_rt::Val> = params
            .iter()
            .zip(args)
            .map(|(t, v)| {
                crate::wasm_rt::Val::from_nanbox(*v, *t)
                    .ok_or_else(|| self.wasm_compile_error("argument not coercible to wasm value"))
            })
            .collect::<Result<_, _>>()?;

        // Resolve imported globals: importObject[mod][field] is a
        // `WebAssembly.Global` (its `.value`) or a plain Number/BigInt, coerced to
        // the imported global's declared type.
        let global_imports: Vec<(String, String, crate::wasm_rt::ValType)> = module
            .global_import_names()
            .iter()
            .map(|(m, f, t)| ((*m).into(), (*f).into(), *t))
            .collect();
        let mut import_global_vals = Vec::with_capacity(global_imports.len());
        for (m, f, ty) in &global_imports {
            let ns = imports_obj
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, m))
                .unwrap_or(NanBox::undefined());
            let entry = ns
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, f))
                .unwrap_or(NanBox::undefined());
            // A `WebAssembly.Global` carries its value in a hidden slot; otherwise
            // the entry is itself the value (a Number/BigInt).
            let raw_val = entry
                .as_handle()
                .map(Handle::from_raw)
                .and_then(|h| self.realm.get_property(h, WASM_GLOBAL_VALUE))
                .unwrap_or(entry);
            let v = crate::wasm_rt::Val::from_nanbox(raw_val, *ty)
                .ok_or_else(|| self.wasm_compile_error("imported global not coercible"))?;
            import_global_vals.push(v);
        }

        let mut inst = if !global_imports.is_empty() {
            crate::wasm_rt::Instance::with_host_imports_and_globals(&module, import_global_vals)
        } else if import_names.is_empty() {
            crate::wasm_rt::Instance::new(&module)
        } else {
            crate::wasm_rt::Instance::with_host_imports(&module)
        }
        .map_err(|e| self.wasm_compile_error(e.0))?;

        // Resume this instance's persistent memory/globals from its prior call, so
        // mutable state (a counter global, written linear memory, …) carries over.
        let inst_id = self
            .realm
            .get_property(data, WASM_INSTANCE_ID)
            .and_then(|v| v.as_number())
            .map(|n| n as u32);
        if let Some(id) = inst_id
            && let Some(state) = self.wasm_states.get(&id)
        {
            inst.import_state(state);
        }

        // The import dispatcher: marshal Vals → JS, call the JS import, marshal the
        // result back. Borrows `self` (the engine) directly — sound because the
        // instance state (`inst`) is a separate object.
        let mut thrown: Option<ExecError> = None;
        let results = {
            let me: &mut Self = self;
            let mut host = |i: usize, wargs: &[crate::wasm_rt::Val]| {
                let nbargs: Vec<NanBox> = wargs.iter().map(|v| v.to_nanbox()).collect();
                match me.call(import_fns[i], &nbargs) {
                    Ok(r) => import_results[i]
                        .iter()
                        .map(|t| {
                            crate::wasm_rt::Val::from_nanbox(r, *t)
                                .ok_or(crate::wasm_rt::WasmRtError("import result not coercible"))
                        })
                        .collect(),
                    Err(e) => {
                        thrown = Some(e);
                        Err(crate::wasm_rt::WasmRtError("host import threw"))
                    }
                }
            };
            inst.call_export_with_host(&name, &val_args, &mut host)
        };
        if let Some(e) = thrown {
            return Err(e); // propagate a JS exception thrown by an import
        }
        let results = results.map_err(|e| self.wasm_compile_error(e.0))?;
        // Persist the post-call memory/globals so the next call sees them.
        if let Some(id) = inst_id {
            self.wasm_states.insert(id, inst.export_state());
        }
        Ok(results
            .first()
            .map_or(NanBox::undefined(), |v| v.to_nanbox()))
    }

    /// A thrown `WebAssembly`-style `TypeError` for compile/instantiate failures.
    fn wasm_compile_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_WASM_COMPILE_ERROR, Some(m)))
    }

    fn wasm_type_error(&mut self, msg: &str) -> ExecError {
        let m = self.new_str(msg);
        ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m)))
    }

    /// Builds an `ArrayBuffer` object of `len` zeroed bytes.
    fn make_array_buffer(&mut self, len: usize) -> Handle {
        let obj = self.realm.new_object();
        let bytes = self.realm.new_array(alloc::vec![NanBox::number(0.0); len]);
        self.realm
            .set_hidden_property(obj, ARRAY_BUFFER_BYTES, NanBox::handle(bytes.to_raw()));
        obj
    }

    /// Builds a `WebAssembly.Global` object wrapping the already-coerced `value`
    /// of type `ty`, with a `.value` accessor (settable only when `mutable`).
    fn make_wasm_global(&mut self, value: NanBox, ty: &str, mutable: bool) -> NanBox {
        let g = self.realm.new_object();
        self.realm.set_hidden_property(g, WASM_GLOBAL_VALUE, value);
        let ty_v = self.new_str(ty);
        self.realm.set_hidden_property(g, WASM_GLOBAL_TYPE, ty_v);
        self.realm
            .set_hidden_property(g, WASM_GLOBAL_MUTABLE, NanBox::boolean(mutable));
        let getter = self.realm.new_bound_native(N_WASM_GLOBAL_GET, g);
        let setter = self.realm.new_bound_native(N_WASM_GLOBAL_SET, g);
        self.realm.define_accessor(
            g,
            "value",
            NanBox::handle(getter.to_raw()),
            NanBox::handle(setter.to_raw()),
        );
        self.realm.mark_hidden(g, "value"); // a prototype accessor in spec: non-enumerable
        // `valueOf()` returns the value (so a Global coerces numerically), reusing
        // the same getter native; it is non-enumerable.
        let value_of = self.realm.new_bound_native(N_WASM_GLOBAL_GET, g);
        self.realm
            .set_property(g, "valueOf", NanBox::handle(value_of.to_raw()));
        self.realm.mark_hidden(g, "valueOf");
        NanBox::handle(g.to_raw())
    }

    /// Coerces a JS value to a WASM `Global`'s value type: `i32` via ToInt32,
    /// `f32`/`f64` via ToNumber, `i64` to a `BigInt`.
    fn wasm_coerce_global(&mut self, ty: &str, v: NanBox) -> NanBox {
        match ty {
            "i32" => {
                // ToInt32 without the std-only `trunc`: truncate toward zero (the
                // `as i64` cast) then take the low 32 bits.
                let n = self.realm.to_number(v);
                let i = if n.is_finite() { n as i64 as i32 } else { 0 };
                NanBox::number(f64::from(i))
            }
            "i64" => {
                if let Some(h) = v.as_handle().map(Handle::from_raw)
                    && self.realm.bigint_at(h).is_some()
                {
                    v
                } else {
                    let n = crate::bignum::BigInt::from_i128(self.realm.to_number(v) as i128);
                    NanBox::handle(self.realm.new_bigint(n).to_raw())
                }
            }
            // f32 (and f64) keep the JS number; f32 precision is not narrowed here.
            _ => NanBox::number(self.realm.to_number(v)),
        }
    }

    /// Decodes/validates `bytes_arr` and builds a `WebAssembly.Module` object that
    /// retains the source bytes (for later instantiation), or throws a `TypeError`.
    fn make_wasm_module(&mut self, bytes_arr: NanBox) -> Result<NanBox, ExecError> {
        let bytes = self
            .wasm_bytes(bytes_arr)
            .ok_or_else(|| self.wasm_compile_error("invalid module source"))?;
        crate::wasm_rt::Module::decode(&bytes)
            .map_err(|_| self.wasm_compile_error("invalid module"))?;
        let module = self.realm.new_object();
        self.realm.set_property(module, WASM_BYTES, bytes_arr);
        self.realm.mark_hidden(module, WASM_BYTES);
        self.realm
            .set_hidden_property(module, WASM_IS_MODULE, NanBox::boolean(true));
        Ok(NanBox::handle(module.to_raw()))
    }

    /// Builds an instance object (`{ exports: {…} }`) from module `bytes_arr` (the
    /// original `BufferSource`, kept for per-call re-decode) and an optional import
    /// object. Each export is a callable wrapper bound to `N_WASM_CALL`. Shared by
    /// `WebAssembly.instantiate` and `new WebAssembly.Instance`.
    fn build_wasm_instance(
        &mut self,
        bytes_arr: NanBox,
        imports_obj: NanBox,
    ) -> Result<NanBox, ExecError> {
        let bytes = self
            .wasm_bytes(bytes_arr)
            .ok_or_else(|| self.wasm_compile_error("invalid module source"))?;
        let module = crate::wasm_rt::Module::decode(&bytes)
            .map_err(|_| self.wasm_compile_error("invalid module"))?;
        let names: Vec<String> = module.export_names().iter().map(|s| (*s).into()).collect();
        // A fresh instance id ties every export wrapper to one persistent state
        // entry, so the instance's memory/globals survive across export calls.
        let inst_id = self.wasm_next_id;
        self.wasm_next_id = self.wasm_next_id.wrapping_add(1);
        let exports = self.realm.new_object();
        for name in names {
            let data = self.realm.new_object();
            self.realm.set_property(data, WASM_BYTES, bytes_arr);
            self.realm.set_property(data, WASM_IMPORTS, imports_obj);
            let name_v = self.new_str(&name);
            self.realm.set_property(data, WASM_EXPORT, name_v);
            self.realm
                .set_property(data, WASM_INSTANCE_ID, NanBox::number(f64::from(inst_id)));
            let f = self.realm.new_bound_native(N_WASM_CALL, data);
            self.realm
                .set_property(exports, &name, NanBox::handle(f.to_raw()));
        }
        // Exported globals become `WebAssembly.Global` objects holding the
        // instance's value, read at instantiation. Supported for modules with no
        // imports (so a fresh `Instance::new` reflects the real initial values).
        let global_exports: Vec<(String, u32)> = module
            .global_exports()
            .iter()
            .map(|(n, i)| ((*n).into(), *i))
            .collect();
        if !global_exports.is_empty()
            && module.import_names().is_empty()
            && module.global_import_names().is_empty()
            && let Ok(inst) = crate::wasm_rt::Instance::new(&module)
        {
            for (gname, gidx) in &global_exports {
                if let Some(val) = inst.global_value(*gidx) {
                    let (value, ty) = self.wasm_global_export_value(val);
                    let mutable = module.global_is_mutable(*gidx);
                    let g = self.make_wasm_global(value, ty, mutable);
                    self.realm.set_property(exports, gname, g);
                }
            }
        }
        let instance = self.realm.new_object();
        self.realm
            .set_property(instance, "exports", NanBox::handle(exports.to_raw()));
        // A (non-enumerable) marker so `instance instanceof WebAssembly.Instance`
        // matches, like the other `WebAssembly.*` boundary objects.
        self.realm.set_hidden_property(
            instance,
            WASM_INSTANCE_ID,
            NanBox::number(f64::from(inst_id)),
        );
        Ok(NanBox::handle(instance.to_raw()))
    }

    /// Converts a WASM global's `Val` to a `(JS value, type string)` pair for
    /// building a `WebAssembly.Global` (an `i64` becomes a `BigInt`).
    fn wasm_global_export_value(&mut self, val: crate::wasm_rt::Val) -> (NanBox, &'static str) {
        match val {
            crate::wasm_rt::Val::I32(_) => (val.to_nanbox(), "i32"),
            crate::wasm_rt::Val::F32(_) => (val.to_nanbox(), "f32"),
            crate::wasm_rt::Val::F64(_) => (val.to_nanbox(), "f64"),
            crate::wasm_rt::Val::I64(n) => {
                let big = crate::bignum::BigInt::from_i128(i128::from(n));
                (NanBox::handle(self.realm.new_bigint(big).to_raw()), "i64")
            }
        }
    }

    /// Extracts a byte vector from a JS `BufferSource`-ish value for the WASM
    /// builtins: an `ArrayBuffer` (its `\0abytes` store) or a plain array of byte
    /// numbers. Returns `None` if `v` isn't byte-like.
    fn wasm_bytes(&self, v: NanBox) -> Option<Vec<u8>> {
        let h = Handle::from_raw(v.as_handle()?);
        // An ArrayBuffer keeps its bytes in a hidden array; fall back to treating
        // the value itself as a byte array.
        let arr = self
            .realm
            .get_property(h, ARRAY_BUFFER_BYTES)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
            .unwrap_or(h);
        let elems = self.realm.array_elements(arr)?;
        Some(
            elems
                .iter()
                .map(|e| e.as_number().unwrap_or(0.0) as u8)
                .collect(),
        )
    }

    fn object_string_tag(&mut self, h: crate::heap::Handle) -> Result<String, ExecError> {
        let tag_sym = self.well_known_symbol("toStringTag");
        let tag_key = self.member_key(tag_sym);
        // Read through the prototype chain so a `Symbol.toStringTag` accessor
        // (e.g. `get [Symbol.toStringTag]() {…}` on a class) is invoked, not just
        // an own data property.
        let v = self.read_member(h, &tag_key)?;
        if let Some(sh) = v.as_handle().map(Handle::from_raw)
            && let Some(s) = self.realm.string_value(sh)
        {
            return Ok(s);
        }
        // A boxed primitive wrapper (`new Number(…)`/`String`/`Boolean`, or the
        // object form `ToObject` produces) reports its primitive's class.
        if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
            return Ok(String::from(match prim.unpack() {
                Unpacked::Number(_) => "Number",
                Unpacked::Bool(_) => "Boolean",
                _ => "String",
            }));
        }
        Ok(if self.realm.is_array(h) {
            String::from("Array")
        } else if self.is_callable(h) || self.realm.class_at(h).is_some() {
            String::from("Function")
        } else if self.realm.string_value(h).is_some() {
            // A primitive string boxes to a `String` exotic object.
            String::from("String")
        } else if let Some(is_set) = self.realm.collection_is_set(h) {
            String::from(if is_set { "Set" } else { "Map" })
        } else if self.realm.date_at(h).is_some() {
            String::from("Date")
        } else if self.realm.regexp_at(h).is_some() {
            String::from("RegExp")
        } else {
            String::from("Object")
        })
    }

    /// Allocates a heap string and returns its boxed handle.
    fn new_str(&mut self, s: &str) -> NanBox {
        NanBox::handle(self.realm.new_string(s).to_raw())
    }

    /// Sorts `elems` with a JS comparator (a negative result orders `a` before
    /// `b`); without one, by the elements' string forms. Insertion sort, so the
    /// comparator can call back into the interpreter.
    fn sort_array(
        &mut self,
        elems: Vec<NanBox>,
        cmp: NanBox,
        numeric: bool,
    ) -> Result<Vec<NanBox>, ExecError> {
        let has_cmp = cmp.as_handle().is_some_and(|raw| {
            let h = Handle::from_raw(raw);
            self.realm.native_at(h).is_some() || self.realm.function_at(h).is_some()
        });
        // `undefined` elements always sort to the end and are never passed to the
        // comparator; only defined values are ordered against each other.
        let undefined_count = elems
            .iter()
            .filter(|e| matches!(e.unpack(), Unpacked::Undefined))
            .count();
        let mut elems: Vec<NanBox> = elems
            .into_iter()
            .filter(|e| !matches!(e.unpack(), Unpacked::Undefined))
            .collect();
        for i in 1..elems.len() {
            let mut j = i;
            while j > 0 {
                let order = if has_cmp {
                    let r = self.call(cmp, &[elems[j - 1], elems[j]])?;
                    self.realm.to_number(r)
                } else if numeric {
                    // A typed array's default comparison is numeric ascending (with
                    // `NaN` sorting to the end).
                    let a = self.realm.to_number(elems[j - 1]);
                    let b = self.realm.to_number(elems[j]);
                    if a < b || b.is_nan() {
                        -1.0
                    } else if a > b || a.is_nan() {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let a = self.realm.to_display_string(elems[j - 1]);
                    let b = self.realm.to_display_string(elems[j]);
                    if a < b {
                        -1.0
                    } else if a > b {
                        1.0
                    } else {
                        0.0
                    }
                };
                if order > 0.0 {
                    elems.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        // Re-append the `undefined` holes after the ordered defined values.
        elems.extend(core::iter::repeat_n(NanBox::undefined(), undefined_count));
        Ok(elems)
    }

    fn run_body(&mut self, body: Body<'a>) -> Result<NanBox, ExecError> {
        match body {
            Body::Expr(e) => self.eval(e),
            Body::Block(stmts) => {
                // Strict mode is inherited from the caller and additionally enabled
                // by a `"use strict"` directive prologue; it propagates to nested
                // bodies and is restored on exit.
                let saved_strict = self.strict;
                self.strict = self.strict || has_use_strict(stmts);
                self.hoist_with(stmts, true)?;
                let mut result = Ok(NanBox::undefined());
                for stmt in stmts {
                    match self.exec(stmt) {
                        Ok(Flow::Return(v)) => {
                            result = Ok(v);
                            break;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
                self.strict = saved_strict;
                result
            }
        }
    }

    // --- statements ---

    fn exec(&mut self, stmt: &'a Stmt) -> Result<Flow, ExecError> {
        match stmt {
            Stmt::Empty { .. } => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Expr { expression, .. } => Ok(Flow::Normal(self.eval(expression)?)),
            Stmt::Var(decl) => {
                self.exec_var(decl)?;
                Ok(Flow::Normal(NanBox::undefined()))
            }
            // Function declarations are handled by hoisting; nothing to do here.
            Stmt::Function(_) => Ok(Flow::Normal(NanBox::undefined())),
            Stmt::Class(class) => {
                let value = self.make_class(class)?;
                if let Some(id) = &class.id {
                    self.current.declare(&id.name, value);
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::Block { body, .. } => self.exec_block(body),
            Stmt::If {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_truthy(test)? {
                    self.exec(consequent)
                } else if let Some(alt) = alternate {
                    self.exec(alt)
                } else {
                    Ok(Flow::Normal(NanBox::undefined()))
                }
            }
            Stmt::While { test, body, .. } => {
                let label = self.pending_label.take();
                while self.eval_truthy(test)? {
                    match loop_action(self.exec(body)?, &label) {
                        LoopAction::Next => {}
                        LoopAction::Stop => break,
                        LoopAction::Propagate(f) => return Ok(f),
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::DoWhile { body, test, .. } => {
                let label = self.pending_label.take();
                loop {
                    match loop_action(self.exec(body)?, &label) {
                        LoopAction::Next => {}
                        LoopAction::Stop => break,
                        LoopAction::Propagate(f) => return Ok(f),
                    }
                    if !self.eval_truthy(test)? {
                        break;
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
            }
            Stmt::Labeled { label, body, .. } => {
                // The label is handed to a *directly* labeled loop (via `pending_label`)
                // so its own `break`/`continue label` see it. For any other body (a
                // block, `if`, …) leaving `pending_label` set would let a nested loop
                // wrongly claim the label, so it stays unset and the break-match below
                // unwinds `break label` out of the whole labeled statement.
                let is_loop = matches!(
                    &**body,
                    Stmt::For { .. }
                        | Stmt::While { .. }
                        | Stmt::DoWhile { .. }
                        | Stmt::ForOf { .. }
                        | Stmt::ForIn { .. }
                );
                if is_loop {
                    self.pending_label = Some(String::from(&*label.name));
                }
                let flow = self.exec(body)?;
                self.pending_label = None;
                // A labeled statement consumes a matching `break label`.
                Ok(match flow {
                    Flow::Break(Some(l)) if l == *label.name => Flow::Normal(NanBox::undefined()),
                    other => other,
                })
            }
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.exec_for(init.as_ref(), test.as_deref(), update.as_deref(), body),
            Stmt::Return { argument, .. } => {
                let v = match argument {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break { label, .. } => {
                Ok(Flow::Break(label.as_ref().map(|l| String::from(&*l.name))))
            }
            Stmt::Continue { label, .. } => Ok(Flow::Continue(
                label.as_ref().map(|l| String::from(&*l.name)),
            )),
            Stmt::Throw { argument, .. } => {
                let v = self.eval(argument)?;
                Err(ExecError::Throw(v))
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => self.exec_try(block, handler.as_ref(), finalizer.as_deref()),
            Stmt::ForOf {
                left,
                right,
                body,
                is_await,
                ..
            } => {
                let iterable = self.eval(right)?;
                // A user iterator (not a built-in array/string/Map/Set or generator)
                // runs lazily: one `next()` per iteration, with `IteratorClose` on an
                // early exit — so `break` calls `return()` and an infinite iterator can
                // be cut short. `for await` keeps the eager path (it awaits each value).
                if !*is_await && let Some(ih) = self.for_of_get_iterator(iterable)? {
                    // A user `[Symbol.iterator]` that is a generator is still drained
                    // eagerly (its `next` is built-in dispatch, not a readable method) —
                    // from this same iterator, so it is not re-created.
                    if self.realm.get_property(ih, GEN_BUF).is_some() {
                        let values = self.iterate_values(NanBox::handle(ih.to_raw()))?;
                        return self.exec_for_each(left, body, values);
                    }
                    return self.exec_for_of_iter(left, body, ih);
                }
                let mut values = self.iterate_values(iterable)?;
                // `for await (…)`: await each iterated value (a non-promise passes
                // through unchanged).
                if *is_await {
                    for v in &mut values {
                        *v = self.await_value(*v)?;
                    }
                }
                self.exec_for_each(left, body, values)
            }
            Stmt::ForIn {
                left, right, body, ..
            } => {
                let obj = self.eval(right)?;
                // A proxy with an `ownKeys` trap enumerates through it; otherwise
                // the normal own + inherited enumerable key walk.
                let keys = if let Some(raw) = obj.as_handle()
                    && let Some(trap_keys) =
                        self.proxy_own_enumerable_keys(Handle::from_raw(raw))?
                {
                    trap_keys.iter().map(|k| self.new_str(k)).collect()
                } else {
                    self.iterate_keys(obj)
                };
                self.exec_for_each(left, body, keys)
            }
            Stmt::Switch {
                discriminant,
                cases,
                ..
            } => self.exec_switch(discriminant, cases),
            _ => Err(ExecError::Unsupported("statement")),
        }
    }

    /// `try { … } catch (e) { … } finally { … }`. A `catch` handles a thrown
    /// value; the `finally` block runs on every exit (normal, thrown, or
    /// `return`/`break`/`continue`), and its own abrupt completion takes over.
    fn exec_try(
        &mut self,
        block: &'a [Stmt],
        handler: Option<&'a crate::ast::CatchClause>,
        finalizer: Option<&'a [Stmt]>,
    ) -> Result<Flow, ExecError> {
        let mut outcome = self.exec_scoped(block);
        // A thrown value is routed to the catch clause, if any.
        if let (Err(ExecError::Throw(value)), Some(catch)) = (&outcome, handler) {
            let thrown = *value;
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            // The catch binding may be a name, a destructuring pattern, or absent
            // (optional catch binding).
            let bound = match &catch.param {
                Some(target) => self.bind_pattern(target, thrown),
                None => Ok(()),
            };
            outcome = bound.and_then(|()| self.exec_seq(&catch.body));
            self.current = saved;
        }
        // `finally` runs regardless; an abrupt finally overrides the outcome.
        if let Some(fin) = finalizer {
            match self.exec_scoped(fin) {
                Ok(Flow::Normal(_)) => outcome,
                other => other, // finally returned/broke/threw → that wins
            }
        } else {
            outcome
        }
    }

    /// Runs a statement list in a fresh child scope.
    fn exec_scoped(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = self.exec_seq(body);
        self.current = saved;
        result
    }

    fn exec_block(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.exec_scoped(body)
    }

    /// Executes a statement sequence in the current scope (with hoisting).
    fn exec_seq(&mut self, body: &'a [Stmt]) -> Result<Flow, ExecError> {
        self.hoist(body)?;
        let mut last = Flow::Normal(NanBox::undefined());
        for stmt in body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = Flow::Normal(v),
                other => return Ok(other),
            }
        }
        Ok(last)
    }

    fn exec_var(&mut self, decl: &'a VarDecl) -> Result<(), ExecError> {
        let is_var = matches!(decl.kind, crate::ast::VarDeclKind::Var);
        for d in &decl.declarations {
            // A bare `var x;` (no initializer) must not clobber the value a
            // hoisted binding may already hold from an earlier assignment.
            if is_var && d.init.is_none() {
                continue;
            }
            let value = match &d.init {
                Some(e) => self.eval(e)?,
                None => NanBox::undefined(),
            };
            // An anonymous function/class assigned to a name takes that name
            // (`const f = function(){}` → `f.name === "f"`).
            if let (Some(init), BindingTarget::Ident(Ident { name, .. })) = (&d.init, &d.target)
                && matches!(init, Expr::Function(_) | Expr::Class(_) | Expr::Arrow(_))
            {
                self.set_fn_name(value, name);
            }
            // `var` assigns to its hoisted binding (in the function/program
            // scope), so a declaration inside a block updates the same variable.
            if is_var && let BindingTarget::Ident(Ident { name, .. }) = &d.target {
                if !self.current.set(name, value) {
                    self.current.declare(name, value);
                }
                continue;
            }
            // A simple `const x = …` binding is tracked so reassignment throws.
            if matches!(decl.kind, crate::ast::VarDeclKind::Const)
                && let BindingTarget::Ident(Ident { name, .. }) = &d.target
            {
                self.current.declare_const(name, value);
                continue;
            }
            self.bind_pattern(&d.target, value)?;
        }
        Ok(())
    }

    /// Binds a (possibly destructuring) target to `value`, declaring the names
    /// it introduces in the current scope.
    fn bind_pattern(&mut self, target: &'a BindingTarget, value: NanBox) -> Result<(), ExecError> {
        match target {
            BindingTarget::Ident(Ident { name, .. }) => self.current.declare(name, value),
            BindingTarget::Array(pat) => {
                // Any iterable destructures (strings, Sets, generators, …); a
                // non-iterable (null, a plain object, a number) is a TypeError.
                let has_rest = pat
                    .elements
                    .iter()
                    .any(|e| matches!(e, ArrayPatternElement::Rest { .. }));
                let needed = pat
                    .elements
                    .iter()
                    .filter(|e| !matches!(e, ArrayPatternElement::Rest { .. }))
                    .count();
                // Without a rest target, a *user* iterator is pulled lazily for only
                // the values the pattern needs, then closed (`IteratorClose`) — so
                // `[a, b] = infiniteIterator` terminates and `return()` runs. Arrays,
                // strings, Sets, and generators take the eager path (no user `next`).
                let elems = if !has_rest && let Some(ih) = self.for_of_get_iterator(value)? {
                    if self.realm.get_property(ih, GEN_BUF).is_some() {
                        // An (eager) generator iterator has no callable `next` property;
                        // its values are already buffered — drain the obtained iterator
                        // (don't re-invoke `Symbol.iterator`, which would re-run it).
                        self.iterate_values(NanBox::handle(ih.to_raw()))?
                    } else {
                        // A plain user iterator: pull only the values the pattern needs,
                        // then close it (so `[a, b] = infiniteIterator` terminates).
                        let iterator = NanBox::handle(ih.to_raw());
                        let mut out = Vec::with_capacity(needed);
                        let mut exhausted = false;
                        for _ in 0..needed {
                            let next_fn = self.read_member(ih, "next")?;
                            let res = self.call_with_this(next_fn, iterator, &[])?;
                            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                                return Err(ExecError::Throw(
                                    self.new_str("iterator result is not an object"),
                                ));
                            };
                            let done = self.read_member(rh, "done")?;
                            if self.realm.truthy(done) {
                                exhausted = true;
                                break;
                            }
                            out.push(self.read_member(rh, "value")?);
                        }
                        if !exhausted {
                            self.iterator_close(ih)?;
                        }
                        out
                    }
                } else {
                    self.iterate_values(value)?
                };
                let mut i = 0;
                for el in &pat.elements {
                    match el {
                        ArrayPatternElement::Hole => i += 1,
                        ArrayPatternElement::Item {
                            target, default, ..
                        } => {
                            let mut v = elems.get(i).copied().unwrap_or(NanBox::undefined());
                            if matches!(v.unpack(), Unpacked::Undefined)
                                && let Some(d) = default
                            {
                                v = self.eval(d)?;
                            }
                            self.bind_pattern(target, v)?;
                            i += 1;
                        }
                        ArrayPatternElement::Rest { target, .. } => {
                            let rest = elems[i.min(elems.len())..].to_vec();
                            let h = self.realm.new_array(rest);
                            self.bind_pattern(target, NanBox::handle(h.to_raw()))?;
                        }
                    }
                }
            }
            BindingTarget::Object(pat) => {
                // Object destructuring requires a coercible value: null/undefined throw
                // a TypeError (RequireObjectCoercible).
                if matches!(value.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    let m = self.new_str("Cannot destructure 'null' or 'undefined' as an object");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let src = value.as_handle().map(Handle::from_raw);
                let mut used: Vec<String> = Vec::new();
                for prop in &pat.properties {
                    // A computed key (`{ [expr]: t }`) is evaluated here.
                    let key = self.eval_prop_key(&prop.key)?;
                    // Read through `read_member` so accessors fire and inherited /
                    // string-length / array-length properties resolve (not just own
                    // data slots).
                    let mut v = match src {
                        Some(h) => self.read_member(h, &key)?,
                        None => NanBox::undefined(),
                    };
                    if matches!(v.unpack(), Unpacked::Undefined)
                        && let Some(d) = &prop.default
                    {
                        v = self.eval(d)?;
                    }
                    used.push(key);
                    self.bind_pattern(&prop.value, v)?;
                }
                if let Some(rest) = &pat.rest {
                    let obj = self.realm.new_object();
                    if let Some(h) = src {
                        for k in self.realm.object_keys(h).unwrap_or_default() {
                            if !used.contains(&k) {
                                let pv = self
                                    .realm
                                    .get_property(h, &k)
                                    .unwrap_or(NanBox::undefined());
                                self.realm.set_property(obj, &k, pv);
                            }
                        }
                    }
                    self.bind_pattern(rest, NanBox::handle(obj.to_raw()))?;
                }
            }
        }
        Ok(())
    }

    /// The values iterated by `for-of`: array elements, string chars, `Set`
    /// values, or `Map` `[key, value]` pairs.
    /// Recursively flattens nested arrays up to `depth` levels (for `flat`).
    fn flatten(&self, elems: &[NanBox], depth: i32) -> Vec<NanBox> {
        let mut out = Vec::new();
        for e in elems {
            if depth > 0
                && let Some(inner) = e
                    .as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.array_elements(h).map(<[_]>::to_vec))
            {
                out.extend(self.flatten(&inner, depth - 1));
            } else {
                out.push(*e);
            }
        }
        out
    }

    fn iterate_values(&mut self, v: NanBox) -> Result<Vec<NanBox>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            let m = self.new_str("is not iterable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // A `String` wrapper object iterates its characters (a `Number`/`Boolean`
        // wrapper is not iterable — falls through to the error).
        if let Some(prim) = self.realm.get_property(h, PRIM_WRAP)
            && let Some(ph) = prim.as_handle().map(Handle::from_raw)
            && self.realm.string_value(ph).is_some()
        {
            return self.iterate_values(prim);
        }
        if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
            return Ok(elems);
        }
        if let Some(s) = self.realm.string_value(h) {
            let chars: Vec<char> = s.chars().collect();
            return Ok(chars
                .iter()
                .map(|c| self.new_str(&String::from(*c)))
                .collect());
        }
        if let Some(entries) = self.realm.collection_entries(h) {
            if self.realm.collection_is_set(h) == Some(true) {
                return Ok(entries.iter().map(|(k, _)| *k).collect());
            }
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push(NanBox::handle(
                    self.realm.new_array(alloc::vec![k, v]).to_raw(),
                ));
            }
            return Ok(out);
        }
        // A generator iterator: its remaining buffered values.
        if let Some(buf) = self
            .realm
            .get_property(h, GEN_BUF)
            .and_then(|b| b.as_handle())
            .map(Handle::from_raw)
        {
            let idx = self
                .realm
                .get_property(h, GEN_IDX)
                .and_then(|n| n.as_number())
                .unwrap_or(0.0) as usize;
            let elems = self
                .realm
                .array_elements(buf)
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            return Ok(elems.into_iter().skip(idx).collect());
        }
        // A custom iterable: call `obj[Symbol.iterator]()` and drain `.next()`.
        // The method may be an own/inherited property or a class method whose
        // computed key is `Symbol.iterator` (`class C { *[Symbol.iterator]() {…} }`).
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let mut iter_fn = self.realm.get_property(h, &iter_key);
        if iter_fn.is_none() {
            iter_fn = self.class_iterator_method(h)?;
        }
        if let Some(f) = iter_fn
            && f.as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
        {
            let iterator = self.call_with_this(f, v, &[])?;
            let Some(ih) = iterator.as_handle().map(Handle::from_raw) else {
                return Err(ExecError::Throw(self.new_str("iterator is not an object")));
            };
            // A generator iterator (its `next` is a built-in method, not a
            // readable property) is drained directly from its buffer.
            if self.realm.get_property(ih, GEN_BUF).is_some() {
                return self.iterate_values(iterator);
            }
            let mut out = Vec::new();
            loop {
                let next_fn = self.read_member(ih, "next")?;
                let res = self.call_with_this(next_fn, iterator, &[])?;
                let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                    return Err(ExecError::Throw(
                        self.new_str("iterator result is not an object"),
                    ));
                };
                let done = self.read_member(rh, "done")?;
                if self.realm.truthy(done) {
                    break;
                }
                out.push(self.read_member(rh, "value")?);
                if out.len() > GEN_CAP {
                    return Err(ExecError::Throw(self.new_str("iterator did not terminate")));
                }
            }
            return Ok(out);
        }
        let m = self.new_str("is not iterable");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// Finds a class instance's `[Symbol.iterator]` method (a method whose
    /// computed key evaluates to the well-known iterator symbol), walking the
    /// `extends` chain. Returns the bound method value, or `None`.
    fn class_iterator_method(
        &mut self,
        h: crate::heap::Handle,
    ) -> Result<Option<NanBox>, ExecError> {
        let Some(tag) = self.realm.class_tag(h) else {
            return Ok(None);
        };
        let iter_sym = self.well_known_symbol("iterator");
        let mut cur = Some(tag);
        while let Some(cid) = cur {
            let class = self.classes[cid as usize];
            let env = self.class_envs[cid as usize].clone();
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && !m.is_static
                    && m.kind == MethodKind::Method
                    && let PropertyKey::Computed(ke) = &m.key
                {
                    let saved = core::mem::replace(&mut self.current, env.clone());
                    let key = self.eval(ke);
                    self.current = saved;
                    if self.realm.strict_equals(key?, iter_sym) {
                        let saved = core::mem::replace(&mut self.current, env.clone());
                        let f = self.make_method(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            m.value.is_generator,
                            Some(cid),
                            false,
                        );
                        self.current = saved;
                        return Ok(Some(f));
                    }
                }
            }
            cur = self.resolve_super(class, &env)?.map(|(p, _)| p);
        }
        Ok(None)
    }

    /// The keys iterated by `for-in`: object property names or array indices,
    /// as strings.
    fn iterate_keys(&mut self, v: NanBox) -> Vec<NanBox> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Vec::new();
        };
        // A proxy with no `ownKeys` trap (the trap case is handled by the caller)
        // enumerates its target's keys.
        let h = self.proxy_key_target(h);
        // `for-in` enumerates own enumerable keys, then enumerable keys inherited
        // through the prototype chain — each name only once, own keys first.
        let mut seen = alloc::collections::BTreeSet::new();
        let mut out = Vec::new();
        // An array's own keys lead with its integer indices (a VM closure's backing
        // cells are not enumerable).
        if !self.realm.is_vm_function(h)
            && let Some(len) = self.realm.array_length(h)
        {
            for i in 0..len {
                let k = alloc::format!("{i}");
                if seen.insert(k.clone()) {
                    out.push(self.new_str(&k));
                }
            }
        }
        let mut cur = Some(h);
        while let Some(c) = cur {
            // Plain objects keep keys in the cell; arrays/functions keep named
            // properties in their auxiliary object.
            let named = self
                .realm
                .object_keys(c)
                .unwrap_or_else(|| self.realm.aux_named_keys(c));
            for k in named {
                if seen.insert(k.clone()) {
                    out.push(self.new_str(&k));
                }
            }
            cur = self.realm.object_proto(c);
        }
        out
    }

    /// Runs `body` once per `item`, binding the loop variable (a fresh scope per
    /// iteration for a declared head).
    /// Obtains a *user* iterable's iterator object (calling `[Symbol.iterator]`
    /// once), for the lazy `for-of` path. Returns `None` for built-in iterables
    /// (arrays/strings/Maps/Sets) and generator values, which `iterate_values`
    /// drains eagerly, and for non-iterables.
    fn for_of_get_iterator(&mut self, v: NanBox) -> Result<Option<Handle>, ExecError> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Ok(None);
        };
        if self.realm.array_elements(h).is_some()
            || self.realm.string_value(h).is_some()
            || self.realm.collection_entries(h).is_some()
            || self.realm.get_property(h, GEN_BUF).is_some()
        {
            return Ok(None);
        }
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let mut iter_fn = self.realm.get_property(h, &iter_key);
        if iter_fn.is_none() {
            iter_fn = self.class_iterator_method(h)?;
        }
        let Some(f) = iter_fn else { return Ok(None) };
        if !f
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            return Ok(None);
        }
        let iterator = self.call_with_this(f, v, &[])?;
        match iterator.as_handle().map(Handle::from_raw) {
            Some(ih) => Ok(Some(ih)),
            None => Err(ExecError::Throw(self.new_str("iterator is not an object"))),
        }
    }

    /// `IteratorClose`: invoke the iterator's `return()` method (if any) on an early
    /// exit, so the iterator can release resources. Errors from `return()` propagate.
    fn iterator_close(&mut self, ih: Handle) -> Result<(), ExecError> {
        let ret = self.read_member(ih, "return")?;
        if ret
            .as_handle()
            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            self.call_with_this(ret, NanBox::handle(ih.to_raw()), &[])?;
        }
        Ok(())
    }

    /// A lazy `for-of` over a user iterator: pull one value per iteration (so an
    /// infinite iterator can be cut short by `break`), and run `IteratorClose` on
    /// every early exit (`break`/`return`/`throw`) — unlike the eager path.
    fn exec_for_of_iter(
        &mut self,
        left: &'a crate::ast::ForLeft,
        body: &'a Stmt,
        ih: Handle,
    ) -> Result<Flow, ExecError> {
        use crate::ast::ForLeft;
        let label = self.pending_label.take();
        let iterator = NanBox::handle(ih.to_raw());
        loop {
            let next_fn = self.read_member(ih, "next")?;
            let res = self.call_with_this(next_fn, iterator, &[])?;
            let Some(rh) = res.as_handle().map(Handle::from_raw) else {
                return Err(ExecError::Throw(
                    self.new_str("iterator result is not an object"),
                ));
            };
            let done = self.read_member(rh, "done")?;
            if self.realm.truthy(done) {
                return Ok(Flow::Normal(NanBox::undefined()));
            }
            let item = self.read_member(rh, "value")?;
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            let r = (|| {
                match left {
                    ForLeft::Decl { target, .. } => self.bind_pattern(target, item)?,
                    ForLeft::Target(expr) => {
                        self.assign_to(expr, item)?;
                    }
                }
                self.exec(body)
            })();
            self.current = saved;
            match r {
                Ok(flow) => match loop_action(flow, &label) {
                    LoopAction::Next => {}
                    LoopAction::Stop => {
                        self.iterator_close(ih)?;
                        return Ok(Flow::Normal(NanBox::undefined()));
                    }
                    LoopAction::Propagate(f) => {
                        self.iterator_close(ih)?;
                        return Ok(f);
                    }
                },
                Err(e) => {
                    // An abrupt body completion still closes the iterator, but its
                    // own error is suppressed in favor of the original.
                    let _ = self.iterator_close(ih);
                    return Err(e);
                }
            }
        }
    }

    fn exec_for_each(
        &mut self,
        left: &'a crate::ast::ForLeft,
        body: &'a Stmt,
        items: Vec<NanBox>,
    ) -> Result<Flow, ExecError> {
        use crate::ast::ForLeft;
        let label = self.pending_label.take();
        for item in items {
            let child = self.current.child();
            let saved = core::mem::replace(&mut self.current, child);
            let r = (|| {
                match left {
                    ForLeft::Decl { target, .. } => self.bind_pattern(target, item)?,
                    ForLeft::Target(expr) => {
                        self.assign_to(expr, item)?;
                    }
                }
                self.exec(body)
            })();
            self.current = saved;
            match loop_action(r?, &label) {
                LoopAction::Next => {}
                LoopAction::Stop => break,
                LoopAction::Propagate(f) => return Ok(f),
            }
        }
        Ok(Flow::Normal(NanBox::undefined()))
    }

    /// Reads the current value of an assignment target (identifier or member).
    fn read_target(&mut self, target: &'a Expr) -> Result<NanBox, ExecError> {
        match target {
            Expr::Ident(id) => Ok(self.current.get(&id.name).unwrap_or(NanBox::undefined())),
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.eval(object)?;
                match obj.as_handle() {
                    Some(raw) => self.member(Handle::from_raw(raw), property),
                    None => Ok(NanBox::undefined()),
                }
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    /// Destructures `value` into an assignment pattern of existing targets
    /// (`[a, b] = …`, `({ x: obj.p } = …)`), recursing into nested patterns.
    fn assign_destructure(&mut self, target: &'a Expr, value: NanBox) -> Result<(), ExecError> {
        match target {
            Expr::Array { elements, .. } => {
                // A non-iterable right-hand side (null, a plain object) is a TypeError.
                let items = self.iterate_values(value)?;
                let mut i = 0;
                for el in elements {
                    match el {
                        ArrayElement::Hole => i += 1,
                        ArrayElement::Item(e) => {
                            let v = items.get(i).copied().unwrap_or(NanBox::undefined());
                            self.assign_destructure(e, v)?;
                            i += 1;
                        }
                        ArrayElement::Spread(e) => {
                            let rest = items[i.min(items.len())..].to_vec();
                            let h = NanBox::handle(self.realm.new_array(rest).to_raw());
                            self.assign_destructure(e, h)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Object { members, .. } => {
                let src = value.as_handle().map(Handle::from_raw);
                let mut used: Vec<String> = Vec::new();
                for m in members {
                    match m {
                        ObjectMember::Property {
                            key, value: tgt, ..
                        } => {
                            let k = self.eval_prop_key(key)?;
                            let v = src
                                .and_then(|h| self.realm.get_property(h, &k))
                                .unwrap_or(NanBox::undefined());
                            used.push(k);
                            self.assign_destructure(tgt, v)?;
                        }
                        ObjectMember::Spread { value: tgt, .. } => {
                            let obj = self.realm.new_object();
                            if let Some(h) = src {
                                for k in self.realm.object_keys(h).unwrap_or_default() {
                                    if !used.contains(&k) {
                                        let pv = self
                                            .realm
                                            .get_property(h, &k)
                                            .unwrap_or(NanBox::undefined());
                                        self.realm.set_property(obj, &k, pv);
                                    }
                                }
                            }
                            self.assign_destructure(tgt, NanBox::handle(obj.to_raw()))?;
                        }
                        ObjectMember::Accessor { .. } => {}
                    }
                }
                Ok(())
            }
            // A defaulted target in a pattern (`[a = 1] = …`, `{ x: a = 1 } = …`):
            // use the default when the source value is `undefined`.
            Expr::Assign {
                op: AssignOp::Assign,
                target: inner,
                value: default_expr,
                ..
            } => {
                let v = if matches!(value.unpack(), Unpacked::Undefined) {
                    self.eval(default_expr)?
                } else {
                    value
                };
                self.assign_destructure(inner, v)
            }
            // A leaf target (identifier or member).
            _ => self.assign_to(target, value),
        }
    }

    /// Assigns `value` to an existing target (an identifier or member).
    fn assign_to(&mut self, target: &'a Expr, value: NanBox) -> Result<(), ExecError> {
        match target {
            Expr::Ident(id) => {
                // Reassigning a `const` binding is a TypeError.
                if self.current.is_const(&id.name) {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                if !self.current.set(&id.name, value) {
                    // Strict mode forbids creating an implicit global.
                    if self.strict {
                        let m = self.new_str(&alloc::format!("{} is not defined", id.name));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    self.current.declare(&id.name, value);
                }
                Ok(())
            }
            Expr::Member {
                object, property, ..
            } => {
                // `super.x = v` invokes the inherited setter with the current `this`.
                if matches!(&**object, Expr::Super(_)) {
                    let name = self.eval_prop_key(property)?;
                    return self.assign_super_member(&name, value);
                }
                let obj = self.eval(object)?;
                if let Some(raw) = obj.as_handle() {
                    self.assign_member(Handle::from_raw(raw), property, value)?;
                }
                Ok(())
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    fn exec_switch(
        &mut self,
        discriminant: &'a Expr,
        cases: &'a [crate::ast::SwitchCase],
    ) -> Result<Flow, ExecError> {
        let value = self.eval(discriminant)?;
        // Find the first matching `case` (strict equality), else `default`.
        let mut start = None;
        for (i, case) in cases.iter().enumerate() {
            if let Some(test) = &case.test {
                let t = self.eval(test)?;
                if self.realm.strict_equals(value, t) {
                    start = Some(i);
                    break;
                }
            }
        }
        if start.is_none() {
            start = cases.iter().position(|c| c.test.is_none());
        }
        let Some(start) = start else {
            return Ok(Flow::Normal(NanBox::undefined()));
        };
        // Run from the matched clause, falling through until `break`.
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        let result = (|| {
            for case in &cases[start..] {
                for stmt in &case.body {
                    match self.exec(stmt)? {
                        // A plain `break` ends the switch; everything else
                        // (labeled break, continue, return) bubbles out.
                        Flow::Break(None) => return Ok(Flow::Normal(NanBox::undefined())),
                        Flow::Normal(_) => {}
                        other => return Ok(other),
                    }
                }
            }
            Ok(Flow::Normal(NanBox::undefined()))
        })();
        self.current = saved;
        result
    }

    fn exec_for(
        &mut self,
        init: Option<&'a ForInit>,
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
    ) -> Result<Flow, ExecError> {
        let label = self.pending_label.take();
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
        // For a `let`/`const` head, each iteration gets a fresh binding (so a
        // closure created in the body captures that iteration's value).
        let per_iter_names: Vec<String> = match init {
            Some(ForInit::Var(decl)) if decl.kind != crate::ast::VarDeclKind::Var => decl
                .declarations
                .iter()
                .filter_map(|d| match &d.target {
                    BindingTarget::Ident(Ident { name, .. }) => Some(String::from(&**name)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        let result = (|| {
            match init {
                Some(ForInit::Var(decl)) => self.exec_var(decl)?,
                Some(ForInit::Expr(e)) => {
                    self.eval(e)?;
                }
                None => {}
            }
            loop {
                let go = match test {
                    Some(t) => self.eval_truthy(t)?,
                    None => true,
                };
                if !go {
                    break;
                }
                // Run the body in a per-iteration scope seeded from the loop
                // variables, then copy any mutations back for test/update.
                let flow = if per_iter_names.is_empty() {
                    self.exec(body)?
                } else {
                    let iter = self.current.child();
                    for name in &per_iter_names {
                        iter.declare(name, self.current.get(name).unwrap_or(NanBox::undefined()));
                    }
                    let loop_scope = core::mem::replace(&mut self.current, iter);
                    let f = self.exec(body);
                    for name in &per_iter_names {
                        if let Some(v) = self.current.get(name) {
                            loop_scope.set(name, v);
                        }
                    }
                    self.current = loop_scope;
                    f?
                };
                match loop_action(flow, &label) {
                    LoopAction::Next => {}
                    LoopAction::Stop => break,
                    LoopAction::Propagate(f) => return Ok(f),
                }
                if let Some(u) = update {
                    self.eval(u)?;
                }
            }
            Ok(Flow::Normal(NanBox::undefined()))
        })();
        self.current = saved;
        result
    }

    // --- expressions ---

    /// Returns the cached well-known symbol `name` (e.g. `iterator`), creating it
    /// on first use. Each is a stable, unique symbol for the realm's lifetime.
    fn well_known_symbol(&mut self, name: &'static str) -> NanBox {
        if let Some(s) = self.well_known_symbols.get(name) {
            return *s;
        }
        let sym = NanBox::handle(
            self.realm
                .new_symbol(&alloc::format!("Symbol.{name}"))
                .to_raw(),
        );
        self.well_known_symbols.insert(name, sym);
        sym
    }

    /// Evaluates `e` and returns its JS truthiness (heap-aware, so an empty
    /// string is falsy).
    fn eval_truthy(&mut self, e: &'a Expr) -> Result<bool, ExecError> {
        let v = self.eval(e)?;
        Ok(self.realm.truthy(v))
    }

    /// Calls `f(args)` and returns the result's truthiness.
    /// Calls `f` with an explicit `this` and returns whether the result is truthy
    /// (for array predicates with a `thisArg`).
    fn call_truthy_this(
        &mut self,
        f: NanBox,
        this: NanBox,
        args: &[NanBox],
    ) -> Result<bool, ExecError> {
        let r = self.call_with_this(f, this, args)?;
        Ok(self.realm.truthy(r))
    }

    /// Resolves an object/class property key to its string name, evaluating a
    /// `[computed]` key expression where present (a symbol maps to its identity
    /// key, any other value to its string form).
    fn eval_prop_key(&mut self, key: &'a PropertyKey) -> Result<String, ExecError> {
        match key {
            PropertyKey::Computed(e) => {
                let v = self.eval(e)?;
                Ok(self.member_key(v))
            }
            _ => static_key(key),
        }
    }

    /// The storage key for a property access value: a symbol becomes a unique,
    /// non-enumerable `"\0sym:<id>"` key (so symbol-keyed properties keep their
    /// identity and stay out of string enumeration); anything else is its string
    /// form.
    fn member_key(&self, k: NanBox) -> String {
        if let Some(raw) = k.as_handle()
            && let Some((_, id)) = self.realm.symbol_at(Handle::from_raw(raw))
        {
            return alloc::format!("\u{0}sym:{id}");
        }
        self.realm.to_display_string(k)
    }

    /// `ToPropertyKey(k)`: like `member_key`, but a non-string, non-symbol object
    /// key is coerced with ToPrimitive(String) so a user `toString` is honored
    /// (`obj[{toString(){return "x"}}]` keys on `"x"`).
    fn coerce_property_key(&mut self, k: NanBox) -> Result<String, ExecError> {
        let is_object_key = k.as_handle().is_some_and(|raw| {
            let h = Handle::from_raw(raw);
            self.realm.symbol_at(h).is_none() && self.realm.string_value(h).is_none()
        });
        if is_object_key {
            let p = self.coerce_object(k, "string")?;
            return Ok(self.realm.to_display_string(p));
        }
        Ok(self.member_key(k))
    }

    /// Invokes a plain object's `[Symbol.toPrimitive](hint)` method, if it has a
    /// callable one. Returns `None` to fall back to `valueOf`/`toString`.
    fn symbol_to_primitive(&mut self, v: NanBox, hint: &str) -> Result<Option<NanBox>, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(None);
        };
        let h = Handle::from_raw(raw);
        let sym = self.well_known_symbol("toPrimitive");
        let key = self.member_key(sym);
        if let Some(f) = self.realm.get_property(h, &key)
            && f.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
        {
            let hint_box = self.new_str(hint);
            let r = self.call_with_this(f, v, &[hint_box])?;
            // `[Symbol.toPrimitive]` must return a primitive, else a TypeError.
            if self.is_object_value(r) {
                let m = self.new_str("Cannot convert object to primitive value");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(Some(r));
        }
        Ok(None)
    }

    /// Whether `v` is an object (a non-primitive heap value: object/array/function/…)
    /// rather than a string/symbol/bigint primitive or an immediate.
    fn is_object_value(&self, v: NanBox) -> bool {
        v.as_handle().map(Handle::from_raw).is_some_and(|h| {
            self.realm.string_value(h).is_none()
                && self.realm.symbol_at(h).is_none()
                && self.realm.bigint_at(h).is_none()
        })
    }

    /// ToPrimitive of an object/array for loose equality: an array becomes its
    /// `join` string; a plain object uses the default-hint ToPrimitive.
    fn coerce_for_eq(&mut self, v: NanBox) -> Result<NanBox, ExecError> {
        self.coerce_object(v, "default")
    }

    /// ToPrimitive of an object/array with `hint`: an array becomes its `join`
    /// string (arrays have no readable `valueOf`/`toString`); a plain object goes
    /// through `coerce_primitive`. Non-objects pass through.
    fn coerce_object(&mut self, v: NanBox, hint: &str) -> Result<NanBox, ExecError> {
        if let Some(h) = v.as_handle().map(Handle::from_raw) {
            if self.realm.is_array(h) {
                let s = self.realm.to_display_string(v);
                return Ok(self.new_str(&s));
            }
            if self.realm.object_keys(h).is_some() {
                return self.coerce_primitive(v, hint);
            }
        }
        Ok(v)
    }

    /// ToPrimitive for a plain object with the given hint: `[Symbol.toPrimitive]`
    /// first, then `valueOf`/`toString` (order depends on the hint), accepting
    /// the first non-object result. Non-objects (and strings/arrays) pass through.
    fn coerce_primitive(&mut self, v: NanBox, hint: &str) -> Result<NanBox, ExecError> {
        let Some(raw) = v.as_handle() else {
            return Ok(v);
        };
        let h = Handle::from_raw(raw);
        // A primitive wrapper's ToPrimitive is simply its boxed value.
        if let Some(prim) = self.realm.get_property(h, PRIM_WRAP) {
            return Ok(prim);
        }
        // Strings and arrays are handled by the arithmetic path directly.
        if self.realm.string_value(h).is_some()
            || self.realm.is_array(h)
            || self.realm.object_keys(h).is_none()
        {
            return Ok(v);
        }
        if let Some(r) = self.symbol_to_primitive(v, hint)? {
            return Ok(r);
        }
        // String hint tries `toString` first; number/default try `valueOf` first.
        let order = if hint == "string" {
            ["toString", "valueOf"]
        } else {
            ["valueOf", "toString"]
        };
        for method in order {
            let m = self.read_member(h, method)?;
            if m.as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let r = self.call_with_this(m, v, &[])?;
                if !self.is_object_value(r) {
                    return Ok(r);
                }
            }
        }
        // Neither `valueOf` nor `toString` produced a primitive — a TypeError.
        let m = self.new_str("Cannot convert object to primitive value");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    /// Coerces `v` to a string, invoking `[Symbol.toPrimitive]("string")` or a
    /// callable `toString` when present (else the default form).
    fn coerce_to_string(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(raw) = v.as_handle() {
            let h = Handle::from_raw(raw);
            // A Symbol has no implicit string conversion (e.g. in a template).
            if self.realm.symbol_at(h).is_some() {
                let m = self.new_str("Cannot convert a Symbol value to a string");
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            if self.realm.string_value(h).is_none()
                && !self.realm.is_array(h)
                && self.realm.object_keys(h).is_some()
            {
                let p = self.coerce_primitive(v, "string")?;
                if p.as_handle() != v.as_handle() {
                    return Ok(self.realm.to_display_string(p));
                }
            }
        }
        Ok(self.realm.to_display_string(v))
    }

    fn eval(&mut self, expr: &'a Expr) -> Result<NanBox, ExecError> {
        match expr {
            Expr::Null(_) => Ok(NanBox::null()),
            Expr::Bool { value, .. } => Ok(NanBox::boolean(*value)),
            Expr::Number { value, .. } => Ok(NanBox::number(*value)),
            Expr::BigInt { digits, .. } => {
                let n = parse_bigint(digits);
                Ok(NanBox::handle(self.realm.new_bigint(n).to_raw()))
            }
            Expr::Str { value, .. } => {
                let h = self.realm.new_string(value);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Ident(id) => match &*id.name {
                "undefined" => Ok(NanBox::undefined()),
                "NaN" => Ok(NanBox::number(f64::NAN)),
                "Infinity" => Ok(NanBox::number(f64::INFINITY)),
                name => match self.current.get(name) {
                    Some(v) => Ok(v),
                    // An unresolved reference throws a catchable ReferenceError.
                    None => {
                        let msg = self.new_str(&alloc::format!("{name} is not defined"));
                        Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(msg)),
                        ))
                    }
                },
            },
            Expr::Regex { pattern, flags, .. } => Ok(NanBox::handle(
                self.realm.new_regexp(pattern, flags).to_raw(),
            )),
            // A template literal: interleave cooked quasis with interpolations.
            Expr::Template(t) => {
                let mut out = String::new();
                for (i, quasi) in t.quasis.iter().enumerate() {
                    if let Some(cooked) = &quasi.cooked {
                        out.push_str(cooked);
                    }
                    if let Some(e) = t.expressions.get(i) {
                        let v = self.eval(e)?;
                        let s = self.coerce_to_string(v)?;
                        out.push_str(&s);
                    }
                }
                Ok(self.new_str(&out))
            }
            // The comma operator: evaluate all, yield the last.
            Expr::Sequence { expressions, .. } => {
                let mut last = NanBox::undefined();
                for e in expressions {
                    last = self.eval(e)?;
                }
                Ok(last)
            }
            // A tagged template: `tag(stringsArray, ...interpolatedValues)`.
            Expr::TaggedTemplate { tag, quasi, .. } => {
                // The frozen strings object is created once per template-literal site
                // and reused on every evaluation (its identity is observable to the tag).
                let cache_key = core::ptr::from_ref(quasi) as usize;
                let strings_arr = if let Some(cached) = self.tagged_template_cache.get(&cache_key) {
                    *cached
                } else {
                    let strings: Vec<NanBox> = quasi
                        .quasis
                        .iter()
                        .map(|q| self.new_str(q.cooked.as_deref().unwrap_or("")))
                        .collect();
                    let raw: Vec<NanBox> =
                        quasi.quasis.iter().map(|q| self.new_str(&q.raw)).collect();
                    let strings_h = self.realm.new_array(strings);
                    // The strings object carries a `.raw` array (for `String.raw` and
                    // tags reading `strings.raw`). Both arrays are frozen, per spec —
                    // freeze `.raw` first and `strings` last so the property write lands.
                    let raw_h = self.realm.new_array(raw);
                    self.realm.freeze_object(raw_h);
                    self.realm
                        .set_property(strings_h, "raw", NanBox::handle(raw_h.to_raw()));
                    self.realm.freeze_object(strings_h);
                    let arr = NanBox::handle(strings_h.to_raw());
                    self.tagged_template_cache.insert(cache_key, arr);
                    arr
                };
                let mut args = alloc::vec![strings_arr];
                for e in &quasi.expressions {
                    args.push(self.eval(e)?);
                }
                // A `recv.tag` tag (e.g. `String.raw`) is dispatched as a method
                // call, so a built-in tag works even if it isn't a readable value.
                if let Expr::Member {
                    object, property, ..
                } = &**tag
                    && let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                {
                    let recv = self.eval(object)?;
                    if let Some(result) = self.call_method(recv, name, &args)? {
                        return Ok(result);
                    }
                    // Fall back to a property-valued tag function.
                    let Some(raw) = recv.as_handle() else {
                        return Err(ExecError::NotCallable);
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    return self.call_with_this(f, recv, &args);
                }
                let tagf = self.eval(tag)?;
                self.call(tagf, &args)
            }
            Expr::This(_) => Ok(self.this_val),
            Expr::NewTarget(_) => Ok(self.new_target),
            Expr::Await { argument, .. } => {
                let v = self.eval(argument)?;
                self.await_value(v)
            }
            // Eager generators: `yield x` appends `x` to the active buffer;
            // `yield* it` appends each value of the iterable. The expression's
            // own value is `undefined` (we cannot thread `next()` arguments back).
            Expr::Yield {
                argument, delegate, ..
            } => {
                let v = match argument {
                    Some(e) => self.eval(e)?,
                    None => NanBox::undefined(),
                };
                if *delegate {
                    let vals = self.iterate_values(v)?;
                    if let Some(sink) = self.gen_sink.as_mut() {
                        if sink.len() + vals.len() > GEN_CAP {
                            return Err(ExecError::Throw(self.new_str("generator yield limit")));
                        }
                        sink.extend(vals);
                    }
                    // `yield* iterable` evaluates to the iterator's final value — a
                    // delegated generator's `return` value (else `undefined`).
                    let ret = v
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.get_property(h, GEN_RET))
                        .unwrap_or(NanBox::undefined());
                    return Ok(ret);
                } else if let Some(sink) = self.gen_sink.as_mut() {
                    if sink.len() >= GEN_CAP {
                        return Err(ExecError::Throw(self.new_str("generator yield limit")));
                    }
                    sink.push(v);
                }
                Ok(NanBox::undefined())
            }
            Expr::Function(func) => Ok(self.eval_fn_expr(func)),
            Expr::Arrow(arrow) => Ok(self.eval_arrow(arrow)),
            Expr::Class(class) => self.make_class(class),
            Expr::Unary { op, argument, .. } => {
                // `delete obj.x` removes a property; `typeof undefinedVar` must
                // not throw — both inspect the operand rather than its value.
                match op {
                    UnaryOp::Delete => {
                        // `delete` returns `false` when the property is
                        // non-configurable (sealed/frozen); `true` otherwise.
                        let mut result = true;
                        if let Expr::Member {
                            object, property, ..
                        } = &**argument
                        {
                            let obj = self.eval(object)?;
                            if let Some(raw) = obj.as_handle() {
                                let h = Handle::from_raw(raw);
                                let name = match property {
                                    PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                                        Some(String::from(&**s))
                                    }
                                    PropertyKey::Computed(e) => {
                                        let k = self.eval(e)?;
                                        Some(self.member_key(k))
                                    }
                                    _ => None,
                                };
                                if let Some(name) = name {
                                    // Proxy `deleteProperty` trap, or forward.
                                    if let Some((target, handler)) = self.realm.proxy_at(h) {
                                        self.guard_revoked(h)?;
                                        let trap = self
                                            .realm
                                            .get_property(handler, "deleteProperty")
                                            .unwrap_or(NanBox::undefined());
                                        if trap
                                            .as_handle()
                                            .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                                        {
                                            let kb = self.new_str(&name);
                                            self.call(
                                                trap,
                                                &[NanBox::handle(target.to_raw()), kb],
                                            )?;
                                        } else {
                                            self.realm.delete_property(target, &name);
                                        }
                                    } else if self.realm.is_array(h) && name == "length" {
                                        // An array's `length` is non-configurable.
                                        result = false;
                                    } else if let (true, Ok(i)) =
                                        (self.realm.is_array(h), name.parse::<usize>())
                                    {
                                        // `delete arr[i]` clears the element (no
                                        // true holes; the slot becomes undefined).
                                        self.realm.set_element(h, i, NanBox::undefined());
                                    } else {
                                        result = self.realm.delete_property(h, &name);
                                    }
                                }
                            }
                        } else if let Expr::Ident(id) = &**argument
                            && self.current.get(&id.name).is_some()
                        {
                            // Deleting a resolvable binding (a declared variable) is a
                            // no-op that returns `false`; an unresolvable name is `true`.
                            result = false;
                        }
                        return Ok(NanBox::boolean(result));
                    }
                    UnaryOp::Typeof => {
                        if let Expr::Ident(id) = &**argument
                            && self.current.get(&id.name).is_none()
                            && !matches!(&*id.name, "undefined" | "NaN" | "Infinity")
                        {
                            return Ok(self.new_str("undefined"));
                        }
                    }
                    _ => {}
                }
                let v = self.eval(argument)?;
                self.unary(*op, v)
            }
            // `x++` / `++x` / `x--` / `--x` on an identifier or member.
            Expr::Update {
                op,
                prefix,
                argument,
                ..
            } => {
                let current = self.read_target(argument)?;
                // A BigInt operand increments/decrements by one BigInt.
                if let Some(big) = current
                    .as_handle()
                    .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)))
                {
                    let one = crate::bignum::BigInt::from_i128(1);
                    let next = match op {
                        crate::ast::UpdateOp::Inc => big.add(&one),
                        crate::ast::UpdateOp::Dec => big.sub(&one),
                    };
                    let next_box = NanBox::handle(self.realm.new_bigint(next).to_raw());
                    self.assign_to(argument, next_box)?;
                    let old_box = NanBox::handle(self.realm.new_bigint(big).to_raw());
                    return Ok(if *prefix { next_box } else { old_box });
                }
                let old = self.realm.to_number(current);
                let next = match op {
                    crate::ast::UpdateOp::Inc => old + 1.0,
                    crate::ast::UpdateOp::Dec => old - 1.0,
                };
                self.assign_to(argument, NanBox::number(next))?;
                Ok(NanBox::number(if *prefix { next } else { old }))
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                // `#x in obj` — the ergonomic brand check (private fields are
                // stored under a `#`-prefixed key).
                if matches!(op, BinaryOp::In)
                    && let Expr::PrivateName(name, _) = &**left
                {
                    let obj = self.eval(right)?;
                    let present = obj
                        .as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|h| self.realm.has_own(h, &alloc::format!("#{name}")));
                    return Ok(NanBox::boolean(present));
                }
                let a = self.eval(left)?;
                let b = self.eval(right)?;
                self.binary(*op, a, b)
            }
            Expr::Logical {
                op, left, right, ..
            } => {
                let l = self.eval(left)?;
                let take_right = match op {
                    LogicalOp::And => self.realm.truthy(l),
                    LogicalOp::Or => !self.realm.truthy(l),
                    LogicalOp::Nullish => {
                        matches!(l.unpack(), Unpacked::Undefined | Unpacked::Null)
                    }
                };
                if take_right { self.eval(right) } else { Ok(l) }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
                ..
            } => {
                if self.eval_truthy(test)? {
                    self.eval(consequent)
                } else {
                    self.eval(alternate)
                }
            }
            Expr::Assign {
                op, target, value, ..
            } => self.eval_assign(*op, target, value),
            Expr::Call {
                callee,
                arguments,
                optional: call_optional,
                ..
            } => {
                // `super(args)` — invoke the base constructor on the current
                // instance.
                if matches!(&**callee, Expr::Super(_)) {
                    let args = self.eval_args(arguments)?;
                    if let Some((pid, penv)) = self.pending_super.clone() {
                        if let Some(raw) = self.this_val.as_handle() {
                            self.run_constructor(pid, &penv, Handle::from_raw(raw), &args)?;
                        }
                        return Ok(NanBox::undefined());
                    }
                    // `super(...)` reaching a native constructor (`extends Error`).
                    if let Some(nid) = self.pending_super_native
                        && let Some(raw) = self.this_val.as_handle()
                    {
                        self.apply_native_super(nid, Handle::from_raw(raw), &args);
                        return Ok(NanBox::undefined());
                    }
                    return Err(ExecError::Unsupported(
                        "super outside a derived constructor",
                    ));
                }
                // `super.method(args)` — invoke the base-class method with the
                // current `this`.
                if let Expr::Member {
                    object, property, ..
                } = &**callee
                    && matches!(&**object, Expr::Super(_))
                {
                    let (PropertyKey::Ident(name) | PropertyKey::Str(name)) = property else {
                        return Err(ExecError::Unsupported("computed super member"));
                    };
                    let args = self.eval_args(arguments)?;
                    let f = self.resolve_super_method(name)?;
                    return self.call_with_this(f, self.this_val, &args);
                }
                // A `recv.method(args)` call: try a built-in method on the
                // receiver before falling back to a property-valued function.
                if let Expr::Member {
                    object,
                    property,
                    optional,
                    ..
                } = &**callee
                {
                    let recv = self.eval(object)?;
                    if *optional && matches!(recv.unpack(), Unpacked::Undefined | Unpacked::Null) {
                        return Err(ExecError::OptShortCircuit);
                    }
                    let args = self.eval_args(arguments)?;
                    if let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                        && let Some(result) = self.call_method(recv, name, &args)?
                    {
                        return Ok(result);
                    }
                    // `obj[Symbol.iterator]()` → an iterator over the receiver.
                    if let PropertyKey::Computed(e) = property {
                        let key = self.eval(e)?;
                        let iter_sym = self.well_known_symbol("iterator");
                        if self.realm.strict_equals(key, iter_sym) {
                            // A generator/iterator is its own iterator (identity).
                            if recv
                                .as_handle()
                                .map(Handle::from_raw)
                                .is_some_and(|h| self.realm.get_property(h, GEN_BUF).is_some())
                            {
                                return Ok(recv);
                            }
                            let vals = self.iterate_values(recv)?;
                            return Ok(self.make_generator(vals));
                        }
                    }
                    // Not a built-in method: read the member and call it.
                    let Some(raw) = recv.as_handle() else {
                        if *call_optional {
                            return Err(ExecError::OptShortCircuit);
                        }
                        return Err(ExecError::NotCallable);
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    // `f?.()` short-circuits when `f` is nullish.
                    if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                    {
                        return Err(ExecError::OptShortCircuit);
                    }
                    // Method call: `this` is the receiver.
                    return self.call_with_this(f, recv, &args);
                }
                let f = self.eval(callee)?;
                if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Err(ExecError::OptShortCircuit);
                }
                let args = self.eval_args(arguments)?;
                self.call(f, &args)
            }
            // The optional-chain boundary: a `?.` short-circuit inside becomes
            // `undefined` here (the rest of the chain was skipped).
            Expr::OptChain { expr, .. } => match self.eval(expr) {
                Err(ExecError::OptShortCircuit) => Ok(NanBox::undefined()),
                other => other,
            },
            Expr::New {
                callee, arguments, ..
            } => {
                let f = self.eval(callee)?;
                let args = self.eval_args(arguments)?;
                self.construct(f, &args)
            }
            Expr::Array { elements, .. } => {
                let mut items = Vec::new();
                for el in elements {
                    match el {
                        ArrayElement::Hole => items.push(NanBox::undefined()),
                        ArrayElement::Item(e) => items.push(self.eval(e)?),
                        ArrayElement::Spread(e) => {
                            let v = self.eval(e)?;
                            items.extend(self.iterate_values(v)?);
                        }
                    }
                }
                let h = self.realm.new_array(items);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Object { members, .. } => {
                let handle = self.realm.new_object();
                for m in members {
                    match m {
                        ObjectMember::Property {
                            key,
                            value,
                            shorthand,
                            ..
                        } => {
                            // `{ __proto__: obj }` — only the *unquoted identifier*
                            // form (not `"__proto__":`, computed, shorthand, or a
                            // method) sets the prototype; a quoted/computed key makes
                            // an ordinary own `__proto__` data property.
                            if !shorthand
                                && !matches!(&**value, Expr::Function(_))
                                && let PropertyKey::Ident(s) = key
                                && &**s == "__proto__"
                            {
                                let v = self.eval(value)?;
                                match v.unpack() {
                                    Unpacked::Null => {
                                        self.realm.set_object_proto(handle, None);
                                    }
                                    _ => {
                                        if let Some(p) = v.as_handle().map(Handle::from_raw) {
                                            self.realm.set_object_proto(handle, Some(p));
                                        }
                                    }
                                }
                                continue;
                            }
                            let k = self.eval_prop_key(key)?;
                            let v = self.eval(value)?;
                            // A method / function-valued property is named after its
                            // (static) key when otherwise anonymous.
                            if matches!(&**value, Expr::Function(_) | Expr::Arrow(_))
                                && let PropertyKey::Ident(s) | PropertyKey::Str(s) = key
                            {
                                self.set_fn_name(v, s);
                            }
                            // A concise method (`{ m() {} }`, not an arrow) records
                            // this object as its `[[HomeObject]]`, so `super.x`
                            // inside it resolves through the object's prototype.
                            if matches!(&**value, Expr::Function(_))
                                && let Some(fv) = v.as_handle().map(Handle::from_raw)
                            {
                                self.realm.set_hidden_property(
                                    fv,
                                    HOME_OBJECT,
                                    NanBox::handle(handle.to_raw()),
                                );
                            }
                            self.realm.set_property(handle, &k, v);
                        }
                        // `{ ...src }` — copy own enumerable properties.
                        ObjectMember::Spread { value, .. } => {
                            let src = self.eval(value)?;
                            if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                                // Spreading an array/string copies its indexed
                                // elements as `"0"`, `"1"`, … properties.
                                if let Some(elems) =
                                    self.realm.array_elements(sh).map(<[_]>::to_vec)
                                {
                                    for (i, e) in elems.iter().enumerate() {
                                        self.realm.set_property(handle, &alloc::format!("{i}"), *e);
                                    }
                                } else if let Some(s) = self.realm.string_value(sh) {
                                    for (i, c) in s.chars().enumerate() {
                                        let cv = self.new_str(&alloc::string::String::from(c));
                                        self.realm.set_property(handle, &alloc::format!("{i}"), cv);
                                    }
                                } else {
                                    // Own enumerable string + symbol keys (and
                                    // accessor getters); the raw key preserves
                                    // symbol identity.
                                    let keys = self.realm.object_keys_with_symbols(sh);
                                    for key in keys {
                                        // `read_member` invokes a getter where present.
                                        let pv = self.read_member(sh, &key)?;
                                        self.realm.set_property(handle, &key, pv);
                                    }
                                }
                            }
                        }
                        // `{ get x() {} }` / `{ set x(v) {} }`.
                        ObjectMember::Accessor {
                            key,
                            is_getter,
                            value,
                            ..
                        } => {
                            let k = self.eval_prop_key(key)?;
                            let f = self.make_function(
                                &value.params,
                                Body::Block(&value.body),
                                false,
                                false,
                            );
                            // An object-literal accessor's `[[HomeObject]]` is this
                            // object, so `super.x` inside it resolves via the proto.
                            if let Some(fh) = f.as_handle().map(Handle::from_raw) {
                                self.realm.set_hidden_property(
                                    fh,
                                    HOME_OBJECT,
                                    NanBox::handle(handle.to_raw()),
                                );
                            }
                            if *is_getter {
                                self.realm
                                    .define_accessor(handle, &k, f, NanBox::undefined());
                            } else {
                                self.realm
                                    .define_accessor(handle, &k, NanBox::undefined(), f);
                            }
                        }
                    }
                }
                Ok(NanBox::handle(handle.to_raw()))
            }
            Expr::Member {
                object,
                property,
                optional,
                ..
            } => {
                // `super.name` reads a super getter/method (not via `this`).
                if matches!(&**object, Expr::Super(_)) {
                    let (PropertyKey::Ident(name) | PropertyKey::Str(name)) = property else {
                        return Err(ExecError::Unsupported("computed super member"));
                    };
                    return self.resolve_super_member(name);
                }
                let obj = self.eval(object)?;
                if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    if *optional {
                        // Short-circuit the rest of the enclosing optional chain.
                        return Err(ExecError::OptShortCircuit);
                    }
                    // `null.x` / `undefined.x` throws a catchable TypeError.
                    let msg = self.new_str("cannot read property of null or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
                }
                let Some(raw) = obj.as_handle() else {
                    // A number/boolean primitive reports its wrapper constructor
                    // (`(5).constructor === Number`); other reads are `undefined`
                    // here (method calls go through the call path).
                    if let PropertyKey::Ident(n) | PropertyKey::Str(n) = property
                        && n.as_ref() == "constructor"
                    {
                        let name = if obj.as_number().is_some() {
                            "Number"
                        } else if matches!(obj.unpack(), Unpacked::Bool(_)) {
                            "Boolean"
                        } else {
                            return Ok(NanBox::undefined());
                        };
                        return Ok(self.current.get(name).unwrap_or(NanBox::undefined()));
                    }
                    return Ok(NanBox::undefined());
                };
                let handle = crate::heap::Handle::from_raw(raw);
                self.member(handle, property)
            }
            _ => Err(ExecError::Unsupported("expression")),
        }
    }

    fn eval_fn_expr(&mut self, func: &'a Function) -> NanBox {
        // A named function expression binds its own name in an intermediate scope
        // that the closure captures, so the body can recurse by that name.
        if let Some(id) = &func.id {
            let inner = self.current.child();
            let saved = core::mem::replace(&mut self.current, inner);
            let f = self.make_function(
                &func.params,
                Body::Block(&func.body),
                func.is_async,
                func.is_generator,
            );
            self.set_fn_name(f, &id.name);
            self.current.declare(&id.name, f);
            self.current = saved;
            return f;
        }
        self.make_function(
            &func.params,
            Body::Block(&func.body),
            func.is_async,
            func.is_generator,
        )
    }

    fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        let f = self.make_function(&arrow.params, body, arrow.is_async, false);
        // Arrows have no own `arguments` binding (they inherit the enclosing one).
        if let Some(raw) = f.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
        {
            self.functions[func_id as usize].is_arrow = true;
        }
        f
    }

    /// Records a function value's name (`fn.name`).
    fn set_fn_name(&mut self, value: NanBox, name: &'a str) {
        if let Some(raw) = value.as_handle()
            && let Some((func_id, _)) = self.realm.function_at(Handle::from_raw(raw))
            // Don't clobber a name the function already has (a named function
            // expression keeps its own name over the binding/key name).
            && self.functions[func_id as usize].name.is_empty()
        {
            self.functions[func_id as usize].name = name;
        }
    }

    fn member(
        &mut self,
        handle: crate::heap::Handle,
        key: &'a PropertyKey,
    ) -> Result<NanBox, ExecError> {
        match key {
            PropertyKey::Number(n) if as_index(*n).is_some() && self.realm.is_array(handle) => {
                Ok(self.realm.get_element(handle, as_index(*n).unwrap()))
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index)
                    && self.realm.is_array(handle)
                {
                    return Ok(self.realm.get_element(handle, i));
                }
                let name = self.coerce_property_key(k)?;
                self.read_member(handle, &name)
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => self.read_member(handle, s),
            PropertyKey::Number(n) => self.read_member(handle, &alloc::format!("{n}")),
            // Private names (`this.#x`) are stored under a `#`-prefixed key.
            PropertyKey::Private(s) => self.read_member(handle, &alloc::format!("#{s}")),
        }
    }

    /// Reads a member by an already-evaluated key value (an array index when the
    /// key is a numeric index and the receiver is an array, else a named read).
    fn read_member_value(
        &mut self,
        handle: crate::heap::Handle,
        key: NanBox,
    ) -> Result<NanBox, ExecError> {
        if let Some(i) = key.as_number().and_then(as_index)
            && self.realm.is_array(handle)
        {
            return Ok(self.realm.get_element(handle, i));
        }
        let name = self.member_key(key);
        self.read_member(handle, &name)
    }

    /// Assigns a member by an already-evaluated key value (used when the target's
    /// computed key must be resolved before the RHS, per spec evaluation order).
    /// Mirrors `assign_member`'s proxy / array-index / setter / length handling.
    fn assign_member_value(
        &mut self,
        handle: crate::heap::Handle,
        key: NanBox,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // Proxy `set` trap (or forward to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "set")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let name = self.member_key(key);
                let key_box = self.new_str(&name);
                let recv = NanBox::handle(handle.to_raw());
                let r = self.call(trap, &[NanBox::handle(target.to_raw()), key_box, new, recv])?;
                // A `set` trap returning a falsy value is a failed [[Set]]: a strict-mode
                // assignment then throws a TypeError (sloppy mode fails silently).
                if self.strict && !self.realm.truthy(r) {
                    let m = self.new_str(&alloc::format!(
                        "'set' on proxy: trap returned falsish for property '{name}'"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(());
            }
            return self.assign_member_value(target, key, new);
        }
        // A numeric index — a number, or a canonical numeric string ("1", not "01"
        // or "1.0") as produced by `Reflect.set`/`arr["1"]=` — addresses array storage.
        if self.realm.is_array(handle) {
            let idx = key.as_number().and_then(as_index).or_else(|| {
                key.as_handle()
                    .map(Handle::from_raw)
                    .and_then(|h| self.realm.string_value(h))
                    .and_then(|s| {
                        s.parse::<usize>()
                            .ok()
                            .filter(|i| alloc::format!("{i}") == s)
                    })
            });
            if let Some(i) = idx {
                self.set_element_coerced(handle, i, new);
                return Ok(());
            }
        }
        let name = self.coerce_property_key(key)?;
        // `regex.lastIndex = n` updates the RegExp's stateful search position.
        if name == "lastIndex" && self.realm.regexp_at(handle).is_some() {
            let n = self.realm.to_number(new).max(0.0) as usize;
            self.realm.set_regex_last_index(handle, n);
            return Ok(());
        }
        // An own accessor setter takes precedence.
        if let Some((_, setter)) = self.realm.accessor(handle, &name) {
            if !matches!(setter.unpack(), Unpacked::Undefined) {
                let this = NanBox::handle(handle.to_raw());
                self.call_with_this(setter, this, &[new])?;
            }
            return Ok(());
        }
        // No own property: an *inherited* accessor on the prototype chain handles the
        // write (its setter runs with `this` = the receiver). An inherited data
        // property, or none, falls through to creating an own data property.
        if !self.realm.has_own(handle, &name) {
            let mut cur = self.realm.object_proto(handle);
            while let Some(p) = cur {
                if let Some((_, setter)) = self.realm.accessor(p, &name) {
                    if !matches!(setter.unpack(), Unpacked::Undefined) {
                        let this = NanBox::handle(handle.to_raw());
                        self.call_with_this(setter, this, &[new])?;
                    }
                    return Ok(());
                }
                if self.realm.has_own(p, &name) {
                    break;
                }
                cur = self.realm.object_proto(p);
            }
        }
        // `arr.length = n` resizes the array.
        if name == "length" && self.realm.is_array(handle) {
            let n = self.realm.to_number(new).max(0.0) as usize;
            self.realm.set_array_length(handle, n);
        } else {
            self.realm.set_property(handle, &name, new);
        }
        Ok(())
    }

    /// Reads a named member, honoring class statics and accessor getters before
    /// ordinary property/length access.
    /// The global constructor a built-in heap value reports as its `.constructor`
    /// (so `[].constructor === Array`), resolved by the value's cell kind. Returns
    /// the actual global binding (identity-equal to `Array`, `Object`, …), or
    /// `None` for kinds without a distinct constructor.
    fn builtin_constructor_for(&mut self, handle: crate::heap::Handle) -> Option<NanBox> {
        let name = if self.realm.is_array(handle) {
            "Array"
        } else if self.realm.string_value(handle).is_some() {
            "String"
        } else if self.realm.regexp_at(handle).is_some() {
            "RegExp"
        } else if self.realm.bigint_at(handle).is_some() {
            "BigInt"
        } else if self.realm.symbol_at(handle).is_some() {
            "Symbol"
        } else if self.realm.date_at(handle).is_some() {
            "Date"
        } else if let Some(is_set) = self.realm.collection_is_set(handle) {
            if is_set { "Set" } else { "Map" }
        } else if self.realm.promise_state(handle).is_some() {
            "Promise"
        } else if self.realm.object_keys(handle).is_some() {
            // A plain object reports `Object`. (Error objects are handled earlier in
            // `read_member`, before their prototype's generic `constructor`.)
            "Object"
        } else {
            return None;
        };
        self.current.get(name)
    }

    fn read_member(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        // String index access (`"abc"[1]`) → the UTF-16 code unit at the index.
        if let Ok(i) = name.parse::<usize>()
            && let Some(s) = self.realm.string_value(handle)
        {
            return Ok(match s.encode_utf16().nth(i) {
                Some(u) => self.new_str(&String::from_utf16_lossy(&[u])),
                None => NanBox::undefined(),
            });
        }
        // A canonical numeric string key on an array (`arr["0"]`) reads the
        // element, exactly like `arr[0]`.
        if self.realm.is_array(handle)
            && let Ok(i) = name.parse::<usize>()
            && alloc::format!("{i}") == name
        {
            return Ok(self.realm.get_element(handle, i));
        }
        // Proxy `get` trap (or forward the read to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "get")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
            {
                let key = self.new_str(name);
                let recv = NanBox::handle(handle.to_raw());
                return self.call(trap, &[NanBox::handle(target.to_raw()), key, recv]);
            }
            return self.read_member(target, name);
        }
        // An error object's `.constructor` is its specific error global — its
        // prototype otherwise reports a generic `Object`. Recognized by an own
        // `name` in the error family plus a `message` (so a user `new Foo()`,
        // whose constructor resolves through its prototype, is never matched).
        if name == "constructor" {
            let nm = self
                .realm
                .get_property(handle, "name")
                .map(|v| self.realm.to_display_string(v))
                .unwrap_or_default();
            if ERROR_NAMES.contains(&nm.as_str())
                && self.realm.get_property(handle, "message").is_some()
                && let Some(ctor) = self.current.get(&nm)
            {
                return Ok(ctor);
            }
        }
        // Well-known `Symbol.iterator` / `Symbol.asyncIterator` (lazily created).
        if self.realm.native_at(handle) == Some(N_SYMBOL)
            && matches!(
                name,
                "iterator"
                    | "asyncIterator"
                    | "hasInstance"
                    | "toPrimitive"
                    | "toStringTag"
                    | "species"
                    | "isConcatSpreadable"
                    | "match"
                    | "matchAll"
                    | "replace"
                    | "search"
                    | "split"
                    | "unscopables"
            )
        {
            // The name is the well-known symbol's key.
            let key: &'static str = match name {
                "iterator" => "iterator",
                "asyncIterator" => "asyncIterator",
                "hasInstance" => "hasInstance",
                "toPrimitive" => "toPrimitive",
                "toStringTag" => "toStringTag",
                "species" => "species",
                "isConcatSpreadable" => "isConcatSpreadable",
                "match" => "match",
                "matchAll" => "matchAll",
                "replace" => "replace",
                "search" => "search",
                "split" => "split",
                _ => "unscopables",
            };
            return Ok(self.well_known_symbol(key));
        }
        // A symbol's `description` (`undefined` for a no-argument `Symbol()`).
        if let Some((desc, _)) = self.realm.symbol_at(handle)
            && name == "description"
        {
            return Ok(if &*desc == SYMBOL_NO_DESC {
                NanBox::undefined()
            } else {
                self.new_str(&desc)
            });
        }
        // A constructor function's `.prototype` (lazily created), so
        // `Fn.prototype.method = …` and prototype-chain inheritance work.
        if name == "prototype"
            && let Some((func_id, _)) = self.realm.function_at(handle)
        {
            let proto = self.realm.function_prototype(func_id);
            return Ok(NanBox::handle(proto.to_raw()));
        }
        // A bound function's `name` is `"bound " + target.name` (recursing so a
        // re-bound function reads `"bound bound …"`); its `length` is the target's
        // length minus the bound arguments (floored at 0).
        if matches!(name, "name" | "length")
            && let Some(target) = self.realm.get_property(handle, BOUND_TARGET)
        {
            let th = target.as_handle().map(Handle::from_raw);
            if name == "name" {
                let tname = match th {
                    Some(t) => {
                        let v = self.read_member(t, "name")?;
                        self.realm.to_display_string(v)
                    }
                    None => String::new(),
                };
                return Ok(self.new_str(&alloc::format!("bound {tname}")));
            }
            // `length`: target.length − number of pre-bound arguments.
            let tlen = match th {
                Some(t) => {
                    let v = self.read_member(t, "length")?;
                    self.realm.to_number(v)
                }
                None => 0.0,
            };
            let bound = self
                .realm
                .get_property(handle, BOUND_ARGS)
                .and_then(|a| a.as_handle().map(Handle::from_raw))
                .and_then(|bh| self.realm.array_length(bh))
                .unwrap_or(0);
            return Ok(NanBox::number((tlen - bound as f64).max(0.0)));
        }
        // `obj.__proto__` reads the prototype link (unless shadowed by an own
        // data property of that name).
        if name == "__proto__" && !self.realm.has_own(handle, "__proto__") {
            return Ok(match self.realm.object_proto(handle) {
                Some(p) => NanBox::handle(p.to_raw()),
                None => NanBox::null(),
            });
        }
        // A class's `name` is its declared identifier (`class C {}` → `"C"`).
        if name == "name"
            && let Some((cid, _)) = self.realm.class_at(handle)
        {
            let cname = self.classes[cid as usize]
                .id
                .as_ref()
                .map_or("", |i| &i.name);
            return Ok(self.new_str(cname));
        }
        // A function's `length` (params before a default/rest) and `name`.
        if matches!(name, "length" | "name")
            && !self.realm.has_own(handle, name)
            && let Some((func_id, _)) = self.realm.function_at(handle)
        {
            let def = self.functions[func_id as usize];
            return Ok(if name == "length" {
                let len = def
                    .params
                    .iter()
                    .take_while(|p| p.default.is_none() && !p.rest)
                    .count();
                NanBox::number(len as f64)
            } else {
                self.new_str(def.name)
            });
        }
        // `Number.*` static constants.
        if self.realm.native_at(handle) == Some(N_NUMBER) {
            match name {
                "MAX_SAFE_INTEGER" => return Ok(NanBox::number(9_007_199_254_740_991.0)),
                "MIN_SAFE_INTEGER" => return Ok(NanBox::number(-9_007_199_254_740_991.0)),
                "MAX_VALUE" => return Ok(NanBox::number(f64::MAX)),
                // The smallest positive value is the least *subnormal* (5e-324),
                // not Rust's `MIN_POSITIVE` (the smallest *normal*, 2.2e-308).
                "MIN_VALUE" => return Ok(NanBox::number(f64::from_bits(1))),
                "EPSILON" => return Ok(NanBox::number(f64::EPSILON)),
                "POSITIVE_INFINITY" => return Ok(NanBox::number(f64::INFINITY)),
                "NEGATIVE_INFINITY" => return Ok(NanBox::number(f64::NEG_INFINITY)),
                "NaN" => return Ok(NanBox::number(f64::NAN)),
                _ => {}
            }
        }
        // A class static — walking the `extends` chain for inherited statics.
        if let Some((cid, _)) = self.realm.class_at(handle) {
            let mut cur = Some(cid);
            while let Some(c) = cur {
                if let Some(v) = self.class_statics[c as usize].get(name) {
                    return Ok(*v);
                }
                // A static getter is called with `this` = the class.
                if let Some(getter) = self.class_static_get[c as usize].get(name).copied() {
                    let this = NanBox::handle(handle.to_raw());
                    return self.call_with_this(getter, this, &[]);
                }
                let class = self.classes[c as usize];
                let env = self.class_envs[c as usize].clone();
                cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
            }
        }
        if let Some((getter, _)) = self.realm.accessor(handle, name) {
            if matches!(getter.unpack(), Unpacked::Undefined) {
                return Ok(NanBox::undefined());
            }
            let this = NanBox::handle(handle.to_raw());
            return self.call_with_this(getter, this, &[]);
        }
        // RegExp introspection properties. (`constructor` falls through to the
        // built-in-constructor fallback below.)
        if name != "constructor"
            && let Some((source, flags)) = self.realm.regexp_at(handle)
        {
            return Ok(match name {
                "source" => self.new_str(&source),
                "flags" => self.new_str(&flags),
                "global" => NanBox::boolean(flags.contains('g')),
                "ignoreCase" => NanBox::boolean(flags.contains('i')),
                "multiline" => NanBox::boolean(flags.contains('m')),
                "sticky" => NanBox::boolean(flags.contains('y')),
                "dotAll" => NanBox::boolean(flags.contains('s')),
                "unicode" => NanBox::boolean(flags.contains('u')),
                "hasIndices" => NanBox::boolean(flags.contains('d')),
                "lastIndex" => NanBox::number(self.realm.regex_last_index(handle) as f64),
                _ => self.member_value(handle, name),
            });
        }
        // `ArrayBuffer.byteLength` (the byte store's length).
        if name == "byteLength"
            && let Some(b) = self.realm.get_property(handle, ARRAY_BUFFER_BYTES)
            && let Some(bh) = b.as_handle().map(Handle::from_raw)
        {
            return Ok(NanBox::number(
                self.realm.array_length(bh).unwrap_or(0) as f64
            ));
        }
        // `DataView.byteLength` / `.buffer` / `.byteOffset`.
        if matches!(name, "byteLength" | "buffer" | "byteOffset")
            && let Some(buf) = self.realm.get_property(handle, DATA_VIEW_BUF)
        {
            return Ok(match name {
                "buffer" => buf,
                "byteOffset" => self
                    .realm
                    .get_property(handle, DATA_VIEW_OFF)
                    .unwrap_or(NanBox::number(0.0)),
                _ => {
                    // An explicit byteLength wins; else the rest of the buffer.
                    if let Some(len) = self
                        .realm
                        .get_property(handle, DATA_VIEW_LEN)
                        .and_then(|n| n.as_number())
                    {
                        return Ok(NanBox::number(len));
                    }
                    let total = buf
                        .as_handle()
                        .map(Handle::from_raw)
                        .and_then(|h| self.realm.get_property(h, ARRAY_BUFFER_BYTES))
                        .and_then(|b| b.as_handle().map(Handle::from_raw))
                        .and_then(|bh| self.realm.array_length(bh))
                        .unwrap_or(0);
                    let off = self
                        .realm
                        .get_property(handle, DATA_VIEW_OFF)
                        .and_then(|n| n.as_number())
                        .unwrap_or(0.0) as usize;
                    NanBox::number(total.saturating_sub(off) as f64)
                }
            });
        }
        // Static `<TypedArray>.BYTES_PER_ELEMENT` (on the constructor itself).
        if name == "BYTES_PER_ELEMENT"
            && let Some(id) = self.realm.native_at(handle)
            && (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
        {
            return Ok(NanBox::number(f64::from(
                TYPED_ARRAY_KINDS[(id - N_TYPED_ARRAY_BASE) as usize].1,
            )));
        }
        // Typed-array introspection (`byteLength`, `BYTES_PER_ELEMENT`).
        if matches!(name, "byteLength" | "BYTES_PER_ELEMENT" | "byteOffset")
            && let Some(k) = self.realm.get_property(handle, TYPED_ARRAY_KIND)
        {
            let bpe = f64::from(TYPED_ARRAY_KINDS[k.as_number().unwrap_or(0.0) as usize].1);
            return Ok(NanBox::number(match name {
                "BYTES_PER_ELEMENT" => bpe,
                "byteOffset" => 0.0,
                _ => self.realm.array_length(handle).unwrap_or(0) as f64 * bpe,
            }));
        }
        // A String wrapper delegates `length` and indexed reads to its boxed
        // string (`new String("hi").length`, `wrapper[0]`).
        if let Some(prim) = self.realm.get_property(handle, PRIM_WRAP)
            && let Some(ph) = prim.as_handle().map(Handle::from_raw)
            && let Some(s) = self.realm.string_value(ph)
        {
            if name == "length" {
                return Ok(NanBox::number(s.encode_utf16().count() as f64));
            }
            if let Ok(i) = name.parse::<usize>() {
                let ch = s.chars().nth(i);
                return Ok(match ch {
                    Some(c) => self.new_str(c.encode_utf8(&mut [0u8; 4])),
                    None => NanBox::undefined(),
                });
            }
            let v = self.member_value(ph, name);
            if !matches!(v.unpack(), Unpacked::Undefined) {
                return Ok(v);
            }
        }
        // Own property (or a built-in like `length`) wins.
        let direct = self.member_value(handle, name);
        if !matches!(direct.unpack(), Unpacked::Undefined) || self.realm.has_own(handle, name) {
            return Ok(direct);
        }
        // Otherwise walk the `[[Prototype]]` chain for an inherited property or
        // accessor (the receiver stays `handle`).
        let mut cur = self.realm.object_proto(handle);
        while let Some(p) = cur {
            if let Some((getter, _)) = self.realm.accessor(p, name) {
                if matches!(getter.unpack(), Unpacked::Undefined) {
                    return Ok(NanBox::undefined());
                }
                let this = NanBox::handle(handle.to_raw());
                return self.call_with_this(getter, this, &[]);
            }
            if self.realm.has_own(p, name) {
                return Ok(self
                    .realm
                    .get_property(p, name)
                    .unwrap_or(NanBox::undefined()));
            }
            cur = self.realm.object_proto(p);
        }
        // A built-in value with no own/inherited `constructor` reports its global
        // constructor (`[].constructor === Array`); user functions/classes resolve
        // theirs through the prototype walk above and never reach here.
        if name == "constructor"
            && let Some(ctor) = self.builtin_constructor_for(handle)
        {
            return Ok(ctor);
        }
        // A built-in array/string/function exposes its prototype's methods as
        // first-class values — so feature detection (`if (arr.flat)`,
        // `typeof str.padStart`) and detached-method access resolve. (Ordinary
        // `recv.m(args)` calls dispatch via `call_method` and never reach here.)
        if let Some(m) = self.builtin_proto_method(handle, name) {
            return Ok(m);
        }
        Ok(direct)
    }

    /// For a built-in array/string/function value, the first-class method `name`
    /// from its constructor's prototype (`Array.prototype` etc.), or `None`.
    fn builtin_proto_method(&mut self, handle: Handle, name: &str) -> Option<NanBox> {
        let ctor_name = if self.realm.string_value(handle).is_some() {
            "String"
        } else if self.realm.is_array(handle) {
            "Array"
        } else if let Some(is_set) = self.realm.collection_is_set(handle) {
            if is_set { "Set" } else { "Map" }
        } else if self.realm.function_at(handle).is_some()
            || self.realm.native_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
        {
            "Function"
        } else {
            return None;
        };
        let proto = self
            .current
            .get(ctor_name)
            .and_then(|v| v.as_handle())
            .map(Handle::from_raw)
            .and_then(|ns| self.realm.get_property(ns, "prototype"))
            .and_then(|p| p.as_handle())
            .map(Handle::from_raw)?;
        let m = self.realm.get_property(proto, name)?;
        (!matches!(m.unpack(), Unpacked::Undefined)).then_some(m)
    }

    fn member_value(&self, handle: crate::heap::Handle, key: &str) -> NanBox {
        if let Some(v) = self.realm.get_property(handle, key) {
            return v;
        }
        if key == "length" {
            if let Some(len) = self.realm.array_length(handle) {
                return NanBox::number(len as f64);
            }
            if let Some(s) = self.realm.string_value(handle) {
                // `String.length` counts UTF-16 code units (astral chars = 2).
                return NanBox::number(s.encode_utf16().count() as f64);
            }
        }
        // `Map`/`Set` expose `size`.
        if key == "size"
            && let Some(n) = self.realm.collection_size(handle)
        {
            return NanBox::number(n as f64);
        }
        NanBox::undefined()
    }

    fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'a Expr,
        value: &'a Expr,
    ) -> Result<NanBox, ExecError> {
        // Logical assignment (`&&=`/`||=`/`??=`) short-circuits: the right side
        // is evaluated and stored only when the current value warrants it.
        if matches!(
            op,
            AssignOp::AndAssign | AssignOp::OrAssign | AssignOp::NullishAssign
        ) {
            let current = self.read_target(target)?;
            let assign = match op {
                AssignOp::AndAssign => self.realm.truthy(current),
                AssignOp::OrAssign => !self.realm.truthy(current),
                _ => matches!(current.unpack(), Unpacked::Undefined | Unpacked::Null),
            };
            if !assign {
                return Ok(current);
            }
            let rhs = self.eval(value)?;
            self.assign_to(target, rhs)?;
            return Ok(rhs);
        }
        // A computed-member target evaluates the object and key *before* the RHS
        // (spec order): `arr[i] = i = 1` writes the original `arr[i]`.
        if let Expr::Member {
            object,
            property: PropertyKey::Computed(key_expr),
            ..
        } = target
        {
            let obj = self.eval(object)?;
            let Some(raw) = obj.as_handle() else {
                return Err(ExecError::Unsupported("member assign to non-object"));
            };
            let handle = crate::heap::Handle::from_raw(raw);
            let key = self.eval(key_expr)?;
            let new = if op == AssignOp::Assign {
                self.eval(value)?
            } else {
                let current = self.read_member_value(handle, key)?;
                let rhs = self.eval(value)?;
                self.binary(compound_op(op)?, current, rhs)?
            };
            self.assign_member_value(handle, key, new)?;
            return Ok(new);
        }
        let rhs = self.eval(value)?;
        // Destructuring assignment: `[a, b] = …` / `({ x } = …)`.
        if op == AssignOp::Assign && matches!(target, Expr::Array { .. } | Expr::Object { .. }) {
            self.assign_destructure(target, rhs)?;
            return Ok(rhs);
        }
        match target {
            Expr::Ident(id) => {
                let name = &*id.name;
                // Reassigning a `const` binding is a TypeError.
                if self.current.is_const(name) {
                    let m = self.new_str("Assignment to constant variable.");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self
                        .current
                        .get(name)
                        .ok_or_else(|| ExecError::NotDefined(String::from(name)))?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                if !self.current.set(name, new) {
                    if self.strict {
                        let m = self.new_str(&alloc::format!("{name} is not defined"));
                        return Err(ExecError::Throw(
                            self.make_error(N_REFERENCE_ERROR, Some(m)),
                        ));
                    }
                    self.current.declare(name, new); // sloppy global
                }
                Ok(new)
            }
            Expr::Member {
                object, property, ..
            } if matches!(&**object, Expr::Super(_)) => {
                // `super.x = v` (and `super.x op= v`) invokes the inherited setter with
                // the current `this`; a compound op reads through `super.x` first.
                let name = self.eval_prop_key(property)?;
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self.resolve_super_member(&name)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.assign_super_member(&name, new)?;
                Ok(new)
            }
            Expr::Member {
                object, property, ..
            } => {
                let obj = self.eval(object)?;
                let Some(raw) = obj.as_handle() else {
                    return Err(ExecError::Unsupported("member assign to non-object"));
                };
                let handle = crate::heap::Handle::from_raw(raw);
                let new = if op == AssignOp::Assign {
                    rhs
                } else {
                    let current = self.member(handle, property)?;
                    self.binary(compound_op(op)?, current, rhs)?
                };
                self.assign_member(handle, property, new)?;
                Ok(new)
            }
            _ => Err(ExecError::Unsupported("assignment target")),
        }
    }

    /// Decides whether a data-property write may proceed. A write to a
    /// non-writable property (its own `writable: false`, or any property of a
    /// frozen object) is a `TypeError` in strict mode and silently ignored
    /// otherwise. Returns `true` when the caller should perform the write.
    fn allow_property_write(
        &mut self,
        handle: crate::heap::Handle,
        key: &str,
    ) -> Result<bool, ExecError> {
        let readonly = self.realm.property_is_readonly(handle, key)
            || (self.realm.is_frozen(handle) && self.realm.get_property(handle, key).is_some());
        if readonly {
            if self.strict {
                let m = self.new_str(&alloc::format!(
                    "Cannot assign to read only property '{key}'"
                ));
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
            return Ok(false); // sloppy mode: the write is silently dropped
        }
        Ok(true)
    }

    fn assign_member(
        &mut self,
        handle: crate::heap::Handle,
        property: &'a PropertyKey,
        new: NanBox,
    ) -> Result<(), ExecError> {
        // `regex.lastIndex = n` updates the RegExp's stateful search position.
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && &**s == "lastIndex"
            && self.realm.regexp_at(handle).is_some()
        {
            let n = self.realm.to_number(new).max(0.0) as usize;
            self.realm.set_regex_last_index(handle, n);
            return Ok(());
        }
        // `obj.__proto__ = proto` updates the prototype link (like
        // `Object.setPrototypeOf`); a non-object, non-null value is ignored.
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && &**s == "__proto__"
        {
            match new.unpack() {
                Unpacked::Null => {
                    self.realm.set_object_proto(handle, None);
                }
                _ => {
                    if let Some(p) = new.as_handle().map(Handle::from_raw) {
                        self.realm.set_object_proto(handle, Some(p));
                    }
                }
            }
            return Ok(());
        }
        // Writing a static on a class (`C.field = v`, `++C.field`): call a static
        // setter if one is defined, else update the class's static table.
        if let Some((cid, _)) = self.realm.class_at(handle) {
            let key = self.eval_prop_key(property)?;
            if let Some(setter) = self.class_static_set[cid as usize].get(&key).copied() {
                let this = NanBox::handle(handle.to_raw());
                self.call_with_this(setter, this, &[new])?;
            } else {
                self.class_statics[cid as usize].insert(key, new);
            }
            return Ok(());
        }
        // Proxy `set` trap (or forward the write to the target).
        if let Some((target, handler)) = self.realm.proxy_at(handle) {
            self.guard_revoked(handle)?;
            let trap = self
                .realm
                .get_property(handler, "set")
                .unwrap_or(NanBox::undefined());
            if trap
                .as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
            {
                let key = self.eval_prop_key(property)?;
                let key_box = self.new_str(&key);
                let recv = NanBox::handle(handle.to_raw());
                let r = self.call(trap, &[NanBox::handle(target.to_raw()), key_box, new, recv])?;
                // A `set` trap returning a falsy value is a failed [[Set]]: a strict-mode
                // assignment then throws a TypeError (sloppy mode fails silently).
                if self.strict && !self.realm.truthy(r) {
                    let m = self.new_str(&alloc::format!(
                        "'set' on proxy: trap returned falsish for property '{key}'"
                    ));
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                }
                return Ok(());
            }
            return self.assign_member(target, property, new);
        }
        // An accessor setter — own or inherited via the prototype chain — takes
        // precedence over creating a data property. A private accessor
        // (`set #x() {…}`) is stored under the `#`-prefixed key, so resolve that.
        let setter_key: Option<alloc::string::String> = match property {
            PropertyKey::Ident(s) | PropertyKey::Str(s) => Some(String::from(&**s)),
            PropertyKey::Private(s) => Some(alloc::format!("#{s}")),
            _ => None,
        };
        if let Some(skey) = setter_key {
            let mut cur = Some(handle);
            while let Some(c) = cur {
                if let Some((_, setter)) = self.realm.accessor(c, &skey) {
                    if !matches!(setter.unpack(), Unpacked::Undefined) {
                        let this = NanBox::handle(handle.to_raw());
                        self.call_with_this(setter, this, &[new])?;
                    } else if self.strict {
                        // Strict mode: writing a getter-only accessor is a TypeError.
                        let m = self.new_str(&alloc::format!(
                            "Cannot set property {skey} which has only a getter"
                        ));
                        return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
                    }
                    // A getter-only accessor still shadows a data assignment.
                    return Ok(());
                }
                // An own data property below shadows an inherited accessor.
                if self.realm.has_own(c, &skey) {
                    break;
                }
                cur = self.realm.object_proto(c);
            }
        }
        match property {
            PropertyKey::Number(n) if as_index(*n).is_some() && self.realm.is_array(handle) => {
                self.set_element_coerced(handle, as_index(*n).unwrap(), new);
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                // A numeric index only addresses array storage; on an object a
                // numeric key is the equivalent string property.
                if let Some(i) = k.as_number().and_then(as_index)
                    && self.realm.is_array(handle)
                {
                    self.set_element_coerced(handle, i, new);
                } else {
                    let name = self.coerce_property_key(k)?;
                    if self.allow_property_write(handle, &name)? {
                        self.realm.set_property(handle, &name, new);
                    }
                }
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                // `arr.length = n` resizes the array (truncate/pad), rather than
                // storing a `length` property.
                if &**s == "length" && self.realm.is_array(handle) {
                    let n = self.realm.to_number(new).max(0.0) as usize;
                    self.realm.set_array_length(handle, n);
                } else if &**s == "prototype"
                    && let Some((func_id, _)) = self.realm.function_at(handle)
                    && let Some(praw) = new.as_handle()
                {
                    // `Fn.prototype = obj` reassigns the constructor's prototype.
                    self.realm
                        .set_function_prototype(func_id, Handle::from_raw(praw));
                } else if self.allow_property_write(handle, s)? {
                    self.realm.set_property(handle, s, new);
                }
            }
            PropertyKey::Number(n) => {
                self.realm.set_property(handle, &alloc::format!("{n}"), new);
            }
            PropertyKey::Private(s) => {
                self.realm
                    .set_property(handle, &alloc::format!("#{s}"), new);
            }
        }
        Ok(())
    }

    fn unary(&mut self, op: UnaryOp, v: NanBox) -> Result<NanBox, ExecError> {
        // BigInt negation / bitwise-not stay BigInt.
        if let Some(big) = v
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)))
        {
            match op {
                UnaryOp::Minus => {
                    return Ok(NanBox::handle(self.realm.new_bigint(big.neg()).to_raw()));
                }
                UnaryOp::BitNot => {
                    // `~x` on a BigInt is `-(x + 1)`.
                    let one = crate::bignum::BigInt::from_i128(1);
                    let nx = big.add(&one).neg();
                    return Ok(NanBox::handle(self.realm.new_bigint(nx).to_raw()));
                }
                UnaryOp::Not => return Ok(NanBox::boolean(big.is_zero())),
                _ => {}
            }
        }
        // A Symbol cannot be converted to a number (unary `+`/`-`/`~`).
        if matches!(op, UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot)
            && v.as_handle()
                .map(Handle::from_raw)
                .is_some_and(|h| self.realm.symbol_at(h).is_some())
        {
            let m = self.new_str("Cannot convert a Symbol value to a number");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        Ok(match op {
            UnaryOp::Plus => {
                let p = self.coerce_object(v, "number")?;
                NanBox::number(self.realm.to_number(p))
            }
            UnaryOp::Minus => {
                let p = self.coerce_object(v, "number")?;
                self.realm.neg(p)
            }
            UnaryOp::Not => self.realm.logical_not(v),
            UnaryOp::Typeof => {
                let t = self.realm.type_of_value(v);
                NanBox::handle(self.realm.new_string(t).to_raw())
            }
            UnaryOp::Void => NanBox::undefined(),
            #[cfg(feature = "std")]
            UnaryOp::BitNot => {
                // ToPrimitive(Number) first, so `~obj` honors a user `valueOf`.
                let p = self.coerce_object(v, "number")?;
                self.realm.bit_not(p)
            }
            #[cfg(not(feature = "std"))]
            UnaryOp::BitNot => return Err(ExecError::Unsupported("~ needs std")),
            UnaryOp::Delete => return Err(ExecError::Unsupported("delete")),
        })
    }

    /// The BigInt operator path. Returns `None` to fall through (e.g. `bigint +
    /// string` is string concatenation). Both operands BigInt → i128 arithmetic;
    /// a mix with a Number throws a `TypeError` for arithmetic but compares
    /// numerically for `<`/`==`.
    fn bigint_binary(
        &mut self,
        op: BinaryOp,
        abig: Option<crate::bignum::BigInt>,
        bbig: Option<crate::bignum::BigInt>,
        a: NanBox,
        b: NanBox,
    ) -> Result<Option<NanBox>, ExecError> {
        // Strict equality: equal only if both are BigInt with the same value.
        match op {
            BinaryOp::EqEqEq => return Ok(Some(NanBox::boolean(abig.is_some() && abig == bbig))),
            BinaryOp::NotEqEq => {
                return Ok(Some(NanBox::boolean(!(abig.is_some() && abig == bbig))));
            }
            _ => {}
        }
        if let (Some(x), Some(y)) = (abig.clone(), bbig.clone()) {
            use core::cmp::Ordering;
            let val = |this: &mut Self, n: crate::bignum::BigInt| {
                NanBox::handle(this.realm.new_bigint(n).to_raw())
            };
            let throw = |this: &mut Self, msg: &str| {
                let m = this.new_str(msg);
                ExecError::Throw(this.make_error(N_TYPE_ERROR, Some(m)))
            };
            let r = match op {
                BinaryOp::Add => val(self, x.add(&y)),
                BinaryOp::Sub => val(self, x.sub(&y)),
                BinaryOp::Mul => val(self, x.mul(&y)),
                BinaryOp::Div => match x.divmod(&y) {
                    Some((q, _)) => val(self, q),
                    None => return Err(throw(self, "Division by zero")),
                },
                BinaryOp::Mod => match x.divmod(&y) {
                    Some((_, rem)) => val(self, rem),
                    None => return Err(throw(self, "Division by zero")),
                },
                BinaryOp::Exp => {
                    if y.is_negative() {
                        return Err(throw(self, "Exponent must be non-negative"));
                    }
                    let e = y.to_i128().and_then(|v| u64::try_from(v).ok()).unwrap_or(0);
                    val(self, x.pow(e))
                }
                // Two's-complement bitwise ops at arbitrary precision.
                BinaryOp::BitAnd => val(self, x.bitand(&y)),
                BinaryOp::BitOr => val(self, x.bitor(&y)),
                BinaryOp::BitXor => val(self, x.bitxor(&y)),
                // `<<`/`>>` as multiply/floor-divide by `2^n` (a negative shift
                // count reverses direction). BigInts have no unsigned `>>>`.
                BinaryOp::Shl | BinaryOp::Shr => {
                    let two = crate::bignum::BigInt::from_i128(2);
                    let count = y.to_i128().unwrap_or(0);
                    // `>>` is `<<` by the negated count, and vice versa.
                    let left = (op == BinaryOp::Shl) == (count >= 0);
                    let mag = u64::try_from(count.unsigned_abs()).unwrap_or(0);
                    let pow2 = two.pow(mag);
                    if left {
                        val(self, x.mul(&pow2))
                    } else {
                        match x.divmod(&pow2) {
                            // Arithmetic shift floors; truncating divmod needs a
                            // `-1` correction for a negative value with a remainder.
                            Some((q, rem)) => {
                                if x.is_negative() && !rem.is_zero() {
                                    val(self, q.sub(&crate::bignum::BigInt::from_i128(1)))
                                } else {
                                    val(self, q)
                                }
                            }
                            None => val(self, crate::bignum::BigInt::zero()),
                        }
                    }
                }
                BinaryOp::Ushr => {
                    return Err(throw(self, "BigInts have no unsigned right shift"));
                }
                BinaryOp::Lt => NanBox::boolean(x.cmp(&y) == Ordering::Less),
                BinaryOp::Gt => NanBox::boolean(x.cmp(&y) == Ordering::Greater),
                BinaryOp::LtEq => NanBox::boolean(x.cmp(&y) != Ordering::Greater),
                BinaryOp::GtEq => NanBox::boolean(x.cmp(&y) != Ordering::Less),
                BinaryOp::EqEq => NanBox::boolean(x == y),
                BinaryOp::NotEq => NanBox::boolean(x != y),
                _ => return Ok(None),
            };
            return Ok(Some(r));
        }
        // Mixed: `bigint + string` (either side a string) → string concat.
        if matches!(op, BinaryOp::Add) {
            let is_str = |this: &Self, v: NanBox| {
                v.as_handle()
                    .is_some_and(|raw| this.realm.string_value(Handle::from_raw(raw)).is_some())
            };
            if is_str(self, a) || is_str(self, b) {
                return Ok(None);
            }
        }
        // BigInt vs Number: compare numerically (`<`/`==` only).
        if matches!(
            op,
            BinaryOp::EqEq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
        ) {
            let to_f = |n: &crate::bignum::BigInt| n.to_i128().map_or(f64::NAN, |v| v as f64);
            let xn = abig.as_ref().map_or_else(|| self.realm.to_number(a), to_f);
            let yn = bbig.as_ref().map_or_else(|| self.realm.to_number(b), to_f);
            let r = match op {
                BinaryOp::EqEq => xn == yn,
                BinaryOp::NotEq => xn != yn,
                BinaryOp::Lt => xn < yn,
                BinaryOp::Gt => xn > yn,
                BinaryOp::LtEq => xn <= yn,
                _ => xn >= yn,
            };
            return Ok(Some(NanBox::boolean(r)));
        }
        // Mixed arithmetic is a TypeError.
        let m = self.new_str("Cannot mix BigInt and other types");
        Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))))
    }

    fn binary(&mut self, op: BinaryOp, a: NanBox, b: NanBox) -> Result<NanBox, ExecError> {
        // BigInt operands take a dedicated path (i128 arithmetic; mixing with
        // other numeric types throws, per the spec).
        let abig = a
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
        let bbig = b
            .as_handle()
            .and_then(|raw| self.realm.bigint_at(Handle::from_raw(raw)));
        if (abig.is_some() || bbig.is_some())
            && let Some(r) = self.bigint_binary(op, abig, bbig, a, b)?
        {
            return Ok(r);
        }
        // Arithmetic and relational operators apply ToPrimitive to object
        // operands (`valueOf`/`toString`); equality/`instanceof`/`in` do not.
        let coerces = matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Mul
                | BinaryOp::Div
                | BinaryOp::Mod
                | BinaryOp::Exp
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::LtEq
                | BinaryOp::GtEq
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Ushr
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
        );
        // `+` uses the "default" hint; the other numeric operators use "number".
        let hint = if matches!(op, BinaryOp::Add) {
            "default"
        } else {
            "number"
        };
        let (a, b) = if coerces && (a.as_handle().is_some() || b.as_handle().is_some()) {
            (
                self.coerce_primitive(a, hint)?,
                self.coerce_primitive(b, hint)?,
            )
        } else {
            (a, b)
        };
        // `==`/`!=` between an object/array and a number/string primitive coerces
        // the object side (arrays via their join; plain objects via ToPrimitive).
        let (a, b) = if matches!(op, BinaryOp::EqEq | BinaryOp::NotEq) {
            // True for a non-string heap value (object or array).
            let obj = |this: &Self, v: NanBox| {
                v.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| this.realm.string_value(h).is_none())
            };
            // True for a number, boolean, or string primitive — the operands
            // against which an object is converted with ToPrimitive (a boolean is
            // first coerced to a number per the `==` algorithm).
            let prim = |this: &Self, v: NanBox| {
                v.as_number().is_some()
                    || matches!(v.unpack(), crate::nanbox::Unpacked::Bool(_))
                    || v.as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|h| this.realm.string_value(h).is_some())
            };
            if obj(self, a) && prim(self, b) {
                (self.coerce_for_eq(a)?, b)
            } else if obj(self, b) && prim(self, a) {
                (a, self.coerce_for_eq(b)?)
            } else {
                (a, b)
            }
        } else {
            (a, b)
        };
        // A Symbol cannot be implicitly converted to a number or string, so any
        // arithmetic/relational operator on one throws a TypeError.
        if coerces {
            let is_sym = |this: &Self, v: NanBox| {
                v.as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| this.realm.symbol_at(h).is_some())
            };
            if is_sym(self, a) || is_sym(self, b) {
                let msg = if matches!(op, BinaryOp::Add) {
                    "Cannot convert a Symbol value to a string"
                } else {
                    "Cannot convert a Symbol value to a number"
                };
                let m = self.new_str(msg);
                return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
            }
        }
        Ok(match op {
            BinaryOp::Add => self.realm.add(a, b),
            BinaryOp::Sub => self.realm.sub(a, b),
            BinaryOp::Mul => self.realm.mul(a, b),
            BinaryOp::Div => self.realm.div(a, b),
            BinaryOp::Mod => self.realm.rem(a, b),
            BinaryOp::Lt => self.realm.less_than(a, b),
            BinaryOp::Gt => self.realm.greater_than(a, b),
            BinaryOp::LtEq => self.realm.less_equal(a, b),
            BinaryOp::GtEq => self.realm.greater_equal(a, b),
            BinaryOp::EqEq => NanBox::boolean(self.realm.loose_equals(a, b)),
            BinaryOp::NotEq => NanBox::boolean(!self.realm.loose_equals(a, b)),
            BinaryOp::EqEqEq => NanBox::boolean(self.realm.strict_equals(a, b)),
            BinaryOp::NotEqEq => NanBox::boolean(!self.realm.strict_equals(a, b)),
            #[cfg(feature = "std")]
            BinaryOp::Exp => self.realm.pow(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shl => self.realm.shl(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Shr => self.realm.shr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::Ushr => self.realm.ushr(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitAnd => self.realm.bit_and(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitOr => self.realm.bit_or(a, b),
            #[cfg(feature = "std")]
            BinaryOp::BitXor => self.realm.bit_xor(a, b),
            #[cfg(not(feature = "std"))]
            BinaryOp::Exp
            | BinaryOp::Shl
            | BinaryOp::Shr
            | BinaryOp::Ushr
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor => return Err(ExecError::Unsupported("** / bitwise need std")),
            BinaryOp::In => {
                let key = self.member_key(a);
                let present = match b.as_handle().map(Handle::from_raw) {
                    // Proxy `has` trap, or forward to the target.
                    Some(h) if self.realm.proxy_at(h).is_some() => {
                        let (target, handler) = self.realm.proxy_at(h).unwrap();
                        self.guard_revoked(h)?;
                        let trap = self
                            .realm
                            .get_property(handler, "has")
                            .unwrap_or(NanBox::undefined());
                        if trap
                            .as_handle()
                            .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
                        {
                            let kb = self.new_str(&key);
                            let r = self.call(trap, &[NanBox::handle(target.to_raw()), kb])?;
                            self.realm.truthy(r)
                        } else {
                            self.realm.has_own(target, &key) || self.realm.is_array(target)
                        }
                    }
                    Some(h) => {
                        // `key in obj` is true for an own *or inherited* property
                        // (walk the prototype chain); arrays also report in-bounds
                        // indices and `length`.
                        let in_chain = || {
                            let mut cur = Some(h);
                            while let Some(c) = cur {
                                if self.realm.has_own(c, &key) {
                                    return true;
                                }
                                cur = self.realm.object_proto(c);
                            }
                            false
                        };
                        if let Some(len) = self.realm.array_length(h) {
                            key == "length"
                                || key.parse::<usize>().is_ok_and(|i| i < len)
                                || in_chain()
                        } else {
                            in_chain()
                        }
                    }
                    None => false,
                };
                NanBox::boolean(present)
            }
            BinaryOp::Instanceof => NanBox::boolean(self.instance_of(a, b)?),
        })
    }

    /// Finds `name` as a method in the superclass chain of the currently-running
    /// method's home class, returning a callable bound to the base definition.
    fn resolve_super_method(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An object-literal method: `super.m()` is `HomeObject.[[Prototype]].m`,
        // called (by the caller) with the current `this`.
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            if let Some(proto) = self.realm.object_proto(home) {
                let f = self.read_member(proto, name)?;
                if f.as_handle()
                    .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
                {
                    return Ok(f);
                }
            }
            return Err(ExecError::Throw(
                self.new_str(&alloc::format!("super method {name} not found")),
            ));
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && m.kind == MethodKind::Method
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        false,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    return Ok(f);
                }
            }
            cur = self.resolve_super(class, &penv)?;
        }
        Err(ExecError::Throw(
            self.new_str(&alloc::format!("super method {name} not found")),
        ))
    }

    /// `super.name` as a value read: a super getter is invoked (with the current
    /// `this`); a super method is returned as a bound function.
    /// `super.name = value`: invoke an inherited setter (found on the home's parent
    /// chain) with `this` = the current receiver; if there is none, assign the property
    /// directly on the receiver.
    fn assign_super_member(&mut self, name: &str, value: NanBox) -> Result<(), ExecError> {
        // An object-literal method: `super.x = v` uses `HomeObject.[[Prototype]]`.
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            if let Some(proto) = self.realm.object_proto(home)
                && let Some((_, setter)) = self.realm.accessor(proto, name)
                && !matches!(setter.unpack(), Unpacked::Undefined)
            {
                self.call_with_this(setter, self.this_val, &[value])?;
                return Ok(());
            }
            if let Some(th) = self.this_val.as_handle().map(Handle::from_raw) {
                self.realm.set_property(th, name, value);
            }
            return Ok(());
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && m.kind == MethodKind::Set
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        false,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    self.call_with_this(f, self.this_val, &[value])?;
                    return Ok(());
                }
            }
            cur = self.resolve_super(class, &penv)?;
        }
        // No inherited setter — the write lands on the receiver (`this`).
        if let Some(th) = self.this_val.as_handle().map(Handle::from_raw) {
            self.realm.set_property(th, name, value);
        }
        Ok(())
    }

    fn resolve_super_member(&mut self, name: &str) -> Result<NanBox, ExecError> {
        // An object-literal method: `super.x` reads `HomeObject.[[Prototype]].x`
        // (a data property, or a getter — invoked through the proto here).
        if self.current_home.is_none()
            && let Some(home) = self.current_home_object
        {
            return match self.realm.object_proto(home) {
                Some(proto) => self.read_member(proto, name),
                None => Ok(NanBox::undefined()),
            };
        }
        let home = self
            .current_home
            .ok_or(ExecError::Unsupported("super outside a method"))?;
        let mut cur = self.resolve_super(
            self.classes[home as usize],
            &self.class_envs[home as usize].clone(),
        )?;
        while let Some((pid, penv)) = cur {
            let class = self.classes[pid as usize];
            for member in &class.body {
                if let ClassMember::Method(m) = member
                    && m.is_static == self.current_home_static
                    && matches!(m.kind, MethodKind::Method | MethodKind::Get)
                    && static_key(&m.key).ok().as_deref() == Some(name)
                {
                    let saved = core::mem::replace(&mut self.current, penv.clone());
                    let f = self.make_method(
                        &m.value.params,
                        Body::Block(&m.value.body),
                        false,
                        m.value.is_generator,
                        Some(pid),
                        self.current_home_static,
                    );
                    self.current = saved;
                    return if m.kind == MethodKind::Get {
                        self.call_with_this(f, self.this_val, &[])
                    } else {
                        Ok(f)
                    };
                }
            }
            cur = self.resolve_super(class, &penv)?;
        }
        Ok(NanBox::undefined())
    }

    /// `obj instanceof Ctor`: true when `obj` was constructed from `Ctor`'s
    /// class or one of its subclasses (via the instance's class tag and the
    /// `extends` chain).
    fn instance_of(&mut self, obj: NanBox, ctor: NanBox) -> Result<bool, ExecError> {
        // A custom `[Symbol.hasInstance]` on the right-hand side overrides the
        // ordinary prototype/cell-kind check (and applies even to a primitive
        // left-hand side, e.g. `4 instanceof Even`). Read via `read_member` so a
        // `static [Symbol.hasInstance]` on a class is found.
        if let Some(ch) = ctor.as_handle().map(Handle::from_raw) {
            let sym = self.well_known_symbol("hasInstance");
            let key = self.member_key(sym);
            let method = self.read_member(ch, &key)?;
            if method
                .as_handle()
                .is_some_and(|r| self.is_callable(Handle::from_raw(r)))
            {
                let result = self.call_with_this(method, ctor, &[obj])?;
                return Ok(self.realm.truthy(result));
            }
        }
        // The RHS must be a callable object (without a `[Symbol.hasInstance]`); a
        // primitive or a non-constructor object is a TypeError.
        let Some(ch) = ctor.as_handle().map(Handle::from_raw) else {
            let m = self.new_str("Right-hand side of 'instanceof' is not an object");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        };
        // A bound function tests `instanceof` against its target function.
        if let Some(target) = self.realm.get_property(ch, BOUND_TARGET) {
            return self.instance_of(obj, target);
        }
        let is_ctor = self.realm.native_at(ch).is_some()
            || self.realm.function_at(ch).is_some()
            || self.realm.class_at(ch).is_some()
            || self.realm.bound_native_at(ch).is_some()
            || self.current.get("Array").and_then(|v| v.as_handle()) == ctor.as_handle()
            || self.current.get("Object").and_then(|v| v.as_handle()) == ctor.as_handle();
        if !is_ctor {
            let m = self.new_str("Right-hand side of 'instanceof' is not callable");
            return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(m))));
        }
        // A primitive left-hand side is not an instance of anything.
        let Some(oh) = obj.as_handle().map(Handle::from_raw) else {
            return Ok(false);
        };
        // Built-in constructors: check the cell kind directly.
        if let Some(id) = self.realm.native_at(ch) {
            // A primitive wrapper (`new Number(…)`) matches its constructor.
            if let Some(wt) = self.realm.get_property(oh, PRIM_WRAP_TYPE)
                && wt.as_number() == Some(f64::from(id))
            {
                return Ok(true);
            }
            // A typed array matches its constructor.
            if (N_TYPED_ARRAY_BASE..N_TYPED_ARRAY_BASE + TYPED_ARRAY_KINDS.len() as u16)
                .contains(&id)
                && let Some(k) = self.realm.get_property(oh, TYPED_ARRAY_KIND)
                && k.as_number() == Some(f64::from(id - N_TYPED_ARRAY_BASE))
            {
                return Ok(true);
            }
            // The `WebAssembly.*` boundary objects match by their marker slot.
            let wasm_marker = match id {
                N_WASM_GLOBAL => Some(WASM_GLOBAL_VALUE),
                N_WASM_MEMORY => Some(WASM_MEM_BUFFER),
                N_WASM_TABLE => Some(WASM_TABLE_ELEMS),
                N_WASM_MODULE => Some(WASM_IS_MODULE),
                N_WASM_INSTANCE => Some(WASM_INSTANCE_ID),
                _ => None,
            };
            if let Some(slot) = wasm_marker
                && self.realm.get_property(oh, slot).is_some()
            {
                return Ok(true);
            }
            // The `Error` family: match by the object's `name` against the
            // constructor (the base `Error` matches any error object).
            if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
                let want = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
                // A user class extending a native error: walk its class chain for
                // a native error super (so `customErr instanceof Error` holds even
                // when the subclass overrides `this.name`).
                if let Some(tag) = self.realm.class_tag(oh) {
                    let mut cur = Some(tag);
                    while let Some(cid) = cur {
                        if let Some(nsup) = self.class_native_super[cid as usize]
                            && (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16)
                                .contains(&nsup)
                        {
                            let have = ERROR_NAMES[(nsup - N_ERROR_BASE) as usize];
                            if want == "Error" || want == have {
                                return Ok(true);
                            }
                        }
                        cur = self
                            .resolve_super(
                                self.classes[cid as usize],
                                &self.class_envs[cid as usize].clone(),
                            )?
                            .map(|(p, _)| p);
                    }
                }
                // Plain error objects: match by the `name` property.
                let obj_name = self
                    .realm
                    .get_property(oh, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if !ERROR_NAMES.contains(&obj_name.as_str()) {
                    return Ok(false);
                }
                return Ok(want == "Error" || obj_name == want);
            }
            return Ok(match id {
                N_REGEXP => self.realm.regexp_at(oh).is_some(),
                N_MAP | N_SET | N_WEAKMAP | N_WEAKSET => self.realm.collection_is_set(oh).is_some(),
                N_DATE => self.realm.date_at(oh).is_some(),
                N_PROMISE => self.realm.promise_state(oh).is_some(),
                // Every callable (function, native, bound) and every class is a
                // `Function`.
                N_FUNCTION => self.is_callable(oh) || self.realm.class_at(oh).is_some(),
                _ => false,
            });
        }
        // `Array`/`Object` are namespace objects (not natives), matched by the
        // identity of the global binding.
        if self.current.get("Array").and_then(|v| v.as_handle()) == ctor.as_handle() {
            return Ok(self.realm.is_array(oh));
        }
        if self.current.get("Object").and_then(|v| v.as_handle()) == ctor.as_handle() {
            // Any non-primitive heap value is an instance of `Object`.
            return Ok(self.realm.string_value(oh).is_none()
                && self.realm.symbol_at(oh).is_none()
                && self.realm.bigint_at(oh).is_none());
        }
        // Plain function constructors: walk the instance's `[[Prototype]]` chain for
        // the constructor's current `.prototype` (so `Object.create(C.prototype)` is an
        // instance, and reassigning `C.prototype` is reflected).
        if let Some((func_id, _)) = self.realm.function_at(ch) {
            let proto = self.realm.function_prototype(func_id);
            // Walk via `get_proto_of` so a proxy's `getPrototypeOf` trap is honored at
            // each step (bounded to guard against a trap returning a cycle).
            let mut cur = oh;
            for _ in 0..100_000 {
                let next = self.get_proto_of(cur)?;
                let Some(p) = next.as_handle().map(Handle::from_raw) else {
                    return Ok(false);
                };
                if p == proto {
                    return Ok(true);
                }
                cur = p;
            }
            return Ok(false);
        }
        let (Some(tag), Some((target_id, _))) = (self.realm.class_tag(oh), self.realm.class_at(ch))
        else {
            return Ok(false);
        };
        // Walk the instance's class chain (its class, then each `extends`).
        let mut cur = Some(tag);
        while let Some(cid) = cur {
            if cid == target_id {
                return Ok(true);
            }
            let class = self.classes[cid as usize];
            // Resolve the superclass in the class's own captured scope.
            let env = self.class_envs[cid as usize].clone();
            cur = self.resolve_super(class, &env)?.map(|(pid, _)| pid);
        }
        Ok(false)
    }
}

/// Collects `var`-declared identifier names in `stmts`, recursing through nested
/// statements but NOT into nested function bodies (which have their own scope) —
/// for hoisting `var` bindings to the enclosing function/program scope.
fn collect_var_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use crate::ast::VarDeclKind;
    fn from_decl<'a>(decl: &'a crate::ast::VarDecl, out: &mut Vec<&'a str>) {
        if matches!(decl.kind, VarDeclKind::Var) {
            for d in &decl.declarations {
                if let BindingTarget::Ident(id) = &d.target {
                    out.push(&id.name);
                }
            }
        }
    }
    for stmt in stmts {
        match stmt {
            Stmt::Var(decl) => from_decl(decl, out),
            Stmt::Block { body, .. } => collect_var_names(body, out),
            Stmt::If {
                consequent,
                alternate,
                ..
            } => {
                collect_var_names(core::slice::from_ref(consequent), out);
                if let Some(alt) = alternate {
                    collect_var_names(core::slice::from_ref(alt), out);
                }
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } | Stmt::Labeled { body, .. } => {
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::For { init, body, .. } => {
                if let Some(crate::ast::ForInit::Var(decl)) = init {
                    from_decl(decl, out);
                }
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::ForIn { left, body, .. } | Stmt::ForOf { left, body, .. } => {
                if let crate::ast::ForLeft::Decl {
                    kind: VarDeclKind::Var,
                    target: BindingTarget::Ident(id),
                    ..
                } = left
                {
                    out.push(&id.name);
                }
                collect_var_names(core::slice::from_ref(body), out);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    collect_var_names(&c.body, out);
                }
            }
            Stmt::Try {
                block,
                handler,
                finalizer,
                ..
            } => {
                collect_var_names(block, out);
                if let Some(h) = handler {
                    collect_var_names(&h.body, out);
                }
                if let Some(f) = finalizer {
                    collect_var_names(f, out);
                }
            }
            // `Stmt::Function` bodies have their own scope — not traversed.
            _ => {}
        }
    }
}

/// Whether a body's directive prologue contains `"use strict"` — a leading run
/// of string-literal expression statements, one of which is exactly `use strict`.
fn has_use_strict(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Expr { expression, .. } => match &**expression {
                Expr::Str { value, .. } => {
                    if &**value == "use strict" {
                        return true;
                    }
                }
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}

/// Collects the names of function declarations that appear **inside a block** (at
/// any nesting depth below the immediate statement list). Per Annex B, such a
/// name is var-hoisted to the enclosing function scope. The immediate top-level
/// functions are excluded — they are bound directly by the hoisting loop.
fn collect_block_function_names<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>) {
    use core::slice::from_ref;
    fn walk<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a str>, in_block: bool) {
        for stmt in stmts {
            match stmt {
                Stmt::Function(f) if in_block => {
                    if let Some(id) = &f.id {
                        out.push(&id.name);
                    }
                }
                Stmt::Block { body, .. } => walk(body, out, true),
                Stmt::If {
                    consequent,
                    alternate,
                    ..
                } => {
                    walk(from_ref(consequent), out, true);
                    if let Some(a) = alternate {
                        walk(from_ref(a), out, true);
                    }
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::Labeled { body, .. }
                | Stmt::For { body, .. }
                | Stmt::ForIn { body, .. }
                | Stmt::ForOf { body, .. } => walk(from_ref(body), out, true),
                Stmt::Try {
                    block, finalizer, ..
                } => {
                    walk(block, out, true);
                    if let Some(f) = finalizer {
                        walk(f, out, true);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    for case in cases {
                        walk(&case.body, out, true);
                    }
                }
                _ => {}
            }
        }
    }
    walk(stmts, out, false);
}

/// The binary operator underlying a compound assignment (`+=` → `+`).
fn compound_op(op: AssignOp) -> Result<BinaryOp, ExecError> {
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
        _ => return Err(ExecError::Unsupported("logical assignment")),
    })
}

/// Computes `[start, end)` char indices for `slice`, handling negative indices
/// (from the end) and an `undefined` end (to the length), clamped to `[0, len]`.
fn slice_bounds(start: f64, end_arg: NanBox, realm: &Realm, len: usize) -> (usize, usize) {
    let clamp = |n: f64| -> usize {
        if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            (n as usize).min(len)
        }
    };
    let a = clamp(start);
    let b = match end_arg.unpack() {
        Unpacked::Undefined => len,
        _ => clamp(realm.to_number(end_arg)),
    };
    (a, b.max(a))
}

/// `padStart`: left-pads `s` with `pad` (repeated/truncated) to `target` chars.
fn pad_start(s: &str, target: usize, pad: &str) -> String {
    let len = s.chars().count();
    if len >= target || pad.is_empty() {
        return String::from(s);
    }
    let need = target - len;
    let mut filler = String::new();
    while filler.chars().count() < need {
        filler.push_str(pad);
    }
    let filler: String = filler.chars().take(need).collect();
    filler + s
}

/// `Number.prototype.toPrecision(p)`: render `n` with `p` significant digits,
/// choosing fixed or exponential notation by magnitude (as the spec does).
fn format_precision(n: f64, p: usize) -> String {
    if n == 0.0 {
        return alloc::format!("{:.*}", p - 1, 0.0);
    }
    if !n.is_finite() {
        return if n.is_nan() {
            String::from("NaN")
        } else if n > 0.0 {
            String::from("Infinity")
        } else {
            String::from("-Infinity")
        };
    }
    // Derive the decimal exponent from the default scientific rendering (no
    // `log10`, which isn't available in the no_std float set).
    let sci = alloc::format!("{:e}", n.abs());
    let e: i32 = sci
        .rfind('e')
        .and_then(|i| sci[i + 1..].parse().ok())
        .unwrap_or(0);
    if e < -6 || e >= p as i32 {
        // Exponential notation with p-1 fractional digits. Rust omits the `+` on
        // a non-negative exponent; JavaScript includes it (`1e+4`, not `1e4`).
        let s = alloc::format!("{:.*e}", p - 1, n);
        return match s.find('e') {
            Some(epos) if s.as_bytes().get(epos + 1) != Some(&b'-') => {
                alloc::format!("{}e+{}", &s[..epos], &s[epos + 1..])
            }
            _ => s,
        };
    }
    let decimals = (p as i32 - 1 - e).max(0) as usize;
    alloc::format!("{:.*}", decimals, n)
}

/// `String.prototype.padEnd`: append `pad` (repeated) until length `target`.
fn pad_end(s: &str, target: usize, pad: &str) -> String {
    let len = s.chars().count();
    if len >= target || pad.is_empty() {
        return String::from(s);
    }
    let need = target - len;
    let mut filler = String::new();
    while filler.chars().count() < need {
        filler.push_str(pad);
    }
    let filler: String = filler.chars().take(need).collect();
    String::from(s) + &filler
}

/// Quotes and escapes a string as a JSON string literal.
fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&alloc::format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Renders the integer part of `n` in `radix` (2–36), with a leading `-` for
/// negatives (matching `Number.prototype.toString(radix)` for integers).
/// A minimal `Number.prototype.toLocaleString` — groups the integer part with
/// `,` thousands separators (no locale data, so this is the en-US-ish default).
/// Coerces `n` to a typed-array element of the given kind index (see
/// [`TYPED_ARRAY_KINDS`]): integer kinds truncate then wrap (or clamp, for
/// `Uint8Clamped`); float kinds narrow precision.
/// Parses a `DataView` accessor name (`getInt32`, `setFloat64`, …) into
/// `(is_set, byte_size, signed, is_float)`, or `None` if it isn't one.
fn dataview_method(method: &str) -> Option<(bool, usize, bool, bool, bool)> {
    let (is_set, t) = if let Some(t) = method.strip_prefix("get") {
        (false, t)
    } else if let Some(t) = method.strip_prefix("set") {
        (true, t)
    } else {
        return None;
    };
    // (size, signed, is_float, is_bigint)
    let (size, signed, is_float, is_bigint) = match t {
        "Int8" => (1, true, false, false),
        "Uint8" => (1, false, false, false),
        "Int16" => (2, true, false, false),
        "Uint16" => (2, false, false, false),
        "Int32" => (4, true, false, false),
        "Uint32" => (4, false, false, false),
        "Float32" => (4, false, true, false),
        "Float64" => (8, false, true, false),
        "BigInt64" => (8, true, false, true),
        "BigUint64" => (8, false, false, true),
        _ => return None,
    };
    Some((is_set, size, signed, is_float, is_bigint))
}

/// Maps a WASM export/import kind byte to its `ExternType` string.
fn wasm_extern_kind(kind: u8) -> &'static str {
    match kind {
        0 => "function",
        1 => "table",
        2 => "memory",
        _ => "global",
    }
}

fn coerce_typed(kind: u16, n: f64) -> f64 {
    match kind {
        7 => f64::from(n as f32), // Float32
        8 => n,                   // Float64
        2 => {
            // Uint8Clamped: clamp to 0..=255 with round-half-to-even (core only).
            if n.is_nan() || n <= 0.0 {
                0.0
            } else if n >= 255.0 {
                255.0
            } else {
                let fl = n as i64;
                let frac = n - fl as f64;
                let r = if frac < 0.5 {
                    fl
                } else if frac > 0.5 || fl % 2 != 0 {
                    fl + 1
                } else {
                    fl
                };
                r as f64
            }
        }
        _ => {
            if !n.is_finite() {
                return 0.0;
            }
            // Truncate toward zero, then reduce into range with integer math (no
            // std float methods, for the `alloc`-only build).
            let i = n as i64;
            let (bits, signed) = match kind {
                0 => (8u32, true), // Int8
                1 => (8, false),   // Uint8
                3 => (16, true),   // Int16
                4 => (16, false),  // Uint16
                5 => (32, true),   // Int32
                _ => (32, false),  // Uint32
            };
            let modulus = 1i64 << bits;
            let mut u = i.rem_euclid(modulus);
            if signed && u >= modulus / 2 {
                u -= modulus;
            }
            u as f64
        }
    }
}

fn group_thousands(n: f64) -> String {
    if n.is_nan() {
        return String::from("NaN");
    }
    if n.is_infinite() {
        return String::from(if n > 0.0 { "∞" } else { "-∞" });
    }
    let neg = n.is_sign_negative() && n != 0.0;
    let abs = if n < 0.0 { -n } else { n };
    let base = alloc::format!("{abs}");
    let (int_part, frac_part) = match base.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (base.as_str(), None),
    };
    let bytes = int_part.as_bytes();
    let len = bytes.len();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    if let Some(f) = frac_part {
        out.push('.');
        out.push_str(f);
    }
    out
}

/// Slices `s` by *character* (Unicode scalar) indices `[st, en)` — the index
/// space the regex engine works in — so multi-byte characters never split a byte
/// boundary (which would panic on `&s[st..en]`).
#[cfg(feature = "regex")]
fn char_substr(s: &str, st: usize, en: usize) -> String {
    s.chars().skip(st).take(en.saturating_sub(st)).collect()
}

/// Slices `s` from character index `st` to the end.
#[cfg(feature = "regex")]
fn char_substr_from(s: &str, st: usize) -> String {
    s.chars().skip(st).collect()
}

fn int_to_radix(n: f64, radix: u32) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let neg = n < 0.0;
    let abs = if neg { -n } else { n };
    // Integer part (the `as u64` cast truncates toward zero — no `std` float math).
    let mut v = abs as u64;
    let mut ibuf = Vec::new();
    if v == 0 {
        ibuf.push(b'0');
    }
    while v > 0 {
        ibuf.push(DIGITS[(v % radix as u64) as usize]);
        v /= radix as u64;
    }
    ibuf.reverse();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    out.push_str(&String::from_utf8(ibuf).unwrap_or_default());
    // Fractional part (bounded digit count to terminate on repeating fractions).
    let mut frac = abs - (abs as u64) as f64;
    if frac > 0.0 {
        out.push('.');
        for _ in 0..20 {
            if frac <= 0.0 {
                break;
            }
            frac *= radix as f64;
            let digit = (frac as usize).min(radix as usize - 1);
            out.push(DIGITS[digit] as char);
            frac -= digit as f64;
        }
    }
    out
}

/// Parses the longest leading decimal-float prefix of `s` (à la `parseFloat`),
/// returning `NaN` if none.
const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `bytes` to a standard (`+`/`/`, `=`-padded) base64 string.
fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[(n >> 18 & 0x3f) as usize] as char);
        out.push(B64_ALPHABET[(n >> 12 & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6 & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Decodes a base64 string (ASCII whitespace ignored), returning `None` on an
/// invalid character or length.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let cleaned: Vec<u8> = s
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(cleaned.len() / 4 * 3);
    for chunk in cleaned.chunks(4) {
        if chunk.len() < 2 {
            return None;
        }
        let mut n = 0u32;
        for &c in chunk {
            n = (n << 6) | val(c)?;
        }
        // Left-align the partial group, then take the available bytes.
        n <<= 6 * (4 - chunk.len());
        out.push((n >> 16 & 0xff) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8 & 0xff) as u8);
        }
        if chunk.len() > 3 {
            out.push((n & 0xff) as u8);
        }
    }
    Some(out)
}

/// Percent-encodes `s`. The unreserved set (`A-Za-z0-9-_.!~*'()`) is always kept;
/// `extra` adds characters preserved by `encodeURI` (the URI reserved set).
fn uri_encode(s: &str, extra: &str) -> String {
    let mut out = String::new();
    let mut buf = [0u8; 4];
    for ch in s.chars() {
        let keep = ch.is_ascii_alphanumeric() || "-_.!~*'()".contains(ch) || extra.contains(ch);
        if keep {
            out.push(ch);
        } else {
            for b in ch.encode_utf8(&mut buf).bytes() {
                out.push('%');
                out.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
                out.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap()
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    out
}

/// Decodes percent-escapes in `s` (`%XX` → byte), returning `None` on a malformed
/// escape or invalid UTF-8.
fn uri_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = (*bytes.get(i + 1)? as char).to_digit(16)?;
            let lo = (*bytes.get(i + 2)? as char).to_digit(16)?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

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
    let mut seen_dot = false;
    let mut seen_e = false;
    while end < bytes.len() {
        let ch = bytes[end] as char;
        let ok = match ch {
            '0'..='9' => true,
            '+' | '-' if end == 0 || matches!(bytes[end - 1] as char, 'e' | 'E') => true,
            '.' if !seen_dot && !seen_e => {
                seen_dot = true;
                true
            }
            'e' | 'E' if !seen_e && end > 0 => {
                seen_e = true;
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

/// Expands a `replace` template: `$&` (whole match), `$1`..`$9` (groups), `$$`
/// (literal `$`), against `caps` over `text`.
#[cfg(feature = "regex")]
fn expand_replacement(
    templ: &str,
    text: &str,
    caps: &crate::regex::Captures,
    group_names: &[(usize, String)],
) -> String {
    let group = |i: usize| {
        caps.groups
            .get(i)
            .and_then(|g| *g)
            .map(|(s, e)| &text[s..e])
    };
    let (m_start, m_end) = caps.groups.first().and_then(|g| *g).unwrap_or((0, 0));
    let mut out = String::new();
    let mut chars = templ.chars().peekable();
    while let Some(c) = chars.next() {
        // `$<name>` — a named-group backreference.
        if c == '$' && chars.peek() == Some(&'<') {
            chars.next(); // `<`
            let mut name = String::new();
            while let Some(&ch) = chars.peek() {
                chars.next();
                if ch == '>' {
                    break;
                }
                name.push(ch);
            }
            if let Some((idx, _)) = group_names.iter().find(|(_, n)| *n == name) {
                out.push_str(group(*idx).unwrap_or(""));
            }
            continue;
        }
        if c != '$' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            Some('$') => {
                out.push('$');
                chars.next();
            }
            Some('&') => {
                out.push_str(group(0).unwrap_or(""));
                chars.next();
            }
            // `` $` `` is the portion before the match; `$'` the portion after.
            Some('`') => {
                out.push_str(&text[..m_start]);
                chars.next();
            }
            Some('\'') => {
                out.push_str(&text[m_end..]);
                chars.next();
            }
            // `$n` for an in-range group → the capture (empty if unmatched);
            // out-of-range `$n` is left literal.
            Some(d)
                if d.is_ascii_digit() && {
                    let n = (*d as u8 - b'0') as usize;
                    n >= 1 && n < caps.groups.len()
                } =>
            {
                let n = (*d as u8 - b'0') as usize;
                chars.next();
                out.push_str(group(n).unwrap_or(""));
            }
            _ => out.push('$'),
        }
    }
    out
}

/// Advances `pos` past JSON whitespace.
fn skip_ws(c: &[char], pos: &mut usize) {
    while c
        .get(*pos)
        .is_some_and(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r'))
    {
        *pos += 1;
    }
}

/// Parses and runs `source` on the new representation, returning the captured
/// `console` output and the program's completion value (as a display string).
///
/// This is the high-level entry point to the new-model engine — the bridge the
/// production pipeline migrates onto.
///
/// # Errors
/// Returns a parse or execution error message on failure.
pub fn eval_source(source: &str) -> Result<(String, String), String> {
    let program =
        crate::parser::Parser::parse_program(source).map_err(|e| alloc::format!("{e}"))?;
    let mut interp = Interp::new();
    let value = match interp.run(&program) {
        Ok(v) => v,
        // Render an uncaught throw readably: an error object as `name: message`,
        // any other thrown value via its display string.
        Err(ExecError::Throw(thrown)) => return Err(format_thrown(&interp, thrown)),
        Err(other) => return Err(alloc::format!("{other:?}")),
    };
    let completion = interp.display(value);
    Ok((String::from(interp.output()), completion))
}

/// Formats an uncaught thrown value for an error message: `name: message` for an
/// error-shaped object, otherwise the value's display string.
fn format_thrown(interp: &Interp, thrown: NanBox) -> String {
    if let Some(raw) = thrown.as_handle() {
        let h = Handle::from_raw(raw);
        let realm = interp.realm();
        if let Some(name) = realm.get_property(h, "name") {
            let name = realm.to_display_string(name);
            let message = realm
                .get_property(h, "message")
                .map(|m| realm.to_display_string(m))
                .unwrap_or_default();
            return if message.is_empty() {
                name
            } else {
                alloc::format!("{name}: {message}")
            };
        }
    }
    interp.display(thrown)
}

/// The current time in milliseconds since the Unix epoch (`0.0` without `std`,
/// which has no clock).
fn now_ms() -> f64 {
    #[cfg(feature = "std")]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0)
    }
    #[cfg(not(feature = "std"))]
    {
        0.0
    }
}

/// A minimal `parseInt`: skips leading whitespace, reads an optional sign and
/// the leading decimal digits, and returns `NaN` if there are none.
fn parse_int(s: &str, radix: u32) -> f64 {
    let mut t = s.trim_start();
    let mut neg = false;
    if let Some(rest) = t.strip_prefix('-') {
        neg = true;
        t = rest;
    } else if let Some(rest) = t.strip_prefix('+') {
        t = rest;
    }
    // Radix 0 means infer: `0x` → 16, else 10. A `0x` prefix is also honored
    // when radix is explicitly 16.
    let mut radix = radix;
    if (radix == 0 || radix == 16)
        && let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
    {
        t = rest;
        radix = 16;
    }
    if radix == 0 {
        radix = 10;
    }
    if !(2..=36).contains(&radix) {
        return f64::NAN;
    }
    // Consume the leading digits valid in this radix.
    let mut value: f64 = 0.0;
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

/// Renders a `BigInt` in the given radix (2..=36), base 10 by default.
fn bigint_to_radix(n: &crate::bignum::BigInt, radix: u32) -> String {
    let radix = if (2..=36).contains(&radix) { radix } else { 10 };
    n.to_str_radix(radix)
}

/// Parses a normalized `BigInt` digit string (decimal, or `0x`/`0o`/`0b`
/// prefixed) into the arbitrary-precision representation.
fn parse_bigint(digits: &str) -> crate::bignum::BigInt {
    let (radix, body) = match digits.get(0..2) {
        Some("0x" | "0X") => (16, &digits[2..]),
        Some("0o" | "0O") => (8, &digits[2..]),
        Some("0b" | "0B") => (2, &digits[2..]),
        _ => (10, digits),
    };
    crate::bignum::BigInt::from_str_radix(body, radix).unwrap_or_else(crate::bignum::BigInt::zero)
}

/// Normalizes an optional `fromIndex` for `indexOf`/`includes`: undefined → 0,
/// negatives count from the end, clamped to `[0, len]`.
/// `ToInteger` for a string index argument: `NaN` (and no-arg) → `Some(0)`, a
/// non-negative integer → `Some(i)`, and a negative index → `None` (out of range,
/// so `charAt` yields `""` and `charCodeAt`/`codePointAt` yield `NaN`/`undefined`).
fn str_char_index(n: f64) -> Option<usize> {
    let n = if n.is_nan() { 0.0 } else { n };
    (n >= 0.0).then_some(n as usize)
}

fn array_from_index(realm: &Realm, arg: NanBox, len: usize) -> usize {
    if matches!(arg.unpack(), Unpacked::Undefined) {
        return 0;
    }
    let n = realm.to_number(arg);
    if n < 0.0 {
        (len as f64 + n).max(0.0) as usize
    } else {
        (n as usize).min(len)
    }
}

/// A non-negative integer array index, if `n` is one.
fn as_index(n: f64) -> Option<usize> {
    if n >= 0.0 && n <= u32::MAX as f64 && (n as u64) as f64 == n {
        Some(n as usize)
    } else {
        None
    }
}

/// A static (non-computed) property key as a string.
/// Expands `$`-patterns in a string-`replace` template (no capture groups, so
/// `$1`…`$9` stay literal): `$&` → match, `` $` `` → prefix, `$'` → suffix,
/// `$$` → `$`.
fn expand_dollar(template: &str, m: &str, before: &str, after: &str) -> String {
    let chars: Vec<char> = template.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '$'
            && i + 1 < chars.len()
            && let Some(rep) = match chars[i + 1] {
                '$' => Some("$"),
                '&' => Some(m),
                '`' => Some(before),
                '\'' => Some(after),
                _ => None,
            }
        {
            out.push_str(rep);
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn static_key(key: &PropertyKey) -> Result<String, ExecError> {
    match key {
        PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(String::from(&**s)),
        PropertyKey::Number(n) => Ok(alloc::format!("{n}")),
        // A private field name (`#x`) maps to a `#`-prefixed storage key.
        PropertyKey::Private(s) => Ok(alloc::format!("#{s}")),
        PropertyKey::Computed(_) => Err(ExecError::Unsupported("computed key")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// Runs `src` and renders the program's final value.
    fn run(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        let value = interp.run(&program).expect("exec");
        interp.realm().to_display_string(value)
    }

    /// Runs `src` and returns its captured `console` output.
    fn out(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    }

    #[test]
    fn variables_assignment_and_control_flow() {
        assert_eq!(run("let x = 1; let y = 2; x + y"), "3");
        assert_eq!(run("let s = 'a'; s += 'b'; s += 'c'; s"), "abc");
        assert_eq!(run("let x = 1; { let x = 99; } x"), "1");
        assert_eq!(
            run("let s = 0; for (let i = 1; i <= 10; i += 1) s += i; s"),
            "55"
        );
        assert_eq!(
            run("let s = 0; for (let i = 0; i < 10; i += 1) { if (i === 5) break; s += i; } s"),
            "10"
        );
    }

    #[test]
    fn functions_and_return() {
        assert_eq!(run("function add(a, b) { return a + b; } add(2, 3)"), "5");
        assert_eq!(run("let sq = function (x) { return x * x; }; sq(7)"), "49");
        assert_eq!(run("let inc = x => x + 1; inc(41)"), "42");
        // Hoisting: callable before its definition.
        assert_eq!(run("f(10); function f(n) { return n; } f(10)"), "10");
    }

    /// The public D′ API (`Interp::snapshot` / `restore_snapshot`): snapshot a
    /// live closure's state in one interpreter and reload it into a *fresh* one
    /// holding the same code, through the supported library surface alone — no
    /// reaching into interpreter internals.
    #[test]
    fn public_snapshot_api_round_trips_across_runtimes() {
        let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

        // Runtime A: advance a counter to n = 2, snapshot it to bytes.
        let mut a = Interp::new();
        let f = a.run(&program).expect("exec A");
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
        let bytes = a.snapshot(&[f]);
        drop(a);

        // Runtime B: a fresh interpreter compiles the same program, then reloads
        // A's snapshot and runs the restored closure — resuming from n = 2.
        let mut b = Interp::new();
        let own = b.run(&program).expect("exec B");
        let restored = b.restore_snapshot(&bytes).expect("restore");
        assert_eq!(restored.len(), 1, "one heap root restored");
        assert_eq!(
            b.call(restored[0], &[]).unwrap().as_number(),
            Some(3.0),
            "restored closure resumes from snapshotted state"
        );
        assert_eq!(
            b.call(own, &[]).unwrap().as_number(),
            Some(1.0),
            "the fresh runtime's own counter is independent"
        );

        // A malformed snapshot is rejected, not panicked on.
        assert!(b.restore_snapshot(b"not a snapshot").is_err());
    }

    /// Cross-runtime D′ reload: snapshot a closure in one runtime, serialize it,
    /// then restore and **execute** it in a *separate, fresh* runtime that holds
    /// the same code — the load → evict → reload scenario. The restored closure
    /// carries the snapshotted captured state and is independent of the fresh
    /// runtime's own instance of the program.
    #[test]
    fn snapshot_reloads_into_a_fresh_runtime() {
        use crate::snapshot::{capture, deserialize, restore, serialize};

        // `makeCounter` (func 0) returns the inner closure (func 1); both runtimes
        // compile the same program, so the snapshot's `func_id`s line up.
        let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

        // Runtime A: build a counter, advance it to n = 2, snapshot it to bytes.
        let mut a = Interp::new();
        let f = a.run(&program).expect("exec A");
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
        let fh = Handle::from_raw(f.as_handle().expect("closure"));
        let bytes = serialize(&capture(&a.realm, &[fh]));
        drop(a); // A is gone — only the bytes survive.

        // Runtime B: a fresh interpreter that compiles the same program (its own
        // counter starts at 0), then reloads A's snapshot and runs the restored
        // closure — which resumes from the snapshotted n = 2.
        let mut b = Interp::new();
        let own = b.run(&program).expect("exec B");
        let snap = deserialize(&bytes).expect("deserialize");
        let restored = restore(&mut b.realm, &snap);
        let f2 = NanBox::handle(restored[0].to_raw());

        assert_eq!(
            b.call(f2, &[]).unwrap().as_number(),
            Some(3.0),
            "restored closure resumes from the snapshotted state in the new runtime"
        );
        assert_eq!(
            b.call(own, &[]).unwrap().as_number(),
            Some(1.0),
            "the fresh runtime's own counter is independent of the reloaded one"
        );
    }

    /// End-to-end D′: a live closure's captured state survives capture →
    /// serialize → deserialize → restore *and remains executable* — the restored
    /// closure runs, carries the snapshotted captured value, and is independent of
    /// the original. (Same interpreter, so its function table already holds the
    /// bodies the snapshot's `func_id`s refer to.)
    #[test]
    fn snapshot_restores_an_executable_closure() {
        use crate::snapshot::{capture, deserialize, restore, serialize};

        let program = Parser::parse_program(
            "function counter(){ var n = 0; return function(){ return ++n; }; } counter()",
        )
        .expect("parse");
        let mut interp = Interp::new();
        let f = interp.run(&program).expect("exec");
        let fh = Handle::from_raw(f.as_handle().expect("closure handle"));

        // Advance the original to n = 2, then snapshot it there.
        assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(1.0));
        assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(2.0));
        let snap = capture(&interp.realm, &[fh]);

        // Full on-disk round-trip.
        let snap = deserialize(&serialize(&snap)).expect("deserialize");

        // Restore into the same runtime (whose function table still has the body).
        let restored = restore(&mut interp.realm, &snap);
        let f2 = NanBox::handle(restored[0].to_raw());

        // The original kept counting (n was 2 → 3); the restored closure starts
        // from the *snapshotted* n = 2 → 3, proving it both executes and carries
        // the captured value.
        assert_eq!(
            interp.call(f, &[]).unwrap().as_number(),
            Some(3.0),
            "original continues"
        );
        assert_eq!(
            interp.call(f2, &[]).unwrap().as_number(),
            Some(3.0),
            "restored from snapshot"
        );

        // Independence: advancing the restored copy does not move the original.
        assert_eq!(
            interp.call(f2, &[]).unwrap().as_number(),
            Some(4.0),
            "restored advances"
        );
        assert_eq!(
            interp.call(f, &[]).unwrap().as_number(),
            Some(4.0),
            "original unaffected by restore"
        );
    }

    #[test]
    fn closures_capture_their_scope() {
        // A returned inner function still sees the enclosing variable.
        assert_eq!(
            run(
                "function adder(n) { return function (x) { return x + n; }; }
                 let add5 = adder(5);
                 add5(10)"
            ),
            "15"
        );
        // The capture is by reference: a mutable counter.
        assert_eq!(
            run("function counter() {
                   let c = 0;
                   return function () { c += 1; return c; };
                 }
                 let next = counter();
                 next(); next(); next()"),
            "3"
        );
    }

    #[test]
    fn recursion() {
        assert_eq!(
            run("function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); } fact(5)"),
            "120"
        );
        assert_eq!(
            run("function fib(n) { return n < 2 ? n : fib(n-1) + fib(n-2); } fib(10)"),
            "55"
        );
    }

    #[test]
    fn higher_order_and_objects() {
        // A function stored on an object, called, mutating closed-over state.
        assert_eq!(
            run("function makeBox(v) {
                   let value = v;
                   return { get: function () { return value; },
                            set: function (x) { value = x; } };
                 }
                 let b = makeBox(1);
                 b.set(99);
                 b.get()"),
            "99"
        );
    }

    #[test]
    fn calling_a_non_function_errors() {
        let program = Parser::parse_program("let x = 5; x()").unwrap();
        let mut interp = Interp::new();
        assert_eq!(interp.run(&program), Err(ExecError::NotCallable));
    }

    #[test]
    fn try_catch_finally() {
        // A thrown value is caught and bound.
        assert_eq!(
            run("let r; try { throw 'boom'; r = 'no'; } catch (e) { r = 'caught:' + e; } r"),
            "caught:boom"
        );
        // No throw: the catch is skipped.
        assert_eq!(
            run("let r = 'ok'; try { r = 'a'; } catch (e) { r = 'b'; } r"),
            "a"
        );
        // finally always runs.
        assert_eq!(
            run(
                "let log = ''; try { log += '1'; throw 0; } catch (e) { log += '2'; } finally { log += '3'; } log"
            ),
            "123"
        );
        // A throw out of a function is caught at the call site.
        assert_eq!(
            run("function boom() { throw 'x'; }
                 let r; try { boom(); } catch (e) { r = 'got:' + e; } r"),
            "got:x"
        );
        // catch without a binding.
        assert_eq!(
            run("let r = 'a'; try { throw 1; } catch { r = 'b'; } r"),
            "b"
        );
    }

    #[test]
    fn uncaught_throw_propagates() {
        let program = Parser::parse_program("throw 'oops'").unwrap();
        let mut interp = Interp::new();
        match interp.run(&program) {
            Err(ExecError::Throw(v)) => {
                assert_eq!(interp.realm().to_display_string(v), "oops");
            }
            other => panic!("expected a throw, got {other:?}"),
        }
    }

    #[test]
    fn finally_return_overrides() {
        // A `return` in finally overrides the try's outcome.
        assert_eq!(
            run("function f() { try { return 'a'; } finally { return 'b'; } } f()"),
            "b"
        );
    }

    #[test]
    fn builtin_functions() {
        // Math methods (variadic).
        assert_eq!(run("Math.max(3, 7, 2)"), "7");
        assert_eq!(run("Math.min(3, 7, 2)"), "2");
        assert_eq!(run("Math.abs(-5)"), "5");
        // Coercion globals.
        assert_eq!(run("String(42)"), "42");
        assert_eq!(run("Number('3.5')"), "3.5");
        assert_eq!(run("Boolean(0)"), "false");
        assert_eq!(run("Boolean('x')"), "true");
        assert_eq!(run("parseInt('42px')"), "42");
        assert_eq!(run("parseInt('  -7 ')"), "-7");
        // typeof a built-in is "function".
        assert_eq!(run("typeof Math.max"), "function");
        // Built-ins compose with user code.
        assert_eq!(
            run(
                "function clamp(x, lo, hi) { return Math.max(lo, Math.min(x, hi)); }
                 clamp(15, 0, 10)"
            ),
            "10"
        );
    }

    #[test]
    fn string_methods() {
        assert_eq!(run("'hello'.toUpperCase()"), "HELLO");
        assert_eq!(run("'HELLO'.toLowerCase()"), "hello");
        assert_eq!(run("'  hi  '.trim()"), "hi");
        assert_eq!(run("'hello'.charAt(1)"), "e");
        assert_eq!(run("'hello'.includes('ell')"), "true");
        assert_eq!(run("'hello'.indexOf('l')"), "2");
        assert_eq!(run("'ab'.repeat(3)"), "ababab");
    }

    #[test]
    fn array_methods() {
        assert_eq!(run("let a = [1, 2]; a.push(3); a.join('-')"), "1-2-3");
        assert_eq!(run("let a = [1, 2, 3]; a.pop()"), "3");
        assert_eq!(run("[1, 2, 3].includes(2)"), "true");
        assert_eq!(run("[1, 2, 3].indexOf(3)"), "2");
        assert_eq!(run("['a', 'b', 'c'].join(', ')"), "a, b, c");
        // splice (remove + insert), unshift, shift.
        assert_eq!(
            run("let a=[1,2,3,4]; a.splice(1,2,'x'); a.join(',')"),
            "1,x,4"
        );
        assert_eq!(run("[1,2,3,4].splice(1,2).join(',')"), "2,3");
        assert_eq!(run("let a=[2,3]; a.unshift(0,1); a.join(',')"), "0,1,2,3");
        assert_eq!(
            run("let a=[1,2,3]; let f=a.shift(); f + ':' + a.join(',')"),
            "1:2,3"
        );
        // Non-mutating: toSorted/toReversed/with leave the original.
        assert_eq!(
            run(
                "let a=[3,1,2]; let s=a.toSorted(function(x,y){return x-y;}); s.join('')+'|'+a.join('')"
            ),
            "123|312"
        );
        assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
        assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
        // includes uses SameValueZero (NaN matches).
        assert_eq!(run("[NaN, 1].includes(NaN)"), "true");
        assert_eq!(run("[1, 2].includes(NaN)"), "false");
    }

    #[test]
    fn define_property_and_locale_compare() {
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:42}); o.x"),
            "42"
        );
        assert_eq!(
            run(
                "let o={n:1}; Object.defineProperty(o,'d',{get:function(){return this.n+1;}}); o.d"
            ),
            "2"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:7}); Object.getOwnPropertyDescriptor(o,'x').value"
            ),
            "7"
        );
        assert_eq!(run("'apple'.localeCompare('banana') < 0"), "true");
        assert_eq!(run("'x'.localeCompare('x')"), "0");
        // fill: whole array, a [start,end) range, and a negative start.
        assert_eq!(run("[0, 0, 0].fill(7).join(',')"), "7,7,7");
        assert_eq!(run("[1, 2, 3, 4].fill(9, 1, 3).join(',')"), "1,9,9,4");
        assert_eq!(run("[1, 2, 3, 4, 5].fill(0, -2).join(',')"), "1,2,3,0,0");
        // reduceRight folds right-to-left, with and without a seed.
        assert_eq!(
            run("['a','b','c'].reduceRight(function(acc,x){ return acc + x; })"),
            "cba"
        );
        assert_eq!(
            run("[1,2,3].reduceRight(function(a,x){ return a + x; }, 10)"),
            "16"
        );
        // findLast / findLastIndex scan right-to-left.
        assert_eq!(
            run("[1,2,3,4].findLast(function(x){ return x % 2 === 1; })"),
            "3"
        );
        assert_eq!(
            run("[1,2,3,4].findLastIndex(function(x){ return x % 2 === 1; })"),
            "2"
        );
        assert_eq!(
            run("[2,4,6].findLast(function(x){ return x > 9; })"),
            "undefined"
        );
        // copyWithin (in place) and flat(depth).
        assert_eq!(run("[1,2,3,4,5].copyWithin(0,3).join(',')"), "4,5,3,4,5");
        assert_eq!(run("[1,[2,[3,[4]]]].flat(2).join(',')"), "1,2,3,4");
    }

    #[test]
    fn math_extras_and_number_coercion() {
        assert_eq!(run("Math.hypot(3, 4)"), "5");
        assert_eq!(run("Math.cbrt(27)"), "3");
        assert_eq!(run("Math.log2(8)"), "3");
        assert_eq!(run("Math.log10(1000)"), "3");
        assert_eq!(run("(1234.5).toExponential(2)"), "1.23e+3");
        // Radix-prefixed string coercion (shared `to_number`, both engines).
        assert_eq!(run("Number('0x1F')"), "31");
        assert_eq!(run("+'0b101'"), "5");
        assert_eq!(run("'0o17' * 1"), "15");
    }

    #[test]
    fn split_limit_and_to_precision() {
        assert_eq!(run("'a,b,c,d'.split(',', 2).length"), "2");
        assert_eq!(run("'aXbXc'.split('X').join('-')"), "a-b-c");
        assert_eq!(run("(123.456).toPrecision(4)"), "123.5");
        assert_eq!(run("(255).toString(2)"), "11111111");
    }

    #[test]
    fn object_freeze_family() {
        // Writes and new properties are no-ops on a frozen object.
        assert_eq!(
            run(
                "let o = Object.freeze({ a: 1 }); o.a = 9; o.b = 2; o.a + ',' + (o.b === undefined)"
            ),
            "1,true"
        );
        assert_eq!(run("Object.isFrozen(Object.freeze({}))"), "true");
        assert_eq!(run("Object.isFrozen({})"), "false");
        assert_eq!(
            run("Object.getOwnPropertyNames({ x: 1, y: 2 }).length"),
            "2"
        );
    }

    #[test]
    fn string_pad_lastindexof_and_number_statics() {
        assert_eq!(run("'5'.padEnd(3, '-')"), "5--");
        assert_eq!(run("'ab'.padEnd(5)"), "ab   ");
        assert_eq!(run("'a-b-c'.lastIndexOf('-')"), "3");
        assert_eq!(run("'abc'.lastIndexOf('x')"), "-1");
        assert_eq!(run("Number.MAX_SAFE_INTEGER"), "9007199254740991");
        assert_eq!(run("Number.POSITIVE_INFINITY"), "Infinity");
        assert_eq!(run("'abc'.concat('def', '!')"), "abcdef!");
    }

    #[test]
    fn console_log_captures_output() {
        let program =
            Parser::parse_program("console.log('hi', 42); let x = [1, 2]; console.log('arr:', x);")
                .unwrap();
        let mut interp = Interp::new();
        interp.run(&program).unwrap();
        assert_eq!(interp.output(), "hi 42\narr: 1,2\n");
    }

    #[test]
    fn json_getters_tojson_and_date_utc() {
        // JSON.stringify invokes getters and honors toJSON.
        assert_eq!(
            run("JSON.stringify({a:1, get b(){ return 2; }})"),
            "{\"a\":1,\"b\":2}"
        );
        assert_eq!(
            run("JSON.stringify({v:42, toJSON(){ return {w:this.v}; }})"),
            "{\"w\":42}"
        );
        assert_eq!(run("JSON.stringify([undefined,1])"), "[null,1]");
        // Date.UTC and getUTC* methods.
        assert_eq!(run("new Date(Date.UTC(2024,0,15)).getUTCDate()"), "15");
        assert_eq!(run("new Date(Date.UTC(2024,0,1)).getUTCDay()"), "1"); // Monday
        assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn date_invalid_toisostring() {
        assert_eq!(
            run("try{new Date(NaN).toISOString();'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        assert_eq!(run("new Date(NaN).toJSON()"), "null");
        assert_eq!(run("JSON.stringify({d:new Date(NaN)})"), "{\"d\":null}");
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15,10,30,45,123)).toISOString()"),
            "2020-06-15T10:30:45.123Z"
        );
        assert_eq!(
            run("try{new Date('garbage').toISOString();'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
    }

    #[test]
    fn objects_inherit_object_prototype() {
        assert_eq!(run("'toString' in {}"), "true");
        assert_eq!(run("'hasOwnProperty' in {}"), "true");
        assert_eq!(
            run("Object.getPrototypeOf({}) === Object.prototype"),
            "true"
        );
        assert_eq!(run("'toString' in Object.create(null)"), "false");
        assert_eq!(
            run("Object.getPrototypeOf(Object.create(null)) === null"),
            "true"
        );
        // Inherited methods are non-enumerable.
        assert_eq!(run("Object.keys({a:1,b:2}).join(',')"), "a,b");
        assert_eq!(
            run("let s=[]; for(let k in {a:1,b:2}) s.push(k); s.join(',')"),
            "a,b"
        );
        assert_eq!(run("JSON.stringify({a:1})"), "{\"a\":1}");
        // hasOwnProperty distinguishes own vs inherited.
        assert_eq!(
            run(
                "let c=Object.create({i:1}); c.o=2; c.hasOwnProperty('o') + ',' + c.hasOwnProperty('i')"
            ),
            "true,false"
        );
    }

    #[test]
    fn object_prototype_tostring_call() {
        assert_eq!(run("typeof Object.prototype"), "object");
        assert_eq!(run("Object.prototype.toString.call({})"), "[object Object]");
        assert_eq!(run("Object.prototype.toString.call([])"), "[object Array]");
        assert_eq!(run("Object.prototype.toString.call(null)"), "[object Null]");
        assert_eq!(
            run("Object.prototype.toString.call(undefined)"),
            "[object Undefined]"
        );
        assert_eq!(
            run("Object.prototype.toString.call(function(){})"),
            "[object Function]"
        );
        assert_eq!(
            run("Object.prototype.toString.call(/x/)"),
            "[object RegExp]"
        );
        assert_eq!(
            run("Object.prototype.toString.call({[Symbol.toStringTag]:'Widget'})"),
            "[object Widget]"
        );
        assert_eq!(
            run("Object.prototype.hasOwnProperty.call({a:1},'a')"),
            "true"
        );
        assert_eq!(
            run("Object.prototype.hasOwnProperty.call({a:1},'b')"),
            "false"
        );
        assert_eq!(
            run("let p={}; Object.prototype.isPrototypeOf.call(p, Object.create(p))"),
            "true"
        );
        assert_eq!(
            run("let o={x:1}; Object.prototype.valueOf.call(o)===o"),
            "true"
        );
    }

    #[test]
    fn webassembly_instantiate_and_call() {
        // `WebAssembly.instantiate` now returns a Promise (per spec); use the
        // synchronous `new Instance(new Module(...))` path to read exports directly.
        let setup = "var b=[0,0x61,0x73,0x6d,1,0,0,0, 1,7,1,0x60,2,0x7f,0x7f,1,0x7f, 3,2,1,0, 7,7,1,3,0x61,0x64,0x64,0,0, 0xa,9,1,7,0,0x20,0,0x20,1,0x6a,0xb]; var e=new WebAssembly.Instance(new WebAssembly.Module(b)).exports; ";
        assert_eq!(run(&alloc::format!("{setup} typeof e.add")), "function");
        assert_eq!(run(&alloc::format!("{setup} e.add(20,22)")), "42");
        assert_eq!(run(&alloc::format!("{setup} e.add(-5,8)")), "3");
    }

    #[test]
    fn webassembly_validate_builtin() {
        assert_eq!(run("typeof WebAssembly"), "object");
        assert_eq!(run("typeof WebAssembly.validate"), "function");
        assert_eq!(
            run("WebAssembly.validate([0,0x61,0x73,0x6d,1,0,0,0])"),
            "true"
        );
        assert_eq!(run("WebAssembly.validate([0,0,0,0,1,0,0,0])"), "false");
        assert_eq!(run("WebAssembly.validate([0,0x61])"), "false");
    }

    #[test]
    fn object_default_tostring_and_tag() {
        assert_eq!(run("({}).toString()"), "[object Object]");
        assert_eq!(run("'abc'.toString()"), "abc");
        assert_eq!(run("({a:1}).valueOf().a"), "1");
        assert_eq!(run("({toString(){return 'custom';}}).toString()"), "custom");
        assert_eq!(
            run("Object.create({toString(){return 'base';}}).toString()"),
            "base"
        );
        assert_eq!(
            run("({[Symbol.toStringTag]:'Widget'}).toString()"),
            "[object Widget]"
        );
        assert_eq!(run("typeof Symbol.toStringTag"), "symbol");
        assert_eq!(run("typeof Symbol.species"), "symbol");
        assert_eq!(
            run("let o={[Symbol.toStringTag]:'X'}; Object.getOwnPropertySymbols(o).length"),
            "1"
        );
        // Existing toStrings unaffected.
        assert_eq!(run("[1,2,3].toString()"), "1,2,3");
        assert_eq!(run("new Error('x').toString()"), "Error: x");
    }

    #[test]
    fn strict_getter_only_write() {
        // Sloppy: silently ignored.
        assert_eq!(run("let o={get x(){return 1;}}; o.x=2; o.x"), "1");
        // Strict: throws TypeError.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={get x(){return 1;}}; try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
        // A setter still works under strict mode.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={_v:0,get x(){return this._v;},set x(v){this._v=v*2;}}; o.x=5; return o.x;})()"
            ),
            "10"
        );
        // Inherited getter-only accessor.
        assert_eq!(
            run(
                "(function(){'use strict'; let o=Object.create({get y(){return 9;}}); try{o.y=1;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
    }

    #[test]
    fn reduce_empty_throws_typeerror() {
        assert_eq!(
            run("try{[].reduce((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
        assert_eq!(
            run("try{[].reduceRight((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
        assert_eq!(run("[].reduce((a,b)=>a+b, 99)"), "99");
        assert_eq!(run("[42].reduce((a,b)=>a+b)"), "42");
        assert_eq!(run("[1,2,3,4].reduce((a,b)=>a+b)"), "10");
    }

    #[test]
    fn function_names_tostring_bound() {
        assert_eq!(run("let myFn=function(){}; myFn.name"), "myFn");
        assert_eq!(run("let arrow=()=>1; arrow.name"), "arrow");
        assert_eq!(run("let x=function inner(){}; x.name"), "inner");
        assert_eq!(
            run("let o={method(){},fn:()=>1}; o.method.name + ':' + o.fn.name"),
            "method:fn"
        );
        assert_eq!(run("typeof (function f(){}).toString()"), "string");
        assert_eq!(
            run("(function f(){}).toString().indexOf('function')>=0"),
            "true"
        );
        assert_eq!(run("function t(a,b,c){} t.bind(null,1).length"), "2");
        assert_eq!(run("function t(a,b,c){} t.bind(null,1,2,3,4).length"), "0");
        assert_eq!(run("function t(){} t.bind(null).name"), "bound t");
    }

    #[test]
    fn proto_accessor() {
        // Object.create + __proto__ read.
        assert_eq!(
            run("let p={greet(){return 'hi';}}; let o=Object.create(p); o.__proto__===p"),
            "true"
        );
        assert_eq!(
            run("let p={greet(){return 'hi';}}; Object.create(p).greet()"),
            "hi"
        );
        // __proto__ assignment relinks.
        assert_eq!(
            run("let o={}; o.__proto__={hello(){return 'yo';}}; o.hello()"),
            "yo"
        );
        assert_eq!(
            run("let b={getX(){return this.x;}}; let o={}; o.__proto__=b; o.x=42; o.getX()"),
            "42"
        );
        // Object-literal __proto__ sets the prototype.
        assert_eq!(
            run("let b={getX(){return this.x;}}; let n={__proto__:b, x:5}; n.getX()"),
            "5"
        );
        // The method form is a regular property, not a prototype set.
        assert_eq!(
            run("typeof ({__proto__(){return 1;}}).__proto__"),
            "function"
        );
        // __proto__ = null clears the chain; a primitive is ignored.
        assert_eq!(
            run("let o={__proto__:{}}; o.__proto__=null; Object.getPrototypeOf(o)===null"),
            "true"
        );
        assert_eq!(
            run("let p={g(){return 1;}}; let o=Object.create(p); o.__proto__=5; o.g()"),
            "1"
        );
    }

    #[test]
    fn proxy_passthrough_keys() {
        assert_eq!(run("Object.keys(new Proxy({a:1,b:2},{})).join(',')"), "a,b");
        assert_eq!(
            run("Object.values(new Proxy({a:1,b:2},{})).join(',')"),
            "1,2"
        );
        assert_eq!(
            run("Object.entries(new Proxy({a:1,b:2},{})).map(e=>e.join(':')).join(',')"),
            "a:1,b:2"
        );
        // Nested trap-less proxies forward through.
        assert_eq!(
            run("Object.keys(new Proxy(new Proxy({x:1},{}),{})).join(',')"),
            "x"
        );
    }

    #[test]
    fn bigint_as_uintn_intn() {
        assert_eq!(run("BigInt.asUintN(8, 256n)"), "0");
        assert_eq!(run("BigInt.asUintN(8, -1n)"), "255");
        assert_eq!(run("BigInt.asUintN(4, -1n)"), "15");
        assert_eq!(run("BigInt.asUintN(64, 18446744073709551617n)"), "1");
        assert_eq!(run("BigInt.asIntN(8, 200n)"), "-56");
        assert_eq!(run("BigInt.asIntN(8, 128n)"), "-128");
        assert_eq!(run("BigInt.asIntN(8, 127n)"), "127");
        assert_eq!(run("BigInt.asIntN(16, 40000n)"), "-25536");
        assert_eq!(run("BigInt.asIntN(32, 4294967295n)"), "-1");
        assert_eq!(
            run("BigInt.asIntN(128, 12345678901234567890n)"),
            "12345678901234567890"
        );
    }

    #[test]
    fn json_number_serialization() {
        assert_eq!(run("JSON.stringify(-0)"), "0");
        assert_eq!(run("JSON.stringify([-0])"), "[0]");
        assert_eq!(run("JSON.stringify({x:-0})"), "{\"x\":0}");
        assert_eq!(run("JSON.stringify(1e21)"), "1e+21");
        assert_eq!(run("JSON.stringify(1e-7)"), "1e-7");
        assert_eq!(run("JSON.stringify(1e20)"), "100000000000000000000");
        assert_eq!(run("JSON.stringify(0.001)"), "0.001");
        assert_eq!(run("JSON.stringify([NaN,Infinity])"), "[null,null]");
        assert_eq!(
            run("JSON.stringify([-0,1e21,0.001,-42])"),
            "[0,1e+21,0.001,-42]"
        );
    }

    #[test]
    fn json_stringify_replacer() {
        // Function replacer: omit keys and transform values (recursively).
        assert_eq!(
            run("JSON.stringify({a:1,b:2}, function(k,v){ return k==='b'?undefined:v; })"),
            "{\"a\":1}"
        );
        assert_eq!(
            run("JSON.stringify({x:{n:1}}, function(k,v){ return typeof v==='number'?v+5:v; })"),
            "{\"x\":{\"n\":6}}"
        );
        // Array replacer: an allowlist applied at every level.
        assert_eq!(
            run("JSON.stringify({a:1,b:2,c:3}, ['a','c'])"),
            "{\"a\":1,\"c\":3}"
        );
        assert_eq!(
            run("JSON.stringify({keep:{a:1,b:2},drop:9}, ['keep','a'])"),
            "{\"keep\":{\"a\":1}}"
        );
    }

    #[test]
    fn json_parse_reviver() {
        assert_eq!(
            run(
                "let o = JSON.parse('{\"a\":1,\"b\":2}', function(k,v){ return typeof v==='number'?v*2:v; }); o.a + ',' + o.b"
            ),
            "2,4"
        );
        assert_eq!(
            run(
                "let o = JSON.parse('{\"keep\":1,\"drop\":2}', function(k,v){ return k==='drop'?undefined:v; }); o.keep + ':' + ('drop' in o)"
            ),
            "1:false"
        );
        assert_eq!(
            run(
                "JSON.parse('[1,2,3]', function(k,v){ return typeof v==='number'?v+10:v; }).join(',')"
            ),
            "11,12,13"
        );
    }

    #[test]
    fn json_stringify() {
        assert_eq!(run("JSON.stringify(42)"), "42");
        assert_eq!(run("JSON.stringify('hi')"), "\"hi\"");
        assert_eq!(run("JSON.stringify(true)"), "true");
        assert_eq!(run("JSON.stringify(null)"), "null");
        assert_eq!(run("JSON.stringify([1, 2, 3])"), "[1,2,3]");
        assert_eq!(
            run("JSON.stringify({ a: 1, b: 'x' })"),
            "{\"a\":1,\"b\":\"x\"}"
        );
        assert_eq!(
            run("JSON.stringify({ nested: { list: [1, true, null] } })"),
            "{\"nested\":{\"list\":[1,true,null]}}"
        );
        // Indentation (the `space` argument): numeric and string, empties inline.
        assert_eq!(
            run("JSON.stringify({a:1,b:2}, null, 2)"),
            "{\n  \"a\": 1,\n  \"b\": 2\n}"
        );
        assert_eq!(run("JSON.stringify([1,2], null, '--')"), "[\n--1,\n--2\n]");
        assert_eq!(run("JSON.stringify({}, null, 2)"), "{}");
        assert_eq!(run("JSON.stringify([], null, 4)"), "[]");
        // A quote in a string is escaped.
        assert_eq!(run("JSON.stringify('a\"b')"), "\"a\\\"b\"");
    }

    #[test]
    fn object_and_array_statics() {
        assert_eq!(run("Object.keys({ a: 1, b: 2 }).join(',')"), "a,b");
        assert_eq!(run("Object.values({ a: 1, b: 2 }).join(',')"), "1,2");
        assert_eq!(run("Array.isArray([1, 2])"), "true");
        assert_eq!(run("Array.isArray('nope')"), "false");
        assert_eq!(run("Array.isArray({})"), "false");
    }

    #[cfg(feature = "std")]
    #[test]
    fn math_float_methods() {
        assert_eq!(run("Math.floor(3.7)"), "3");
        assert_eq!(run("Math.ceil(3.2)"), "4");
        assert_eq!(run("Math.round(3.5)"), "4");
        assert_eq!(run("Math.sqrt(144)"), "12");
    }

    #[test]
    fn classes_and_this() {
        // A class with a constructor and a method using `this`.
        assert_eq!(
            run("class Point {
                   constructor(x, y) { this.x = x; this.y = y; }
                   sum() { return this.x + this.y; }
                 }
                 let p = new Point(3, 4);
                 p.sum()"),
            "7"
        );
        // Methods mutate instance state via `this`.
        assert_eq!(
            run("class Counter {
                   constructor() { this.n = 0; }
                   inc() { this.n += 1; return this.n; }
                 }
                 let c = new Counter();
                 c.inc(); c.inc(); c.inc()"),
            "3"
        );
        // A field initializer.
        assert_eq!(
            run("class Box { value = 42; get() { return this.value; } }
                 new Box().get()"),
            "42"
        );
        // A method calling another method on `this`.
        assert_eq!(
            run("class Calc {
                   constructor(v) { this.v = v; }
                   double() { return this.v * 2; }
                   quadruple() { return this.double() * 2; }
                 }
                 new Calc(5).quadruple()"),
            "20"
        );
        // typeof a class is function.
        assert_eq!(run("class A {} typeof A"), "function");
        // A class expression.
        assert_eq!(
            run("let C = class { constructor() { this.k = 9; } }; new C().k"),
            "9"
        );
    }

    #[test]
    fn class_getters_setters_statics() {
        // A getter computes from instance state.
        assert_eq!(
            run("class C { constructor(w, h) { this.w = w; this.h = h; }
                   get area() { return this.w * this.h; } }
                 new C(3, 4).area"),
            "12"
        );
        // A setter mutates instance state.
        assert_eq!(
            run("class Temp {
                   constructor() { this.c = 0; }
                   get fahrenheit() { return this.c * 9 / 5 + 32; }
                   set fahrenheit(f) { this.c = (f - 32) * 5 / 9; }
                 }
                 let t = new Temp(); t.fahrenheit = 212; t.c"),
            "100"
        );
        // Static methods and fields.
        assert_eq!(
            run(
                "class MathX { static square(n) { return n * n; } static pi = 3; }
                 MathX.square(5) + MathX.pi"
            ),
            "28"
        );
        // A static factory returning an instance.
        assert_eq!(
            run("class Point {
                   constructor(x) { this.x = x; }
                   static origin() { return new Point(0); }
                 }
                 Point.origin().x"),
            "0"
        );
    }

    #[test]
    fn class_inheritance() {
        // A subclass inherits a base method.
        assert_eq!(
            run("class Animal {
                   constructor(name) { this.name = name; }
                   describe() { return this.name; }
                 }
                 class Dog extends Animal {}
                 new Dog('Rex').describe()"),
            "Rex"
        );
        // `super(...)` calls the base constructor; the derived adds state.
        assert_eq!(
            run("class Animal {
                   constructor(name) { this.name = name; }
                 }
                 class Dog extends Animal {
                   constructor(name, breed) { super(name); this.breed = breed; }
                   tag() { return this.name + ':' + this.breed; }
                 }
                 new Dog('Rex', 'Lab').tag()"),
            "Rex:Lab"
        );
        // A derived method overrides the base.
        assert_eq!(
            run("class A { kind() { return 'A'; } }
                 class B extends A { kind() { return 'B'; } }
                 new B().kind() + new A().kind()"),
            "BA"
        );
        // Implicit super (no derived constructor) forwards the args.
        assert_eq!(
            run("class Base { constructor(v) { this.v = v; } }
                 class Sub extends Base { get() { return this.v; } }
                 new Sub(7).get()"),
            "7"
        );
        // Three-level chain.
        assert_eq!(
            run("class A { constructor() { this.a = 1; } }
                 class B extends A { constructor() { super(); this.b = 2; } }
                 class C extends B { constructor() { super(); this.c = 3; } }
                 let o = new C(); o.a + o.b + o.c"),
            "6"
        );
    }

    #[test]
    fn object_array_statics_and_number_methods() {
        // Object.assign / entries.
        assert_eq!(
            run("let t = Object.assign({}, { a: 1 }, { b: 2 }); t.a + t.b"),
            "3"
        );
        assert_eq!(
            run("Object.entries({ a: 1, b: 2 }).map(e => e[0] + '=' + e[1]).join(',')"),
            "a=1,b=2"
        );
        // Array.from (string / array) and Array.of.
        assert_eq!(run("Array.from('abc').join('-')"), "a-b-c");
        assert_eq!(
            run("Array.from([1, 2, 3]).map(x => x * 2).join(',')"),
            "2,4,6"
        );
        assert_eq!(run("Array.of(1, 2, 3).join(',')"), "1,2,3");
        // Number methods.
        assert_eq!(run("(255).toString()"), "255");
    }

    #[cfg(feature = "std")]
    #[test]
    fn number_tofixed() {
        assert_eq!(run("(3.14159).toFixed(2)"), "3.14");
        assert_eq!(run("(1).toFixed(3)"), "1.000");
    }

    #[test]
    fn promises_and_microtasks() {
        // The `then` reactions run on the microtask queue, drained after the
        // script — observed via captured `console.log` output.
        let out = |src: &str| {
            let program = Parser::parse_program(src).expect("parse");
            let mut interp = Interp::new();
            interp.run(&program).expect("exec");
            String::from(interp.output())
        };
        // then runs after the synchronous code.
        assert_eq!(
            out("console.log('sync');
                 Promise.resolve(42).then(v => console.log('got:' + v));"),
            "sync\ngot:42\n"
        );
        // Chaining transforms the value through each then.
        assert_eq!(
            out("Promise.resolve(1).then(v => v + 1).then(v => v * 10).then(v => console.log(v));"),
            "20\n"
        );
        // catch handles a rejection.
        assert_eq!(
            out("Promise.reject('boom').catch(e => console.log('caught:' + e));"),
            "caught:boom\n"
        );
        // A throw in a then handler rejects the chain to the next catch.
        assert_eq!(
            out(
                "Promise.resolve(1).then(() => { throw 'x'; }).catch(e => console.log('rej:' + e));"
            ),
            "rej:x\n"
        );
        // `new Promise(executor)` with resolve.
        assert_eq!(
            out("new Promise((resolve) => { resolve(7); }).then(v => console.log(v));"),
            "7\n"
        );
        // Adoption: resolving with a promise chains its value.
        assert_eq!(
            out("Promise.resolve(Promise.resolve(99)).then(v => console.log(v));"),
            "99\n"
        );
        // typeof a promise is object.
        assert_eq!(run("typeof Promise.resolve(1)"), "object");
    }

    #[test]
    fn async_await() {
        // observe results via console.log after the microtask drain.
        let out = |src: &str| {
            let program = Parser::parse_program(src).expect("parse");
            let mut interp = Interp::new();
            interp.run(&program).expect("exec");
            String::from(interp.output())
        };
        // An async function returns a promise; awaiting unwraps values.
        assert_eq!(
            out(
                "async function f() { let x = await Promise.resolve(10); return x + 5; }
                 f().then(v => console.log(v));"
            ),
            "15\n"
        );
        // Awaiting in sequence.
        assert_eq!(
            out("async function g() {
                   let a = await Promise.resolve(2);
                   let b = await Promise.resolve(3);
                   return a * b;
                 }
                 g().then(v => console.log(v));"),
            "6\n"
        );
        // try/catch around a rejected await.
        assert_eq!(
            out("async function h() {
                   try { await Promise.reject('boom'); return 'no'; }
                   catch (e) { return 'caught:' + e; }
                 }
                 h().then(v => console.log(v));"),
            "caught:boom\n"
        );
        // An async arrow, awaiting a plain value.
        assert_eq!(
            out("let f = async (x) => (await x) + 1;
                 f(41).then(v => console.log(v));"),
            "42\n"
        );
        // typeof an async function is function; its call returns a promise.
        assert_eq!(run("async function a() {} typeof a"), "function");
        assert_eq!(run("async function a() { return 1; } typeof a()"), "object");
    }

    #[cfg(feature = "regex")]
    #[test]
    fn regexp() {
        // Regex literal + test.
        assert_eq!(run("/ab+c/.test('xxabbbcyy')"), "true");
        assert_eq!(run("/^\\d+$/.test('12345')"), "true");
        assert_eq!(run("/^\\d+$/.test('12a45')"), "false");
        // case-insensitive flag.
        assert_eq!(run("/hello/i.test('HELLO')"), "true");
        // new RegExp(...).
        assert_eq!(run("new RegExp('a.c').test('axc')"), "true");
        // exec returns the matched substring (or null).
        assert_eq!(run("/b+/.exec('aabbbc')[0]"), "bbb");
        assert_eq!(run("/zzz/.exec('abc') === null"), "true");
        // A regex renders as /source/flags; typeof is object.
        assert_eq!(run("'' + /ab/gi"), "/ab/gi");
        assert_eq!(run("typeof /x/"), "object");
    }

    #[test]
    fn json_parse() {
        // Scalars.
        assert_eq!(run("JSON.parse('42')"), "42");
        assert_eq!(run("JSON.parse('true')"), "true");
        assert_eq!(run("JSON.parse('null') === null"), "true");
        assert_eq!(run("JSON.parse('\"hi\\\\nthere\"')"), "hi\nthere");
        // Arrays and objects.
        assert_eq!(run("JSON.parse('[1, 2, 3]').length"), "3");
        assert_eq!(run("JSON.parse('[1, 2, 3]')[1]"), "2");
        assert_eq!(run("JSON.parse('{\"a\": 1, \"b\": 2}').b"), "2");
        // Nested.
        assert_eq!(
            run("JSON.parse('{\"items\": [{\"id\": 7}, {\"id\": 9}]}').items[1].id"),
            "9"
        );
        // Round-trip with stringify.
        assert_eq!(
            run("let o = JSON.parse('{\"x\": 10, \"y\": 20}'); JSON.stringify(o)"),
            "{\"x\":10,\"y\":20}"
        );
        // Negative / float numbers.
        assert_eq!(run("JSON.parse('-3.5')"), "-3.5");
        // Malformed input throws (caught).
        assert_eq!(
            run("try { JSON.parse('{bad}'); 'no'; } catch (e) { 'threw'; }"),
            "threw"
        );
    }

    #[test]
    fn dates() {
        // A fixed timestamp (2021-06-15T12:30:45.500Z = 1623760245500 ms).
        let ts = "1623760245500";
        assert_eq!(
            run(&alloc::format!("new Date({ts}).getTime()")),
            "1623760245500"
        );
        assert_eq!(run(&alloc::format!("new Date({ts}).getFullYear()")), "2021");
        assert_eq!(run(&alloc::format!("new Date({ts}).getMonth()")), "5"); // June, 0-based
        assert_eq!(run(&alloc::format!("new Date({ts}).getDate()")), "15");
        assert_eq!(run(&alloc::format!("new Date({ts}).getHours()")), "12");
        assert_eq!(run(&alloc::format!("new Date({ts}).getMinutes()")), "30");
        assert_eq!(run(&alloc::format!("new Date({ts}).getSeconds()")), "45");
        assert_eq!(run(&alloc::format!("new Date({ts}).getDay()")), "2"); // Tuesday
        assert_eq!(
            run(&alloc::format!("new Date({ts}).toISOString()")),
            "2021-06-15T12:30:45.500Z"
        );
        // The epoch.
        assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
        assert_eq!(run("new Date(0).getDay()"), "4"); // Thursday
        // typeof a date is object.
        assert_eq!(run("typeof new Date(0)"), "object");
    }

    #[test]
    fn eval_source_entry_point() {
        // Captured console output + completion value.
        let (out, completion) = eval_source("console.log('hi'); 1 + 2").expect("ok");
        assert_eq!(out, "hi\n");
        assert_eq!(completion, "3");
        // A program with no trailing expression yields `undefined`.
        let (_, c) = eval_source("let x = 5;").expect("ok");
        assert_eq!(c, "undefined");
        // A thrown error surfaces as an Err.
        assert!(eval_source("throw 'boom'").is_err());
        // A parse error surfaces as an Err.
        assert!(eval_source("let = ;").is_err());
    }

    #[test]
    fn object_is_and_safe_integer_and_computed_methods() {
        assert_eq!(run("Object.is(NaN, NaN)"), "true");
        assert_eq!(run("Object.is(0, -0)"), "false");
        assert_eq!(run("Object.is(-0, -0)"), "true");
        assert_eq!(
            run("Object.is('a', 'a') + ':' + Object.is(1, 2)"),
            "true:false"
        );
        assert_eq!(run("Number.isSafeInteger(9007199254740991)"), "true");
        assert_eq!(run("Number.isSafeInteger(9007199254740992)"), "false");
        assert_eq!(run("Number.isSafeInteger(1.5)"), "false");
        // Computed class method name.
        assert_eq!(
            run("let k='go'; class C { [k](){ return 42; } } new C().go()"),
            "42"
        );
    }

    #[test]
    fn object_spread_of_array_and_string() {
        assert_eq!(
            run("let o={...[10,20,30]}; o[0] + ':' + o[2] + ':' + Object.keys(o).join(',')"),
            "10:30:0,1,2"
        );
        assert_eq!(run("let o={...'ab'}; o[0] + o[1]"), "ab");
        // Mixed with explicit keys.
        assert_eq!(
            run("let o={...[1,2], a:9}; o[0] + ':' + o[1] + ':' + o.a"),
            "1:2:9"
        );
    }

    #[test]
    fn object_spread_invokes_getters() {
        assert_eq!(
            run(
                "let s={a:1, get b(){ return this.a + 1; }}; let c={...s, d:3}; c.a + ',' + c.b + ',' + c.d"
            ),
            "1,2,3"
        );
        // Later keys win; both sources merged.
        assert_eq!(
            run("JSON.stringify({...{x:1},...{y:2},x:9})"),
            "{\"x\":9,\"y\":2}"
        );
    }

    #[test]
    fn custom_symbol_iterator() {
        // for-of and spread drive a user `[Symbol.iterator]`.
        let src = "let o = { [Symbol.iterator]() { let i = 0; return { next() { return i < 3 ? { value: i++, done: false } : { value: undefined, done: true }; } }; } };";
        assert_eq!(
            run(&alloc::format!(
                "{src} let s=[]; for (let x of o) s.push(x); s.join(',')"
            )),
            "0,1,2"
        );
        assert_eq!(run(&alloc::format!("{src} [...o].join('-')")), "0-1-2");
    }

    #[test]
    fn computed_object_literal_keys() {
        // `{ [expr]: v }` evaluates and coerces the key.
        assert_eq!(run("let k = 'a' + 'b'; let o = { [k]: 7 }; o.ab"), "7");
        // A numeric computed key coerces to its string form.
        assert_eq!(run("let o = { [1 + 1]: 'two' }; o['2']"), "two");
    }

    #[test]
    fn private_class_fields() {
        // Private fields store and read through `this.#x`.
        assert_eq!(
            run(
                "class C { #n = 0; bump(){ this.#n++; return this.#n; } } let c = new C(); c.bump(); c.bump()"
            ),
            "2"
        );
        // ...and are non-enumerable.
        assert_eq!(
            run("class C { #s = 1; constructor(){ this.p = 2; } } Object.keys(new C()).join(',')"),
            "p"
        );
    }

    #[test]
    fn bigints() {
        assert_eq!(run("typeof 5n"), "bigint");
        assert_eq!(run("(2n + 3n).toString()"), "5");
        assert_eq!(run("100n * 100n === 10000n"), "true");
        assert_eq!(run("2n ** 16n === 65536n"), "true");
        assert_eq!(run("10n / 3n === 3n"), "true");
        assert_eq!(run("-7n === 0n - 7n"), "true");
        assert_eq!(run("BigInt(99) === 99n"), "true");
        assert_eq!(run("10n === 10"), "false");
        assert_eq!(run("10n == 10"), "true");
        assert_eq!(run("!!0n"), "false");
        // Mixing BigInt and Number in arithmetic throws.
        assert_eq!(
            run("let r='ok'; try { 1n + 1; } catch (e) { r = 'threw'; } r"),
            "threw"
        );
        // Arbitrary precision: results far beyond i128 are exact.
        assert_eq!(
            run("(2n ** 200n).toString()"),
            "1606938044258990275541962092341162602522202993782792835301376"
        );
        assert_eq!(
            run("let f=1n; for(let i=1n;i<=25n;i++) f*=i; f.toString()"),
            "15511210043330985984000000"
        );
        assert_eq!(
            run("((2n ** 128n) - 1n).toString()"),
            "340282366920938463463374607431768211455"
        );
        assert_eq!(run("(~5n).toString()"), "-6");
        // Two's-complement bitwise, including beyond i128.
        assert_eq!(
            run("(12n & 10n).toString() + ',' + (12n | 10n) + ',' + (12n ^ 10n)"),
            "8,14,6"
        );
        assert_eq!(run("(-1n & 255n).toString()"), "255");
        assert_eq!(run("(((2n ** 100n) | 1n) - (2n ** 100n)).toString()"), "1");
    }

    #[test]
    fn new_on_bound_function() {
        assert_eq!(
            run(
                "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null); let p=new B(3,4); p.x + ':' + p.y"
            ),
            "3:4"
        );
        assert_eq!(
            run("function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,10); new B(20).x"),
            "10"
        );
        assert_eq!(
            run("function P(x){this.x=x;} let B=P.bind(null); (new B(1)) instanceof P"),
            "true"
        );
        // Re-bound: bound args accumulate.
        assert_eq!(
            run(
                "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,5).bind(null,6); let p=new B(); p.x + ':' + p.y"
            ),
            "5:6"
        );
        // A class can be bound and constructed.
        assert_eq!(
            run("class C{constructor(v){this.v=v;}} new (C.bind(null))(42).v"),
            "42"
        );
        assert_eq!(
            run("class C{constructor(v){this.v=v;}} new (C.bind(null,7))().v"),
            "7"
        );
    }

    #[test]
    fn apply_arraylike_and_bound_name() {
        // apply accepts an array-like (length + indexed properties).
        assert_eq!(
            run("function f(){return arguments.length;} f.apply(null,{length:3,0:1,1:2,2:3})"),
            "3"
        );
        assert_eq!(
            run(
                "function s(){let t=0;for(let i=0;i<arguments.length;i++)t+=arguments[i];return t;} s.apply(null,{length:2,0:10,1:20})"
            ),
            "30"
        );
        // A bound function's name.
        assert_eq!(run("function foo(){} foo.bind(null).name"), "bound foo");
        assert_eq!(
            run("function foo(){} foo.bind(null).bind(null).name"),
            "bound bound foo"
        );
    }

    #[test]
    fn function_length_and_name() {
        assert_eq!(run("function f(a, b, c){} f.length"), "3");
        assert_eq!(run("function f(){} f.length"), "0");
        // length counts params before the first default/rest.
        assert_eq!(run("function f(a, b = 1, c){} f.length"), "1");
        assert_eq!(run("function f(a, ...r){} f.length"), "1");
        // name from a declaration and a named function expression.
        assert_eq!(run("function greet(){} greet.name"), "greet");
        assert_eq!(run("let g = function inner(){}; g.name"), "inner");
    }

    #[test]
    fn object_seal_and_extensibility() {
        // preventExtensions: no new props, existing still writable.
        assert_eq!(
            run(
                "let o={a:1}; Object.preventExtensions(o); o.b=2; o.a=9; String(o.b) + ':' + o.a + ':' + Object.isExtensible(o)"
            ),
            "undefined:9:false"
        );
        // seal: no new props, no delete, existing writable.
        assert_eq!(
            run(
                "let o={x:1}; Object.seal(o); o.y=2; o.x=5; delete o.x; o.x + ':' + String(o.y) + ':' + Object.isSealed(o)"
            ),
            "5:undefined:true"
        );
        // freeze implies sealed + non-extensible.
        assert_eq!(
            run("let o={a:1}; Object.freeze(o); Object.isSealed(o) + ':' + Object.isExtensible(o)"),
            "true:false"
        );
    }

    #[test]
    fn non_writable_and_join_nullish() {
        // defineProperty writable:false ignores writes; descriptor reports it.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,writable:false,enumerable:true}); o.x=9; o.x"
            ),
            "1"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); Object.getOwnPropertyDescriptor(o,'x').writable"
            ),
            "false"
        );
        // Non-enumerable stays out of Object.keys but readable.
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'h',{value:2,enumerable:false}); Object.keys(o).join(',') + ':' + o.h"
            ),
            "a:2"
        );
        // Array.join renders null/undefined as empty.
        assert_eq!(run("[1,null,2,undefined,3].join('-')"), "1--2--3");
    }

    #[test]
    fn map_group_by_and_well_formed() {
        assert_eq!(
            run(
                "let g=Map.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); (g instanceof Map) + ':' + g.get('odd').join(',') + ':' + g.size"
            ),
            "true:1,3,5:2"
        );
        // Object keys are preserved (not stringified, unlike Object.groupBy).
        assert_eq!(
            run("let k={}; let g=Map.groupBy([1,2], x=>k); g.get(k).join(',')"),
            "1,2"
        );
        assert_eq!(
            run("'abc'.isWellFormed() + ':' + '\u{1f600}'.toWellFormed()"),
            "true:\u{1f600}"
        );
    }

    #[test]
    fn get_own_property_symbols_and_reflect_ownkeys() {
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={a:1}; o[s]=2; let g=Object.getOwnPropertySymbols(o); g.length + ':' + (g[0]===s) + ':' + o[g[0]]"
            ),
            "1:true:2"
        );
        assert_eq!(run("Object.getOwnPropertySymbols({}).length"), "0");
        // Reflect.ownKeys: string keys then symbol keys.
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={a:1,b:2}; o[s]=3; let k=Reflect.ownKeys(o); k.length + ':' + k[0] + ':' + (k[2]===s)"
            ),
            "3:a:true"
        );
    }

    #[test]
    fn assign_and_spread_copy_symbol_keys() {
        assert_eq!(
            run(
                "let s=Symbol('k'); let src={a:1}; src[s]=9; let t=Object.assign({},src); t.a + ':' + t[s]"
            ),
            "1:9"
        );
        assert_eq!(
            run("let s=Symbol('k'); let src={a:1}; src[s]=9; let t={...src}; t[s]"),
            "9"
        );
        // Object.keys still excludes the symbol.
        assert_eq!(
            run("let s=Symbol('k'); let src={a:1}; src[s]=9; Object.keys({...src}).join(',')"),
            "a"
        );
    }

    #[test]
    fn object_group_by() {
        assert_eq!(
            run(
                "let g=Object.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); g.odd.join(',') + '|' + g.even.join(',')"
            ),
            "1,3,5|2,4"
        );
        assert_eq!(run("Object.groupBy(['a','ab','b'], s=>s[0]).a.length"), "2");
        assert_eq!(run("Object.keys(Object.groupBy([], x=>x)).length"), "0");
        // Works over any iterable + uses the index.
        assert_eq!(run("Object.groupBy('aab', c=>c).a.length"), "2");
    }

    #[test]
    fn integer_key_ordering_and_array_tostring() {
        // Integer keys come first (ascending), then string keys in insertion order.
        assert_eq!(
            run("let o={2:'a',1:'b',3:'c'}; Object.keys(o).join(',')"),
            "1,2,3"
        );
        assert_eq!(
            run("let o={z:1, 2:2, a:3, 1:4}; Object.keys(o).join(',')"),
            "1,2,z,a"
        );
        assert_eq!(
            run("let o={}; o.b=1; o.a=2; Object.keys(o).join(',')"),
            "b,a"
        );
        assert_eq!(
            run("Object.values({10:'x',2:'y',1:'z'}).join(',')"),
            "z,y,x"
        );
        // Array toString joins with comma.
        assert_eq!(run("['a','b','c'].toString()"), "a,b,c");
        assert_eq!(run("[1,[2,3],4].toString()"), "1,2,3,4");
    }

    #[test]
    fn inherited_setter_is_called() {
        // Assigning to a property with an *inherited* setter calls it (rather
        // than shadowing it with an own data property).
        assert_eq!(
            run(
                "let base={_d:0, get c(){return this._d;}, set c(v){this._d=v;}}; let d=Object.create(base); d.c=10; d._d + ':' + base._d"
            ),
            "10:0"
        );
        // A getter-only inherited accessor shadows the data assignment.
        assert_eq!(
            run("let base={get x(){return 1;}}; let d=Object.create(base); d.x=99; d.x"),
            "1"
        );
        // An own data property still assigns normally.
        assert_eq!(run("let o={a:1}; o.a=2; o.a"), "2");
    }

    #[test]
    fn defineproperty_invariants() {
        // Redefining a non-configurable property throws; value is retained.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); try{ Object.defineProperty(o,'x',{value:2}); 'no' }catch(e){ (e instanceof TypeError)+':'+o.x }"
            ),
            "true:1"
        );
        // A configurable property can be redefined (attributes reset).
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.defineProperty(o,'x',{value:2,configurable:true}); o.x"
            ),
            "2"
        );
        // Defining a new property on a non-extensible object throws.
        assert_eq!(
            run(
                "let o={}; Object.preventExtensions(o); try{ Object.defineProperty(o,'z',{value:1}); 'no' }catch(e){ e instanceof TypeError }"
            ),
            "true"
        );
        // Non-configurable but writable: value may still change.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'w',{value:1,writable:true,configurable:false}); Object.defineProperty(o,'w',{value:2,writable:true,configurable:false}); o.w"
            ),
            "2"
        );
    }

    #[test]
    fn descriptor_reports_configurable() {
        // defineProperty defaults to non-configurable.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1}); Object.getOwnPropertyDescriptor(o,'x').configurable"
            ),
            "false"
        );
        // Explicit configurable: true is reported.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.getOwnPropertyDescriptor(o,'x').configurable"
            ),
            "true"
        );
        // A plain literal property is configurable; a frozen one is not.
        assert_eq!(
            run("Object.getOwnPropertyDescriptor({a:1},'a').configurable"),
            "true"
        );
        assert_eq!(
            run("Object.getOwnPropertyDescriptor(Object.freeze({a:1}),'a').configurable"),
            "false"
        );
    }

    #[test]
    fn delete_respects_configurable() {
        assert_eq!(run("let o={a:1}; delete o.a"), "true");
        assert_eq!(run("let o={}; delete o.missing"), "true");
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x"),
            "false"
        );
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x; o.x"
            ),
            "1"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'y',{value:2,configurable:true}); delete o.y"),
            "true"
        );
        assert_eq!(run("let o=Object.freeze({a:1}); delete o.a"), "false");
    }

    #[test]
    fn redefine_accessor_as_data() {
        // Accessor → accessor.
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{get(){return 2;},configurable:true}); o.x"
            ),
            "2"
        );
        // Accessor → data property (the old getter must no longer apply).
        assert_eq!(
            run(
                "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{value:42,configurable:true}); o.x"
            ),
            "42"
        );
    }

    #[test]
    fn enumerable_accessor_keys() {
        // Object-literal getters are enumerable (appear in Object.keys/JSON).
        assert_eq!(
            run("Object.keys({x:1, get y(){return 2;}}).join(',')"),
            "x,y"
        );
        assert_eq!(
            run("JSON.stringify({a:1, get b(){return 2;}})"),
            "{\"a\":1,\"b\":2}"
        );
        // defineProperty accessor: enumerable per the descriptor.
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;},enumerable:true}); Object.keys(o).join(',')"
            ),
            "a,c"
        );
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;}}); Object.keys(o).join(',')"
            ),
            "a"
        );
        // Class accessors are non-enumerable.
        assert_eq!(
            run(
                "class C{ constructor(){this.a=1;} get b(){return 2;} } Object.keys(new C()).join(',')"
            ),
            "a"
        );
    }

    #[test]
    fn get_own_property_descriptors() {
        assert_eq!(
            run(
                "let o={a:1}; Object.defineProperty(o,'b',{value:2,writable:false,enumerable:true}); let d=Object.getOwnPropertyDescriptors(o); d.a.value + ',' + d.a.writable + ',' + d.b.value + ',' + d.b.writable"
            ),
            "1,true,2,false"
        );
        assert_eq!(
            run("Object.keys(Object.getOwnPropertyDescriptors({a:1,b:2})).join(',')"),
            "a,b"
        );
        assert_eq!(
            run(
                "let o={get x(){return 5;}}; let d=Object.getOwnPropertyDescriptors(o); typeof d.x.get"
            ),
            "function"
        );
    }

    #[test]
    fn object_define_properties() {
        assert_eq!(
            run(
                "let o={}; Object.defineProperties(o, { x:{value:1}, y:{get:function(){return 2;}} }); o.x + ',' + o.y"
            ),
            "1,2"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'a',{value:42}); o.a"),
            "42"
        );
    }

    #[test]
    fn computed_key_destructuring() {
        // Declaration form.
        assert_eq!(
            run("let k='name'; let {[k]: v} = {name:'Alice'}; v"),
            "Alice"
        );
        assert_eq!(
            run("let p='x'; let {[p]: a, ...rest} = {x:1, y:2}; a + ':' + rest.y"),
            "1:2"
        );
        // Assignment form.
        assert_eq!(
            run("let k='m'; let v; ({[k]: v} = {m:42}); String(v)"),
            "42"
        );
    }

    #[test]
    fn destructuring_assignment_with_defaults() {
        assert_eq!(run("let a,b; [a,b]=[1,2]; a+','+b"), "1,2");
        assert_eq!(run("let a,b; [a,b]=[1,2]; [a,b]=[b,a]; a+','+b"), "2,1");
        assert_eq!(run("let a,b; ({x:a,y:b}={x:10,y:20}); a+','+b"), "10,20");
        // Default in an assignment pattern.
        assert_eq!(run("let a,b,c; [a,b,c=99]=[1,2]; String(c)"), "99");
        assert_eq!(run("let x; ({p:x=7}={}); String(x)"), "7");
    }

    #[test]
    fn date_multi_arg_and_subtraction() {
        assert_eq!(
            run("let d=new Date(2026,5,5); d.getFullYear()+'/'+d.getMonth()+'/'+d.getDate()"),
            "2026/5/5"
        );
        assert_eq!(run("let d=new Date(0); d.getTime()"), "0");
        assert_eq!(run("(new Date(2000)) - (new Date(1000))"), "1000");
    }

    #[test]
    fn utf16_string_indexing() {
        assert_eq!(run("'café'.length"), "4");
        assert_eq!(run("'\\u{1F600}'.length"), "2");
        assert_eq!(run("'a\\u{1F600}b'.length"), "4");
        assert_eq!(run("'\\u{1F600}'.charCodeAt(0)"), "55357");
        assert_eq!(run("'\\u{1F600}'.charCodeAt(1)"), "56832");
        assert_eq!(run("'\\u{1F600}'.codePointAt(0)"), "128512");
        assert_eq!(run("'a\\u{1F600}b'.codePointAt(1)"), "128512");
        assert_eq!(run("'hello'.charCodeAt(0)"), "104");
    }

    #[test]
    fn array_call_and_unary_plus_array() {
        // Array(...) without new.
        assert_eq!(run("Array(3).length"), "3");
        assert_eq!(run("Array(1,2,3).join(',')"), "1,2,3");
        assert_eq!(run("Array().length"), "0");
        // Unary + coerces arrays via their string form.
        assert_eq!(run("+[]"), "0");
        assert_eq!(run("+[5]"), "5");
        assert_eq!(run("Number.isNaN(+[1,2])"), "true");
        // Symbol.toPrimitive still gets the number hint for unary +.
        assert_eq!(
            run("+{[Symbol.toPrimitive](h){ return h==='number'?9:0; }}"),
            "9"
        );
    }

    #[test]
    fn reverse_inplace_new_array_string_index() {
        // reverse mutates in place and returns the same array.
        assert_eq!(
            run("let a=[1,2,3]; let b=a.reverse(); (a===b) + ':' + a.join(',')"),
            "true:3,2,1"
        );
        // new Array(n) and new Array(...elements).
        assert_eq!(run("new Array(3).fill(7).join(',')"), "7,7,7");
        assert_eq!(run("new Array(1,2,3).join(',')"), "1,2,3");
        assert_eq!(run("new Array(3).length"), "3");
        // String index access.
        assert_eq!(run("'hello'[0] + 'hello'[4]"), "ho");
        assert_eq!(run("String('abc'[5])"), "undefined");
    }

    #[test]
    fn array_immutable_methods() {
        assert_eq!(
            run("let a=[3,1,2]; a.toSorted().join(',') + '|' + a.join(',')"),
            "1,2,3|3,1,2"
        );
        assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
        assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
        assert_eq!(run("[1,2,3].with(-1,9).join(',')"), "1,2,9");
        assert_eq!(
            run("[1,2,3,4,5].toSpliced(1,2,'a','b').join(',')"),
            "1,a,b,4,5"
        );
        assert_eq!(run("[1,2,3,4].toSpliced(2).join(',')"), "1,2");
        // with out-of-range → RangeError.
        assert_eq!(
            run("try { [1,2,3].with(10,0); 'no' } catch(e){ e instanceof RangeError }"),
            "true"
        );
    }

    #[test]
    fn reduce_args_and_sort_in_place() {
        // reduce callback gets (acc, cur, index, array).
        assert_eq!(
            run(
                "let ix=[]; [10,20,30].reduce(function(a,c,i,arr){ ix.push(i + ':' + arr.length); return a+c; }, 0); ix.join(',')"
            ),
            "0:3,1:3,2:3"
        );
        assert_eq!(
            run("['a','b','c'].reduceRight(function(a,c){return a+c;})"),
            "cba"
        );
        // sort is in place and returns the same array.
        assert_eq!(
            run("let a=[3,1,2]; let b=a.sort(); (a===b) + ':' + a.join(',')"),
            "true:1,2,3"
        );
        assert_eq!(run("[3,1,2].sort((x,y)=>y-x).join(',')"), "3,2,1");
    }

    #[cfg(feature = "intl")]
    #[test]
    fn string_normalize_forms() {
        // "é" composed (1 cp) vs decomposed (e + U+0301).
        assert_eq!(run("'\u{e9}'.normalize('NFD').length"), "2");
        assert_eq!(run("'e\u{301}'.normalize('NFC').length"), "1");
        assert_eq!(
            run("'\u{e9}'.normalize() === 'e\u{301}'.normalize()"),
            "true"
        );
        // NFKC expands the ﬁ ligature.
        assert_eq!(run("'\u{fb01}'.normalize('NFKC')"), "fi");
        assert_eq!(run("'abc'.normalize()"), "abc");
        // An unsupported form throws a RangeError *object* (not a bare string).
        assert_eq!(
            run("try{'x'.normalize('BAD');'no'}catch(e){e instanceof RangeError}"),
            "true"
        );
        assert_eq!(
            run("try{'x'.normalize('BAD');'no'}catch(e){e.name}"),
            "RangeError"
        );
    }

    #[test]
    fn string_raw_and_member_tag() {
        assert_eq!(run("String.raw`a\\nb`"), "a\\nb");
        assert_eq!(run("String.raw`${1}+${2}=${3}`"), "1+2=3");
        assert_eq!(run("String.raw`line\\tend`.length"), "9"); // backslash + t kept raw
        // A tag read as a member of a plain object also dispatches.
        assert_eq!(
            run("let o={ t(s){ return s.raw[0]; } }; o.t`x\\ny`"),
            "x\\ny"
        );
    }

    #[test]
    fn generator_return_value() {
        // The return value is surfaced once, with done:true, after the yields.
        assert_eq!(
            run(
                "function* g(){ yield 1; yield 2; return 99; } let it=g(); it.next(); it.next(); let r=it.next(); r.value + ':' + r.done"
            ),
            "99:true"
        );
        // Subsequent next() calls yield undefined/done.
        assert_eq!(
            run(
                "function* g(){ yield 1; return 7; } let it=g(); it.next(); it.next(); String(it.next().value) + ':' + it.next().done"
            ),
            "undefined:true"
        );
        // Spread excludes the return value.
        assert_eq!(
            run("function* g(){ yield 1; yield 2; return 9; } [...g()].join(',')"),
            "1,2"
        );
    }

    #[test]
    fn array_iterators() {
        assert_eq!(run("[...['a','b','c'].keys()].join(',')"), "0,1,2");
        assert_eq!(run("[...['a','b','c'].values()].join(',')"), "a,b,c");
        assert_eq!(
            run("let o=[]; for (let [i,v] of ['x','y'].entries()) o.push(i+':'+v); o.join(',')"),
            "0:x,1:y"
        );
        // The iterator supports next().
        assert_eq!(
            run("let it=['p','q'].values(); it.next().value + it.next().value"),
            "pq"
        );
    }

    #[test]
    fn matchall_replaceall_require_global() {
        assert_eq!(
            run("try{ 'aaa'.replaceAll(/a/,'b'); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(
            run("try{ [...'aaa'.matchAll(/a/)]; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("'aaa'.replaceAll(/a/g,'b')"), "bbb");
        assert_eq!(run("[...'a1b2'.matchAll(/[a-z]\\d/g)].length"), "2");
        assert_eq!(
            run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, '$<m>/$<y>')"),
            "06/2024"
        );
    }

    #[test]
    fn replace_groups_and_split_limit() {
        // The replace function receives a `groups` object for named captures.
        assert_eq!(
            run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, (m,y,mo,o,s,g)=>g.y+'/'+g.m)"),
            "2024/06"
        );
        // Regex split honors the limit.
        assert_eq!(run("'a1b2c3'.split(/(\\d)/,3).join('|')"), "a|1|b");
        // Empty-regex split has no trailing empty; capture split keeps its trailing.
        assert_eq!(run("'abc'.split(/(?:)/).join(',')"), "a,b,c");
        assert_eq!(run("'a1'.split(/(\\d)/).length"), "3");
    }

    #[test]
    fn regex_unicode_property_categories() {
        // Robust across the intl / no-intl matchers.
        assert_eq!(run(r#"'Hello World'.match(/\p{Lu}/g).join('')"#), "HW");
        assert_eq!(run(r#"'Hello'.match(/\p{Ll}/g).join('')"#), "ello");
        assert_eq!(run(r#"'abc123'.match(/\p{N}/g).join('')"#), "123");
        assert_eq!(run(r#"'a.b!c'.match(/\p{P}/g).join('')"#), ".!");
        assert_eq!(run(r#"'中文字'.match(/\p{Lo}/g).length"#), "3");
        assert_eq!(run(r#"'a1b2'.match(/\P{N}/g).join('')"#), "ab");
        // The full subcategory set compiles (matching may need Unicode tables).
        assert_eq!(
            run(r#"'x'.match(/\p{Sm}|\p{Sc}|\p{Mn}|\p{Pd}/g)===null"#),
            "true"
        );
    }

    #[cfg(feature = "intl")]
    #[test]
    fn regex_unicode_property_precise_with_intl() {
        assert_eq!(run(r#"'3+5'.match(/\p{Sm}/)[0]"#), "+");
        assert_eq!(run(r#"'$5'.match(/\p{Sc}/)[0]"#), "$");
        assert_eq!(run(r#"'(a)'.match(/\p{Ps}/)[0]"#), "(");
        assert_eq!(run(r#"'a-b'.match(/\p{Pd}/)[0]"#), "-");
    }

    #[test]
    fn regex_on_multibyte_strings() {
        // These previously panicked (char-index spans used as byte ranges).
        assert_eq!(run("'café'.match(/é/)[0]"), "é");
        assert_eq!(run("'café'.match(/(.+)/)[1]"), "café");
        assert_eq!(run("'café'.replace(/é/, 'e')"), "cafe");
        assert_eq!(run("'a→b→c'.split(/→/).join('|')"), "a|b|c");
        assert_eq!(run("'über 123'.match(/\\d+/)[0]"), "123");
        assert_eq!(run("'café'.match(/(?<r>.+)/).groups.r"), "café");
        assert_eq!(run("[...'café déjà'.matchAll(/é/g)].length"), "2");
    }

    #[test]
    fn regex_empty_and_zerowidth_matches() {
        // Empty-match global replace keeps the characters.
        assert_eq!(run("'abc'.replace(/x*/g, '-')"), "-a-b-c-");
        // Zero-width (lookahead) split keeps the boundary character.
        assert_eq!(
            run("'camelCaseWord'.split(/(?=[A-Z])/).join('|')"),
            "camel|Case|Word"
        );
        // Capture-group split still splices the captures.
        assert_eq!(run("'a1b2'.split(/(\\d)/).join(',')"), "a,1,b,2,");
        // Lookahead-based number grouping replace.
        assert_eq!(
            run("'1234567'.replace(/(?<=\\d)(?=(?:\\d{3})+$)/g, ',')"),
            "1,234,567"
        );
    }

    #[test]
    fn replace_dollar_patterns() {
        // String-pattern replace.
        assert_eq!(run("'hello'.replace('l', '[$&]')"), "he[l]lo");
        assert_eq!(run("'abc'.replace('b', '$`')"), "aac"); // prefix
        assert_eq!(run("'abc'.replace('b', \"$'\")"), "acc"); // suffix
        assert_eq!(run("'test'.replaceAll('t', '$$')"), "$es$"); // literal $
        // Regex-pattern replace.
        assert_eq!(
            run("'2024-06'.replace(/(\\d+)-(\\d+)/, '$2/$1')"),
            "06/2024"
        );
        assert_eq!(run("'x'.replace(/x/, '$1')"), "$1"); // no group 1 → literal
        assert_eq!(run("'abc'.replace(/b/, '$`')"), "aac");
    }

    #[test]
    fn regex_stateful_last_index() {
        // Global exec advances lastIndex and resets on a miss.
        assert_eq!(
            run(
                "let r=/\\d/g; r.exec('a1b2')[0] + ':' + r.lastIndex + ':' + r.exec('a1b2')[0] + ':' + r.lastIndex"
            ),
            "1:2:2:4"
        );
        assert_eq!(
            run("let r=/\\d/g; r.exec('a1'); String(r.exec('a1')) + ':' + r.lastIndex"),
            "null:0"
        );
        // Writing lastIndex resumes from there.
        assert_eq!(run("let r=/\\d/g; r.lastIndex=3; r.exec('12345')[0]"), "4");
        // test() advances; non-global never does.
        assert_eq!(run("let r=/x/g; r.test('axbx'); r.lastIndex"), "2");
        assert_eq!(run("let r=/\\d/; r.exec('a1'); r.lastIndex"), "0");
    }

    #[test]
    fn regex_lookbehind() {
        assert_eq!(run("/(?<=\\$)\\d+/.test('$100')"), "true");
        assert_eq!(run("/(?<=\\$)\\d+/.test('100')"), "false");
        assert_eq!(run("'$100'.match(/(?<=\\$)\\d+/)[0]"), "100");
        assert_eq!(
            run("'price: $50'.replace(/(?<=\\$)\\d+/, 'X')"),
            "price: $X"
        );
        assert_eq!(run("/(?<!a)b/.test('ab')"), "false");
        assert_eq!(run("/(?<!a)b/.test('xb')"), "true");
    }

    #[test]
    fn regex_lookahead_and_backref() {
        // Positive / negative lookahead (zero-width).
        assert_eq!(run("/foo(?=bar)/.test('foobar')"), "true");
        assert_eq!(run("/foo(?=bar)/.test('foobaz')"), "false");
        assert_eq!(run("/foo(?!bar)/.test('foobaz')"), "true");
        assert_eq!(run("'foobar'.replace(/foo(?=bar)/, 'X')"), "Xbar");
        // Backreferences.
        assert_eq!(run("/(\\w)\\1/.test('hello')"), "true");
        assert_eq!(run("/(\\w)\\1/.test('abc')"), "false");
        assert_eq!(run("'hello'.match(/(.)\\1/)[0]"), "ll");
    }

    #[test]
    fn regex_named_groups() {
        assert_eq!(
            run(
                "let m='2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/); m.groups.y + ':' + m.groups.mo"
            ),
            "2024:06"
        );
        // Still positionally indexable.
        assert_eq!(
            run("'2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/)[1]"),
            "2024"
        );
        // Named backreference in replacement.
        assert_eq!(
            run("'John Smith'.replace(/(?<first>\\w+) (?<last>\\w+)/, '$<last>, $<first>')"),
            "Smith, John"
        );
        // No named groups → .groups is undefined.
        assert_eq!(run("String('ab'.match(/(a)(b)/).groups)"), "undefined");
    }

    #[test]
    fn match_all_named_groups() {
        assert_eq!(
            run(
                "let m=[...'2024-06 2025-12'.matchAll(/(?<y>\\d{4})-(?<mo>\\d{2})/g)]; m[0].groups.y + ':' + m[1].groups.mo"
            ),
            "2024:12"
        );
        // Positional access + index still work on matchAll results.
        assert_eq!(run("[...'a1b2'.matchAll(/([a-z])(\\d)/g)][1][2]"), "2");
        assert_eq!(
            run("[...'xy'.matchAll(/(?<c>.)/g)].map(m=>m.groups.c).join('')"),
            "xy"
        );
    }

    #[test]
    fn string_match_all() {
        assert_eq!(run("[...'a1b2c3'.matchAll(/([a-z])(\\d)/g)].length"), "3");
        assert_eq!(
            run(
                "let m=[...'a1b2'.matchAll(/([a-z])(\\d)/g)]; m[0][0] + ':' + m[0][1] + ':' + m[0][2]"
            ),
            "a1:a:1"
        );
        assert_eq!(
            run("[...'hello world'.matchAll(/\\w+/g)].map(m=>m[0]).join(',')"),
            "hello,world"
        );
        assert_eq!(run("[...'abc'.matchAll(/\\d/g)].length"), "0");
    }

    #[test]
    fn array_thisarg_and_split_captures() {
        assert_eq!(
            run("[1,2,3].map(function(x){return x*this.m;},{m:3}).join(',')"),
            "3,6,9"
        );
        assert_eq!(
            run("[1,2,3,4].filter(function(x){return x>this.t;},{t:2}).join(',')"),
            "3,4"
        );
        assert_eq!(
            run("[1,2,3].some(function(x){return x===this.g;},{g:2})"),
            "true"
        );
        assert_eq!(
            run("[1,2,3].every(function(x){return x<=this.mx;},{mx:3})"),
            "true"
        );
        // split with a capturing separator splices the captures in.
        assert_eq!(run("'a1b2c3'.split(/(\\d)/).join('|')"), "a|1|b|2|c|3|");
    }

    #[test]
    fn array_last_index_of_from_index() {
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(20)"), "3");
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(10,3)"), "0");
        assert_eq!(run("[10,20,30,20,10].lastIndexOf(20,-3)"), "1");
        assert_eq!(run("[1,2,3].lastIndexOf(9)"), "-1");
        assert_eq!(run("[1,2,3,4].findLastIndex(x => x < 3)"), "1");
    }

    #[test]
    fn frozen_object_blocks_delete() {
        assert_eq!(
            run("let o={a:1,b:2}; Object.freeze(o); delete o.b; o.b"),
            "2"
        );
        assert_eq!(
            run("let o={a:1}; Object.freeze(o); o.a=9; o.c=3; o.a + ':' + String(o.c)"),
            "1:undefined"
        );
        assert_eq!(run("let o={a:1,b:2}; delete o.b; String(o.b)"), "undefined");
    }

    #[test]
    fn array_and_function_named_properties() {
        assert_eq!(
            run("let a=[1,2,3]; a.tag='x'; a.tag + ':' + a.length + ':' + a[0]"),
            "x:3:1"
        );
        assert_eq!(run("let a=[1]; a.tag='y'; a.hasOwnProperty('tag')"), "true");
        assert_eq!(run("function f(){} f.meta=42; f.meta"), "42");
        // Tagged template strings carry `.raw`.
        assert_eq!(run("function t(s){ return s.raw[0]; } t`a\\tb`"), "a\\tb");
    }

    #[test]
    fn error_to_string() {
        assert_eq!(run("new Error('boom').toString()"), "Error: boom");
        assert_eq!(run("new TypeError('bad').toString()"), "TypeError: bad");
        assert_eq!(run("new Error().toString()"), "Error");
        // A user toString still wins.
        assert_eq!(
            run("({ name:'X', message:'y', toString(){ return 'custom'; } }).toString()"),
            "custom"
        );
    }

    #[test]
    fn symbol_to_primitive_hints() {
        let o =
            "let o={[Symbol.toPrimitive](h){ return h==='number'?42:h==='string'?'str':'def'; }};";
        assert_eq!(run(&alloc::format!("{o} +o")), "42");
        assert_eq!(run(&alloc::format!("{o} `${{o}}`")), "str");
        assert_eq!(run(&alloc::format!("{o} o + ''")), "def");
        // Symbol.toPrimitive takes precedence over valueOf/toString.
        assert_eq!(
            run("let o={[Symbol.toPrimitive](){ return 9; }, valueOf(){ return 1; }}; o + 0"),
            "9"
        );
    }

    #[test]
    fn loose_equality_object_coercion() {
        assert_eq!(run("[] == 0"), "true");
        assert_eq!(run("[1] == 1"), "true");
        assert_eq!(run("[1,2] == '1,2'"), "true");
        assert_eq!(run("({}) == ({})"), "false"); // distinct objects
        assert_eq!(run("let o={valueOf(){return 5;}}; o == 5"), "true");
        assert_eq!(run("'' == 0"), "true");
        assert_eq!(run("null == 0"), "false");
    }

    #[test]
    fn to_primitive_in_operators() {
        assert_eq!(run("let o={valueOf(){return 42;}}; o + 8"), "50");
        assert_eq!(run("let o={valueOf(){return 6;}}; o * 7"), "42");
        assert_eq!(run("let o={toString(){return 'x';}}; '' + o"), "x");
        // valueOf is preferred for the default/number hint.
        assert_eq!(
            run("let o={valueOf(){return 5;}, toString(){return 'five';}}; o + 1"),
            "6"
        );
        // Identity comparison does not coerce.
        assert_eq!(run("let o={valueOf(){return 1;}}; o === o"), "true");
    }

    #[test]
    fn template_invokes_custom_tostring() {
        assert_eq!(
            run("let o = { toString() { return 'custom'; } }; `val=${o}`"),
            "val=custom"
        );
        // A plain object with no toString still renders the default form.
        assert_eq!(run("`${ {a:1} }`"), "[object Object]");
        // Arrays/numbers/booleans coerce as usual.
        assert_eq!(run("`${[1,2,3]}-${true}-${null}`"), "1,2,3-true-null");
    }

    #[test]
    fn coercion_string_number_join_freeze_tofixed() {
        // String()/Number() honor toString/valueOf; join too.
        assert_eq!(run("String({toString(){return 'x';}})"), "x");
        assert_eq!(run("Number({valueOf(){return 42;}})"), "42");
        assert_eq!(
            run("[{toString(){return 'a';}},{toString(){return 'b';}}].join(',')"),
            "a,b"
        );
        // Frozen array rejects push.
        assert_eq!(
            run("let a=[1,2,3]; Object.freeze(a); a.push(4); a.length + ':' + Object.isFrozen(a)"),
            "3:true"
        );
        // toFixed rounds half away from zero.
        assert_eq!(run("(0.5).toFixed(0)"), "1");
        assert_eq!(run("(2.5).toFixed(0)"), "3");
        assert_eq!(run("(123.456).toFixed(2)"), "123.46");
    }

    #[test]
    fn math_abs_round_and_create_descriptors() {
        // Math.round rounds half toward +Infinity (not away from zero).
        assert_eq!(run("Math.round(-2.5)"), "-2");
        assert_eq!(run("Math.round(2.5)"), "3");
        assert_eq!(run("Math.round(-0.5) === 0"), "true");
        // Math.abs(-0) is +0.
        assert_eq!(run("Object.is(Math.abs(-0), 0)"), "true");
        // Object.create with a descriptors map.
        assert_eq!(
            run(
                "let p={g(){return 'hi';}}; let o=Object.create(p, {n:{value:5,enumerable:true}}); o.n + ':' + o.g() + ':' + Object.keys(o).join(',')"
            ),
            "5:hi:n"
        );
    }

    #[test]
    fn negative_zero_stringifies_to_zero() {
        assert_eq!(run("String(-0)"), "0");
        assert_eq!(run("(-0).toString()"), "0");
        assert_eq!(run("'' + -0"), "0");
        assert_eq!(run("`${-0}`"), "0");
        assert_eq!(run("[-0, 0].join(',')"), "0,0");
        // But Object.is still distinguishes the bit pattern.
        assert_eq!(run("Object.is(-0, 0)"), "false");
    }

    #[test]
    fn math_minus_zero_indexof_fromindex_number_exponential() {
        // Math.max/min treat +0 > -0.
        assert_eq!(run("Object.is(Math.max(-0, 0), 0)"), "true");
        assert_eq!(run("Object.is(Math.min(0, -0), -0)"), "true");
        // String.indexOf honors fromIndex.
        assert_eq!(run("'hello world'.indexOf('o', 5)"), "7");
        assert_eq!(run("'hello world'.indexOf('o')"), "4");
        // Number.toString exponential thresholds.
        assert_eq!(run("(1e21).toString()"), "1e+21");
        assert_eq!(run("(1e-7).toString()"), "1e-7");
        assert_eq!(run("(1e20).toString()"), "100000000000000000000"); // not exponential
        assert_eq!(run("(0.000001).toString()"), "0.000001"); // 1e-6 stays decimal
    }

    #[test]
    fn math_trig_and_extra() {
        assert_eq!(
            run("Math.sin(0) + ':' + Math.cos(0) + ':' + Math.tan(0)"),
            "0:1:0"
        );
        assert_eq!(run("Math.round(Math.sin(Math.PI/2))"), "1");
        assert_eq!(run("Math.round(Math.atan2(1,1)*4/Math.PI)"), "1");
        assert_eq!(
            run("Math.cosh(0) + ':' + Math.tanh(0) + ':' + Math.expm1(0)"),
            "1:0:0"
        );
        assert_eq!(
            run("Math.fround(1.5) + ':' + (Math.fround(1.1) !== 1.1)"),
            "1.5:true"
        );
        assert_eq!(run("Math.clz32(1) + ':' + Math.clz32(0)"), "31:32");
        assert_eq!(run("Math.imul(3,4) + ':' + Math.imul(-1,8)"), "12:-8");
    }

    #[test]
    fn number_formatting() {
        assert_eq!(run("(3.5).toString(2)"), "11.1");
        assert_eq!(run("(255.5).toString(16)"), "ff.8");
        assert_eq!(run("(-255.5).toString(16)"), "-ff.8");
        assert_eq!(run("(12345).toPrecision(1)"), "1e+4");
        assert_eq!(run("(0.0000001234).toPrecision(2)"), "1.2e-7");
        assert_eq!(run("(1234567).toLocaleString()"), "1,234,567");
        assert_eq!(run("(-1234.5).toLocaleString()"), "-1,234.5");
    }

    #[test]
    fn math_random_in_range() {
        // In [0, 1), and consecutive calls differ (the PRNG advances).
        assert_eq!(run("let a=Math.random(); a >= 0 && a < 1"), "true");
        assert_eq!(run("Math.random() !== Math.random()"), "true");
        assert_eq!(
            run(
                "let xs=[]; for (let i=0;i<100;i++) xs.push(Math.random()); xs.every(x=>x>=0&&x<1)"
            ),
            "true"
        );
    }

    #[test]
    fn math_constants() {
        assert_eq!(run("Math.PI > 3.14 && Math.PI < 3.15"), "true");
        assert_eq!(run("Math.E > 2.71 && Math.E < 2.72"), "true");
        assert_eq!(run("Math.SQRT2 * Math.SQRT2 > 1.999"), "true");
        assert_eq!(run("Math.floor(Math.LN2 * 1000)"), "693");
    }

    #[test]
    fn private_in_brand_check() {
        assert_eq!(
            run(
                "class H{ #s=1; static check(o){ return #s in o; } } H.check(new H()) + ':' + H.check({})"
            ),
            "true:false"
        );
        // Works for an inherited brand too (subclass instances have the field).
        assert_eq!(
            run(
                "class H{ #s=1; static check(o){ return #s in o; } } class D extends H{} H.check(new D())"
            ),
            "true"
        );
    }

    #[test]
    fn class_static_blocks() {
        assert_eq!(
            run("class C{ static x=1; static { C.y = C.x + 1; } } C.y"),
            "2"
        );
        // Multiple blocks run in order.
        assert_eq!(
            run("class C{ static n=0; static { C.n=10; } static { C.n+=5; } } C.n"),
            "15"
        );
        // `this` is the class inside a static block.
        assert_eq!(
            run("class C{ static x=1; static { this.y = this.x + 100; } } C.y"),
            "101"
        );
    }

    #[test]
    fn static_setters_and_symbol_description() {
        // Static setter then getter.
        assert_eq!(
            run(
                "class T{ static _c=0; static get c(){return T._c;} static set c(v){T._c=v;} } T.c=25; T.c"
            ),
            "25"
        );
        // Symbol description: undefined for no-arg, the string otherwise.
        assert_eq!(run("String(Symbol().description)"), "undefined");
        assert_eq!(run("Symbol('d').description"), "d");
        assert_eq!(run("Symbol('').description"), ""); // explicit empty
    }

    #[test]
    fn object_hasown_static_accessors_replaceall_fn() {
        // Object.hasOwn.
        assert_eq!(
            run("Object.hasOwn({a:1},'a') + ':' + Object.hasOwn({a:1},'b')"),
            "true:false"
        );
        assert_eq!(run("Object.hasOwn(Object.create({x:1}),'x')"), "false");
        // Static field write-back and static getter.
        assert_eq!(
            run(
                "class C{ static n=0; static inc(){ return ++C.n; } static get cur(){ return C.n; } } C.inc(); C.inc(); C.cur"
            ),
            "2"
        );
        // replaceAll with a function replacer.
        assert_eq!(
            run("'AAA'.replaceAll('A', function(){ return 'B'; })"),
            "BBB"
        );
        assert_eq!(
            run("'a1b2'.replace('1', function(m){ return '['+m+']'; })"),
            "a[1]b2"
        );
    }

    #[test]
    fn object_reflection_and_static_inheritance() {
        assert_eq!(run("({a:1}).hasOwnProperty('a')"), "true");
        assert_eq!(run("({a:1}).hasOwnProperty('b')"), "false");
        // Static methods are inherited down the `extends` chain.
        assert_eq!(
            run("class A { static make(){ return 'made'; } } class B extends A {} B.make()"),
            "made"
        );
        // `static m(){ return new this(); }` uses the receiver class.
        assert_eq!(
            run(
                "class A { static create(){ return new this(); } get tag(){ return 'a'; } } class B extends A {} B.create().tag"
            ),
            "a"
        );
        // String.raw interleaves a raw-bearing object with substitutions.
        assert_eq!(run("String.raw({ raw: ['a','b','c'] }, 1, 2)"), "a1b2c");
    }

    #[test]
    fn constructor_function_prototype() {
        // Method on the prototype, resolved through the instance.
        assert_eq!(
            run(
                "function A(n){this.n=n;} A.prototype.m=function(){return this.n*2;}; new A(5).m()"
            ),
            "10"
        );
        // Two-level prototype chain via Object.create.
        assert_eq!(
            run(
                "function A(){} A.prototype.greet=function(){return 'hi';}; function B(){} B.prototype=Object.create(A.prototype); new B().greet()"
            ),
            "hi"
        );
        // `.prototype` is a stable object across reads.
        assert_eq!(run("function A(){} A.prototype.x=1; A.prototype.x"), "1");
    }

    #[test]
    fn named_function_expression_recurses() {
        assert_eq!(
            run("let f = function fac(n){ return n <= 1 ? 1 : n * fac(n-1); }; f(5)"),
            "120"
        );
        // The name is scoped to the expression, not visible outside.
        assert_eq!(
            run("let f = function self(n){ return n===0?0:n+self(n-1); }; f(4)"),
            "10"
        );
    }

    #[test]
    fn array_length_assignment_resizes() {
        assert_eq!(run("let a=[1,2,3,4,5]; a.length=3; a.join(',')"), "1,2,3");
        assert_eq!(
            run("let a=[1,2]; a.length=4; String(a[3]) + ':' + a.length"),
            "undefined:4"
        );
        assert_eq!(run("let a=[1,2,3]; a.length=0; a.length"), "0");
        // String.fromCodePoint.
        assert_eq!(run("String.fromCodePoint(97, 98, 99)"), "abc");
    }

    #[test]
    fn var_hoisting() {
        // A `var` read before its declaration line yields `undefined`.
        assert_eq!(
            run("function f(){ var a = b; var b = 5; return String(a); } f()"),
            "undefined"
        );
        assert_eq!(
            run("function f(){ return typeof later; var later = 1; } f()"),
            "undefined"
        );
        // A var inside a block still hoists to the function scope.
        assert_eq!(run("function f(){ { var x = 9; } return x; } f()"), "9");
    }

    #[test]
    fn arrow_inherits_lexical_this() {
        assert_eq!(
            run("let o = { v: 42, m: function(){ let f = () => this.v; return f(); } }; o.m()"),
            "42"
        );
        // Nested arrows keep inheriting.
        assert_eq!(
            run("let o = { v: 7, m: function(){ return (() => (() => this.v)())(); } }; o.m()"),
            "7"
        );
    }

    #[test]
    fn computed_class_members() {
        // Computed instance method, field, and getter names.
        assert_eq!(
            run(
                "let m='go'; class C{ [m](){return 1;} [m+'V']=2; get [m+'G'](){return 3;} } let c=new C(); c.go() + ':' + c.goV + ':' + c.goG"
            ),
            "1:2:3"
        );
        // Computed static method, field, and getter names.
        assert_eq!(
            run(
                "let s='mk'; class C{ static [s](){return 'a';} static [s+'N']=4; static get [s+'G'](){return 'b';} } C.mk() + ':' + C.mkN + ':' + C.mkG"
            ),
            "a:4:b"
        );
    }

    // The recursion guard (infinite recursion → RangeError, deep finite recursion
    // works) is covered by the `recursion-guard` Test262 corpus test, which runs
    // on a large stack; unit tests here run on the default ~2 MB thread stack,
    // too small for the deep recursion the guard permits.

    #[test]
    fn super_member_read() {
        // super.getter (invoked) and super.method (returned then called).
        assert_eq!(
            run(
                "class B{ constructor(){this._v=10;} get d(){return this._v*2;} m(){return this._v;} } class D extends B{ get d(){return super.d+1;} m(){return super.m()+5;} } let x=new D(); x.d + ':' + x.m()"
            ),
            "21:15"
        );
        // super property as a function value.
        assert_eq!(
            run(
                "class A{ greet(){return 'A';} } class C extends A{ greet(){ let f=super.greet; return f.call(this)+'C'; } } new C().greet()"
            ),
            "AC"
        );
    }

    #[test]
    fn date_setters_and_parse() {
        assert_eq!(
            run("let d=new Date(0); d.setUTCFullYear(2000); d.getUTCFullYear()"),
            "2000"
        );
        assert_eq!(
            run("let d=new Date(0); d.setUTCMonth(5); d.getUTCMonth()"),
            "5"
        );
        assert_eq!(
            run("let d=new Date(0); d.setTime(86400000); d.getUTCDate()"),
            "2"
        );
        assert_eq!(run("Date.parse('1970-01-01T00:00:00.000Z')"), "0");
        assert_eq!(
            run("Date.parse('2000-01-01T00:00:00.000Z') === Date.UTC(2000,0,1)"),
            "true"
        );
        assert_eq!(
            run("new Date('2000-01-01T12:00:00.000Z').getUTCHours()"),
            "12"
        );
        assert_eq!(run("Number.isNaN(Date.parse('garbage'))"), "true");
    }

    #[test]
    fn json_date_and_bigint() {
        assert_eq!(
            run("JSON.stringify(new Date(0))"),
            "\"1970-01-01T00:00:00.000Z\""
        );
        assert_eq!(
            run("JSON.stringify({d:new Date(0)})"),
            "{\"d\":\"1970-01-01T00:00:00.000Z\"}"
        );
        assert_eq!(
            run("try{ JSON.stringify(10n); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(
            run("try{ JSON.stringify({a:1n}); 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("JSON.stringify({a:1,b:'x'})"), "{\"a\":1,\"b\":\"x\"}");
    }

    #[test]
    fn iterator_helpers() {
        let g = "function* g(){yield 1;yield 2;yield 3;yield 4;} ";
        assert_eq!(
            run(&alloc::format!("{g}[...g().map(x=>x*10)].join(',')")),
            "10,20,30,40"
        );
        assert_eq!(
            run(&alloc::format!("{g}[...g().filter(x=>x%2===0)].join(',')")),
            "2,4"
        );
        assert_eq!(run(&alloc::format!("{g}[...g().take(2)].join(',')")), "1,2");
        assert_eq!(run(&alloc::format!("{g}[...g().drop(2)].join(',')")), "3,4");
        assert_eq!(
            run(&alloc::format!("{g}g().toArray().join(',')")),
            "1,2,3,4"
        );
        assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b,0)")), "10");
        assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b)")), "10");
        assert_eq!(run(&alloc::format!("{g}g().some(x=>x>3)")), "true");
        assert_eq!(run(&alloc::format!("{g}g().every(x=>x>2)")), "false");
        assert_eq!(run(&alloc::format!("{g}g().find(x=>x>2)")), "3");
        assert_eq!(
            run(&alloc::format!(
                "{g}[...g().map(x=>x*2).filter(x=>x>4)].join(',')"
            )),
            "6,8"
        );
        // A helper over the remaining values after one `next()`.
        assert_eq!(
            run(&alloc::format!(
                "{g}let it=g(); it.next(); it.map(x=>x).toArray().join(',')"
            )),
            "2,3,4"
        );
    }

    #[test]
    fn labeled_block_and_class_name() {
        // break out of a labeled block.
        assert_eq!(
            run(
                "let r=[]; blk:{ r.push(1); if(true)break blk; r.push(2); } r.push(3); r.join(',')"
            ),
            "1,3"
        );
        assert_eq!(run("let h='no'; a:{ b:{ break a; } h='in'; } h"), "no");
        // continue to a loop label still works.
        assert_eq!(
            run(
                "let r=[]; outer: for(let i=0;i<3;i++){ for(let j=0;j<3;j++){ if(j===1)continue outer; r.push(i+','+j); } } r.join(';')"
            ),
            "0,0;1,0;2,0"
        );
        // Named class self-reference and `.name`.
        assert_eq!(
            run("let C=class Named{ who(){return Named===C;} }; new C().who()"),
            "true"
        );
        assert_eq!(
            run("let C=class Named{ n(){return Named.name;} }; new C().n()"),
            "Named"
        );
        assert_eq!(run("class Declared{} Declared.name"), "Declared");
    }

    #[test]
    fn arraybuffer_and_dataview() {
        assert_eq!(run("new ArrayBuffer(8).byteLength"), "8");
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,42); v.getInt32(0)"),
            "42"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,-1); v.getUint32(0)"),
            "4294967295"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setUint8(0,255); v.getInt8(0)"),
            "-1"
        );
        assert_eq!(
            run(
                "let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,true)"
            ),
            "1000"
        );
        assert_eq!(
            run(
                "let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,false)"
            ),
            "-6141"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setFloat64(0,3.14159); v.getFloat64(0)"),
            "3.14159"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setFloat32(0,1.5); v.getFloat32(0)"),
            "1.5"
        );
        assert_eq!(
            run("let v=new DataView(new ArrayBuffer(8)); v.setInt8(0,300); v.getInt8(0)"),
            "44"
        );
        // Offset view shares the buffer.
        assert_eq!(
            run(
                "let b=new ArrayBuffer(8); let v=new DataView(b); new DataView(b,2).setInt32(0,7); v.getInt32(2)"
            ),
            "7"
        );
    }

    #[test]
    fn typed_arrays() {
        assert_eq!(run("new Uint8Array(3).length"), "3");
        assert_eq!(run("let a=new Uint8Array(1); a[0]=256; a[0]"), "0");
        assert_eq!(run("let a=new Uint8Array(1); a[0]=-1; a[0]"), "255");
        assert_eq!(run("let a=new Int8Array(1); a[0]=200; a[0]"), "-56");
        assert_eq!(
            run("new Uint8ClampedArray([300,-5,100]).join(',')"),
            "255,0,100"
        );
        assert_eq!(run("new Int16Array([70000])[0]"), "4464");
        assert_eq!(run("let f=new Float64Array(1); f[0]=3.14; f[0]"), "3.14");
        assert_eq!(run("new Uint8Array([1,2,3])[1]"), "2");
        assert_eq!(run("new Uint16Array(4).byteLength"), "8");
        assert_eq!(run("new Uint8Array(1).BYTES_PER_ELEMENT"), "1");
        assert_eq!(run("new Uint8Array([1,2,3]) instanceof Uint8Array"), "true");
        assert_eq!(
            run("new Uint8Array([1,2,3]).map(x=>x*2).join(',')"),
            "2,4,6"
        );
        assert_eq!(run("[...new Uint8Array([8,9])].join(',')"), "8,9");
    }

    #[test]
    fn primitive_wrapper_objects() {
        // Number wrapper.
        assert_eq!(run("typeof new Number(5)"), "object");
        assert_eq!(run("new Number(5).valueOf()"), "5");
        assert_eq!(run("new Number(5) + 3"), "8");
        assert_eq!(run("new Number(255).toString(16)"), "ff");
        assert_eq!(run("new Number(5) instanceof Number"), "true");
        // String wrapper.
        assert_eq!(run("new String('hello').length"), "5");
        assert_eq!(run("new String('abc')[1]"), "b");
        assert_eq!(run("new String('HELLO').toLowerCase()"), "hello");
        assert_eq!(run("new String('a') + 'b'"), "ab");
        assert_eq!(run("new String('x') instanceof String"), "true");
        // Boolean wrapper.
        assert_eq!(run("new Boolean(false).valueOf()"), "false");
        assert_eq!(run("new Boolean(false) ? 'truthy' : 'falsy'"), "truthy");
        assert_eq!(run("new Boolean(true) instanceof Boolean"), "true");
        // Defaults.
        assert_eq!(run("new Number().valueOf()"), "0");
        assert_eq!(run("new String().valueOf()"), "");
    }

    #[test]
    fn sloppy_this_is_global_object() {
        // Sloppy plain call: `this` is the global object.
        assert_eq!(
            run("(function(){ function f(){return this===globalThis;} return f(); })()"),
            "true"
        );
        assert_eq!(
            run("(function(){ function f(){return typeof this;} return f(); })()"),
            "object"
        );
        assert_eq!(
            run("(function(){ function f(){return this===globalThis;} return f.call(null); })()"),
            "true"
        );
        // Nested plain function.
        assert_eq!(
            run(
                "(function(){ var o={m(){ function inner(){return this===globalThis;} return inner(); }}; return o.m(); })()"
            ),
            "true"
        );
        // Strict (lexical) keeps `this` undefined.
        assert_eq!(
            run(
                "(function(){'use strict'; function f(){return this===undefined;} return f(); })()"
            ),
            "true"
        );
        assert_eq!(
            run("(function(){'use strict'; function f(){return this;} return f.call(null); })()"),
            "null"
        );
        // A method receiver and a lexical arrow `this` are unaffected.
        assert_eq!(
            run("(function(){ var o={x:5,m(){return this.x;}}; return o.m(); })()"),
            "5"
        );
        assert_eq!(
            run("(function(){ var o={x:9,m(){var a=()=>this.x;return a();}}; return o.m(); })()"),
            "9"
        );
    }

    #[test]
    fn strict_mode_undeclared_assignment() {
        // Strict mode: an implicit-global assignment throws ReferenceError.
        assert_eq!(
            run(
                "(function(){'use strict'; try{ undeclaredX=1; return 'no'; }catch(e){ return e instanceof ReferenceError ? 'ref' : 'other'; }})()"
            ),
            "ref"
        );
        // Sloppy mode still creates the global.
        assert_eq!(
            run("(function(){ sloppyG=5; return typeof sloppyG; })()"),
            "number"
        );
        // Strict propagates to a nested function.
        assert_eq!(
            run(
                "(function(){'use strict'; return (function(){ try{nx=1;return 'no';}catch(e){return 'ref';} })(); })()"
            ),
            "ref"
        );
        // A declared binding is assignable under strict mode.
        assert_eq!(
            run("(function(){'use strict'; let x=1; x=2; return x; })()"),
            "2"
        );
        // Program-level `use strict`.
        assert_eq!(run("'use strict'; var ok='y'; ok"), "y");
        // Strict: writing a read-only property throws; sloppy silently ignores it.
        assert_eq!(
            run(
                "(function(){'use strict'; let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
        assert_eq!(
            run("let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); o.x=2; o.x"),
            "1"
        );
        // Strict: a frozen object rejects writes.
        assert_eq!(
            run(
                "(function(){'use strict'; let o=Object.freeze({a:1}); try{o.a=9;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
            ),
            "te"
        );
    }

    #[test]
    fn block_level_function_hoisting() {
        assert_eq!(
            run("(function(){ {function g(){return 1;}} return typeof g; })()"),
            "function"
        );
        assert_eq!(
            run("(function(){ {function g(){return 42;}} return g(); })()"),
            "42"
        );
        assert_eq!(
            run("(function(){ if(true){function h(){return 5;}} return h(); })()"),
            "5"
        );
        assert_eq!(
            run("(function(){ {{function d(){return 9;}}} return d(); })()"),
            "9"
        );
        // A later block declaration overrides the outer one (function-scoped).
        assert_eq!(
            run(
                "(function(){ function f(){return 'o';} {function f(){return 'i';}} return f(); })()"
            ),
            "i"
        );
        // Top-level hoisting is unaffected.
        assert_eq!(
            run("(function(){ return e(); function e(){return 'h';} })()"),
            "h"
        );
    }

    #[test]
    fn for_await_of_parses_and_runs() {
        // `for await` parses inside an async function and the call yields a promise.
        assert_eq!(
            run(
                "async function f(){ let s=0; for await(const x of [1,2,3]) s+=x; return s; } typeof f()"
            ),
            "object"
        );
        // An async generator is iterable with for-await.
        assert_eq!(
            run(
                "async function* g(){ yield 1; } async function f(){ for await(const x of g()){} } typeof f"
            ),
            "function"
        );
        // A regular for-of (no await) is unaffected by the AST change.
        assert_eq!(run("let s=0; for(const x of [1,2,3]) s+=x; s"), "6");
    }

    #[test]
    fn intl_number_and_datetime_format() {
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(1234.5)"),
            "1,234.5"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(1000000)"),
            "1,000,000"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(1234.5)"),
            "$1,234.50"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'currency',currency:'JPY'}).format(1234)"),
            "¥1,234"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{style:'percent'}).format(0.25)"),
            "25%"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{minimumFractionDigits:2}).format(5)"),
            "5.00"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US',{useGrouping:false}).format(1234567)"),
            "1234567"
        );
        assert_eq!(
            run("new Intl.NumberFormat('en-US').format(-1234.5)"),
            "-1,234.5"
        );
        // Callable without `new`.
        assert_eq!(run("Intl.NumberFormat('en-US').format(42)"), "42");
        assert_eq!(
            run("new Intl.DateTimeFormat('en-US').format(new Date(Date.UTC(2020,5,15)))"),
            "6/15/2020"
        );
    }

    #[test]
    fn date_string_methods() {
        assert_eq!(run("new Date(0).toDateString()"), "Thu Jan 01 1970");
        assert_eq!(
            run("new Date(0).toUTCString()"),
            "Thu, 01 Jan 1970 00:00:00 GMT"
        );
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15,10,30,45)).toLocaleString()"),
            "6/15/2020, 10:30:45"
        );
        assert_eq!(
            run("new Date(Date.UTC(2020,5,15)).toLocaleDateString()"),
            "6/15/2020"
        );
        assert_eq!(
            run("new Date(Date.UTC(2021,11,25)).toDateString()"),
            "Sat Dec 25 2021"
        );
    }

    #[test]
    fn base64_btoa_atob() {
        assert_eq!(run("btoa('hi')"), "aGk=");
        assert_eq!(run("btoa('Man')"), "TWFu");
        assert_eq!(run("btoa('M')"), "TQ==");
        assert_eq!(run("btoa('')"), "");
        assert_eq!(run("atob('aGVsbG8=')"), "hello");
        assert_eq!(run("atob(btoa('round trip!'))"), "round trip!");
        assert_eq!(run("btoa('é')"), "6Q==");
        assert_eq!(run("atob('aG k=')"), "hi"); // whitespace ignored
        assert_eq!(
            run("try{btoa('\\u{1F600}');'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
    }

    #[test]
    fn structured_clone_deep_copy() {
        assert_eq!(
            run("let o={b:{c:2}}; let c=structuredClone(o); c.b.c=9; o.b.c"),
            "2"
        );
        assert_eq!(
            run("let c=structuredClone([1,[2,3]]); c[1][0]=9; c[1][0]"),
            "9"
        );
        assert_eq!(run("structuredClone(new Map([['k',1]])).get('k')"), "1");
        assert_eq!(
            run("[...structuredClone(new Set([1,2,3]))].join(',')"),
            "1,2,3"
        );
        assert_eq!(run("structuredClone(new Date(1000)).getTime()"), "1000");
        // Cycles and shared references.
        assert_eq!(
            run("let o={}; o.self=o; let c=structuredClone(o); c.self===c"),
            "true"
        );
        assert_eq!(
            run("let s={v:1}; let c=structuredClone({x:s,y:s}); c.x===c.y"),
            "true"
        );
        // Primitives pass through; functions throw.
        assert_eq!(run("structuredClone(42)"), "42");
        assert_eq!(
            run("try{structuredClone({f:function(){}});'no'}catch(e){e instanceof TypeError}"),
            "true"
        );
    }

    #[test]
    fn uri_encoding_functions() {
        assert_eq!(run("encodeURIComponent('a b&c=d')"), "a%20b%26c%3Dd");
        assert_eq!(run("decodeURIComponent('a%20b%26c')"), "a b&c");
        assert_eq!(run("encodeURI('http://a.com/x y')"), "http://a.com/x%20y");
        assert_eq!(run("encodeURIComponent('café')"), "caf%C3%A9");
        assert_eq!(run("decodeURIComponent('caf%C3%A9')"), "café");
        assert_eq!(run("encodeURIComponent(\"-_.!~*'()\")"), "-_.!~*'()");
        assert_eq!(run("decodeURIComponent('%2f')"), "/");
        // Malformed percent-encoding throws a URIError (a subclass of Error).
        assert_eq!(
            run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof URIError}"),
            "true"
        );
        assert_eq!(
            run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof Error}"),
            "true"
        );
    }

    #[test]
    fn global_this_object() {
        assert_eq!(run("typeof globalThis"), "object");
        assert_eq!(run("globalThis.globalThis === globalThis"), "true");
        assert_eq!(run("globalThis.Math.max(1,2,3)"), "3");
        assert_eq!(run("globalThis.parseInt('42px')"), "42");
        assert_eq!(run("globalThis.Array.isArray([])"), "true");
        assert_eq!(run("globalThis.Infinity"), "Infinity");
        assert_eq!(run("globalThis.x = 7; globalThis.x"), "7");
    }

    #[test]
    fn map_set_samevaluezero_and_set_ops() {
        // SameValueZero key matching.
        assert_eq!(run("let m=new Map(); m.set(NaN,'y'); m.get(NaN)"), "y");
        assert_eq!(run("new Set([NaN,NaN,1]).size"), "2");
        assert_eq!(run("let m=new Map(); m.set(-0,'n'); m.get(0)"), "n");
        // ES2025 Set composition.
        assert_eq!(
            run("[...new Set([1,2,3]).union(new Set([3,4]))].join(',')"),
            "1,2,3,4"
        );
        assert_eq!(
            run("[...new Set([1,2,3]).intersection(new Set([2,3,4]))].join(',')"),
            "2,3"
        );
        assert_eq!(
            run("[...new Set([1,2,3]).difference(new Set([2]))].join(',')"),
            "1,3"
        );
        assert_eq!(
            run("[...new Set([1,2]).symmetricDifference(new Set([2,3]))].join(',')"),
            "1,3"
        );
        assert_eq!(run("new Set([1,2]).isSubsetOf(new Set([1,2,3]))"), "true");
        assert_eq!(run("new Set([1,2,3]).isSupersetOf(new Set([1,2]))"), "true");
        assert_eq!(run("new Set([1,2]).isDisjointFrom(new Set([3,4]))"), "true");
        // The argument may be any iterable.
        assert_eq!(
            run("[...new Set([1,2,3]).intersection([2,3,9])].join(',')"),
            "2,3"
        );
    }

    #[test]
    fn parse_float_infinity() {
        assert_eq!(run("parseFloat('Infinity')"), "Infinity");
        assert_eq!(run("parseFloat('-Infinity')"), "-Infinity");
        assert_eq!(run("parseFloat('  +Infinity x')"), "Infinity");
        assert_eq!(run("parseFloat('InfinityX')"), "Infinity");
        assert_eq!(run("Number.isNaN(parseFloat('Inf'))"), "true");
        assert_eq!(run("parseFloat('3.14abc')"), "3.14");
    }

    #[test]
    fn define_property_with_symbol_key() {
        assert_eq!(
            run("let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); o[s]"),
            "42"
        );
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertyDescriptor(o,s).value"
            ),
            "42"
        );
        // A non-enumerable symbol (defineProperty's default) still appears here.
        assert_eq!(
            run(
                "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertySymbols(o).length"
            ),
            "1"
        );
        // A symbol-keyed accessor.
        assert_eq!(
            run(
                "let s=Symbol('a'); let o={}; let v=0; Object.defineProperty(o,s,{get(){return v;},set(n){v=n;}}); o[s]=7; o[s]"
            ),
            "7"
        );
        assert_eq!(
            run(
                "let s=Symbol('r'); let o={}; Reflect.defineProperty(o,s,{value:9}); Reflect.getOwnPropertyDescriptor(o,s).value"
            ),
            "9"
        );
    }

    #[test]
    fn error_stack_and_aggregate() {
        assert_eq!(run("typeof new Error('x').stack"), "string");
        assert_eq!(run("new Error('boom').stack.indexOf('boom') >= 0"), "true");
        assert_eq!(run("Object.keys(new Error('x')).indexOf('stack')"), "-1");
        // AggregateError: message is the 2nd arg, `.errors` collects the 1st.
        assert_eq!(
            run(
                "let a=new AggregateError([new Error('a'),new TypeError('b')],'m'); a.message + ':' + a.errors.length + ':' + a.name"
            ),
            "m:2:AggregateError"
        );
        assert_eq!(run("new AggregateError([],'x') instanceof Error"), "true");
        assert_eq!(
            run("new AggregateError(new Set([new Error('x')]),'s').errors.length"),
            "1"
        );
    }

    #[test]
    fn error_cause_option() {
        assert_eq!(run("new Error('m',{cause:'r'}).cause"), "r");
        assert_eq!(run("new TypeError('t',{cause:42}).cause"), "42");
        assert_eq!(run("String(new Error('m').cause)"), "undefined");
        assert_eq!(run("String(new Error('m',{}).cause)"), "undefined");
        assert_eq!(
            run("new Error('o',{cause:new Error('i')}).cause.message"),
            "i"
        );
    }

    #[test]
    fn class_extends_native_error() {
        assert_eq!(
            run(
                "class E extends Error{ constructor(m,c){ super(m); this.name='E'; this.c=c; } } let e=new E('x',5); e.message + ':' + e.c + ':' + e.name"
            ),
            "x:5:E"
        );
        assert_eq!(
            run("class E extends Error{} (new E('m')) instanceof Error"),
            "true"
        );
        assert_eq!(
            run(
                "class E extends Error{ constructor(m){super(m);} } let e=new E('m'); (e instanceof Error) + ',' + (e instanceof E) + ',' + (e instanceof TypeError)"
            ),
            "true,true,false"
        );
        assert_eq!(
            run(
                "class V extends RangeError{} let v=new V(); (v instanceof RangeError) + ',' + (v instanceof Error)"
            ),
            "true,true"
        );
    }

    #[test]
    fn class_field_init_order_and_computed_fields() {
        // A field declared without an initializer must not clobber a constructor
        // write (fields init before the constructor body).
        assert_eq!(
            run(
                "class A{ #b; constructor(v){ this.#b=v; } get b(){ return this.#b; } } new A(100).b"
            ),
            "100"
        );
        assert_eq!(
            run(
                "class A{ #b; constructor(v){ this.#b=v; } add(n){ this.#b+=n; return this.#b; } } let a=new A(100); a.add(50)"
            ),
            "150"
        );
        // Computed instance field names.
        assert_eq!(run("let k='x'; class C{ [k+'1']=7; } new C().x1"), "7");
    }

    #[test]
    fn class_rest_params_and_string_positions() {
        // Class constructor rest parameter (with spread).
        assert_eq!(
            run("class V{constructor(...c){this.c=c;}} new V(...[1,2,3]).c.length"),
            "3"
        );
        assert_eq!(
            run("class V{constructor(a, ...r){this.r=r;}} new V(1,2,3).r.join(',')"),
            "2,3"
        );
        // Class constructor default parameter.
        assert_eq!(
            run("class P{constructor(x=7){this.x=x;}} new P().x + ':' + new P(2).x"),
            "7:2"
        );
        // String prefix/suffix with positions.
        assert_eq!(run("'hello world'.startsWith('world', 6)"), "true");
        assert_eq!(run("'hello world'.endsWith('hello', 5)"), "true");
        assert_eq!(
            run("'hello'.includes('lo', 3) + ':' + 'hello'.includes('he', 1)"),
            "true:false"
        );
    }

    #[test]
    fn arithmetic_object_coercion() {
        assert_eq!(run("[5] - 2"), "3");
        assert_eq!(run("[10] / 2"), "5");
        assert_eq!(run("[6] & 3"), "2");
        assert_eq!(run("[2] ** 3"), "8");
        assert_eq!(run("String({} - 1)"), "NaN");
        assert_eq!(run("-[5]"), "-5");
        assert_eq!(run("new Date(5000) - new Date(2000)"), "3000");
    }

    #[test]
    fn tostring_in_concat_and_property_key() {
        // String.concat honors a user toString.
        assert_eq!(run("'x'.concat({toString(){return 'TS';}})"), "xTS");
        // An object property key is coerced via ToString (toString).
        assert_eq!(
            run("let k={toString(){return 'key';}}; let m={}; m[k]=42; m.key + ':' + m[k]"),
            "42:42"
        );
    }

    #[test]
    fn relational_object_coercion() {
        assert_eq!(run("String([5] < 10)"), "true");
        assert_eq!(run("String([20] > 10)"), "true");
        assert_eq!(run("String([1] < [2])"), "true"); // "1" < "2"
        assert_eq!(run("String([10] < [9])"), "true"); // lexicographic
        assert_eq!(run("String({} < 1)"), "false"); // NaN
        assert_eq!(run("String(new Date(1) < new Date(2))"), "true"); // by timestamp
    }

    #[test]
    fn loose_eq_object_coercion() {
        assert_eq!(run("String([] == false)"), "true"); // []→""→0, false→0
        assert_eq!(run("String([] == 0)"), "true");
        assert_eq!(run("String([0] == false)"), "true"); // [0]→"0"→0
        assert_eq!(run("String({} == 0)"), "false"); // "[object Object]"→NaN
        assert_eq!(run("String({} == {})"), "false"); // distinct objects
        assert_eq!(run("String([1,2] == '1,2')"), "true");
    }

    #[test]
    fn array_string_index_access() {
        assert_eq!(run("let a=[10,20,30]; a['0'] + ':' + a['2']"), "10:30");
        assert_eq!(run("let a=[10,20,30]; let k='1'; a[k]"), "20");
        assert_eq!(
            run("let a=[10,20,30]; String(a['00']) + ':' + String(a['01'])"),
            "undefined:undefined"
        );
        assert_eq!(run("[[1,2],[3,4]]['0']['1']"), "2");
    }

    #[test]
    fn object_literal_async_methods_parse() {
        // `async`/`get`/`set` remain usable as property names.
        assert_eq!(
            run("let async=5; let o={async, get:6, set:7}; o.async + ':' + o.get + ':' + o.set"),
            "5:6:7"
        );
        // An async method is a function whose call yields a promise (object).
        assert_eq!(
            run("let o={ async f(){return 1;} }; typeof o.f"),
            "function"
        );
        assert_eq!(
            run("let o={ async f(){return 1;} }; typeof o.f()"),
            "object"
        );
        assert_eq!(
            run("let k='m'; let o={ async [k](){return 1;} }; typeof o.m"),
            "function"
        );
    }

    #[test]
    fn object_literal_generator_methods() {
        assert_eq!(
            run("let o={ *g(){yield 1;yield 2;} }; [...o.g()].join(',')"),
            "1,2"
        );
        assert_eq!(
            run("let o={ *[Symbol.iterator](){yield 'a';yield 'b';} }; [...o].join(',')"),
            "a,b"
        );
        assert_eq!(
            run("let k='m'; let o={ *[k](){yield 9;} }; [...o.m()].join(',')"),
            "9"
        );
        // The generator method reads `this`.
        assert_eq!(
            run("let o={ v:5, *items(){yield this.v;yield this.v*2;} }; [...o.items()].join(',')"),
            "5,10"
        );
    }

    #[test]
    fn class_symbol_iterator_method() {
        assert_eq!(
            run("class C{ *[Symbol.iterator](){yield 'x';yield 'y';} } [...new C()].join(',')"),
            "x,y"
        );
        // A non-generator iterator method (manual iterator object).
        assert_eq!(
            run(
                "class C{ [Symbol.iterator](){let i=0;return{next:()=>i<3?{value:i++,done:false}:{done:true}};} } [...new C()].join(',')"
            ),
            "0,1,2"
        );
        // for-of uses it too.
        assert_eq!(
            run(
                "class C{ *[Symbol.iterator](){yield 1;yield 2;} } let s=0; for(let v of new C())s+=v; s"
            ),
            "3"
        );
    }

    #[test]
    fn generator_is_its_own_iterator() {
        assert_eq!(
            run("function* g(){yield 1;} let it=g(); it[Symbol.iterator]() === it"),
            "true"
        );
        assert_eq!(
            run(
                "function* g(){yield 1;yield 2;} let it=g(); it[Symbol.iterator]().next().value + ':' + it.next().value"
            ),
            "1:2"
        );
        assert_eq!(
            run("function* g(){yield* [1,2]; yield* 'ab';} [...g()].join(',')"),
            "1,2,a,b"
        );
    }

    #[test]
    fn explicit_symbol_iterator_call() {
        assert_eq!(
            run("let it=[10,20,30][Symbol.iterator](); it.next().value + ',' + it.next().value"),
            "10,20"
        );
        assert_eq!(run("'abc'[Symbol.iterator]().next().value"), "a");
        assert_eq!(
            run("let m=new Map([['k','v']])[Symbol.iterator]().next().value; m[0] + '=' + m[1]"),
            "k=v"
        );
        assert_eq!(run("new Set([1,2])[Symbol.iterator]().next().value"), "1");
        assert_eq!(run("[...[1,2,3][Symbol.iterator]()].join(',')"), "1,2,3");
    }

    #[test]
    fn in_operator_walks_prototype_chain() {
        assert_eq!(run("'a' in {a:1}"), "true");
        assert_eq!(run("'z' in {a:1}"), "false");
        assert_eq!(run("let o=Object.create({x:1}); 'x' in o"), "true");
        assert_eq!(
            run("let o=Object.create(Object.create({deep:1})); 'deep' in o"),
            "true"
        );
        assert_eq!(run("0 in [10,20]"), "true");
        assert_eq!(run("5 in [10,20]"), "false");
    }

    #[test]
    fn for_in_inherited_enumeration() {
        assert_eq!(
            run(
                "let p={a:1}; let o=Object.create(p); o.b=2; let k=[]; for(let x in o)k.push(x); k.sort().join(',')"
            ),
            "a,b"
        );
        // Non-enumerable prototype methods are not enumerated.
        assert_eq!(run("let k=[]; for(let x in {})k.push(x); k.length"), "0");
        // A shadowed inherited key appears once.
        assert_eq!(
            run("let o=Object.create({v:1}); o.v=2; let k=[]; for(let x in o)k.push(x); k.length"),
            "1"
        );
    }

    #[test]
    fn const_reassignment_throws() {
        assert_eq!(
            run("const x=1; try{ x=2; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        assert_eq!(run("const x=1; try{ x=2; }catch(e){} x"), "1");
        assert_eq!(
            run("const n=10; try{ n+=5; 'no' }catch(e){ e instanceof TypeError }"),
            "true"
        );
        // Mutation through a const reference is allowed; let is reassignable.
        assert_eq!(run("const a=[1]; a.push(2); a.length"), "2");
        assert_eq!(run("let y=1; y=2; y"), "2");
        // An inner const shadows without affecting the outer.
        assert_eq!(run("const a=1; { const a=2; } a"), "1");
    }

    #[test]
    fn destructure_any_iterable() {
        // Array binding patterns destructure any iterable, not just arrays.
        assert_eq!(run("let [a,b,c]='xyz'; a+b+c"), "xyz");
        assert_eq!(
            run("let [f,...r]=new Set([1,2,3,4]); f + ':' + r.join(',')"),
            "1:2,3,4"
        );
        assert_eq!(
            run("function* g(){yield 10;yield 20;} let [x,y]=g(); x+y"),
            "30"
        );
        assert_eq!(run("let [[k,v]]=new Map([['a',1]]); k + ':' + v"), "a:1");
    }

    #[test]
    fn computed_member_assignment_eval_order() {
        // The index is resolved before the RHS (which mutates it).
        assert_eq!(
            run("let a=[0,0]; let i=0; a[i] = i = 1; a[0] + ',' + a[1]"),
            "1,0"
        );
        // Compound assignment on a computed element still works.
        assert_eq!(run("let a=[1,2,3]; a[1] *= 10; a.join(',')"), "1,20,3");
        assert_eq!(run("let o={x:5}; let k='x'; o[k] += 3; o.x"), "8");
        // Computed key honoring a setter.
        assert_eq!(run("let o={set v(n){this._v=n*2;}}; o['v']=10; o._v"), "20");
    }

    #[test]
    fn in_operator_array_bounds_and_delete() {
        assert_eq!(run("0 in [1,2,3]"), "true");
        assert_eq!(run("5 in [1,2,3]"), "false"); // out of bounds
        assert_eq!(run("'length' in [1,2,3]"), "true");
        assert_eq!(run("'a' in {a:1}"), "true");
        assert_eq!(run("'b' in {a:1}"), "false");
        // delete clears an array element.
        assert_eq!(
            run("let a=[1,2,3]; delete a[1]; String(a[1]) + ':' + a.length"),
            "undefined:3"
        );
        assert_eq!(run("let o={a:1}; delete o.a; 'a' in o"), "false");
    }

    #[test]
    fn catch_binding_forms() {
        // Destructured catch binding (object and array patterns).
        assert_eq!(
            run(
                "let r; try { throw {code:42, text:'x'}; } catch({code,text}){ r=code+':'+text; } r"
            ),
            "42:x"
        );
        assert_eq!(
            run("let r; try { throw [1,2,3]; } catch([a,b]){ r=a+b; } r"),
            "3"
        );
        // Optional catch binding (no parameter).
        assert_eq!(
            run("let r=false; try { throw 1; } catch { r=true; } r"),
            "true"
        );
        // Named binding still works.
        assert_eq!(
            run("let r; try { throw new Error('m'); } catch(e){ r=e.message; } r"),
            "m"
        );
    }

    #[test]
    fn arguments_object() {
        assert_eq!(
            run(
                "function s(){ var t=0; for (var i=0;i<arguments.length;i++) t+=arguments[i]; return t; } s(1,2,3,4)"
            ),
            "10"
        );
        assert_eq!(
            run("function f(){ return arguments[1]; } f('a','b','c')"),
            "b"
        );
        assert_eq!(run("function f(){ return arguments.length; } f()"), "0");
        // An arrow inherits the enclosing `arguments`.
        assert_eq!(
            run("function outer(){ var a = () => arguments[0]; return a(); } outer('Z')"),
            "Z"
        );
    }

    #[test]
    fn function_call_apply_bind() {
        assert_eq!(
            run("function f(p){return p + ':' + this.n;} f.call({n:7}, 'a')"),
            "a:7"
        );
        assert_eq!(
            run("function f(a,b){return a+b+this.n;} f.apply({n:1}, [2,3])"),
            "6"
        );
        assert_eq!(
            run(
                "function f(a,b){return a+b+this.n;} let g=f.bind({n:10}, 5); g(20) + ':' + typeof g"
            ),
            "35:function"
        );
        assert_eq!(run("Math.max.apply(null, [3,9,2])"), "9");
    }

    #[test]
    fn prototype_chains() {
        // Inherited data property and method (this-bound), own shadows inherited.
        assert_eq!(
            run(
                "let p={k:'base',m:function(){return this.n;}}; let o=Object.create(p); o.n=7; o.k + ':' + o.m()"
            ),
            "base:7"
        );
        // Object.keys excludes inherited; getPrototypeOf identity.
        assert_eq!(
            run("let p={a:1}; let o=Object.create(p); o.b=2; Object.keys(o).join(',')"),
            "b"
        );
        assert_eq!(
            run("let p={}; Object.getPrototypeOf(Object.create(p)) === p"),
            "true"
        );
        assert_eq!(
            run("Object.getPrototypeOf(Object.create(null)) === null"),
            "true"
        );
        // Two-level chain: nearest prototype wins.
        assert_eq!(
            run("let a={x:1}; let b=Object.create(a); b.x=2; let c=Object.create(b); c.x"),
            "2"
        );
        // setPrototypeOf installs the link.
        assert_eq!(run("let o={}; Object.setPrototypeOf(o,{v:9}); o.v"), "9");
    }

    #[test]
    fn proxy_get_set_traps() {
        // get trap with fallthrough; set trap transforming the value.
        assert_eq!(
            run(
                "let t={a:1}; let p=new Proxy(t,{get:function(o,k){return k in o?o[k]:'def';}}); p.a + ':' + p.zzz"
            ),
            "1:def"
        );
        assert_eq!(
            run(
                "let t={}; let p=new Proxy(t,{set:function(o,k,v){o[k]=v*3;return true;}}); p.n=4; t.n"
            ),
            "12"
        );
        // No-trap handler forwards to the target.
        assert_eq!(
            run("let p=new Proxy({x:5},{}); p.y=6; '' + p.x + p.y"),
            "56"
        );
        assert_eq!(run("typeof new Proxy({}, {})"), "object");
        // has trap (for `in`) and deleteProperty trap (for `delete`).
        assert_eq!(
            run(
                "let p=new Proxy({a:1},{has:function(t,k){return k==='magic'||k in t;}}); '' + ('a' in p) + ('magic' in p) + ('z' in p)"
            ),
            "truetruefalse"
        );
        assert_eq!(
            run(
                "let seen=''; let p=new Proxy({a:1},{deleteProperty:function(t,k){seen=k; delete t[k]; return true;}}); delete p.a; seen + ':' + ('a' in p)"
            ),
            "a:false"
        );
        // Forwarding `in`/`delete` with no traps.
        assert_eq!(
            run("let p=new Proxy({x:1},{}); let r = 'x' in p; delete p.x; '' + r + ('x' in p)"),
            "truefalse"
        );
    }

    #[test]
    fn instanceof_array_object_and_fromentries_map() {
        assert_eq!(run("[] instanceof Array"), "true");
        assert_eq!(run("({}) instanceof Array"), "false");
        assert_eq!(run("[] instanceof Object"), "true");
        assert_eq!(run("({}) instanceof Object"), "true");
        assert_eq!(run("'str' instanceof Object"), "false"); // primitive
        // Object.fromEntries from a Map.
        assert_eq!(
            run("let m=new Map([['x',10],['y',20]]); let o=Object.fromEntries(m); o.x + ':' + o.y"),
            "10:20"
        );
    }

    #[test]
    fn collection_foreach_thisarg() {
        assert_eq!(
            run(
                "let r=[]; new Map([['a',1],['b',2]]).forEach(function(v,k){ r.push(k+':'+v*this.m); }, {m:10}); r.join(',')"
            ),
            "a:10,b:20"
        );
        assert_eq!(
            run("let s=0; new Set([1,2,3]).forEach(function(v){ s+=v*this.m; }, {m:2}); s"),
            "12"
        );
        // The callback also gets the collection as the third argument.
        assert_eq!(
            run("let n; new Set([1]).forEach((v,k,coll)=>{ n=coll.size; }); n"),
            "1"
        );
    }

    #[test]
    fn map_set_clear_and_assign_getters() {
        assert_eq!(
            run("let m=new Map(); m.set('a',1).set('b',2); m.clear(); m.size + ':' + m.has('a')"),
            "0:false"
        );
        assert_eq!(
            run("let s=new Set([1,2,3]); s.clear(); s.add(5).add(5); s.size"),
            "1"
        );
        // Object.assign invokes getters.
        assert_eq!(
            run(
                "let src={a:1, get b(){ return this.a + 1; }}; let t=Object.assign({}, src); t.a + ',' + t.b"
            ),
            "1,2"
        );
        assert_eq!(run("Object.assign({}, {x:1}, {y:2}, {x:9}).x"), "9");
    }

    #[test]
    fn bigint_shifts() {
        assert_eq!(run("(1n << 8n).toString()"), "256");
        assert_eq!(run("(256n >> 2n).toString()"), "64");
        assert_eq!(run("(2n ** 32n).toString()"), "4294967296");
        assert_eq!(run("(-8n >> 1n).toString()"), "-4");
        assert_eq!(run("(-7n >> 1n).toString()"), "-4"); // arithmetic floor
        assert_eq!(run("(5n << -1n).toString()"), "2"); // negative count reverses
    }

    #[test]
    fn number_of_bigint_and_collection_from_iterable() {
        // Number(bigint) → the double value.
        assert_eq!(run("Number(100n)"), "100");
        assert_eq!(run("Number(-7n)"), "-7");
        assert_eq!(run("Number(2n ** 10n)"), "1024");
        // Set/Map seed from any iterable, incl. a string.
        assert_eq!(run("[...new Set('hello')].join('')"), "helo");
        assert_eq!(run("new Set([1,1,2,3]).size"), "3");
        assert_eq!(
            run("let m=new Map([['a',1],['b',2]]); m.get('a') + m.get('b')"),
            "3"
        );
    }

    // `Promise.any` (first-fulfilled / AggregateError-when-all-reject) is covered
    // by the `promise-allsettled-any` Test262 corpus test, which awaits the
    // settled value (microtask timing isn't observable synchronously here).

    #[test]
    fn weakref_and_finalization_registry() {
        assert_eq!(
            run("let o={x:1}; let r=new WeakRef(o); (r.deref()===o) + ':' + r.deref().x"),
            "true:1"
        );
        assert_eq!(
            run("typeof WeakRef + ',' + typeof FinalizationRegistry"),
            "function,function"
        );
        assert_eq!(
            run(
                "let reg=new FinalizationRegistry(()=>{}); reg.register({}, 'h'); reg.unregister('t')"
            ),
            "false"
        );
        assert_eq!(run("new WeakRef([1,2,3]).deref().length"), "3");
    }

    #[test]
    fn math_exp_log_reflect_assign_array() {
        // Math.exp / Math.log.
        assert_eq!(run("Math.round(Math.exp(0))"), "1");
        assert_eq!(run("Math.round(Math.log(Math.E))"), "1");
        // Object.assign spreads an array source's indices.
        assert_eq!(
            run(
                "let o=Object.assign({}, ['a','b','c']); o[0] + o[2] + ':' + Object.keys(o).join(',')"
            ),
            "ac:0,1,2"
        );
        // Reflect.has walks the chain; Reflect.set updates array storage.
        assert_eq!(run("Reflect.has(Object.create({k:1}), 'k')"), "true");
        assert_eq!(
            run("let a=[1,2,3]; Reflect.set(a, 3, 4); a[3] + ':' + a.length"),
            "4:4"
        );
        assert_eq!(
            run("Reflect.defineProperty({}, 'x', {value:5}) === true"),
            "true"
        );
        assert_eq!(
            run(
                "let o={}; Reflect.defineProperty(o,'x',{value:9,enumerable:true}); Reflect.getOwnPropertyDescriptor(o,'x').value"
            ),
            "9"
        );
    }

    #[test]
    fn reflect_and_weak_collections() {
        // Reflect mirrors the fundamental operations.
        assert_eq!(run("let o={a:1}; Reflect.get(o,'a')"), "1");
        assert_eq!(run("let o={}; Reflect.set(o,'x',5); o.x"), "5");
        assert_eq!(
            run("Reflect.has({a:1},'a') + ':' + Reflect.has({a:1},'z')"),
            "true:false"
        );
        assert_eq!(run("Reflect.ownKeys({a:1,b:2,c:3}).length"), "3");
        assert_eq!(
            run("function f(a,b){return a+b+this.n;} Reflect.apply(f,{n:10},[1,2])"),
            "13"
        );
        assert_eq!(
            run("function B(v){this.v=v;} Reflect.construct(B,[9]).v"),
            "9"
        );
        // WeakMap / WeakSet (object-keyed; bounded — no true weakness).
        assert_eq!(
            run("let k={}; let m=new WeakMap(); m.set(k,'v'); m.get(k) + ':' + m.has(k)"),
            "v:true"
        );
        // Weak collections are recognized by instanceof and chain from set/add.
        assert_eq!(run("(new WeakMap()).set({}, 1) instanceof WeakMap"), "true");
        assert_eq!(run("(new WeakSet()).add({}) instanceof WeakSet"), "true");
        assert_eq!(
            run("let s=new WeakSet(); let o={}; s.add(o); s.has(o) + ':' + s.has({})"),
            "true:false"
        );
    }

    #[test]
    fn proxy_revocable() {
        // Works before revoke; every operation throws after.
        assert_eq!(
            run(
                "let r=Proxy.revocable({a:1},{get:function(t,k){return t[k];}}); let b=r.proxy.a; r.revoke(); let after='ok'; try { r.proxy.a; } catch(e){ after='threw'; } b + ':' + after"
            ),
            "1:threw"
        );
        assert_eq!(
            run("let r=Proxy.revocable({},{}); typeof r.proxy + ',' + typeof r.revoke"),
            "object,function"
        );
    }

    #[test]
    fn proxy_apply_construct_traps() {
        // apply trap intercepts a call; typeof a function proxy is "function".
        assert_eq!(
            run(
                "function f(a,b){return a+b;} let p=new Proxy(f,{apply:function(t,th,a){return a[0]*a[1];}}); p(3,4) + ':' + typeof p"
            ),
            "12:function"
        );
        assert_eq!(
            run("function f(a){return a+1;} let p=new Proxy(f,{}); p(9)"),
            "10"
        );
        // construct trap intercepts `new`.
        assert_eq!(
            run(
                "function B(v){this.v=v;} let p=new Proxy(B,{construct:function(t,a){return {v:a[0]*2};}}); (new p(5)).v"
            ),
            "10"
        );
        assert_eq!(
            run("function B(v){this.v=v;} let p=new Proxy(B,{}); (new p(7)).v"),
            "7"
        );
    }

    #[test]
    fn symbols() {
        assert_eq!(run("typeof Symbol('x')"), "symbol");
        assert_eq!(run("Symbol('hi').toString()"), "Symbol(hi)");
        assert_eq!(run("Symbol('hi').description"), "hi");
        assert_eq!(run("Symbol('a') === Symbol('a')"), "false");
        assert_eq!(run("let s = Symbol(); s === s"), "true");
        assert_eq!(run("Symbol.for('k') === Symbol.for('k')"), "true");
        assert_eq!(run("Symbol.keyFor(Symbol.for('k2'))"), "k2");
        assert_eq!(run("typeof Symbol.iterator"), "symbol");
    }

    #[test]
    fn symbol_keyed_properties() {
        // Distinct symbols are distinct keys; symbol keys are non-enumerable.
        assert_eq!(
            run(
                "let a=Symbol('k'),b=Symbol('k'),o={}; o[a]=1; o[b]=2; o.p=3; '' + o[a] + o[b] + Object.keys(o).join('')"
            ),
            "12p"
        );
        assert_eq!(run("let s=Symbol(); let o={}; o[s]='v'; s in o"), "true");
        assert_eq!(
            run("let s=Symbol(); let o={}; o[s]=1; delete o[s]; o[s]"),
            "undefined"
        );
        assert_eq!(
            run("let o={}; o[Symbol.iterator]='it'; o[Symbol.iterator]"),
            "it"
        );
    }

    #[test]
    fn regex_replace_with_function() {
        assert_eq!(
            run("'a1b2'.replace(/[0-9]/g, function(m){ return '<'+m+'>'; })"),
            "a<1>b<2>"
        );
        assert_eq!(
            run("'1-2'.replace(/(\\d)-(\\d)/, function(_, a, b){ return b+'-'+a; })"),
            "2-1"
        );
        // A string replacement still works.
        assert_eq!(run("'foo'.replace(/o/g, '0')"), "f00");
    }

    #[test]
    fn promise_combinators() {
        // Drive the combinators through output (await/then resolve eagerly).
        assert_eq!(
            out(
                "Promise.all([Promise.resolve(1), 2, Promise.resolve(3)]).then(r => console.log(r.join(',')));"
            ),
            "1,2,3\n"
        );
        assert_eq!(
            out(
                "Promise.race([Promise.resolve('a'), Promise.resolve('b')]).then(v => console.log(v));"
            ),
            "a\n"
        );
        assert_eq!(
            out(
                "Promise.all([Promise.resolve(1), Promise.reject('boom')]).catch(e => console.log('caught:' + e));"
            ),
            "caught:boom\n"
        );
    }

    #[test]
    fn empty_string_is_falsy() {
        assert_eq!(run("!!''"), "false");
        assert_eq!(run("!!'x'"), "true");
        assert_eq!(run("'' || 'fallback'"), "fallback");
        assert_eq!(run("if ('') { 'T' } else { 'F' }"), "F");
        assert_eq!(run("Boolean('')"), "false");
        assert_eq!(
            run("[0, '', null, 1, 'a'].filter(function(x){ return x; }).join(',')"),
            "1,a"
        );
    }

    #[test]
    fn array_from_array_like() {
        // `{ length }` array-like with a map callback (tree-walker path).
        assert_eq!(
            run("Array.from({length:3}, function(_,i){ return i*i; }).join(',')"),
            "0,1,4"
        );
        // Array-like with indexed props, no map fn.
        assert_eq!(run("Array.from({length:2, 0:'a', 1:'b'}).join('-')"), "a-b");
        // Still works for real iterables.
        assert_eq!(
            run("Array.from([1,2,3], function(x){ return x*2; }).join(',')"),
            "2,4,6"
        );
    }

    #[test]
    fn array_from_index_and_string_search() {
        assert_eq!(run("[1,2,3,2,1].indexOf(2, 2)"), "3");
        assert_eq!(run("[1,2,3].includes(2, 2)"), "false");
        assert_eq!(run("[5,6,7].indexOf(5, 1)"), "-1");
        assert_eq!(run("'hello world'.search('world')"), "6");
        assert_eq!(run("'abc'.search('z')"), "-1");
    }

    #[test]
    fn collection_iterators() {
        // Map keys/values/entries.
        assert_eq!(
            run("[...new Map([['a',1],['b',2]]).keys()].join(',')"),
            "a,b"
        );
        assert_eq!(
            run("[...new Map([['a',1],['b',2]]).values()].join(',')"),
            "1,2"
        );
        assert_eq!(run("new Map([['a',1],['b',2]]).entries().length"), "2");
        // Set values/keys are its elements.
        assert_eq!(run("[...new Set([1,2,3,2]).values()].join(',')"), "1,2,3");
        assert_eq!(run("[...new Set([5,6]).keys()].join(',')"), "5,6");
    }

    #[test]
    fn eager_generators() {
        // for-of and spread over a finite generator.
        assert_eq!(
            run(
                "function* g(n){ for (let i=0;i<n;i++) yield i*i; } let s=[]; for (let v of g(4)) s.push(v); s.join(',')"
            ),
            "0,1,4,9"
        );
        assert_eq!(
            run("function* g(){ yield 'a'; yield 'b'; } [...g()].join('-')"),
            "a-b"
        );
        // The next() iterator protocol.
        assert_eq!(
            run(
                "function* g(){ yield 1; yield 2; } let it=g(); '' + it.next().value + it.next().value + it.next().done"
            ),
            "12true"
        );
        // yield* delegation.
        assert_eq!(
            run(
                "function* inner(){ yield 2; yield 3; } function* outer(){ yield 1; yield* inner(); yield 4; } [...outer()].join(',')"
            ),
            "1,2,3,4"
        );
    }

    #[test]
    fn new_on_constructor_functions() {
        // `this` binding + implicit instance return.
        assert_eq!(run("function P(x){ this.x = x; } new P(7).x"), "7");
        // `instanceof` matches the constructing function, not others.
        assert_eq!(
            run(
                "function P(){} function Q(){} let p = new P(); '' + (p instanceof P) + (p instanceof Q)"
            ),
            "truefalse"
        );
        // An explicit object return overrides the new instance.
        assert_eq!(
            run("function F(){ this.a = 1; return { b: 2 }; } let o = new F(); '' + o.a + o.b"),
            "undefined2"
        );
        // The hidden constructor tag does not enumerate.
        assert_eq!(
            run("function P(){ this.v = 1; } Object.keys(new P()).join(',')"),
            "v"
        );
    }

    #[test]
    fn class_methods_are_non_enumerable() {
        // Methods are callable but absent from enumeration (only public fields
        // show up), and `{...obj}` spread skips them too.
        assert_eq!(
            run(
                "class C { m(){ return 1; } constructor(){ this.a = 1; this.b = 2; } } Object.keys(new C()).join(',')"
            ),
            "a,b"
        );
        assert_eq!(
            run(
                "class C { greet(){ return 'hi'; } } let c = new C(); c.greet() + ':' + Object.keys({ ...c }).length"
            ),
            "hi:0"
        );
    }

    #[test]
    fn optional_calls_destructuring_assign_and_coercion() {
        // Optional calls short-circuit on a nullish callee.
        assert_eq!(run("let o = { f: () => 7 }; o.f?.()"), "7");
        assert_eq!(run("let o = {}; String(o.missing?.())"), "undefined");
        // Destructuring assignment (swap, rest, member targets).
        assert_eq!(run("let a = 1, b = 2; [a, b] = [b, a]; a + ',' + b"), "2,1");
        assert_eq!(
            run("let h, t; [h, ...t] = [1, 2, 3, 4]; h + '|' + t.join(',')"),
            "1|2,3,4"
        );
        assert_eq!(
            run("let p = {}; ({ x: p.px, y: p.py } = { x: 10, y: 20 }); p.px + ',' + p.py"),
            "10,20"
        );
        // `+` ToPrimitive: arrays/objects stringify.
        assert_eq!(run("'' + [1, 2, 3]"), "1,2,3");
        assert_eq!(run("String([1, 2] + [3, 4])"), "1,23,4");
        assert_eq!(run("({}) + '!'"), "[object Object]!");
        // instanceof on error objects.
        assert_eq!(
            run("try { null.x; } catch (e) { '' + (e instanceof TypeError); }"),
            "true"
        );
        assert_eq!(
            run("try { nope; } catch (e) { '' + (e instanceof ReferenceError); }"),
            "true"
        );
    }

    #[test]
    fn labeled_loops_and_do_while() {
        // `continue label` to an outer loop.
        assert_eq!(
            run("let count = 0;
                 outer: for (let i = 0; i < 3; i++) {
                   for (let j = 0; j < 3; j++) {
                     if (j === 1) continue outer;
                     count++;
                   }
                 }
                 count"),
            "3"
        );
        // `break label` out of nested loops.
        assert_eq!(
            run("let hits = 0;
                 search: for (let i = 0; i < 5; i++) {
                   for (let j = 0; j < 5; j++) {
                     hits++;
                     if (i === 1 && j === 1) break search;
                   }
                 }
                 hits"),
            "7"
        );
        // do/while runs the body at least once.
        assert_eq!(
            run("let n = 0, s = 0; do { s += n; n++; } while (n < 4); s"),
            "6"
        );
        assert_eq!(run("let r = 0; do { r++; } while (false); r"), "1");
    }

    #[test]
    fn for_of_for_in_switch() {
        // for-of over an array, a string, a Set, and a Map.
        assert_eq!(run("let s = 0; for (const x of [1, 2, 3]) s += x; s"), "6");
        assert_eq!(
            run("let r = ''; for (const c of 'abc') r += c + '.'; r"),
            "a.b.c."
        );
        assert_eq!(
            run("let s = 0; for (const v of new Set([1, 2, 3, 2])) s += v; s"),
            "6"
        );
        assert_eq!(
            run("let r = ''; for (const [k, v] of new Map([['a', 1], ['b', 2]])) r += k + v; r"),
            "a1b2"
        );
        // for-of with break/continue.
        assert_eq!(
            run("let s = 0; for (const x of [1, 2, 3, 4]) { if (x === 3) break; s += x; } s"),
            "3"
        );
        // for-in over object keys and array indices.
        assert_eq!(
            run("let r = ''; for (const k in { a: 1, b: 2 }) r += k; r"),
            "ab"
        );
        assert_eq!(
            run("let r = ''; for (const i in ['x', 'y', 'z']) r += i; r"),
            "012"
        );
        // for-of binding to an existing variable (no declaration).
        assert_eq!(run("let x; let s = 0; for (x of [10, 20]) s += x; s"), "30");
        // switch with fall-through and default.
        assert_eq!(
            run("function f(n) {
                   let r = '';
                   switch (n) {
                     case 1: r += 'one';
                     case 2: r += 'two'; break;
                     case 3: r += 'three'; break;
                     default: r += 'other';
                   }
                   return r;
                 }
                 f(1) + '|' + f(2) + '|' + f(3) + '|' + f(9)"),
            "onetwo|two|three|other"
        );
    }

    #[test]
    fn destructuring() {
        // Array destructuring with defaults, holes, and rest.
        assert_eq!(run("let [a, b] = [1, 2]; a + b"), "3");
        assert_eq!(run("let [a, , c] = [1, 2, 3]; a + c"), "4");
        assert_eq!(run("let [a, b = 9] = [1]; a + b"), "10");
        assert_eq!(
            run("let [first, ...rest] = [1, 2, 3, 4]; rest.join(',')"),
            "2,3,4"
        );
        // Object destructuring with shorthand, rename, default, and rest.
        assert_eq!(run("let { x, y } = { x: 1, y: 2 }; x + y"), "3");
        assert_eq!(run("let { a: p, b: q } = { a: 10, b: 20 }; p + q"), "30");
        assert_eq!(run("let { m = 7 } = {}; m"), "7");
        assert_eq!(
            run("let { a, ...others } = { a: 1, b: 2, c: 3 }; Object.keys(others).join(',')"),
            "b,c"
        );
        // Nested.
        assert_eq!(run("let { p: { q } } = { p: { q: 42 } }; q"), "42");
        assert_eq!(run("let [[a], [b]] = [[1], [2]]; a + b"), "3");
        // Destructuring function parameters.
        assert_eq!(run("function f([a, b]) { return a * b; } f([3, 4])"), "12");
        assert_eq!(
            run("function g({ x, y }) { return x + y; } g({ x: 5, y: 6 })"),
            "11"
        );
        // Default and rest parameters.
        assert_eq!(run("function h(a, b = 10) { return a + b; } h(5)"), "15");
        assert_eq!(
            run("function r(...xs) { return xs.length; } r(1, 2, 3)"),
            "3"
        );
    }

    #[test]
    fn maps_and_sets() {
        // Map: set/get/has/size/delete.
        assert_eq!(
            run("let m = new Map(); m.set('a', 1); m.set('b', 2); m.get('a') + m.get('b')"),
            "3"
        );
        assert_eq!(run("let m = new Map(); m.set('x', 1); m.has('x')"), "true");
        assert_eq!(run("let m = new Map(); m.set('x', 1); m.size"), "1");
        assert_eq!(
            run("let m = new Map(); m.set('x', 1); m.delete('x'); m.has('x')"),
            "false"
        );
        // set returns the map (chainable); overwriting a key keeps size.
        assert_eq!(
            run("let m = new Map(); m.set('a', 1); m.set('a', 9); m.get('a') + ':' + m.size"),
            "9:1"
        );
        // Map seeded from pairs.
        assert_eq!(
            run("let m = new Map([['a', 1], ['b', 2]]); m.get('b')"),
            "2"
        );
        // Set: add/has/size, dedup, and seeding from an array.
        assert_eq!(
            run("let s = new Set(); s.add(1); s.add(1); s.add(2); s.size"),
            "2"
        );
        assert_eq!(run("let s = new Set([1, 2, 2, 3]); s.size"), "3");
        assert_eq!(run("new Set([1, 2, 3]).has(2)"), "true");
        // forEach over a Map accumulating values.
        assert_eq!(
            run("let m = new Map([['a', 10], ['b', 20]]);
                 let t = 0; m.forEach(v => { t += v; }); t"),
            "30"
        );
        // typeof a Map is object.
        assert_eq!(run("typeof new Map()"), "object");
    }

    #[test]
    fn more_string_methods() {
        assert_eq!(run("'hello world'.slice(0, 5)"), "hello");
        assert_eq!(run("'hello'.slice(-3)"), "llo");
        assert_eq!(run("'a,b,c'.split(',').join('|')"), "a|b|c");
        assert_eq!(run("'hello'.startsWith('he')"), "true");
        assert_eq!(run("'hello'.endsWith('lo')"), "true");
        assert_eq!(run("'a-b-a'.replace('a', 'X')"), "X-b-a"); // first only
        assert_eq!(run("'5'.padStart(3, '0')"), "005");
    }

    #[test]
    fn more_array_methods() {
        assert_eq!(run("[1, 2, 3, 4].slice(1, 3).join(',')"), "2,3");
        assert_eq!(run("[1, 2].concat([3, 4], 5).join(',')"), "1,2,3,4,5");
        assert_eq!(run("[1, 2, 3].reverse().join(',')"), "3,2,1");
        assert_eq!(run("[1, 2, 3, 4].find(x => x > 2)"), "3");
        assert_eq!(run("[1, 2, 3, 4].findIndex(x => x > 2)"), "2");
        assert_eq!(run("[1, 2, 3].some(x => x === 2)"), "true");
        assert_eq!(run("[1, 2, 3].every(x => x > 0)"), "true");
        assert_eq!(run("[1, 2, 3].every(x => x > 1)"), "false");
        // Default sort (string order) and comparator sort.
        assert_eq!(run("[3, 1, 2].sort().join(',')"), "1,2,3");
        assert_eq!(run("[10, 9, 100].sort().join(',')"), "10,100,9"); // string order
        assert_eq!(
            run("[10, 9, 100].sort((a, b) => a - b).join(',')"),
            "9,10,100"
        );
        assert_eq!(run("[3, 1, 2].sort((a, b) => b - a).join(',')"), "3,2,1");
    }

    #[test]
    fn higher_order_array_methods() {
        // map / filter / reduce with closures.
        assert_eq!(run("[1, 2, 3].map(x => x * 2).join(',')"), "2,4,6");
        assert_eq!(
            run("[1, 2, 3, 4].filter(x => x % 2 === 0).join(',')"),
            "2,4"
        );
        assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
        assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b)"), "10"); // no initial
        // forEach with a closed-over accumulator.
        assert_eq!(
            run("let total = 0; [10, 20, 30].forEach(x => { total += x; }); total"),
            "60"
        );
        // Chained, with a captured multiplier.
        assert_eq!(
            run("let k = 3;
                 [1, 2, 3, 4].filter(x => x > 1).map(x => x * k).reduce((a, b) => a + b, 0)"),
            "27"
        );
    }
}
