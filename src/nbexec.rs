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
    /// The function's name (`fn.name`); empty for anonymous functions.
    name: &'a str,
    /// The class this is a method of (for `super.method()`), if any.
    home_class: Option<u32>,
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
    /// Per-class static getter functions (`static get x() {}`), called on read.
    class_static_get: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class static setter functions (`static set x(v) {}`), called on write.
    class_static_set: Vec<alloc::collections::BTreeMap<String, NanBox>>,
    /// Per-class captured definition scope, parallel to `classes`.
    class_envs: Vec<Scope>,
    /// The current `this` binding (method/constructor receiver).
    this_val: NanBox,
    /// When running a generator body eagerly, the buffer `yield` appends to.
    gen_sink: Option<Vec<NanBox>>,
    /// The `Symbol.for` global registry: shared symbols keyed by string.
    symbol_registry: alloc::collections::BTreeMap<String, NanBox>,
    /// Cached well-known symbols (e.g. `Symbol.iterator`), created on first use.
    well_known_symbols: alloc::collections::BTreeMap<&'static str, NanBox>,
    /// The superclass to invoke for `super(...)` inside the running constructor.
    pending_super: Option<(u32, Scope)>,
    /// The class of the currently-running method (for `super.method()`).
    current_home: Option<u32>,
    /// A label attached to the next loop (for `break`/`continue label`).
    pending_label: Option<String>,
    /// The promise-reaction microtask queue, drained after the script.
    microtasks: Vec<Job>,
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
const N_OBJECT_CREATE: u16 = 107;
const N_OBJECT_GET_PROTO: u16 = 108;
const N_OBJECT_SET_PROTO: u16 = 109;
const N_OBJECT_DEFINE_PROP: u16 = 110;
const N_OBJECT_GET_OWN_DESC: u16 = 111;
const N_WEAKMAP: u16 = 112;
const N_OBJECT_IS: u16 = 123;
const N_OBJECT_HAS_OWN: u16 = 129;
const N_OBJECT_GET_OWN_DESCS: u16 = 130;
const N_OBJECT_DEFINE_PROPS: u16 = 124;
const N_WEAKSET: u16 = 113;
const N_REFLECT_GET: u16 = 114;
const N_REFLECT_SET: u16 = 115;
const N_REFLECT_HAS: u16 = 116;
const N_REFLECT_OWN_KEYS: u16 = 117;
const N_REFLECT_DELETE: u16 = 118;
const N_REFLECT_APPLY: u16 = 119;
const N_REFLECT_CONSTRUCT: u16 = 120;
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
const ERROR_NAMES: [&str; 5] = [
    "Error",
    "TypeError",
    "RangeError",
    "SyntaxError",
    "ReferenceError",
];
const N_TYPE_ERROR: u16 = N_ERROR_BASE + 1;
const N_REFERENCE_ERROR: u16 = N_ERROR_BASE + 4;
/// A reserved, non-identifier key under which a `new fn()` instance records its
/// constructor function (a hidden, GC-traced slot) so `instanceof` can match it.
const CTOR_KEY: &str = "\u{0}ctor";
/// Reserved hidden keys for an eager generator's result object: the buffer of
/// yielded values and the current `next()` cursor.
/// Sentinel description for a `Symbol()` created with no argument (so its
/// `.description` is `undefined`, distinct from `Symbol("")`).
const SYMBOL_NO_DESC: &str = "\u{0}nodesc";
const GEN_BUF: &str = "\u{0}gbuf";
const GEN_IDX: &str = "\u{0}gidx";
/// A generator's `return` value, surfaced once after its yields are exhausted.
const GEN_RET: &str = "\u{0}gret";
/// Reserved hidden keys for a bound function (`Function.prototype.bind`).
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
            class_static_get: Vec::new(),
            class_static_set: Vec::new(),
            class_envs: Vec::new(),
            this_val: NanBox::undefined(),
            gen_sink: None,
            symbol_registry: alloc::collections::BTreeMap::new(),
            well_known_symbols: alloc::collections::BTreeMap::new(),
            pending_super: None,
            current_home: None,
            pending_label: None,
            microtasks: Vec::new(),
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

    /// Installs a small built-in library: the `Math` object and the global
    /// coercion/parse functions. (A token stdlib to prove the native-call path;
    /// the full port is the remaining migration work.)
    fn install_globals(&mut self) {
        // An object whose properties are native methods, bound to `global_name`.
        let install_namespace = |this: &mut Self, global_name: &str, methods: &[(&str, u16)]| {
            let obj = this.realm.new_object();
            for (name, id) in methods {
                let f = this.realm.new_native(*id);
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
                ("trunc", N_MATH_TRUNC),
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
        let promise_ctor = self.realm.new_native(N_PROMISE);
        self.current
            .declare("Promise", NanBox::handle(promise_ctor.to_raw()));
        // `Date` is a native constructor; `Date.now()` is a static.
        let date_ctor = self.realm.new_native(N_DATE);
        self.current
            .declare("Date", NanBox::handle(date_ctor.to_raw()));
        // `RegExp` is a native constructor.
        let regexp_ctor = self.realm.new_native(N_REGEXP);
        self.current
            .declare("RegExp", NanBox::handle(regexp_ctor.to_raw()));
        // The `Error` family — native constructors producing `{ name, message }`.
        for (i, name) in ERROR_NAMES.iter().enumerate() {
            let ctor = self.realm.new_native(N_ERROR_BASE + i as u16);
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
                ("create", N_OBJECT_CREATE),
                ("getPrototypeOf", N_OBJECT_GET_PROTO),
                ("setPrototypeOf", N_OBJECT_SET_PROTO),
                ("defineProperty", N_OBJECT_DEFINE_PROP),
                ("defineProperties", N_OBJECT_DEFINE_PROPS),
                ("getOwnPropertyDescriptor", N_OBJECT_GET_OWN_DESC),
                ("getOwnPropertyDescriptors", N_OBJECT_GET_OWN_DESCS),
                ("is", N_OBJECT_IS),
                ("hasOwn", N_OBJECT_HAS_OWN),
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
                ("deleteProperty", N_REFLECT_DELETE),
                ("apply", N_REFLECT_APPLY),
                ("construct", N_REFLECT_CONSTRUCT),
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
            ("Proxy", N_PROXY),
            ("WeakMap", N_WEAKMAP),
            ("WeakSet", N_WEAKSET),
        ] {
            let f = self.realm.new_native(id);
            self.current.declare(name, NanBox::handle(f.to_raw()));
        }
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
                    if n > m {
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
                    if n < m {
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
                // `Number(obj)` runs the object through ToNumber (number hint),
                // honoring a custom `valueOf`.
                let p = self.coerce_object(arg(0), "number")?;
                NanBox::number(self.realm.to_number(p))
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
                    crate::bignum::BigInt::from_i128(self.realm.to_number(v) as i128)
                };
                NanBox::handle(self.realm.new_bigint(n).to_raw())
            }
            N_PARSE_INT => {
                let s = self.realm.to_display_string(arg(0));
                let radix = match args.get(1) {
                    Some(r) if !matches!(r.unpack(), Unpacked::Undefined) => {
                        self.realm.to_number(*r) as u32
                    }
                    _ => 0,
                };
                NanBox::number(parse_int(&s, radix))
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
                    crate::json::stringify_pretty(&self.realm, value, &indent)
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
                    return Err(ExecError::Throw(self.new_str("Unexpected token in JSON")));
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
                let keys = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.object_keys(Handle::from_raw(raw)))
                    .unwrap_or_default();
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
            N_OBJECT_IS_SEALED => NanBox::boolean(
                arg(0)
                    .as_handle()
                    .is_some_and(|raw| self.realm.is_sealed(Handle::from_raw(raw))),
            ),
            N_OBJECT_IS_EXTENSIBLE => NanBox::boolean(
                arg(0)
                    .as_handle()
                    .is_some_and(|raw| self.realm.is_extensible(Handle::from_raw(raw))),
            ),
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
                            self.apply_descriptor(obj, &key, Handle::from_raw(d));
                        }
                    }
                }
                NanBox::handle(obj.to_raw())
            }
            N_OBJECT_GET_PROTO => arg(0)
                .as_handle()
                .and_then(|raw| self.realm.object_proto(Handle::from_raw(raw)))
                .map_or(NanBox::null(), |p| NanBox::handle(p.to_raw())),
            N_OBJECT_SET_PROTO => {
                if let Some(raw) = arg(0).as_handle() {
                    let proto = arg(1).as_handle().map(Handle::from_raw);
                    self.realm.set_object_proto(Handle::from_raw(raw), proto);
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
                    let key = self.realm.to_display_string(arg(1));
                    self.apply_descriptor(obj, &key, Handle::from_raw(draw));
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
                            self.apply_descriptor(obj, &key, Handle::from_raw(d));
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
            // --- Reflect.* ---
            N_REFLECT_GET => {
                if let Some(raw) = arg(0).as_handle() {
                    let key = self.member_key(arg(1));
                    return self.read_member(Handle::from_raw(raw), &key);
                }
                NanBox::undefined()
            }
            N_REFLECT_SET => {
                if let Some(raw) = arg(0).as_handle() {
                    let key = self.member_key(arg(1));
                    self.realm.set_property(Handle::from_raw(raw), &key, arg(2));
                }
                NanBox::boolean(true)
            }
            N_REFLECT_HAS => {
                let key = self.member_key(arg(1));
                NanBox::boolean(
                    arg(0)
                        .as_handle()
                        .map(Handle::from_raw)
                        .is_some_and(|h| self.realm.has_own(h, &key) || self.realm.is_array(h)),
                )
            }
            N_REFLECT_DELETE => {
                if let Some(raw) = arg(0).as_handle() {
                    let key = self.member_key(arg(1));
                    self.realm.delete_property(Handle::from_raw(raw), &key);
                }
                NanBox::boolean(true)
            }
            N_REFLECT_OWN_KEYS => {
                let names = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.own_property_names(Handle::from_raw(raw)))
                    .unwrap_or_default();
                let boxed: Vec<NanBox> = names.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
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
                return self.construct(arg(0), &list);
            }
            N_OBJECT_GET_OWN_DESC => match arg(0).as_handle().map(Handle::from_raw) {
                Some(obj) => {
                    let key = self.realm.to_display_string(arg(1));
                    self.build_descriptor(obj, &key)
                        .unwrap_or(NanBox::undefined())
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
            N_OBJECT_IS_FROZEN => NanBox::boolean(
                arg(0)
                    .as_handle()
                    .is_some_and(|raw| self.realm.is_frozen(Handle::from_raw(raw))),
            ),
            N_OBJECT_GET_OWN_NAMES => {
                let names = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.own_property_names(Handle::from_raw(raw)))
                    .unwrap_or_default();
                let boxed: Vec<NanBox> = names.iter().map(|k| self.new_str(k)).collect();
                NanBox::handle(self.realm.new_array(boxed).to_raw())
            }
            N_OBJECT_VALUES => {
                let mut vals = Vec::new();
                if let Some(raw) = arg(0).as_handle() {
                    let h = Handle::from_raw(raw);
                    for k in self.realm.object_keys(h).unwrap_or_default() {
                        vals.push(
                            self.realm
                                .get_property(h, &k)
                                .unwrap_or(NanBox::undefined()),
                        );
                    }
                }
                NanBox::handle(self.realm.new_array(vals).to_raw())
            }
            N_ARRAY_IS_ARRAY => NanBox::boolean(
                arg(0)
                    .as_handle()
                    .is_some_and(|raw| self.realm.is_array(Handle::from_raw(raw))),
            ),
            N_OBJECT_ASSIGN => {
                let target = arg(0);
                if let Some(t) = target.as_handle().map(Handle::from_raw) {
                    for src in &args[1.min(args.len())..] {
                        if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                            // Data keys plus accessor (getter) keys, read via
                            // `read_member` so getters are invoked.
                            let mut keys = self.realm.object_keys(sh).unwrap_or_default();
                            keys.extend(self.realm.object_accessor_keys(sh));
                            for k in keys {
                                let v = self.read_member(sh, &k)?;
                                self.realm.set_property(t, &k, v);
                            }
                        }
                    }
                }
                target
            }
            N_OBJECT_ENTRIES => {
                let mut pairs = Vec::new();
                if let Some(h) = arg(0).as_handle().map(Handle::from_raw) {
                    for k in self.realm.object_keys(h).unwrap_or_default() {
                        let v = self
                            .realm
                            .get_property(h, &k)
                            .unwrap_or(NanBox::undefined());
                        let key = self.new_str(&k);
                        let pair = self.realm.new_array(alloc::vec![key, v]);
                        pairs.push(NanBox::handle(pair.to_raw()));
                    }
                }
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
                    let mut out = Vec::with_capacity(items.len());
                    for (i, e) in items.iter().enumerate() {
                        out.push(self.call(f, &[*e, NanBox::number(i as f64)])?);
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
                NanBox::number(if n.is_finite() { (n + 0.5).floor() } else { n })
            }
            #[cfg(feature = "std")]
            N_MATH_SQRT => NanBox::number(self.realm.to_number(arg(0)).sqrt()),
            #[cfg(not(feature = "std"))]
            N_MATH_FLOOR | N_MATH_CEIL | N_MATH_ROUND | N_MATH_SQRT => {
                return Err(ExecError::Unsupported("Math float ops need std"));
            }
            #[cfg(feature = "std")]
            N_MATH_POW => NanBox::number(
                self.realm
                    .to_number(arg(0))
                    .powf(self.realm.to_number(arg(1))),
            ),
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
                let sum: f64 = args.iter().map(|a| self.realm.to_number(*a).powi(2)).sum();
                NanBox::number(sum.sqrt())
            }
            #[cfg(feature = "std")]
            N_MATH_CBRT => NanBox::number(self.realm.to_number(arg(0)).cbrt()),
            #[cfg(feature = "std")]
            N_MATH_LOG2 => NanBox::number(self.realm.to_number(arg(0)).log2()),
            #[cfg(feature = "std")]
            N_MATH_LOG10 => NanBox::number(self.realm.to_number(arg(0)).log10()),
            #[cfg(not(feature = "std"))]
            N_MATH_HYPOT | N_MATH_CBRT | N_MATH_LOG2 | N_MATH_LOG10 => {
                return Err(ExecError::Unsupported("Math fns need std"));
            }
            N_PARSE_FLOAT => {
                let s = self.realm.to_display_string(arg(0));
                NanBox::number(parse_float_prefix(s.trim()))
            }
            N_IS_NAN => NanBox::boolean(self.realm.to_number(arg(0)).is_nan()),
            N_IS_FINITE => NanBox::boolean(self.realm.to_number(arg(0)).is_finite()),
            // `Error(msg)` called without `new` builds the same object.
            id if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) => {
                self.make_error(id, args.first().copied())
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
                for k in keys {
                    if allow.iter().any(|a| a == &k) {
                        let pv = self
                            .realm
                            .get_property(vh, &k)
                            .unwrap_or(NanBox::undefined());
                        let nv = self.json_filter_keys(pv, allow);
                        self.realm.set_property(no, &k, nv);
                    }
                }
                return NanBox::handle(no.to_raw());
            }
        }
        value
    }

    fn json_parse(&mut self, c: &[char], pos: &mut usize) -> Result<NanBox, ExecError> {
        skip_ws(c, pos);
        let err = |s: &mut Self| ExecError::Throw(s.new_str("Unexpected end of JSON input"));
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
                        _ => return Err(ExecError::Throw(self.new_str("Expected ',' or ']'"))),
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
                        return Err(ExecError::Throw(self.new_str("Expected property name")));
                    }
                    let key = self.json_string(c, pos)?;
                    skip_ws(c, pos);
                    if c.get(*pos) != Some(&':') {
                        return Err(ExecError::Throw(self.new_str("Expected ':'")));
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
                        _ => return Err(ExecError::Throw(self.new_str("Expected ',' or '}'"))),
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
                    .map_err(|_| ExecError::Throw(self.new_str("Invalid number in JSON")))
            }
            _ => Err(ExecError::Throw(self.new_str("Unexpected token in JSON"))),
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
            Err(ExecError::Throw(self.new_str("Unexpected token in JSON")))
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
                    return Err(ExecError::Throw(
                        self.new_str("Unterminated string in JSON"),
                    ));
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
                                .ok_or_else(|| {
                                    ExecError::Throw(self.new_str("Invalid \\u escape in JSON"))
                                })?;
                            out.push(code);
                            *pos += 4;
                        }
                        _ => return Err(ExecError::Throw(self.new_str("Invalid escape in JSON"))),
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
                let r = self.call_with_this(tj, v, &[])?;
                return self.json_to_string(r);
            }
        }
        match v.unpack() {
            Unpacked::Undefined => Ok(None),
            Unpacked::Null => Ok(Some(String::from("null"))),
            Unpacked::Bool(b) => Ok(Some(String::from(if b { "true" } else { "false" }))),
            Unpacked::Number(n) => Ok(Some(if n.is_finite() {
                alloc::format!("{n}")
            } else {
                String::from("null")
            })),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(s) = self.realm.string_value(h) {
                    return Ok(Some(json_quote(&s)));
                }
                if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                    let mut parts = Vec::with_capacity(elems.len());
                    for e in elems {
                        parts.push(
                            self.json_to_string(e)?
                                .unwrap_or_else(|| String::from("null")),
                        );
                    }
                    return Ok(Some(alloc::format!("[{}]", parts.join(","))));
                }
                if self.realm.object_keys(h).is_some() {
                    // Data keys plus accessor (getter) keys, read via read_member.
                    let mut keys = self.realm.object_keys(h).unwrap_or_default();
                    keys.extend(self.realm.object_accessor_keys(h));
                    let mut parts = Vec::new();
                    for k in keys {
                        let val = self.read_member(h, &k)?;
                        if let Some(s) = self.json_to_string(val)? {
                            parts.push(alloc::format!("{}:{}", json_quote(&k), s));
                        }
                    }
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

    /// Runs a whole program, returning the value of its last expression
    /// statement (or `undefined`).
    pub fn run(&mut self, program: &'a Program) -> Result<NanBox, ExecError> {
        self.hoist_with(&program.body, true)?;
        let mut last = NanBox::undefined();
        for stmt in &program.body {
            match self.exec(stmt)? {
                Flow::Normal(v) => last = v,
                Flow::Return(v) => {
                    self.drain_microtasks()?;
                    return Ok(v);
                }
                Flow::Break(_) | Flow::Continue(_) => {}
            }
        }
        // Drain the promise microtask queue (the event loop) before returning.
        self.drain_microtasks()?;
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
                self.current.declare(&id.name, value);
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
        self.make_method(params, body, is_async, is_generator, None)
    }

    fn make_method(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        is_generator: bool,
        home_class: Option<u32>,
    ) -> NanBox {
        let func_id = self.functions.len() as u32;
        self.functions.push(FnDef {
            params,
            body,
            is_async,
            is_generator,
            is_arrow: false,
            name: "",
            home_class,
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
            );
        } else {
            self.settle(handle, value, true);
        }
    }

    /// Registers `then` reactions on `handle`, returning a new dependent promise.
    fn promise_then(&mut self, handle: Handle, on_f: NanBox, on_r: NanBox) -> NanBox {
        let result = self.register_then(handle, on_f, on_r);
        NanBox::handle(result.to_raw())
    }

    fn register_then(&mut self, handle: Handle, on_f: NanBox, on_r: NanBox) -> Handle {
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
            }),
            Some((fulfilled, value)) => {
                let handler = if fulfilled { on_f } else { on_r };
                self.microtasks.push(Job {
                    handler,
                    value,
                    result,
                    fulfilled,
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

    /// Runs the next queued promise reaction.
    fn run_one_microtask(&mut self) -> Result<(), ExecError> {
        let job = self.microtasks.remove(0);
        if job
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
    fn await_value(&mut self, value: NanBox) -> Result<NanBox, ExecError> {
        use crate::cell::PromiseStatus::{Fulfilled, Pending, Rejected};
        let Some(state) = value
            .as_handle()
            .and_then(|raw| self.realm.promise_state(Handle::from_raw(raw)))
        else {
            return Ok(value); // not a promise
        };
        while state.borrow().status == Pending && !self.microtasks.is_empty() {
            self.run_one_microtask()?;
        }
        let s = state.borrow();
        match s.status {
            Fulfilled => Ok(s.value),
            Rejected => Err(ExecError::Throw(s.value)),
            Pending => Ok(NanBox::undefined()), // never settled (no timers)
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
        if let Some((g, s)) = self.realm.accessor(obj, key) {
            let d = self.realm.new_object();
            self.realm.set_property(d, "get", g);
            self.realm.set_property(d, "set", s);
            self.realm.set_property(d, "enumerable", t);
            self.realm.set_property(d, "configurable", t);
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
            self.realm.set_property(d, "configurable", t);
            Some(NanBox::handle(d.to_raw()))
        } else {
            None
        }
    }

    fn apply_descriptor(&mut self, obj: Handle, key: &str, desc: Handle) {
        let getter = self.realm.get_property(desc, "get");
        let setter = self.realm.get_property(desc, "set");
        if getter.is_some() || setter.is_some() {
            self.realm.define_accessor(
                obj,
                key,
                getter.unwrap_or(NanBox::undefined()),
                setter.unwrap_or(NanBox::undefined()),
            );
        } else {
            let value = self
                .realm
                .get_property(desc, "value")
                .unwrap_or(NanBox::undefined());
            // Set the value first, then apply the attribute flags.
            self.realm.set_property(obj, key, value);
            if matches!(
                self.realm
                    .get_property(desc, "writable")
                    .map(|v| self.realm.truthy(v)),
                Some(false)
            ) {
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
        // A built-in function dispatches directly.
        if let Some(id) = self.realm.native_at(handle) {
            return self.call_native(id, args);
        }
        // A bound native (promise resolve/reject) carries its target.
        if let Some((id, target)) = self.realm.bound_native_at(handle) {
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
        self.invoke(def, captured, this_val, args)
    }

    /// Runs a function body with `this` and the parameters bound in a fresh
    /// child of `captured`.
    fn invoke(
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
            core::mem::replace(&mut self.this_val, this_val)
        };
        let saved_home = core::mem::replace(&mut self.current_home, def.home_class);
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
        // `new UserClass(...)`.
        if let Some((class_id, env)) = self.realm.class_at(handle) {
            return self.instantiate(class_id, &env, args);
        }
        // `new constructorFunction(...)`: bind a fresh object as `this`, run the
        // body, and return it — unless the function explicitly returned an object
        // (the spec's constructor return rule).
        if let Some((func_id, _)) = self.realm.function_at(handle) {
            // The instance's `[[Prototype]]` is the constructor's `.prototype`,
            // so inherited methods/getters resolve through the chain.
            let proto = self.realm.function_prototype(func_id);
            let instance = self.realm.new_object_with_proto(Some(proto));
            let this = NanBox::handle(instance.to_raw());
            // Record the constructor for `instanceof` (hidden, GC-traced slot).
            self.realm.set_hidden_property(instance, CTOR_KEY, callee);
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
                alloc::vec![NanBox::undefined(); n.max(0.0) as usize]
            } else {
                args.to_vec()
            };
            return Ok(NanBox::handle(self.realm.new_array(elems).to_raw()));
        }
        let id = self
            .realm
            .native_at(handle)
            .ok_or(ExecError::Unsupported("new on this value"))?;
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
                    Some(a) => self.realm.to_number(*a),
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
            let r = self.realm.new_regexp(&pat, &flags);
            return Ok(NanBox::handle(r.to_raw()));
        }
        // `new Error(message)` and friends → `{ name, message }`.
        if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
            return Ok(self.make_error(id, args.first().copied()));
        }
        // `WeakMap`/`WeakSet` reuse the collection cell (no true weak refs here).
        let is_set = match id {
            N_SET | N_WEAKSET => true,
            N_MAP | N_WEAKMAP => false,
            _ => return Err(ExecError::Unsupported("new on this constructor")),
        };
        let handle = self.realm.new_collection(is_set);
        // Seed from an iterable: a `Set` from array elements, a `Map` from
        // `[key, value]` pairs.
        if let Some(seed) = args
            .first()
            .copied()
            .and_then(NanBox::as_handle)
            .and_then(|r| self.realm.array_elements(Handle::from_raw(r)))
            .map(<[_]>::to_vec)
        {
            for item in seed {
                if is_set {
                    self.realm.collection_set(handle, item, item);
                } else if let Some(pr) = item
                    .as_handle()
                    .and_then(|r| self.realm.array_elements(Handle::from_raw(r)))
                    .map(<[_]>::to_vec)
                {
                    let k = pr.first().copied().unwrap_or(NanBox::undefined());
                    let v = pr.get(1).copied().unwrap_or(NanBox::undefined());
                    self.realm.collection_set(handle, k, v);
                }
            }
        }
        Ok(NanBox::handle(handle.to_raw()))
    }

    /// Builds a match-result object (`[0..n]` groups, plus `index`, `input`,
    /// `length`) from regex captures over `text`.
    #[cfg(feature = "regex")]
    fn regex_match_object(&mut self, text: &str, caps: &crate::regex::Captures) -> NanBox {
        let obj = self.realm.new_object();
        for (i, g) in caps.groups.iter().enumerate() {
            let v = match g {
                Some((s, e)) => self.new_str(&text[*s..*e]),
                None => NanBox::undefined(),
            };
            self.realm.set_property(obj, &alloc::format!("{i}"), v);
        }
        let index = caps.groups.first().and_then(|g| *g).map_or(0, |(s, _)| s);
        self.realm
            .set_property(obj, "index", NanBox::number(index as f64));
        let input = self.new_str(text);
        self.realm.set_property(obj, "input", input);
        self.realm
            .set_property(obj, "length", NanBox::number(caps.groups.len() as f64));
        NanBox::handle(obj.to_raw())
    }

    /// Builds an error object `{ name, message }` for the constructor `id`.
    fn make_error(&mut self, id: u16, message: Option<NanBox>) -> NanBox {
        let name = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
        let obj = self.realm.new_object();
        let name_v = self.new_str(name);
        self.realm.set_property(obj, "name", name_v);
        let msg = match message {
            Some(m) if !matches!(m.unpack(), Unpacked::Undefined) => {
                let s = self.realm.to_display_string(m);
                self.new_str(&s)
            }
            _ => self.new_str(""),
        };
        self.realm.set_property(obj, "message", msg);
        NanBox::handle(obj.to_raw())
    }

    /// Registers a class and allocates a class value capturing the current scope.
    fn make_class(&mut self, class: &'a Class) -> NanBox {
        let class_id = self.classes.len() as u32;
        self.classes.push(class);
        // Build the static members (`static foo() {}` / `static x = …`).
        let mut statics = alloc::collections::BTreeMap::new();
        let mut static_getters = alloc::collections::BTreeMap::new();
        let mut static_setters = alloc::collections::BTreeMap::new();
        for member in &class.body {
            match member {
                ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                    if let Ok(key) = static_key(&m.key) {
                        let f = self.make_function(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            m.value.is_generator,
                        );
                        statics.insert(key, f);
                    }
                }
                ClassMember::Field(field) if field.is_static => {
                    if let Ok(key) = static_key(&field.key) {
                        let v = match &field.value {
                            Some(e) => self.eval(e).unwrap_or(NanBox::undefined()),
                            None => NanBox::undefined(),
                        };
                        statics.insert(key, v);
                    }
                }
                // `static get x() {}` / `static set x(v) {}` — accessors.
                ClassMember::Method(m)
                    if m.is_static && matches!(m.kind, MethodKind::Get | MethodKind::Set) =>
                {
                    if let Ok(key) = static_key(&m.key) {
                        let f = self.make_function(
                            &m.value.params,
                            Body::Block(&m.value.body),
                            false,
                            false,
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
        self.class_static_get.push(static_getters);
        self.class_static_set.push(static_setters);
        self.class_envs.push(self.current.clone());
        let handle = self.realm.new_class(class_id, self.current.clone());
        NanBox::handle(handle.to_raw())
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
        self.realm
            .class_at(Handle::from_raw(raw))
            .map(Some)
            .ok_or(ExecError::Unsupported("extends a non-class"))
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
                    }
                    MethodKind::Set => {
                        self.realm
                            .define_accessor(instance, &key, NanBox::undefined(), f);
                    }
                    MethodKind::Constructor => {}
                }
            }
        }

        self.realm.set_class_tag(instance, class_id);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        let result = self.run_constructor(class_id, env, instance, args);
        self.this_val = saved_this;
        result?;
        Ok(this_val)
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
    ) -> Result<(), ExecError> {
        let class = self.classes[class_id as usize];
        let parent = self.resolve_super(class, env)?;
        let saved_super = core::mem::replace(&mut self.pending_super, parent.clone());
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
                    let r = (|| {
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
                        for stmt in &ctor.value.body {
                            if let Flow::Return(_) = self.exec(stmt)? {
                                break;
                            }
                        }
                        Ok(())
                    })();
                    self.current = saved;
                    r?;
                }
                // No own constructor but a base: implicit `super(args)`, then
                // this class's own field initializers.
                (None, Some((pid, penv))) => {
                    self.run_constructor(*pid, &penv.clone(), instance, args)?;
                    self.init_instance_fields(class_id, instance)?;
                }
                (None, None) => {
                    self.init_instance_fields(class_id, instance)?;
                }
            }
            Ok(())
        })();
        self.current = saved_scope;
        self.pending_super = saved_super;
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
                #[cfg(feature = "std")]
                "toFixed" => {
                    let digits = (self.realm.to_number(arg(0)) as usize).min(100);
                    // JS rounds half away from zero (Rust's formatter rounds half
                    // to even), so pre-round at the target scale.
                    let s = if n.is_finite() {
                        let factor = 10f64.powi(digits as i32);
                        let rounded = (n * factor).round() / factor;
                        alloc::format!("{rounded:.digits$}")
                    } else {
                        alloc::format!("{n}")
                    };
                    Some(self.new_str(&s))
                }
                // `toExponential(d)` — exponential notation with `d` fractional
                // digits and a signed exponent (`1.23e+3`).
                "toExponential" => {
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

        // --- universal `Object.prototype` methods (own/inherited reflection) ---
        match method {
            "hasOwnProperty" => {
                let key = self.realm.to_display_string(arg(0));
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
                let key = self.realm.to_display_string(arg(0));
                return Ok(Some(NanBox::boolean(self.realm.has_own(handle, &key))));
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
                let s = if msg.is_empty() {
                    name
                } else {
                    alloc::format!("{name}: {msg}")
                };
                return Ok(Some(self.new_str(&s)));
            }
            _ => {}
        }

        // --- `Function.prototype.call`/`apply`/`bind` on a callable receiver ---
        if self.is_callable(handle) {
            match method {
                "call" => {
                    let this = arg(0);
                    let rest: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return self.call_with_this(recv, this, &rest).map(Some);
                }
                "apply" => {
                    let this = arg(0);
                    let list = match arg(1).as_handle().map(Handle::from_raw) {
                        Some(h) => self
                            .realm
                            .array_elements(h)
                            .map(<[_]>::to_vec)
                            .unwrap_or_default(),
                        None => Vec::new(),
                    };
                    return self.call_with_this(recv, this, &list).map(Some);
                }
                "bind" => {
                    let this = arg(0);
                    let bound: Vec<NanBox> = args.iter().skip(1).copied().collect();
                    return Ok(Some(self.make_bound_function(recv, this, bound)));
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
                _ => {}
            }
        }

        // --- `Date.now()` static ---
        if self.realm.native_at(handle) == Some(N_DATE) && method == "now" {
            return Ok(Some(NanBox::number(now_ms())));
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
            return Ok(Some(self.new_str(&alloc::format!("Symbol({desc})"))));
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
                let s: String = args
                    .iter()
                    .filter_map(|a| char::from_u32(self.realm.to_number(*a) as u32))
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
                "toISOString" | "toJSON" => self.new_str(&crate::realm::date_to_iso(ms)),
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
                match (method, re) {
                    ("test", Ok(re)) => return Ok(Some(NanBox::boolean(re.is_match(&text)))),
                    ("exec", Ok(re)) => {
                        return Ok(Some(match re.captures_from(&text, 0) {
                            Some(caps) => self.regex_match_object(&text, &caps),
                            None => NanBox::null(),
                        }));
                    }
                    _ => {}
                }
            }
            #[cfg(not(feature = "regex"))]
            if matches!(method, "test" | "exec") {
                return Err(ExecError::Unsupported("RegExp needs the regex feature"));
            }
        }
        // --- `Promise.resolve` / `Promise.reject` statics (on the constructor) ---
        if self.realm.native_at(handle) == Some(N_PROMISE) {
            match method {
                "resolve" => {
                    let p = self.realm.new_promise();
                    self.resolve_with(p, arg(0));
                    return Ok(Some(NanBox::handle(p.to_raw())));
                }
                "reject" => {
                    let p = self.realm.new_promise();
                    self.settle(p, arg(0), false);
                    return Ok(Some(NanBox::handle(p.to_raw())));
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
                // `Promise.race(iterable)`: settle with the first input to settle
                // (eager — the first list element).
                "race" => {
                    let items = self.iterate_values(arg(0))?;
                    let p = self.realm.new_promise();
                    if let Some(item) = items.into_iter().next() {
                        match self.await_value(item) {
                            Ok(v) => self.resolve_with(p, v),
                            Err(ExecError::Throw(e)) => self.settle(p, e, false),
                            Err(other) => return Err(other),
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
                    // Simplified: run the callback on either settlement, passing
                    // the value through.
                    return Ok(Some(self.promise_then(handle, arg(0), arg(0))));
                }
                _ => {}
            }
        }

        // --- regex-backed String methods (when the argument is a RegExp) ---
        #[cfg(feature = "regex")]
        if let Some(s) = self.realm.string_value(handle)
            && matches!(
                method,
                "match" | "search" | "replace" | "replaceAll" | "split"
            )
            && let Some((src, flags)) = arg(0)
                .as_handle()
                .and_then(|raw| self.realm.regexp_at(Handle::from_raw(raw)))
            && let Ok(re) = crate::regex::Regex::new(&src, &flags)
        {
            let global = flags.contains('g');
            match method {
                "search" => {
                    let idx = re.find_from(&s, 0).map_or(-1.0, |(st, _)| st as f64);
                    return Ok(Some(NanBox::number(idx)));
                }
                "match" if !global => {
                    return Ok(Some(match re.captures_from(&s, 0) {
                        Some(caps) => self.regex_match_object(&s, &caps),
                        None => NanBox::null(),
                    }));
                }
                "match" => {
                    // Global: an array of all whole matches (or null).
                    let mut out = Vec::new();
                    let mut at = 0;
                    while let Some((st, en)) = re.find_from(&s, at) {
                        out.push(self.new_str(&s[st..en]));
                        at = if en > st { en } else { en + 1 };
                    }
                    return Ok(Some(if out.is_empty() {
                        NanBox::null()
                    } else {
                        NanBox::handle(self.realm.new_array(out).to_raw())
                    }));
                }
                "split" => {
                    let mut parts = Vec::new();
                    let mut at = 0;
                    while let Some(caps) = re.captures_from(&s, at) {
                        let Some((st, en)) = caps.groups[0] else {
                            break;
                        };
                        if en == at && st == at {
                            at += 1;
                            if at > s.len() {
                                break;
                            }
                            continue;
                        }
                        parts.push(self.new_str(&s[at..st]));
                        // The separator's capture groups are spliced into the
                        // result (`"a1b".split(/(\d)/)` → `["a","1","b"]`).
                        for g in &caps.groups[1..] {
                            match g {
                                Some((gs, ge)) => parts.push(self.new_str(&s[*gs..*ge])),
                                None => parts.push(NanBox::undefined()),
                            }
                        }
                        at = en;
                    }
                    parts.push(self.new_str(&s[at..]));
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
                        out.push_str(&s[at..st]);
                        if is_fn {
                            let mut call_args = alloc::vec![self.new_str(&s[st..en])];
                            for g in caps.groups.iter().skip(1) {
                                call_args.push(match g {
                                    Some((gs, ge)) => self.new_str(&s[*gs..*ge]),
                                    None => NanBox::undefined(),
                                });
                            }
                            call_args.push(NanBox::number(st as f64));
                            call_args.push(self.new_str(&s));
                            let r = self.call(replacer, &call_args)?;
                            let rep = self.realm.to_display_string(r);
                            out.push_str(&rep);
                        } else {
                            out.push_str(&expand_replacement(&templ, &s, &caps));
                        }
                        at = if en > st { en } else { en + 1 };
                        if !global || at > s.len() {
                            break;
                        }
                    }
                    out.push_str(&s[at.min(s.len())..]);
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
                    let i = self.realm.to_number(arg(0)) as usize;
                    // UTF-16-indexed: the unit at `i` as a one-unit string (a
                    // lone surrogate renders as U+FFFD via lossy decoding).
                    let units: Vec<u16> = s.encode_utf16().collect();
                    let out = units
                        .get(i)
                        .map(|&u| String::from_utf16_lossy(&[u]))
                        .unwrap_or_default();
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
                    let idx = s
                        .find(&needle)
                        .map_or(-1.0, |b| s[..b].chars().count() as f64);
                    Some(NanBox::number(idx))
                }
                "repeat" => {
                    let n = self.realm.to_number(arg(0));
                    let n = if n >= 0.0 { n as usize } else { 0 };
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
                        Some(self.new_str(&s.replacen(&from, &to, 1)))
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
                    } else {
                        let to = self.realm.to_display_string(repl);
                        Some(self.new_str(&s.replace(&from, &to)))
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
                // `charCodeAt(i)` is the UTF-16 code unit at index `i` (NaN if
                // out of range); a surrogate half reads as that 16-bit value.
                "charCodeAt" => {
                    let i = self.realm.to_number(arg(0)) as usize;
                    Some(
                        s.encode_utf16()
                            .nth(i)
                            .map_or(NanBox::number(f64::NAN), |u| NanBox::number(f64::from(u))),
                    )
                }
                // `codePointAt(i)` combines a surrogate pair at UTF-16 index `i`.
                "codePointAt" => {
                    let i = self.realm.to_number(arg(0)) as usize;
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
                    let idx = s
                        .rfind(&needle)
                        .map_or(-1.0, |b| s[..b].chars().count() as f64);
                    Some(NanBox::number(idx))
                }
                // `concat` appends each argument's string form.
                "concat" => {
                    let mut out = s.clone();
                    for a in args {
                        out.push_str(&self.realm.to_display_string(*a));
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
                "normalize" => Some(self.new_str(&s)),
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
                    // through ToString (so a custom `toString` is honored).
                    let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                    for e in &elems {
                        let s = match e.unpack() {
                            Unpacked::Null | Unpacked::Undefined => String::new(),
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
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                "with" => {
                    let mut out = elems.clone();
                    let i = self.realm.to_number(arg(0));
                    let idx = if i < 0.0 { out.len() as f64 + i } else { i } as usize;
                    if idx < out.len() {
                        out[idx] = arg(1);
                    }
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                "toSorted" => {
                    let sorted = self.sort_array(elems.clone(), arg(0))?;
                    return Ok(Some(NanBox::handle(self.realm.new_array(sorted).to_raw())));
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
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
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
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
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
                        return Err(ExecError::Throw(
                            self.new_str("Reduce of empty array with no initial value"),
                        ));
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
                        return Err(ExecError::Throw(
                            self.new_str("Reduce of empty array with no initial value"),
                        ));
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
                    let h = self.realm.new_array(elems[a..b].to_vec());
                    return Ok(Some(NanBox::handle(h.to_raw())));
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
                    for a in args {
                        if let Some(other) = a
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            out.extend(other);
                        } else {
                            out.push(*a);
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
                    // Sorts in place and returns the same array.
                    let sorted = self.sort_array(elems, arg(0))?;
                    self.realm.array_set_all(handle, sorted);
                    return Ok(Some(NanBox::handle(handle.to_raw())));
                }
                _ => {}
            }
        }

        // --- Map / Set methods ---
        if let Some(size) = self.realm.collection_size(handle) {
            match method {
                "set" => {
                    self.realm.collection_set(handle, arg(0), arg(1));
                    return Ok(Some(recv)); // Map.set returns the map (chainable)
                }
                "add" => {
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
                    for (k, v) in self.realm.collection_entries(handle).unwrap_or_default() {
                        // (value, key) per the JS signature.
                        self.call(f, &[v, k])?;
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
                _ => {
                    let _ = size;
                }
            }
        }
        Ok(None)
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
        mut elems: Vec<NanBox>,
        cmp: NanBox,
    ) -> Result<Vec<NanBox>, ExecError> {
        let has_cmp = cmp.as_handle().is_some_and(|raw| {
            let h = Handle::from_raw(raw);
            self.realm.native_at(h).is_some() || self.realm.function_at(h).is_some()
        });
        for i in 1..elems.len() {
            let mut j = i;
            while j > 0 {
                let order = if has_cmp {
                    let r = self.call(cmp, &[elems[j - 1], elems[j]])?;
                    self.realm.to_number(r)
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
        Ok(elems)
    }

    fn run_body(&mut self, body: Body<'a>) -> Result<NanBox, ExecError> {
        match body {
            Body::Expr(e) => self.eval(e),
            Body::Block(stmts) => {
                self.hoist_with(stmts, true)?;
                for stmt in stmts {
                    match self.exec(stmt)? {
                        Flow::Return(v) => return Ok(v),
                        Flow::Normal(_) | Flow::Break(_) | Flow::Continue(_) => {}
                    }
                }
                Ok(NanBox::undefined())
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
                let value = self.make_class(class);
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
                self.pending_label = Some(String::from(&*label.name));
                let flow = self.exec(body)?;
                self.pending_label = None;
                // A labeled non-loop block consumes a matching `break label`.
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
                left, right, body, ..
            } => {
                let iterable = self.eval(right)?;
                let values = self.iterate_values(iterable)?;
                self.exec_for_each(left, body, values)
            }
            Stmt::ForIn {
                left, right, body, ..
            } => {
                let obj = self.eval(right)?;
                let keys = self.iterate_keys(obj);
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
            if let Some(BindingTarget::Ident(Ident { name, .. })) = &catch.param {
                child.declare(name, thrown);
            }
            let saved = core::mem::replace(&mut self.current, child);
            outcome = self.exec_seq(&catch.body);
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
            // `var` assigns to its hoisted binding (in the function/program
            // scope), so a declaration inside a block updates the same variable.
            if is_var && let BindingTarget::Ident(Ident { name, .. }) = &d.target {
                if !self.current.set(name, value) {
                    self.current.declare(name, value);
                }
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
                let elems = value
                    .as_handle()
                    .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                    .map(<[_]>::to_vec)
                    .unwrap_or_default();
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
                let src = value.as_handle().map(Handle::from_raw);
                let mut used: Vec<String> = Vec::new();
                for prop in &pat.properties {
                    // A computed key (`{ [expr]: t }`) is evaluated here.
                    let key = self.eval_prop_key(&prop.key)?;
                    let mut v = src
                        .and_then(|h| self.realm.get_property(h, &key))
                        .unwrap_or(NanBox::undefined());
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
            return Err(ExecError::Throw(self.new_str("value is not iterable")));
        };
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
        let iter_sym = self.well_known_symbol("iterator");
        let iter_key = self.member_key(iter_sym);
        let iter_fn = self.realm.get_property(h, &iter_key);
        if let Some(f) = iter_fn
            && f.as_handle()
                .is_some_and(|raw| self.is_callable(Handle::from_raw(raw)))
        {
            let iterator = self.call_with_this(f, v, &[])?;
            let Some(ih) = iterator.as_handle().map(Handle::from_raw) else {
                return Err(ExecError::Throw(self.new_str("iterator is not an object")));
            };
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
        Err(ExecError::Throw(self.new_str("value is not iterable")))
    }

    /// The keys iterated by `for-in`: object property names or array indices,
    /// as strings.
    fn iterate_keys(&mut self, v: NanBox) -> Vec<NanBox> {
        let Some(h) = v.as_handle().map(Handle::from_raw) else {
            return Vec::new();
        };
        if let Some(len) = self.realm.array_length(h) {
            return (0..len)
                .map(|i| self.new_str(&alloc::format!("{i}")))
                .collect();
        }
        self.realm
            .object_keys(h)
            .unwrap_or_default()
            .iter()
            .map(|k| self.new_str(k))
            .collect()
    }

    /// Runs `body` once per `item`, binding the loop variable (a fresh scope per
    /// iteration for a declared head).
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
                let items = self.iterate_values(value).unwrap_or_default();
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
                if !self.current.set(&id.name, value) {
                    self.current.declare(&id.name, value);
                }
                Ok(())
            }
            Expr::Member {
                object, property, ..
            } => {
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
            return Ok(Some(self.call_with_this(f, v, &[hint_box])?));
        }
        Ok(None)
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
        let is_object = |this: &Self, r: NanBox| {
            r.as_handle().is_some_and(|rr| {
                let rh = Handle::from_raw(rr);
                this.realm.string_value(rh).is_none()
                    && (this.realm.object_keys(rh).is_some() || this.realm.is_array(rh))
            })
        };
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
                if !is_object(self, r) {
                    return Ok(r);
                }
            }
        }
        Ok(v)
    }

    /// Coerces `v` to a string, invoking `[Symbol.toPrimitive]("string")` or a
    /// callable `toString` when present (else the default form).
    fn coerce_to_string(&mut self, v: NanBox) -> Result<String, ExecError> {
        if let Some(raw) = v.as_handle() {
            let h = Handle::from_raw(raw);
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
                let strings: Vec<NanBox> = quasi
                    .quasis
                    .iter()
                    .map(|q| self.new_str(q.cooked.as_deref().unwrap_or("")))
                    .collect();
                let raw: Vec<NanBox> = quasi.quasis.iter().map(|q| self.new_str(&q.raw)).collect();
                let strings_h = self.realm.new_array(strings);
                // The strings object carries a `.raw` array (for `String.raw` and
                // tags reading `strings.raw`).
                let raw_arr = NanBox::handle(self.realm.new_array(raw).to_raw());
                self.realm.set_property(strings_h, "raw", raw_arr);
                let strings_arr = NanBox::handle(strings_h.to_raw());
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
            Expr::Class(class) => Ok(self.make_class(class)),
            Expr::Unary { op, argument, .. } => {
                // `delete obj.x` removes a property; `typeof undefinedVar` must
                // not throw — both inspect the operand rather than its value.
                match op {
                    UnaryOp::Delete => {
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
                                    } else if let (true, Ok(i)) =
                                        (self.realm.is_array(h), name.parse::<usize>())
                                    {
                                        // `delete arr[i]` clears the element (no
                                        // true holes; the slot becomes undefined).
                                        self.realm.set_element(h, i, NanBox::undefined());
                                    } else {
                                        self.realm.delete_property(h, &name);
                                    }
                                }
                            }
                        }
                        return Ok(NanBox::boolean(true));
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
                    let Some((pid, penv)) = self.pending_super.clone() else {
                        return Err(ExecError::Unsupported(
                            "super outside a derived constructor",
                        ));
                    };
                    if let Some(raw) = self.this_val.as_handle() {
                        self.run_constructor(pid, &penv, Handle::from_raw(raw), &args)?;
                    }
                    return Ok(NanBox::undefined());
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
                        return Ok(NanBox::undefined());
                    }
                    let args = self.eval_args(arguments)?;
                    if let PropertyKey::Ident(name) | PropertyKey::Str(name) = property
                        && let Some(result) = self.call_method(recv, name, &args)?
                    {
                        return Ok(result);
                    }
                    // Not a built-in method: read the member and call it.
                    let Some(raw) = recv.as_handle() else {
                        if *call_optional {
                            return Ok(NanBox::undefined());
                        }
                        return Err(ExecError::NotCallable);
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    // `f?.()` short-circuits when `f` is nullish.
                    if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null)
                    {
                        return Ok(NanBox::undefined());
                    }
                    // Method call: `this` is the receiver.
                    return self.call_with_this(f, recv, &args);
                }
                let f = self.eval(callee)?;
                if *call_optional && matches!(f.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Ok(NanBox::undefined());
                }
                let args = self.eval_args(arguments)?;
                self.call(f, &args)
            }
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
                        ObjectMember::Property { key, value, .. } => {
                            let k = self.eval_prop_key(key)?;
                            let v = self.eval(value)?;
                            self.realm.set_property(handle, &k, v);
                        }
                        // `{ ...src }` — copy own enumerable properties.
                        ObjectMember::Spread { value, .. } => {
                            let src = self.eval(value)?;
                            if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                                let mut keys = self.realm.object_keys(sh).unwrap_or_default();
                                // Accessor (getter) properties are enumerable too.
                                keys.extend(self.realm.object_accessor_keys(sh));
                                for key in keys {
                                    // `read_member` invokes a getter where present.
                                    let pv = self.read_member(sh, &key)?;
                                    self.realm.set_property(handle, &key, pv);
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
                let obj = self.eval(object)?;
                if matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    if *optional {
                        return Ok(NanBox::undefined());
                    }
                    // `null.x` / `undefined.x` throws a catchable TypeError.
                    let msg = self.new_str("cannot read property of null or undefined");
                    return Err(ExecError::Throw(self.make_error(N_TYPE_ERROR, Some(msg))));
                }
                let Some(raw) = obj.as_handle() else {
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
                let name = self.member_key(k);
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
                self.call(trap, &[NanBox::handle(target.to_raw()), key_box, new, recv])?;
                return Ok(());
            }
            return self.assign_member_value(target, key, new);
        }
        // A numeric index addresses array storage directly.
        if let Some(i) = key.as_number().and_then(as_index)
            && self.realm.is_array(handle)
        {
            self.realm.set_element(handle, i, new);
            return Ok(());
        }
        let name = self.member_key(key);
        // An accessor setter takes precedence.
        if let Some((_, setter)) = self.realm.accessor(handle, &name) {
            if !matches!(setter.unpack(), Unpacked::Undefined) {
                let this = NanBox::handle(handle.to_raw());
                self.call_with_this(setter, this, &[new])?;
            }
            return Ok(());
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
        // Well-known `Symbol.iterator` / `Symbol.asyncIterator` (lazily created).
        if self.realm.native_at(handle) == Some(N_SYMBOL)
            && matches!(
                name,
                "iterator" | "asyncIterator" | "hasInstance" | "toPrimitive"
            )
        {
            let key: &'static str = match name {
                "iterator" => "iterator",
                "asyncIterator" => "asyncIterator",
                "hasInstance" => "hasInstance",
                _ => "toPrimitive",
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
        // A function's `length` (params before a default/rest) and `name`.
        if matches!(name, "length" | "name")
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
                "MIN_VALUE" => return Ok(NanBox::number(f64::MIN_POSITIVE)),
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
        // RegExp introspection properties.
        if let Some((source, flags)) = self.realm.regexp_at(handle) {
            return Ok(match name {
                "source" => self.new_str(&source),
                "flags" => self.new_str(&flags),
                "global" => NanBox::boolean(flags.contains('g')),
                "ignoreCase" => NanBox::boolean(flags.contains('i')),
                "multiline" => NanBox::boolean(flags.contains('m')),
                "sticky" => NanBox::boolean(flags.contains('y')),
                _ => self.member_value(handle, name),
            });
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
        Ok(direct)
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
                    self.current.declare(name, new); // sloppy global
                }
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

    fn assign_member(
        &mut self,
        handle: crate::heap::Handle,
        property: &'a PropertyKey,
        new: NanBox,
    ) -> Result<(), ExecError> {
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
                self.call(trap, &[NanBox::handle(target.to_raw()), key_box, new, recv])?;
                return Ok(());
            }
            return self.assign_member(target, property, new);
        }
        // An accessor setter takes precedence for a named property.
        if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property
            && let Some((_, setter)) = self.realm.accessor(handle, s)
        {
            if !matches!(setter.unpack(), Unpacked::Undefined) {
                let this = NanBox::handle(handle.to_raw());
                self.call_with_this(setter, this, &[new])?;
            }
            return Ok(());
        }
        match property {
            PropertyKey::Number(n) if as_index(*n).is_some() && self.realm.is_array(handle) => {
                self.realm.set_element(handle, as_index(*n).unwrap(), new);
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                // A numeric index only addresses array storage; on an object a
                // numeric key is the equivalent string property.
                if let Some(i) = k.as_number().and_then(as_index)
                    && self.realm.is_array(handle)
                {
                    self.realm.set_element(handle, i, new);
                } else {
                    let name = self.member_key(k);
                    self.realm.set_property(handle, &name, new);
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
                } else {
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
            UnaryOp::BitNot => self.realm.bit_not(v),
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
            // True for a number or string primitive.
            let prim = |this: &Self, v: NanBox| {
                v.as_number().is_some()
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
                        if let Some(len) = self.realm.array_length(h) {
                            // An array index is present iff it is in bounds.
                            key == "length"
                                || key.parse::<usize>().is_ok_and(|i| i < len)
                                || self.realm.has_own(h, &key)
                        } else {
                            self.realm.has_own(h, &key)
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
                    && !m.is_static
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

    /// `obj instanceof Ctor`: true when `obj` was constructed from `Ctor`'s
    /// class or one of its subclasses (via the instance's class tag and the
    /// `extends` chain).
    fn instance_of(&mut self, obj: NanBox, ctor: NanBox) -> Result<bool, ExecError> {
        let (Some(oh), Some(ch)) = (
            obj.as_handle().map(Handle::from_raw),
            ctor.as_handle().map(Handle::from_raw),
        ) else {
            return Ok(false);
        };
        // Built-in constructors: check the cell kind directly.
        if let Some(id) = self.realm.native_at(ch) {
            // The `Error` family: match by the object's `name` against the
            // constructor (the base `Error` matches any error object).
            if (N_ERROR_BASE..N_ERROR_BASE + ERROR_NAMES.len() as u16).contains(&id) {
                let obj_name = self
                    .realm
                    .get_property(oh, "name")
                    .map(|v| self.realm.to_display_string(v))
                    .unwrap_or_default();
                if !ERROR_NAMES.contains(&obj_name.as_str()) {
                    return Ok(false);
                }
                let want = ERROR_NAMES[(id - N_ERROR_BASE) as usize];
                return Ok(want == "Error" || obj_name == want);
            }
            return Ok(match id {
                N_REGEXP => self.realm.regexp_at(oh).is_some(),
                N_MAP | N_SET | N_WEAKMAP | N_WEAKSET => self.realm.collection_is_set(oh).is_some(),
                N_DATE => self.realm.date_at(oh).is_some(),
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
        // Plain function constructors: match the instance's recorded constructor.
        if self.realm.function_at(ch).is_some() {
            return Ok(self
                .realm
                .get_property(oh, CTOR_KEY)
                .and_then(|t| t.as_handle())
                == ctor.as_handle());
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
        // Exponential notation with p-1 fractional digits.
        return alloc::format!("{:.*e}", p - 1, n);
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
fn int_to_radix(n: f64, radix: u32) -> String {
    let neg = n < 0.0;
    let mut v = n.abs() as u64;
    if v == 0 {
        return String::from("0");
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::new();
    while v > 0 {
        buf.push(DIGITS[(v % radix as u64) as usize]);
        v /= radix as u64;
    }
    if neg {
        buf.push(b'-');
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_default()
}

/// Parses the longest leading decimal-float prefix of `s` (à la `parseFloat`),
/// returning `NaN` if none.
fn parse_float_prefix(s: &str) -> f64 {
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
fn expand_replacement(templ: &str, text: &str, caps: &crate::regex::Captures) -> String {
    let group = |i: usize| {
        caps.groups
            .get(i)
            .and_then(|g| *g)
            .map(|(s, e)| &text[s..e])
    };
    let mut out = String::new();
    let mut chars = templ.chars().peekable();
    while let Some(c) = chars.next() {
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
            Some(d) if d.is_ascii_digit() => {
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
    fn math_constants() {
        assert_eq!(run("Math.PI > 3.14 && Math.PI < 3.15"), "true");
        assert_eq!(run("Math.E > 2.71 && Math.E < 2.72"), "true");
        assert_eq!(run("Math.SQRT2 * Math.SQRT2 > 1.999"), "true");
        assert_eq!(run("Math.floor(Math.LN2 * 1000)"), "693");
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
