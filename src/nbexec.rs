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
    /// Per-class captured definition scope, parallel to `classes`.
    class_envs: Vec<Scope>,
    /// The current `this` binding (method/constructor receiver).
    this_val: NanBox,
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
// Bound natives (carry a target promise handle):
const N_RESOLVE: u16 = 100;
const N_REJECT: u16 = 101;

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
            class_envs: Vec::new(),
            this_val: NanBox::undefined(),
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
                ("trunc", N_MATH_TRUNC),
            ],
        );
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
            N_MATH_ABS => {
                let n = self.realm.to_number(arg(0));
                NanBox::number(if n < 0.0 { -n } else { n })
            }
            N_STRING => {
                let s = self.realm.to_display_string(arg(0));
                NanBox::handle(self.realm.new_string(&s).to_raw())
            }
            N_NUMBER => NanBox::number(self.realm.to_number(arg(0))),
            N_BOOLEAN => NanBox::boolean(arg(0).to_boolean()),
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
            N_JSON_STRINGIFY => match self.json_stringify(arg(0)) {
                Some(s) => NanBox::handle(self.realm.new_string(&s).to_raw()),
                None => NanBox::undefined(),
            },
            N_JSON_PARSE => {
                let text = self.realm.to_display_string(arg(0));
                let chars: Vec<char> = text.chars().collect();
                let mut pos = 0;
                let value = self.json_parse(&chars, &mut pos)?;
                skip_ws(&chars, &mut pos);
                if pos != chars.len() {
                    return Err(ExecError::Throw(self.new_str("Unexpected token in JSON")));
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
                            for k in self.realm.object_keys(sh).unwrap_or_default() {
                                let v = self
                                    .realm
                                    .get_property(sh, &k)
                                    .unwrap_or(NanBox::undefined());
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
                // optional map callback applied to each element.
                let items = self.iterate_values(arg(0)).unwrap_or_default();
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
                if let Some(pairs) = arg(0)
                    .as_handle()
                    .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                    .map(<[_]>::to_vec)
                {
                    for pair in pairs {
                        if let Some(kv) = pair
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            let k = self.realm.to_display_string(
                                kv.first().copied().unwrap_or(NanBox::undefined()),
                            );
                            let v = kv.get(1).copied().unwrap_or(NanBox::undefined());
                            self.realm.set_property(obj, &k, v);
                        }
                    }
                }
                NanBox::handle(obj.to_raw())
            }
            #[cfg(feature = "std")]
            N_MATH_FLOOR => NanBox::number(self.realm.to_number(arg(0)).floor()),
            #[cfg(feature = "std")]
            N_MATH_CEIL => NanBox::number(self.realm.to_number(arg(0)).ceil()),
            #[cfg(feature = "std")]
            N_MATH_ROUND => NanBox::number(self.realm.to_number(arg(0)).round()),
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

    fn json_stringify(&self, v: NanBox) -> Option<String> {
        match v.unpack() {
            Unpacked::Undefined => None,
            Unpacked::Null => Some(String::from("null")),
            Unpacked::Bool(b) => Some(String::from(if b { "true" } else { "false" })),
            Unpacked::Number(n) => Some(if n.is_finite() {
                alloc::format!("{n}")
            } else {
                String::from("null")
            }),
            Unpacked::Handle(raw) => {
                let h = Handle::from_raw(raw);
                if let Some(s) = self.realm.string_value(h) {
                    return Some(json_quote(&s));
                }
                if let Some(elems) = self.realm.array_elements(h).map(<[_]>::to_vec) {
                    let parts: Vec<String> = elems
                        .iter()
                        .map(|e| {
                            self.json_stringify(*e)
                                .unwrap_or_else(|| String::from("null"))
                        })
                        .collect();
                    return Some(alloc::format!("[{}]", parts.join(",")));
                }
                if let Some(keys) = self.realm.object_keys(h) {
                    let mut parts = Vec::new();
                    for k in keys {
                        let val = self
                            .realm
                            .get_property(h, &k)
                            .unwrap_or(NanBox::undefined());
                        if let Some(s) = self.json_stringify(val) {
                            parts.push(alloc::format!("{}:{}", json_quote(&k), s));
                        }
                    }
                    return Some(alloc::format!("{{{}}}", parts.join(",")));
                }
                None // a function
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
        self.hoist(&program.body)?;
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
    fn hoist(&mut self, stmts: &'a [Stmt]) -> Result<(), ExecError> {
        for stmt in stmts {
            if let Stmt::Function(func) = stmt
                && let Some(id) = &func.id
            {
                let value =
                    self.make_function(&func.params, Body::Block(&func.body), func.is_async);
                self.current.declare(&id.name, value);
            }
        }
        Ok(())
    }

    /// Registers a function definition and allocates a closure capturing the
    /// current scope.
    fn make_function(&mut self, params: &'a [Param], body: Body<'a>, is_async: bool) -> NanBox {
        self.make_method(params, body, is_async, None)
    }

    fn make_method(
        &mut self,
        params: &'a [Param],
        body: Body<'a>,
        is_async: bool,
        home_class: Option<u32>,
    ) -> NanBox {
        let func_id = self.functions.len() as u32;
        self.functions.push(FnDef {
            params,
            body,
            is_async,
            home_class,
        });
        let handle = self.realm.new_function(func_id, self.current.clone());
        NanBox::handle(handle.to_raw())
    }

    /// Calls `callee` with `args`.
    fn call(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        self.call_with_this(callee, NanBox::undefined(), args)
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

    fn is_callable(&self, handle: Handle) -> bool {
        self.realm.native_at(handle).is_some()
            || self.realm.function_at(handle).is_some()
            || self.realm.bound_native_at(handle).is_some()
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
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        let saved_home = core::mem::replace(&mut self.current_home, def.home_class);
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
            self.run_body(def.body)
        })();
        self.current = saved;
        self.this_val = saved_this;
        self.current_home = saved_home;
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
        // `new UserClass(...)`.
        if let Some((class_id, env)) = self.realm.class_at(handle) {
            return self.instantiate(class_id, &env, args);
        }
        let id = self
            .realm
            .native_at(handle)
            .ok_or(ExecError::Unsupported("new on this value"))?;
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
            let ms = match args.first() {
                Some(a) => self.realm.to_number(*a),
                None => now_ms(),
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
        let is_set = match id {
            N_SET => true,
            N_MAP => false,
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
        for member in &class.body {
            match member {
                ClassMember::Method(m) if m.is_static && m.kind == MethodKind::Method => {
                    if let Ok(key) = static_key(&m.key) {
                        let f =
                            self.make_function(&m.value.params, Body::Block(&m.value.body), false);
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
                _ => {}
            }
        }
        self.class_statics.push(statics);
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
                let key = static_key(&m.key)?;
                let saved = core::mem::replace(&mut self.current, cenv.clone());
                let f = self.make_method(
                    &m.value.params,
                    Body::Block(&m.value.body),
                    false,
                    Some(*cid),
                );
                self.current = saved;
                match m.kind {
                    MethodKind::Method => {
                        self.realm.set_property(instance, &key, f);
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
                    let scope = self.current.child();
                    for (i, param) in ctor.value.params.iter().enumerate() {
                        if let BindingTarget::Ident(Ident { name, .. }) = &param.target {
                            let v = args.get(i).copied().unwrap_or(NanBox::undefined());
                            scope.declare(name, v);
                        }
                    }
                    let saved = core::mem::replace(&mut self.current, scope);
                    let r = (|| {
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
                // No own constructor but a base: implicit `super(args)`.
                (None, Some((pid, penv))) => {
                    self.run_constructor(*pid, &penv.clone(), instance, args)?;
                }
                (None, None) => {}
            }
            // Field initializers (after the constructor body / super).
            for member in &class.body {
                if let ClassMember::Field(field) = member
                    && !field.is_static
                {
                    let key = static_key(&field.key)?;
                    let v = match &field.value {
                        Some(e) => self.eval(e)?,
                        None => NanBox::undefined(),
                    };
                    self.realm.set_property(instance, &key, v);
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
                    let digits = self.realm.to_number(arg(0)) as usize;
                    Some(self.new_str(&alloc::format!("{n:.digits$}")))
                }
                _ => None,
            });
        }

        let Some(raw) = recv.as_handle() else {
            return Ok(None);
        };
        let handle = Handle::from_raw(raw);

        // --- `Date.now()` static ---
        if self.realm.native_at(handle) == Some(N_DATE) && method == "now" {
            return Ok(Some(NanBox::number(now_ms())));
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
            _ => {}
        }
        // --- Date instance methods ---
        if let Some(ms) = self.realm.date_at(handle) {
            let t = ms as i64;
            let day = t.div_euclid(86_400_000);
            let tod = t.rem_euclid(86_400_000);
            let (y, mo, d) = crate::realm::civil_from_days(day);
            return Ok(Some(match method {
                "getTime" | "valueOf" => NanBox::number(ms),
                "getFullYear" => NanBox::number(y as f64),
                "getMonth" => NanBox::number((mo - 1) as f64), // 0-based
                "getDate" => NanBox::number(d as f64),
                "getDay" => NanBox::number((day.rem_euclid(7) + 4).rem_euclid(7) as f64),
                "getHours" => NanBox::number((tod / 3_600_000) as f64),
                "getMinutes" => NanBox::number((tod / 60_000 % 60) as f64),
                "getSeconds" => NanBox::number((tod / 1000 % 60) as f64),
                "getMilliseconds" => NanBox::number((tod % 1000) as f64),
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
                    while let Some((st, en)) = re.find_from(&s, at) {
                        if en == at && st == at {
                            at += 1;
                            if at > s.len() {
                                break;
                            }
                            continue;
                        }
                        parts.push(self.new_str(&s[at..st]));
                        at = en;
                    }
                    parts.push(self.new_str(&s[at..]));
                    return Ok(Some(NanBox::handle(self.realm.new_array(parts).to_raw())));
                }
                // replace / replaceAll: substitute `$1`..`$9` / `$&` from groups.
                _ => {
                    let templ = self.realm.to_display_string(arg(1));
                    let mut out = String::new();
                    let mut at = 0;
                    while let Some(caps) = re.captures_from(&s, at) {
                        let (st, en) = caps.groups[0].unwrap_or((at, at));
                        out.push_str(&s[at..st]);
                        out.push_str(&expand_replacement(&templ, &s, &caps));
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
                    Some(self.new_str(&s.chars().nth(i).map(String::from).unwrap_or_default()))
                }
                "includes" => Some(NanBox::boolean(
                    s.contains(&self.realm.to_display_string(arg(0))),
                )),
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
                "startsWith" => Some(NanBox::boolean(
                    s.starts_with(&self.realm.to_display_string(arg(0))),
                )),
                "endsWith" => Some(NanBox::boolean(
                    s.ends_with(&self.realm.to_display_string(arg(0))),
                )),
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
                    let parts: Vec<NanBox> = if sep.is_empty() {
                        let chars: Vec<char> = s.chars().collect();
                        chars
                            .iter()
                            .map(|c| self.new_str(&String::from(*c)))
                            .collect()
                    } else {
                        s.split(&sep).map(|p| self.new_str(p)).collect()
                    };
                    Some(NanBox::handle(self.realm.new_array(parts).to_raw()))
                }
                "replace" => {
                    let from = self.realm.to_display_string(arg(0));
                    let to = self.realm.to_display_string(arg(1));
                    Some(self.new_str(&s.replacen(&from, &to, 1)))
                }
                "replaceAll" => {
                    let from = self.realm.to_display_string(arg(0));
                    let to = self.realm.to_display_string(arg(1));
                    Some(self.new_str(&s.replace(&from, &to)))
                }
                "at" => {
                    let i = self.realm.to_number(arg(0));
                    let chars: Vec<char> = s.chars().collect();
                    let idx = if i < 0.0 { chars.len() as f64 + i } else { i };
                    Some(match as_index(idx).and_then(|u| chars.get(u)) {
                        Some(c) => self.new_str(&String::from(*c)),
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
                "charCodeAt" | "codePointAt" => {
                    let i = self.realm.to_number(arg(0)) as usize;
                    Some(match s.chars().nth(i) {
                        Some(c) => NanBox::number(u32::from(c) as f64),
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
                    for a in args {
                        len = self.realm.array_push(handle, *a).unwrap_or(len);
                    }
                    return Ok(Some(NanBox::number(len as f64)));
                }
                "pop" => return Ok(Some(self.realm.array_pop(handle))),
                "join" => {
                    let sep = if matches!(arg(0).unpack(), Unpacked::Undefined) {
                        String::from(",")
                    } else {
                        self.realm.to_display_string(arg(0))
                    };
                    let parts: Vec<String> = elems
                        .iter()
                        .map(|e| self.realm.to_display_string(*e))
                        .collect();
                    return Ok(Some(self.new_str(&parts.join(&sep))));
                }
                "includes" => {
                    let target = arg(0);
                    let found = elems.iter().any(|e| self.realm.strict_equals(*e, target));
                    return Ok(Some(NanBox::boolean(found)));
                }
                "indexOf" => {
                    let target = arg(0);
                    let idx = elems
                        .iter()
                        .position(|e| self.realm.strict_equals(*e, target))
                        .map_or(-1.0, |i| i as f64);
                    return Ok(Some(NanBox::number(idx)));
                }
                "map" => {
                    let f = arg(0);
                    let mut out = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        out.push(self.call(f, &[*e, NanBox::number(i as f64)])?);
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                "filter" => {
                    let f = arg(0);
                    let mut out = Vec::new();
                    for (i, e) in elems.iter().enumerate() {
                        if self.call(f, &[*e, NanBox::number(i as f64)])?.to_boolean() {
                            out.push(*e);
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                "forEach" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        self.call(f, &[*e, NanBox::number(i as f64)])?;
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
                    for (i, e) in elems.iter().enumerate().skip(start) {
                        acc = self.call(f, &[acc, *e, NanBox::number(i as f64)])?;
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
                    let mut out = elems.clone();
                    out.reverse();
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
                }
                // One level of flattening (`[1, [2, 3]].flat()`).
                "flat" => {
                    let mut out = Vec::new();
                    for e in &elems {
                        match e
                            .as_handle()
                            .and_then(|raw| self.realm.array_elements(Handle::from_raw(raw)))
                            .map(<[_]>::to_vec)
                        {
                            Some(inner) => out.extend(inner),
                            None => out.push(*e),
                        }
                    }
                    let h = self.realm.new_array(out);
                    return Ok(Some(NanBox::handle(h.to_raw())));
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
                    let found = elems
                        .iter()
                        .rposition(|e| self.realm.strict_equals(*e, target));
                    return Ok(Some(NanBox::number(found.map_or(-1.0, |i| i as f64))));
                }
                // Iterators, materialized as arrays (spread / for-of consume them).
                "keys" => {
                    let ks = (0..elems.len()).map(|i| NanBox::number(i as f64)).collect();
                    return Ok(Some(NanBox::handle(self.realm.new_array(ks).to_raw())));
                }
                "values" => {
                    let v = elems.clone();
                    return Ok(Some(NanBox::handle(self.realm.new_array(v).to_raw())));
                }
                "entries" => {
                    let mut out = Vec::with_capacity(elems.len());
                    for (i, e) in elems.iter().enumerate() {
                        let pair = self
                            .realm
                            .new_array(alloc::vec![NanBox::number(i as f64), *e]);
                        out.push(NanBox::handle(pair.to_raw()));
                    }
                    return Ok(Some(NanBox::handle(self.realm.new_array(out).to_raw())));
                }
                "find" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call(f, &[*e, NanBox::number(i as f64)])?.to_boolean() {
                            return Ok(Some(*e));
                        }
                    }
                    return Ok(Some(NanBox::undefined()));
                }
                "findIndex" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call(f, &[*e, NanBox::number(i as f64)])?.to_boolean() {
                            return Ok(Some(NanBox::number(i as f64)));
                        }
                    }
                    return Ok(Some(NanBox::number(-1.0)));
                }
                "some" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if self.call(f, &[*e, NanBox::number(i as f64)])?.to_boolean() {
                            return Ok(Some(NanBox::boolean(true)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(false)));
                }
                "every" => {
                    let f = arg(0);
                    for (i, e) in elems.iter().enumerate() {
                        if !self.call(f, &[*e, NanBox::number(i as f64)])?.to_boolean() {
                            return Ok(Some(NanBox::boolean(false)));
                        }
                    }
                    return Ok(Some(NanBox::boolean(true)));
                }
                "sort" => {
                    let sorted = self.sort_array(elems, arg(0))?;
                    let h = self.realm.new_array(sorted);
                    return Ok(Some(NanBox::handle(h.to_raw())));
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
                self.hoist(stmts)?;
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
                if self.eval(test)?.to_boolean() {
                    self.exec(consequent)
                } else if let Some(alt) = alternate {
                    self.exec(alt)
                } else {
                    Ok(Flow::Normal(NanBox::undefined()))
                }
            }
            Stmt::While { test, body, .. } => {
                let label = self.pending_label.take();
                while self.eval(test)?.to_boolean() {
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
                    if !self.eval(test)?.to_boolean() {
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
        for d in &decl.declarations {
            let value = match &d.init {
                Some(e) => self.eval(e)?,
                None => NanBox::undefined(),
            };
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
                    let key = static_key(&prop.key)?;
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
                            let k = static_key(key)?;
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
                    Some(t) => self.eval(t)?.to_boolean(),
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

    fn eval(&mut self, expr: &'a Expr) -> Result<NanBox, ExecError> {
        match expr {
            Expr::Null(_) => Ok(NanBox::null()),
            Expr::Bool { value, .. } => Ok(NanBox::boolean(*value)),
            Expr::Number { value, .. } => Ok(NanBox::number(*value)),
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
                        out.push_str(&self.realm.to_display_string(v));
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
                let tagf = self.eval(tag)?;
                let strings: Vec<NanBox> = quasi
                    .quasis
                    .iter()
                    .map(|q| self.new_str(q.cooked.as_deref().unwrap_or("")))
                    .collect();
                let strings_arr = NanBox::handle(self.realm.new_array(strings).to_raw());
                let mut args = alloc::vec![strings_arr];
                for e in &quasi.expressions {
                    args.push(self.eval(e)?);
                }
                self.call(tagf, &args)
            }
            Expr::This(_) => Ok(self.this_val),
            Expr::Await { argument, .. } => {
                let v = self.eval(argument)?;
                self.await_value(v)
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
                                if let PropertyKey::Ident(s) | PropertyKey::Str(s) = property {
                                    self.realm.delete_property(h, s);
                                } else if let PropertyKey::Computed(e) = property {
                                    let k = self.eval(e)?;
                                    let name = self.realm.to_display_string(k);
                                    self.realm.delete_property(h, &name);
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
                    LogicalOp::And => l.to_boolean(),
                    LogicalOp::Or => !l.to_boolean(),
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
                if self.eval(test)?.to_boolean() {
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
                            let k = static_key(key)?;
                            let v = self.eval(value)?;
                            self.realm.set_property(handle, &k, v);
                        }
                        // `{ ...src }` — copy own enumerable properties.
                        ObjectMember::Spread { value, .. } => {
                            let src = self.eval(value)?;
                            if let Some(sh) = src.as_handle().map(Handle::from_raw) {
                                for key in self.realm.object_keys(sh).unwrap_or_default() {
                                    let pv = self
                                        .realm
                                        .get_property(sh, &key)
                                        .unwrap_or(NanBox::undefined());
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
                            let k = static_key(key)?;
                            let f =
                                self.make_function(&value.params, Body::Block(&value.body), false);
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
        self.make_function(&func.params, Body::Block(&func.body), func.is_async)
    }

    fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        self.make_function(&arrow.params, body, arrow.is_async)
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
                let name = self.realm.to_display_string(k);
                self.read_member(handle, &name)
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => self.read_member(handle, s),
            PropertyKey::Number(n) => self.read_member(handle, &alloc::format!("{n}")),
            PropertyKey::Private(_) => Err(ExecError::Unsupported("private member")),
        }
    }

    /// Reads a named member, honoring class statics and accessor getters before
    /// ordinary property/length access.
    fn read_member(
        &mut self,
        handle: crate::heap::Handle,
        name: &str,
    ) -> Result<NanBox, ExecError> {
        if let Some((cid, _)) = self.realm.class_at(handle)
            && let Some(v) = self.class_statics[cid as usize].get(name)
        {
            return Ok(*v);
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
        Ok(self.member_value(handle, name))
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
                return NanBox::number(s.chars().count() as f64);
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
                AssignOp::AndAssign => current.to_boolean(),
                AssignOp::OrAssign => !current.to_boolean(),
                _ => matches!(current.unpack(), Unpacked::Undefined | Unpacked::Null),
            };
            if !assign {
                return Ok(current);
            }
            let rhs = self.eval(value)?;
            self.assign_to(target, rhs)?;
            return Ok(rhs);
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
                    let name = self.realm.to_display_string(k);
                    self.realm.set_property(handle, &name, new);
                }
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => {
                self.realm.set_property(handle, s, new);
            }
            PropertyKey::Number(n) => {
                self.realm.set_property(handle, &alloc::format!("{n}"), new);
            }
            PropertyKey::Private(_) => return Err(ExecError::Unsupported("private assign")),
        }
        Ok(())
    }

    fn unary(&mut self, op: UnaryOp, v: NanBox) -> Result<NanBox, ExecError> {
        Ok(match op {
            UnaryOp::Plus => NanBox::number(self.realm.to_number(v)),
            UnaryOp::Minus => self.realm.neg(v),
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

    fn binary(&mut self, op: BinaryOp, a: NanBox, b: NanBox) -> Result<NanBox, ExecError> {
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
                let key = self.realm.to_display_string(a);
                let present = b
                    .as_handle()
                    .map(Handle::from_raw)
                    .is_some_and(|h| self.realm.has_own(h, &key) || self.realm.is_array(h));
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
                N_MAP | N_SET => self.realm.collection_is_set(oh).is_some(),
                N_DATE => self.realm.date_at(oh).is_some(),
                _ => false,
            });
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
        PropertyKey::Computed(_) => Err(ExecError::Unsupported("computed key")),
        PropertyKey::Private(_) => Err(ExecError::Unsupported("private key")),
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
