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
    Argument, ArrayElement, Arrow, ArrowBody, AssignOp, BinaryOp, BindingTarget, Class,
    ClassMember, Expr, ForInit, Function, Ident, LogicalOp, MethodKind, ObjectMember, Param,
    Program, PropertyKey, Stmt, UnaryOp, VarDecl,
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
    /// A `break`.
    Break,
    /// A `continue`.
    Continue,
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
    /// The current `this` binding (method/constructor receiver).
    this_val: NanBox,
    /// The superclass to invoke for `super(...)` inside the running constructor.
    pending_super: Option<(u32, Scope)>,
    /// Captured `console.log` output (a line per call).
    output: String,
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

impl<'a> Interp<'a> {
    /// A fresh interpreter with a single (global) scope and a starter stdlib.
    #[must_use]
    pub fn new() -> Self {
        let mut interp = Self {
            realm: Realm::new(),
            current: Scope::root(),
            functions: Vec::new(),
            classes: Vec::new(),
            this_val: NanBox::undefined(),
            pending_super: None,
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
            ],
        );
        install_namespace(self, "console", &[("log", N_CONSOLE_LOG)]);
        install_namespace(self, "JSON", &[("stringify", N_JSON_STRINGIFY)]);
        install_namespace(
            self,
            "Object",
            &[
                ("keys", N_OBJECT_KEYS),
                ("values", N_OBJECT_VALUES),
                ("assign", N_OBJECT_ASSIGN),
                ("entries", N_OBJECT_ENTRIES),
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
                NanBox::number(parse_int(&s))
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
                let items: Vec<NanBox> = match arg(0).as_handle().map(Handle::from_raw) {
                    Some(h) if self.realm.is_array(h) => self
                        .realm
                        .array_elements(h)
                        .map(<[_]>::to_vec)
                        .unwrap_or_default(),
                    Some(h) => match self.realm.string_value(h) {
                        Some(s) => {
                            let chars: Vec<char> = s.chars().collect();
                            chars
                                .iter()
                                .map(|c| self.new_str(&String::from(*c)))
                                .collect()
                        }
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                };
                NanBox::handle(self.realm.new_array(items).to_raw())
            }
            N_ARRAY_OF => NanBox::handle(self.realm.new_array(args.to_vec()).to_raw()),
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
            _ => return Err(ExecError::NotCallable),
        })
    }

    /// Serializes a value to JSON (`None` when the value is `undefined` or a
    /// function — which `JSON.stringify` omits / drops).
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
                Flow::Return(v) => return Ok(v),
                Flow::Break | Flow::Continue => {}
            }
        }
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
                let value = self.make_function(&func.params, Body::Block(&func.body));
                self.current.declare(&id.name, value);
            }
        }
        Ok(())
    }

    /// Registers a function definition and allocates a closure capturing the
    /// current scope.
    fn make_function(&mut self, params: &'a [Param], body: Body<'a>) -> NanBox {
        let func_id = self.functions.len() as u32;
        self.functions.push(FnDef { params, body });
        let handle = self.realm.new_function(func_id, self.current.clone());
        NanBox::handle(handle.to_raw())
    }

    /// Calls `callee` with `args`.
    fn call(&mut self, callee: NanBox, args: &[NanBox]) -> Result<NanBox, ExecError> {
        self.call_with_this(callee, NanBox::undefined(), args)
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
        for (i, param) in def.params.iter().enumerate() {
            let BindingTarget::Ident(Ident { name, .. }) = &param.target else {
                return Err(ExecError::Unsupported("destructuring parameter"));
            };
            let value = args.get(i).copied().unwrap_or(NanBox::undefined());
            call_scope.declare(name, value);
        }
        let saved = core::mem::replace(&mut self.current, call_scope);
        let saved_this = core::mem::replace(&mut self.this_val, this_val);
        let result = self.run_body(def.body);
        self.current = saved;
        self.this_val = saved_this;
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

    /// Registers a class and allocates a class value capturing the current scope.
    fn make_class(&mut self, class: &'a Class) -> NanBox {
        let class_id = self.classes.len() as u32;
        self.classes.push(class);
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
                if let ClassMember::Method(m) = member
                    && !m.is_static
                    && m.kind == MethodKind::Method
                {
                    let key = static_key(&m.key)?;
                    let saved = core::mem::replace(&mut self.current, cenv.clone());
                    let f = self.make_function(&m.value.params, Body::Block(&m.value.body));
                    self.current = saved;
                    self.realm.set_property(instance, &key, f);
                }
            }
        }

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
                Argument::Spread(_) => return Err(ExecError::Unsupported("call spread")),
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
                "toString" => Some(self.new_str(&self.realm.to_display_string(recv))),
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
                        Flow::Normal(_) | Flow::Break | Flow::Continue => {}
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
                while self.eval(test)?.to_boolean() {
                    match self.exec(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        Flow::Normal(_) | Flow::Continue => {}
                    }
                }
                Ok(Flow::Normal(NanBox::undefined()))
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
            Stmt::Break { label: None, .. } => Ok(Flow::Break),
            Stmt::Continue { label: None, .. } => Ok(Flow::Continue),
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
            let BindingTarget::Ident(Ident { name, .. }) = &d.target else {
                return Err(ExecError::Unsupported("destructuring binding"));
            };
            let value = match &d.init {
                Some(e) => self.eval(e)?,
                None => NanBox::undefined(),
            };
            self.current.declare(name, value);
        }
        Ok(())
    }

    fn exec_for(
        &mut self,
        init: Option<&'a ForInit>,
        test: Option<&'a Expr>,
        update: Option<&'a Expr>,
        body: &'a Stmt,
    ) -> Result<Flow, ExecError> {
        let child = self.current.child();
        let saved = core::mem::replace(&mut self.current, child);
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
                match self.exec(body)? {
                    Flow::Break => break,
                    Flow::Return(v) => return Ok(Flow::Return(v)),
                    Flow::Normal(_) | Flow::Continue => {}
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
                name => self
                    .current
                    .get(name)
                    .ok_or_else(|| ExecError::NotDefined(String::from(name))),
            },
            Expr::This(_) => Ok(self.this_val),
            Expr::Function(func) => Ok(self.eval_fn_expr(func)),
            Expr::Arrow(arrow) => Ok(self.eval_arrow(arrow)),
            Expr::Class(class) => Ok(self.make_class(class)),
            Expr::Unary { op, argument, .. } => {
                let v = self.eval(argument)?;
                self.unary(*op, v)
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
                callee, arguments, ..
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
                        return Err(ExecError::NotCallable);
                    };
                    let f = self.member(Handle::from_raw(raw), property)?;
                    // Method call: `this` is the receiver.
                    return self.call_with_this(f, recv, &args);
                }
                let f = self.eval(callee)?;
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
                        ArrayElement::Spread(_) => {
                            return Err(ExecError::Unsupported("array spread"));
                        }
                    }
                }
                let h = self.realm.new_array(items);
                Ok(NanBox::handle(h.to_raw()))
            }
            Expr::Object { members, .. } => {
                let handle = self.realm.new_object();
                for m in members {
                    let ObjectMember::Property { key, value, .. } = m else {
                        return Err(ExecError::Unsupported("object member"));
                    };
                    let k = static_key(key)?;
                    let v = self.eval(value)?;
                    self.realm.set_property(handle, &k, v);
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
                if *optional && matches!(obj.unpack(), Unpacked::Undefined | Unpacked::Null) {
                    return Ok(NanBox::undefined());
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
        self.make_function(&func.params, Body::Block(&func.body))
    }

    fn eval_arrow(&mut self, arrow: &'a Arrow) -> NanBox {
        let body = match &arrow.body {
            ArrowBody::Expr(e) => Body::Expr(e),
            ArrowBody::Block(b) => Body::Block(b),
        };
        self.make_function(&arrow.params, body)
    }

    fn member(
        &mut self,
        handle: crate::heap::Handle,
        key: &'a PropertyKey,
    ) -> Result<NanBox, ExecError> {
        match key {
            PropertyKey::Number(n) if as_index(*n).is_some() => {
                Ok(self.realm.get_element(handle, as_index(*n).unwrap()))
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index) {
                    return Ok(self.realm.get_element(handle, i));
                }
                let name = self.realm.to_display_string(k);
                Ok(self.member_value(handle, &name))
            }
            PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(self.member_value(handle, s)),
            PropertyKey::Number(n) => Ok(self.member_value(handle, &alloc::format!("{n}"))),
            PropertyKey::Private(_) => Err(ExecError::Unsupported("private member")),
        }
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
        let rhs = self.eval(value)?;
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
        match property {
            PropertyKey::Number(n) if as_index(*n).is_some() => {
                self.realm.set_element(handle, as_index(*n).unwrap(), new);
            }
            PropertyKey::Computed(e) => {
                let k = self.eval(e)?;
                if let Some(i) = k.as_number().and_then(as_index) {
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
            BinaryOp::In | BinaryOp::Instanceof => {
                return Err(ExecError::Unsupported("in / instanceof"));
            }
        })
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

/// A minimal `parseInt`: skips leading whitespace, reads an optional sign and
/// the leading decimal digits, and returns `NaN` if there are none.
fn parse_int(s: &str) -> f64 {
    let t = s.trim_start();
    let mut chars = t.chars().peekable();
    let mut out = String::new();
    if matches!(chars.peek(), Some('+' | '-')) {
        out.push(chars.next().unwrap());
    }
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            out.push(*c);
            chars.next();
        } else {
            break;
        }
    }
    // Just a sign (or empty) → NaN.
    if out.is_empty() || out == "+" || out == "-" {
        return f64::NAN;
    }
    out.parse::<f64>().unwrap_or(f64::NAN)
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
