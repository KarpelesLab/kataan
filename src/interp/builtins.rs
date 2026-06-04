//! A first slice of the ECMAScript standard library for the tree-walking
//! interpreter: the value-returning globals (`Math`, `JSON`, `Object`,
//! `parseInt`, …) installed at startup, and the prototype methods on arrays
//! and strings dispatched at the call site (so the higher-order ones can call
//! back into the evaluator).
//!
//! These are intentionally pragmatic implementations to exercise real
//! programs; the spec-complete builtins land with the bytecode VM and the real
//! object model.

use super::value::{NativeFn, Obj, Value};
use super::{Completion, Interp};
use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Builds a native-function value from a name and a Rust callback.
fn native<'a>(
    name: &'static str,
    f: impl Fn(&[Value<'a>]) -> Completion<'a, Value<'a>> + 'a,
) -> Value<'a> {
    Value::Native(Rc::new(NativeFn {
        name,
        call: Box::new(f),
    }))
}

/// `args[i]`, or `undefined`.
fn arg<'a>(args: &[Value<'a>], i: usize) -> Value<'a> {
    args.get(i).cloned().unwrap_or(Value::Undefined)
}

/// Builds a callable constructor *object* — invokable like a function (so
/// `String(x)` / `new Map()` work) while also able to hold static members as
/// ordinary properties (`Number.isInteger`, `String.fromCharCode`, …).
fn constructor_object<'a>(
    name: &'static str,
    f: impl Fn(&[Value<'a>]) -> Completion<'a, Value<'a>> + 'a,
) -> Rc<Obj<'a>> {
    let obj = Obj::object();
    obj.set_callable(native(name, f));
    obj
}

impl<'a> Interp<'a> {
    /// Installs the standard-library globals into the global scope.
    pub(super) fn install_stdlib(&self) {
        self.install_math();
        self.install_json();
        self.install_object_ctor();
        self.install_array_ctor();
        self.install_number_globals();
        self.install_errors();
        self.install_collections();
    }

    fn install_collections(&self) {
        // `new Map([[k, v], …])`
        self.define_global(
            "Map",
            native("Map", |a| {
                let map = Obj::collection(false);
                if let Value::Object(init) = arg(a, 0)
                    && let Some(elems) = init.elements()
                {
                    for entry in elems.borrow().iter() {
                        if let Value::Object(pair) = entry {
                            collection_set(&map, pair.get("0"), pair.get("1"));
                        }
                    }
                }
                Ok(Value::Object(map))
            }),
        );
        // `new Set([v, …])`
        self.define_global(
            "Set",
            native("Set", |a| {
                let set = Obj::collection(true);
                if let Value::Object(init) = arg(a, 0)
                    && let Some(elems) = init.elements()
                {
                    for v in elems.borrow().iter() {
                        collection_set(&set, v.clone(), v.clone());
                    }
                }
                Ok(Value::Object(set))
            }),
        );
    }

    fn install_errors(&self) {
        for name in [
            "Error",
            "TypeError",
            "RangeError",
            "SyntaxError",
            "ReferenceError",
            "EvalError",
            "URIError",
        ] {
            self.define_global(
                name,
                native(name, move |a| {
                    let obj = Obj::object();
                    obj.set("name", Value::str(name));
                    let msg = if a.is_empty() {
                        String::new()
                    } else {
                        arg(a, 0).to_js_string()
                    };
                    obj.set("message", Value::str(msg));
                    Ok(Value::Object(obj))
                }),
            );
        }
    }

    fn install_math(&self) {
        let math = Obj::object();
        math.set("PI", Value::Number(core::f64::consts::PI));
        math.set("E", Value::Number(core::f64::consts::E));
        macro_rules! unary {
            ($name:literal, $f:expr) => {
                math.set(
                    $name,
                    native($name, move |a| Ok(Value::Number($f(arg(a, 0).to_number())))),
                );
            };
        }
        unary!("abs", f64::abs);
        unary!("floor", f64::floor);
        unary!("ceil", f64::ceil);
        unary!("round", |n: f64| (n + 0.5).floor()); // JS rounds .5 toward +∞
        unary!("trunc", f64::trunc);
        unary!("sqrt", f64::sqrt);
        unary!("cbrt", f64::cbrt);
        unary!("sign", f64::signum);
        unary!("exp", f64::exp);
        unary!("log", f64::ln);
        unary!("log2", f64::log2);
        unary!("log10", f64::log10);
        unary!("sin", f64::sin);
        unary!("cos", f64::cos);
        unary!("tan", f64::tan);
        math.set(
            "pow",
            native("pow", |a| {
                Ok(Value::Number(
                    arg(a, 0).to_number().powf(arg(a, 1).to_number()),
                ))
            }),
        );
        math.set(
            "max",
            native("max", |a| {
                Ok(Value::Number(
                    a.iter()
                        .map(Value::to_number)
                        .fold(f64::NEG_INFINITY, f64::max),
                ))
            }),
        );
        math.set(
            "min",
            native("min", |a| {
                Ok(Value::Number(
                    a.iter().map(Value::to_number).fold(f64::INFINITY, f64::min),
                ))
            }),
        );
        self.define_global("Math", Value::Object(math));
    }

    fn install_json(&self) {
        let json = Obj::object();
        json.set(
            "stringify",
            native("stringify", |a| {
                Ok(json_stringify(&arg(a, 0)).map_or(Value::Undefined, Value::str))
            }),
        );
        json.set(
            "parse",
            native("parse", |a| {
                let s = arg(a, 0).to_js_string();
                json_parse(&s)
            }),
        );
        self.define_global("JSON", Value::Object(json));
    }

    fn install_object_ctor(&self) {
        let object = Obj::object();
        object.set(
            "keys",
            native("keys", |a| Ok(object_entries(&arg(a, 0), EntryKind::Key))),
        );
        object.set(
            "values",
            native("values", |a| {
                Ok(object_entries(&arg(a, 0), EntryKind::Value))
            }),
        );
        object.set(
            "entries",
            native("entries", |a| {
                Ok(object_entries(&arg(a, 0), EntryKind::Pair))
            }),
        );
        object.set(
            "fromEntries",
            native("fromEntries", |a| {
                let obj = Obj::object();
                let mut entries = Vec::new();
                iterate_into(&arg(a, 0), &mut entries);
                for entry in entries {
                    if let Value::Object(pair) = entry {
                        obj.set(&pair.get("0").to_js_string(), pair.get("1"));
                    }
                }
                Ok(Value::Object(obj))
            }),
        );
        object.set(
            "create",
            native("create", |a| match arg(a, 0) {
                Value::Object(proto) => Ok(Value::Object(Obj::with_proto(proto))),
                _ => Ok(Value::Object(Obj::object())),
            }),
        );
        // `freeze`/`isFrozen` are accepted but not yet enforced.
        object.set("freeze", native("freeze", |a| Ok(arg(a, 0))));
        object.set(
            "assign",
            native("assign", |a| {
                if let Value::Object(target) = arg(a, 0) {
                    for src in &a[1.min(a.len())..] {
                        if let Value::Object(s) = src {
                            for k in s.own_keys() {
                                target.set(&k, s.get(&k));
                            }
                        }
                    }
                    return Ok(Value::Object(target));
                }
                Ok(arg(a, 0))
            }),
        );
        self.define_global("Object", Value::Object(object));
    }

    fn install_array_ctor(&self) {
        let array = Obj::object();
        array.set(
            "isArray",
            native("isArray", |a| {
                Ok(Value::Bool(
                    matches!(arg(a, 0), Value::Object(o) if o.is_array()),
                ))
            }),
        );
        // `Array.from(arrayLike)` — arrays, strings, Sets/Maps, and
        // array-likes (objects with a `length`). The mapping-function overload
        // is deferred (it needs the evaluator).
        array.set(
            "from",
            native("from", |a| {
                let mut out = Vec::new();
                iterate_into(&arg(a, 0), &mut out);
                Ok(Value::Object(Obj::array(out)))
            }),
        );
        // `Array.of(...args)`.
        array.set(
            "of",
            native("of", |a| Ok(Value::Object(Obj::array(a.to_vec())))),
        );
        self.define_global("Array", Value::Object(array));
    }

    fn install_number_globals(&self) {
        self.define_global(
            "parseInt",
            native("parseInt", |a| {
                let s = arg(a, 0).to_js_string();
                let radix = arg(a, 1).to_number();
                Ok(Value::Number(parse_int(s.trim(), radix)))
            }),
        );
        self.define_global(
            "parseFloat",
            native("parseFloat", |a| {
                Ok(Value::Number(parse_float(arg(a, 0).to_js_string().trim())))
            }),
        );
        self.define_global(
            "isNaN",
            native("isNaN", |a| Ok(Value::Bool(arg(a, 0).to_number().is_nan()))),
        );
        self.define_global(
            "isFinite",
            native("isFinite", |a| {
                Ok(Value::Bool(arg(a, 0).to_number().is_finite()))
            }),
        );
        // `Number(x)` plus its statics.
        let number = constructor_object("Number", |a| Ok(Value::Number(arg(a, 0).to_number())));
        number.set(
            "isInteger",
            native("isInteger", |a| {
                Ok(Value::Bool(
                    matches!(arg(a, 0), Value::Number(n) if n.is_finite() && n.fract() == 0.0),
                ))
            }),
        );
        number.set(
            "isFinite",
            native("isFinite", |a| {
                Ok(Value::Bool(
                    matches!(arg(a, 0), Value::Number(n) if n.is_finite()),
                ))
            }),
        );
        number.set(
            "isNaN",
            native("isNaN", |a| {
                Ok(Value::Bool(
                    matches!(arg(a, 0), Value::Number(n) if n.is_nan()),
                ))
            }),
        );
        number.set(
            "isSafeInteger",
            native("isSafeInteger", |a| {
                Ok(Value::Bool(matches!(arg(a, 0), Value::Number(n)
                if n.is_finite() && n.fract() == 0.0 && n.abs() <= 9_007_199_254_740_991.0)))
            }),
        );
        number.set(
            "parseInt",
            native("parseInt", |a| {
                Ok(Value::Number(parse_int(
                    arg(a, 0).to_js_string().trim(),
                    arg(a, 1).to_number(),
                )))
            }),
        );
        number.set(
            "parseFloat",
            native("parseFloat", |a| {
                Ok(Value::Number(parse_float(arg(a, 0).to_js_string().trim())))
            }),
        );
        number.set("MAX_SAFE_INTEGER", Value::Number(9_007_199_254_740_991.0));
        number.set("MIN_SAFE_INTEGER", Value::Number(-9_007_199_254_740_991.0));
        number.set("MAX_VALUE", Value::Number(f64::MAX));
        number.set("MIN_VALUE", Value::Number(f64::MIN_POSITIVE));
        number.set("EPSILON", Value::Number(f64::EPSILON));
        number.set("POSITIVE_INFINITY", Value::Number(f64::INFINITY));
        number.set("NEGATIVE_INFINITY", Value::Number(f64::NEG_INFINITY));
        number.set("NaN", Value::Number(f64::NAN));
        self.define_global("Number", Value::Object(number));

        // `Boolean(x)`.
        self.define_global(
            "Boolean",
            Value::Object(constructor_object("Boolean", |a| {
                Ok(Value::Bool(arg(a, 0).to_boolean()))
            })),
        );

        // `String(x)` plus its statics.
        let string = constructor_object("String", |a| {
            Ok(Value::str(if a.is_empty() {
                String::new()
            } else {
                arg(a, 0).to_js_string()
            }))
        });
        string.set(
            "fromCharCode",
            native("fromCharCode", |a| {
                let s: String = a
                    .iter()
                    .filter_map(|v| char::from_u32(v.to_number() as u32))
                    .collect();
                Ok(Value::str(s))
            }),
        );
        string.set(
            "fromCodePoint",
            native("fromCodePoint", |a| {
                let s: String = a
                    .iter()
                    .filter_map(|v| char::from_u32(v.to_number() as u32))
                    .collect();
                Ok(Value::str(s))
            }),
        );
        // `String.raw(strings, ...subs)` — joins the `raw` segments with the
        // substitutions interleaved.
        string.set(
            "raw",
            native("raw", |a| {
                let mut out = String::new();
                if let Value::Object(strings) = arg(a, 0)
                    && let Value::Object(raw) = strings.get("raw")
                    && let Some(parts) = raw.elements()
                {
                    for (i, p) in parts.borrow().iter().enumerate() {
                        out.push_str(&p.to_js_string());
                        if let Some(sub) = a.get(i + 1) {
                            out.push_str(&sub.to_js_string());
                        }
                    }
                }
                Ok(Value::str(out))
            }),
        );
        self.define_global("String", Value::Object(string));
    }

    /// Dispatches a method call on an array or string receiver. Returns
    /// `Ok(Some(result))` if it handled the method, `Ok(None)` otherwise.
    pub(super) fn call_builtin_method(
        &mut self,
        recv: &Value<'a>,
        name: &str,
        args: &[Value<'a>],
    ) -> Completion<'a, Option<Value<'a>>> {
        match recv {
            Value::Object(o) if o.is_array() => self.array_method(o, name, args),
            Value::Object(o) if o.as_collection().is_some() => {
                self.collection_method(o, name, args)
            }
            Value::Str(s) => Ok(string_method(s, name, args)),
            Value::Number(n) => Ok(number_method(*n, name, args)),
            _ => Ok(None),
        }
    }

    /// `Map`/`Set` prototype methods.
    fn collection_method(
        &mut self,
        obj: &Rc<Obj<'a>>,
        name: &str,
        args: &[Value<'a>],
    ) -> Completion<'a, Option<Value<'a>>> {
        use super::value::same_value_zero;
        let cell = obj.as_collection().expect("collection");
        let result = match name {
            // Map.set(k, v) / Set.add(v)
            "set" => {
                collection_set(obj, arg(args, 0), arg(args, 1));
                Value::Object(Rc::clone(obj))
            }
            "add" => {
                collection_set(obj, arg(args, 0), arg(args, 0));
                Value::Object(Rc::clone(obj))
            }
            "get" => {
                let key = arg(args, 0);
                cell.borrow()
                    .entries
                    .iter()
                    .find(|(k, _)| same_value_zero(k, &key))
                    .map_or(Value::Undefined, |(_, v)| v.clone())
            }
            "has" => {
                let key = arg(args, 0);
                Value::Bool(
                    cell.borrow()
                        .entries
                        .iter()
                        .any(|(k, _)| same_value_zero(k, &key)),
                )
            }
            "delete" => {
                let key = arg(args, 0);
                let mut c = cell.borrow_mut();
                let before = c.entries.len();
                c.entries.retain(|(k, _)| !same_value_zero(k, &key));
                Value::Bool(c.entries.len() != before)
            }
            "clear" => {
                cell.borrow_mut().entries.clear();
                Value::Undefined
            }
            "keys" => {
                let ks: Vec<Value<'a>> = cell
                    .borrow()
                    .entries
                    .iter()
                    .map(|(k, _)| k.clone())
                    .collect();
                Value::Object(Obj::array(ks))
            }
            "values" => {
                let vs: Vec<Value<'a>> = cell
                    .borrow()
                    .entries
                    .iter()
                    .map(|(_, v)| v.clone())
                    .collect();
                Value::Object(Obj::array(vs))
            }
            "forEach" => {
                let callback = arg(args, 0);
                let is_set = cell.borrow().is_set;
                let snapshot: Vec<(Value<'a>, Value<'a>)> = cell.borrow().entries.clone();
                for (k, v) in snapshot {
                    // Map: (value, key, map); Set: (value, value, set).
                    let call_args = if is_set {
                        alloc::vec![k.clone(), k, Value::Object(Rc::clone(obj))]
                    } else {
                        alloc::vec![v, k, Value::Object(Rc::clone(obj))]
                    };
                    self.call_with_this(callback.clone(), Value::Undefined, call_args)?;
                }
                Value::Undefined
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }

    fn array_method(
        &mut self,
        arr: &Rc<Obj<'a>>,
        name: &str,
        args: &[Value<'a>],
    ) -> Completion<'a, Option<Value<'a>>> {
        let elements = arr.elements().expect("array");
        let result = match name {
            "push" => {
                let mut v = elements.borrow_mut();
                v.extend_from_slice(args);
                Value::Number(v.len() as f64)
            }
            "pop" => elements.borrow_mut().pop().unwrap_or(Value::Undefined),
            "shift" => {
                let mut v = elements.borrow_mut();
                if v.is_empty() {
                    Value::Undefined
                } else {
                    v.remove(0)
                }
            }
            "join" => {
                let sep = if args.is_empty() {
                    ",".into()
                } else {
                    arg(args, 0).to_js_string()
                };
                let parts: Vec<String> = elements
                    .borrow()
                    .iter()
                    .map(|v| match v {
                        Value::Undefined | Value::Null => String::new(),
                        other => other.to_js_string(),
                    })
                    .collect();
                Value::str(parts.join(&sep))
            }
            "includes" => {
                let needle = arg(args, 0);
                let found = elements
                    .borrow()
                    .iter()
                    .any(|v| super::value::strict_equals(v, &needle));
                Value::Bool(found)
            }
            "indexOf" => {
                let needle = arg(args, 0);
                let idx = elements
                    .borrow()
                    .iter()
                    .position(|v| super::value::strict_equals(v, &needle));
                Value::Number(idx.map_or(-1.0, |i| i as f64))
            }
            "slice" => {
                let v = elements.borrow();
                let (start, end) = slice_bounds(args, v.len());
                Value::Object(Obj::array(v[start..end].to_vec()))
            }
            "at" => {
                let v = elements.borrow();
                let n = arg(args, 0).to_number();
                let idx = if n < 0.0 { v.len() as f64 + n } else { n };
                if idx < 0.0 {
                    Value::Undefined
                } else {
                    v.get(idx as usize).cloned().unwrap_or(Value::Undefined)
                }
            }
            "concat" => {
                let mut out = elements.borrow().clone();
                for a in args {
                    match a {
                        Value::Object(o) if o.is_array() => {
                            out.extend(o.elements().expect("array").borrow().iter().cloned());
                        }
                        other => out.push(other.clone()),
                    }
                }
                Value::Object(Obj::array(out))
            }
            "reverse" => {
                elements.borrow_mut().reverse();
                Value::Object(Rc::clone(arr))
            }
            "fill" => {
                let val = arg(args, 0);
                for slot in elements.borrow_mut().iter_mut() {
                    *slot = val.clone();
                }
                Value::Object(Rc::clone(arr))
            }
            "flat" => {
                let mut out = Vec::new();
                for v in elements.borrow().iter() {
                    match v {
                        Value::Object(o) if o.is_array() => {
                            out.extend(o.elements().expect("array").borrow().iter().cloned());
                        }
                        other => out.push(other.clone()),
                    }
                }
                Value::Object(Obj::array(out))
            }
            "lastIndexOf" => {
                let needle = arg(args, 0);
                let v = elements.borrow();
                let idx = v
                    .iter()
                    .rposition(|x| super::value::strict_equals(x, &needle));
                Value::Number(idx.map_or(-1.0, |i| i as f64))
            }
            "unshift" => {
                let mut v = elements.borrow_mut();
                for (i, a) in args.iter().enumerate() {
                    v.insert(i, a.clone());
                }
                Value::Number(v.len() as f64)
            }
            "map" | "filter" | "forEach" | "find" | "findIndex" | "some" | "every" => {
                return self.array_iter_method(arr, name, args).map(Some);
            }
            "reduce" => {
                return self.array_reduce(arr, args).map(Some);
            }
            "sort" => {
                return self.array_sort(arr, args).map(Some);
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }

    /// `Array.prototype.sort`, optionally with a comparator.
    fn array_sort(&mut self, arr: &Rc<Obj<'a>>, args: &[Value<'a>]) -> Completion<'a, Value<'a>> {
        let mut items: Vec<Value<'a>> = arr.elements().expect("array").borrow().clone();
        let comparator = arg(args, 0);
        let has_cmp = comparator.is_callable();
        // Insertion sort so the (fallible, evaluator-driven) comparator can be
        // called without fighting the borrow checker.
        for i in 1..items.len() {
            let mut j = i;
            while j > 0 {
                let order = if has_cmp {
                    self.call_with_this(
                        comparator.clone(),
                        Value::Undefined,
                        alloc::vec![items[j - 1].clone(), items[j].clone()],
                    )?
                    .to_number()
                } else {
                    // Default order: compare by string value.
                    let a = items[j - 1].to_js_string();
                    let b = items[j].to_js_string();
                    if a > b { 1.0 } else { -1.0 }
                };
                if order > 0.0 {
                    items.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
        *arr.elements().expect("array").borrow_mut() = items;
        Ok(Value::Object(Rc::clone(arr)))
    }

    /// The callback-driven array methods (`map`/`filter`/`forEach`/`find`/
    /// `some`/`every`).
    fn array_iter_method(
        &mut self,
        arr: &Rc<Obj<'a>>,
        name: &str,
        args: &[Value<'a>],
    ) -> Completion<'a, Value<'a>> {
        let callback = arg(args, 0);
        let snapshot: Vec<Value<'a>> = arr.elements().expect("array").borrow().clone();
        let mut out: Vec<Value<'a>> = Vec::new();
        for (i, item) in snapshot.iter().enumerate() {
            let call_args = alloc::vec![
                item.clone(),
                Value::Number(i as f64),
                Value::Object(Rc::clone(arr)),
            ];
            let r = self.call_with_this(callback.clone(), Value::Undefined, call_args)?;
            match name {
                "map" => out.push(r),
                "filter" => {
                    if r.to_boolean() {
                        out.push(item.clone());
                    }
                }
                "forEach" => {}
                "find" => {
                    if r.to_boolean() {
                        return Ok(item.clone());
                    }
                }
                "findIndex" => {
                    if r.to_boolean() {
                        return Ok(Value::Number(i as f64));
                    }
                }
                "some" => {
                    if r.to_boolean() {
                        return Ok(Value::Bool(true));
                    }
                }
                "every" => {
                    if !r.to_boolean() {
                        return Ok(Value::Bool(false));
                    }
                }
                _ => unreachable!(),
            }
        }
        Ok(match name {
            "map" | "filter" => Value::Object(Obj::array(out)),
            "find" => Value::Undefined,
            "findIndex" => Value::Number(-1.0),
            "some" => Value::Bool(false),
            "every" => Value::Bool(true),
            _ => Value::Undefined, // forEach
        })
    }

    fn array_reduce(&mut self, arr: &Rc<Obj<'a>>, args: &[Value<'a>]) -> Completion<'a, Value<'a>> {
        let callback = arg(args, 0);
        let snapshot: Vec<Value<'a>> = arr.elements().expect("array").borrow().clone();
        let mut iter = snapshot.iter().enumerate();
        let mut acc = if args.len() >= 2 {
            arg(args, 1)
        } else {
            match iter.next() {
                Some((_, v)) => v.clone(),
                None => return Err(Value::str("reduce of empty array with no initial value")),
            }
        };
        for (i, item) in iter {
            let call_args = alloc::vec![
                acc,
                item.clone(),
                Value::Number(i as f64),
                Value::Object(Rc::clone(arr)),
            ];
            acc = self.call_with_this(callback.clone(), Value::Undefined, call_args)?;
        }
        Ok(acc)
    }
}

/// Appends the iterable contents of `value` into `out` (arrays, strings,
/// Sets/Maps, and array-likes with a numeric `length`).
fn iterate_into<'a>(value: &Value<'a>, out: &mut Vec<Value<'a>>) {
    match value {
        Value::Object(o) if o.is_array() => {
            out.extend(o.elements().expect("array").borrow().iter().cloned());
        }
        Value::Object(o) if o.as_collection().is_some() => {
            let c = o.as_collection().unwrap().borrow();
            if c.is_set {
                out.extend(c.entries.iter().map(|(k, _)| k.clone()));
            } else {
                out.extend(
                    c.entries
                        .iter()
                        .map(|(k, v)| Value::Object(Obj::array(alloc::vec![k.clone(), v.clone()]))),
                );
            }
        }
        Value::Str(s) => out.extend(s.chars().map(|c| Value::str(c.to_string()))),
        // Array-like: an object with a numeric `length`.
        Value::Object(o) => {
            let len = o.get("length").to_number();
            if len.is_finite() && len >= 0.0 {
                for i in 0..len as usize {
                    out.push(o.get(&i.to_string()));
                }
            }
        }
        _ => {}
    }
}

/// Inserts or updates `(key, value)` in a `Map`/`Set` (SameValueZero key).
fn collection_set<'a>(obj: &Rc<Obj<'a>>, key: Value<'a>, value: Value<'a>) {
    use super::value::same_value_zero;
    let cell = obj.as_collection().expect("collection");
    let mut c = cell.borrow_mut();
    if let Some(slot) = c.entries.iter_mut().find(|(k, _)| same_value_zero(k, &key)) {
        slot.1 = value;
    } else {
        c.entries.push((key, value));
    }
}

/// String prototype methods (none need the evaluator).
fn string_method<'a>(s: &str, name: &str, args: &[Value<'a>]) -> Option<Value<'a>> {
    let chars: Vec<char> = s.chars().collect();
    let result = match name {
        "toUpperCase" => Value::str(s.to_uppercase()),
        "toLowerCase" => Value::str(s.to_lowercase()),
        "trim" => Value::str(s.trim().to_string()),
        "includes" => Value::Bool(s.contains(&arg(args, 0).to_js_string())),
        "startsWith" => Value::Bool(s.starts_with(&arg(args, 0).to_js_string())),
        "endsWith" => Value::Bool(s.ends_with(&arg(args, 0).to_js_string())),
        "repeat" => {
            let n = arg(args, 0).to_number();
            if n < 0.0 || !n.is_finite() {
                return Some(Value::str("")); // RangeError elided for the MVP
            }
            Value::str(s.repeat(n as usize))
        }
        "charAt" => {
            let i = arg(args, 0).to_number() as usize;
            Value::str(chars.get(i).map(|c| c.to_string()).unwrap_or_default())
        }
        "charCodeAt" => {
            let i = arg(args, 0).to_number() as usize;
            chars
                .get(i)
                .map_or(Value::Number(f64::NAN), |c| Value::Number(*c as u32 as f64))
        }
        "indexOf" => {
            let needle = arg(args, 0).to_js_string();
            Value::Number(
                s.find(&needle)
                    .map_or(-1.0, |b| s[..b].chars().count() as f64),
            )
        }
        "slice" => {
            let (start, end) = slice_bounds(args, chars.len());
            Value::str(chars[start..end].iter().collect::<String>())
        }
        "split" => {
            let sep = arg(args, 0);
            let parts: Vec<Value<'a>> = match sep {
                Value::Undefined => alloc::vec![Value::str(s.to_string())],
                _ => {
                    let sep = sep.to_js_string();
                    if sep.is_empty() {
                        s.chars().map(|c| Value::str(c.to_string())).collect()
                    } else {
                        s.split(&sep).map(|p| Value::str(p.to_string())).collect()
                    }
                }
            };
            Value::Object(Obj::array(parts))
        }
        "at" => {
            let n = arg(args, 0).to_number();
            let idx = if n < 0.0 { chars.len() as f64 + n } else { n };
            if idx < 0.0 {
                Value::Undefined
            } else {
                chars
                    .get(idx as usize)
                    .map_or(Value::Undefined, |c| Value::str(c.to_string()))
            }
        }
        "trimStart" => Value::str(s.trim_start().to_string()),
        "trimEnd" => Value::str(s.trim_end().to_string()),
        "concat" => {
            let mut out = String::from(s);
            for a in args {
                out.push_str(&a.to_js_string());
            }
            Value::str(out)
        }
        "replace" => {
            let from = arg(args, 0).to_js_string();
            let to = arg(args, 1).to_js_string();
            Value::str(s.replacen(&from, &to, 1))
        }
        "replaceAll" => {
            let from = arg(args, 0).to_js_string();
            let to = arg(args, 1).to_js_string();
            Value::str(s.replace(&from, &to))
        }
        "padStart" | "padEnd" => {
            let target = arg(args, 0).to_number() as usize;
            let pad = if matches!(arg(args, 1), Value::Undefined) {
                " ".to_string()
            } else {
                arg(args, 1).to_js_string()
            };
            let cur = chars.len();
            if cur >= target || pad.is_empty() {
                Value::str(s.to_string())
            } else {
                let need = target - cur;
                let mut fill = String::new();
                while fill.chars().count() < need {
                    fill.push_str(&pad);
                }
                let fill: String = fill.chars().take(need).collect();
                Value::str(if name == "padStart" {
                    fill + s
                } else {
                    String::from(s) + &fill
                })
            }
        }
        "toString" => Value::str(s.to_string()),
        _ => return None,
    };
    Some(result)
}

/// Number prototype methods.
fn number_method<'a>(n: f64, name: &str, args: &[Value<'a>]) -> Option<Value<'a>> {
    let result = match name {
        "toFixed" => {
            let digits = arg(args, 0).to_number();
            let d = if digits.is_finite() && digits >= 0.0 {
                digits as usize
            } else {
                0
            };
            Value::str(alloc::format!("{n:.d$}"))
        }
        "toString" => {
            let radix = arg(args, 0).to_number();
            if radix == 10.0 || matches!(arg(args, 0), Value::Undefined) {
                Value::str(Value::Number(n).to_js_string())
            } else {
                Value::str(to_radix_string(n, radix as u32))
            }
        }
        _ => return None,
    };
    Some(result)
}

/// Renders an integer-valued number in the given radix (2–36); fractional
/// parts are truncated (the common `(n).toString(radix)` case).
fn to_radix_string(n: f64, radix: u32) -> String {
    if !(2..=36).contains(&radix) || !n.is_finite() {
        return Value::Number(n).to_js_string();
    }
    let neg = n < 0.0;
    let mut int = n.abs().trunc() as u64;
    if int == 0 {
        return "0".into();
    }
    let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut buf = Vec::new();
    while int > 0 {
        buf.push(digits[(int % u64::from(radix)) as usize]);
        int /= u64::from(radix);
    }
    if neg {
        buf.push(b'-');
    }
    buf.reverse();
    String::from_utf8(buf).expect("ascii digits")
}

/// Resolves the `(start, end)` of a `slice(start, end)` call against `len`,
/// applying the negative-index and clamping rules.
fn slice_bounds<'a>(args: &[Value<'a>], len: usize) -> (usize, usize) {
    let norm = |v: Value<'a>, default: usize| -> usize {
        if matches!(v, Value::Undefined) {
            return default;
        }
        let n = v.to_number();
        if n.is_nan() {
            return 0;
        }
        if n < 0.0 {
            (len as f64 + n).max(0.0) as usize
        } else {
            (n as usize).min(len)
        }
    };
    let start = norm(arg(args, 0), 0);
    let end = norm(arg(args, 1), len);
    (start, end.max(start))
}

enum EntryKind {
    Key,
    Value,
    Pair,
}

fn object_entries<'a>(value: &Value<'a>, kind: EntryKind) -> Value<'a> {
    let Value::Object(o) = value else {
        return Value::Object(Obj::array(Vec::new()));
    };
    let items: Vec<Value<'a>> = o
        .own_keys()
        .into_iter()
        .map(|k| match kind {
            EntryKind::Key => Value::str(k),
            EntryKind::Value => o.get(&k),
            EntryKind::Pair => {
                Value::Object(Obj::array(alloc::vec![Value::str(k.clone()), o.get(&k)]))
            }
        })
        .collect();
    Value::Object(Obj::array(items))
}

/// `JSON.stringify` for the supported value set. Returns `None` for values that
/// serialize to nothing at the top level (`undefined`, functions).
fn json_stringify(value: &Value<'_>) -> Option<String> {
    match value {
        Value::Undefined | Value::Function(_) | Value::Native(_) | Value::Class(_) => None,
        Value::Null => Some("null".into()),
        Value::Bool(b) => Some(if *b { "true" } else { "false" }.into()),
        Value::Number(n) => Some(if n.is_finite() {
            value.to_js_string()
        } else {
            "null".into()
        }),
        Value::Str(s) => Some(json_quote(s)),
        Value::Object(o) if o.is_array() => {
            let parts: Vec<String> = o
                .elements()
                .expect("array")
                .borrow()
                .iter()
                .map(|v| json_stringify(v).unwrap_or_else(|| "null".into()))
                .collect();
            Some(format!("[{}]", parts.join(",")))
        }
        Value::Object(o) => {
            let mut parts = Vec::new();
            for k in o.own_keys() {
                if let Some(v) = json_stringify(&o.get(&k)) {
                    parts.push(format!("{}:{}", json_quote(&k), v));
                }
            }
            Some(format!("{{{}}}", parts.join(",")))
        }
    }
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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A small recursive-descent `JSON.parse`.
fn json_parse<'a>(s: &str) -> Completion<'a, Value<'a>> {
    let mut p = JsonParser {
        chars: s.chars().collect(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(Value::str("Unexpected token in JSON"));
    }
    Ok(v)
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
}

impl JsonParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.pos += 1;
        }
    }
    fn value<'a>(&mut self) -> Completion<'a, Value<'a>> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Value::str(self.string()?)),
            Some('t') => self.literal("true", Value::Bool(true)),
            Some('f') => self.literal("false", Value::Bool(false)),
            Some('n') => self.literal("null", Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            _ => Err(Value::str("Unexpected token in JSON")),
        }
    }
    fn literal<'a>(&mut self, word: &str, v: Value<'a>) -> Completion<'a, Value<'a>> {
        for expected in word.chars() {
            if self.bump() != Some(expected) {
                return Err(Value::str("Unexpected token in JSON"));
            }
        }
        Ok(v)
    }
    fn number<'a>(&mut self) -> Completion<'a, Value<'a>> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || c.is_ascii_digit())
        {
            self.pos += 1;
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| Value::str("Invalid number in JSON"))
    }
    fn string<'a>(&mut self) -> Completion<'a, String> {
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(Value::str("Unterminated string in JSON")),
                Some('"') => return Ok(out),
                Some('\\') => match self.bump() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{08}'),
                    Some('f') => out.push('\u{0C}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let d = self
                                .bump()
                                .and_then(|c| c.to_digit(16))
                                .ok_or_else(|| Value::str("Invalid \\u escape in JSON"))?;
                            code = code * 16 + d;
                        }
                        out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    _ => return Err(Value::str("Invalid escape in JSON")),
                },
                Some(c) => out.push(c),
            }
        }
    }
    fn array<'a>(&mut self) -> Completion<'a, Value<'a>> {
        self.bump(); // `[`
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.bump();
            return Ok(Value::Object(Obj::array(items)));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.bump() {
                Some(',') => self.skip_ws(),
                Some(']') => return Ok(Value::Object(Obj::array(items))),
                _ => return Err(Value::str("Expected ',' or ']' in JSON")),
            }
        }
    }
    fn object<'a>(&mut self) -> Completion<'a, Value<'a>> {
        self.bump(); // `{`
        let obj = Obj::object();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(Value::Object(obj));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                return Err(Value::str("Expected string key in JSON"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.bump() != Some(':') {
                return Err(Value::str("Expected ':' in JSON"));
            }
            let v = self.value()?;
            obj.set(&key, v);
            self.skip_ws();
            match self.bump() {
                Some(',') => {}
                Some('}') => return Ok(Value::Object(obj)),
                _ => return Err(Value::str("Expected ',' or '}' in JSON")),
            }
        }
    }
}

/// `parseInt` over a trimmed string with an optional radix.
fn parse_int(s: &str, radix: f64) -> f64 {
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut radix = if radix == 0.0 || radix.is_nan() {
        10
    } else {
        radix as u32
    };
    let rest = if (radix == 16 || radix == 10) && (rest.starts_with("0x") || rest.starts_with("0X"))
    {
        radix = 16;
        &rest[2..]
    } else {
        rest
    };
    let digits: String = rest.chars().take_while(|c| c.is_digit(radix)).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    let mut value = 0.0_f64;
    for c in digits.chars() {
        value = value * f64::from(radix) + f64::from(c.to_digit(radix).unwrap());
    }
    if neg { -value } else { value }
}

/// `parseFloat` over a trimmed string: consumes the longest valid float prefix.
fn parse_float(s: &str) -> f64 {
    let bytes = s.as_bytes();
    let mut end = 0;
    let mut seen_dot = false;
    let mut seen_e = false;
    while end < bytes.len() {
        let c = bytes[end] as char;
        let ok = c.is_ascii_digit()
            || (end == 0 && (c == '-' || c == '+'))
            || (c == '.' && !seen_dot && !seen_e)
            || ((c == 'e' || c == 'E') && !seen_e && end > 0)
            || ((c == '-' || c == '+') && end > 0 && matches!(bytes[end - 1] as char, 'e' | 'E'));
        if !ok {
            break;
        }
        if c == '.' {
            seen_dot = true;
        }
        if c == 'e' || c == 'E' {
            seen_e = true;
        }
        end += 1;
    }
    s[..end].parse::<f64>().unwrap_or(f64::NAN)
}
