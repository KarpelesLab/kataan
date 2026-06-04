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

impl<'a> Interp<'a> {
    /// Installs the standard-library globals into the global scope.
    pub(super) fn install_stdlib(&self) {
        self.install_math();
        self.install_json();
        self.install_object_ctor();
        self.install_array_ctor();
        self.install_number_globals();
        self.install_errors();
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
        self.define_global(
            "Number",
            native("Number", |a| Ok(Value::Number(arg(a, 0).to_number()))),
        );
        self.define_global(
            "String",
            native("String", |a| {
                Ok(Value::str(if a.is_empty() {
                    String::new()
                } else {
                    arg(a, 0).to_js_string()
                }))
            }),
        );
        self.define_global(
            "Boolean",
            native("Boolean", |a| Ok(Value::Bool(arg(a, 0).to_boolean()))),
        );
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
            Value::Str(s) => Ok(string_method(s, name, args)),
            _ => Ok(None),
        }
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
            "map" | "filter" | "forEach" | "find" | "some" | "every" => {
                return self.array_iter_method(arr, name, args).map(Some);
            }
            "reduce" => {
                return self.array_reduce(arr, args).map(Some);
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
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
        _ => return None,
    };
    Some(result)
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
