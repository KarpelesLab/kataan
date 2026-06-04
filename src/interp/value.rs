//! Runtime values for the tree-walking interpreter, plus the ECMAScript
//! abstract conversions over them.
//!
//! This is the *interpreter-era* value representation: a plain tagged enum with
//! reference-counted heap payloads. It validates semantics ahead of the
//! performance-oriented NaN-boxed representation that replaces it once the
//! object model and GC land (see `ROADMAP.md`).
//!
//! Values borrow the AST (`'a`) so that function closures can reference their
//! definition without cloning the body.

use super::env::Env;
use crate::ast::{Arrow, Function};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

/// A JavaScript runtime value (interpreter era; primitives + functions).
#[derive(Clone)]
pub enum Value<'a> {
    /// `undefined`.
    Undefined,
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// An IEEE-754 double.
    Number(f64),
    /// An immutable string.
    Str(Rc<str>),
    /// A user-defined function closure.
    Function(Rc<Closure<'a>>),
    /// A native (Rust-implemented) function.
    Native(Rc<NativeFn<'a>>),
    /// An ordinary object or array.
    Object(Rc<Obj<'a>>),
}

/// An ordinary object (or array) in the interpreter-era object model: an
/// insertion-ordered set of string-keyed properties, with an optional dense
/// element vector for arrays. (Prototype chains, accessors, and hidden classes
/// arrive with the real object model.)
pub struct Obj<'a> {
    props: RefCell<Vec<(Box<str>, Value<'a>)>>,
    array: Option<RefCell<Vec<Value<'a>>>>,
}

impl<'a> Obj<'a> {
    /// Creates an empty ordinary object.
    #[must_use]
    pub fn object() -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: None,
        })
    }

    /// Creates an array object from its initial elements.
    #[must_use]
    pub fn array(elements: Vec<Value<'a>>) -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: Some(RefCell::new(elements)),
        })
    }

    /// Whether this is an array.
    #[must_use]
    pub fn is_array(&self) -> bool {
        self.array.is_some()
    }

    /// The array elements, if this is an array.
    #[must_use]
    pub fn elements(&self) -> Option<&RefCell<Vec<Value<'a>>>> {
        self.array.as_ref()
    }

    /// Gets a property by string key, resolving array indices and `length`.
    /// Returns `undefined` for absent properties.
    #[must_use]
    pub fn get(&self, key: &str) -> Value<'a> {
        if let Some(arr) = &self.array {
            if key == "length" {
                return Value::Number(arr.borrow().len() as f64);
            }
            if let Ok(i) = key.parse::<usize>() {
                return arr.borrow().get(i).cloned().unwrap_or(Value::Undefined);
            }
        }
        self.props
            .borrow()
            .iter()
            .find(|(k, _)| **k == *key)
            .map(|(_, v)| v.clone())
            .unwrap_or(Value::Undefined)
    }

    /// Whether the object has an own property with `key`.
    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        if let Some(arr) = &self.array {
            if key == "length" {
                return true;
            }
            if let Ok(i) = key.parse::<usize>() {
                return i < arr.borrow().len();
            }
        }
        self.props.borrow().iter().any(|(k, _)| **k == *key)
    }

    /// Sets a property by string key, growing the array (with holes filled by
    /// `undefined`) for in-bounds-or-beyond array indices.
    pub fn set(&self, key: &str, value: Value<'a>) {
        if let Some(arr) = &self.array {
            if let Ok(i) = key.parse::<usize>() {
                let mut v = arr.borrow_mut();
                if i >= v.len() {
                    v.resize(i + 1, Value::Undefined);
                }
                v[i] = value;
                return;
            }
            if key == "length" {
                return; // length assignment is not yet modeled
            }
        }
        let mut props = self.props.borrow_mut();
        if let Some(slot) = props.iter_mut().find(|(k, _)| **k == *key) {
            slot.1 = value;
        } else {
            props.push((key.into(), value));
        }
    }

    /// The own enumerable string keys, in order (array indices first).
    #[must_use]
    pub fn own_keys(&self) -> Vec<alloc::string::String> {
        let mut keys = Vec::new();
        if let Some(arr) = &self.array {
            for i in 0..arr.borrow().len() {
                keys.push(alloc::format!("{i}"));
            }
        }
        for (k, _) in self.props.borrow().iter() {
            keys.push(k.as_ref().into());
        }
        keys
    }
}

/// A user-defined function paired with the environment it closed over.
pub struct Closure<'a> {
    /// The callable AST node (a function or an arrow).
    pub def: Callable<'a>,
    /// The captured (lexical) environment.
    pub env: Env<'a>,
}

/// The two callable AST shapes a [`Closure`] can wrap.
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub enum Callable<'a> {
    Function(&'a Function),
    Arrow(&'a Arrow),
}

/// A native function: a name and a Rust callback.
pub struct NativeFn<'a> {
    /// The function's `name` (for diagnostics and `Function.prototype.name`).
    pub name: &'static str,
    /// The callback. Returns either a value or a thrown value.
    #[allow(clippy::type_complexity)]
    pub call: alloc::boxed::Box<dyn Fn(&[Value<'a>]) -> Result<Value<'a>, Value<'a>> + 'a>,
}

impl<'a> Value<'a> {
    /// Builds a string value from anything string-like.
    pub fn str(s: impl Into<Rc<str>>) -> Self {
        Value::Str(s.into())
    }

    /// The `typeof` operator's result string.
    #[must_use]
    pub fn type_of(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::Null => "object",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::Str(_) => "string",
            Value::Function(_) | Value::Native(_) => "function",
            Value::Object(_) => "object",
        }
    }

    /// Whether the value is callable.
    #[must_use]
    pub fn is_callable(&self) -> bool {
        matches!(self, Value::Function(_) | Value::Native(_))
    }

    /// `ToBoolean` (the truthiness coercion).
    #[must_use]
    pub fn to_boolean(&self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0 && !n.is_nan(),
            Value::Str(s) => !s.is_empty(),
            Value::Function(_) | Value::Native(_) | Value::Object(_) => true,
        }
    }

    /// `ToNumber`.
    #[must_use]
    pub fn to_number(&self) -> f64 {
        match self {
            Value::Undefined => f64::NAN,
            Value::Null => 0.0,
            Value::Bool(b) => f64::from(u8::from(*b)),
            Value::Number(n) => *n,
            Value::Str(s) => string_to_number(s),
            Value::Function(_) | Value::Native(_) => f64::NAN,
            // ToPrimitive on an array yields its join; other objects → NaN.
            Value::Object(o) if o.is_array() => string_to_number(&self.to_js_string()),
            Value::Object(_) => f64::NAN,
        }
    }

    /// `ToString`.
    #[must_use]
    pub fn to_js_string(&self) -> String {
        match self {
            Value::Undefined => "undefined".into(),
            Value::Null => "null".into(),
            Value::Bool(b) => if *b { "true" } else { "false" }.into(),
            Value::Number(n) => number_to_string(*n),
            Value::Str(s) => s.as_ref().into(),
            Value::Function(c) => match c.def {
                Callable::Function(f) => {
                    let name = f.id.as_ref().map_or("", |id| &id.name);
                    alloc::format!("function {name}() {{ … }}")
                }
                Callable::Arrow(_) => "() => { … }".into(),
            },
            Value::Native(n) => alloc::format!("function {}() {{ [native code] }}", n.name),
            Value::Object(o) if o.is_array() => {
                // Array `toString` is the elements joined by ",".
                let elems = o.elements().expect("array");
                let parts: Vec<String> = elems
                    .borrow()
                    .iter()
                    .map(|v| match v {
                        Value::Undefined | Value::Null => String::new(),
                        other => other.to_js_string(),
                    })
                    .collect();
                parts.join(",")
            }
            Value::Object(_) => "[object Object]".into(),
        }
    }

    /// `ToInt32` (for the bitwise operators).
    #[must_use]
    pub fn to_int32(&self) -> i32 {
        to_int32(self.to_number())
    }

    /// `ToUint32` (for `>>>`).
    #[must_use]
    pub fn to_uint32(&self) -> u32 {
        to_int32(self.to_number()) as u32
    }
}

/// `7.1.4 ToNumber` applied to a string: trims ECMAScript whitespace, accepts
/// the empty string as `0`, decimal/hex/octal/binary literals, and `Infinity`.
fn string_to_number(s: &str) -> f64 {
    let t = s.trim_matches(|c: char| c.is_whitespace());
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u128::from_str_radix(hex, 16).map_or(f64::NAN, |v| v as f64);
    }
    if let Some(oct) = t.strip_prefix("0o").or_else(|| t.strip_prefix("0O")) {
        return u128::from_str_radix(oct, 8).map_or(f64::NAN, |v| v as f64);
    }
    if let Some(bin) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        return u128::from_str_radix(bin, 2).map_or(f64::NAN, |v| v as f64);
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// `6.1.6.1.20 Number::toString` (radix 10), pragmatic edition: correct for the
/// special values and integers; finite non-integers use Rust's shortest
/// round-tripping `Display`, which matches JS for the vast majority of inputs.
fn number_to_string(n: f64) -> String {
    if n.is_nan() {
        return "NaN".into();
    }
    if n.is_infinite() {
        return if n > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if n == 0.0 {
        return "0".into(); // also collapses -0 to "0"
    }
    alloc::format!("{n}")
}

/// `7.1.6 ToInt32`.
fn to_int32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    let m = n.trunc();
    let two32 = 4_294_967_296.0_f64;
    let mut int32 = m.rem_euclid(two32);
    if int32 >= two32 / 2.0 {
        int32 -= two32;
    }
    int32 as i64 as i32
}

/// Strict equality (`===`).
#[must_use]
pub fn strict_equals<'a>(a: &Value<'a>, b: &Value<'a>) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y, // NaN != NaN, +0 == -0
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Function(x), Value::Function(y)) => Rc::ptr_eq(x, y),
        (Value::Native(x), Value::Native(y)) => Rc::ptr_eq(x, y),
        (Value::Object(x), Value::Object(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// Abstract (loose) equality (`==`) for the primitive value set.
#[must_use]
pub fn loose_equals<'a>(a: &Value<'a>, b: &Value<'a>) -> bool {
    use Value::{Bool, Null, Number, Str, Undefined};
    match (a, b) {
        // Same-type comparisons reduce to strict equality.
        _ if core::mem::discriminant(a) == core::mem::discriminant(b) => strict_equals(a, b),
        // null and undefined are loosely equal to each other and nothing else.
        (Null | Undefined, Null | Undefined) => true,
        (Null | Undefined, _) | (_, Null | Undefined) => false,
        // Number/String: coerce the string to a number.
        (Number(_), Str(_)) | (Str(_), Number(_)) => a.to_number() == b.to_number(),
        // A boolean operand is coerced to a number, then compared again.
        (Bool(_), _) => loose_equals(&Number(a.to_number()), b),
        (_, Bool(_)) => loose_equals(a, &Number(b.to_number())),
        _ => false,
    }
}

impl fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s:?}"),
            other => write!(f, "{}", other.to_js_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthiness() {
        assert!(!Value::Undefined.to_boolean());
        assert!(!Value::Number(0.0).to_boolean());
        assert!(!Value::Number(f64::NAN).to_boolean());
        assert!(Value::Number(1.0).to_boolean());
        assert!(!Value::str("").to_boolean());
        assert!(Value::str("x").to_boolean());
    }

    #[test]
    fn numeric_coercion() {
        assert_eq!(Value::str("  42 ").to_number(), 42.0);
        assert_eq!(Value::str("0xFF").to_number(), 255.0);
        assert_eq!(Value::str("").to_number(), 0.0);
        assert!(Value::str("abc").to_number().is_nan());
        assert_eq!(Value::Bool(true).to_number(), 1.0);
        assert_eq!(Value::Null.to_number(), 0.0);
    }

    #[test]
    fn stringification() {
        assert_eq!(Value::Number(1.0).to_js_string(), "1");
        assert_eq!(Value::Number(1.5).to_js_string(), "1.5");
        assert_eq!(Value::Number(-0.0).to_js_string(), "0");
        assert_eq!(Value::Number(f64::NAN).to_js_string(), "NaN");
        assert_eq!(Value::Number(f64::INFINITY).to_js_string(), "Infinity");
    }

    #[test]
    fn int32_conversion() {
        assert_eq!(Value::Number(4_294_967_297.0).to_int32(), 1);
        assert_eq!(Value::Number(-1.0).to_uint32(), u32::MAX);
        assert_eq!(Value::str("3.9").to_int32(), 3);
    }

    #[test]
    fn equality() {
        assert!(loose_equals(&Value::Null, &Value::Undefined));
        assert!(loose_equals(&Value::Number(1.0), &Value::str("1")));
        assert!(loose_equals(&Value::Bool(true), &Value::Number(1.0)));
        assert!(!strict_equals(&Value::Number(1.0), &Value::str("1")));
        assert!(!strict_equals(
            &Value::Number(f64::NAN),
            &Value::Number(f64::NAN)
        ));
    }
}
