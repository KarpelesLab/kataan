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
    /// A class (a constructor plus its shared prototype and statics).
    Class(Rc<ClassValue<'a>>),
}

/// A class value: its AST definition, the environment it closed over, the
/// shared instance `prototype` object (carrying the methods), the object
/// holding its static members, and a resolved superclass for `extends`.
pub struct ClassValue<'a> {
    /// The class's AST definition.
    pub def: &'a crate::ast::Class,
    /// The captured environment.
    pub env: Env<'a>,
    /// The environment methods close over — the defining env plus a `%class%`
    /// binding, so `super` resolves inside methods/the constructor.
    pub method_env: Env<'a>,
    /// The shared prototype object holding instance methods.
    pub prototype: Rc<Obj<'a>>,
    /// The object holding static members.
    pub statics: Rc<Obj<'a>>,
    /// The resolved superclass, if this class `extends` another.
    pub super_class: Option<Rc<ClassValue<'a>>>,
    /// The constructor closure, if the class declares one.
    pub ctor: RefCell<Option<Value<'a>>>,
}

/// An ordinary object (or array) in the interpreter-era object model: an
/// insertion-ordered set of string-keyed properties, with an optional dense
/// element vector for arrays. (Prototype chains, accessors, and hidden classes
/// arrive with the real object model.)
pub struct Obj<'a> {
    props: RefCell<Vec<(Box<str>, Value<'a>)>>,
    array: Option<RefCell<Vec<Value<'a>>>>,
    proto: RefCell<Option<Rc<Obj<'a>>>>,
    collection: Option<RefCell<Collection<'a>>>,
    accessors: RefCell<Vec<(Box<str>, Accessor<'a>)>>,
    /// For built-in constructor objects (`String`, `Number`, …): the underlying
    /// native invoked when the object is called or `new`'d. Lets a constructor
    /// also carry static members as ordinary properties.
    callable: RefCell<Option<Value<'a>>>,
    /// For `Promise` instances: the internal settlement state.
    promise: RefCell<Option<Rc<RefCell<super::promise::PromiseState<'a>>>>>,
    /// For bytecode functions: the compiled module + chunk + captured values.
    bytecode: RefCell<Option<Rc<BytecodeFn<'a>>>>,
}

/// A compiled (bytecode) function value: the module it lives in, the index of
/// its chunk, and any values captured from enclosing scopes (upvalues).
pub struct BytecodeFn<'a> {
    /// The module containing this function's chunk (and any nested ones).
    pub module: Rc<crate::bytecode::Module>,
    /// The chunk index within the module.
    pub chunk: u32,
    /// Captured upvalue cells, addressed by capture index.
    pub captures: Vec<Value<'a>>,
}

/// A getter/setter accessor property.
#[derive(Clone, Default)]
pub struct Accessor<'a> {
    /// The getter function, invoked on read.
    pub get: Option<Value<'a>>,
    /// The setter function, invoked on write.
    pub set: Option<Value<'a>>,
}

/// The backing store of a `Map` or `Set`: insertion-ordered key/value entries
/// compared by SameValueZero. For a `Set`, the value equals the key.
pub struct Collection<'a> {
    /// Whether this is a `Set` (vs a `Map`) — affects `forEach`/iteration.
    pub is_set: bool,
    /// The entries, in insertion order.
    pub entries: Vec<(Value<'a>, Value<'a>)>,
}

impl<'a> Obj<'a> {
    /// Creates an empty ordinary object.
    #[must_use]
    pub fn object() -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: None,
            proto: RefCell::new(None),
            collection: None,
            accessors: RefCell::new(Vec::new()),
            callable: RefCell::new(None),
            promise: RefCell::new(None),
            bytecode: RefCell::new(None),
        })
    }

    /// Creates an empty object whose prototype is `proto`.
    #[must_use]
    pub fn with_proto(proto: Rc<Obj<'a>>) -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: None,
            proto: RefCell::new(Some(proto)),
            collection: None,
            accessors: RefCell::new(Vec::new()),
            callable: RefCell::new(None),
            promise: RefCell::new(None),
            bytecode: RefCell::new(None),
        })
    }

    /// Creates an array object from its initial elements.
    #[must_use]
    pub fn array(elements: Vec<Value<'a>>) -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: Some(RefCell::new(elements)),
            proto: RefCell::new(None),
            collection: None,
            accessors: RefCell::new(Vec::new()),
            callable: RefCell::new(None),
            promise: RefCell::new(None),
            bytecode: RefCell::new(None),
        })
    }

    /// Creates an empty `Map` (`is_set = false`) or `Set` (`is_set = true`).
    #[must_use]
    pub fn collection(is_set: bool) -> Rc<Obj<'a>> {
        Rc::new(Obj {
            props: RefCell::new(Vec::new()),
            array: None,
            proto: RefCell::new(None),
            collection: Some(RefCell::new(Collection {
                is_set,
                entries: Vec::new(),
            })),
            accessors: RefCell::new(Vec::new()),
            callable: RefCell::new(None),
            promise: RefCell::new(None),
            bytecode: RefCell::new(None),
        })
    }

    /// The `Map`/`Set` backing store, if this object is one.
    #[must_use]
    pub fn as_collection(&self) -> Option<&RefCell<Collection<'a>>> {
        self.collection.as_ref()
    }

    /// Marks this object as a callable constructor backed by `f`.
    pub fn set_callable(&self, f: Value<'a>) {
        *self.callable.borrow_mut() = Some(f);
    }

    /// The native this object delegates calls to, if it is a callable
    /// constructor object.
    #[must_use]
    pub fn callable(&self) -> Option<Value<'a>> {
        self.callable.borrow().clone()
    }

    /// Marks this object as a `Promise` with the given internal state.
    pub fn set_promise_state(&self, state: Rc<RefCell<super::promise::PromiseState<'a>>>) {
        *self.promise.borrow_mut() = Some(state);
    }

    /// This object's promise state, if it is a `Promise`.
    #[must_use]
    pub fn promise_state(&self) -> Option<Rc<RefCell<super::promise::PromiseState<'a>>>> {
        self.promise.borrow().clone()
    }

    /// Marks this object as a bytecode function.
    pub fn set_bytecode_fn(&self, f: Rc<BytecodeFn<'a>>) {
        *self.bytecode.borrow_mut() = Some(f);
    }

    /// The compiled function this object wraps, if it is a bytecode function.
    #[must_use]
    pub fn bytecode_fn(&self) -> Option<Rc<BytecodeFn<'a>>> {
        self.bytecode.borrow().clone()
    }

    /// Defines (or extends) the getter for `key`.
    pub fn define_getter(&self, key: &str, getter: Value<'a>) {
        let mut accs = self.accessors.borrow_mut();
        if let Some(slot) = accs.iter_mut().find(|(k, _)| **k == *key) {
            slot.1.get = Some(getter);
        } else {
            accs.push((
                key.into(),
                Accessor {
                    get: Some(getter),
                    set: None,
                },
            ));
        }
    }

    /// Defines (or extends) the setter for `key`.
    pub fn define_setter(&self, key: &str, setter: Value<'a>) {
        let mut accs = self.accessors.borrow_mut();
        if let Some(slot) = accs.iter_mut().find(|(k, _)| **k == *key) {
            slot.1.set = Some(setter);
        } else {
            accs.push((
                key.into(),
                Accessor {
                    get: None,
                    set: Some(setter),
                },
            ));
        }
    }

    /// Deletes an own property (data or accessor) by `key`. Array indices are
    /// cleared to `undefined`. Returns `true` (deletion of an absent property is
    /// also `true`, per the spec for configurable/absent properties).
    pub fn delete_key(&self, key: &str) -> bool {
        if let Some(arr) = &self.array
            && let Ok(i) = key.parse::<usize>()
        {
            let mut v = arr.borrow_mut();
            if i < v.len() {
                v[i] = Value::Undefined;
            }
            return true;
        }
        self.props.borrow_mut().retain(|(k, _)| **k != *key);
        self.accessors.borrow_mut().retain(|(k, _)| **k != *key);
        true
    }

    /// Finds an accessor for `key`, walking the prototype chain.
    #[must_use]
    pub fn find_accessor(&self, key: &str) -> Option<Accessor<'a>> {
        if let Some((_, acc)) = self.accessors.borrow().iter().find(|(k, _)| **k == *key) {
            return Some(acc.clone());
        }
        let mut proto = self.proto();
        while let Some(p) = proto {
            if let Some((_, acc)) = p.accessors.borrow().iter().find(|(k, _)| **k == *key) {
                return Some(acc.clone());
            }
            proto = p.proto();
        }
        None
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

    /// This object's prototype, if any.
    #[must_use]
    pub fn proto(&self) -> Option<Rc<Obj<'a>>> {
        self.proto.borrow().clone()
    }

    /// Sets this object's prototype.
    pub fn set_proto(&self, proto: Option<Rc<Obj<'a>>>) {
        *self.proto.borrow_mut() = proto;
    }

    /// Looks up an *own* property only (no prototype walk).
    #[must_use]
    pub fn get_own(&self, key: &str) -> Option<Value<'a>> {
        if let Some(arr) = &self.array {
            if key == "length" {
                return Some(Value::Number(arr.borrow().len() as f64));
            }
            if let Ok(i) = key.parse::<usize>() {
                return arr.borrow().get(i).cloned();
            }
        }
        self.props
            .borrow()
            .iter()
            .find(|(k, _)| **k == *key)
            .map(|(_, v)| v.clone())
    }

    /// Gets a property by string key, walking the prototype chain. Returns
    /// `undefined` for absent properties.
    #[must_use]
    pub fn get(&self, key: &str) -> Value<'a> {
        if let Some(v) = self.get_own(key) {
            return v;
        }
        let mut proto = self.proto();
        while let Some(p) = proto {
            if let Some(v) = p.get_own(key) {
                return v;
            }
            proto = p.proto();
        }
        Value::Undefined
    }

    /// Whether `key` is present on the object or anywhere in its prototype
    /// chain (the `in` operator).
    #[must_use]
    pub fn has_property(&self, key: &str) -> bool {
        if self.has(key) {
            return true;
        }
        let mut proto = self.proto();
        while let Some(p) = proto {
            if p.has(key) {
                return true;
            }
            proto = p.proto();
        }
        false
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
            || self.accessors.borrow().iter().any(|(k, _)| **k == *key)
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
        for (k, _) in self.accessors.borrow().iter() {
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
            Value::Function(_) | Value::Native(_) | Value::Class(_) => "function",
            // A bytecode function (or callable constructor object) is a function.
            Value::Object(o) if o.bytecode_fn().is_some() || o.callable().is_some() => "function",
            Value::Object(_) => "object",
        }
    }

    /// Whether the value is callable (as a plain call). Classes are *not*
    /// callable without `new`; a constructor *object* (`String`, …) is.
    #[must_use]
    pub fn is_callable(&self) -> bool {
        match self {
            Value::Function(_) | Value::Native(_) => true,
            Value::Object(o) => o.callable().is_some() || o.bytecode_fn().is_some(),
            _ => false,
        }
    }

    /// `ToBoolean` (the truthiness coercion).
    #[must_use]
    pub fn to_boolean(&self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0 && !n.is_nan(),
            Value::Str(s) => !s.is_empty(),
            Value::Function(_) | Value::Native(_) | Value::Object(_) | Value::Class(_) => true,
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
            Value::Function(_) | Value::Native(_) | Value::Class(_) => f64::NAN,
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
            Value::Class(c) => {
                let name = c.def.id.as_ref().map_or("", |id| &id.name);
                alloc::format!("class {name} {{ … }}")
            }
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
            // Error-like objects (own `name` + `message`) render `name: message`.
            Value::Object(o) => match (o.get_own("name"), o.get_own("message")) {
                (Some(name), Some(message)) => {
                    let m = message.to_js_string();
                    if m.is_empty() {
                        name.to_js_string()
                    } else {
                        alloc::format!("{}: {m}", name.to_js_string())
                    }
                }
                _ => "[object Object]".into(),
            },
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
        (Value::Class(x), Value::Class(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// SameValueZero — strict equality except that `NaN` equals `NaN` (the
/// comparison `Map`/`Set` and `Array.prototype.includes` use).
#[must_use]
pub fn same_value_zero<'a>(a: &Value<'a>, b: &Value<'a>) -> bool {
    if let (Value::Number(x), Value::Number(y)) = (a, b)
        && x.is_nan()
        && y.is_nan()
    {
        return true;
    }
    strict_equals(a, b)
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
