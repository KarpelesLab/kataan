//! Evaluating the real [`ast::Expr`](crate::ast::Expr) over the
//! [`Realm`]/[`NanBox`] model (`ROADMAP.md` §3 → Phase D migration).
//!
//! [`Realm`]: crate::realm::Realm
//! [`NanBox`]: crate::nanbox::NanBox
//!
//! Where [`nbvm`](crate::nbvm) executes a hand-written op stream, this walks the
//! **parser's actual AST** — the bridge from the existing front end to the new
//! representation. It covers the side-effect-free expression subset (literals,
//! every operator, logical short-circuit, the conditional, and array/object
//! literals + member access), evaluating each node straight onto the realm's
//! value operations: numbers/booleans are NaN-boxed immediates, strings/arrays/
//! objects are GC-managed heap cells.
//!
//! Variables, calls, and assignment are out of scope here — they need the
//! environment/closure machinery that lands with the structural VM migration;
//! this proves the value-and-operator layer evaluates real parsed expressions.
//!
//! Pure, safe `alloc`-only Rust.

use crate::ast::{ArrayElement, BinaryOp, Expr, LogicalOp, ObjectMember, PropertyKey, UnaryOp};
use crate::nanbox::NanBox;
use crate::realm::Realm;
use alloc::string::String;
use alloc::vec::Vec;

/// Why an expression could not be evaluated by this subset evaluator.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EvalError {
    /// The expression uses a construct outside the supported subset (a variable,
    /// call, assignment, …).
    Unsupported(&'static str),
}

/// Evaluates `expr` over `realm`, returning its value.
pub fn eval(realm: &mut Realm, expr: &Expr) -> Result<NanBox, EvalError> {
    match expr {
        Expr::Null(_) => Ok(NanBox::null()),
        Expr::Bool { value, .. } => Ok(NanBox::boolean(*value)),
        Expr::Number { value, .. } => Ok(NanBox::number(*value)),
        Expr::Str { value, .. } => {
            // The cooked value is WTF-8 bytes; preserve any lone surrogates.
            let h = realm.new_string_wtf8(value.to_vec());
            Ok(NanBox::handle(h.to_raw()))
        }

        Expr::Unary { op, argument, .. } => {
            let v = eval(realm, argument)?;
            Ok(match op {
                UnaryOp::Plus => NanBox::number(realm.to_number(v)),
                UnaryOp::Minus => realm.neg(v),
                UnaryOp::Not => realm.logical_not(v),
                UnaryOp::Typeof => {
                    let t = realm.type_of_value(v);
                    let h = realm.new_string(t);
                    NanBox::handle(h.to_raw())
                }
                UnaryOp::Void => NanBox::undefined(),
                #[cfg(feature = "std")]
                UnaryOp::BitNot => realm.bit_not(v),
                #[cfg(not(feature = "std"))]
                UnaryOp::BitNot => return Err(EvalError::Unsupported("~ needs std")),
                UnaryOp::Delete => return Err(EvalError::Unsupported("delete")),
            })
        }

        Expr::Logical {
            op, left, right, ..
        } => {
            let l = eval(realm, left)?;
            // Short-circuit on the left operand.
            match op {
                LogicalOp::And => {
                    if l.to_boolean() {
                        eval(realm, right)
                    } else {
                        Ok(l)
                    }
                }
                LogicalOp::Or => {
                    if l.to_boolean() {
                        Ok(l)
                    } else {
                        eval(realm, right)
                    }
                }
                LogicalOp::Nullish => {
                    if matches!(
                        l.unpack(),
                        crate::nanbox::Unpacked::Undefined | crate::nanbox::Unpacked::Null
                    ) {
                        eval(realm, right)
                    } else {
                        Ok(l)
                    }
                }
            }
        }

        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => {
            if eval(realm, test)?.to_boolean() {
                eval(realm, consequent)
            } else {
                eval(realm, alternate)
            }
        }

        Expr::Binary {
            op, left, right, ..
        } => {
            let a = eval(realm, left)?;
            let b = eval(realm, right)?;
            eval_binary(realm, *op, a, b)
        }

        Expr::Array { elements, .. } => {
            let mut items: Vec<NanBox> = Vec::new();
            for el in elements {
                match el {
                    ArrayElement::Hole => items.push(NanBox::undefined()),
                    ArrayElement::Item(e) => items.push(eval(realm, e)?),
                    ArrayElement::Spread(_) => {
                        return Err(EvalError::Unsupported("array spread"));
                    }
                }
            }
            let h = realm.new_array(items);
            Ok(NanBox::handle(h.to_raw()))
        }

        Expr::Object { members, .. } => {
            let handle = realm.new_object();
            for m in members {
                match m {
                    ObjectMember::Property { key, value, .. } => {
                        let k = static_key(key)?;
                        let v = eval(realm, value)?;
                        realm.set_property(handle, &k, v);
                    }
                    ObjectMember::Spread { .. } => {
                        return Err(EvalError::Unsupported("object spread"));
                    }
                    ObjectMember::Accessor { .. } => {
                        return Err(EvalError::Unsupported("object accessor"));
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
            let obj = eval(realm, object)?;
            if *optional
                && matches!(
                    obj.unpack(),
                    crate::nanbox::Unpacked::Undefined | crate::nanbox::Unpacked::Null
                )
            {
                return Ok(NanBox::undefined());
            }
            let Some(raw) = obj.as_handle() else {
                return Ok(NanBox::undefined());
            };
            let handle = crate::heap::Handle::from_raw(raw);
            // Resolve the key, treating a non-negative integer as an array index.
            match property {
                PropertyKey::Number(n) if as_index(*n).is_some() => {
                    Ok(realm.get_element(handle, as_index(*n).unwrap()))
                }
                PropertyKey::Computed(e) => {
                    let k = eval(realm, e)?;
                    if let Some(i) = k.as_number().and_then(as_index) {
                        return Ok(realm.get_element(handle, i));
                    }
                    let key = realm.to_display_string(k);
                    Ok(member_value(realm, handle, &key))
                }
                PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(member_value(realm, handle, s)),
                PropertyKey::Number(n) => Ok(member_value(realm, handle, &alloc::format!("{n}"))),
                PropertyKey::Private(_) => Err(EvalError::Unsupported("private member")),
            }
        }

        // `undefined`/`NaN`/`Infinity` are global identifiers, not literals.
        Expr::Ident(id) => match &*id.name {
            "undefined" => Ok(NanBox::undefined()),
            "NaN" => Ok(NanBox::number(f64::NAN)),
            "Infinity" => Ok(NanBox::number(f64::INFINITY)),
            _ => Err(EvalError::Unsupported("variable reference")),
        },
        // The optional-chain boundary; this evaluator already yields `undefined`
        // for member access on a nullish value, so the chain just evaluates.
        Expr::OptChain { expr, .. } => eval(realm, expr),
        _ => Err(EvalError::Unsupported("expression")),
    }
}

/// Applies a binary operator via the realm's value operations.
fn eval_binary(realm: &mut Realm, op: BinaryOp, a: NanBox, b: NanBox) -> Result<NanBox, EvalError> {
    Ok(match op {
        BinaryOp::Add => realm.add(a, b),
        BinaryOp::Sub => realm.sub(a, b),
        BinaryOp::Mul => realm.mul(a, b),
        BinaryOp::Div => realm.div(a, b),
        BinaryOp::Mod => realm.rem(a, b),
        BinaryOp::Lt => realm.less_than(a, b),
        BinaryOp::Gt => realm.greater_than(a, b),
        BinaryOp::LtEq => realm.less_equal(a, b),
        BinaryOp::GtEq => realm.greater_equal(a, b),
        BinaryOp::EqEq => NanBox::boolean(realm.loose_equals(a, b)),
        BinaryOp::NotEq => NanBox::boolean(!realm.loose_equals(a, b)),
        BinaryOp::EqEqEq => NanBox::boolean(realm.strict_equals(a, b)),
        BinaryOp::NotEqEq => NanBox::boolean(!realm.strict_equals(a, b)),
        #[cfg(feature = "std")]
        BinaryOp::Exp => realm.pow(a, b),
        #[cfg(feature = "std")]
        BinaryOp::Shl => realm.shl(a, b),
        #[cfg(feature = "std")]
        BinaryOp::Shr => realm.shr(a, b),
        #[cfg(feature = "std")]
        BinaryOp::Ushr => realm.ushr(a, b),
        #[cfg(feature = "std")]
        BinaryOp::BitAnd => realm.bit_and(a, b),
        #[cfg(feature = "std")]
        BinaryOp::BitOr => realm.bit_or(a, b),
        #[cfg(feature = "std")]
        BinaryOp::BitXor => realm.bit_xor(a, b),
        #[cfg(not(feature = "std"))]
        BinaryOp::Exp
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::Ushr
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor => {
            return Err(EvalError::Unsupported("** / bitwise need std"));
        }
        BinaryOp::In | BinaryOp::Instanceof => {
            return Err(EvalError::Unsupported("in / instanceof"));
        }
    })
}

/// A static (non-computed) property key as a string.
fn static_key(key: &PropertyKey) -> Result<String, EvalError> {
    match key {
        PropertyKey::Ident(s) | PropertyKey::Str(s) => Ok(String::from(&**s)),
        PropertyKey::Number(n) => Ok(alloc::format!("{n}")),
        PropertyKey::Computed(_) => Err(EvalError::Unsupported("computed key")),
        PropertyKey::Private(_) => Err(EvalError::Unsupported("private key")),
    }
}

/// A non-negative integer array index, if `n` is one (no `fract`, so the core
/// build is happy): truncate to `u64` and check it round-trips.
fn as_index(n: f64) -> Option<usize> {
    if n >= 0.0 && n <= u32::MAX as f64 && (n as u64) as f64 == n {
        Some(n as usize)
    } else {
        None
    }
}

/// Reads a named member: an object property, or the special `length` of an
/// array/string (those are heap cells, not property-bearing objects).
fn member_value(realm: &Realm, handle: crate::heap::Handle, key: &str) -> NanBox {
    if let Some(v) = realm.get_property(handle, key) {
        return v;
    }
    if key == "length" {
        if let Some(len) = realm.array_length(handle) {
            return NanBox::number(len as f64);
        }
        if let Some(s) = realm.string_value(handle) {
            return NanBox::number(s.chars().count() as f64);
        }
    }
    NanBox::undefined()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    /// Parses an expression, evaluates it, and renders the result as a string.
    fn run(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        // The program is one expression statement; pull its expression out.
        let expr = match &program.body[0] {
            crate::ast::Stmt::Expr { expression, .. } => expression,
            _ => panic!("expected an expression statement"),
        };
        let mut realm = Realm::new();
        let value = eval(&mut realm, expr).expect("eval");
        realm.to_display_string(value)
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(run("2 + 3 * 4"), "14");
        assert_eq!(run("(2 + 3) * 4"), "20");
        assert_eq!(run("10 - 2 - 3"), "5");
        assert_eq!(run("7 % 3"), "1");
        assert_eq!(run("-(3 + 4)"), "-7");
    }

    #[test]
    fn strings_and_concatenation() {
        assert_eq!(run("'foo' + 'bar'"), "foobar");
        assert_eq!(run("'n=' + (1 + 2)"), "n=3");
        assert_eq!(run("typeof 'x'"), "string");
        assert_eq!(run("typeof 42"), "number");
        assert_eq!(run("typeof true"), "boolean");
    }

    #[test]
    fn comparisons_and_equality() {
        assert_eq!(run("1 < 2"), "true");
        assert_eq!(run("2 <= 2"), "true");
        assert_eq!(run("3 > 5"), "false");
        assert_eq!(run("1 == '1'"), "true"); // loose
        assert_eq!(run("1 === '1'"), "false"); // strict
        assert_eq!(run("'a' === 'a'"), "true"); // strings by value
        assert_eq!(run("null == undefined"), "true");
    }

    #[test]
    fn logical_short_circuit_and_ternary() {
        assert_eq!(run("true && 'yes'"), "yes");
        assert_eq!(run("false && 'yes'"), "false");
        assert_eq!(run("0 || 'fallback'"), "fallback");
        assert_eq!(run("null ?? 'default'"), "default");
        assert_eq!(run("0 ?? 'default'"), "0"); // 0 is not nullish
        assert_eq!(run("1 < 2 ? 'a' : 'b'"), "a");
    }

    #[test]
    fn arrays_objects_and_member_access() {
        assert_eq!(run("[1, 2, 3]"), "1,2,3");
        assert_eq!(run("[1, 2, 3][1]"), "2");
        assert_eq!(run("[10, 20, 30].length"), "3");
        assert_eq!(run("({ x: 1, y: 2 }).y"), "2");
        assert_eq!(run("({ a: 1 + 2 }).a"), "3");
        assert_eq!(run("({ nested: { v: 9 } }).nested.v"), "9");
    }

    #[test]
    fn unsupported_constructs_report_cleanly() {
        let program = Parser::parse_program("x + 1").unwrap();
        let crate::ast::Stmt::Expr { expression, .. } = &program.body[0] else {
            panic!()
        };
        let mut realm = Realm::new();
        assert_eq!(
            eval(&mut realm, expression),
            Err(EvalError::Unsupported("variable reference"))
        );
    }
}
