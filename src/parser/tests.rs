//! Expression-parser tests.
//!
//! Most tests render the parsed AST to a compact S-expression string and
//! compare against an expected shape — this keeps the assertions readable and
//! independent of span numbers.

use super::Parser;
use crate::ast::{Argument, ArrayElement, Expr, ObjectMember, PropertyKey, TemplateLiteral};
use alloc::string::String;
use alloc::vec::Vec;

/// Parses a single expression (asserting the whole input is consumed).
fn parse(src: &str) -> Expr {
    Parser::parse_expression_entry(src).expect("parse ok")
}

/// Parses and renders to an S-expression.
fn sx(src: &str) -> String {
    sexpr(&parse(src))
}

/// Parses, asserting failure, and returns the error message.
fn perr(src: &str) -> String {
    use alloc::string::ToString;
    Parser::parse_expression_entry(src)
        .expect_err("expected parse error")
        .to_string()
}

/// Renders an expression as an S-expression for terse structural assertions.
fn sexpr(e: &Expr) -> String {
    use alloc::format;
    match e {
        Expr::Null(_) => "null".into(),
        Expr::Bool { value, .. } => format!("{value}"),
        Expr::Number { value, .. } => format!("{value}"),
        Expr::BigInt { digits, .. } => format!("{digits}n"),
        Expr::Str { value, .. } => format!("{value:?}"),
        Expr::Regex { pattern, flags, .. } => format!("/{pattern}/{flags}"),
        Expr::Ident(id) => id.name.clone().into_string(),
        Expr::This(_) => "this".into(),
        Expr::Super(_) => "super".into(),
        Expr::Template(t) => sexpr_template(t),
        Expr::TaggedTemplate { tag, quasi, .. } => {
            format!("(tagged {} {})", sexpr(tag), sexpr_template(quasi))
        }
        Expr::Array { elements, .. } => {
            let mut parts = Vec::new();
            for el in elements {
                parts.push(match el {
                    ArrayElement::Hole => "hole".into(),
                    ArrayElement::Item(e) => sexpr(e),
                    ArrayElement::Spread(e) => format!("(... {})", sexpr(e)),
                });
            }
            format!("(array {})", parts.join(" "))
        }
        Expr::Object { members, .. } => {
            let mut parts = Vec::new();
            for m in members {
                parts.push(match m {
                    ObjectMember::Spread { value, .. } => format!("(... {})", sexpr(value)),
                    ObjectMember::Property {
                        key,
                        value,
                        shorthand,
                        ..
                    } => {
                        let k = sexpr_key(key);
                        if *shorthand {
                            format!("(short {k})")
                        } else {
                            format!("({k} {})", sexpr(value))
                        }
                    }
                });
            }
            format!("(object {})", parts.join(" "))
        }
        Expr::Member {
            object,
            property,
            optional,
            ..
        } => {
            let dot = if *optional { "?." } else { "." };
            format!("(member {dot} {} {})", sexpr(object), sexpr_key(property))
        }
        Expr::Call {
            callee,
            arguments,
            optional,
            ..
        } => {
            let q = if *optional { "?call" } else { "call" };
            format!("({q} {}{})", sexpr(callee), sexpr_args(arguments))
        }
        Expr::New {
            callee, arguments, ..
        } => format!("(new {}{})", sexpr(callee), sexpr_args(arguments)),
        Expr::Unary { op, argument, .. } => format!("({} {})", op.as_str(), sexpr(argument)),
        Expr::Update {
            op,
            prefix,
            argument,
            ..
        } => {
            let pos = if *prefix { "pre" } else { "post" };
            format!("({pos}{} {})", op.as_str(), sexpr(argument))
        }
        Expr::Binary {
            op, left, right, ..
        } => format!("({} {} {})", op.as_str(), sexpr(left), sexpr(right)),
        Expr::Logical {
            op, left, right, ..
        } => format!("({} {} {})", op.as_str(), sexpr(left), sexpr(right)),
        Expr::Conditional {
            test,
            consequent,
            alternate,
            ..
        } => format!(
            "(?: {} {} {})",
            sexpr(test),
            sexpr(consequent),
            sexpr(alternate)
        ),
        Expr::Assign {
            op, target, value, ..
        } => format!("({} {} {})", op.as_str(), sexpr(target), sexpr(value)),
        Expr::Sequence { expressions, .. } => {
            let parts: Vec<String> = expressions.iter().map(sexpr).collect();
            format!("(seq {})", parts.join(" "))
        }
    }
}

fn sexpr_key(key: &PropertyKey) -> String {
    use alloc::format;
    match key {
        PropertyKey::Ident(n) => n.clone().into_string(),
        PropertyKey::Private(n) => format!("#{n}"),
        PropertyKey::Str(s) => format!("{s:?}"),
        PropertyKey::Number(n) => format!("{n}"),
        PropertyKey::Computed(e) => format!("[{}]", sexpr(e)),
    }
}

fn sexpr_args(args: &[Argument]) -> String {
    use alloc::format;
    let mut s = String::new();
    for a in args {
        s.push(' ');
        match a {
            Argument::Item(e) => s.push_str(&sexpr(e)),
            Argument::Spread(e) => s.push_str(&format!("(... {})", sexpr(e))),
        }
    }
    s
}

fn sexpr_template(t: &TemplateLiteral) -> String {
    use alloc::format;
    let mut parts = Vec::new();
    for (i, q) in t.quasis.iter().enumerate() {
        let cooked = q.cooked.as_deref().unwrap_or("<bad>");
        parts.push(format!("{cooked:?}"));
        if let Some(e) = t.expressions.get(i) {
            parts.push(format!("${{{}}}", sexpr(e)));
        }
    }
    format!("(tmpl {})", parts.join(" "))
}

// --- literals -----------------------------------------------------------

#[test]
fn literals() {
    assert_eq!(sx("null"), "null");
    assert_eq!(sx("true"), "true");
    assert_eq!(sx("false"), "false");
    assert_eq!(sx("this"), "this");
    assert_eq!(sx("42"), "42");
    assert_eq!(sx("2.5"), "2.5");
    assert_eq!(sx("0xFF"), "255");
    assert_eq!(sx("123n"), "123n");
    assert_eq!(sx(r#""hi\n""#), r#""hi\n""#);
    assert_eq!(sx("foo"), "foo");
    assert_eq!(sx("/ab+c/gi"), "/ab+c/gi");
}

// --- precedence & associativity ----------------------------------------

#[test]
fn arithmetic_precedence() {
    assert_eq!(sx("1 + 2 * 3"), "(+ 1 (* 2 3))");
    assert_eq!(sx("1 * 2 + 3"), "(+ (* 1 2) 3)");
    assert_eq!(sx("(1 + 2) * 3"), "(* (+ 1 2) 3)");
}

#[test]
fn left_associative_subtraction() {
    assert_eq!(sx("1 - 2 - 3"), "(- (- 1 2) 3)");
}

#[test]
fn exponent_right_associative() {
    assert_eq!(sx("2 ** 3 ** 2"), "(** 2 (** 3 2))");
    assert_eq!(sx("(-2) ** 2"), "(** (- 2) 2)");
    assert_eq!(sx("++x ** 2"), "(** (pre++ x) 2)");
}

#[test]
fn exponent_unary_operand_is_error() {
    assert!(perr("-2 ** 2").contains("parenthesized"));
    assert!(perr("typeof x ** 2").contains("parenthesized"));
}

#[test]
fn comparison_and_logical() {
    assert_eq!(sx("a < b == c"), "(== (< a b) c)");
    assert_eq!(sx("a || b && c"), "(|| a (&& b c))");
    assert_eq!(sx("a && b || c"), "(|| (&& a b) c)");
    assert_eq!(sx("a | b & c ^ d"), "(| a (^ (& b c) d))");
}

#[test]
fn relational_keywords() {
    assert_eq!(sx("x in y"), "(in x y)");
    assert_eq!(sx("a instanceof B"), "(instanceof a B)");
}

#[test]
fn nullish_mixing_requires_parens() {
    assert!(perr("a ?? b || c").contains("??"));
    assert!(perr("a || b ?? c").contains("??"));
    assert!(perr("a ?? b && c").contains("??"));
    // Parenthesized is fine.
    assert_eq!(sx("(a ?? b) || c"), "(|| (?? a b) c)");
    assert_eq!(sx("a ?? b ?? c"), "(?? (?? a b) c)");
}

#[test]
fn conditional_and_assignment() {
    assert_eq!(sx("a ? b : c ? d : e"), "(?: a b (?: c d e))");
    assert_eq!(sx("a = b = c"), "(= a (= b c))");
    assert_eq!(sx("x += 1"), "(+= x 1)");
    assert_eq!(sx("a.b ??= c"), "(??= (member . a b) c)");
}

#[test]
fn invalid_assignment_target() {
    assert!(perr("1 = 2").contains("invalid assignment target"));
    assert!(perr("a + b = c").contains("invalid assignment target"));
}

#[test]
fn sequence() {
    assert_eq!(sx("a, b, c"), "(seq a b c)");
}

// --- unary & update -----------------------------------------------------

#[test]
fn unary_and_update() {
    assert_eq!(sx("!x"), "(! x)");
    assert_eq!(sx("-+x"), "(- (+ x))");
    assert_eq!(sx("typeof x"), "(typeof x)");
    assert_eq!(sx("delete a.b"), "(delete (member . a b))");
    assert_eq!(sx("++x"), "(pre++ x)");
    assert_eq!(sx("x++"), "(post++ x)");
    assert_eq!(sx("x--"), "(post-- x)");
}

#[test]
fn postfix_blocked_by_newline() {
    // ASI: a newline before `++` means it is NOT a postfix operator, so `++b`
    // begins a fresh token the single-expression entry treats as trailing.
    assert!(perr("a\n++b").contains("trailing"));
}

// --- member / call / new ------------------------------------------------

#[test]
fn member_and_call() {
    assert_eq!(sx("a.b.c"), "(member . (member . a b) c)");
    assert_eq!(sx("a[b]"), "(member . a [b])");
    assert_eq!(sx("f(1, 2)"), "(call f 1 2)");
    assert_eq!(sx("a.b(c)"), "(call (member . a b) c)");
    assert_eq!(sx("f(...xs)"), "(call f (... xs))");
    assert_eq!(sx("obj.#priv"), "(member . obj #priv)");
    // Reserved word as a property name.
    assert_eq!(sx("a.class"), "(member . a class)");
}

#[test]
fn new_expression() {
    assert_eq!(sx("new X()"), "(new X)");
    assert_eq!(sx("new X"), "(new X)");
    assert_eq!(sx("new X(1, 2)"), "(new X 1 2)");
    assert_eq!(sx("new a.b.C()"), "(new (member . (member . a b) C))");
    // Call binds outside the `new`: new X().y
    assert_eq!(sx("new X().y"), "(member . (new X) y)");
    assert_eq!(sx("new new X()()"), "(new (new X))");
}

#[test]
fn optional_chaining() {
    assert_eq!(sx("a?.b"), "(member ?. a b)");
    assert_eq!(sx("a?.[b]"), "(member ?. a [b])");
    assert_eq!(sx("a?.(x)"), "(?call a x)");
    assert_eq!(sx("a?.b.c"), "(member . (member ?. a b) c)");
}

#[test]
fn tagged_template() {
    assert_eq!(sx("tag`hi`"), r#"(tagged tag (tmpl "hi"))"#);
}

// --- arrays / objects / templates --------------------------------------

#[test]
fn array_literals() {
    assert_eq!(sx("[1, 2, 3]"), "(array 1 2 3)");
    assert_eq!(sx("[1, , 3]"), "(array 1 hole 3)");
    assert_eq!(sx("[1, ...xs]"), "(array 1 (... xs))");
    assert_eq!(sx("[,]"), "(array hole)");
    assert_eq!(sx("[1,]"), "(array 1)");
}

#[test]
fn object_literals() {
    assert_eq!(sx("{a: 1, b: 2}"), "(object (a 1) (b 2))");
    assert_eq!(sx("{x}"), "(object (short x))");
    assert_eq!(sx("{[k]: v}"), "(object ([k] v))");
    assert_eq!(sx(r#"{"a-b": 1}"#), r#"(object ("a-b" 1))"#);
    assert_eq!(sx("{0: 1}"), "(object (0 1))");
    assert_eq!(sx("{...rest}"), "(object (... rest))");
    // Reserved word allowed as a `key: value` key but not as shorthand.
    assert_eq!(sx("{if: 1}"), "(object (if 1))");
    assert!(perr("{if}").contains("shorthand"));
}

#[test]
fn templates() {
    assert_eq!(sx("`hello`"), r#"(tmpl "hello")"#);
    assert_eq!(sx("`a${x}b`"), r#"(tmpl "a" ${x} "b")"#);
    assert_eq!(sx("`${a + b}`"), r#"(tmpl "" ${(+ a b)} "")"#);
    assert_eq!(sx("`${`${x}`}`"), r#"(tmpl "" ${(tmpl "" ${x} "")} "")"#);
}

// --- deferred features give clear errors -------------------------------

#[test]
fn deferred_features() {
    assert!(perr("function(){}").contains("later increment"));
    assert!(perr("() => 1").contains("empty parentheses"));
    assert!(perr("new.target").contains("new.target"));
}
