//! Expression-parser tests.
//!
//! Most tests render the parsed AST to a compact S-expression string and
//! compare against an expected shape — this keeps the assertions readable and
//! independent of span numbers.

use super::Parser;
use crate::ast::{
    Argument, ArrayElement, ArrayPatternElement, ArrowBody, BindingTarget, Class, ClassMember,
    Expr, ForInit, ForLeft, Function, MethodKind, ObjectMember, Param, Program, PropertyKey, Stmt,
    TemplateLiteral, VarDeclarator,
};
use alloc::string::String;
use alloc::vec::Vec;

/// Renders a binding target (identifier or destructuring pattern).
fn sexpr_target(t: &BindingTarget) -> String {
    use alloc::format;
    match t {
        BindingTarget::Ident(id) => id.name.clone().into_string(),
        BindingTarget::Array(p) => {
            let parts: Vec<String> = p
                .elements
                .iter()
                .map(|el| match el {
                    ArrayPatternElement::Hole => "hole".into(),
                    ArrayPatternElement::Rest { target, .. } => {
                        format!("...{}", sexpr_target(target))
                    }
                    ArrayPatternElement::Item {
                        target, default, ..
                    } => match default {
                        Some(d) => format!("{}={}", sexpr_target(target), sexpr(d)),
                        None => sexpr_target(target),
                    },
                })
                .collect();
            format!("[{}]", parts.join(" "))
        }
        BindingTarget::Object(p) => {
            let mut parts: Vec<String> = p
                .properties
                .iter()
                .map(|pr| {
                    let k = sexpr_key(&pr.key);
                    let base = if pr.shorthand {
                        k
                    } else {
                        format!("{k}:{}", sexpr_target(&pr.value))
                    };
                    match &pr.default {
                        Some(d) => format!("{base}={}", sexpr(d)),
                        None => base,
                    }
                })
                .collect();
            if let Some(r) = &p.rest {
                parts.push(format!("...{}", sexpr_target(r)));
            }
            format!("{{{}}}", parts.join(" "))
        }
    }
}

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
        Expr::Function(f) => sexpr_function(f),
        Expr::Arrow(a) => {
            let body = match &a.body {
                ArrowBody::Expr(e) => sexpr(e),
                ArrowBody::Block(b) => format!("(block {})", stmts(b)),
            };
            let kw = if a.is_async { "async-arrow" } else { "arrow" };
            format!("({kw} ({}) {body})", params(&a.params))
        }
        Expr::Class(c) => sexpr_class(c),
    }
}

fn sexpr_class(c: &Class) -> String {
    use alloc::format;
    let mut head = String::from("class");
    if let Some(id) = &c.id {
        head.push(' ');
        head.push_str(&id.name);
    }
    if let Some(s) = &c.super_class {
        head.push_str(&format!(" extends {}", sexpr(s)));
    }
    let members: Vec<String> = c
        .body
        .iter()
        .map(|m| match m {
            ClassMember::StaticBlock { body, .. } => format!("(static-block {})", stmts(body)),
            ClassMember::Field(f) => {
                let st = if f.is_static { "static-" } else { "" };
                let v = f
                    .value
                    .as_ref()
                    .map_or(String::new(), |e| format!(" {}", sexpr(e)));
                format!("({st}field {}{v})", sexpr_key(&f.key))
            }
            ClassMember::Method(m) => {
                let st = if m.is_static { "static-" } else { "" };
                let kind = match m.kind {
                    MethodKind::Method => "method",
                    MethodKind::Get => "get",
                    MethodKind::Set => "set",
                    MethodKind::Constructor => "ctor",
                };
                let mut flags = String::new();
                if m.value.is_async {
                    flags.push_str("async-");
                }
                if m.value.is_generator {
                    flags.push('*');
                }
                format!(
                    "({st}{flags}{kind} {} ({}) (block {}))",
                    sexpr_key(&m.key),
                    params(&m.value.params),
                    stmts(&m.value.body)
                )
            }
        })
        .collect();
    format!("({head} {})", members.join(" "))
}

fn sexpr_function(f: &Function) -> String {
    use alloc::format;
    let name = f.id.as_ref().map_or("", |id| &id.name);
    let mut tag = String::from("fn");
    if f.is_async {
        tag = "async-fn".into();
    }
    if f.is_generator {
        tag.push('*');
    }
    format!(
        "({tag} {name} ({}) (block {}))",
        params(&f.params),
        stmts(&f.body)
    )
}

fn params(ps: &[Param]) -> String {
    use alloc::format;
    let parts: Vec<String> = ps
        .iter()
        .map(|p| {
            let mut s = sexpr_target(&p.target);
            if p.rest {
                s = format!("...{s}");
            }
            if let Some(d) = &p.default {
                s = format!("{s}={}", sexpr(d));
            }
            s
        })
        .collect();
    parts.join(" ")
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
    assert!(perr("new.target").contains("new.target"));
    // Assignment-target destructuring (`[a,b] = c` as an expression) is not yet
    // reinterpreted from an array literal.
    assert!(perr("[a, b] = c").contains("invalid assignment target"));
    // Module syntax is still deferred (program context).
    assert!(sperr("import x from 'm';").contains("import/export"));
}

// === statements =========================================================

/// Parses a program and renders its statements (space-joined).
fn prog(src: &str) -> String {
    let program: Program = Parser::parse_program(src).expect("parse program ok");
    let parts: Vec<String> = program.body.iter().map(sexpr_stmt).collect();
    parts.join(" ")
}

/// Parses a program asserting failure; returns the error message.
fn sperr(src: &str) -> String {
    use alloc::string::ToString;
    Parser::parse_program(src)
        .expect_err("expected parse error")
        .to_string()
}

fn sexpr_stmt(s: &Stmt) -> String {
    use alloc::format;
    match s {
        Stmt::Expr { expression, .. } => format!("(expr {})", sexpr(expression)),
        Stmt::Empty { .. } => "(empty)".into(),
        Stmt::Debugger { .. } => "(debugger)".into(),
        Stmt::Block { body, .. } => format!("(block {})", stmts(body)),
        Stmt::Var(decl) => {
            let ds: Vec<String> = decl.declarations.iter().map(sexpr_declarator).collect();
            format!("({} {})", decl.kind.as_str(), ds.join(" "))
        }
        Stmt::If {
            test,
            consequent,
            alternate,
            ..
        } => match alternate {
            Some(a) => format!(
                "(if {} {} {})",
                sexpr(test),
                sexpr_stmt(consequent),
                sexpr_stmt(a)
            ),
            None => format!("(if {} {})", sexpr(test), sexpr_stmt(consequent)),
        },
        Stmt::While { test, body, .. } => format!("(while {} {})", sexpr(test), sexpr_stmt(body)),
        Stmt::DoWhile { body, test, .. } => {
            format!("(do {} {})", sexpr_stmt(body), sexpr(test))
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            let i = match init {
                None => "_".into(),
                Some(ForInit::Expr(e)) => sexpr(e),
                Some(ForInit::Var(d)) => {
                    let ds: Vec<String> = d.declarations.iter().map(sexpr_declarator).collect();
                    format!("({} {})", d.kind.as_str(), ds.join(" "))
                }
            };
            let t = test.as_ref().map_or("_".into(), |e| sexpr(e));
            let u = update.as_ref().map_or("_".into(), |e| sexpr(e));
            format!("(for {i} {t} {u} {})", sexpr_stmt(body))
        }
        Stmt::ForIn {
            left, right, body, ..
        } => format!(
            "(for-in {} {} {})",
            sexpr_for_left(left),
            sexpr(right),
            sexpr_stmt(body)
        ),
        Stmt::ForOf {
            left, right, body, ..
        } => format!(
            "(for-of {} {} {})",
            sexpr_for_left(left),
            sexpr(right),
            sexpr_stmt(body)
        ),
        Stmt::Switch {
            discriminant,
            cases,
            ..
        } => {
            let cs: Vec<String> = cases
                .iter()
                .map(|c| {
                    let label = c.test.as_ref().map_or("default".into(), sexpr);
                    format!("(case {label} {})", stmts(&c.body))
                })
                .collect();
            format!("(switch {} {})", sexpr(discriminant), cs.join(" "))
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
            ..
        } => {
            let h = handler.as_ref().map_or("_".into(), |c| {
                let p = c.param.as_ref().map_or("_".into(), sexpr_target);
                format!("(catch {p} {})", stmts(&c.body))
            });
            let f = finalizer
                .as_ref()
                .map_or("_".into(), |b| format!("(finally {})", stmts(b)));
            format!("(try ({}) {h} {f})", stmts(block))
        }
        Stmt::Return { argument, .. } => match argument {
            Some(a) => format!("(return {})", sexpr(a)),
            None => "(return)".into(),
        },
        Stmt::Break { label, .. } => match label {
            Some(l) => format!("(break {})", l.name),
            None => "(break)".into(),
        },
        Stmt::Continue { label, .. } => match label {
            Some(l) => format!("(continue {})", l.name),
            None => "(continue)".into(),
        },
        Stmt::Throw { argument, .. } => format!("(throw {})", sexpr(argument)),
        Stmt::Labeled { label, body, .. } => {
            format!("(label {} {})", label.name, sexpr_stmt(body))
        }
        Stmt::With { object, body, .. } => {
            format!("(with {} {})", sexpr(object), sexpr_stmt(body))
        }
        Stmt::Function(f) => sexpr_function(f),
        Stmt::Class(c) => sexpr_class(c),
    }
}

fn stmts(body: &[Stmt]) -> String {
    let parts: Vec<String> = body.iter().map(sexpr_stmt).collect();
    parts.join(" ")
}

fn sexpr_declarator(d: &VarDeclarator) -> String {
    use alloc::format;
    let target = sexpr_target(&d.target);
    match &d.init {
        Some(e) => format!("({target} {})", sexpr(e)),
        None => target,
    }
}

fn sexpr_for_left(left: &ForLeft) -> String {
    use alloc::format;
    match left {
        ForLeft::Decl { kind, target, .. } => {
            format!("({} {})", kind.as_str(), sexpr_target(target))
        }
        ForLeft::Target(e) => sexpr(e),
    }
}

#[test]
fn simple_statements() {
    assert_eq!(prog("x;"), "(expr x)");
    assert_eq!(prog(";"), "(empty)");
    assert_eq!(prog("debugger;"), "(debugger)");
    assert_eq!(prog("{ a; b; }"), "(block (expr a) (expr b))");
    assert_eq!(prog("a; b; c;"), "(expr a) (expr b) (expr c)");
}

#[test]
fn declarations() {
    assert_eq!(prog("let x = 1;"), "(let (x 1))");
    assert_eq!(prog("var a, b = 2, c;"), "(var a (b 2) c)");
    assert_eq!(prog("const k = 0;"), "(const (k 0))");
    assert!(sperr("const k;").contains("must be initialized"));
}

#[test]
fn if_else_and_dangling_else() {
    assert_eq!(prog("if (a) b;"), "(if a (expr b))");
    assert_eq!(prog("if (a) b; else c;"), "(if a (expr b) (expr c))");
    // The `else` binds to the inner `if`.
    assert_eq!(
        prog("if (a) if (b) c; else d;"),
        "(if a (if b (expr c) (expr d)))"
    );
}

#[test]
fn loops() {
    assert_eq!(prog("while (a) b;"), "(while a (expr b))");
    assert_eq!(prog("do a; while (b);"), "(do (expr a) b)");
    assert_eq!(prog("for (;;) a;"), "(for _ _ _ (expr a))");
    assert_eq!(
        prog("for (let i = 0; i < n; i++) f(i);"),
        "(for (let (i 0)) (< i n) (post++ i) (expr (call f i)))"
    );
    assert_eq!(
        prog("for (i = 0; i < 3; i += 1) {}"),
        "(for (= i 0) (< i 3) (+= i 1) (block ))"
    );
}

#[test]
fn for_in_and_of() {
    // Declaration head (the `Decl` arm renders `(kind name)`).
    assert_eq!(
        prog("for (const k in obj) f(k);"),
        "(for-in (const k) obj (expr (call f k)))"
    );
    assert_eq!(
        prog("for (let x of xs) g(x);"),
        "(for-of (let x) xs (expr (call g x)))"
    );
    // Bare assignment target (no declaration keyword) renders the expression.
    assert_eq!(prog("for (k in obj) ;"), "(for-in k obj (empty))");
}

#[test]
fn for_in_uses_no_in_rule() {
    // The `in` here introduces a for-in loop rather than a relational operator.
    assert_eq!(prog("for (a in b) ;"), "(for-in a b (empty))");
}

#[test]
fn switch_statement() {
    assert_eq!(
        prog("switch (x) { case 1: a; break; default: b; }"),
        "(switch x (case 1 (expr a) (break)) (case default (expr b)))"
    );
    assert!(sperr("switch (x) { default: ; default: ; }").contains("multiple `default`"));
}

#[test]
fn try_catch_finally() {
    assert_eq!(
        prog("try { a; } catch (e) { b; } finally { c; }"),
        "(try ((expr a)) (catch e (expr b)) (finally (expr c)))"
    );
    assert_eq!(
        prog("try { a; } catch { b; }"),
        "(try ((expr a)) (catch _ (expr b)) _)"
    );
    assert_eq!(
        prog("try { a; } finally { c; }"),
        "(try ((expr a)) _ (finally (expr c)))"
    );
    assert!(sperr("try { a; }").contains("catch"));
}

#[test]
fn jumps_and_labels() {
    assert_eq!(prog("return;"), "(return)");
    assert_eq!(prog("return a + b;"), "(return (+ a b))");
    assert_eq!(prog("throw e;"), "(throw e)");
    assert_eq!(
        prog("loop: for (;;) break loop;"),
        "(label loop (for _ _ _ (break loop)))"
    );
    assert_eq!(
        prog("outer: while (a) continue outer;"),
        "(label outer (while a (continue outer)))"
    );
    // `{ a: 1 }` at statement position is a block with a labeled statement.
    assert_eq!(prog("{ a: 1 }"), "(block (label a (expr 1)))");
}

#[test]
fn asi_inserts_semicolons() {
    // Newline-separated statements need no explicit `;`.
    assert_eq!(prog("a\nb\nc"), "(expr a) (expr b) (expr c)");
    // `return` with a newline before its operand returns undefined (restricted
    // production): two statements, not `return a`.
    assert_eq!(prog("return\na"), "(return) (expr a)");
    // A newline after `throw` is an error.
    assert!(sperr("throw\ne").contains("newline after `throw`"));
    // Missing separator without a newline is an error.
    assert!(sperr("a b").contains("semicolon"));
}

// === functions & arrows =================================================

#[test]
fn function_declarations() {
    assert_eq!(prog("function f() {}"), "(fn f () (block ))");
    assert_eq!(
        prog("function add(a, b) { return a + b; }"),
        "(fn add (a b) (block (return (+ a b))))"
    );
    assert_eq!(prog("function* g() {}"), "(fn* g () (block ))");
    assert_eq!(prog("async function h() {}"), "(async-fn h () (block ))");
    assert!(sperr("function () {}").contains("requires a name"));
}

#[test]
fn parameters() {
    assert_eq!(
        prog("function f(a, b = 1, ...rest) {}"),
        "(fn f (a b=1 ...rest) (block ))"
    );
    // `async\nfunction` is `async;` then a function declaration (ASI).
    assert_eq!(
        prog("async\nfunction f() {}"),
        "(expr async) (fn f () (block ))"
    );
}

#[test]
fn function_expressions() {
    assert_eq!(
        prog("let f = function () {};"),
        "(let (f (fn  () (block ))))"
    );
    assert_eq!(
        prog("let f = function named(x) { return x; };"),
        "(let (f (fn named (x) (block (return x)))))"
    );
    assert_eq!(
        prog("(async function () {})"),
        "(expr (async-fn  () (block )))"
    );
}

#[test]
fn arrow_functions() {
    assert_eq!(sx("x => x + 1"), "(arrow (x) (+ x 1))");
    assert_eq!(sx("() => 1"), "(arrow () 1)");
    assert_eq!(sx("(a, b) => a * b"), "(arrow (a b) (* a b))");
    assert_eq!(sx("(a, b = 2) => a"), "(arrow (a b=2) a)");
    assert_eq!(sx("(...xs) => xs"), "(arrow (...xs) xs)");
    assert_eq!(sx("x => { return x; }"), "(arrow (x) (block (return x)))");
    assert_eq!(sx("async x => x"), "(async-arrow (x) x)");
    assert_eq!(sx("async (a) => a"), "(async-arrow (a) a)");
}

#[test]
fn arrow_cover_grammar() {
    // Parenthesized expression vs. arrow params — only `=>` makes it an arrow.
    assert_eq!(sx("(a + b) * c"), "(* (+ a b) c)");
    assert_eq!(sx("(a)"), "a");
    // Nested / right-associative arrows.
    assert_eq!(sx("a => b => a + b"), "(arrow (a) (arrow (b) (+ a b)))");
    // Arrow as a call argument.
    assert_eq!(sx("f(x => x, 1)"), "(call f (arrow (x) x) 1)");
    // Arrow as a conditional branch.
    assert_eq!(
        sx("c ? x => x : y => y"),
        "(?: c (arrow (x) x) (arrow (y) y))"
    );
}

#[test]
fn iife() {
    assert_eq!(
        prog("(function () { return 1; })();"),
        "(expr (call (fn  () (block (return 1)))))"
    );
    assert_eq!(prog("(() => 1)();"), "(expr (call (arrow () 1)))");
}

// === destructuring patterns ============================================

#[test]
fn array_destructuring() {
    assert_eq!(prog("let [a, b] = xs;"), "(let ([a b] xs))");
    assert_eq!(prog("let [a, , c] = xs;"), "(let ([a hole c] xs))");
    assert_eq!(
        prog("let [a = 1, ...rest] = xs;"),
        "(let ([a=1 ...rest] xs))"
    );
    assert_eq!(prog("const [[a], [b]] = m;"), "(const ([[a] [b]] m))");
}

#[test]
fn object_destructuring() {
    assert_eq!(prog("let {a, b} = o;"), "(let ({a b} o))");
    assert_eq!(prog("let {a: x, b: y} = o;"), "(let ({a:x b:y} o))");
    assert_eq!(prog("let {a = 1} = o;"), "(let ({a=1} o))");
    assert_eq!(prog("let {a, ...rest} = o;"), "(let ({a ...rest} o))");
    assert_eq!(prog("let {[k]: v} = o;"), "(let ({[k]:v} o))");
    // Nested object/array patterns.
    assert_eq!(prog("let {a: [b, c]} = o;"), "(let ({a:[b c]} o))");
}

#[test]
fn destructuring_in_params_and_catch() {
    assert_eq!(
        prog("function f({a, b}, [c]) {}"),
        "(fn f ({a b} [c]) (block ))"
    );
    assert_eq!(sx("({a, b}) => a + b"), "(arrow ({a b}) (+ a b))");
    assert_eq!(
        prog("try {} catch ({message}) {}"),
        "(try () (catch {message} ) _)"
    );
    assert_eq!(
        prog("for (const [k, v] of m) ;"),
        "(for-of (const [k v]) m (empty))"
    );
}

// === classes ============================================================

#[test]
fn empty_and_basic_class() {
    assert_eq!(prog("class C {}"), "(class C )");
    assert_eq!(prog("class C extends B {}"), "(class C extends B )");
    assert!(sperr("class {}").contains("requires a name"));
    assert_eq!(sx("(class {})"), "(class )");
}

#[test]
fn class_methods() {
    assert_eq!(
        prog("class C { m(a) { return a; } }"),
        "(class C (method m (a) (block (return a))))"
    );
    assert_eq!(
        prog("class C { constructor(x) { this.x = x; } }"),
        "(class C (ctor constructor (x) (block (expr (= (member . this x) x)))))"
    );
    assert_eq!(
        prog("class C { static s() {} m() {} }"),
        "(class C (static-method s () (block )) (method m () (block )))"
    );
    assert_eq!(
        prog("class C { *gen() {} async a() {} }"),
        "(class C (*method gen () (block )) (async-method a () (block )))"
    );
}

#[test]
fn class_accessors() {
    assert_eq!(
        prog("class C { get x() { return 1; } set x(v) {} }"),
        "(class C (get x () (block (return 1))) (set x (v) (block )))"
    );
    // `get`/`set`/`static`/`async` are usable as plain method names.
    assert_eq!(
        prog("class C { get() {} static() {} }"),
        "(class C (method get () (block )) (method static () (block )))"
    );
}

#[test]
fn class_fields_and_private() {
    assert_eq!(
        prog("class C { x = 1; y; static z = 2; }"),
        "(class C (field x 1) (field y) (static-field z 2))"
    );
    assert_eq!(
        prog("class C { #count = 0; #inc() { this.#count++; } }"),
        "(class C (field #count 0) (method #inc () (block (expr (post++ (member . this #count))))))"
    );
    assert_eq!(
        prog("class C { static { init(); } }"),
        "(class C (static-block (expr (call init))))"
    );
    // Computed keys.
    assert_eq!(
        prog("class C { [k]() {} }"),
        "(class C (method [k] () (block )))"
    );
}
