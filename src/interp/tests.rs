//! Tree-walking interpreter tests.

use super::Interp;
use crate::parser::Parser;
use alloc::string::String;

/// Evaluates a program and returns its completion value as a JS string.
fn eval(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse ok");
    let mut interp = Interp::new();
    match interp.run(&program) {
        Ok(v) => v.to_js_string(),
        Err(e) => panic!("uncaught: {}", e.to_js_string()),
    }
}

/// Evaluates a program expected to throw; returns the thrown value's string.
fn eval_throw(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse ok");
    let mut interp = Interp::new();
    match interp.run(&program) {
        Ok(_) => panic!("expected a throw"),
        Err(e) => e.to_js_string(),
    }
}

#[test]
fn arithmetic_and_precedence() {
    assert_eq!(eval("1 + 2 * 3"), "7");
    assert_eq!(eval("(1 + 2) * 3"), "9");
    assert_eq!(eval("2 ** 3 ** 2"), "512");
    assert_eq!(eval("7 % 3"), "1");
    assert_eq!(eval("-5 % 3"), "-2");
    assert_eq!(eval("10 / 4"), "2.5");
}

#[test]
fn string_and_coercion() {
    assert_eq!(eval("'a' + 'b' + 'c'"), "abc");
    assert_eq!(eval("1 + '2'"), "12");
    assert_eq!(eval("'3' * 2"), "6");
    assert_eq!(eval("`sum=${1 + 2}`"), "sum=3");
    assert_eq!(eval("typeof 1"), "number");
    assert_eq!(eval("typeof 'x'"), "string");
    assert_eq!(eval("typeof undefinedVar"), "undefined");
}

#[test]
fn equality_and_logic() {
    assert_eq!(eval("1 == '1'"), "true");
    assert_eq!(eval("1 === '1'"), "false");
    assert_eq!(eval("null == undefined"), "true");
    assert_eq!(eval("0 / 0 === 0 / 0"), "false"); // NaN !== NaN
    assert_eq!(eval("true && 'yes'"), "yes");
    assert_eq!(eval("false || 'fallback'"), "fallback");
    assert_eq!(eval("null ?? 'default'"), "default");
    assert_eq!(eval("0 ?? 'default'"), "0");
}

#[test]
fn bitwise() {
    assert_eq!(eval("5 & 3"), "1");
    assert_eq!(eval("5 | 2"), "7");
    assert_eq!(eval("5 ^ 1"), "4");
    assert_eq!(eval("1 << 4"), "16");
    assert_eq!(eval("-1 >>> 28"), "15");
    assert_eq!(eval("~0"), "-1");
}

#[test]
fn variables_and_scoping() {
    assert_eq!(eval("let x = 10; x + 5"), "15");
    assert_eq!(eval("let x = 1; { let x = 2; } x"), "1");
    assert_eq!(eval("let x = 1; { x = 2; } x"), "2");
    assert_eq!(eval("var a = 1, b = 2; a + b"), "3");
    assert_eq!(
        eval_throw("const k = 1; k = 2;"),
        "assignment to constant variable k"
    );
}

#[test]
fn control_flow() {
    assert_eq!(
        eval("let s = 0; for (let i = 0; i < 5; i++) s += i; s"),
        "10"
    );
    assert_eq!(
        eval("let n = 0, i = 0; while (i < 10) { i++; if (i % 2) continue; n++; } n"),
        "5"
    );
    assert_eq!(
        eval("let x = 3; let r; if (x > 2) r = 'big'; else r = 'small'; r"),
        "big"
    );
    assert_eq!(
        eval("let out = 0; for (let i = 0; i < 100; i++) { if (i === 7) { out = i; break; } } out"),
        "7"
    );
    assert_eq!(
        eval(
            "let r; switch (2) { case 1: r = 'a'; break; case 2: r = 'b'; break; default: r = 'c'; } r"
        ),
        "b"
    );
}

#[test]
fn labeled_loops() {
    assert_eq!(
        eval(
            "let hits = 0;
             outer: for (let i = 0; i < 3; i++) {
               for (let j = 0; j < 3; j++) {
                 if (j === 1) continue outer;
                 hits++;
               }
             }
             hits"
        ),
        "3"
    );
    assert_eq!(
        eval(
            "let found = '';
             search: for (let i = 0; i < 3; i++)
               for (let j = 0; j < 3; j++)
                 if (i + j === 3) { found = `${i},${j}`; break search; }
             found"
        ),
        "1,2"
    );
}

#[test]
fn functions_and_closures() {
    assert_eq!(eval("function add(a, b) { return a + b; } add(2, 3)"), "5");
    assert_eq!(eval("const sq = x => x * x; sq(9)"), "81");
    assert_eq!(eval("(function (n) { return n + 1; })(41)"), "42");
    // Recursion (Fibonacci).
    assert_eq!(
        eval("function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); } fib(10)"),
        "55"
    );
    // Closures capture their environment.
    assert_eq!(
        eval(
            "function counter() { let c = 0; return () => ++c; } let n = counter(); n(); n(); n()"
        ),
        "3"
    );
    // Default parameters.
    assert_eq!(eval("function f(a, b = 10) { return a + b; } f(5)"), "15");
    // Mutual recursion via hoisting.
    assert_eq!(
        eval(
            "function ev(n){return n===0?true:od(n-1)} function od(n){return n===0?false:ev(n-1)} ev(10)"
        ),
        "true"
    );
}

#[test]
fn try_catch_finally() {
    assert_eq!(eval("try { throw 'boom'; } catch (e) { e; }"), "boom");
    assert_eq!(
        eval(
            "let log = ''; try { log += 'a'; throw 1; } catch { log += 'b'; } finally { log += 'c'; } log"
        ),
        "abc"
    );
    // `finally` runs even with a return in `try`.
    assert_eq!(
        eval("function f() { try { return 'x'; } finally { } } f()"),
        "x"
    );
    // An uncaught throw propagates.
    assert_eq!(eval_throw("throw 'uncaught';"), "uncaught");
}

#[test]
fn update_and_compound_assignment() {
    assert_eq!(eval("let x = 5; x++; x"), "6");
    assert_eq!(eval("let x = 5; let y = x++; `${x},${y}`"), "6,5");
    assert_eq!(eval("let x = 5; let y = ++x; `${x},${y}`"), "6,6");
    assert_eq!(eval("let x = 10; x -= 3; x *= 2; x"), "14");
    assert_eq!(eval("let x = 0; x ||= 5; x"), "5");
    assert_eq!(eval("let x = 1; x &&= 7; x"), "7");
    assert_eq!(eval("let x = null; x ??= 9; x"), "9");
}
