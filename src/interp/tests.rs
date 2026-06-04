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
        "TypeError: assignment to constant variable k"
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
fn objects() {
    assert_eq!(eval("let o = { a: 1, b: 2 }; o.a + o.b"), "3");
    assert_eq!(eval("let o = {}; o.x = 5; o['y'] = 7; o.x + o.y"), "12");
    assert_eq!(eval("let k = 'key'; let o = { [k]: 42 }; o.key"), "42");
    assert_eq!(
        eval("let o = { a: 1 }; let p = { ...o, b: 2 }; p.a + p.b"),
        "3"
    );
    assert_eq!(eval("let o = { n: 0 }; o.n++; o.n += 10; o.n"), "11");
    assert_eq!(
        eval_throw("let o = null; o.x"),
        "TypeError: cannot read properties of null (reading 'x')"
    );
}

#[test]
fn arrays() {
    assert_eq!(eval("let a = [1, 2, 3]; a.length"), "3");
    assert_eq!(eval("let a = [10, 20, 30]; a[1]"), "20");
    assert_eq!(eval("let a = [1, 2]; a[5] = 9; a.length"), "6");
    assert_eq!(eval("let a = [1, ...[2, 3], 4]; a.length"), "4");
    assert_eq!(eval("[1, 2, 3] + ''"), "1,2,3");
    assert_eq!(
        eval("let a = []; a[0] = 'x'; a[1] = 'y'; a[0] + a[1]"),
        "xy"
    );
}

#[test]
fn methods_and_this() {
    assert_eq!(
        eval("let o = { x: 10, get() { return this.x; } }; o.get()"),
        "10"
    );
    assert_eq!(
        eval("let o = { v: 3, add(n) { return this.v + n; } }; o.add(4)"),
        "7"
    );
    // An arrow inside a method captures the method's `this` lexically.
    assert_eq!(
        eval("let o = { v: 5, run() { let f = () => this.v; return f(); } }; o.run()"),
        "5"
    );
}

#[test]
fn destructuring_runtime() {
    assert_eq!(eval("let [a, b] = [1, 2]; a + b"), "3");
    assert_eq!(eval("let [a, , c] = [1, 2, 3]; a + c"), "4");
    assert_eq!(
        eval("let [first, ...rest] = [1, 2, 3, 4]; rest.length"),
        "3"
    );
    assert_eq!(eval("let { x, y } = { x: 7, y: 8 }; x * y"), "56");
    assert_eq!(eval("let { a: p, b = 100 } = { a: 1 }; p + b"), "101");
    assert_eq!(eval("function f([a, b]) { return a - b; } f([9, 4])"), "5");
    assert_eq!(
        eval("function g({ n }) { return n * 2; } g({ n: 21 })"),
        "42"
    );
}

#[test]
fn for_of_and_for_in() {
    assert_eq!(
        eval("let s = 0; for (const x of [1, 2, 3, 4]) s += x; s"),
        "10"
    );
    assert_eq!(
        eval("let out = ''; for (const c of 'abc') out += c + '.'; out"),
        "a.b.c."
    );
    assert_eq!(
        eval("let o = { a: 1, b: 2, c: 3 }; let keys = ''; for (const k in o) keys += k; keys"),
        "abc"
    );
}

#[test]
fn classes_basic() {
    assert_eq!(
        eval(
            "class Point { constructor(x, y) { this.x = x; this.y = y; } sum() { return this.x + this.y; } } new Point(3, 4).sum()"
        ),
        "7"
    );
    // Instance fields.
    assert_eq!(
        eval(
            "class C { n = 10; bump() { this.n++; return this.n; } } let c = new C(); c.bump(); c.bump()"
        ),
        "12"
    );
    // Static members.
    assert_eq!(
        eval(
            "class M { static origin() { return 0; } static label = 'M'; } M.origin() + ' ' + M.label"
        ),
        "0 M"
    );
    // typeof + instanceof.
    assert_eq!(eval("class A {} typeof A"), "function");
    assert_eq!(eval("class A {} new A() instanceof A"), "true");
    assert_eq!(eval("class A {} class B {} new A() instanceof B"), "false");
}

#[test]
fn classes_inheritance() {
    assert_eq!(
        eval(
            "class Animal {
               constructor(name) { this.name = name; }
               speak() { return this.name + ' makes a sound'; }
             }
             class Dog extends Animal {
               constructor(name) { super(name); }
               speak() { return this.name + ' barks'; }
             }
             new Dog('Rex').speak()"
        ),
        "Rex barks"
    );
    // super.method() calls the parent implementation.
    assert_eq!(
        eval(
            "class Base { greet() { return 'hello'; } }
             class Sub extends Base { greet() { return super.greet() + ' world'; } }
             new Sub().greet()"
        ),
        "hello world"
    );
    // instanceof walks the chain.
    assert_eq!(
        eval("class Base {} class Sub extends Base {} new Sub() instanceof Base"),
        "true"
    );
    // Inherited methods.
    assert_eq!(
        eval("class A { m() { return 42; } } class B extends A {} new B().m()"),
        "42"
    );
}

#[test]
fn error_objects() {
    assert_eq!(eval("let e = new Error('boom'); e.message"), "boom");
    assert_eq!(eval("new TypeError('bad').name"), "TypeError");
    assert_eq!(eval("String(new RangeError('out'))"), "RangeError: out");
    assert_eq!(
        eval("try { throw new Error('caught'); } catch (e) { e.message; }"),
        "caught"
    );
    assert_eq!(eval("`${new Error('x')}`"), "Error: x");
    // Runtime errors are catchable objects with a name and message.
    assert_eq!(eval("try { null.x; } catch (e) { e.name; }"), "TypeError");
    assert_eq!(
        eval("try { missingFn(); } catch (e) { e.message; }"),
        "missingFn is not defined"
    );
    assert_eq!(
        eval("try { let n = 1; n(); } catch (e) { e.name; }"),
        "TypeError"
    );
}

#[test]
fn more_array_methods() {
    assert_eq!(eval("[3, 1, 2].sort().join(',')"), "1,2,3");
    assert_eq!(eval("[3, 1, 2].sort((a, b) => b - a).join(',')"), "3,2,1");
    assert_eq!(eval("[10, 1, 2].sort((a, b) => a - b).join(',')"), "1,2,10");
    assert_eq!(eval("[1, 2, 3].reverse().join(',')"), "3,2,1");
    assert_eq!(eval("[1, 2].concat([3, 4], 5).join(',')"), "1,2,3,4,5");
    assert_eq!(eval("[[1, 2], [3], 4].flat().join(',')"), "1,2,3,4");
    assert_eq!(eval("[1, 2, 3].at(-1)"), "3");
    assert_eq!(eval("[5, 6, 7].findIndex(x => x === 6)"), "1");
    assert_eq!(eval("let a = [1, 2]; a.unshift(0); a.join(',')"), "0,1,2");
}

#[test]
fn more_string_number_methods() {
    assert_eq!(eval("(3.14159).toFixed(2)"), "3.14");
    assert_eq!(eval("(255).toString(16)"), "ff");
    assert_eq!(eval("(5).toString(2)"), "101");
    assert_eq!(eval("'5'.padStart(3, '0')"), "005");
    assert_eq!(eval("'5'.padEnd(3, '.')"), "5..");
    assert_eq!(eval("'a-b-c'.replace('-', '+')"), "a+b-c");
    assert_eq!(eval("'a-b-c'.replaceAll('-', '+')"), "a+b+c");
    assert_eq!(eval("'hello'.at(-1)"), "o");
}

#[test]
fn maps() {
    assert_eq!(
        eval("let m = new Map(); m.set('a', 1); m.set('b', 2); m.get('a') + m.get('b')"),
        "3"
    );
    assert_eq!(eval("let m = new Map(); m.set('x', 9); m.has('x')"), "true");
    assert_eq!(
        eval("let m = new Map(); m.set('x', 1); m.delete('x'); m.size"),
        "0"
    );
    assert_eq!(eval("new Map([['a', 1], ['b', 2]]).size"), "2");
    assert_eq!(
        eval("let m = new Map([['a', 1], ['b', 2]]); let s = 0; m.forEach(v => s += v); s"),
        "3"
    );
    // NaN keys are SameValueZero-equal.
    assert_eq!(
        eval("let m = new Map(); m.set(0 / 0, 'x'); m.get(0 / 0)"),
        "x"
    );
}

#[test]
fn sets() {
    assert_eq!(
        eval("let s = new Set(); s.add(1); s.add(1); s.add(2); s.size"),
        "2"
    );
    assert_eq!(eval("new Set([1, 2, 2, 3, 3, 3]).size"), "3");
    assert_eq!(eval("let s = new Set([1, 2]); s.has(2)"), "true");
    assert_eq!(
        eval("let s = new Set([1, 2, 3]); s.delete(2); s.has(2)"),
        "false"
    );
    assert_eq!(
        eval("let s = new Set([1, 2, 3]); let t = 0; s.forEach(v => t += v); t"),
        "6"
    );
    // Dedup an array via a Set.
    assert_eq!(
        eval("let s = new Set([3, 1, 3, 2, 1]); s.values().join(',')"),
        "3,1,2"
    );
}

#[test]
fn getters_and_setters() {
    // Object-literal accessors.
    assert_eq!(
        eval(
            "let o = { _n: 1, get n() { return this._n; }, set n(v) { this._n = v * 2; } }; o.n = 5; o.n"
        ),
        "10"
    );
    assert_eq!(
        eval(
            "let o = { first: 'Ada', last: 'L', get full() { return this.first + ' ' + this.last; } }; o.full"
        ),
        "Ada L"
    );
    // Class accessors (on the prototype, inherited by instances).
    assert_eq!(
        eval(
            "class C { constructor() { this._x = 0; } get x() { return this._x; } set x(v) { this._x = v + 1; } } let c = new C(); c.x = 9; c.x"
        ),
        "10"
    );
    // Inherited accessor through extends.
    assert_eq!(
        eval("class A { get v() { return 42; } } class B extends A {} new B().v"),
        "42"
    );
}

#[test]
fn in_operator() {
    assert_eq!(eval("'a' in { a: 1 }"), "true");
    assert_eq!(eval("'b' in { a: 1 }"), "false");
    assert_eq!(eval("class A { m() {} } 'm' in new A()"), "true");
}

#[test]
fn math_builtins() {
    assert_eq!(eval("Math.abs(-5)"), "5");
    assert_eq!(eval("Math.floor(3.7)"), "3");
    assert_eq!(eval("Math.ceil(3.2)"), "4");
    assert_eq!(eval("Math.max(1, 9, 4, 2)"), "9");
    assert_eq!(eval("Math.min(3, -2, 8)"), "-2");
    assert_eq!(eval("Math.pow(2, 10)"), "1024");
    assert_eq!(eval("Math.sqrt(144)"), "12");
    assert_eq!(eval("Math.round(2.5)"), "3");
}

#[test]
fn number_globals() {
    assert_eq!(eval("parseInt('42px')"), "42");
    assert_eq!(eval("parseInt('ff', 16)"), "255");
    assert_eq!(eval("parseInt('0x1F')"), "31");
    assert_eq!(eval("parseFloat('3.14abc')"), "3.14");
    assert_eq!(eval("isNaN(0 / 0)"), "true");
    assert_eq!(eval("isFinite(1 / 0)"), "false");
    assert_eq!(eval("Number('12') + 3"), "15");
    assert_eq!(eval("String(42) + '!'"), "42!");
}

#[test]
fn array_methods() {
    assert_eq!(eval("let a = [1, 2]; a.push(3, 4); a.length"), "4");
    assert_eq!(eval("let a = [1, 2, 3]; a.pop()"), "3");
    assert_eq!(eval("[1, 2, 3].join('-')"), "1-2-3");
    assert_eq!(eval("[1, 2, 3].includes(2)"), "true");
    assert_eq!(eval("[1, 2, 3].indexOf(3)"), "2");
    assert_eq!(eval("[1, 2, 3, 4].slice(1, 3).join(',')"), "2,3");
    assert_eq!(eval("[1, 2, 3].map(x => x * x).join(',')"), "1,4,9");
    assert_eq!(
        eval("[1, 2, 3, 4].filter(x => x % 2 === 0).join(',')"),
        "2,4"
    );
    assert_eq!(eval("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
    assert_eq!(eval("[1, 2, 3, 4].reduce((a, b) => a + b)"), "10");
    assert_eq!(eval("let s = 0; [1, 2, 3].forEach(x => s += x); s"), "6");
    assert_eq!(eval("[1, 2, 3].find(x => x > 1)"), "2");
    assert_eq!(eval("[1, 2, 3].some(x => x > 2)"), "true");
    assert_eq!(eval("[2, 4, 6].every(x => x % 2 === 0)"), "true");
    assert_eq!(eval("Array.isArray([1, 2])"), "true");
    assert_eq!(eval("Array.isArray('nope')"), "false");
}

#[test]
fn string_methods() {
    assert_eq!(eval("'hello'.toUpperCase()"), "HELLO");
    assert_eq!(eval("'HELLO'.toLowerCase()"), "hello");
    assert_eq!(eval("'  hi  '.trim()"), "hi");
    assert_eq!(eval("'hello world'.includes('world')"), "true");
    assert_eq!(eval("'hello'.indexOf('l')"), "2");
    assert_eq!(eval("'hello'.slice(1, 4)"), "ell");
    assert_eq!(eval("'a,b,c'.split(',').length"), "3");
    assert_eq!(eval("'ab'.repeat(3)"), "ababab");
    assert_eq!(eval("'hello'.charAt(1)"), "e");
    assert_eq!(eval("'hello'.startsWith('he')"), "true");
}

#[test]
fn object_statics() {
    assert_eq!(eval("Object.keys({ a: 1, b: 2 }).join(',')"), "a,b");
    assert_eq!(eval("Object.values({ a: 1, b: 2 }).join(',')"), "1,2");
    assert_eq!(
        eval("Object.entries({ a: 1 }).map(e => e[0] + '=' + e[1]).join(',')"),
        "a=1"
    );
    assert_eq!(
        eval("let t = Object.assign({}, { a: 1 }, { b: 2 }); t.a + t.b"),
        "3"
    );
}

#[test]
fn json() {
    assert_eq!(eval("JSON.stringify(42)"), "42");
    assert_eq!(eval("JSON.stringify('hi')"), "\"hi\"");
    assert_eq!(eval("JSON.stringify([1, 2, 3])"), "[1,2,3]");
    assert_eq!(
        eval("JSON.stringify({ a: 1, b: 'x' })"),
        "{\"a\":1,\"b\":\"x\"}"
    );
    assert_eq!(eval("JSON.parse('[1, 2, 3]').length"), "3");
    assert_eq!(eval("JSON.parse('{\"n\": 42}').n"), "42");
    assert_eq!(
        eval("let o = JSON.parse(JSON.stringify({ x: [1, 2], y: 'z' })); o.x[1] + o.y"),
        "2z"
    );
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
