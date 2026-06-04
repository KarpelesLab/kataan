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

/// Runs a program (draining microtasks) and returns the string value of a
/// top-level binding afterward — used to observe asynchronous (Promise)
/// effects, since `run` flushes the microtask queue before returning.
fn eval_global(src: &str, name: &str) -> String {
    let program = Parser::parse_program(src).expect("parse ok");
    let mut interp = Interp::new();
    interp.run(&program).expect("run ok");
    interp
        .global()
        .get(name)
        .map_or_else(|| String::from("<unbound>"), |v| v.to_js_string())
}

/// Compiles a single-expression program to bytecode, runs it through the VM,
/// and returns the result as a string. Panics if the expression is outside the
/// bytecode compiler's supported subset or throws.
fn eval_bc(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse ok");
    let mut interp = Interp::new();
    match interp.eval_via_bytecode(&program.body) {
        Ok(Ok(v)) => v.to_js_string(),
        Ok(Err(thrown)) => panic!("uncaught: {}", thrown.to_js_string()),
        Err(e) => panic!("compile: {}", e.message),
    }
}

#[test]
fn bytecode_vm_matches_tree_walker() {
    // Arithmetic, precedence, and the compact-int path.
    assert_eq!(eval_bc("1 + 2 * 3"), "7");
    assert_eq!(eval_bc("(1 + 2) * 3"), "9");
    assert_eq!(eval_bc("2 ** 3 ** 2"), "512");
    assert_eq!(eval_bc("-5 % 3"), "-2");
    assert_eq!(eval_bc("10 / 4"), "2.5");
    // String + coercion (reuses the tree-walker's operator semantics).
    assert_eq!(eval_bc("'a' + 'b' + 'c'"), "abc");
    assert_eq!(eval_bc("1 + '2'"), "12");
    // Comparison and logical short-circuit.
    assert_eq!(eval_bc("3 < 5"), "true");
    assert_eq!(eval_bc("2 === 2"), "true");
    assert_eq!(eval_bc("true && 'yes'"), "yes");
    assert_eq!(eval_bc("false || 'fallback'"), "fallback");
    assert_eq!(eval_bc("0 && 'no'"), "0"); // short-circuits, leaves left
    // Unary.
    assert_eq!(eval_bc("-(3 + 4)"), "-7");
    assert_eq!(eval_bc("!false"), "true");
    // Globals, member access, indexing, and calls (into the real stdlib).
    assert_eq!(eval_bc("Math.max(1, 9, 4)"), "9");
    assert_eq!(eval_bc("Math.floor(3.9) + Math.ceil(1.1)"), "5");
    assert_eq!(eval_bc("'hello'.length"), "5");
    assert_eq!(eval_bc("Number.MAX_SAFE_INTEGER"), "9007199254740991");
    // A reference error propagates through the VM.
    let program = Parser::parse_program("missingGlobalXyz").unwrap();
    let mut interp = Interp::new();
    let result = interp.eval_via_bytecode(&program.body).unwrap();
    assert!(result.is_err());
}

#[test]
fn bytecode_vm_statements() {
    // Locals, assignment, and a final-expression completion value.
    assert_eq!(eval_bc("let x = 10; x + 5"), "15");
    assert_eq!(eval_bc("let x = 1; x = x + 41; x"), "42");
    assert_eq!(eval_bc("let a = 2; let b = 3; a * b"), "6");
    // Compound assignment.
    assert_eq!(eval_bc("let n = 5; n += 10; n *= 2; n"), "30");
    // Block scoping (inner shadow does not leak).
    assert_eq!(eval_bc("let x = 1; { let x = 99; } x"), "1");
    // if / else.
    assert_eq!(
        eval_bc("let r = 0; if (3 > 2) r = 'big'; else r = 'small'; r"),
        "big"
    );
    assert_eq!(
        eval_bc("let r = 0; if (1 > 2) r = 'a'; else r = 'b'; r"),
        "b"
    );
    // while loop summing 0..5.
    assert_eq!(
        eval_bc("let s = 0; let i = 0; while (i < 5) { s += i; i += 1; } s"),
        "10"
    );
    // A loop that calls into the real stdlib.
    assert_eq!(
        eval_bc("let m = 0; let i = 1; while (i <= 4) { m = Math.max(m, i * i); i += 1; } m"),
        "16"
    );
    // Globals are written through SetGlobal when not declared local.
    assert_eq!(eval_bc("g = 7; g + 1"), "8");
}

#[test]
fn bytecode_vm_functions() {
    // Declaration, parameters, return.
    assert_eq!(
        eval_bc("function add(a, b) { return a + b; } add(3, 4)"),
        "7"
    );
    // Recursion (the declaration is hoisted to a global).
    assert_eq!(
        eval_bc("function fib(n) { if (n < 2) return n; return fib(n - 1) + fib(n - 2); } fib(10)"),
        "55"
    );
    // Local function called within a loop.
    assert_eq!(
        eval_bc(
            "function sq(x) { return x * x; }
             let total = 0; let i = 1;
             while (i <= 4) { total += sq(i); i += 1; }
             total"
        ),
        "30" // 1 + 4 + 9 + 16
    );
    // A function expression assigned to a local, then called.
    assert_eq!(
        eval_bc("let f = function (n) { return n * 10; }; f(5)"),
        "50"
    );
    // An arrow with an expression body.
    assert_eq!(eval_bc("let inc = x => x + 1; inc(inc(inc(0)))"), "3");
    // A function may call another by name (mutual reference via globals).
    assert_eq!(
        eval_bc(
            "function ev(n) { if (n === 0) return true; return od(n - 1); }
             function od(n) { if (n === 0) return false; return ev(n - 1); }
             ev(8)"
        ),
        "true"
    );
}

#[test]
fn bytecode_vm_object_array_literals() {
    // Array literals and indexing.
    assert_eq!(eval_bc("let a = [10, 20, 30]; a[1]"), "20");
    assert_eq!(eval_bc("[1, 2, 3].length"), "3");
    assert_eq!(eval_bc("let a = [1, 2]; a[2] = 9; a[2]"), "9");
    // Object literals (identifier, string, numeric, computed keys).
    assert_eq!(eval_bc("let o = { x: 1, y: 2 }; o.x + o.y"), "3");
    assert_eq!(eval_bc("let o = { 'a-b': 7 }; o['a-b']"), "7");
    assert_eq!(eval_bc("let k = 'key'; let o = { [k]: 42 }; o.key"), "42");
    // Property writes and a small computed build.
    assert_eq!(eval_bc("let o = {}; o.n = 5; o.n += 10; o.n"), "15");
    assert_eq!(
        eval_bc(
            "let a = []; let i = 0; while (i < 4) { a[i] = i * i; i += 1; } a[3] + ',' + a.length"
        ),
        "9,4"
    );
    // Nested structures.
    assert_eq!(
        eval_bc("let o = { list: [1, 2, { v: 3 }] }; o.list[2].v"),
        "3"
    );
}

/// Compiles to bytecode, serializes it, deserializes it, and runs the reloaded
/// module — the export/reload round-trip.
fn eval_bc_reloaded(src: &str) -> String {
    use crate::bytecode::{deserialize, serialize};
    let program = Parser::parse_program(src).expect("parse ok");
    let module = crate::interp::compile_program(&program.body).expect("compile ok");
    let bytes = serialize(&module);
    let reloaded = deserialize(&bytes).expect("deserialize ok");
    let mut interp = Interp::new();
    match interp.run_module(alloc::rc::Rc::new(reloaded)) {
        Ok(v) => v.to_js_string(),
        Err(e) => panic!("uncaught: {}", e.to_js_string()),
    }
}

#[test]
fn bytecode_serialize_reload_run_roundtrip() {
    // Programs run identically after a serialize → deserialize → run round-trip.
    assert_eq!(eval_bc_reloaded("1 + 2 * 3"), "7");
    assert_eq!(
        eval_bc_reloaded("function fib(n) { return n < 2 ? n : fib(n-1) + fib(n-2); } fib(10)"),
        "55"
    );
    assert_eq!(
        eval_bc_reloaded("let s = 0; for (let i = 1; i <= 100; i += 1) s += i; s"),
        "5050"
    );
    assert_eq!(
        eval_bc_reloaded("[1, 2, 3, 4].map(x => x * x).reduce((a, b) => a + b, 0)"),
        "30"
    );
    assert_eq!(
        eval_bc_reloaded("let o = { v: 21, dbl() { return this.v * 2; } }; o.dbl()"),
        "42"
    );
    assert_eq!(
        eval_bc_reloaded("let r = 0; try { throw 'e'; } catch (x) { r = x; } `caught:${r}`"),
        "caught:e"
    );
}

#[test]
fn bytecode_vm_compound_and_logical_assignment() {
    // Bitwise / shift compound assignment.
    assert_eq!(eval_bc("let n = 8; n &= 12; n"), "8");
    assert_eq!(eval_bc("let n = 8; n |= 3; n"), "11");
    assert_eq!(eval_bc("let n = 6; n ^= 3; n"), "5");
    assert_eq!(eval_bc("let n = 1; n <<= 4; n"), "16");
    assert_eq!(eval_bc("let n = 64; n >>= 2; n"), "16");
    // Logical assignment short-circuits.
    assert_eq!(eval_bc("let o = {}; o.x ??= 5; o.x ??= 9; o.x"), "5");
    assert_eq!(eval_bc("let a = null; a ??= 7; a"), "7");
    assert_eq!(eval_bc("let b = 0; b ||= 42; b"), "42");
    assert_eq!(eval_bc("let c = 3; c ||= 99; c"), "3");
    assert_eq!(eval_bc("let d = 1; d &&= 2; d &&= 3; d"), "3");
    assert_eq!(eval_bc("let e = 5; e &&= 0; e &&= 9; e"), "0");
    // The right-hand side is not evaluated when the assignment doesn't fire.
    assert_eq!(
        eval_bc(
            "let calls = 0; const side = () => { calls += 1; return 1; };
             let v = 10; v ??= side(); calls"
        ),
        "0"
    );
    // Logical assignment on a member.
    assert_eq!(
        eval_bc("let cfg = { count: 0 }; cfg.count ||= 5; cfg.count ||= 9; cfg.count"),
        "5"
    );
}

#[test]
fn bytecode_vm_bitwise_in_instanceof() {
    // Bitwise / shift via the generic Binary op.
    assert_eq!(eval_bc("5 & 3"), "1");
    assert_eq!(eval_bc("5 | 2"), "7");
    assert_eq!(eval_bc("5 ^ 1"), "4");
    assert_eq!(eval_bc("1 << 4"), "16");
    assert_eq!(eval_bc("256 >> 2"), "64");
    assert_eq!(eval_bc("-1 >>> 28"), "15");
    // `in` and `instanceof`.
    assert_eq!(eval_bc("let o = { a: 1 }; 'a' in o"), "true");
    assert_eq!(eval_bc("let o = { a: 1 }; 'b' in o"), "false");
    assert_eq!(eval_bc("new Map() instanceof Map"), "true");
    assert_eq!(eval_bc("new TypeError('x') instanceof Error"), "true");
}

#[test]
fn bytecode_vm_typeof_void() {
    assert_eq!(eval_bc("typeof 42"), "number");
    assert_eq!(eval_bc("typeof 'x'"), "string");
    assert_eq!(eval_bc("typeof true"), "boolean");
    assert_eq!(eval_bc("typeof undefined"), "undefined");
    assert_eq!(eval_bc("typeof {}"), "object");
    assert_eq!(eval_bc("typeof Math"), "object");
    assert_eq!(eval_bc("typeof Math.max"), "function");
    assert_eq!(eval_bc("let f = x => x; typeof f"), "function");
    assert_eq!(eval_bc("let n = 5; typeof n"), "number");
    // typeof an unbound global does not throw.
    assert_eq!(eval_bc("typeof someUnboundThing"), "undefined");
    assert_eq!(eval_bc("typeof someUnboundThing === 'undefined'"), "true");
    // void always yields undefined.
    assert_eq!(eval_bc("void 0"), "undefined");
    assert_eq!(eval_bc("let x = 5; void (x = 10); x"), "10"); // side effect runs
    // Unary plus coerces to number.
    assert_eq!(eval_bc("+'42'"), "42");
    assert_eq!(eval_bc("+true"), "1");
    // delete removes a property and returns a boolean.
    assert_eq!(
        eval_bc("let o = { a: 1, b: 2 }; delete o.a; o.a === undefined && o.b === 2"),
        "true"
    );
    assert_eq!(
        eval_bc("let o = { x: 1 }; delete o['x']; 'x' in o"),
        "false"
    );
    assert_eq!(eval_bc("delete ({}).missing"), "true");
}

#[test]
fn bytecode_vm_for_of() {
    // Over an array.
    assert_eq!(
        eval_bc("let s = 0; for (const x of [1, 2, 3, 4]) s += x; s"),
        "10"
    );
    // Over a string (characters).
    assert_eq!(
        eval_bc("let out = ''; for (const c of 'abc') out = c + out; out"),
        "cba"
    );
    // Over a Set (deduplicated values).
    assert_eq!(
        eval_bc("let n = 0; for (const v of new Set([1, 1, 2, 3, 3])) n += v; n"),
        "6"
    );
    // Over a Map ([key, value] pairs) with break/continue.
    assert_eq!(
        eval_bc(
            "let out = '';
             for (const pair of new Map([['a', 1], ['b', 2], ['c', 3]])) {
               if (pair[0] === 'b') continue;
               out += pair[0] + pair[1];
             }
             out"
        ),
        "a1c3"
    );
    // break exits the loop.
    assert_eq!(
        eval_bc("let r = 0; for (const x of [1, 2, 3, 4, 5]) { if (x === 3) break; r += x; } r"),
        "3"
    );
    // Computed work inside for-of.
    assert_eq!(
        eval_bc(
            "let words = ['hi', 'there']; let total = 0; for (const w of words) total += w.length; total"
        ),
        "7"
    );
    // A non-iterable throws.
    let program = Parser::parse_program("for (const x of 42) {}").unwrap();
    let mut interp = Interp::new();
    assert!(interp.eval_via_bytecode(&program.body).unwrap().is_err());
}

#[test]
fn bytecode_vm_accessors() {
    // Object-literal getter.
    assert_eq!(
        eval_bc("let o = { a: 3, get double() { return this.a * 2; } }; o.double"),
        "6"
    );
    // Object-literal getter + setter.
    assert_eq!(
        eval_bc(
            "let o = { _v: 0, get v() { return this._v; }, set v(x) { this._v = x + 1; } };
             o.v = 41; o.v"
        ),
        "42"
    );
    // Class getter computed from a field.
    assert_eq!(
        eval_bc(
            "class Circle {
               constructor(r) { this.r = r; }
               get area() { return Math.round(this.r * this.r * 3.14); }
             }
             new Circle(2).area"
        ),
        "13"
    );
    // Class getter + setter pair.
    assert_eq!(
        eval_bc(
            "class Box {
               constructor() { this._w = 1; }
               get width() { return this._w; }
               set width(x) { this._w = x < 0 ? 0 : x; }
             }
             let b = new Box(); b.width = -5; let neg = b.width; b.width = 9; neg + ',' + b.width"
        ),
        "0,9"
    );
    // A static getter.
    assert_eq!(
        eval_bc("class Config { static get version() { return 3; } } Config.version"),
        "3"
    );
    // An inherited getter.
    assert_eq!(
        eval_bc(
            "class Base { get kind() { return 'base'; } }
             class Sub extends Base {}
             new Sub().kind"
        ),
        "base"
    );
}

#[test]
fn bytecode_vm_classes() {
    // Constructor + instance methods + `this`.
    assert_eq!(
        eval_bc(
            "class Point {
               constructor(x, y) { this.x = x; this.y = y; }
               norm() { return Math.sqrt(this.x * this.x + this.y * this.y); }
             }
             new Point(3, 4).norm()"
        ),
        "5"
    );
    // Instance fields with initializers.
    assert_eq!(
        eval_bc(
            "class Box { value = 10; label = 'b'; total() { return this.value; } }
             let b = new Box(); b.value += 5; b.total() + ',' + b.label"
        ),
        "15,b"
    );
    // A method mutating a field via `++`.
    assert_eq!(
        eval_bc(
            "class Counter { count = 0; inc() { return ++this.count; } }
             let c = new Counter(); c.inc(); c.inc(); c.inc()"
        ),
        "3"
    );
    // Static methods and fields.
    assert_eq!(
        eval_bc(
            "class Registry { static items = []; static add(x) { Registry.items.push(x); return Registry.items.length; } }
             Registry.add('a'); Registry.add('b')"
        ),
        "2"
    );
    // instanceof on a compiled class.
    assert_eq!(
        eval_bc("class A {} class B {} let a = new A(); (a instanceof A) + ',' + (a instanceof B)"),
        "true,false"
    );
    // A method calling another method on `this`.
    assert_eq!(
        eval_bc(
            "class Calc {
               constructor(n) { this.n = n; }
               double() { return this.n * 2; }
               quadruple() { return this.double() * 2; }
             }
             new Calc(5).quadruple()"
        ),
        "20"
    );
    // A class round-trips through serialize → reload → run.
    assert_eq!(
        eval_bc_reloaded(
            "class Adder { constructor(b) { this.b = b; } add(x) { return x + this.b; } }
             new Adder(100).add(23)"
        ),
        "123"
    );
}

#[test]
fn bytecode_vm_class_inheritance() {
    // super() in the subclass constructor, plus inherited + overridden methods.
    assert_eq!(
        eval_bc(
            "class Animal {
               constructor(name) { this.name = name; }
               speak() { return this.name + ' makes a sound'; }
               describe() { return 'I am ' + this.name; }
             }
             class Dog extends Animal {
               constructor(name) { super(name); this.legs = 4; }
               speak() { return this.name + ' barks'; }
             }
             let d = new Dog('Rex');
             d.speak() + '|' + d.describe() + '|' + d.legs"
        ),
        "Rex barks|I am Rex|4"
    );
    // super.method() calls the parent's implementation.
    assert_eq!(
        eval_bc(
            "class Base { greet() { return 'hi'; } }
             class Sub extends Base { greet() { return super.greet() + '!'; } }
             new Sub().greet()"
        ),
        "hi!"
    );
    // instanceof walks the inheritance chain.
    assert_eq!(
        eval_bc(
            "class A {} class B extends A {} class C extends B {}
             let c = new C();
             (c instanceof A) + ',' + (c instanceof B) + ',' + (c instanceof C)"
        ),
        "true,true,true"
    );
    // A three-level chain with super calls and accumulation.
    assert_eq!(
        eval_bc(
            "class L1 { constructor() { this.tag = 'a'; } }
             class L2 extends L1 { constructor() { super(); this.tag += 'b'; } }
             class L3 extends L2 { constructor() { super(); this.tag += 'c'; } }
             new L3().tag"
        ),
        "abc"
    );
    // Inheritance survives serialize → reload → run.
    assert_eq!(
        eval_bc_reloaded(
            "class Shape { constructor(n) { this.n = n; } kind() { return 'shape'; } }
             class Circle extends Shape { kind() { return super.kind() + ':circle'; } }
             let c = new Circle(7); c.n + ',' + c.kind()"
        ),
        "7,shape:circle"
    );
}

#[test]
fn bytecode_vm_closures() {
    // A counter closing over a mutable local (shared cell).
    assert_eq!(
        eval_bc(
            "function makeCounter() { let c = 0; return () => ++c; }
             let inc = makeCounter();
             inc(); inc(); inc()"
        ),
        "3"
    );
    // Read-only capture in a higher-order call.
    assert_eq!(
        eval_bc("let base = 100; [1, 2, 3].map(x => x + base).join(',')"),
        "101,102,103"
    );
    // Curried function expression.
    assert_eq!(
        eval_bc("function adder(n) { return function (x) { return x + n; }; } adder(10)(5)"),
        "15"
    );
    // Two closures sharing the same captured cell.
    assert_eq!(
        eval_bc(
            "function makePair() {
               let v = 0;
               return { inc: () => { v += 1; return v; }, get: () => v };
             }
             let p = makePair();
             p.inc(); p.inc();
             p.get()"
        ),
        "2"
    );
    // Capture of a function parameter.
    assert_eq!(
        eval_bc(
            "function sumWith(offset) { return arr => arr.reduce((a, b) => a + b + offset, 0); }
             sumWith(1)([10, 20, 30])"
        ),
        "63" // 10+1 + 20+1 + 30+1
    );
    // Transitive capture through nested arrows.
    assert_eq!(eval_bc("let f = a => b => c => a + b + c; f(1)(2)(3)"), "6");
    // A captured variable mutated after the closure is created is observed.
    assert_eq!(eval_bc("let x = 1; let get = () => x; x = 42; get()"), "42");
    // A closure survives the serialize → reload → run round-trip.
    assert_eq!(
        eval_bc_reloaded(
            "function adder(n) { return x => x + n; } let add5 = adder(5); add5(add5(0))"
        ),
        "10"
    );
}

#[test]
fn function_apply_and_call() {
    // Tree-walker.
    assert_eq!(
        eval("function add(a, b, c) { return a + b + c; } add.apply(null, [1, 2, 3])"),
        "6"
    );
    assert_eq!(
        eval("function add(a, b, c) { return a + b + c; } add.call(null, 4, 5, 6)"),
        "15"
    );
    assert_eq!(eval("Math.max.apply(null, [7, 2, 9, 1])"), "9");
    // `this` rebinding via call/apply.
    assert_eq!(
        eval("const o = { v: 10, get() { return this.v; } }; o.get.call({ v: 99 })"),
        "99"
    );
    assert_eq!(
        eval("function id() { return this.x; } id.apply({ x: 'hi' })"),
        "hi"
    );
    // apply with no/empty argument list.
    assert_eq!(eval("function n() { return 7; } n.apply(null)"), "7");
    // Through the bytecode VM.
    assert_eq!(
        eval_bc("function mul(a, b) { return a * b; } mul.apply(null, [6, 7])"),
        "42"
    );
    assert_eq!(
        eval_bc("function mul(a, b) { return a * b; } mul.call(null, 3, 4)"),
        "12"
    );
    // bind: partial application and `this` binding.
    assert_eq!(
        eval("function greet(g, name) { return g + ', ' + name; } greet.bind(null, 'Hi')('Ada')"),
        "Hi, Ada"
    );
    assert_eq!(
        eval("const o = { v: 42, get() { return this.v; } }; o.get.bind({ v: 99 })()"),
        "99"
    );
    assert_eq!(
        eval("const add = (a, b, c) => a + b + c; add.bind(null, 1, 2)(3)"),
        "6"
    );
    assert_eq!(eval("typeof (function () {}).bind(null)"), "function");
    // bind through the bytecode VM.
    assert_eq!(
        eval_bc("function mul(a, b) { return a * b; } let d = mul.bind(null, 2); d(21)"),
        "42"
    );
}

#[test]
fn bytecode_vm_update_operator() {
    // Prefix and postfix, increment and decrement, on locals.
    assert_eq!(eval_bc("let x = 5; ++x"), "6");
    assert_eq!(eval_bc("let x = 5; x++"), "5"); // postfix yields the old value
    assert_eq!(eval_bc("let x = 5; x++; x"), "6");
    assert_eq!(eval_bc("let x = 5; --x"), "4");
    assert_eq!(eval_bc("let x = 5; x--; x"), "4");
    // As a loop step.
    assert_eq!(
        eval_bc("let s = 0; for (let i = 0; i < 5; i++) s += i; s"),
        "10"
    );
    // On a member.
    assert_eq!(eval_bc("let o = { n: 10 }; o.n++; o.n"), "11");
    assert_eq!(eval_bc("let a = [1, 2, 3]; ++a[1]; a[1]"), "3");
    // String coercion: '5' becomes 5 then increments.
    assert_eq!(eval_bc("let x = '5'; x++; x"), "6");
}

#[test]
fn bytecode_vm_rest_parameters() {
    // Pure variadic.
    assert_eq!(
        eval_bc(
            "function sum(...nums) { let t = 0; for (const n of nums) t += n; return t; } sum(1, 2, 3, 4, 5)"
        ),
        "15"
    );
    // Leading fixed parameters then a rest.
    assert_eq!(
        eval_bc(
            "function tail(first, ...rest) { return first + ':' + rest.join(','); } tail('a', 'b', 'c', 'd')"
        ),
        "a:b,c,d"
    );
    // Empty rest is an empty array.
    assert_eq!(eval_bc("function f(...xs) { return xs.length; } f()"), "0");
    assert_eq!(
        eval_bc("function f(a, ...xs) { return xs.length; } f(1)"),
        "0"
    );
    // Arrow with rest.
    assert_eq!(
        eval_bc("let count = (...xs) => xs.length; count(1, 2, 3)"),
        "3"
    );
    // Rest + spread round-trips an argument list.
    assert_eq!(
        eval_bc("function relay(...args) { return Math.max(...args); } relay(3, 9, 2, 7)"),
        "9"
    );
    // Rest survives the serialize → reload → run round-trip.
    assert_eq!(
        eval_bc_reloaded(
            "function sum(...ns) { return ns.reduce((a, b) => a + b, 0); } sum(10, 20, 30)"
        ),
        "60"
    );
}

#[test]
fn bytecode_vm_object_spread() {
    assert_eq!(
        eval_bc("let b = { a: 1, b: 2 }; let m = { ...b, c: 3 }; m.a + m.b + m.c"),
        "6"
    );
    // A later property overrides a spread one.
    assert_eq!(
        eval_bc("let b = { x: 1 }; let o = { ...b, x: 99 }; o.x"),
        "99"
    );
    // A later spread overrides earlier keys.
    assert_eq!(
        eval_bc("let o = { a: 1, ...{ a: 2, b: 3 } }; o.a + ',' + o.b"),
        "2,3"
    );
    // Spreading from a function result.
    assert_eq!(
        eval_bc(
            "function defaults() { return { color: 'red', size: 1 }; } let o = { ...defaults(), size: 5 }; o.color + o.size"
        ),
        "red5"
    );
    // The original is not mutated.
    assert_eq!(
        eval_bc("let orig = { n: 1 }; let copy = { ...orig, n: 2 }; orig.n + ',' + copy.n"),
        "1,2"
    );
}

#[test]
fn bytecode_vm_call_spread() {
    // Spread into a plain call, mixed with positional args.
    assert_eq!(
        eval_bc(
            "function sum(a, b, c, d) { return a + b + c + d; } let xs = [2, 3]; sum(1, ...xs, 4)"
        ),
        "10"
    );
    assert_eq!(
        eval_bc("function f(a, b, c) { return a*100 + b*10 + c; } f(...[1, 2, 3])"),
        "123"
    );
    // Spread into a built-in.
    assert_eq!(eval_bc("Math.max(...[5, 9, 2, 7])"), "9");
    assert_eq!(eval_bc("Math.min(1, ...[8, 0, 4])"), "0");
    // Spread a string's characters as args.
    assert_eq!(
        eval_bc("function j(a, b, c) { return a + b + c; } j(...'xyz')"),
        "xyz"
    );
    // Method call with spread keeps the receiver as `this`.
    assert_eq!(
        eval_bc(
            "let o = { base: 100, add(a, b) { return this.base + a + b; } }; let v = [2, 3]; o.add(...v)"
        ),
        "105"
    );
}

#[test]
fn bytecode_vm_array_spread() {
    assert_eq!(eval_bc("[...[1, 2, 3]].length"), "3");
    assert_eq!(eval_bc("[0, ...[1, 2], 3].join(',')"), "0,1,2,3");
    assert_eq!(
        eval_bc("let a = [2, 3]; let b = [1, ...a, 4]; b.join(',')"),
        "1,2,3,4"
    );
    // Spread of a string (characters) and a Set (values).
    assert_eq!(eval_bc("[...'abc'].join('-')"), "a-b-c");
    assert_eq!(eval_bc("[...new Set([1, 1, 2, 3, 3])].join(',')"), "1,2,3");
    // Concatenating two arrays via spread.
    assert_eq!(
        eval_bc("let x = [1, 2]; let y = [3, 4]; [...x, ...y].reduce((a, b) => a + b, 0)"),
        "10"
    );
}

#[test]
fn bytecode_vm_object_rest_pattern() {
    assert_eq!(
        eval_bc(
            "let { a, b, ...rest } = { a: 1, b: 2, c: 3, d: 4 }; a + ',' + b + ',' + rest.c + ',' + rest.d"
        ),
        "1,2,3,4"
    );
    // The rest object excludes the named keys.
    assert_eq!(
        eval_bc("let { x, ...others } = { x: 10, y: 20 }; ('x' in others) + ',' + others.y"),
        "false,20"
    );
    // No remaining keys: an empty rest object.
    assert_eq!(
        eval_bc("let { a, ...rest } = { a: 1 }; Object.keys(rest).length"),
        "0"
    );
}

#[test]
fn bytecode_vm_destructuring() {
    // Array destructuring.
    assert_eq!(eval_bc("let [a, b] = [1, 2]; a + b"), "3");
    assert_eq!(eval_bc("let [a, , c] = [1, 2, 3]; a + c"), "4");
    assert_eq!(
        eval_bc("let [first, ...rest] = [1, 2, 3, 4]; rest.length"),
        "3"
    );
    assert_eq!(eval_bc("let [a, b = 10] = [5]; a + b"), "15");
    // Object destructuring.
    assert_eq!(eval_bc("let { x, y } = { x: 7, y: 8 }; x * y"), "56");
    assert_eq!(eval_bc("let { a: p, b: q } = { a: 1, b: 2 }; p - q"), "-1");
    assert_eq!(eval_bc("let { x, y = 99 } = { x: 1 }; x + y"), "100");
    // Nested destructuring.
    assert_eq!(
        eval_bc("let { p: [a, b], q } = { p: [1, 2], q: 3 }; a + b + q"),
        "6"
    );
    assert_eq!(eval_bc("let [{ v }] = [{ v: 42 }]; v"), "42");
    // Destructuring from a function result.
    assert_eq!(
        eval_bc("function pair() { return [10, 20]; } let [a, b] = pair(); a + b"),
        "30"
    );
    // Parameter defaults and destructuring.
    assert_eq!(
        eval_bc("function f(a, b = 10) { return a + b; } f(5)"),
        "15"
    );
    assert_eq!(
        eval_bc("function f(a, b = 10) { return a + b; } f(5, 20)"),
        "25"
    );
    assert_eq!(
        eval_bc("let dist = ([x1, y1], [x2, y2]) => (x2-x1) + (y2-y1); dist([0, 0], [3, 4])"),
        "7"
    );
    assert_eq!(
        eval_bc("function area({ w, h }) { return w * h; } area({ w: 4, h: 5 })"),
        "20"
    );
    assert_eq!(
        eval_bc("function greet({ name = 'world' } = {}) { return 'hi ' + name; } greet({})"),
        "hi world"
    );
}

#[test]
fn bytecode_vm_for_in() {
    // Own keys of an object.
    assert_eq!(
        eval_bc("let o = { a: 1, b: 2, c: 3 }; let out = ''; for (const k in o) out += k; out"),
        "abc"
    );
    // Sum the values by key.
    assert_eq!(
        eval_bc("let o = { a: 1, b: 2, c: 3 }; let s = 0; for (const k in o) s += o[k]; s"),
        "6"
    );
    // Array indices.
    assert_eq!(
        eval_bc("let a = [10, 20, 30]; let out = ''; for (const i in a) out += i; out"),
        "012"
    );
    // for-in over a non-object yields nothing (no throw).
    assert_eq!(
        eval_bc("let count = 0; for (const k in 42) count += 1; count"),
        "0"
    );
}

#[test]
fn bytecode_vm_switch() {
    let sw = |n: &str| {
        alloc::format!(
            "let r = '?'; switch ({n}) {{ \
               case 1: r = 'one'; break; \
               case 2: r = 'two'; break; \
               default: r = 'other'; \
             }} r"
        )
    };
    assert_eq!(eval_bc(&sw("1")), "one");
    assert_eq!(eval_bc(&sw("2")), "two");
    assert_eq!(eval_bc(&sw("9")), "other");
    // Fall-through (no break).
    assert_eq!(
        eval_bc(
            "let r = ''; switch (2) { case 1: r += 'a'; case 2: r += 'b'; case 3: r += 'c'; break; case 4: r += 'd'; } r"
        ),
        "bc"
    );
    // `continue` inside a switch targets the enclosing loop, not the switch.
    assert_eq!(
        eval_bc(
            "let out = '';
             for (let i = 0; i < 4; i += 1) {
               switch (i) { case 1: continue; case 2: out += 'two'; break; default: out += i; }
             }
             out"
        ),
        "0two3"
    );
    // String discriminant.
    assert_eq!(
        eval_bc(
            "let x = 'b'; let r = 0; switch (x) { case 'a': r = 1; break; case 'b': r = 2; break; } r"
        ),
        "2"
    );
}

#[test]
fn bytecode_vm_new_operator() {
    // Built-in constructors via `new`.
    assert_eq!(eval_bc("let m = new Map(); m.set('k', 1); m.get('k')"), "1");
    assert_eq!(eval_bc("let s = new Set([1, 1, 2]); s.size"), "2");
    assert_eq!(eval_bc("new Error('oops').message"), "oops");
    assert_eq!(eval_bc("new Date(0).getFullYear()"), "1970");
    assert_eq!(eval_bc("new TypeError('x') instanceof Error"), "true");
    // `new` then a method call on the result.
    assert_eq!(
        eval_bc("let m = new Map([['a', 1], ['b', 2]]); m.has('a') && m.size === 2"),
        "true"
    );
}

#[test]
fn bytecode_vm_try_catch_throw() {
    // A throw is caught and its value bound.
    assert_eq!(
        eval_bc("let r = 0; try { throw 'boom'; } catch (e) { r = e; } r"),
        "boom"
    );
    // Catch without a binding.
    assert_eq!(
        eval_bc("let r = 'a'; try { throw 1; } catch { r = 'b'; } r"),
        "b"
    );
    // No throw: the catch is skipped.
    assert_eq!(
        eval_bc("let r = 'init'; try { r = 'ok'; } catch (e) { r = 'caught'; } r"),
        "ok"
    );
    // An engine-raised error (calling a non-function) is catchable.
    assert_eq!(
        eval_bc("let r = ''; try { let n = 5; n(); } catch (e) { r = e.name; } r"),
        "TypeError"
    );
    // Engine-thrown errors match their type (and Error) via instanceof.
    assert_eq!(
        eval_bc(
            "let r = '';
             try { null.x; } catch (e) {
               r = (e instanceof TypeError) + ',' + (e instanceof Error) + ',' + (e instanceof RangeError);
             }
             r"
        ),
        "true,true,false"
    );
    assert_eq!(
        eval(
            "try { undefinedThing; } catch (e) { '' + (e instanceof ReferenceError) + (e instanceof Error); }"
        ),
        "truetrue"
    );
    // A throw from a called function unwinds to the caller's handler.
    assert_eq!(
        eval_bc(
            "function boom() { throw 'deep'; }
             let r = 0; try { boom(); } catch (e) { r = e; } r"
        ),
        "deep"
    );
    // Rethrow from catch, caught by an outer try.
    assert_eq!(
        eval_bc(
            "let r = 0;
             try { try { throw 'x'; } catch (e) { throw 'y'; } } catch (e) { r = e; }
             r"
        ),
        "y"
    );
    // throw new Error(...) and read .message.
    assert_eq!(
        eval_bc("let r = ''; try { throw new Error('msg'); } catch (e) { r = e.message; } r"),
        "msg"
    );
}

#[test]
fn bytecode_vm_finally() {
    // finally runs after a caught throw.
    assert_eq!(
        eval_bc(
            "let log = '';
             try { log += 'T'; throw 'e'; } catch (x) { log += 'C'; } finally { log += 'F'; }
             log"
        ),
        "TCF"
    );
    // finally runs on normal completion (no catch).
    assert_eq!(
        eval_bc("let log = ''; try { log += 'T'; } finally { log += 'F'; } log"),
        "TF"
    );
    // finally runs even when nothing is caught, before the error propagates.
    assert_eq!(
        eval_bc(
            "let log = '';
             try {
               try { throw 'boom'; } finally { log += 'inner;'; }
             } catch (e) { log += 'outer:' + e; }
             log"
        ),
        "inner;outer:boom"
    );
    // finally observes a value mutated in catch.
    assert_eq!(
        eval_bc(
            "let r = 0;
             try { throw 1; } catch (e) { r = e; } finally { r += 10; }
             r"
        ),
        "11"
    );
    // A throw inside catch still runs finally, then propagates.
    assert_eq!(
        eval_bc(
            "let log = '';
             try {
               try { throw 'a'; } catch (e) { log += 'c;'; throw 'b'; } finally { log += 'f;'; }
             } catch (e) { log += 'outer:' + e; }
             log"
        ),
        "c;f;outer:b"
    );
}

#[test]
fn bytecode_vm_inequality_and_nullish() {
    assert_eq!(eval_bc("1 != 2"), "true");
    assert_eq!(eval_bc("1 != 1"), "false");
    assert_eq!(eval_bc("1 !== '1'"), "true");
    assert_eq!(eval_bc("2 !== 2"), "false");
    assert_eq!(eval_bc("null ?? 'default'"), "default");
    assert_eq!(eval_bc("undefined ?? 'd'"), "d");
    assert_eq!(eval_bc("0 ?? 'd'"), "0"); // 0 is not nullish
    assert_eq!(eval_bc("'' ?? 'd'"), ""); // empty string is not nullish
    assert_eq!(eval_bc("let o = { a: 0 }; o.a ?? 99"), "0");
    assert_eq!(eval_bc("let o = {}; o.missing ?? 99"), "99");
}

#[test]
fn bytecode_vm_ternary_and_templates() {
    // Ternary chooses a branch value.
    assert_eq!(eval_bc("3 > 2 ? 'yes' : 'no'"), "yes");
    assert_eq!(eval_bc("let n = 7; n % 2 === 0 ? 'even' : 'odd'"), "odd");
    assert_eq!(eval_bc("let a = 1; let b = 2; (a > b ? a : b) * 10"), "20");
    // Template literals interpolate and coerce.
    assert_eq!(eval_bc("let x = 5; `value=${x}`"), "value=5");
    assert_eq!(
        eval_bc("let a = 2; let b = 3; `${a} + ${b} = ${a + b}`"),
        "2 + 3 = 5"
    );
    assert_eq!(eval_bc("`plain text`"), "plain text");
    assert_eq!(
        eval_bc("let name = 'world'; `Hello, ${name}! (${name.length} chars)`"),
        "Hello, world! (5 chars)"
    );
    // Ternary + template together.
    assert_eq!(
        eval_bc("let n = 3; `n is ${n > 0 ? 'positive' : 'non-positive'}`"),
        "n is positive"
    );
}

#[test]
fn bytecode_vm_method_calls() {
    // Built-in array/string methods dispatch with the receiver as `this`.
    assert_eq!(
        eval_bc("[1, 2, 3, 4].map(x => x * x).join(',')"),
        "1,4,9,16"
    );
    assert_eq!(eval_bc("[5, 3, 8, 1].filter(n => n > 3).length"), "2");
    assert_eq!(eval_bc("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
    assert_eq!(eval_bc("'Hello World'.toUpperCase()"), "HELLO WORLD");
    assert_eq!(eval_bc("'a,b,c'.split(',').length"), "3");
    assert_eq!(eval_bc("'  trim  '.trim()"), "trim");
    assert_eq!(eval_bc("[3, 1, 2].sort((a, b) => a - b).join('')"), "123");
    // A user method uses `this` correctly.
    assert_eq!(
        eval_bc("let o = { v: 10, get() { return this.v; } }; o.get()"),
        "10"
    );
    assert_eq!(
        eval_bc("let o = { base: 100, add(n) { return this.base + n; } }; o.add(23)"),
        "123"
    );
    // Computed method name.
    assert_eq!(eval_bc("let m = 'toUpperCase'; 'hi'[m]()"), "HI");
    // Chained method calls.
    assert_eq!(
        eval_bc("'a-b-c'.split('-').map(s => s + '!').join('')"),
        "a!b!c!"
    );
}

#[test]
fn bytecode_vm_loops() {
    // C-style for loop.
    assert_eq!(
        eval_bc("let s = 0; for (let i = 0; i < 5; i += 1) s += i; s"),
        "10"
    );
    // for with break.
    assert_eq!(
        eval_bc(
            "let out = 0; for (let i = 0; i < 100; i += 1) { if (i === 7) { out = i; break; } } out"
        ),
        "7"
    );
    // for with continue (sum of evens 0..10).
    assert_eq!(
        eval_bc(
            "let s = 0; for (let i = 0; i < 10; i += 1) { if (i % 2 === 1) continue; s += i; } s"
        ),
        "20"
    );
    // while with break/continue.
    assert_eq!(
        eval_bc(
            "let n = 0; let i = 0; while (i < 10) { i += 1; if (i % 2 === 1) continue; n += 1; } n"
        ),
        "5"
    );
    // do-while runs the body at least once.
    assert_eq!(
        eval_bc("let i = 0; let count = 0; do { count += 1; i += 1; } while (i < 3); count"),
        "3"
    );
    // A factorial via a for loop calling nothing.
    assert_eq!(
        eval_bc("let f = 1; for (let i = 1; i <= 5; i += 1) f *= i; f"),
        "120"
    );
}

#[test]
fn bytecode_vm_falls_back_on_unsupported() {
    // A generator is still outside the bytecode compiler's subset (it needs
    // suspendable frames), so it is reported as unsupported and the caller falls
    // back to the tree-walker.
    let program = Parser::parse_program("function* g() { yield 1; } [...g()]").unwrap();
    let mut interp = Interp::new();
    assert!(interp.eval_via_bytecode(&program.body).is_err());
}

#[test]
fn destructuring_assignment() {
    // Swap via array destructuring assignment.
    assert_eq!(
        eval("let a = 1, b = 2; [a, b] = [b, a]; a + ',' + b"),
        "2,1"
    );
    // Array rest into an existing variable.
    assert_eq!(
        eval("let x, y, z; [x, y, ...z] = [1, 2, 3, 4]; x + ',' + y + ',' + z.join('-')"),
        "1,2,3-4"
    );
    // Holes are skipped.
    assert_eq!(eval("let s; [, s] = [10, 20]; s"), "20");
    // Defaults in array destructuring assignment.
    assert_eq!(eval("let a, b; [a = 5, b = 6] = [1]; a + ',' + b"), "1,6");
    // Object destructuring assignment with renaming.
    assert_eq!(
        eval("let p, q; ({ x: p, y: q } = { x: 7, y: 8 }); p + ',' + q"),
        "7,8"
    );
    // A member as a destructuring target.
    assert_eq!(eval("let o = {}; ({ v: o.val } = { v: 42 }); o.val"), "42");
    // Object rest in destructuring assignment.
    assert_eq!(
        eval("let f, r; ({ f, ...r } = { f: 1, a: 2, b: 3 }); f + ',' + JSON.stringify(r)"),
        "1,{\"a\":2,\"b\":3}"
    );
    // Nested destructuring assignment.
    assert_eq!(
        eval("let a, b, c; [a, [b, c]] = [1, [2, 3]]; a + ',' + b + ',' + c"),
        "1,2,3"
    );
}

#[test]
fn optional_chaining_and_calls() {
    // Optional member access.
    assert_eq!(eval("const o = { a: { b: 42 } }; o?.a?.b"), "42");
    assert_eq!(eval("const o = {}; String(o?.x?.y)"), "undefined");
    assert_eq!(eval("const o = null; String(o?.a)"), "undefined");
    // Optional call: present method runs; absent one yields undefined (no throw).
    assert_eq!(eval("const o = { f() { return 7; } }; o.f?.()"), "7");
    assert_eq!(eval("const o = {}; String(o.missing?.())"), "undefined");
    assert_eq!(eval("const o = {}; o.missing?.() ?? 'none'"), "none");
    // Optional call doesn't evaluate arguments when skipped.
    assert_eq!(
        eval(
            "let calls = 0; const arg = () => { calls += 1; return 1; };
             const o = {}; o.fn?.(arg()); calls"
        ),
        "0"
    );
    // Optional call on a plain (non-member) callee.
    assert_eq!(eval("let f; String(f?.())"), "undefined");
}

#[test]
fn bytecode_vm_recursive_and_mutual_closures() {
    // A closure that references itself in its own initializer (memoized fib).
    assert_eq!(
        eval_bc(
            "const memo = (f) => { const c = {}; return (n) => n in c ? c[n] : (c[n] = f(n)); };
             const fib = memo((n) => (n < 2 ? n : fib(n - 1) + fib(n - 2)));
             fib(20)"
        ),
        "6765"
    );
    // Mutually-recursive closures via forward reference.
    assert_eq!(
        eval_bc(
            "const isEven = (n) => (n === 0 ? true : isOdd(n - 1));
             const isOdd = (n) => (n === 0 ? false : isEven(n - 1));
             '' + isEven(10) + ',' + isOdd(7)"
        ),
        "true,true"
    );
    // Self-recursion inside a function body (block scope).
    assert_eq!(
        eval_bc(
            "function run() {
               const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
               return fact(6);
             }
             run()"
        ),
        "720"
    );
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
    // `+` does ToPrimitive: arrays/objects coerce to strings and concatenate.
    assert_eq!(eval("'' + [1, 2, 3]"), "1,2,3");
    assert_eq!(eval("String([1, 2] + [3, 4])"), "1,23,4");
    assert_eq!(eval("String([] + [])"), "");
    assert_eq!(eval("[1] + 1"), "11");
    assert_eq!(eval("1 + [2]"), "12");
    assert_eq!(eval("({}) + '!'"), "[object Object]!");
    // Pure-numeric addition is unaffected.
    assert_eq!(eval("1 + 2"), "3");
    assert_eq!(eval("5 + null"), "5");
    assert_eq!(eval("true + 1"), "2");
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
    // instanceof works through the built-in Error hierarchy.
    assert_eq!(eval("new TypeError('x') instanceof TypeError"), "true");
    assert_eq!(eval("new TypeError('x') instanceof Error"), "true");
    assert_eq!(eval("new RangeError('x') instanceof TypeError"), "false");
    assert_eq!(eval("new Map() instanceof Map"), "true");
    assert_eq!(eval("new Set() instanceof Set"), "true");
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
    assert_eq!(
        eval("[1, 2, 3].flatMap(x => [x, -x]).join(',')"),
        "1,-1,2,-2,3,-3"
    );
    assert_eq!(eval("[1, 2, 3, 4].some(x => x > 3)"), "true");
    assert_eq!(eval("[1, 2, 3, 4].every(x => x > 0)"), "true");
    assert_eq!(eval("[1, 2, 3, 4].every(x => x > 2)"), "false");
    assert_eq!(eval("[1, 2, 3, 4, 5].findLast(x => x % 2 === 1)"), "5");
    assert_eq!(eval("[1, 2, 3, 4, 5].findLastIndex(x => x % 2 === 0)"), "3");
    assert_eq!(eval("[1, 2, 3].findLast(x => x > 9)"), "undefined");
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
    assert_eq!(eval("(1234.5678).toExponential(2)"), "1.23e+3");
    assert_eq!(eval("(0.00012).toExponential()"), "1.2e-4");
    assert_eq!(eval("(123.456).toPrecision(4)"), "123.5");
    assert_eq!(eval("(255).toString(16)"), "ff");
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
    // Spread and for-of over a Set / Map.
    assert_eq!(eval("[...new Set([1, 1, 2, 3])].join(',')"), "1,2,3");
    assert_eq!(
        eval("let t = 0; for (const x of new Set([4, 5])) t += x; t"),
        "9"
    );
    assert_eq!(
        eval("let out = ''; for (const [k, v] of new Map([['a', 1], ['b', 2]])) out += k + v; out"),
        "a1b2"
    );
}

#[test]
fn delete_operator() {
    assert_eq!(
        eval("let o = { a: 1, b: 2 }; delete o.a; o.a === undefined && o.b === 2"),
        "true"
    );
    assert_eq!(eval("let o = { a: 1 }; delete o.a"), "true");
    assert_eq!(eval("let o = { x: 1 }; delete o['x']; 'x' in o"), "false");
    assert_eq!(eval("delete ({}).missing"), "true");
}

#[test]
fn tagged_templates() {
    assert_eq!(
        eval("function t(s, ...v) { return s.join('|') + '#' + v.join(','); } t`a${1}b${2}c`"),
        "a|b|c#1,2"
    );
    // The cooked strings array and its `raw` sibling.
    assert_eq!(
        eval("function t(s) { return s.length + ':' + s[0] + ':' + s.raw[0]; } t`x\\ny`"),
        "1:x\ny:x\\ny"
    );
    // Substitution values arrive after the strings array.
    assert_eq!(
        eval("function sum(s, a, b) { return a + b; } sum`${10}+${20}`"),
        "30"
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
    assert_eq!(eval("Math.hypot(3, 4)"), "5");
    assert_eq!(eval("Math.sign(-7)"), "-1");
    assert_eq!(eval("Math.sign(0)"), "0");
    assert_eq!(eval("Math.cbrt(27)"), "3");
    assert_eq!(eval("Math.clz32(1)"), "31");
    assert_eq!(eval("Math.atan2(0, -1).toFixed(5)"), "3.14159");
    assert_eq!(eval("Math.trunc(-4.7)"), "-4");
}

#[cfg(feature = "regex")]
#[test]
fn regexp() {
    assert_eq!(eval(r"/\d+/.test('abc123')"), "true");
    assert_eq!(eval(r"/^\d+$/.test('abc')"), "false");
    assert_eq!(eval(r"'2023-11-14'.match(/(\d+)-(\d+)/)[2]"), "11");
    assert_eq!(eval(r"'a1b22c333'.match(/\d+/g).join(',')"), "1,22,333");
    assert_eq!(eval(r"'hello'.replace(/l/g, 'L')"), "heLLo");
    assert_eq!(eval(r"'a@b'.replace(/(\w)@(\w)/, '$2-$1')"), "b-a");
    assert_eq!(eval(r"'a,b;c'.split(/[,;]/).join('|')"), "a|b|c");
    assert_eq!(eval(r"'find cat'.search(/cat/)"), "5");
    assert_eq!(eval(r"new RegExp('[a-z]+', 'i').test('ABC')"), "true");
    assert_eq!(eval(r"/x/ instanceof RegExp"), "true");
    assert_eq!(eval(r"/(\w)(\w)/.exec('hi')[2]"), "i");
}

#[test]
fn promises() {
    // Basic resolution.
    assert_eq!(
        eval_global("let r; Promise.resolve(40).then(v => r = v + 2);", "r"),
        "42"
    );
    // `.then` chaining threads the return value.
    assert_eq!(
        eval_global(
            "let r; Promise.resolve(1).then(v => v + 10).then(v => v * 2).then(v => r = v);",
            "r"
        ),
        "22"
    );
    // Rejection flows to `.catch`.
    assert_eq!(
        eval_global(
            "let r; Promise.reject('boom').catch(e => r = 'caught:' + e);",
            "r"
        ),
        "caught:boom"
    );
    // The executor's resolve settles the promise.
    assert_eq!(
        eval_global("let r; new Promise(res => res(7)).then(v => r = v);", "r"),
        "7"
    );
    // A throw in the executor rejects.
    assert_eq!(
        eval_global(
            "let r; new Promise(() => { throw 'x'; }).catch(e => r = e);",
            "r"
        ),
        "x"
    );
    // Thenable adoption: a handler returning a promise is awaited.
    assert_eq!(
        eval_global(
            "let r; Promise.resolve(5).then(v => Promise.resolve(v * 4)).then(v => r = v);",
            "r"
        ),
        "20"
    );
    // Microtasks run after synchronous code.
    assert_eq!(
        eval_global(
            "let log = ''; Promise.resolve().then(() => log += 'async'); log += 'sync';",
            "log"
        ),
        "syncasync"
    );
    // A throw in a handler is caught downstream.
    assert_eq!(
        eval_global(
            "let r; Promise.resolve().then(() => { throw 'oops'; }).catch(e => r = e);",
            "r"
        ),
        "oops"
    );
    // Promise.all preserves order and mixes plain values with promises.
    assert_eq!(
        eval_global(
            "let r; Promise.all([Promise.resolve(1), 2, Promise.resolve(3)]).then(a => r = a.join(','));",
            "r"
        ),
        "1,2,3"
    );
    // Promise.all rejects with the first rejection.
    assert_eq!(
        eval_global(
            "let r; Promise.all([Promise.resolve(1), Promise.reject('e')]).catch(e => r = e);",
            "r"
        ),
        "e"
    );
    assert_eq!(
        eval_global("let r; Promise.all([]).then(a => r = a.length);", "r"),
        "0"
    );
    // Promise.race settles with the first.
    assert_eq!(
        eval_global(
            "let r; Promise.race([Promise.resolve('a'), 'b']).then(v => r = v);",
            "r"
        ),
        "a"
    );
    // instanceof Promise.
    assert_eq!(eval("Promise.resolve(1) instanceof Promise"), "true");
    // Promise.allSettled records every outcome and never rejects.
    assert_eq!(
        eval_global(
            "let r; Promise.allSettled([Promise.resolve(1), Promise.reject('e')]).then(a => r = a.map(d => d.status).join(','));",
            "r"
        ),
        "fulfilled,rejected"
    );
    // queueMicrotask runs after synchronous code.
    assert_eq!(
        eval_global(
            "let log = 's'; queueMicrotask(() => log += 'm'); log += 'y';",
            "log"
        ),
        "sym"
    );
}

#[test]
fn timers_and_event_loop() {
    // Ordering: synchronous, then microtasks, then timers by delay.
    assert_eq!(
        eval_global(
            "let log = '';
             setTimeout(() => log += 'T0', 0);
             setTimeout(() => log += 'T9', 9);
             setTimeout(() => log += 'T5', 5);
             Promise.resolve().then(() => log += 'M');
             log += 'S';",
            "log"
        ),
        "SMT0T5T9"
    );
    // Extra arguments are forwarded to the callback.
    assert_eq!(
        eval_global("let r; setTimeout((a, b) => r = a + b, 0, 3, 4);", "r"),
        "7"
    );
    // clearTimeout cancels a pending timer.
    assert_eq!(
        eval_global(
            "let r = 'no'; let id = setTimeout(() => r = 'yes', 5); clearTimeout(id);",
            "r"
        ),
        "no"
    );
    // A timer can schedule a microtask, which runs before the next timer.
    assert_eq!(
        eval_global(
            "let log = '';
             setTimeout(() => { log += 'A'; Promise.resolve().then(() => log += 'a'); }, 0);
             setTimeout(() => log += 'B', 1);",
            "log"
        ),
        "AaB"
    );
}

#[test]
fn dates() {
    // Fixed timestamps keep the tests deterministic (no `Date.now`).
    assert_eq!(
        eval("new Date(0).toISOString()"),
        "1970-01-01T00:00:00.000Z"
    );
    assert_eq!(
        eval("new Date(1700000000000).toISOString()"),
        "2023-11-14T22:13:20.000Z"
    );
    assert_eq!(eval("new Date(1700000000000).getFullYear()"), "2023");
    assert_eq!(eval("new Date(1700000000000).getMonth()"), "10"); // 0-indexed Nov
    assert_eq!(eval("new Date(1700000000000).getDate()"), "14");
    assert_eq!(eval("new Date(86400000).getTime()"), "86400000");
    assert_eq!(eval("new Date(0).getDay()"), "4"); // 1970-01-01 was Thursday
    assert_eq!(eval("typeof Date.now()"), "number");
    assert_eq!(eval("new Date(0) instanceof Date"), "true");
}

#[test]
fn constructor_statics() {
    // Number statics.
    assert_eq!(eval("Number.isInteger(5)"), "true");
    assert_eq!(eval("Number.isInteger(5.5)"), "false");
    assert_eq!(eval("Number.isNaN(0 / 0)"), "true");
    assert_eq!(eval("Number.isFinite(1 / 0)"), "false");
    assert_eq!(eval("Number.MAX_SAFE_INTEGER"), "9007199254740991");
    assert_eq!(eval("Number.parseInt('ff', 16)"), "255");
    // String statics.
    assert_eq!(eval("String.fromCharCode(72, 105)"), "Hi");
    assert_eq!(eval("String.raw`a\\tb${1}c`"), "a\\tb1c");
    // Callable-as-function still works.
    assert_eq!(eval("String(42) + Number('8') + Boolean(0)"), "428false");
    // And `new`-able where it builds an object (Map/Set).
    assert_eq!(eval("let m = new Map(); m.set('k', 1); m.get('k')"), "1");
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
    // keys / values / entries (usable in for-of and spread).
    assert_eq!(eval("[...[10, 20, 30].keys()].join(',')"), "0,1,2");
    assert_eq!(eval("[...['a', 'b'].values()].join(',')"), "a,b");
    assert_eq!(
        eval("[...['a', 'b'].entries()].map(([i, v]) => i + ':' + v).join(',')"),
        "0:a,1:b"
    );
    assert_eq!(
        eval("let out = ''; for (const [i, v] of ['x', 'y'].entries()) out += i + v; out"),
        "0x1y"
    );
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
    // substring clamps to [0, len] and swaps when start > end.
    assert_eq!(eval("'hello'.substring(1, 3)"), "el");
    assert_eq!(eval("'hello'.substring(3, 1)"), "el");
    assert_eq!(eval("'hello'.substring(2)"), "llo");
    assert_eq!(eval("'hello'.substring(-5, 3)"), "hel");
    // substr(start, length); a negative start counts from the end.
    assert_eq!(eval("'hello'.substr(1, 3)"), "ell");
    assert_eq!(eval("'hello'.substr(-2)"), "lo");
    assert_eq!(eval("'hello'.substr(2)"), "llo");
}

#[test]
fn array_object_statics() {
    assert_eq!(eval("Array.from('abc').join('-')"), "a-b-c");
    assert_eq!(eval("Array.from([1, 2, 3]).length"), "3");
    assert_eq!(eval("Array.from(new Set([1, 1, 2, 3])).join(',')"), "1,2,3");
    assert_eq!(eval("Array.of(1, 2, 3).reduce((a, b) => a + b, 0)"), "6");
    assert_eq!(
        eval("let o = Object.fromEntries([['a', 1], ['b', 2]]); o.a + o.b"),
        "3"
    );
    assert_eq!(
        eval("let o = Object.fromEntries(new Map([['x', 9]])); o.x"),
        "9"
    );
    // Object.create sets the prototype.
    assert_eq!(
        eval("let proto = { greet() { return 'hi'; } }; let o = Object.create(proto); o.greet()"),
        "hi"
    );
    // defineProperty: data and accessor descriptors.
    assert_eq!(
        eval("let o = {}; Object.defineProperty(o, 'x', { value: 42 }); o.x"),
        "42"
    );
    assert_eq!(
        eval(
            "let o = { n: 5 }; Object.defineProperty(o, 'd', { get() { return this.n * 2; } }); o.d"
        ),
        "10"
    );
    // getPrototypeOf / setPrototypeOf / getOwnPropertyNames.
    assert_eq!(
        eval("let p = { a: 1 }; let o = Object.create(p); Object.getPrototypeOf(o) === p"),
        "true"
    );
    assert_eq!(
        eval("Object.getOwnPropertyNames({ x: 1, y: 2, z: 3 }).join(',')"),
        "x,y,z"
    );
    assert_eq!(
        eval("let o = {}; Object.setPrototypeOf(o, { hi() { return 'yo'; } }); o.hi()"),
        "yo"
    );
    // structuredClone deep-copies (mutating the clone leaves the original).
    assert_eq!(
        eval(
            "let a = { x: 1, nested: { list: [1, 2] } }; let b = structuredClone(a); b.nested.list.push(3); a.nested.list.length + ':' + b.nested.list.length"
        ),
        "2:3"
    );
    assert_eq!(
        eval(
            "let m = new Map([['k', 1]]); let c = structuredClone(m); c.set('k', 9); m.get('k') + ',' + c.get('k')"
        ),
        "1,9"
    );
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
    // Indentation via the `space` argument.
    assert_eq!(
        eval("JSON.stringify({ a: 1 }, null, 2)"),
        "{\n  \"a\": 1\n}"
    );
    assert_eq!(eval("JSON.stringify([1, 2], null, 1)"), "[\n 1,\n 2\n]");
    assert_eq!(eval("JSON.stringify({}, null, 2)"), "{}"); // empty stays inline
    assert_eq!(
        eval("JSON.stringify({ a: 1 }, null, '__').split('\\n')[1]"),
        "__\"a\": 1"
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
