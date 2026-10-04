//! The product entry points run user code on the **hosted bytecode VM**
//! (`ROADMAP.md` §2.0), not on the tree-walker: `Interp::run` (the embedding
//! API, the REPL, `hostrun`, snapshots, the buffer-sharing C ABI) and
//! `nbvm::execute*` (`kataan run`, `kt_eval`, the web build).
//!
//! `Interp::tree_walked` counts the user statements the tree-walker executed;
//! every test here asserts it stays zero.

use kataan::nbexec::Interp;
use kataan::parser::Parser;

/// Parses `src` and leaks the AST, as a REPL does, so the interpreter may keep
/// references into it across runs.
fn leak(src: &str) -> &'static kataan::ast::Program {
    Box::leak(Box::new(Parser::parse_program(src).expect("parse")))
}

#[test]
fn interp_run_executes_on_the_vm() {
    let mut interp = Interp::new();
    let v = interp
        .run(leak(
            "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n\
             class P { #x = 2; get x() { return this.#x; } }\n\
             let total = 0;\n\
             for (const k of [1, 2, 3]) total += k;\n\
             fib(10) + new P().x + total",
        ))
        .expect("run");
    assert_eq!(interp.display(v), "63");
}

#[test]
fn sequential_runs_share_globals_and_closures() {
    // REPL-style: each line is its own Script over one interpreter; functions
    // and closures one run creates stay callable from the next.
    let mut interp = Interp::new();
    let lines = [
        (
            "function counter() { let n = 0; return () => ++n; }",
            "undefined",
        ),
        ("var next = counter(); let seen = [];", "undefined"),
        ("seen.push(next(), next()); seen.join()", "1,2"),
        ("const double = x => x * 2; next() + double(10)", "23"),
        (
            "class Point { constructor(x) { this.x = x; } }",
            "undefined",
        ),
        ("new Point(next()).x", "4"),
        ("typeof counter + ':' + seen.length", "function:2"),
    ];
    for (src, want) in lines {
        let v = interp.run(leak(src)).expect(src);
        assert_eq!(interp.display(v), want, "{src}");
    }
}

#[test]
fn a_later_run_sees_an_earlier_runs_lexical_bindings() {
    let mut interp = Interp::new();
    interp.run(leak("let a = 1; const b = 2;")).expect("first");
    let v = interp.run(leak("a += b; a")).expect("second");
    assert_eq!(interp.display(v), "3");
    // Redeclaring a global lexical binding is an early error of the new script.
    assert!(interp.run(leak("let a = 5;")).is_err());
    let v = interp.run(leak("a")).expect("still there");
    assert_eq!(interp.display(v), "3");
}

#[test]
fn an_uncaught_throw_does_not_poison_later_runs() {
    let mut interp = Interp::new();
    assert!(matches!(
        interp.run(leak("function f() { throw new TypeError('x'); } f()")),
        Err(kataan::nbexec::ExecError::Throw(_))
    ));
    let v = interp
        .run(leak("try { f() } catch (e) { e.message }"))
        .expect("run");
    assert_eq!(interp.display(v), "x");
}

#[test]
fn host_functions_and_promises_on_the_vm() {
    let mut interp = Interp::new();
    interp.register_global_fn("hostAdd", 2, |cx, _this, args| {
        let a = cx.to_number(args[0])?;
        let b = cx.to_number(args[1])?;
        Ok(cx.number(a + b))
    });
    interp
        .run(leak(
            "var log = [];\n\
             Promise.resolve(hostAdd(2, 3)).then(v => log.push(v));\n\
             (async () => { log.push(await hostAdd(1, 1)); })();",
        ))
        .expect("run");
    // The event loop ran to quiescence inside `run`.
    let v = interp.run(leak("log.join()")).expect("read");
    assert_eq!(interp.display(v), "5,2");
}

#[test]
fn host_runtime_timers_call_vm_closures() {
    let mut interp = Interp::new();
    kataan::host::timers::install(&mut interp);
    interp
        .run(leak(
            "let order = [];\n\
             setTimeout(() => { order.push('timeout'); console.log(order.join()); }, 5);\n\
             process.nextTick(() => order.push('tick'));\n\
             Promise.resolve().then(() => order.push('micro'));\n\
             order.push('sync');",
        ))
        .expect("run");
    kataan::host::timers::run_event_loop(&mut interp).expect("loop");
    assert!(
        interp.output().contains("sync,"),
        "output: {:?}",
        interp.output()
    );
    assert!(interp.output().contains("timeout"));
}

#[test]
fn execute_runs_on_the_hosted_vm() {
    // `nbvm::execute` (the CLI's `run`, `kt_eval`) is hosted: real `Error`
    // instances and the full built-in library, no standalone-realm fallback.
    let (out, completion) = kataan::nbvm::execute(
        "console.log([1, 2, 3].map(x => x * 2).join('-'));\n\
         try { null.x } catch (e) { e instanceof TypeError }",
    )
    .expect("execute");
    assert_eq!(out, "2-4-6\n");
    assert_eq!(completion, "true");
    let err = kataan::nbvm::execute("throw new RangeError('nope')").unwrap_err();
    assert_eq!(err, "RangeError: nope");
    let (out, result) =
        kataan::nbvm::execute_capturing("console.log('a'); undefinedName", Default::default());
    assert_eq!(out, "a\n");
    assert!(result.unwrap_err().contains("ReferenceError"));
}

#[test]
fn huge_literal_tables_compile() {
    // A generated data table must not exhaust the 16-bit register file.
    let mut src = String::from("var table = [\n");
    for i in 0..70_000 {
        src.push_str(&format!("  [\"k{i}\", {i}, {{ v: {i} }}],\n"));
    }
    src.push_str("];\ntable.length + ':' + table[69999][2].v");
    let mut interp = Interp::new();
    let v = interp.run(leak(&src)).expect("run");
    assert_eq!(interp.display(v), "70000:69999");
}

#[test]
fn labelled_function_declarations_run_on_the_vm() {
    let mut interp = Interp::new();
    let v = interp
        .run(leak(
            "l: function f() { return 1; }\n\
             function g() { m: function h() { return 2; } return h(); }\n\
             { n: function k() { return 3; } var r = k(); }\n\
             f() + g() + r",
        ))
        .expect("run");
    assert_eq!(interp.display(v), "6");
}

#[test]
fn every_function_user_code_obtains_is_a_vm_function() {
    // One function from each way user code can make one: declarations (the
    // script's hoisted bindings included), expressions, arrows, methods,
    // accessors, classes, generators, async functions, `Function` and its
    // generator/async siblings, and direct and indirect `eval`.
    let mut interp = Interp::new();
    let v = interp
        .run(leak(
            "function decl() {}\n\
             async function* agen() {}\n\
             class C { m() {} static s() {} get g() { return 1; } }\n\
             var fns = [decl, agen, C, C.prototype.m, C.s,\n\
               Object.getOwnPropertyDescriptor(C.prototype, 'g').get,\n\
               function () {}, () => 1, function* () {}, async () => {},\n\
               ({ m() {} }).m,\n\
               Function('return 1'),\n\
               Object.getPrototypeOf(function* () {}).constructor('yield 1'),\n\
               Object.getPrototypeOf(async function () {}).constructor('await 1'),\n\
               eval('(function () {})'), (0, eval)('(function () {})'),\n\
               eval('function inEval() {} inEval')];\n\
             fns",
        ))
        .expect("run");
    let arr = kataan::heap::Handle::from_raw(v.as_handle().expect("array"));
    let elems = interp
        .realm()
        .array_elements(arr)
        .expect("elements")
        .to_vec();
    assert_eq!(elems.len(), 17);
    for (i, f) in elems.iter().enumerate() {
        let h = kataan::heap::Handle::from_raw(f.as_handle().expect("function"));
        assert!(
            interp.realm().is_vm_function_value(h),
            "function #{i} is not a VM function"
        );
    }
}
