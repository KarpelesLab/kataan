//! Deeply nested programs and data must fail with a catchable error, never a
//! native stack overflow, and realistic nesting must work.
//!
//! The parser, the compiler, the validator and the built-ins that recurse
//! over user data are bounded by native stack headroom ([`kataan::stack`]),
//! not by a fixed level count. These tests run each case on a thread whose
//! stack size they declare, in debug builds (CI) as well as release, where a
//! level of recursion is many times larger.

use kataan::Interp;
use kataan::parser::Parser;

/// Generates a program nesting `kind` `n` levels deep. Every program prints
/// nothing and completes with a value that [`expected`] predicts.
fn program(kind: &str, n: usize) -> String {
    let r = |s: &str, k: usize| s.repeat(k);
    match kind {
        "parens" => format!("{}1{}", r("(", n), r(")", n)),
        "arrays" => format!("{}{}.length", r("[", n), r("]", n)),
        "objects" => format!("typeof {}1{}", r("({a:", n), r("})", n)),
        "calls" => format!("var f=x=>x;{}1{}", r("f(", n), r(")", n)),
        "ifs" => format!("var v=0;{}v=1;v", r("if(1)", n)),
        "elseifs" => {
            let mut s = format!("var x={n},v=0;");
            for i in 0..n {
                s.push_str(&format!("if(x=={i})v={i};else "));
            }
            s.push_str("v=1;v");
            s
        }
        "blocks" => format!("var v=0;{}v=1{}v", r("{", n), r("}", n)),
        "funcs" => format!("{}1{}", r("(function(){return ", n), r("})()", n)),
        "arrows" => format!("{}1{}", r("(()=>", n), r(")()", n)),
        "unary" => format!("{}1", r("- ", n & !1)),
        "nots" => format!("{}1", r("!", n & !1)),
        "plus" => format!("0{}", r("+0", n)),
        "strplus" => format!("(\"\"{}).length*0", r("+\"ab\"", n)),
        "ternary" => format!("{}0{}", r("1?", n), r(":1", n)),
        "assign" => format!("var a;{}0", r("a=", n)),
        "member" => format!("var o={{}};o.o=o;typeof o{}", r(".o", n)),
        "callchain" => format!("var o={{f(){{return o}}}};typeof o{}", r(".f()", n)),
        "andand" => format!("1{}", r("&&1", n)),
        "comma" => format!("(0{})", r(",0", n)),
        "template" => format!("{}0{}", r("`${", n), r("}`", n)),
        "dstr" => format!(
            "var {}a{}={}0{};a",
            r("[", n),
            r("]", n),
            r("[", n),
            r("]", n)
        ),
        "tries" => format!("var v=0;{}v=1{}v", r("try{", n), r("}finally{}", n)),
        "evalparens" => format!("eval(\"{}1{}\")", r("(", n), r(")", n)),
        "evalcalls" => format!("var f=x=>x;eval(\"{}1{}\")", r("f(", n), r(")", n)),
        _ => panic!("unknown kind {kind}"),
    }
}

fn expected(kind: &str) -> &'static str {
    match kind {
        "arrays" => "1",
        "objects" => "object",
        "member" | "callchain" => "object",
        "ifs" | "elseifs" | "blocks" | "tries" | "parens" | "calls" | "funcs" | "arrows"
        | "evalparens" | "evalcalls" | "andand" => "1",
        "unary" => "1",
        "nots" => "true",
        _ => "0",
    }
}

const KINDS: &[&str] = &[
    "parens",
    "arrays",
    "objects",
    "calls",
    "ifs",
    "elseifs",
    "blocks",
    "funcs",
    "arrows",
    "unary",
    "nots",
    "plus",
    "strplus",
    "ternary",
    "assign",
    "member",
    "callchain",
    "andand",
    "comma",
    "template",
    "dstr",
    "tries",
    "evalparens",
    "evalcalls",
];

/// Parses and runs `src`: `Ok(display)` of the completion value, `Err` with
/// the error's description on any parse/compile/run failure.
fn run(src: &str) -> Result<String, String> {
    let program = Parser::parse_program(src).map_err(|e| format!("parse: {e}"))?;
    let mut interp = Interp::new();
    match interp.run(&program) {
        Ok(v) => Ok(interp.realm().to_display_string(v)),
        Err(e) => Err(format!("run: {e:?}")),
    }
}

/// Runs `f` on a fresh thread with a `stack`-byte stack that the engine is
/// told about.
fn on_stack<T: Send + 'static>(stack: usize, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            kataan::stack::set_thread_stack_size(stack);
            f()
        })
        .expect("spawn")
        .join()
        .expect("join")
}

/// Every kind at a depth node accepts must run on the CLI's stack size.
#[test]
fn realistic_nesting_runs() {
    let out = on_stack(512 << 20, || {
        let mut bad = Vec::new();
        for &kind in KINDS {
            let n = match kind {
                // Each level is a closure call at run time; the VM's call
                // depth limit (not the parser) bounds these.
                "funcs" | "arrows" | "calls" | "evalcalls" => 1000,
                _ => 2000,
            };
            let got = run(&program(kind, n));
            if got.as_deref() != Ok(expected(kind)) {
                bad.push(format!("{kind}@{n}: {got:?}"));
            }
        }
        bad
    });
    assert!(out.is_empty(), "{out:#?}");
}

/// Far past any limit, every kind fails cleanly (or works) on a small stack:
/// the process must not abort.
#[test]
fn absurd_nesting_fails_cleanly() {
    on_stack(4 << 20, || {
        for &kind in KINDS {
            let _ = run(&program(kind, 200_000));
        }
    });
}

/// The same on a thread that never declares its stack — an embedder's
/// `std::thread::spawn` default of 2 MiB — where the engine falls back to
/// its conservative default budget.
#[test]
fn absurd_nesting_fails_cleanly_on_an_undeclared_default_stack() {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            for &kind in KINDS {
                let _ = run(&program(kind, 200_000));
            }
        })
        .expect("spawn")
        .join()
        .expect("join");
}

/// Prints the native stack each level costs (`cargo test --test deep_nesting
/// -- --ignored --nocapture stack_per_level`).
#[test]
#[ignore = "measurement, not a check"]
fn stack_per_level() {
    on_stack(1 << 30, || {
        const BUDGET: usize = 16 << 20;
        println!(
            "size_of Expr {} Stmt {}",
            size_of::<kataan::ast::Expr>(),
            size_of::<kataan::ast::Stmt>()
        );
        let only = std::env::var("DEEP_KINDS").unwrap_or_default();
        for &kind in KINDS {
            if !only.is_empty() && !only.split(',').any(|k| k == kind) {
                continue;
            }
            let fits = |n: usize| {
                kataan::stack::set_stack_limit(kataan::stack::current_sp() - BUDGET);
                let ok = Parser::parse_program(&program(kind, n)).is_ok();
                let full = ok && run(&program(kind, n)).as_deref() == Ok(expected(kind));
                (ok, full)
            };
            let max = |want: &dyn Fn((bool, bool)) -> bool| {
                let (mut lo, mut hi) = (
                    0usize,
                    if matches!(kind, "arrows" | "funcs") {
                        600
                    } else {
                        1usize << 16
                    },
                );
                while lo < hi {
                    let m = (lo + hi).div_ceil(2);
                    if want(fits(m)) { lo = m } else { hi = m - 1 }
                }
                lo
            };
            let p = max(&|r| r.0);
            let f = if std::env::var_os("DEEP_PARSE_ONLY").is_some() {
                0
            } else {
                max(&|r| r.1)
            };
            println!(
                "{kind:>10}: parse {p:>7} levels ({:>6} B/level)  run {f:>7} levels",
                BUDGET / p.max(1)
            );
        }
    });
}

/// Recursion that user *data* drives through built-ins (no deep source):
/// cases node answers with a RangeError must throw one, never abort, and
/// deep but valid JSON must parse (iteratively, bounded by memory).
const DATA_CASES: &str = r#"
function k(f) { try { return String(f()); } catch (e) { return e.name; } }
function deepArr(n) { let a = []; for (let i = 0; i < n; i++) a = [a]; return a; }
function deepObj(n) { let o = {}; for (let i = 0; i < n; i++) o = { a: o }; return o; }
[
  k(() => String(deepArr(100000)).length),
  k(() => deepArr(100000).join("-").length),
  k(() => deepArr(40).join("-").length),
  k(() => deepArr(100000).flat(Infinity).length),
  k(() => { let o = { toString() { return "x"; } }; for (let i = 0; i < 100000; i++) { const p = o; o = { toString() { return String(p); } }; } return String(o); }),
  k(() => { let o = { valueOf() { return 1; } }; for (let i = 0; i < 100000; i++) { const p = o; o = { valueOf() { return +p + 1; } }; } return +o; }),
  k(() => { let v = JSON.parse("[".repeat(1e5) + "]".repeat(1e5)), d = 0; while (v.length) { v = v[0]; d++; } return d; }),
  k(() => { let v = JSON.parse('{"a":'.repeat(1e5) + "0" + "}".repeat(1e5)), d = 0; while (typeof v === "object") { v = v.a; d++; } return d; }),
  k(() => { let v = JSON.parse('[{"k":'.repeat(5e4) + "1" + "}]".repeat(5e4)), d = 0; while (Array.isArray(v)) { v = v[0].k; d++; } return d + ":" + v; }),
  k(() => JSON.parse("[".repeat(1e5) + "]".repeat(1e5 - 1))),
  k(() => JSON.stringify(JSON.parse('{"a\\n":[1,{"b":[]}],"c":{"d":{}},"a\\n":2}'))),
  k(() => JSON.stringify(deepObj(100000)).length),
  k(() => { let o = {}; for (let i = 0; i < 1e5; i++) o = Object.create(o); return o.zz + ":" + ("zz" in o); }),
].join(" ")
"#;

#[test]
fn deep_data_recursion_is_a_range_error_and_deep_json_parses() {
    let got = on_stack(256 << 20, || run(DATA_CASES));
    assert_eq!(
        got.as_deref(),
        Ok(
            "RangeError RangeError 0 RangeError RangeError RangeError 99999 100000 50000:1 \
            SyntaxError {\"a\\n\":2,\"c\":{\"d\":{}}} RangeError undefined:false"
        )
    );
    // On an undeclared default-size thread the answers may come sooner, but
    // nothing aborts.
    let small = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| run(DATA_CASES).is_ok())
        .expect("spawn")
        .join()
        .expect("join");
    assert!(small);
}

/// Code that needs more registers than a frame has (each pending call of
/// `f(f(f(…)))` holds its callee and arguments) is a catchable `RangeError`
/// from `eval`, not an uncatchable engine fault.
#[test]
fn register_exhaustion_is_a_catchable_range_error() {
    let src = r#"
        var f = x => x;
        var src = "f(".repeat(20000) + "1" + ")".repeat(20000);
        try { eval(src); "no error" } catch (e) { e.constructor.name }
    "#;
    // A debug build's parser frames are large enough that its nesting limit
    // (a SyntaxError) comes first; either way the error is catchable.
    let want = if cfg!(debug_assertions) {
        "SyntaxError"
    } else {
        "RangeError"
    };
    assert_eq!(on_stack(512 << 20, || run(src)).as_deref(), Ok(want));
}
