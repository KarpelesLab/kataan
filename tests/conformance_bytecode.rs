//! Bytecode-VM coverage harness: runs each self-checking `.js` fixture in
//! `testdata/conformance/` through the **bytecode VM** (`kataan::nbvm::execute`,
//! the primary engine with a tree-walker fallback), and separately measures how
//! many fixtures the bytecode compiler handles *fully* — i.e. compile without
//! routing to the tree-walker.
//!
//! The first assertion guards correctness (the cutover must not change results);
//! the second is a ratcheting coverage metric for the fold — as more of the
//! language compiles to bytecode, more fixtures run on the fast path. (See
//! `ROADMAP.md` Phase D and the Test262 milestone.)

use kataan::parser::Parser;

/// JS prelude defining the assertion helpers the fixtures rely on (as plain
/// functions, so the bytecode compiler can lower them).
const PRELUDE: &str = r#"
function assert(c, m) { if (!c) { throw 'AssertionError: ' + (m || 'assertion failed'); } }
function assertEq(a, b) { if (a !== b) { throw 'AssertionError: ' + a + ' !== ' + b; } }
"#;

const FIXTURES: &[&str] = &[
    "language.js",
    "functions_classes.js",
    "builtins.js",
    "advanced.js",
    "regexp.js",
    "vm_features.js",
    "vm_modern.js",
    "vm_closures.js",
    "vm_classes.js",
    "vm_realworld.js",
];

#[test]
fn bytecode_runs_and_compiles_real_fixtures() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/conformance/");
    let mut ran_ok = 0;
    let mut compiled_fully = 0;
    let mut results = Vec::new();
    for name in FIXTURES {
        let source = std::fs::read_to_string(format!("{dir}{name}")).expect("read fixture");
        let combined = format!("{PRELUDE}\n{source}");

        // 1) Correctness: it must run (bytecode or fallback) without an uncaught
        //    throw.
        let ran = kataan::nbvm::execute(&combined).is_ok();
        if ran {
            ran_ok += 1;
        }

        // 2) Coverage: does the whole program compile to bytecode (no fallback)?
        let program = Parser::parse_program(&combined).expect("parse");
        let fully = kataan::nbvm::compile_program(&program).is_ok();
        if fully {
            compiled_fully += 1;
        }
        results.push(format!(
            "  {} run={} bytecode={}",
            name,
            if ran { "ok" } else { "FAIL" },
            if fully { "full" } else { "fallback" },
        ));
    }
    eprintln!(
        "bytecode harness: {ran_ok}/{} run ok, {compiled_fully}/{} compile fully to bytecode\n{}",
        FIXTURES.len(),
        FIXTURES.len(),
        results.join("\n"),
    );

    // Every fixture must still run correctly via the bytecode-first engine —
    // the cutover's safety net (bytecode where it compiles, tree-walker
    // otherwise).
    assert_eq!(ran_ok, FIXTURES.len(), "a fixture failed to run");
    // Ratchet: at least this many whole real-world programs compile fully to
    // bytecode (no fallback). Raise as the fold widens; never let it regress.
    assert!(
        compiled_fully >= 9,
        "bytecode coverage regressed: only {compiled_fully} fixtures compile fully"
    );
}
