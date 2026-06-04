//! New-model conformance harness: runs each self-checking `.js` fixture in
//! `testdata/conformance/` through the **new representation** interpreter
//! (`kataan::nbexec`, over `Realm`/`NanBox`) rather than the production
//! tree-walker.
//!
//! This quantifies how much of real JavaScript the new model executes — the
//! concrete coverage metric behind the Phase-D migration (see `ROADMAP.md`). A
//! fixture passes if it runs to completion without an uncaught throw; the
//! `assert`/`assertEq` helpers are supplied as a small JS prelude (the new-model
//! interpreter has no host-native injection yet, but it has functions, `throw`,
//! and strict equality, which is all the harness needs).
//!
//! The asserted threshold ratchets up as the new model gains coverage; the
//! per-fixture breakdown is printed so regressions and new passes are visible.

use kataan::nbexec::Interp;
use kataan::parser::Parser;

/// JS prelude defining the assertion helpers the fixtures rely on.
const PRELUDE: &str = r#"
function assert(c, m) { if (!c) { throw 'AssertionError: ' + (m || 'assertion failed'); } }
function assertEq(a, b) { if (a !== b) { throw 'AssertionError: ' + a + ' !== ' + b; } }
"#;

/// Runs one fixture through the new model; `Ok` iff it completed without an
/// uncaught throw (or other execution error).
fn run_fixture(source: &str) -> Result<(), String> {
    let combined = alloc_combined(source);
    let program = Parser::parse_program(&combined).map_err(|e| format!("parse: {e}"))?;
    let mut interp = Interp::new();
    interp.run(&program).map_err(|e| format!("exec: {e:?}"))?;
    Ok(())
}

fn alloc_combined(source: &str) -> String {
    let mut s = String::with_capacity(PRELUDE.len() + source.len() + 1);
    s.push_str(PRELUDE);
    s.push('\n');
    s.push_str(source);
    s
}

/// Each fixture and whether the new model is currently expected to run it. As
/// the new interpreter gains features, flip `false → true` here (and the total
/// passing count below rises).
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
fn new_model_conformance_coverage() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/conformance/");
    let mut passed = 0;
    let mut results = Vec::new();
    for name in FIXTURES {
        let path = format!("{dir}{name}");
        let source = std::fs::read_to_string(&path).expect("read fixture");
        match run_fixture(&source) {
            Ok(()) => {
                passed += 1;
                results.push(format!("  PASS  {name}"));
            }
            Err(e) => results.push(format!("  fail  {name}  ({e})")),
        }
    }
    eprintln!(
        "new-model conformance: {passed}/{} fixtures pass\n{}",
        FIXTURES.len(),
        results.join("\n")
    );
    // Ratchet: the new model runs the entire real-world conformance corpus.
    // Never let it regress.
    assert!(
        passed >= 10,
        "new-model conformance regressed: only {passed} fixtures pass"
    );
}
