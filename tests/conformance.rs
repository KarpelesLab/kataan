//! Conformance harness: runs each self-checking `.js` fixture in
//! `testdata/conformance/` through the interpreter. A fixture passes if it runs
//! to completion without an uncaught throw; the fixtures assert expected
//! behavior via the injected `assert(cond, msg)` and `assertEq(actual,
//! expected)` helpers.
//!
//! This is the seed of the eventual Test262 integration (see `ROADMAP.md`):
//! the same runner shape will drive the real suite once the engine is
//! conformant enough to report a meaningful pass-rate.

use kataan::interp::{Interp, NativeFn, Value, strict_equals};
use kataan::parser::Parser;
use std::rc::Rc;

/// Builds a native function value from a name and a callback.
fn native<'a>(
    name: &'static str,
    f: impl Fn(&[Value<'a>]) -> Result<Value<'a>, Value<'a>> + 'a,
) -> Value<'a> {
    Value::Native(Rc::new(NativeFn {
        name,
        call: Box::new(f),
    }))
}

/// Installs the `assert` / `assertEq` harness (plus a no-op `console.log`) into
/// the interpreter's globals.
fn install_harness(interp: &Interp<'_>) {
    interp.define_global(
        "assert",
        native("assert", |args| {
            let cond = args.first().is_some_and(Value::to_boolean);
            if cond {
                Ok(Value::Undefined)
            } else {
                let msg = args
                    .get(1)
                    .map_or_else(|| "assertion failed".to_string(), Value::to_js_string);
                Err(Value::str(format!("AssertionError: {msg}")))
            }
        }),
    );
    interp.define_global(
        "assertEq",
        native("assertEq", |args| {
            let a = args.first().cloned().unwrap_or(Value::Undefined);
            let b = args.get(1).cloned().unwrap_or(Value::Undefined);
            if strict_equals(&a, &b) {
                Ok(Value::Undefined)
            } else {
                Err(Value::str(format!(
                    "AssertionError: {:?} !== {:?}",
                    a.to_js_string(),
                    b.to_js_string()
                )))
            }
        }),
    );
    interp.define_global("console", {
        let log = native("log", |_args| Ok(Value::Undefined));
        let console = kataan::interp::Obj::object();
        console.set("log", log);
        Value::Object(console)
    });
}

/// Parses and runs one fixture through both execution paths — the tree-walker
/// and the bytecode VM (which falls back to the tree-walker for unsupported
/// constructs) — returning an uncaught throw / mismatch message on failure.
fn run_fixture(source: &str) -> Result<(), String> {
    let program = Parser::parse_program(source).map_err(|e| e.to_string())?;

    // Tree-walker (the reference).
    let mut tree = Interp::new();
    install_harness(&tree);
    tree.run(&program)
        .map_err(|thrown| format!("tree-walker: {}", thrown.to_js_string()))?;

    // Bytecode VM (with automatic tree-walker fallback for unsupported syntax).
    let mut vm = Interp::new();
    install_harness(&vm);
    vm.run_with_vm(&program)
        .map_err(|thrown| format!("vm: {}", thrown.to_js_string()))?;

    Ok(())
}

#[test]
fn conformance_fixtures() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/conformance");
    let mut ran = 0;
    let mut failures = Vec::new();

    for entry in std::fs::read_dir(dir).expect("read conformance dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("js") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read fixture");
        ran += 1;
        if let Err(message) = run_fixture(&source) {
            failures.push(format!("{}: {message}", path.display()));
        }
    }

    assert!(ran > 0, "no conformance fixtures found in {dir}");
    assert!(
        failures.is_empty(),
        "{}/{} conformance fixtures failed:\n{}",
        failures.len(),
        ran,
        failures.join("\n")
    );
    eprintln!("conformance: {ran}/{ran} fixtures passed");
}
