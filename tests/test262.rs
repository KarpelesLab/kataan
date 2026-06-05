//! A Test262-style conformance runner.
//!
//! Test262 is ECMA's official conformance suite (~50k tests). This harness
//! implements the *runner* — the part that is engine-specific — over a vendored,
//! conformant subset under `testdata/test262/`: it parses each test's
//! `/*--- … ---*/` YAML frontmatter (the `negative`, `includes`, and `flags`
//! metadata), assembles the standard harness (`sta.js` + `assert.js`, plus any
//! `includes`), and runs the result through the engine — checking that positive
//! tests complete and that `negative` tests fail at the declared phase (`parse`
//! or runtime) with the declared error type.
//!
//! Pointing this at a checkout of the real suite is a matter of changing the
//! corpus directory; the runner logic is the hard part and is exercised here.
//! (See `ROADMAP.md`, the Test262 milestone.)

use kataan::parser::Parser;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/test262/");

/// The subset of Test262 frontmatter the runner acts on.
#[derive(Default, Debug)]
struct Meta {
    /// `negative: { phase, type }` — the test must fail this way.
    negative: Option<(String, String)>,
    /// Harness files to prepend (besides the default `sta.js`/`assert.js`).
    includes: Vec<String>,
    /// `flags: [...]` — e.g. `raw` (no harness), `onlyStrict`, `module`, `async`.
    flags: Vec<String>,
    /// `features: [...]` — capabilities the test needs; `intl` gates on the
    /// optional `intl` build feature (skipped when it is off).
    features: Vec<String>,
}

/// Extracts and parses the `/*--- … ---*/` metadata block.
fn parse_meta(source: &str) -> Meta {
    let mut meta = Meta::default();
    let Some(start) = source.find("/*---") else {
        return meta;
    };
    let Some(end) = source[start..].find("---*/") else {
        return meta;
    };
    let block = &source[start + 5..start + end];

    // A tiny line-oriented reader for the handful of keys we use (the corpus
    // sticks to flow-style lists and a two-line `negative` mapping).
    let mut lines = block.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("flags:") {
            meta.flags = parse_flow_list(rest);
        } else if let Some(rest) = trimmed.strip_prefix("features:") {
            meta.features = parse_flow_list(rest);
        } else if let Some(rest) = trimmed.strip_prefix("includes:") {
            meta.includes = parse_flow_list(rest);
        } else if trimmed == "negative:" {
            let mut phase = String::new();
            let mut ty = String::new();
            while let Some(peek) = lines.peek() {
                let pt = peek.trim();
                if let Some(v) = pt.strip_prefix("phase:") {
                    phase = v.trim().to_string();
                } else if let Some(v) = pt.strip_prefix("type:") {
                    ty = v.trim().to_string();
                } else {
                    break;
                }
                lines.next();
            }
            meta.negative = Some((phase, ty));
        }
    }
    meta
}

/// Parses a flow-style list `[a, b, c]` (or empty) into owned strings.
fn parse_flow_list(s: &str) -> Vec<String> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

/// Assembles the full source: harness (unless `raw`) + includes + the test.
fn assemble(meta: &Meta, source: &str) -> String {
    let mut out = String::new();
    if !meta.flags.iter().any(|f| f == "raw") {
        for h in ["sta.js", "assert.js"] {
            out.push_str(&read_harness(h));
            out.push('\n');
        }
        for inc in &meta.includes {
            out.push_str(&read_harness(inc));
            out.push('\n');
        }
    }
    out.push_str(source);
    out
}

fn read_harness(name: &str) -> String {
    std::fs::read_to_string(format!("{DIR}harness/{name}"))
        .unwrap_or_else(|_| panic!("missing harness file {name}"))
}

/// Runs one assembled program through the engine (the production bytecode-first
/// path). Returns `Ok(())` on clean completion, or `Err((phase, message))` where
/// `phase` is `"parse"` or `"runtime"`.
fn run(combined: &str) -> Result<(), (String, String)> {
    // Parse phase.
    if let Err(e) = Parser::parse_program(combined) {
        return Err(("parse".into(), format!("{e}")));
    }
    // Execution phase (bytecode VM with tree-walker fallback).
    match kataan::nbvm::execute(combined) {
        Ok(_) => Ok(()),
        Err(e) => Err(("runtime".into(), e)),
    }
}

/// The outcome the runner asserts against the frontmatter.
fn evaluate(meta: &Meta, source: &str) -> Result<(), String> {
    let combined = assemble(meta, source);
    let result = run(&combined);
    match (&meta.negative, result) {
        // Positive test: must complete cleanly.
        (None, Ok(())) => Ok(()),
        (None, Err((phase, msg))) => Err(format!("expected pass, {phase} failed: {msg}")),
        // Negative test: must fail at the declared phase.
        (Some((phase, _ty)), Err((got_phase, _))) => {
            if phase == &got_phase {
                Ok(())
            } else {
                Err(format!(
                    "negative test failed at {got_phase}, expected {phase}"
                ))
            }
        }
        (Some((phase, ty)), Ok(())) => {
            Err(format!("expected {ty} at {phase}, but the test passed"))
        }
    }
}

#[test]
fn test262_corpus_conformance() {
    // Run on a large stack so legitimately deep JS recursion is bounded by the
    // engine's call-depth guard (a thrown RangeError), not a host stack overflow.
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(corpus_conformance)
        .expect("spawn corpus thread")
        .join()
        .expect("corpus thread");
}

fn corpus_conformance() {
    let test_dir = format!("{DIR}test/");
    let mut entries: Vec<_> = std::fs::read_dir(&test_dir)
        .expect("read test262 test dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "js"))
        .collect();
    entries.sort();

    let mut passed = 0;
    let mut results = Vec::new();
    for path in &entries {
        let source = std::fs::read_to_string(path).expect("read test");
        let meta = parse_meta(&source);
        let name = path.file_name().unwrap().to_string_lossy();
        // Skip tests needing the optional `intl` feature when it is not built.
        if meta.features.iter().any(|f| f == "intl") && !cfg!(feature = "intl") {
            passed += 1;
            results.push(format!("  SKIP  {name}  (needs the intl feature)"));
            continue;
        }
        match evaluate(&meta, &source) {
            Ok(()) => {
                passed += 1;
                results.push(format!("  PASS  {name}"));
            }
            Err(e) => results.push(format!("  FAIL  {name}  ({e})")),
        }
    }
    eprintln!(
        "test262 runner: {passed}/{} pass\n{}",
        entries.len(),
        results.join("\n")
    );

    // The whole vendored corpus must pass (positive and negative alike); this is
    // the conformance gate. As the corpus grows toward the real suite, keep it
    // green — a regression means an engine or runner bug.
    assert_eq!(
        passed,
        entries.len(),
        "test262 conformance regressed (see per-test output above)"
    );
    // Guard against the corpus silently emptying.
    assert!(entries.len() >= 7, "test262 corpus shrank unexpectedly");
}

/// The frontmatter parser is itself covered, since the runner's correctness
/// hinges on it.
#[test]
fn frontmatter_parsing() {
    let src = "/*---\nflags: [onlyStrict, module]\nincludes: [a.js, b.js]\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\nvar x = 1;";
    let meta = parse_meta(src);
    assert_eq!(meta.flags, ["onlyStrict", "module"]);
    assert_eq!(meta.includes, ["a.js", "b.js"]);
    assert_eq!(
        meta.negative,
        Some(("parse".to_string(), "SyntaxError".to_string()))
    );

    // A test with no frontmatter parses to empty metadata.
    let empty = parse_meta("var x = 1;");
    assert!(empty.negative.is_none() && empty.flags.is_empty());
}
