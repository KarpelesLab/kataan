//! The **official** Test262 conformance runner.
//!
//! Unlike the fast curated gate in `tests/test262.rs` (a small vendored subset
//! that must stay 100% green), this runner executes the real tc39/test262 corpus
//! pinned as a git submodule at `vendor/test262/` (~53k tests). It is the
//! instrument we use to measure and drive true ECMAScript conformance.
//!
//! It is `#[ignore]`d so the default `cargo test` PR gate stays fast; run it
//! explicitly:
//!
//! ```text
//! git submodule update --init --depth 1 vendor/test262
//! cargo test --test test262_official -- --ignored --nocapture
//! ```
//!
//! ## What it implements (a conformant runner, not a toy)
//! - Full `/*--- … ---*/` frontmatter: `negative {phase,type}`, `includes`,
//!   `flags`, `features` (flow- and block-style lists).
//! - **Negative tests check the error *type***, not just the failure phase.
//! - **Strict + sloppy dual execution** per `flags` (`onlyStrict`/`noStrict`/
//!   `raw`); a test passes only if every required mode passes.
//! - **`async`** tests: inject `print`, include `doneprintHandle.js`, and require
//!   the `Test262:AsyncTestComplete` sentinel (and absence of `…Failure`).
//! - A `$262` host object + `print` global, injected as a JS prelude.
//! - **Feature/path skip gating** for areas the engine does not implement yet
//!   (modules, Temporal, intl402 without the `intl` feature, staging, agents…).
//!
//! ## Status ledger (regression gate)
//! `tests/test262-status.txt` records the *known-failing* tests (paths relative
//! to `vendor/test262/test/`). The runner fails if any test fails that is **not**
//! in the ledger (a regression), and reports ledger entries that now pass (stale —
//! remove them). Regenerate the ledger from the current run with
//! `KATAAN_TEST262_BLESS=1`. The ledger's target is **empty** (literal 100%).

use std::collections::HashSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use kataan::limits::Limits;
use kataan::nbexec::ErrorPhase;

/// Env var that puts the test binary into *worker* mode: `"start:step:outpath"`.
const WORKER_ENV: &str = "KATAAN_T262_WORKER";

/// Repo-root-relative location of the pinned submodule corpus.
fn corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/test262")
}

/// Path to the checked-in known-failures ledger.
fn ledger_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test262-status.txt")
}

/// JS prelude providing the `print` global and a best-effort `$262` host object.
/// The native-only `$262` members that the engine cannot yet support throw, so
/// tests needing them fail into the ledger rather than passing spuriously.
const HOST_PRELUDE: &str = r#"
var print = function () { var s = ''; for (var i = 0; i < arguments.length; i++) { if (i) s += ' '; s += arguments[i]; } console.log(s); };
var $262 = {
  global: this,
  gc: function () {},
  evalScript: function (src) { return (0, eval)(src); },
  detachArrayBuffer: function (b) { return $262_detachArrayBuffer(b); },
  createRealm: function () { throw new TypeError('$262.createRealm is not supported'); },
  IsHTMLDDA: undefined
};
"#;

/// `features:` tags whose tests we skip until the corresponding engine support
/// lands. (intl402 is gated separately on the build feature.)
const SKIP_FEATURES: &[&str] = &[
    // "Temporal", // implemented (ZonedDateTime/Now skipped via path-check above)
    "decorators",
    "tail-call-optimization",
    "import-assertions",
    "import-attributes",
    // Atomics / SharedArrayBuffer: the single-agent deterministic core is
    // implemented (namespace, ops, SAB construct/grow/slice), so non-gated probes
    // like `Atomics.pause` already pass. The bulk stays skipped pending the
    // concurrent part (wait/notify + `$262.agent` threading) and BigInt Atomics —
    // a local `KATAAN_TEST262_FILTER=Atomics` run currently shows Atomics 34% /
    // SharedArrayBuffer 79%, too many failures to un-skip cleanly yet.
    "Atomics.waitAsync",
    "cross-realm",
    "IsHTMLDDA",
];

/// The subset of frontmatter the runner acts on.
#[derive(Default, Debug, Clone)]
struct Meta {
    negative: Option<(String, String)>, // (phase, type)
    includes: Vec<String>,
    flags: Vec<String>,
    features: Vec<String>,
}

impl Meta {
    fn has_flag(&self, f: &str) -> bool {
        self.flags.iter().any(|x| x == f)
    }
}

/// Parses the `/*--- … ---*/` YAML frontmatter (flow- and block-style lists, and
/// the `negative` mapping).
fn parse_meta(source: &str) -> Meta {
    let mut meta = Meta::default();
    let Some(start) = source.find("/*---") else {
        return meta;
    };
    let Some(rel_end) = source[start..].find("---*/") else {
        return meta;
    };
    let block = &source[start + 5..start + rel_end];

    let lines: Vec<&str> = block.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let collect_list = |key: &str| -> Option<&str> { trimmed.strip_prefix(key) };

        if let Some(rest) = collect_list("flags:") {
            meta.flags = parse_list(rest, &lines, &mut i);
        } else if let Some(rest) = collect_list("features:") {
            meta.features = parse_list(rest, &lines, &mut i);
        } else if let Some(rest) = collect_list("includes:") {
            meta.includes = parse_list(rest, &lines, &mut i);
        } else if trimmed == "negative:" {
            let mut phase = String::new();
            let mut ty = String::new();
            let mut j = i + 1;
            while j < lines.len() {
                let pt = lines[j].trim();
                if let Some(v) = pt.strip_prefix("phase:") {
                    phase = v.trim().to_string();
                } else if let Some(v) = pt.strip_prefix("type:") {
                    ty = v.trim().to_string();
                } else {
                    break;
                }
                j += 1;
            }
            i = j - 1;
            meta.negative = Some((phase, ty));
        }
        i += 1;
    }
    meta
}

/// Parses a list value that may be flow-style (`[a, b]` on the same line) or
/// block-style (`- a` / `- b` on following indented lines). Advances `i` past any
/// consumed block lines.
fn parse_list(same_line: &str, lines: &[&str], i: &mut usize) -> Vec<String> {
    let s = same_line.trim();
    if s.starts_with('[') {
        return s
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect();
    }
    if !s.is_empty() {
        // Scalar on the same line (rare for these keys) — treat as one item.
        return vec![s.to_string()];
    }
    // Block style: consume following `- item` lines.
    let mut out = Vec::new();
    let mut j = *i + 1;
    while j < lines.len() {
        let t = lines[j].trim();
        if let Some(item) = t.strip_prefix("- ") {
            out.push(item.trim().to_string());
            j += 1;
        } else {
            break;
        }
    }
    *i = j - 1;
    out
}

/// Reads every harness file once into memory.
fn load_harness(root: &Path) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let dir = root.join("harness");
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "js")
                && let Ok(s) = std::fs::read_to_string(&p)
            {
                map.insert(p.file_name().unwrap().to_string_lossy().into_owned(), s);
            }
        }
    }
    map
}

/// Assembles the full program for one mode: optional strict directive, host
/// prelude, harness (`sta.js` + `assert.js` + async/includes), then the test.
fn assemble(
    strict: bool,
    harness: &std::collections::HashMap<String, String>,
    meta: &Meta,
    src: &str,
) -> String {
    let mut out = String::new();
    if meta.has_flag("raw") {
        out.push_str(src);
        return out;
    }
    if strict {
        out.push_str("\"use strict\";\n");
    }
    out.push_str(HOST_PRELUDE);
    for h in ["sta.js", "assert.js"] {
        if let Some(s) = harness.get(h) {
            out.push_str(s);
            out.push('\n');
        }
    }
    if meta.has_flag("async")
        && let Some(s) = harness.get("doneprintHandle.js")
    {
        out.push_str(s);
        out.push('\n');
    }
    for inc in &meta.includes {
        if let Some(s) = harness.get(inc) {
            out.push_str(s);
            out.push('\n');
        }
    }
    out.push_str(src);
    out
}

/// Decides whether a test is out of scope for the current engine/build.
fn skip_reason(rel: &str, meta: &Meta) -> Option<&'static str> {
    if rel.starts_with("staging/") {
        return Some("staging");
    }
    // Temporal is implemented for the plain/instant/duration types; ZonedDateTime
    // (time-zone database) and Now (system clock) are not yet, so skip those two
    // subtrees rather than ledger their whole surface as failing.
    if rel.starts_with("built-ins/Temporal/ZonedDateTime/")
        || rel.starts_with("built-ins/Temporal/Now/")
        || rel.contains("/ZonedDateTime/")
    {
        return Some("Temporal ZonedDateTime/Now not yet implemented");
    }
    if rel.starts_with("intl402/") && !cfg!(feature = "intl") {
        return Some("intl402 (build without the intl feature)");
    }
    for f in &meta.features {
        if SKIP_FEATURES.iter().any(|s| s == f) {
            return Some("unimplemented feature");
        }
    }
    None
}

/// Builds the harness *prelude* (everything `assemble` prepends except the test
/// source itself) for a module test: the host prelude, `sta.js` + `assert.js`,
/// `doneprintHandle.js` for an async test, and any `includes`. Evaluated as a
/// script in the module realm's global so the module body sees `assert` etc.
fn module_prelude(harness: &std::collections::HashMap<String, String>, meta: &Meta) -> String {
    let mut out = String::new();
    out.push_str(HOST_PRELUDE);
    for h in ["sta.js", "assert.js"] {
        if let Some(s) = harness.get(h) {
            out.push_str(s);
            out.push('\n');
        }
    }
    if meta.has_flag("async")
        && let Some(s) = harness.get("doneprintHandle.js")
    {
        out.push_str(s);
        out.push('\n');
    }
    for inc in &meta.includes {
        if let Some(s) = harness.get(inc) {
            out.push_str(s);
            out.push('\n');
        }
    }
    out
}

/// Runs a `flags: [module]` test: the test FILE is the entry module, evaluated
/// (after the harness prelude) with file-relative resolution of its imports
/// (`import "./x_FIXTURE.js"` loads the sibling). Classifies the outcome against
/// the test's expectation exactly like [`run_mode`].
fn run_module_test(
    path: &std::path::Path,
    harness: &std::collections::HashMap<String, String>,
    meta: &Meta,
) -> Result<(), String> {
    let prelude = module_prelude(harness, meta);
    let key = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned());
    let res = kataan::nbvm::execute_module_typed_with_prelude(
        &key,
        &kataan::nbexec::module::FileModuleHost,
        &prelude,
        Limits::default(),
    );
    classify(res, meta)
}

/// Classifies an engine result `(output, _)` / `Thrown` against the test's
/// positive/negative + async expectation. Shared by the script and module paths.
fn classify(
    res: Result<(String, String), kataan::nbexec::Thrown>,
    meta: &Meta,
) -> Result<(), String> {
    match (&meta.negative, res) {
        (None, Ok((output, _))) => {
            if meta.has_flag("async") {
                if output.contains("Test262:AsyncTestComplete")
                    && !output.contains("Test262:AsyncTestFailure")
                {
                    Ok(())
                } else {
                    Err(format!("async did not complete (output: {:.120})", output))
                }
            } else {
                Ok(())
            }
        }
        (None, Err(t)) => Err(format!(
            "expected pass, {:?} {}: {}",
            t.phase, t.name, t.message
        )),
        (Some((phase, ty)), Err(t)) => {
            let want_parse = phase == "parse" || phase == "early" || phase == "resolution";
            let phase_ok = (want_parse && t.phase == ErrorPhase::Parse)
                || (!want_parse && t.phase == ErrorPhase::Runtime);
            if !phase_ok {
                return Err(format!(
                    "negative: failed at {:?}, expected phase {}",
                    t.phase, phase
                ));
            }
            if t.name == *ty {
                Ok(())
            } else {
                Err(format!("negative: threw {}, expected {}", t.name, ty))
            }
        }
        (Some((_phase, ty)), Ok(_)) => Err(format!("expected {} but the test passed", ty)),
    }
}

/// Runs one assembled program through the production engine path and classifies
/// it against the test's expectation.
fn run_mode(combined: &str, meta: &Meta) -> Result<(), String> {
    classify(
        kataan::nbvm::execute_typed(combined, Limits::default()),
        meta,
    )
}

/// Runs all required modes for one test; passes only if every mode passes.
/// `path` is the absolute test-file path (for file-relative module / dynamic-
/// import resolution).
fn run_test(
    path: &std::path::Path,
    harness: &std::collections::HashMap<String, String>,
    meta: &Meta,
    src: &str,
) -> Result<(), String> {
    // A `flags: [module]` test runs as an ES module (the file is the entry
    // module), with the harness installed as a global prelude. Modules are always
    // strict and run once (no sloppy/strict dual mode).
    if meta.has_flag("module") {
        return run_module_test(path, harness, meta);
    }
    // Dynamic-import tests are scripts that call `import("./x_FIXTURE.js")`; the
    // specifier must resolve relative to the test file, so run them with the
    // file as the import base (still strict/sloppy dual mode + harness).
    let dynamic_import =
        path.to_string_lossy().contains("dynamic-import") && src.contains("import(");
    let base = std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned());
    let run = |combined: &str| -> Result<(), String> {
        if dynamic_import {
            classify(
                kataan::nbvm::execute_script_typed_with_import_base(
                    combined,
                    &base,
                    Limits::default(),
                ),
                meta,
            )
        } else {
            run_mode(combined, meta)
        }
    };
    let raw = meta.has_flag("raw");
    let (do_sloppy, do_strict) = if raw {
        (true, false)
    } else if meta.has_flag("onlyStrict") {
        (false, true)
    } else if meta.has_flag("noStrict") {
        (true, false)
    } else {
        (true, true)
    };
    if do_sloppy {
        run(&assemble(false, harness, meta, src)).map_err(|e| format!("[sloppy] {e}"))?;
    }
    if do_strict {
        run(&assemble(true, harness, meta, src)).map_err(|e| format!("[strict] {e}"))?;
    }
    Ok(())
}

/// Collects every test file (excluding `_FIXTURE.js`), relative to `test/`.
fn collect_tests(root: &Path) -> Vec<(String, PathBuf)> {
    let test_dir = root.join("test");
    let mut out = Vec::new();
    let mut stack = vec![test_dir.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "js") {
                let name = p.file_name().unwrap().to_string_lossy();
                if name.ends_with("_FIXTURE.js") {
                    continue;
                }
                let rel = p
                    .strip_prefix(&test_dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((rel, p));
            }
        }
    }
    out.sort();
    // `KATAAN_TEST262_FILTER=substr` restricts the run to test paths containing
    // `substr` (a local convenience for iterating on one area; applied in both the
    // coordinator and workers so their indexing agrees). Unset in CI.
    if let Ok(filter) = std::env::var("KATAAN_TEST262_FILTER")
        && !filter.is_empty()
    {
        out.retain(|(rel, _)| rel.contains(&filter));
    }
    out
}

fn read_ledger() -> HashSet<String> {
    let mut set = HashSet::new();
    if let Ok(text) = std::fs::read_to_string(ledger_path()) {
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if !line.is_empty() {
                set.insert(line.to_string());
            }
        }
    }
    set
}

#[test]
#[ignore = "runs the full ~53k official Test262 corpus; needs the vendor/test262 submodule"]
fn official_test262() {
    // Large stack so deep-but-legal recursion is bounded by the engine's
    // call-depth guard (a thrown RangeError), not a host overflow.
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(dispatch)
        .expect("spawn corpus thread")
        .join()
        .expect("corpus thread");
}

/// Worker mode (a child process) vs. coordinator mode (the top-level run).
fn dispatch() {
    if let Ok(spec) = std::env::var(WORKER_ENV) {
        run_worker(&spec);
    } else {
        coordinate();
    }
}

/// Worker: run the shard `start, start+step, …` and append a progress line per
/// test to `outpath` (`S` before running, `R` after), so the coordinator can
/// attribute a *native stack overflow* — which aborts the process and cannot be
/// caught with `catch_unwind` — to the exact in-flight test and resume past it.
fn run_worker(spec: &str) {
    let parts: Vec<&str> = spec.splitn(3, ':').collect();
    let start: usize = parts[0].parse().unwrap();
    let step: usize = parts[1].parse().unwrap();
    let outpath = parts[2];

    let root = corpus_root();
    let harness = load_harness(&root);
    let tests = collect_tests(&root);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(outpath)
        .expect("open out");

    let sanitize = |s: &str| s.replace(['\t', '\n', '\r'], " ");
    let mut idx = start;
    while idx < tests.len() {
        let (rel, path) = &tests[idx];
        let cur = idx;
        idx += step;
        let Ok(src) = std::fs::read_to_string(path) else {
            let _ = writeln!(f, "R\t{cur}\t{rel}\tFAIL\tunreadable");
            continue;
        };
        let meta = parse_meta(&src);
        // A test that drives `$262.agent` needs the cooperative agent scheduler
        // (concurrent Atomics: one agent blocks in `Atomics.wait` until another
        // `notify`s). That scheduler is not implemented, so skip these regardless
        // of path — they surface across built-ins/{Atomics,TypedArray,…} via the
        // `SharedArrayBuffer` feature tag, not just under a single directory.
        // `flags: [CanBlockIsFalse]` tests require a host whose main agent has
        // [[CanBlock]] = false (so `Atomics.wait` throws). This engine models a
        // shell host (CanBlock = true, `wait` returns), so skip them.
        if skip_reason(rel, &meta).is_some()
            || src.contains("$262.agent")
            || meta.has_flag("CanBlockIsFalse")
        {
            let _ = writeln!(f, "R\t{cur}\t{rel}\tSKIP\t");
            continue;
        }
        let _ = writeln!(f, "S\t{cur}\t{rel}");
        let _ = f.flush();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_test(path, &harness, &meta, &src)
        }));
        let (st, reason) = match res {
            Ok(Ok(())) => ("PASS", String::new()),
            Ok(Err(e)) => ("FAIL", e),
            Err(_) => ("FAIL", String::from("panic")),
        };
        let _ = writeln!(f, "R\t{cur}\t{rel}\t{st}\t{}", sanitize(&reason));
        let _ = f.flush();
    }
    let _ = writeln!(f, "DONE");
    let _ = f.flush();
}

/// Coordinator: shard the corpus across child processes (for native-stack
/// isolation) and tally their progress files.
fn coordinate() {
    let root = corpus_root();
    if !root.join("test").is_dir() {
        eprintln!(
            "SKIP: official Test262 corpus not found at {} — run:\n  \
             git submodule update --init --depth 1 vendor/test262",
            root.display()
        );
        return;
    }

    let total = collect_tests(&root).len();
    eprintln!("test262 official: {total} test files at {}", root.display());

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let exe = std::env::current_exe().expect("current_exe");
    let pid = std::process::id();
    let tmp = std::env::temp_dir();

    let mut handles = Vec::new();
    for w in 0..workers {
        let exe = exe.clone();
        let outpath = tmp.join(format!("kataan_t262_{pid}_{w}.txt"));
        handles.push(
            std::thread::Builder::new()
                .spawn(move || manage_shard(w, workers, total, &exe, &outpath))
                .expect("spawn coordinator thread"),
        );
    }

    let mut pass = 0usize;
    let mut skip = 0usize;
    let mut fails: Vec<(String, String)> = Vec::new();
    for h in handles {
        let (p, s, f) = h.join().expect("shard join");
        pass += p;
        skip += s;
        fails.extend(f);
    }
    fails.sort();

    let ran = pass + fails.len();
    eprintln!(
        "\n=== test262 official baseline ===\n\
         total={total} ran={ran} pass={pass} fail={} skip={skip}\n\
         pass-rate (of ran): {:.2}%",
        fails.len(),
        if ran > 0 {
            100.0 * pass as f64 / ran as f64
        } else {
            0.0
        },
    );

    // Per-area failure breakdown (first two path components).
    let mut areas: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (rel, _) in &fails {
        let area = rel.splitn(3, '/').take(2).collect::<Vec<_>>().join("/");
        *areas.entry(area).or_default() += 1;
    }
    let mut area_vec: Vec<_> = areas.into_iter().collect();
    area_vec.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    eprintln!("top failing areas:");
    for (area, n) in area_vec.iter().take(25) {
        eprintln!("  {n:>5}  {area}");
    }

    let fail_set: HashSet<String> = fails.iter().map(|(r, _)| r.clone()).collect();

    // Bless mode: (re)write the ledger from the current failures.
    if std::env::var("KATAAN_TEST262_BLESS").is_ok() {
        let mut body = String::from(
            "# Test262 known-failures ledger (paths relative to vendor/test262/test/).\n\
             # Regenerate with KATAAN_TEST262_BLESS=1. Target: empty (literal 100%).\n",
        );
        for (rel, reason) in &fails {
            body.push_str(rel);
            body.push_str("  # ");
            body.push_str(&reason.replace('\n', " "));
            body.push('\n');
        }
        std::fs::write(ledger_path(), body).expect("write ledger");
        eprintln!("\nBLESSED ledger with {} known failures.", fails.len());
        return;
    }

    // Gate mode: compare against the ledger.
    let ledger = read_ledger();
    let regressions: Vec<&String> = fail_set.iter().filter(|r| !ledger.contains(*r)).collect();
    let newly_passing: Vec<&String> = ledger.iter().filter(|r| !fail_set.contains(*r)).collect();

    if !newly_passing.is_empty() {
        eprintln!(
            "\n{} ledger entries now PASS (remove them from the ledger):",
            newly_passing.len()
        );
        for r in newly_passing.iter().take(40) {
            eprintln!("  + {r}");
        }
    }
    if !regressions.is_empty() {
        let mut sorted = regressions.clone();
        sorted.sort();
        eprintln!("\n{} REGRESSIONS (failing, not in ledger):", sorted.len());
        for r in sorted.iter().take(80) {
            let reason = fails
                .iter()
                .find(|(p, _)| &p == r)
                .map(|(_, m)| m.as_str())
                .unwrap_or("");
            eprintln!("  - {r}  ({reason})");
        }
    }
    assert!(
        regressions.is_empty(),
        "{} new Test262 failures not in the ledger (see above)",
        regressions.len()
    );
}

/// What a shard's progress file reveals after a child exits.
struct Scan {
    /// The child finished its whole shard (wrote the `DONE` marker).
    done: bool,
    /// An in-flight test (`S` with no matching `R`) — the crasher, if any.
    crashed: Option<(usize, String)>,
}

fn scan_progress(path: &Path) -> Scan {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut resolved = HashSet::new();
    let mut started: Vec<(usize, String)> = Vec::new();
    let mut done = false;
    for line in text.lines() {
        if line == "DONE" {
            done = true;
            continue;
        }
        let parts: Vec<&str> = line.splitn(5, '\t').collect();
        match parts.first() {
            Some(&"R") => {
                if let Ok(i) = parts[1].parse::<usize>() {
                    resolved.insert(i);
                }
            }
            Some(&"S") => {
                if let Ok(i) = parts[1].parse::<usize>() {
                    started.push((i, parts.get(2).unwrap_or(&"").to_string()));
                }
            }
            _ => {}
        }
    }
    let crashed = started.into_iter().find(|(i, _)| !resolved.contains(i));
    Scan { done, crashed }
}

/// Drives one shard to completion: spawn the worker child, and on a native crash
/// record the in-flight test as a failure and relaunch past it.
fn manage_shard(
    w: usize,
    workers: usize,
    total: usize,
    exe: &Path,
    outpath: &Path,
) -> (usize, usize, Vec<(String, String)>) {
    // A test producing no progress for this long is treated as hung (an infinite
    // loop or catastrophic backtracking) — the child is killed and the in-flight
    // test recorded as a failure, exactly like a native crash.
    const IDLE_TIMEOUT_SECS: u64 = 20;
    // Hard per-worker address-space cap (KiB) enforced via `ulimit -v`. A test
    // with an unbounded allocation (e.g. a typed array / ArrayBuffer / string of
    // pathological size) hits this ceiling, its allocation is refused, the worker
    // aborts, and the crash machinery records it as a failure — instead of the
    // process ballooning and OOM-ing the whole host. 4 GiB is far above any
    // legitimate single Test262 test yet bounds a runaway.
    const MAX_ADDRESS_SPACE_KIB: u64 = 4 * 1024 * 1024;

    let _ = std::fs::remove_file(outpath);
    let mut start = w;
    let mut guard = 0usize;
    while start < total {
        guard += 1;
        if guard > total + 16 {
            break;
        }
        // Launch the worker through `sh` so we can apply `ulimit -v` before exec.
        let spawn = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "ulimit -v {MAX_ADDRESS_SPACE_KIB} 2>/dev/null; exec \"$@\""
            ))
            .arg("sh") // $0 for the inner shell
            .arg(exe)
            .args([
                "--ignored",
                "--exact",
                "official_test262",
                "--test-threads",
                "1",
            ])
            .env(
                WORKER_ENV,
                format!("{start}:{workers}:{}", outpath.display()),
            )
            .env_remove("KATAAN_TEST262_BLESS")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = spawn else {
            start += workers;
            continue;
        };
        // Poll the child, watching the progress file grow. No growth for
        // IDLE_TIMEOUT_SECS while still alive ⇒ hung ⇒ kill it.
        let file_len = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let mut last_len = file_len(outpath);
        let mut idle = 0u64;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Err(_) => break,
                Ok(None) => {}
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            let len = file_len(outpath);
            if len != last_len {
                last_len = len;
                idle = 0;
            } else {
                idle += 1;
            }
            if idle >= IDLE_TIMEOUT_SECS {
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }

        let scan = scan_progress(outpath);
        if scan.done {
            break;
        }
        match scan.crashed {
            Some((idx, rel)) => {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(outpath)
                {
                    let _ = writeln!(
                        f,
                        "R\t{idx}\t{rel}\tFAIL\tcrash or timeout (native abort / hang)"
                    );
                }
                start = idx + workers;
            }
            // Child died without an in-flight test and without DONE — advance to
            // avoid spinning on the same spot.
            None => start += workers,
        }
    }
    tally_file(outpath)
}

fn tally_file(path: &Path) -> (usize, usize, Vec<(String, String)>) {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut pass = 0;
    let mut skip = 0;
    let mut fails = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.splitn(5, '\t').collect();
        if parts.first() != Some(&"R") {
            continue;
        }
        match parts.get(3) {
            Some(&"PASS") => pass += 1,
            Some(&"SKIP") => skip += 1,
            Some(&"FAIL") => {
                let rel = parts.get(2).unwrap_or(&"").to_string();
                let reason = parts.get(4).unwrap_or(&"").to_string();
                fails.push((rel, reason));
            }
            _ => {}
        }
    }
    (pass, skip, fails)
}

// ---------------------------------------------------------------------------
// Runner-logic unit tests (no corpus needed).
// ---------------------------------------------------------------------------

#[test]
fn frontmatter_flow_and_block() {
    let flow = "/*---\nflags: [onlyStrict, async]\nincludes: [a.js, b.js]\nfeatures: [Temporal]\nnegative:\n  phase: parse\n  type: SyntaxError\n---*/\n";
    let m = parse_meta(flow);
    assert!(m.has_flag("onlyStrict") && m.has_flag("async"));
    assert_eq!(m.includes, ["a.js", "b.js"]);
    assert_eq!(m.features, ["Temporal"]);
    assert_eq!(m.negative, Some(("parse".into(), "SyntaxError".into())));

    let block =
        "/*---\nincludes:\n  - propertyHelper.js\n  - compareArray.js\nflags: [raw]\n---*/\n";
    let mb = parse_meta(block);
    assert_eq!(mb.includes, ["propertyHelper.js", "compareArray.js"]);
    assert!(mb.has_flag("raw"));
}

#[test]
fn mode_selection_and_skips() {
    let only_strict = parse_meta("/*---\nflags: [onlyStrict]\n---*/\n");
    assert!(only_strict.has_flag("onlyStrict") && !only_strict.has_flag("noStrict"));

    // Module-flagged tests now RUN (the ES-module executor landed); they are no
    // longer skipped.
    let module = parse_meta("/*---\nflags: [module]\n---*/\n");
    assert_eq!(skip_reason("language/module-code/x.js", &module), None);

    let temporal = parse_meta("/*---\nfeatures: [Temporal]\n---*/\n");
    assert_eq!(
        skip_reason("built-ins/Temporal/x.js", &temporal),
        Some("unimplemented feature")
    );

    let plain = parse_meta("/*---\ndescription: x\n---*/\n");
    assert_eq!(skip_reason("language/types/x.js", &plain), None);
}
