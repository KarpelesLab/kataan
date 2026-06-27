//! Dev-only harness: run a single Test262 `flags: [module]` file with the
//! standard prelude + file-relative import resolution, and print its output.
//! Usage: `cargo run --example modrun -- path/to/test.js`
//!
//! Not part of the engine — exists so module-goal proposals (e.g. import-defer)
//! can be iterated on without running the whole corpus.

use std::collections::HashMap;
use std::path::Path;

const HOST_PRELUDE: &str = r#"
var print = function () { var s = ''; for (var i = 0; i < arguments.length; i++) { if (i) s += ' '; s += arguments[i]; } console.log(s); };
var $262 = { global: this, gc: function () {}, evalScript: function (src) { return eval(src); }, detachArrayBuffer: function (b) { return $262_detachArrayBuffer(b); }, createRealm: function () { throw new TypeError('no'); }, IsHTMLDDA: undefined };
"#;

fn main() {
    let path = std::env::args().nth(1).expect("usage: modrun <test.js>");
    let src = std::fs::read_to_string(&path).expect("read test");
    let harness_dir = "vendor/test262/harness";
    let mut harness: HashMap<String, String> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir(harness_dir) {
        for e in rd.flatten() {
            if let Ok(s) = std::fs::read_to_string(e.path()) {
                harness.insert(e.file_name().to_string_lossy().into_owned(), s);
            }
        }
    }

    let is_async = src.contains("flags:") && src.contains("async");
    // Crude `includes: [a.js, b.js]` extraction (flow style only — enough here).
    let mut includes: Vec<String> = Vec::new();
    if let Some(i) = src.find("includes:") {
        if let Some(lb) = src[i..].find('[') {
            if let Some(rb) = src[i + lb..].find(']') {
                let list = &src[i + lb + 1..i + lb + rb];
                includes = list
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }

    let mut prelude = String::from(HOST_PRELUDE);
    for h in ["sta.js", "assert.js"] {
        if let Some(s) = harness.get(h) {
            prelude.push_str(s);
            prelude.push('\n');
        }
    }
    if is_async {
        if let Some(s) = harness.get("doneprintHandle.js") {
            prelude.push_str(s);
            prelude.push('\n');
        }
    }
    for inc in &includes {
        if let Some(s) = harness.get(inc) {
            prelude.push_str(s);
            prelude.push('\n');
        }
    }

    let key = std::fs::canonicalize(Path::new(&path))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or(path.clone());
    let res = kataan::nbvm::execute_module_typed_with_prelude(
        &key,
        &kataan::nbexec::module::FileModuleHost,
        &prelude,
        kataan::limits::Limits::default(),
    );
    match res {
        Ok((output, _completion)) => {
            print!("{output}");
            if is_async {
                if output.contains("Test262:AsyncTestComplete")
                    && !output.contains("Test262:AsyncTestFailure")
                {
                    eprintln!("[modrun] PASS (async)");
                } else {
                    eprintln!("[modrun] FAIL (async did not complete)");
                }
            } else {
                eprintln!("[modrun] PASS (no throw)");
            }
        }
        Err(t) => {
            eprintln!("[modrun] THROW {:?} {}: {}", t.phase, t.name, t.message);
        }
    }
}
