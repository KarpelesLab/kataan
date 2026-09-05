//! Dev-only: run a sloppy script file and print output.
//! Usage: cargo run --example srun -- path/to/file.js
const PRELUDE: &str = r#"
var print = function () { var s = ''; for (var i = 0; i < arguments.length; i++) { if (i) s += ' '; s += arguments[i]; } console.log(s); };
"#;
fn main() {
    let path = std::env::args().nth(1).expect("usage: srun <file.js>");
    let src = std::fs::read_to_string(&path).expect("read");
    let combined = format!("{PRELUDE}\n{src}");
    // `KATAAN_SRUN_TIER=nbexec` pins the reference tree-walker, so a snippet can be
    // checked on both tiers (the default entry silently falls back to nbexec, which
    // otherwise hides which tier produced a result).
    let force_nbexec = std::env::var("KATAAN_SRUN_TIER").is_ok_and(|v| v == "nbexec");
    let result = if force_nbexec {
        kataan::nbexec::eval_source_typed(&combined, kataan::limits::Limits::default())
    } else {
        kataan::nbvm::execute_typed(&combined, kataan::limits::Limits::default())
    };
    match result {
        Ok((output, _)) => {
            print!("{output}");
            eprintln!("[srun] PASS (no throw)");
        }
        Err(t) => eprintln!("[srun] THROW {:?} {}: {}", t.phase, t.name, t.message),
    }
}
