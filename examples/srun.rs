//! Dev-only: run a sloppy script file and print output.
//! Usage: cargo run --example srun -- path/to/file.js
const PRELUDE: &str = r#"
var print = function () { var s = ''; for (var i = 0; i < arguments.length; i++) { if (i) s += ' '; s += arguments[i]; } console.log(s); };
"#;
fn main() {
    let path = std::env::args().nth(1).expect("usage: srun <file.js>");
    let src = std::fs::read_to_string(&path).expect("read");
    let combined = format!("{PRELUDE}\n{src}");
    match kataan::nbvm::execute_typed(&combined, kataan::limits::Limits::default()) {
        Ok((output, _)) => {
            print!("{output}");
            eprintln!("[srun] PASS (no throw)");
        }
        Err(t) => eprintln!("[srun] THROW {:?} {}: {}", t.phase, t.name, t.message),
    }
}
