//! Dev-only: run a sloppy script file and print output.
//! Usage: cargo run --example srun -- path/to/file.js
const PRELUDE: &str = r#"
var print = function () { var s = ''; for (var i = 0; i < arguments.length; i++) { if (i) s += ' '; s += arguments[i]; } console.log(s); };
"#;
fn main() {
    let path = std::env::args().nth(1).expect("usage: srun <file.js>");
    let src = std::fs::read_to_string(&path).expect("read");
    let combined = format!("{PRELUDE}\n{src}");
    // A file holding several scripts separated by `//---SCRIPT---` lines runs
    // them as consecutive Scripts over one realm (as the Test262 runner does).
    let scripts: Vec<&str> = combined.split("//---SCRIPT---\n").collect();
    let result = if scripts.len() > 1 {
        kataan::nbvm::execute_scripts_typed(&scripts, kataan::limits::Limits::default())
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
