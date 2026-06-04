//! The `kataan` command-line tool.
//!
//! The CLI exposes each stage of the engine: `lex` tokenizes, `parse` produces
//! an AST, `eval`/`run` execute a program (with a minimal `console`), and
//! `repl` starts an interactive session. A fuller host runtime (event loop,
//! modules, `fetch`) arrives in a later phase (see `ROADMAP.md`).
//!
//! ```text
//! kataan lex   [-e] SOURCE|FILE   # print the tokens
//! kataan parse [-e] SOURCE|FILE   # dump the AST
//! kataan run   [-e] SOURCE|FILE   # evaluate (alias: eval)
//! kataan repl                     # interactive REPL
//! kataan --version
//! ```

use kataan::interp::Interp;
use kataan::lexer::{Lexer, TokenKind};
use kataan::parser::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();

    match args.as_slice() {
        [] | ["-h" | "--help" | "help"] => {
            print_usage();
            ExitCode::SUCCESS
        }
        ["-V" | "--version" | "version"] => {
            println!("kataan {}", kataan::VERSION);
            ExitCode::SUCCESS
        }
        ["lex", "-e", source] => run_lex(source, "<argv>"),
        ["lex", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_lex(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        ["parse", "-e", source] => run_parse(source, "<argv>"),
        ["parse", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_parse(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        ["eval" | "run", "-e", source] => run_eval(source, "<argv>"),
        ["eval" | "run", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_eval(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        // Run through the new-representation engine (`ROADMAP.md` §3).
        ["nbrun", "-e", source] => run_eval_nb(source, "<argv>"),
        ["nbrun", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_eval_nb(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        ["repl"] => run_repl(),
        ["disasm", "-e", source] => run_disasm(source, "<argv>"),
        ["disasm", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_disasm(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        ["bcrun", "-e", source] => run_eval_vm(source, "<argv>"),
        ["bcrun", path] => match std::fs::read_to_string(path) {
            Ok(source) => run_eval_vm(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("kataan: unrecognized arguments: {}", args.join(" "));
            eprintln!("try `kataan --help`");
            ExitCode::FAILURE
        }
    }
}

/// Tokenizes `source` and prints one token per line. `origin` is shown in
/// error messages.
fn run_lex(source: &str, origin: &str) -> ExitCode {
    match Lexer::new(source).tokenize() {
        Ok(tokens) => {
            for tok in &tokens {
                if tok.kind == TokenKind::Eof {
                    println!("{:>4}..{:<4} Eof", tok.span.start, tok.span.end);
                    continue;
                }
                let nl = if tok.newline_before { " ⏎" } else { "" };
                println!(
                    "{:>4}..{:<4} {:<24} {:?}{nl}",
                    tok.span.start,
                    tok.span.end,
                    format!("{:?}", tok.kind),
                    tok.text(source),
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parses `source` as a program and prints its AST. `origin` is shown in error
/// messages.
fn run_parse(source: &str, origin: &str) -> ExitCode {
    match Parser::parse_program(source) {
        Ok(program) => {
            println!("{program:#?}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Parses and evaluates `source`, printing the completion value (REPL-style).
/// `origin` is shown in error messages.
fn run_eval(source: &str, origin: &str) -> ExitCode {
    let program = match Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut interp = Interp::new();
    install_console(&interp);
    // The bytecode VM is the primary execution path (it falls back to the
    // tree-walker for constructs it doesn't yet compile).
    match interp.run_with_vm(&program) {
        Ok(value) => {
            // Print non-undefined completion values, REPL-style.
            if !matches!(value, kataan::interp::Value::Undefined) {
                println!("{}", value.to_js_string());
            }
            ExitCode::SUCCESS
        }
        Err(thrown) => {
            eprintln!("kataan: {origin}: Uncaught {}", thrown.to_js_string());
            ExitCode::FAILURE
        }
    }
}

/// Parses and evaluates `source` through the **bytecode VM** (falling back to
/// the tree-walker for unsupported constructs), printing the completion value.
/// (Kept as an explicit subcommand; `run`/`eval` use the same path now.)
/// Runs `source` through the new-representation engine — the bytecode VM with a
/// tree-walker fallback (`kataan::nbvm::execute`) — printing its captured
/// `console` output and a non-empty completion value.
fn run_eval_nb(source: &str, origin: &str) -> ExitCode {
    match kataan::nbvm::execute(source) {
        Ok((output, completion)) => {
            print!("{output}");
            if !completion.is_empty() && completion != "undefined" {
                println!("{completion}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_eval_vm(source: &str, origin: &str) -> ExitCode {
    let program = match Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut interp = Interp::new();
    install_console(&interp);
    match interp.run_with_vm(&program) {
        Ok(value) => {
            if !matches!(value, kataan::interp::Value::Undefined) {
                println!("{}", value.to_js_string());
            }
            ExitCode::SUCCESS
        }
        Err(thrown) => {
            eprintln!("kataan: {origin}: Uncaught {}", thrown.to_js_string());
            ExitCode::FAILURE
        }
    }
}

/// Compiles `source` to bytecode and prints its disassembly. Falls back with a
/// message for constructs outside the bytecode compiler's current subset.
fn run_disasm(source: &str, origin: &str) -> ExitCode {
    use kataan::interp::compile_program;

    let program = match Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            return ExitCode::FAILURE;
        }
    };
    match compile_program(&program.body) {
        Ok(module) => {
            print!("{}", module.disassemble());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: {origin}: {}", e.message);
            ExitCode::FAILURE
        }
    }
}

/// Runs a read-eval-print loop. Each entered line is parsed to an owned
/// `Program` which is leaked to `&'static` so the persistent interpreter (and
/// the values stored in its globals) can reference it across iterations — fine
/// for an interactive session.
fn run_repl() -> ExitCode {
    use std::io::{self, BufRead, Write};

    let mut interp: Interp<'static> = Interp::new();
    install_console(&interp);
    println!(
        "kataan {} REPL — type JavaScript, Ctrl-D to exit",
        kataan::VERSION
    );

    let stdin = io::stdin();
    loop {
        print!("> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("kataan: read error: {e}");
                break;
            }
        }
        if line.trim().is_empty() {
            continue;
        }
        let program = match Parser::parse_program(&line) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                continue;
            }
        };
        // Leak the AST so its `&'static` borrow outlives values kept in globals.
        let program: &'static kataan::ast::Program = Box::leak(Box::new(program));
        match interp.run(program) {
            Ok(value) => {
                if !matches!(value, kataan::interp::Value::Undefined) {
                    println!("{value:?}");
                }
            }
            Err(thrown) => eprintln!("Uncaught {}", thrown.to_js_string()),
        }
    }
    println!();
    ExitCode::SUCCESS
}

/// Installs a minimal `console` global (`log`/`info`/`warn`/`error`) that
/// prints its arguments space-separated. This is the first sliver of the host
/// runtime; a fuller one arrives in Phase F.
fn install_console(interp: &Interp) {
    use kataan::interp::{NativeFn, Obj, Value};
    use std::rc::Rc;

    let console = Obj::object();
    for name in ["log", "info", "warn", "error"] {
        let to_stderr = matches!(name, "warn" | "error");
        let native = Value::Native(Rc::new(NativeFn {
            name,
            call: Box::new(move |args: &[Value]| {
                let line = args
                    .iter()
                    .map(Value::to_js_string)
                    .collect::<Vec<_>>()
                    .join(" ");
                if to_stderr {
                    eprintln!("{line}");
                } else {
                    println!("{line}");
                }
                Ok(Value::Undefined)
            }),
        }));
        console.set(name, native);
    }
    interp.define_global("console", Value::Object(console));
}

fn print_usage() {
    println!(
        "kataan {} — a JavaScript engine in pure Rust\n\
         \n\
         USAGE:\n    \
         kataan lex <FILE>         tokenize a script file\n    \
         kataan lex -e <SOURCE>    tokenize a source string\n    \
         kataan parse <FILE>       parse a program and dump its AST\n    \
         kataan parse -e <SOURCE>  parse a source string and dump its AST\n    \
         kataan eval <FILE>        evaluate a program (prints completion value)\n    \
         kataan eval -e <SOURCE>   evaluate a source string\n    \
         kataan repl               start an interactive REPL\n    \
         kataan disasm <FILE>      compile to bytecode and print the disassembly\n    \
         kataan bcrun <FILE>       run via the bytecode VM (tree-walker fallback)\n    \
         kataan nbrun <FILE>       run via the new-representation engine (ROADMAP.md \u{a7}3)\n    \
         kataan nbrun -e <SOURCE>  run a source string on the new engine\n    \
         kataan --version          print the version\n    \
         kataan --help             show this help\n\
         \n\
         A full host runtime (event loop, fetch, modules) arrives in later phases — see ROADMAP.md.",
        kataan::VERSION
    );
}
