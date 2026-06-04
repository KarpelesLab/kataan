//! The `kataan` command-line tool.
//!
//! The engine front end is being built bottom-up, and the CLI exposes each
//! stage as it lands: `lex` tokenizes, and `parse` produces an AST. Later
//! phases add full-program parsing, `run`, and a REPL (see `ROADMAP.md`).
//!
//! ```text
//! kataan lex FILE            # print the tokens of FILE
//! kataan lex -e 'SOURCE'     # tokenize SOURCE from the command line
//! kataan parse FILE          # parse a program and dump its AST
//! kataan parse -e 'SOURCE'   # parse SOURCE (a program) and dump its AST
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
    match interp.run(&program) {
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
         kataan --version          print the version\n    \
         kataan --help             show this help\n\
         \n\
         A full host runtime and REPL arrive in later phases — see ROADMAP.md.",
        kataan::VERSION
    );
}
