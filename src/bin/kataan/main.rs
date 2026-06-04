//! The `kataan` command-line tool.
//!
//! At this stage (Phase A) the engine front end is the lexer, so the CLI can
//! tokenize a script and print the token stream — useful for inspecting the
//! lexer and as the first end-to-end demonstration of the pipeline. Later
//! phases add `parse`, `run`, and a REPL (see `ROADMAP.md`).
//!
//! ```text
//! kataan lex FILE          # print the tokens of FILE
//! kataan lex -e 'SOURCE'   # tokenize SOURCE from the command line
//! kataan --version
//! ```

use kataan::lexer::{Lexer, TokenKind};
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

fn print_usage() {
    println!(
        "kataan {} — a JavaScript engine in pure Rust\n\
         \n\
         USAGE:\n    \
         kataan lex <FILE>         tokenize a script file\n    \
         kataan lex -e <SOURCE>    tokenize a source string\n    \
         kataan --version          print the version\n    \
         kataan --help             show this help\n\
         \n\
         More subcommands (parse, run, repl) arrive in later phases — see ROADMAP.md.",
        kataan::VERSION
    );
}
