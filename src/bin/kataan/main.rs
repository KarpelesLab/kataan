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
        ["eval" | "run", "-e", source] => run_eval_nb(source, "<argv>"),
        // `run` accepts either JS source or a compiled `.ktbc` artifact (detected
        // by its `KTBC` magic).
        ["eval" | "run", path] => match std::fs::read(path) {
            Ok(bytes) if bytes.starts_with(b"KTBC") => run_bytecode(&bytes, path),
            Ok(bytes) => run_eval_nb(&String::from_utf8_lossy(&bytes), path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        // Compile JS to a portable `.ktbc` bytecode artifact (Phase D′).
        ["compile", path, "-o", out] | ["compile", "-o", out, path] => run_compile(path, out),
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

/// Compiles `path`'s JavaScript to a portable `.ktbc` bytecode artifact written
/// to `out` (Phase D′ — `kataan compile app.js -o app.ktbc`).
fn run_compile(path: &str, out: &str) -> ExitCode {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("kataan: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let program = match Parser::parse_program(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let protos = match kataan::nbvm::compile_program(&program) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {path}: cannot compile to bytecode: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = kataan::bytecode::serialize(&protos);
    match std::fs::write(out, &bytes) {
        Ok(()) => {
            eprintln!("kataan: wrote {} ({} bytes)", out, bytes.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: cannot write {out}: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Loads and runs a compiled `.ktbc` bytecode artifact (the inverse of
/// `compile`), printing any `console` output and a non-empty completion value.
fn run_bytecode(bytes: &[u8], origin: &str) -> ExitCode {
    let protos = match kataan::bytecode::deserialize(bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {origin}: invalid bytecode artifact: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let mut realm = kataan::realm::Realm::new();
    match kataan::nbvm::run_program_capturing(&mut realm, &protos, 0, &[]) {
        Ok((value, output)) => {
            print!("{output}");
            let completion = realm.to_display_string(value);
            if !completion.is_empty() && completion != "undefined" {
                println!("{completion}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("kataan: {origin}: {e:?}");
            ExitCode::FAILURE
        }
    }
}

/// Runs a read-eval-print loop on the new-representation tree-walker
/// (`nbexec::Interp`), which carries its own `console` and persists globals
/// across lines. Each line's AST is leaked to `&'static` so values the
/// persistent interpreter keeps in its globals can reference it.
fn run_repl() -> ExitCode {
    use std::io::{self, BufRead, Write};

    let mut interp: kataan::nbexec::Interp<'static> = kataan::nbexec::Interp::new();
    println!(
        "kataan {} REPL — type JavaScript, Ctrl-D to exit",
        kataan::VERSION
    );

    let stdin = io::stdin();
    let mut shown = 0usize; // bytes of `console` output already printed
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
                // Flush any new `console` output, then the completion value.
                let out = interp.output();
                if out.len() > shown {
                    print!("{}", &out[shown..]);
                    shown = out.len();
                }
                let disp = interp.display(value);
                if !disp.is_empty() && disp != "undefined" {
                    println!("{disp}");
                }
            }
            Err(e) => eprintln!("Uncaught {e:?}"),
        }
    }
    println!();
    ExitCode::SUCCESS
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
         kataan compile <FILE> -o <OUT.ktbc>  compile JS to a bytecode artifact\n    \
         kataan run <FILE.ktbc>    run a compiled bytecode artifact\n    \
         kataan nbrun <FILE>       alias for `run` (the new-representation engine)\n    \
         kataan nbrun -e <SOURCE>  run a source string on the new engine\n    \
         kataan --version          print the version\n    \
         kataan --help             show this help\n\
         \n\
         A full host runtime (event loop, fetch, modules) arrives in later phases — see ROADMAP.md.",
        kataan::VERSION
    );
}
