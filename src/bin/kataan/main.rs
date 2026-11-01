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

/// The engine's own pooling allocator (pure Rust; see `kataan::pool_alloc`).
#[cfg(feature = "pool-alloc")]
#[global_allocator]
static GLOBAL: kataan::pool_alloc::PoolAlloc = kataan::pool_alloc::PoolAlloc;

/// A hard address-space limit for the whole process, so a runaway script (or a
/// compiler blowup) fails its own allocation instead of exhausting the machine.
/// `KATAAN_MAX_MEMORY_MB` overrides the default of 8 GiB. It can never exceed a
/// limit already inherited, and child processes cannot raise it.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
#[allow(unsafe_code)]
fn limit_address_space() {
    const DEFAULT_MB: u64 = 8 * 1024;
    const RLIMIT_AS: i32 = 9;
    #[repr(C)]
    struct RLimit {
        cur: u64,
        max: u64,
    }
    unsafe extern "C" {
        fn getrlimit(resource: i32, rlim: *mut RLimit) -> i32;
        fn setrlimit(resource: i32, rlim: *const RLimit) -> i32;
    }
    let mb = std::env::var("KATAAN_MAX_MEMORY_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MB);
    let bytes = mb.saturating_mul(1024 * 1024);
    let mut lim = RLimit { cur: 0, max: 0 };
    // SAFETY: both calls only read or write the `RLimit` passed by pointer,
    // whose layout matches the C `struct rlimit` on 64-bit Linux.
    unsafe {
        if getrlimit(RLIMIT_AS, &mut lim) == 0 {
            let cap = bytes.min(lim.max);
            setrlimit(RLIMIT_AS, &RLimit { cur: cap, max: cap });
        }
    }
}

fn main() -> ExitCode {
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    limit_address_space();
    // Run on a large stack so deep JS recursion is bounded by the engine's
    // call-depth guard (a thrown RangeError), not a host stack overflow.
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(run_main)
        .expect("spawn main thread")
        .join()
        .expect("main thread")
}

fn run_main() -> ExitCode {
    kataan::stack::set_thread_stack_size(512 * 1024 * 1024);
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
            // An ES module (a `.mjs` file, or module syntax in another file,
            // as Node detects it) runs as the entry of a module graph whose
            // imports resolve relative to it.
            Ok(bytes) if runs_as_module(path, &String::from_utf8_lossy(&bytes)) => run_module(path),
            Ok(bytes) => run_eval_nb(&String::from_utf8_lossy(&bytes), path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        // Compile JS to a portable `.ktbc` bytecode artifact (Phase D′).
        ["compile", path, "-o", out] | ["compile", "-o", out, path] => run_compile(path, out),
        // `nbrun` is a legacy alias of `run` (there is one engine).
        ["nbrun", "-e", source] => run_eval_nb(source, "<argv>"),
        ["nbrun", path] => match std::fs::read_to_string(path) {
            Ok(source) if runs_as_module(path, &source) => run_module(path),
            Ok(source) => run_eval_nb(&source, path),
            Err(e) => {
                eprintln!("kataan: cannot read {path}: {e}");
                ExitCode::FAILURE
            }
        },
        // Run *with the §4.1 host runtime* (timers / event loop): after the
        // top-level script completes, drive the event loop so
        // `setTimeout`/`setInterval`/`process.nextTick` work runs before the
        // process exits.
        ["hostrun", "-e", source] => run_host(source, "<argv>"),
        ["hostrun", path] => match std::fs::read_to_string(path) {
            Ok(source) if runs_as_module(path, &source) => run_host_module(path),
            Ok(source) => run_host(&source, path),
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
                    tok.text(source.as_bytes()),
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

/// Runs `source` on the bytecode VM hosted by an interpreter
/// (`kataan::nbvm::execute_capturing`, i.e. `Interp::run`), printing its captured
/// `console` output (even when the script then throws) and a non-empty
/// completion value. Backs `run`, `eval` and
/// `nbrun`.
fn run_eval_nb(source: &str, origin: &str) -> ExitCode {
    // The script's `console` output is printed on the error path too: an
    // uncaught throw does not un-print what ran before it.
    let limits = kataan::limits::Limits::default();
    let (output, outcome) = if origin == "<argv>" {
        kataan::nbvm::execute_capturing_then_exit(source, limits)
    } else {
        kataan::nbvm::execute_file_capturing_then_exit(source, &script_key(origin), limits)
    };
    print!("{output}");
    match outcome {
        Ok(completion) => {
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

/// The canonical path of the script or module file at `path` (the referrer its
/// imports resolve against), or `path` itself when it cannot be canonicalised.
fn script_key(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string())
}

/// Whether the file at `path` runs as an ES module: a `.mjs` file always, a
/// `.cjs` file never, and any other file when its text is module code (Node's
/// syntax detection): it has a top-level `import`/`export` declaration, or it
/// does not parse as a script but does as a module (top-level `await`,
/// `import.meta`).
fn runs_as_module(path: &str, source: &str) -> bool {
    if path.ends_with(".mjs") {
        return true;
    }
    if path.ends_with(".cjs")
        || !["import", "export", "await"]
            .iter()
            .any(|k| source.contains(k))
    {
        return false;
    }
    match kataan::parser::Parser::parse_program(source) {
        Ok(program) => program.source_type == kataan::ast::SourceType::Module,
        Err(_) => kataan::parser::Parser::parse_module(source).is_ok(),
    }
}

/// `name: message` for an uncaught error (just the name when there is no
/// message).
fn thrown_text(t: &kataan::nbexec::Thrown) -> String {
    if t.message.is_empty() {
        t.name.clone()
    } else {
        format!("{}: {}", t.name, t.message)
    }
}

/// Prints what the program logged, then reports the uncaught error `e` of a
/// `hostrun` program by its name and message (not the raw thrown handle).
fn host_uncaught(
    interp: &kataan::nbexec::Interp,
    origin: &str,
    e: kataan::nbexec::ExecError,
) -> ExitCode {
    print!("{}", interp.output());
    let t = interp.exec_error_to_thrown(e, kataan::nbexec::ErrorPhase::Runtime);
    eprintln!("kataan: {origin}: uncaught {}", thrown_text(&t));
    ExitCode::FAILURE
}

/// Runs the `.mjs` file at `path` as the entry module of an ES-module graph.
fn run_module(path: &str) -> ExitCode {
    let key = script_key(path);
    let (output, outcome) = kataan::nbvm::execute_module_capturing(
        &key,
        &kataan::nbexec::module::FileModuleHost,
        kataan::limits::Limits::default(),
    );
    print!("{output}");
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kataan: {path}: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `hostrun` for a `.mjs` entry: the module graph runs with the host timers
/// installed, then the event loop drains to quiescence.
fn run_host_module(path: &str) -> ExitCode {
    let key = script_key(path);
    let mut interp: kataan::nbexec::Interp = kataan::nbexec::Interp::new();
    kataan::host::timers::install(&mut interp);
    if let Err(t) =
        kataan::nbvm::evaluate_module_on(&mut interp, &key, &kataan::nbexec::module::FileModuleHost)
    {
        print!("{}", interp.output());
        eprintln!("kataan: {path}: {}", thrown_text(&t));
        return ExitCode::FAILURE;
    }
    if let Err(e) = kataan::host::timers::run_event_loop(&mut interp) {
        return host_uncaught(&interp, path, e);
    }
    print!("{}", interp.output());
    ExitCode::SUCCESS
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
///
/// Artifacts run on the *bare* VM tier — a self-contained realm with no host
/// interpreter (`ROADMAP.md` §2.0): a `.ktbc` file is untrusted, verified input
/// compiled without the host's global-declaration pass.
fn run_bytecode(bytes: &[u8], origin: &str) -> ExitCode {
    // Decode *and* verify — a `.ktbc` file is untrusted input.
    let protos = match kataan::bytecode::deserialize_verified(bytes) {
        Ok(p) => p,
        Err(Err(e)) => {
            eprintln!("kataan: {origin}: invalid bytecode artifact: {e:?}");
            return ExitCode::FAILURE;
        }
        Err(Ok(e)) => {
            eprintln!("kataan: {origin}: bytecode failed verification: {e:?}");
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

/// Evaluates `source` (on the hosted bytecode VM, `Interp::run`) with the §4.1
/// host runtime installed, then drives [`kataan::host::timers::run_event_loop`]
/// so timer- and `nextTick`-scheduled callbacks run before returning. Prints any
/// `console` output and a non-empty completion value.
fn run_host(source: &str, origin: &str) -> ExitCode {
    let program = match Parser::parse_program(source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("kataan: {origin}: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Leak the AST so its `&'static` borrow outlives values kept in globals
    // (timer callbacks pinned across the event loop).
    let program: &'static kataan::ast::Program = Box::leak(Box::new(program));
    let mut interp: kataan::nbexec::Interp = kataan::nbexec::Interp::new();
    kataan::host::timers::install(&mut interp);
    if origin != "<argv>" {
        interp.set_script_import_base(Some(script_key(origin)));
    }

    let completion = match interp.run(program) {
        Ok(value) => value,
        Err(e) => return host_uncaught(&interp, origin, e),
    };
    // Drain timers / microtasks / nextTick queue to quiescence.
    if let Err(e) = kataan::host::timers::run_event_loop(&mut interp) {
        return host_uncaught(&interp, origin, e);
    }
    print!("{}", interp.output());
    let disp = interp.display(completion);
    if !disp.is_empty() && disp != "undefined" {
        println!("{disp}");
    }
    ExitCode::SUCCESS
}

/// Runs a read-eval-print loop on one persistent interpreter: each line runs on
/// the hosted bytecode VM (`Interp::run`), sharing the global environment and
/// the function table with every earlier line. Each line's AST is leaked to
/// `&'static` so values the persistent interpreter keeps in its globals can
/// reference it.
fn run_repl() -> ExitCode {
    use std::io::{self, BufRead, Write};

    let mut interp: kataan::nbexec::Interp = kataan::nbexec::Interp::new();
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
         kataan run <FILE>         run a script, a .mjs module or a .ktbc artifact\n    \
         kataan run -e <SOURCE>    run a source string\n    \
         kataan eval <FILE>        alias for `run`\n    \
         kataan eval -e <SOURCE>   alias for `run -e`\n    \
         kataan repl               start an interactive REPL\n    \
         kataan compile <FILE> -o <OUT.ktbc>  compile JS to a bytecode artifact\n    \
         kataan hostrun <FILE>     run with the host runtime (timers / event loop)\n    \
         kataan hostrun -e <SRC>   run a source string with the host event loop\n    \
         kataan --version          print the version\n    \
         kataan --help             show this help\n\
         \n\
         A full host runtime (event loop, fetch, modules) arrives in later phases — see ROADMAP.md.",
        kataan::VERSION
    );
}
