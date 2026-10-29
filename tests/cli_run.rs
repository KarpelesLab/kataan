//! The `kataan` command-line binary, run as a child process.
#![cfg(feature = "cli")]

use std::process::Command;

/// Runs `kataan run -e <source>` and returns (stdout, stderr, success).
fn run_e(source: &str) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_kataan"))
        .args(["run", "-e", source])
        .output()
        .expect("spawn kataan");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// An uncaught exception does not swallow what the program printed before
/// it: the earlier `console.log` lines reach stdout, the error reaches
/// stderr, and the exit status is a failure (all output was dropped).
#[test]
fn uncaught_throw_keeps_earlier_output() {
    let (stdout, stderr, ok) = run_e(
        r#"console.log("before"); (function () { console.log("in f"); throw new TypeError("boom"); })();"#,
    );
    assert_eq!(stdout, "before\nin f\n");
    assert!(stderr.contains("TypeError: boom"), "stderr: {stderr}");
    assert!(!ok);
}

/// A program that completes prints its output and its completion value.
#[test]
fn completed_program_prints_output_and_completion() {
    let (stdout, _, ok) = run_e(r#"console.log("x"); 1 + 2"#);
    assert_eq!(stdout, "x\n3\n");
    assert!(ok);
}

/// `kataan run` does not expose the Test262 host hooks: they are harness
/// plumbing (`$262_IsHTMLDDA`, `$262_agent_start`, …), not language globals.
#[test]
fn no_test262_host_hooks_in_cli() {
    let (stdout, _, ok) = run_e(
        r#"console.log(Object.getOwnPropertyNames(globalThis).filter(function (n) { return n.indexOf("262") >= 0; }).join(), typeof $262_IsHTMLDDA, typeof $262_createRealm, typeof $262_agent_start)"#,
    );
    assert!(
        stdout.starts_with(" undefined undefined undefined\n"),
        "stdout: {stdout}"
    );
    assert!(ok);
}
