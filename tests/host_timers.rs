//! §4.1 host timers & event loop — integration tests.
//!
//! Each test builds an [`Interp`], installs the timer / event-loop host layer
//! (`kataan::host::timers::install`), runs a JS snippet that schedules timers,
//! microtasks, and `process.nextTick` jobs, drives the loop to quiescence with
//! `kataan::host::timers::run_event_loop`, and asserts the observed ordering via
//! `interp.output()` (the captured `console.log` lines).
//!
//! The engine's `Interp::run` already drains the Promise **microtask** queue at
//! top level, so the synchronous batch resolves `sync → microtasks` on its own;
//! `run_event_loop` then owns the **macrotask** (timer) turns and the
//! `nextTick`-before-microtask ordering for callbacks scheduled *inside* the
//! loop.

use kataan::host::timers;
use kataan::nbexec::Interp;
use kataan::parser::Parser;

/// Installs the timer layer, runs `src`, drives the event loop, and returns the
/// captured `console.log` lines (blank lines dropped).
fn run(src: &str) -> Vec<String> {
    let program = Parser::parse_program(src).expect("parse snippet");
    let mut interp = Interp::new();
    timers::install(&mut interp);
    interp.run(&program).expect("run snippet");
    timers::run_event_loop(&mut interp).expect("event loop drains cleanly");
    interp
        .output()
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Position of the first line equal to `needle` (panics if absent).
fn idx(lines: &[String], needle: &str) -> usize {
    lines
        .iter()
        .position(|l| l == needle)
        .unwrap_or_else(|| panic!("missing line {needle:?} in {lines:?}"))
}

#[test]
fn sync_then_microtask_then_macrotask() {
    let out = run(
        "Promise.resolve().then(function () { console.log('micro'); });\
         setTimeout(function () { console.log('macro'); });\
         console.log('sync');",
    );
    assert_eq!(out, ["sync", "micro", "macro"]);
}

#[test]
fn nexttick_runs_before_microtask() {
    // Scheduled *inside* a timer so the ordering is governed entirely by
    // run_event_loop's per-turn drain (nextTick strictly before microtasks).
    let out = run("setTimeout(function () {\
             Promise.resolve().then(function () { console.log('promise'); });\
             process.nextTick(function () { console.log('nextTick'); });\
         });");
    assert!(
        idx(&out, "nextTick") < idx(&out, "promise"),
        "nextTick must run before the microtask: {out:?}"
    );
}

#[test]
fn interval_fires_n_times_then_cleared() {
    let out = run("var n = 0;\
         var id = setInterval(function () {\
             n++; console.log('tick' + n);\
             if (n === 3) clearInterval(id);\
         }, 10);");
    assert_eq!(out, ["tick1", "tick2", "tick3"]);
}

#[test]
fn clear_timeout_cancels_pending_callback() {
    let out = run(
        "var id = setTimeout(function () { console.log('should-not-run'); }, 10);\
         clearTimeout(id);\
         console.log('done');",
    );
    assert_eq!(out, ["done"]);
}

#[test]
fn setimmediate_and_extra_arguments() {
    // setImmediate fires as a zero-delay macrotask; extra args after the delay
    // are forwarded to the callback.
    let out = run(
        "setImmediate(function (a, b) { console.log('immediate:' + (a + b)); }, 2, 3);\
         setTimeout(function (x) { console.log('timeout:' + x); }, 0, 40);\
         console.log('sync');",
    );
    // 'sync' is synchronous; the two zero-delay macrotasks fire in scheduling
    // order (immediate was scheduled first).
    assert_eq!(idx(&out, "sync"), 0);
    assert!(out.contains(&"immediate:5".to_string()), "{out:?}");
    assert!(out.contains(&"timeout:40".to_string()), "{out:?}");
    assert!(
        idx(&out, "immediate:5") < idx(&out, "timeout:40"),
        "{out:?}"
    );
}

#[test]
fn queue_microtask_runs_before_timers() {
    let out = run("setTimeout(function () { console.log('timer'); }, 0);\
         queueMicrotask(function () { console.log('micro'); });\
         console.log('sync');");
    assert_eq!(out, ["sync", "micro", "timer"]);
}

#[test]
fn nested_timeouts_fire_in_due_order() {
    // A later-scheduled shorter delay fires before an earlier-scheduled longer
    // one (the virtual clock orders by due time, not insertion).
    let out = run("setTimeout(function () { console.log('slow'); }, 100);\
         setTimeout(function () { console.log('fast'); }, 10);");
    assert_eq!(out, ["fast", "slow"]);
}

#[test]
fn abort_controller_fires_listener_synchronously() {
    let out = run("var c = new AbortController();\
         c.signal.addEventListener('abort', function () {\
             console.log('aborted:' + c.signal.reason);\
         });\
         console.log('before:' + c.signal.aborted);\
         c.abort('stop');\
         console.log('after:' + c.signal.aborted);");
    assert_eq!(out, ["before:false", "aborted:stop", "after:true"]);
}

#[test]
fn abort_signal_static_abort() {
    let out = run("var s = AbortSignal.abort('reason-x');\
         console.log(s.aborted + ':' + s.reason);");
    assert_eq!(out, ["true:reason-x"]);
}

#[test]
fn abort_signal_timeout_aborts_via_event_loop() {
    let out = run("var s = AbortSignal.timeout(5);\
         s.addEventListener('abort', function () { console.log('timedout:' + s.reason.name); });\
         console.log('start:' + s.aborted);");
    assert_eq!(out, ["start:false", "timedout:TimeoutError"]);
}

#[test]
fn process_nexttick_is_additive() {
    // A pre-existing `process` object (e.g. one another host area populated with
    // `platform`/`argv`) must survive install — only `nextTick` is added.
    let program = Parser::parse_program(
        "console.log(process.foo + ':' + (typeof process.nextTick));\
         process.nextTick(function () { console.log('ticked'); });",
    )
    .expect("parse");
    let seed = Parser::parse_program("globalThis.process = { foo: 42 };").expect("parse seed");

    let mut interp = Interp::new();
    interp.run(&seed).expect("seed process");
    timers::install(&mut interp);
    interp.run(&program).expect("run");
    timers::run_event_loop(&mut interp).expect("loop");

    let lines: Vec<String> = interp
        .output()
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    assert_eq!(lines, ["42:function", "ticked"]);
}

#[test]
fn clear_interval_from_outside_before_first_fire() {
    let out = run(
        "var id = setInterval(function () { console.log('nope'); }, 10);\
         clearInterval(id);\
         console.log('cleared');",
    );
    assert_eq!(out, ["cleared"]);
}
