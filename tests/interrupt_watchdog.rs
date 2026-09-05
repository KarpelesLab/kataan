//! End-to-end test of the host interrupt hook: a watchdog thread trips the flag
//! and the engine must stop promptly, without the script being able to swallow
//! the deadline.

use kataan::interrupt::Interrupt;
use kataan::limits::Limits;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Runs `src` on a worker thread with a watchdog that trips `after`, returning
/// how long the engine took to stop.
///
/// The engine itself is `!Send` (it holds `Rc`s), which is the whole reason the
/// flag lives in its own `Arc` allocation — so the interpreter is *created on
/// the worker thread* and only the `Interrupt` handle crosses between them.
fn run_until_interrupted(src: &'static str, after: Duration) -> Duration {
    let flag = Interrupt::new();
    let watchdog = flag.clone();
    let (tx, rx) = mpsc::channel();

    let worker = thread::spawn(move || {
        let started = Instant::now();
        let _ = kataan::nbvm::execute_typed_interruptible(src, Limits::default(), Some(flag));
        let _ = tx.send(started.elapsed());
    });

    thread::sleep(after);
    watchdog.trip();
    let elapsed = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("engine did not stop after the watchdog tripped");
    worker.join().expect("worker panicked");
    elapsed
}

/// The base case: an unbounded loop must stop when the watchdog trips.
#[test]
fn infinite_loop_is_interrupted() {
    let took = run_until_interrupted("while (true) {}", Duration::from_millis(150));
    assert!(
        took < Duration::from_secs(5),
        "engine ran {took:?} after a 150ms deadline"
    );
}

/// The case the whole design turns on: a script must not be able to catch the
/// interrupt and keep going. If this ever regresses, the watchdog silently
/// becomes advisory.
#[test]
fn interrupt_cannot_be_swallowed_by_catch() {
    let took = run_until_interrupted(
        "while (true) { try { let x = 1; } catch (e) { /* swallow */ } }",
        Duration::from_millis(150),
    );
    assert!(
        took < Duration::from_secs(5),
        "a try/catch inside the loop swallowed the interrupt ({took:?})"
    );
}

/// `finally` must not be able to resurrect execution either.
#[test]
fn interrupt_is_not_deferred_by_finally() {
    let took = run_until_interrupted(
        "while (true) { try { let x = 1; } finally { let y = 2; } }",
        Duration::from_millis(150),
    );
    assert!(
        took < Duration::from_secs(5),
        "a finally block deferred the interrupt ({took:?})"
    );
}

/// A `for-of` over a materialized iterable is a different loop driver from the
/// bytecode back-edge, so it needs its own check point.
#[test]
fn for_of_loop_is_interrupted() {
    let took = run_until_interrupted(
        "const s='x'.repeat(2000000); let n=0; for(;;){ for (const c of s) n++; }",
        Duration::from_millis(200),
    );
    assert!(
        took < Duration::from_secs(8),
        "for-of did not observe the interrupt ({took:?})"
    );
}

/// An untripped flag must not disturb a normal program, and the engine must not
/// clear the caller's flag behind its back.
#[test]
fn untripped_flag_runs_normally_and_is_not_cleared() {
    let flag = Interrupt::new();
    let (out, _) = kataan::nbvm::execute_typed_interruptible(
        "let s=0; for (let i=0;i<100000;i++) s+=i; console.log(s);",
        Limits::default(),
        Some(flag.clone()),
    )
    .expect("uninterrupted program should complete");
    assert_eq!(out.trim(), "4999950000");
    assert!(!flag.is_tripped(), "engine tripped the flag on its own");

    // A tripped flag stays tripped until the host clears it — re-arming is the
    // embedder's explicit decision, never an engine side effect.
    flag.trip();
    assert!(flag.is_tripped());
    flag.clear();
    assert!(!flag.is_tripped());
}
