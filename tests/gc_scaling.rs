//! Heap-growth regression gate for the bytecode tier.
//!
//! The Test262 corpus is blind to memory: a test that allocates 8 GB and gets
//! the right answer passes. That is not hypothetical — the SpiderMonkey
//! `Date/dst-offset-caching-*` staging tests peaked at **8.45 GB** and were
//! written off as harness noise, because the only symptom the runner could
//! report was "worker exited mid-test".
//!
//! The cause was a documented restriction rather than a slip: collection was
//! confined to the *outermost* activation, because a suspended frame's register
//! window is a `Vec` on the Rust stack that the collector cannot see. So a loop
//! inside a function — which is to say nearly every real program — never
//! collected. Frames now publish their registers on the audited JS-to-JS call
//! path (`Ctx::frame_shadow`), and the safepoint collects whenever the whole
//! descent published.
//!
//! # What these tests assert
//!
//! Live object count after the program returns, at `n` and `4n`. A collecting
//! engine finishes with roughly the same small heap either way; a
//! non-collecting one finishes holding every object it ever made, so the count
//! scales with `n`. The assertion is on the *ratio*, so it does not depend on
//! the GC threshold, the growth heuristic, or how many objects the harness
//! itself allocates.
//!
//! These deliberately test the tier directly (`compile_program` +
//! `run_program_capturing` over a caller-owned `Realm`) rather than
//! `execute_typed`, which owns its realm and drops it before returning — there
//! would be nothing left to measure.

use kataan::nbvm::{compile_program, run_program_capturing};
use kataan::parser::Parser;
use kataan::realm::Realm;

/// Runs `src` on the bytecode tier and returns the live object count afterwards.
///
/// Panics rather than returning a `Result`: a workload here that fails to
/// compile or throws is a broken test, not a measurement.
fn live_objects_after(src: &str) -> usize {
    let program = Parser::parse_program(src).expect("parse");
    let protos = compile_program(&program).expect("compile to bytecode");
    let mut realm = Realm::new();
    run_program_capturing(&mut realm, &protos, 0, &[]).expect("run");
    realm.object_count()
}

/// Asserts that four times the work does not leave four times the heap.
///
/// The bound is generous (2x for a 4x workload) because it is a leak detector,
/// not a budget: a tier that collects lands near 1.0, one that does not lands
/// near 4.0, and nothing sits in between.
fn assert_flat(label: &str, workload: impl Fn(usize) -> String, n: usize) {
    let small = live_objects_after(&workload(n));
    let large = live_objects_after(&workload(n * 4));
    let ratio = large as f64 / small.max(1) as f64;
    println!(
        "{label}: n={n} -> {small} objects, n={} -> {large} objects (x{ratio:.2})",
        n * 4
    );
    assert!(
        ratio < 2.0,
        "{label}: 4x the iterations left {ratio:.2}x the live objects \
         ({small} -> {large}); the collector is not reclaiming inside function bodies"
    );
}

#[test]
fn loop_inside_a_function_collects_its_garbage() {
    assert_flat(
        "object literal in a function-local loop",
        |n| {
            format!(
                "function f(){{var s=0;for(var i=0;i<{n};i++){{var o={{a:i}};s+=o.a;}}return s;}} f();"
            )
        },
        20_000,
    );
}

#[test]
fn garbage_from_a_nested_call_is_collected() {
    // The allocation happens one frame deeper than the loop, so the collector
    // has to see both the loop frame's registers and the callee's.
    assert_flat(
        "callee-allocated object",
        |n| {
            format!(
                "function mk(i){{return {{a:i,b:[i]}};}}\
                 function f(){{var s=0;for(var i=0;i<{n};i++){{s+=mk(i).a;}}return s;}} f();"
            )
        },
        20_000,
    );
}

#[test]
fn garbage_three_frames_deep_is_collected() {
    assert_flat(
        "three-deep call chain",
        |n| {
            format!(
                "function c(i){{return {{v:i}};}}\
                 function b(i){{return c(i).v;}}\
                 function a(i){{return b(i)+1;}}\
                 function f(){{var s=0;for(var i=0;i<{n};i++){{s+=a(i);}}return s;}} f();"
            )
        },
        20_000,
    );
}

#[test]
fn strings_built_in_a_function_loop_are_collected() {
    assert_flat(
        "concatenated strings",
        |n| {
            format!(
                "function f(){{var s=0;for(var i=0;i<{n};i++){{var t=\"k\"+i;s+=t.length;}}return s;}} f();"
            )
        },
        20_000,
    );
}

#[test]
fn a_top_level_loop_still_collects() {
    // The outermost activation always could collect; this guards against the
    // frame-publishing bookkeeping breaking the case that already worked.
    assert_flat(
        "top-level loop",
        |n| format!("var s=0;for(var i=0;i<{n};i++){{var o={{a:i}};s+=o.a;}}"),
        20_000,
    );
}

#[test]
fn live_objects_survive_collection_inside_a_function() {
    // Reclaiming garbage is only half of it: anything still reachable from a
    // deeper frame's registers must come through intact. A missing root would
    // show up here as corrupted data rather than as a smaller heap.
    let src = "
        function inner(o, i) { return { v: o.a + i, arr: [o, i] }; }
        function mid(i) { var o = { a: i }; var r = inner(o, i); return r.v + r.arr[1]; }
        function outer(n) {
          var acc = 0, keep = [];
          for (var i = 0; i < n; i++) {
            acc += mid(i);
            if (i % 500 === 0) { keep.push({ i: i, s: 'k' + i }); }
          }
          for (var j = 0; j < keep.length; j++) {
            if (keep[j].i !== j * 500 || keep[j].s !== 'k' + (j * 500)) { return -1; }
          }
          return keep.length;
        }
        outer(40000);
    ";
    let program = Parser::parse_program(src).expect("parse");
    let protos = compile_program(&program).expect("compile to bytecode");
    let mut realm = Realm::new();
    let (value, _) = run_program_capturing(&mut realm, &protos, 0, &[]).expect("run");
    assert_eq!(
        realm.to_display_string(value),
        "80",
        "an object retained across collections was corrupted or lost"
    );
}
