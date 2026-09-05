//! Asymptotic-complexity regression gate.
//!
//! The Test262 corpus is structurally blind to performance: a test that passes
//! in 23 seconds passes. Three severe cliffs landed and survived clean corpus
//! runs before anyone noticed them by hand —
//!
//! | | before | after | how it surfaced |
//! | --- | --- | --- | --- |
//! | `Array.prototype.sort` (insertion sort) | 23 s | 23 ms | one test *timing out* |
//! | anchored regex scanning the whole subject | 13.4 s | 25 ms | a benchmark re-run by habit |
//! | every string property access materializing | 3.9 s | 78 ms | an ad-hoc scaling probe |
//!
//! Two of those three were found by luck. This file makes the check routine.
//!
//! # Why ratios, not milliseconds
//!
//! These tests assert on the *shape* of the cost curve, never on wall-clock
//! time. Each workload runs at `n` and `4n` and asserts on `t(4n) / t(n)`: ~1
//! for O(1), ~4 for linear, ~4.3 for `n log n`, ~16 for quadratic. Both
//! measurements move together under machine load, so the ratio survives a box
//! that swings between load 9 and load 78 — which this one does. An
//! absolute-time budget would either flake constantly or be set so loose it
//! catches nothing.
//!
//! # Why the script times itself
//!
//! Each workload brackets *only* its hot loop with `Date.now()` and prints the
//! elapsed milliseconds. Timing the whole process instead would fold in `O(n)`
//! setup (`'x'.repeat(n)`, filling an array), which at large `n` dominates and
//! makes every workload look linear; subtracting a separately-measured setup
//! run is worse still, because the noise in two independent measurements swamps
//! the difference. Both drafts of this file produced false results that way —
//! the second reported `delete` as quadratic when direct measurement showed it
//! linear.
//!
//! Note this pins the workloads to the tree-walker: an in-script `Date.now()`
//! makes the whole program fall back from the bytecode VM. That is fine here —
//! an asymptotic bug shows in either tier, and the alternative measures setup
//! rather than the operation.
//!
//! Bounds are deliberately slack: they catch a *category* error (linear became
//! quadratic, O(1) became O(n)), not constant-factor drift.

use kataan::limits::Limits;

/// Runs `src`, which must print exactly one number: the milliseconds its hot
/// loop took.
fn timed_ms(src: &str) -> f64 {
    let (out, _) = kataan::nbvm::execute_typed(src, Limits::default())
        .unwrap_or_else(|e| panic!("workload threw: {e:?}"));
    out.trim()
        .lines()
        .next_back()
        .and_then(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("workload printed no timing: {out:?}"))
        .max(1.0) // floor: sub-millisecond noise must not inflate the ratio
}

/// Best-of-3 — the minimum is the least noisy estimator under contention.
fn best_ms(src: &str) -> f64 {
    (0..3).map(|_| timed_ms(src)).fold(f64::INFINITY, f64::min)
}

/// Asserts that scaling `n` by 4 does not scale the timed loop by more than
/// `bound`. `make(n)` must print the loop's elapsed milliseconds.
fn assert_scaling(name: &str, n: usize, bound: f64, make: impl Fn(usize) -> String) {
    let small = best_ms(&make(n));
    let large = best_ms(&make(n * 4));
    let ratio = large / small;
    assert!(
        ratio <= bound,
        "{name}: scaling n x4 scaled the work x{ratio:.1} (bound {bound:.1}) — \
         {small:.0}ms at n={n} vs {large:.0}ms at n={}. Linear is ~4, quadratic ~16.",
        n * 4
    );
}

/// Wraps `body` so only it is timed, and prints the elapsed milliseconds.
fn timed(setup: &str, body: &str) -> String {
    format!("{setup}\nconst __t=Date.now();\n{body}\nconsole.log(Date.now()-__t);")
}

/// `Array.prototype.sort` must be `n log n` (~4.3), not the insertion sort it
/// once was: a 262 144-element typed array took 23 s.
#[test]
fn sort_is_linearithmic() {
    assert_scaling("Array.prototype.sort", 20_000, 8.0, |n| {
        timed(
            &format!("const a=new Array({n}); for(let i=0;i<{n};i++)a[i]=(i*7919)%{n};"),
            "a.sort();",
        )
    });
}

/// A `^`-anchored match can only start at one offset, so growing the *subject*
/// must not grow the cost. This regressed to a full scan per call when the
/// class start-filter searched past the last candidate offset.
#[test]
fn anchored_regex_ignores_subject_length() {
    assert_scaling("anchored regex", 200_000, 2.5, |n| {
        timed(
            &format!("const re=/^(?:(?:a)+|(?:b)+)/, s='x'.repeat({n});"),
            "for(let i=0;i<5000;i++) re.test(s);",
        )
    });
}

/// `s.length` is O(1). It was O(n) — every string property access materialized
/// the string — which made every `for (i = 0; i < s.length; i++)` quadratic.
#[test]
fn string_length_is_constant_time() {
    assert_scaling("String.length", 100_000, 2.5, |n| {
        timed(
            &format!("const s='x'.repeat({n});"),
            "let c=0; for(let i=0;i<100000;i++) c+=s.length;",
        )
    });
}

/// Indexed string reads are O(1) on an ASCII string (byte index == unit index).
#[test]
fn string_index_is_constant_time() {
    assert_scaling("String charCodeAt", 100_000, 2.5, |n| {
        timed(
            &format!("const s='x'.repeat({n});"),
            "let c=0; for(let i=0;i<100000;i++) c+=s.charCodeAt(i%1000);",
        )
    });
}

/// Tearing down a dictionary-mode object is linear. `delete` was a linear scan
/// of the insertion-order vector, so this was quadratic.
#[test]
fn property_delete_is_linear() {
    assert_scaling("delete", 20_000, 8.0, |n| {
        timed(
            &format!("const o={{}}; for(let i=0;i<{n};i++)o['k'+i]=i;"),
            &format!("for(let i=0;i<{n};i++)delete o['k'+i];"),
        )
    });
}

/// Reading one array element is O(1). `at`/`indexOf` copied the whole backing
/// store per call.
#[test]
fn array_element_read_is_constant_time() {
    assert_scaling("Array.at", 100_000, 2.5, |n| {
        timed(
            &format!("const a=new Array({n}).fill(1);"),
            "let c=0; for(let i=0;i<100000;i++) c+=a.at(0);",
        )
    });
}

/// Appending is amortised O(1); `push` paid for a whole-array hole mask built
/// before its own fast path.
#[test]
fn array_push_is_linear() {
    assert_scaling("Array.push", 20_000, 8.0, |n| {
        timed("", &format!("const a=[]; for(let i=0;i<{n};i++)a.push(i);"))
    });
}

/// Draining a built-in iterator is linear; `next` cloned the whole backing
/// buffer per step, in two separate implementations.
#[test]
fn iterator_drain_is_linear() {
    assert_scaling("iterator drain", 20_000, 8.0, |n| {
        timed(
            &format!("const s='x'.repeat({n});"),
            "let c=0; for(const ch of s) c++;",
        )
    });
}

/// Template and `concat` string building must be as linear as `+=` — both once
/// copied into a flat buffer per append.
#[test]
fn string_building_is_linear() {
    assert_scaling("string building", 20_000, 8.0, |n| {
        timed(
            "",
            &format!(
                "let s=''; for(let i=0;i<{n};i++) s=`${{s}}x`; \
                 let t=''; for(let i=0;i<{n};i++) t=t.concat('y');"
            ),
        )
    })
}
