//! Regression tests for divergences found by differential testing: small
//! deterministic programs whose results were compared against an independent
//! engine and the spec. Each test pins the spec-mandated result.
use super::*;
use crate::parser::Parser;

/// Runs `src` and renders the program's final value.
fn run(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

/// `continue` must reach the innermost enclosing *loop*, past any `switch`
/// or labeled block in between, and `continue L` must reach loop `L` even
/// when a `switch` encloses that loop (the compiler's continue-target stack
/// was not parallel to its break-target stack, so the label resolved to the
/// wrong loop, and a `continue` in a labeled block was dropped, looping
/// forever). A label chain `A: B: for` names the loop with both labels.
#[test]
fn continue_targets_past_switch_and_labeled_blocks() {
    let src = r#"
        var r = [];
        switch (1) { case 1:
            L1: for (var i = 0; i < 2; i++) {
                for (var j = 0; j < 3; j++) { r.push(i + ":" + j); if (j === 0) continue L1; r.push("x"); }
                r.push("after");
            }
        }
        function g() {
            var r = [];
            L0: for (var k = 0; k < 2; k++) {
                switch (k) { case 0: case 1:
                    for (var j = 0; j < 3; j++) { r.push(k + ":" + j); if (j === 1) continue L0; }
                    r.push("after");
                }
            }
            return r.join();
        }
        function h() {
            var r = [];
            for (var i = 0; i < 3; i++) { B: { r.push(i); if (i < 2) continue; r.push("b"); } r.push("e" + i); }
            for (var i = 0; i < 3; i++) { switch (i) { case 0: continue; case 1: r.push("one"); } r.push("e" + i); }
            return r.join();
        }
        function n() {
            var r = [];
            A: B: for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 3; j++) { if (j == 1) continue A; if (i == 2) break B; r.push(i + "" + j); }
                r.push("no");
            }
            return r.join();
        }
        r.join() + "|" + g() + "|" + h() + "|" + n()
    "#;
    assert_eq!(
        run(src),
        "0:0,1:0|0:0,0:1,1:0,1:1|0,1,2,b,e2,one,e1,e2|00,10"
    );
}

/// Number→String rounding. `ToString` picks the shortest round-tripping
/// digits and, between two equally short candidates equally close to the
/// value, the even one; `toPrecision`/`toExponential` round an exact tie up
/// (and a carry adds a decade: `9.99` to 2 digits is `10`); a radix other
/// than 10 gives the shortest numeral that reads back exactly (it truncated
/// fractions to 20 digits and saturated integer parts at 2^64).
#[test]
fn number_to_string_rounding() {
    let src = r#"
        [String(2 ** -25), String(-1482983220584872.25), (2 ** -25).toExponential(),
         (2 ** -25).toPrecision(17), (8.225e18).toPrecision(3), (2.5e17).toPrecision(1),
         (9.99).toPrecision(2), (99.99).toPrecision(1), (0).toPrecision(3), (-1.5).toPrecision(1),
         (1.25).toExponential(1), (0).toExponential(2), (123.456).toPrecision(5),
         (1e-10).toString(2).length, (2 ** 70 + 2 ** 20).toString(2).slice(-21),
         (255.5).toString(16), (1e21).toString(36), (0.1).toString(3),
         (1.550650161452515e-266).toString(21).length].join(" ")
    "#;
    assert_eq!(
        run(src),
        "2.9802322387695312e-8 -1482983220584872.2 2.9802322387695312e-8 \
         2.9802322387695313e-8 8.23e+18 3e+17 10 1e+2 0.00 -2 1.3e+0 0.00e+0 123.46 \
         88 100000000000000000000 ff.8 5v1j4f4ds7a000 0.0022002200220022002200220022002201 215"
    );
}
