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

/// IsIntegralNumber is `n % 1 == 0`. An `i64` round-trip cast saturates
/// past 2^63, so `Number.isInteger(1e20)` was false (and `Atomics.pause`
/// rejected it; RegExp `@@split`'s ToUint32(limit) truncated past 2^63).
#[test]
fn integral_number_predicates_beyond_i64() {
    let src = r#"
        var p; try { Atomics.pause(1e20); p = "ok"; } catch (e) { p = e.name; }
        [Number.isInteger(1e20), Number.isInteger(-1e300), Number.isInteger(2 ** 60),
         Number.isInteger(1.5), Number.isSafeInteger(1e20), Number.isSafeInteger(2 ** 53 - 1),
         p, "a,b,c".split(/,/, 2 ** 64 + 2 ** 12).length].join()
    "#;
    assert_eq!(run(src), "true,true,true,false,false,true,ok,3");
}

/// A strict-mode write that would add an index to a non-extensible array
/// (past its end, or into a hole) throws a TypeError; the dense element store
/// used to grow or fill it silently (only seal/freeze were caught). Sloppy
/// mode drops the write. Hot loops (JIT tiers) behave the same.
#[test]
fn strict_index_add_on_non_extensible_array_throws() {
    let src = r#"
        "use strict";
        function t(f) { try { f(); return "ok"; } catch (e) { return e.name; } }
        var r = [
            t(() => { var a = [1, 2]; Object.preventExtensions(a); a[2] = 0; }),
            t(() => { var a = [1, 2]; Object.preventExtensions(a); a[5] = 0; }),
            t(() => { var a = [1, , 3]; Object.preventExtensions(a); a[1] = 0; }),
            t(() => { var a = [1, 2]; Object.preventExtensions(a); a[0] = 9; }),
            t(() => { var a = [1, 2]; Object.preventExtensions(a); a["2"] = 0; }),
        ];
        function hot() {
            var a = [1, 2]; Object.preventExtensions(a); var c = 0;
            for (var i = 0; i < 2000; i++) { try { a[2 + (i % 3)] = i; } catch (e) { c++; } }
            return c + ":" + a.length;
        }
        var sloppy = Function("var a = [1, , 3]; Object.preventExtensions(a); a[1] = 1; a[4] = 1;"
            + " return a.length + ':' + (1 in a);")();
        r.join() + "|" + hot() + "|" + sloppy
    "#;
    assert_eq!(
        run(src),
        "TypeError,TypeError,TypeError,ok,TypeError|2000:2|3:false"
    );
}

/// Lone surrogates survive string builtins, and halves meeting at a seam
/// join into one code point. `replace`/`replaceAll` with a string pattern,
/// the host `join` (`Array.prototype.toString`), `String(v)`, `String.raw`,
/// `Error` messages and `Error.prototype.toString` went through Rust
/// `String`s and folded each lone half to U+FFFD; `padStart`/`padEnd` and
/// RegExp `@@replace` concatenated without pairing (`[...s].length` was 2).
#[test]
fn lone_surrogates_survive_string_builtins() {
    let src = r#"
        var h = "\uD800", l = "\uDC00", q = h + "x" + l;
        var hex = s => [...s].map(c => c.codePointAt(0).toString(16)).join(".");
        [q.replaceAll("b", "$&"), q.replace("x", "[$&]"), q.replaceAll("x", () => h),
         "\u{10000}".replace(l, "y"), q.replace(/x/g, "$`"), q.replace(/x/, h + "$'"),
         [q].toString(), String([h, [h]]), String.raw({raw: [h, l]}, ""), new Error(h).message,
         String(new Error(h)), h.padEnd(2, l), l.padStart(2, h)].map(hex).join(" ")
    "#;
    assert_eq!(
        run(src),
        "d800.78.dc00 d800.5b.78.5d.dc00 d800.10000 d800.79 d800.10000 d800.10000.dc00 \
         d800.78.dc00 d800.2c.d800 10000 d800 45.72.72.6f.72.3a.20.d800 10000 10000"
    );
}

/// `Array.prototype.join` pairs a lone high and a lone low surrogate meeting
/// at a seam (element/element, element/separator) into one code point.
#[test]
fn join_pairs_surrogate_halves_at_seams() {
    let src = r#"
        var h = "\uD800", l = "\uDC00";
        var hex = s => [...s].map(c => c.codePointAt(0).toString(16)).join(".");
        [[h, l].join(""), [h, l].join(h), [h + "a", l].join(h), [1, l].join(h),
         [{toString() { return l; }}].join(h), new Int8Array(2).join(h)].map(hex).join(" ")
    "#;
    assert_eq!(
        run(src),
        "10000 d800.10000 d800.61.10000 31.10000 dc00 30.d800.30"
    );
}

/// A RegExp `searchValue` whose `@@replace` is undefined is a string pattern:
/// ToString runs its (possibly user) `toString`, so `/./iyg` searches for
/// `"/./giy"` (the canonical flags order), not for its literal source text.
#[test]
fn replace_tostrings_a_regexp_pattern_without_replace() {
    let src = r#"
        var r = /./iyg, r2 = /x/g;
        Object.defineProperty(r, Symbol.replace, { value: undefined });
        Object.defineProperty(r2, Symbol.replace, { value: undefined });
        r2.toString = function () { return "Q"; };
        ["a /./giy /./iyg a".replaceAll(r, "z"), "aQb".replace(r2, "z")].join("|")
    "#;
    assert_eq!(run(src), "a z /./iyg a|azb");
}

/// The default `sort` comparator orders by UTF-16 code unit: an astral
/// character (surrogate units 0xD800..) sorts before U+E000..U+FFFF, and lone
/// surrogates keep their own order instead of all folding to U+FFFD.
#[test]
fn default_sort_orders_by_utf16_code_unit() {
    let src = r#"
        var hex = a => a.map(s => String(s).codePointAt(0).toString(16)).join(",");
        [hex(["￿", "\u{10000}"].sort()), hex(["\uDC00", "\uD800", "a"].sort()),
         hex(["￿", "\u{10000}", {toString() { return ""; }}].toSorted()),
         hex(["\uDC00", "\uD800", 1].sort())].join(" ")
    "#;
    assert_eq!(
        run(src),
        "10000,ffff 61,d800,dc00 10000,e000,ffff 31,d800,dc00"
    );
}
