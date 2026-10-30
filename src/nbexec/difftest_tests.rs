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

/// A global `let`/`const`/`class` is not a property of the global object, so
/// `globalThis.name = v` creates a separate property and leaves the binding
/// alone (the write-side mirror applied to every global binding).
#[test]
fn global_object_write_leaves_lexical_binding() {
    let src = r#"
        let w = 1; const c = 2; class K {}
        function rw() { return w; }
        for (var i = 0; i < 3; i++) rw();
        globalThis.w = 5; globalThis.c = 7; globalThis.K = 3;
        var v = 1; globalThis.v = 9;
        [w, globalThis.w, c, globalThis.c, typeof K, globalThis.K, rw(), v].join()
    "#;
    assert_eq!(run(src), "1,5,2,7,function,3,1,9");
}

/// `delete globalThis.x` of a configurable global property (an indirect-eval
/// `var`, an implicit global) also removes its binding mirror: the bare name
/// is unresolvable afterwards, in code compiled before the delete as well.
#[test]
fn delete_global_object_property_drops_binding() {
    let src = r#"
        (0, eval)("var ev = 1"); (0, eval)("var ev2 = 2");
        function rd(f) { try { return f(); } catch (e) { return e.name; } }
        function r2() { return ev2; }
        for (var i = 0; i < 5; i++) rd(r2);
        var a = delete globalThis.ev, b = Reflect.deleteProperty(globalThis, "ev2");
        var vv = 1;
        [a, b, typeof ev, rd(() => ev), rd(r2), "ev" in globalThis, delete globalThis.vv, vv].join()
    "#;
    assert_eq!(
        run(src),
        "true,true,undefined,ReferenceError,ReferenceError,false,false,1"
    );
}

/// A bare assignment to a name the global object holds as an own accessor
/// runs its setter, in sloppy code and in a function, rather than dropping
/// the write (a script with a function declaration took that path).
#[test]
fn sloppy_write_to_global_accessor_calls_setter() {
    let src = r#"
        var log = [];
        Object.defineProperty(globalThis, "acc", {
            set(v) { log.push("set" + v); }, get() { log.push("get"); return 42; },
            configurable: true });
        acc = 3;
        function wr() { acc = 4; }
        wr();
        Object.defineProperty(globalThis, "ro", { get() { return 1; }, configurable: true });
        var t = (function () { "use strict"; try { ro = 2; return "no"; } catch (e) { return e.name; } })();
        ro = 3;
        [log.join(), acc, t, ro].join("|")
    "#;
    assert_eq!(run(src), "set3,set4|42|TypeError|1");
}

/// A built-in converts only the arguments its algorithm reads: an argument
/// past `Math.abs`'s one parameter (or `Date.UTC`'s and `new Date`'s seven) is not
/// ToNumber'd, so a Symbol there is no TypeError and a `valueOf` there does
/// not run; `padStart`/`padEnd` on a string already long enough never
/// ToString the fill string.
#[test]
fn builtins_coerce_only_the_arguments_they_read() {
    let src = r#"
        var log = [], o = { valueOf() { log.push("v"); return 1; }, toString() { log.push("s"); return "-"; } };
        var s = Symbol("s");
        var r = [Math.abs(-2, s), Math.clz32(1, s), Math.pow(2, 3, s), Math.atan2(0, 1, s),
                 Math.imul(3, 4, s), Date.UTC(1970, 0, 1, 0, 0, 0, 5, s),
                 "abc".padStart(2, o), "abc".padEnd(3, o), "ab".padEnd(4, o),
                 new Date(1970, 0, 1, 0, 0, 0, 7, s).getTime()];
        r.join() + "|" + log.join() + "|" + Math.hypot(3, 4, o)
    "#;
    assert_eq!(
        run(src),
        "2,31,8,0,12,5,abc,abc,ab--,7|s|5.0990195135927845"
    );
}

/// A Proxy's internal methods are each called exactly as the algorithm says:
/// `Object.hasOwn(proxy, k)` asks its `getOwnPropertyDescriptor` trap (it
/// was false for every proxy), `Object.seal` defines `{configurable: false}`
/// per key without reading descriptors (only `freeze` reads them), and
/// `bind` asks a proxy target's `getPrototypeOf` trap for the bound
/// function's prototype.
#[test]
fn proxy_traps_for_has_own_seal_and_bind() {
    let src = r#"
        var log = [];
        function P(t) {
            return new Proxy(t, {
                getOwnPropertyDescriptor(t, k) { log.push("gopd:" + k); return Reflect.getOwnPropertyDescriptor(t, k); },
                defineProperty(t, k, d) { log.push("def:" + k); return Reflect.defineProperty(t, k, d); },
                getPrototypeOf(t) { log.push("gpo"); return Reflect.getPrototypeOf(t); },
            });
        }
        var r = [Object.hasOwn(P({ a: 1 }), "a"), Object.hasOwn(P({}), "a")];
        r.push(log.splice(0).join());
        Object.seal(P({ a: 1 }));
        r.push(log.splice(0).join());
        Object.freeze(P({ a: 1 }));
        r.push(log.splice(0).join());
        var g = P(function f(x) {}).bind(null);
        r.push(log.splice(0).join(), g.name, Object.getPrototypeOf(g) === Function.prototype);
        r.join("|")
    "#;
    assert_eq!(
        run(src),
        "true|false|gopd:a,gopd:a|def:a|gopd:a,def:a|gpo,gopd:length|bound f|true"
    );
}

/// `Array.prototype` methods over a generic array-like read what their
/// algorithm reads, in order: the callback methods read `length` once (it was
/// read twice), `at`/`with` read only the indices they name (not every index),
/// and `copyWithin` interleaves each `Has`/`Get` with its `Set` (it read the
/// whole source range first, which also matters when the ranges overlap).
#[test]
fn array_generic_methods_read_only_what_they_name() {
    let src = r#"
        var log = [];
        function P(t) {
            return new Proxy(t, {
                get(t, k, r) { log.push("get:" + String(k)); return Reflect.get(t, k, r); },
                set(t, k, v, r) { log.push("set:" + String(k)); return Reflect.set(t, k, v, r); },
                has(t, k) { log.push("has:" + String(k)); return Reflect.has(t, k); },
            });
        }
        var r = [];
        Array.prototype.forEach.call(P({ length: 1, 0: 1 }), function () {});
        r.push(log.splice(0).join());
        var n = 0;
        Array.prototype.some.call({ get length() { n++; return 0; } }, function () {});
        r.push(n);
        r.push(Array.prototype.at.call(P([1, 2, 3]), -1), log.splice(0).join());
        r.push(Array.prototype.with.call(P([1, 2]), 0, 5).join(), log.splice(0).join());
        Array.prototype.copyWithin.call(P({ length: 3, 0: "a", 1: "b", 2: "c" }), 0, 1);
        r.push(log.splice(0).filter(function (e) { return e.indexOf("set:") == 0 || e.indexOf("has:") == 0; }).join());
        var o = { length: 4, 0: "a", 1: "b", 2: "c", 3: "d" };
        Array.prototype.copyWithin.call(o, 1, 0, 3);
        r.push(Array.prototype.join.call(o, ""));
        r.join("|")
    "#;
    assert_eq!(
        run(src),
        "get:length,has:0,get:0|1|3|get:length,get:2|5,2|get:length,get:1\
         |has:1,set:0,has:2,set:1|aabc"
    );
}

/// `for-in` walks the whole prototype chain: a Proxy on it is asked through
/// its traps (`getPrototypeOf` included, so what it inherits is enumerated),
/// and an array, String or typed-array *prototype* contributes its index keys
/// (they were dropped: `for (k in Object.create([7, 8]))` enumerated nothing).
#[test]
fn for_in_walks_proxies_and_indexed_prototypes() {
    let src = r#"
        function keys(o) { var r = []; for (var k in o) r.push(k); return r.join(); }
        var proto = { inh: 1 };
        var log = [];
        var p = new Proxy({ a: 1 }, { getPrototypeOf(t) { log.push("gpo"); return proto; } });
        var a = [1, 2]; a.q = 1; var o2 = Object.create(a); o2[5] = 1;
        var s = Object.create(new String("ab")); s.x = 1;
        [keys(p), log.length > 0, keys(Object.create([7, 8])), keys(o2), keys(s),
         keys(Object.create(new Uint8Array(2))), keys(new Proxy([1, 2], {})),
         keys(Object.create(new Proxy([5], {})))].join("|")
    "#;
    assert_eq!(run(src), "a,inh|true|0,1|5,0,1,q|x,0,1|0,1|0,1|0");
}

/// An Error instance has no own `name`: engine-thrown and constructed errors
/// alike inherit it from their prototype, so reassigning
/// `TypeError.prototype.name` renames them and `getOwnPropertyNames` lists
/// only `message` (every error carried an own `name` data property).
#[test]
fn error_instances_inherit_their_name() {
    let src = r#"
        var e = new TypeError("m"), r = [];
        r.push(Object.hasOwn(e, "name"), Object.getOwnPropertyNames(e).join(), e.name, String(e));
        try { null.x; } catch (t) { r.push(Object.hasOwn(t, "name"), t.name); }
        class MyErr extends RangeError {}
        r.push(Object.hasOwn(new MyErr(), "name"), new MyErr("q").toString());
        TypeError.prototype.name = "TE";
        try { undefined.y; } catch (t) { r.push(t.name); }
        r.push(new TypeError("z").toString());
        TypeError.prototype.name = "TypeError";
        r.join("|")
    "#;
    assert_eq!(
        run(src),
        "false|message|TypeError|TypeError: m|false|TypeError|false|RangeError: q|TE|TE: z"
    );
}

/// ToInt32 / ToUint32 / ToUint16 are modular for every finite Number: a
/// magnitude past 2^63 (`1e100`, `2**64 + 20480`) wraps rather than saturating
/// (an `as i64` cast) in `Math.clz32`, `Math.imul`, DataView setters,
/// `String.fromCharCode`, the `split` limit and integer typed-array stores.
#[test]
fn integer_conversions_wrap_past_two_to_the_63() {
    let src = r#"
        // X is exactly 2^64 + 20480: its low 32 bits are 20480.
        var X = 2 ** 64 + 4096 * 5, dv = new DataView(new ArrayBuffer(4)), r = [];
        dv.setInt32(0, 1e100); r.push(dv.getInt32(0));
        dv.setUint16(0, X); r.push(dv.getUint16(0));
        dv.setInt8(0, -1e100); r.push(dv.getInt8(0));
        r.push(Math.clz32(1e100), Math.clz32(X), Math.imul(X, 5),
               String.fromCharCode(1e100, X).length, String.fromCharCode(X).charCodeAt(0),
               "a,b,c".split(",", X).length, new Int32Array([1e100])[0], new Uint16Array([X])[0]);
        r.join()
    "#;
    assert_eq!(run(src), "0,20480,0,32,17,102400,2,20480,3,0,20480");
}

/// `splice()` / `toSpliced()` with no arguments at all remove nothing (a
/// missing `start` makes the delete count 0), while a lone `start`, even an
/// explicit `undefined`, removes through the end. Both removed everything.
#[test]
fn splice_without_start_removes_nothing() {
    let src = r#"
        var a = [1, 2, 3], h = [1, , 3];
        [a.splice().length, a.length, [1, 2].toSpliced().join(), h.toSpliced().length,
         a.splice(undefined).length, a.length, [1, 2, 3].toSpliced(1).join()].join("|")
    "#;
    assert_eq!(run(src), "0|3|1,2|3|3|0|1");
}

/// `Math.hypot` does not overflow (or underflow) in its intermediate squares:
/// `hypot(1e200, 1e200)` is `1e200 * Math.SQRT2`, not Infinity, and
/// `hypot(1e-200, 1e-200)` is not 0.
#[test]
fn hypot_scales_huge_and_tiny_arguments() {
    let src = r#"
        function h(f) { return f(1e200, 1e200) + "," + f(3e-200, 4e-200) + "," + f(3, 4) + "," + f(-0) + "," + f(NaN, -Infinity); }
        h(Math.hypot) + "|" + h(function (a, b) { return Math.hypot.apply(null, arguments); })
    "#;
    assert_eq!(
        run(src),
        "1.414213562373095e+200,5e-200,5,0,Infinity|1.414213562373095e+200,5e-200,5,0,Infinity"
    );
}

/// `Object.entries` / `Object.values` on a Proxy ask its
/// `getOwnPropertyDescriptor` trap for each key even when it has no `ownKeys`
/// trap (the trapless path skipped `[[GetOwnProperty]]` altogether).
#[test]
fn entries_values_ask_gopd_without_own_keys_trap() {
    let src = r#"
        var log = [];
        var p = new Proxy({ a: 1, b: 2 }, {
            getOwnPropertyDescriptor(t, k) { log.push(k); return Reflect.getOwnPropertyDescriptor(t, k); },
        });
        var r = [JSON.stringify(Object.entries(p)), log.splice(0).join(), Object.values(p).join(), log.join()];
        r.join("|")
    "#;
    assert_eq!(run(src), r#"[["a",1],["b",2]]|a,b|1,2|a,b"#);
}

/// The `defineProperty` trap invariant judges the target's property by its
/// `[[GetOwnProperty]]` descriptor: an array index made non-configurable by
/// the trap's own `Reflect.defineProperty` is non-configurable (it read as
/// configurable, so `Object.freeze` of a proxy over an array threw).
#[test]
fn define_property_invariant_sees_array_index_attributes() {
    let src = r#"
        var t = [1, 2, 3];
        var p = new Proxy(t, { defineProperty(t, k, d) { return Reflect.defineProperty(t, k, d); } });
        Object.freeze(p);
        [Object.isFrozen(t), Object.isFrozen(p)].join()
    "#;
    assert_eq!(run(src), "true,true");
}

/// An array-index store past the dense storage cap with a *string* key
/// (`a["4294967294"] = v`, as `a[4294967294] = v` already did) is stored
/// sparsely and grows `length`; it raised a RangeError.
#[test]
fn string_keyed_array_index_past_cap_is_stored() {
    let src = r#"
        var a = [], k = "4294967294";
        a["4294967294"] = 1;
        var b = [1]; b[k] = 2;
        var c = [1]; c["100000000"] = 3;
        [a.length, b.length, b[k], c.length, Object.keys(c).join()].join()
    "#;
    assert_eq!(run(src), "4294967295,4294967295,2,100000001,0,100000000");
}

/// `JSON.stringify` of a typed array serializes its own enumerable named
/// properties after its indices, like any object (they were dropped).
#[test]
fn json_stringify_typed_array_named_properties() {
    let src = r#"
        var u = new Uint8Array([1, 2]); u.foo = 3; u["01"] = 4;
        JSON.stringify(u) + "|" + JSON.stringify({ u: u }, null, 0)
    "#;
    assert_eq!(
        run(src),
        r#"{"0":1,"1":2,"foo":3,"01":4}|{"u":{"0":1,"1":2,"foo":3,"01":4}}"#
    );
}

/// `Object.isFrozen` / `Object.isSealed` of an array run TestIntegrityLevel
/// over its descriptors: an array frozen index by index, or an empty
/// non-extensible one (sealed: its `length` is non-configurable), was
/// reported as neither.
#[test]
fn array_integrity_level_tests_its_descriptors() {
    let src = r#"
        var a = [1, 2]; Object.preventExtensions(a);
        Object.defineProperty(a, "0", { configurable: false, writable: false });
        Object.defineProperty(a, "1", { configurable: false, writable: false });
        var r = [Object.isFrozen(a), Object.isSealed(a)];
        Object.defineProperty(a, "length", { writable: false });
        r.push(Object.isFrozen(a), Object.isSealed(a));
        var c = []; Object.preventExtensions(c);
        r.push(Object.isFrozen(c), Object.isSealed(c), Object.isFrozen([]), Object.isFrozen(Object.freeze([1])));
        r.join()
    "#;
    assert_eq!(run(src), "false,true,true,true,false,true,false,true");
}

/// Each tagged-template site has its own template object, also inside arrow
/// functions: the compiler lowers an arrow body through a temporary clone,
/// so keying the site by the AST node's address let a later arrow reuse an
/// earlier one's address and receive *its* strings. A site called again
/// still gets the same object.
#[test]
fn tagged_template_sites_in_arrows_stay_distinct() {
    let src = r#"
        var tag = function (s) { return s; };
        var fs = [];
        for (var i = 0; i < 30; i++) {
            fs.push(eval("() => [tag`a" + i + "${0}b`, () => tag`c" + i + "`]"));
        }
        var ok = true;
        for (var i = 0; i < 30; i++) {
            var r = fs[i]();
            ok = ok && r[0].raw.join("|") === "a" + i + "|b" && r[1]()[0] === "c" + i;
        }
        var f = () => tag`same`;
        var g = (x) => { return String.raw`x${x}y`; };
        [ok, f() === f(), g(1), (() => tag`q`)()[0]].join()
    "#;
    assert_eq!(run(src), "true,true,x1y,q");
}

/// `String.prototype.{split,replace,replaceAll,match,search,matchAll}` do
/// `GetMethod(searchValue, @@method)` once (it ran twice), and `match` /
/// `search` / `matchAll` build their RegExp with RegExpCreate, which
/// ToStrings the argument instead of asking IsRegExp (an extra `@@match`
/// read); `concat` ToStrings each argument, so a Symbol is a TypeError.
#[test]
fn string_methods_look_up_symbol_methods_once() {
    let src = r#"
        var log = [];
        var o = { toString() { log.push("ts"); return ","; } };
        ["split", "replace", "match", "search"].forEach(function (s) {
            Object.defineProperty(o, Symbol[s], { get() { log.push(s); return undefined; }, configurable: true });
        });
        var r = [];
        "a,b".split(o); r.push(log.splice(0).join());
        "a,b".replace(o, "x"); r.push(log.splice(0).join());
        "a,b".match(o); r.push(log.splice(0).join());
        "a,b".search(o); r.push(log.splice(0).join());
        String.prototype.split.call("a,b", o); r.push(log.splice(0).join());
        var c; try { "a".concat(Symbol()); c = "no"; } catch (e) { c = e.name; }
        r.push(c, "a".concat(1, { toString() { return "z"; } }));
        r.join("|")
    "#;
    assert_eq!(
        run(src),
        "split,ts|replace,ts|match,ts|search,ts|split,ts|TypeError|a1z"
    );
}

/// `Object.freeze(arguments)` on a mapped arguments object goes through its
/// `[[DefineOwnProperty]]`: each index keeps the parameter's current value and
/// stops aliasing it (a later parameter write showed through the frozen
/// object).
#[test]
fn freezing_mapped_arguments_severs_the_alias() {
    let src = r#"
        function f(a, b) {
            a = 5;
            Object.freeze(arguments);
            a = 7; b = 8;
            arguments[0] = 9;
            return [arguments[0], arguments[1], a, b, Object.isFrozen(arguments)].join();
        }
        f(1, 2)
    "#;
    assert_eq!(run(src), "5,2,7,8,true");
}

/// Runs `src` (draining its promise jobs) and returns its `console` output.
fn out(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    interp.run(&program).expect("exec");
    String::from(interp.output())
}

/// An arrow in a class field initializer sees the instance as `this` (and
/// `super`/`new.target` through it), also when it runs during the
/// initialization: the fields closure did not box its `this` for nested
/// arrows, so an immediately-called arrow read `undefined`, and `new.target`
/// in one was refused by the compiler.
#[test]
fn arrow_in_field_initializer_sees_this() {
    let src = r#"
        class A { constructor() { this.t = 1; } m() { return 'A.m:' + this.tag; } }
        class B { tag = 'b'; af = (() => this.tag)(); }
        class D extends A { tag = 'd'; af = (() => super.m())(); bf = (() => (() => this.t)())(); #p = (() => this.t + 1)(); p() { return this.#p; } }
        class N extends A { nt = (() => new.target)(); }
        var d = new D();
        [new B().af, d.af, d.bf, d.p(), new N().nt].join()
    "#;
    assert_eq!(run(src), "b,A.m:d,1,2,");
}

/// AsyncIteratorClose of a `for await` over a sync iterable awaits the
/// promise from %AsyncFromSyncIteratorPrototype%.return, even when the sync
/// iterator has no `return` (one tick) and after a sync `return()` (two: the
/// result's value is resolved first); the loop exit was synchronous.
#[test]
fn for_await_close_over_sync_iterator_awaits_the_wrapper() {
    let src = r#"
        function ticks(tag) { var i = 0; (function f() { if (i < 4) { console.log(tag + i++); Promise.resolve().then(f); } })(); }
        (async () => { for await (var x of [1, 2]) { console.log('A body'); break; } console.log('A after'); })(); ticks('A');
        (async () => { for (var i = 0; i < 12; i++) await null;
          ticks('C'); var it = { [Symbol.iterator]() { return { next() { return { value: 1, done: false }; }, return() { console.log('C ret'); return {}; } }; } };
          for await (var x of it) { console.log('C body'); break; } console.log('C after'); })();
        (async () => { for (var i = 0; i < 24; i++) await null;
          ticks('D'); try { for await (var x of [1]) { throw 'e'; } } catch (e) { console.log('D caught'); } })();
    "#;
    assert_eq!(
        out(src),
        "A0\nA1\nA body\nA2\nA after\nA3\n\
         C0\nC1\nC2\nC body\nC ret\nC3\nC after\n\
         D0\nD1\nD2\nD3\nD caught\n"
    );
}

/// `Iterator.prototype.drop` skips with IteratorStep, which reads `done`
/// but never `value` (a throwing `value` getter on a skipped result must
/// not run).
#[test]
fn iterator_drop_skips_without_reading_values() {
    let src = r#"
        var log = [];
        var src = { i: 0, next() { var k = this.i++; return { get value() { log.push('v' + k); if (k === 0) throw 'boom'; return k; }, get done() { log.push('d' + k); return k > 2; } }; } };
        var r = Iterator.from(src).drop(1).toArray();
        r.join() + '|' + log.join()
    "#;
    assert_eq!(run(src), "1,2|d0,d1,v1,d2,v2,d3");
}

/// `then` on a promise whose species is a subclass settles the subclass's
/// capability from the reaction job itself: a bridge through an internal
/// promise delayed every link of a subclass chain by one tick.
#[test]
fn subclass_promise_then_takes_no_extra_tick() {
    let src = r#"
        function ticks(n) { var i = 0; (function f() { if (i < n) { console.log('t' + i++); Promise.resolve().then(f); } })(); }
        class MP extends Promise { then(a, b) { return super.then(a, b); } }
        ticks(4); MP.resolve(2).then(v => console.log('c', v)).then(() => console.log('fin'));
    "#;
    assert_eq!(out(src), "t0\nt1\nc 2\nt2\nfin\nt3\n");
}

/// A promise's resolving functions share one `[[AlreadyResolved]]` flag:
/// after `resolve(promise)` (whose adoption is still pending) a later
/// `reject` or `resolve` of the same pair is a no-op; the fresh pair a
/// PromiseResolveThenableJob hands the thenable has a flag of its own.
#[test]
fn resolving_functions_honor_already_resolved() {
    let src = r#"
        var log = [];
        new Promise((r, j) => { r(Promise.resolve(1)); j(2); }).then(v => log.push('a' + v), e => log.push('ae' + e));
        new Promise((r, j) => { r(Promise.resolve(3)); r(4); }).then(v => log.push('b' + v));
        new Promise(r => r({ then(a, b) { a(5); b(6); a(7); throw 8; } })).then(v => log.push('c' + v), e => log.push('ce' + e));
        new Promise(r => r({ then(a, b) { throw 9; } })).then(v => log.push('d' + v), e => log.push('de' + e));
        var keep; new Promise(r => r({ then(a) { keep = a; } })).then(v => log.push('e' + v));
        Promise.resolve().then(() => { keep(10); keep(11); });
        var save; new Promise(r => { save = r; r({ then() {} }); }).then(v => log.push('never' + v));
        Promise.resolve().then(() => save(12));
        var done = Promise.resolve().then(() => 0).then(() => 0).then(() => 0).then(() => 0);
    "#;
    let program = Parser::parse_program(&alloc::format!(
        "{src}; done.then(() => {{ globalThis.R = log.join(); }});"
    ))
    .expect("parse");
    let mut interp = Interp::new();
    interp.run(&program).expect("exec");
    let r = Parser::parse_program("R").expect("parse");
    let v = interp.run(&r).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "c5,de9,e10,a1,b3");
}

/// `queueMicrotask` throws a TypeError for a non-callable argument and calls
/// its callback with no arguments.
#[test]
fn queue_microtask_checks_and_calls_bare() {
    let src = r#"
        try { queueMicrotask(5); } catch (e) { console.log(e.constructor.name); }
        queueMicrotask(function () { console.log('args', arguments.length); });
    "#;
    assert_eq!(out(src), "TypeError\nargs 0\n");
}

/// `Array.fromAsync` over a sync iterable steps it through the
/// async-from-sync wrapper: each `next` resolves its value, then is awaited,
/// so the result settles on the spec's tick (it finished synchronously for
/// an empty array, a tick early per element otherwise).
#[test]
fn array_from_async_over_sync_iterables_keeps_spec_ticks() {
    let src = r#"
        function ticks(tag, n) { var i = 0; (function f() { if (i < n) { console.log(tag + i++); Promise.resolve().then(f); } })(); }
        ticks('a', 5); Array.fromAsync([]).then(v => console.log('A', v.length));
        Promise.resolve().then(() => 0).then(() => 0).then(() => 0).then(() => 0).then(() => 0).then(() => {
            ticks('b', 7); Array.fromAsync([1, Promise.resolve(2)]).then(v => console.log('B', v.join()));
        });
    "#;
    assert_eq!(
        out(src),
        "a0\na1\na2\na3\nA 0\na4\nb0\nb1\nb2\nb3\nb4\nb5\nb6\nB 1,2\n"
    );
}

/// When `Array.fromAsync`'s `mapfn` throws, AsyncIteratorClose awaits the
/// iterator's `return()` (a sync iterator's through the async-from-sync
/// wrapper) before the rejection; it used to close synchronously and reject at
/// once.
#[test]
fn array_from_async_close_awaits_return_before_rejecting() {
    let src = r#"
        function ticks(n) { var i = 0; (function f() { if (i < n) { console.log('t' + i++); Promise.resolve().then(f); } })(); }
        var it = { i: 0, [Symbol.iterator]() { return this; }, next() { return { value: this.i++, done: false }; }, return() { console.log('return'); return {}; } };
        ticks(8); Array.fromAsync(it, x => { if (x === 1) throw 'mf'; return x; }).catch(e => console.log('c', e));
    "#;
    assert_eq!(out(src), "t0\nt1\nt2\nt3\nt4\nt5\nreturn\nt6\nt7\nc mf\n");
}

/// A dynamic `import()` resolves its promise with the namespace (so a module
/// exporting `then` is adopted as a thenable), it does not fulfil with it.
#[cfg(all(feature = "module", feature = "std"))]
#[test]
fn dynamic_import_resolves_with_a_thenable_namespace() {
    let dir = std::env::temp_dir().join(alloc::format!(
        "kataan-difftest-dynimport-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("mkdir");
    std::fs::write(
        dir.join("thenable.mjs"),
        "export function then(r) { console.log('then called'); r('by-then'); } export const x = 1;\n",
    )
    .expect("write");
    let main = dir.join("main.mjs");
    std::fs::write(
        &main,
        "const t = await import('./thenable.mjs'); console.log(typeof t, t);\n",
    )
    .expect("write");
    let (output, _) = crate::nbvm::execute_module_typed(
        main.to_str().expect("utf-8 path"),
        &crate::nbexec::module::FileModuleHost,
        crate::limits::Limits::default(),
    )
    .expect("module runs");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output, "then called\nstring by-then\n");
}

/// `Promise.withResolvers.length` is 0 (it was 1).
#[test]
fn promise_with_resolvers_length_is_zero() {
    assert_eq!(run("Promise.withResolvers.length"), "0");
}

/// `SharedArrayBuffer[Symbol.species]` exists, a getter returning `this` like
/// `ArrayBuffer`'s (it was missing).
#[test]
fn shared_array_buffer_has_species_getter() {
    let src = r#"
        var d = Object.getOwnPropertyDescriptor(SharedArrayBuffer, Symbol.species);
        class S extends SharedArrayBuffer {}
        [typeof d.get, d.set, d.configurable, d.get.name,
         SharedArrayBuffer[Symbol.species] === SharedArrayBuffer, S[Symbol.species] === S].join()
    "#;
    assert_eq!(run(src), "function,,true,get [Symbol.species],true,true");
}

/// `Object.defineProperties` / `Object.create` convert every descriptor
/// (ToPropertyDescriptor) before defining any, so a bad descriptor later in
/// the list leaves the target untouched (the earlier ones were defined); the
/// descriptors' getters still run in key order, before any define.
#[test]
fn define_properties_validates_all_descriptors_first() {
    let src = r#"
        var log = [], o = {}, r = [];
        try { Object.defineProperties(o, { a: { value: 1 }, b: { get: 5 } }); } catch (e) { r.push(e.name); }
        r.push(Object.getOwnPropertyNames(o).length);
        try { Object.defineProperties(o, { a: { value: 1 }, b: { value: 1, get: undefined } }); } catch (e) { r.push(e.name); }
        r.push(Object.getOwnPropertyNames(o).length);
        try { Object.create(null, { a: { value: 1 }, b: 5 }); } catch (e) { r.push(e.name); }
        var d = { get a() { log.push("A"); return { get value() { log.push("va"); return 1; } }; },
                  get b() { log.push("B"); return { value: 2 }; } };
        Object.defineProperty(d, "a", { enumerable: true });
        Object.defineProperty(d, "b", { enumerable: true });
        var t = Object.defineProperties({}, d);
        r.push(log.join(""), t.a + t.b);
        r.join("|")
    "#;
    assert_eq!(run(src), "TypeError|0|TypeError|0|TypeError|AvaB|3");
}

/// `Date.parse` honors the UTC offset of the `toString` format
/// (`GMT+0100` was ignored), and reads the common human forms: a month name
/// before or after the day (`Mar 1 2022`, `1 March 2022`), and slash dates
/// (`2022/03/01`, month-first `03/01/2022`), with an optional time (12-hour
/// with `AM`/`PM`) and zone. A two-digit year is in 1950..2049, and a leading
/// number past 31 is the year (`45/1/2`, `may 45 1`), as other engines read them.
#[test]
fn date_parse_reads_offsets_and_human_forms() {
    let src = r#"
        ["Thu Jan 01 1970 00:00:00 GMT+0100 (CET)", "Thu Jan 01 1970 01:30:00 GMT-02:30",
         "Mar 1 2022", "1 March 2022", "Tuesday, March 1, 2022 10:00", "2022/03/01", "03/01/2022 00:00:01",
         "Thu, 01 Jan 1970 00:00:00 GMT", "Jan 32 2020", "Foo 1 2020", "13/01/2020",
         "Jan 1, 2020 GMT", "1 Jan 2020 UTC+0100", "Jan 1 2020 GMTx",
         "5/1/49", "may 45 1", "45/1/2", "Jan 1 2000 10:00 PM", "Jan 1 2000 12:00 AM", "1 may 50",
         "13/1/2", "Jan 1 2000 13:00 PM"]
            .map(function (s) { return Date.parse(s); }).join()
    "#;
    assert_eq!(
        run(src),
        "-3600000,14400000,1646092800000,1646092800000,1646128800000,1646092800000,1646092801000,0,NaN,NaN,NaN,\
         1577836800000,1577833200000,NaN,\
         2503440000000,2377209600000,2366928000000,946764000000,946684800000,-620784000000,NaN,NaN"
    );
}

/// Default compact notation rounds to two significant digits whatever the
/// numbering system (a locale formatting with native digits, `ar-EG`, kept the
/// crate's full mantissa), carries a mantissa that rounds up to a new power of
/// ten to the next magnitude (`999999` is `1M`, not `1000K`), and reports its
/// `morePrecision` rounding in resolvedOptions.
#[cfg(feature = "intl")]
#[test]
fn intl_compact_rounding_is_numbering_system_independent() {
    let src = r#"
        var en = new Intl.NumberFormat("en", { notation: "compact" });
        var ar = new Intl.NumberFormat("ar-EG", { notation: "compact" });
        var arab = new Intl.NumberFormat("en-u-nu-arab", { notation: "compact" });
        var o = en.resolvedOptions();
        [en.format(1234567), en.format(999999), en.format(-999999), en.format(999.96),
         /^\u0661\u066b\u0662[^\u0660-\u0669]+$/.test(ar.format(1234567)),
         arab.format(12345) === "\u0661\u0662K", arab.format(999999) === "\u0661M",
         en.formatToParts(999999).map(function (p) { return p.type + ":" + p.value; }).join(" "),
         o.roundingPriority, o.minimumSignificantDigits, o.maximumSignificantDigits,
         o.minimumFractionDigits, o.maximumFractionDigits,
         new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).resolvedOptions().roundingPriority
        ].join()
    "#;
    assert_eq!(
        run(src),
        "1.2M,1M,-1M,1K,true,true,true,integer:1 compact:M,morePrecision,1,2,0,0,auto"
    );
}

/// A String wrapper's read-only index properties are only its *canonical*
/// indices: `"01"`, `"+1"` and `"00"` are ordinary keys, so assigning one
/// creates a property (it was dropped, and threw in strict code).
#[test]
fn string_wrapper_non_canonical_index_keys_are_writable() {
    let src = r#"
        var r = [];
        ["01", "+1", "00", "1"].forEach(function (k) {
            var s = new String("abc");
            s[k] = 7;
            var strict = (function () { "use strict"; var t = new String("abc");
                try { t[k] = 5; return t[k]; } catch (e) { return e.name; } })();
            r.push(k + ":" + s[k] + ":" + Reflect.set(new String("abc"), k, 1) + ":" + strict
                   + ":" + Object.keys(s).join("/"));
        });
        r.join(" ")
    "#;
    assert_eq!(
        run(src),
        "01:7:true:5:0/1/2/01 +1:7:true:5:0/1/2/+1 00:7:true:5:0/1/2/00 1:b:false:TypeError:0/1/2"
    );
}

/// An array whose `length` is non-writable cannot gain an index at or past
/// it (ArrayDefineOwnProperty): `a[5] = 1` and `Reflect.set` fail (strict code
/// throws) instead of growing the array, in a hot loop too. In-range writes
/// still succeed.
#[test]
fn array_with_readonly_length_does_not_grow() {
    let src = r#"
        var r = [];
        var a = [1]; Object.defineProperty(a, "length", { writable: false });
        a[5] = 1; r.push(a.length, 5 in a);
        a["1"] = 2; r.push(a.length, 1 in a);
        a[0] = 9; r.push(a[0], Reflect.set(a, 3, 1), Reflect.defineProperty(a, 4, { value: 1 }), a.length);
        (function () { "use strict"; try { a[2] = 1; r.push("nothrow"); } catch (e) { r.push(e.name); } })();
        var b = []; Object.defineProperty(b, "length", { value: 2, writable: false });
        b[1] = "x"; b[2] = "y"; r.push(b.length, b[1], 2 in b);
        function f(arr, n) { for (var i = 0; i < n; i++) arr[i] = i; return arr.length; }
        for (var k = 0; k < 500; k++) f([], 4);
        r.push(f(b, 4), 3 in b);
        r.join()
    "#;
    assert_eq!(
        run(src),
        "1,false,1,false,9,false,false,1,TypeError,2,x,false,2,false"
    );
}

/// An anonymous class named after an object-literal key or a computed
/// instance-field key (NamedEvaluation) gets that name *before* its static
/// elements run: a `static name()` method or `static name` field replaces it
/// (it was overwritten by the key afterwards), and a static block sees it (it
/// saw `""`). Symbol and computed keys included.
#[test]
fn class_named_by_property_key_before_static_elements() {
    let src = r#"
        var s = Symbol("sym"), r = [];
        var o = { j: class { static name() {} }, ["k"]: class { static name() {} },
                  l: class { static get name() { return "g"; } }, m: class { static name = "x"; },
                  n: class { static { r.push(this.name); } }, [s]: class { static { r.push(this.name); } },
                  p: class {}, q: function () {} };
        r.push(typeof o.j.name, typeof o.k.name, o.l.name, o.m.name, o.p.name, o.q.name, o[s].name);
        class I { ["f" + 1] = class { static { r.push("blk:" + this.name); } }; [s] = class { static name() {} }; }
        var i = new I();
        r.push(i.f1.name, typeof i[s].name, Object.keys(o).join("/"));
        r.join()
    "#;
    assert_eq!(
        run(src),
        "n,[sym],function,function,g,x,p,q,[sym],blk:f1,f1,function,j/k/l/m/n/p/q"
    );
}

/// `Function.prototype.toString` on a callable Proxy renders the
/// NativeFunction form without reading `name` through the proxy's `get` trap
/// (the name comes from the innermost target's own data property).
#[test]
fn function_to_string_on_proxy_makes_no_trap_call() {
    let src = r#"
        var log = [], h = { get(t, k) { log.push(String(k)); return t[k]; } };
        var p = new Proxy(function f(a) {}, h), pp = new Proxy(p, h);
        var q = new Proxy(class C {}, h), m = new Proxy(Math.max, h);
        [Function.prototype.toString.call(p), Function.prototype.toString.call(pp),
         Function.prototype.toString.call(q),
         Function.prototype.toString.call(m), log.length].join("|")
    "#;
    assert_eq!(
        run(src),
        "function f() { [native code] }|function f() { [native code] }|function C() { [native code] }|function max() { [native code] }|0"
    );
}

/// `++`/`--` on an object whose ToPrimitive is a BigInt step a BigInt
/// (ToNumeric keeps it): the result was a Number.
#[test]
fn update_of_object_with_bigint_value_stays_bigint() {
    let src = r#"
        var o = { valueOf() { return 1n; } }, r = [];
        var x = o; x++; r.push(typeof x, x);
        x = o; --x; r.push(typeof x, x);
        x = o; var old = x++; r.push(typeof old, old);
        var w = Object(5n); w++; r.push(typeof w, w);
        var obj = { p: o }; obj.p++; var arr = [o]; arr[0]--;
        r.push(typeof obj.p, obj.p, typeof arr[0], arr[0]);
        function f() { var y = o; y++; return y; }
        r.push(typeof f());
        r.join()
    "#;
    assert_eq!(
        run(src),
        "bigint,2,bigint,0,bigint,1,bigint,6,bigint,2,bigint,0,bigint"
    );
}

/// Negative zero in scientific/engineering notation is `-0E0` (the internal
/// negative-zero stand-in leaked as `-4.941E-324`), in `formatToParts` too.
#[cfg(feature = "intl")]
#[test]
fn scientific_negative_zero_is_zero() {
    let src = r#"
        function f(o) { return new Intl.NumberFormat("en", o).format(-0); }
        [f({ notation: "scientific" }), f({ notation: "engineering" }),
         f({ notation: "scientific", minimumFractionDigits: 2 }), f({ notation: "scientific", signDisplay: "negative" }),
         new Intl.NumberFormat("en", { notation: "scientific" }).formatToParts(-0).map(function (p) { return p.type + ":" + p.value; }).join(" ")].join("|")
    "#;
    assert_eq!(
        run(src),
        "-0E0|-0E0|-0.00E0|0E0|minusSign:- integer:0 exponentSeparator:E exponentInteger:0"
    );
}

/// `Intl.PluralRules#select` takes its operands from the number *formatted*
/// under the instance's digit options (ResolvePlural): `1` with
/// `minimumFractionDigits: 1` is "1.0" (`other`), `1.4` with
/// `maximumSignificantDigits: 1` is "1" (`one`), the default three fraction
/// digits round `1.0001` to "1", and `roundingPriority` picks between the two
/// digit sets. The raw value was used.
#[cfg(feature = "intl")]
#[test]
fn plural_rules_select_uses_formatted_operands() {
    let src = r#"
        function s(loc, o, n) { return new Intl.PluralRules(loc, o).select(n); }
        [s("en", { minimumFractionDigits: 1 }, 1), s("en", { maximumSignificantDigits: 1 }, 1.4),
         s("en", { maximumFractionDigits: 0 }, 1.4), s("en", {}, 1.0001), s("en", {}, -1),
         s("ru", { minimumFractionDigits: 1 }, 2), s("ru", {}, 2),
         s("en", { type: "ordinal", minimumFractionDigits: 1 }, 2),
         s("en", { maximumSignificantDigits: 2, roundingPriority: "lessPrecision", maximumFractionDigits: 0 }, 1.4),
         s("en", { maximumSignificantDigits: 1, roundingPriority: "morePrecision", maximumFractionDigits: 1 }, 1.04),
         s("fr", { notation: "compact" }, 1500000)].join()
    "#;
    assert_eq!(run(src), "other,one,one,one,one,other,few,two,one,one,many");
}

/// A currency's default fraction digits are its ISO 4217 minor units
/// (CurrencyDigits): `BHD` has 3, so 1.005 is "BHD 1.005" (it was rounded at
/// two digits and padded, "BHD 1.010"), and resolvedOptions reports 3.
#[cfg(feature = "intl")]
#[test]
fn currency_default_digits_follow_iso_4217() {
    let src = r#"
        function f(c, v) { return new Intl.NumberFormat("en", { style: "currency", currency: c }).format(v); }
        var o = new Intl.NumberFormat("en", { style: "currency", currency: "KWD" }).resolvedOptions();
        [f("BHD", 1.005), f("BHD", 12345.6789), f("JPY", 1.5), f("CLF", 1.23456), f("USD", 1.005),
         o.minimumFractionDigits, o.maximumFractionDigits].join("|")
    "#;
    assert_eq!(
        run(src),
        "BHD\u{a0}1.005|BHD\u{a0}12,345.679|¥2|CLF\u{a0}1.2346|$1.01|3|3"
    );
}

/// A `roundingIncrement` other than 1 needs equal *resolved* fraction
/// digits: an omitted bound takes the style default, so
/// `{roundingIncrement: 5}` alone (0..3) or with only `maximumFractionDigits: 1`
/// (0..1) is a RangeError (only two explicit unequal bounds were checked).
/// Compact notation's default rounding is not fraction digits (a TypeError).
#[cfg(feature = "intl")]
#[test]
fn rounding_increment_checks_resolved_fraction_digits() {
    let src = r#"
        function t(f) { try { return f(); } catch (e) { return e.name; } }
        [t(() => new Intl.NumberFormat("en", { roundingIncrement: 5 }).format(1)),
         t(() => new Intl.NumberFormat("en", { roundingIncrement: 5, maximumFractionDigits: 1 }).format(1)),
         t(() => new Intl.PluralRules("en", { roundingIncrement: 5, maximumFractionDigits: 1 }).select(1)),
         t(() => new Intl.NumberFormat("en", { roundingIncrement: 5, maximumFractionDigits: 1, minimumFractionDigits: 1 }).format(1.02)),
         t(() => new Intl.NumberFormat("en", { roundingIncrement: 5, maximumFractionDigits: 0 }).format(12)),
         t(() => new Intl.NumberFormat("en", { notation: "compact", roundingIncrement: 5 }).format(1))].join()
    "#;
    assert_eq!(
        run(src),
        "RangeError,RangeError,RangeError,1.0,10,TypeError"
    );
}

/// `style: "percent"` rounds the *scaled* value in decimal, so a
/// `roundingIncrement` applies (`0.12` → "10%" by 5s; it was "12%"), an
/// omitted maximum is at least the minimum (`minimumFractionDigits: 1` keeps
/// "12.5%", not "13.0%"), and `stripIfInteger` sees a whole percentage.
#[cfg(feature = "intl")]
#[test]
fn percent_rounds_the_scaled_value() {
    let src = r#"
        function f(o, v) { return new Intl.NumberFormat("en", Object.assign({ style: "percent" }, o)).format(v); }
        [f({ roundingIncrement: 5 }, 0.12), f({ minimumFractionDigits: 1 }, 0.125), f({ minimumFractionDigits: 1 }, 0.0105),
         f({}, 0.125), f({ maximumFractionDigits: 2 }, 0.0105), f({ roundingIncrement: 25, maximumFractionDigits: 2, minimumFractionDigits: 2 }, 0.1235),
         f({ trailingZeroDisplay: "stripIfInteger", minimumFractionDigits: 2 }, 0.07),
         f({ trailingZeroDisplay: "stripIfInteger", minimumFractionDigits: 2 }, 0.0105),
         new Intl.NumberFormat("en", { style: "percent", roundingIncrement: 5 }).formatToParts(0.12).map(function (p) { return p.value; }).join("")].join("|")
    "#;
    assert_eq!(run(src), "10%|12.5%|1.1%|13%|1.05%|12.25%|7%|1.05%|10%");
}
/// A plain `Interp` (an embedder, `kataan run`) has none of the Test262 host
/// hooks (`$262_IsHTMLDDA`, `$262_createRealm`, `$262_agent_*`, …): they are
/// installed only when `Limits::test262_host_hooks` asks, as the conformance
/// runner does, and then also in the realms `$262_createRealm` makes.
#[test]
fn test262_host_hooks_are_opt_in() {
    let probe = r#"
        var own = Object.getOwnPropertyNames(globalThis).filter(function (n) { return n.indexOf("262") >= 0; });
        [own.length, typeof $262_IsHTMLDDA, typeof $262_createRealm, typeof $262_evalScript,
         typeof $262_detachArrayBuffer, typeof $262_agent_start, typeof $262_AbstractModuleSource].join()
    "#;
    let program = Parser::parse_program(probe).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(value),
        "0,undefined,undefined,undefined,undefined,undefined,undefined"
    );
    let limits = crate::limits::Limits::default().with_test262_host_hooks();
    let program = Parser::parse_program(
        "var r = $262_createRealm(); [typeof $262_IsHTMLDDA, typeof r.global.$262_createRealm].join()",
    )
    .expect("parse");
    let mut interp = Interp::new_with_limits(limits);
    let value = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(value), "function,function");
}

/// An anonymous class is named inside its own ClassDefinitionEvaluation
/// (`SetFunctionName(F, className)` precedes the elements), so its static
/// blocks and static field initializers already see `this.name`, and a static
/// `name` method overrides the name. A class used as a static field
/// initializer (which compiles as a function of its own), under a computed
/// class-field key, or as an object-literal property value was named only
/// after its static elements had run, and the late naming also clobbered a
/// static `name` method. (V8 still reports a string `name` for
/// `{ [k]: class { static name() {} } }`; the spec gives the method.)
#[test]
fn anonymous_class_is_named_before_its_static_elements() {
    let src = r#"
        var r = [];
        var k = "kk", s = Symbol("sy");
        class A {
            static C = class { static name() { return 1; } };
            static D = class { static { r.push("D:" + this.name); } };
            static E = class { static x = this.name; };
            F = class { static { r.push("F:" + this.name); } };
            static ["H"] = class { static { r.push("H:" + this.name); } };
            static #I = class { static { r.push("I:" + this.name); } };
            static [k] = class { static { r.push("A1:" + this.name); } };
            [s] = class { static { r.push("A2:" + this.name); } };
            static getI() { return A.#I.name; }
        }
        new A();
        r.push(typeof A.C.name, A.E.x, A.getI(), A.D.name, A.kk.name);
        var o = { J: class { static { r.push("J:" + this.name); } },
                  [k]: class { static { r.push("o1:" + this.name); } },
                  [s]: class { static x = this.name; },
                  n: class { static name() {} },
                  [k + 1]: class { static name() {} },
                  g: function () {}, [k + 2]: () => 0 };
        r.push(o[s].x, typeof o.n.name, typeof o.kk1.name, o.g.name, o.kk2.name, o.kk.name);
        r.join("|")
    "#;
    assert_eq!(
        run(src),
        "D:D|H:H|I:#I|A1:kk|F:F|A2:[sy]|function|E|#I|D|kk|J:J|o1:kk|[sy]|function|function|g|kk2|kk"
    );
}

/// A labelled `for (let …)` loop whose body has a closure over the loop
/// variable: the bound-names scan stopped at the label (it only looked
/// through labelled *functions*), so the captured head binding was not made
/// a cell and the compiler refused the whole program.
#[test]
fn labelled_let_loop_with_a_capturing_closure_compiles() {
    let src = r#"
        var r = [];
        l: for (let i = 0; i < 4; i++) { let f = () => i; if (i % 2) continue l; r.push(f()); }
        m: for (let j = 0; j < 4; j++) { var g = () => j; if (j == 2) break m; r.push(g()); }
        a: b: for (const k of [7, 8]) { r.push((() => k)()); }
        n: { let z = 5; r.push((() => z)()); break n; }
        r.join()
    "#;
    assert_eq!(run(src), "0,2,0,1,7,8,5");
}

/// CalendarDateUntil counts a month only while the *unregulated* date does
/// not surpass the end (ISODateSurpasses compares the day before constraining
/// it): from Jan 31, one month is Feb "31", past Feb 29, so Jan 31 → Feb 29 is
/// 29 days, not one month. The difference constrained the day first.
#[test]
fn date_until_does_not_count_a_month_the_day_overshoots() {
    let src = r#"
        var d = Temporal.PlainDate.from("2020-01-31");
        [d.until("2020-02-29", { largestUnit: "months" }), d.until("2020-03-30", { largestUnit: "months" }),
         d.until("2021-02-28", { largestUnit: "years" }), d.until("2020-03-31", { largestUnit: "months" }),
         Temporal.PlainDate.from("2020-02-29").until("2021-02-28", { largestUnit: "years" }),
         Temporal.PlainDate.from("2020-03-31").until("2020-02-29", { largestUnit: "months" }),
         Temporal.PlainDate.from("2020-03-31").since("2020-02-29", { largestUnit: "months" }),
         Temporal.PlainDate.from("2019-12-31").until("2021-03-01", { largestUnit: "months" }),
         Temporal.Duration.from("P1M").round({ smallestUnit: "day", relativeTo: "2020-01-31" }),
         Temporal.PlainDateTime.from("2020-01-31T12:00").until("2020-02-29T12:00", { largestUnit: "months" }),
         Temporal.PlainDate.from("1970-01-01").until("+275760-09-13", { largestUnit: "years" }),
         Temporal.PlainDate.from("+275760-09-13").until("-271821-04-19", { largestUnit: "years" })].join()
    "#;
    assert_eq!(
        run(src),
        "P29D,P1M30D,P1Y28D,P2M,P11M30D,-P1M,P1M,P14M1D,P29D,P29D,P273790Y8M12D,-P547581Y4M24D"
    );
}

/// The difference no longer counts Jan 31 + P1M10H as one month (it is 29
/// days 10 h), so the destination lies past the `[0, 1]`-month rounding
/// window; ComputeNudgeWindow then shifts the window one month further,
/// `[1, 2]` (tc39/proposal-temporal#3168), for both `round` and `total`.
#[test]
fn calendar_rounding_window_shifts_past_an_overshooting_day() {
    let src = r#"
        var d = new Temporal.Duration(0, 1, 0, 0, 10);
        var relativeTo = new Temporal.PlainDate(2020, 1, 31);
        var n = Temporal.Duration.from("-P1MT10H");
        [d.round({ smallestUnit: "months", roundingMode: "expand", relativeTo }),
         d.round({ smallestUnit: "months", relativeTo }),
         d.total({ unit: "months", relativeTo }),
         n.round({ smallestUnit: "months", roundingMode: "expand", relativeTo: "2020-03-31" }),
         n.total({ unit: "months", relativeTo: "2020-03-31" })].join()
    "#;
    assert_eq!(
        run(src),
        "P2M,P1M,1.0134408602150538,-P2M,-1.014367816091954"
    );
}

/// With parameter expressions the body's variable environment holds only the
/// body's own `var` names, and only those that are also parameters start with
/// the parameter's value (FunctionDeclarationInstantiation step 28). A body
/// with a direct eval used to copy every parameter-scope binding into it, so
/// a body assignment to a parameter was invisible to a default's closure, and
/// a `var` declared by an eval in a default leaked its value into the body's
/// own `var` of that name.
#[test]
fn body_var_env_holds_only_body_vars_with_parameter_expressions() {
    let src = r#"
        function f1(a, b = () => a) { eval(''); a = 2; return [a, b()]; }
        function f2(a = eval('var zz = 3'), b = () => zz) { var zz; return [zz, b()]; }
        function f3(a = eval('var zz = 3'), b = () => zz) { zz = 4; return [zz, b()]; }
        function f4(a, b = () => arguments) { eval(''); arguments = 3; return typeof b(); }
        function f5(a, b = (x) => { a = x; }) { eval(''); b(7); return a; }
        function f6(a, b = () => a) { eval(''); var a = 2; return [a, b()]; }
        [f1(1), f2(), f3(), f4(1), f5(1), f6(1)].join(";")
    "#;
    assert_eq!(run(src), "2,2;,3;4,4;number;7;2,1");
}

/// An object argument that an Array method coerces (a `fromIndex`, a
/// `start`/`end`, a `join` separator, a `flat` depth) runs user code after
/// `len` was read and before any element is: the elements must then be read
/// live (`HasProperty`/`Get` up to that `len`). The dense path used a snapshot
/// taken before the coercion, so writes, deletions and truncations made by a
/// `valueOf`/`toString` were invisible.
#[test]
fn array_methods_read_elements_after_coercing_object_arguments() {
    let src = r#"
        function shrink(a, v) { return { valueOf() { a.length = 1; return v; }, toString() { a.length = 1; return v; } }; }
        function poke(a, v) { return { valueOf() { a[2] = 9; return v; }, toString() { a[2] = 9; return v; } }; }
        var r = [];
        var a;
        a = [1, 2, 3]; r.push(a.includes(undefined, shrink(a, 0)));
        a = [1, 2, 3]; r.push(a.join(shrink(a, "+")));
        a = [1, 2, 3]; r.push(JSON.stringify(a.slice(shrink(a, 0))));
        a = [1, 2, 3]; r.push(JSON.stringify(a.with(shrink(a, 0), "x")));
        a = [1, 2, 3]; r.push(String(a.at(shrink(a, 2))));
        a = [1, 2, 3]; r.push(JSON.stringify(a.splice(shrink(a, 0))));
        a = [1, 2, 3]; r.push(a.includes(9, poke(a, 0)));
        a = [1, 2, 3]; r.push(JSON.stringify(a.toSpliced(poke(a, 1))));
        a = [1, 2, 3]; r.push(JSON.stringify(a.copyWithin(poke(a, 0), 1)));
        a = [1, 2, 3]; r.push(JSON.stringify(a.flat(poke(a, 1))));
        var o = { length: 3, 0: "a", 1: "b", 2: "c" };
        r.push(Array.prototype.join.call(o, { toString() { delete o[1]; o.length = 1; return "+"; } }));
        r.join(" ")
    "#;
    assert_eq!(
        run(src),
        "true 1++ [1,null,null] [\"x\",null,null] undefined [1,null,null] true [1] [2,9,9] [1,2,9] a++c"
    );
}

/// FlattenIntoArray reads every source array live: a hole is `HasProperty`
/// through the prototype chain (an inherited index is flattened in), an
/// accessor index runs its getter, and a proxy of an array is an array
/// (IsArray). `flat` only did that for the receiver's own holes and
/// `flatMap` never did, skipping every hole or accessor slot of a nested or
/// mapped array and appending a proxy whole.
#[test]
fn flat_and_flat_map_read_nested_arrays_live() {
    let src = r#"
        var r = [];
        Array.prototype[1] = "P";
        r.push(JSON.stringify([0, , 2].flat()));
        r.push(JSON.stringify([[1, , 3]].flat()));
        r.push(JSON.stringify([1, , 3].flatMap(x => [x])));
        r.push(JSON.stringify([1].flatMap(() => [1, , 3])));
        delete Array.prototype[1];
        var g = [1, 2];
        Object.defineProperty(g, 0, { get() { return "g"; } });
        r.push(JSON.stringify([[0, g]].flat(2)));
        r.push(JSON.stringify([1].flatMap(() => g)));
        r.push(JSON.stringify([1].flatMap(() => new Proxy([1, 2], {}))));
        r.push(JSON.stringify([[1], new Proxy([3, [4]], {})].flat(2)));
        r.join(" ")
    "#;
    assert_eq!(
        run(src),
        "[0,\"P\",2] [1,\"P\",3] [1,\"P\",3] [1,\"P\",3] [0,\"g\",2] [\"g\",2] [1,2] [1,3,4]"
    );
}

/// A proxy has `[[Call]]` only if its target is callable and `[[Construct]]`
/// only if its target is a constructor (ProxyCreate step 7). A direct call
/// of a proxy over a plain object reached its `apply` trap, and `new` of a
/// proxy over a plain object or an arrow reached its `construct` trap,
/// instead of throwing a TypeError.
#[test]
fn proxy_call_and_construct_need_a_callable_or_constructor_target() {
    let src = r#"
        function k(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
        var pa = new Proxy({}, { apply() { return 1; } });
        var pc = new Proxy({}, { construct() { return {}; } });
        var pf = new Proxy(() => 1, { construct() { return {}; } });
        var pg = new Proxy(function () { return 2; }, { construct() { return { c: 3 }; } });
        [k(() => pa()), k(() => new Proxy(pa, {})()), k(() => new pc()), k(() => new pf()),
         k(() => pf()), k(() => pg()), k(() => new pg().c), typeof pa, typeof pf].join()
    "#;
    assert_eq!(
        run(src),
        "TypeError,TypeError,TypeError,TypeError,1,2,3,object,function"
    );
}

/// `typeof x` resolves the reference `x` once: inside `with (o)` that is one
/// HasBinding (`has` + `@@unscopables`), then GetValue's own `HasProperty`
/// and `Get`. The binding was resolved a second time before the read, so a
/// proxy `with` object saw its `has` trap and `@@unscopables` read twice.
#[test]
fn typeof_in_with_resolves_the_reference_once() {
    let src = r#"
        var log = [];
        var o = new Proxy({ a: 1 }, {
            has(t, k) { log.push("has:" + String(k)); return Reflect.has(t, k); },
            get(t, k, r) { log.push("get:" + String(k)); return Reflect.get(t, k, r); }
        });
        var r;
        with (o) { r = typeof a + typeof zz; }
        r + " " + log.join()
    "#;
    assert_eq!(
        run(src),
        "numberundefined has:r,has:a,get:Symbol(Symbol.unscopables),has:a,get:a,has:zz"
    );
}

/// A static class method's `[[SourceText]]` is its MethodDefinition, the
/// member text after the `static` modifier, and `Function.prototype.toString`
/// returns it. Static methods and accessors fell back to the NativeFunction
/// form because the member's span began at `static`.
#[test]
fn static_class_methods_keep_their_source_text() {
    let src = r#"
        class A {
            static m ( a ) { return a; }
            static async *g() {}
            static get q() { return 1; }
            static ['c' + 1]() {}
            static #p() {}
            static p() { return A.#p; }
            n() {}
        }
        [String(A.m), String(A.g), String(Object.getOwnPropertyDescriptor(A, "q").get),
         String(A.c1), String(A.p()), String(A.prototype.n)].join(" | ")
    "#;
    assert_eq!(
        run(src),
        "m ( a ) { return a; } | async *g() {} | get q() { return 1; } | ['c' + 1]() {} | #p() {} | n() {}"
    );
}

/// A `JSON.stringify` replacer (or `toJSON`) that returns a fresh array at
/// every level never closes a cycle, so the serialization recurses without
/// end; it now stops at `max_json_depth` with a RangeError (as a stack
/// overflow would) instead of running forever with a quadratic cycle check.
/// Run on a generous stack, as the CLI and the Test262 harness threads do:
/// each level re-enters the VM for the replacer.
#[test]
fn json_stringify_runaway_replacer_recursion_is_a_range_error() {
    let src = r#"
        function k(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
        function nest(n) { var a = []; for (var i = 0; i < n; i++) a = [a]; return a; }
        [k(() => JSON.stringify(1, function (key) { return [key]; })),
         k(() => JSON.stringify({ toJSON() { return { a: this }; } }.toJSON())),
         k(() => JSON.stringify({ get a() { return { a: { toJSON() { return [this]; } } }; } })),
         k(() => JSON.stringify(nest(1500)).length)].join()
    "#;
    let out = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            crate::stack::set_thread_stack_size(64 * 1024 * 1024);
            run(src)
        })
        .expect("spawn")
        .join()
        .expect("join");
    assert_eq!(out, "RangeError,RangeError,RangeError,3002");
}

/// The descriptor object a proxy's `getOwnPropertyDescriptor` trap yields is
/// built by FromPropertyDescriptor, whose fields come in a fixed order
/// (value, writable, get, set, enumerable, configurable). It kept
/// ToPropertyDescriptor's read order (enumerable and configurable first).
#[test]
fn proxy_descriptor_fields_come_in_from_property_descriptor_order() {
    let src = r#"
        var t = Object.freeze({ a: 1 });
        var p = new Proxy(t, { getOwnPropertyDescriptor(t, k) { return Reflect.getOwnPropertyDescriptor(t, k); } });
        var q = new Proxy({}, { getOwnPropertyDescriptor() { return { configurable: true, set: undefined, get() {} }; } });
        [Object.keys(Object.getOwnPropertyDescriptor(p, "a")).join(),
         Object.keys(Reflect.getOwnPropertyDescriptor(q, "x")).join()].join(" ")
    "#;
    assert_eq!(
        run(src),
        "value,writable,enumerable,configurable get,set,enumerable,configurable"
    );
}

/// A typed array's `[[Delete]]` of a canonical numeric key fails exactly when
/// the key names an element (10.4.5.6). `Reflect.deleteProperty(ta, "0")`
/// reported true (the `delete` operator already got it right).
#[test]
fn typed_array_delete_of_an_element_fails() {
    let src = r#"
        var ta = new Uint8Array(2);
        [Reflect.deleteProperty(ta, "0"), Reflect.deleteProperty(ta, 1),
         Reflect.deleteProperty(ta, "5"), Reflect.deleteProperty(ta, "-0"),
         Reflect.deleteProperty(ta, "x")].join()
    "#;
    assert_eq!(run(src), "false,false,true,true,true");
}

/// A write that reaches a typed array on the prototype chain with a
/// canonical numeric key is that typed array's integer-indexed `[[Set]]`:
/// an out-of-bounds (or `-0`, fractional) index is a dropped write, and only
/// a valid index creates an own data property on the receiver. The VM's
/// property-add fast paths created the own property for every such key.
#[test]
fn writes_through_a_typed_array_prototype_honour_its_set() {
    let src = r#"
        var a = new Uint8Array(2);
        function keysAfter(f) { var o = Object.create(a); f(o); return Object.keys(o).join("+"); }
        var k = "5";
        [keysAfter(o => { o[5] = 1; }), keysAfter(o => { o["5"] = 1; }), keysAfter(o => { o[k] = 1; }),
         keysAfter(o => { o["-0"] = 1; }), keysAfter(o => { o[1.5] = 1; }), keysAfter(o => { o[1] = 1; }),
         keysAfter(o => { o.x = 1; }), String(a[1])].join()
    "#;
    assert_eq!(run(src), ",,,,,1,x,0");
}

/// A String wrapper on the prototype chain owns `length` and its index keys
/// (StringGetOwnProperty): `Object.create(new String("ab"))[0]` is "a" and
/// `.length` is 2, and since those properties are read-only, assigning them
/// through the child fails instead of creating own properties. The reads saw
/// `undefined` and the writes created own `length` / index properties.
#[test]
fn string_wrapper_prototype_exposes_its_length_and_indices() {
    let src = r#"
        var s = new String("ab");
        var o = Object.create(s);
        var r = [o[0], o["1"], o.length, Reflect.get(o, "0")];
        o.length = 9; o[1] = "z"; var k = "0"; o[k] = "y";
        var strict;
        try { (function () { "use strict"; Object.create(s).length = 9; })(); strict = "no error"; }
        catch (e) { strict = e.constructor.name; }
        r.concat([Object.keys(o).length, o.length, o[1], strict]).join()
    "#;
    assert_eq!(run(src), "a,b,2,a,0,2,b,TypeError");
}

/// `x instanceof Function` is OrdinaryHasInstance: a walk of `x`'s prototype
/// chain for `Function.prototype`, not a callability check. A function whose
/// prototype was set to null (or to a plain object) is no `Function`
/// instance, and an object inheriting `Function.prototype` is one.
#[test]
fn instanceof_function_walks_the_prototype_chain() {
    let src = r#"
        var f = function () {}; Object.setPrototypeOf(f, null);
        var g = function () {}; Object.setPrototypeOf(g, {});
        var o = Object.create(Function.prototype);
        class A {}
        [f instanceof Function, g instanceof Function, o instanceof Function,
         A instanceof Function, (() => 1) instanceof Function, Math.max instanceof Function,
         (function () {}).bind() instanceof Function].join()
    "#;
    assert_eq!(run(src), "false,false,true,true,true,true,true");
}

/// Global declaration instantiation creates the global object's properties
/// in the spec's order (GlobalDeclarationInstantiation and
/// EvalDeclarationInstantiation with Annex B.3.2.2 / B.3.2.3): Annex B block
/// functions first (those that are neither a top-level function nor a
/// `var`), then the top-level functions (each placed by its *last*
/// declaration), then the `var` names. The engine created the `var`s first.
#[test]
fn global_declarations_create_properties_in_spec_order() {
    let src = r#"
        var mine = /^(x|f|b|e)\d$|^pick$/;
        function pick() { return Object.getOwnPropertyNames(globalThis).filter(function (k) { return mine.test(k); }).join(); }
        var script = pick();
        var x1 = 1;
        function f1() {}
        { function b1() {} }
        var x2;
        function f2() {}
        function f1() {}
        if (true) { function b2() {} }
        { function f2() {} }
        { function x1() {} }
        var keys = Object.keys(globalThis).filter(function (k) { return mine.test(k); }).join();
        (0, eval)("var e1; function e3(){} { function e5(){} } var e2; function e4(){} function e3(){}");
        [script, keys, pick().slice(keys.length + 1)].join(" | ")
    "#;
    assert_eq!(
        run(src),
        "b1,b2,pick,f2,f1,x1,x2 | b1,b2,pick,f2,f1,x1,x2 | e5,e4,e3,e1,e2"
    );
}

/// A sloppy `eval` whose declarations are rejected (EvalDeclarationInstantiation
/// throws before creating any binding) must leave the caller's Annex B
/// block-function bookkeeping intact: a later block function at that level
/// still becomes visible in the enclosing variable scope. It was lost, so
/// the call threw "is not a function".
#[test]
fn rejected_eval_declarations_keep_annex_b_block_functions() {
    let src = r#"
        var r = [];
        try { (0, eval)("var zz; function NaN() {}"); } catch (e) { r.push(e.constructor.name, "zz" in globalThis); }
        { function blk() { return 3; } }
        r.push(blk());
        (function () {
            let q;
            { try { eval("var q;"); } catch (e) { r.push(e.constructor.name); } }
            { function inner() { return 4; } }
            r.push(inner());
        })();
        r.join()
    "#;
    assert_eq!(run(src), "TypeError,false,3,SyntaxError,4");
}

/// Annex B.3.2.2 / B.3.2.3: a block-level function that cannot var-hoist
/// simply stays block-scoped. In sloppy eval code a lexical binding of its
/// name between the eval and its variable environment (`bindingExists`; a
/// catch parameter is exempt, B.3.4) skips the hoisting, and at global scope
/// so does a name CanDeclareGlobalVar rejects; the copy into an existing
/// non-writable global (`NaN`) fails silently. The engine threw a
/// SyntaxError for the first case and a TypeError for `{ function NaN(){} }`.
#[test]
fn annex_b_block_functions_skip_instead_of_throwing() {
    let src = r#"
        var r = [];
        r.push((function () { let q = 1; eval("{ function q() {} }"); return typeof q; })());
        r.push((function () { { let q = 1; eval("{ function q() {} }"); var i = typeof q; } return i + "/" + typeof q; })());
        r.push((function () { const q = 1; eval("{ function q() {} } { function z() {} }"); return typeof q + "/" + typeof z; })());
        r.push((function () { try { throw 1; } catch (q) { eval("{ function q() {} }"); var i = typeof q; } return i + "/" + typeof q; })());
        r.push((function () { { function q() { return 1; } eval("{ function q() { return 2; } }"); var i = q(); } return i + "/" + q(); })());
        r.push((function () { eval("{ function q() {} }"); return delete q; })());
        r.push((0, eval)("{ function NaN() {} } typeof NaN"));
        { function NaN() {} }
        r.push(typeof NaN);
        r.join()
    "#;
    assert_eq!(
        run(src),
        "number,number/undefined,number/function,number/function,1/1,true,number,number"
    );
}

/// `Array.prototype.includes` / `indexOf` on an empty array-like return
/// before ToIntegerOrInfinity(fromIndex), so its `valueOf` never runs (the
/// generic array-like path coerced it first).
#[test]
fn includes_and_index_of_skip_from_index_on_empty_array_likes() {
    let src = r#"
        var n = 0; var fi = { valueOf() { n++; return 0; } };
        var r = [Array.prototype.includes.call({ length: 0 }, 1, fi),
                 Array.prototype.indexOf.call({ length: -3 }, 1, fi),
                 Array.prototype.includes.call(function () {}, 1, fi),
                 Array.prototype.lastIndexOf.call({ length: 0 }, 1, fi)];
        r.concat(n).join()
    "#;
    assert_eq!(run(src), "false,-1,false,-1,0");
}

/// `Date.prototype.setYear` with a year far past the time-value range is
/// NaN (MakeDay / TimeClip); the engine overflowed its integer day count
/// and stored a wrapped date.
#[test]
fn set_year_out_of_range_is_nan() {
    let src = r#"
        var r = [];
        for (var y of [1e20, -1e20, Infinity, 275761, 99]) { var d = new Date(0); r.push(d.setYear(y)); }
        r.join()
    "#;
    assert_eq!(run(src), "NaN,NaN,NaN,NaN,915148800000");
}

/// EvaluateCall reads a method (its getter, a proxy `get` trap) before the
/// arguments are evaluated, and calls the method it read: an argument that
/// replaces or deletes it does not change the call. The engine read the
/// method after its arguments (`Op::CallMethod`); it now reads it first
/// whenever the arguments could tell (`Op::GetMethod` + `Op::CallWithMethod`,
/// behind an `Op::PlainPrims` guard for operators over plain values).
#[test]
fn method_call_reads_the_callee_before_its_arguments() {
    let src = r#"
        var log = [];
        function A(x) { log.push("arg" + x); return x; }
        var o = { get f() { log.push("get"); return function (x) { log.push("f" + x); }; } };
        o.f(A(1)); o["f"](A(2));
        var p = new Proxy({ g(x) { log.push("g" + x); } },
          { get(t, k) { log.push("trap"); return t[k]; } });
        p.g(A(3));
        var q = { m(x) { log.push("old" + x); } };
        q.m((q.m = function (x) { log.push("new" + x); }, 4)); q.m(5);
        var r = { m(x) { log.push("r" + x); } };
        r.m((delete r.m, 6));
        var t = { get tag() { log.push("tag"); return function (s, x) { log.push("t" + x); }; } };
        t.tag`a${A(7)}`;
        var v = { m(x) { log.push("v" + x); } };
        var w = { valueOf() { v.m = function () { log.push("vnew"); }; return 8; } };
        v.m(w + 0);
        var a = []; a.push((a.push = function () { log.push("fake"); }, 9));
        var s = 0, u = { add(x) { return x + 1; } };
        for (var i = 0; i < 50; i++) s += u.add(i * 2);
        log.join() + "|" + a.join() + "|" + s
    "#;
    assert_eq!(
        run(src),
        "get,arg1,f1,get,arg2,f2,trap,arg3,g3,old4,new5,r6,tag,arg7,t7,v8|9|2500"
    );
}

/// A quantified group iterated thousands of times still matches. The regex
/// backtracker recursed once per choice point and capture save, and past a
/// fixed recursion cap it reported a silent "no match": `/(?:ab)+c/` failed
/// on `"ab".repeat(2000) + "c"`. Choice points now live on an explicit heap
/// stack, so only lookarounds recurse.
#[test]
fn long_quantified_group_iterations_still_match() {
    let src = r#"
        var r = [];
        var s = "ab".repeat(20000) + "c";
        r.push(/(?:ab)+c/.test(s), /(ab)+c/.exec(s)[1], /^(?:ab)*c$/.test(s), /(?:a|b)+c/.test(s));
        r.push(/(?:ab)+?c/.test(s), /(?:a.)+c/.exec(s).index, /((?:ab){2})+c/.exec(s)[0].length);
        r.push(/(?:(?=a)ab)+c/.test(s), /(?:(?<=b|^)ab)+c/.test(s), /(?:(?i:AB))+c/.test(s));
        var csv = "x,".repeat(5000) + "end";
        r.push(/^(?:[^,]*,)+end$/.test(csv), csv.match(/(?:([^,]*),)+/)[1]);
        r.push(/(a+)+b/.test("a".repeat(30)));
        r.join()
    "#;
    assert_eq!(
        run(src),
        "true,ab,true,true,true,0,40001,true,true,true,true,x,false"
    );
}

/// `Array.prototype.join` reads `length` once and then `Get`s each element in
/// turn, so an element's `toString` that changes the array is seen by the
/// elements after it (and likewise each element's `toLocaleString`). The
/// engine joined a snapshot of the elements taken before any user code ran.
#[test]
fn join_reads_elements_after_user_code_live() {
    let src = r#"
        var r = [];
        var f = [{ toString() { f.length = 1; return "x"; } }, "y", "z"]; r.push(f.join());
        var g = [{ toString() { g[2] = "changed"; return "a"; } }, "b", "c"]; r.push(g.join("-"));
        var t = [{ toLocaleString() { t.length = 1; return "L"; } }, 5, 6]; r.push(t.toLocaleString());
        var u = [1, { toLocaleString() { u[2] = { toLocaleString() { return "new"; } }; return "m"; } }, 3];
        r.push(u.toLocaleString());
        var p = [{ toString() { Array.prototype[2] = "proto"; p.length = 2; p.length = 3; return "s"; } }, 1, 2];
        r.push(p.join()); delete Array.prototype[2];
        r.join("|")
    "#;
    assert_eq!(run(src), "x,,|a-b-changed|L,,|1,m,new|s,1,proto");
}

/// A join cycle through more than one array (`a = [b]; b = [a]`) renders the
/// revisited array as `""`, as a direct self-reference already did, matching
/// every web engine. Each level is a native-to-native call with no call-depth
/// guard, so the engine recursed until the native stack overflowed and the
/// process aborted.
#[test]
fn join_cycle_through_several_arrays_renders_empty() {
    let src = r#"
        var r = [];
        var a = [1]; var b = [a]; a.push(b); r.push(String(a), a.join("-"), a.toLocaleString(), String([a]));
        var n = [1, [2, [3, [4]]]]; n[1][1][1].push(n); r.push(String(n));
        var s = [1]; s.push(s); r.push(String(s), [s, s].join());
        var w = [[1, 2], [3]]; r.push(String(w), String([w, w]));
        r.join("|")
    "#;
    assert_eq!(run(src), "1,|1-|1,|1,|1,2,3,4,|1,|1,,1,|1,2,3|1,2,3,1,2,3");
}

/// `bind` builds the bound name `"bound " + targetName` as a rope join: a
/// chain of re-binds stays linear (it copied every name, quadratic in the
/// chain length) and a lone surrogate in the target's name survives (it was
/// replaced by U+FFFD).
#[test]
fn bound_function_name_is_a_rope_join() {
    let src = r#"
        var f = function () { return 7; };
        for (var i = 0; i < 20000; i++) f = f.bind(null);
        var k = function () {}; Object.defineProperty(k, "name", { value: "a\uD800b" });
        var n = k.bind().name;
        var g = function () {}; Object.defineProperty(g, "name", { value: 42 });
        [f.name.length, f.name.slice(0, 12), n.length, n.charCodeAt(7), JSON.stringify(g.bind().name)].join()
    "#;
    assert_eq!(run(src), r#"120001,bound bound ,9,55296,"bound ""#);
}

/// `at`/`with` count from the array's real `length`, even when it lies past
/// the dense store (`a.length = 2**32 - 1`, or an index set beyond the dense
/// cap): `at(-1)` read the last *stored* element, and `with` built an array
/// of the stored elements only. `with` on such a length is now the engine's
/// "Invalid array length" RangeError rather than a short array.
#[test]
fn at_and_with_honor_a_length_past_the_dense_store() {
    let src = r#"
        function S(v) { return v === undefined ? "u" : String(v); }
        function k(f) { try { return S(f()); } catch (e) { return e.constructor.name; } }
        var L = 2 ** 32 - 1;
        function mk() { var a = [1, 2, 3]; a.length = L; return a; }
        var s = []; s[2e8] = "x";
        [k(() => mk().at(-1)), k(() => mk().at(-L)), k(() => mk().at(2)),
         k(() => { var a = mk(); a[L - 1] = "last"; return a.at(-1); }),
         k(() => mk().with(-1, 9).length), k(() => s.at(-1)), k(() => s.at(0)),
         k(() => [1, 2, 3].at(-1))].join()
    "#;
    assert_eq!(run(src), "u,1,3,last,RangeError,x,u,3");
}

/// A user String key that starts with NUL is an ordinary key: it enumerates,
/// serializes and round-trips through every key path, distinct from the
/// engine's internal `"\0…"` slots (which used to swallow `"\0a"`-style names
/// from `Object.keys`, `for-in`, `JSON.stringify` and `Reflect.ownKeys`).
#[test]
fn nul_prefixed_string_keys_are_ordinary_keys() {
    let src = r#"
        var esc = function (a) { return JSON.stringify(a); };
        var ks = ["\0a", "\0", "\0\0", "\0#x", "\0sym:1", "\0prim", "\0\v", "a\0b"];
        var o = {};
        for (var i = 0; i < ks.length; i++) o[ks[i]] = i;
        var fi = []; for (var k in o) fi.push(k);
        var lit = { "\0lit": 1, ["\0comp"]: 2 };
        var parsed = JSON.parse('{"\\u0000p":1,"\\u0000sym:2":2}');
        var d = {}; Object.defineProperty(d, "\0d", { value: 1, enumerable: true });
        var trap = [];
        var px = new Proxy({ "\0t": 1 }, { get: function (t, k) { trap.push(k); return t[k]; } });
        px["\0t"];
        var hits = ks.filter(function (k) { return k in o && o.hasOwnProperty(k) && o[k] === ks.indexOf(k); }).length;
        [esc(Object.keys(o)), esc(fi), esc(Reflect.ownKeys(o)), esc(o), esc(Object.keys(lit)),
         esc(Object.keys(parsed)), esc(Object.keys(d)), esc(trap), hits,
         esc(Object.keys(JSON.parse(JSON.stringify(o))))].join("|")
    "#;
    let keys = r#"["\u0000a","\u0000","\u0000\u0000","\u0000#x","\u0000sym:1","\u0000prim","\u0000\u000b","a\u0000b"]"#;
    let obj = r#"{"\u0000a":0,"\u0000":1,"\u0000\u0000":2,"\u0000#x":3,"\u0000sym:1":4,"\u0000prim":5,"\u0000\u000b":6,"a\u0000b":7}"#;
    let want = [
        keys,
        keys,
        keys,
        obj,
        r#"["\u0000lit","\u0000comp"]"#,
        r#"["\u0000p","\u0000sym:2"]"#,
        r#"["\u0000d"]"#,
        r#"["\u0000t"]"#,
        "8",
        keys,
    ]
    .join("|");
    assert_eq!(run(src), want);
}

/// No String key can name an internal slot: a guessed private-name storage key
/// neither reads nor overwrites `#x` nor passes its brand check, a
/// `"\0sym:<id>"` string does not alias the Symbol with that id, and a
/// `"\0prim"` / `"\0PromiseState"` own property does not make an object a
/// primitive wrapper or a promise.
#[test]
fn string_keys_cannot_forge_internal_slots() {
    let src = r#"
        class A { #x = 1; static get(o) { return o.#x; } static has(o) { return #x in o; } }
        var a = new A(), b = {};
        for (var i = 0; i < 4000; i++) {
            var k = "\0#x@v" + i.toString(16);
            a[k] = 42; a[k + "f"] = 42; b[k] = 1; b[k + "f"] = 1;
        }
        var forged;
        try { A.get(b); forged = "read"; } catch (e) { forged = e instanceof TypeError; }
        var c = {};
        for (var j = 0; j < 100; j++) c["\0sym:" + j] = function () { return { next: function () { return { done: true }; } }; };
        var iter;
        try { for (var _ of c); iter = "iterated"; } catch (e) { iter = e instanceof TypeError; }
        var w = { "\0prim": 5 };
        var n = new Number(3); n["\0prim"] = 9;
        var pr;
        try { Promise.prototype.then.call({ "\0PromiseState": 1 }); pr = "then"; } catch (e) { pr = e instanceof TypeError; }
        [A.get(a), A.has(b), forged, iter, Object.getOwnPropertySymbols(c).length,
         Object.prototype.toString.call(w), n.valueOf(), pr].join()
    "#;
    assert_eq!(run(src), "1,false,true,true,0,[object Object],3,true");
}

/// A Symbol produced by an object key's `@@toPrimitive` is still that Symbol
/// (the host's ToPropertyKey hands the VM the key *value*, not its storage
/// string), and functions named after NUL string keys keep the key's text.
#[test]
fn key_values_round_trip_symbols_and_names() {
    let src = r#"
        var s = Symbol("k");
        var o = {}; o[{ [Symbol.toPrimitive]: function () { return s; } }] = 1;
        var m = { "\0m": function () {}, get "\0g"() { return 1; }, ["\0c"]: () => 0 };
        class K { static "\0sm"() {} "\0f" = function () {}; }
        [o[s], Object.getOwnPropertySymbols(o).length, Object.keys(o).length,
         JSON.stringify([m["\0m"].name, Object.getOwnPropertyDescriptor(m, "\0g").get.name,
                         m["\0c"].name, K["\0sm"].name, new K()["\0f"].name])].join("|")
    "#;
    assert_eq!(
        run(src),
        r#"1|1|0|["\u0000m","get \u0000g","\u0000c","\u0000sm","\u0000f"]"#
    );
}

/// The `Array.prototype` built-ins on a real array whose `length` lies past
/// its dense store (`a.length = 2**32 - 1`) run their index-by-index
/// algorithms against the real `length` (they took the dense store's size,
/// so `fill(7, -1)` wrote index 2 and `slice(-2)` returned the last two
/// *stored* elements), stepping over the absent region past the dense store
/// so each still finishes at once. `delete` of an element stored past the
/// dense cap removes it (it was a silent no-op), and such an element on an
/// array *prototype* is inherited (it read as `undefined`).
#[test]
fn array_builtins_honor_a_length_past_the_dense_store() {
    let src = r#"
        var L = 2 ** 32 - 1;
        function mk() { var a = [1, 2, 3]; a.length = L; return a; }
        function K(a) { return Object.keys(a).map(function (k) { return k + "=" + a[k]; }).join(",") + "|" + a.length; }
        var r = [];
        r.push(K(mk().slice(-2)), K(mk().slice(0, 5)));
        r.push(mk().indexOf(3, -1), mk().includes(3, -2), mk().includes(undefined));
        var a = mk(); a[L - 3] = 3; r.push(a.lastIndexOf(3), a.lastIndexOf(3, -4));
        a = mk(); a.fill(7, -1); r.push(K(a));
        a = mk(); a[L - 1] = 9; a.copyWithin(0, L - 2); r.push(K(a));
        a = mk(); a[L - 1] = 9; var s = a.splice(1, 1); r.push(K(s), K(a));
        a = mk(); a[L - 2] = 5; s = a.splice(0); r.push(K(s), K(a));
        a = mk(); a[L - 2] = "r"; a.reverse(); r.push(K(a));
        a = mk(); a[L - 2] = 0; a.sort(); r.push(K(a));
        a = mk(); a[L - 1] = 4; r.push(a.shift(), K(a));
        a = mk(); a.length = L - 1; r.push(a.unshift(0), K(a));
        a = mk(); a[L - 1] = 5; var seen = []; a.forEach(function (v, i) { seen.push(i); });
        r.push(seen.join(" "), a.reduce(function (x, y) { return x + y; }));
        a = mk(); a[L - 2] = 0; r.push(delete a[L - 2], K(a));
        var p = []; p[L - 2] = "P"; var b = []; Object.setPrototypeOf(b, p);
        r.push(b[L - 2], Object.create(p)[L - 2]);
        r.join(" ; ")
    "#;
    assert_eq!(
        run(src),
        "|2 ; 0=1,1=2,2=3|5 ; -1 ; false ; true ; 4294967292 ; 2 ; \
         0=1,1=2,2=3,4294967294=7|4294967295 ; 1=9,2=3,4294967294=9|4294967295 ; \
         0=2|1 ; 0=1,1=3,4294967293=9|4294967294 ; \
         0=1,1=2,2=3,4294967293=5|4294967295 ; |0 ; \
         1=r,4294967292=3,4294967293=2,4294967294=1|4294967295 ; \
         0=0,1=1,2=2,3=3|4294967295 ; 1 ; 0=2,1=3,4294967293=4|4294967294 ; \
         4294967295 ; 0=0,1=1,2=2,3=3|4294967295 ; 0 1 2 4294967294 ; 11 ; \
         true ; 0=1,1=2,2=3|4294967295 ; P ; P"
    );
}

/// A mutator on an array whose `length` is non-writable still performs every
/// element read, move and delete its algorithm makes before the final
/// `Set(O, "length", …)` throws: `shift` moved the elements down, `pop`
/// deleted the last one, and `splice` throws even when the length would not
/// change (the dense fast paths threw up front, leaving the array untouched,
/// and `splice` with an unchanged length did not throw at all).
#[test]
fn mutators_move_elements_before_a_readonly_length_throws() {
    let src = r#"
        function show(a) { var s = []; for (var i = 0; i < a.length; i++) s.push(i in a ? a[i] : "_"); return a.length + "[" + s.join(",") + "]"; }
        function ro() { var a = [0, 1, 2, 3]; Object.defineProperty(a, "length", { writable: false }); return a; }
        var r = [];
        [function (a) { return a.shift(); }, function (a) { return a.pop(); },
         function (a) { return a.splice(1, 1); }, function (a) { return a.splice(1, 1, 8); },
         function (a) { return a.splice(0, 0); }].forEach(function (f) {
            var a = ro(), e = "ok";
            try { f(a); } catch (x) { e = x.constructor.name; }
            r.push(e + show(a));
        });
        r.join(" ")
    "#;
    assert_eq!(
        run(src),
        "TypeError4[1,2,3,_] TypeError4[0,1,2,_] TypeError4[0,2,3,_] TypeError4[0,8,2,3] TypeError4[0,1,2,3]"
    );
}
