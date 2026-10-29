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
