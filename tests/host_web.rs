//! Integration tests for the §4.3 web-platform globals installed by
//! `kataan::host::web::install` — `TextEncoder`/`TextDecoder`, `atob`/`btoa`,
//! `URL`/`URLSearchParams`, `structuredClone`, `performance`, and `console`.
//!
//! Each test builds a fresh `Interp`, installs the web globals, runs a JS
//! snippet, and asserts on the completion value (`interp.display`) or captured
//! `console` output (`interp.output`).

use kataan::Interp;
use kataan::parser::Parser;

/// Run `src`, returning `(console_output, completion_display)`.
fn run(src: &str) -> (String, String) {
    let mut interp = Interp::new();
    kataan::host::web::install(&mut interp);
    let program = Parser::parse_program(src).expect("parse");
    let result = interp.run(&program).expect("run");
    let display = interp.display(result);
    (interp.output().to_string(), display)
}

/// The completion value of `src` as a display string.
fn eval(src: &str) -> String {
    run(src).1
}

/// The captured `console` output of `src`.
fn out(src: &str) -> String {
    run(src).0
}

// ---------------------------------------------------------------------------
// TextEncoder / TextDecoder
// ---------------------------------------------------------------------------

#[test]
fn text_encoder_encode_ascii() {
    assert_eq!(eval(r#"new TextEncoder().encode("ABC").length + ''"#), "3");
    assert_eq!(eval(r#"new TextEncoder().encode("ABC")[0] + ''"#), "65");
    assert_eq!(eval(r#"new TextEncoder().encoding"#), "utf-8");
}

#[test]
fn text_encoder_returns_uint8array() {
    assert_eq!(
        eval(r#"(new TextEncoder().encode("x") instanceof Uint8Array) + ''"#),
        "true"
    );
}

#[test]
fn text_encoder_utf8_multibyte() {
    // "é" is 2 UTF-8 bytes (0xC3 0xA9); "€" is 3 bytes.
    assert_eq!(eval(r#"new TextEncoder().encode("é").length + ''"#), "2");
    assert_eq!(eval(r#"new TextEncoder().encode("€").length + ''"#), "3");
}

#[test]
fn text_encode_decode_roundtrip() {
    let src = r#"
        var enc = new TextEncoder();
        var dec = new TextDecoder();
        var s = "héllo, 世界 €";
        dec.decode(enc.encode(s)) === s ? "ok" : "fail"
    "#;
    assert_eq!(eval(src), "ok");
}

#[test]
fn text_encoder_encode_into() {
    let src = r#"
        var enc = new TextEncoder();
        var buf = new Uint8Array(10);
        var r = enc.encodeInto("AB", buf);
        r.read + "," + r.written + "," + buf[0] + "," + buf[1]
    "#;
    assert_eq!(eval(src), "2,2,65,66");
}

#[test]
fn text_decoder_from_arraybuffer_and_view() {
    // Decode straight from an ArrayBuffer, and from a subarray view.
    let src = r#"
        var bytes = new Uint8Array([72, 105]); // "Hi"
        var dec = new TextDecoder();
        var a = dec.decode(bytes.buffer);
        var b = dec.decode(bytes.subarray(1));
        a + "|" + b
    "#;
    assert_eq!(eval(src), "Hi|i");
}

#[test]
fn text_decoder_utf16le() {
    // "Hi" as UTF-16LE: 0x48 0x00 0x69 0x00.
    let src = r#"
        var bytes = new Uint8Array([0x48, 0x00, 0x69, 0x00]);
        new TextDecoder("utf-16le").decode(bytes)
    "#;
    assert_eq!(eval(src), "Hi");
}

#[test]
fn text_decoder_fatal_throws_on_invalid_utf8() {
    let src = r#"
        try {
            new TextDecoder("utf-8", { fatal: true }).decode(new Uint8Array([0xFF, 0xFE]));
            "no-throw"
        } catch (e) { "threw" }
    "#;
    assert_eq!(eval(src), "threw");
}

#[test]
fn text_decoder_invalid_label_throws() {
    let src = r#"
        try { new TextDecoder("no-such-encoding"); "no-throw" }
        catch (e) { "threw" }
    "#;
    assert_eq!(eval(src), "threw");
}

// ---------------------------------------------------------------------------
// atob / btoa
// ---------------------------------------------------------------------------

#[test]
fn btoa_basic() {
    assert_eq!(eval(r#"btoa("hello")"#), "aGVsbG8=");
    assert_eq!(eval(r#"btoa("hello world")"#), "aGVsbG8gd29ybGQ=");
    assert_eq!(eval(r#"btoa("")"#), "");
}

#[test]
fn atob_basic() {
    assert_eq!(eval(r#"atob("aGVsbG8=")"#), "hello");
    assert_eq!(eval(r#"atob("aGVsbG8gd29ybGQ=")"#), "hello world");
}

#[test]
fn base64_roundtrip() {
    let src = r#"
        var s = "The quick brown fox: 0123456789!";
        atob(btoa(s)) === s ? "ok" : "fail"
    "#;
    assert_eq!(eval(src), "ok");
}

#[test]
fn atob_invalid_throws_named_error() {
    let src = r#"
        try { atob("not base64!!!"); "no-throw" }
        catch (e) { e.name }
    "#;
    assert_eq!(eval(src), "InvalidCharacterError");
}

#[test]
fn btoa_non_latin1_throws_named_error() {
    let src = r#"
        try { btoa("😀"); "no-throw" }
        catch (e) { e.name }
    "#;
    assert_eq!(eval(src), "InvalidCharacterError");
}

// ---------------------------------------------------------------------------
// URL
// ---------------------------------------------------------------------------

#[test]
fn url_full_parse_href() {
    let href = "https://user:pass@example.com:8080/path/to?a=1&b=2#frag";
    assert_eq!(eval(&format!(r#"new URL("{href}").href"#)), href);
}

#[test]
fn url_component_getters() {
    let src = r#"
        var u = new URL("https://user:pass@example.com:8080/p/q?a=1#h");
        [u.protocol, u.username, u.password, u.hostname, u.port, u.host,
         u.pathname, u.search, u.hash].join("|")
    "#;
    assert_eq!(
        eval(src),
        "https:|user|pass|example.com|8080|example.com:8080|/p/q|?a=1|#h"
    );
}

#[test]
fn url_origin() {
    assert_eq!(
        eval(r#"new URL("https://example.com/x").origin"#),
        "https://example.com"
    );
    assert_eq!(
        eval(r#"new URL("http://example.com:8080/x").origin"#),
        "http://example.com:8080"
    );
}

#[test]
fn url_default_port_dropped() {
    assert_eq!(eval(r#"new URL("http://a.com:80/").port"#), "");
    assert_eq!(
        eval(r#"new URL("https://a.com:443/").href"#),
        "https://a.com/"
    );
}

#[test]
fn url_relative_resolution() {
    assert_eq!(
        eval(r#"new URL("../c", "http://a.com/b/d").href"#),
        "http://a.com/c"
    );
    assert_eq!(
        eval(r#"new URL("/x", "http://a.com/b/c").href"#),
        "http://a.com/x"
    );
    assert_eq!(
        eval(r#"new URL("g", "http://a.com/b/c").href"#),
        "http://a.com/b/g"
    );
    assert_eq!(
        eval(r#"new URL("?y", "http://a.com/b/c?x").href"#),
        "http://a.com/b/c?y"
    );
}

#[test]
fn url_setters_update_href() {
    let src = r##"
        var u = new URL("http://a.com/old?q=1#f");
        u.pathname = "/new";
        u.search = "?k=2";
        u.hash = "#g";
        u.href
    "##;
    assert_eq!(eval(src), "http://a.com/new?k=2#g");
}

#[test]
fn url_protocol_setter() {
    let src = r#"
        var u = new URL("http://a.com/");
        u.protocol = "https:";
        u.href
    "#;
    assert_eq!(eval(src), "https://a.com/");
}

#[test]
fn url_tostring_and_tojson() {
    assert_eq!(
        eval(r#"new URL("http://a.com/x").toString()"#),
        "http://a.com/x"
    );
    assert_eq!(
        eval(r#"JSON.stringify({ u: new URL("http://a.com/x") })"#),
        r#"{"u":"http://a.com/x"}"#
    );
}

#[test]
fn url_opaque_scheme() {
    assert_eq!(
        eval(r#"new URL("mailto:foo@bar.com").href"#),
        "mailto:foo@bar.com"
    );
    assert_eq!(eval(r#"new URL("mailto:foo@bar.com").protocol"#), "mailto:");
}

#[test]
fn url_invalid_throws() {
    let src = r#"
        try { new URL("not a url"); "no-throw" } catch (e) { "threw" }
    "#;
    assert_eq!(eval(src), "threw");
}

#[test]
fn url_search_params_accessor() {
    let src = r#"
        var u = new URL("http://a.com/?a=1&b=2&a=3");
        u.searchParams.getAll("a").join(",") + "|" + u.searchParams.get("b")
    "#;
    assert_eq!(eval(src), "1,3|2");
}

// ---------------------------------------------------------------------------
// URLSearchParams
// ---------------------------------------------------------------------------

#[test]
fn usp_from_string() {
    let src = r#"
        var p = new URLSearchParams("a=1&b=2&a=3");
        p.get("a") + "|" + p.getAll("a").join(",") + "|" + p.has("b") + "|" + p.has("z")
    "#;
    assert_eq!(eval(src), "1|1,3|true|false");
}

#[test]
fn usp_leading_question_mark() {
    assert_eq!(eval(r#"new URLSearchParams("?x=1").get("x")"#), "1");
}

#[test]
fn usp_append_set_delete() {
    let src = r#"
        var p = new URLSearchParams();
        p.append("a", "1");
        p.append("a", "2");
        p.append("b", "3");
        p.set("a", "9");
        p.delete("b");
        p.toString()
    "#;
    assert_eq!(eval(src), "a=9");
}

#[test]
fn usp_tostring_encodes() {
    let src = r#"
        var p = new URLSearchParams();
        p.append("k v", "a&b");
        p.toString()
    "#;
    assert_eq!(eval(src), "k+v=a%26b");
}

#[test]
fn usp_sort() {
    let src = r#"
        var p = new URLSearchParams("c=3&a=1&b=2");
        p.sort();
        p.toString()
    "#;
    assert_eq!(eval(src), "a=1&b=2&c=3");
}

#[test]
fn usp_from_array_and_object() {
    assert_eq!(
        eval(r#"new URLSearchParams([["a", "1"], ["b", "2"]]).toString()"#),
        "a=1&b=2"
    );
    assert_eq!(
        eval(r#"new URLSearchParams({ a: "1", b: "2" }).toString()"#),
        "a=1&b=2"
    );
}

#[test]
fn usp_entries_iterator() {
    let src = r#"
        var p = new URLSearchParams("a=1&b=2");
        var out = [];
        for (var e of p.entries()) { out.push(e[0] + "=" + e[1]); }
        out.join("&")
    "#;
    assert_eq!(eval(src), "a=1&b=2");
}

#[test]
fn usp_direct_iteration() {
    let src = r#"
        var p = new URLSearchParams("a=1&b=2");
        var out = [];
        for (var [k, v] of p) { out.push(k + ":" + v); }
        out.join(",")
    "#;
    assert_eq!(eval(src), "a:1,b:2");
}

#[test]
fn usp_keys_values() {
    let src = r#"
        var p = new URLSearchParams("a=1&b=2");
        [...p.keys()].join(",") + "|" + [...p.values()].join(",")
    "#;
    assert_eq!(eval(src), "a,b|1,2");
}

#[test]
fn usp_foreach() {
    let src = r#"
        var p = new URLSearchParams("a=1&b=2");
        var out = [];
        p.forEach(function (v, k) { out.push(k + "=" + v); });
        out.join("&")
    "#;
    assert_eq!(eval(src), "a=1&b=2");
}

// ---------------------------------------------------------------------------
// structuredClone
// ---------------------------------------------------------------------------

#[test]
fn structured_clone_primitives() {
    assert_eq!(eval(r#"structuredClone(42) + ''"#), "42");
    assert_eq!(eval(r#"structuredClone("hi")"#), "hi");
    assert_eq!(eval(r#"structuredClone(true) + ''"#), "true");
    assert_eq!(eval(r#"structuredClone(null) + ''"#), "null");
}

#[test]
fn structured_clone_object_deep() {
    let src = r#"
        var o = { a: 1, b: { c: [2, 3], d: "x" } };
        var c = structuredClone(o);
        var equal = c.a === 1 && c.b.c[0] === 2 && c.b.c[1] === 3 && c.b.d === "x";
        var distinct = c !== o && c.b !== o.b && c.b.c !== o.b.c;
        (equal && distinct) ? "ok" : "fail"
    "#;
    assert_eq!(eval(src), "ok");
}

#[test]
fn structured_clone_mutation_independence() {
    let src = r#"
        var o = { arr: [1, 2, 3] };
        var c = structuredClone(o);
        c.arr[0] = 99;
        o.arr[0] + "|" + c.arr[0]
    "#;
    assert_eq!(eval(src), "1|99");
}

#[test]
fn structured_clone_map_set() {
    let src = r#"
        var m = new Map([["a", 1], ["b", 2]]);
        var s = new Set([1, 2, 3]);
        var cm = structuredClone(m);
        var cs = structuredClone(s);
        (cm instanceof Map) + "|" + cm.get("a") + "|" + cm.size + "|" +
        (cs instanceof Set) + "|" + cs.has(2) + "|" + cs.size + "|" + (cm !== m)
    "#;
    assert_eq!(eval(src), "true|1|2|true|true|3|true");
}

#[test]
fn structured_clone_date_regexp() {
    let src = r#"
        var d = new Date(1000000);
        var r = /ab+c/gi;
        var cd = structuredClone(d);
        var cr = structuredClone(r);
        (cd instanceof Date) + "|" + (cd.getTime() === 1000000) + "|" +
        (cr instanceof RegExp) + "|" + cr.source + "|" + cr.flags
    "#;
    assert_eq!(eval(src), "true|true|true|ab+c|gi");
}

#[test]
fn structured_clone_typed_array() {
    let src = r#"
        var a = new Uint8Array([1, 2, 3]);
        var c = structuredClone(a);
        c[0] = 9;
        (c instanceof Uint8Array) + "|" + c.length + "|" + a[0] + "|" + c[0]
    "#;
    assert_eq!(eval(src), "true|3|1|9");
}

#[test]
fn structured_clone_cycle() {
    let src = r#"
        var o = { name: "root" };
        o.self = o;
        var c = structuredClone(o);
        (c.self === c) + "|" + (c !== o) + "|" + c.name
    "#;
    assert_eq!(eval(src), "true|true|root");
}

#[test]
fn structured_clone_function_throws() {
    let src = r#"
        try { structuredClone(function () {}); "no-throw" }
        catch (e) { e.name }
    "#;
    assert_eq!(eval(src), "DataCloneError");
}

#[test]
fn structured_clone_symbol_throws() {
    let src = r#"
        try { structuredClone(Symbol("x")); "no-throw" }
        catch (e) { "threw" }
    "#;
    assert_eq!(eval(src), "threw");
}

// ---------------------------------------------------------------------------
// performance
// ---------------------------------------------------------------------------

#[test]
fn performance_now_is_number_and_monotonic() {
    let src = r#"
        var a = performance.now();
        var b = performance.now();
        (typeof a === "number") + "|" + (b >= a)
    "#;
    assert_eq!(eval(src), "true|true");
}

#[test]
fn performance_time_origin() {
    assert_eq!(
        eval(r#"(typeof performance.timeOrigin === "number") + ''"#),
        "true"
    );
}

#[test]
fn performance_mark_measure_entries() {
    let src = r#"
        performance.mark("start");
        performance.mark("end");
        performance.measure("dur", "start", "end");
        var e = performance.getEntriesByName("dur");
        e.length + "|" + e[0].name + "|" + e[0].entryType + "|" + (typeof e[0].duration)
    "#;
    assert_eq!(eval(src), "1|dur|measure|number");
}

#[test]
fn performance_get_entries_by_type() {
    let src = r#"
        performance.mark("m1");
        performance.mark("m2");
        performance.getEntriesByType("mark").length + ""
    "#;
    assert_eq!(eval(src), "2");
}

// ---------------------------------------------------------------------------
// console
// ---------------------------------------------------------------------------

#[test]
fn console_log_multiple_args() {
    assert_eq!(out(r#"console.log("a", "b", "c");"#), "a b c\n");
}

#[test]
fn console_log_numbers_and_bools() {
    assert_eq!(out(r#"console.log(1, true, null);"#), "1 true null\n");
}

#[test]
fn console_format_substitution() {
    assert_eq!(out(r#"console.log("x=%s y=%d", "hi", 5);"#), "x=hi y=5\n");
    assert_eq!(out(r#"console.log("%i and %f", 3.9, 2.5);"#), "3 and 2.5\n");
    assert_eq!(out(r#"console.log("100%% done");"#), "100% done\n");
}

#[test]
fn console_format_json_directive() {
    assert_eq!(out(r#"console.log("%j", { a: 1 });"#), "{\"a\":1}\n");
}

#[test]
fn console_object_inspection() {
    assert_eq!(out(r#"console.log({ a: 1, b: 2 });"#), "{ a: 1, b: 2 }\n");
    assert_eq!(out(r#"console.log([1, 2, 3]);"#), "[ 1, 2, 3 ]\n");
    assert_eq!(
        out(r#"console.log({ s: "x", n: [1] });"#),
        "{ s: 'x', n: [ 1 ] }\n"
    );
}

#[test]
fn console_info_warn_error_debug() {
    let o = out(r#"
        console.info("i");
        console.warn("w");
        console.error("e");
        console.debug("d");
    "#);
    assert_eq!(o, "i\nw\ne\nd\n");
}

#[test]
fn console_group_indentation() {
    let o = out(r#"
        console.group("A");
        console.log("inner");
        console.groupEnd();
        console.log("outer");
    "#);
    assert_eq!(o, "A\n  inner\nouter\n");
}

#[test]
fn console_count() {
    let o = out(r#"
        console.count("x");
        console.count("x");
        console.count("y");
    "#);
    assert_eq!(o, "x: 1\nx: 2\ny: 1\n");
}

#[test]
fn console_assert() {
    let o = out(r#"
        console.assert(true, "should not show");
        console.assert(false, "boom");
    "#);
    assert_eq!(o, "Assertion failed: boom\n");
}

#[test]
fn console_map_set_inspection() {
    assert_eq!(
        out(r#"console.log(new Map([["a", 1]]));"#),
        "Map(1) { 'a' => 1 }\n"
    );
    assert_eq!(out(r#"console.log(new Set([1, 2]));"#), "Set(2) { 1, 2 }\n");
}

#[test]
fn console_does_not_double_install() {
    // Only one console binding exists; a plain log still produces one line.
    assert_eq!(out(r#"console.log("once");"#), "once\n");
}
