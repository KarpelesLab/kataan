//! Unit tests for the tree-walk interpreter (moved out of mod.rs).
use super::*;
use crate::parser::Parser;

/// Runs `src` and renders the program's final value.
fn run(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

#[test]
fn lone_surrogates_round_trip_through_string_ops() {
    // Creation preserves a lone surrogate; length is in UTF-16 units.
    assert_eq!(run(r#""\uD800".length === 1"#), "true");
    assert_eq!(run(r#""\uD800".charCodeAt(0) === 0xD800"#), "true");
    assert_eq!(run(r#""\u{D834}".charCodeAt(0) === 0xD834"#), "true");
    // An astral char is two units, one code point.
    assert_eq!(
        run(
            r#""😀".length === 2 && "😀".codePointAt(0) === 0x1F600 && "😀".charCodeAt(0) === 0xD83D"#
        ),
        "true"
    );
    // slice over UTF-16 units keeps a lone surrogate.
    assert_eq!(
        run(r#""a\uD800b".slice(1,2).charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(
        run(r#""a\uD800b".substring(1,2).charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(run(r#""a\uD800b".at(1) === "\uD800""#), "true");
    assert_eq!(run(r#""a\uD800b"[1].charCodeAt(0) === 0xD800"#), "true");
    // charAt of a lone surrogate is a one-unit string carrying the surrogate.
    assert_eq!(
        run(r#""\uD800".charAt(0).charCodeAt(0) === 0xD800"#),
        "true"
    );
    // fromCharCode preserves a lone surrogate.
    assert_eq!(
        run("String.fromCharCode(0xD800).charCodeAt(0) === 0xD800"),
        "true"
    );
    assert_eq!(
        run("String.fromCharCode(0xD83D, 0xDE00).codePointAt(0) === 0x1F600"),
        "true"
    );
}

/// P1/P5/P6: the string-method fast path (single rope flatten, lazy lossy
/// `String`, running UTF-16 counts) must not change observable behaviour on
/// ordinary or surrogate-bearing strings.
#[test]
fn string_method_correctness_after_perf_rework() {
    // charCodeAt / slice / indexOf on ordinary strings.
    assert_eq!(run(r#""abc".charCodeAt(1)"#), "98");
    assert_eq!(run(r#""hello".slice(1,3)"#), "el");
    assert_eq!(run(r#""hello".indexOf("ll")"#), "2");
    assert_eq!(run(r#""hello".lastIndexOf("l")"#), "3");
    // The lazily-built lossy `String` arms still work.
    assert_eq!(run(r#""  hi  ".trim()"#), "hi");
    assert_eq!(run(r#""  hi".trimStart()"#), "hi");
    assert_eq!(run(r#""hi  ".trimEnd()"#), "hi");
    assert_eq!(run(r#""abcabc".search("ca")"#), "2");
    assert_eq!(run(r#""a".localeCompare("b") < 0"#), "true");
    // replace / replaceAll, including the function callback whose match-offset
    // is now produced by a running UTF-16 unit count (P6).
    assert_eq!(run(r#""a-b-c".replace("-", "+")"#), "a+b-c");
    assert_eq!(run(r#""a-b-c".replaceAll("-", "+")"#), "a+b+c");
    // The callback receives the correct UTF-16 offsets (1 and 3) at each match.
    assert_eq!(run(r#""a-b-c".replaceAll("-", (m,o)=>o)"#), "a1b3c");
    // P6 offsets stay correct past an astral char (the 😀 counts as 2 UTF-16
    // units, so the first `-` is at unit 2 and the second at unit 4).
    assert_eq!(run(r#""😀-x-y".replaceAll("-", (m,o)=>o)"#), "😀2x4y");
    // Surrogate-bearing strings round-trip through the byte-based ops.
    assert_eq!(run(r#""\uD800".length === 1"#), "true");
    assert_eq!(run(r#""\uD800".charCodeAt(0) === 0xD800"#), "true");
    assert_eq!(
        run(r#""a\uD800b".slice(1,2).charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(run(r#""😀".length"#), "2");
}

/// RE-P1: a `RegExp` whose compiled program is now cached on its cell must
/// still behave identically when reused across many calls — the cache returns
/// a consistent program, `lastIndex` keeps advancing for `g`/`y`, and two
/// regexes that share a source but differ in flags must not collide.
#[cfg(feature = "regex")]
#[test]
fn regex_compiled_cache_preserves_behaviour() {
    // Reusing one regex across a loop yields the same result every call (the
    // cached program is used, not recompiled into something different).
    assert_eq!(
        run(r#"{
                    let re = /a(\d)/;
                    let out = [];
                    for (let i = 0; i < 5; i++) out.push(re.test("a7") + "" + (re.exec("a7")[1]));
                    out.join(",")
                }"#),
        "true7,true7,true7,true7,true7"
    );

    // A global regex reused via String.match collects every occurrence.
    assert_eq!(
        run(r#"{ let re=/(\d+)/g; "a1b22c333".match(re).join(",") }"#),
        "1,22,333"
    );

    // `lastIndex` advances across repeated stateful `exec`/`test` calls and
    // resets to 0 after the final miss — unaffected by the program cache.
    assert_eq!(
        run(r#"{
                    let re = /\d/g;
                    let s = "a1b2";
                    let idx = [];
                    re.exec(s); idx.push(re.lastIndex);
                    re.exec(s); idx.push(re.lastIndex);
                    re.exec(s); idx.push(re.lastIndex);   // miss -> reset to 0
                    idx.join(",")
                }"#),
        "2,4,0"
    );

    // A sticky regex's lastIndex advances exactly at the match boundary.
    assert_eq!(
        run(r#"{
                    let re = /\d/y;
                    re.lastIndex = 1;
                    let m = re.test("a1b2");
                    m + ":" + re.lastIndex
                }"#),
        "true:2"
    );

    // `lastIndex` is a real own data property (RegExpAlloc's DefinePropertyOrThrow):
    // hasOwnProperty is true even before any assignment, its descriptor is
    // { writable:true, enumerable:false, configurable:false }, and it appears in
    // getOwnPropertyNames — while still reading/writing the compact cell field.
    assert_eq!(run(r#"/x/.hasOwnProperty("lastIndex")"#), "true");
    assert_eq!(
        run(
            r#"var d=Object.getOwnPropertyDescriptor(/x/g,"lastIndex"); [d.value,d.writable,d.enumerable,d.configurable].join(",")"#
        ),
        "0,true,false,false"
    );
    assert_eq!(
        run(r#"Object.getOwnPropertyNames(/x/).indexOf("lastIndex")>=0"#),
        "true"
    );
    // A materialized non-writable descriptor is honored by an assignment.
    assert_eq!(
        run(
            r#"{ let re=/x/; Object.defineProperty(re,"lastIndex",{value:3,writable:false}); re.lastIndex=9; re.lastIndex }"#
        ),
        "3"
    );

    // Same source, different flags are distinct programs and must not collide
    // through the cache: `/x/u` (unicode) vs `/x/` (plain) behave per their
    // own flags. `/😀/u` matches the astral char as one unit-pair; `/./` only
    // ever spans one code unit, while `/./u` spans the whole astral char.
    assert_eq!(run(r#"/x/u.test("x") && !/x/u.global"#), "true");
    assert_eq!(run(r#"/x/.test("x") && /x/g.global"#), "true");
    // `.` with and without `u` over an astral subject: non-`u` `.` matches one
    // code unit (length-1 match), `u` `.` matches the whole code point (2).
    assert_eq!(run(r#""😀".match(/./)[0].length"#), "1");
    assert_eq!(run(r#""😀".match(/./u)[0].length"#), "2");

    // Two regexes built from the same source string but different flags, used
    // in the same scope, keep independent compiled programs.
    assert_eq!(
        run(r#"{
                    let a = /\w+/;
                    let b = /\w+/g;
                    let r1 = "foo bar".match(a).length;     // non-global: 1 match
                    let r2 = "foo bar".match(b).length;     // global: 2 matches
                    r1 + "," + r2
                }"#),
        "1,2"
    );
}

/// C1: a dense-array element *write* to a valid array index (`< 2^32-1`) past the
/// configured `max_array_len` cap is served *sparsely* (stored as an aux named
/// property + a logical-length bump), never a `RangeError` — a plain `arr[i] = v`
/// is spec-conformant and grows `length` to `i + 1`. A *length* set to a valid
/// uint32 above the cap is likewise a sparse length (no allocation); only a length
/// above the uint32 ceiling (2^32-1) is invalid.
#[test]
fn oversized_array_growth_throws_range_error() {
    // `a[1e9] = 1` (index 1e9 > the 100M default cap) stores sparsely and grows
    // `length` to 1e9+1 — a valid array index write never throws.
    assert_eq!(
        run("var a=[1]; a[1e9]=1; String(a.length)+','+a[1e9]"),
        "1000000001,1"
    );
    // `a.length = 1e9` is a valid uint32: a sparse length, reported as-is, no throw.
    assert_eq!(
        run("var a=[1]; a.length=1e9; String(a.length)"),
        "1000000000"
    );
    // Computed `a["length"] = 1e9` behaves the same.
    assert_eq!(
        run("var a=[1]; a['length']=1e9; String(a.length)"),
        "1000000000"
    );
    // A length above the uint32 ceiling (2^32) is invalid → RangeError.
    assert_eq!(
        run("var a=[1]; try{a.length=4294967296;'noThrow'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    // A within-cap grow / length set still works (no regression).
    assert_eq!(
        run("var a=[1]; a[5]=9; JSON.stringify(a)"),
        "[1,null,null,null,null,9]"
    );
    assert_eq!(
        run("var a=[1,2,3,4,5]; a.length=2; JSON.stringify(a)"),
        "[1,2]"
    );
}

/// C2: a deeply nested expression (shallow in the AST via the precedence loop,
/// but thousands of native `eval` recursions) throws a catchable `RangeError`
/// rather than overflowing the host stack. Run on a generous stack so the
/// `max_eval_depth` guard fires before the (much larger) real overflow point,
/// exactly as the production / test262 harness threads do.
#[test]
fn deep_expression_throws_instead_of_overflowing() {
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let src = core::iter::repeat_n("1", 20_000)
                .collect::<alloc::vec::Vec<_>>()
                .join("+");
            // Leak the AST: dropping a 20k-deep boxed expression chain would
            // itself recurse and is unrelated to what we are asserting.
            let program = alloc::boxed::Box::leak(alloc::boxed::Box::new(
                Parser::parse_program(&src).expect("parse"),
            ));
            let mut interp = Interp::new();
            let threw = matches!(interp.run(program), Err(ExecError::Throw(_)));
            core::mem::forget(interp);
            threw
        })
        .expect("spawn")
        .join()
        .expect("join");
    assert!(handle, "deep expression should throw, not abort");
}

/// L1: `ArrayBuffer.prototype.transfer(n)` with an enormous length throws a
/// catchable `RangeError` (via `validate_alloc_len`) instead of attempting a
/// `usize::MAX` allocation that aborts the process.
#[test]
fn array_buffer_transfer_huge_length_throws() {
    assert_eq!(
        run(
            "var b=new ArrayBuffer(4); try{b.transfer(1e309);'noThrow'}catch(e){e.constructor.name}"
        ),
        "RangeError"
    );
    // A reasonable resize still works.
    assert_eq!(
        run("var b=new ArrayBuffer(4); b.transfer(8).byteLength"),
        "8"
    );
}

#[test]
fn surrogate_search_pad_split_iteration_units() {
    // indexOf/includes over UTF-16 units (astral char shifts the index by 2).
    assert_eq!(run(r#""😀x".indexOf("x") === 2"#), "true");
    assert_eq!(run(r#""😀x".includes("x")"#), "true");
    assert_eq!(run(r#""a😀b".lastIndexOf("b") === 3"#), "true");
    assert_eq!(run(r#""😀b".startsWith("😀")"#), "true");
    assert_eq!(run(r#""a😀".endsWith("😀")"#), "true");
    // padStart/padEnd count UTF-16 units; an astral pad char counts as two.
    assert_eq!(run(r#""x".padStart(3, "😀").length === 3"#), "true");
    assert_eq!(run(r#""x".padEnd(2).length === 2"#), "true");
    // split('') yields one entry per UTF-16 unit (astral → two halves).
    assert_eq!(run(r#""😀".split("").length === 2"#), "true");
    assert_eq!(run(r#""a\uD800b".split("").length === 3"#), "true");
    // for-of yields code points: an astral char is a single iteration.
    assert_eq!(run(r#"[..."😀"].length === 1"#), "true");
    assert_eq!(run(r#"[..."a\uD800b"].length === 3"#), "true");
    // repeat/concat preserve surrogates losslessly.
    assert_eq!(run(r#""\uD800".repeat(2).length === 2"#), "true");
    assert_eq!(
        run(r#""\uD800".repeat(2).charCodeAt(1) === 0xD800"#),
        "true"
    );
    assert_eq!(
        run(r#"("a".concat("\uD800")).charCodeAt(1) === 0xD800"#),
        "true"
    );
}

#[cfg(feature = "regex")]
#[test]
fn regex_u16_code_unit_indices_and_uflag() {
    // u-flag: `.` matches the whole astral char (one code point) but reports
    // its span in code units (length 2) at code-unit index 0.
    assert_eq!(run(r#"/./u.exec("😀")[0].length === 2"#), "true");
    assert_eq!(run(r#"/./u.exec("😀").index === 0"#), "true");
    // Non-u: `.` matches one code unit, so an astral subject yields two whole
    // matches and the first match is one unit long.
    assert_eq!(run(r#"/./.exec("😀")[0].length === 1"#), "true");
    assert_eq!(run(r#""😀".match(/./g).length === 2"#), "true");
    // A literal astral match under the u-flag, spliced back, is byte-stable.
    assert_eq!(run(r#""a😀b".replace(/😀/u, "X") === "aXb""#), "true");
    // `.index` / `lastIndex` / matchAll index are code-unit indices on an
    // astral subject.
    assert_eq!(run(r#""a😀b".search(/b/) === 3"#), "true");
    assert_eq!(
        run(r#"{ const r=/b/g; r.exec("😀b"); r.lastIndex === 3 }"#),
        "true"
    );
    assert_eq!(
        run(r#"[..."a😀b".matchAll(/(.)/gu)].map(m=>m.index).join(",") === "0,1,3""#),
        "true"
    );
    // split over an astral subject keeps surrounding text whole.
    assert_eq!(run(r#""a😀b".split(/😀/).join("|") === "a|b""#), "true");
    // `$&`/`$1`/`` $` ``/`$'` substitutions operate on code-unit slices and
    // re-encode astral characters losslessly.
    assert_eq!(
        run(r#""x😀y".replace(/(😀)/, "[$1]") === "x[😀]y""#),
        "true"
    );
    assert_eq!(run(r#""a😀b".replace(/😀/, "$`$'") === "aabb""#), "true");
    // A surrogate-bearing subject (forces the tree-walker path) matches via
    // its code units and the captured slice carries the lone surrogate.
    assert_eq!(
        run(r#""a\uD800b".replace(/\uD800/u, "X") === "aXb""#),
        "true"
    );
}

#[test]
fn case_and_normalize_preserve_surrogates() {
    // A lone surrogate has no case and survives toUpperCase/toLowerCase.
    assert_eq!(
        run(r#""\uD800".toUpperCase().charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(
        run(r#""\uDC00".toLowerCase().charCodeAt(0) === 0xDC00"#),
        "true"
    );
    // Surrounding scalars still case-map; the surrogate stays put.
    assert_eq!(run(r#""a\uD800b".toUpperCase() === "A\uD800B""#), "true");
    // The surrogate-free fast path is unchanged, including `ß`→`SS`.
    assert_eq!(run(r#""abc".toUpperCase() === "ABC""#), "true");
    assert_eq!(run(r#""ß".toUpperCase() === "SS""#), "true");
    assert_eq!(run(r#""ABC".toLowerCase() === "abc""#), "true");
    // normalize is the identity on a lone surrogate (it round-trips).
    assert_eq!(
        run(r#""\uD800".normalize().charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(
        run(r#""a\uD800é".normalize("NFC").charCodeAt(1) === 0xD800"#),
        "true"
    );
    // A surrogate-free string still normalizes (NFC composes here).
    assert_eq!(run(r#""é".normalize("NFC") === "é""#), "true");
}

#[test]
fn json_preserves_lone_surrogates() {
    // stringify escapes a lone surrogate as `\uXXXX` (well-formed JSON).
    assert_eq!(run(r#"JSON.stringify("\uD800") === '"\\ud800"'"#), "true");
    // A valid astral char round-trips as the character.
    assert_eq!(run(r#"JSON.stringify("😀") === '"😀"'"#), "true");
    // parse of a `\uXXXX` lone surrogate preserves it.
    assert_eq!(
        run(r#"JSON.parse('"\\ud800"').charCodeAt(0) === 0xD800"#),
        "true"
    );
    assert_eq!(run(r#"JSON.parse('"\\ud800"').length === 1"#), "true");
    // parse pairs `😀` into one astral code point.
    assert_eq!(
        run(r#"JSON.parse('"\\ud83d\\ude00"').codePointAt(0) === 0x1F600"#),
        "true"
    );
    // Round-trip a string with an embedded lone surrogate.
    assert_eq!(
        run(r#"JSON.parse(JSON.stringify("a\uD800b")).charCodeAt(1) === 0xD800"#),
        "true"
    );
}

#[test]
fn non_surrogate_strings_behave_as_before() {
    // A plain corpus must be unchanged by the WTF-8 storage move.
    assert_eq!(run(r#""hello".length"#), "5");
    assert_eq!(run(r#""héllo 中".length"#), "7");
    assert_eq!(run(r#""abcde".slice(1,3)"#), "bc");
    assert_eq!(run(r#""a,b,c".split(",").length"#), "3");
    assert_eq!(run(r#""banana".indexOf("na")"#), "2");
    assert_eq!(run(r#""banana".lastIndexOf("na")"#), "4");
    assert_eq!(run(r#""x".padStart(3, "ab")"#), "abx");
    assert_eq!(
        run(r#"JSON.stringify({a:1,b:"hi"})"#),
        r#"{"a":1,"b":"hi"}"#
    );
    assert_eq!(run(r#"`a${1}b${2}c`"#), "a1b2c");
}

#[test]
fn limits_override_changes_runtime_caps() {
    use crate::limits::Limits;
    // A lowered `max_string_len` rejects a concatenation the default accepts,
    // proving the cap is read live from `realm.limits` rather than a constant.
    let src = "'abcde'.repeat(3)"; // 15 chars
    assert_eq!(eval_source(src).expect("default ok").1, "abcdeabcdeabcde");
    let low = Limits {
        max_string_len: 10,
        ..Limits::default()
    };
    let err = eval_source_with_limits(src, low).expect_err("should exceed length");
    assert!(err.contains("Invalid string length"), "unexpected: {err}");

    // A low object→dictionary threshold forces the conversion early yet keeps
    // correct property semantics (count, values, insertion order preserved).
    let dict = Limits {
        object_dictionary_threshold: 4,
        ..Limits::default()
    };
    let keys_src = "let o={}; for(let i=0;i<10;i++) o['k'+i]=i; [Object.keys(o).length, o.k0, o.k9, Object.keys(o)[0]].join(',')";
    assert_eq!(
        eval_source_with_limits(keys_src, dict).expect("dict ok").1,
        "10,0,9,k0"
    );
}

/// C2 follow-up: a custom low `max_eval_depth` (via `Realm::with_limits`,
/// threaded through `eval_source_with_limits`) is honored live. The tree-walk
/// recursion that the interpreter performs on a deeply nested expression
/// trips the dedicated knob — a depth the *default* realm evaluates fine is
/// rejected once the cap is lowered, proving `max_eval_depth` bounds the
/// eval/exec recursion independently of `max_call_depth`.
#[test]
fn max_eval_depth_override_honored() {
    // Each tree-walk level burns a lot of native stack, so run on a generous
    // stack (like the production / test262 threads) where the *guard*, not a
    // real overflow, is the limiting factor.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            use crate::limits::Limits;
            // A left-deep `1+1+…+1`: shallow allocations but `depth` nested
            // native `eval` recursions (one per `+` term), driving
            // `eval_depth` up by one per level within a single frame.
            fn deep_add(depth: usize) -> String {
                core::iter::repeat_n("1", depth)
                    .collect::<alloc::vec::Vec<_>>()
                    .join("+")
            }

            // 600 terms evaluate cleanly under the default cap…
            let src = deep_add(600);
            assert_eq!(eval_source(&src).expect("default ok").1, "600");

            // …but a realm whose `max_eval_depth` is lowered below that depth
            // rejects the very same source with a catchable stack-overflow
            // `RangeError`, while `max_call_depth` is left at its (much
            // higher) default — proving the dedicated knob is honored live.
            let low = Limits {
                max_eval_depth: 100,
                ..Limits::default()
            };
            let err = eval_source_with_limits(&src, low).expect_err("should exceed eval depth");
            assert!(
                err.contains("Maximum call stack size exceeded"),
                "unexpected error: {err}"
            );
        })
        .expect("spawn")
        .join()
        .expect("join");
}

/// C2 follow-up: interpreter recursion past `max_eval_depth` throws a
/// catchable `RangeError` (caught by a JS `try/catch`, surfacing as
/// `RangeError`) instead of crashing the host. Run on a generous native
/// stack so the guard — not a real overflow — is what stops the recursion.
#[test]
fn deep_eval_recursion_throws_range_error_catchable() {
    let kind = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            // A left-deep `1+1+…+1` far past the default `max_eval_depth`
            // (1500): each `+` term is a native `eval` recursion, so the
            // guard fires mid-evaluation and the throw is caught by JS.
            let deep = core::iter::repeat_n("1", 20_000)
                .collect::<alloc::vec::Vec<_>>()
                .join("+");
            let src =
                alloc::format!("try {{ {deep}; 'noThrow' }} catch (e) {{ e.constructor.name }}");
            // Leak the AST: dropping a 20k-deep boxed expression chain would
            // itself recurse and is unrelated to what we are asserting.
            let program = alloc::boxed::Box::leak(alloc::boxed::Box::new(
                Parser::parse_program(&src).expect("parse"),
            ));
            let mut interp = Interp::new();
            let res = interp.run(program).map(|v| interp.display(v));
            core::mem::forget(interp);
            res
        })
        .expect("spawn")
        .join()
        .expect("join");
    assert_eq!(kind.expect("eval ok"), "RangeError");
}

/// Runs `src` and returns its captured `console` output.
fn out(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    interp.run(&program).expect("exec");
    String::from(interp.output())
}

/// With the `intl` crate, `Intl.Segmenter` uses real UAX-29 grapheme clusters — an emoji
/// stays a single segment (the no-`intl` fallback splits per code point).
#[cfg(feature = "intl")]
#[test]
fn intl_segmenter_real_grapheme_clusters() {
    assert_eq!(
        out(
            r#"console.log([...new Intl.Segmenter("en").segment("a😀b")].map(s=>s.segment).join("|"))"#
        ),
        "a|😀|b\n"
    );
}

/// `Intl.Segmenter.resolvedOptions()` reports `{locale, granularity}`, and the
/// object `segment()` returns supports `containing(index)`.
#[cfg(feature = "intl")]
#[test]
fn intl_segmenter_resolved_options_and_containing() {
    assert_eq!(
        run("new Intl.Segmenter('en',{granularity:'word'}).resolvedOptions().granularity"),
        "word",
    );
    assert_eq!(
        run("new Intl.Segmenter('en').resolvedOptions().granularity"),
        "grapheme",
    );
    assert_eq!(
        run("Object.keys(new Intl.Segmenter('en').resolvedOptions()).join(',')"),
        "locale,granularity",
    );
    let s = "new Intl.Segmenter('en',{granularity:'word'}).segment('hello world')";
    assert_eq!(run(&alloc::format!("{s}.containing(2).segment")), "hello");
    assert_eq!(run(&alloc::format!("{s}.containing(6).segment")), "world");
    assert_eq!(
        run(&alloc::format!("{s}.containing(100)===undefined")),
        "true",
    );
}

/// `String.prototype.localeCompare` honors the `numeric`/`sensitivity` options.
#[cfg(feature = "intl")]
#[test]
fn intl_locale_compare_options() {
    assert_eq!(
        run("'10'.localeCompare('9',undefined,{numeric:true})>0"),
        "true"
    );
    assert_eq!(run("'10'.localeCompare('9')<0"), "true"); // default: code-point
    assert_eq!(
        run("'a10'.localeCompare('a9',undefined,{numeric:true})>0"),
        "true"
    );
    assert_eq!(
        run("'A'.localeCompare('a',undefined,{sensitivity:'base'})"),
        "0"
    );
}

/// The `Intl` namespace object is an ordinary object: its `[[Prototype]]` is
/// `%Object.prototype%` and it carries an own `[Symbol.toStringTag]` of `"Intl"`.
#[test]
fn intl_namespace_prototype_and_tostringtag() {
    assert_eq!(
        run("Object.getPrototypeOf(Intl)===Object.prototype"),
        "true"
    );
    assert_eq!(run("Object.prototype.toString.call(Intl)"), "[object Intl]");
    assert_eq!(run("Intl[Symbol.toStringTag]"), "Intl");
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(Intl,Symbol.toStringTag);\
             [d.writable,d.enumerable,d.configurable].join(',')"
        ),
        "false,false,true"
    );
    assert_eq!(run("Object.isExtensible(Intl)"), "true");
}

/// `Intl.Locale.prototype.variants` returns the base-name variant subtags (joined
/// by `-`), or `undefined` when there are none; the accessor is named
/// `"get variants"`.
#[test]
fn intl_locale_variants_getter() {
    assert_eq!(
        run("new Intl.Locale('en-US-1996-fonipa').variants"),
        "1996-fonipa"
    );
    assert_eq!(run("typeof new Intl.Locale('sv').variants"), "undefined");
    assert_eq!(
        run("new Intl.Locale('sl-rozaj-biske-1994').variants"), // sorted
        "1994-biske-rozaj"
    );
    assert_eq!(
        run("Object.getOwnPropertyDescriptor(Intl.Locale.prototype,'variants').get.name"),
        "get variants"
    );
    // Private-use subtags are preserved; a repeated `-u-` key keeps the first.
    assert_eq!(
        run("new Intl.Locale('en-x-u-foo').toString()"),
        "en-x-u-foo"
    );
    assert_eq!(
        run("new Intl.Locale('da-u-ca-gregory-ca-buddhist').toString()"),
        "da-u-ca-gregory"
    );
}

/// `Intl.Locale.prototype.maximize`/`minimize` apply the CLDR likely-subtags data
/// (only meaningful with the `intl` crate).
#[cfg(feature = "intl")]
#[test]
fn intl_locale_maximize_minimize() {
    assert_eq!(
        run("new Intl.Locale('en').maximize().toString()"),
        "en-Latn-US"
    );
    assert_eq!(
        run("new Intl.Locale('zh').maximize().toString()"),
        "zh-Hans-CN"
    );
    assert_eq!(run("new Intl.Locale('und').minimize().toString()"), "en");
    // `Hant` is the likely script for `zh-TW`, so it drops on minimization.
    assert_eq!(
        run("new Intl.Locale('zh-Hant-TW').minimize().toString()"),
        "zh-TW"
    );
    assert_eq!(
        run("new Intl.Locale('en-Latn-US').minimize().toString()"),
        "en"
    );
    // Extensions survive maximization.
    assert_eq!(
        run("new Intl.Locale('en-u-ca-gregory').maximize().toString()"),
        "en-Latn-US-u-ca-gregory"
    );
}

/// Regular grandfathered tags and CLDR `bcp47` `-u-` type-value aliases
/// canonicalize to their preferred forms (both via `getCanonicalLocales` and the
/// `Intl.Locale` constructor).
#[test]
fn intl_grandfathered_and_type_aliases() {
    // Regular grandfathered tags → canonical.
    assert_eq!(run("Intl.getCanonicalLocales('art-lojban')[0]"), "jbo");
    assert_eq!(run("Intl.getCanonicalLocales('cel-gaulish')[0]"), "xtg");
    assert_eq!(run("Intl.getCanonicalLocales('zh-guoyu')[0]"), "zh");
    assert_eq!(run("Intl.getCanonicalLocales('zh-xiang')[0]"), "hsn");
    // Irregular grandfathered forms remain structurally invalid → RangeError.
    assert_eq!(
        run("try{Intl.getCanonicalLocales('i-klingon');'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    // -u- type-value aliases.
    assert_eq!(
        run("Intl.getCanonicalLocales('und-u-ca-ethiopic-amete-alem')[0]"),
        "und-u-ca-ethioaa"
    );
    assert_eq!(
        run("Intl.getCanonicalLocales('und-u-ca-islamicc')[0]"),
        "und-u-ca-islamic-civil"
    );
    assert_eq!(
        run("Intl.getCanonicalLocales('und-u-ks-primary')[0]"),
        "und-u-ks-level1"
    );
    assert_eq!(
        run("Intl.getCanonicalLocales('und-u-ms-imperial')[0]"),
        "und-u-ms-uksystem"
    );
    // The `calendar` option is canonicalized like the `-u-ca-` type.
    assert_eq!(
        run("new Intl.Locale('en',{calendar:'islamicc'}).calendar"),
        "islamic-civil"
    );
    assert_eq!(
        run("new Intl.Locale('en',{calendar:'ethiopic-amete-alem'}).toString()"),
        "en-u-ca-ethioaa"
    );
    // A grandfathered base produced by options (`cel` + variant `gaulish`).
    assert_eq!(
        run("new Intl.Locale('cel',{variants:'gaulish'}).baseName"),
        "xtg"
    );
}

/// `-u-` extension leading attributes (keyword-less subtags) are preserved and
/// sorted; a `Locale` object is a single-element locale list; a `null` options
/// argument is a TypeError.
#[test]
fn intl_locale_attributes_list_and_null_options() {
    assert_eq!(
        run("new Intl.Locale('en-u-attr-co-phonebk').toString()"),
        "en-u-attr-co-phonebk"
    );
    assert_eq!(
        run("new Intl.Locale('pt-u-attr2-attr1-ca-gregory').toString()"),
        "pt-u-attr1-attr2-ca-gregory"
    );
    assert_eq!(
        run("new Intl.Locale('en-u-baz-a-bar-x-u-foo').toString()"),
        "en-a-bar-u-baz-x-u-foo"
    );
    // A `Locale` value passed to `getCanonicalLocales` is a single-element list.
    assert_eq!(
        run("Intl.getCanonicalLocales(new Intl.Locale('ar-EG'))[0]"),
        "ar-EG"
    );
    assert_eq!(
        run("Intl.getCanonicalLocales(new Intl.Locale('ar-EG')).length"),
        "1"
    );
    // `null` options → TypeError; `undefined` is fine.
    assert_eq!(
        run("try{new Intl.Locale('en',null);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(run("new Intl.Locale('en',undefined).toString()"), "en");
    // `CanonicalizeLocaleList` uses proxy-aware `[[HasProperty]]`: a throwing
    // `has` trap propagates.
    assert_eq!(
        run(
            "var p=new Proxy({0:'en',length:1},{has(){throw new TypeError('h')}});\
             try{Intl.getCanonicalLocales(p);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
}

/// `Number.prototype.toLocaleString` applies every NumberFormat option.
#[cfg(feature = "intl")]
#[test]
fn intl_number_to_locale_string_all_options() {
    assert_eq!(
        run("(1234567).toLocaleString('en-US',{notation:'compact'})"),
        "1.2M"
    );
    assert_eq!(
        run("(5).toLocaleString('en-US',{style:'unit',unit:'kilometer-per-hour'})"),
        "5 km/h"
    );
    assert_eq!(
        run("(1).toLocaleString('en-US',{minimumSignificantDigits:3})"),
        "1.00"
    );
    assert_eq!(run("(1234567).toLocaleString('en-US')"), "1,234,567"); // default unchanged
    assert_eq!(
        run("try{(5).toLocaleString('en-US',{style:'bad'});'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
}

/// With the `intl` crate, `Intl.PluralRules` applies real CLDR rules — Polish has the
/// `few`/`many` categories the en-only fallback can't express.
#[cfg(feature = "intl")]
#[test]
fn intl_plural_rules_real_cldr_categories() {
    assert_eq!(
        out(
            r#"var p=new Intl.PluralRules("pl");console.log([1,2,5,22].map(n=>p.select(n)).join(","))"#
        ),
        "one,few,many,few\n"
    );
}

/// With the `intl` crate, `Intl.Collator` does real UCA collation: `numeric` order sorts
/// "a2" before "a10", and accents sort after their base letter (the fallback is code-point
/// order, where "a10" < "a2" and accents sort far from their base).
#[cfg(feature = "intl")]
#[test]
fn intl_collator_real_uca() {
    assert_eq!(
        out(r#"console.log(new Intl.Collator("en",{numeric:true}).compare("a2","a10"))"#),
        "-1\n"
    );
    assert_eq!(
        out(r#"console.log(["é","a","z","b"].sort(new Intl.Collator("en").compare).join(","))"#),
        "a,b,é,z\n"
    );
}

/// With the `intl` crate, `Intl.NumberFormat` is locale-aware — German currency uses the
/// `1.234,50 €` pattern (comma decimal, trailing symbol), unlike the en-only fallback.
#[cfg(feature = "intl")]
#[test]
fn intl_numberformat_is_locale_aware() {
    assert_eq!(
        out(
            r#"console.log(new Intl.NumberFormat("de-DE",{style:"currency",currency:"EUR"}).format(1234.5))"#
        ),
        "1.234,50\u{a0}€\n"
    );
}

/// With the `intl` crate, `Intl.DateTimeFormat` is locale-aware (German month name +
/// day-month-year order), which the en-only fallback can't produce.
#[cfg(feature = "intl")]
#[test]
fn intl_datetime_is_locale_aware() {
    assert_eq!(
        out(
            r#"console.log(new Intl.DateTimeFormat("de",{timeZone:"UTC",dateStyle:"long"}).format(new Date(Date.UTC(2024,0,15))))"#
        ),
        "15. Januar 2024\n"
    );
}

/// `Intl.DateTimeFormat` field shapes the CLDR `availableFormats` table has no
/// entry for: the flexible day period on its own, and a lone `minute`/`second`/
/// `fractionalSecondDigits`. `intl` 0.5.3 synthesizes a pattern for these rather
/// than falling back to a date pattern and stripping it to nothing.
///
/// A lone `second: "2-digit"` gives `6`, matching V8/node: ECMA-402's format
/// matcher reports back the *chosen pattern's* field width, and the synthesized
/// pattern for a lone second is `s`.
#[cfg(feature = "intl")]
#[test]
fn intl_datetime_lone_field_options() {
    let src = r#"
        const t = new Date(Date.UTC(2024,0,1,2,35,6,789));
        const f = o => new Intl.DateTimeFormat("en", Object.assign({timeZone:"UTC"}, o)).format(t);
        console.log([f({dayPeriod:"long"}), f({minute:"numeric"}), f({second:"2-digit"}),
                     f({fractionalSecondDigits:3}), f({second:"numeric",fractionalSecondDigits:2})].join("|"))
    "#;
    assert_eq!(out(src), "in the morning|35|6|789|6.78\n");
    // CLDR dropped the `midnight` day-period *format* rule; 00:00 is `morning1`.
    assert_eq!(
        out(
            r#"console.log(new Intl.DateTimeFormat("en",{timeZone:"UTC",dayPeriod:"long"}).format(new Date(0)))"#
        ),
        "in the morning\n"
    );
}

/// The proleptic Gregorian calendar has no year 0: the `year` field is
/// era-relative, so ISO year 0 renders as `1 BC` (and `01` two-digit).
#[cfg(feature = "intl")]
#[test]
fn intl_datetime_bce_era_year() {
    let src = r#"
        const d = new Date(-62151602400000);
        const a = new Intl.DateTimeFormat("en-US",{timeZone:"UTC",year:"numeric",era:"short"}).format(d);
        const b = new Intl.DateTimeFormat("en-US",{timeZone:"UTC",year:"2-digit"}).format(d);
        console.log(a + "|" + b)
    "#;
    assert_eq!(out(src), "1 BC|01\n");
}

/// A `formatRange` whose endpoints share a date and differ only in the time of
/// day renders the date once (UTS #35 §2.6.2 date+time interval composition).
#[cfg(feature = "intl")]
#[test]
fn intl_datetime_same_day_range() {
    let src = r#"
        const f = new Intl.DateTimeFormat("en-US",{timeZone:"UTC",year:"numeric",month:"numeric",
            day:"numeric",hour:"numeric",minute:"numeric"});
        console.log(f.formatRange(Date.UTC(2021,7,4,0,30), Date.UTC(2021,7,4,23,30)))
    "#;
    assert_eq!(
        out(src),
        "8/4/2021, 12:30 AM\u{2009}\u{2013}\u{2009}11:30 PM\n"
    );
}

/// The ECMA-402 normative-optional legacy constructor mode: calling
/// `Intl.NumberFormat` as a function on an existing instance stashes the new
/// formatter under `%Intl%.[[FallbackSymbol]]` and returns the receiver, and
/// `resolvedOptions` unwraps it again on a receiver that has no
/// `[[InitializedNumberFormat]]` of its own.
#[cfg(feature = "intl")]
#[test]
fn intl_legacy_constructed_symbol() {
    let src = r#"
        const bare = Object.create(Intl.NumberFormat.prototype);
        const r = Intl.NumberFormat.call(bare, "de");
        const s = Object.getOwnPropertySymbols(r);
        const d = Object.getOwnPropertyDescriptor(r, s[0]);
        // A receiver that *is* already a NumberFormat keeps its own slots.
        const own = new Intl.NumberFormat("fr");
        Intl.NumberFormat.call(own, "de");
        console.log([r === bare, s.length, s[0].description, d.writable, d.enumerable, d.configurable,
                     Intl.NumberFormat.prototype.resolvedOptions.call(r).locale,
                     own.resolvedOptions().locale].join("|"))
    "#;
    assert_eq!(
        out(src),
        "true|1|IntlLegacyConstructedSymbol|false|false|false|de|fr\n"
    );
}

/// With the `intl` crate, `Intl.DisplayNames` / `Intl.ListFormat` are locale-aware (the
/// en-only fallback ignores the locale argument).
#[cfg(feature = "intl")]
#[test]
fn intl_display_and_list_are_locale_aware() {
    assert_eq!(
        out(r#"console.log(new Intl.DisplayNames("de",{type:"region"}).of("US"))"#),
        "Vereinigte Staaten\n"
    );
    assert_eq!(
        out(r#"console.log(new Intl.ListFormat("es").format(["a","b","c"]))"#),
        "a, b y c\n"
    );
}

#[test]
fn variables_assignment_and_control_flow() {
    assert_eq!(run("let x = 1; let y = 2; x + y"), "3");
    assert_eq!(run("let s = 'a'; s += 'b'; s += 'c'; s"), "abc");
    assert_eq!(run("let x = 1; { let x = 99; } x"), "1");
    assert_eq!(
        run("let s = 0; for (let i = 1; i <= 10; i += 1) s += i; s"),
        "55"
    );
    assert_eq!(
        run("let s = 0; for (let i = 0; i < 10; i += 1) { if (i === 5) break; s += i; } s"),
        "10"
    );
}

#[test]
fn functions_and_return() {
    assert_eq!(run("function add(a, b) { return a + b; } add(2, 3)"), "5");
    assert_eq!(run("let sq = function (x) { return x * x; }; sq(7)"), "49");
    assert_eq!(run("let inc = x => x + 1; inc(41)"), "42");
    // Hoisting: callable before its definition.
    assert_eq!(run("f(10); function f(n) { return n; } f(10)"), "10");
}

/// The public D′ API (`Interp::snapshot` / `restore_snapshot`): snapshot a
/// live closure's state in one interpreter and reload it into a *fresh* one
/// holding the same code, through the supported library surface alone — no
/// reaching into interpreter internals.
#[test]
fn public_snapshot_api_round_trips_across_runtimes() {
    let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

    // Runtime A: advance a counter to n = 2, snapshot it to bytes.
    let mut a = Interp::new();
    let f = a.run(&program).expect("exec A");
    assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
    assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
    let bytes = a.snapshot(&[f]);
    drop(a);

    // Runtime B: a fresh interpreter compiles the same program, then reloads
    // A's snapshot and runs the restored closure — resuming from n = 2.
    let mut b = Interp::new();
    let own = b.run(&program).expect("exec B");
    let restored = b.restore_snapshot(&bytes).expect("restore");
    assert_eq!(restored.len(), 1, "one heap root restored");
    assert_eq!(
        b.call(restored[0], &[]).unwrap().as_number(),
        Some(3.0),
        "restored closure resumes from snapshotted state"
    );
    assert_eq!(
        b.call(own, &[]).unwrap().as_number(),
        Some(1.0),
        "the fresh runtime's own counter is independent"
    );

    // A malformed snapshot is rejected, not panicked on.
    assert!(b.restore_snapshot(b"not a snapshot").is_err());
}

/// Cross-runtime D′ reload: snapshot a closure in one runtime, serialize it,
/// then restore and **execute** it in a *separate, fresh* runtime that holds
/// the same code — the load → evict → reload scenario. The restored closure
/// carries the snapshotted captured state and is independent of the fresh
/// runtime's own instance of the program.
#[test]
fn snapshot_reloads_into_a_fresh_runtime() {
    use crate::snapshot::{capture, deserialize, restore, serialize};

    // `makeCounter` (func 0) returns the inner closure (func 1); both runtimes
    // compile the same program, so the snapshot's `func_id`s line up.
    let program = Parser::parse_program(
            "function makeCounter(start){ var n = start; return function(){ return ++n; }; } makeCounter(0)",
        )
        .expect("parse");

    // Runtime A: build a counter, advance it to n = 2, snapshot it to bytes.
    let mut a = Interp::new();
    let f = a.run(&program).expect("exec A");
    assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(1.0));
    assert_eq!(a.call(f, &[]).unwrap().as_number(), Some(2.0));
    let fh = Handle::from_raw(f.as_handle().expect("closure"));
    let bytes = serialize(&capture(&a.realm, &[fh]));
    drop(a); // A is gone — only the bytes survive.

    // Runtime B: a fresh interpreter that compiles the same program (its own
    // counter starts at 0), then reloads A's snapshot and runs the restored
    // closure — which resumes from the snapshotted n = 2.
    let mut b = Interp::new();
    let own = b.run(&program).expect("exec B");
    let snap = deserialize(&bytes).expect("deserialize");
    let restored = restore(&mut b.realm, &snap);
    let f2 = NanBox::handle(restored[0].to_raw());

    assert_eq!(
        b.call(f2, &[]).unwrap().as_number(),
        Some(3.0),
        "restored closure resumes from the snapshotted state in the new runtime"
    );
    assert_eq!(
        b.call(own, &[]).unwrap().as_number(),
        Some(1.0),
        "the fresh runtime's own counter is independent of the reloaded one"
    );
}

/// End-to-end D′: a live closure's captured state survives capture →
/// serialize → deserialize → restore *and remains executable* — the restored
/// closure runs, carries the snapshotted captured value, and is independent of
/// the original. (Same interpreter, so its function table already holds the
/// bodies the snapshot's `func_id`s refer to.)
#[test]
fn snapshot_restores_an_executable_closure() {
    use crate::snapshot::{capture, deserialize, restore, serialize};

    let program = Parser::parse_program(
        "function counter(){ var n = 0; return function(){ return ++n; }; } counter()",
    )
    .expect("parse");
    let mut interp = Interp::new();
    let f = interp.run(&program).expect("exec");
    let fh = Handle::from_raw(f.as_handle().expect("closure handle"));

    // Advance the original to n = 2, then snapshot it there.
    assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(1.0));
    assert_eq!(interp.call(f, &[]).unwrap().as_number(), Some(2.0));
    let snap = capture(&interp.realm, &[fh]);

    // Full on-disk round-trip.
    let snap = deserialize(&serialize(&snap)).expect("deserialize");

    // Restore into the same runtime (whose function table still has the body).
    let restored = restore(&mut interp.realm, &snap);
    let f2 = NanBox::handle(restored[0].to_raw());

    // The original kept counting (n was 2 → 3); the restored closure starts
    // from the *snapshotted* n = 2 → 3, proving it both executes and carries
    // the captured value.
    assert_eq!(
        interp.call(f, &[]).unwrap().as_number(),
        Some(3.0),
        "original continues"
    );
    assert_eq!(
        interp.call(f2, &[]).unwrap().as_number(),
        Some(3.0),
        "restored from snapshot"
    );

    // Independence: advancing the restored copy does not move the original.
    assert_eq!(
        interp.call(f2, &[]).unwrap().as_number(),
        Some(4.0),
        "restored advances"
    );
    assert_eq!(
        interp.call(f, &[]).unwrap().as_number(),
        Some(4.0),
        "original unaffected by restore"
    );
}

#[test]
fn closures_capture_their_scope() {
    // A returned inner function still sees the enclosing variable.
    assert_eq!(
        run(
            "function adder(n) { return function (x) { return x + n; }; }
                 let add5 = adder(5);
                 add5(10)"
        ),
        "15"
    );
    // The capture is by reference: a mutable counter.
    assert_eq!(
        run("function counter() {
                   let c = 0;
                   return function () { c += 1; return c; };
                 }
                 let next = counter();
                 next(); next(); next()"),
        "3"
    );
}

#[test]
fn recursion() {
    assert_eq!(
        run("function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); } fact(5)"),
        "120"
    );
    assert_eq!(
        run("function fib(n) { return n < 2 ? n : fib(n-1) + fib(n-2); } fib(10)"),
        "55"
    );
}

#[test]
fn higher_order_and_objects() {
    // A function stored on an object, called, mutating closed-over state.
    assert_eq!(
        run("function makeBox(v) {
                   let value = v;
                   return { get: function () { return value; },
                            set: function (x) { value = x; } };
                 }
                 let b = makeBox(1);
                 b.set(99);
                 b.get()"),
        "99"
    );
}

#[test]
fn calling_a_non_function_errors() {
    // Calling a non-callable is a *catchable* JS `TypeError` (an
    // `ExecError::Throw`), not an internal `NotCallable` — so user `try/catch`
    // can handle it (per ECMA-262 Call, step 2).
    let program = Parser::parse_program("let x = 5; x()").unwrap();
    let mut interp = Interp::new();
    assert!(matches!(interp.run(&program), Err(ExecError::Throw(_))));
}

#[test]
fn try_catch_finally() {
    // A thrown value is caught and bound.
    assert_eq!(
        run("let r; try { throw 'boom'; r = 'no'; } catch (e) { r = 'caught:' + e; } r"),
        "caught:boom"
    );
    // No throw: the catch is skipped.
    assert_eq!(
        run("let r = 'ok'; try { r = 'a'; } catch (e) { r = 'b'; } r"),
        "a"
    );
    // finally always runs.
    assert_eq!(
        run(
            "let log = ''; try { log += '1'; throw 0; } catch (e) { log += '2'; } finally { log += '3'; } log"
        ),
        "123"
    );
    // A throw out of a function is caught at the call site.
    assert_eq!(
        run("function boom() { throw 'x'; }
                 let r; try { boom(); } catch (e) { r = 'got:' + e; } r"),
        "got:x"
    );
    // catch without a binding.
    assert_eq!(
        run("let r = 'a'; try { throw 1; } catch { r = 'b'; } r"),
        "b"
    );
}

#[test]
fn uncaught_throw_propagates() {
    let program = Parser::parse_program("throw 'oops'").unwrap();
    let mut interp = Interp::new();
    match interp.run(&program) {
        Err(ExecError::Throw(v)) => {
            assert_eq!(interp.realm().to_display_string(v), "oops");
        }
        other => panic!("expected a throw, got {other:?}"),
    }
}

#[test]
fn finally_return_overrides() {
    // A `return` in finally overrides the try's outcome.
    assert_eq!(
        run("function f() { try { return 'a'; } finally { return 'b'; } } f()"),
        "b"
    );
}

#[test]
fn builtin_functions() {
    // Math methods (variadic).
    assert_eq!(run("Math.max(3, 7, 2)"), "7");
    assert_eq!(run("Math.min(3, 7, 2)"), "2");
    assert_eq!(run("Math.abs(-5)"), "5");
    // Coercion globals.
    assert_eq!(run("String(42)"), "42");
    assert_eq!(run("Number('3.5')"), "3.5");
    assert_eq!(run("Boolean(0)"), "false");
    assert_eq!(run("Boolean('x')"), "true");
    assert_eq!(run("parseInt('42px')"), "42");
    assert_eq!(run("parseInt('  -7 ')"), "-7");
    // typeof a built-in is "function".
    assert_eq!(run("typeof Math.max"), "function");
    // Built-ins compose with user code.
    assert_eq!(
        run(
            "function clamp(x, lo, hi) { return Math.max(lo, Math.min(x, hi)); }
                 clamp(15, 0, 10)"
        ),
        "10"
    );
}

#[test]
fn string_methods() {
    assert_eq!(run("'hello'.toUpperCase()"), "HELLO");
    assert_eq!(run("'HELLO'.toLowerCase()"), "hello");
    assert_eq!(run("'  hi  '.trim()"), "hi");
    assert_eq!(run("'hello'.charAt(1)"), "e");
    assert_eq!(run("'hello'.includes('ell')"), "true");
    assert_eq!(run("'hello'.indexOf('l')"), "2");
    assert_eq!(run("'ab'.repeat(3)"), "ababab");
}

#[test]
fn array_methods() {
    assert_eq!(run("let a = [1, 2]; a.push(3); a.join('-')"), "1-2-3");
    assert_eq!(run("let a = [1, 2, 3]; a.pop()"), "3");
    assert_eq!(run("[1, 2, 3].includes(2)"), "true");
    assert_eq!(run("[1, 2, 3].indexOf(3)"), "2");
    assert_eq!(run("['a', 'b', 'c'].join(', ')"), "a, b, c");
    // splice (remove + insert), unshift, shift.
    assert_eq!(
        run("let a=[1,2,3,4]; a.splice(1,2,'x'); a.join(',')"),
        "1,x,4"
    );
    assert_eq!(run("[1,2,3,4].splice(1,2).join(',')"), "2,3");
    assert_eq!(run("let a=[2,3]; a.unshift(0,1); a.join(',')"), "0,1,2,3");
    assert_eq!(
        run("let a=[1,2,3]; let f=a.shift(); f + ':' + a.join(',')"),
        "1:2,3"
    );
    // Non-mutating: toSorted/toReversed/with leave the original.
    assert_eq!(
        run(
            "let a=[3,1,2]; let s=a.toSorted(function(x,y){return x-y;}); s.join('')+'|'+a.join('')"
        ),
        "123|312"
    );
    assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
    assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
    // includes uses SameValueZero (NaN matches).
    assert_eq!(run("[NaN, 1].includes(NaN)"), "true");
    assert_eq!(run("[1, 2].includes(NaN)"), "false");
}

#[test]
fn define_property_and_locale_compare() {
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'x',{value:42}); o.x"),
        "42"
    );
    assert_eq!(
        run("let o={n:1}; Object.defineProperty(o,'d',{get:function(){return this.n+1;}}); o.d"),
        "2"
    );
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:7}); Object.getOwnPropertyDescriptor(o,'x').value"
        ),
        "7"
    );
    assert_eq!(run("'apple'.localeCompare('banana') < 0"), "true");
    assert_eq!(run("'x'.localeCompare('x')"), "0");
    // fill: whole array, a [start,end) range, and a negative start.
    assert_eq!(run("[0, 0, 0].fill(7).join(',')"), "7,7,7");
    assert_eq!(run("[1, 2, 3, 4].fill(9, 1, 3).join(',')"), "1,9,9,4");
    assert_eq!(run("[1, 2, 3, 4, 5].fill(0, -2).join(',')"), "1,2,3,0,0");
    // reduceRight folds right-to-left, with and without a seed.
    assert_eq!(
        run("['a','b','c'].reduceRight(function(acc,x){ return acc + x; })"),
        "cba"
    );
    assert_eq!(
        run("[1,2,3].reduceRight(function(a,x){ return a + x; }, 10)"),
        "16"
    );
    // findLast / findLastIndex scan right-to-left.
    assert_eq!(
        run("[1,2,3,4].findLast(function(x){ return x % 2 === 1; })"),
        "3"
    );
    assert_eq!(
        run("[1,2,3,4].findLastIndex(function(x){ return x % 2 === 1; })"),
        "2"
    );
    assert_eq!(
        run("[2,4,6].findLast(function(x){ return x > 9; })"),
        "undefined"
    );
    // copyWithin (in place) and flat(depth).
    assert_eq!(run("[1,2,3,4,5].copyWithin(0,3).join(',')"), "4,5,3,4,5");
    assert_eq!(run("[1,[2,[3,[4]]]].flat(2).join(',')"), "1,2,3,4");
}

#[test]
fn math_extras_and_number_coercion() {
    assert_eq!(run("Math.hypot(3, 4)"), "5");
    assert_eq!(run("Math.cbrt(27)"), "3");
    assert_eq!(run("Math.log2(8)"), "3");
    assert_eq!(run("Math.log10(1000)"), "3");
    assert_eq!(run("(1234.5).toExponential(2)"), "1.23e+3");
    // Radix-prefixed string coercion (shared `to_number`, both engines).
    assert_eq!(run("Number('0x1F')"), "31");
    assert_eq!(run("+'0b101'"), "5");
    assert_eq!(run("'0o17' * 1"), "15");
}

#[test]
fn split_limit_and_to_precision() {
    assert_eq!(run("'a,b,c,d'.split(',', 2).length"), "2");
    assert_eq!(run("'aXbXc'.split('X').join('-')"), "a-b-c");
    assert_eq!(run("(123.456).toPrecision(4)"), "123.5");
    assert_eq!(run("(255).toString(2)"), "11111111");
}

#[test]
fn object_freeze_family() {
    // Writes and new properties are no-ops on a frozen object.
    assert_eq!(
        run("let o = Object.freeze({ a: 1 }); o.a = 9; o.b = 2; o.a + ',' + (o.b === undefined)"),
        "1,true"
    );
    assert_eq!(run("Object.isFrozen(Object.freeze({}))"), "true");
    assert_eq!(run("Object.isFrozen({})"), "false");
    assert_eq!(
        run("Object.getOwnPropertyNames({ x: 1, y: 2 }).length"),
        "2"
    );
}

#[test]
fn string_pad_lastindexof_and_number_statics() {
    assert_eq!(run("'5'.padEnd(3, '-')"), "5--");
    assert_eq!(run("'ab'.padEnd(5)"), "ab   ");
    assert_eq!(run("'a-b-c'.lastIndexOf('-')"), "3");
    assert_eq!(run("'abc'.lastIndexOf('x')"), "-1");
    assert_eq!(run("Number.MAX_SAFE_INTEGER"), "9007199254740991");
    assert_eq!(run("Number.POSITIVE_INFINITY"), "Infinity");
    assert_eq!(run("'abc'.concat('def', '!')"), "abcdef!");
}

#[test]
fn console_log_captures_output() {
    let program =
        Parser::parse_program("console.log('hi', 42); let x = [1, 2]; console.log('arr:', x);")
            .unwrap();
    let mut interp = Interp::new();
    interp.run(&program).unwrap();
    assert_eq!(interp.output(), "hi 42\narr: 1,2\n");
}

#[test]
fn json_getters_tojson_and_date_utc() {
    // JSON.stringify invokes getters and honors toJSON.
    assert_eq!(
        run("JSON.stringify({a:1, get b(){ return 2; }})"),
        "{\"a\":1,\"b\":2}"
    );
    assert_eq!(
        run("JSON.stringify({v:42, toJSON(){ return {w:this.v}; }})"),
        "{\"w\":42}"
    );
    assert_eq!(run("JSON.stringify([undefined,1])"), "[null,1]");
    // Date.UTC and getUTC* methods.
    assert_eq!(run("new Date(Date.UTC(2024,0,15)).getUTCDate()"), "15");
    assert_eq!(run("new Date(Date.UTC(2024,0,1)).getUTCDay()"), "1"); // Monday
    assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
}

#[test]
fn date_invalid_toisostring() {
    assert_eq!(
        run("try{new Date(NaN).toISOString();'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    assert_eq!(run("new Date(NaN).toJSON()"), "null");
    assert_eq!(run("JSON.stringify({d:new Date(NaN)})"), "{\"d\":null}");
    assert_eq!(
        run("new Date(Date.UTC(2020,5,15,10,30,45,123)).toISOString()"),
        "2020-06-15T10:30:45.123Z"
    );
    assert_eq!(
        run("try{new Date('garbage').toISOString();'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
}

#[test]
fn objects_inherit_object_prototype() {
    assert_eq!(run("'toString' in {}"), "true");
    assert_eq!(run("'hasOwnProperty' in {}"), "true");
    assert_eq!(
        run("Object.getPrototypeOf({}) === Object.prototype"),
        "true"
    );
    assert_eq!(run("'toString' in Object.create(null)"), "false");
    assert_eq!(
        run("Object.getPrototypeOf(Object.create(null)) === null"),
        "true"
    );
    // Inherited methods are non-enumerable.
    assert_eq!(run("Object.keys({a:1,b:2}).join(',')"), "a,b");
    assert_eq!(
        run("let s=[]; for(let k in {a:1,b:2}) s.push(k); s.join(',')"),
        "a,b"
    );
    assert_eq!(run("JSON.stringify({a:1})"), "{\"a\":1}");
    // hasOwnProperty distinguishes own vs inherited.
    assert_eq!(
        run(
            "let c=Object.create({i:1}); c.o=2; c.hasOwnProperty('o') + ',' + c.hasOwnProperty('i')"
        ),
        "true,false"
    );
}

#[test]
fn object_prototype_tostring_call() {
    assert_eq!(run("typeof Object.prototype"), "object");
    assert_eq!(run("Object.prototype.toString.call({})"), "[object Object]");
    assert_eq!(run("Object.prototype.toString.call([])"), "[object Array]");
    assert_eq!(run("Object.prototype.toString.call(null)"), "[object Null]");
    assert_eq!(
        run("Object.prototype.toString.call(undefined)"),
        "[object Undefined]"
    );
    assert_eq!(
        run("Object.prototype.toString.call(function(){})"),
        "[object Function]"
    );
    assert_eq!(
        run("Object.prototype.toString.call(/x/)"),
        "[object RegExp]"
    );
    assert_eq!(
        run("Object.prototype.toString.call({[Symbol.toStringTag]:'Widget'})"),
        "[object Widget]"
    );
    assert_eq!(
        run("Object.prototype.hasOwnProperty.call({a:1},'a')"),
        "true"
    );
    assert_eq!(
        run("Object.prototype.hasOwnProperty.call({a:1},'b')"),
        "false"
    );
    assert_eq!(
        run("let p={}; Object.prototype.isPrototypeOf.call(p, Object.create(p))"),
        "true"
    );
    assert_eq!(
        run("let o={x:1}; Object.prototype.valueOf.call(o)===o"),
        "true"
    );
}

#[test]
fn webassembly_instantiate_and_call() {
    // `WebAssembly.instantiate` now returns a Promise (per spec); use the
    // synchronous `new Instance(new Module(...))` path to read exports directly.
    let setup = "var b=[0,0x61,0x73,0x6d,1,0,0,0, 1,7,1,0x60,2,0x7f,0x7f,1,0x7f, 3,2,1,0, 7,7,1,3,0x61,0x64,0x64,0,0, 0xa,9,1,7,0,0x20,0,0x20,1,0x6a,0xb]; var e=new WebAssembly.Instance(new WebAssembly.Module(b)).exports; ";
    assert_eq!(run(&alloc::format!("{setup} typeof e.add")), "function");
    assert_eq!(run(&alloc::format!("{setup} e.add(20,22)")), "42");
    assert_eq!(run(&alloc::format!("{setup} e.add(-5,8)")), "3");
}

#[test]
fn webassembly_validate_builtin() {
    assert_eq!(run("typeof WebAssembly"), "object");
    assert_eq!(run("typeof WebAssembly.validate"), "function");
    assert_eq!(
        run("WebAssembly.validate([0,0x61,0x73,0x6d,1,0,0,0])"),
        "true"
    );
    assert_eq!(run("WebAssembly.validate([0,0,0,0,1,0,0,0])"), "false");
    assert_eq!(run("WebAssembly.validate([0,0x61])"), "false");
}

#[test]
fn object_default_tostring_and_tag() {
    assert_eq!(run("({}).toString()"), "[object Object]");
    assert_eq!(run("'abc'.toString()"), "abc");
    assert_eq!(run("({a:1}).valueOf().a"), "1");
    assert_eq!(run("({toString(){return 'custom';}}).toString()"), "custom");
    assert_eq!(
        run("Object.create({toString(){return 'base';}}).toString()"),
        "base"
    );
    assert_eq!(
        run("({[Symbol.toStringTag]:'Widget'}).toString()"),
        "[object Widget]"
    );
    assert_eq!(run("typeof Symbol.toStringTag"), "symbol");
    assert_eq!(run("typeof Symbol.species"), "symbol");
    assert_eq!(
        run("let o={[Symbol.toStringTag]:'X'}; Object.getOwnPropertySymbols(o).length"),
        "1"
    );
    // Existing toStrings unaffected.
    assert_eq!(run("[1,2,3].toString()"), "1,2,3");
    assert_eq!(run("new Error('x').toString()"), "Error: x");
}

#[test]
fn strict_getter_only_write() {
    // Sloppy: silently ignored.
    assert_eq!(run("let o={get x(){return 1;}}; o.x=2; o.x"), "1");
    // Strict: throws TypeError.
    assert_eq!(
        run(
            "(function(){'use strict'; let o={get x(){return 1;}}; try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
        ),
        "te"
    );
    // A setter still works under strict mode.
    assert_eq!(
        run(
            "(function(){'use strict'; let o={_v:0,get x(){return this._v;},set x(v){this._v=v*2;}}; o.x=5; return o.x;})()"
        ),
        "10"
    );
    // Inherited getter-only accessor.
    assert_eq!(
        run(
            "(function(){'use strict'; let o=Object.create({get y(){return 9;}}); try{o.y=1;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
        ),
        "te"
    );
}

#[test]
fn reduce_empty_throws_typeerror() {
    assert_eq!(
        run("try{[].reduce((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
    assert_eq!(
        run("try{[].reduceRight((a,b)=>a+b);'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
    assert_eq!(run("[].reduce((a,b)=>a+b, 99)"), "99");
    assert_eq!(run("[42].reduce((a,b)=>a+b)"), "42");
    assert_eq!(run("[1,2,3,4].reduce((a,b)=>a+b)"), "10");
}

#[test]
fn function_names_tostring_bound() {
    assert_eq!(run("let myFn=function(){}; myFn.name"), "myFn");
    assert_eq!(run("let arrow=()=>1; arrow.name"), "arrow");
    assert_eq!(run("let x=function inner(){}; x.name"), "inner");
    assert_eq!(
        run("let o={method(){},fn:()=>1}; o.method.name + ':' + o.fn.name"),
        "method:fn"
    );
    assert_eq!(run("typeof (function f(){}).toString()"), "string");
    assert_eq!(
        run("(function f(){}).toString().indexOf('function')>=0"),
        "true"
    );
    assert_eq!(run("function t(a,b,c){} t.bind(null,1).length"), "2");
    assert_eq!(run("function t(a,b,c){} t.bind(null,1,2,3,4).length"), "0");
    assert_eq!(run("function t(){} t.bind(null).name"), "bound t");
}

#[test]
fn proto_accessor() {
    // Object.create + __proto__ read.
    assert_eq!(
        run("let p={greet(){return 'hi';}}; let o=Object.create(p); o.__proto__===p"),
        "true"
    );
    assert_eq!(
        run("let p={greet(){return 'hi';}}; Object.create(p).greet()"),
        "hi"
    );
    // __proto__ assignment relinks.
    assert_eq!(
        run("let o={}; o.__proto__={hello(){return 'yo';}}; o.hello()"),
        "yo"
    );
    assert_eq!(
        run("let b={getX(){return this.x;}}; let o={}; o.__proto__=b; o.x=42; o.getX()"),
        "42"
    );
    // Object-literal __proto__ sets the prototype.
    assert_eq!(
        run("let b={getX(){return this.x;}}; let n={__proto__:b, x:5}; n.getX()"),
        "5"
    );
    // The method form is a regular property, not a prototype set.
    assert_eq!(
        run("typeof ({__proto__(){return 1;}}).__proto__"),
        "function"
    );
    // __proto__ = null clears the chain; a primitive is ignored.
    assert_eq!(
        run("let o={__proto__:{}}; o.__proto__=null; Object.getPrototypeOf(o)===null"),
        "true"
    );
    assert_eq!(
        run("let p={g(){return 1;}}; let o=Object.create(p); o.__proto__=5; o.g()"),
        "1"
    );
}

#[test]
fn proxy_passthrough_keys() {
    assert_eq!(run("Object.keys(new Proxy({a:1,b:2},{})).join(',')"), "a,b");
    assert_eq!(
        run("Object.values(new Proxy({a:1,b:2},{})).join(',')"),
        "1,2"
    );
    assert_eq!(
        run("Object.entries(new Proxy({a:1,b:2},{})).map(e=>e.join(':')).join(',')"),
        "a:1,b:2"
    );
    // Nested trap-less proxies forward through.
    assert_eq!(
        run("Object.keys(new Proxy(new Proxy({x:1},{}),{})).join(',')"),
        "x"
    );
}

#[test]
fn bigint_as_uintn_intn() {
    assert_eq!(run("BigInt.asUintN(8, 256n)"), "0");
    assert_eq!(run("BigInt.asUintN(8, -1n)"), "255");
    assert_eq!(run("BigInt.asUintN(4, -1n)"), "15");
    assert_eq!(run("BigInt.asUintN(64, 18446744073709551617n)"), "1");
    assert_eq!(run("BigInt.asIntN(8, 200n)"), "-56");
    assert_eq!(run("BigInt.asIntN(8, 128n)"), "-128");
    assert_eq!(run("BigInt.asIntN(8, 127n)"), "127");
    assert_eq!(run("BigInt.asIntN(16, 40000n)"), "-25536");
    assert_eq!(run("BigInt.asIntN(32, 4294967295n)"), "-1");
    assert_eq!(
        run("BigInt.asIntN(128, 12345678901234567890n)"),
        "12345678901234567890"
    );
}

#[test]
fn json_number_serialization() {
    assert_eq!(run("JSON.stringify(-0)"), "0");
    assert_eq!(run("JSON.stringify([-0])"), "[0]");
    assert_eq!(run("JSON.stringify({x:-0})"), "{\"x\":0}");
    assert_eq!(run("JSON.stringify(1e21)"), "1e+21");
    assert_eq!(run("JSON.stringify(1e-7)"), "1e-7");
    assert_eq!(run("JSON.stringify(1e20)"), "100000000000000000000");
    assert_eq!(run("JSON.stringify(0.001)"), "0.001");
    assert_eq!(run("JSON.stringify([NaN,Infinity])"), "[null,null]");
    assert_eq!(
        run("JSON.stringify([-0,1e21,0.001,-42])"),
        "[0,1e+21,0.001,-42]"
    );
}

#[test]
fn json_stringify_replacer() {
    // Function replacer: omit keys and transform values (recursively).
    assert_eq!(
        run("JSON.stringify({a:1,b:2}, function(k,v){ return k==='b'?undefined:v; })"),
        "{\"a\":1}"
    );
    assert_eq!(
        run("JSON.stringify({x:{n:1}}, function(k,v){ return typeof v==='number'?v+5:v; })"),
        "{\"x\":{\"n\":6}}"
    );
    // Array replacer: an allowlist applied at every level.
    assert_eq!(
        run("JSON.stringify({a:1,b:2,c:3}, ['a','c'])"),
        "{\"a\":1,\"c\":3}"
    );
    assert_eq!(
        run("JSON.stringify({keep:{a:1,b:2},drop:9}, ['keep','a'])"),
        "{\"keep\":{\"a\":1}}"
    );
}

#[test]
fn json_parse_reviver() {
    assert_eq!(
        run(
            "let o = JSON.parse('{\"a\":1,\"b\":2}', function(k,v){ return typeof v==='number'?v*2:v; }); o.a + ',' + o.b"
        ),
        "2,4"
    );
    assert_eq!(
        run(
            "let o = JSON.parse('{\"keep\":1,\"drop\":2}', function(k,v){ return k==='drop'?undefined:v; }); o.keep + ':' + ('drop' in o)"
        ),
        "1:false"
    );
    assert_eq!(
        run("JSON.parse('[1,2,3]', function(k,v){ return typeof v==='number'?v+10:v; }).join(',')"),
        "11,12,13"
    );
}

#[test]
fn json_stringify() {
    assert_eq!(run("JSON.stringify(42)"), "42");
    assert_eq!(run("JSON.stringify('hi')"), "\"hi\"");
    assert_eq!(run("JSON.stringify(true)"), "true");
    assert_eq!(run("JSON.stringify(null)"), "null");
    assert_eq!(run("JSON.stringify([1, 2, 3])"), "[1,2,3]");
    assert_eq!(
        run("JSON.stringify({ a: 1, b: 'x' })"),
        "{\"a\":1,\"b\":\"x\"}"
    );
    assert_eq!(
        run("JSON.stringify({ nested: { list: [1, true, null] } })"),
        "{\"nested\":{\"list\":[1,true,null]}}"
    );
    // Indentation (the `space` argument): numeric and string, empties inline.
    assert_eq!(
        run("JSON.stringify({a:1,b:2}, null, 2)"),
        "{\n  \"a\": 1,\n  \"b\": 2\n}"
    );
    assert_eq!(run("JSON.stringify([1,2], null, '--')"), "[\n--1,\n--2\n]");
    assert_eq!(run("JSON.stringify({}, null, 2)"), "{}");
    assert_eq!(run("JSON.stringify([], null, 4)"), "[]");
    // A quote in a string is escaped.
    assert_eq!(run("JSON.stringify('a\"b')"), "\"a\\\"b\"");
}

#[test]
fn object_and_array_statics() {
    assert_eq!(run("Object.keys({ a: 1, b: 2 }).join(',')"), "a,b");
    assert_eq!(run("Object.values({ a: 1, b: 2 }).join(',')"), "1,2");
    assert_eq!(run("Array.isArray([1, 2])"), "true");
    assert_eq!(run("Array.isArray('nope')"), "false");
    assert_eq!(run("Array.isArray({})"), "false");
}

#[cfg(feature = "std")]
#[test]
fn math_float_methods() {
    assert_eq!(run("Math.floor(3.7)"), "3");
    assert_eq!(run("Math.ceil(3.2)"), "4");
    assert_eq!(run("Math.round(3.5)"), "4");
    assert_eq!(run("Math.sqrt(144)"), "12");
}

#[test]
fn classes_and_this() {
    // A class with a constructor and a method using `this`.
    assert_eq!(
        run("class Point {
                   constructor(x, y) { this.x = x; this.y = y; }
                   sum() { return this.x + this.y; }
                 }
                 let p = new Point(3, 4);
                 p.sum()"),
        "7"
    );
    // Methods mutate instance state via `this`.
    assert_eq!(
        run("class Counter {
                   constructor() { this.n = 0; }
                   inc() { this.n += 1; return this.n; }
                 }
                 let c = new Counter();
                 c.inc(); c.inc(); c.inc()"),
        "3"
    );
    // A field initializer.
    assert_eq!(
        run("class Box { value = 42; get() { return this.value; } }
                 new Box().get()"),
        "42"
    );
    // A method calling another method on `this`.
    assert_eq!(
        run("class Calc {
                   constructor(v) { this.v = v; }
                   double() { return this.v * 2; }
                   quadruple() { return this.double() * 2; }
                 }
                 new Calc(5).quadruple()"),
        "20"
    );
    // typeof a class is function.
    assert_eq!(run("class A {} typeof A"), "function");
    // A class expression.
    assert_eq!(
        run("let C = class { constructor() { this.k = 9; } }; new C().k"),
        "9"
    );
}

#[test]
fn class_getters_setters_statics() {
    // A getter computes from instance state.
    assert_eq!(
        run("class C { constructor(w, h) { this.w = w; this.h = h; }
                   get area() { return this.w * this.h; } }
                 new C(3, 4).area"),
        "12"
    );
    // A setter mutates instance state.
    assert_eq!(
        run("class Temp {
                   constructor() { this.c = 0; }
                   get fahrenheit() { return this.c * 9 / 5 + 32; }
                   set fahrenheit(f) { this.c = (f - 32) * 5 / 9; }
                 }
                 let t = new Temp(); t.fahrenheit = 212; t.c"),
        "100"
    );
    // Static methods and fields.
    assert_eq!(
        run(
            "class MathX { static square(n) { return n * n; } static pi = 3; }
                 MathX.square(5) + MathX.pi"
        ),
        "28"
    );
    // A static factory returning an instance.
    assert_eq!(
        run("class Point {
                   constructor(x) { this.x = x; }
                   static origin() { return new Point(0); }
                 }
                 Point.origin().x"),
        "0"
    );
}

#[test]
fn class_inheritance() {
    // A subclass inherits a base method.
    assert_eq!(
        run("class Animal {
                   constructor(name) { this.name = name; }
                   describe() { return this.name; }
                 }
                 class Dog extends Animal {}
                 new Dog('Rex').describe()"),
        "Rex"
    );
    // `super(...)` calls the base constructor; the derived adds state.
    assert_eq!(
        run("class Animal {
                   constructor(name) { this.name = name; }
                 }
                 class Dog extends Animal {
                   constructor(name, breed) { super(name); this.breed = breed; }
                   tag() { return this.name + ':' + this.breed; }
                 }
                 new Dog('Rex', 'Lab').tag()"),
        "Rex:Lab"
    );
    // A derived method overrides the base.
    assert_eq!(
        run("class A { kind() { return 'A'; } }
                 class B extends A { kind() { return 'B'; } }
                 new B().kind() + new A().kind()"),
        "BA"
    );
    // Implicit super (no derived constructor) forwards the args.
    assert_eq!(
        run("class Base { constructor(v) { this.v = v; } }
                 class Sub extends Base { get() { return this.v; } }
                 new Sub(7).get()"),
        "7"
    );
    // Three-level chain.
    assert_eq!(
        run("class A { constructor() { this.a = 1; } }
                 class B extends A { constructor() { super(); this.b = 2; } }
                 class C extends B { constructor() { super(); this.c = 3; } }
                 let o = new C(); o.a + o.b + o.c"),
        "6"
    );
}

#[test]
fn object_array_statics_and_number_methods() {
    // Object.assign / entries.
    assert_eq!(
        run("let t = Object.assign({}, { a: 1 }, { b: 2 }); t.a + t.b"),
        "3"
    );
    assert_eq!(
        run("Object.entries({ a: 1, b: 2 }).map(e => e[0] + '=' + e[1]).join(',')"),
        "a=1,b=2"
    );
    // Array.from (string / array) and Array.of.
    assert_eq!(run("Array.from('abc').join('-')"), "a-b-c");
    assert_eq!(
        run("Array.from([1, 2, 3]).map(x => x * 2).join(',')"),
        "2,4,6"
    );
    assert_eq!(run("Array.of(1, 2, 3).join(',')"), "1,2,3");
    // Number methods.
    assert_eq!(run("(255).toString()"), "255");
}

#[cfg(feature = "std")]
#[test]
fn number_tofixed() {
    assert_eq!(run("(3.14159).toFixed(2)"), "3.14");
    assert_eq!(run("(1).toFixed(3)"), "1.000");
}

#[test]
fn promises_and_microtasks() {
    // The `then` reactions run on the microtask queue, drained after the
    // script — observed via captured `console.log` output.
    let out = |src: &str| {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    };
    // then runs after the synchronous code.
    assert_eq!(
        out("console.log('sync');
                 Promise.resolve(42).then(v => console.log('got:' + v));"),
        "sync\ngot:42\n"
    );
    // Chaining transforms the value through each then.
    assert_eq!(
        out("Promise.resolve(1).then(v => v + 1).then(v => v * 10).then(v => console.log(v));"),
        "20\n"
    );
    // catch handles a rejection.
    assert_eq!(
        out("Promise.reject('boom').catch(e => console.log('caught:' + e));"),
        "caught:boom\n"
    );
    // A throw in a then handler rejects the chain to the next catch.
    assert_eq!(
        out("Promise.resolve(1).then(() => { throw 'x'; }).catch(e => console.log('rej:' + e));"),
        "rej:x\n"
    );
    // `new Promise(executor)` with resolve.
    assert_eq!(
        out("new Promise((resolve) => { resolve(7); }).then(v => console.log(v));"),
        "7\n"
    );
    // Adoption: resolving with a promise chains its value.
    assert_eq!(
        out("Promise.resolve(Promise.resolve(99)).then(v => console.log(v));"),
        "99\n"
    );
    // typeof a promise is object.
    assert_eq!(run("typeof Promise.resolve(1)"), "object");
}

#[test]
fn async_await() {
    // observe results via console.log after the microtask drain.
    let out = |src: &str| {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    };
    // An async function returns a promise; awaiting unwraps values.
    assert_eq!(
        out(
            "async function f() { let x = await Promise.resolve(10); return x + 5; }
                 f().then(v => console.log(v));"
        ),
        "15\n"
    );
    // Awaiting in sequence.
    assert_eq!(
        out("async function g() {
                   let a = await Promise.resolve(2);
                   let b = await Promise.resolve(3);
                   return a * b;
                 }
                 g().then(v => console.log(v));"),
        "6\n"
    );
    // try/catch around a rejected await.
    assert_eq!(
        out("async function h() {
                   try { await Promise.reject('boom'); return 'no'; }
                   catch (e) { return 'caught:' + e; }
                 }
                 h().then(v => console.log(v));"),
        "caught:boom\n"
    );
    // An async arrow, awaiting a plain value.
    assert_eq!(
        out("let f = async (x) => (await x) + 1;
                 f(41).then(v => console.log(v));"),
        "42\n"
    );
    // typeof an async function is function; its call returns a promise.
    assert_eq!(run("async function a() {} typeof a"), "function");
    assert_eq!(run("async function a() { return 1; } typeof a()"), "object");
}

#[cfg(feature = "regex")]
#[test]
fn regexp() {
    // Regex literal + test.
    assert_eq!(run("/ab+c/.test('xxabbbcyy')"), "true");
    assert_eq!(run("/^\\d+$/.test('12345')"), "true");
    assert_eq!(run("/^\\d+$/.test('12a45')"), "false");
    // case-insensitive flag.
    assert_eq!(run("/hello/i.test('HELLO')"), "true");
    // new RegExp(...).
    assert_eq!(run("new RegExp('a.c').test('axc')"), "true");
    // exec returns the matched substring (or null).
    assert_eq!(run("/b+/.exec('aabbbc')[0]"), "bbb");
    assert_eq!(run("/zzz/.exec('abc') === null"), "true");
    // A regex renders as /source/flags; typeof is object.
    assert_eq!(run("'' + /ab/gi"), "/ab/gi");
    assert_eq!(run("typeof /x/"), "object");
}

#[test]
fn json_parse() {
    // Scalars.
    assert_eq!(run("JSON.parse('42')"), "42");
    assert_eq!(run("JSON.parse('true')"), "true");
    assert_eq!(run("JSON.parse('null') === null"), "true");
    assert_eq!(run("JSON.parse('\"hi\\\\nthere\"')"), "hi\nthere");
    // Arrays and objects.
    assert_eq!(run("JSON.parse('[1, 2, 3]').length"), "3");
    assert_eq!(run("JSON.parse('[1, 2, 3]')[1]"), "2");
    assert_eq!(run("JSON.parse('{\"a\": 1, \"b\": 2}').b"), "2");
    // Nested.
    assert_eq!(
        run("JSON.parse('{\"items\": [{\"id\": 7}, {\"id\": 9}]}').items[1].id"),
        "9"
    );
    // Round-trip with stringify.
    assert_eq!(
        run("let o = JSON.parse('{\"x\": 10, \"y\": 20}'); JSON.stringify(o)"),
        "{\"x\":10,\"y\":20}"
    );
    // Negative / float numbers.
    assert_eq!(run("JSON.parse('-3.5')"), "-3.5");
    // Malformed input throws (caught).
    assert_eq!(
        run("try { JSON.parse('{bad}'); 'no'; } catch (e) { 'threw'; }"),
        "threw"
    );
}

#[test]
fn dates() {
    // A fixed timestamp (2021-06-15T12:30:45.500Z = 1623760245500 ms).
    let ts = "1623760245500";
    assert_eq!(
        run(&alloc::format!("new Date({ts}).getTime()")),
        "1623760245500"
    );
    assert_eq!(run(&alloc::format!("new Date({ts}).getFullYear()")), "2021");
    assert_eq!(run(&alloc::format!("new Date({ts}).getMonth()")), "5"); // June, 0-based
    assert_eq!(run(&alloc::format!("new Date({ts}).getDate()")), "15");
    assert_eq!(run(&alloc::format!("new Date({ts}).getHours()")), "12");
    assert_eq!(run(&alloc::format!("new Date({ts}).getMinutes()")), "30");
    assert_eq!(run(&alloc::format!("new Date({ts}).getSeconds()")), "45");
    assert_eq!(run(&alloc::format!("new Date({ts}).getDay()")), "2"); // Tuesday
    assert_eq!(
        run(&alloc::format!("new Date({ts}).toISOString()")),
        "2021-06-15T12:30:45.500Z"
    );
    // The epoch.
    assert_eq!(run("new Date(0).toISOString()"), "1970-01-01T00:00:00.000Z");
    assert_eq!(run("new Date(0).getDay()"), "4"); // Thursday
    // typeof a date is object.
    assert_eq!(run("typeof new Date(0)"), "object");
}

#[test]
fn eval_source_entry_point() {
    // Captured console output + completion value.
    let (out, completion) = eval_source("console.log('hi'); 1 + 2").expect("ok");
    assert_eq!(out, "hi\n");
    assert_eq!(completion, "3");
    // A program with no trailing expression yields `undefined`.
    let (_, c) = eval_source("let x = 5;").expect("ok");
    assert_eq!(c, "undefined");
    // A thrown error surfaces as an Err.
    assert!(eval_source("throw 'boom'").is_err());
    // A parse error surfaces as an Err.
    assert!(eval_source("let = ;").is_err());
}

#[test]
fn object_is_and_safe_integer_and_computed_methods() {
    assert_eq!(run("Object.is(NaN, NaN)"), "true");
    assert_eq!(run("Object.is(0, -0)"), "false");
    assert_eq!(run("Object.is(-0, -0)"), "true");
    assert_eq!(
        run("Object.is('a', 'a') + ':' + Object.is(1, 2)"),
        "true:false"
    );
    assert_eq!(run("Number.isSafeInteger(9007199254740991)"), "true");
    assert_eq!(run("Number.isSafeInteger(9007199254740992)"), "false");
    assert_eq!(run("Number.isSafeInteger(1.5)"), "false");
    // Computed class method name.
    assert_eq!(
        run("let k='go'; class C { [k](){ return 42; } } new C().go()"),
        "42"
    );
}

#[test]
fn object_spread_of_array_and_string() {
    assert_eq!(
        run("let o={...[10,20,30]}; o[0] + ':' + o[2] + ':' + Object.keys(o).join(',')"),
        "10:30:0,1,2"
    );
    assert_eq!(run("let o={...'ab'}; o[0] + o[1]"), "ab");
    // Mixed with explicit keys.
    assert_eq!(
        run("let o={...[1,2], a:9}; o[0] + ':' + o[1] + ':' + o.a"),
        "1:2:9"
    );
}

#[test]
fn object_spread_invokes_getters() {
    assert_eq!(
        run(
            "let s={a:1, get b(){ return this.a + 1; }}; let c={...s, d:3}; c.a + ',' + c.b + ',' + c.d"
        ),
        "1,2,3"
    );
    // Later keys win; both sources merged.
    assert_eq!(
        run("JSON.stringify({...{x:1},...{y:2},x:9})"),
        "{\"x\":9,\"y\":2}"
    );
}

#[test]
fn custom_symbol_iterator() {
    // for-of and spread drive a user `[Symbol.iterator]`.
    let src = "let o = { [Symbol.iterator]() { let i = 0; return { next() { return i < 3 ? { value: i++, done: false } : { value: undefined, done: true }; } }; } };";
    assert_eq!(
        run(&alloc::format!(
            "{src} let s=[]; for (let x of o) s.push(x); s.join(',')"
        )),
        "0,1,2"
    );
    assert_eq!(run(&alloc::format!("{src} [...o].join('-')")), "0-1-2");
}

#[test]
fn computed_object_literal_keys() {
    // `{ [expr]: v }` evaluates and coerces the key.
    assert_eq!(run("let k = 'a' + 'b'; let o = { [k]: 7 }; o.ab"), "7");
    // A numeric computed key coerces to its string form.
    assert_eq!(run("let o = { [1 + 1]: 'two' }; o['2']"), "two");
}

#[test]
fn private_class_fields() {
    // Private fields store and read through `this.#x`.
    assert_eq!(
        run(
            "class C { #n = 0; bump(){ this.#n++; return this.#n; } } let c = new C(); c.bump(); c.bump()"
        ),
        "2"
    );
    // ...and are non-enumerable.
    assert_eq!(
        run("class C { #s = 1; constructor(){ this.p = 2; } } Object.keys(new C()).join(',')"),
        "p"
    );
}

#[test]
fn bigints() {
    assert_eq!(run("typeof 5n"), "bigint");
    assert_eq!(run("(2n + 3n).toString()"), "5");
    assert_eq!(run("100n * 100n === 10000n"), "true");
    assert_eq!(run("2n ** 16n === 65536n"), "true");
    assert_eq!(run("10n / 3n === 3n"), "true");
    assert_eq!(run("-7n === 0n - 7n"), "true");
    assert_eq!(run("BigInt(99) === 99n"), "true");
    assert_eq!(run("10n === 10"), "false");
    assert_eq!(run("10n == 10"), "true");
    assert_eq!(run("!!0n"), "false");
    // Mixing BigInt and Number in arithmetic throws.
    assert_eq!(
        run("let r='ok'; try { 1n + 1; } catch (e) { r = 'threw'; } r"),
        "threw"
    );
    // Arbitrary precision: results far beyond i128 are exact.
    assert_eq!(
        run("(2n ** 200n).toString()"),
        "1606938044258990275541962092341162602522202993782792835301376"
    );
    assert_eq!(
        run("let f=1n; for(let i=1n;i<=25n;i++) f*=i; f.toString()"),
        "15511210043330985984000000"
    );
    assert_eq!(
        run("((2n ** 128n) - 1n).toString()"),
        "340282366920938463463374607431768211455"
    );
    assert_eq!(run("(~5n).toString()"), "-6");
    // Two's-complement bitwise, including beyond i128.
    assert_eq!(
        run("(12n & 10n).toString() + ',' + (12n | 10n) + ',' + (12n ^ 10n)"),
        "8,14,6"
    );
    assert_eq!(run("(-1n & 255n).toString()"), "255");
    assert_eq!(run("(((2n ** 100n) | 1n) - (2n ** 100n)).toString()"), "1");
}

#[test]
fn new_on_bound_function() {
    assert_eq!(
        run(
            "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null); let p=new B(3,4); p.x + ':' + p.y"
        ),
        "3:4"
    );
    assert_eq!(
        run("function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,10); new B(20).x"),
        "10"
    );
    assert_eq!(
        run("function P(x){this.x=x;} let B=P.bind(null); (new B(1)) instanceof P"),
        "true"
    );
    // Re-bound: bound args accumulate.
    assert_eq!(
        run(
            "function P(x,y){this.x=x;this.y=y;} let B=P.bind(null,5).bind(null,6); let p=new B(); p.x + ':' + p.y"
        ),
        "5:6"
    );
    // A class can be bound and constructed.
    assert_eq!(
        run("class C{constructor(v){this.v=v;}} new (C.bind(null))(42).v"),
        "42"
    );
    assert_eq!(
        run("class C{constructor(v){this.v=v;}} new (C.bind(null,7))().v"),
        "7"
    );
}

#[test]
fn apply_arraylike_and_bound_name() {
    // apply accepts an array-like (length + indexed properties).
    assert_eq!(
        run("function f(){return arguments.length;} f.apply(null,{length:3,0:1,1:2,2:3})"),
        "3"
    );
    assert_eq!(
        run(
            "function s(){let t=0;for(let i=0;i<arguments.length;i++)t+=arguments[i];return t;} s.apply(null,{length:2,0:10,1:20})"
        ),
        "30"
    );
    // A bound function's name.
    assert_eq!(run("function foo(){} foo.bind(null).name"), "bound foo");
    assert_eq!(
        run("function foo(){} foo.bind(null).bind(null).name"),
        "bound bound foo"
    );
}

#[test]
fn function_length_and_name() {
    assert_eq!(run("function f(a, b, c){} f.length"), "3");
    assert_eq!(run("function f(){} f.length"), "0");
    // length counts params before the first default/rest.
    assert_eq!(run("function f(a, b = 1, c){} f.length"), "1");
    assert_eq!(run("function f(a, ...r){} f.length"), "1");
    // name from a declaration and a named function expression.
    assert_eq!(run("function greet(){} greet.name"), "greet");
    assert_eq!(run("let g = function inner(){}; g.name"), "inner");
}

#[test]
fn object_seal_and_extensibility() {
    // preventExtensions: no new props, existing still writable.
    assert_eq!(
        run(
            "let o={a:1}; Object.preventExtensions(o); o.b=2; o.a=9; String(o.b) + ':' + o.a + ':' + Object.isExtensible(o)"
        ),
        "undefined:9:false"
    );
    // seal: no new props, no delete, existing writable.
    assert_eq!(
        run(
            "let o={x:1}; Object.seal(o); o.y=2; o.x=5; delete o.x; o.x + ':' + String(o.y) + ':' + Object.isSealed(o)"
        ),
        "5:undefined:true"
    );
    // freeze implies sealed + non-extensible.
    assert_eq!(
        run("let o={a:1}; Object.freeze(o); Object.isSealed(o) + ':' + Object.isExtensible(o)"),
        "true:false"
    );
}

#[test]
fn non_writable_and_join_nullish() {
    // defineProperty writable:false ignores writes; descriptor reports it.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1,writable:false,enumerable:true}); o.x=9; o.x"
        ),
        "1"
    );
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); Object.getOwnPropertyDescriptor(o,'x').writable"
        ),
        "false"
    );
    // Non-enumerable stays out of Object.keys but readable.
    assert_eq!(
        run(
            "let o={a:1}; Object.defineProperty(o,'h',{value:2,enumerable:false}); Object.keys(o).join(',') + ':' + o.h"
        ),
        "a:2"
    );
    // Array.join renders null/undefined as empty.
    assert_eq!(run("[1,null,2,undefined,3].join('-')"), "1--2--3");
}

#[test]
fn map_group_by_and_well_formed() {
    assert_eq!(
        run(
            "let g=Map.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); (g instanceof Map) + ':' + g.get('odd').join(',') + ':' + g.size"
        ),
        "true:1,3,5:2"
    );
    // Object keys are preserved (not stringified, unlike Object.groupBy).
    assert_eq!(
        run("let k={}; let g=Map.groupBy([1,2], x=>k); g.get(k).join(',')"),
        "1,2"
    );
    assert_eq!(
        run("'abc'.isWellFormed() + ':' + '\u{1f600}'.toWellFormed()"),
        "true:\u{1f600}"
    );
}

#[test]
fn get_own_property_symbols_and_reflect_ownkeys() {
    assert_eq!(
        run(
            "let s=Symbol('k'); let o={a:1}; o[s]=2; let g=Object.getOwnPropertySymbols(o); g.length + ':' + (g[0]===s) + ':' + o[g[0]]"
        ),
        "1:true:2"
    );
    assert_eq!(run("Object.getOwnPropertySymbols({}).length"), "0");
    // Reflect.ownKeys: string keys then symbol keys.
    assert_eq!(
        run(
            "let s=Symbol('k'); let o={a:1,b:2}; o[s]=3; let k=Reflect.ownKeys(o); k.length + ':' + k[0] + ':' + (k[2]===s)"
        ),
        "3:a:true"
    );
}

#[test]
fn assign_and_spread_copy_symbol_keys() {
    assert_eq!(
        run(
            "let s=Symbol('k'); let src={a:1}; src[s]=9; let t=Object.assign({},src); t.a + ':' + t[s]"
        ),
        "1:9"
    );
    assert_eq!(
        run("let s=Symbol('k'); let src={a:1}; src[s]=9; let t={...src}; t[s]"),
        "9"
    );
    // Object.keys still excludes the symbol.
    assert_eq!(
        run("let s=Symbol('k'); let src={a:1}; src[s]=9; Object.keys({...src}).join(',')"),
        "a"
    );
}

#[test]
fn object_group_by() {
    assert_eq!(
        run(
            "let g=Object.groupBy([1,2,3,4,5], x=>x%2?'odd':'even'); g.odd.join(',') + '|' + g.even.join(',')"
        ),
        "1,3,5|2,4"
    );
    assert_eq!(run("Object.groupBy(['a','ab','b'], s=>s[0]).a.length"), "2");
    assert_eq!(run("Object.keys(Object.groupBy([], x=>x)).length"), "0");
    // Works over any iterable + uses the index.
    assert_eq!(run("Object.groupBy('aab', c=>c).a.length"), "2");
}

#[test]
fn integer_key_ordering_and_array_tostring() {
    // Integer keys come first (ascending), then string keys in insertion order.
    assert_eq!(
        run("let o={2:'a',1:'b',3:'c'}; Object.keys(o).join(',')"),
        "1,2,3"
    );
    assert_eq!(
        run("let o={z:1, 2:2, a:3, 1:4}; Object.keys(o).join(',')"),
        "1,2,z,a"
    );
    assert_eq!(
        run("let o={}; o.b=1; o.a=2; Object.keys(o).join(',')"),
        "b,a"
    );
    assert_eq!(
        run("Object.values({10:'x',2:'y',1:'z'}).join(',')"),
        "z,y,x"
    );
    // Array toString joins with comma.
    assert_eq!(run("['a','b','c'].toString()"), "a,b,c");
    assert_eq!(run("[1,[2,3],4].toString()"), "1,2,3,4");
}

#[test]
fn inherited_setter_is_called() {
    // Assigning to a property with an *inherited* setter calls it (rather
    // than shadowing it with an own data property).
    assert_eq!(
        run(
            "let base={_d:0, get c(){return this._d;}, set c(v){this._d=v;}}; let d=Object.create(base); d.c=10; d._d + ':' + base._d"
        ),
        "10:0"
    );
    // A getter-only inherited accessor shadows the data assignment.
    assert_eq!(
        run("let base={get x(){return 1;}}; let d=Object.create(base); d.x=99; d.x"),
        "1"
    );
    // An own data property still assigns normally.
    assert_eq!(run("let o={a:1}; o.a=2; o.a"), "2");
}

#[test]
fn defineproperty_invariants() {
    // Redefining a non-configurable property throws; value is retained.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); try{ Object.defineProperty(o,'x',{value:2}); 'no' }catch(e){ (e instanceof TypeError)+':'+o.x }"
        ),
        "true:1"
    );
    // A configurable property can be redefined (attributes reset).
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.defineProperty(o,'x',{value:2,configurable:true}); o.x"
        ),
        "2"
    );
    // Defining a new property on a non-extensible object throws.
    assert_eq!(
        run(
            "let o={}; Object.preventExtensions(o); try{ Object.defineProperty(o,'z',{value:1}); 'no' }catch(e){ e instanceof TypeError }"
        ),
        "true"
    );
    // Non-configurable but writable: value may still change.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'w',{value:1,writable:true,configurable:false}); Object.defineProperty(o,'w',{value:2,writable:true,configurable:false}); o.w"
        ),
        "2"
    );
}

#[test]
fn descriptor_reports_configurable() {
    // defineProperty defaults to non-configurable.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1}); Object.getOwnPropertyDescriptor(o,'x').configurable"
        ),
        "false"
    );
    // Explicit configurable: true is reported.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{value:1,configurable:true}); Object.getOwnPropertyDescriptor(o,'x').configurable"
        ),
        "true"
    );
    // A plain literal property is configurable; a frozen one is not.
    assert_eq!(
        run("Object.getOwnPropertyDescriptor({a:1},'a').configurable"),
        "true"
    );
    assert_eq!(
        run("Object.getOwnPropertyDescriptor(Object.freeze({a:1}),'a').configurable"),
        "false"
    );
}

#[test]
fn delete_respects_configurable() {
    assert_eq!(run("let o={a:1}; delete o.a"), "true");
    assert_eq!(run("let o={}; delete o.missing"), "true");
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x"),
        "false"
    );
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'x',{value:1,configurable:false}); delete o.x; o.x"),
        "1"
    );
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'y',{value:2,configurable:true}); delete o.y"),
        "true"
    );
    assert_eq!(run("let o=Object.freeze({a:1}); delete o.a"), "false");
}

#[test]
fn redefine_accessor_as_data() {
    // Accessor → accessor.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{get(){return 2;},configurable:true}); o.x"
        ),
        "2"
    );
    // Accessor → data property (the old getter must no longer apply).
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'x',{get(){return 1;},configurable:true}); Object.defineProperty(o,'x',{value:42,configurable:true}); o.x"
        ),
        "42"
    );
}

#[test]
fn enumerable_accessor_keys() {
    // Object-literal getters are enumerable (appear in Object.keys/JSON).
    assert_eq!(
        run("Object.keys({x:1, get y(){return 2;}}).join(',')"),
        "x,y"
    );
    assert_eq!(
        run("JSON.stringify({a:1, get b(){return 2;}})"),
        "{\"a\":1,\"b\":2}"
    );
    // defineProperty accessor: enumerable per the descriptor.
    assert_eq!(
        run(
            "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;},enumerable:true}); Object.keys(o).join(',')"
        ),
        "a,c"
    );
    assert_eq!(
        run(
            "let o={a:1}; Object.defineProperty(o,'c',{get(){return 9;}}); Object.keys(o).join(',')"
        ),
        "a"
    );
    // Class accessors are non-enumerable.
    assert_eq!(
        run(
            "class C{ constructor(){this.a=1;} get b(){return 2;} } Object.keys(new C()).join(',')"
        ),
        "a"
    );
}

#[test]
fn own_key_order_data_accessor_interleave() {
    // A getter defined between two data properties keeps its chronological slot in
    // `[[OwnPropertyKeys]]` (integer-index-then-insertion order, regardless of
    // data-vs-accessor).
    assert_eq!(
        run(
            "let o={}; o.a=1; Object.defineProperty(o,'b',{get(){return 2;},enumerable:true,configurable:true}); o.c=3; Object.getOwnPropertyNames(o).join(',')"
        ),
        "a,b,c"
    );
    // Redefining an existing *data* property as an accessor must NOT move it to the
    // end — it retains its original insertion position (the fix: seed unified key
    // order before the data slot is deleted).
    assert_eq!(
        run(
            "let o={}; o.x=1; o.y=2; o.z=3; Object.defineProperty(o,'y',{get(){return 9;},enumerable:true,configurable:true}); Object.getOwnPropertyNames(o).join(',')"
        ),
        "x,y,z"
    );
    assert_eq!(
        run(
            "let o={}; o.x=1; o.y=2; o.z=3; Object.defineProperty(o,'y',{get(){return 9;},enumerable:true,configurable:true}); Reflect.ownKeys(o).join(',')"
        ),
        "x,y,z"
    );
    // The reverse (accessor redefined as data) also keeps its slot.
    assert_eq!(
        run(
            "let o={}; Object.defineProperty(o,'m',{get(){return 1;},enumerable:true,configurable:true}); o.n=2; Object.defineProperty(o,'m',{value:5,enumerable:true,configurable:true}); Object.getOwnPropertyNames(o).join(',')+'|'+o.m"
        ),
        "m,n|5"
    );
    // Integer-index keys still sort ahead of string keys, then insertion order.
    assert_eq!(
        run(
            "let o={}; o.b=1; o['2']=2; Object.defineProperty(o,'a',{get(){return 0;},enumerable:true,configurable:true}); o['1']=4; Object.getOwnPropertyNames(o).join(',')"
        ),
        "1,2,b,a"
    );
    // getOwnPropertyDescriptors and for-in observe the same interleaved order.
    assert_eq!(
        run(
            "let o={}; o.a=1; Object.defineProperty(o,'b',{get(){return 2;},enumerable:true,configurable:true}); o.c=3; Object.keys(Object.getOwnPropertyDescriptors(o)).join(',')"
        ),
        "a,b,c"
    );
    assert_eq!(
        run(
            "let o={}; o.a=1; Object.defineProperty(o,'b',{get(){return 2;},enumerable:true,configurable:true}); o.c=3; let r=[]; for(let k in o) r.push(k); r.join(',')"
        ),
        "a,b,c"
    );
}

#[test]
fn get_own_property_descriptors() {
    assert_eq!(
        run(
            "let o={a:1}; Object.defineProperty(o,'b',{value:2,writable:false,enumerable:true}); let d=Object.getOwnPropertyDescriptors(o); d.a.value + ',' + d.a.writable + ',' + d.b.value + ',' + d.b.writable"
        ),
        "1,true,2,false"
    );
    assert_eq!(
        run("Object.keys(Object.getOwnPropertyDescriptors({a:1,b:2})).join(',')"),
        "a,b"
    );
    assert_eq!(
        run(
            "let o={get x(){return 5;}}; let d=Object.getOwnPropertyDescriptors(o); typeof d.x.get"
        ),
        "function"
    );
}

#[test]
fn object_define_properties() {
    assert_eq!(
        run(
            "let o={}; Object.defineProperties(o, { x:{value:1}, y:{get:function(){return 2;}} }); o.x + ',' + o.y"
        ),
        "1,2"
    );
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'a',{value:42}); o.a"),
        "42"
    );
}

#[test]
fn computed_key_destructuring() {
    // Declaration form.
    assert_eq!(
        run("let k='name'; let {[k]: v} = {name:'Alice'}; v"),
        "Alice"
    );
    assert_eq!(
        run("let p='x'; let {[p]: a, ...rest} = {x:1, y:2}; a + ':' + rest.y"),
        "1:2"
    );
    // Assignment form.
    assert_eq!(
        run("let k='m'; let v; ({[k]: v} = {m:42}); String(v)"),
        "42"
    );
}

#[test]
fn destructuring_assignment_with_defaults() {
    assert_eq!(run("let a,b; [a,b]=[1,2]; a+','+b"), "1,2");
    assert_eq!(run("let a,b; [a,b]=[1,2]; [a,b]=[b,a]; a+','+b"), "2,1");
    assert_eq!(run("let a,b; ({x:a,y:b}={x:10,y:20}); a+','+b"), "10,20");
    // Default in an assignment pattern.
    assert_eq!(run("let a,b,c; [a,b,c=99]=[1,2]; String(c)"), "99");
    assert_eq!(run("let x; ({p:x=7}={}); String(x)"), "7");
}

#[test]
fn date_multi_arg_and_subtraction() {
    assert_eq!(
        run("let d=new Date(2026,5,5); d.getFullYear()+'/'+d.getMonth()+'/'+d.getDate()"),
        "2026/5/5"
    );
    assert_eq!(run("let d=new Date(0); d.getTime()"), "0");
    assert_eq!(run("(new Date(2000)) - (new Date(1000))"), "1000");
    // A two-digit year maps to 1900 + year.
    assert_eq!(run("Date.UTC(70,0,1)"), "0");
    assert_eq!(run("Date.UTC(2020,0,1)"), "1577836800000");
}

#[test]
fn date_utc_ieee754_arithmetic() {
    // MakeTime/MakeDate arithmetic follows IEEE-754 float rules exactly (the spec
    // is explicit that `*`/`+` are the ECMAScript operators): once magnitudes
    // exceed 2^53, exact-integer math would round differently or overflow an i64.
    assert_eq!(
        run("Date.UTC(1970, 0, 1, 80063993375, 29, 1, -288230376151711740)"),
        "29312"
    );
    assert_eq!(
        run("Date.UTC(1970, 0, 213503982336, 0, 0, 0, -18446744073709552000)"),
        "34447360"
    );
    // The same float path backs `new Date(y, m, …)`.
    assert_eq!(
        run("new Date(1970, 0, 1, 80063993375, 29, 1, -288230376151711740).getTime()"),
        "29312"
    );
}

#[test]
fn utf16_string_indexing() {
    assert_eq!(run("'café'.length"), "4");
    assert_eq!(run("'\\u{1F600}'.length"), "2");
    assert_eq!(run("'a\\u{1F600}b'.length"), "4");
    assert_eq!(run("'\\u{1F600}'.charCodeAt(0)"), "55357");
    assert_eq!(run("'\\u{1F600}'.charCodeAt(1)"), "56832");
    assert_eq!(run("'\\u{1F600}'.codePointAt(0)"), "128512");
    assert_eq!(run("'a\\u{1F600}b'.codePointAt(1)"), "128512");
    assert_eq!(run("'hello'.charCodeAt(0)"), "104");
}

#[test]
fn array_call_and_unary_plus_array() {
    // Array(...) without new.
    assert_eq!(run("Array(3).length"), "3");
    assert_eq!(run("Array(1,2,3).join(',')"), "1,2,3");
    assert_eq!(run("Array().length"), "0");
    // Unary + coerces arrays via their string form.
    assert_eq!(run("+[]"), "0");
    assert_eq!(run("+[5]"), "5");
    assert_eq!(run("Number.isNaN(+[1,2])"), "true");
    // Symbol.toPrimitive still gets the number hint for unary +.
    assert_eq!(
        run("+{[Symbol.toPrimitive](h){ return h==='number'?9:0; }}"),
        "9"
    );
}

#[test]
fn reverse_inplace_new_array_string_index() {
    // reverse mutates in place and returns the same array.
    assert_eq!(
        run("let a=[1,2,3]; let b=a.reverse(); (a===b) + ':' + a.join(',')"),
        "true:3,2,1"
    );
    // new Array(n) and new Array(...elements).
    assert_eq!(run("new Array(3).fill(7).join(',')"), "7,7,7");
    assert_eq!(run("new Array(1,2,3).join(',')"), "1,2,3");
    assert_eq!(run("new Array(3).length"), "3");
    // String index access.
    assert_eq!(run("'hello'[0] + 'hello'[4]"), "ho");
    assert_eq!(run("String('abc'[5])"), "undefined");
}

#[test]
fn array_immutable_methods() {
    assert_eq!(
        run("let a=[3,1,2]; a.toSorted().join(',') + '|' + a.join(',')"),
        "1,2,3|3,1,2"
    );
    assert_eq!(run("[1,2,3].toReversed().join(',')"), "3,2,1");
    assert_eq!(run("[1,2,3].with(1,9).join(',')"), "1,9,3");
    assert_eq!(run("[1,2,3].with(-1,9).join(',')"), "1,2,9");
    assert_eq!(
        run("[1,2,3,4,5].toSpliced(1,2,'a','b').join(',')"),
        "1,a,b,4,5"
    );
    assert_eq!(run("[1,2,3,4].toSpliced(2).join(',')"), "1,2");
    // with out-of-range → RangeError.
    assert_eq!(
        run("try { [1,2,3].with(10,0); 'no' } catch(e){ e instanceof RangeError }"),
        "true"
    );
}

#[test]
fn reduce_args_and_sort_in_place() {
    // reduce callback gets (acc, cur, index, array).
    assert_eq!(
        run(
            "let ix=[]; [10,20,30].reduce(function(a,c,i,arr){ ix.push(i + ':' + arr.length); return a+c; }, 0); ix.join(',')"
        ),
        "0:3,1:3,2:3"
    );
    assert_eq!(
        run("['a','b','c'].reduceRight(function(a,c){return a+c;})"),
        "cba"
    );
    // sort is in place and returns the same array.
    assert_eq!(
        run("let a=[3,1,2]; let b=a.sort(); (a===b) + ':' + a.join(',')"),
        "true:1,2,3"
    );
    assert_eq!(run("[3,1,2].sort((x,y)=>y-x).join(',')"), "3,2,1");
}

#[cfg(feature = "intl")]
#[test]
fn string_normalize_forms() {
    // "é" composed (1 cp) vs decomposed (e + U+0301).
    assert_eq!(run("'\u{e9}'.normalize('NFD').length"), "2");
    assert_eq!(run("'e\u{301}'.normalize('NFC').length"), "1");
    assert_eq!(
        run("'\u{e9}'.normalize() === 'e\u{301}'.normalize()"),
        "true"
    );
    // NFKC expands the ﬁ ligature.
    assert_eq!(run("'\u{fb01}'.normalize('NFKC')"), "fi");
    assert_eq!(run("'abc'.normalize()"), "abc");
    // An unsupported form throws a RangeError *object* (not a bare string).
    assert_eq!(
        run("try{'x'.normalize('BAD');'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    assert_eq!(
        run("try{'x'.normalize('BAD');'no'}catch(e){e.name}"),
        "RangeError"
    );
}

#[test]
fn string_raw_and_member_tag() {
    assert_eq!(run("String.raw`a\\nb`"), "a\\nb");
    assert_eq!(run("String.raw`${1}+${2}=${3}`"), "1+2=3");
    assert_eq!(run("String.raw`line\\tend`.length"), "9"); // backslash + t kept raw
    // A tag read as a member of a plain object also dispatches.
    assert_eq!(
        run("let o={ t(s){ return s.raw[0]; } }; o.t`x\\ny`"),
        "x\\ny"
    );
}

#[test]
fn generator_return_value() {
    // The return value is surfaced once, with done:true, after the yields.
    assert_eq!(
        run(
            "function* g(){ yield 1; yield 2; return 99; } let it=g(); it.next(); it.next(); let r=it.next(); r.value + ':' + r.done"
        ),
        "99:true"
    );
    // Subsequent next() calls yield undefined/done.
    assert_eq!(
        run(
            "function* g(){ yield 1; return 7; } let it=g(); it.next(); it.next(); String(it.next().value) + ':' + it.next().done"
        ),
        "undefined:true"
    );
    // Spread excludes the return value.
    assert_eq!(
        run("function* g(){ yield 1; yield 2; return 9; } [...g()].join(',')"),
        "1,2"
    );
}

#[test]
fn array_iterators() {
    assert_eq!(run("[...['a','b','c'].keys()].join(',')"), "0,1,2");
    assert_eq!(run("[...['a','b','c'].values()].join(',')"), "a,b,c");
    assert_eq!(
        run("let o=[]; for (let [i,v] of ['x','y'].entries()) o.push(i+':'+v); o.join(',')"),
        "0:x,1:y"
    );
    // The iterator supports next().
    assert_eq!(
        run("let it=['p','q'].values(); it.next().value + it.next().value"),
        "pq"
    );
}

#[test]
fn matchall_replaceall_require_global() {
    assert_eq!(
        run("try{ 'aaa'.replaceAll(/a/,'b'); 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(
        run("try{ [...'aaa'.matchAll(/a/)]; 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(run("'aaa'.replaceAll(/a/g,'b')"), "bbb");
    assert_eq!(run("[...'a1b2'.matchAll(/[a-z]\\d/g)].length"), "2");
    assert_eq!(
        run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, '$<m>/$<y>')"),
        "06/2024"
    );
}

#[test]
fn replace_groups_and_split_limit() {
    // The replace function receives a `groups` object for named captures.
    assert_eq!(
        run("'2024-06'.replace(/(?<y>\\d+)-(?<m>\\d+)/, (m,y,mo,o,s,g)=>g.y+'/'+g.m)"),
        "2024/06"
    );
    // Regex split honors the limit.
    assert_eq!(run("'a1b2c3'.split(/(\\d)/,3).join('|')"), "a|1|b");
    // Empty-regex split has no trailing empty; capture split keeps its trailing.
    assert_eq!(run("'abc'.split(/(?:)/).join(',')"), "a,b,c");
    assert_eq!(run("'a1'.split(/(\\d)/).length"), "3");
}

#[test]
fn regex_unicode_property_categories() {
    // Robust across the intl / no-intl matchers. Property escapes require the
    // `u` flag (without it `\p` is the literal `p`, per Annex B).
    assert_eq!(run(r#"'Hello World'.match(/\p{Lu}/gu).join('')"#), "HW");
    assert_eq!(run(r#"'Hello'.match(/\p{Ll}/gu).join('')"#), "ello");
    assert_eq!(run(r#"'abc123'.match(/\p{N}/gu).join('')"#), "123");
    assert_eq!(run(r#"'a.b!c'.match(/\p{P}/gu).join('')"#), ".!");
    assert_eq!(run(r#"'中文字'.match(/\p{Lo}/gu).length"#), "3");
    assert_eq!(run(r#"'a1b2'.match(/\P{N}/gu).join('')"#), "ab");
    // The full subcategory set compiles (matching may need Unicode tables).
    assert_eq!(
        run(r#"'x'.match(/\p{Sm}|\p{Sc}|\p{Mn}|\p{Pd}/gu)===null"#),
        "true"
    );
}

#[cfg(feature = "intl")]
#[test]
fn regex_unicode_property_precise_with_intl() {
    assert_eq!(run(r#"'3+5'.match(/\p{Sm}/u)[0]"#), "+");
    assert_eq!(run(r#"'$5'.match(/\p{Sc}/u)[0]"#), "$");
    assert_eq!(run(r#"'(a)'.match(/\p{Ps}/u)[0]"#), "(");
    assert_eq!(run(r#"'a-b'.match(/\p{Pd}/u)[0]"#), "-");
}

#[test]
fn regex_on_multibyte_strings() {
    // These previously panicked (char-index spans used as byte ranges).
    assert_eq!(run("'café'.match(/é/)[0]"), "é");
    assert_eq!(run("'café'.match(/(.+)/)[1]"), "café");
    assert_eq!(run("'café'.replace(/é/, 'e')"), "cafe");
    assert_eq!(run("'a→b→c'.split(/→/).join('|')"), "a|b|c");
    assert_eq!(run("'über 123'.match(/\\d+/)[0]"), "123");
    assert_eq!(run("'café'.match(/(?<r>.+)/).groups.r"), "café");
    assert_eq!(run("[...'café déjà'.matchAll(/é/g)].length"), "2");
    // Regex-template `$&`/`` $` ``/`$'` over a multibyte subject previously
    // byte-indexed char offsets and panicked; the template now slices chars
    // (RE-7 refactor).
    assert_eq!(run("'café'.replace(/f/, '[$&]')"), "ca[f]é");
    assert_eq!(run("'café'.replace(/f/, '$`')"), "cacaé");
    assert_eq!(run("'café'.replace(/f/, \"$'\")"), "caéé");
    assert_eq!(run("'aéb'.replace(/(é)/, '<$1>')"), "a<é>b");
}

#[test]
fn regex_empty_and_zerowidth_matches() {
    // Empty-match global replace keeps the characters.
    assert_eq!(run("'abc'.replace(/x*/g, '-')"), "-a-b-c-");
    // Zero-width (lookahead) split keeps the boundary character.
    assert_eq!(
        run("'camelCaseWord'.split(/(?=[A-Z])/).join('|')"),
        "camel|Case|Word"
    );
    // Capture-group split still splices the captures.
    assert_eq!(run("'a1b2'.split(/(\\d)/).join(',')"), "a,1,b,2,");
    // Lookahead-based number grouping replace.
    assert_eq!(
        run("'1234567'.replace(/(?<=\\d)(?=(?:\\d{3})+$)/g, ',')"),
        "1,234,567"
    );
}

#[test]
fn replace_dollar_patterns() {
    // String-pattern replace.
    assert_eq!(run("'hello'.replace('l', '[$&]')"), "he[l]lo");
    assert_eq!(run("'abc'.replace('b', '$`')"), "aac"); // prefix
    assert_eq!(run("'abc'.replace('b', \"$'\")"), "acc"); // suffix
    assert_eq!(run("'test'.replaceAll('t', '$$')"), "$es$"); // literal $
    // Regex-pattern replace.
    assert_eq!(
        run("'2024-06'.replace(/(\\d+)-(\\d+)/, '$2/$1')"),
        "06/2024"
    );
    assert_eq!(run("'x'.replace(/x/, '$1')"), "$1"); // no group 1 → literal
    assert_eq!(run("'abc'.replace(/b/, '$`')"), "aac");
}

#[test]
fn regex_stateful_last_index() {
    // Global exec advances lastIndex and resets on a miss.
    assert_eq!(
        run(
            "let r=/\\d/g; r.exec('a1b2')[0] + ':' + r.lastIndex + ':' + r.exec('a1b2')[0] + ':' + r.lastIndex"
        ),
        "1:2:2:4"
    );
    assert_eq!(
        run("let r=/\\d/g; r.exec('a1'); String(r.exec('a1')) + ':' + r.lastIndex"),
        "null:0"
    );
    // Writing lastIndex resumes from there.
    assert_eq!(run("let r=/\\d/g; r.lastIndex=3; r.exec('12345')[0]"), "4");
    // test() advances; non-global never does.
    assert_eq!(run("let r=/x/g; r.test('axbx'); r.lastIndex"), "2");
    assert_eq!(run("let r=/\\d/; r.exec('a1'); r.lastIndex"), "0");
}

#[test]
fn regex_lookbehind() {
    assert_eq!(run("/(?<=\\$)\\d+/.test('$100')"), "true");
    assert_eq!(run("/(?<=\\$)\\d+/.test('100')"), "false");
    assert_eq!(run("'$100'.match(/(?<=\\$)\\d+/)[0]"), "100");
    assert_eq!(
        run("'price: $50'.replace(/(?<=\\$)\\d+/, 'X')"),
        "price: $X"
    );
    assert_eq!(run("/(?<!a)b/.test('ab')"), "false");
    assert_eq!(run("/(?<!a)b/.test('xb')"), "true");
}

#[test]
fn regex_lookahead_and_backref() {
    // Positive / negative lookahead (zero-width).
    assert_eq!(run("/foo(?=bar)/.test('foobar')"), "true");
    assert_eq!(run("/foo(?=bar)/.test('foobaz')"), "false");
    assert_eq!(run("/foo(?!bar)/.test('foobaz')"), "true");
    assert_eq!(run("'foobar'.replace(/foo(?=bar)/, 'X')"), "Xbar");
    // Backreferences.
    assert_eq!(run("/(\\w)\\1/.test('hello')"), "true");
    assert_eq!(run("/(\\w)\\1/.test('abc')"), "false");
    assert_eq!(run("'hello'.match(/(.)\\1/)[0]"), "ll");
}

#[test]
fn regex_named_groups() {
    assert_eq!(
        run("let m='2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/); m.groups.y + ':' + m.groups.mo"),
        "2024:06"
    );
    // Still positionally indexable.
    assert_eq!(
        run("'2024-06'.match(/(?<y>\\d{4})-(?<mo>\\d{2})/)[1]"),
        "2024"
    );
    // Named backreference in replacement.
    assert_eq!(
        run("'John Smith'.replace(/(?<first>\\w+) (?<last>\\w+)/, '$<last>, $<first>')"),
        "Smith, John"
    );
    // No named groups → .groups is undefined.
    assert_eq!(run("String('ab'.match(/(a)(b)/).groups)"), "undefined");
}

#[test]
fn match_all_named_groups() {
    assert_eq!(
        run(
            "let m=[...'2024-06 2025-12'.matchAll(/(?<y>\\d{4})-(?<mo>\\d{2})/g)]; m[0].groups.y + ':' + m[1].groups.mo"
        ),
        "2024:12"
    );
    // Positional access + index still work on matchAll results.
    assert_eq!(run("[...'a1b2'.matchAll(/([a-z])(\\d)/g)][1][2]"), "2");
    assert_eq!(
        run("[...'xy'.matchAll(/(?<c>.)/g)].map(m=>m.groups.c).join('')"),
        "xy"
    );
}

#[test]
fn string_match_all() {
    assert_eq!(run("[...'a1b2c3'.matchAll(/([a-z])(\\d)/g)].length"), "3");
    assert_eq!(
        run("let m=[...'a1b2'.matchAll(/([a-z])(\\d)/g)]; m[0][0] + ':' + m[0][1] + ':' + m[0][2]"),
        "a1:a:1"
    );
    assert_eq!(
        run("[...'hello world'.matchAll(/\\w+/g)].map(m=>m[0]).join(',')"),
        "hello,world"
    );
    assert_eq!(run("[...'abc'.matchAll(/\\d/g)].length"), "0");
}

#[test]
fn array_thisarg_and_split_captures() {
    assert_eq!(
        run("[1,2,3].map(function(x){return x*this.m;},{m:3}).join(',')"),
        "3,6,9"
    );
    assert_eq!(
        run("[1,2,3,4].filter(function(x){return x>this.t;},{t:2}).join(',')"),
        "3,4"
    );
    assert_eq!(
        run("[1,2,3].some(function(x){return x===this.g;},{g:2})"),
        "true"
    );
    assert_eq!(
        run("[1,2,3].every(function(x){return x<=this.mx;},{mx:3})"),
        "true"
    );
    // split with a capturing separator splices the captures in.
    assert_eq!(run("'a1b2c3'.split(/(\\d)/).join('|')"), "a|1|b|2|c|3|");
}

#[test]
fn array_last_index_of_from_index() {
    assert_eq!(run("[10,20,30,20,10].lastIndexOf(20)"), "3");
    assert_eq!(run("[10,20,30,20,10].lastIndexOf(10,3)"), "0");
    assert_eq!(run("[10,20,30,20,10].lastIndexOf(20,-3)"), "1");
    assert_eq!(run("[1,2,3].lastIndexOf(9)"), "-1");
    assert_eq!(run("[1,2,3,4].findLastIndex(x => x < 3)"), "1");
}

#[test]
fn frozen_object_blocks_delete() {
    assert_eq!(
        run("let o={a:1,b:2}; Object.freeze(o); delete o.b; o.b"),
        "2"
    );
    assert_eq!(
        run("let o={a:1}; Object.freeze(o); o.a=9; o.c=3; o.a + ':' + String(o.c)"),
        "1:undefined"
    );
    assert_eq!(run("let o={a:1,b:2}; delete o.b; String(o.b)"), "undefined");
}

#[test]
fn array_and_function_named_properties() {
    assert_eq!(
        run("let a=[1,2,3]; a.tag='x'; a.tag + ':' + a.length + ':' + a[0]"),
        "x:3:1"
    );
    assert_eq!(run("let a=[1]; a.tag='y'; a.hasOwnProperty('tag')"), "true");
    assert_eq!(run("function f(){} f.meta=42; f.meta"), "42");
    // Tagged template strings carry `.raw`.
    assert_eq!(run("function t(s){ return s.raw[0]; } t`a\\tb`"), "a\\tb");
}

#[test]
fn error_to_string() {
    assert_eq!(run("new Error('boom').toString()"), "Error: boom");
    assert_eq!(run("new TypeError('bad').toString()"), "TypeError: bad");
    assert_eq!(run("new Error().toString()"), "Error");
    // A user toString still wins.
    assert_eq!(
        run("({ name:'X', message:'y', toString(){ return 'custom'; } }).toString()"),
        "custom"
    );
}

#[test]
fn symbol_to_primitive_hints() {
    let o = "let o={[Symbol.toPrimitive](h){ return h==='number'?42:h==='string'?'str':'def'; }};";
    assert_eq!(run(&alloc::format!("{o} +o")), "42");
    assert_eq!(run(&alloc::format!("{o} `${{o}}`")), "str");
    assert_eq!(run(&alloc::format!("{o} o + ''")), "def");
    // Symbol.toPrimitive takes precedence over valueOf/toString.
    assert_eq!(
        run("let o={[Symbol.toPrimitive](){ return 9; }, valueOf(){ return 1; }}; o + 0"),
        "9"
    );
}

#[test]
fn loose_equality_object_coercion() {
    assert_eq!(run("[] == 0"), "true");
    assert_eq!(run("[1] == 1"), "true");
    assert_eq!(run("[1,2] == '1,2'"), "true");
    assert_eq!(run("({}) == ({})"), "false"); // distinct objects
    assert_eq!(run("let o={valueOf(){return 5;}}; o == 5"), "true");
    assert_eq!(run("'' == 0"), "true");
    assert_eq!(run("null == 0"), "false");
}

#[test]
fn to_primitive_in_operators() {
    assert_eq!(run("let o={valueOf(){return 42;}}; o + 8"), "50");
    assert_eq!(run("let o={valueOf(){return 6;}}; o * 7"), "42");
    assert_eq!(run("let o={toString(){return 'x';}}; '' + o"), "x");
    // valueOf is preferred for the default/number hint.
    assert_eq!(
        run("let o={valueOf(){return 5;}, toString(){return 'five';}}; o + 1"),
        "6"
    );
    // Identity comparison does not coerce.
    assert_eq!(run("let o={valueOf(){return 1;}}; o === o"), "true");
}

#[test]
fn template_invokes_custom_tostring() {
    assert_eq!(
        run("let o = { toString() { return 'custom'; } }; `val=${o}`"),
        "val=custom"
    );
    // A plain object with no toString still renders the default form.
    assert_eq!(run("`${ {a:1} }`"), "[object Object]");
    // Arrays/numbers/booleans coerce as usual.
    assert_eq!(run("`${[1,2,3]}-${true}-${null}`"), "1,2,3-true-null");
}

#[test]
fn coercion_string_number_join_freeze_tofixed() {
    // String()/Number() honor toString/valueOf; join too.
    assert_eq!(run("String({toString(){return 'x';}})"), "x");
    assert_eq!(run("Number({valueOf(){return 42;}})"), "42");
    assert_eq!(
        run("[{toString(){return 'a';}},{toString(){return 'b';}}].join(',')"),
        "a,b"
    );
    // Frozen array rejects push: `Array.prototype.push` does `Set(…, Throw=true)`,
    // so it throws a TypeError (even in sloppy mode) and leaves the array unchanged.
    assert_eq!(
        run(
            "let a=[1,2,3]; let e=''; try{a=Object.freeze(a); a.push(4)}catch(x){e=x.name}; a.length + ':' + Object.isFrozen(a) + ':' + e"
        ),
        "3:true:TypeError"
    );
    // toFixed rounds half away from zero.
    assert_eq!(run("(0.5).toFixed(0)"), "1");
    assert_eq!(run("(2.5).toFixed(0)"), "3");
    assert_eq!(run("(123.456).toFixed(2)"), "123.46");
}

#[test]
fn math_abs_round_and_create_descriptors() {
    // Math.round rounds half toward +Infinity (not away from zero).
    assert_eq!(run("Math.round(-2.5)"), "-2");
    assert_eq!(run("Math.round(2.5)"), "3");
    assert_eq!(run("Math.round(-0.5) === 0"), "true");
    // Math.abs(-0) is +0.
    assert_eq!(run("Object.is(Math.abs(-0), 0)"), "true");
    // Object.create with a descriptors map.
    assert_eq!(
        run(
            "let p={g(){return 'hi';}}; let o=Object.create(p, {n:{value:5,enumerable:true}}); o.n + ':' + o.g() + ':' + Object.keys(o).join(',')"
        ),
        "5:hi:n"
    );
}

#[test]
fn negative_zero_stringifies_to_zero() {
    assert_eq!(run("String(-0)"), "0");
    assert_eq!(run("(-0).toString()"), "0");
    assert_eq!(run("'' + -0"), "0");
    assert_eq!(run("`${-0}`"), "0");
    assert_eq!(run("[-0, 0].join(',')"), "0,0");
    // But Object.is still distinguishes the bit pattern.
    assert_eq!(run("Object.is(-0, 0)"), "false");
}

#[test]
fn math_minus_zero_indexof_fromindex_number_exponential() {
    // Math.max/min treat +0 > -0.
    assert_eq!(run("Object.is(Math.max(-0, 0), 0)"), "true");
    assert_eq!(run("Object.is(Math.min(0, -0), -0)"), "true");
    // String.indexOf honors fromIndex.
    assert_eq!(run("'hello world'.indexOf('o', 5)"), "7");
    assert_eq!(run("'hello world'.indexOf('o')"), "4");
    // Number.toString exponential thresholds.
    assert_eq!(run("(1e21).toString()"), "1e+21");
    assert_eq!(run("(1e-7).toString()"), "1e-7");
    assert_eq!(run("(1e20).toString()"), "100000000000000000000"); // not exponential
    assert_eq!(run("(0.000001).toString()"), "0.000001"); // 1e-6 stays decimal
}

#[test]
fn math_trig_and_extra() {
    assert_eq!(
        run("Math.sin(0) + ':' + Math.cos(0) + ':' + Math.tan(0)"),
        "0:1:0"
    );
    assert_eq!(run("Math.round(Math.sin(Math.PI/2))"), "1");
    assert_eq!(run("Math.round(Math.atan2(1,1)*4/Math.PI)"), "1");
    assert_eq!(
        run("Math.cosh(0) + ':' + Math.tanh(0) + ':' + Math.expm1(0)"),
        "1:0:0"
    );
    assert_eq!(
        run("Math.fround(1.5) + ':' + (Math.fround(1.1) !== 1.1)"),
        "1.5:true"
    );
    assert_eq!(run("Math.clz32(1) + ':' + Math.clz32(0)"), "31:32");
    assert_eq!(run("Math.imul(3,4) + ':' + Math.imul(-1,8)"), "12:-8");
}

#[test]
fn number_formatting() {
    assert_eq!(run("(3.5).toString(2)"), "11.1");
    assert_eq!(run("(255.5).toString(16)"), "ff.8");
    assert_eq!(run("(-255.5).toString(16)"), "-ff.8");
    assert_eq!(run("(12345).toPrecision(1)"), "1e+4");
    assert_eq!(run("(0.0000001234).toPrecision(2)"), "1.2e-7");
    assert_eq!(run("(1234567).toLocaleString()"), "1,234,567");
    assert_eq!(run("(-1234.5).toLocaleString()"), "-1,234.5");
}

#[test]
fn math_random_in_range() {
    // In [0, 1), and consecutive calls differ (the PRNG advances).
    assert_eq!(run("let a=Math.random(); a >= 0 && a < 1"), "true");
    assert_eq!(run("Math.random() !== Math.random()"), "true");
    assert_eq!(
        run("let xs=[]; for (let i=0;i<100;i++) xs.push(Math.random()); xs.every(x=>x>=0&&x<1)"),
        "true"
    );
}

#[test]
fn math_constants() {
    assert_eq!(run("Math.PI > 3.14 && Math.PI < 3.15"), "true");
    assert_eq!(run("Math.E > 2.71 && Math.E < 2.72"), "true");
    assert_eq!(run("Math.SQRT2 * Math.SQRT2 > 1.999"), "true");
    assert_eq!(run("Math.floor(Math.LN2 * 1000)"), "693");
}

#[test]
fn private_in_brand_check() {
    assert_eq!(
        run(
            "class H{ #s=1; static check(o){ return #s in o; } } H.check(new H()) + ':' + H.check({})"
        ),
        "true:false"
    );
    // Works for an inherited brand too (subclass instances have the field).
    assert_eq!(
        run(
            "class H{ #s=1; static check(o){ return #s in o; } } class D extends H{} H.check(new D())"
        ),
        "true"
    );
}

#[test]
fn class_static_blocks() {
    assert_eq!(
        run("class C{ static x=1; static { C.y = C.x + 1; } } C.y"),
        "2"
    );
    // Multiple blocks run in order.
    assert_eq!(
        run("class C{ static n=0; static { C.n=10; } static { C.n+=5; } } C.n"),
        "15"
    );
    // `this` is the class inside a static block.
    assert_eq!(
        run("class C{ static x=1; static { this.y = this.x + 100; } } C.y"),
        "101"
    );
}

#[test]
fn static_setters_and_symbol_description() {
    // Static setter then getter.
    assert_eq!(
        run(
            "class T{ static _c=0; static get c(){return T._c;} static set c(v){T._c=v;} } T.c=25; T.c"
        ),
        "25"
    );
    // Symbol description: undefined for no-arg, the string otherwise.
    assert_eq!(run("String(Symbol().description)"), "undefined");
    assert_eq!(run("Symbol('d').description"), "d");
    assert_eq!(run("Symbol('').description"), ""); // explicit empty
}

#[test]
fn object_hasown_static_accessors_replaceall_fn() {
    // Object.hasOwn.
    assert_eq!(
        run("Object.hasOwn({a:1},'a') + ':' + Object.hasOwn({a:1},'b')"),
        "true:false"
    );
    assert_eq!(run("Object.hasOwn(Object.create({x:1}),'x')"), "false");
    // Static field write-back and static getter.
    assert_eq!(
        run(
            "class C{ static n=0; static inc(){ return ++C.n; } static get cur(){ return C.n; } } C.inc(); C.inc(); C.cur"
        ),
        "2"
    );
    // replaceAll with a function replacer.
    assert_eq!(
        run("'AAA'.replaceAll('A', function(){ return 'B'; })"),
        "BBB"
    );
    assert_eq!(
        run("'a1b2'.replace('1', function(m){ return '['+m+']'; })"),
        "a[1]b2"
    );
}

#[test]
fn object_reflection_and_static_inheritance() {
    assert_eq!(run("({a:1}).hasOwnProperty('a')"), "true");
    assert_eq!(run("({a:1}).hasOwnProperty('b')"), "false");
    // Static methods are inherited down the `extends` chain.
    assert_eq!(
        run("class A { static make(){ return 'made'; } } class B extends A {} B.make()"),
        "made"
    );
    // `static m(){ return new this(); }` uses the receiver class.
    assert_eq!(
        run(
            "class A { static create(){ return new this(); } get tag(){ return 'a'; } } class B extends A {} B.create().tag"
        ),
        "a"
    );
    // String.raw interleaves a raw-bearing object with substitutions.
    assert_eq!(run("String.raw({ raw: ['a','b','c'] }, 1, 2)"), "a1b2c");
}

#[test]
fn constructor_function_prototype() {
    // Method on the prototype, resolved through the instance.
    assert_eq!(
        run("function A(n){this.n=n;} A.prototype.m=function(){return this.n*2;}; new A(5).m()"),
        "10"
    );
    // Two-level prototype chain via Object.create.
    assert_eq!(
        run(
            "function A(){} A.prototype.greet=function(){return 'hi';}; function B(){} B.prototype=Object.create(A.prototype); new B().greet()"
        ),
        "hi"
    );
    // `.prototype` is a stable object across reads.
    assert_eq!(run("function A(){} A.prototype.x=1; A.prototype.x"), "1");
}

#[test]
fn named_function_expression_recurses() {
    assert_eq!(
        run("let f = function fac(n){ return n <= 1 ? 1 : n * fac(n-1); }; f(5)"),
        "120"
    );
    // The name is scoped to the expression, not visible outside.
    assert_eq!(
        run("let f = function self(n){ return n===0?0:n+self(n-1); }; f(4)"),
        "10"
    );
}

#[test]
fn array_length_assignment_resizes() {
    assert_eq!(run("let a=[1,2,3,4,5]; a.length=3; a.join(',')"), "1,2,3");
    assert_eq!(
        run("let a=[1,2]; a.length=4; String(a[3]) + ':' + a.length"),
        "undefined:4"
    );
    assert_eq!(run("let a=[1,2,3]; a.length=0; a.length"), "0");
    // String.fromCodePoint.
    assert_eq!(run("String.fromCodePoint(97, 98, 99)"), "abc");
}

#[test]
fn var_hoisting() {
    // A `var` read before its declaration line yields `undefined`.
    assert_eq!(
        run("function f(){ var a = b; var b = 5; return String(a); } f()"),
        "undefined"
    );
    assert_eq!(
        run("function f(){ return typeof later; var later = 1; } f()"),
        "undefined"
    );
    // A var inside a block still hoists to the function scope.
    assert_eq!(run("function f(){ { var x = 9; } return x; } f()"), "9");
}

#[test]
fn arrow_inherits_lexical_this() {
    assert_eq!(
        run("let o = { v: 42, m: function(){ let f = () => this.v; return f(); } }; o.m()"),
        "42"
    );
    // Nested arrows keep inheriting.
    assert_eq!(
        run("let o = { v: 7, m: function(){ return (() => (() => this.v)())(); } }; o.m()"),
        "7"
    );
}

#[test]
fn computed_class_members() {
    // Computed instance method, field, and getter names.
    assert_eq!(
        run(
            "let m='go'; class C{ [m](){return 1;} [m+'V']=2; get [m+'G'](){return 3;} } let c=new C(); c.go() + ':' + c.goV + ':' + c.goG"
        ),
        "1:2:3"
    );
    // Computed static method, field, and getter names.
    assert_eq!(
        run(
            "let s='mk'; class C{ static [s](){return 'a';} static [s+'N']=4; static get [s+'G'](){return 'b';} } C.mk() + ':' + C.mkN + ':' + C.mkG"
        ),
        "a:4:b"
    );
}

// The recursion guard (infinite recursion → RangeError, deep finite recursion
// works) is covered by the `recursion-guard` Test262 corpus test, which runs
// on a large stack; unit tests here run on the default ~2 MB thread stack,
// too small for the deep recursion the guard permits.

#[test]
fn super_member_read() {
    // super.getter (invoked) and super.method (returned then called).
    assert_eq!(
        run(
            "class B{ constructor(){this._v=10;} get d(){return this._v*2;} m(){return this._v;} } class D extends B{ get d(){return super.d+1;} m(){return super.m()+5;} } let x=new D(); x.d + ':' + x.m()"
        ),
        "21:15"
    );
    // super property as a function value.
    assert_eq!(
        run(
            "class A{ greet(){return 'A';} } class C extends A{ greet(){ let f=super.greet; return f.call(this)+'C'; } } new C().greet()"
        ),
        "AC"
    );
}

#[test]
fn date_setters_and_parse() {
    assert_eq!(
        run("let d=new Date(0); d.setUTCFullYear(2000); d.getUTCFullYear()"),
        "2000"
    );
    assert_eq!(
        run("let d=new Date(0); d.setUTCMonth(5); d.getUTCMonth()"),
        "5"
    );
    assert_eq!(
        run("let d=new Date(0); d.setTime(86400000); d.getUTCDate()"),
        "2"
    );
    assert_eq!(run("Date.parse('1970-01-01T00:00:00.000Z')"), "0");
    assert_eq!(
        run("Date.parse('2000-01-01T00:00:00.000Z') === Date.UTC(2000,0,1)"),
        "true"
    );
    assert_eq!(
        run("new Date('2000-01-01T12:00:00.000Z').getUTCHours()"),
        "12"
    );
    assert_eq!(run("Number.isNaN(Date.parse('garbage'))"), "true");
}

#[test]
fn json_date_and_bigint() {
    assert_eq!(
        run("JSON.stringify(new Date(0))"),
        "\"1970-01-01T00:00:00.000Z\""
    );
    assert_eq!(
        run("JSON.stringify({d:new Date(0)})"),
        "{\"d\":\"1970-01-01T00:00:00.000Z\"}"
    );
    assert_eq!(
        run("try{ JSON.stringify(10n); 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(
        run("try{ JSON.stringify({a:1n}); 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(run("JSON.stringify({a:1,b:'x'})"), "{\"a\":1,\"b\":\"x\"}");
}

#[test]
fn iterator_helpers() {
    let g = "function* g(){yield 1;yield 2;yield 3;yield 4;} ";
    assert_eq!(
        run(&alloc::format!("{g}[...g().map(x=>x*10)].join(',')")),
        "10,20,30,40"
    );
    assert_eq!(
        run(&alloc::format!("{g}[...g().filter(x=>x%2===0)].join(',')")),
        "2,4"
    );
    assert_eq!(run(&alloc::format!("{g}[...g().take(2)].join(',')")), "1,2");
    assert_eq!(run(&alloc::format!("{g}[...g().drop(2)].join(',')")), "3,4");
    assert_eq!(
        run(&alloc::format!("{g}g().toArray().join(',')")),
        "1,2,3,4"
    );
    assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b,0)")), "10");
    assert_eq!(run(&alloc::format!("{g}g().reduce((a,b)=>a+b)")), "10");
    assert_eq!(run(&alloc::format!("{g}g().some(x=>x>3)")), "true");
    assert_eq!(run(&alloc::format!("{g}g().every(x=>x>2)")), "false");
    assert_eq!(run(&alloc::format!("{g}g().find(x=>x>2)")), "3");
    assert_eq!(
        run(&alloc::format!(
            "{g}[...g().map(x=>x*2).filter(x=>x>4)].join(',')"
        )),
        "6,8"
    );
    // A helper over the remaining values after one `next()`.
    assert_eq!(
        run(&alloc::format!(
            "{g}let it=g(); it.next(); it.map(x=>x).toArray().join(',')"
        )),
        "2,3,4"
    );
}

#[test]
fn labeled_block_and_class_name() {
    // break out of a labeled block.
    assert_eq!(
        run("let r=[]; blk:{ r.push(1); if(true)break blk; r.push(2); } r.push(3); r.join(',')"),
        "1,3"
    );
    assert_eq!(run("let h='no'; a:{ b:{ break a; } h='in'; } h"), "no");
    // continue to a loop label still works.
    assert_eq!(
        run(
            "let r=[]; outer: for(let i=0;i<3;i++){ for(let j=0;j<3;j++){ if(j===1)continue outer; r.push(i+','+j); } } r.join(';')"
        ),
        "0,0;1,0;2,0"
    );
    // Named class self-reference and `.name`.
    assert_eq!(
        run("let C=class Named{ who(){return Named===C;} }; new C().who()"),
        "true"
    );
    assert_eq!(
        run("let C=class Named{ n(){return Named.name;} }; new C().n()"),
        "Named"
    );
    assert_eq!(run("class Declared{} Declared.name"), "Declared");
}

#[test]
fn arraybuffer_and_dataview() {
    assert_eq!(run("new ArrayBuffer(8).byteLength"), "8");
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,42); v.getInt32(0)"),
        "42"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setInt32(0,-1); v.getUint32(0)"),
        "4294967295"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setUint8(0,255); v.getInt8(0)"),
        "-1"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,true)"),
        "1000"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setInt16(0,1000,true); v.getInt16(0,false)"),
        "-6141"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setFloat64(0,3.14159); v.getFloat64(0)"),
        "3.14159"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setFloat32(0,1.5); v.getFloat32(0)"),
        "1.5"
    );
    assert_eq!(
        run("let v=new DataView(new ArrayBuffer(8)); v.setInt8(0,300); v.getInt8(0)"),
        "44"
    );
    // Offset view shares the buffer.
    assert_eq!(
        run(
            "let b=new ArrayBuffer(8); let v=new DataView(b); new DataView(b,2).setInt32(0,7); v.getInt32(2)"
        ),
        "7"
    );
}

#[test]
fn typed_arrays() {
    assert_eq!(run("new Uint8Array(3).length"), "3");
    assert_eq!(run("let a=new Uint8Array(1); a[0]=256; a[0]"), "0");
    assert_eq!(run("let a=new Uint8Array(1); a[0]=-1; a[0]"), "255");
    assert_eq!(run("let a=new Int8Array(1); a[0]=200; a[0]"), "-56");
    assert_eq!(
        run("new Uint8ClampedArray([300,-5,100]).join(',')"),
        "255,0,100"
    );
    assert_eq!(run("new Int16Array([70000])[0]"), "4464");
    assert_eq!(run("let f=new Float64Array(1); f[0]=3.14; f[0]"), "3.14");
    assert_eq!(run("new Uint8Array([1,2,3])[1]"), "2");
    assert_eq!(run("new Uint16Array(4).byteLength"), "8");
    assert_eq!(run("new Uint8Array(1).BYTES_PER_ELEMENT"), "1");
    assert_eq!(run("new Uint8Array([1,2,3]) instanceof Uint8Array"), "true");
    assert_eq!(
        run("new Uint8Array([1,2,3]).map(x=>x*2).join(',')"),
        "2,4,6"
    );
    assert_eq!(run("[...new Uint8Array([8,9])].join(',')"), "8,9");
}

#[test]
fn bigint_typed_arrays() {
    // Both kinds exist, are 64-bit, and share the %TypedArray% hierarchy.
    assert_eq!(run("typeof BigInt64Array"), "function");
    assert_eq!(run("typeof BigUint64Array"), "function");
    assert_eq!(run("BigInt64Array.BYTES_PER_ELEMENT"), "8");
    assert_eq!(run("BigUint64Array.BYTES_PER_ELEMENT"), "8");
    assert_eq!(run("new BigInt64Array(2).BYTES_PER_ELEMENT"), "8");
    assert_eq!(run("BigInt64Array.name"), "BigInt64Array");
    assert_eq!(
        run("Object.getPrototypeOf(BigInt64Array)===Object.getPrototypeOf(Int8Array)"),
        "true"
    );
    assert_eq!(run("new BigInt64Array(3).length"), "3");
    assert_eq!(run("new BigInt64Array(2)[0] === 0n"), "true");
    // Elements are BigInt; reading yields a BigInt, writing accepts BigInt.
    assert_eq!(run("var a=new BigInt64Array([1n,2n]); a[1]===2n"), "true");
    assert_eq!(run("new BigInt64Array([-1n])[0]"), "-1");
    // Little-endian i64 / u64 codec with low-64-bit two's-complement wrapping.
    assert_eq!(
        run("new BigUint64Array([18446744073709551615n])[0]"),
        "18446744073709551615"
    );
    assert_eq!(
        run("var a=new BigUint64Array(1); a[0]=-1n; a[0]"),
        "18446744073709551615"
    );
    assert_eq!(
        run("var a=new BigInt64Array(1); a[0]=18446744073709551617n; a[0]"),
        "1"
    );
    // ToBigInt on write: a Boolean / String coerces; a Number throws TypeError.
    assert_eq!(
        run("var a=new BigInt64Array(2); a[0]=true; a[1]='5'; a.join(',')"),
        "1,5"
    );
    assert_eq!(
        run("try{var a=new BigInt64Array(1);a[0]=5;'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{new BigInt64Array([1,2]);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Methods are BigInt-aware.
    assert_eq!(
        run("new BigInt64Array([5n,6n,7n]).slice(1).join(',')"),
        "6,7"
    );
    assert_eq!(
        run("new BigInt64Array([5n,6n,7n]).subarray(1).join(',')"),
        "6,7"
    );
    assert_eq!(
        run("var a=new BigInt64Array(3); a.set([9n,8n],1); a.join(',')"),
        "0,9,8"
    );
    assert_eq!(run("new BigInt64Array(2).fill(7n).join(',')"), "7,7");
    assert_eq!(
        run("try{new BigInt64Array(2).fill(3);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(run("BigInt64Array.of(3n,4n).join(',')"), "3,4");
    assert_eq!(
        run("var s=0n; for(var x of new BigInt64Array([1n,2n,3n])) s+=x; s===6n"),
        "true"
    );
    // A typed array constructed from another BigInt typed array copies values.
    assert_eq!(
        run(
            "var a=new BigUint64Array([10n,20n]); var b=new BigUint64Array(a); b[0]===10n && b!==a"
        ),
        "true"
    );
    // DataView round-trips the 64-bit BigInt accessors (little-endian arg).
    assert_eq!(
        run(
            "var dv=new DataView(new ArrayBuffer(8)); dv.setBigInt64(0,-7n); dv.getBigInt64(0)===-7n"
        ),
        "true"
    );
    assert_eq!(
        run(
            "var dv=new DataView(new ArrayBuffer(8)); dv.setBigUint64(0,5n,true); dv.getBigUint64(0,true)===5n"
        ),
        "true"
    );
    assert_eq!(
        run(
            "try{new DataView(new ArrayBuffer(8)).setBigInt64(0,5);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
}

#[test]
fn construct_throwing_prototype_getter_propagates_not_recurses() {
    // `reflect_new_target` is a one-shot: a `new X()` inside a throwing
    // `prototype` getter used by `Reflect.construct` must not re-observe the
    // outer newTarget (which previously caused unbounded recursion → a spurious
    // "Maximum call stack size exceeded" instead of the getter's own throw).
    for target in ["Uint8Array", "Float64Array", "Array", "(function C(){})"] {
        let src = alloc::format!(
            "function E(){{ this.m='boom'; }} \
             var nt = function(){{}}.bind(null); \
             Object.defineProperty(nt,'prototype',{{get(){{ throw new E(); }}}}); \
             var out='?'; \
             try {{ Reflect.construct({target}, [], nt); }} catch(e){{ out = e.m; }} \
             out"
        );
        assert_eq!(run(&src), "boom", "target = {target}");
    }
}

#[test]
fn typed_array_to_string_is_array_to_string() {
    // %TypedArray%.prototype.toString is the SAME function object as
    // Array.prototype.toString (23.2.3.30), and still joins the view.
    let ta = "Object.getPrototypeOf(Int8Array).prototype";
    assert_eq!(
        run(&alloc::format!(
            "{ta}.toString === Array.prototype.toString"
        )),
        "true"
    );
    assert_eq!(run("new Uint8Array([1,2,3]).toString()"), "1,2,3");
}

#[test]
fn typed_array_from_array_honors_overridden_iterator_next() {
    // `new TA(array)` drains the array via its @@iterator's actual `next`
    // (InitializeTypedArrayFromList), honoring a patched
    // %ArrayIteratorPrototype%.next rather than a raw element grab.
    assert_eq!(
        run("var P=Object.getPrototypeOf([].values()); \
             var vals=[1,2,3,4]; \
             P.next=function(){ var d=vals.length===0; return {value:vals.pop(),done:d}; }; \
             var a=new Uint16Array([0]); \
             a.length+':'+a[0]+','+a[1]+','+a[2]+','+a[3]"),
        "4:4,3,2,1"
    );
}

#[test]
fn reflect_set_typed_array_receiver_respects_integer_index_bounds() {
    // OrdinarySet's CreateDataProperty on a typed-array receiver: an invalid
    // index fails (false) without coercing the value; a valid index coerces
    // and writes.
    assert_eq!(
        run(
            "var t=new Int8Array([0,0]); var r=new Int8Array([9]); var c=0; \
             var v={valueOf(){c++; return 5;}}; \
             var ok=Reflect.set(t,1,v,r); ok+','+t[1]+','+r.hasOwnProperty(1)+','+c"
        ),
        "false,0,false,0"
    );
    // Receiver on the target's prototype chain (O === Receiver) → the value is
    // coerced exactly once and the out-of-bounds write is a silent success.
    assert_eq!(
        run(
            "var r=new Int32Array(10); var o=Object.create(r); var c=0; \
             var v={valueOf(){c++; return 1;}}; \
             Reflect.set(o,100,v,r)+','+c"
        ),
        "true,1"
    );
}

#[test]
fn typed_array_view_aliasing() {
    // Sibling views over one ArrayBuffer share bytes intrinsically.
    assert_eq!(
        run(
            "let b=new ArrayBuffer(8); let u=new Uint8Array(b); let f=new Float64Array(b); u[0]=255; f[0]>0"
        ),
        "true"
    );
    // A DataView write is seen by a typed-array view over the same buffer.
    assert_eq!(
        run(
            "let b=new ArrayBuffer(8); let u=new Uint8Array(b); let dv=new DataView(b); dv.setUint8(1,9); u[1]===9"
        ),
        "true"
    );
    // An offset/length view aliases the right window of the buffer.
    assert_eq!(
        run("let b=new ArrayBuffer(8); let u=new Uint8Array(b,2,4); u[0]=42; new Uint8Array(b)[2]"),
        "42"
    );
    // `subarray` shares the parent's buffer (not a copy).
    assert_eq!(
        run("let u=new Uint8Array([1,2,3,4]); let s=u.subarray(1,3); s[0]=99; u[1]"),
        "99"
    );
    // `.set`, `.fill`, `.copyWithin`, `.byteOffset`, and object-form JSON.
    assert_eq!(
        run("let u=new Uint8Array(4); u.set([5,6],1); u.join(',')"),
        "0,5,6,0"
    );
    assert_eq!(run("new Uint8Array([1,2,3]).fill(7).join(',')"), "7,7,7");
    assert_eq!(
        run("new Uint8Array([1,2,3,4]).copyWithin(0,2).join(',')"),
        "3,4,3,4"
    );
    assert_eq!(
        run("new Uint8Array(b=new ArrayBuffer(4),2).byteOffset"),
        "2"
    );
    assert_eq!(run("Array.isArray(new Uint8Array([1]))"), "false");
    assert_eq!(
        run("JSON.stringify(new Uint8Array([1,2,3]))"),
        "{\"0\":1,\"1\":2,\"2\":3}"
    );
    // BigInt64 round-trips through a DataView.
    assert_eq!(
        run(
            "let dv=new DataView(new ArrayBuffer(8)); dv.setBigInt64(0,-1n); dv.getBigInt64(0).toString()"
        ),
        "-1"
    );
}

#[test]
fn primitive_wrapper_objects() {
    // Number wrapper.
    assert_eq!(run("typeof new Number(5)"), "object");
    assert_eq!(run("new Number(5).valueOf()"), "5");
    assert_eq!(run("new Number(5) + 3"), "8");
    assert_eq!(run("new Number(255).toString(16)"), "ff");
    assert_eq!(run("new Number(5) instanceof Number"), "true");
    // String wrapper.
    assert_eq!(run("new String('hello').length"), "5");
    assert_eq!(run("new String('abc')[1]"), "b");
    assert_eq!(run("new String('HELLO').toLowerCase()"), "hello");
    assert_eq!(run("new String('a') + 'b'"), "ab");
    assert_eq!(run("new String('x') instanceof String"), "true");
    // Boolean wrapper.
    assert_eq!(run("new Boolean(false).valueOf()"), "false");
    assert_eq!(run("new Boolean(false) ? 'truthy' : 'falsy'"), "truthy");
    assert_eq!(run("new Boolean(true) instanceof Boolean"), "true");
    // Defaults.
    assert_eq!(run("new Number().valueOf()"), "0");
    assert_eq!(run("new String().valueOf()"), "");
}

#[test]
fn sloppy_this_is_global_object() {
    // Sloppy plain call: `this` is the global object.
    assert_eq!(
        run("(function(){ function f(){return this===globalThis;} return f(); })()"),
        "true"
    );
    assert_eq!(
        run("(function(){ function f(){return typeof this;} return f(); })()"),
        "object"
    );
    assert_eq!(
        run("(function(){ function f(){return this===globalThis;} return f.call(null); })()"),
        "true"
    );
    // Nested plain function.
    assert_eq!(
        run(
            "(function(){ var o={m(){ function inner(){return this===globalThis;} return inner(); }}; return o.m(); })()"
        ),
        "true"
    );
    // Strict (lexical) keeps `this` undefined.
    assert_eq!(
        run("(function(){'use strict'; function f(){return this===undefined;} return f(); })()"),
        "true"
    );
    assert_eq!(
        run("(function(){'use strict'; function f(){return this;} return f.call(null); })()"),
        "null"
    );
    // A method receiver and a lexical arrow `this` are unaffected.
    assert_eq!(
        run("(function(){ var o={x:5,m(){return this.x;}}; return o.m(); })()"),
        "5"
    );
    assert_eq!(
        run("(function(){ var o={x:9,m(){var a=()=>this.x;return a();}}; return o.m(); })()"),
        "9"
    );
}

#[test]
fn strict_mode_undeclared_assignment() {
    // Strict mode: an implicit-global assignment throws ReferenceError.
    assert_eq!(
        run(
            "(function(){'use strict'; try{ undeclaredX=1; return 'no'; }catch(e){ return e instanceof ReferenceError ? 'ref' : 'other'; }})()"
        ),
        "ref"
    );
    // Sloppy mode still creates the global.
    assert_eq!(
        run("(function(){ sloppyG=5; return typeof sloppyG; })()"),
        "number"
    );
    // Strict propagates to a nested function.
    assert_eq!(
        run(
            "(function(){'use strict'; return (function(){ try{nx=1;return 'no';}catch(e){return 'ref';} })(); })()"
        ),
        "ref"
    );
    // A declared binding is assignable under strict mode.
    assert_eq!(
        run("(function(){'use strict'; let x=1; x=2; return x; })()"),
        "2"
    );
    // Program-level `use strict`.
    assert_eq!(run("'use strict'; var ok='y'; ok"), "y");
    // Strict: writing a read-only property throws; sloppy silently ignores it.
    assert_eq!(
        run(
            "(function(){'use strict'; let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); try{o.x=2;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
        ),
        "te"
    );
    assert_eq!(
        run("let o={}; Object.defineProperty(o,'x',{value:1,writable:false}); o.x=2; o.x"),
        "1"
    );
    // Strict: a frozen object rejects writes.
    assert_eq!(
        run(
            "(function(){'use strict'; let o=Object.freeze({a:1}); try{o.a=9;return 'no';}catch(e){return e instanceof TypeError?'te':'other';}})()"
        ),
        "te"
    );
}

#[test]
fn block_level_function_hoisting() {
    assert_eq!(
        run("(function(){ {function g(){return 1;}} return typeof g; })()"),
        "function"
    );
    assert_eq!(
        run("(function(){ {function g(){return 42;}} return g(); })()"),
        "42"
    );
    assert_eq!(
        run("(function(){ if(true){function h(){return 5;}} return h(); })()"),
        "5"
    );
    assert_eq!(
        run("(function(){ {{function d(){return 9;}}} return d(); })()"),
        "9"
    );
    // A later block declaration overrides the outer one (function-scoped).
    assert_eq!(
        run("(function(){ function f(){return 'o';} {function f(){return 'i';}} return f(); })()"),
        "i"
    );
    // Top-level hoisting is unaffected.
    assert_eq!(
        run("(function(){ return e(); function e(){return 'h';} })()"),
        "h"
    );
}

#[test]
fn for_await_of_parses_and_runs() {
    // `for await` parses inside an async function and the call yields a promise.
    assert_eq!(
        run(
            "async function f(){ let s=0; for await(const x of [1,2,3]) s+=x; return s; } typeof f()"
        ),
        "object"
    );
    // An async generator is iterable with for-await.
    assert_eq!(
        run(
            "async function* g(){ yield 1; } async function f(){ for await(const x of g()){} } typeof f"
        ),
        "function"
    );
    // A regular for-of (no await) is unaffected by the AST change.
    assert_eq!(run("let s=0; for(const x of [1,2,3]) s+=x; s"), "6");
}

#[test]
fn intl_number_and_datetime_format() {
    assert_eq!(
        run("new Intl.NumberFormat('en-US').format(1234.5)"),
        "1,234.5"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US').format(1000000)"),
        "1,000,000"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(1234.5)"),
        "$1,234.50"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{style:'currency',currency:'JPY'}).format(1234)"),
        "¥1,234"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{style:'percent'}).format(0.25)"),
        "25%"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{minimumFractionDigits:2}).format(5)"),
        "5.00"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{useGrouping:false}).format(1234567)"),
        "1234567"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US').format(-1234.5)"),
        "-1,234.5"
    );
    // Callable without `new`.
    assert_eq!(run("Intl.NumberFormat('en-US').format(42)"), "42");
    assert_eq!(
        run("new Intl.DateTimeFormat('en-US').format(new Date(Date.UTC(2020,5,15)))"),
        "6/15/2020"
    );
}

#[test]
fn date_string_methods() {
    assert_eq!(run("new Date(0).toDateString()"), "Thu Jan 01 1970");
    assert_eq!(
        run("new Date(0).toUTCString()"),
        "Thu, 01 Jan 1970 00:00:00 GMT"
    );
    // toLocaleString now routes through a real DateTimeFormat: the en-US default
    // is 12-hour. CLDR's U+202F before the day period is folded to a plain space
    // to match the reference engine (see `dtf_pad_time_parts`).
    assert_eq!(
        run("new Date(Date.UTC(2020,5,15,10,30,45)).toLocaleString()"),
        "6/15/2020, 10:30:45 AM"
    );
    assert_eq!(
        run("new Date(Date.UTC(2020,5,15)).toLocaleDateString()"),
        "6/15/2020"
    );
    assert_eq!(
        run("new Date(Date.UTC(2021,11,25)).toDateString()"),
        "Sat Dec 25 2021"
    );
}

#[test]
fn base64_btoa_atob() {
    assert_eq!(run("btoa('hi')"), "aGk=");
    assert_eq!(run("btoa('Man')"), "TWFu");
    assert_eq!(run("btoa('M')"), "TQ==");
    assert_eq!(run("btoa('')"), "");
    assert_eq!(run("atob('aGVsbG8=')"), "hello");
    assert_eq!(run("atob(btoa('round trip!'))"), "round trip!");
    assert_eq!(run("btoa('é')"), "6Q==");
    assert_eq!(run("atob('aG k=')"), "hi"); // whitespace ignored
    assert_eq!(
        run("try{btoa('\\u{1F600}');'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
}

#[test]
fn structured_clone_deep_copy() {
    assert_eq!(
        run("let o={b:{c:2}}; let c=structuredClone(o); c.b.c=9; o.b.c"),
        "2"
    );
    assert_eq!(
        run("let c=structuredClone([1,[2,3]]); c[1][0]=9; c[1][0]"),
        "9"
    );
    assert_eq!(run("structuredClone(new Map([['k',1]])).get('k')"), "1");
    assert_eq!(
        run("[...structuredClone(new Set([1,2,3]))].join(',')"),
        "1,2,3"
    );
    assert_eq!(run("structuredClone(new Date(1000)).getTime()"), "1000");
    // Cycles and shared references.
    assert_eq!(
        run("let o={}; o.self=o; let c=structuredClone(o); c.self===c"),
        "true"
    );
    assert_eq!(
        run("let s={v:1}; let c=structuredClone({x:s,y:s}); c.x===c.y"),
        "true"
    );
    // Primitives pass through; functions throw.
    assert_eq!(run("structuredClone(42)"), "42");
    assert_eq!(
        run("try{structuredClone({f:function(){}});'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
}

#[test]
fn uri_encoding_functions() {
    assert_eq!(run("encodeURIComponent('a b&c=d')"), "a%20b%26c%3Dd");
    assert_eq!(run("decodeURIComponent('a%20b%26c')"), "a b&c");
    assert_eq!(run("encodeURI('http://a.com/x y')"), "http://a.com/x%20y");
    assert_eq!(run("encodeURIComponent('café')"), "caf%C3%A9");
    assert_eq!(run("decodeURIComponent('caf%C3%A9')"), "café");
    assert_eq!(run("encodeURIComponent(\"-_.!~*'()\")"), "-_.!~*'()");
    assert_eq!(run("decodeURIComponent('%2f')"), "/");
    // Malformed percent-encoding throws a URIError (a subclass of Error).
    assert_eq!(
        run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof URIError}"),
        "true"
    );
    assert_eq!(
        run("try{decodeURIComponent('%zz');'no'}catch(e){e instanceof Error}"),
        "true"
    );
    // `decodeURI` preserves the reserved set (`;/?:@&=+$,#`); `decodeURIComponent`
    // decodes it.
    assert_eq!(run("decodeURI('%3B')"), "%3B");
    assert_eq!(run("decodeURIComponent('%3B')"), ";");
    assert_eq!(run("decodeURI('http:%2f%2Fa')"), "http:%2f%2Fa");
    assert_eq!(run("decodeURIComponent('%2f')"), "/");
    // Multi-byte UTF-8 assembles into the code point (Cyrillic).
    assert_eq!(run("decodeURI('%D0%AE') === '\\u042E'"), "true");
    // Invalid/overlong UTF-8 → URIError.
    assert_eq!(
        run("try{decodeURI('%C0%80');'no'}catch(e){e instanceof URIError}"),
        "true"
    );
    // A lone surrogate is preserved verbatim by decode (not a `%` escape) …
    assert_eq!(
        run("decodeURI('\\uD800').charCodeAt(0).toString(16)"),
        "d800"
    );
    // … but encoding an unpaired surrogate throws a URIError.
    assert_eq!(
        run("try{encodeURI('\\uD800');'no'}catch(e){e instanceof URIError}"),
        "true"
    );
    assert_eq!(
        run("try{encodeURIComponent('\\uDC00');'no'}catch(e){e instanceof URIError}"),
        "true"
    );
    // A valid surrogate pair encodes to its 4-byte UTF-8 form.
    assert_eq!(run("encodeURIComponent('\\u{1F600}')"), "%F0%9F%98%80");
}

#[test]
fn uri_globals_resolve_inside_try() {
    // Regression: the URI/escape globals must be resolvable as bare identifiers
    // everywhere (they were missing from the VM's known-globals list, so a
    // reference inside a `try` compiled to an inline ReferenceError throw).
    for g in [
        "decodeURI",
        "decodeURIComponent",
        "encodeURI",
        "encodeURIComponent",
        "escape",
        "unescape",
    ] {
        assert_eq!(
            run(&alloc::format!("try{{typeof {g}}}catch(e){{'threw'}}")),
            "function",
            "{g} should resolve inside a try block"
        );
    }
}

#[test]
fn class_prototype_constructor_is_first() {
    // `MakeConstructor` defines `prototype.constructor` before the ClassElements,
    // so `constructor` precedes the methods in `[[OwnPropertyKeys]]` order.
    assert_eq!(
        run("class C{a(){} b(){}} Object.getOwnPropertyNames(C.prototype).join(',')"),
        "constructor,a,b"
    );
    assert_eq!(
        run("class C{a(){} ['b'](){} c(){}} Object.getOwnPropertyNames(C.prototype).join(',')"),
        "constructor,a,b,c"
    );
    // A class instance inherits `constructor` from the prototype — it has no own
    // `constructor` property.
    assert_eq!(
        run("class C{} new C().hasOwnProperty('constructor')"),
        "false"
    );
    assert_eq!(run("class C{} new C().constructor === C"), "true");
    // A computed `['constructor']` method is an ordinary prototype method, not the
    // class constructor, so it is what an instance reads.
    assert_eq!(
        run("class C{['constructor'](){return 1;}} new C().constructor()"),
        "1"
    );
    assert_eq!(
        run("class C{['constructor'](){return 1;}} C.prototype.constructor !== C"),
        "true"
    );
}

#[test]
fn match_all_iterator_honors_custom_exec() {
    // `%RegExpStringIteratorPrototype%.next` calls `RegExpExec` lazily, so a
    // custom `exec` installed after the iterator is created is honored.
    assert_eq!(
        run("var it=/./g[Symbol.matchAll]('abc');\
             RegExp.prototype.exec=function(){return ['ZZ'];};\
             var r=it.next(); r.value[0]+':'+r.done"),
        "ZZ:false"
    );
    // A `null` result ends iteration.
    assert_eq!(
        run("var it=/./g[Symbol.matchAll]('abc');\
             RegExp.prototype.exec=function(){return null;};\
             var r=it.next(); String(r.value)+':'+r.done"),
        "undefined:true"
    );
    // A throwing `exec` propagates out of `next()`.
    assert_eq!(
        run("var it=/./[Symbol.matchAll]('');\
             RegExp.prototype.exec=function(){throw new TypeError('x');};\
             try{it.next();'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
}

/// Runs `src` with a minimal `$262` host object (exposing `createRealm` /
/// `evalScript`), so cross-realm behavior can be exercised in unit tests.
fn run262(src: &str) -> String {
    const PRELUDE: &str = "var $262 = { global: this, \
        createRealm: function () { return $262_createRealm(); }, \
        evalScript: function (s) { return $262_evalScript(s); } };\n";
    let full = alloc::format!("{PRELUDE}{src}");
    let program = Parser::parse_program(&full).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

#[test]
fn cross_realm_eval_runs_in_target_realm() {
    // A created realm's `eval` resolves *that realm's* globals, not the caller's.
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             other.eval('Object') === Object"
        ),
        "false",
    );
    // And identity holds: the other realm's `Object` is what its `eval` sees.
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             other.eval('Object') === other.Object"
        ),
        "true",
    );
}

#[test]
fn cross_realm_dynamic_function_body_globals() {
    // A function built by another realm's `Function` constructor runs (sloppy
    // `this`, global reads) in that realm's global environment.
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             var f = new other.Function('return this');\
             f() === other"
        ),
        "true",
    );
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             var f = new other.Function('return Object');\
             f() === other.Object"
        ),
        "true",
    );
}

#[test]
fn cross_realm_generator_function_is_distinct() {
    // `%GeneratorFunction%` produced inside another realm's `eval` belongs to that
    // realm (its `.constructor` differs from the main realm's).
    assert_eq!(
        run262(
            "var GF = Object.getPrototypeOf(function*(){}).constructor;\
             var other = $262.createRealm().global;\
             var OGF = Object.getPrototypeOf(other.eval('(0, function*(){})')).constructor;\
             OGF === GF"
        ),
        "false",
    );
}

#[test]
fn cross_realm_created_realm_iterator_proto_chains_own_object_proto() {
    // A created realm's `%IteratorPrototype%` inherits *that realm's*
    // `Object.prototype`, not the parent realm's (regression: `new_object()` in
    // `install_globals` picked up the previous realm's `default_object_proto`).
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             Object.getPrototypeOf(other.Iterator.prototype) === other.Object.prototype"
        ),
        "true",
    );
    // …and it is NOT the main realm's `Object.prototype`.
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             Object.getPrototypeOf(other.Iterator.prototype) === Object.prototype"
        ),
        "false",
    );
}

#[test]
fn cross_realm_dynamic_generator_prototype_and_body_realm() {
    // `Reflect.construct(otherRealm.GeneratorFunction, [body], newTarget)`: the
    // created function's `.prototype` object inherits the *constructor* realm's
    // `%GeneratorPrototype%`, and its body reads the constructor realm's globals.
    let src = "var A = $262.createRealm().global;\
         A.calls = 0;\
         var AGF = A.eval('(0, function*(){})').constructor;\
         var aGenProto = Object.getPrototypeOf(A.eval('(0, function*(){})').prototype);\
         var B = $262.createRealm().global;\
         var nt = new B.Function(); nt.prototype = null;\
         var fn = Reflect.construct(AGF, ['calls += 1;'], nt);\
         var protoOk = Object.getPrototypeOf(fn.prototype) === aGenProto;\
         var g = fn();\
         var instOk = g instanceof A.Object;\
         g.next();\
         protoOk && instOk && A.calls === 1";
    assert_eq!(run262(src), "true");
}

#[test]
fn cross_realm_new_non_constructor_throws_current_realm_type_error() {
    // `new otherRealm.parseInt(0)` (a non-constructor) throws the *current* realm's
    // TypeError — the "not a constructor" check precedes any [[Construct]].
    assert_eq!(
        run262(
            "var other = $262.createRealm().global;\
             var ok = false;\
             try { new other.parseInt(0); } catch (e) { ok = e instanceof TypeError; }\
             ok"
        ),
        "true",
    );
}

#[test]
fn global_this_object() {
    assert_eq!(run("typeof globalThis"), "object");
    assert_eq!(run("globalThis.globalThis === globalThis"), "true");
    assert_eq!(run("globalThis.Math.max(1,2,3)"), "3");
    assert_eq!(run("globalThis.parseInt('42px')"), "42");
    assert_eq!(run("globalThis.Array.isArray([])"), "true");
    assert_eq!(run("globalThis.Infinity"), "Infinity");
    assert_eq!(run("globalThis.x = 7; globalThis.x"), "7");
}

#[test]
fn map_set_samevaluezero_and_set_ops() {
    // SameValueZero key matching.
    assert_eq!(run("let m=new Map(); m.set(NaN,'y'); m.get(NaN)"), "y");
    assert_eq!(run("new Set([NaN,NaN,1]).size"), "2");
    assert_eq!(run("let m=new Map(); m.set(-0,'n'); m.get(0)"), "n");
    // ES2025 Set composition.
    assert_eq!(
        run("[...new Set([1,2,3]).union(new Set([3,4]))].join(',')"),
        "1,2,3,4"
    );
    assert_eq!(
        run("[...new Set([1,2,3]).intersection(new Set([2,3,4]))].join(',')"),
        "2,3"
    );
    assert_eq!(
        run("[...new Set([1,2,3]).difference(new Set([2]))].join(',')"),
        "1,3"
    );
    assert_eq!(
        run("[...new Set([1,2]).symmetricDifference(new Set([2,3]))].join(',')"),
        "1,3"
    );
    assert_eq!(run("new Set([1,2]).isSubsetOf(new Set([1,2,3]))"), "true");
    assert_eq!(run("new Set([1,2,3]).isSupersetOf(new Set([1,2]))"), "true");
    assert_eq!(run("new Set([1,2]).isDisjointFrom(new Set([3,4]))"), "true");
    // The argument must be a *set-like* record (GetSetRecord), not a bare
    // iterable: a plain array is a TypeError (no `has`, `size` is `undefined`).
    assert_eq!(
        run("try{new Set([1,2,3]).intersection([2,3,9]); 'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
    // A genuine set-like object (numeric `size`, callable `has`/`keys`) works.
    assert_eq!(
        run(
            "let sl={size:2,has:v=>v===2||v===3,keys:()=>[2,3][Symbol.iterator]()}; [...new Set([1,2,3]).intersection(sl)].join(',')"
        ),
        "2,3"
    );
}

#[test]
fn parse_float_infinity() {
    assert_eq!(run("parseFloat('Infinity')"), "Infinity");
    assert_eq!(run("parseFloat('-Infinity')"), "-Infinity");
    assert_eq!(run("parseFloat('  +Infinity x')"), "Infinity");
    assert_eq!(run("parseFloat('InfinityX')"), "Infinity");
    assert_eq!(run("Number.isNaN(parseFloat('Inf'))"), "true");
    assert_eq!(run("parseFloat('3.14abc')"), "3.14");
}

#[test]
fn define_property_with_symbol_key() {
    assert_eq!(
        run("let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); o[s]"),
        "42"
    );
    assert_eq!(
        run(
            "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertyDescriptor(o,s).value"
        ),
        "42"
    );
    // A non-enumerable symbol (defineProperty's default) still appears here.
    assert_eq!(
        run(
            "let s=Symbol('k'); let o={}; Object.defineProperty(o,s,{value:42}); Object.getOwnPropertySymbols(o).length"
        ),
        "1"
    );
    // A symbol-keyed accessor.
    assert_eq!(
        run(
            "let s=Symbol('a'); let o={}; let v=0; Object.defineProperty(o,s,{get(){return v;},set(n){v=n;}}); o[s]=7; o[s]"
        ),
        "7"
    );
    assert_eq!(
        run(
            "let s=Symbol('r'); let o={}; Reflect.defineProperty(o,s,{value:9}); Reflect.getOwnPropertyDescriptor(o,s).value"
        ),
        "9"
    );
}

#[test]
fn error_stack_and_aggregate() {
    assert_eq!(run("typeof new Error('x').stack"), "string");
    assert_eq!(run("new Error('boom').stack.indexOf('boom') >= 0"), "true");
    assert_eq!(run("Object.keys(new Error('x')).indexOf('stack')"), "-1");
    // AggregateError: message is the 2nd arg, `.errors` collects the 1st.
    assert_eq!(
        run(
            "let a=new AggregateError([new Error('a'),new TypeError('b')],'m'); a.message + ':' + a.errors.length + ':' + a.name"
        ),
        "m:2:AggregateError"
    );
    assert_eq!(run("new AggregateError([],'x') instanceof Error"), "true");
    assert_eq!(
        run("new AggregateError(new Set([new Error('x')]),'s').errors.length"),
        "1"
    );
}

#[test]
fn error_cause_option() {
    assert_eq!(run("new Error('m',{cause:'r'}).cause"), "r");
    assert_eq!(run("new TypeError('t',{cause:42}).cause"), "42");
    assert_eq!(run("String(new Error('m').cause)"), "undefined");
    assert_eq!(run("String(new Error('m',{}).cause)"), "undefined");
    assert_eq!(
        run("new Error('o',{cause:new Error('i')}).cause.message"),
        "i"
    );
}

#[test]
fn class_extends_native_error() {
    assert_eq!(
        run(
            "class E extends Error{ constructor(m,c){ super(m); this.name='E'; this.c=c; } } let e=new E('x',5); e.message + ':' + e.c + ':' + e.name"
        ),
        "x:5:E"
    );
    assert_eq!(
        run("class E extends Error{} (new E('m')) instanceof Error"),
        "true"
    );
    assert_eq!(
        run(
            "class E extends Error{ constructor(m){super(m);} } let e=new E('m'); (e instanceof Error) + ',' + (e instanceof E) + ',' + (e instanceof TypeError)"
        ),
        "true,true,false"
    );
    assert_eq!(
        run(
            "class V extends RangeError{} let v=new V(); (v instanceof RangeError) + ',' + (v instanceof Error)"
        ),
        "true,true"
    );
}

#[test]
fn class_field_init_order_and_computed_fields() {
    // A field declared without an initializer must not clobber a constructor
    // write (fields init before the constructor body).
    assert_eq!(
        run("class A{ #b; constructor(v){ this.#b=v; } get b(){ return this.#b; } } new A(100).b"),
        "100"
    );
    assert_eq!(
        run(
            "class A{ #b; constructor(v){ this.#b=v; } add(n){ this.#b+=n; return this.#b; } } let a=new A(100); a.add(50)"
        ),
        "150"
    );
    // Computed instance field names.
    assert_eq!(run("let k='x'; class C{ [k+'1']=7; } new C().x1"), "7");
}

#[test]
fn class_rest_params_and_string_positions() {
    // Class constructor rest parameter (with spread).
    assert_eq!(
        run("class V{constructor(...c){this.c=c;}} new V(...[1,2,3]).c.length"),
        "3"
    );
    assert_eq!(
        run("class V{constructor(a, ...r){this.r=r;}} new V(1,2,3).r.join(',')"),
        "2,3"
    );
    // Class constructor default parameter.
    assert_eq!(
        run("class P{constructor(x=7){this.x=x;}} new P().x + ':' + new P(2).x"),
        "7:2"
    );
    // String prefix/suffix with positions.
    assert_eq!(run("'hello world'.startsWith('world', 6)"), "true");
    assert_eq!(run("'hello world'.endsWith('hello', 5)"), "true");
    assert_eq!(
        run("'hello'.includes('lo', 3) + ':' + 'hello'.includes('he', 1)"),
        "true:false"
    );
}

#[test]
fn arithmetic_object_coercion() {
    assert_eq!(run("[5] - 2"), "3");
    assert_eq!(run("[10] / 2"), "5");
    assert_eq!(run("[6] & 3"), "2");
    assert_eq!(run("[2] ** 3"), "8");
    assert_eq!(run("String({} - 1)"), "NaN");
    assert_eq!(run("-[5]"), "-5");
    assert_eq!(run("new Date(5000) - new Date(2000)"), "3000");
}

#[test]
fn tostring_in_concat_and_property_key() {
    // String.concat honors a user toString.
    assert_eq!(run("'x'.concat({toString(){return 'TS';}})"), "xTS");
    // An object property key is coerced via ToString (toString).
    assert_eq!(
        run("let k={toString(){return 'key';}}; let m={}; m[k]=42; m.key + ':' + m[k]"),
        "42:42"
    );
}

#[test]
fn relational_object_coercion() {
    assert_eq!(run("String([5] < 10)"), "true");
    assert_eq!(run("String([20] > 10)"), "true");
    assert_eq!(run("String([1] < [2])"), "true"); // "1" < "2"
    assert_eq!(run("String([10] < [9])"), "true"); // lexicographic
    assert_eq!(run("String({} < 1)"), "false"); // NaN
    assert_eq!(run("String(new Date(1) < new Date(2))"), "true"); // by timestamp
}

#[test]
fn loose_eq_object_coercion() {
    assert_eq!(run("String([] == false)"), "true"); // []→""→0, false→0
    assert_eq!(run("String([] == 0)"), "true");
    assert_eq!(run("String([0] == false)"), "true"); // [0]→"0"→0
    assert_eq!(run("String({} == 0)"), "false"); // "[object Object]"→NaN
    assert_eq!(run("String({} == {})"), "false"); // distinct objects
    assert_eq!(run("String([1,2] == '1,2')"), "true");
}

#[test]
fn array_string_index_access() {
    assert_eq!(run("let a=[10,20,30]; a['0'] + ':' + a['2']"), "10:30");
    assert_eq!(run("let a=[10,20,30]; let k='1'; a[k]"), "20");
    assert_eq!(
        run("let a=[10,20,30]; String(a['00']) + ':' + String(a['01'])"),
        "undefined:undefined"
    );
    assert_eq!(run("[[1,2],[3,4]]['0']['1']"), "2");
}

#[test]
fn object_literal_async_methods_parse() {
    // `async`/`get`/`set` remain usable as property names.
    assert_eq!(
        run("let async=5; let o={async, get:6, set:7}; o.async + ':' + o.get + ':' + o.set"),
        "5:6:7"
    );
    // An async method is a function whose call yields a promise (object).
    assert_eq!(
        run("let o={ async f(){return 1;} }; typeof o.f"),
        "function"
    );
    assert_eq!(
        run("let o={ async f(){return 1;} }; typeof o.f()"),
        "object"
    );
    assert_eq!(
        run("let k='m'; let o={ async [k](){return 1;} }; typeof o.m"),
        "function"
    );
}

#[test]
fn object_literal_generator_methods() {
    assert_eq!(
        run("let o={ *g(){yield 1;yield 2;} }; [...o.g()].join(',')"),
        "1,2"
    );
    assert_eq!(
        run("let o={ *[Symbol.iterator](){yield 'a';yield 'b';} }; [...o].join(',')"),
        "a,b"
    );
    assert_eq!(
        run("let k='m'; let o={ *[k](){yield 9;} }; [...o.m()].join(',')"),
        "9"
    );
    // The generator method reads `this`.
    assert_eq!(
        run("let o={ v:5, *items(){yield this.v;yield this.v*2;} }; [...o.items()].join(',')"),
        "5,10"
    );
}

#[test]
fn class_symbol_iterator_method() {
    assert_eq!(
        run("class C{ *[Symbol.iterator](){yield 'x';yield 'y';} } [...new C()].join(',')"),
        "x,y"
    );
    // A non-generator iterator method (manual iterator object).
    assert_eq!(
        run(
            "class C{ [Symbol.iterator](){let i=0;return{next:()=>i<3?{value:i++,done:false}:{done:true}};} } [...new C()].join(',')"
        ),
        "0,1,2"
    );
    // for-of uses it too.
    assert_eq!(
        run(
            "class C{ *[Symbol.iterator](){yield 1;yield 2;} } let s=0; for(let v of new C())s+=v; s"
        ),
        "3"
    );
}

#[test]
fn generator_is_its_own_iterator() {
    assert_eq!(
        run("function* g(){yield 1;} let it=g(); it[Symbol.iterator]() === it"),
        "true"
    );
    assert_eq!(
        run(
            "function* g(){yield 1;yield 2;} let it=g(); it[Symbol.iterator]().next().value + ':' + it.next().value"
        ),
        "1:2"
    );
    assert_eq!(
        run("function* g(){yield* [1,2]; yield* 'ab';} [...g()].join(',')"),
        "1,2,a,b"
    );
}

#[test]
fn explicit_symbol_iterator_call() {
    assert_eq!(
        run("let it=[10,20,30][Symbol.iterator](); it.next().value + ',' + it.next().value"),
        "10,20"
    );
    assert_eq!(run("'abc'[Symbol.iterator]().next().value"), "a");
    assert_eq!(
        run("let m=new Map([['k','v']])[Symbol.iterator]().next().value; m[0] + '=' + m[1]"),
        "k=v"
    );
    assert_eq!(run("new Set([1,2])[Symbol.iterator]().next().value"), "1");
    assert_eq!(run("[...[1,2,3][Symbol.iterator]()].join(',')"), "1,2,3");
}

#[test]
fn in_operator_walks_prototype_chain() {
    assert_eq!(run("'a' in {a:1}"), "true");
    assert_eq!(run("'z' in {a:1}"), "false");
    assert_eq!(run("let o=Object.create({x:1}); 'x' in o"), "true");
    assert_eq!(
        run("let o=Object.create(Object.create({deep:1})); 'deep' in o"),
        "true"
    );
    assert_eq!(run("0 in [10,20]"), "true");
    assert_eq!(run("5 in [10,20]"), "false");
}

#[test]
fn for_in_inherited_enumeration() {
    assert_eq!(
        run(
            "let p={a:1}; let o=Object.create(p); o.b=2; let k=[]; for(let x in o)k.push(x); k.sort().join(',')"
        ),
        "a,b"
    );
    // Non-enumerable prototype methods are not enumerated.
    assert_eq!(run("let k=[]; for(let x in {})k.push(x); k.length"), "0");
    // A shadowed inherited key appears once.
    assert_eq!(
        run("let o=Object.create({v:1}); o.v=2; let k=[]; for(let x in o)k.push(x); k.length"),
        "1"
    );
}

#[test]
fn const_reassignment_throws() {
    assert_eq!(
        run("const x=1; try{ x=2; 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(run("const x=1; try{ x=2; }catch(e){} x"), "1");
    assert_eq!(
        run("const n=10; try{ n+=5; 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    // Mutation through a const reference is allowed; let is reassignable.
    assert_eq!(run("const a=[1]; a.push(2); a.length"), "2");
    assert_eq!(run("let y=1; y=2; y"), "2");
    // An inner const shadows without affecting the outer.
    assert_eq!(run("const a=1; { const a=2; } a"), "1");
}

#[test]
fn destructure_any_iterable() {
    // Array binding patterns destructure any iterable, not just arrays.
    assert_eq!(run("let [a,b,c]='xyz'; a+b+c"), "xyz");
    assert_eq!(
        run("let [f,...r]=new Set([1,2,3,4]); f + ':' + r.join(',')"),
        "1:2,3,4"
    );
    assert_eq!(
        run("function* g(){yield 10;yield 20;} let [x,y]=g(); x+y"),
        "30"
    );
    assert_eq!(run("let [[k,v]]=new Map([['a',1]]); k + ':' + v"), "a:1");
}

#[test]
fn computed_member_assignment_eval_order() {
    // The index is resolved before the RHS (which mutates it).
    assert_eq!(
        run("let a=[0,0]; let i=0; a[i] = i = 1; a[0] + ',' + a[1]"),
        "1,0"
    );
    // Compound assignment on a computed element still works.
    assert_eq!(run("let a=[1,2,3]; a[1] *= 10; a.join(',')"), "1,20,3");
    assert_eq!(run("let o={x:5}; let k='x'; o[k] += 3; o.x"), "8");
    // Computed key honoring a setter.
    assert_eq!(run("let o={set v(n){this._v=n*2;}}; o['v']=10; o._v"), "20");
}

#[test]
fn in_operator_array_bounds_and_delete() {
    assert_eq!(run("0 in [1,2,3]"), "true");
    assert_eq!(run("5 in [1,2,3]"), "false"); // out of bounds
    assert_eq!(run("'length' in [1,2,3]"), "true");
    assert_eq!(run("'a' in {a:1}"), "true");
    assert_eq!(run("'b' in {a:1}"), "false");
    // delete clears an array element.
    assert_eq!(
        run("let a=[1,2,3]; delete a[1]; String(a[1]) + ':' + a.length"),
        "undefined:3"
    );
    assert_eq!(run("let o={a:1}; delete o.a; 'a' in o"), "false");
}

#[test]
fn catch_binding_forms() {
    // Destructured catch binding (object and array patterns).
    assert_eq!(
        run("let r; try { throw {code:42, text:'x'}; } catch({code,text}){ r=code+':'+text; } r"),
        "42:x"
    );
    assert_eq!(
        run("let r; try { throw [1,2,3]; } catch([a,b]){ r=a+b; } r"),
        "3"
    );
    // Optional catch binding (no parameter).
    assert_eq!(
        run("let r=false; try { throw 1; } catch { r=true; } r"),
        "true"
    );
    // Named binding still works.
    assert_eq!(
        run("let r; try { throw new Error('m'); } catch(e){ r=e.message; } r"),
        "m"
    );
}

#[test]
fn arguments_object() {
    assert_eq!(
        run(
            "function s(){ var t=0; for (var i=0;i<arguments.length;i++) t+=arguments[i]; return t; } s(1,2,3,4)"
        ),
        "10"
    );
    assert_eq!(
        run("function f(){ return arguments[1]; } f('a','b','c')"),
        "b"
    );
    assert_eq!(run("function f(){ return arguments.length; } f()"), "0");
    // An arrow inherits the enclosing `arguments`.
    assert_eq!(
        run("function outer(){ var a = () => arguments[0]; return a(); } outer('Z')"),
        "Z"
    );
}

#[test]
fn function_call_apply_bind() {
    assert_eq!(
        run("function f(p){return p + ':' + this.n;} f.call({n:7}, 'a')"),
        "a:7"
    );
    assert_eq!(
        run("function f(a,b){return a+b+this.n;} f.apply({n:1}, [2,3])"),
        "6"
    );
    assert_eq!(
        run("function f(a,b){return a+b+this.n;} let g=f.bind({n:10}, 5); g(20) + ':' + typeof g"),
        "35:function"
    );
    assert_eq!(run("Math.max.apply(null, [3,9,2])"), "9");
}

#[test]
fn prototype_chains() {
    // Inherited data property and method (this-bound), own shadows inherited.
    assert_eq!(
        run(
            "let p={k:'base',m:function(){return this.n;}}; let o=Object.create(p); o.n=7; o.k + ':' + o.m()"
        ),
        "base:7"
    );
    // Object.keys excludes inherited; getPrototypeOf identity.
    assert_eq!(
        run("let p={a:1}; let o=Object.create(p); o.b=2; Object.keys(o).join(',')"),
        "b"
    );
    assert_eq!(
        run("let p={}; Object.getPrototypeOf(Object.create(p)) === p"),
        "true"
    );
    assert_eq!(
        run("Object.getPrototypeOf(Object.create(null)) === null"),
        "true"
    );
    // Two-level chain: nearest prototype wins.
    assert_eq!(
        run("let a={x:1}; let b=Object.create(a); b.x=2; let c=Object.create(b); c.x"),
        "2"
    );
    // setPrototypeOf installs the link.
    assert_eq!(run("let o={}; Object.setPrototypeOf(o,{v:9}); o.v"), "9");
}

#[test]
fn proxy_get_set_traps() {
    // get trap with fallthrough; set trap transforming the value.
    assert_eq!(
        run(
            "let t={a:1}; let p=new Proxy(t,{get:function(o,k){return k in o?o[k]:'def';}}); p.a + ':' + p.zzz"
        ),
        "1:def"
    );
    assert_eq!(
        run(
            "let t={}; let p=new Proxy(t,{set:function(o,k,v){o[k]=v*3;return true;}}); p.n=4; t.n"
        ),
        "12"
    );
    // No-trap handler forwards to the target.
    assert_eq!(
        run("let p=new Proxy({x:5},{}); p.y=6; '' + p.x + p.y"),
        "56"
    );
    assert_eq!(run("typeof new Proxy({}, {})"), "object");
    // has trap (for `in`) and deleteProperty trap (for `delete`).
    assert_eq!(
        run(
            "let p=new Proxy({a:1},{has:function(t,k){return k==='magic'||k in t;}}); '' + ('a' in p) + ('magic' in p) + ('z' in p)"
        ),
        "truetruefalse"
    );
    assert_eq!(
        run(
            "let seen=''; let p=new Proxy({a:1},{deleteProperty:function(t,k){seen=k; delete t[k]; return true;}}); delete p.a; seen + ':' + ('a' in p)"
        ),
        "a:false"
    );
    // Forwarding `in`/`delete` with no traps.
    assert_eq!(
        run("let p=new Proxy({x:1},{}); let r = 'x' in p; delete p.x; '' + r + ('x' in p)"),
        "truefalse"
    );
}

#[test]
fn instanceof_array_object_and_fromentries_map() {
    assert_eq!(run("[] instanceof Array"), "true");
    assert_eq!(run("({}) instanceof Array"), "false");
    assert_eq!(run("[] instanceof Object"), "true");
    assert_eq!(run("({}) instanceof Object"), "true");
    assert_eq!(run("'str' instanceof Object"), "false"); // primitive
    // Object.fromEntries from a Map.
    assert_eq!(
        run("let m=new Map([['x',10],['y',20]]); let o=Object.fromEntries(m); o.x + ':' + o.y"),
        "10:20"
    );
}

#[test]
fn collection_foreach_thisarg() {
    assert_eq!(
        run(
            "let r=[]; new Map([['a',1],['b',2]]).forEach(function(v,k){ r.push(k+':'+v*this.m); }, {m:10}); r.join(',')"
        ),
        "a:10,b:20"
    );
    assert_eq!(
        run("let s=0; new Set([1,2,3]).forEach(function(v){ s+=v*this.m; }, {m:2}); s"),
        "12"
    );
    // The callback also gets the collection as the third argument.
    assert_eq!(
        run("let n; new Set([1]).forEach((v,k,coll)=>{ n=coll.size; }); n"),
        "1"
    );
}

#[test]
fn map_set_clear_and_assign_getters() {
    assert_eq!(
        run("let m=new Map(); m.set('a',1).set('b',2); m.clear(); m.size + ':' + m.has('a')"),
        "0:false"
    );
    assert_eq!(
        run("let s=new Set([1,2,3]); s.clear(); s.add(5).add(5); s.size"),
        "1"
    );
    // Object.assign invokes getters.
    assert_eq!(
        run(
            "let src={a:1, get b(){ return this.a + 1; }}; let t=Object.assign({}, src); t.a + ',' + t.b"
        ),
        "1,2"
    );
    assert_eq!(run("Object.assign({}, {x:1}, {y:2}, {x:9}).x"), "9");
}

#[test]
fn bigint_shifts() {
    assert_eq!(run("(1n << 8n).toString()"), "256");
    assert_eq!(run("(256n >> 2n).toString()"), "64");
    assert_eq!(run("(2n ** 32n).toString()"), "4294967296");
    assert_eq!(run("(-8n >> 1n).toString()"), "-4");
    assert_eq!(run("(-7n >> 1n).toString()"), "-4"); // arithmetic floor
    assert_eq!(run("(5n << -1n).toString()"), "2"); // negative count reverses
}

#[test]
fn number_of_bigint_and_collection_from_iterable() {
    // Number(bigint) → the double value.
    assert_eq!(run("Number(100n)"), "100");
    assert_eq!(run("Number(-7n)"), "-7");
    assert_eq!(run("Number(2n ** 10n)"), "1024");
    // Set/Map seed from any iterable, incl. a string.
    assert_eq!(run("[...new Set('hello')].join('')"), "helo");
    assert_eq!(run("new Set([1,1,2,3]).size"), "3");
    assert_eq!(
        run("let m=new Map([['a',1],['b',2]]); m.get('a') + m.get('b')"),
        "3"
    );
}

// `Promise.any` (first-fulfilled / AggregateError-when-all-reject) is covered
// by the `promise-allsettled-any` Test262 corpus test, which awaits the
// settled value (microtask timing isn't observable synchronously here).

#[test]
fn weakref_and_finalization_registry() {
    assert_eq!(
        run("let o={x:1}; let r=new WeakRef(o); (r.deref()===o) + ':' + r.deref().x"),
        "true:1"
    );
    assert_eq!(
        run("typeof WeakRef + ',' + typeof FinalizationRegistry"),
        "function,function"
    );
    // `register`/`unregister` now maintain a real `[[Cells]]` list with spec
    // brand/argument validation (the cleanup callback still never fires — no GC).
    assert_eq!(
        run(
            "let reg=new FinalizationRegistry(()=>{}); let t={}; reg.register({}, 'h', t); \
             String(reg.unregister(t)) + ',' + String(reg.unregister(t))"
        ),
        "true,false"
    );
    // An unregister token that cannot be held weakly (a string) is a TypeError.
    assert_eq!(
        run(
            "let reg=new FinalizationRegistry(()=>{}); try { reg.unregister('t'); 'no throw' } \
             catch (e) { e.constructor.name }"
        ),
        "TypeError"
    );
    // Brand checks + prototype shape.
    assert_eq!(
        run(
            "typeof WeakRef.prototype.deref + ',' + WeakRef.prototype[Symbol.toStringTag] + ',' + \
             FinalizationRegistry.prototype[Symbol.toStringTag]"
        ),
        "function,WeakRef,FinalizationRegistry"
    );
    assert_eq!(run("new WeakRef([1,2,3]).deref().length"), "3");
}

#[test]
fn symbol_prototype_methods() {
    // `Symbol.prototype` exists with brand-checking methods; instance behavior
    // is unchanged.
    assert_eq!(run("typeof Symbol.prototype"), "object");
    assert_eq!(run("Symbol.prototype[Symbol.toStringTag]"), "Symbol");
    assert_eq!(
        run("Symbol.prototype.toString.call(Symbol('x'))"),
        "Symbol(x)"
    );
    assert_eq!(
        run("let s=Symbol('y'); Symbol.prototype.valueOf.call(s)===s"),
        "true"
    );
    assert_eq!(
        run("typeof Symbol.prototype[Symbol.toPrimitive]"),
        "function"
    );
    assert_eq!(
        run("Object.getPrototypeOf(Symbol('66'))===Symbol.prototype"),
        "true"
    );
    assert_eq!(run("Symbol.prototype.constructor===Symbol"), "true");
    // A non-symbol `this` is a TypeError.
    assert_eq!(
        run(
            "try { Symbol.prototype.valueOf.call({}); 'no throw' } catch (e) { e.constructor.name }"
        ),
        "TypeError"
    );
    // Instance fast paths intact; `Symbol() instanceof Symbol` stays false.
    assert_eq!(run("Symbol('a').description"), "a");
    assert_eq!(run("Symbol('a') instanceof Symbol"), "false");
    assert_eq!(run("Symbol().hasOwnProperty('description')"), "false");
}

#[test]
fn math_exp_log_reflect_assign_array() {
    // Math.exp / Math.log.
    assert_eq!(run("Math.round(Math.exp(0))"), "1");
    assert_eq!(run("Math.round(Math.log(Math.E))"), "1");
    // Object.assign spreads an array source's indices.
    assert_eq!(
        run("let o=Object.assign({}, ['a','b','c']); o[0] + o[2] + ':' + Object.keys(o).join(',')"),
        "ac:0,1,2"
    );
    // Reflect.has walks the chain; Reflect.set updates array storage.
    assert_eq!(run("Reflect.has(Object.create({k:1}), 'k')"), "true");
    assert_eq!(
        run("let a=[1,2,3]; Reflect.set(a, 3, 4); a[3] + ':' + a.length"),
        "4:4"
    );
    assert_eq!(
        run("Reflect.defineProperty({}, 'x', {value:5}) === true"),
        "true"
    );
    assert_eq!(
        run(
            "let o={}; Reflect.defineProperty(o,'x',{value:9,enumerable:true}); Reflect.getOwnPropertyDescriptor(o,'x').value"
        ),
        "9"
    );
}

#[test]
fn reflect_and_weak_collections() {
    // Reflect mirrors the fundamental operations.
    assert_eq!(run("let o={a:1}; Reflect.get(o,'a')"), "1");
    assert_eq!(run("let o={}; Reflect.set(o,'x',5); o.x"), "5");
    assert_eq!(
        run("Reflect.has({a:1},'a') + ':' + Reflect.has({a:1},'z')"),
        "true:false"
    );
    assert_eq!(run("Reflect.ownKeys({a:1,b:2,c:3}).length"), "3");
    assert_eq!(
        run("function f(a,b){return a+b+this.n;} Reflect.apply(f,{n:10},[1,2])"),
        "13"
    );
    assert_eq!(
        run("function B(v){this.v=v;} Reflect.construct(B,[9]).v"),
        "9"
    );
    // WeakMap / WeakSet (object-keyed; bounded — no true weakness).
    assert_eq!(
        run("let k={}; let m=new WeakMap(); m.set(k,'v'); m.get(k) + ':' + m.has(k)"),
        "v:true"
    );
    // Weak collections are recognized by instanceof and chain from set/add.
    assert_eq!(run("(new WeakMap()).set({}, 1) instanceof WeakMap"), "true");
    assert_eq!(run("(new WeakSet()).add({}) instanceof WeakSet"), "true");
    assert_eq!(
        run("let s=new WeakSet(); let o={}; s.add(o); s.has(o) + ':' + s.has({})"),
        "true:false"
    );
}

#[test]
fn proxy_revocable() {
    // Works before revoke; every operation throws after.
    assert_eq!(
        run(
            "let r=Proxy.revocable({a:1},{get:function(t,k){return t[k];}}); let b=r.proxy.a; r.revoke(); let after='ok'; try { r.proxy.a; } catch(e){ after='threw'; } b + ':' + after"
        ),
        "1:threw"
    );
    assert_eq!(
        run("let r=Proxy.revocable({},{}); typeof r.proxy + ',' + typeof r.revoke"),
        "object,function"
    );
}

#[test]
fn proxy_apply_construct_traps() {
    // apply trap intercepts a call; typeof a function proxy is "function".
    assert_eq!(
        run(
            "function f(a,b){return a+b;} let p=new Proxy(f,{apply:function(t,th,a){return a[0]*a[1];}}); p(3,4) + ':' + typeof p"
        ),
        "12:function"
    );
    assert_eq!(
        run("function f(a){return a+1;} let p=new Proxy(f,{}); p(9)"),
        "10"
    );
    // construct trap intercepts `new`.
    assert_eq!(
        run(
            "function B(v){this.v=v;} let p=new Proxy(B,{construct:function(t,a){return {v:a[0]*2};}}); (new p(5)).v"
        ),
        "10"
    );
    assert_eq!(
        run("function B(v){this.v=v;} let p=new Proxy(B,{}); (new p(7)).v"),
        "7"
    );
}

#[test]
fn symbols() {
    assert_eq!(run("typeof Symbol('x')"), "symbol");
    assert_eq!(run("Symbol('hi').toString()"), "Symbol(hi)");
    assert_eq!(run("Symbol('hi').description"), "hi");
    assert_eq!(run("Symbol('a') === Symbol('a')"), "false");
    assert_eq!(run("let s = Symbol(); s === s"), "true");
    assert_eq!(run("Symbol.for('k') === Symbol.for('k')"), "true");
    assert_eq!(run("Symbol.keyFor(Symbol.for('k2'))"), "k2");
    assert_eq!(run("typeof Symbol.iterator"), "symbol");
}

#[test]
fn symbol_keyed_properties() {
    // Distinct symbols are distinct keys; symbol keys are non-enumerable.
    assert_eq!(
        run(
            "let a=Symbol('k'),b=Symbol('k'),o={}; o[a]=1; o[b]=2; o.p=3; '' + o[a] + o[b] + Object.keys(o).join('')"
        ),
        "12p"
    );
    assert_eq!(run("let s=Symbol(); let o={}; o[s]='v'; s in o"), "true");
    assert_eq!(
        run("let s=Symbol(); let o={}; o[s]=1; delete o[s]; o[s]"),
        "undefined"
    );
    assert_eq!(
        run("let o={}; o[Symbol.iterator]='it'; o[Symbol.iterator]"),
        "it"
    );
}

#[test]
fn regex_replace_with_function() {
    assert_eq!(
        run("'a1b2'.replace(/[0-9]/g, function(m){ return '<'+m+'>'; })"),
        "a<1>b<2>"
    );
    assert_eq!(
        run("'1-2'.replace(/(\\d)-(\\d)/, function(_, a, b){ return b+'-'+a; })"),
        "2-1"
    );
    // A string replacement still works.
    assert_eq!(run("'foo'.replace(/o/g, '0')"), "f00");
}

#[test]
fn promise_combinators() {
    // Drive the combinators through output (await/then resolve eagerly).
    assert_eq!(
        out(
            "Promise.all([Promise.resolve(1), 2, Promise.resolve(3)]).then(r => console.log(r.join(',')));"
        ),
        "1,2,3\n"
    );
    assert_eq!(
        out(
            "Promise.race([Promise.resolve('a'), Promise.resolve('b')]).then(v => console.log(v));"
        ),
        "a\n"
    );
    assert_eq!(
        out(
            "Promise.all([Promise.resolve(1), Promise.reject('boom')]).catch(e => console.log('caught:' + e));"
        ),
        "caught:boom\n"
    );
}

#[test]
fn empty_string_is_falsy() {
    assert_eq!(run("!!''"), "false");
    assert_eq!(run("!!'x'"), "true");
    assert_eq!(run("'' || 'fallback'"), "fallback");
    assert_eq!(run("if ('') { 'T' } else { 'F' }"), "F");
    assert_eq!(run("Boolean('')"), "false");
    assert_eq!(
        run("[0, '', null, 1, 'a'].filter(function(x){ return x; }).join(',')"),
        "1,a"
    );
}

#[test]
fn array_from_array_like() {
    // `{ length }` array-like with a map callback (tree-walker path).
    assert_eq!(
        run("Array.from({length:3}, function(_,i){ return i*i; }).join(',')"),
        "0,1,4"
    );
    // Array-like with indexed props, no map fn.
    assert_eq!(run("Array.from({length:2, 0:'a', 1:'b'}).join('-')"), "a-b");
    // Still works for real iterables.
    assert_eq!(
        run("Array.from([1,2,3], function(x){ return x*2; }).join(',')"),
        "2,4,6"
    );
}

#[test]
fn array_from_index_and_string_search() {
    assert_eq!(run("[1,2,3,2,1].indexOf(2, 2)"), "3");
    assert_eq!(run("[1,2,3].includes(2, 2)"), "false");
    assert_eq!(run("[5,6,7].indexOf(5, 1)"), "-1");
    assert_eq!(run("'hello world'.search('world')"), "6");
    assert_eq!(run("'abc'.search('z')"), "-1");
}

#[test]
fn collection_iterators() {
    // Map keys/values/entries.
    assert_eq!(
        run("[...new Map([['a',1],['b',2]]).keys()].join(',')"),
        "a,b"
    );
    assert_eq!(
        run("[...new Map([['a',1],['b',2]]).values()].join(',')"),
        "1,2"
    );
    // `entries()` is a real iterator object (with `.next`), not an array.
    assert_eq!(
        run(
            "let e=new Map([['a',1],['b',2]]).entries(); let r=e.next(); r.value.join(':')+'/'+r.done"
        ),
        "a:1/false"
    );
    assert_eq!(
        run("[...new Map([['a',1],['b',2]]).entries()].map(p=>p.join(':')).join(',')"),
        "a:1,b:2"
    );
    // Set values/keys are its elements.
    assert_eq!(run("[...new Set([1,2,3,2]).values()].join(',')"), "1,2,3");
    assert_eq!(run("[...new Set([5,6]).keys()].join(',')"), "5,6");
}

#[test]
fn eager_generators() {
    // for-of and spread over a finite generator.
    assert_eq!(
        run(
            "function* g(n){ for (let i=0;i<n;i++) yield i*i; } let s=[]; for (let v of g(4)) s.push(v); s.join(',')"
        ),
        "0,1,4,9"
    );
    assert_eq!(
        run("function* g(){ yield 'a'; yield 'b'; } [...g()].join('-')"),
        "a-b"
    );
    // The next() iterator protocol.
    assert_eq!(
        run(
            "function* g(){ yield 1; yield 2; } let it=g(); '' + it.next().value + it.next().value + it.next().done"
        ),
        "12true"
    );
    // yield* delegation.
    assert_eq!(
        run(
            "function* inner(){ yield 2; yield 3; } function* outer(){ yield 1; yield* inner(); yield 4; } [...outer()].join(',')"
        ),
        "1,2,3,4"
    );
}

#[test]
fn new_on_constructor_functions() {
    // `this` binding + implicit instance return.
    assert_eq!(run("function P(x){ this.x = x; } new P(7).x"), "7");
    // `instanceof` matches the constructing function, not others.
    assert_eq!(
        run(
            "function P(){} function Q(){} let p = new P(); '' + (p instanceof P) + (p instanceof Q)"
        ),
        "truefalse"
    );
    // An explicit object return overrides the new instance.
    assert_eq!(
        run("function F(){ this.a = 1; return { b: 2 }; } let o = new F(); '' + o.a + o.b"),
        "undefined2"
    );
    // The hidden constructor tag does not enumerate.
    assert_eq!(
        run("function P(){ this.v = 1; } Object.keys(new P()).join(',')"),
        "v"
    );
}

#[test]
fn class_methods_are_non_enumerable() {
    // Methods are callable but absent from enumeration (only public fields
    // show up), and `{...obj}` spread skips them too.
    assert_eq!(
        run(
            "class C { m(){ return 1; } constructor(){ this.a = 1; this.b = 2; } } Object.keys(new C()).join(',')"
        ),
        "a,b"
    );
    assert_eq!(
        run(
            "class C { greet(){ return 'hi'; } } let c = new C(); c.greet() + ':' + Object.keys({ ...c }).length"
        ),
        "hi:0"
    );
}

#[test]
fn optional_calls_destructuring_assign_and_coercion() {
    // Optional calls short-circuit on a nullish callee.
    assert_eq!(run("let o = { f: () => 7 }; o.f?.()"), "7");
    assert_eq!(run("let o = {}; String(o.missing?.())"), "undefined");
    // Destructuring assignment (swap, rest, member targets).
    assert_eq!(run("let a = 1, b = 2; [a, b] = [b, a]; a + ',' + b"), "2,1");
    assert_eq!(
        run("let h, t; [h, ...t] = [1, 2, 3, 4]; h + '|' + t.join(',')"),
        "1|2,3,4"
    );
    assert_eq!(
        run("let p = {}; ({ x: p.px, y: p.py } = { x: 10, y: 20 }); p.px + ',' + p.py"),
        "10,20"
    );
    // `+` ToPrimitive: arrays/objects stringify.
    assert_eq!(run("'' + [1, 2, 3]"), "1,2,3");
    assert_eq!(run("String([1, 2] + [3, 4])"), "1,23,4");
    assert_eq!(run("({}) + '!'"), "[object Object]!");
    // instanceof on error objects.
    assert_eq!(
        run("try { null.x; } catch (e) { '' + (e instanceof TypeError); }"),
        "true"
    );
    assert_eq!(
        run("try { nope; } catch (e) { '' + (e instanceof ReferenceError); }"),
        "true"
    );
}

#[test]
fn labeled_loops_and_do_while() {
    // `continue label` to an outer loop.
    assert_eq!(
        run("let count = 0;
                 outer: for (let i = 0; i < 3; i++) {
                   for (let j = 0; j < 3; j++) {
                     if (j === 1) continue outer;
                     count++;
                   }
                 }
                 count"),
        "3"
    );
    // `break label` out of nested loops.
    assert_eq!(
        run("let hits = 0;
                 search: for (let i = 0; i < 5; i++) {
                   for (let j = 0; j < 5; j++) {
                     hits++;
                     if (i === 1 && j === 1) break search;
                   }
                 }
                 hits"),
        "7"
    );
    // do/while runs the body at least once.
    assert_eq!(
        run("let n = 0, s = 0; do { s += n; n++; } while (n < 4); s"),
        "6"
    );
    assert_eq!(run("let r = 0; do { r++; } while (false); r"), "1");
}

#[test]
fn for_of_for_in_switch() {
    // for-of over an array, a string, a Set, and a Map.
    assert_eq!(run("let s = 0; for (const x of [1, 2, 3]) s += x; s"), "6");
    assert_eq!(
        run("let r = ''; for (const c of 'abc') r += c + '.'; r"),
        "a.b.c."
    );
    assert_eq!(
        run("let s = 0; for (const v of new Set([1, 2, 3, 2])) s += v; s"),
        "6"
    );
    assert_eq!(
        run("let r = ''; for (const [k, v] of new Map([['a', 1], ['b', 2]])) r += k + v; r"),
        "a1b2"
    );
    // for-of with break/continue.
    assert_eq!(
        run("let s = 0; for (const x of [1, 2, 3, 4]) { if (x === 3) break; s += x; } s"),
        "3"
    );
    // for-in over object keys and array indices.
    assert_eq!(
        run("let r = ''; for (const k in { a: 1, b: 2 }) r += k; r"),
        "ab"
    );
    assert_eq!(
        run("let r = ''; for (const i in ['x', 'y', 'z']) r += i; r"),
        "012"
    );
    // for-of binding to an existing variable (no declaration).
    assert_eq!(run("let x; let s = 0; for (x of [10, 20]) s += x; s"), "30");
    // switch with fall-through and default.
    assert_eq!(
        run("function f(n) {
                   let r = '';
                   switch (n) {
                     case 1: r += 'one';
                     case 2: r += 'two'; break;
                     case 3: r += 'three'; break;
                     default: r += 'other';
                   }
                   return r;
                 }
                 f(1) + '|' + f(2) + '|' + f(3) + '|' + f(9)"),
        "onetwo|two|three|other"
    );
}

#[test]
fn destructuring() {
    // Array destructuring with defaults, holes, and rest.
    assert_eq!(run("let [a, b] = [1, 2]; a + b"), "3");
    assert_eq!(run("let [a, , c] = [1, 2, 3]; a + c"), "4");
    assert_eq!(run("let [a, b = 9] = [1]; a + b"), "10");
    assert_eq!(
        run("let [first, ...rest] = [1, 2, 3, 4]; rest.join(',')"),
        "2,3,4"
    );
    // Object destructuring with shorthand, rename, default, and rest.
    assert_eq!(run("let { x, y } = { x: 1, y: 2 }; x + y"), "3");
    assert_eq!(run("let { a: p, b: q } = { a: 10, b: 20 }; p + q"), "30");
    assert_eq!(run("let { m = 7 } = {}; m"), "7");
    assert_eq!(
        run("let { a, ...others } = { a: 1, b: 2, c: 3 }; Object.keys(others).join(',')"),
        "b,c"
    );
    // Nested.
    assert_eq!(run("let { p: { q } } = { p: { q: 42 } }; q"), "42");
    assert_eq!(run("let [[a], [b]] = [[1], [2]]; a + b"), "3");
    // Destructuring function parameters.
    assert_eq!(run("function f([a, b]) { return a * b; } f([3, 4])"), "12");
    assert_eq!(
        run("function g({ x, y }) { return x + y; } g({ x: 5, y: 6 })"),
        "11"
    );
    // Default and rest parameters.
    assert_eq!(run("function h(a, b = 10) { return a + b; } h(5)"), "15");
    assert_eq!(
        run("function r(...xs) { return xs.length; } r(1, 2, 3)"),
        "3"
    );
}

#[test]
fn maps_and_sets() {
    // Map: set/get/has/size/delete.
    assert_eq!(
        run("let m = new Map(); m.set('a', 1); m.set('b', 2); m.get('a') + m.get('b')"),
        "3"
    );
    assert_eq!(run("let m = new Map(); m.set('x', 1); m.has('x')"), "true");
    assert_eq!(run("let m = new Map(); m.set('x', 1); m.size"), "1");
    assert_eq!(
        run("let m = new Map(); m.set('x', 1); m.delete('x'); m.has('x')"),
        "false"
    );
    // set returns the map (chainable); overwriting a key keeps size.
    assert_eq!(
        run("let m = new Map(); m.set('a', 1); m.set('a', 9); m.get('a') + ':' + m.size"),
        "9:1"
    );
    // Map seeded from pairs.
    assert_eq!(
        run("let m = new Map([['a', 1], ['b', 2]]); m.get('b')"),
        "2"
    );
    // Set: add/has/size, dedup, and seeding from an array.
    assert_eq!(
        run("let s = new Set(); s.add(1); s.add(1); s.add(2); s.size"),
        "2"
    );
    assert_eq!(run("let s = new Set([1, 2, 2, 3]); s.size"), "3");
    assert_eq!(run("new Set([1, 2, 3]).has(2)"), "true");
    // forEach over a Map accumulating values.
    assert_eq!(
        run("let m = new Map([['a', 10], ['b', 20]]);
                 let t = 0; m.forEach(v => { t += v; }); t"),
        "30"
    );
    // typeof a Map is object.
    assert_eq!(run("typeof new Map()"), "object");
}

#[test]
fn more_string_methods() {
    assert_eq!(run("'hello world'.slice(0, 5)"), "hello");
    assert_eq!(run("'hello'.slice(-3)"), "llo");
    assert_eq!(run("'a,b,c'.split(',').join('|')"), "a|b|c");
    assert_eq!(run("'hello'.startsWith('he')"), "true");
    assert_eq!(run("'hello'.endsWith('lo')"), "true");
    assert_eq!(run("'a-b-a'.replace('a', 'X')"), "X-b-a"); // first only
    assert_eq!(run("'5'.padStart(3, '0')"), "005");
}

#[test]
fn more_array_methods() {
    assert_eq!(run("[1, 2, 3, 4].slice(1, 3).join(',')"), "2,3");
    assert_eq!(run("[1, 2].concat([3, 4], 5).join(',')"), "1,2,3,4,5");
    assert_eq!(run("[1, 2, 3].reverse().join(',')"), "3,2,1");
    assert_eq!(run("[1, 2, 3, 4].find(x => x > 2)"), "3");
    assert_eq!(run("[1, 2, 3, 4].findIndex(x => x > 2)"), "2");
    assert_eq!(run("[1, 2, 3].some(x => x === 2)"), "true");
    assert_eq!(run("[1, 2, 3].every(x => x > 0)"), "true");
    assert_eq!(run("[1, 2, 3].every(x => x > 1)"), "false");
    // Default sort (string order) and comparator sort.
    assert_eq!(run("[3, 1, 2].sort().join(',')"), "1,2,3");
    assert_eq!(run("[10, 9, 100].sort().join(',')"), "10,100,9"); // string order
    assert_eq!(
        run("[10, 9, 100].sort((a, b) => a - b).join(',')"),
        "9,10,100"
    );
    assert_eq!(run("[3, 1, 2].sort((a, b) => b - a).join(',')"), "3,2,1");
}

#[test]
fn higher_order_array_methods() {
    // map / filter / reduce with closures.
    assert_eq!(run("[1, 2, 3].map(x => x * 2).join(',')"), "2,4,6");
    assert_eq!(
        run("[1, 2, 3, 4].filter(x => x % 2 === 0).join(',')"),
        "2,4"
    );
    assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b, 0)"), "10");
    assert_eq!(run("[1, 2, 3, 4].reduce((a, b) => a + b)"), "10"); // no initial
    // forEach with a closed-over accumulator.
    assert_eq!(
        run("let total = 0; [10, 20, 30].forEach(x => { total += x; }); total"),
        "60"
    );
    // Chained, with a captured multiplier.
    assert_eq!(
        run("let k = 3;
                 [1, 2, 3, 4].filter(x => x > 1).map(x => x * k).reduce((a, b) => a + b, 0)"),
        "27"
    );
}

// --- A5: WebAssembly.Memory shares the byte store (#11) ------------------

/// Hand-assembled wasm module:
///   (module
///     (memory (export "mem") 1)
///     (func (export "store") (param i32 i32) local.get 0 local.get 1 i32.store)
///     (func (export "load")  (param i32) (result i32) local.get 0 i32.load))
fn mem_module_bytes() -> alloc::vec::Vec<u8> {
    let mut m = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    // Type section: type0 (i32,i32)->(), type1 (i32)->(i32)
    m.extend([
        0x01, 0x0b, 0x02, 0x60, 0x02, 0x7f, 0x7f, 0x00, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    ]);
    // Function section: func0:type0, func1:type1
    m.extend([0x03, 0x03, 0x02, 0x00, 0x01]);
    // Memory section: one memory, min 1
    m.extend([0x05, 0x03, 0x01, 0x00, 0x01]);
    // Export section: "mem" mem0, "store" func0, "load" func1
    m.extend([
        0x07, 0x16, 0x03, 0x03, b'm', b'e', b'm', 0x02, 0x00, 0x05, b's', b't', b'o', b'r', b'e',
        0x00, 0x00, 0x04, b'l', b'o', b'a', b'd', 0x00, 0x01,
    ]);
    // Code section: func0 stores, func1 loads
    m.extend([
        0x0a, 0x13, 0x02, // section, size, count
        0x09, 0x00, 0x20, 0x00, 0x20, 0x01, 0x36, 0x02, 0x00, 0x0b, // func0
        0x07, 0x00, 0x20, 0x00, 0x28, 0x02, 0x00, 0x0b, // func1
    ]);
    m
}

/// A module exporting one memory (`mem`, min 1 page) and one function
/// (`grow_store`, `(param i32) -> i32`) that grows linear memory by a page,
/// stores the parameter as a byte at address 70000 (inside the freshly-grown
/// region), and returns the new page count. Used by the T6 grow-during-call
/// regression. Hand-assembled because `wat_to_binary` only emits function
/// exports (not memory exports).
fn mem_grow_module_bytes() -> alloc::vec::Vec<u8> {
    let mut m = alloc::vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    // Type section: type0 (i32)->(i32)
    m.extend([0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f]);
    // Function section: func0:type0
    m.extend([0x03, 0x02, 0x01, 0x00]);
    // Memory section: one memory, min 1 page (no max).
    m.extend([0x05, 0x03, 0x01, 0x00, 0x01]);
    // Export section: "mem" mem0, "grow_store" func0
    m.extend([
        0x07, 0x14, 0x02, // section id, size, count
        0x03, b'm', b'e', b'm', 0x02, 0x00, // "mem" -> memory 0
        0x0a, b'g', b'r', b'o', b'w', b'_', b's', b't', b'o', b'r', b'e', 0x00,
        0x00, // "grow_store" -> func 0
    ]);
    // Code section: func0 grows by a page, stores param at 70000, returns size.
    m.extend([
        0x0a, 0x14, 0x01, // section id, size, count
        0x12, // body size (18 bytes)
        0x00, // 0 local declarations
        0x41, 0x01, // i32.const 1
        0x40, 0x00, // memory.grow
        0x1a, // drop (the old page count)
        0x41, 0xf0, 0xa2, 0x04, // i32.const 70000
        0x20, 0x00, // local.get 0
        0x3a, 0x00, 0x00, // i32.store8 align=0 offset=0
        0x3f, 0x00, // memory.size
        0x0b, // end
    ]);
    m
}

/// Renders `bytes` as a JS array literal (so a test can build a `Uint8Array`).
fn js_byte_array(bytes: &[u8]) -> alloc::string::String {
    let mut s = alloc::string::String::from("[");
    for (i, b) in bytes.iter().enumerate() {
        if i != 0 {
            s.push(',');
        }
        s.push_str(&alloc::format!("{b}"));
    }
    s.push(']');
    s
}

/// Runs `src` with the memory module's bytes pre-installed as the global
/// `MOD` (a JS array of byte numbers), returning the completion display.
fn run_wasm(src: &str) -> alloc::string::String {
    let combined = alloc::format!("const MOD = {}; {src}", js_byte_array(&mem_module_bytes()));
    let program = Parser::parse_program(&combined).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

#[test]
fn wasm_memory_shares_byte_store_with_js() {
    // A `Uint8Array` over `mem.buffer` sees a write the exported wasm fn made.
    assert_eq!(
        run_wasm(
            "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const u8 = new Uint8Array(mem.buffer);
                 inst.exports.store(16, 0x41);     // wasm writes mem[16] = 65
                 u8[16];"
        ),
        "65"
    );
}

#[test]
fn wasm_reads_js_write_before_call() {
    // A JS write through a view (before the call) is read back by wasm.
    assert_eq!(
        run_wasm(
            "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const u8 = new Uint8Array(mem.buffer);
                 u8[20] = 0;
                 // Store 0x12345678 LE at addr 20 from JS via DataView, then wasm loads it.
                 const dv = new DataView(mem.buffer);
                 dv.setInt32(20, 0x12345678, true);
                 inst.exports.load(20);"
        ),
        "305419896" // 0x12345678
    );
}

#[test]
fn wasm_memory_grow_keeps_same_buffer_object_and_shares() {
    // `mem.grow(1)` keeps the SAME ArrayBuffer object; a view over the grown
    // buffer works and still shares the store with wasm.
    assert_eq!(
        run_wasm(
            "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const before = mem.buffer;
                 const old = mem.grow(1);             // grow by one 64KiB page
                 const same = (mem.buffer === before);
                 const u8 = new Uint8Array(mem.buffer);
                 // Write near the top of the newly-grown region from wasm, read in JS.
                 inst.exports.store(70000, 0x7e);
                 [old, mem.buffer.byteLength, same, u8[70000]].join(',');"
        ),
        "1,131072,true,126"
    );
}

#[test]
fn wasm_exported_table_is_introspectable_from_js() {
    // A module exporting a `funcref` table (slot 0 = the exported `add`, slot 1
    // uninitialized) plus the `add` function. From JS, `tbl.get(0)` is the `add`
    // wrapper (callable), `tbl.length` is 2, and `tbl.get(1)` is null.
    // Hand-encoded: (type (i32 i32)->i32) (func add) (table 2 funcref)
    //   (elem (i32.const 0) 0) (export "tbl" table 0) (export "add" func 0)
    let module_bytes: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // header
        0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // type (i32 i32)->i32
        0x03, 0x02, 0x01, 0x00, // func 0: type 0
        0x04, 0x04, 0x01, 0x70, 0x00, 0x02, // table: funcref min 2
        0x07, 0x0d, 0x02, // exports: 2
        0x03, 0x74, 0x62, 0x6c, 0x01, 0x00, // "tbl" table 0
        0x03, 0x61, 0x64, 0x64, 0x00, 0x00, // "add" func 0
        0x09, 0x07, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00, // elem active t0 off0 [func0]
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b, // code: add
    ];
    let src = "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
         const t = inst.exports.tbl;
         const f = t.get(0);
         // slot 1 is an uninitialized funcref (null); `join` renders it as empty.
         [t.length, typeof f, f(20, 22), t.get(1) === null].join(',');";
    let combined = alloc::format!("const MOD = {}; {src}", js_byte_array(module_bytes));
    let program = Parser::parse_program(&combined).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(value),
        "2,function,42,true"
    );
}

/// Runs `src` with the module compiled from `wat` pre-installed as the global
/// `MOD` (a JS array of byte numbers), returning the completion display.
fn run_wasm_wat(wat: &str, src: &str) -> alloc::string::String {
    let bin = crate::wasm_spec::wat_to_binary(wat).expect("compile WAT");
    let combined = alloc::format!("const MOD = {}; {src}", js_byte_array(&bin));
    let program = Parser::parse_program(&combined).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

/// Like [`run_wasm`] but installs the bytes of [`mem_grow_module_bytes`].
fn run_wasm_grow(src: &str) -> alloc::string::String {
    let combined = alloc::format!(
        "const MOD = {}; {src}",
        js_byte_array(&mem_grow_module_bytes())
    );
    let program = Parser::parse_program(&combined).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

#[test]
fn wasm_grow_during_call_persists_new_page(/* T6 */) {
    // An export that GROWS memory by a page *inside the call* and then stores a
    // byte into the freshly-grown region. After the call, a JS `Uint8Array` over
    // `Memory.buffer` must observe both the larger byteLength and the new byte —
    // i.e. the boundary copy-out must not truncate to the pre-call size, and the
    // canonical store must be grown to match the instance's enlarged memory.
    assert_eq!(
        run_wasm_grow(
            "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const mem = inst.exports.mem;
                 const pages = inst.exports.grow_store(0x5a);   // store 90 at 70000
                 const u8 = new Uint8Array(mem.buffer);
                 [pages, mem.buffer.byteLength, u8[70000]].join(',');"
        ),
        "2,131072,90"
    );
}

#[test]
fn wasm_repeat_calls_reuse_instance_state() {
    // A mutable global counter incremented by an export must persist across
    // separate JS→wasm calls (proving the same cached module + carried-over
    // instance state are reused rather than re-initialized each call).
    let wat = "(module
            (global $c (mut i32) (i32.const 0))
            (func (export \"inc\") (result i32)
              (global.set $c (i32.add (global.get $c) (i32.const 1)))
              (global.get $c)))";
    assert_eq!(
        run_wasm_wat(
            wat,
            "const inst = new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array(MOD)));
                 const a = inst.exports.inc();   // 1
                 const b = inst.exports.inc();   // 2
                 const c = inst.exports.inc();   // 3
                 [a, b, c].join(',');"
        ),
        "1,2,3"
    );
}

// --- A6: embedder buffer-creation API (#11) -----------------------------

#[test]
fn embedder_array_buffer_from_bytes_round_trips() {
    // An ArrayBuffer built from owned bytes is visible to JS and round-trips:
    // JS reads the seeded bytes, mutates one, and the owned store reflects it.
    let mut interp = Interp::new();
    let buf = interp.array_buffer_from_bytes(&[10, 20, 30, 40]);
    interp.declare_global("buf", NanBox::handle(buf.to_raw()));
    let program = Parser::parse_program(
        "const v = new Uint8Array(buf); const sum = v[0]+v[1]+v[2]+v[3]; v[1] = 99; sum",
    )
    .expect("parse");
    let value = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(value), "100");
    // The owned store now reads back the JS mutation.
    let bytes_h = interp.array_buffer_bytes_handle(buf).expect("bytes");
    assert_eq!(interp.realm().bytes_at(bytes_h).unwrap(), &[10, 99, 30, 40]);
}

#[test]
#[allow(unsafe_code)] // wraps a leaked 'static region zero-copy (A6)
fn embedder_array_buffer_from_external_is_zero_copy() {
    // A `'static`/leaked external region wrapped zero-copy: a JS write through
    // a view changes the *external region itself* (proving no copy was made).
    let region: &'static mut [u8] = alloc::vec![0u8; 8].leak();
    region[0] = 1;
    let ptr = region.as_mut_ptr();
    let len = region.len();
    let mut interp = Interp::new();
    // SAFETY: `region` is a leaked `'static` allocation; it stays valid for the
    // realm's lifetime and is never aliased mutably elsewhere during the run.
    let buf = unsafe { interp.array_buffer_from_external(ptr, len, None) };
    interp.declare_global("ext", NanBox::handle(buf.to_raw()));
    let program = Parser::parse_program(
        "const v = new Uint8Array(ext); const seen = v[0]; v[3] = 222; v[7] = 111; seen",
    )
    .expect("parse");
    let value = interp.run(&program).expect("exec");
    // JS saw the externally-seeded byte...
    assert_eq!(interp.realm().to_display_string(value), "1");
    // ...and the external region itself observed the JS writes (zero-copy).
    assert_eq!(region[3], 222);
    assert_eq!(region[7], 111);
}

#[test]
#[allow(unsafe_code)] // wraps a leaked 'static region zero-copy (A6)
fn embedder_typed_array_over_external_buffer() {
    // `typed_array_over` builds a Float64 view over an external buffer; a JS
    // store is reflected in the raw region (decoded via the same view).
    let region: &'static mut [u8] = alloc::vec![0u8; 16].leak();
    let ptr = region.as_mut_ptr();
    let mut interp = Interp::new();
    // SAFETY: leaked `'static`, uniquely owned for the run.
    let buf = unsafe { interp.array_buffer_from_external(ptr, 16, None) };
    let view = interp.typed_array_over(buf, 8, 0, 2).expect("float64 view");
    // Write 3.5 into element 1 through the realm API, read it back.
    interp.realm_mut().typed_set(view, 1, NanBox::number(3.5));
    let back = interp.realm_mut().typed_get(view, 1).unwrap();
    assert_eq!(back.as_number(), Some(3.5));
    // The external region bytes are the IEEE-754 encoding of 3.5 at offset 8.
    assert_eq!(&region[8..16], &3.5f64.to_le_bytes());
}

#[test]
fn typed_array_view_ctor_validates_bounds() {
    // H2/T1: a length that overruns the buffer is a RangeError.
    assert_eq!(
        run("try{new Uint32Array(new ArrayBuffer(8),0,100);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // A misaligned byteOffset (not a multiple of the element size) is a RangeError.
    assert_eq!(
        run("try{new Uint16Array(new ArrayBuffer(8),1);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // A byteOffset past the buffer end is a RangeError.
    assert_eq!(
        run("try{new Uint8Array(new ArrayBuffer(4),8);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // A trailing-bytes length not divisible by the element size is a RangeError.
    assert_eq!(
        run("try{new Uint32Array(new ArrayBuffer(6));'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // A valid aligned view still constructs.
    assert_eq!(
        run("let v=new Uint16Array(new ArrayBuffer(8),2,3); v.length===3 && v.byteOffset===2"),
        "true"
    );
}

#[test]
fn dataview_ctor_and_access_validate_bounds() {
    // M1: an explicit byteLength past the buffer is rejected at construction.
    assert_eq!(
        run("try{new DataView(new ArrayBuffer(8),0,100);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // M1: a stored over-long length cannot be smuggled into an access either;
    // a valid view's out-of-range access still throws.
    assert_eq!(
        run(
            "try{new DataView(new ArrayBuffer(8),0,8).getInt32(6);'no'}catch(e){e instanceof RangeError}"
        ),
        "true"
    );
    // A negative offset is a RangeError.
    assert_eq!(
        run("try{new DataView(new ArrayBuffer(8),-1);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // A valid DataView access round-trips.
    assert_eq!(
        run("let dv=new DataView(new ArrayBuffer(8)); dv.setInt32(0,0x01020304); dv.getInt32(0)"),
        "16909060"
    );
}

#[test]
fn typed_set_same_kind_fast_path_and_overlap() {
    // Same-kind copy.
    assert_eq!(
        run(
            "let a=new Uint8Array([1,2,3,4]); let b=new Uint8Array([9,8]); a.set(b,1); a.join(',')"
        ),
        "1,9,8,4"
    );
    // Overlapping copy within the same backing buffer (sibling views).
    assert_eq!(
        run(
            "let buf=new ArrayBuffer(4); let full=new Uint8Array(buf); full.set([1,2,3,4]); \
                 let dst=new Uint8Array(buf,1,3); let src=new Uint8Array(buf,0,3); dst.set(src); full.join(',')"
        ),
        "1,1,2,3"
    );
    // Out-of-range set is a RangeError.
    assert_eq!(
        run("try{new Uint8Array(2).set([1,2,3]);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // T2: a saturated offset throws rather than panicking.
    assert_eq!(
        run("try{new Uint8Array(4).set([1,2],1e308);'no'}catch(e){e instanceof RangeError}"),
        "true"
    );
    // Different-kind set still coerces correctly (generic path).
    assert_eq!(
        run("let a=new Uint8Array(3); a.set(new Float64Array([1.9,2.9,3.9])); a.join(',')"),
        "1,2,3"
    );
}

#[test]
fn typed_fill_and_copy_within_all_kinds() {
    // fill on each element kind.
    for ctor in [
        "Int8Array",
        "Uint8Array",
        "Uint8ClampedArray",
        "Int16Array",
        "Uint16Array",
        "Int32Array",
        "Uint32Array",
        "Float32Array",
        "Float64Array",
    ] {
        assert_eq!(
            run(&alloc::format!(
                "let a=new {ctor}(4); a.fill(7,1,3); a.join(',')"
            )),
            "0,7,7,0",
            "fill failed for {ctor}"
        );
        assert_eq!(
            run(&alloc::format!(
                "let a=new {ctor}([1,2,3,4]); a.copyWithin(0,2); a.join(',')"
            )),
            "3,4,3,4",
            "copyWithin failed for {ctor}"
        );
    }
    // Uint8Clamped fill clamps.
    assert_eq!(
        run("let a=new Uint8ClampedArray(2); a.fill(300); a.join(',')"),
        "255,255"
    );
    // fill with negative bounds (count from the end).
    assert_eq!(
        run("let a=new Int32Array(5); a.fill(9,-2); a.join(',')"),
        "0,0,0,9,9"
    );
}

#[test]
fn typed_array_aliasing_sees_bulk_writes() {
    // A sibling view over the same buffer observes fill/set/copyWithin writes.
    assert_eq!(
        run(
            "let buf=new ArrayBuffer(8); let a=new Uint8Array(buf); let b=new Uint8Array(buf); \
                 a.fill(5); b.join(',')"
        ),
        "5,5,5,5,5,5,5,5"
    );
    assert_eq!(
        run(
            "let buf=new ArrayBuffer(4); let a=new Uint8Array(buf); let b=new Uint8Array(buf); \
                 a.set([10,20,30,40]); a.copyWithin(0,2); b.join(',')"
        ),
        "30,40,30,40"
    );
    // A DataView aliases a typed-array fill.
    assert_eq!(
        run(
            "let buf=new ArrayBuffer(4); let a=new Uint8Array(buf); let dv=new DataView(buf); \
                 a.fill(0xFF); dv.getUint32(0).toString(16)"
        ),
        "ffffffff"
    );
}

// --- Host-function registration API (`register_fn`, ROADMAP §4.0) -----------

#[test]
fn host_fn_basic_call_and_return() {
    let program = Parser::parse_program("addOne(41)").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("addOne", 1, |cx, _this, args| {
        let n = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))?;
        Ok(cx.number(n + 1.0))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "42");
}

#[test]
fn host_fn_typeof_name_length() {
    fn check(src: &str) -> String {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.register_global_fn("greet", 2, |cx, _t, _a| Ok(cx.undefined()));
        let v = interp.run(&program).expect("exec");
        interp.realm().to_display_string(v)
    }
    assert_eq!(check("typeof greet"), "function");
    assert_eq!(check("greet.name"), "greet");
    assert_eq!(check("greet.length"), "2");
    // `Function.prototype.toString` shape for a native.
    assert_eq!(
        check("greet.toString()"),
        "function greet() { [native code] }"
    );
}

#[test]
fn host_fn_array_set_writes_and_grows() {
    // A host fn writes array elements via array_set (in-range and past-the-end,
    // which grows the length); array_set on a non-array returns false.
    let mut interp = Interp::new();
    interp.register_global_fn("fill", 1, |cx, _t, args| {
        let a = args.first().copied().unwrap_or(cx.undefined());
        let ten = cx.number(10.0);
        let wrote = cx.array_set(a, 1, ten); // in range
        let twenty = cx.number(20.0);
        cx.array_set(a, 5, twenty); // past the end → grows to length 6
        let obj = cx.new_object();
        let non_array = cx.array_set(obj, 0, ten);
        Ok(cx.boolean(wrote && !non_array))
    });
    let program = Parser::parse_program(
        "var a = [1, 2, 3]; var ok = fill(a); ok + ',' + a.length + ',' + a[1] + ',' + a[5]",
    )
    .expect("parse");
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "true,6,10,20");
}

#[test]
fn host_fn_value_inspection_and_array_access() {
    // A host fn that inspects its argument via the Ctx introspection API and, for
    // an Array, sums its elements through array_len/array_get.
    let program = Parser::parse_program(
        "inspect([10,20,30]) + '|' + inspect('hi') + '|' + inspect(fn) + '|' + inspect({})",
    )
    .expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("fn", 0, |cx, _t, _a| Ok(cx.undefined()));
    interp.register_global_fn("inspect", 1, |cx, _t, args| {
        let v = args.first().copied().unwrap_or(cx.undefined());
        if cx.is_array(v) {
            let n = cx.array_len(v).unwrap_or(0);
            let mut sum = 0.0;
            for i in 0..n {
                let e = cx.array_get(v, i);
                sum += cx.to_number(e)?;
            }
            return Ok(cx.string(&alloc::format!("array[{n}]={sum}")));
        }
        let s = alloc::format!(
            "{}/callable={}/object={}",
            cx.type_of(v),
            cx.is_callable(v),
            cx.is_object(v)
        );
        Ok(cx.string(&s))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(v),
        "array[3]=60|string/callable=false/object=false|\
         function/callable=true/object=true|object/callable=false/object=true"
    );
}

#[test]
fn host_fn_property_api_has_delete_keys() {
    // A host fn exercising has / has_own / delete / own_keys on an object.
    let program = Parser::parse_program("var o = { a: 1, b: 2 }; probe(o)").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("probe", 1, |cx, _t, args| {
        let o = args.first().copied().unwrap_or(cx.undefined());
        // Inherited (`toString`) is found by `has` but not `has_own`.
        let has_inherited = cx.has(o, "toString");
        let has_own_inherited = cx.has_own(o, "toString");
        let keys_before = cx.own_keys(o).join(",");
        let deleted = cx.delete(o, "a");
        let has_a_after = cx.has(o, "a");
        let keys_after = cx.own_keys(o).join(",");
        Ok(cx.string(&alloc::format!(
            "{has_inherited}/{has_own_inherited}/{keys_before}/{deleted}/{has_a_after}/{keys_after}"
        )))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(v),
        "true/false/a,b/true/false/b"
    );
}

#[test]
fn host_fn_native_state_wrap() {
    // napi_wrap-style: attach opaque Rust state to a JS object, read it back.
    let mut interp = Interp::new();
    interp.register_global_fn("wrap", 1, |cx, _t, args| {
        let o = cx.new_object();
        let n = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))? as i64;
        cx.set_native_state(o, n);
        Ok(o)
    });
    interp.register_global_fn("unwrap", 1, |cx, _t, args| {
        let o = args.first().copied().unwrap_or(cx.undefined());
        let v = cx.native_state::<i64>(o).copied().unwrap_or(-1);
        Ok(cx.number(v as f64))
    });
    let p =
        Parser::parse_program("var w = wrap(42); [unwrap(w), typeof w].join(',')").expect("parse");
    let v = interp.run(&p).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "42,object");
}

#[test]
fn host_deferred_promise_settled_later() {
    use core::cell::RefCell;
    // A host fn hands JS a deferred promise; the host settles it from "outside"
    // after the run, and the reaction runs on the drained microtask queue.
    let token: alloc::rc::Rc<RefCell<Option<u32>>> = alloc::rc::Rc::new(RefCell::new(None));
    let mut interp = Interp::new();
    let tc = token.clone();
    interp.register_global_fn("later", 0, move |cx, _t, _a| {
        let (promise, tok) = cx.deferred()?;
        *tc.borrow_mut() = Some(tok);
        Ok(promise)
    });
    // All programs must outlive the interp (its lifetime param binds them), so
    // parse them up front.
    let p_attach = Parser::parse_program(
        "globalThis.__out='pending'; later().then(v=>{globalThis.__out='got:'+v}); 'ok'",
    )
    .expect("parse");
    let p_read = Parser::parse_program("globalThis.__out").expect("parse");

    // Attach a reaction; it must not run yet (promise still pending).
    let v = interp.run(&p_attach).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "ok");
    let v = interp.run(&p_read).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "pending");
    // Host settles the deferred promise from "outside"; the reaction drains.
    let tok = token.borrow().unwrap();
    interp
        .resolve_deferred(tok, NanBox::number(42.0))
        .expect("settle");
    let v = interp.run(&p_read).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "got:42");
    // The token is released — a second settle is a harmless no-op.
    interp
        .resolve_deferred(tok, NanBox::number(7.0))
        .expect("noop");
    let v = interp.run(&p_read).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "got:42");
}

#[test]
fn host_fn_persistent_handle_across_calls() {
    // A host fn pins an object in one call and reads it back in a later call via a
    // persistent handle — surviving the GC that runs between them.
    let mut interp = Interp::new();
    interp.register_global_fn("stash", 1, |cx, _t, args| {
        let o = cx.new_object();
        let v = args.first().copied().unwrap_or(cx.undefined());
        cx.set(o, "x", v);
        let idx = cx.persist(o);
        Ok(cx.number(f64::from(idx)))
    });
    interp.register_global_fn("readX", 1, |cx, _t, args| {
        let idx = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))? as u32;
        let o = cx.persistent(idx);
        cx.get(o, "x")
    });
    // Persist in one run; allocate garbage (to move the heap) in another; read back.
    let stash = Parser::parse_program("stash(99)").expect("parse");
    let idx = interp.run(&stash).expect("exec");
    let idx = interp.realm().to_display_string(idx);
    let churn =
        Parser::parse_program("var s=''; for (var i=0;i<500;i++) s+={a:i}.a;").expect("parse");
    interp.run(&churn).expect("exec");
    let read = Parser::parse_program(&alloc::format!("readX({idx})")).expect("parse");
    let v = interp.run(&read).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "99");
}

#[test]
#[cfg(feature = "std")]
fn host_fn_panic_is_trapped_as_error() {
    // A panicking host closure becomes a catchable JS Error rather than unwinding
    // across the engine, and the registry is not corrupted (a later call works).
    let mut interp = Interp::new();
    interp.register_global_fn("boom", 0, |_cx, _t, _a| panic!("host kaboom"));
    interp.register_global_fn("ok", 0, |cx, _t, _a| Ok(cx.number(42.0)));
    // (The trapped panic's message is captured by the test harness and shown only
    // if this test fails.)
    let program = Parser::parse_program(
        "var caught = false;\
         try { boom(); } catch (e) { caught = e instanceof Error; }\
         caught + ',' + ok()",
    )
    .expect("parse");
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "true,42");
}

#[test]
fn host_fn_register_constructor() {
    // A host constructor: `new Vec2(x,y)` binds a fresh `this`, the closure sets
    // fields, instanceof works via the auto-created prototype, and a prototype
    // method sees the instance.
    let mut interp = Interp::new();
    interp.register_global_constructor("Vec2", 2, |cx, this, args| {
        let x = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))?;
        let y = cx.to_number(args.get(1).copied().unwrap_or(cx.undefined()))?;
        let xv = cx.number(x);
        cx.set(this, "x", xv);
        let yv = cx.number(y);
        cx.set(this, "y", yv);
        Ok(cx.undefined())
    });
    let program = Parser::parse_program(
        "Vec2.prototype.sum = function () { return this.x + this.y; };\
         var v = new Vec2(3, 4);\
         [v.x, v.y, v.sum(), v instanceof Vec2, v.constructor === Vec2, typeof Vec2].join(',')",
    )
    .expect("parse");
    let val = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(val),
        "3,4,7,true,true,function"
    );
}

#[test]
fn host_fn_register_constructor_return_object_and_plain_call() {
    // The constructor return rule: a returned object replaces `this`. And a plain
    // `register_fn` remains non-constructable (`new` → TypeError).
    let mut interp = Interp::new();
    interp.register_global_constructor("Boxed", 1, |cx, _this, args| {
        let obj = cx.new_object();
        let v = args.first().copied().unwrap_or(cx.undefined());
        cx.set(obj, "wrapped", v);
        Ok(obj) // returned object wins over the fresh `this`
    });
    interp.register_global_fn("plain", 0, |cx, _t, _a| Ok(cx.undefined()));
    let program = Parser::parse_program(
        "var b = new Boxed(42);\
         var threw = false; try { new plain(); } catch (e) { threw = e instanceof TypeError; }\
         b.wrapped + ',' + threw",
    )
    .expect("parse");
    let val = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(val), "42,true");
}

#[test]
fn host_fn_set_property_invokes_setter() {
    // cx.set_property runs an inherited accessor setter (full [[Set]]); cx.set
    // writes an own data property, shadowing the accessor.
    let program = Parser::parse_program(
        "var log = [];\
         var o = Object.create({ set v(x) { log.push('setter:' + x); } });\
         writeVia(o); \
         log.join(',') + '|own=' + Object.prototype.hasOwnProperty.call(o, 'v')",
    )
    .expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("writeVia", 1, |cx, _t, args| {
        let o = args.first().copied().unwrap_or(cx.undefined());
        let five = cx.number(5.0);
        cx.set_property(o, "v", five)?; // runs the inherited setter, no own prop
        Ok(cx.undefined())
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "setter:5|own=false");
}

#[test]
fn host_fn_construct_reenters_js() {
    // A host fn `new`s a JS class via cx.construct and reads back a field, and
    // reports is_constructor for a class vs a plain function.
    let program = Parser::parse_program(
        "class Point { constructor(x, y) { this.x = x; this.y = y; } }\
         var arrow = () => {};\
         make(Point, arrow)",
    )
    .expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("make", 2, |cx, _t, args| {
        let ctor = args.first().copied().unwrap_or(cx.undefined());
        let arrow = args.get(1).copied().unwrap_or(cx.undefined());
        let three = cx.number(3.0);
        let four = cx.number(4.0);
        let p = cx.construct(ctor, &[three, four])?;
        let px = cx.get(p, "x")?;
        let py = cx.get(p, "y")?;
        let x = cx.to_number(px)?;
        let y = cx.to_number(py)?;
        Ok(cx.string(&alloc::format!(
            "{}+{}={}/class:{}/arrow:{}",
            x,
            y,
            x + y,
            cx.is_constructor(ctor),
            cx.is_constructor(arrow)
        )))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(v),
        "3+4=7/class:true/arrow:false"
    );
}

#[test]
fn host_fn_returns_promises() {
    // A host fn returns a resolved / rejected promise; JS observes them via
    // then/catch after the microtask drain, and is_promise inspects the value.
    let mut interp = Interp::new();
    interp.register_global_fn("asyncOk", 1, |cx, _t, args| {
        let v = args.first().copied().unwrap_or(cx.undefined());
        Ok(cx.resolved_promise(v))
    });
    interp.register_global_fn("asyncFail", 1, |cx, _t, args| {
        let v = args.first().copied().unwrap_or(cx.undefined());
        Ok(cx.rejected_promise(v))
    });
    interp.register_global_fn("isP", 1, |cx, _t, args| {
        let v = args.first().copied().unwrap_or(cx.undefined());
        Ok(cx.boolean(cx.is_promise(v)))
    });
    let program = Parser::parse_program(
        "var log=[];\
         asyncOk(7).then(v=>log.push('ok:'+v));\
         asyncFail('bad').catch(e=>log.push('rej:'+e));\
         log.push('sync:'+isP(asyncOk(1))+','+isP(5));\
         log",
    )
    .expect("parse");
    let v = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(v),
        "sync:true,false,ok:7,rej:bad"
    );
}

#[test]
fn host_fn_throw_is_catchable() {
    let program =
        Parser::parse_program("try { boom(); 'no' } catch (e) { e.message }").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("boom", 0, |cx, _t, _a| Err(cx.type_error("kaboom")));
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "kaboom");
}

#[test]
fn host_fn_receives_this_and_args() {
    // Installed as a method so `this` is the receiver object.
    let program =
        Parser::parse_program("var o = { x: 10 }; o.sum = theSum; o.sum(1, 2, 3)").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("theSum", 0, |cx, this, args| {
        let base = cx.get(this, "x")?;
        let mut total = cx.to_number(base)?;
        for a in args {
            total += cx.to_number(*a)?;
        }
        Ok(cx.number(total))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "16");
}

#[test]
fn host_fn_can_build_objects_and_arrays() {
    let program =
        Parser::parse_program("var p = makePair(3, 4); JSON.stringify(p)").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("makePair", 2, |cx, _t, args| {
        let a = args.first().copied().unwrap_or(cx.undefined());
        let b = args.get(1).copied().unwrap_or(cx.undefined());
        let obj = cx.new_object();
        cx.set(obj, "first", a);
        cx.set(obj, "second", b);
        let arr = cx.new_array(alloc::vec![a, b]);
        cx.set(obj, "both", arr);
        Ok(obj)
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(
        interp.realm().to_display_string(v),
        r#"{"first":3,"second":4,"both":[3,4]}"#
    );
}

#[test]
fn host_fn_can_reenter_js() {
    // The host function calls a JS callback passed as an argument (ctx.call).
    let program =
        Parser::parse_program("applyTwice(function (n) { return n * 2 }, 5)").expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("applyTwice", 2, |cx, _t, args| {
        let f = args.first().copied().unwrap_or(cx.undefined());
        let x = args.get(1).copied().unwrap_or(cx.undefined());
        let once = cx.call(f, cx.undefined(), &[x])?;
        cx.call(f, cx.undefined(), &[once])
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "20");
}

#[test]
fn host_fn_mutable_state_persists_across_calls() {
    let program = Parser::parse_program("counter(); counter(); counter()").expect("parse");
    let mut interp = Interp::new();
    let mut n = 0.0_f64;
    interp.register_global_fn("counter", 0, move |cx, _t, _a| {
        n += 1.0;
        Ok(cx.number(n))
    });
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "3");
}

#[test]
fn host_fn_new_is_type_error() {
    let program =
        Parser::parse_program("try { new plain(); 'no' } catch (e) { e instanceof TypeError }")
            .expect("parse");
    let mut interp = Interp::new();
    interp.register_global_fn("plain", 0, |cx, _t, _a| Ok(cx.undefined()));
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "true");
}

#[test]
fn host_fn_self_reentrancy_throws() {
    // A host function that calls *itself* while a call is in flight gets a
    // clean TypeError rather than aliasing its own &mut closure.
    let program = Parser::parse_program(
        "try { recur(); 'no' } catch (e) { String(e).indexOf('re-entrant') >= 0 }",
    )
    .expect("parse");
    let mut interp = Interp::new();
    // Capture the function value so the closure can call back into itself.
    let self_val = interp.register_fn("recur", 0, |cx, _t, _a| {
        let g = cx.global();
        let me = cx.get(g, "recur")?;
        cx.call(me, cx.undefined(), &[])
    });
    interp.declare_global("recur", self_val);
    let v = interp.run(&program).expect("exec");
    assert_eq!(interp.realm().to_display_string(v), "true");
}

// --- Trap-less Proxy forwarding through iteration (ROADMAP §3.7) ------------

#[test]
fn proxy_of_array_iterates_through_get() {
    // `[...proxy]` / `Array.from(proxy)` must read each index through the
    // proxy's `[[Get]]` (HasProperty must forward to the target), not snapshot
    // the target array and read holes.
    assert_eq!(run("JSON.stringify([...new Proxy([1,2,3],{})])"), "[1,2,3]");
    assert_eq!(
        run("JSON.stringify(Array.from(new Proxy([1,2,3],{})))"),
        "[1,2,3]"
    );
    // The `get` trap observes the numeric indices (not just `length`).
    assert_eq!(
        run(
            "var log=[]; var p=new Proxy([1,2,3],{get(t,k,r){if(typeof k==='string')log.push(k);return Reflect.get(t,k,r);}}); \
             [...p]; JSON.stringify(log.filter(k=>k==='0'||k==='1'||k==='2'))"
        ),
        r#"["0","1","2"]"#
    );
}

#[test]
fn proxy_of_array_generic_methods() {
    // Generic array-like algorithms applied to a proxy read through `[[Get]]`.
    assert_eq!(
        run(
            "var s=[];Array.prototype.forEach.call(new Proxy([4,5,6],{}),x=>s.push(x));JSON.stringify(s)"
        ),
        "[4,5,6]"
    );
    assert_eq!(
        run(r#"Array.prototype.join.call(new Proxy([7,8,9],{}), "-")"#),
        "7-8-9"
    );
    assert_eq!(
        run("Array.prototype.map.call(new Proxy([1,2,3],{}),x=>x*2).join(',')"),
        "2,4,6"
    );
    assert_eq!(
        run("Array.prototype.indexOf.call(new Proxy([1,2,3],{}), 2)"),
        "1"
    );
}

#[test]
fn prototype_methods_are_first_class_function_objects() {
    // A `<Ctor>.prototype.<method>` value reference is a real callable function
    // object with `typeof === "function"` and the spec `name`/`length` — not a
    // call-site-only dispatch. Covers Array/String/%TypedArray% prototypes.
    assert_eq!(run("typeof Array.prototype.map"), "function");
    assert_eq!(run("Array.prototype.map.name"), "map");
    assert_eq!(run("Array.prototype.map.length"), "1");
    assert_eq!(run("typeof String.prototype.slice"), "function");
    assert_eq!(run("String.prototype.slice.name"), "slice");
    assert_eq!(run("String.prototype.charAt.length"), "1");
    assert_eq!(
        run("typeof Object.getPrototypeOf(Uint8Array.prototype).map"),
        "function"
    );
    assert_eq!(
        run("Object.getPrototypeOf(Uint8Array.prototype).map.name"),
        "map"
    );

    // The value read from a live receiver is the very same function object as
    // the one on the prototype (identity, not a fresh per-read thunk).
    assert_eq!(run("[1,2].map === Array.prototype.map"), "true");
    assert_eq!(run("'ab'.slice === String.prototype.slice"), "true");

    // The materialized method is non-enumerable, writable, configurable — the
    // built-in method attributes ({ writable:true, enumerable:false,
    // configurable:true }).
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(Array.prototype,'map');\
             [d.writable,d.enumerable,d.configurable].join(',')"
        ),
        "true,false,true"
    );

    // The fast path `[].map(...)` is unchanged and still produces the same
    // result as the first-class function applied via `.call`.
    assert_eq!(run("[1,2,3].map(x=>x*2).join(',')"), "2,4,6");
    assert_eq!(
        run("Array.prototype.map.call([1,2,3],x=>x*2).join(',')"),
        "2,4,6"
    );
}

#[test]
fn prototype_methods_work_via_call_apply_reflect_bind() {
    // `.call` on an array-like object (no `Array` involved) runs the generic
    // algorithm on the ToObject'd `this`.
    assert_eq!(
        run("var al={0:'a',1:'b',length:2};\
             Array.prototype.map.call(al,x=>x+x).join(',')"),
        "aa,bb"
    );
    // `Reflect.apply` invokes the stored first-class method.
    assert_eq!(
        run("Reflect.apply(Array.prototype.slice,[1,2,3,4],[1,3]).join(',')"),
        "2,3"
    );
    // A mutating method (`push`) stored in a variable and applied via `.call`.
    assert_eq!(
        run("var p=Array.prototype.push;var a=[1];p.call(a,2,3);a.join(',')"),
        "1,2,3"
    );
    // `Function.prototype.call.bind(method)` — the classic uncurry-this idiom.
    assert_eq!(
        run(
            "var boundSlice=Function.prototype.call.bind(Array.prototype.slice);\
             boundSlice([9,8,7],1).join(',')"
        ),
        "8,7"
    );
    // String method via `.call` on a primitive `this`.
    assert_eq!(run("String.prototype.slice.call('hello',1,3)"), "el");
    // A %TypedArray% prototype method via `.call` on a typed-array receiver.
    assert_eq!(
        run("var m=Object.getPrototypeOf(Uint8Array.prototype).map;\
             Array.from(m.call(new Uint8Array([5,6]),x=>x+1)).join(',')"),
        "6,7"
    );
    // The array iterator method is first-class too.
    assert_eq!(
        run("Array.prototype[Symbol.iterator].call([5,6]).next().value"),
        "5"
    );
}

#[test]
fn proxy_object_spread_copies_own_enumerable() {
    // `{...proxy}` (CopyDataProperties) must enumerate the proxy's own keys
    // through the ownKeys/getOwnPropertyDescriptor/get protocol.
    assert_eq!(
        run("JSON.stringify({...new Proxy({a:1,b:2},{})})"),
        r#"{"a":1,"b":2}"#
    );
    // A proxy over an array spreads its indices (length is non-enumerable).
    assert_eq!(
        run("JSON.stringify({...new Proxy([1,2,3],{})})"),
        r#"{"0":1,"1":2,"2":3}"#
    );
    // Non-enumerable own properties are skipped; the get trap fires per key.
    assert_eq!(
        run(
            "var b={};Object.defineProperty(b,'h',{value:9,enumerable:false});b.x=1;\
             JSON.stringify({...new Proxy(b,{})})"
        ),
        r#"{"x":1}"#
    );
    // Symbol keys are copied too.
    assert_eq!(
        run("var s=Symbol();var o={[s]:42,a:1};var out={...new Proxy(o,{})};out[s]+','+out.a"),
        "42,1"
    );
}

#[test]
fn proxy_object_rest_destructuring() {
    // Object-rest patterns (binding, assignment target, param, generator) run
    // CopyDataProperties through the proxy protocol, and copy symbol keys.
    assert_eq!(
        run("var {...r} = new Proxy({a:1,b:2},{}); JSON.stringify(r)"),
        r#"{"a":1,"b":2}"#
    );
    assert_eq!(
        run("var {a, ...rest} = new Proxy({a:1,b:2,c:3},{}); a + '|' + JSON.stringify(rest)"),
        r#"1|{"b":2,"c":3}"#
    );
    assert_eq!(
        run("function f({...p}){ return JSON.stringify(p); } f(new Proxy({x:1,y:2},{}))"),
        r#"{"x":1,"y":2}"#
    );
    assert_eq!(
        run("var g; ({...g} = new Proxy({m:5},{})); JSON.stringify(g)"),
        r#"{"m":5}"#
    );
    // A symbol own key is copied by object rest (was previously dropped).
    assert_eq!(
        run("var s=Symbol(); var {...q} = {[s]:7, a:1}; q[s] + ',' + q.a"),
        "7,1"
    );
    // Generator with yield through an object-rest destructuring over a proxy.
    assert_eq!(
        run(
            "function* gen(){ var {...z} = new Proxy({p:1,q:2},{}); yield z; } JSON.stringify(gen().next().value)"
        ),
        r#"{"p":1,"q":2}"#
    );
}

#[test]
fn proxy_json_and_descriptors() {
    // JSON.stringify enumerates a proxy through its ownKeys/get protocol; a
    // proxy over an array serializes as an array (IsArray unwraps the proxy).
    assert_eq!(
        run("JSON.stringify(new Proxy({a:1,b:2},{}))"),
        r#"{"a":1,"b":2}"#
    );
    assert_eq!(run("JSON.stringify(new Proxy([1,2,3],{}))"), "[1,2,3]");
    assert_eq!(
        run("JSON.stringify({x:new Proxy([1,2],{}),y:new Proxy({z:3},{})})"),
        r#"{"x":[1,2],"y":{"z":3}}"#
    );
    // An ownKeys trap restricts the serialized keys.
    assert_eq!(
        run(
            "JSON.stringify(new Proxy({a:1,b:2},{ownKeys(){return ['a'];},\
             getOwnPropertyDescriptor(t,k){return {value:t[k],enumerable:true,configurable:true};}}))"
        ),
        r#"{"a":1}"#
    );
    // Object.getOwnPropertyDescriptors drives the proxy's [[GetOwnProperty]].
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyDescriptors(new Proxy({a:1},{})))"),
        r#"{"a":{"value":1,"writable":true,"enumerable":true,"configurable":true}}"#
    );
}

#[test]
fn proxy_reflect_ownkeys_trapless_forwards() {
    // Reflect.ownKeys on a trap-less proxy forwards [[OwnPropertyKeys]] to the
    // target (was returning []).
    assert_eq!(
        run("JSON.stringify(Reflect.ownKeys(new Proxy({a:1,b:2},{})))"),
        r#"["a","b"]"#
    );
    // An array proxy reports indices then "length".
    assert_eq!(
        run("JSON.stringify(Reflect.ownKeys(new Proxy([9,8],{})))"),
        r#"["0","1","length"]"#
    );
    // A defined ownKeys trap still drives the result.
    assert_eq!(
        run("JSON.stringify(Reflect.ownKeys(new Proxy({a:1},{ownKeys(){return ['x','y'];}})))"),
        r#"["x","y"]"#
    );
}

#[test]
fn proxy_as_set_target_delegates() {
    // Object.assign onto a trap-less proxy target forwards writes to the target
    // (was throwing "object is not extensible" from the cell-level gate).
    assert_eq!(
        run("var p=new Proxy({},{});Object.assign(p,{x:1,y:2});p.x+','+p.y"),
        "1,2"
    );
    // A set trap on the target fires (receiver forwarding aside).
    assert_eq!(
        run(
            "var log=[];var p=new Proxy({},{set(t,k,v,r){log.push(k);t[k]=v;return true;}});\
             Object.assign(p,{a:1,b:2});log.join(',')"
        ),
        "a,b"
    );
    // A genuinely frozen (non-proxy) target still throws.
    assert_eq!(
        run("try{Object.assign(Object.freeze({}),{x:1});'no'}catch(e){e instanceof TypeError}"),
        "true"
    );
}

#[test]
fn proxy_reflect_set_receiver_semantics() {
    // The canonical passthrough set trap `set(t,k,v,r){return Reflect.set(t,k,v,r)}`
    // (receiver = the proxy) writes to the target via [[DefineOwnProperty]] on the
    // receiver, without recursing into the set trap.
    assert_eq!(
        run("var q=new Proxy({},{set(t,k,v,r){return Reflect.set(t,k,v,r);}});q.z=9;q.z"),
        "9"
    );
    assert_eq!(
        run(
            "var n=0;var p=new Proxy({},{set(t,k,v,r){n++;return Reflect.set(t,k,v,r);}});\
             p.a=1;p.a+','+n"
        ),
        "1,1"
    );
    // Reflect.set on a proxy target (receiver defaults to the proxy) works.
    assert_eq!(run("var p=new Proxy({},{});Reflect.set(p,'x',5);p.x"), "5");
    // An ordinary receiver still receives the write.
    assert_eq!(run("var o={};Reflect.set({},'k',7,o);o.k"), "7");
    // A non-writable data property on the target rejects the receiver write.
    assert_eq!(
        run(
            "var t={};Object.defineProperty(t,'r',{value:1,writable:false});\
             var p=new Proxy(t,{});Reflect.set(t,'r',2,p)"
        ),
        "false"
    );
    // A setter found on the chain runs with the receiver as `this`.
    assert_eq!(
        run(
            "var got;var p;var t={set v(x){got=(this===p);}};p=new Proxy(t,{});\
             Reflect.set(t,'v',3,p);got"
        ),
        "true"
    );
}

#[test]
fn array_of_honors_constructor_receiver() {
    // Array.of with a constructor `this` builds via Construct + CreateDataProperty.
    assert_eq!(
        run("class S extends Array{}; var r=Array.of.call(S,1,2,3); (r instanceof S)+','+r.length"),
        "true,3"
    );
    assert_eq!(
        run("class S extends Array{}; var r=S.of(9,8); (r instanceof S)+','+r[0]"),
        "true,9"
    );
    // A custom constructor receives the element count and the defined elements.
    assert_eq!(
        run("var seen;var r=Array.of.call(function(n){seen=n;},10,20);seen+','+r[0]+','+r[1]"),
        "2,10,20"
    );
    // Default Array / non-constructor receiver still builds a plain array.
    assert_eq!(run("JSON.stringify(Array.of(1,2,3))"), "[1,2,3]");
    assert_eq!(run("var of=Array.of;JSON.stringify(of(5,6))"), "[5,6]");
}

#[test]
fn sparse_array_holes_in_haspropety_and_flat() {
    // `in` / HasProperty treat an array hole as absent (elision, delete, and
    // `new Array(n)` all create holes).
    assert_eq!(run("1 in [2,,3]"), "false");
    assert_eq!(run("var a=[2,9,3];delete a[1];1 in a"), "false");
    assert_eq!(run("0 in new Array(3)"), "false");
    assert_eq!(run("[2,,3].hasOwnProperty(1)"), "false");
    // Generic iteration that probes presence with HasProperty skips holes.
    assert_eq!(run("var c=0;[2,,3].forEach(()=>c++);c"), "2");
    assert_eq!(
        run("var r=[1,,3].map(x=>x*2);(1 in r)+','+r.length"),
        "false,3"
    );
    // Array.prototype.flat/flatMap skip holes (FlattenIntoArray uses HasProperty).
    assert_eq!(run("[1,[2,,3]].flat().join(',')"), "1,2,3");
    assert_eq!(run("var r=[1,,2].flat();r.length+','+r.join(',')"), "2,1,2");
    assert_eq!(run("[1,2,3].flatMap(x=>x===2?[]:[x]).join(',')"), "1,3");
    // Object.keys unaffected (already hole-aware).
    assert_eq!(run("JSON.stringify(Object.keys([2,,3]))"), r#"["0","2"]"#);
}

#[test]
fn flat_flatmap_skip_absent_array_like_indices() {
    // FlattenIntoArray uses HasProperty: an absent generic-array-like index is
    // skipped, and a poisoned getter past `length` is never read.
    assert_eq!(
        run("var a={length:3,0:1,2:21,get 3(){throw 'no'}};\
             JSON.stringify([].flatMap.call(a,function(e){return [39,e*2];}))"),
        "[39,2,39,42]"
    );
    assert_eq!(
        run("var b={length:3,0:1,2:[2,3],get 3(){throw 'no'}};JSON.stringify([].flat.call(b))"),
        "[1,2,3]"
    );
    // Real arrays with holes unaffected.
    assert_eq!(run("JSON.stringify([1,,3].flatMap(x=>[x]))"), "[1,3]");
    assert_eq!(run("JSON.stringify([1,[2,,3]].flat())"), "[1,2,3]");
    // flatMap passes the source object as the 3rd callback argument.
    assert_eq!(
        run("[1,2].flatMap((x,i,arr)=>[arr.length]).join(',')"),
        "2,2"
    );
}

#[test]
fn flat_flatmap_array_species_create() {
    // flat/flatMap use ArraySpeciesCreate(O, 0): a non-constructor @@species is a
    // TypeError, a subclass species builds that subclass.
    assert_eq!(
        run(
            "try{var a=[1,[2]];a.constructor={[Symbol.species]:42};a.flat();'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "try{var a=[1,2];a.constructor={[Symbol.species]:42};a.flatMap(x=>[x]);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "class S extends Array{};var a=new S(1,[2,3]);var r=a.flat();(r instanceof S)+','+JSON.stringify(Array.from(r))"
        ),
        r#"true,[1,2,3]"#
    );
    // Normal behavior unchanged.
    assert_eq!(run("JSON.stringify([1,[2,[3]]].flat())"), "[1,2,[3]]");
    assert_eq!(run("JSON.stringify([1,[2,[3]]].flat(2))"), "[1,2,3]");
    assert_eq!(
        run("JSON.stringify([1,2].flatMap(x=>[x,x*10]))"),
        "[1,10,2,20]"
    );
    // A hole in a mapped array is skipped when flattened.
    assert_eq!(run("JSON.stringify([1].flatMap(x=>[x,,x*2]))"), "[1,2]");
}

#[test]
fn with_statement_consults_proxy_has_trap() {
    // The `with` object environment's HasBinding is a proxy-aware HasProperty, so
    // a `has` trap decides whether a name is a binding (and `get`/`set` traps run).
    assert_eq!(
        run("var p=new Proxy({x:5},{has(t,k){return k==='x'},get(){return 42}});with(p){x}"),
        "42"
    );
    assert_eq!(
        run("var attr=7;var p=new Proxy({},{has(){return false}});with(p){attr}"),
        "7"
    );
    // A trapless proxy forwards HasProperty to its target.
    assert_eq!(run("var p=new Proxy({y:9},{});with(p){y}"), "9");
    // @@unscopables still blocks a binding the proxy would otherwise provide.
    assert_eq!(
        run("var a=1;var p=new Proxy({a:5,[Symbol.unscopables]:{a:true}},{});with(p){a}"),
        "1"
    );
    // Plain-object `with` unaffected.
    assert_eq!(run("var o={a:1,b:2};with(o){a+b}"), "3");
}

#[test]
fn with_unscopables_non_object_is_ignored() {
    // A non-Object `@@unscopables` (`''` — a heap string, still a `Handle`) must
    // NOT block bindings: `toString` still resolves to the with-object's property.
    assert_eq!(
        run("var t={};var env={toString:t};env[Symbol.unscopables]='';\
             var r;with(env){r=(toString===t)};r"),
        "true"
    );
}

#[test]
fn generator_array_destructuring_closes_iterator_on_return() {
    // `[ {} = yield ] = iter` inside a generator: an `iter.return()` at the yield
    // is an abrupt completion that must `IteratorClose` the (not-done) iterator —
    // calling its `return()` once — and the destructuring must pull lazily (one
    // step per element) rather than draining an infinite iterator.
    assert_eq!(
        run("var nextCount=0,returnCount=0,unreachable=0;\
             var iterator={next:function(){nextCount++;return{done:false,value:undefined};},\
                           return:function(){returnCount++;return{};}};\
             var iterable={};iterable[Symbol.iterator]=function(){return iterator;};\
             function* g(){var result;result=[ {} = yield ] = iterable;unreachable++;}\
             var it=g();it.next();var r=it.return(777);\
             (nextCount===1 && returnCount===1 && unreachable===0 && r.value===777 && r.done===true)"),
        "true"
    );
}

#[test]
fn delete_sloppy_global_is_configurable() {
    // A sloppy assignment to an unresolvable reference creates a *configurable*
    // global-object property (not a declarative binding), so `delete` removes it
    // and the name becomes unresolvable again.
    assert_eq!(
        run("(function(){zzTemp=1;var d=delete zzTemp;var gone;\
              try{zzTemp;gone=false;}catch(e){gone=(e instanceof ReferenceError);}\
              return d===true && gone===true;})()"),
        "true"
    );
}

#[test]
fn parenthesized_optional_chain_call_preserves_this() {
    // `(a?.b)()` — a parenthesized optional chain is reference-transparent, so the
    // call keeps `this` = the member's base (unlike a bare `a?.b()` chain).
    assert_eq!(
        run("var a={b:function(){return this._b;},_b:{c:42}};\
             ((a?.b)().c===42 && (a?.b)?.().c===42)"),
        "true"
    );
    // A nullish base makes the parenthesized value `undefined`; the *outer* call
    // then throws (not part of the chain) — but an outer `?.()` short-circuits.
    assert_eq!(run("var a=null;((a?.b)?.()===undefined)"), "true");
}

#[test]
fn reflect_set_proxy_target_returns_trap_boolean() {
    // Reflect.set on a proxy target returns the [[Set]] boolean: a falsy `set`
    // trap result is `false` (not a throw), a truthy one is `true`.
    assert_eq!(
        run("Reflect.set(new Proxy({},{set(){return false}}),'a','x')"),
        "false"
    );
    assert_eq!(
        run("Reflect.set(new Proxy({},{set(){return null}}),'a','x')"),
        "false"
    );
    assert_eq!(
        run("Reflect.set(new Proxy({},{set(){return 0}}),'a','x')"),
        "false"
    );
    assert_eq!(
        run("Reflect.set(new Proxy({},{set(t,k,v){t[k]=v;return true}}),'a','x')"),
        "true"
    );
    // A trap-less proxy whose target is itself a proxy forwards [[Set]] to it.
    assert_eq!(
        run(
            "var log=[];var inner=new Proxy({},{set(t,k,v){log.push(k);t[k]=v;return true}});\
             var outer=new Proxy(inner,{});Reflect.set(outer,'z',1);log.join(',')"
        ),
        "z"
    );
    // Truthy trap over a non-writable target property (same value) succeeds.
    assert_eq!(
        run(
            "var tg={};Object.defineProperty(tg,'a',{value:1,writable:false,configurable:false});\
             Reflect.set(new Proxy(tg,{set(){return true}}),'a',1)"
        ),
        "true"
    );
}

#[test]
fn computed_set_walks_proxy_and_setter_prototype() {
    // OrdinarySet: a computed write (`o[k]=v`, `arr[i]=v`) whose own slot is
    // absent runs parent.[[Set]] — an inherited setter or a proxy prototype
    // handles it (dot-key `assign_member` already did; now the computed path too).
    assert_eq!(
        run(
            "globalThis.__log=[];var pr=new Proxy({},{set(t,k,v){__log.push(k+'='+v);return true}});\
             var o=Object.create(pr);o['x']=1;__log.join(',')+'|'+o.hasOwnProperty('x')"
        ),
        "x=1|false"
    );
    // An array index whose prototype is a proxy forwards to the trap.
    assert_eq!(
        run(
            "globalThis.__log=[];var pr=new Proxy({},{set(t,k,v){__log.push(k+'='+v);return true}});\
             var a=[];Object.setPrototypeOf(a,pr);a[0]=9;__log.join(',')+'|'+a.hasOwnProperty('0')"
        ),
        "0=9|false"
    );
    // An inherited index setter runs with the array as receiver.
    assert_eq!(
        run(
            "var got;var a=[];Object.setPrototypeOf(a,{set 0(v){got=v;}});a[0]=7;got+'|'+a.hasOwnProperty('0')"
        ),
        "7|false"
    );
    // Fast path preserved: default-prototype arrays, present indices, hole fills.
    assert_eq!(run("var a=[1,2,3];a[1]=9;a.join(',')"), "1,9,3");
    assert_eq!(
        run("var a=[1];a[1]=2;a[2]=3;a.join(',')+'|'+a.length"),
        "1,2,3|3"
    );
    assert_eq!(
        run("var a=[1,,3];a[1]=2;a.join(',')+'|'+a.hasOwnProperty('1')"),
        "1,2,3|true"
    );
    // A *present* own index on a proxy-proto array writes the own slot (no walk).
    assert_eq!(
        run(
            "globalThis.__log=[];var pr=new Proxy({},{set(t,k,v){__log.push(k);return true}});\
             var a=[1,2,3];Object.setPrototypeOf(a,pr);a[1]=9;a.join(',')+'|'+__log.length"
        ),
        "1,9,3|0"
    );
}

#[test]
fn function_constructor_tostring_coerces_arguments() {
    // CreateDynamicFunction ToString's each argument: a custom toString runs and
    // a thrown value propagates (not stringified to "[object Object]").
    assert_eq!(run("new Function({toString:()=>'a'},'return a')(5)"), "5");
    assert_eq!(
        run("try{new Function({toString:()=>{throw 1}},'return 1')();'no'}catch(e){e}"),
        "1"
    );
    assert_eq!(
        run("try{new Function('a',{toString:()=>{throw 'body'}});'no'}catch(e){e}"),
        "body"
    );
    // valueOf is the fallback when toString is not callable.
    assert_eq!(
        run("new Function({toString:null,valueOf:()=>'x'},'return x')(9)"),
        "9"
    );
    // Normal string arguments unaffected.
    assert_eq!(run("new Function('a','b','return a+b')(2,3)"), "5");
}

#[test]
fn string_replace_coerces_replacement_via_tostring() {
    // replace / replaceAll ToString the replacement value and the function
    // replacer's result (custom toString / @@toPrimitive runs; a throw propagates)
    // instead of rendering "[object Object]".
    assert_eq!(run("'aa'.replace('a',{toString:()=>'z'})"), "za");
    assert_eq!(
        run("'aa'.replaceAll('a',{[Symbol.toPrimitive]:()=>'z'})"),
        "zz"
    );
    assert_eq!(run("'aa'.replace('a',()=>({toString:()=>'Q'}))"), "Qa");
    assert_eq!(run("'aa'.replaceAll('a',()=>({toString:()=>'Q'}))"), "QQ");
    assert_eq!(
        run("try{'aa'.replace('a',{toString:()=>{throw 'x'}});'no'}catch(e){e}"),
        "x"
    );
    assert_eq!(
        run("try{'aa'.replaceAll('a',{toString:()=>{throw 'y'}});'no'}catch(e){e}"),
        "y"
    );
    // Normal string replacement and $-patterns unaffected.
    assert_eq!(run("'a.b'.replace('.','-')"), "a-b");
    assert_eq!(run("'abc'.replaceAll('b','[$&]')"), "a[b]c");
}

#[test]
fn string_raw_validates_template_and_raw() {
    // ToObject(template) + ToObject(Get(template,"raw")) are throwing ToObject,
    // and the `raw` Get fires an inherited getter (throw propagates).
    assert_eq!(run("String.raw`a${1}b`"), "a1b");
    assert_eq!(
        run("try{String.raw(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{String.raw({});'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{String.raw(null);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{String.raw({get raw(){throw 'x'}});'no'}catch(e){e}"),
        "x"
    );
    // Manual raw objects and array-like raw still work.
    assert_eq!(run("String.raw({raw:['a','b','c']},1,2)"), "a1b2c");
    assert_eq!(run("String.raw({raw:{length:2,0:'x',1:'y'}},9)"), "x9y");
}

#[test]
fn split_tostring_before_limit_zero() {
    // `ToString(separator)` (spec step 7) runs before the `lim === 0`
    // short-circuit (step 8), so a throwing separator toString throws even at
    // limit 0; an object separator is ToString'd.
    assert_eq!(
        run("try{'x'.split({toString:()=>{throw 's'}},0);'no'}catch(e){e}"),
        "s"
    );
    assert_eq!(
        run("JSON.stringify('axbxc'.split({toString:()=>'x'}))"),
        r#"["a","b","c"]"#
    );
    assert_eq!(run("JSON.stringify('a,b'.split(',',0))"), "[]");
    assert_eq!(
        run("JSON.stringify('a,b,c'.split(','))"),
        r#"["a","b","c"]"#
    );
    assert_eq!(run("JSON.stringify('x'.split(undefined))"), r#"["x"]"#);
    assert_eq!(run("JSON.stringify('ab'.split(''))"), r#"["a","b"]"#);
    assert_eq!(
        run("JSON.stringify('a1b2c'.split(/\\d/))"),
        r#"["a","b","c"]"#
    );
}

#[test]
fn reflect_set_trapless_chain_to_ordinary_getter_only() {
    // Reflect.set forwarding through a trap-less proxy chain to an ordinary
    // target: an inherited getter-only accessor fails (false); a setter runs
    // (true); a data write succeeds (true).
    assert_eq!(
        run("var re=/(?:)/g;var rp=new Proxy(new Proxy(re,{}),{});Reflect.set(rp,'global',true)"),
        "false"
    );
    assert_eq!(
        run(
            "var got;var o={set x(v){got=v}};var rp=new Proxy(new Proxy(o,{}),{});Reflect.set(rp,'x',5)+','+got"
        ),
        "true,5"
    );
    assert_eq!(
        run("var o={};var rp=new Proxy(new Proxy(o,{}),{});Reflect.set(rp,'y',9)+','+o.y"),
        "true,9"
    );
    // Plain assignment through a proxy-of-proxy still reaches a setter accessor.
    assert_eq!(
        run("var bar;var o={set bar(v){bar=v}};var p=new Proxy(new Proxy(o,{}),{});p.bar=1;bar"),
        "1"
    );
}

#[test]
fn promise_subclass_construction_and_statics() {
    // `class P extends Promise {}`: super(executor) runs the executor with a
    // callable resolve/reject and gives the instance promise state (via a hidden
    // backing-cell slot), so construction, static combinators, then/catch, and
    // await all work.
    assert_eq!(
        run(
            "class P extends Promise{};var g=[];new P((res,rej)=>{g.push(typeof res,typeof rej)});g.join(',')"
        ),
        "function,function"
    );
    assert_eq!(
        run("class P extends Promise{};P.resolve(1) instanceof P"),
        "true"
    );
    assert_eq!(
        run("class P extends Promise{};P.all([]) instanceof P"),
        "true"
    );
    assert_eq!(
        run("class P extends Promise{};P.race([P.resolve(1)]) instanceof P"),
        "true"
    );
    assert_eq!(
        run("class P extends Promise{};P.allSettled([]) instanceof P"),
        "true"
    );
    // A non-callable executor is a TypeError.
    assert_eq!(
        run("class P extends Promise{};try{new P(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Plain Promise unaffected.
    assert_eq!(run("Promise.resolve(1) instanceof Promise"), "true");
}

#[test]
fn promise_subclass_async_delivery() {
    // Resolution/rejection/await on a subclass promise deliver values through the
    // microtask queue (observed via console.log output after the drain).
    let out = |src: &str| {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    };
    assert_eq!(
        out("class P extends Promise{};P.resolve(5).then(x=>console.log('r:'+x));"),
        "r:5\n"
    );
    assert_eq!(
        out("class P extends Promise{};new P(res=>res(10)).then(x=>console.log('n:'+x));"),
        "n:10\n"
    );
    assert_eq!(
        out("class P extends Promise{};P.reject('e').catch(x=>console.log('c:'+x));"),
        "c:e\n"
    );
    assert_eq!(
        out(
            "class P extends Promise{};(async()=>{var v=await P.resolve(99);console.log('a:'+v);})();"
        ),
        "a:99\n"
    );
    assert_eq!(
        out(
            "class P extends Promise{};P.all([P.resolve(1),P.resolve(2)]).then(a=>console.log('all:'+a));"
        ),
        "all:1,2\n"
    );
}

#[test]
fn promise_symbol_species() {
    // Promise carries `get [Symbol.species]` (returns `this`), so a subclass
    // inherits it and `then`/`catch` produce subclass instances via
    // SpeciesConstructor.
    assert_eq!(run("Promise[Symbol.species]===Promise"), "true");
    assert_eq!(
        run("class P extends Promise{};P[Symbol.species]===P"),
        "true"
    );
    assert_eq!(
        run("class P extends Promise{};P.resolve(1).then(x=>x) instanceof P"),
        "true"
    );
    assert_eq!(
        run("class P extends Promise{};P.reject(1).catch(x=>x) instanceof P"),
        "true"
    );
    // An explicit `@@species` override is honored.
    assert_eq!(
        run(
            "class P extends Promise{static get [Symbol.species](){return Promise}};\
             var r=P.resolve(1).then(x=>x);(r instanceof Promise)+','+(r instanceof P)"
        ),
        "true,false"
    );
    // The accessor shape: { get, set: undefined, enumerable: false, configurable: true }.
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(Promise,Symbol.species);\
             (typeof d.get)+','+(d.set===undefined)+','+d.enumerable+','+d.configurable"
        ),
        "function,true,false,true"
    );
}

#[test]
fn arraybuffer_symbol_species() {
    // ArrayBuffer carries `get [Symbol.species]` (returns `this`); a subclass
    // inherits it; the accessor shape matches the spec.
    assert_eq!(run("ArrayBuffer[Symbol.species]===ArrayBuffer"), "true");
    assert_eq!(
        run("class B extends ArrayBuffer{};B[Symbol.species]===B"),
        "true"
    );
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(ArrayBuffer,Symbol.species);\
             (typeof d.get)+','+(d.set===undefined)+','+d.enumerable+','+d.configurable"
        ),
        "function,true,false,true"
    );
    // `slice` honors an explicit species override.
    assert_eq!(
        run(
            "class B extends ArrayBuffer{static get [Symbol.species](){return ArrayBuffer}};\
             new B(8).slice(0,4) instanceof ArrayBuffer"
        ),
        "true"
    );
}

#[test]
fn arraybuffer_slice_species_constructor() {
    // ArrayBuffer.prototype.slice allocates the result via SpeciesConstructor.
    assert_eq!(
        run(
            "class B extends ArrayBuffer{};var s=new B(8).slice(0,4);(s instanceof B)+','+s.byteLength"
        ),
        "true,4"
    );
    assert_eq!(
        run(
            "class B extends ArrayBuffer{static get [Symbol.species](){return ArrayBuffer}};\
             var s=new B(8).slice(0,4);(s instanceof ArrayBuffer)+','+(s instanceof B)"
        ),
        "true,false"
    );
    // A non-constructor species is a TypeError.
    assert_eq!(
        run(
            "class B extends ArrayBuffer{static get [Symbol.species](){return 42}};\
             try{new B(8).slice(0,4);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // Data is copied; plain ArrayBuffer slice unaffected.
    assert_eq!(
        run("var b=new ArrayBuffer(4);new Uint8Array(b).set([1,2,3,4]);\
             JSON.stringify(Array.from(new Uint8Array(b.slice(1,3))))"),
        "[2,3]"
    );
    assert_eq!(
        run("var s=new ArrayBuffer(8).slice(2,6);s.byteLength+','+(s instanceof ArrayBuffer)"),
        "4,true"
    );
}

#[test]
fn sort_indexed_properties_precise() {
    // Array.prototype.sort on an array with accessor/hole indices runs
    // SortIndexedProperties: index getters fire on read, setters on write-back,
    // and trailing holes are deleted.
    assert_eq!(
        run(
            "var log=[];var a=[3,1,2];Object.defineProperty(a,'1',{get(){log.push('g1');return 1},set(v){},configurable:true});a.sort();JSON.stringify(log)"
        ),
        r#"["g1"]"#
    );
    assert_eq!(
        run(
            "var log=[];var a=[3,1,2];Object.defineProperty(a,'0',{get(){return 5},set(v){log.push('s0:'+v)},configurable:true});a.sort();JSON.stringify(log)"
        ),
        r#"["s0:1"]"#
    );
    // A trailing hole is deleted (present count < length).
    assert_eq!(
        run("var a=[3,,1];a.sort();(2 in a)+','+a.length+','+a.join(',')"),
        "false,3,1,3,"
    );
    // Holes/undefined sort after present defined values; the extra hole is deleted.
    assert_eq!(
        run("var a=[3,,1,undefined,2];a.sort();a.join(',')+'|'+a.length+'|'+(4 in a)"),
        "1,2,3,,|5|false"
    );
    // comparefn validation happens on the precise path too.
    assert_eq!(
        run("var a=[3,,1];try{a.sort(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Dense arrays and typed arrays keep the fast path.
    assert_eq!(run("JSON.stringify([3,1,2].sort())"), "[1,2,3]");
    assert_eq!(run("JSON.stringify([10,2,1].sort((a,b)=>a-b))"), "[1,2,10]");
    assert_eq!(
        run("Array.from(new Uint8Array([3,1,2]).sort()).join(',')"),
        "1,2,3"
    );
}

#[test]
fn reverse_precise_for_hole_accessor_arrays() {
    // reverse on a hole/accessor array fires index getters/setters and preserves
    // holes by move-and-delete; dense/typed arrays keep the fast path.
    assert_eq!(
        run(
            "var log=[];var a=[1,2,3];Object.defineProperty(a,'0',{get(){log.push('g0');return 1},set(v){log.push('s0:'+v)},configurable:true,enumerable:true});a.reverse();JSON.stringify(log)"
        ),
        r#"["g0","s0:3"]"#
    );
    assert_eq!(
        run("var a=[1,,3];a.reverse();a.join(',')+'|'+(0 in a)+(1 in a)+(2 in a)"),
        "3,,1|truefalsetrue"
    );
    assert_eq!(run("JSON.stringify([1,2,3,4].reverse())"), "[4,3,2,1]");
    assert_eq!(run("JSON.stringify([1,2,3,4,5].reverse())"), "[5,4,3,2,1]");
    assert_eq!(
        run("Array.from(new Uint8Array([1,2,3]).reverse()).join(',')"),
        "3,2,1"
    );
}

#[test]
fn copy_within_precise_for_hole_accessor_arrays() {
    // copyWithin on a hole/accessor array copies through [[Get]]/[[Set]]/Delete
    // (getters/setters fire, holes propagate) with the correct overlap direction.
    assert_eq!(
        run(
            "var log=[];var a=[1,2,3,4];Object.defineProperty(a,'3',{get(){log.push('g3');return 4},configurable:true,enumerable:true});\
             Object.defineProperty(a,'0',{set(v){log.push('s0:'+v)},get(){return 1},configurable:true,enumerable:true});\
             a.copyWithin(0,3);JSON.stringify(log)"
        ),
        r#"["g3","s0:4"]"#
    );
    assert_eq!(
        run("var a=[1,2,3,4,5];delete a[3];a.copyWithin(0,3);a.join(',')+'|'+(0 in a)"),
        ",5,3,,5|false"
    );
    // Overlapping copy is not clobbered (backward direction).
    assert_eq!(
        run(
            "var a=[1,2,3,4,5];Object.defineProperty(a,'0',{value:1,writable:true,enumerable:true,configurable:true});JSON.stringify(a.copyWithin(1,0,3))"
        ),
        "[1,1,2,3,5]"
    );
    // Dense fast path unchanged.
    assert_eq!(
        run("JSON.stringify([1,2,3,4,5].copyWithin(0,3))"),
        "[4,5,3,4,5]"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4,5].copyWithin(1,0,3))"),
        "[1,1,2,3,5]"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4,5].copyWithin(-2,-3,-1))"),
        "[1,2,3,3,4]"
    );
}

#[test]
fn precise_readers_through_inherited_index() {
    // join / toString read a hole via [[Get]] — an inherited Array.prototype index
    // resolves through the prototype chain instead of rendering empty.
    assert_eq!(
        run("Array.prototype[1]=1;var x=[0];x.length=2;x.join()"),
        "0,1"
    );
    assert_eq!(
        run("Array.prototype[1]=1;var x=[0];x.length=2;x.toString()"),
        "0,1"
    );
    // toLocaleString invokes the *inherited* element's toLocaleString too (n===2).
    assert_eq!(
        run(
            "var n=0;var o={toLocaleString(){n++;return ''}};Array.prototype[1]=o;\
             var x=[o];x.length=2;x.toLocaleString();n"
        ),
        "2"
    );
    // slice reads the inherited hole via HasProperty+[[Get]] and creates an own
    // property in the copy.
    assert_eq!(
        run("Array.prototype[1]=1;var x=[0];x.length=2;var a=x.slice();\
             a[0]+','+a[1]+','+a.hasOwnProperty('1')"),
        "0,1,true"
    );
    // Dense fast path unchanged (no holes, no proto pollution).
    assert_eq!(run("[1,2,3].join(',')"), "1,2,3");
    assert_eq!(run("[1,2,3].toString()"), "1,2,3");
    assert_eq!(run("['a','b'].toLocaleString()"), "a,b");
    assert_eq!(run("[1,,3].join(',')"), "1,,3");
    assert_eq!(run("JSON.stringify([0,1,2,3].slice(1,3))"), "[1,2]");
}

#[test]
fn precise_length_mutators_through_prototype_and_frozen_length() {
    // pop reads the inherited hole value; shift moves it down; unshift shifts the
    // inherited index up — all via [[Get]]/[[Set]]/Delete through the prototype.
    assert_eq!(
        run("Array.prototype[1]=1;var x=[0];x.length=2;x.pop()"),
        "1"
    );
    assert_eq!(
        run("Array.prototype[1]=1;var x=[0];x.length=2;x.shift();x[0]"),
        "1"
    );
    assert_eq!(
        run("Array.prototype[0]=1;var x=[];x.length=1;x.unshift(0);x[0]+','+x[1]"),
        "0,1"
    );
    // pop on an empty frozen array: the closing Set(O,"length",…,true) throws even
    // though the value is unchanged (ordinary [[Set]] of a non-writable property
    // returns false regardless of same-value).
    assert_eq!(
        run("var a=[];Object.freeze(a);try{a.pop();'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("var a=[];Object.freeze(a);try{a.push(1);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // pop on `new Array(1)` whose inherited getter freezes mid-operation: the getter
    // fires once, then the length Set throws and length is unchanged.
    assert_eq!(
        run("var log=0;var a=new Array(1);\
             Object.defineProperty(Array.prototype,'0',{get(){Object.freeze(a);log++},configurable:true});\
             try{a.pop()}catch(e){};log+','+a.length"),
        "1,1"
    );
    // push on an array whose inherited Array.prototype[0] *setter* fires (freezing
    // the array), then the length Set throws — no own index property is created.
    assert_eq!(
        run(
            "var a=[];Object.defineProperty(Array.prototype,'0',{set(_v){Object.freeze(a)},configurable:true});\
             var t;try{a.push(1)}catch(e){t=e.constructor.name};t+','+a.hasOwnProperty(0)+','+a.length"
        ),
        "TypeError,false,0"
    );
    // An *own* accessor at an index still shadows an inherited one on write (the own
    // setter fires, not the prototype's).
    assert_eq!(
        run(
            "var hit=0;Object.defineProperty(Array.prototype,'0',{get(){return 11},configurable:true});\
             var a=[];Object.defineProperty(a,'0',{set(v){hit=v},get(){return hit},configurable:true});\
             a[0]=7;hit"
        ),
        "7"
    );
    // Dense fast path unchanged.
    assert_eq!(run("var a=[1,2,3];a.pop();JSON.stringify(a)"), "[1,2]");
    assert_eq!(run("var a=[1,2,3];a.shift();JSON.stringify(a)"), "[2,3]");
    assert_eq!(
        run("var a=[1,2,3];a.unshift(0);JSON.stringify(a)"),
        "[0,1,2,3]"
    );
    assert_eq!(
        run("var a=[1,2,3];a.push(4);JSON.stringify(a)"),
        "[1,2,3,4]"
    );
}

#[test]
fn sort_write_back_fires_inherited_setter() {
    // SortIndexedProperties write-back goes through [[Set]]: an inherited setter on
    // Object.prototype fires (no own property is created at that index), and the
    // inherited getter is read during collection.
    assert_eq!(
        run("var log=[];Object.defineProperty(Object.prototype,'2',\
               {get(){log.push('g');return 4},set(v){log.push('s'+v)},configurable:true});\
             var a=[undefined,3,,2,undefined,,1];a.sort();\
             log.join(',')+'|'+a[0]+a[1]+a[3]+'|'+a.hasOwnProperty('2')"),
        "g,s3|124|false"
    );
    // Dense sort unchanged.
    assert_eq!(run("JSON.stringify([3,1,2].sort())"), "[1,2,3]");
}

#[test]
fn splice_coerces_args_with_tointegerorinfinity() {
    // splice's start/deleteCount are ToIntegerOrInfinity (valueOf, not toString);
    // a throwing coercion propagates; Infinity clamps to the length.
    assert_eq!(
        run(
            "var x=[0,1,2,3];var a=x.splice(0,{valueOf:()=>3,toString:()=>0});a.length+'|'+JSON.stringify(a)"
        ),
        "3|[0,1,2]"
    );
    assert_eq!(
        run("var x=[0,1,2,3];x.splice({valueOf:()=>1},2);JSON.stringify(x)"),
        "[0,3]"
    );
    assert_eq!(
        run("var x=[1,2,3];try{x.splice({valueOf:()=>{throw 'e'}},1);'no'}catch(e){e}"),
        "e"
    );
    assert_eq!(
        run("var x=[1,2,3,4];x.splice(1,Infinity);JSON.stringify(x)"),
        "[1]"
    );
    // Ordinary splice unaffected.
    assert_eq!(
        run("var a=[1,2,3,4,5];a.splice(1,2,'a','b','c');JSON.stringify(a)"),
        r#"[1,"a","b","c",4,5]"#
    );
    assert_eq!(
        run("var a=[1,2,3,4,5];JSON.stringify(a.splice(-2,1))"),
        "[4]"
    );
}

#[test]
fn array_index_args_use_tointegerorinfinity() {
    // fill/flat/copyWithin coerce their index/depth args via ToIntegerOrInfinity
    // (valueOf, not the string form); a throwing coercion propagates.
    assert_eq!(
        run("JSON.stringify([1,2,3,4].fill(0,{valueOf:()=>2}))"),
        "[1,2,0,0]"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4].fill(0,1,{valueOf:()=>3}))"),
        "[1,0,0,4]"
    );
    assert_eq!(
        run("try{[1,2,3].fill(0,{valueOf:()=>{throw 'e'}});'no'}catch(e){e}"),
        "e"
    );
    assert_eq!(
        run("JSON.stringify([1,[2,[3]]].flat({valueOf:()=>2}))"),
        "[1,2,3]"
    );
    assert_eq!(
        run("try{[1,[2]].flat({valueOf:()=>{throw 'd'}});'no'}catch(e){e}"),
        "d"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4,5].copyWithin({valueOf:()=>1},3))"),
        "[1,4,5,4,5]"
    );
    assert_eq!(
        run("try{[1,2,3].copyWithin({valueOf:()=>{throw 'c'}},0);'no'}catch(e){e}"),
        "c"
    );
}

#[test]
fn slice_args_use_tointegerorinfinity() {
    // slice's start/end coerce via ToIntegerOrInfinity (valueOf); throws propagate.
    assert_eq!(
        run("JSON.stringify([1,2,3,4].slice({valueOf:()=>1}))"),
        "[2,3,4]"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4].slice(0,{valueOf:()=>2}))"),
        "[1,2]"
    );
    assert_eq!(
        run("try{[1,2,3].slice({valueOf:()=>{throw 's'}});'no'}catch(e){e}"),
        "s"
    );
    assert_eq!(run("JSON.stringify([1,2,3,4,5].slice(-2))"), "[4,5]");
    assert_eq!(run("JSON.stringify([1,2,3,4].slice(1,3))"), "[2,3]");
    assert_eq!(run("JSON.stringify([1,2,3].slice())"), "[1,2,3]");
}

#[test]
fn to_spliced_args_use_tointegerorinfinity() {
    // toSpliced (dense path) coerces start/deleteCount via ToIntegerOrInfinity.
    assert_eq!(
        run("JSON.stringify([1,2,3,4].toSpliced({valueOf:()=>1},1,'x'))"),
        r#"[1,"x",3,4]"#
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4].toSpliced(0,{valueOf:()=>2}))"),
        "[3,4]"
    );
    assert_eq!(
        run("try{[1,2,3].toSpliced({valueOf:()=>{throw 't'}},1);'no'}catch(e){e}"),
        "t"
    );
    assert_eq!(
        run("JSON.stringify([1,2,3,4].toSpliced(1,1,'x'))"),
        r#"[1,"x",3,4]"#
    );
}

#[test]
fn typed_array_from_array_like_uses_tolength() {
    // `new TypedArray(object)` with no @@iterator reads ToLength(Get(O,"length")):
    // an absent/NaN/negative length clamps to 0 (empty), not a RangeError.
    assert_eq!(run("new Uint8Array({}).length"), "0");
    assert_eq!(run("new Uint8Array({valueOf:()=>3}).length"), "0");
    assert_eq!(run("new Uint8Array({length:-1}).length"), "0");
    assert_eq!(run("new Uint8Array({length:NaN}).length"), "0");
    assert_eq!(
        run("Array.from(new Uint8Array({length:3,0:9})).join(',')"),
        "9,0,0"
    );
    assert_eq!(
        run("Array.from(new Uint8Array({length:{valueOf:()=>2},0:1,1:2})).join(',')"),
        "1,2"
    );
    // A Symbol length is a TypeError; the numeric-length path still RangeErrors.
    assert_eq!(
        run("try{new Uint8Array({length:Symbol()});'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(run("new Uint8Array(3).length"), "3");
    assert_eq!(
        run("try{new Uint8Array(-1);'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
}

#[test]
fn bigint_primitive_prototype_chain() {
    // A BigInt primitive's [[Prototype]] is %BigInt.prototype%, so property reads
    // resolve through it (fixing Object.prototype.toString and getPrototypeOf).
    assert_eq!(run("Object.prototype.toString.call(1n)"), "[object BigInt]");
    assert_eq!(
        run("Object.prototype.toString.call(10n)"),
        "[object BigInt]"
    );
    assert_eq!(run("Object.getPrototypeOf(1n)===BigInt.prototype"), "true");
    // Symbol primitive still works; a non-string @@toStringTag falls back to Object.
    assert_eq!(
        run("Object.prototype.toString.call(Symbol())"),
        "[object Symbol]"
    );
    assert_eq!(run("(255n).toString(16)"), "ff");
}

#[test]
fn string_exotic_own_index_and_length() {
    // A String object (primitive or wrapper) has own `length` and index
    // "0".."length-1" properties (StringGetOwnProperty), so hasOwnProperty / `in`
    // recognize them.
    assert_eq!(run("'abc'.hasOwnProperty(0)"), "true");
    assert_eq!(run("'abc'.hasOwnProperty('2')"), "true");
    assert_eq!(run("'abc'.hasOwnProperty('length')"), "true");
    assert_eq!(run("'abc'.hasOwnProperty(5)"), "false");
    assert_eq!(run("new String('abc').hasOwnProperty(1)"), "true");
    assert_eq!(
        run("var s=new String('ab');(0 in s)+','+(2 in s)"),
        "true,false"
    );
    // `in` on a string *primitive* is a TypeError (the RHS must be an object).
    assert_eq!(
        run("try{2 in 'ab';'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // A Number wrapper is not a String; arrays and plain objects unaffected.
    assert_eq!(run("new Number(5).hasOwnProperty(0)"), "false");
    assert_eq!(
        run("[1,2,3].hasOwnProperty(1)+','+[1,,3].hasOwnProperty(1)"),
        "true,false"
    );
    assert_eq!(
        run("({a:1}).hasOwnProperty('a')+','+({a:1}).hasOwnProperty('b')"),
        "true,false"
    );
}

#[test]
fn string_exotic_property_is_enumerable() {
    // A String object's index properties are enumerable; `length` is not.
    assert_eq!(run("'abc'.propertyIsEnumerable(0)"), "true");
    assert_eq!(run("'abc'.propertyIsEnumerable('length')"), "false");
    assert_eq!(run("'abc'.propertyIsEnumerable(5)"), "false");
    assert_eq!(run("new String('ab').propertyIsEnumerable(1)"), "true");
    assert_eq!(
        run("JSON.stringify(Object.keys('abc'))"),
        r#"["0","1","2"]"#
    );
    // Arrays and plain objects unaffected.
    assert_eq!(
        run("[1,2].propertyIsEnumerable(0)+','+[1,2].propertyIsEnumerable('length')"),
        "true,false"
    );
    assert_eq!(run("({a:1}).propertyIsEnumerable('a')"), "true");
}

#[test]
fn string_exotic_descriptors_and_own_names() {
    // getOwnPropertyDescriptor gives the spec String descriptor; getOwnPropertyNames
    // lists indices then length.
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyDescriptor('abc',0))"),
        r#"{"value":"a","writable":false,"enumerable":true,"configurable":false}"#
    );
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyDescriptor('abc','length'))"),
        r#"{"value":3,"writable":false,"enumerable":false,"configurable":false}"#
    );
    assert_eq!(run("Object.getOwnPropertyDescriptor('abc',5)"), "undefined");
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyNames('abc'))"),
        r#"["0","1","2","length"]"#
    );
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyNames(new String('ab')))"),
        r#"["0","1","length"]"#
    );
    // A wrapper's named own props come after the exotic keys.
    assert_eq!(
        run(
            "var s=new String('x');Object.defineProperty(s,'foo',{value:1});JSON.stringify(Object.getOwnPropertyNames(s))"
        ),
        r#"["0","length","foo"]"#
    );
    // Arrays and plain objects unaffected.
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyNames([1,2]))"),
        r#"["0","1","length"]"#
    );
    assert_eq!(
        run("JSON.stringify(Object.getOwnPropertyNames({a:1,b:2}))"),
        r#"["a","b"]"#
    );
}

#[test]
fn for_in_enumerates_string_indices() {
    // for-in over a String object yields its index keys "0".."length-1"
    // (`length` is non-enumerable); a wrapper's named own props follow.
    assert_eq!(
        run("var k=[];for(var i in 'abc')k.push(i);JSON.stringify(k)"),
        r#"["0","1","2"]"#
    );
    assert_eq!(
        run("var s=new String('ab');s.x=1;var k=[];for(var i in s)k.push(i);JSON.stringify(k)"),
        r#"["0","1","x"]"#
    );
    // Arrays and plain objects unaffected.
    assert_eq!(
        run("var k=[];for(var i in [10,20])k.push(i);JSON.stringify(k)"),
        r#"["0","1"]"#
    );
    assert_eq!(
        run("var k=[];for(var i in {a:1,b:2})k.push(i);JSON.stringify(k)"),
        r#"["a","b"]"#
    );
}

#[test]
fn error_subclass_constructor_is_the_subclass() {
    // A subclass of a native error resolves `.constructor` to the subclass (its
    // own/prototype-chain link), not the base error global — the error-name
    // fallback only fires when nothing before Object.prototype defines it.
    assert_eq!(
        run("class E extends Error{}new E('m').constructor===E"),
        "true"
    );
    assert_eq!(
        run("class E extends TypeError{}new E().constructor===E"),
        "true"
    );
    assert_eq!(
        run("class E extends Error{}class F extends E{}new F().constructor===F"),
        "true"
    );
    assert_eq!(
        run("class E extends Error{}var e=new E('hi');e.message+','+(e instanceof Error)"),
        "hi,true"
    );
    // Direct / thrown native errors still resolve to their own global.
    assert_eq!(run("new Error().constructor===Error"), "true");
    assert_eq!(run("new TypeError().constructor===TypeError"), "true");
    assert_eq!(run("new RangeError().constructor===RangeError"), "true");
    assert_eq!(
        run("try{null.x}catch(e){e.constructor===TypeError}"),
        "true"
    );
}

#[test]
fn weakref_and_finalization_registry_subclassing() {
    // `class W extends WeakRef {}`: super() validates + stamps the target, and
    // W.prototype links to WeakRef.prototype (so deref / brand / @@toStringTag work).
    assert_eq!(
        run(
            "class W extends WeakRef{}var o={};var w=new W(o);(w instanceof W)+','+(w instanceof WeakRef)+','+(w.deref()===o)"
        ),
        "true,true,true"
    );
    assert_eq!(
        run("class W extends WeakRef{}try{new W(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("class W extends WeakRef{}Object.prototype.toString.call(new W({}))"),
        "[object WeakRef]"
    );
    // FinalizationRegistry subclassing: super() validates the callback + brands.
    assert_eq!(
        run(
            "class F extends FinalizationRegistry{}var f=new F(()=>{});var o={},t={};f.register(o,5,t);f.unregister(t);(f instanceof F)+','+(f instanceof FinalizationRegistry)"
        ),
        "true,true"
    );
    assert_eq!(
        run("class F extends FinalizationRegistry{}try{new F(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Direct construction unaffected.
    assert_eq!(run("var o={};new WeakRef(o).deref()===o"), "true");
}

#[test]
fn aggregate_error_and_non_constructor_subclassing() {
    // AggregateError subclass: super(errors, message) drains the errors iterable
    // into an own `errors` array and takes the message second.
    assert_eq!(
        run(
            "class A extends AggregateError{}var a=new A([1,2,3],'m');(a instanceof A)+','+a.errors.join(',')+','+a.message"
        ),
        "true,1,2,3,m"
    );
    assert_eq!(
        run("class A extends AggregateError{}new A(new Set([9,8])).errors.join(',')"),
        "9,8"
    );
    assert_eq!(
        run("class A extends AggregateError{}try{new A(5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Symbol/BigInt are callable but not constructors: `new Subclass()` throws.
    assert_eq!(
        run("class S extends Symbol{}try{new S();'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("class B extends BigInt{}try{new B();'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // Direct AggregateError and ordinary error subclassing unaffected.
    assert_eq!(
        run("var a=new AggregateError([1,2],'x');a.errors.join(',')+','+a.message"),
        "1,2,x"
    );
    assert_eq!(run("class E extends Error{}new E('m').message"), "m");
}

#[test]
fn array_from_subclass_uses_dense_storage() {
    // `C.from(...)` (subclass constructor `this`) must CreateDataPropertyOrThrow
    // each element so it lands in the dense element store — a raw set_property
    // stashed them as named props that join/reduce/spread (dense reads) missed.
    assert_eq!(
        run("class A extends Array{}A.from([1,2,3]).join(',')"),
        "1,2,3"
    );
    assert_eq!(
        run("class A extends Array{}A.from([1,2,3],x=>x*2).join(',')"),
        "2,4,6"
    );
    assert_eq!(
        run("class A extends Array{}A.from(new Set([5,6,7])).join(',')"),
        "5,6,7"
    );
    assert_eq!(
        run("class A extends Array{}A.from([1,2,3,4]).reduce((a,b)=>a+b,0)"),
        "10"
    );
    assert_eq!(
        run("class A extends Array{}[...A.from([1,2,3])].join(',')"),
        "1,2,3"
    );
    assert_eq!(
        run("class A extends Array{}var a=A.from([1,2]);(a instanceof A)+','+Array.isArray(a)"),
        "true,true"
    );
    // Plain Array.from unaffected.
    assert_eq!(run("Array.from([1,2,3]).join(',')"), "1,2,3");
}

#[test]
fn live_set_map_iteration() {
    // Set/Map iterators are live: a mutation mid-iteration is observed (added
    // entries are visited, deleted ones skipped), per %MapIteratorPrototype%.
    assert_eq!(
        run("var s=new Set([1]);var o=[];for(var x of s){o.push(x);if(x===1)s.add(2)}o.join(',')"),
        "1,2"
    );
    assert_eq!(
        run(
            "var s=new Set([1,2,3]);var o=[];for(var x of s){o.push(x);if(x===1)s.delete(2)}o.join(',')"
        ),
        "1,3"
    );
    assert_eq!(
        run(
            "var m=new Map([[1,1],[2,2],[3,3]]);var o=[];for(var[k]of m){o.push(k);if(k===1)m.delete(2)}o.join(',')"
        ),
        "1,3"
    );
    // A manual iterator (via .values() and via [Symbol.iterator]) is live too.
    assert_eq!(
        run(
            "var s=new Set([1,2]);var it=s.values();var r=[it.next().value];s.add(3);var n;while(!(n=it.next()).done)r.push(n.value);r.join(',')"
        ),
        "1,2,3"
    );
    assert_eq!(
        run(
            "var s=new Set([1,2]);var it=s[Symbol.iterator]();var r=[it.next().value];s.add(3);var n;while(!(n=it.next()).done)r.push(n.value);r.join(',')"
        ),
        "1,2,3"
    );
    // Deleting the just-yielded key still advances to the successor.
    assert_eq!(
        run("var s=new Set([1,2,3]);var o=[];for(var x of s){o.push(x);s.delete(x)}o.join(',')"),
        "1,2,3"
    );
    // Once exhausted the iterator detaches (a later add is not resumed).
    assert_eq!(
        run(
            "var s=new Set([1]);var it=s[Symbol.iterator]();it.next();it.next();s.add(2);it.next().done"
        ),
        "true"
    );
    // An iterator is its own iterable; ordinary iteration/spread unaffected.
    assert_eq!(
        run("var it=new Set([1]).values();it[Symbol.iterator]()===it"),
        "true"
    );
    assert_eq!(run("[...new Set([1,2,3])].join(',')"), "1,2,3");
    assert_eq!(
        run("JSON.stringify([...new Map([['a',1],['b',2]])])"),
        r#"[["a",1],["b",2]]"#
    );
    assert_eq!(
        run("Object.prototype.toString.call(new Set([1]).values())"),
        "[object Set Iterator]"
    );
}

#[test]
fn string_symbol_method_only_for_object_arg() {
    // String.prototype.match/replace/search/split access @@match/@@replace/etc.
    // ONLY when the argument is an Object — a BigInt/Symbol primitive (whose
    // prototype may carry the symbol) is coerced to a string pattern instead.
    assert_eq!(
        run(
            "Object.defineProperty(BigInt.prototype,Symbol.match,{get(){throw new Error('x')},configurable:true});\
             var m='a1b1c'.match(1n);var r=m.index+','+JSON.stringify(m);delete BigInt.prototype[Symbol.match];r"
        ),
        r#"1,["1"]"#
    );
    assert_eq!(run("'a1b1c'.split(1n).join('|')"), "a|b|c");
    assert_eq!(run("'a1b1c'.replace(1n,'X')"), "aXb1c");
    // A RegExp or a custom-@@match object still delegates.
    assert_eq!(run("'a1b'.match(/\\d/)[0]"), "1");
    assert_eq!(run("'x'.match({[Symbol.match](s){return 'c:'+s}})"), "c:x");
}

#[test]
fn private_methods_install_after_super() {
    // Private methods/accessors are installed by InitializeInstanceElements —
    // after super() returns — so they're unreachable while a base constructor
    // runs (called via this.f() before super completes → TypeError).
    assert_eq!(
        run(
            "var C=class{constructor(){this.f()}};class D extends C{f(){this.#m()}#m(){return 42}}try{new D();'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "var C=class{constructor(){this.f()}};class D extends C{f(){return this.#x}get #x(){return 1}}try{new D();'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // Normal private-member behavior is intact.
    assert_eq!(
        run("class A{#m(){return 42}call(){return this.#m()}}new A().call()"),
        "42"
    );
    assert_eq!(
        run(
            "var C=class{constructor(){}};class D extends C{f(){return this.#m()}#m(){return 7}}new D().f()"
        ),
        "7"
    );
    assert_eq!(
        run(
            "class A{#a(){return 1}ga(){return this.#a()}}class B extends A{#b(){return 2}gb(){return this.#b()}}var b=new B();b.ga()+','+b.gb()"
        ),
        "1,2"
    );
    // A field initializer may reference this class's private method (installed first).
    assert_eq!(run("class A{#m(){return 9}x=this.#m()}new A().x"), "9");
    // Shared per class (c1.#m === c2.#m), static private, and `#x in o` still work.
    assert_eq!(
        run(
            "class A{#m(){}chk(o){return this.gm()===o.gm()}gm(){return this.#m}}var a=new A();a.chk(new A())"
        ),
        "true"
    );
    assert_eq!(
        run("class A{static #s=10;static get(){return A.#s}}A.get()"),
        "10"
    );
    assert_eq!(
        run("class A{#x=1;static has(o){return #x in o}}A.has(new A())+','+A.has({})"),
        "true,false"
    );
}

#[test]
fn static_private_not_inherited_by_subclass() {
    // Static private methods/getters/fields are OWN to their class and NOT
    // inherited: accessing one on a subclass constructor (whose [[Prototype]] is
    // the base) is a TypeError, even though the base holds the element.
    assert_eq!(
        run("class A{static #m(){return 42}static call(){return A.#m()}}A.call()"),
        "42"
    );
    assert_eq!(
        run(
            "class A{static #m(){return 1}static call(o){return o.#m()}}class B extends A{}try{A.call(B)}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "class A{static get #x(){return 1}static get(o){return o.#x}}class B extends A{}try{A.get(B)}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "class A{static #f=1;static get(o){return o.#f}}class B extends A{}try{A.get(B)}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // The declaring class itself still resolves it.
    assert_eq!(
        run("class A{static #m(){return 5}static call(o){return o.#m()}}A.call(A)"),
        "5"
    );
    // Instance privates unaffected.
    assert_eq!(
        run("class A{#m(){return 42}call(){return this.#m()}}new A().call()"),
        "42"
    );
    assert_eq!(
        run(
            "class A{#m(){return 1}call(o){return o.#m()}}try{new A().call({})}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
}

#[test]
fn direct_eval_sees_enclosing_private_names() {
    // A direct `eval` inside a method may reference the enclosing class's private
    // names — resolution walks the lexical class chain unchanged.
    assert_eq!(
        run(r#"class C{#m(){return "ok"}g(){return eval("this.#m()")}}new C().g()"#),
        "ok"
    );
    assert_eq!(
        run(r#"class C{#x=42;g(){return eval("this.#x")}}new C().g()"#),
        "42"
    );
    // A brand check still throws for a wrong receiver reached through eval.
    assert_eq!(
        run(
            r#"class C{#m(){return 1}g(){return eval("this.#m()")}}try{C.prototype.g.call({});'no'}catch(e){e.constructor.name}"#
        ),
        "TypeError"
    );
    // An indirect eval does NOT see the private names (SyntaxError at parse).
    assert_eq!(
        run(
            r#"var e=eval;class C{#m(){}g(){try{return e("this.#m")}catch(err){return err.constructor.name}}}new C().g()"#
        ),
        "SyntaxError"
    );
}

#[test]
fn direct_eval_arguments_in_field_initializer_is_early_error() {
    // ContainsArguments: `arguments` in a direct eval inside a field initializer
    // is a SyntaxError (thrown when the initializer runs).
    assert_eq!(
        run(r#"class C{x=eval("1;arguments;")}try{new C();'no'}catch(e){e.constructor.name}"#),
        "SyntaxError"
    );
    // Transparent through arrows inside the eval body.
    assert_eq!(
        run(r#"class C{x=eval("(()=>arguments)")}try{new C();'no'}catch(e){e.constructor.name}"#),
        "SyntaxError"
    );
    // And through arrows the initializer stores and invokes later.
    assert_eq!(
        run(
            r#"class C{x=()=>{var t=()=>eval("arguments");t()}}try{new C().x();'no'}catch(e){e.constructor.name}"#
        ),
        "SyntaxError"
    );
    // A nested *non-arrow* function shields its own `arguments` (no early error).
    assert_eq!(
        run(r#"class C{x=eval("(function(){return arguments.length})(1,2)")}new C().x"#),
        "2"
    );
    // Outside a field initializer, `arguments` in a direct eval is fine.
    assert_eq!(
        run(r#"function f(){return eval("arguments.length")}f(1,2,3)"#),
        "3"
    );
}

#[test]
fn private_element_double_initialization_throws() {
    // A base that returns the same object twice re-runs private installation on
    // it — PrivateMethodOrAccessorAdd / PrivateFieldAdd throw on the second pass.
    let dbl = |body: &str| {
        alloc::format!(
            "var o={{}};class B{{constructor(a){{return a}}}};class C extends B{{{body}}};new C(o);try{{new C(o);'no'}}catch(e){{e.constructor.name}}"
        )
    };
    assert_eq!(run(&dbl("#m(){}")), "TypeError");
    assert_eq!(run(&dbl("get #a(){return 1}")), "TypeError");
    assert_eq!(run(&dbl("set #a(v){}")), "TypeError");
    assert_eq!(run(&dbl("get #a(){return 1}\nset #a(v){}")), "TypeError");
    assert_eq!(run(&dbl("#f=1;")), "TypeError");
    // A get/set pair in one class body is a single element — first init is fine.
    assert_eq!(
        run("class C{get #a(){return 7}set #a(v){}rd(){return this.#a}}new C().rd()"),
        "7"
    );
}

#[test]
fn class_field_create_data_property_or_throw() {
    // A public field on a frozen `this` throws (CreateDataPropertyOrThrow).
    assert_eq!(
        run("class T{f=Object.freeze(this);g=1}try{new T();'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // A private field / method / accessor on a non-extensible instance throws.
    assert_eq!(
        run(
            "class B{constructor(s){if(s)Object.preventExtensions(this)}}class C extends B{#v=1;constructor(s){super(s)}}try{new C(true);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // Normal (extensible) construction is unaffected.
    assert_eq!(run("class C{a=1;b=2}var c=new C();c.a+','+c.b"), "1,2");
}

#[test]
fn native_error_subclass_message_own_property() {
    // `new Sub(msg)` installs an own `message`; `new Sub()` does not (it inherits
    // the prototype's).
    assert_eq!(
        run("class E extends TypeError{}new E('x').hasOwnProperty('message')"),
        "true"
    );
    assert_eq!(run("class E extends TypeError{}new E('x').message"), "x");
    assert_eq!(
        run("class E extends TypeError{}new E().hasOwnProperty('message')"),
        "false"
    );
    assert_eq!(
        run("class E extends Error{}E.prototype.message='d';new E().message"),
        "d"
    );
}

#[test]
fn private_access_on_primitive_receiver_throws() {
    // PrivateFieldGet/Set step 2: a private read/write with a primitive `this`
    // (reached via `method.call(primitive)`) is a TypeError.
    assert_eq!(
        run(
            "class C{#p=1;get(){return this.#p}}try{C.prototype.get.call(15);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "class C{#p=1;set(){this.#p=2}}try{C.prototype.set.call('s');'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // A normal primitive read still reports its wrapper constructor.
    assert_eq!(run("(5).constructor===Number"), "true");
}

#[test]
fn class_name_inner_binding_is_const() {
    // The class name is an immutable inner binding: reassigning it inside the
    // class body is a TypeError (both class expressions and declarations).
    assert_eq!(
        run(
            "var C=class Foo{m(){try{Foo=1;return 'assigned'}catch(e){return e.constructor.name}}};new C().m()"
        ),
        "TypeError"
    );
    assert_eq!(
        run(
            "class Bar{m(){try{Bar=1;return 'assigned'}catch(e){return e.constructor.name}}}new Bar().m()"
        ),
        "TypeError"
    );
    // A static initializer sees the same immutable binding.
    assert_eq!(
        run(
            "class Q{static x=(()=>{try{Q=1;return 'assigned'}catch(e){return e.constructor.name}})()}Q.x"
        ),
        "TypeError"
    );
    // The class name is still readable, and the OUTER declaration binding stays mutable.
    assert_eq!(run("class Baz{m(){return Baz.name}}new Baz().m()"), "Baz");
    assert_eq!(run("class D{};D=5;D"), "5");
    // Self-reference (recursion) still works.
    assert_eq!(
        run("class F{static go(n){return n<=0?0:1+F.go(n-1)}}F.go(3)"),
        "3"
    );
    assert_eq!(
        run("var f=class Rec{go(n){return n<=0?'done':this.go(n-1)}};new f().go(2)"),
        "done"
    );
}

#[test]
fn static_method_sees_class_name() {
    // A static method captures the class scope, so a named class expression can
    // reference itself by name from a static method (matching instance methods).
    assert_eq!(
        run("var f=class Rec{static typ(){return typeof Rec}};f.typ()"),
        "function"
    );
    assert_eq!(
        run("var f=class Rec{static go(n){return n<=0?'done':Rec.go(n-1)}};f.go(3)"),
        "done"
    );
    // The self-name is still the immutable const inside a static method.
    assert_eq!(
        run(
            "var f=class Rec{static m(){try{Rec=1;return 'assigned'}catch(e){return e.constructor.name}}};f.m()"
        ),
        "TypeError"
    );
    // Outer bindings still chain through; a static field initializer sees the name.
    assert_eq!(run("var y=99;class C{static m(){return y}}C.m()"), "99");
    assert_eq!(run("class Q{static x=Q.name}Q.x"), "Q");
}

#[test]
fn live_typed_array_iteration() {
    // Typed-array iterators are live: a resizable-buffer resize or an element
    // write mid-iteration is observed (per %ArrayIteratorPrototype% re-reading
    // the live length/elements). Plain arrays keep their snapshot fast path.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:16});var ta=new Uint8Array(rab);ta[0]=1;ta[1]=2;ta[2]=3;ta[3]=4;\
             var out=[];var i=0;for(var x of ta){out.push(x);if(i===0)rab.resize(8);i++}out.join(',')"
        ),
        "1,2,3,4,0,0,0,0"
    );
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(8,{maxByteLength:16});var ta=new Uint8Array(rab);for(var k=0;k<8;k++)ta[k]=k;\
             var out=[];var i=0;for(var x of ta){out.push(x);if(i===2)rab.resize(4);i++}out.join(',')"
        ),
        "0,1,2,3"
    );
    assert_eq!(
        run(
            "var ta=new Uint8Array([1,2,3,4]);var out=[];var i=0;for(var x of ta){out.push(x);if(i===0)ta[2]=64;i++}out.join(',')"
        ),
        "1,2,64,4"
    );
    // keys/values/entries + manual next + own-iterable, and spread, still work.
    assert_eq!(
        run("[...new Uint8Array([9,8,7]).keys()].join(',')"),
        "0,1,2"
    );
    assert_eq!(
        run("JSON.stringify([...new Uint8Array([5,6]).entries()])"),
        "[[0,5],[1,6]]"
    );
    assert_eq!(
        run("var it=new Uint8Array([1]).values();it[Symbol.iterator]()===it"),
        "true"
    );
    assert_eq!(run("[...new Float64Array([1.5,2.5])].join(',')"), "1.5,2.5");
    // A plain-array `for…of` is also **live**: the default `%ArrayIterator%`
    // re-reads `length` each step (`CreateArrayIterator`), so an element pushed
    // during the loop is visited (a value appended before the cursor reaches it).
    assert_eq!(
        run(
            "var a=[1,2,3];var out=[];var i=0;for(var x of a){out.push(x);if(i===0)a.push(99);i++}out.join(',')"
        ),
        "1,2,3,99"
    );
    // A `pop` during iteration contracts the live length (test262 for-of/array-contract).
    assert_eq!(
        run("var a=[0,1];var n=0;for(var x of a){a.pop();n++}n"),
        "1"
    );
}

#[test]
fn construct_returning_a_function_yields_that_function() {
    // ECMA-262 10.2.2 [[Construct]]: a constructor that returns an *object* makes
    // `new` yield that object — a returned *function* is an object too (test262
    // language/statements/function/S13.2.2_A8_*).
    assert_eq!(
        run(
            "function F(){this.a=1;g.b=2;return g;function g(x){return x+1}} var o=new F(); o.a===undefined && o.b===2 && o(4)===5"
        ),
        "true"
    );
    // A returned primitive is ignored; the fresh instance wins.
    assert_eq!(run("function F(){this.a=1;return 7} new F().a"), "1");
}

#[test]
fn var_for_in_of_binding_writes_the_hoisted_var() {
    // A `var` ForBinding has no per-iteration binding: it PutValue()s the hoisted
    // function-scope var, so the name survives the loop with the last value.
    assert_eq!(run("var a=5;for(var a in {x:1,y:2}){}a"), "y");
    assert_eq!(
        run("(function(){var a;for(var a of [1,2,3]){}return a})()"),
        "3"
    );
    // Redeclaring the catch parameter via a `var` for-in writes through to it
    // (test262 annexB try/catch-redeclared-for-in-var).
    assert_eq!(
        run(
            "var after;try{throw 'e'}catch(err){for(var err in {propertyName:null}){}after=err}after"
        ),
        "propertyName"
    );
    // A `let`/`const` head keeps its own per-iteration binding (unchanged).
    assert_eq!(run("var a=5;for(let a of [1,2]){}a"), "5");
}

#[test]
fn for_of_in_head_bound_names_are_in_tdz() {
    // ForIn/OfHeadEvaluation: the ForDeclaration's bound names are created in a
    // TDZ *before* the iterable/enumerated expression is evaluated, so a
    // reference to a bound name inside that expression throws a ReferenceError
    // (test262 for-of/for-in head-*-bound-names-fordecl-tdz).
    let is_ref_err = "e instanceof ReferenceError";
    assert_eq!(
        run(&alloc::format!(
            "var a=1;try{{let a=2;for(let a of [a]){{}}}}catch(e){{a={is_ref_err}}}a"
        )),
        "true"
    );
    assert_eq!(
        run(&alloc::format!(
            "var a=1;try{{let a=2;for(let a in {{a:a}}){{}}}}catch(e){{a={is_ref_err}}}a"
        )),
        "true"
    );
    // `const` head, same rule.
    assert_eq!(
        run(&alloc::format!(
            "var a=1;try{{for(const x of [x]){{}}}}catch(e){{a={is_ref_err}}}a"
        )),
        "true"
    );
    // A closure created in the head expression captures the TDZ binding, so a
    // later `typeof x` in it still throws (test262 scope-head-lex-open).
    assert_eq!(
        run(&alloc::format!(
            "var p;for(let x of (p=function(){{typeof x;}},[]));\
             var a=0;try{{p()}}catch(e){{a={is_ref_err}}}a"
        )),
        "true"
    );
    // Negatives: an expression that does not reference a bound name is fine, and
    // the per-iteration body binding is unaffected.
    assert_eq!(run("var s=0;for(let i of [1,2,3])s+=i;s"), "6");
    assert_eq!(run("var arr=[1,2];var n=0;for(let y of arr)n++;n"), "2");
    // A `var` head is *not* in a TDZ (the name is hoisted to `undefined`); the
    // array `[v]` reads that hoisted `undefined` without throwing.
    assert_eq!(run("var n=0;for(var v of [v]){n++}n"), "1");
    // Per-iteration binding still gives each closure its own value.
    assert_eq!(
        run("var f=[];for(let i of [0,1,2]){f.push(function(){return i})}f[0]()+f[1]()+f[2]()"),
        "3"
    );
}

#[test]
fn for_of_over_plain_array_is_live() {
    // test262 language/statements/for-of/array-expand and array-contract.
    assert_eq!(
        run("var a=[0];var n=0;var f=0,s=1;for(var x of a){f=s;s=null;if(f!==null)a.push(1);n++}n"),
        "2"
    );
    assert_eq!(
        run("var a=[0,1];var n=0;for(var x of a){a.pop();n++}n"),
        "1"
    );
}

#[test]
fn sloppy_let_as_identifier_statement() {
    // At a StatementListItem `let` heads a LexicalDeclaration only when a binding
    // follows; `let = 1`, `let;`, `let.x` are expression statements over the
    // ordinary identifier `let` (test262 for/head-lhs-let, for-in/head-lhs-let).
    assert_eq!(run("var let;let=1;let"), "1");
    assert_eq!(run("var let=3;let+4"), "7");
    assert_eq!(run("var let=1,a;for(let;;){a='ran';break}a"), "ran");
    // A real declaration still declares.
    assert_eq!(run("let x=5;x"), "5");
    // `async of => …` is an async arrow, valid as a classic-`for` initializer.
    assert_eq!(run("var i=0,c=0;for(async of => {}; i<3; ++i){c++}c"), "3");
    assert_eq!(run("typeof (async of => of)"), "function");
}

#[test]
fn assignment_resolves_lhs_reference_before_rhs() {
    // PutValue resolves the LHS reference *before* evaluating the RHS. A RHS that
    // deletes the `with`-object binding still writes through to that object
    // (SetMutableBinding recreates it in sloppy mode) — test262 S11.13.1_A5.
    assert_eq!(
        run("var x=0;var scope={x:1};with(scope){x=(delete scope.x,2)}''+scope.x+','+x"),
        "2,0"
    );
    // A RHS whose direct `eval` creates a shadowing `var` does not capture the
    // assignment: the LHS still names the outer binding — test262 S11.13.1_A6.
    assert_eq!(
        run("var x=0;var innerX=(function(){x=(eval('var x;'),1);return x})();''+innerX+','+x"),
        "undefined,1"
    );
    // The ordinary case is unaffected.
    assert_eq!(run("var a=1;a=a+1;a"), "2");
}

#[test]
fn annexb_html_like_comments() {
    // Annex B B.1.3: `<!--` opens a single-line comment anywhere; `-->` opens one
    // at a line start (or input start). `x-->0` (postfix decrement) is unaffected.
    assert_eq!(run("var y=1; <!-- comment\ny+5"), "6");
    assert_eq!(run("var z=10;\n--> comment\nz*2"), "20");
    assert_eq!(run("--> leading comment\n42"), "42");
    assert_eq!(run("var x=3; (x-->0)+','+x"), "true,2");
    assert_eq!(run("var a=5; (a --> 0)+','+a"), "true,4");
    assert_eq!(run("var q=1;<!--\nq=99;\nq"), "99");
    assert_eq!(run("1 // c\n+2"), "3");
    assert_eq!(run("1 /* c */ +2"), "3");
}

#[test]
fn regex_backspace_in_char_class() {
    // Inside a character class `\b` is a backspace (U+0008), not a word boundary.
    assert_eq!(run("/[\\b]/.test('\\b')"), "true");
    assert_eq!(run("/[\\b]/.test('b')"), "false");
    assert_eq!(
        run("/[a\\bc]/.test('\\b')+','+/[a\\bc]/.test('a')"),
        "true,true"
    );
    // Outside a class, `\b` is still a word boundary.
    assert_eq!(run("/\\bword\\b/.test('a word here')"), "true");
    // Other class escapes unaffected.
    assert_eq!(
        run("/[\\n\\t]/.test('\\n')+','+/[\\n\\t]/.test('\\t')"),
        "true,true"
    );
}

#[test]
fn annexb_legacy_octal_string_escapes() {
    // Annex B B.1.2 legacy octal string escapes (sloppy mode).
    assert_eq!(run("'\\101'"), "A"); // octal 101 = 65
    assert_eq!(run("'\\7'.charCodeAt(0)"), "7");
    assert_eq!(run("'\\12'.charCodeAt(0)"), "10"); // octal 12 = newline
    assert_eq!(run("'\\377'.charCodeAt(0)"), "255");
    assert_eq!(run("'\\0'.charCodeAt(0)"), "0");
    // A leading 0-3 admits 3 digits; a leading 4-7, two (so \400 = \40 + "0").
    assert_eq!(
        run("var s='\\400';s.length+','+s.charCodeAt(0)+','+s.charCodeAt(1)"),
        "2,32,48"
    );
    // \0 followed by a digit is octal 0 (NUL) then the literal digit.
    assert_eq!(
        run("var s='\\08';s.length+','+s.charCodeAt(0)+','+s.charCodeAt(1)"),
        "2,0,56"
    );
    // \8 and \9 are not octal; strict mode rejects legacy octal; \0 stays valid.
    assert_eq!(run("'\\8'"), "8");
    assert_eq!(
        run("try{eval(\"'use strict';'\\\\101'\");'no'}catch(e){e.constructor.name}"),
        "SyntaxError"
    );
    assert_eq!(
        run("(function(){'use strict';return '\\0'.charCodeAt(0)})()"),
        "0"
    );
}

#[test]
fn regex_legacy_octal_escapes() {
    // Annex B legacy octal escapes in a non-`u` regex: `\101` = char 65 ('A').
    // A numeric escape is a backreference only when it names an existing group
    // (pre-scanned total group count); otherwise it is a legacy octal escape.
    assert_eq!(
        run("/\\101/.test('A')+','+/\\101/.test('101')"),
        "true,false"
    );
    assert_eq!(run("/\\12/.test('\\n')"), "true"); // octal 12 = newline
    assert_eq!(run("/\\1/.test('\\x01')"), "true"); // no group → octal 1
    assert_eq!(run("/\\8/.test('8')+','+/\\9/.test('9')"), "true,true"); // not octal
    // A backreference still works (existing or forward group).
    assert_eq!(
        run("/(a)\\1/.test('aa')+','+/(a)\\1/.test('ab')"),
        "true,false"
    );
    assert_eq!(run("/(a)(b)\\2/.test('abb')"), "true");
    // Inside a class `\101` is octal (no backreferences there); ranges decode both ends.
    assert_eq!(
        run("/[\\101]/.test('A')+','+/[\\101]/.test('1')"),
        "true,false"
    );
    assert_eq!(
        run("/[\\101-\\103]/.test('B')+','+/[\\101-\\103]/.test('D')"),
        "true,false"
    );
    // `\0` NUL and word-boundary `\b` are unaffected.
    assert_eq!(run("/\\0/.test('\\0')"), "true");
    assert_eq!(run("/\\bword\\b/.test('a word')"), "true");
}

#[test]
fn intl_service_constructors_non_enumerable() {
    // ECMA-402: every Intl service constructor is a non-enumerable property of
    // Intl (writable:true, enumerable:false, configurable:true), so
    // Object.keys(Intl) is empty.
    assert_eq!(run("JSON.stringify(Object.keys(Intl))"), "[]");
    assert_eq!(
        run(
            "['NumberFormat','DateTimeFormat','Collator','PluralRules','ListFormat','RelativeTimeFormat','DisplayNames','Segmenter','Locale'].every(n=>!Object.getOwnPropertyDescriptor(Intl,n).enumerable)"
        ),
        "true"
    );
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(Intl,'NumberFormat');d.writable+','+d.configurable"
        ),
        "true,true"
    );
    // The constructors still work.
    assert_eq!(
        run("typeof new Intl.NumberFormat('en-US').format"),
        "function"
    );
}

#[test]
fn intl_formatter_subclassing() {
    // `class M extends Intl.NumberFormat {}` links M.prototype to
    // NumberFormat.prototype (instanceof), super() initializes the internal slots
    // (format/resolvedOptions work), and a Reflect.construct newTarget's prototype
    // is honored.
    assert_eq!(
        run("class M extends Intl.NumberFormat{}new M('en-US').format(1234.5)"),
        "1,234.5"
    );
    assert_eq!(
        run(
            "class M extends Intl.NumberFormat{}var f=new M('en');(f instanceof M)+','+(f instanceof Intl.NumberFormat)"
        ),
        "true,true"
    );
    assert_eq!(
        run("class M extends Intl.NumberFormat{}new M('en-US').resolvedOptions().locale"),
        "en-US"
    );
    assert_eq!(
        run(
            "class M extends Intl.NumberFormat{fmt2(x){return this.format(x)+'!'}}new M('en-US').fmt2(5)"
        ),
        "5!"
    );
    assert_eq!(
        run("class M extends Intl.NumberFormat{}new M('en-US',{style:'percent'}).format(0.5)"),
        "50%"
    );
    assert_eq!(
        run("class M extends Intl.DateTimeFormat{}typeof new M('en').format(new Date(0))"),
        "string"
    );
    assert_eq!(
        run(
            "function D(){}var f=Reflect.construct(Intl.NumberFormat,['en-US'],D);f.format(5)+','+(Object.getPrototypeOf(f)===D.prototype)"
        ),
        "5,true"
    );
    // Direct construction unaffected.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US').format(99)+','+(new Intl.NumberFormat() instanceof Intl.NumberFormat)"
        ),
        "99,true"
    );
}

#[test]
fn intl_number_format_signdisplay_negative_zero() {
    // `signDisplay: "always"` on -0 is "-0" (not "-+0"): the negative sign
    // replaces the positive one the +0 formatting produced.
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{signDisplay:'always'}).format(-0)"),
        "-0"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{signDisplay:'always'}).format(0)"),
        "+0"
    );
    assert_eq!(run("new Intl.NumberFormat('en-US').format(-0)"), "-0");
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{signDisplay:'never'}).format(-0)"),
        "0"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{signDisplay:'exceptZero'}).format(-0)"),
        "0"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{signDisplay:'always'}).format(-5)"),
        "-5"
    );
}

#[test]
fn intl_number_format_range() {
    // formatRange splices both endpoints into the locale's CLDR `range` pattern;
    // two ends that render alike collapse to the `approximately` form instead.
    assert_eq!(
        run("new Intl.NumberFormat('en-US').formatRange(3,5)"),
        "3\u{2013}5"
    );
    assert_eq!(run("new Intl.NumberFormat('en-US').formatRange(5,5)"), "~5");
    // The `$` affix is one code point, so ICU's AUTO collapse repeats it on both
    // ends and pads the separator.
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).formatRange(3,5)"),
        "$3.00 \u{2013} $5.00"
    );
    // A longer shared affix is factored out instead: "+$2.90–3.10".
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',signDisplay:'always'}).formatRange(2.9,3.1)"
        ),
        "+$2.90\u{2013}3.10"
    );
    // ToIntlMathematicalValue keeps a high-precision string endpoint exact (the
    // two values below are the same f64, so an f64 range would collapse them).
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US').formatRange('987654321987654321','987654321987654322')"
        ),
        "987,654,321,987,654,321\u{2013}987,654,321,987,654,322"
    );
    // Both arguments are required and finite.
    assert_eq!(
        run("try{new Intl.NumberFormat('en').formatRange(NaN,5);'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    assert_eq!(
        run(
            "try{new Intl.NumberFormat('en').formatRange(1,undefined);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // Structural: length 2, name, not a constructor, brand-checked receiver.
    assert_eq!(run("Intl.NumberFormat.prototype.formatRange.length"), "2");
    assert_eq!(
        run("Intl.NumberFormat.prototype.formatRange.name"),
        "formatRange"
    );
    assert_eq!(
        run(
            "try{Intl.NumberFormat.prototype.formatRange.call({},1,2);'no'}catch(e){e.constructor.name}"
        ),
        "TypeError"
    );
    // formatRangeToParts tags each endpoint's source.
    assert_eq!(
        run("new Intl.NumberFormat('en').formatRangeToParts(3,5).map(x=>x.source).join('|')"),
        "startRange|shared|endRange"
    );
    // …over *field-level* number parts, not one opaque literal per endpoint.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).formatRangeToParts(3,5).map(x=>x.type+':'+x.value+':'+x.source).join('|')"
        ),
        "currency:$:startRange|integer:3:startRange|literal: \u{2013} :shared|currency:$:endRange|integer:5:endRange"
    );
    // Two ends that render alike collapse to the all-shared `approximately` form.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).formatRangeToParts(2.999,3.001).map(x=>x.type+':'+x.value+':'+x.source).join('|')"
        ),
        "approximatelySign:~:shared|currency:$:shared|integer:3:shared"
    );
}

#[cfg(feature = "intl")]
#[test]
fn intl_datetime_format_range_to_parts() {
    // A DateTimeFormat range emits *field-level* parts (year/month/day/… with
    // `literal` separators), not one opaque literal per endpoint. A differing
    // range shows each endpoint's fields tagged startRange/endRange around a
    // shared separator.
    assert_eq!(
        run(
            "new Intl.DateTimeFormat('en-US').formatRangeToParts(new Date(Date.UTC(2019,0,3)),new Date(Date.UTC(2019,0,5))).map(p=>p.type).join('|')"
        ),
        "month|literal|day|literal|year|literal|month|literal|day|literal|year"
    );
    // Equal displayed fields collapse: every part is tagged "shared" and the
    // list is byte-for-byte formatToParts (same type/value sequence).
    assert_eq!(
        run(
            "var d=new Date(Date.UTC(2019,7,10)); var f=new Intl.DateTimeFormat('en',{year:'numeric',month:'short',day:'numeric'}); var r=f.formatRangeToParts(d,d); var p=f.formatToParts(d); r.every((e,i)=>e.type===p[i].type&&e.value===p[i].value&&e.source==='shared')&&r.length===p.length"
        ),
        "true"
    );
    // There is an inner `literal` immediately before the `dayPeriod` field (this
    // is what the resolved-time-zone tests probe; a single opaque literal used to
    // read past the array end and throw).
    assert_eq!(
        run(
            "var parts=new Intl.DateTimeFormat('en-US',{timeStyle:'short'}).formatRangeToParts(0,86400); parts.some((part,i)=>part.type==='literal'&&parts[i+1]&&parts[i+1].type==='dayPeriod')"
        ),
        "true"
    );
    // ToDateTimeFormattable coerces both operands (running valueOf) *before* the
    // SameTemporalType check: a NaN-returning valueOf paired with a Temporal
    // object still calls valueOf, then throws TypeError (not RangeError).
    assert_eq!(
        run(
            "var n=0; var bad={valueOf(){n++;return NaN;}}; try{new Intl.DateTimeFormat().formatRange(bad,new Temporal.PlainDate(1970,1,1));}catch(e){} n"
        ),
        "1"
    );
}

#[test]
fn intl_display_names_type_required() {
    // Intl.DisplayNames requires a valid `type` option (absent → TypeError,
    // invalid → RangeError); a primitive options argument is a TypeError.
    assert_eq!(
        run("try{new Intl.DisplayNames('en',{});'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{new Intl.DisplayNames('en');'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{new Intl.DisplayNames('en',{type:'bogus'});'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    assert_eq!(
        run("try{new Intl.DisplayNames('en',5);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    // A valid type works.
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'language'}).of('fr')"),
        "French"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'region'}).of('US')"),
        "United States"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'currency'}).of('USD')"),
        "US Dollar"
    );
}

#[test]
fn intl_number_format_significant_digit_defaults() {
    // SetNumberFormatDigitOptions: a lone significant-digit option defaults the
    // other (min→1, max→21), so the minimum padding actually applies.
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{minimumSignificantDigits:3}).format(1)"),
        "1.00"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{minimumSignificantDigits:3}).format(12)"),
        "12.0"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en-US',{maximumSignificantDigits:2}).format(123.4)"),
        "120"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{minimumSignificantDigits:3}).resolvedOptions().maximumSignificantDigits"
        ),
        "21"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{maximumSignificantDigits:5}).resolvedOptions().minimumSignificantDigits"
        ),
        "1"
    );
    // No significant-digit option → no significant path (fraction digits default).
    assert_eq!(
        run("new Intl.NumberFormat('en-US').resolvedOptions().minimumSignificantDigits"),
        "undefined"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en-US',{minimumSignificantDigits:2,maximumSignificantDigits:4}).format(1.23456)"
        ),
        "1.235"
    );
}

#[test]
fn intl_number_format_trailing_zero_display() {
    // trailingZeroDisplay:"stripIfInteger" drops forced trailing zeros for an
    // integer value; a real fraction is kept, and "auto" is unaffected.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{minimumFractionDigits:2,trailingZeroDisplay:'stripIfInteger'}).format(5)"
        ),
        "5"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{minimumFractionDigits:2,trailingZeroDisplay:'stripIfInteger'}).format(5.5)"
        ),
        "5.50"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{minimumFractionDigits:2}).format(5)"),
        "5.00"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{minimumSignificantDigits:3,trailingZeroDisplay:'stripIfInteger'}).format(5)"
        ),
        "5"
    );
    assert_eq!(run("new Intl.NumberFormat('en').format(1234.5)"), "1,234.5");
}

#[test]
fn intl_number_format_rounding_increment() {
    // roundingIncrement rounds to the nearest increment × 10^-maxFrac step.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{maximumFractionDigits:2,minimumFractionDigits:2,roundingIncrement:5}).format(1.23)"
        ),
        "1.25"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{maximumFractionDigits:2,minimumFractionDigits:2,roundingIncrement:5}).format(1.27)"
        ),
        "1.25"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{maximumFractionDigits:2,minimumFractionDigits:2,roundingIncrement:25}).format(1.30)"
        ),
        "1.25"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{maximumFractionDigits:0,minimumFractionDigits:0,roundingIncrement:10}).format(143)"
        ),
        "140"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{maximumFractionDigits:2}).format(1.234)"),
        "1.23"
    );
}

#[test]
fn intl_number_format_accounting_sign() {
    // currencySign:"accounting" wraps a negative currency amount in parentheses.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{style:'currency',currency:'USD',currencySign:'accounting'}).format(-5)"
        ),
        "($5.00)"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{style:'currency',currency:'USD',currencySign:'accounting'}).format(5)"
        ),
        "$5.00"
    );
    // Standard sign and non-accounting formats are unchanged.
    assert_eq!(
        run("new Intl.NumberFormat('en',{style:'currency',currency:'USD'}).format(-5)"),
        "-$5.00"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{style:'currency',currency:'USD',currencySign:'accounting'}).resolvedOptions().currencySign"
        ),
        "accounting"
    );
}

#[test]
fn intl_number_format_signdisplay_negative_mode() {
    // signDisplay:"negative" shows a minus for negative values but NOT for -0.
    assert_eq!(
        run("new Intl.NumberFormat('en',{signDisplay:'negative'}).format(-0)"),
        "0"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{signDisplay:'negative'}).format(-5)"),
        "-5"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{signDisplay:'negative'}).format(5)"),
        "5"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{style:'percent',signDisplay:'negative'}).format(-0)"),
        "0%"
    );
    // auto/always still sign a negative zero.
    assert_eq!(run("new Intl.NumberFormat('en').format(-0)"), "-0");
    assert_eq!(
        run("new Intl.NumberFormat('en',{signDisplay:'always'}).format(-0)"),
        "-0"
    );
}

#[test]
fn intl_display_names_resolved_options() {
    // resolvedOptions reports { locale, style, type, fallback[, languageDisplay] }
    // with the defaults (style "long", fallback "code", languageDisplay "dialect").
    assert_eq!(
        run(
            "var o=new Intl.DisplayNames('en',{type:'region'}).resolvedOptions();o.type+','+o.style+','+o.fallback"
        ),
        "region,long,code"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'language'}).resolvedOptions().languageDisplay"),
        "dialect"
    );
    assert_eq!(
        run(
            "var o=new Intl.DisplayNames('en',{type:'region',style:'short',fallback:'none'}).resolvedOptions();o.style+','+o.fallback"
        ),
        "short,none"
    );
    assert_eq!(
        run("Object.keys(new Intl.DisplayNames('en',{type:'region'}).resolvedOptions()).join(',')"),
        "locale,style,type,fallback"
    );
    // Invalid style/fallback are RangeErrors.
    assert_eq!(
        run(
            "try{new Intl.DisplayNames('en',{type:'region',style:'bogus'});'no'}catch(e){e.constructor.name}"
        ),
        "RangeError"
    );
    assert_eq!(
        run(
            "try{new Intl.DisplayNames('en',{type:'region',fallback:'bogus'});'no'}catch(e){e.constructor.name}"
        ),
        "RangeError"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'region'}).of('US')"),
        "United States"
    );
}

#[test]
fn intl_display_names_of_fallback() {
    // For a crate-backed type with no match, `fallback:"none"` returns undefined
    // and "code" returns the code (works where the data reports "not found").
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'language',fallback:'none'}).of('xx')===undefined"),
        "true"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'language',fallback:'code'}).of('xx')"),
        "xx"
    );
    // Known names and other types are unaffected.
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'language'}).of('fr')"),
        "French"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'region'}).of('US')"),
        "United States"
    );
    assert_eq!(
        run("new Intl.DisplayNames('en',{type:'currency'}).of('USD')"),
        "US Dollar"
    );
}

#[test]
fn array_from_async_subclass_uses_dense_storage() {
    // `C.fromAsync(...)` (subclass constructor `this`) must CreateDataPropertyOrThrow
    // each element into the dense store, like C.from — a raw set_property stashed
    // them as named props that join/reduce missed. (Observed after the microtask
    // drain via console.log, since fromAsync returns a promise.)
    assert_eq!(
        out(
            "class A extends Array{}Array.fromAsync.call(A,[1,2,3]).then(a=>console.log((a instanceof A)+','+a.join(',')+','+a.reduce((x,y)=>x+y,0)))"
        ),
        "true,1,2,3,6\n"
    );
    assert_eq!(
        out(
            "class A extends Array{}Array.fromAsync.call(A,[1,2],x=>x*10).then(a=>console.log(a.join(',')))"
        ),
        "10,20\n"
    );
    assert_eq!(
        out("Array.fromAsync([1,2,3]).then(a=>console.log(a.join(',')))"),
        "1,2,3\n"
    );
}

#[test]
fn float16_array_read_write_and_rounding() {
    // Float16Array (ES2025): construction, name/size, exact IEEE-754 half rounding,
    // and the generic typed-array methods over the new kind.
    assert_eq!(run("typeof Float16Array"), "function");
    assert_eq!(run("Float16Array.BYTES_PER_ELEMENT"), "2");
    assert_eq!(run("Float16Array.name"), "Float16Array");
    assert_eq!(
        run("var a=new Float16Array(3);a[0]=1.5;a[1]=2.25;a[0]+','+a[1]+','+a[2]"),
        "1.5,2.25,0"
    );
    // f16 cannot represent 1.0001 (rounds to 1) but holds its max 65504 exactly.
    assert_eq!(run("var a=new Float16Array(1);a[0]=1.0001;a[0]"), "1");
    assert_eq!(run("var a=new Float16Array(1);a[0]=65504;a[0]"), "65504");
    assert_eq!(
        run("var a=new Float16Array(1);a[0]=0.1;a[0]"),
        "0.0999755859375"
    );
    assert_eq!(
        run("Float16Array.from([1,2,3]).map(x=>x*2).join(',')"),
        "2,4,6"
    );
    assert_eq!(
        run("new Float16Array([1,2,3,4]).subarray(1,3).join(',')"),
        "2,3"
    );
    assert_eq!(
        run("var a=new Float16Array(new ArrayBuffer(4));a[0]=3.5;a[0]+'/'+a.length"),
        "3.5/2"
    );
    assert_eq!(run("new Float16Array(1) instanceof Float16Array"), "true");
    assert_eq!(
        run("Object.prototype.toString.call(new Float16Array(1))"),
        "[object Float16Array]"
    );
    // The moved Object.prototype.toString id is unaffected.
    assert_eq!(run("({}).toString()"), "[object Object]");
}

#[test]
fn atomics_single_agent_integer_ops() {
    // Single-agent Atomics over an integer TypedArray: RMW returns the old value,
    // store returns (and wraps) the written value, load reads, isLockFree checks
    // the byte size, and non-integer arrays / OOB indices / non-arrays throw.
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=5;Atomics.add(a,0,3)+','+a[0]"),
        "5,8"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=10;Atomics.sub(a,0,4)+','+a[0]"),
        "10,6"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=12;Atomics.and(a,0,10)+','+a[0]"),
        "12,8"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=12;Atomics.or(a,0,3);a[0]"),
        "15"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=12;Atomics.xor(a,0,10);a[0]"),
        "6"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=7;Atomics.exchange(a,0,99)+','+a[0]"),
        "7,99"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=5;Atomics.compareExchange(a,0,5,50)+','+a[0]"),
        "5,50"
    );
    assert_eq!(
        run("var a=new Int32Array(1);a[0]=5;Atomics.compareExchange(a,0,9,50)+','+a[0]"),
        "5,5"
    );
    assert_eq!(
        run("var a=new Uint8Array(1);Atomics.store(a,0,256)+','+a[0]"),
        "256,0"
    ); // Atomics.store returns the integer value; the element wraps to 0
    assert_eq!(
        run("var a=new Uint8Array(1);a[0]=42;Atomics.load(a,0)"),
        "42"
    );
    assert_eq!(
        run("[Atomics.isLockFree(4),Atomics.isLockFree(3)].join(',')"),
        "true,false"
    );
    assert_eq!(
        run("try{Atomics.add(new Float64Array(1),0,1);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{Atomics.load(new Int32Array(2),5);'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    assert_eq!(
        run("Object.prototype.toString.call(Atomics)"),
        "[object Atomics]"
    );
}

#[test]
fn shared_array_buffer_core() {
    // SharedArrayBuffer (single-agent): construction + byteLength, the
    // growable/maxByteLength accessors, typed-array + Atomics + DataView backing,
    // the [Symbol.toStringTag], and a slot-requiring accessor read on the
    // prototype throwing.
    assert_eq!(run("new SharedArrayBuffer(16).byteLength"), "16");
    assert_eq!(
        run("Object.prototype.toString.call(new SharedArrayBuffer(8))"),
        "[object SharedArrayBuffer]"
    );
    assert_eq!(
        run("var s=new SharedArrayBuffer(8);var a=new Int32Array(s);a[0]=42;a[0]"),
        "42"
    );
    assert_eq!(
        run(
            "var s=new SharedArrayBuffer(8);var a=new Int32Array(s);Atomics.store(a,0,5);Atomics.add(a,0,3)+','+a[0]"
        ),
        "5,8"
    );
    assert_eq!(run("new SharedArrayBuffer(8).growable"), "false");
    assert_eq!(
        run("new SharedArrayBuffer(8,{maxByteLength:16}).growable"),
        "true"
    );
    assert_eq!(
        run("new SharedArrayBuffer(8,{maxByteLength:16}).maxByteLength"),
        "16"
    );
    assert_eq!(
        run("new SharedArrayBuffer(8).constructor===SharedArrayBuffer"),
        "true"
    );
    assert_eq!(
        run("new SharedArrayBuffer(8) instanceof SharedArrayBuffer"),
        "true"
    );
    assert_eq!(
        run("SharedArrayBuffer.name+','+SharedArrayBuffer.length"),
        "SharedArrayBuffer,1"
    );
    assert_eq!(
        run("try{SharedArrayBuffer.prototype.byteLength;'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run("try{new SharedArrayBuffer(16,{maxByteLength:8});'no'}catch(e){e.constructor.name}"),
        "RangeError"
    );
    assert_eq!(
        run("var s=new SharedArrayBuffer(8);var d=new DataView(s);d.setInt32(0,99);d.getInt32(0)"),
        "99"
    );
}

#[test]
fn shared_array_buffer_grow_and_slice() {
    // grow (growable SABs only, increase-only, data-preserving) and slice
    // (returns a *SharedArrayBuffer*, leaving ArrayBuffer.slice unchanged).
    assert_eq!(
        run("var s=new SharedArrayBuffer(8,{maxByteLength:16});s.grow(12);s.byteLength"),
        "12"
    );
    assert_eq!(
        run(
            "var s=new SharedArrayBuffer(8,{maxByteLength:16});var a=new Int32Array(s);a[0]=77;s.grow(16);new Int32Array(s)[0]+','+s.byteLength"
        ),
        "77,16"
    );
    assert_eq!(
        run(
            "var s=new SharedArrayBuffer(8,{maxByteLength:16});try{s.grow(4);'no'}catch(e){e.constructor.name}"
        ),
        "RangeError"
    );
    assert_eq!(
        run(
            "var s=new SharedArrayBuffer(8,{maxByteLength:16});try{s.grow(20);'no'}catch(e){e.constructor.name}"
        ),
        "RangeError"
    );
    assert_eq!(
        run("var s=new SharedArrayBuffer(8);try{s.grow(16);'no'}catch(e){e.constructor.name}"),
        "TypeError"
    );
    assert_eq!(
        run(
            "var s=new SharedArrayBuffer(8);var a=new Uint8Array(s);a[2]=5;var s2=s.slice(2,4);new Uint8Array(s2)[0]+','+s2.byteLength"
        ),
        "5,2"
    );
    assert_eq!(
        run("Object.prototype.toString.call(new SharedArrayBuffer(8).slice(0,4))"),
        "[object SharedArrayBuffer]"
    );
    assert_eq!(
        run("new SharedArrayBuffer(8).slice(0,4) instanceof SharedArrayBuffer"),
        "true"
    );
    assert_eq!(
        run("Object.prototype.toString.call(new ArrayBuffer(8).slice(0,4))"),
        "[object ArrayBuffer]"
    );
}

#[test]
#[cfg(feature = "intl")]
fn intl_numbering_system_digit_substitution() {
    // The resolved numbering system substitutes the ASCII digits (consecutive-
    // codepoint systems); separators stay, latn/hanidec are left ASCII.
    assert_eq!(
        run("new Intl.NumberFormat('en',{numberingSystem:'arab'}).format(123)"),
        "\u{0661}\u{0662}\u{0663}"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{numberingSystem:'deva'}).format(123)"),
        "\u{0967}\u{0968}\u{0969}"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en',{numberingSystem:'thai'}).format(789)"),
        "\u{0e57}\u{0e58}\u{0e59}"
    );
    assert_eq!(run("new Intl.NumberFormat('en').format(123)"), "123");
    // hanidec is the one non-consecutive system (explicit digit array): 5 → 五.
    assert_eq!(
        run("new Intl.NumberFormat('en',{numberingSystem:'hanidec'}).format(5)"),
        "\u{4e94}"
    );
    // Digits substituted but the grouping/decimal separators are untouched.
    assert_eq!(
        run("(1234).toLocaleString('en',{numberingSystem:'arab'})"),
        "\u{0661},\u{0662}\u{0663}\u{0664}"
    );
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{numberingSystem:'arab'}).resolvedOptions().numberingSystem"
        ),
        "arab"
    );
}

#[test]
#[cfg(feature = "intl")]
fn intl_default_numbering_system_from_locale() {
    // Absent an explicit option, NumberFormat resolves the locale's CLDR default
    // numbering system (via the intl crate's per-locale data) instead of always
    // latn — so e.g. Persian renders Extended Arabic-Indic digits.
    assert_eq!(
        run("new Intl.NumberFormat('fa').resolvedOptions().numberingSystem"),
        "arabext"
    );
    assert_eq!(
        run("new Intl.NumberFormat('fa').format(123)"),
        "\u{06f1}\u{06f2}\u{06f3}"
    );
    assert_eq!(
        run("new Intl.NumberFormat('en').resolvedOptions().numberingSystem"),
        "latn"
    );
    assert_eq!(run("new Intl.NumberFormat('en').format(1234)"), "1,234");
    assert_eq!(
        run("new Intl.NumberFormat('de-DE').format(1234.5)"),
        "1.234,5"
    );
    // An explicit option still wins.
    assert_eq!(
        run(
            "new Intl.NumberFormat('en',{numberingSystem:'deva'}).resolvedOptions().numberingSystem"
        ),
        "deva"
    );
}

#[test]
#[cfg(feature = "intl")]
fn intl_datetime_numbering_system() {
    // DateTimeFormat renders its digits in the resolved numbering system (explicit
    // option and the locale's CLDR default), separators untouched; latn unchanged.
    assert_eq!(
        run(
            "new Intl.DateTimeFormat('en',{numberingSystem:'arab',year:'numeric',month:'numeric',day:'numeric'}).format(new Date(2020,5,15))"
        ),
        "\u{0666}/\u{0661}\u{0665}/\u{0662}\u{0660}\u{0662}\u{0660}"
    );
    assert_eq!(
        run(
            "new Intl.DateTimeFormat('fa',{year:'numeric',month:'numeric',day:'numeric',calendar:'gregory'}).format(new Date(2020,5,15))"
        ),
        "\u{06f2}\u{06f0}\u{06f2}\u{06f0}/\u{06f6}/\u{06f1}\u{06f5}"
    );
    assert_eq!(
        run("new Intl.DateTimeFormat('fa').resolvedOptions().numberingSystem"),
        "arabext"
    );
    assert_eq!(
        run(
            "new Intl.DateTimeFormat('en',{year:'numeric',month:'numeric',day:'numeric'}).format(new Date(2020,5,15))"
        ),
        "6/15/2020"
    );
}

#[test]
fn function_to_string_is_consistent_across_coercion_paths() {
    // `Function.prototype.toString` reached indirectly (`String(fn)`, `fn + ""`,
    // `.call`) must render the same source-representation as `fn.toString()`,
    // not be misrouted to `Array.prototype.toString` ("[object Function]").
    assert_eq!(
        run(r#"var f=function g(){}; String(f) === f.toString()"#),
        "true"
    );
    assert_eq!(
        run(r#"var f=function g(){}; (""+f) === f.toString()"#),
        "true"
    );
    assert_eq!(
        run(r#"var f=function(){}; Function.prototype.toString.call(f) === f.toString()"#),
        "true"
    );
    // A function used as a computed property key stringifies consistently, so a
    // later lookup with the same function resolves.
    assert_eq!(
        run(r#"var o={[function(){}]:1}; o[String(function(){})] === 1 && o[function(){}] === 1"#),
        "true"
    );
    // Non-callable `this` throws a TypeError.
    assert_eq!(
        run(r#"try{Function.prototype.toString.call({});false}catch(e){e instanceof TypeError}"#),
        "true"
    );
}

#[test]
fn iterators_inherit_object_prototype() {
    // `%IteratorPrototype%.[[Prototype]]` is `%Object.prototype%`, so every
    // iterator inherits `toString`/`valueOf`; ToPrimitive on an iterator (e.g. a
    // generator object used as a computed key) yields "[object <Tag>]" instead of
    // throwing "Cannot convert object to primitive value".
    assert_eq!(
        run(r#"typeof [][Symbol.iterator]().toString === "function""#),
        "true"
    );
    assert_eq!(
        run(r#"function* g(){}; String(g()) === "[object Generator]""#),
        "true"
    );
    assert_eq!(
        run(r#"function* g(){}; Object.keys({[g()]:1})[0] === "[object Generator]""#),
        "true"
    );
}

#[test]
fn bigint_literal_property_keys() {
    // A BigInt literal key is ToString of its value (a canonical decimal string),
    // exact even beyond 2^53, and supports non-decimal bases.
    assert_eq!(
        run(r#"({999999999999999999n: true})["999999999999999999"] === true"#),
        "true"
    );
    assert_eq!(run(r#"({0x10n: "s"})["16"] === "s""#), "true");
    assert_eq!(run(r#"({1n(){return "m";}})["1"]() === "m""#), "true");
    assert_eq!(
        run(r#"class C{1n(){return "c";}}; new C()["1"]() === "c""#),
        "true"
    );
    assert_eq!(run(r#"var {1n: a} = {"1":"v"}; a === "v""#), "true");
    // A plain numeric-keyed concise method still works (regression guard).
    assert_eq!(run(r#"({5(){return "five";}})["5"]() === "five""#), "true");
}

#[test]
fn class_field_named_get_set_before_generator_asi() {
    // `get`/`set` followed by `*` is a plain field/method name (a getter is never
    // a generator); the `*m(){}` is a separate generator method via ASI.
    assert_eq!(
        run(r#"class A{ get
 *a(){} }; A.prototype.hasOwnProperty("a") && new A().hasOwnProperty("get")"#),
        "true"
    );
    // Ordinary accessors and async generators still parse correctly.
    assert_eq!(
        run(
            r#"class C{ get x(){return 1;} async *g(){} }; new C().x === 1 && typeof new C().g === "function""#
        ),
        "true"
    );
}

#[test]
fn tagged_template_raw_is_non_enumerable() {
    // The template object's `raw` property is
    // `{ writable:false, enumerable:false, configurable:false }`.
    assert_eq!(
        run(r#"var t; (function tag(x){t=x;})`${1}`;
               var d=Object.getOwnPropertyDescriptor(t,"raw");
               !d.enumerable && !d.writable && !d.configurable"#),
        "true"
    );
}

#[test]
fn object_proto_valueof_boxes_primitive_this() {
    // `Object.prototype.valueOf` is `ToObject(this)`: a primitive is boxed to a
    // wrapper object (typeof "object"), and null/undefined throw a TypeError.
    assert_eq!(run("typeof Object.prototype.valueOf.call(true)"), "object");
    assert_eq!(run("typeof Object.prototype.valueOf.call(5)"), "object");
    assert_eq!(
        run(
            "try { Object.prototype.valueOf.call(undefined); 'no' } catch (e) { e instanceof TypeError }"
        ),
        "true"
    );
}

#[test]
fn object_prototype_tostring_honors_symbol_tostringtag_on_primitive() {
    // A string `Symbol.toStringTag` on the boxed primitive's prototype overrides
    // the builtin tag (`toString.call(true)` becomes "[object test262]").
    assert_eq!(
        run(r#"Boolean.prototype[Symbol.toStringTag] = 'test262';
               Object.prototype.toString.call(true)"#),
        "[object test262]"
    );
}

#[test]
fn object_values_entries_snapshot_and_run_getters() {
    // EnumerableOwnProperties snapshots the key list once, then per key re-checks
    // existence/enumerability and reads via `[[Get]]` (invoking a getter). A key
    // the getter adds is excluded; a deleted future key drops out.
    assert_eq!(
        run("var o={a:'A', get b(){ this.c='C'; return 'B'; }};\
             var v=Object.values(o); v.length===2 && v[1]==='B'"),
        "true"
    );
    assert_eq!(
        run(
            "var o={a:'A', get b(){ delete this.c; return 'B'; }, c:'C'};\
             Object.values(o).length===2"
        ),
        "true"
    );
}

#[test]
fn object_create_length_is_two() {
    assert_eq!(run("Object.create.length"), "2");
}

#[test]
fn define_property_large_array_index_is_stored() {
    // A canonical array index beyond the dense cap is stored sparsely and grows
    // `length`; 2**32-1 is not an index (an ordinary named property).
    assert_eq!(
        run(
            "var a=[]; Object.defineProperty(a,4294967294,{value:100,configurable:true});\
             a.hasOwnProperty('4294967294') && a[4294967294]===100 && a.length===4294967295"
        ),
        "true"
    );
    assert_eq!(
        run(
            "var a=[]; Object.defineProperty(a,4294967295,{value:1,configurable:true});\
             a.hasOwnProperty('4294967295') && a.length===0"
        ),
        "true"
    );
}

#[test]
fn is_frozen_computes_test_integrity_level() {
    // A non-extensible object with no own properties is frozen; one with only
    // non-configurable non-writable data + non-configurable accessors is frozen.
    assert_eq!(
        run("var o={}; Object.preventExtensions(o); Object.isFrozen(o)"),
        "true"
    );
    assert_eq!(
        run(
            "var o={}; Object.defineProperty(o,'x',{value:1,writable:false,configurable:false});\
             Object.defineProperty(o,'y',{get(){return 1;},configurable:false});\
             Object.preventExtensions(o); Object.isFrozen(o)"
        ),
        "true"
    );
    // Merely sealed (data still writable) is not frozen.
    assert_eq!(
        run("var o={x:1}; Object.seal(o); Object.isFrozen(o)"),
        "false"
    );
}

#[test]
fn object_subclass_ignores_value_argument() {
    // `new (class extends Object {})(value)` yields a fresh object with the
    // subclass prototype, ignoring the value argument (spec step 1).
    assert_eq!(
        run("class O extends Object {}; var o=new O({a:1});\
             o.a===undefined && Object.getPrototypeOf(o)===O.prototype"),
        "true"
    );
}

#[test]
fn getownpropertynames_orders_string_index_before_length() {
    // A String exotic object's extra index key sorts ascending before `length`.
    assert_eq!(
        run("var s=new String('abc'); s[5]='de';\
             Object.getOwnPropertyNames(s).join(',')"),
        "0,1,2,5,length"
    );
}

#[test]
fn unary_numeric_on_symbol_and_bigint_wrappers() {
    // ToNumber of a boxed Symbol/BigInt throws under `+` (and `<`/etc.).
    assert_eq!(
        run("try{ +Object(Symbol()); 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    assert_eq!(
        run("try{ +Object(3n); 'no' }catch(e){ e instanceof TypeError }"),
        "true"
    );
    // `-`/`~` on a boxed BigInt stay BigInt (ToNumeric), don't throw.
    assert_eq!(run("(-Object(3n)).toString()"), "-3");
    assert_eq!(run("(~Object(3n)).toString()"), "-4");
    // Number/string/boolean wrappers and dates are unaffected.
    assert_eq!(run("+Object(5)"), "5");
    assert_eq!(run("+new Date(1000)"), "1000");
}

#[test]
fn json_stringify_symbol_escape_and_empty_replacer() {
    // Symbols are ignored as values and keys.
    assert_eq!(run("JSON.stringify(Symbol())"), "undefined");
    assert_eq!(run("JSON.stringify([Symbol()])"), "[null]");
    assert_eq!(run("JSON.stringify({key: Symbol()})"), "{}");
    // 0x08/0x0C escape to \\b / \\f.
    assert_eq!(run(r#"JSON.stringify("\b\f")"#), r#""\b\f""#);
    // An empty PropertyList replacer yields {} for objects, and propagates into
    // nested objects within arrays.
    assert_eq!(run("JSON.stringify({a:1,b:2}, [])"), "{}");
    assert_eq!(run("JSON.stringify([1, {a: 2}], [])"), "[1,{}]");
}

#[test]
fn json_parse_reviver_source_context() {
    // A primitive leaf's context.source is its exact JSON text.
    assert_eq!(
        run("var s; JSON.parse('1.5e2', function(k,v,c){ s=c.source; return v; }); s"),
        "1.5e2"
    );
    // Objects/arrays get a bare context (no source).
    assert_eq!(
        run(
            "var s='x'; JSON.parse('[1]', function(k,v,c){ if(Array.isArray(v)) s=c.source; return v; }); String(s)"
        ),
        "undefined"
    );
    // A forward-substituted value loses its source (SameValue check).
    assert_eq!(
        run(
            "var s='x'; JSON.parse('[1,2]', function(k,v,c){ if(k==='0'){this[1]=42;} if(k==='1'){s=c.source;} return this[k]===undefined?v:this[k]; }); String(s)"
        ),
        "undefined"
    );
}

#[test]
fn set_composition_lazy_keys_short_circuit() {
    // isSupersetOf / isDisjointFrom over a set-like stop iterating the argument's
    // keys as soon as the answer is known (and still return correctly).
    assert_eq!(
        run("new Set([1,2,3]).isSupersetOf(new Set([1,9]))"),
        "false"
    );
    assert_eq!(
        run("new Set([1,2]).isDisjointFrom(new Set([2,3,4]))"),
        "false"
    );
    assert_eq!(
        run("new Set([1,2]).isDisjointFrom(new Set([3,4,5]))"),
        "true"
    );
}

// The `run` helper drains the microtask queue before rendering the final value,
// but that value is captured *before* draining — so async results are observed
// via a mutated object/array reference (whose contents reflect the post-drain
// state), not via a freshly computed primitive.

#[test]
fn async_gen_yield_star_sync_iterator_unwraps_promise_values() {
    // `yield* syncIter` in an async generator wraps the sync iterator as an
    // AsyncFromSyncIterator: each yielded value is unwrapped (awaited).
    assert_eq!(
        run("var out = [];
             function* g() { yield Promise.resolve(1); yield 2; }
             async function* ag() { yield* g(); }
             (async () => { for await (var v of ag()) out.push(v); })();
             out"),
        "1,2"
    );
}

#[test]
fn async_gen_yield_star_native_async_does_not_unwrap() {
    // A native async iterator's promise value is re-yielded verbatim (NOT
    // unwrapped) — AsyncGeneratorYield does not itself await the value.
    assert_eq!(
        run(
            "var out = [];
             var p = Promise.resolve('x');
             var asyncIter = {
               [Symbol.asyncIterator]() { return this; },
               _done: false,
               next() { var d = this._done; this._done = true; return Promise.resolve({ value: p, done: d }); },
             };
             async function* ag() { yield* asyncIter; }
             (async () => { for await (var v of ag()) out.push(v === p); })();
             out"
        ),
        "true"
    );
}

#[test]
fn for_await_over_infinite_sync_iterator_breaks_and_closes() {
    // A lazy `for await` over an infinite sync iterator terminates on `break`
    // (it does NOT drain eagerly) and calls the iterator's `return`.
    assert_eq!(
        run("var out = []; var count = 0;
             var it = {
               [Symbol.iterator]() { return this; },
               next() { count += 1; return { value: count, done: false }; },
               return() { out.push('returned'); return { done: true }; },
             };
             (async () => { for await (var v of it) { if (v >= 3) break; } out.push(count); })();
             out"),
        "returned,3"
    );
}

#[test]
fn async_gen_yield_star_value_rejection_closes_sync_iterator() {
    // When the unwrapped value rejects, the sync iterator is closed (its `return`
    // is called) and the rejection propagates.
    assert_eq!(
        run("var out = [];
             var it = {
               [Symbol.iterator]() { return this; },
               next() { return { value: Promise.reject('boom'), done: false }; },
               return() { out.push('returned'); return { done: true }; },
             };
             async function* ag() { yield* it; }
             (async () => {
               try { for await (var v of ag()) {} } catch (e) { out.push(e); }
             })();
             out"),
        "returned,boom"
    );
}

#[test]
fn for_await_over_sync_array_yields_in_order() {
    // A plain sync array in `for await` is driven lazily and awaits each value.
    assert_eq!(
        run(
            "var out = [];
             (async () => { for await (var v of [Promise.resolve('a'), 'b', Promise.resolve('c')]) out.push(v); })();
             out"
        ),
        "a,b,c"
    );
}

#[test]
fn resizable_length_tracking_view_tracks_buffer() {
    // A view with no explicit length over a resizable buffer re-spans on resize;
    // its `.length` follows the buffer's current byte length.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab); \
             var a=ta.length; rab.resize(2); var b=ta.length; rab.resize(8); var c=ta.length; \
             a+','+b+','+c"
        ),
        "4,2,8"
    );
    // `.resizable`/`.maxByteLength` reflect the resizable allocation.
    assert_eq!(
        run("var rab=new ArrayBuffer(4,{maxByteLength:8}); \
             rab.resizable+','+rab.maxByteLength+','+(new ArrayBuffer(4)).resizable"),
        "true,8,false"
    );
}

#[test]
fn resizable_out_of_bounds_fixed_view_reads_empty() {
    // A fixed-length view whose resizable buffer shrinks below its extent is
    // out of bounds: `.length` collapses to 0 and integer reads are undefined,
    // then it becomes valid again when the buffer is grown back. The bytes past
    // the shrink point were dropped, so on regrow they read back zero-filled.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab,0,4); \
             ta[0]=1; ta[3]=9; rab.resize(2); \
             var oob=ta.length+','+ta[0]+','+ta[3]; \
             rab.resize(4); var back=ta.length+','+ta[0]+','+ta[3]; \
             oob+'|'+back"
        ),
        "0,undefined,undefined|4,1,0"
    );
}

#[test]
fn resizable_fill_out_of_bounds_throws_typeerror() {
    // `%TypedArray%.prototype.fill` re-validates after argument coercion: a
    // `valueOf` that shrinks the buffer below a fixed-length view is a TypeError.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab,0,4); \
             var evil={valueOf(){ rab.resize(2); return 3; }}; \
             var e='no'; try { ta.fill(evil,1,2); } catch(x){ e=x.constructor.name; } e"
        ),
        "TypeError"
    );
    // A length-tracking view is not out of bounds on a plain shrink: no throw,
    // and the write is clamped to the new (shorter) length.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab); \
             var evil={valueOf(){ rab.resize(2); return 5; }}; \
             ta.fill(evil); ta.length+':'+ta.join(',')"
        ),
        "2:5,5"
    );
}

#[test]
fn generic_array_fill_on_shrunk_typed_array_is_noop_not_throw() {
    // The *generic* `Array.prototype.fill.call(ta)` runs the ordinary array-like
    // algorithm: an out-of-bounds `Set` is a silent no-op, never a TypeError.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab,0,4); \
             var evil={valueOf(){ rab.resize(2); return 3; }}; \
             var e='no'; try { Array.prototype.fill.call(ta,evil,1,2); } catch(x){ e=x.constructor.name; } e"
        ),
        "no"
    );
}

#[test]
fn resizable_typed_iterator_throws_on_oob_and_latches_done() {
    // Iterating a fixed-length view whose buffer shrinks out of bounds mid-loop
    // throws a TypeError at the next step.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(4,{maxByteLength:8}); var ta=new Uint8Array(rab,0,4); \
             var e='no'; var seen=0; \
             try { for (var v of ta) { seen++; if (seen===2) rab.resize(2); } } \
             catch(x){ e=x.constructor.name; } seen+','+e"
        ),
        "2,TypeError"
    );
    // Once the cursor passes the current length the iterator is done and stays
    // done even after the buffer grows back (length-tracking view).
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(3,{maxByteLength:5}); var ta=new Int8Array(rab); \
             ta[0]=11; var it=ta.values(); var r0=it.next().value; \
             rab.resize(0); var d1=it.next().done; rab.resize(5); var d2=it.next().done; \
             r0+','+d1+','+d2"
        ),
        "11,true,true"
    );
}

#[test]
fn resizable_with_uses_original_length_and_current_index_bound() {
    // `%TypedArray%.prototype.with` sizes the result from the pre-coercion length
    // but validates the index against the current (post-coercion) length.
    assert_eq!(
        run(
            "var rab=new ArrayBuffer(2,{maxByteLength:5}); var ta=new Int8Array(rab); \
             ta[0]=11; ta[1]=22; \
             var value={valueOf(){ rab.resize(5); return 123; }}; \
             var r=ta.with(4,value); \
             r.length+','+r.join(',')+'|'+ta.length"
        ),
        "2,11,22|5"
    );
}

// --- Function.prototype.toString source-text (ECMA-262 20.2.3.5) -------------

#[test]
fn to_string_reproduces_function_declaration_source() {
    // A function declaration reproduces its exact source, comments/whitespace
    // included — not a synthesized signature.
    assert_eq!(
        run(
            "/* x */function /* a */ f /* b */ ( /* c */ y /* d */ ) /* e */ { /* g */ ; }\nf.toString()"
        ),
        "function /* a */ f /* b */ ( /* c */ y /* d */ ) /* e */ { /* g */ ; }"
    );
    // `String(f)` and `"" + f` take the same path.
    assert_eq!(
        run("function g(a,b){return a}\nString(g)"),
        "function g(a,b){return a}"
    );
    assert_eq!(run("function h(){}\n'' + h"), "function h(){}");
}

#[test]
fn to_string_reproduces_arrow_source() {
    assert_eq!(run("var f = (a) => a + 1\n'' + f"), "(a) => a + 1");
    assert_eq!(
        run("var g = /* z */a /* a */ => /* b */ 0\n'' + g"),
        "a /* a */ => /* b */ 0"
    );
    assert_eq!(
        run("var h = async /* a */ ( /* b */ ) /* c */ => /* d */ 0\n'' + h"),
        "async /* a */ ( /* b */ ) /* c */ => /* d */ 0"
    );
}

#[test]
fn to_string_reproduces_class_source() {
    // A class has no valid NativeFunction fallback, so it MUST reproduce source.
    assert_eq!(
        run("class /* a */ A /* b */ { /* c */ }\nA.toString()"),
        "class /* a */ A /* b */ { /* c */ }"
    );
    assert_eq!(
        run("class A{}; class B extends A { m(){} }\n'' + B"),
        "class B extends A { m(){} }"
    );
    // `String(C)` path too.
    assert_eq!(
        run("class C { constructor(){} n(){} }\nString(C)"),
        "class C { constructor(){} n(){} }"
    );
}

#[test]
fn to_string_reproduces_object_method_and_accessor_source() {
    // A concise method's source is the whole MethodDefinition (key + body).
    assert_eq!(
        run("var o = { /* p */f /* a */ ( /* b */ ) /* c */ { /* d */ } }\n'' + o.f"),
        "f /* a */ ( /* b */ ) /* c */ { /* d */ }"
    );
    // A computed-key method: its exact source, so it can even round-trip as a key.
    assert_eq!(
        run("var g = { [ { a(){} }.a ](){ } }['a(){}']\n'' + g"),
        "[ { a(){} }.a ](){ }"
    );
    // A generator method includes its `*` prefix.
    assert_eq!(run("var o = { *gen(){} }\n'' + o.gen"), "*gen(){}");
    // A getter includes its `get` prefix; source read off the descriptor.
    assert_eq!(
        run(
            "var o = { get /* a */ x /* b */ ( /* c */ ) /* d */ { } }\n'' + Object.getOwnPropertyDescriptor(o, 'x').get"
        ),
        "get /* a */ x /* b */ ( /* c */ ) /* d */ { }"
    );
}

#[test]
fn to_string_eval_body_uses_its_own_source() {
    // A function/class defined inside `eval` slices the eval body's source, not
    // the enclosing program's.
    assert_eq!(
        run("eval('var f = function ff(){ return 1 }'); '' + f"),
        "function ff(){ return 1 }"
    );
    assert_eq!(
        run("eval('var C = class CC { m(){} }'); '' + C"),
        "class CC { m(){} }"
    );
    // A definition in the enclosing code after an eval still uses the outer source.
    assert_eq!(
        run("eval('1'); function main(){}\n'' + main"),
        "function main(){}"
    );
}

#[test]
fn to_string_proxy_of_callable_is_native_syntax() {
    // A Proxy has no `[[SourceText]]`; `Function.prototype.toString` yields the
    // NativeFunction form, never the wrapped function/class source.
    assert_eq!(
        run("class A{}; '' + (new Proxy(A, {}))"),
        "function A() { [native code] }"
    );
    assert_eq!(
        run("function foo(){}; (new Proxy(foo, {})).toString()"),
        "function foo() { [native code] }"
    );
    // Calling `Function.prototype.toString` on a non-callable proxy throws.
    assert_eq!(
        run(
            "var threw=false; try { Function.prototype.toString.call(new Proxy({}, {})) } catch(e){ threw = e instanceof TypeError }\nthrew"
        ),
        "true"
    );
}

#[test]
fn to_string_dynamic_and_builtin_functions_are_native_syntax() {
    // CreateDynamicFunction step 20 sets `[[SourceText]]` to the *assembled*
    // source, so a dynamically-built function stringifies to that text — not the
    // NativeFunction form.
    assert_eq!(
        run("'' + Function('a', 'return a')"),
        "function anonymous(a\n) {\nreturn a\n}"
    );
    // The keyword comes from the assembled wrapper, so an `async function` keeps
    // its prefix and the text stays valid syntax (slicing the extracted function
    // node instead would drop the `async`).
    assert_eq!(
        run(
            "var AsyncFunction = (async function(){}).constructor; '' + (new AsyncFunction('return 1'))"
        ),
        "async function anonymous(\n) {\nreturn 1\n}"
    );
    // A real built-in retains no source, so it still takes the NativeFunction form.
    assert_eq!(run("'' + Math.max"), "function max() { [native code] }");
}

// --- Atomics agents + virtual clock (see `nbexec::agent`) ------------------

#[test]
fn atomics_sync_wait_finite_timeout_times_out_and_advances_clock() {
    // A synchronous `Atomics.wait` on a matching value with a finite timeout
    // blocks for the whole timeout, then times out. The virtual clock advances by
    // the timeout so a `monotonicNow()`-measured duration observes the elapsed ms.
    assert_eq!(
        out(r#"
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            var before = $262_agent_monotonicNow();
            var r = Atomics.wait(i32, 0, 0, 250);
            var after = $262_agent_monotonicNow();
            console.log(r + " " + (after - before));
        "#),
        "timed-out 250\n"
    );
}

#[test]
fn atomics_sync_wait_value_mismatch_is_not_equal_and_no_time_passes() {
    // A value mismatch returns "not-equal" at once — the clock does not advance.
    assert_eq!(
        out(r#"
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            var before = $262_agent_monotonicNow();
            var r = Atomics.wait(i32, 0, 7, 250);
            var after = $262_agent_monotonicNow();
            console.log(r + " " + (after - before));
        "#),
        "not-equal 0\n"
    );
}

#[test]
fn atomics_wait_async_resolves_ok_on_notify() {
    // `Atomics.waitAsync` parks a waiter; a matching same-agent `Atomics.notify`
    // wakes it, settling the promise "ok" (drained as a microtask after the script).
    assert_eq!(
        out(r#"
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            var res = Atomics.waitAsync(i32, 0, 0, 1000);
            res.value.then(function (v) { console.log("settled " + v); });
            var woken = Atomics.notify(i32, 0, 1);
            console.log("notified " + woken + " async " + res.async);
        "#),
        "notified 1 async true\nsettled ok\n"
    );
}

#[test]
fn atomics_wait_async_resolves_timed_out_and_advances_clock() {
    // With no notify, a finite-timeout `waitAsync` settles "timed-out" when its
    // timeout macrotask fires, advancing the virtual clock by the timeout.
    assert_eq!(
        out(r#"
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            var before = $262_agent_monotonicNow();
            Atomics.waitAsync(i32, 0, 0, 300).value.then(function (v) {
                console.log(v + " " + ($262_agent_monotonicNow() - before));
            });
        "#),
        "timed-out 300\n"
    );
}

/// The main-agent side of an agent test: spin on the shared `RUNNING` counter
/// until `n` workers have checked in, exactly as `$262.agent.waitUntil` does.
#[cfg(feature = "std")]
const AGENT_WAIT_UNTIL: &str = r"
function waitUntil(ta, index, expected) {
  while (Atomics.load(ta, index) !== expected) ;
}
function report() {
  var r;
  while ((r = $262_agent_getReport()) === null) $262_agent_sleep(1);
  return r;
}
";

#[test]
#[cfg(feature = "std")]
fn agent_worker_blocks_in_atomics_wait_until_notified() {
    // The point of the threaded agent model: the worker really *parks* inside
    // `Atomics.wait` (mid-callback, on a buffer broadcast from the main agent),
    // and the main agent's `Atomics.notify` finds it on the waiter list, reports
    // one wake, and releases it with "ok".
    assert_eq!(
        out(&alloc::format!(
            r#"{AGENT_WAIT_UNTIL}
            $262_agent_start(`
              $262.agent.receiveBroadcast(function (sab) {{
                var a = new Int32Array(sab);
                Atomics.add(a, 1, 1);
                $262.agent.report(Atomics.wait(a, 0, 0, 30000));
              }});
            `);
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            $262_agent_broadcast(i32.buffer);
            waitUntil(i32, 1, 1);
            console.log("woke " + Atomics.notify(i32, 0, 1));
            console.log("report " + report());
            console.log("again " + Atomics.notify(i32, 0, 1));
        "#
        )),
        "woke 1\nreport ok\nagain 0\n"
    );
}

#[test]
#[cfg(feature = "std")]
fn agent_notify_wakes_exactly_one_of_two_parked_workers() {
    // Two workers park on the same location; `notify(…, 1)` wakes exactly one and
    // the other runs out its (short) timeout. Sorting makes the pair order-free.
    assert_eq!(
        out(&alloc::format!(
            r#"{AGENT_WAIT_UNTIL}
            for (var i = 0; i < 2; i++) {{
              $262_agent_start(`
                $262.agent.receiveBroadcast(function (sab) {{
                  var a = new Int32Array(sab);
                  Atomics.add(a, 1, 1);
                  $262.agent.report(Atomics.wait(a, 0, 0, 250));
                }});
              `);
            }}
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            $262_agent_broadcast(i32.buffer);
            waitUntil(i32, 1, 2);
            console.log("woke " + Atomics.notify(i32, 0, 1));
            console.log([report(), report()].sort().join(","));
        "#
        )),
        "woke 1\nok,timed-out\n"
    );
}

#[test]
#[cfg(feature = "std")]
fn agent_broadcast_shares_the_same_data_block() {
    // A broadcast hands over the buffer's *Shared Data Block*, not a copy: the
    // worker's `Atomics.store` through its own `SharedArrayBuffer` object is
    // visible to the main agent reading the original one.
    assert_eq!(
        out(&alloc::format!(
            r#"{AGENT_WAIT_UNTIL}
            $262_agent_start(`
              $262.agent.receiveBroadcast(function (sab) {{
                var a = new Int32Array(sab);
                Atomics.store(a, 0, 42);
                Atomics.add(a, 1, 1);
              }});
            `);
            var i32 = new Int32Array(new SharedArrayBuffer(8));
            $262_agent_broadcast(i32.buffer);
            waitUntil(i32, 1, 1);
            console.log("shared " + Atomics.load(i32, 0));
        "#
        )),
        "shared 42\n"
    );
}

#[test]
fn virtual_clock_set_timeout_fires_in_delay_order_and_advances() {
    // Macrotasks fire earliest-virtual-fire-time first, and dispatching one
    // advances the virtual clock read by `monotonicNow()`.
    assert_eq!(
        out(r#"
            setTimeout(function () { console.log("b " + $262_agent_monotonicNow()); }, 200);
            setTimeout(function () { console.log("a " + $262_agent_monotonicNow()); }, 50);
        "#),
        "a 50\nb 200\n"
    );
}

// --- Cross-realm intrinsic identity / species (GetFunctionRealm) ------------

#[test]
fn cross_realm_intrinsic_throws_its_own_realms_type_error() {
    // A method belonging to another realm, called on a bad `this`, must throw
    // *that realm's* `%TypeError%` — the value assert.throws inspects is the
    // error's `.constructor`, so it must be `other.TypeError`, not the main one.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var e;
            try { other.String.prototype.valueOf.call(false); } catch (x) { e = x; }
            (e.constructor === other.TypeError) && (e.constructor !== TypeError)
        "#),
        "true"
    );
    // The same for a value-of on a null `this` reached via Reflect.apply.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var e;
            try { Reflect.apply(other.String.prototype.toString, null, []); }
            catch (x) { e = x; }
            e.constructor === other.TypeError
        "#),
        "true"
    );
}

#[test]
fn cross_realm_class_brand_check_throws_defining_realms_type_error() {
    // A class evaluated in another realm (via that realm's indirect `eval`) whose
    // private-method brand check fails throws the *defining realm's* TypeError.
    assert_eq!(
        run(r#"
            var r1 = $262_createRealm();
            var C1 = r1.global.eval("(class { #m(){return 1;} access(o){ return o.#m(); } })");
            var c1 = new C1();
            var okSelf = (c1.access(c1) === 1);
            var e;
            try { c1.access({}); } catch (x) { e = x; }
            okSelf && (e.constructor === r1.global.TypeError)
        "#),
        "true"
    );
}

#[test]
fn array_species_create_nulls_cross_realm_array_constructor() {
    // ArraySpeciesCreate: when the original array's `constructor` is *another
    // realm's* `%Array%`, it is treated as undefined — the current realm's
    // `%Array%` builds the result and the foreign `@@species` is never read.
    assert_eq!(
        run(r#"
            var arr = [];
            var OArray = $262_createRealm().global.Array;
            var called = 0;
            arr.constructor = OArray;
            Object.defineProperty(OArray, Symbol.species, { get: function(){ called++; } });
            var r = arr.map(function(){});
            called + ":" + (Object.getPrototypeOf(r) === Array.prototype)
        "#),
        "0:true"
    );
}

#[test]
fn same_realm_array_species_still_honored() {
    // A same-realm subclass `@@species` override is still consulted (no
    // over-eager cross-realm nullification of the current realm's constructor).
    assert_eq!(
        run(r#"
            class MyArr extends Array {}
            var a = new MyArr(1, 2, 3);
            var r = a.map(function(x){ return x; });
            r instanceof MyArr
        "#),
        "true"
    );
}

#[test]
fn ordinary_construct_null_prototype_falls_back_to_object_prototype() {
    // GetPrototypeFromConstructor: `new C()` where `C.prototype` was reassigned to
    // a non-object (`null`) links the instance to `%Object.prototype%`, not the
    // stale intrinsic `.prototype` object.
    assert_eq!(
        run(r#"
            function C(){}
            C.prototype = null;
            Object.getPrototypeOf(new C()) === Object.prototype
        "#),
        "true"
    );
    // A reassigned *object* `.prototype` is still honored.
    assert_eq!(
        run(r#"
            var p = {};
            function D(){}
            D.prototype = p;
            Object.getPrototypeOf(new D()) === p
        "#),
        "true"
    );
    // A normal function's instance still inherits the function's own `.prototype`.
    assert_eq!(
        run(r#"
            function E(){}
            Object.getPrototypeOf(new E()) === E.prototype
        "#),
        "true"
    );
}

#[test]
fn ordinary_construct_cross_realm_derives_new_targets_object_prototype() {
    // `Construct(C)` / `Reflect.construct(fn, [], C)` where `C` belongs to another
    // realm and has a non-object `.prototype` links the result to *that realm's*
    // `%Object.prototype%` (GetPrototypeFromConstructor step 4).
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var C = new other.Function();
            C.prototype = null;
            var a = Array.of.call(C, 1, 2, 3);
            var b = Array.from.call(C, []);
            var d = Reflect.construct(function(){}, [], C);
            (Object.getPrototypeOf(a) === other.Object.prototype) &&
            (Object.getPrototypeOf(b) === other.Object.prototype) &&
            (Object.getPrototypeOf(d) === other.Object.prototype)
        "#),
        "true"
    );
}

#[test]
fn iterator_abstract_construct_and_cross_realm_proto() {
    // `new Iterator()` (newTarget is `%Iterator%` itself) is a TypeError.
    assert_eq!(
        run(r#"
            var threw = false;
            try { new Iterator(); } catch (e) { threw = (e instanceof TypeError); }
            threw
        "#),
        "true"
    );
    // `Reflect.construct(Iterator, [], newTarget)` with a cross-realm `newTarget`
    // whose `.prototype` is not an object derives that realm's `%Iterator.prototype%`
    // (resolved by name, since `%Iterator.prototype%.constructor` is an accessor).
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var nt = new other.Function();
            nt.prototype = undefined;
            var ai = Reflect.construct(Iterator, [1], nt);
            var np = Reflect.construct(Iterator, [1], (function(){ var f = new other.Function(); f.prototype = null; return f; })());
            (Object.getPrototypeOf(ai) === other.Iterator.prototype) &&
            (Object.getPrototypeOf(np) === other.Iterator.prototype)
        "#),
        "true"
    );
    // An object `.prototype` on the newTarget is used directly.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var nt = new other.Function();
            var op = {};
            nt.prototype = op;
            Object.getPrototypeOf(Reflect.construct(Iterator, [1], nt)) === op
        "#),
        "true"
    );
}

#[test]
fn dynamic_function_cross_realm_derives_new_targets_function_prototype() {
    // `Reflect.construct(Function, [], C)` where `C` belongs to another realm and
    // has a non-object `.prototype` links the built function to *that realm's*
    // `%Function.prototype%` (CreateDynamicFunction → GetPrototypeFromConstructor
    // step 4, resolving the intrinsic in `GetFunctionRealm(C)`'s realm).
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var C = new other.Function();
            C.prototype = null;
            var o = Reflect.construct(Function, [], C);
            Object.getPrototypeOf(o) === other.Function.prototype
        "#),
        "true"
    );
    // A non-null non-object `.prototype` (e.g. a number) still derives the realm's
    // intrinsic default.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var C = new other.Function();
            C.prototype = 1;
            Object.getPrototypeOf(Reflect.construct(Function, [], C)) === other.Function.prototype
        "#),
        "true"
    );
    // An object `.prototype` on the cross-realm newTarget is used directly.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var C = new other.Function();
            var op = {};
            C.prototype = op;
            Object.getPrototypeOf(Reflect.construct(Function, [], C)) === op
        "#),
        "true"
    );
    // Same-realm construction (plain `new Function()` / `Function()`) is untouched:
    // the built function keeps the current realm's `%Function.prototype%`.
    assert_eq!(
        run(r#"
            var a = new Function("return 1");
            var b = Function("return 2");
            (Object.getPrototypeOf(a) === Function.prototype) &&
            (Object.getPrototypeOf(b) === Function.prototype) && (b() === 2)
        "#),
        "true"
    );
}

#[cfg(feature = "intl")]
#[test]
fn intl_cross_realm_derives_new_targets_service_prototype() {
    // Each `$262.createRealm()` realm builds its own distinct `%Intl.X.prototype%`
    // intrinsics (the cache is swapped per realm), and a cross-realm `newTarget`
    // with a non-object `.prototype` derives that realm's service prototype.
    assert_eq!(
        run(r#"
            var other = $262_createRealm().global;
            var nt = new other.Function();
            nt.prototype = undefined;
            var nf = Reflect.construct(Intl.NumberFormat, [], nt);
            var loc = Reflect.construct(Intl.Locale, ['en'], nt);
            (typeof other.Intl.NumberFormat.prototype === 'object') &&
            (Object.getPrototypeOf(nf) === other.Intl.NumberFormat.prototype) &&
            (Object.getPrototypeOf(loc) === other.Intl.Locale.prototype) &&
            (Object.getPrototypeOf(new Intl.NumberFormat()) === Intl.NumberFormat.prototype)
        "#),
        "true"
    );
}

// ---------------------------------------------------------------------------
// Regression tests for the contained Test262 fixes in this change set.
// ---------------------------------------------------------------------------

#[test]
fn class_extends_requires_constructor_with_object_prototype() {
    // `extends <non-constructor>` is a TypeError (ClassDefinitionEvaluation).
    let t = |src: &str| {
        run(&alloc::format!(
            "(function(){{ try {{ {src}; return 'ok'; }} catch(e) {{ return e.constructor.name; }} }})()"
        ))
    };
    assert_eq!(t("class C extends 42 {}"), "TypeError");
    assert_eq!(t("class C extends Math.abs {}"), "TypeError"); // callable, not a constructor
    // A bound function that is a constructor but has no `prototype` → TypeError.
    assert_eq!(
        t("var B = function(){}.bind(); class C extends B {}"),
        "TypeError"
    );
    // A `prototype` getter is evaluated exactly once; a primitive result throws.
    assert_eq!(
        run(r#"
            var calls = 0;
            var B = function(){}.bind();
            Object.defineProperty(B, 'prototype', { get(){ calls++; return 42; }, configurable: true });
            var threw = false;
            try { eval('(class extends B {})'); } catch(e) { threw = e instanceof TypeError; }
            threw + ':' + calls
        "#),
        "true:1"
    );
    // A valid subclass still works.
    assert_eq!(
        run("class A { m(){return 5;} } class B extends A {} new B().m()"),
        "5"
    );
    // `extends null` is valid (null prototype).
    assert_eq!(
        run("class C extends null {} Object.getPrototypeOf(C.prototype) === null"),
        "true"
    );
}

#[test]
fn class_constructor_own_key_order() {
    // `length, name, prototype` precede static members in source order, even when
    // a static member overrides `name`/`length`.
    assert_eq!(
        run(r#"class A { static method(){} static length(){} }
               JSON.stringify(Object.getOwnPropertyNames(A))"#),
        r#"["length","name","prototype","method"]"#
    );
    assert_eq!(
        run(r#"class N { static name(){} }
               JSON.stringify(Object.getOwnPropertyNames(N))"#),
        r#"["length","name","prototype"]"#
    );
    assert_eq!(
        run(r#"class C { static get length(){} }
               JSON.stringify(Object.getOwnPropertyNames(C))"#),
        r#"["length","name","prototype"]"#
    );
}

#[test]
fn map_foreach_delete_then_readd_is_revisited() {
    // A key deleted then re-added during `forEach` is visited again at its new
    // position (live-cursor iteration over the compacting backing store).
    assert_eq!(
        run(r#"
            var m = new Map([['foo',0],['bar',1]]);
            var count = 0, out = [];
            m.forEach(function(v,k){
                if (count === 0) { m.delete('foo'); m.set('foo','baz'); }
                out.push(k+'='+v); count++;
            });
            count + '|' + out.join(',') + '|' + m.size
        "#),
        "3|foo=0,bar=1,foo=baz|2"
    );
}

#[test]
fn set_composition_iterates_this_live() {
    // `isSubsetOf` iterates `this` live: an element deleted by the argument's
    // `has` is skipped rather than re-tested.
    assert_eq!(
        run(r#"
            var base = new Set(['a','b','c']);
            var evil = { size: 3, has(v){ if (v==='a') base.delete('c'); return ['x','a','b'].includes(v); }, keys(){ throw new Error('nope'); } };
            base.isSubsetOf(evil) + '|' + [...base].join(',')
        "#),
        "true|a,b"
    );
}

#[test]
fn reflect_apply_normalizes_array_holes() {
    // A hole in the args array reaches the callee as `undefined`, not the internal
    // hole sentinel (so `Object.is(args[2], undefined)` holds).
    assert_eq!(
        run(r#"
            var got;
            function f(){ got = arguments; }
            Reflect.apply(f, null, ['a', 2, , null]);
            got.length + '|' + Object.is(got[2], undefined) + '|' + (got[3] === null)
        "#),
        "4|true|true"
    );
}

#[test]
fn string_coercion_honors_overridden_array_tostring() {
    assert_eq!(
        run(r#"
            Array.prototype.toString = function(){ return '__A__'; };
            String([1,2]) + '|' + String(new Array)
        "#),
        "__A__|__A__"
    );
}

#[test]
fn replace_all_passes_this_object_to_symbol_replace() {
    // `O` handed to `@@replace` is the `this` value of `replaceAll` — the wrapper
    // object for a `new String(...)` receiver, not the boxed primitive.
    assert_eq!(
        run(r#"
            var sv = /./g;
            var str = new String('Leo');
            var seen;
            Object.defineProperty(sv, Symbol.replace, { value: function(O){ seen = (O === str); return 42; } });
            str.replaceAll(sv, {}) + '|' + seen
        "#),
        "42|true"
    );
}

#[test]
fn private_field_on_proxy_and_returned_object() {
    // A private field brands a Proxy returned by a base constructor (bypassing its
    // traps) and is readable.
    assert_eq!(
        run(r#"
            var trapped = [];
            class Base { constructor(){ return new Proxy(this, { get(o,p){ trapped.push(p); return o[p]; } }); } }
            class Test extends Base { #f = 3; method(){ return this.#f; } }
            new Test().method() + '|' + JSON.stringify(trapped)
        "#),
        r#"3|["method"]"#
    );
    // A base constructor that returns an object rebinds `this`; the derived class's
    // private field lands on that object.
    assert_eq!(
        run(r#"
            class TB { constructor(o){ return o; } }
            class C extends TB { #val = 42; static val(o){ return o.#val; } constructor(o){ super(o); } }
            var t = new C({});
            C.val(t)
        "#),
        "42"
    );
}

#[test]
fn public_hash_named_field_enumerates() {
    // A computed public field whose name starts with `#` is an ordinary enumerable
    // property (only the `\0`-sentinel internal slots are hidden).
    assert_eq!(
        run(r##"
            class C { ["#constructor"] = 42; }
            var c = new C();
            JSON.stringify(Object.keys(c)) + '|' + c.propertyIsEnumerable('#constructor')
        "##),
        r##"["#constructor"]|true"##
    );
    // A real private field stays hidden.
    assert_eq!(
        run(r##"class P { #secret = 1; pub = 2; } JSON.stringify(Object.keys(new P()))"##),
        r#"["pub"]"#
    );
}

#[test]
fn iterator_from_primitive_string_fires_symbol_iterator_getter() {
    // `Iterator.from` on a primitive string reads `@@iterator` with the string
    // primitive as the receiver (GetV semantics).
    assert_eq!(
        run(r#"
            var seen;
            var orig = String.prototype[Symbol.iterator];
            Object.defineProperty(String.prototype, Symbol.iterator, {
                get(){ 'use strict'; seen = typeof this; return orig; }, configurable: true
            });
            Iterator.from('');
            var a = seen;
            Iterator.from(new String(''));
            a + '|' + seen
        "#),
        "string|object"
    );
}

#[test]
#[cfg(feature = "intl")]
fn intl_collator_subclass_super_carries_slot() {
    // `class X extends Intl.Collator { constructor(l,o){ super(l,o); } }`: the
    // derived instance must carry the Collator internal slot so its
    // `compare`/`resolvedOptions` bound-function getters accept the receiver
    // (the `super()` into the native ctor initializes the slot in place).
    assert_eq!(
        run(r#"
            class MyCollator extends Intl.Collator {
                constructor(locales, options) { super(locales, options); }
            }
            var c = new MyCollator(['en']);
            var a = ['B', 'A'];
            a.sort(c.compare);
            a.join(',') + '|' + (c instanceof MyCollator) + '|' + (c instanceof Intl.Collator)
        "#),
        "A,B|true|true"
    );
    // A constructor-less subclass works the same way.
    assert_eq!(
        run(r#"class C extends Intl.Collator {} typeof new C('en').resolvedOptions().locale"#),
        "string"
    );
}

#[test]
fn monkey_patched_promise_then_honored_on_direct_call() {
    // A user reassignment of the *inherited* `Promise.prototype.then` must be
    // observed on a direct `promise.then(...)` call (the native fast-path used to
    // bypass the prototype chain). Regression coverage for the reaction/species
    // call-counting fixes.
    assert_eq!(
        run(r#"
            var orig = Promise.prototype.then;
            var called = false;
            Promise.prototype.then = function (a, b) { called = true; return orig.call(this, a, b); };
            Promise.resolve(1).then(function () {});
            Promise.prototype.then = orig;
            called
        "#),
        "true"
    );
}

#[test]
fn promise_finally_then_call_counting() {
    // `finally` on a resolved subclass produces the spec number of `then` calls
    // (the thenable-adoption path now goes through `Promise.prototype.then`).
    let out = |src: &str| {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        interp.run(&program).expect("exec");
        String::from(interp.output())
    };
    assert_eq!(
        out(r#"
            class MyPromise extends Promise {}
            var mp1 = MyPromise.resolve({});
            var mp2 = MyPromise.resolve(42);
            var thenCalls = [];
            var then = Promise.prototype.then;
            Promise.prototype.then = function (a, b) { thenCalls.push(this); return then.call(this, a, b); };
            mp1.finally(function () { return mp2; }).then(function () {
                console.log('count=' + thenCalls.length);
            }).then(function () {}, function () {});
        "#),
        "count=5\n"
    );
}

#[test]
fn static_block_has_own_variable_environment() {
    // A `var` inside a `static { }` block does not leak to the enclosing scope,
    // and each static block is independent.
    assert_eq!(
        run(r#"var t = 'outer'; class C { static { var t = 'inner'; } } t"#),
        "outer"
    );
    assert_eq!(
        run(r#"
            var p1, p2;
            class C {
                static { var t = 'first'; p1 = t; }
                static { var t = 'second'; p2 = t; }
            }
            p1 + ',' + p2
        "#),
        "first,second"
    );
}

#[test]
fn constructor_body_hoists_vars_like_function_code() {
    // A strict class constructor body is function code: its `var` bindings hoist,
    // so a write from a nested `catch` scope updates the hoisted binding rather
    // than throwing a spurious ReferenceError.
    assert_eq!(
        run(r#"
            class C {
                constructor() {
                    var e;
                    try { throw new TypeError('x'); } catch (err) { e = err; }
                    this.n = e.constructor.name;
                }
            }
            new C().n
        "#),
        "TypeError"
    );
}

#[test]
fn super_property_before_super_call_throws_reference_error() {
    // `super.m()` in a derived constructor before `super(...)` does GetThisBinding,
    // which throws a ReferenceError (not a raw string) when `this` is uninitialized.
    let out = |src: &str| {
        let program = Parser::parse_program(src).expect("parse");
        let mut interp = Interp::new();
        let _ = interp.run(&program);
        String::from(interp.output())
    };
    assert_eq!(
        out(r#"
            class B { m() { return 1; } }
            class D extends B {
                constructor() {
                    var r = 'none';
                    try { super.m(); } catch (e) { r = (e instanceof ReferenceError); }
                    console.log(r);
                    super();
                }
            }
            new D();
        "#),
        "true\n"
    );
}

#[test]
fn class_constructor_caller_arguments_are_poisoned() {
    // A class constructor inherits the poisoned `Function.prototype.caller`/
    // `.arguments` accessors: both read and write throw a TypeError (and no own
    // property is created by the write).
    assert_eq!(
        run(r#"
            class C {}
            var r = 'ok';
            try { C.caller = {}; r = 'no-throw'; } catch (e) { r = e.constructor.name; }
            r + '|' + C.hasOwnProperty('caller')
        "#),
        "TypeError|false"
    );
}

#[test]
fn concatenated_boundary_surrogates_coalesce() {
    // WTF-8 canonicalization: a high surrogate concatenated with a low surrogate
    // pairs into the astral scalar, so the two encodings compare equal, iterate as
    // one code point, and slice splits them back into lone surrogates.
    assert_eq!(run(r#"('\uD83D' + '\uDCA9') === '\u{1F4A9}'"#), "true");
    assert_eq!(run(r#"('\uD83D' + '\uDCA9').length"#), "2");
    assert_eq!(run(r#"[...('\uD83D' + '\uDCA9')].length"#), "1");
    // slice() splits an astral character into its lone surrogate halves.
    assert_eq!(
        run(r#""\u{1F4A9}".slice(0, 1).charCodeAt(0) === 0xD83D"#),
        "true"
    );
    assert_eq!(
        run(r#""\u{1F4A9}".slice(1).charCodeAt(0) === 0xDCA9"#),
        "true"
    );
    assert_eq!(run(r#""\u{1F4A9}".slice(0, 1).isWellFormed()"#), "false");
    assert_eq!(
        run(r#"('a' + '\uD83D' + '\uDCA9' + 'd').toWellFormed() === 'a\u{1F4A9}d'"#),
        "true"
    );
}

/// ECMA-402 Temporal formatting protocol: `Intl.DateTimeFormat.prototype.format`
/// formats a Temporal object WITHOUT calling its (throwing) `valueOf`, restricting
/// the output to the type's data model.
#[cfg(feature = "intl")]
#[test]
fn dtf_format_temporal_plain_types() {
    // PlainDate → date only; PlainTime → time only; PlainMonthDay drops the year;
    // PlainYearMonth drops the day; PlainDateTime keeps both.
    assert_eq!(
        run(r#"new Intl.DateTimeFormat("en-US").format(new Temporal.PlainDate(2021,8,4))"#),
        "8/4/2021"
    );
    assert_eq!(
        run(
            r#"new Intl.DateTimeFormat("en-US").format(new Temporal.PlainMonthDay(8,4,"gregory"))"#
        ),
        "8/4"
    );
    assert_eq!(
        run(
            r#"new Intl.DateTimeFormat("en-US").format(new Temporal.PlainYearMonth(2021,8,"gregory"))"#
        ),
        "8/2021"
    );
    assert_eq!(
        run(
            r#"new Intl.DateTimeFormat("en-US").format(new Temporal.PlainDateTime(2021,8,4,0,30,45,123)).includes("8/4/2021")"#
        ),
        "true"
    );
}

/// A `Temporal.ZonedDateTime` is rejected by `format` with a `TypeError`, and a
/// PlainTime with a date-only formatter (no overlap) is a `TypeError`.
#[cfg(feature = "intl")]
#[test]
fn dtf_format_temporal_errors() {
    assert_eq!(
        run(
            r#"try{new Intl.DateTimeFormat("en").format(new Temporal.ZonedDateTime(0n,"UTC"));"no"}catch(e){e.constructor.name}"#
        ),
        "TypeError"
    );
    // A year-only formatter cannot format a PlainTime (no field overlap).
    assert_eq!(
        run(
            r#"try{new Intl.DateTimeFormat("en",{year:"numeric"}).format(new Temporal.PlainTime(1,2));"no"}catch(e){e.constructor.name}"#
        ),
        "TypeError"
    );
}

/// `Temporal.<Type>.prototype.toLocaleString` formats through the same protocol,
/// matching `Intl.DateTimeFormat.prototype.format`.
#[cfg(feature = "intl")]
#[test]
fn temporal_to_locale_string_matches_format() {
    assert_eq!(
        run(r#"var d=new Temporal.PlainDate(1976,11,18);
               d.toLocaleString("en-US")===new Intl.DateTimeFormat("en-US").format(d)"#),
        "true"
    );
    assert_eq!(
        run(r#"new Temporal.PlainYearMonth(2021,8,"gregory").toLocaleString("en-US")"#),
        "8/2021"
    );
}

/// A `-u-ca-` locale extension resolves the DateTimeFormat calendar so an iso8601
/// Temporal object is calendar-compatible.
#[cfg(feature = "intl")]
#[test]
fn dtf_temporal_locale_calendar_extension() {
    assert_eq!(
        run(
            r#"new Temporal.PlainYearMonth(2024,12,"iso8601",26).toLocaleString("en-u-ca-iso8601",{timeZone:"UTC"})"#
        ),
        "12/2024"
    );
}

/// `Intl.ListFormat` honors `type`/`style` in `format` and `formatToParts`
/// (English CLDR reference patterns for short/narrow and `unit`).
#[cfg(feature = "intl")]
#[test]
fn intl_list_format_type_style() {
    assert_eq!(
        run(r#"new Intl.ListFormat("en",{style:"short"}).format(["a","b"])"#),
        "a & b"
    );
    assert_eq!(
        run(r#"new Intl.ListFormat("en",{type:"unit",style:"narrow"}).format(["a","b","c"])"#),
        "a b c"
    );
    // formatToParts uses the same style-aware connectors.
    assert_eq!(
        run(r#"new Intl.ListFormat("en",{style:"short"}).formatToParts(["a","b"])[1].value"#),
        " & "
    );
}

/// `Intl.Collator` ResolveLocale: `-u-kn-`/`-u-kf-` set numeric/caseFirst, while
/// non-relevant or unsupported Unicode extensions are dropped from the resolved
/// locale (leaving resolvedOptions identical to the plain locale).
#[cfg(feature = "intl")]
#[test]
fn intl_collator_resolve_locale_extensions() {
    assert_eq!(
        run(r#"new Intl.Collator("en-u-kn-true").resolvedOptions().numeric"#),
        "true"
    );
    assert_eq!(
        run(r#"new Intl.Collator("en-u-kf-upper").resolvedOptions().caseFirst"#),
        "upper"
    );
    // An irrelevant/unsupported extension key does not affect the resolved locale.
    assert_eq!(
        run(
            r#"new Intl.Collator("en-u-kb-true").resolvedOptions().locale===new Intl.Collator("en").resolvedOptions().locale"#
        ),
        "true"
    );
    // A private-use `-x-u-...` is not a real extension.
    assert_eq!(
        run(r#"new Intl.Collator("de-x-u-co-phonebk").resolvedOptions().collation"#),
        "default"
    );
    // ignorePunctuation:false keeps spaces significant (NonIgnorable weighting).
    assert_eq!(
        run(r#"new Intl.Collator("en",{ignorePunctuation:false}).compare("a","a ")<0"#),
        "true"
    );
}

/// `Intl.supportedValuesOf("numberingSystem")` returns the full CLDR set, and the
/// numbering system resolves through the `-u-nu-` extension in the formatters.
#[cfg(feature = "intl")]
#[test]
fn intl_numbering_system_resolution() {
    assert_eq!(
        run(r#"Intl.supportedValuesOf("numberingSystem").includes("armn")"#),
        "true"
    );
    assert_eq!(
        run(r#"new Intl.RelativeTimeFormat("en-u-nu-arab").resolvedOptions().numberingSystem"#),
        "arab"
    );
    // A known option overrides the extension; an unknown one falls back to it.
    assert_eq!(
        run(
            r#"new Intl.RelativeTimeFormat("en-u-nu-arab",{numberingSystem:"invalid"}).resolvedOptions().numberingSystem"#
        ),
        "arab"
    );
    assert_eq!(
        run(
            r#"new Intl.DurationFormat("en",{numberingSystem:"adlm"}).resolvedOptions().numberingSystem"#
        ),
        "adlm"
    );
}

/// `Intl.PluralRules` with compact notation derives the compact-exponent operand,
/// so French selects `many` for 1.5e6 (which is `other` in standard notation).
#[cfg(feature = "intl")]
#[test]
fn intl_plural_rules_compact_notation() {
    assert_eq!(
        run(r#"new Intl.PluralRules("fr",{notation:"compact"}).select(1.5e6)"#),
        "many"
    );
    assert_eq!(
        run(r#"new Intl.PluralRules("fr",{notation:"standard"}).select(1.5e6)"#),
        "other"
    );
}

/// `String.prototype.localeCompare` initializes a Collator (validating locales /
/// options) and uses real UCA collation (lowercase sorts before uppercase).
#[cfg(feature = "intl")]
#[test]
fn intl_locale_compare_collator_semantics() {
    assert_eq!(run(r#"("a").localeCompare("A")"#), "-1");
    // Invalid `locales`/`options` throw the same errors as `new Intl.Collator`.
    assert_eq!(
        run(r#"try{("").localeCompare("",null);"ok"}catch(e){e.constructor.name}"#),
        "TypeError"
    );
    assert_eq!(
        run(
            r#"try{("").localeCompare("","de",{usage:"invalid"});"ok"}catch(e){e.constructor.name}"#
        ),
        "RangeError"
    );
}

/// `Intl.DurationFormat` treats a negative-zero field as `+0` (no minus sign).
#[cfg(feature = "intl")]
#[test]
fn intl_duration_format_negative_zero() {
    assert_eq!(
        run(
            r#"new Intl.DurationFormat("en",{yearsDisplay:"always"}).format({years:-0})===new Intl.DurationFormat("en",{yearsDisplay:"always"}).format({years:0})"#
        ),
        "true"
    );
}

/// `Intl.Segmenter`: `segment` does `ToString` (Symbol throws) and
/// `%Segments.prototype%.containing` brand-checks its receiver.
#[cfg(feature = "intl")]
#[test]
fn intl_segmenter_string_and_branding() {
    assert_eq!(
        run(r#"try{new Intl.Segmenter().segment(Symbol());"ok"}catch(e){e.constructor.name}"#),
        "TypeError"
    );
    assert_eq!(
        run(
            r#"var c=new Intl.Segmenter().segment("x").containing; try{c.call({});"ok"}catch(e){e.constructor.name}"#
        ),
        "TypeError"
    );
}

/// `String.prototype.toLocaleUpperCase`/`toLocaleLowerCase` validate the locale
/// argument and apply Turkic case tailoring.
#[cfg(feature = "intl")]
#[test]
fn intl_to_locale_case() {
    assert_eq!(
        run(
            r#"try{("x").toLocaleUpperCase("not a valid locale");"ok"}catch(e){e.constructor.name}"#
        ),
        "RangeError"
    );
    // Turkish dotless-i tailoring for uppercase.
    assert_eq!(run(r#"("i").toLocaleUpperCase("tr")"#), "\u{130}");
}

#[test]
fn flatmap_forwards_this_arg() {
    // `flatMap(callbackfn, thisArg)` forwards the second argument as `this`.
    assert_eq!(
        run("var m={tag:7};[1].flatMap(function(){return [this.tag];}, m)[0]"),
        "7"
    );
}

#[test]
fn array_species_create_through_proxy_receiver() {
    // `Array.prototype.{map,filter,slice,splice,concat}` on a Proxy whose target is
    // an Array must run ArraySpeciesCreate against the *original* receiver (reading
    // `constructor`/`@@species` through the proxy), not a materialized snapshot.
    let harness = "var array=[1,2,3];\
        var proxy=new Proxy(new Proxy(array,{}),{});\
        var Ctor=function(){};\
        array.constructor=function(){};\
        array.constructor[Symbol.species]=Ctor;";
    for m in ["map", "filter"] {
        assert_eq!(
            run(&alloc::format!(
                "{harness}Object.getPrototypeOf(Array.prototype.{m}.call(proxy,function(){{return true;}}))===Ctor.prototype"
            )),
            "true",
            "method {m}"
        );
    }
    for m in ["slice", "splice", "concat"] {
        assert_eq!(
            run(&alloc::format!(
                "{harness}Object.getPrototypeOf(Array.prototype.{m}.call(proxy))===Ctor.prototype"
            )),
            "true",
            "method {m}"
        );
    }
}

#[test]
fn array_prototype_has_own_length_zero() {
    // `Array.prototype` exposes an own `length` (0, writable, non-enumerable,
    // non-configurable), so `"length" in Object.create(Array.prototype)` holds.
    assert_eq!(run("Array.prototype.length"), "0");
    assert_eq!(run(r#""length" in Object.create(Array.prototype)"#), "true");
    assert_eq!(
        run(
            "var d=Object.getOwnPropertyDescriptor(Array.prototype,'length');\
             [d.value,d.writable,d.enumerable,d.configurable].join(',')"
        ),
        "0,true,false,false"
    );
    // Not enumerable — does not appear in Object.keys.
    assert_eq!(run("Object.keys(Array.prototype).length"), "0");
}

#[test]
fn weak_collection_tag_and_constructor_not_conflated_with_strong() {
    // A WeakMap/WeakSet inherits from its own prototype, not Map/Set: deleting the
    // prototype's `@@toStringTag` falls back to "[object Object]" (not "[object
    // Map]"), and `.constructor` is WeakMap/WeakSet.
    assert_eq!(run("(new WeakMap()).constructor===WeakMap"), "true");
    assert_eq!(run("(new WeakSet()).constructor===WeakSet"), "true");
    assert_eq!(
        run(
            "var wm=new WeakMap();delete WeakMap.prototype[Symbol.toStringTag];\
             Object.prototype.toString.call(wm)"
        ),
        "[object Object]"
    );
    assert_eq!(
        run(
            "var ws=new WeakSet();delete WeakSet.prototype[Symbol.toStringTag];\
             Object.prototype.toString.call(ws)"
        ),
        "[object Object]"
    );
    // The weak collection's own methods still resolve (from its prototype).
    assert_eq!(run("typeof (new WeakMap()).set"), "function");
}

#[test]
fn async_function_prototype_tostringtag_and_ctor() {
    // A (non-generator) async function reports "[object AsyncFunction]" — even
    // through a Proxy wrapper (the tag is read from %AsyncFunction.prototype%).
    assert_eq!(
        run("Object.prototype.toString.call(async function(){})"),
        "[object AsyncFunction]"
    );
    assert_eq!(
        run("Object.prototype.toString.call(new Proxy(async function(){}, {}))"),
        "[object AsyncFunction]"
    );
    assert_eq!(
        run("Object.prototype.toString.call(async () => {})"),
        "[object AsyncFunction]"
    );
    // %AsyncFunction% is a distinct constructor reachable via `.constructor`.
    assert_eq!(
        run("(async function(){}).constructor.name"),
        "AsyncFunction"
    );
    assert_eq!(
        run("(async function(){}).constructor === Function"),
        "false"
    );
    // Its prototype carries the tag; removing it falls back to "[object Function]".
    assert_eq!(
        run("var af=(async function(){}).constructor;\
             Object.defineProperty(af.prototype, Symbol.toStringTag, {value:undefined});\
             Object.prototype.toString.call(async function(){})"),
        "[object Function]"
    );
    // A plain async function's [[Prototype]] is NOT %Function.prototype%.
    assert_eq!(
        run("Object.getPrototypeOf(async function(){}) === Function.prototype"),
        "false"
    );
}

#[test]
fn proxy_set_preserves_receiver() {
    // A trapless-forward [[Set]] keeps the Receiver, so the inherited
    // `Object.prototype.__proto__` setter runs against the proxy and its
    // [[SetPrototypeOf]] trap fires (abrupt completion propagates).
    assert_eq!(
        run("var thrown=false;\
             var p=new Proxy({}, {setPrototypeOf(){ throw new Error('t'); }});\
             try { p.__proto__ = {}; } catch(e){ thrown = e.message==='t'; }\
             thrown"),
        "true"
    );
    // OrdinarySetWithOwnDescriptor writes to the Receiver: a proxy Receiver with
    // no set trap but a defineProperty trap routes the write through it.
    assert_eq!(
        run("var log=[];\
             var p=new Proxy({foo:1}, {defineProperty(t,k,d){ log.push(k); return Reflect.defineProperty(t,k,d); }});\
             p.foo=2; p.foo=2;\
             log.join(',')"),
        "foo,foo"
    );
    // A trapless proxy whose *target* is itself a proxy forwards the set.
    assert_eq!(
        run("var hit='';\
             var inner=new Proxy({}, {set(t,k,v){ hit=k; return true; }});\
             var outer=new Proxy(inner, {});\
             outer.x=5; hit"),
        "x"
    );
}

#[test]
fn proxy_get_preserves_receiver() {
    // A trapless-forward [[Get]] runs an inherited getter with `this` = the proxy.
    assert_eq!(
        run("var t={get attr(){ return this; }};\
             var p=new Proxy(t, {get:null});\
             p.attr === p"),
        "true"
    );
    // A proxy on the prototype chain also preserves the original receiver.
    assert_eq!(
        run("var t={get attr(){ return this; }};\
             var pp=Object.create(new Proxy(t, {}));\
             pp.attr === pp"),
        "true"
    );
}

#[test]
fn object_keys_proxy_calls_get_own_property() {
    // EnumerableOwnPropertyNames calls [[GetOwnProperty]] per key even with no
    // getOwnPropertyDescriptor trap — the trap *lookup* on a proxy-wrapped handler
    // is observable. An empty array's own key list is ["length"].
    assert_eq!(
        run("var log=[];\
             Object.keys(new Proxy([], new Proxy({}, {get(t,pk){ log.push(pk); }})));\
             log.join(',')"),
        "ownKeys,getOwnPropertyDescriptor"
    );
}

#[test]
fn proxy_is_extensible_invariant() {
    // The isExtensible trap result must equal the target's actual extensibility.
    assert_eq!(
        run(
            "var p=new Proxy(Object.freeze({}), {isExtensible(){ return true; }});\
             var threw=false;\
             try { Object.isExtensible(p); } catch(e){ threw = e instanceof TypeError; }\
             threw"
        ),
        "true"
    );
    // A trap that agrees with the target is fine.
    assert_eq!(
        run(
            "var p=new Proxy({}, {isExtensible(t){ return Reflect.isExtensible(t); }});\
             Object.isExtensible(p)"
        ),
        "true"
    );
}

#[test]
fn string_wrapper_defined_index_is_read() {
    // String-exotic [[GetOwnProperty]] falls back to OrdinaryGetOwnProperty for an
    // out-of-range index, so a defined own property at that index is readable.
    assert_eq!(
        run("var s=new String('str');\
             Object.defineProperty(s,'4',{value:4,writable:true,enumerable:true,configurable:true});\
             s[4]"),
        "4"
    );
    // An out-of-range index with no own property is still undefined.
    assert_eq!(run("(new String('str'))[9]"), "undefined");
    // A primitive string is unaffected.
    assert_eq!(run("'str'[4]"), "undefined");
}

#[test]
fn regexp_last_index_partial_define_keeps_writable() {
    // A `{ value }`-only redefine of the synthesized `lastIndex` keeps
    // writable:true (it must not fall to the new-property writable:false default).
    assert_eq!(
        run("var re=/x/;\
             Reflect.defineProperty(re,'lastIndex',{value:'foo'});\
             Object.getOwnPropertyDescriptor(re,'lastIndex').writable"),
        "true"
    );
    assert_eq!(
        run("var re=/x/;\
             Reflect.defineProperty(re,'lastIndex',{value:7});\
             re.lastIndex"),
        "7"
    );
    // exec still resets lastIndex through the materialized slot.
    assert_eq!(
        run("var re=/x/g;\
             Object.defineProperty(re,'lastIndex',{value:0});\
             re.exec('xx'); re.lastIndex"),
        "1"
    );
}

#[test]
fn object_tolocalestring_primitive_this_receiver() {
    // `Object.prototype.toLocaleString` invokes `toString` with Receiver = the
    // original `this`. A strict `toString` accessor getter reading `typeof this`
    // therefore sees the primitive ("boolean"), not a boxed wrapper ("object").
    assert_eq!(
        run("'use strict';\
             Object.defineProperty(Boolean.prototype, 'toString', {\
               get: function(){ var v = typeof this; return function(){ return v; }; }, configurable:true });\
             true.toLocaleString()"),
        "boolean"
    );
}

#[test]
fn proxy_set_typed_array_integer_index_exotic() {
    // A canonical numeric index reaching a TypedArray in the receiver's prototype
    // chain is the integer-indexed exotic [[Set]] — it never walks past the view
    // to an inherited setter. An out-of-bounds/invalid index is a silent no-op.
    assert_eq!(
        run("Object.defineProperty(Float64Array.prototype, '1', {\
               set: function(){ throw new Error('unreachable'); }, configurable:true });\
             var target=new Float64Array([0]);\
             var recv=new Proxy(Object.create(target), {\
               defineProperty(){ throw new Error('define unreachable'); } });\
             recv[1] = 2.3;\
             (!target.hasOwnProperty('1')) + ',' + (!Object.prototype.hasOwnProperty.call(recv,'1'))"),
        "true,true"
    );
    // A valid index on a proxy directly wrapping a TypedArray writes the element
    // (through the trapless [[DefineOwnProperty]] forward to the view).
    assert_eq!(
        run("var ta=new Float64Array([0,0,0]);\
             var p=new Proxy(ta, {});\
             p[1]=5; ta[1]"),
        "5"
    );
}

#[test]
fn abstract_module_source_intrinsic_shape() {
    // The `%AbstractModuleSource%` intrinsic (source-phase-imports proposal) is
    // exposed to Test262 via the `$262_AbstractModuleSource()` host hook. Only the
    // intrinsic *shape* is materialized (no loadable module sources yet). A single
    // program captures one instance in `C` and checks the full intrinsic shape:
    //   - it is a function whose `[[Prototype]]` is %FunctionPrototype%;
    //   - `name` = "AbstractModuleSource" and `length` = 0, each
    //     { writable: false, enumerable: false, configurable: true };
    //   - the abstract constructor throws a TypeError on `[[Construct]]`;
    //   - `prototype` is { writable: false, enumerable: false, configurable: false }
    //     and its `[[Prototype]]` is %Object.prototype%;
    //   - `prototype.constructor` is a data property back to `C`
    //     { writable: true, enumerable: false, configurable: true };
    //   - `prototype[@@toStringTag]` is a getter-only accessor
    //     { enumerable: false, configurable: true } returning undefined for any
    //     receiver lacking a `[[ModuleSourceClassName]]` slot.
    let program = r#"
        var C = $262_AbstractModuleSource();
        var ok = typeof C === 'function'
            && Object.getPrototypeOf(C) === Function.prototype;
        var nd = Object.getOwnPropertyDescriptor(C, 'name');
        ok = ok && nd.value === 'AbstractModuleSource' && nd.writable === false
            && nd.enumerable === false && nd.configurable === true;
        var ld = Object.getOwnPropertyDescriptor(C, 'length');
        ok = ok && ld.value === 0 && ld.writable === false
            && ld.enumerable === false && ld.configurable === true;
        var threw = false;
        try { new C(); } catch (e) { threw = e instanceof TypeError; }
        ok = ok && threw;
        var pd = Object.getOwnPropertyDescriptor(C, 'prototype');
        ok = ok && pd.writable === false && pd.enumerable === false
            && pd.configurable === false;
        ok = ok && Object.getPrototypeOf(C.prototype) === Object.prototype;
        var cd = Object.getOwnPropertyDescriptor(C.prototype, 'constructor');
        ok = ok && cd.value === C && cd.writable === true
            && cd.enumerable === false && cd.configurable === true;
        var td = Object.getOwnPropertyDescriptor(C.prototype, Symbol.toStringTag);
        ok = ok && typeof td.get === 'function' && td.set === undefined
            && td.enumerable === false && td.configurable === true
            && td.get.call(262) === undefined
            && td.get.call(C.prototype) === undefined;
        ok
    "#;
    assert_eq!(run(program), "true");
}

#[test]
fn shadow_realm_evaluate_isolates_globalthis_side_effects() {
    // A `ShadowRealm` has its own genuinely-distinct `globalThis`: assignments in
    // the shadow realm never touch the host realm's globals, and successive
    // `evaluate` calls on the same instance share the shadow realm's globals.
    let program = r#"
        globalThis.myValue = 'host';
        const r = new ShadowRealm();
        r.evaluate('globalThis.myValue = "shadow";');
        const inShadow = r.evaluate('globalThis.myValue');
        // Host is untouched; shadow retained its own write.
        (globalThis.myValue === 'host') && (inShadow === 'shadow')
    "#;
    assert_eq!(run(program), "true");
}

#[test]
fn shadow_realm_instances_do_not_share_globals() {
    // Two distinct `ShadowRealm` instances each have their own `globalThis`.
    let program = r#"
        const a = new ShadowRealm();
        const b = new ShadowRealm();
        a.evaluate('globalThis.x = 1;');
        b.evaluate('globalThis.x = 2;');
        (a.evaluate('globalThis.x') === 1) && (b.evaluate('globalThis.x') === 2)
    "#;
    assert_eq!(run(program), "true");
}

#[test]
fn shadow_realm_globalthis_properties_are_configurable() {
    // Every host-added `globalThis` property in a shadow realm is configurable
    // (deletable), except the ES non-configurable value trio.
    let program = r#"
        const r = new ShadowRealm();
        r.evaluate(`
          const names = Object.keys(Object.getOwnPropertyDescriptors(globalThis));
          const nonConfig = ['undefined', 'Infinity', 'NaN'];
          const missed = Object.entries(Object.getOwnPropertyDescriptors(globalThis))
            .filter(e => e[1].configurable === false)
            .map(e => e[0])
            .filter(n => !nonConfig.includes(n));
          missed.join(',');
        `)
    "#;
    assert_eq!(run(program), "");
}

#[test]
fn shadow_realm_evaluate_wraps_callable_and_marshals_primitives() {
    // A callable result is exposed as a wrapped function; calling it marshals
    // primitives across the boundary. A non-primitive, non-callable argument or
    // return value is a TypeError.
    let program = r#"
        const r = new ShadowRealm();
        const add = r.evaluate('(a, b) => a + b');
        let ok = (typeof add === 'function') && (add(2, 3) === 5);
        // Passing an object argument across the boundary throws.
        let threwArg = false;
        try { add({}, 1); } catch (e) { threwArg = e instanceof TypeError; }
        // Returning a non-callable object across the boundary throws.
        let threwRet = false;
        try { r.evaluate('({})'); } catch (e) { threwRet = e instanceof TypeError; }
        ok && threwArg && threwRet
    "#;
    assert_eq!(run(program), "true");
}

#[test]
fn shadow_realm_nested_realms_stay_isolated() {
    // A shadow realm can create its own nested shadow realm; the three realms
    // (host, realm1, realm2) each keep an independent `globalThis`.
    let program = r#"
        globalThis.myValue = 'a';
        const realm1 = new ShadowRealm();
        realm1.evaluate('globalThis.myValue = "b";');
        const realm2Evaluate = realm1.evaluate(`
          const realm2 = new ShadowRealm();
          (str) => realm2.evaluate(str);
        `);
        realm2Evaluate('globalThis.myValue = "c";');
        (globalThis.myValue === 'a')
          && (realm1.evaluate('globalThis.myValue') === 'b')
          && (realm2Evaluate('globalThis.myValue') === 'c')
    "#;
    assert_eq!(run(program), "true");
}

#[test]
fn per_iteration_let_env_is_flat_and_captures_per_iteration() {
    // The classic per-iteration capture: each closure sees its own `i`.
    assert_eq!(
        run("var f=[]; for (let i=0;i<3;i++) f.push(()=>i); f.map(g=>g()).join(',')"),
        "0,1,2"
    );
    // The increment and the body see the same (current) iteration binding.
    assert_eq!(run("var s=0; for (let i=0;i<5;i++) { s+=i; } s"), "10");
    // …and a closure made in the *increment* also captures per-iteration.
    assert_eq!(
        run("var f=[]; for (let i=0;i<3;i++,f.push(()=>i)); f.map(g=>g()).join(',')"),
        "1,2,3"
    );
    // Each iteration environment is a sibling parented on the loop's OUTER
    // scope, not on the previous iteration (14.7.4.4 step 1.b) — so an outer
    // binding stays exactly one lookup away however many iterations have run.
    // Chaining them made every lookup in the loop walk one link per completed
    // iteration; this trip count is large enough that the quadratic form takes
    // minutes while the correct one is instant.
    assert_eq!(
        run("var s=0; for (let i=0;i<20000;i++) { s+=1; } s"),
        "20000"
    );
    // A `const` head keeps its immutability across the per-iteration copy.
    assert_eq!(
        run(
            "var ok=false; try { for (const c=0;;) { c=1; } } catch (e) { ok = e instanceof TypeError; } ok"
        ),
        "true"
    );
}

#[test]
fn array_length_shrink_from_sparse_length_terminates() {
    // ArraySetLength's stop-at-non-configurable scan must be bounded by the
    // indices that can actually exist, not by the *logical* length: shrinking
    // back from 2**32-1 on an array carrying a non-configurable index used to
    // walk every one of the 4.29 billion indices in between.
    assert_eq!(
        run(
            "var a=[0,1]; Object.defineProperty(a,'1',{value:1,configurable:false});
             a.length = 4294967295; a.length = 2; a.length"
        ),
        "2"
    );
    // The stop-at behaviour itself is unchanged: a shrink past a
    // non-configurable index leaves the length one above it.
    assert_eq!(
        run(
            "var a=[0,1,2]; Object.defineProperty(a,'1',{value:1,configurable:false});
             a.length = 0; a.length"
        ),
        "2"
    );
    // …and defining `length` below it is a TypeError.
    assert_eq!(
        run(
            "var a=[0,1]; Object.defineProperty(a,'1',{value:1,configurable:false});
             var ok=false; try { Object.defineProperty(a,'length',{value:1}); }
             catch (e) { ok = e instanceof TypeError; } ok && a.length === 2"
        ),
        "true"
    );
    // A *sparse* (past-the-dense-cap) non-configurable index still stops it.
    assert_eq!(
        run(
            "var a=[]; a[3]=1; Object.defineProperty(a,'3',{value:1,configurable:false});
             a.length = 0; a.length"
        ),
        "4"
    );
}

#[test]
fn string_concat_is_linear_not_quadratic() {
    // `+=` builds a rope in O(1), but the surrounding `is a string?` type tests
    // used to go through `string_value`, which MATERIALIZES the rope — turning
    // every append into a full copy. This many appends takes seconds if the
    // quadratic behaviour returns, and milliseconds otherwise.
    assert_eq!(
        run("var s=''; for (var i=0;i<40000;i++) s+='x'; s.length"),
        "40000"
    );
    // The rope still reads back correctly through the type-test paths.
    assert_eq!(
        run("var s=''; for (var i=0;i<5;i++) s+=i; s + '|' + (typeof s) + '|' + s.charAt(3)"),
        "01234|string|3"
    );
    // A lone-surrogate boundary pair still coalesces across a concat.
    assert_eq!(
        run(r#"var s="\uD83D"; s += "\uDE00"; s.length + ":" + s.codePointAt(0)"#),
        "2:128512"
    );
}

#[test]
fn legacy_regexp_statics_are_lazy_but_correct() {
    // Annex B.2.5 statics reflect the last successful match. They are recorded
    // as index ranges and materialized on read; a global match must not pay for
    // building them per match (that made `replace(/x/g, …)` quadratic).
    assert_eq!(
        run(
            r#"/b(c)(d)/.exec("abcde"); [RegExp.lastMatch, RegExp.$1, RegExp.$2,
             RegExp.leftContext, RegExp.rightContext, RegExp.lastParen,
             RegExp.input].join("|")"#
        ),
        "bcd|c|d|a|e|d|abcde"
    );
    // An absent group is the empty string, and `$3` (never present) too.
    assert_eq!(
        run(
            r#"/a(x)?(b)/.exec("ab"); "[" + RegExp.$1 + "][" + RegExp.$2 + "][" + RegExp.$3 + "]""#
        ),
        "[][b][]"
    );
    // `lastParen` is the highest-index participating group.
    assert_eq!(run(r#"/(a)(b)?/.exec("a"); RegExp.lastParen"#), "a");
    // A failed match leaves the previous record intact.
    assert_eq!(
        run(r#"/a/.exec("xay"); /zzz/.exec("xay"); RegExp.lastMatch + RegExp.leftContext"#),
        "ax"
    );
    // `RegExp.input = …` overrides only `input`, not the match record.
    assert_eq!(
        run(r#"/b/.exec("abc"); RegExp.input = "zz"; RegExp.input + "|" + RegExp.lastMatch"#),
        "zz|b"
    );
    // A global replace over a long subject stays linear.
    assert_eq!(
        run(r#"var s = "ab".repeat(20000); s.replace(/a/g, "c").length"#),
        "40000"
    );
    // …and the statics still reflect its final match.
    assert_eq!(
        run(r#""ab".repeat(3).replace(/a(b)/g, "x"); RegExp.$1"#),
        "b"
    );
}

#[test]
fn array_push_pop_fast_path_preserves_semantics() {
    // The no-copy `push`/`pop` path must agree with the precise generic one.
    assert_eq!(
        run("var a=[1,2]; a.push(3,4) + ':' + a.join(',')"),
        "4:1,2,3,4"
    );
    assert_eq!(run("var a=[1,2,3]; a.pop() + ':' + a.join(',')"), "3:1,2");
    assert_eq!(run("var a=[]; a.pop() + ':' + a.length"), "undefined:0");
    // Holes below the write positions do not affect `push`…
    assert_eq!(
        run("var a=[1]; a[3]=4; a.push(5); a.length + ':' + (1 in a)"),
        "5:false"
    );
    // …but a hole at the index `pop` reads must go through the precise path so
    // an inherited getter fires.
    assert_eq!(
        run(
            "Object.defineProperty(Array.prototype,'2',{get(){return 'proto';},configurable:true});
             var a=[0,1]; a.length=3; var r=a.pop();
             delete Array.prototype[2]; r + ':' + a.length"
        ),
        "proto:2"
    );
    // A non-writable `length` still throws on push/pop.
    assert_eq!(
        run(
            "var a=[1,2]; Object.defineProperty(a,'length',{writable:false});
             var n=0; try { a.push(3); } catch (e) { n += e instanceof TypeError; }
             try { a.pop(); } catch (e) { n += e instanceof TypeError; } n"
        ),
        "2"
    );
    // A widened logical length routes to the precise path (appends at `length`).
    assert_eq!(
        run("var a=[1]; a.length=5; a.push(9); a.length + ':' + a[5]"),
        "6:9"
    );
    // Repeated push stays linear — quadratic behaviour here takes seconds.
    assert_eq!(
        run("var a=[]; for (var i=0;i<40000;i++) a.push(i); a.length + ':' + a[39999]"),
        "40000:39999"
    );
}

#[test]
fn collection_hash_index_matches_same_value_zero() {
    // The index hash must agree with SameValueZero everywhere the two differ
    // from `===`: NaN is its own key, and -0/+0 are one key.
    assert_eq!(
        run("var m=new Map(); m.set(NaN,'n'); m.get(NaN) + ':' + m.has(NaN) + ':' + m.size"),
        "n:true:1"
    );
    assert_eq!(
        run(
            "var m=new Map(); m.set(-0,'z'); m.set(0,'p'); m.size + ':' + m.get(-0) + ':' +
             Object.is(m.keys().next().value, 0)"
        ),
        "1:p:true"
    );
    // Strings and BigInts are value types: distinct cells, one key.
    assert_eq!(
        run("var m=new Map(); m.set('a'+'b', 1); m.get('ab') + ':' + m.has(['a','b'].join(''))"),
        "1:true"
    );
    assert_eq!(
        run("var m=new Map(); m.set(10n**20n, 'big'); m.get(BigInt('1' + '0'.repeat(20)))"),
        "big"
    );
    // Objects and symbols are identity keys — equal-looking ones stay distinct.
    assert_eq!(
        run("var m=new Map(); var a={},b={}; m.set(a,1); m.set(b,2);
             m.size + ':' + m.get(a) + ':' + m.get(b)"),
        "2:1:2"
    );
    assert_eq!(
        run("var m=new Map(); m.set(Symbol('x'),1); m.set(Symbol('x'),2); m.size"),
        "2"
    );
    // Primitives that are not numbers still discriminate.
    assert_eq!(
        run(
            "var m=new Map(); m.set(true,1); m.set(false,2); m.set(null,3); m.set(undefined,4);
             [m.get(true),m.get(false),m.get(null),m.get(undefined),m.size].join(',')"
        ),
        "1,2,3,4,4"
    );
    // A number key and its string form are different keys.
    assert_eq!(
        run("var m=new Map(); m.set(1,'n'); m.set('1','s'); m.size + ':' + m.get(1) + m.get('1')"),
        "2:ns"
    );
}

#[test]
fn collection_index_survives_mutation() {
    // Delete renumbers the entries, so the index must be invalidated.
    assert_eq!(
        run("var m=new Map(); for (var i=0;i<50;i++) m.set(i,i);
             m.delete(10); m.delete(0);
             [m.has(10), m.has(0), m.get(11), m.get(49), m.size].join(',')"),
        "false,false,11,49,48"
    );
    // …and re-inserting a deleted key works (it appends at the end).
    assert_eq!(
        run(
            "var m=new Map(); m.set('a',1); m.set('b',2); m.delete('a'); m.set('a',3);
             [...m.keys()].join('') + ':' + m.get('a')"
        ),
        "ba:3"
    );
    // clear() drops everything, and the map is usable afterwards.
    assert_eq!(
        run(
            "var m=new Map(); for (var i=0;i<30;i++) m.set(i,i); m.clear();
             m.size + ':' + m.has(5) + ':' + (m.set(5,'x'), m.get(5))"
        ),
        "0:false:x"
    );
    // Overwriting an existing key updates in place and keeps insertion order.
    assert_eq!(
        run("var m=new Map(); m.set('a',1); m.set('b',2); m.set('a',9);
             m.size + ':' + [...m.values()].join(',')"),
        "2:9,2"
    );
    // Sets behave the same way.
    assert_eq!(
        run("var s=new Set(); for (var i=0;i<40;i++) s.add(i%20);
             s.size + ':' + s.has(19) + ':' + (s.delete(19), s.has(19))"),
        "20:true:false"
    );
    // Live iteration still observes mutations made during the walk.
    assert_eq!(
        run("var s=new Set([1,2]); var out=[]; for (const v of s) { out.push(v); if (v===1) s.add(3); }
             out.join(',')"),
        "1,2,3"
    );
    // A large map stays linear — quadratic behaviour here takes seconds.
    assert_eq!(
        run("var m=new Map(); for (var i=0;i<40000;i++) m.set(i,i);
             m.size + ':' + m.get(39999) + ':' + m.has(-1)"),
        "40000:39999:false"
    );
}
