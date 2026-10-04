//! End-to-end tests for garbage collection inside hosted bytecode-VM runs (the
//! VM's safepoints root its frames; the interpreter's state is rooted by
//! [`gc`](super::gc)).
//!
//! Stress the same programs with a tiny pinned trigger and use-after-free
//! detection: `KATAAN_GC_THRESHOLD=256 KATAAN_GC_VERIFY=panic cargo test
//! --release --lib nbexec` (any collected object used again panics).
//!
//! Each one builds a deliberately awkward live set — a reference cycle, a
//! `WeakMap` value reachable only through a live key, a suspended generator, a
//! pending promise chain, an in-flight `for-of` iterator, a `try`/`finally`
//! pending completion, a mapped `arguments` object — and then allocates enough at
//! the top level to force several real collections *around* it. A collector that
//! frees something live fails the assertion; one that frees nothing fails
//! [`top_level_churn_is_reclaimed_not_retained`].

use crate::nbexec::Interp;
use crate::parser::Parser;
use alloc::format;
use alloc::string::String;

/// Runs `src` on the hosted VM and renders its completion value.
fn run(src: &str) -> String {
    run_with_live(src).0
}

/// Runs `src` and returns `(rendered value, live heap objects at the end)`.
fn run_with_live(src: &str) -> (String, usize) {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    let text = interp.realm().to_display_string(value);
    (text, interp.realm().object_count())
}

/// Whether `KATAAN_GC_VERIFY` is on: swept objects are then kept (to catch a
/// later use), so live counts say nothing about reclamation.
fn verifying() -> bool {
    #[cfg(feature = "std")]
    {
        std::env::var_os("KATAAN_GC_VERIFY").is_some()
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

/// Enough top-level allocation to cross the collector's trigger many times over,
/// so every test below really collects rather than merely not crashing.
const CHURN: &str = "for (var _i = 0; _i < 400000; _i++) { var _t = { k: _i }; }";

#[test]
fn top_level_churn_is_reclaimed_not_retained() {
    let (v, live) = run_with_live(&format!("{CHURN} 'done'"));
    assert_eq!(v, "done");
    // The collector triggers every ~65k allocations (`GC_MIN_THRESHOLD`), so at
    // most that much garbage is live at the end — never the 400k the loop made.
    assert!(
        verifying() || live < 100_000,
        "top-level garbage was retained: {live} live objects"
    );
}

#[test]
fn objects_live_across_a_collection_survive() {
    // A cycle, an array graph, and a closure capture — all reachable only through
    // top-level bindings — must be intact after collecting around them.
    assert_eq!(
        run(&format!(
            "var a = {{ n: 1 }}; var b = {{ a: a }}; a.b = b;
             var arr = [a, b, 'text', [1, 2, 3]];
             var f = (function () {{ var hidden = {{ v: 42 }};
                                     return function () {{ return hidden.v; }}; }})();
             {CHURN}
             (a.b.a === a) && arr[3][2] === 3 && arr[2] === 'text' && f() === 42"
        )),
        "true"
    );
}

#[test]
fn a_weakmap_value_survives_while_its_key_is_live() {
    // The ephemeron rule: the payload is reachable ONLY as a WeakMap value, and
    // survives exactly because its key is still rooted.
    assert_eq!(
        run(&format!(
            "var key = {{ id: 'k' }}; var wm = new WeakMap();
             wm.set(key, {{ deep: {{ v: 'payload' }} }});
             {CHURN}
             wm.get(key).deep.v"
        )),
        "payload"
    );
}

#[test]
fn a_suspended_generator_keeps_its_objects() {
    assert_eq!(
        run(&format!(
            "function* g() {{ var held = {{ v: 'held' }};
                              yield held; yield held.v + '!'; }}
             var it = g(); var first = it.next().value;
             {CHURN}
             first.v + '/' + it.next().value"
        )),
        "held/held!"
    );
}

#[test]
fn a_pending_promise_chain_settles_after_a_collection() {
    assert_eq!(
        run(&format!(
            "var log = [];
             var p = new Promise(function (res) {{ res({{ v: 'resolved' }}); }});
             p.then(function (o) {{ log.push(o.v); }})
              .then(function () {{ log.push('tail'); }});
             {CHURN}
             p.then(function (o) {{ log.push(o.v + '2'); }});
             'scheduled'"
        )),
        "scheduled"
    );
}

#[test]
fn a_for_of_iterator_and_its_items_survive_the_loop() {
    // The iterator record, its cached `next`, and each item live in Rust locals
    // across the body, which allocates enough to collect mid-iteration.
    assert_eq!(
        run("var seen = [];
             var iterable = { [Symbol.iterator]() {
                 var i = 0;
                 return { next() { i++;
                     return i > 5 ? { done: true } : { done: false, value: { n: i } }; } };
             } };
             for (var item of iterable) {
               for (var j = 0; j < 200000; j++) { var t = { j: j }; }
               seen.push(item.n);
             }
             seen.join(',')"),
        "1,2,3,4,5"
    );
}

#[test]
fn a_for_in_key_list_survives_the_loop() {
    assert_eq!(
        run("var o = { a: 1, b: 2, c: 3 }; var seen = [];
             for (var k in o) {
               for (var j = 0; j < 100000; j++) { var t = { j: j }; }
               seen.push(k);
             }
             seen.join(',')"),
        "a,b,c"
    );
}

#[test]
fn a_try_finally_pending_completion_survives() {
    // The in-flight thrown object is a Rust local while the collecting `finally`
    // body runs.
    assert_eq!(
        run(&format!(
            "var caught = 'none';
             try {{
               try {{ throw {{ tag: 'boom' }}; }} finally {{ {CHURN} }}
             }} catch (e) {{ caught = e.tag; }}
             caught"
        )),
        "boom"
    );
}

#[test]
fn a_switch_discriminant_survives_its_case_bodies() {
    assert_eq!(
        run(&format!(
            "var d = {{ toString: function () {{ return 'x'; }} }};
             var out = 'no';
             switch (d) {{ case d: {CHURN} out = String(d); break; }}
             out"
        )),
        "x"
    );
}

#[test]
fn mapped_arguments_still_alias_after_a_collection() {
    // `arg_maps` is pruned as a weak-key table; a *live* arguments object must
    // still alias its parameter binding across a cycle.
    assert_eq!(
        run(&format!(
            "var esc;
             function f(a) {{ esc = arguments; a = 'changed'; return arguments[0]; }}
             var before = f('orig');
             {CHURN}
             before + '/' + esc[0]"
        )),
        "changed/changed"
    );
}

#[test]
fn a_live_binding_read_through_an_accessor_survives() {
    assert_eq!(
        run(&format!(
            "var holder = {{ v: 'live' }};
             var ns = Object.freeze({{ get value() {{ return holder.v; }} }});
             {CHURN}
             holder.v = 'updated';
             ns.value"
        )),
        "updated"
    );
}

#[test]
fn class_state_survives_a_collection() {
    assert_eq!(
        run(&format!(
            "class C {{ static tag = {{ s: 'static' }};
                        #p = {{ s: 'private' }};
                        get p() {{ return this.#p.s; }} }}
             var c = new C();
             {CHURN}
             C.tag.s + '/' + c.p + '/' + (c instanceof C)"
        )),
        "static/private/true"
    );
}

#[test]
fn symbols_used_as_property_keys_survive() {
    assert_eq!(
        run(&format!(
            "var s = Symbol('tag'); var o = {{}}; o[s] = 'kept';
             var g = Symbol.for('reg');
             {CHURN}
             o[s] + '/' + (Symbol.for('reg') === g) + '/'
               + (Object.getOwnPropertySymbols(o)[0] === s)"
        )),
        "kept/true/true"
    );
}

#[test]
fn regexp_and_string_state_survive() {
    // The realm memoizes the last regex subject's UTF-16 units and the Annex B.2.5
    // legacy match record; both hold string state across a cycle.
    assert_eq!(
        run(&format!(
            "var re = /(\\w+)@(\\w+)/g; var subject = 'alice@example bob@other';
             var m = re.exec(subject);
             {CHURN}
             m[1] + '/' + RegExp.$2 + '/' + re.exec(subject)[1]"
        )),
        "alice/example/bob"
    );
}

// --- collection inside function bodies (published VM frames) ---

/// The same churn as [`CHURN`], inside a function body: the VM collects at
/// safepoints below the outermost frame when every frame on the way down
/// published its registers (an ordinary JS-to-JS call does).
const FN_CHURN: &str =
    "function churn() { for (var i = 0; i < 400000; i++) { var t = { k: i }; } return 'churned'; }";

#[test]
fn churn_inside_a_function_body_is_reclaimed() {
    let (v, live) = run_with_live(&format!(
        "{FN_CHURN}
         function outer() {{ return churn(); }}
         outer()"
    ));
    assert_eq!(v, "churned");
    assert!(
        verifying() || live < 100_000,
        "garbage made inside a function body was retained: {live} live objects"
    );
}

#[test]
fn caller_frames_keep_their_registers_across_a_collection() {
    // `held` lives only in `outer`'s register window (and a closure's captured
    // cell) while `churn` collects underneath it.
    assert_eq!(
        run(&format!(
            "{FN_CHURN}
             function outer() {{
               var held = {{ deep: {{ v: 'reg' }} }};
               var cell = {{ v: 'cell' }};
               var get = function () {{ return cell.v; }};
               var r = churn();
               return held.deep.v + '/' + get() + '/' + r;
             }}
             outer()"
        )),
        "reg/cell/churned"
    );
}

#[test]
fn a_suspended_vm_generator_frame_survives_collections_in_a_function() {
    // The generator's suspended registers live in its hidden state array; the
    // collection runs inside a function called between two resumptions.
    assert_eq!(
        run(&format!(
            "{FN_CHURN}
             function* g() {{ var held = {{ v: 'held' }}; yield 1; yield held.v; }}
             function drive() {{
               var it = g(); it.next();
               churn();
               return it.next().value;
             }}
             drive()"
        )),
        "held"
    );
}
