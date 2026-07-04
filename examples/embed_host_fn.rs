//! Embedding demo: register Rust closures as JavaScript functions and call them
//! from a script (`ROADMAP.md` §4.0 — the host-function registration API).
//!
//! Usage: `cargo run --example embed_host_fn`
//!
//! Shows the `register_fn` surface end to end: building values, reading
//! arguments and `this`, throwing a catchable JS error, re-entering JS by
//! calling a script-supplied callback or `construct`ing a class, introspecting
//! values (`type_of`/`is_array`/`array_get`/`own_keys`), and attaching opaque
//! Rust state to a JS object with `set_native_state`/`native_state` (napi_wrap).

use kataan::parser::Parser;
use kataan::{Ctx, Interp, NanBox};

fn main() {
    let mut interp = Interp::new();

    // 1. A pure numeric host function.
    interp.register_global_fn("hypot", 2, |cx: &mut Ctx, _this, args| {
        let a = cx.to_number(args.first().copied().unwrap_or(cx.undefined()))?;
        let b = cx.to_number(args.get(1).copied().unwrap_or(cx.undefined()))?;
        Ok(cx.number((a * a + b * b).sqrt()))
    });

    // 2. A host function that builds an object and throws on bad input.
    interp.register_global_fn("parseHex", 1, |cx: &mut Ctx, _this, args| {
        let s = cx.to_string(args.first().copied().unwrap_or(cx.undefined()))?;
        match i64::from_str_radix(s.trim_start_matches("0x"), 16) {
            Ok(n) => {
                let obj = cx.new_object();
                let dec = cx.number(n as f64);
                cx.set(obj, "decimal", dec);
                let hex = cx.string(&s);
                cx.set(obj, "source", hex);
                Ok(obj)
            }
            Err(_) => Err(cx.range_error(&format!("not hex: {s}"))),
        }
    });

    // 3. A host function that re-enters JS, invoking a callback argument.
    interp.register_global_fn("times", 2, |cx: &mut Ctx, _this, args| {
        let f = args.first().copied().unwrap_or(cx.undefined());
        let n = cx.to_number(args.get(1).copied().unwrap_or(cx.undefined()))? as i64;
        let mut acc: Vec<NanBox> = Vec::new();
        for i in 0..n {
            let idx = cx.number(i as f64);
            acc.push(cx.call(f, cx.undefined(), &[idx])?);
        }
        Ok(cx.new_array(acc))
    });

    // 4. Introspect any value with the inspection / array / property API.
    interp.register_global_fn("describe", 1, |cx: &mut Ctx, _this, args| {
        let v = args.first().copied().unwrap_or(cx.undefined());
        let s = if cx.is_array(v) {
            let n = cx.array_len(v).unwrap_or(0);
            let mut sum = 0.0;
            for i in 0..n {
                let e = cx.array_get(v, i);
                sum += cx.to_number(e).unwrap_or(0.0);
            }
            format!("array of {n}, sum {sum}")
        } else if cx.is_object(v) {
            format!("object with keys [{}]", cx.own_keys(v).join(", "))
        } else {
            format!("{} {}", cx.type_of(v), cx.to_string(v)?)
        };
        Ok(cx.string(&s))
    });

    // 5. Construct a script-supplied class, then read a field back through [[Get]].
    interp.register_global_fn("originOf", 1, |cx: &mut Ctx, _this, args| {
        let ctor = args.first().copied().unwrap_or(cx.undefined());
        if !cx.is_constructor(ctor) {
            return Err(cx.type_error("originOf expects a constructor"));
        }
        let zero = cx.number(0.0);
        let p = cx.construct(ctor, &[zero, zero])?;
        cx.get(p, "x")
    });

    // 6. napi_wrap-style native state: `makeCounter()` returns a JS object with an
    // opaque Rust `u32` attached; `bump(c)` reads and increments it. (The state is
    // dropped as a finalizer when the object is GC'd.)
    interp.register_global_fn("makeCounter", 0, |cx: &mut Ctx, _this, _args| {
        let o = cx.new_object();
        cx.set_native_state(o, 0u32);
        Ok(o)
    });
    interp.register_global_fn("bump", 1, |cx: &mut Ctx, _this, args| {
        let o = args.first().copied().unwrap_or(cx.undefined());
        let next = cx.native_state::<u32>(o).copied().unwrap_or(0) + 1;
        cx.set_native_state(o, next);
        Ok(cx.number(f64::from(next)))
    });

    let src = r#"
        var out = [];
        out.push("hypot(3,4) = " + hypot(3, 4));
        out.push("parseHex('0xff') = " + JSON.stringify(parseHex("0xff")));
        try { parseHex("nope"); } catch (e) { out.push("threw: " + e.message); }
        out.push("times(sq,5) = " + JSON.stringify(times(function (i) { return i * i; }, 5)));
        out.push("describe([1,2,3]) = " + describe([1, 2, 3]));
        out.push("describe({a:1,b:2}) = " + describe({ a: 1, b: 2 }));
        out.push("describe('hi') = " + describe("hi"));
        class Point { constructor(x, y) { this.x = x; this.y = y; } }
        out.push("originOf(Point) = " + originOf(Point));
        var c = makeCounter();
        out.push("counter = " + bump(c) + "," + bump(c) + "," + bump(c));
        out.join("\n");
    "#;

    let program = Parser::parse_program(src).expect("parse");
    let value = interp.run(&program).expect("run");
    println!("{}", interp.realm().to_display_string(value));
}
