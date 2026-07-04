//! Embedding demo: register Rust closures as JavaScript functions and call them
//! from a script (`ROADMAP.md` §4.0 — the host-function registration API).
//!
//! Usage: `cargo run --example embed_host_fn`
//!
//! Shows the whole `register_fn` surface end to end: building values, reading
//! arguments and `this`, throwing a catchable JS error, and re-entering JS by
//! calling a script-supplied callback.

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

    let src = r#"
        var out = [];
        out.push("hypot(3,4) = " + hypot(3, 4));
        out.push("parseHex('0xff') = " + JSON.stringify(parseHex("0xff")));
        try { parseHex("nope"); } catch (e) { out.push("threw: " + e.message); }
        out.push("times(sq,5) = " + JSON.stringify(times(function (i) { return i * i; }, 5)));
        out.join("\n");
    "#;

    let program = Parser::parse_program(src).expect("parse");
    let value = interp.run(&program).expect("run");
    println!("{}", interp.realm().to_display_string(value));
}
