//! Programs whose hot loops the compiled tier runs with integer values or
//! carried fields in registers (`jit::numeric`, `jit::numeric::int32`):
//! each result is pinned to the spec's (checked against another engine),
//! so a fast path that keeps a value in the wrong representation, or
//! forwards a field it should have read, shows here whether or not the
//! loop compiles.
use super::*;
use crate::parser::Parser;

/// Runs `src` and renders the program's final value.
fn run(src: &str) -> String {
    let program = Parser::parse_program(src).expect("parse");
    let mut interp = Interp::new();
    let value = interp.run(&program).expect("exec");
    interp.realm().to_display_string(value)
}

/// A carried field whose receiver register is a move of a loop-invariant
/// one (an inlined accessor's `this`, `const q = o` in the loop): aliases,
/// a field that stops being a Number, values that are no int32, a moved
/// receiver whose source binding the loop reassigns, and an accessor
/// replaced partway.
#[test]
fn carried_fields_through_moved_receivers() {
    let src = r#"
        const r = [];
        class P { constructor() { this._x = 0; } get x() { return this._x; } set x(v) { this._x = v; } }
        const p = new P();
        for (let i = 0; i < 30000; i++) { p.x = p.x + 1; }
        r.push(p.x);
        const o = { n: 0, m: 0 };
        function f1(n) { const a = o, b = o; for (let i = 0; i < n; i++) { a.n = b.n + 2; b.m = a.n - 1; } return a.n + ':' + b.m; }
        r.push(f1(20000));
        const z = { v: 0 };
        function f2(n) { const q = z; for (let i = 0; i < n; i++) { q.v = q.v + 1; if (i === 9000) z.v = 'k'; if (i === 15000) z.v = { valueOf() { return 7; } }; } return String(q.v).length + ' ' + typeof q.v; }
        r.push(f2(20000));
        const w = { v: 2147483600 };
        function f3(n) { const q = w; for (let i = 0; i < n; i++) { q.v = q.v + 1; } return q.v; }
        r.push(f3(100));
        w.v = 0.25; r.push(f3(30000));
        w.v = NaN; r.push(f3(30000));
        let g = { v: 3 };
        function f5(n) { for (let i = 0; i < n; i++) { const h = g; h.v = h.v + 3; } return g.v; }
        r.push(f5(20000)); g = { v: -5 }; r.push(f5(20000));
        let cnt = 0;
        class S { constructor() { this._x = 0; } get x() { if (++cnt === 15000) Object.defineProperty(S.prototype, 'x', { get() { return -1; }, set(v) {}, configurable: true }); return this._x; } set x(v) { this._x = v; } }
        const s = new S(); for (let i = 0; i < 30000; i++) { s.x = s.x + 1; } r.push(s.x, s._x, cnt);
        let gg = { v: 0 }; const g2 = { v: 1000 };
        function f6(n) { for (let i = 0; i < n; i++) { const h = gg; h.v = h.v + 1; if (i === 7000) gg = g2; } return gg.v + ' ' + g2.v; }
        r.push(f6(20000));
        r.join(' ')
    "#;
    assert_eq!(
        run(src),
        "30000 40000:39999 4 number 2147483700 30000.25 NaN 60003 59995 -1 14999 15000 13999 13999"
    );
}
