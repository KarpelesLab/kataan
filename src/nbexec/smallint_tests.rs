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

/// Named property reads a loop's integer version holds as int32s
/// (`int32::plan`'s `GetProp`, checked after the read): values that are no
/// int32 (a fraction, `-0`, NaN, `2^31 - 1` past a `* 3`, a string, an
/// object with `valueOf`), a field turned into a getter partway, a read the
/// fast version copies from an earlier read of the same key, an index read
/// from a field, and a second layout.
#[test]
fn int_property_reads() {
    let src = r#"
        const r = [];
        function sum(a) { let s = 0; for (let i = 0; i < a.length; i++) { const o = a[i]; if (o.v > 3) s = (s + o.v) | 0; } return s; }
        const a = []; for (let i = 0; i < 400; i++) a.push({ v: i & 15 });
        let t = 0; for (let k = 0; k < 200; k++) t = (t + sum(a)) | 0;
        r.push(t);
        a[17].v = 2.5; a[18].v = -0; a[19].v = NaN; a[20].v = 2147483647; a[21].v = '9'; a[22].v = { valueOf() { return 11; } };
        r.push(sum(a));
        function acc(a) { let s = 0; for (let i = 0; i < a.length; i++) { s = s + a[i].w * 3; } return s; }
        const b = []; for (let i = 0; i < 300; i++) b.push({ w: 1000000 + i });
        let u = 0; for (let k = 0; k < 100; k++) u = acc(b);
        r.push(u);
        b[5].w = 2147483647; r.push(acc(b)); b[6].w = -0.5; r.push(acc(b));
        function neg(a) { let z = 1; for (let i = 0; i < a.length; i++) { z = a[i].n * 2; } return Object.is(z, -0) ? 'm0' : z; }
        const c = []; for (let i = 0; i < 300; i++) c.push({ n: i }); for (let k = 0; k < 50; k++) neg(c);
        c[299].n = -0; r.push(neg(c));
        function gather(objs, tab) { let s = 0; for (let i = 0; i < objs.length; i++) { s = (s + tab[objs[i].k & 7]) | 0; } return s; }
        const objs = []; for (let i = 0; i < 300; i++) objs.push({ k: i * 7 }); const tab = [1, 2, 3, 4, 5, 6, 7, 8];
        let g = 0; for (let k = 0; k < 100; k++) g = gather(objs, tab); r.push(g);
        objs[3].k = 1.5; r.push(gather(objs, tab)); objs[4].k = '3'; r.push(gather(objs, tab));
        let calls = 0;
        function ga(a) { let s = 0; for (let i = 0; i < a.length; i++) { s = (s + a[i].v) | 0; if (i === 150) Object.defineProperty(a[200], 'v', { get() { calls++; return 1e9; } }); } return s; }
        const d = []; for (let i = 0; i < 300; i++) d.push({ v: i });
        for (let k = 0; k < 60; k++) { d[200] = { v: 200 }; ga(d); }
        r.push(ga(d), calls);
        function bits(a) { let s = 0; for (let i = 0; i < a.length; i++) { const o = a[i]; if (o.m > 0) s ^= (o.m << 1) + i; } return s; }
        const e = []; for (let i = 0; i < 300; i++) e.push({ m: i * 13 }); let q = 0; for (let k = 0; k < 100; k++) q = bits(e); r.push(q);
        e[7].m = 3.75; e[8].m = -1; e[9].m = 4294967296; r.push(bits(e));
        function poly(a) { let s = 0; for (let i = 0; i < a.length; i++) s = (s + a[i].v * 2) | 0; return s; }
        const f = []; for (let i = 0; i < 300; i++) f.push({ v: i }); for (let k = 0; k < 80; k++) poly(f);
        f[10] = { z: 1, v: 5 }; f[11] = { v: 6, y: 2 }; r.push(poly(f));
        r.join(' ')
    "#;
    assert_eq!(
        run(src),
        "570000 3853 900134550 7339585476 7336585456.5 m0 1354 1350 1349 1000044650 61 420 310 89680"
    );
}

/// Reads the fast version replaces by a copy of an int register
/// (`plan_reuse` in the integer version): a field stored from an int sum and
/// read back after the sum's last use (where no GPR holds it any more), a
/// key read twice around a store of another, and values that stop being
/// int32s.
#[test]
fn int_property_read_copies() {
    let src = r#"
        const r = [];
        function strs(x, y) { const o = { a: x, c: 's' }; let s = 0; for (let i = 0; i < 4; i = i + 1) { s = s + o.c; o.c = o.a + i; s = s + o.c * y; } return s.length; }
        function nums(x, y, n) { const o = { a: x, c: 0 }; let s = 0; for (let i = 0; i < n; i = i + 1) { o.c = o.a + i; s = s + o.c * y; } return s; }
        function twice(x, n) { const o = { a: x, b: 1 }; let s = 0; for (let i = 0; i < n; i++) { const p = o.a + 1; o.b = 2; const q = o.a + 2; s = (s + p + q) | 0; } return s; }
        function back(x, n) { const o = { a: x, c: 0 }; let s = 0; for (let i = 0; i < n; i++) { const t = o.a + i; o.c = t; const u = o.c; s = (s + (u & 255)) | 0; } return s; }
        let acc = 0;
        for (let k = 0; k < 3000; k++) { acc += strs(1, 2) + nums(k, 3, 20) + twice(k, 10) + back(k, 10); }
        r.push(acc);
        r.push(strs(2147483647, 0.5), nums(2147483647, 2, 30), nums(0.5, -0, 30), nums('7', 1, 5));
        r.push(twice(2147483646, 30), twice(-0, 3), twice(1.5, 3), back(2147483640, 30), back(NaN, 3));
        r.join(' ')
    "#;
    assert_eq!(run(src), "365474040 76 128849019690 0 360 -30 9 18 2243 0");
}

/// Carried fields in loops with branches of their own
/// (`plan_carried_fields`): a store on some iterations only, a value
/// computed on every path but stored on one, two stores of the key on
/// different paths, a store through a register that is sometimes the same
/// object, a field that stops being a Number, an exit after the store, and
/// an accessor installed partway.
#[test]
fn carried_fields_in_branchy_loops() {
    let src = r#"
        const r = [];
        const o = { n: 0, m: 0, k: 0 };
        function step(o, k) { o.n = o.n + 1; if (k & 1) o.m++; if (k % 7 === 3) o.k = o.k + o.n; }
        for (let k = 0; k < 40000; k++) step(o, k);
        r.push(o.n, o.m, o.k);
        const p = { v: 1 };
        function f1(n) { const q = p; for (let i = 0; i < n; i++) { const t = q.v + 1; if (i % 3 === 0) q.v = t; } return q.v; }
        r.push(f1(30000));
        const s = { v: 0 };
        function f2(n) { const q = s; for (let i = 0; i < n; i++) { if (i & 1) q.v = q.v + 1; else q.v = q.v + 2; } return q.v; }
        r.push(f2(30000));
        const a1 = { v: 0 }, a2 = { v: 0 };
        function f3(n) { const q = a1; for (let i = 0; i < n; i++) { const w = (i % 5 === 0) ? a1 : a2; q.v = q.v + 1; w.v = w.v + 10; } return a1.v + ' ' + a2.v; }
        r.push(f3(30000));
        const b = { v: 0 };
        function f4(n) { const q = b; for (let i = 0; i < n; i++) { if (i === 12345) b.v = 'x'; if (i === 12350) b.v = 3.5; q.v = q.v + 1; if (i % 1000 === 999 && q.v > 1e9) break; } return q.v; }
        r.push(f4(30000));
        const c = { v: 0 };
        function f5(n) { const q = c; for (let i = 0; i < n; i++) { q.v = q.v + 3; if (q.v > 50000) return 'exit ' + i + ' ' + q.v; if (i & 2) continue; q.v = q.v; } return q.v; }
        r.push(f5(30000)); c.v = -0; r.push(f5(10));
        const e = { v: 0 }; let gets = 0;
        function f7(n) { const q = e; for (let i = 0; i < n; i++) { if (i === 20000) Object.defineProperty(e, 'v', { get() { gets++; return 7; }, set(x) {} }); if (i & 1) q.v = q.v + 1; } return q.v + ' ' + gets; }
        r.push(f7(30000));
        r.join(' ')
    "#;
    assert_eq!(
        run(src),
        "40000 20000 114277143 10001 45000 90000 240000 17653.5 exit 16666 50001 30 7 5001"
    );
}
