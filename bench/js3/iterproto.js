// Iterator-protocol plumbing (original): lazy pipelines of generator
// combinators (map/filter/take/zip/chunk/flatten), hand-written iterator
// objects with return() cleanup, a tree walked by recursive yield*, spread and
// Array.from over iterables, and the built-in Iterator helpers.
const N = typeof IP_N === 'number' ? IP_N : 120000;

function* range(a, b) { for (let i = a; i < b; i++) yield i; }
function* map(it, f) { for (const x of it) yield f(x); }
function* filter(it, p) { for (const x of it) if (p(x)) yield x; }
function* take(it, n) { if (n <= 0) return; for (const x of it) { yield x; if (--n <= 0) return; } }
function* zip(a, b) {
  const ia = a[Symbol.iterator](), ib = b[Symbol.iterator]();
  while (true) { const x = ia.next(), y = ib.next(); if (x.done || y.done) return; yield [x.value, y.value]; }
}
function* chunk(it, n) { let buf = []; for (const x of it) { buf.push(x); if (buf.length === n) { yield buf; buf = []; } } if (buf.length) yield buf; }
function* flatten(it) { for (const x of it) { if (Array.isArray(x)) yield* flatten(x); else yield x; } }

let closed = 0;
class Countdown {
  constructor(n) { this.n = n; }
  [Symbol.iterator]() {
    let n = this.n;
    return { next: () => (n > 0 ? { value: n--, done: false } : { value: undefined, done: true }), return: () => { closed++; return { done: true }; } };
  }
}
class Tree {
  constructor(v, l, r) { this.v = v; this.l = l; this.r = r; }
  *[Symbol.iterator]() { if (this.l) yield* this.l; yield this.v; if (this.r) yield* this.r; }
}
function build(lo, hi) { if (lo > hi) return null; const m = (lo + hi) >> 1; return new Tree(m, build(lo, m - 1), build(m + 1, hi)); }

let s1 = 0;
for (const x of take(filter(map(range(0, N * 4), (x) => x * 7 % 1001), (x) => x & 1), N)) s1 += x;
let s2 = 0;
for (const [a, b] of zip(range(0, N), map(range(0, N), (x) => x * x))) s2 = (s2 + a * 3 + b) % 1000000007;
let s3 = 0, chunks = 0;
for (const c of chunk(range(0, N), 17)) { chunks++; s3 += c[c.length - 1]; }
const nested = [];
for (let i = 0; i < N / 20; i++) nested.push([i, [i + 1, [i + 2]], i + 3]);
let s4 = 0;
for (const x of flatten(nested)) s4 += x;
let s5 = 0;
for (let k = 0; k < N / 100; k++) for (const x of new Countdown(50)) { if (x === 10) break; s5 += x; }
const tree = build(0, (N >> 3) - 1);
let s6 = 0, prev = -1, ordered = true;
for (const v of tree) { if (v <= prev) ordered = false; prev = v; s6 += v; }
const spread = [...take(range(0, N), 5000), ...new Countdown(100)];
const fromSet = Array.from(new Set(map(range(0, 20000), (x) => x % 977)));
const helper = Iterator.from(range(0, N)).filter((x) => x % 3 === 0).map((x) => x / 3).drop(10).take(20000).reduce((a, b) => a + b, 0);
const entries = new Map([...take(range(0, 3000), 3000)].map((i) => [i, i * 2]));
let s7 = 0;
for (const [k, v] of entries) s7 += k ^ v;
for (const v of entries.values()) s7 += v & 3;
console.log(s1, s2, s3, chunks, s4, s5, closed, s6, ordered, spread.length, fromSet.length, helper, s7);
