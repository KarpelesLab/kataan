// Identity-keyed caches (original): memoization keyed by object identity in
// WeakMaps, Symbol-keyed hidden metadata on objects, a WeakRef-backed lookaside
// cache whose targets stay strongly reachable, a WeakSet visited-marker graph
// walk, and well-known symbols (toPrimitive, iterator, hasInstance).
const N = typeof WC_N === 'number' ? WC_N : 300000;

const META = Symbol('meta');
const VERSION = Symbol.for('app.version');
const memo = new WeakMap();
function area(shape) {
  let v = memo.get(shape);
  if (v === undefined) { v = shape.w * shape.h + (shape[META] ? shape[META].bonus : 0); memo.set(shape, v); }
  return v;
}
const shapes = [];
for (let i = 0; i < 5000; i++) {
  const s = { w: i % 17 + 1, h: i % 23 + 1 };
  if (i % 3 === 0) s[META] = { bonus: i % 5 };
  s[VERSION] = i & 7;
  shapes.push(s);
}
let acc = 0;
for (let i = 0; i < N; i++) {
  const s = shapes[(i * 7919) % shapes.length];
  acc = (acc + area(s) + (s[VERSION] | 0)) % 1000003;
}
// WeakRef lookaside: everything is still held strongly in `shapes`.
const refs = new Map();
let derefHits = 0;
for (let i = 0; i < N; i++) {
  const k = i % 1500;
  let r = refs.get(k);
  if (r === undefined) { r = new WeakRef(shapes[k]); refs.set(k, r); }
  const o = r.deref();
  if (o !== undefined && o.w > 0) derefHits++;
}
// Graph walk with a WeakSet of visited nodes.
const nodes = [];
for (let i = 0; i < 4000; i++) nodes.push({ id: i, out: [] });
for (let i = 0; i < 4000; i++) for (let j = 1; j <= 3; j++) nodes[i].out.push(nodes[(i * j * 31 + 7) % 4000]);
let reached = 0;
for (let root = 0; root < 12; root++) {
  const seen = new WeakSet();
  const stack = [nodes[root * 97]];
  while (stack.length) {
    const n = stack.pop();
    if (seen.has(n)) continue;
    seen.add(n); reached++;
    for (const m of n.out) if (!seen.has(m)) stack.push(m);
  }
}
// Symbol-keyed protocol objects.
class Money {
  constructor(c) { this.c = c; }
  [Symbol.toPrimitive](hint) { return hint === 'string' ? '$' + (this.c / 100).toFixed(2) : this.c / 100; }
  static [Symbol.hasInstance](x) { return typeof x === 'object' && x !== null && 'c' in x; }
}
let money = 0, strs = 0, inst = 0;
for (let i = 0; i < N / 4; i++) {
  const m = new Money(i);
  money += +m;
  if (i % 100 === 0) strs += `${m}`.length;
  if ({ c: 1 } instanceof Money) inst++;
}
const symKeys = Object.getOwnPropertySymbols(shapes[3]).map((s) => s.description).join(',');
const reg = new FinalizationRegistry(() => {});
for (let i = 0; i < 1000; i++) reg.register(shapes[i], i);
console.log(acc, derefHits, reached, money.toFixed(2), strs, inst, symKeys, Symbol.keyFor(VERSION), memo.has(shapes[0]));
