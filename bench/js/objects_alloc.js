// Allocation-heavy: object/array literals, short-lived garbage.
let keep = null, s = 0;
for (let i = 0; i < 2000000; i++) {
  const o = { a: i, b: [i, i + 1], c: { d: i & 3 } };
  s += o.b[1] - o.a + o.c.d;
  if ((i & 1023) === 0) keep = o;
}
console.log(s, keep.a);
