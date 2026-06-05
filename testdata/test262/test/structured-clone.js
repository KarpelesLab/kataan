/*---
description: structuredClone deep-copies objects, arrays, Maps, Sets, Dates, and cycles
esid: sec-structuredclone
---*/
var o = { a: 1, b: { c: 2 }, arr: [1, 2, { d: 3 }] };
var c = structuredClone(o);
assert.sameValue(c.a, 1, "shallow value copied");
assert.sameValue(c.b.c, 2, "nested value copied");
c.b.c = 99;
c.arr[2].d = 99;
assert.sameValue(o.b.c, 2, "deep copy is independent (object)");
assert.sameValue(o.arr[2].d, 3, "deep copy is independent (array)");
assert.sameValue(Array.isArray(c.arr), true, "arrays stay arrays");
var m = structuredClone(new Map([["k", { x: 1 }]]));
assert.sameValue(m.get("k").x, 1, "Map cloned");
m.get("k").x = 9;
var s = structuredClone(new Set([1, 2, 3]));
assert.sameValue([...s].join(","), "1,2,3", "Set cloned");
assert.sameValue(s.has(2), true, "Set membership");
var d = structuredClone(new Date(1000));
assert.sameValue(d.getTime(), 1000, "Date cloned");
// Cycles and shared references are preserved.
var cyc = {};
cyc.self = cyc;
var cc = structuredClone(cyc);
assert.sameValue(cc.self === cc, true, "cycle preserved");
var shared = { v: 1 };
var pair = structuredClone({ x: shared, y: shared });
assert.sameValue(pair.x === pair.y, true, "shared reference stays shared");
// Primitives pass through.
assert.sameValue(structuredClone(42), 42, "number");
assert.sameValue(structuredClone("hi"), "hi", "string");
assert.sameValue(structuredClone(null), null, "null");
// Functions are not cloneable.
var threw = false;
try { structuredClone({ f: function () {} }); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "functions throw");
