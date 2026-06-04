/*---
description: Object.fromEntries and entries round-trip
esid: sec-object.fromentries
---*/
var o = Object.fromEntries([["a", 1], ["b", 2]]);
assert.sameValue(o.a, 1);
assert.sameValue(o.b, 2);
var pairs = Object.entries({ x: 10, y: 20 });
assert.sameValue(pairs.length, 2);
assert.sameValue(pairs[0][0], "x");
assert.sameValue(pairs[0][1], 10);
