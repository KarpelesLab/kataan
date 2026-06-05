/*---
description: Object.entries, values, and fromEntries round trips
esid: sec-object.entries
---*/
var o = { a: 1, b: 2, c: 3 };
assert.sameValue(Object.entries(o).length, 3);
assert.sameValue(Object.entries(o).map(function (e) { return e[0] + "=" + e[1]; }).join(","), "a=1,b=2,c=3");
assert.sameValue(Object.values(o).join(","), "1,2,3");
assert.sameValue(Object.keys(o).join(","), "a,b,c");
var doubled = Object.fromEntries(Object.entries(o).map(function (e) { return [e[0], e[1] * 2]; }));
assert.sameValue(doubled.b, 4);
var total = Object.values(o).reduce(function (a, b) { return a + b; }, 0);
assert.sameValue(total, 6);
var inherited = Object.create({ inherited: 1 });
inherited.own = 2;
assert.sameValue(Object.keys(inherited).join(","), "own", "only own enumerable");
assert.sameValue(Object.entries(inherited).length, 1);
