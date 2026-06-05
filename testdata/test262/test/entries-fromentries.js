/*---
description: Object.entries and fromEntries with various values
esid: sec-object.entries
---*/
var o = { a: 1, b: "two", c: true, d: null };
var entries = Object.entries(o);
assert.sameValue(entries.length, 4);
assert.sameValue(entries[0].join("="), "a=1");
assert.sameValue(entries[1].join("="), "b=two");
var reconstructed = Object.fromEntries(entries);
assert.sameValue(reconstructed.a, 1);
assert.sameValue(reconstructed.c, true);
assert.sameValue(reconstructed.d, null);
var doubled = Object.fromEntries(Object.entries({ x: 1, y: 2 }).map(function (e) { return [e[0], e[1] * 2]; }));
assert.sameValue(doubled.x, 2);
assert.sameValue(doubled.y, 4);
var filtered = Object.fromEntries(Object.entries({ a: 1, b: 2, c: 3 }).filter(function (e) { return e[1] > 1; }));
assert.sameValue(Object.keys(filtered).join(","), "b,c");
var swapped = Object.fromEntries(Object.entries({ a: "x", b: "y" }).map(function (e) { return [e[1], e[0]]; }));
assert.sameValue(swapped.x, "a");
assert.sameValue(swapped.y, "b");
var fromMap = Object.fromEntries(new Map([["k1", 1], ["k2", 2]]));
assert.sameValue(fromMap.k1, 1);
assert.sameValue(Object.values({ a: 1, b: 2, c: 3 }).reduce(function (a, b) { return a + b; }, 0), 6);
