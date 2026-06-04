/*---
description: Spread in various positions; array and object merging
esid: sec-array-initializer
---*/
assert.sameValue([0, ...[1, 2], 3, ...[4, 5]].join(","), "0,1,2,3,4,5", "spread in the middle");
function f() { return [...arguments].length; }
assert.sameValue(f(1, 2, 3), 3, "spread arguments");
var base = { a: 1, b: 2 };
var ext = { ...base, c: 3 };
assert.sameValue(ext.a + ext.b + ext.c, 6);
var override = { ...base, b: 20 };
assert.sameValue(override.b, 20, "later property wins");
assert.sameValue(Math.max(...[3, 1, 4, 1, 5, 9, 2, 6]), 9);
