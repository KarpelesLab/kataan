/*---
description: Array.isArray distinguishing arrays from array-likes
esid: sec-array.isarray
---*/
assert.sameValue(Array.isArray([]), true);
assert.sameValue(Array.isArray([1, 2, 3]), true);
assert.sameValue(Array.isArray(new Array(5)), true);
assert.sameValue(Array.isArray("string"), false);
assert.sameValue(Array.isArray({ length: 0 }), false, "array-like is not array");
assert.sameValue(Array.isArray({ 0: "a", length: 1 }), false);
assert.sameValue(Array.isArray(null), false);
assert.sameValue(Array.isArray(undefined), false);
assert.sameValue(Array.isArray(42), false);
assert.sameValue(Array.isArray(function () {}), false);
assert.sameValue(typeof Array.isArray, "function");
assert.sameValue(Array.isArray([[], []]), true, "nested arrays");
assert.sameValue(Array.from({ length: 2 }).length, 2);
