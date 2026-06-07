/*---
description: Array.isArray rejects functions (and accepts only real arrays)
esid: sec-array.isarray
---*/
assert.sameValue(Array.isArray(function () {}), false, "function expression");
assert.sameValue(Array.isArray(() => 1), false, "arrow");
function g() {}
assert.sameValue(Array.isArray(g), false, "function declaration");
var o = { m() {} };
assert.sameValue(Array.isArray(o.m), false, "method");
assert.sameValue(Array.isArray(parseInt), false, "native function");

assert.sameValue(Array.isArray([]), true, "empty array");
assert.sameValue(Array.isArray([1, 2, 3]), true, "non-empty array");
assert.sameValue(Array.isArray(new Array(3)), true, "Array constructor");
assert.sameValue(Array.isArray({}), false, "object");
assert.sameValue(Array.isArray("str"), false, "string");
assert.sameValue(Array.isArray(null), false, "null");
