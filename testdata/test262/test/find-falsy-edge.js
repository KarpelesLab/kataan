/*---
description: find/findIndex with falsy matches, every/some short-circuit
esid: sec-array.prototype.find
---*/
assert.sameValue([0, 1, 2].find(function (x) { return x === 0; }), 0, "finds falsy value 0");
assert.sameValue([0, 1, 2].findIndex(function (x) { return x === 0; }), 0);
assert.sameValue([1, 2, 3].find(function (x) { return x > 10; }), undefined);
assert.sameValue([1, 2, 3].findIndex(function (x) { return x > 10; }), -1);
assert.sameValue([false, 0, ""].find(function (x) { return !x; }), false, "finds first falsy");
var calls = 0;
[1, 2, 3, 4].some(function (x) { calls++; return x === 2; });
assert.sameValue(calls, 2, "some short-circuits");
calls = 0;
[1, 2, 3, 4].every(function (x) { calls++; return x < 3; });
assert.sameValue(calls, 3, "every short-circuits at first false");
