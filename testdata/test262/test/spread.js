/*---
description: Spread in array literals and calls
esid: sec-array-initializer
---*/
var base = [2, 3, 4];
var arr = [1, ...base, 5];
assert.sameValue(arr.length, 5);
assert.sameValue(arr[0], 1);
assert.sameValue(arr[4], 5);
function sum(a, b, c) { return a + b + c; }
assert.sameValue(sum(...base), 9, "spread into a call");
