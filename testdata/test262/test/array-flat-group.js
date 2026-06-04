/*---
description: Array some/every short-circuit, indexOf NaN, fill with bounds
esid: sec-properties-of-the-array-prototype-object
---*/
var calls = 0;
[1, 2, 3, 4].some(function (x) { calls++; return x === 2; });
assert.sameValue(calls, 2, "some stops at the first truthy");
assert.sameValue([NaN].indexOf(NaN), -1, "indexOf uses strict equality");
assert.sameValue([NaN].includes(NaN), true, "includes uses SameValueZero");
assert.sameValue([0, 0, 0, 0].fill(1, 1, 3).join(","), "0,1,1,0");
