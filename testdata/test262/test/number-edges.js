/*---
description: Number boundary values and special arithmetic
esid: sec-numbers
---*/
assert.sameValue(1 / 0, Infinity);
assert.sameValue(-1 / 0, -Infinity);
assert.sameValue(0 / 0 !== 0 / 0, true, "NaN is not equal to itself");
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.isNaN(NaN), true);
assert.sameValue(Number.isNaN(5), false);
assert.sameValue(Math.max(), -Infinity, "Math.max of nothing");
assert.sameValue(Math.min(), Infinity);
assert.sameValue(parseInt("0x1F", 16), 31);
assert.sameValue((0.1 + 0.2 === 0.3), false, "floating point");
assert.sameValue(Math.abs(-5), 5);
assert.sameValue(10 % 3, 1);
assert.sameValue(-10 % 3, -1, "sign follows dividend");
