/*---
description: Number boundary values and constants
esid: sec-properties-of-the-number-constructor
---*/
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.MIN_SAFE_INTEGER, -9007199254740991);
assert.sameValue(Number.MAX_VALUE > 1e308, true);
assert.sameValue(Number.MIN_VALUE > 0, true);
assert.sameValue(Number.MIN_VALUE < 1e-300, true);
assert.sameValue(Number.EPSILON > 0, true);
assert.sameValue(Number.EPSILON < 0.001, true);
assert.sameValue(Number.POSITIVE_INFINITY, Infinity);
assert.sameValue(Number.NEGATIVE_INFINITY, -Infinity);
assert.sameValue(Number.isNaN(Number.NaN), true);
assert.sameValue(1 / Number.MAX_VALUE > 0, true);
assert.sameValue(Number.MAX_SAFE_INTEGER + 1 === Number.MAX_SAFE_INTEGER + 2, true);
assert.sameValue(Number.isSafeInteger(Number.MAX_SAFE_INTEGER), true);
assert.sameValue(Number.isSafeInteger(Number.MAX_SAFE_INTEGER + 1), false);
assert.sameValue(2 ** 53 - 1, Number.MAX_SAFE_INTEGER);
assert.sameValue(Number.isInteger(Number.MAX_SAFE_INTEGER), true);
