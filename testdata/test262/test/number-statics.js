/*---
description: Number static properties and methods
esid: sec-properties-of-the-number-constructor
---*/
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.MIN_SAFE_INTEGER, -9007199254740991);
assert.sameValue(Number.EPSILON > 0, true);
assert.sameValue(Number.POSITIVE_INFINITY, Infinity);
assert.sameValue(Number.NEGATIVE_INFINITY, -Infinity);
assert.sameValue(Number.isNaN(Number.NaN), true);
assert.sameValue(Number.isInteger(5), true);
assert.sameValue(Number.isInteger(5.5), false);
assert.sameValue(Number.isSafeInteger(2 ** 53 - 1), true);
assert.sameValue(Number.isSafeInteger(2 ** 53), false);
assert.sameValue(Number.parseFloat("3.14"), 3.14);
assert.sameValue(Number.parseInt("42px"), 42);
assert.sameValue(Number.isFinite(Infinity), false);
assert.sameValue(Number.MAX_VALUE > 0, true);
