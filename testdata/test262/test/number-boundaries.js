/*---
description: Number boundaries, isInteger, isSafeInteger, isFinite
esid: sec-number-objects
---*/
assert.sameValue(Number.isInteger(5), true);
assert.sameValue(Number.isInteger(5.5), false);
assert.sameValue(Number.isInteger("5"), false, "no coercion");
assert.sameValue(Number.isSafeInteger(Math.pow(2, 53)), false);
assert.sameValue(Number.isSafeInteger(Math.pow(2, 53) - 1), true);
assert.sameValue(Number.isFinite(Infinity), false);
assert.sameValue(Number.isFinite(42), true);
assert.sameValue(Number.isFinite("42"), false, "no coercion");
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.MIN_SAFE_INTEGER, -9007199254740991);
assert.sameValue(isNaN(NaN), true);
assert.sameValue(isFinite("100"), true, "global isFinite coerces");
