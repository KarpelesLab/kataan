/*---
description: Number static methods and parsing
esid: sec-number-constructor
---*/
assert.sameValue(Number.isInteger(5), true);
assert.sameValue(Number.isInteger(5.5), false);
assert.sameValue(Number.isInteger("5"), false, "string is not integer");
assert.sameValue(Number.isInteger(Infinity), false);
assert.sameValue(Number.isNaN(NaN), true);
assert.sameValue(Number.isNaN(5), false);
assert.sameValue(Number.isNaN("NaN"), false, "string not coerced");
assert.sameValue(Number.isFinite(5), true);
assert.sameValue(Number.isFinite(Infinity), false);
assert.sameValue(Number.isFinite("5"), false, "no coercion");
assert.sameValue(Number.isSafeInteger(2 ** 53 - 1), true);
assert.sameValue(Number.isSafeInteger(2 ** 53), false);
assert.sameValue(Number.parseInt("42"), 42);
assert.sameValue(Number.parseInt("0xFF", 16), 255);
assert.sameValue(Number.parseFloat("3.14"), 3.14);
assert.sameValue(Number.parseFloat("1.5e3"), 1500);
assert.sameValue(Number("42"), 42);
assert.sameValue(Number("3.14"), 3.14);
assert.sameValue(Number(true), 1);
assert.sameValue(Number(""), 0);
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.EPSILON > 0, true);
