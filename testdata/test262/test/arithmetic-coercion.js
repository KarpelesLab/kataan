/*---
description: Arithmetic and bitwise operators coerce object operands via ToPrimitive
esid: sec-tonumber
---*/
assert.sameValue([5] - 2, 3, "array minus number");
assert.sameValue([5] * 2, 10);
assert.sameValue([10] / 2, 5);
assert.sameValue([10] % 3, 1);
assert.sameValue([2] ** 3, 8);
assert.sameValue("5" - 2, 3, "string minus number");
assert.sameValue("3" * "4", 12);
assert.sameValue([6] & 3, 2, "bitwise and");
assert.sameValue([4] | 1, 5);
assert.sameValue([5] ^ 1, 4);
assert.sameValue([1] << 3, 8);
assert.sameValue([16] >> 2, 4);
assert.sameValue([2] ** [3], 8, "both arrays");
assert.sameValue(({ valueOf: function () { return 8; } }) - 3, 5, "valueOf in subtraction");
assert.sameValue(true - false, 1, "booleans");
assert.sameValue(null * 5, 0, "null is 0");
assert.sameValue(Number.isNaN({} - 1), true, "plain object is NaN");
assert.sameValue(Number.isNaN([1, 2] - 1), true, "multi-element array is NaN");
assert.sameValue(new Date(5000) - new Date(2000), 3000, "date difference");
assert.sameValue(-[5], -5, "unary negation of array");
assert.sameValue(+[42], 42, "unary plus of single-element array");
