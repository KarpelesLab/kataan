/*---
description: Math.pow / ** and Math.hypot special-value handling per spec (not IEEE)
esid: sec-numeric-types-number-exponentiate
---*/
// A NaN exponent is always NaN (IEEE pow(1, NaN) is 1; ECMAScript is NaN).
assert.sameValue(Math.pow(1, NaN), NaN, "pow(1, NaN)");
assert.sameValue(1 ** NaN, NaN, "1 ** NaN");
assert.sameValue(Math.pow(NaN, NaN), NaN, "pow(NaN, NaN)");

// |base| == 1 with an infinite exponent is NaN.
assert.sameValue(Math.pow(1, Infinity), NaN, "pow(1, Infinity)");
assert.sameValue(Math.pow(-1, Infinity), NaN, "pow(-1, Infinity)");
assert.sameValue(Math.pow(1, -Infinity), NaN, "pow(1, -Infinity)");
assert.sameValue((-1) ** Infinity, NaN, "(-1) ** Infinity");

// Anything ** ±0 is 1, even NaN ** 0.
assert.sameValue(Math.pow(NaN, 0), 1, "pow(NaN, 0)");
assert.sameValue(NaN ** 0, 1, "NaN ** 0");
assert.sameValue(Math.pow(Infinity, 0), 1, "pow(Infinity, 0)");

// Ordinary cases unchanged.
assert.sameValue(Math.pow(2, 10), 1024, "pow(2, 10)");
assert.sameValue(2 ** 10, 1024, "2 ** 10");
assert.sameValue(Math.pow(0, 0), 1, "pow(0, 0)");
assert.sameValue(Math.pow(0, -1), Infinity, "pow(0, -1)");
assert.sameValue(Math.pow(-8, 1 / 3), NaN, "pow(-8, 1/3)");

// Math.hypot: any infinite argument -> Infinity, even alongside a NaN.
assert.sameValue(Math.hypot(Infinity, NaN), Infinity, "hypot(Infinity, NaN)");
assert.sameValue(Math.hypot(NaN, Infinity), Infinity, "hypot(NaN, Infinity)");
assert.sameValue(Math.hypot(-Infinity, 5), Infinity, "hypot(-Infinity, 5)");
assert.sameValue(Math.hypot(NaN, 3), NaN, "hypot(NaN, 3) without an infinity");
assert.sameValue(Math.hypot(3, 4), 5, "hypot(3, 4)");
assert.sameValue(Math.hypot(), 0, "hypot()");
