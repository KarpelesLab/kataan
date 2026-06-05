/*---
description: Number precision and floating point edge cases
esid: sec-numbers
---*/
assert.sameValue(0.1 + 0.2 === 0.3, false, "floating point imprecision");
assert.sameValue(Math.abs(0.1 + 0.2 - 0.3) < Number.EPSILON, true, "within epsilon");
assert.sameValue((0.1 + 0.2).toFixed(1), "0.3");
assert.sameValue(0.1 * 3 !== 0.3, true);
assert.sameValue(1 / 3 * 3, 1, "division round trip");
assert.sameValue((1).toFixed(20).length, 22, "many decimals");
assert.sameValue(9999999999999999 === 10000000000000000, true, "precision loss at scale");
assert.sameValue(0.5 + 0.5, 1);
assert.sameValue(2 ** 53 === 2 ** 53 + 1, true, "beyond safe integer");
assert.sameValue((123.456).toPrecision(4), "123.5");
assert.sameValue((0.000123).toExponential(2), "1.23e-4");
assert.sameValue(1.5 % 1, 0.5);
assert.sameValue(-1.5 % 1, -0.5);
assert.sameValue(5 % 0 !== 5 % 0, true, "modulo by zero is NaN");
