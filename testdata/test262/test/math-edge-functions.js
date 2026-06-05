/*---
description: Math functions with edge cases
esid: sec-math
---*/
assert.sameValue(Math.max(), -Infinity);
assert.sameValue(Math.min(), Infinity);
assert.sameValue(Math.max(1, 2, 3, 4, 5), 5);
assert.sameValue(Math.abs(-0), 0);
assert.sameValue(Math.sign(-5), -1);
assert.sameValue(Math.sign(0), 0);
assert.sameValue(Math.trunc(4.9), 4);
assert.sameValue(Math.trunc(-4.9), -4);
assert.sameValue(Math.floor(-0.5), -1);
assert.sameValue(Math.ceil(-0.5) === 0, true, "ceil(-0.5) is zero");
assert.sameValue(Math.round(2.5), 3);
assert.sameValue(Math.round(-2.5), -2, "round half toward +Infinity");
assert.sameValue(Math.hypot(3, 4), 5);
assert.sameValue(Math.cbrt(-8), -2);
assert.sameValue(Math.max(NaN, 1) !== Math.max(NaN, 1), true, "NaN propagates");
