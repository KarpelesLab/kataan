/*---
description: Additional Math functions
esid: sec-math-object
---*/
assert.sameValue(Math.hypot(3, 4), 5);
assert.sameValue(Math.cbrt(27), 3);
assert.sameValue(Math.log2(8), 3);
assert.sameValue(Math.log10(1000), 3);
assert.sameValue(Math.sign(-0.5), -1);
assert.sameValue(Math.max(), -Infinity);
assert.sameValue(Math.min(), Infinity);
assert.sameValue(Math.trunc(-4.7), -4);
