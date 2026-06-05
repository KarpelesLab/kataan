/*---
description: Math constants and less-common functions
esid: sec-math
---*/
assert.sameValue(Math.PI > 3.14 && Math.PI < 3.15, true);
assert.sameValue(Math.E > 2.71 && Math.E < 2.72, true);
assert.sameValue(Math.SQRT2 > 1.41 && Math.SQRT2 < 1.42, true);
assert.sameValue(Math.floor(Math.LN2 * 1000), 693);
assert.sameValue(Math.log2(8), 3);
assert.sameValue(Math.log10(1000), 3);
assert.sameValue(Math.cbrt(27), 3);
assert.sameValue(Math.hypot(3, 4), 5);
assert.sameValue(Math.sign(-5), -1);
assert.sameValue(Math.trunc(-4.7), -4);
assert.sameValue(Math.max(1, 2, 3), 3);
assert.sameValue(Math.min(3, 2, 1), 1);
assert.sameValue(Math.pow(2, 8), 256);
