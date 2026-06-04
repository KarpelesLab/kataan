/*---
description: Math functions and constants
esid: sec-math
---*/
assert.sameValue(Math.floor(4.7), 4);
assert.sameValue(Math.ceil(4.1), 5);
assert.sameValue(Math.round(4.5), 5);
assert.sameValue(Math.trunc(-4.7), -4);
assert.sameValue(Math.sign(-3), -1);
assert.sameValue(Math.sign(3), 1);
assert.sameValue(Math.sign(0), 0);
assert.sameValue(Math.abs(-7), 7);
assert.sameValue(Math.min(3, 1, 2), 1);
assert.sameValue(Math.max(3, 1, 2), 3);
assert.sameValue(Math.pow(2, 10), 1024);
assert.sameValue(Math.sqrt(144), 12);
assert.sameValue(Math.cbrt(27), 3);
assert.sameValue(Math.hypot(3, 4), 5);
assert.sameValue(Math.PI > 3.14 && Math.PI < 3.15, true);
