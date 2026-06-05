/*---
description: Math trig, log, exp, and rounding functions
esid: sec-math
---*/
assert.sameValue(Math.floor(4.7), 4);
assert.sameValue(Math.ceil(4.1), 5);
assert.sameValue(Math.round(4.5), 5);
assert.sameValue(Math.round(4.4), 4);
assert.sameValue(Math.trunc(4.9), 4);
assert.sameValue(Math.trunc(-4.9), -4);
assert.sameValue(Math.sqrt(144), 12);
assert.sameValue(Math.cbrt(27), 3);
assert.sameValue(Math.pow(2, 10), 1024);
assert.sameValue(Math.abs(-7), 7);
assert.sameValue(Math.sign(-3), -1);
assert.sameValue(Math.sign(3), 1);
assert.sameValue(Math.min(1, 2, 3), 1);
assert.sameValue(Math.max(1, 2, 3), 3);
assert.sameValue(Math.hypot(3, 4), 5);
assert.sameValue(Math.log2(16), 4);
assert.sameValue(Math.log10(1000), 3);
assert.sameValue(Math.round(Math.exp(0)), 1);
assert.sameValue(Math.round(Math.log(Math.E)), 1);
assert.sameValue(Math.floor(Math.PI), 3);
assert.sameValue(Math.max(...[5, 2, 9, 1]), 9);
