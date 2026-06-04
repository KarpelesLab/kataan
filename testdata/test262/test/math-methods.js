/*---
description: Math methods
esid: sec-math-object
---*/
assert.sameValue(Math.max(1, 5, 3), 5);
assert.sameValue(Math.min(4, 2, 8), 2);
assert.sameValue(Math.abs(-7), 7);
assert.sameValue(Math.floor(3.9), 3);
assert.sameValue(Math.ceil(3.1), 4);
assert.sameValue(Math.round(2.5), 3);
assert.sameValue(Math.sqrt(16), 4);
assert.sameValue(Math.pow(2, 8), 256);
assert.sameValue(Math.sign(-3), -1);
assert.sameValue(Math.trunc(4.7), 4);
