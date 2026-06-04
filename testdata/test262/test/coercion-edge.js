/*---
description: More coercion and operator edge cases
esid: sec-abstract-equality-comparison
---*/
assert.sameValue([] + [], "");
assert.sameValue([] + {}, "[object Object]");
assert.sameValue(1 < 2 < 3, true);
assert.sameValue(typeof NaN, "number");
assert.sameValue(10 % 3, 1);
assert.sameValue(-5 % 3, -2);
assert.sameValue(2 ** 3 ** 2, 512);
assert.sameValue("5" - 3, 2);
assert.sameValue(true + true, 2);
assert.sameValue(null + 1, 1);
