/*---
description: Number precision, toFixed, and floating point behavior
esid: sec-number.prototype.tofixed
---*/
assert.sameValue((0.1 + 0.2).toFixed(2), "0.30");
assert.sameValue((0.1 + 0.2 === 0.3), false, "floating point imprecision");
assert.sameValue(Math.abs(0.1 + 0.2 - 0.3) < 1e-10, true, "close enough");
assert.sameValue((1.999999).toFixed(2), "2.00");
assert.sameValue((100).toFixed(2), "100.00");
assert.sameValue((0).toFixed(0), "0");
assert.sameValue((123.456).toFixed(1), "123.5");
assert.sameValue(Number((1.5).toFixed(0)), 2);
assert.sameValue(Number.parseFloat("3.14159").toFixed(2), "3.14");
assert.sameValue((255).toString(16), "ff");
assert.sameValue((10).toString(2), "1010");
