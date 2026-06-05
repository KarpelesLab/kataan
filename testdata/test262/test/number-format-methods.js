/*---
description: Number toExponential, toPrecision, toFixed
esid: sec-number.prototype.toexponential
---*/
assert.sameValue((1234.5678).toFixed(2), "1234.57");
assert.sameValue((0).toFixed(2), "0.00");
assert.sameValue((1.005).toFixed(2).length, 4, "toFixed gives a 4-char string");
assert.sameValue((123456).toExponential(2), "1.23e+5");
assert.sameValue((0.000123).toExponential(2), "1.23e-4");
assert.sameValue((123.456).toPrecision(4), "123.5");
assert.sameValue((0.0001234).toPrecision(2), "0.00012");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((3.14159).toFixed(0), "3");
