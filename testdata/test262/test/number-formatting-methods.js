/*---
description: Number toExponential, toPrecision, toFixed
esid: sec-number.prototype.toexponential
---*/
assert.sameValue((1234.5678).toFixed(2), "1234.57");
assert.sameValue((0).toFixed(2), "0.00");
assert.sameValue((1.005).toFixed(2).length, 4);
assert.sameValue((123456).toExponential(2), "1.23e+5");
assert.sameValue((0.00001234).toExponential(3), "1.234e-5");
assert.sameValue((5).toExponential(0), "5e+0");
assert.sameValue((123.456).toPrecision(4), "123.5");
assert.sameValue((0.0001234).toPrecision(2), "0.00012");
assert.sameValue((123).toPrecision(5), "123.00");
assert.sameValue((255).toString(16), "ff");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((3.14159).toFixed(0), "3");
assert.sameValue((2.5).toFixed(0), "3", "rounds half up");
assert.sameValue((1000000).toExponential(), "1e+6");
