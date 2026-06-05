/*---
description: Number toPrecision, toFixed, toExponential, toString radix
esid: sec-number.prototype.toprecision
---*/
assert.sameValue((123.456).toFixed(2), "123.46");
assert.sameValue((123.456).toPrecision(4), "123.5");
assert.sameValue((0.0001234).toPrecision(2), "0.00012");
assert.sameValue((255).toString(16), "ff");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((8).toString(8), "10");
assert.sameValue((1000000).toString(), "1000000");
assert.sameValue((3.14159).toFixed(0), "3");
assert.sameValue((0.5).toFixed(0), "1", "rounds half up");
assert.sameValue((1.005).toFixed(2).length, 4);
assert.sameValue((100).toExponential(2), "1.00e+2");
assert.sameValue(Number("1e3"), 1000);
