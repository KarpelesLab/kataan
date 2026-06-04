/*---
description: Number toExponential, toString radix, and conversions
esid: sec-number.prototype.toexponential
---*/
assert.sameValue((1234.5).toExponential(2), "1.23e+3");
assert.sameValue((0).toString(), "0");
assert.sameValue((-255).toString(16), "-ff");
assert.sameValue(Number("0x1F"), 31);
assert.sameValue(Number("  3.14  "), 3.14);
assert.sameValue(Number(""), 0);
