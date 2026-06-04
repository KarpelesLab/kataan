/*---
description: Number toFixed, toPrecision, and parseFloat
esid: sec-number.prototype.tofixed
---*/
assert.sameValue((3.14159).toFixed(2), "3.14");
assert.sameValue((1).toFixed(3), "1.000");
assert.sameValue((1234.5678).toFixed(0), "1235");
assert.sameValue((0.1 + 0.2).toFixed(1), "0.3");
assert.sameValue(parseFloat("3.14abc"), 3.14);
assert.sameValue(parseFloat(".5"), 0.5);
assert.sameValue((255).toString(16), "ff");
assert.sameValue(Number("  42  "), 42);
