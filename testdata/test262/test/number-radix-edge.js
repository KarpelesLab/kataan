/*---
description: Number.prototype.toString with radix and parseInt
esid: sec-number.prototype.tostring
---*/
assert.sameValue((255).toString(16), "ff");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((100).toString(8), "144");
assert.sameValue((1295).toString(36), "zz");
assert.sameValue((0).toString(2), "0");
assert.sameValue((-255).toString(16), "-ff");
assert.sameValue((10).toString(), "10", "default radix 10");
assert.sameValue(parseInt("ff", 16), 255);
assert.sameValue(parseInt("zz", 36), 1295);
assert.sameValue(parseInt("777", 8), 511);
assert.sameValue(parseInt("101", 2), 5);
assert.sameValue(Number.isNaN(parseInt("g", 16)), true);
assert.sameValue(parseInt("10", 16), 16);
assert.sameValue(parseInt("0", 10), 0);
assert.sameValue((3.14).toString(), "3.14");
assert.sameValue((1000000).toString(), "1000000");
