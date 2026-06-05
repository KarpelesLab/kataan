/*---
description: Number.prototype.toString with various radixes, parseInt
esid: sec-number.prototype.tostring
---*/
assert.sameValue((255).toString(16), "ff");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((8).toString(8), "10");
assert.sameValue((35).toString(36), "z");
assert.sameValue((1000).toString(16), "3e8");
assert.sameValue((0).toString(2), "0");
assert.sameValue(parseInt("ff", 16), 255);
assert.sameValue(parseInt("11111111", 2), 255);
assert.sameValue(parseInt("z", 36), 35);
assert.sameValue(parseInt("777", 8), 511);
assert.sameValue(parseInt("100", 2), 4);
assert.sameValue(parseInt("deadbeef", 16), 3735928559);
assert.sameValue((-255).toString(16), "-ff", "negative");
assert.sameValue(Number.parseInt("0x1F", 16), 31);
