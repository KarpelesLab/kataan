/*---
description: Number toString in various radixes and parseInt round-trips
esid: sec-number.prototype.tostring
---*/
assert.sameValue((255).toString(16), "ff");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((-10).toString(2), "-1010");
assert.sameValue(parseInt("ff", 16), 255);
assert.sameValue(parseInt("11111111", 2), 255);
assert.sameValue((1000).toString(36), "rs");
