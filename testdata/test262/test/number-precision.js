/*---
description: Number toPrecision and toString radix
esid: sec-number.prototype.toprecision
---*/
assert.sameValue((123.456).toPrecision(4), "123.5");
assert.sameValue((255).toString(2), "11111111");
assert.sameValue((10).toString(8), "12");
