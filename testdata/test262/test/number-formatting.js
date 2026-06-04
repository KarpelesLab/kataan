/*---
description: Number static properties and formatting
esid: sec-properties-of-the-number-constructor
---*/
assert.sameValue(Number.MAX_SAFE_INTEGER, 9007199254740991);
assert.sameValue(Number.isFinite(Infinity), false);
assert.sameValue(Number("3.14"), 3.14);
assert.sameValue((1000000).toString(), "1000000");
assert.sameValue((0.1 + 0.2).toFixed(1), "0.3");
