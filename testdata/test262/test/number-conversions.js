/*---
description: Number conversions, NaN, and special values
esid: sec-tonumber
---*/
assert.sameValue(+"5", 5, "unary plus coerces");
assert.sameValue(+"", 0);
assert.sameValue(+"abc" !== +"abc", true, "NaN");
assert.sameValue(+true, 1);
assert.sameValue(+null, 0);
assert.sameValue(+[], 0, "empty array to 0");
assert.sameValue(+[5], 5, "single-element array");
assert.sameValue(Number.isNaN(+[1, 2]), true, "multi-element array is NaN");
assert.sameValue(parseInt("0xFF", 16), 255);
assert.sameValue(parseInt("11", 2), 3);
assert.sameValue(Number("0b101"), 5);
assert.sameValue(Number("Infinity"), Infinity);
assert.sameValue(1 / 0, Infinity);
assert.sameValue(Number.MAX_VALUE > 1e307, true);
