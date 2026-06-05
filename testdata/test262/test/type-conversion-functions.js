/*---
description: Number, String, Boolean, Array as conversion functions
esid: sec-number-constructor
---*/
assert.sameValue(Number("42"), 42);
assert.sameValue(Number("3.14"), 3.14);
assert.sameValue(Number(""), 0);
assert.sameValue(Number("  10  "), 10);
assert.sameValue(Number(true), 1);
assert.sameValue(Number(false), 0);
assert.sameValue(Number(null), 0);
assert.sameValue(Number.isNaN(Number(undefined)), true);
assert.sameValue(String(42), "42");
assert.sameValue(String(true), "true");
assert.sameValue(String(null), "null");
assert.sameValue(String([1, 2, 3]), "1,2,3");
assert.sameValue(Boolean(0), false);
assert.sameValue(Boolean(""), false);
assert.sameValue(Boolean("x"), true);
assert.sameValue(Boolean([]), true, "empty array is truthy");
assert.sameValue(Boolean(null), false);
