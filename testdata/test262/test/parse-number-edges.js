/*---
description: parseInt/parseFloat edge cases and radix handling
esid: sec-parseint-string-radix
---*/
assert.sameValue(parseInt("123", 10), 123);
assert.sameValue(parseInt("z", 36), 35);
assert.sameValue(parseInt("10", 2), 2);
assert.sameValue(parseInt("0x1F"), 31, "auto hex");
assert.sameValue(parseInt("  42  "), 42, "leading whitespace");
assert.sameValue(parseInt("3.99"), 3, "stops at decimal");
assert.sameValue(Number.isNaN(parseInt("")), true);
assert.sameValue(parseFloat("3.14e2"), 314);
assert.sameValue(parseFloat("-0.5"), -0.5);
assert.sameValue(parseFloat(".25"), 0.25);
assert.sameValue(parseInt("100", 16), 256);
assert.sameValue(parseInt("ff", 16), 255);
