/*---
description: parseInt, parseFloat edge cases
esid: sec-parseint-string-radix
---*/
assert.sameValue(parseInt("42"), 42);
assert.sameValue(parseInt("42.9"), 42, "parseInt truncates");
assert.sameValue(parseInt("  42  "), 42, "leading whitespace");
assert.sameValue(parseInt("42abc"), 42, "trailing non-digits");
assert.sameValue(Number.isNaN(parseInt("abc")), true);
assert.sameValue(parseInt("0x1F"), 31, "hex auto-detect");
assert.sameValue(parseInt("-42"), -42);
assert.sameValue(parseInt("+42"), 42);
assert.sameValue(parseFloat("3.14"), 3.14);
assert.sameValue(parseFloat("3.14.15"), 3.14, "stops at second dot");
assert.sameValue(parseFloat(".5"), 0.5);
assert.sameValue(parseFloat("1e3"), 1000, "exponent");
assert.sameValue(parseFloat("  2.5xyz"), 2.5);
assert.sameValue(Number.isNaN(parseFloat("abc")), true);
assert.sameValue(parseInt("11", 2), 3, "binary radix");
