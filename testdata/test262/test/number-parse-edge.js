/*---
description: parseInt radix, parseFloat, and Number edge cases
esid: sec-parseint-string-radix
---*/
assert.sameValue(parseInt("ff", 16), 255);
assert.sameValue(parseInt("0x1A"), 26);
assert.sameValue(parseInt("101", 2), 5);
assert.sameValue(parseFloat("3.14abc"), 3.14);
assert.sameValue(parseInt("   42   "), 42);
assert.sameValue(isNaN(parseInt("xyz")), true);
