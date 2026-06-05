/*---
description: All bitwise operators including unsigned shift
esid: sec-bitwise-operators
---*/
assert.sameValue(5 & 3, 1);
assert.sameValue(5 | 3, 7);
assert.sameValue(5 ^ 3, 6);
assert.sameValue(~5, -6);
assert.sameValue(1 << 4, 16);
assert.sameValue(256 >> 2, 64);
assert.sameValue(-1 >>> 28, 15, "unsigned right shift");
assert.sameValue(-8 >> 1, -4, "signed preserves sign");
assert.sameValue((0xFF & 0x0F), 15);
assert.sameValue((0xF0 | 0x0F), 255);
assert.sameValue(1 << 31, -2147483648, "overflow to negative");
assert.sameValue((-1 >>> 0), 4294967295, "full unsigned");
assert.sameValue(7 & ~3, 4, "clear bits");
assert.sameValue(0 | 3.9, 3, "truncates to int32");
