/*---
description: Numeric separators and various number literal forms
esid: sec-literals-numeric-literals
---*/
assert.sameValue(1_000_000, 1000000);
assert.sameValue(0xff, 255);
assert.sameValue(0b1010, 10);
assert.sameValue(0o17, 15);
assert.sameValue(1e3, 1000);
assert.sameValue(1.5e-2, 0.015);
assert.sameValue(.5, 0.5);
