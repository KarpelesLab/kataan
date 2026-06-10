/*---
description: parseInt rejects a radix outside [2,36] (including negative) as NaN
esid: sec-parseint-string-radix
---*/
// A negative radix is invalid (must not saturate to 0 / default to base 10).
assert.sameValue(isNaN(parseInt("10", -5)), true, "radix -5 invalid");
assert.sameValue(isNaN(parseInt("10", -1)), true, "radix -1 invalid");
// Other out-of-range radices.
assert.sameValue(isNaN(parseInt("10", 1)), true, "radix 1 invalid");
assert.sameValue(isNaN(parseInt("10", 37)), true, "radix 37 invalid");
assert.sameValue(isNaN(parseInt("10", 100)), true, "radix 100 invalid");

// Radix 0 (or omitted) infers from the string.
assert.sameValue(parseInt("10", 0), 10, "radix 0 -> base 10");
assert.sameValue(parseInt("0x1F", 0), 31, "radix 0 with 0x -> base 16");
assert.sameValue(parseInt("42"), 42, "no radix");

// Valid radices, and a fractional radix truncates.
assert.sameValue(parseInt("ff", 16), 255, "hex");
assert.sameValue(parseInt("z", 36), 35, "base 36");
assert.sameValue(parseInt("10", 2), 2, "binary");
assert.sameValue(parseInt("ff", 16.9), 255, "fractional radix truncates to 16");

// Number.parseInt shares the behavior.
assert.sameValue(isNaN(Number.parseInt("10", -5)), true, "Number.parseInt radix -5");
assert.sameValue(Number.parseInt("ff", 16), 255, "Number.parseInt hex");
