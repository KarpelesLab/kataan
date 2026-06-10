/*---
description: toFixed/toExponential render Infinity/NaN as words; toFixed validates the digit range
esid: sec-number.prototype.tofixed
---*/
// Non-finite numbers stringify to words, not the host formatter's "inf".
assert.sameValue((Infinity).toFixed(2), "Infinity", "toFixed Infinity");
assert.sameValue((-Infinity).toFixed(2), "-Infinity", "toFixed -Infinity");
assert.sameValue((NaN).toFixed(2), "NaN", "toFixed NaN");
assert.sameValue((Infinity).toExponential(2), "Infinity", "toExponential Infinity");
assert.sameValue((NaN).toExponential(), "NaN", "toExponential NaN");

// Digit range [0,100]; out of range is a RangeError.
assert.throws(RangeError, function () { (1).toFixed(101); }, "digits 101");
assert.throws(RangeError, function () { (1).toFixed(-1); }, "digits -1");
assert.sameValue((1).toFixed(100).length, 102, "digits 100 allowed");
assert.sameValue((1).toFixed(0), "1", "digits 0 allowed");

// Ordinary formatting and rounding are unaffected.
assert.sameValue((2.345).toFixed(2), "2.35", "rounding");
assert.sameValue((5).toFixed(3), "5.000", "padding");
assert.sameValue((3.7).toFixed(), "4", "no-arg -> 0 digits");
assert.sameValue((12345).toExponential(2), "1.23e+4", "exponential");
