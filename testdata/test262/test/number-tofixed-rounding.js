/*---
description: Number.prototype.toFixed rounds the exact double value, ties away from zero
esid: sec-number.prototype.tofixed
---*/
// Near-half doubles (slightly below .X5) round DOWN — not up.
assert.sameValue((2.355).toFixed(2), "2.35", "2.355 is 2.35499...");
assert.sameValue((0.615).toFixed(2), "0.61", "0.615 is 0.61499...");
assert.sameValue((1.555).toFixed(2), "1.55", "1.555 is 1.55499...");
assert.sameValue((1.005).toFixed(2), "1.00", "1.005 is 1.00499...");
assert.sameValue((8.575).toFixed(2), "8.57", "8.575 is 8.57499...");
assert.sameValue((35.855).toFixed(2), "35.85", "35.855 is 35.85499...");
// A double slightly above .X5 rounds up.
assert.sameValue((2.345).toFixed(2), "2.35", "2.345 is 2.34500...");

// Exact halves (representable) round half AWAY from zero (the spec's larger n).
assert.sameValue((0.5).toFixed(0), "1", "0.5 -> 1");
assert.sameValue((1.5).toFixed(0), "2", "1.5 -> 2");
assert.sameValue((2.5).toFixed(0), "3", "2.5 -> 3");
assert.sameValue((0.25).toFixed(1), "0.3", "0.25 -> 0.3");
assert.sameValue((0.75).toFixed(1), "0.8", "0.75 -> 0.8");
assert.sameValue((0.125).toFixed(2), "0.13", "0.125 -> 0.13");

// Signs, zero, integers, and the maximum precision.
assert.sameValue((-2.355).toFixed(2), "-2.35", "negative");
assert.sameValue((-1.5).toFixed(1), "-1.5", "negative exact");
assert.sameValue((0).toFixed(2), "0.00", "zero");
assert.sameValue((-0).toFixed(2), "0.00", "negative zero has no sign");
assert.sameValue((123.456).toFixed(0), "123", "to integer");
assert.sameValue((1.1).toFixed(100).slice(0, 20), "1.100000000000000088", "full expansion");
