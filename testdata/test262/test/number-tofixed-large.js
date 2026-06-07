/*---
description: Number.prototype.toFixed uses ToString for magnitudes >= 1e21
esid: sec-number.prototype.tofixed
---*/
// A magnitude of 1e21 or more is rendered as its ToString (exponential), not as a
// full decimal expansion.
assert.sameValue((1e21).toFixed(2), "1e+21", "1e21");
assert.sameValue((1e21).toFixed(0), "1e+21", "1e21 zero digits");
assert.sameValue((-1e21).toFixed(2), "-1e+21", "negative 1e21");
assert.sameValue((1e30).toFixed(5), "1e+30", "1e30");

// Below 1e21, toFixed still expands to fixed-point with rounding (half away
// from zero).
assert.sameValue((123.456).toFixed(2), "123.46", "rounds");
assert.sameValue((0).toFixed(2), "0.00", "zero");
assert.sameValue((1e20).toFixed(1), "100000000000000000000.0", "just below 1e21 expands");
assert.sameValue((2.5).toFixed(0), "3", "half away from zero");
