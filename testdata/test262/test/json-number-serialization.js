/*---
description: JSON.stringify serializes numbers with the spec Number ToString
esid: sec-serializejsonproperty
---*/
// Negative zero serializes as "0".
assert.sameValue(JSON.stringify(-0), "0", "negative zero");
assert.sameValue(JSON.stringify([-0]), "[0]", "negative zero in an array");
assert.sameValue(JSON.stringify({ x: -0 }), '{"x":0}', "negative zero in an object");
// Large magnitudes use exponential form (>= 1e21).
assert.sameValue(JSON.stringify(1e21), "1e+21", "1e21 uses exponential");
assert.sameValue(JSON.stringify(1e-7), "1e-7", "small magnitude exponential");
assert.sameValue(JSON.stringify(1.5e300), "1.5e+300", "very large");
// Ordinary numbers are unchanged.
assert.sameValue(JSON.stringify(100), "100", "integer");
assert.sameValue(JSON.stringify(1.5), "1.5", "fraction");
assert.sameValue(JSON.stringify(0.001), "0.001", "small fraction (not exponential)");
assert.sameValue(JSON.stringify(-42), "-42", "negative");
assert.sameValue(JSON.stringify(1e20), "100000000000000000000", "1e20 stays decimal");
// Non-finite numbers are null.
assert.sameValue(JSON.stringify(NaN), "null", "NaN");
assert.sameValue(JSON.stringify(Infinity), "null", "Infinity");
assert.sameValue(JSON.stringify([NaN, Infinity, -Infinity]), "[null,null,null]", "non-finite in array");
// A round-trip of a mixed array.
assert.sameValue(JSON.stringify([-0, 1e21, 0.001, -42]), "[0,1e+21,0.001,-42]", "mixed");
