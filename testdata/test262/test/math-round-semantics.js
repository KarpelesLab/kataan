/*---
description: Math.round rounds half toward +Infinity (not away from zero)
esid: sec-math.round
---*/
// Positive halves round up.
assert.sameValue(Math.round(2.5), 3, "round(2.5)");
assert.sameValue(Math.round(0.5), 1, "round(0.5)");
assert.sameValue(Math.round(2.4), 2, "round(2.4)");

// Negative halves round toward +Infinity (so -2.5 -> -2, not -3).
assert.sameValue(Math.round(-2.5), -2, "round(-2.5) is -2, not -3");
assert.sameValue(Math.round(-1.5), -1, "round(-1.5)");
assert.sameValue(Math.round(-2.6), -3, "round(-2.6)");

// The sign of zero is preserved for x in [-0.5, -0).
assert.sameValue(Object.is(Math.round(-0.5), -0), true, "round(-0.5) is -0");
assert.sameValue(Object.is(Math.round(-0.4), -0), true, "round(-0.4) is -0");
assert.sameValue(Object.is(Math.round(0.4), 0), true, "round(0.4) is +0");

// The floating-point edge: the largest double below 0.5 must round to 0.
assert.sameValue(Math.round(0.49999999999999994), 0, "round(0.4999...94) is 0, not 1");

// Specials pass through.
assert.sameValue(Object.is(Math.round(-0), -0), true, "round(-0) is -0");
assert.sameValue(Math.round(Infinity), Infinity, "round(Infinity)");
assert.sameValue(Number.isNaN(Math.round(NaN)), true, "round(NaN)");
