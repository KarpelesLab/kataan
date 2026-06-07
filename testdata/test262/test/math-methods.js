/*---
description: Math methods (hypot, cbrt, log2/log10, clz32, imul, fround, sign, ...)
esid: sec-math
---*/
assert.sameValue(Math.hypot(3, 4), 5, "hypot 3,4");
assert.sameValue(Math.hypot(5, 12), 13, "hypot 5,12");
assert.sameValue(Math.cbrt(27), 3, "cbrt 27");
assert.sameValue(Math.cbrt(-8), -2, "cbrt -8");
assert.sameValue(Math.log2(8), 3, "log2 8");
assert.sameValue(Math.log10(1000), 3, "log10 1000");
assert.sameValue(Math.clz32(1), 31, "clz32 1");
assert.sameValue(Math.clz32(0), 32, "clz32 0");
assert.sameValue(Math.imul(3, 4), 12, "imul 3,4");
assert.sameValue(Math.imul(-5, 3), -15, "imul signed");
assert.sameValue(Math.fround(1.5), 1.5, "fround exact");
assert.sameValue(Math.sign(-5), -1, "sign negative");
assert.sameValue(Math.sign(3), 1, "sign positive");
assert.sameValue(Math.sign(0), 0, "sign zero");

// Sign-of-zero preservation (rendered identically, distinguished by Object.is).
assert.sameValue(Object.is(Math.sign(-0), -0), true, "sign(-0) is -0");
assert.sameValue(Object.is(Math.round(-0.1), -0), true, "round(-0.1) is -0");

assert.sameValue(Math.trunc(4.7), 4, "trunc positive");
assert.sameValue(Math.trunc(-4.7), -4, "trunc negative");
assert.sameValue(Math.round(2.5), 3, "round half up");
assert.sameValue(Math.round(-2.5), -2, "round half toward +Inf");

// Variadic min/max with empty -> identity element.
assert.sameValue(Math.max(), -Infinity, "max() identity");
assert.sameValue(Math.min(), Infinity, "min() identity");
assert.sameValue(Math.max(1, 2, 3), 3, "max(1,2,3)");

// Number statics.
assert.sameValue(Number.parseFloat("3.14abc"), 3.14, "Number.parseFloat");
assert.sameValue(Number.isNaN(NaN), true, "Number.isNaN");
assert.sameValue(Number.isFinite(Infinity), false, "Number.isFinite");

// Core arithmetic intrinsics (kept from the original suite).
assert.sameValue(Math.abs(-7), 7, "abs");
assert.sameValue(Math.floor(3.9), 3, "floor");
assert.sameValue(Math.ceil(3.1), 4, "ceil");
assert.sameValue(Math.sqrt(16), 4, "sqrt");
assert.sameValue(Math.pow(2, 8), 256, "pow");
assert.sameValue(Math.min(4, 2, 8), 2, "min(4,2,8)");
