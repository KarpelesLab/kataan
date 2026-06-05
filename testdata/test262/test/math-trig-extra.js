/*---
description: Math trig, hyperbolic, and extra functions
esid: sec-math
---*/
assert.sameValue(Math.sin(0), 0);
assert.sameValue(Math.cos(0), 1);
assert.sameValue(Math.tan(0), 0);
assert.sameValue(Math.round(Math.sin(Math.PI / 2)), 1);
assert.sameValue(Math.round(Math.cos(Math.PI)), -1);
assert.sameValue(Math.asin(0), 0);
assert.sameValue(Math.acos(1), 0);
assert.sameValue(Math.atan(0), 0);
assert.sameValue(Math.round(Math.atan2(1, 1) * 4 / Math.PI), 1, "atan2(1,1) = PI/4");
assert.sameValue(Math.atan2(0, 1), 0);
assert.sameValue(Math.sinh(0), 0);
assert.sameValue(Math.cosh(0), 1);
assert.sameValue(Math.tanh(0), 0);
assert.sameValue(Math.asinh(0), 0);
assert.sameValue(Math.acosh(1), 0);
assert.sameValue(Math.atanh(0), 0);
assert.sameValue(Math.expm1(0), 0);
assert.sameValue(Math.log1p(0), 0);
assert.sameValue(Math.fround(1.5), 1.5);
assert.sameValue(Math.fround(1.1) !== 1.1, true, "fround loses double precision");
assert.sameValue(Math.clz32(1), 31);
assert.sameValue(Math.clz32(0), 32);
assert.sameValue(Math.clz32(0xFFFFFFFF), 0);
assert.sameValue(Math.imul(3, 4), 12);
assert.sameValue(Math.imul(-1, 8), -8);
assert.sameValue(Math.round(Math.sinh(1) * 1000), 1175, "sinh(1)");
