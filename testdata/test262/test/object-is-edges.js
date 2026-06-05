/*---
description: Object.is special cases vs ===
esid: sec-object.is
---*/
assert.sameValue(Object.is(NaN, NaN), true);
assert.sameValue(NaN === NaN, false);
assert.sameValue(Object.is(0, -0), false);
assert.sameValue(0 === -0, true);
assert.sameValue(Object.is(-0, -0), true);
assert.sameValue(Object.is(1, 1), true);
var o = {};
assert.sameValue(Object.is(o, o), true);
assert.sameValue(Object.is({}, {}), false, "distinct objects");
assert.sameValue(Object.is("x", "x"), true);
assert.sameValue(Object.is(null, null), true);
assert.sameValue(Object.is(undefined, undefined), true);
