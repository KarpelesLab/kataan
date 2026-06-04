/*---
description: Object.is and Number predicate functions
esid: sec-object.is
---*/
assert.sameValue(Object.is(NaN, NaN), true, "Object.is treats NaN as equal");
assert.sameValue(Object.is(0, -0), false, "Object.is distinguishes +0 and -0");
assert.sameValue(Object.is(1, 1), true);
assert.sameValue(Object.is("a", "a"), true);
assert.sameValue(Number.isSafeInteger(9007199254740991), true);
assert.sameValue(Number.isSafeInteger(9007199254740992), false);
assert.sameValue(Number.isSafeInteger(1.5), false);
