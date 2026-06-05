/*---
description: Strict equality and SameValueZero in array methods
esid: sec-strict-equality-comparison
---*/
assert.sameValue(1 === 1, true);
assert.sameValue(1 === "1", false, "no coercion");
assert.sameValue(null === null, true);
assert.sameValue(undefined === undefined, true);
assert.sameValue(NaN === NaN, false);
assert.sameValue([1, 2, NaN].includes(NaN), true, "includes uses SameValueZero");
assert.sameValue([1, 2, NaN].indexOf(NaN), -1, "indexOf uses strict equality");
assert.sameValue([0, -0].indexOf(-0), 0);
var o = {};
assert.sameValue([o].indexOf(o), 0, "objects by identity");
assert.sameValue([{}].indexOf({}), -1, "distinct objects");
assert.sameValue("abc" === "abc", true, "strings by value");
