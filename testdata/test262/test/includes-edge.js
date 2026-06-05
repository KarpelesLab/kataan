/*---
description: includes with NaN, -0, and various types
esid: sec-array.prototype.includes
---*/
assert.sameValue([1, 2, NaN].includes(NaN), true, "SameValueZero finds NaN");
assert.sameValue([1, 2, 3].includes(2), true);
assert.sameValue([1, 2, 3].includes(4), false);
assert.sameValue([0].includes(-0), true, "0 and -0 are SameValueZero");
assert.sameValue(["a", "b"].includes("a"), true);
assert.sameValue([undefined].includes(undefined), true);
assert.sameValue([null].includes(null), true);
assert.sameValue("hello".includes("ell"), true);
assert.sameValue("hello".includes("xyz"), false);
var obj = {};
assert.sameValue([obj].includes(obj), true, "by identity");
assert.sameValue([{}].includes({}), false);
