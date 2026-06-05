/*---
description: Array includes/indexOf with fromIndex and NaN
esid: sec-array.prototype.includes
---*/
var a = [1, 2, 3, 2, 1];
assert.sameValue(a.includes(2), true);
assert.sameValue(a.includes(2, 2), true, "fromIndex finds later");
assert.sameValue(a.includes(2, 4), false, "past last 2");
assert.sameValue(a.includes(1, -1), true, "negative fromIndex");
assert.sameValue(a.includes(3, -2), false);
assert.sameValue([NaN].includes(NaN), true, "includes finds NaN");
assert.sameValue([NaN].indexOf(NaN), -1, "indexOf does not");
assert.sameValue([1, 2, 3].indexOf(2, 2), -1, "indexOf fromIndex");
assert.sameValue([1, 2, 3].indexOf(2, -2), 1);
assert.sameValue(a.lastIndexOf(2), 3);
assert.sameValue(a.lastIndexOf(2, 2), 1, "lastIndexOf fromIndex");
assert.sameValue([1, 2, 3].includes(4), false);
assert.sameValue([].includes(1), false);
assert.sameValue(["a", "b"].includes("a"), true);
