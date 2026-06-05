/*---
description: indexOf, lastIndexOf, includes with negative fromIndex
esid: sec-array.prototype.indexof
---*/
var a = [1, 2, 3, 2, 1];
assert.sameValue(a.indexOf(2), 1);
assert.sameValue(a.indexOf(2, 2), 3, "fromIndex skips earlier");
assert.sameValue(a.indexOf(2, -2), 3, "negative fromIndex");
assert.sameValue(a.indexOf(1, -1), 4);
assert.sameValue(a.includes(3, 3), false, "includes with fromIndex past it");
assert.sameValue(a.includes(1, -1), true);
assert.sameValue([1, 2, 3].indexOf(5), -1);
assert.sameValue(["a", "b", "a"].lastIndexOf("a"), 2);
assert.sameValue(["a", "b", "a"].lastIndexOf("a", 1), 0);
