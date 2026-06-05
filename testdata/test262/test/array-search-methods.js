/*---
description: indexOf/lastIndexOf/includes with fromIndex, and find variants
esid: sec-array.prototype.indexof
---*/
var a = [10, 20, 30, 20, 10];
assert.sameValue(a.indexOf(20), 1);
assert.sameValue(a.indexOf(20, 2), 3, "fromIndex");
assert.sameValue(a.lastIndexOf(20), 3);
assert.sameValue(a.lastIndexOf(10, 3), 0);
assert.sameValue(a.includes(30), true);
assert.sameValue(a.includes(99), false);
assert.sameValue(a.includes(10, 1), true);
assert.sameValue([1, 2, 3].indexOf(4), -1);
assert.sameValue(["a", "b", "c"].indexOf("b"), 1);
assert.sameValue([1, 2, 3, 4].findLastIndex(function (x) { return x < 3; }), 1);
