/*---
description: toSorted, toReversed, with (non-mutating array methods)
esid: sec-array.prototype.tosorted
---*/
var a = [3, 1, 2];
var sorted = a.toSorted(function (x, y) { return x - y; });
assert.sameValue(sorted.join(","), "1,2,3");
assert.sameValue(a.join(","), "3,1,2", "toSorted does not mutate");
assert.sameValue([1, 2, 3].toReversed().join(","), "3,2,1");
assert.sameValue([1, 2, 3].with(1, 9).join(","), "1,9,3");
