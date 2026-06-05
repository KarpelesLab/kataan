/*---
description: Default sort (string comparison) vs numeric comparator
esid: sec-array.prototype.sort
---*/
assert.sameValue([3, 1, 2].sort().join(","), "1,2,3");
assert.sameValue([10, 1, 2].sort().join(","), "1,10,2", "default is lexicographic");
assert.sameValue([10, 1, 2].sort(function (a, b) { return a - b; }).join(","), "1,2,10");
assert.sameValue(["banana", "apple", "cherry"].sort().join(","), "apple,banana,cherry");
assert.sameValue([-1, -10, -2].sort(function (a, b) { return a - b; }).join(","), "-10,-2,-1");
assert.sameValue([true, false, true].sort().join(","), "false,true,true");
assert.sameValue([3, 1, 2].sort(function (a, b) { return b - a; }).join(","), "3,2,1");
var mixed = ["10", "9", "100", "1"].sort();
assert.sameValue(mixed.join(","), "1,10,100,9", "string sort");
var empty = [].sort();
assert.sameValue(empty.length, 0);
assert.sameValue([1].sort().join(""), "1");
