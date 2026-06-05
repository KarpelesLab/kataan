/*---
description: Array sort with undefined and mixed values
esid: sec-array.prototype.sort
---*/
assert.sameValue([3, undefined, 1, undefined, 2].sort().join(","), "1,2,3,,", "undefined sorts to end");
assert.sameValue([3, 1, 2].sort().length, 3);
var withComparator = [3, 1, 2].sort(function (a, b) { return a - b; });
assert.sameValue(withComparator.join(","), "1,2,3");
assert.sameValue(["c", "a", "b"].sort().join(""), "abc");
var stable = [{ k: 1, v: "a" }, { k: 1, v: "b" }, { k: 0, v: "c" }];
stable.sort(function (x, y) { return x.k - y.k; });
assert.sameValue(stable.map(function (o) { return o.v; }).join(""), "cab", "stable sort");
assert.sameValue([10, 2, 1].sort().join(","), "1,10,2", "default lexicographic");
assert.sameValue([5].sort().join(""), "5");
assert.sameValue([].sort().length, 0);
var nums = [40, 1, 5, 200];
assert.sameValue(nums.sort(function (a, b) { return a - b; }).join(","), "1,5,40,200");
