/*---
description: sort with comparator returning various signs, in place
esid: sec-array.prototype.sort
---*/
var a = [3, 1, 4, 1, 5, 9, 2, 6];
var sorted = a.sort(function (x, y) { return x - y; });
assert.sameValue(sorted.join(","), "1,1,2,3,4,5,6,9");
assert.sameValue(a, sorted, "sort returns the same array (in place)");
assert.sameValue([3, 1, 2].sort(function (x, y) { return y - x; }).join(","), "3,2,1", "descending");
var byLength = ["ccc", "a", "bb"].sort(function (x, y) { return x.length - y.length; });
assert.sameValue(byLength.join(","), "a,bb,ccc");
var stable = [{ k: 2, i: 0 }, { k: 1, i: 1 }, { k: 2, i: 2 }, { k: 1, i: 3 }];
stable.sort(function (x, y) { return x.k - y.k; });
assert.sameValue(stable.map(function (o) { return o.i; }).join(""), "1302", "stable");
