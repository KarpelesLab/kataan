/*---
description: Array sort with negative numbers and stable comparators
esid: sec-array.prototype.sort
---*/
assert.sameValue([-3, 1, -1, 2, 0].sort(function (a, b) { return a - b; }).join(","), "-3,-1,0,1,2");
assert.sameValue([3, 1, 2].sort(function (a, b) { return b - a; }).join(","), "3,2,1");
var items = [{ n: "a", p: 2 }, { n: "b", p: 1 }, { n: "c", p: 2 }, { n: "d", p: 1 }];
var sorted = items.sort(function (x, y) { return x.p - y.p; });
assert.sameValue(sorted.map(function (i) { return i.n; }).join(""), "bdac", "stable sort");
assert.sameValue([10, 9, 100, 1].sort(function (a, b) { return a - b; }).join(","), "1,9,10,100");
assert.sameValue([].sort().length, 0);
assert.sameValue([5].sort().join(","), "5");
