/*---
description: Array sort with comparators, stability, and numbers
esid: sec-array.prototype.sort
---*/
var data = [{ k: 1, v: "a" }, { k: 1, v: "b" }, { k: 0, v: "c" }];
var sorted = data.sort(function (x, y) { return x.k - y.k; });
assert.sameValue(sorted[0].v, "c", "lowest key first");
assert.sameValue(sorted[1].v + sorted[2].v, "ab", "stable: a before b");
assert.sameValue([5, 3, 8, 1].sort(function (a, b) { return a - b; }).join(","), "1,3,5,8");
assert.sameValue([5, 3, 8, 1].sort(function (a, b) { return b - a; }).join(","), "8,5,3,1");
assert.sameValue(["c", "a", "b"].sort().join(""), "abc");
