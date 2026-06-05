/*---
description: Sort with comparators returning various values
esid: sec-array.prototype.sort
---*/
assert.sameValue([3, 1, 2].sort(function (a, b) { return a - b; }).join(","), "1,2,3");
assert.sameValue([3, 1, 2].sort(function (a, b) { return a > b ? 1 : -1; }).join(","), "1,2,3", "1/-1 comparator");
assert.sameValue([3, 1, 2].sort(function (a, b) { return b - a; }).join(","), "3,2,1");
var byKey = [{ k: 3 }, { k: 1 }, { k: 2 }].sort(function (a, b) { return a.k - b.k; });
assert.sameValue(byKey.map(function (o) { return o.k; }).join(""), "123");
assert.sameValue([10, 100, 1].sort(function (a, b) { return a - b; }).join(","), "1,10,100");
assert.sameValue(["c", "a", "b"].sort(function (a, b) { return a < b ? -1 : a > b ? 1 : 0; }).join(""), "abc");
var floats = [3.14, 1.41, 2.71];
assert.sameValue(floats.sort(function (a, b) { return a - b; })[0], 1.41);
var negatives = [-1, -5, -3, -2];
assert.sameValue(negatives.sort(function (a, b) { return a - b; }).join(","), "-5,-3,-2,-1");
assert.sameValue([5].sort(function (a, b) { return a - b; }).join(""), "5");
assert.sameValue([].sort(function (a, b) { return a - b; }).length, 0);
var mixed = [3, 1, 4, 1, 5, 9, 2, 6].sort(function (a, b) { return a - b; });
assert.sameValue(mixed.join(","), "1,1,2,3,4,5,6,9");
