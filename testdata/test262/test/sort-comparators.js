/*---
description: Array sort with comparators and stability of equal keys
esid: sec-array.prototype.sort
---*/
assert.sameValue([3, 1, 2].sort(function (a, b) { return a - b; }).join(","), "1,2,3");
assert.sameValue([3, 1, 2].sort(function (a, b) { return b - a; }).join(","), "3,2,1");
assert.sameValue(["bb", "a", "ccc"].sort(function (a, b) { return a.length - b.length; }).join(","), "a,bb,ccc");
assert.sameValue([10, 9, 1, 20].sort().join(","), "1,10,20,9");
