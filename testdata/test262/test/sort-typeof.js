/*---
description: Array sort with comparator and typeof results
esid: sec-array.prototype.sort
---*/
assert.sameValue([3, 1, 2].sort().join(","), "1,2,3", "default sort");
assert.sameValue([10, 2, 1].sort().join(","), "1,10,2", "default sort is lexicographic");
assert.sameValue([10, 2, 1].sort(function (a, b) { return a - b; }).join(","), "1,2,10", "numeric sort");
assert.sameValue(["banana", "apple", "cherry"].sort().join(","), "apple,banana,cherry");
assert.sameValue(typeof undefined, "undefined");
assert.sameValue(typeof 1, "number");
assert.sameValue(typeof "s", "string");
assert.sameValue(typeof true, "boolean");
assert.sameValue(typeof function () {}, "function");
assert.sameValue(typeof {}, "object");
assert.sameValue(typeof [], "object");
assert.sameValue(typeof null, "object");
assert.sameValue(typeof Symbol(), "symbol");
