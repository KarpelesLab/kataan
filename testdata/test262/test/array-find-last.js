/*---
description: Array findLast, findLastIndex, copyWithin, flatMap
esid: sec-properties-of-the-array-prototype-object
---*/
assert.sameValue([1, 2, 3, 4].findLast(function (x) { return x % 2 === 1; }), 3);
assert.sameValue([1, 2, 3, 4].findLastIndex(function (x) { return x % 2 === 1; }), 2);
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, x * 10]; }).join(","), "1,10,2,20,3,30");
