/*---
description: Array findLast, findLastIndex, flat, and flatMap
esid: sec-array.prototype.findlast
---*/
assert.sameValue([1, 2, 3, 4, 5].findLast(function (x) { return x % 2 === 0; }), 4);
assert.sameValue([1, 2, 3, 4, 5].findLastIndex(function (x) { return x % 2 === 0; }), 3);
assert.sameValue([1, 2, 3].findLast(function (x) { return x > 10; }), undefined);
assert.sameValue([[1, 2], [3, [4, 5]]].flat().length, 4, "flat depth 1");
assert.sameValue([[1, 2], [3, [4, 5]]].flat(2).join(","), "1,2,3,4,5");
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, x]; }).join(","), "1,1,2,2,3,3");
assert.sameValue([1, 2, 3, 4].filter(function (x) { return x > 2; }).join(","), "3,4");
