/*---
description: every, some, find, findIndex, forEach, and reduceRight
esid: sec-array.prototype.every
---*/
assert.sameValue([2, 4, 6].every(function (x) { return x % 2 === 0; }), true);
assert.sameValue([2, 4, 5].every(function (x) { return x % 2 === 0; }), false);
assert.sameValue([1, 3, 4].some(function (x) { return x % 2 === 0; }), true);
assert.sameValue([1, 2, 3, 4].find(function (x) { return x > 2; }), 3);
assert.sameValue([1, 2, 3, 4].findIndex(function (x) { return x > 2; }), 2);
var sum = 0;
[1, 2, 3].forEach(function (x) { sum += x; });
assert.sameValue(sum, 6);
assert.sameValue(["a", "b", "c"].reduceRight(function (acc, x) { return acc + x; }), "cba");
assert.sameValue([1, 2, 3].map(function (x, i) { return x + i; }).join(","), "1,3,5");
