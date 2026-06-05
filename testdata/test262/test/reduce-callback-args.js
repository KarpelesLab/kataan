/*---
description: reduce/reduceRight callback receives (acc, cur, index, array)
esid: sec-array.prototype.reduce
---*/
var indices = [];
var total = [10, 20, 30].reduce(function (acc, cur, i, arr) {
  indices.push(i);
  return acc + cur;
}, 0);
assert.sameValue(total, 60);
assert.sameValue(indices.join(","), "0,1,2", "index passed");
var arrSeen = null;
[1].reduce(function (acc, cur, i, arr) { arrSeen = arr; return acc; }, 0);
assert.sameValue(arrSeen.length, 1, "array passed to callback");
var withoutInitial = [5, 10, 15].reduce(function (acc, cur) { return acc + cur; });
assert.sameValue(withoutInitial, 30, "no initial uses first element");
var concat = ["a", "b", "c"].reduceRight(function (acc, cur) { return acc + cur; });
assert.sameValue(concat, "cba");
var max = [3, 7, 2, 8, 5].reduce(function (a, b) { return Math.max(a, b); });
assert.sameValue(max, 8);
