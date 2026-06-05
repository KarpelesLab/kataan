/*---
description: Combinations of array methods producing aggregates
esid: sec-array.prototype
---*/
var data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
var evenSquaresSum = data
  .filter(function (x) { return x % 2 === 0; })
  .map(function (x) { return x * x; })
  .reduce(function (a, b) { return a + b; }, 0);
assert.sameValue(evenSquaresSum, 4 + 16 + 36 + 64 + 100);
var grouped = data.reduce(function (acc, x) {
  var key = x % 3;
  if (!acc[key]) acc[key] = [];
  acc[key].push(x);
  return acc;
}, {});
assert.sameValue(grouped[0].join(","), "3,6,9");
assert.sameValue(grouped[1].join(","), "1,4,7,10");
var flat = [[1, 2], [3, 4], [5]].reduce(function (a, b) { return a.concat(b); }, []);
assert.sameValue(flat.join(","), "1,2,3,4,5");
var max = data.reduce(function (a, b) { return Math.max(a, b); }, -Infinity);
assert.sameValue(max, 10);
var unique = [1, 1, 2, 3, 3, 3, 4].filter(function (x, i, arr) { return arr.indexOf(x) === i; });
assert.sameValue(unique.join(","), "1,2,3,4", "dedup via indexOf");
