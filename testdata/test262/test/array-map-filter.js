/*---
description: Array.prototype.map and filter produce expected results
esid: sec-array.prototype.map
includes: [compareArray.js]
---*/
var doubled = [1, 2, 3].map(function (x) { return x * 2; });
assert.sameValue(doubled.join(","), "2,4,6", "map doubles each element");
var evens = [1, 2, 3, 4, 5].filter(function (x) { return x % 2 === 0; });
assert.sameValue(evens.join(","), "2,4", "filter keeps even numbers");
assert.sameValue([1, 2, 3, 4].reduce(function (a, b) { return a + b; }, 0), 10, "reduce sums");
