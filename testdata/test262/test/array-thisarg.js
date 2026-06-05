/*---
description: Array iteration methods with a thisArg
esid: sec-array.prototype.map
---*/
var ctx = { multiplier: 3 };
var result = [1, 2, 3].map(function (x) { return x * this.multiplier; }, ctx);
assert.sameValue(result.join(","), "3,6,9", "map thisArg");
var sum = 0;
[1, 2, 3].forEach(function (x) { sum += x * this.multiplier; }, ctx);
assert.sameValue(sum, 18, "forEach thisArg");
var filtered = [1, 2, 3, 4].filter(function (x) { return x > this.threshold; }, { threshold: 2 });
assert.sameValue(filtered.join(","), "3,4", "filter thisArg");
assert.sameValue([1, 2, 3].some(function (x) { return x === this.target; }, { target: 2 }), true);
assert.sameValue([1, 2, 3].every(function (x) { return x <= this.max; }, { max: 3 }), true);
