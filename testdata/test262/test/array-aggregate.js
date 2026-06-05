/*---
description: Array aggregation patterns with reduce
esid: sec-array.prototype.reduce
---*/
var nums = [1, 2, 3, 4, 5];
assert.sameValue(nums.reduce(function (a, b) { return a + b; }, 0), 15, "sum");
assert.sameValue(nums.reduce(function (a, b) { return a * b; }, 1), 120, "product");
assert.sameValue(nums.reduce(function (a, b) { return Math.max(a, b); }), 5, "max");
assert.sameValue(nums.reduce(function (a, b) { return Math.min(a, b); }), 1, "min");
var words = ["hello", "world"];
assert.sameValue(words.reduce(function (a, b) { return a + " " + b; }), "hello world");
var counts = ["a", "b", "a", "c", "a"].reduce(function (acc, x) { acc[x] = (acc[x] || 0) + 1; return acc; }, {});
assert.sameValue(counts.a, 3);
var grouped = [1, 2, 3, 4, 5, 6].reduce(function (acc, x) { (x % 2 ? acc.odd : acc.even).push(x); return acc; }, { odd: [], even: [] });
assert.sameValue(grouped.odd.join(","), "1,3,5");
assert.sameValue(grouped.even.join(","), "2,4,6");
var flattened = [[1, 2], [3, 4], [5]].reduce(function (a, b) { return a.concat(b); }, []);
assert.sameValue(flattened.join(","), "1,2,3,4,5");
var avg = nums.reduce(function (a, b) { return a + b; }, 0) / nums.length;
assert.sameValue(avg, 3);
var reversed = nums.reduce(function (acc, x) { return [x].concat(acc); }, []);
assert.sameValue(reversed.join(","), "5,4,3,2,1");
