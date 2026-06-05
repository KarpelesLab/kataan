/*---
description: findLast, findLastIndex, at with negatives
esid: sec-array.prototype.findlast
---*/
var nums = [1, 2, 3, 4, 5, 6];
assert.sameValue(nums.findLast(function (x) { return x % 2 === 0; }), 6);
assert.sameValue(nums.findLastIndex(function (x) { return x % 2 === 0; }), 5);
assert.sameValue(nums.findLast(function (x) { return x > 10; }), undefined);
assert.sameValue(nums.findLastIndex(function (x) { return x > 10; }), -1);
assert.sameValue(nums.find(function (x) { return x > 3; }), 4);
assert.sameValue(nums.findLast(function (x) { return x < 4; }), 3, "last matching");
assert.sameValue(nums.at(-1), 6);
assert.sameValue(nums.at(-2), 5);
var words = ["apple", "banana", "cherry"];
assert.sameValue(words.findLast(function (w) { return w.length > 5; }), "cherry");
assert.sameValue(words.findIndex(function (w) { return w.startsWith("b"); }), 1);
