/*---
description: Sort stability and numeric comparator correctness
esid: sec-array.prototype.sort
---*/
var data = [];
for (var i = 0; i < 50; i++) data.push({ key: i % 5, order: i });
data.sort(function (a, b) { return a.key - b.key; });
var stable = true;
for (var j = 1; j < data.length; j++) {
  if (data[j].key === data[j - 1].key && data[j].order < data[j - 1].order) stable = false;
}
assert.sameValue(stable, true, "stable sort");
var nums = [38, 27, 43, 3, 9, 82, 10];
nums.sort(function (a, b) { return a - b; });
assert.sameValue(nums.join(","), "3,9,10,27,38,43,82");
var desc = [1, 5, 2, 8, 3].sort(function (a, b) { return b - a; });
assert.sameValue(desc.join(","), "8,5,3,2,1");
var strings = ["banana", "apple", "cherry", "date"].sort();
assert.sameValue(strings.join(","), "apple,banana,cherry,date");
var byLength = ["aaa", "a", "aa", "aaaa"].sort(function (a, b) { return a.length - b.length; });
assert.sameValue(byLength.join(","), "a,aa,aaa,aaaa");
var mixed = [3, 1, 2].sort();
assert.sameValue(mixed.join(""), "123");
