/*---
description: sort stability and correctness at scale
esid: sec-array.prototype.sort
---*/
var nums = [];
for (var i = 0; i < 100; i++) nums.push((i * 37) % 100);
nums.sort(function (a, b) { return a - b; });
var ok = true;
for (var j = 1; j < nums.length; j++) if (nums[j] < nums[j - 1]) ok = false;
assert.sameValue(ok, true, "fully sorted");
var records = [];
for (var k = 0; k < 20; k++) records.push({ key: k % 3, orig: k });
records.sort(function (a, b) { return a.key - b.key; });
var stable = true;
for (var m = 1; m < records.length; m++) {
  if (records[m].key === records[m - 1].key && records[m].orig < records[m - 1].orig) stable = false;
}
assert.sameValue(stable, true, "stable sort preserves order");
assert.sameValue(["banana", "apple", "cherry"].sort().join(","), "apple,banana,cherry", "default lexicographic");
assert.sameValue([10, 9, 100, 1].sort().join(","), "1,10,100,9", "default is string sort");
