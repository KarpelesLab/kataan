/*---
description: Array.isArray and sort with undefined/mixed values
esid: sec-array.isarray
---*/
assert.sameValue(Array.isArray([]), true);
assert.sameValue(Array.isArray([1, 2]), true);
assert.sameValue(Array.isArray({}), false);
assert.sameValue(Array.isArray("abc"), false);
assert.sameValue(Array.isArray(null), false);
assert.sameValue(Array.isArray(Array.from("ab")), true, "Array.from yields a real array");
assert.sameValue([3, 1, 2].sort().join(","), "1,2,3");
assert.sameValue([20, 3, 100].sort().join(","), "100,20,3", "default lexicographic");
assert.sameValue([20, 3, 100].sort(function (a, b) { return a - b; }).join(","), "3,20,100");
assert.sameValue(["banana", "apple"].sort().join(","), "apple,banana");
