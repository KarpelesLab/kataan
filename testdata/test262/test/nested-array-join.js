/*---
description: Array join with nested arrays and various element types
esid: sec-array.prototype.join
---*/
assert.sameValue([1, [2, 3], 4].join(","), "1,2,3,4", "nested array stringified");
assert.sameValue([[1, 2], [3, 4]].join(";"), "1,2;3,4");
assert.sameValue([1, 2, 3].join(""), "123");
assert.sameValue([1, 2, 3].join(), "1,2,3", "default separator is comma");
assert.sameValue([true, false].join("-"), "true-false");
assert.sameValue([null, undefined, 1].join(","), ",,1");
assert.sameValue(["a", "b"].toString(), "a,b", "toString joins with comma");
assert.sameValue([[1], [2], [3]].join("|"), "1|2|3");
