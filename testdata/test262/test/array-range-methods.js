/*---
description: Array fill, copyWithin, and slice with ranges
esid: sec-array.prototype.fill
---*/
assert.sameValue([1, 2, 3, 4].fill(0, 1, 3).join(","), "1,0,0,4", "fill range");
assert.sameValue([1, 2, 3, 4].fill(9).join(","), "9,9,9,9", "fill all");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3, 5).join(","), "4,5,3,4,5");
assert.sameValue([1, 2, 3, 4, 5].slice(1, 4).join(","), "2,3,4");
assert.sameValue([1, 2, 3, 4, 5].slice(-2).join(","), "4,5", "negative slice");
assert.sameValue([1, 2, 3, 4, 5].splice(1, 2).join(","), "2,3", "splice returns removed");
var a = [1, 2, 3];
a.splice(1, 0, "x", "y");
assert.sameValue(a.join(","), "1,x,y,2,3", "splice inserts");
