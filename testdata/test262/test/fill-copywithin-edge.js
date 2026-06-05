/*---
description: Array fill and copyWithin edge cases
esid: sec-array.prototype.fill
---*/
assert.sameValue([1, 2, 3, 4].fill(0).join(","), "0,0,0,0", "fill all");
assert.sameValue([1, 2, 3, 4].fill(0, 2).join(","), "1,2,0,0", "fill from index");
assert.sameValue([1, 2, 3, 4].fill(0, 1, 3).join(","), "1,0,0,4", "fill range");
assert.sameValue([1, 2, 3, 4].fill(0, -2).join(","), "1,2,0,0", "negative start");
assert.sameValue([1, 2, 3, 4].fill(9, 1, -1).join(","), "1,9,9,4", "negative end");
assert.sameValue(new Array(3).fill(7).join(","), "7,7,7");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3).join(","), "4,5,3,4,5");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(1, 3).join(","), "1,4,5,4,5");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3, 4).join(","), "4,2,3,4,5");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(-2, 0).join(","), "1,2,3,1,2");
var arr = [1, 2, 3];
arr.fill(0);
assert.sameValue(arr.join(","), "0,0,0", "fill mutates");
