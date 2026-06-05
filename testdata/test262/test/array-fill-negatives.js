/*---
description: Array fill and copyWithin with negative indices
esid: sec-array.prototype.fill
---*/
assert.sameValue([1, 2, 3, 4, 5].fill(0, -2).join(","), "1,2,3,0,0", "fill from negative");
assert.sameValue([1, 2, 3, 4, 5].fill(0, 1, -1).join(","), "1,0,0,0,5", "fill range with negative end");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(-2, 0).join(","), "1,2,3,1,2", "copyWithin negative target");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, -2).join(","), "4,5,3,4,5");
assert.sameValue([1, 2, 3].fill(9).join(","), "9,9,9");
assert.sameValue([1, 2, 3, 4].slice(-3, -1).join(","), "2,3");
