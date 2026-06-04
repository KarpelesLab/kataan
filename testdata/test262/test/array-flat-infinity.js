/*---
description: Array flat with depth Infinity and deeply nested arrays
esid: sec-array.prototype.flat
---*/
assert.sameValue([1, [2, [3, [4, [5]]]]].flat(Infinity).join(","), "1,2,3,4,5");
assert.sameValue([1, [2, [3]]].flat(0).length, 2);
assert.sameValue([[1], [2], [3]].flat().join(","), "1,2,3");
