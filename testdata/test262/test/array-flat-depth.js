/*---
description: Array flat/flatMap with depth and copyWithin
esid: sec-array.prototype.flat
---*/
assert.sameValue([1, [2, [3, [4]]]].flat().join(","), "1,2,3,4", "flat default depth 1");
assert.sameValue([1, [2, [3, [4]]]].flat(2).join(","), "1,2,3,4", "flat depth 2");
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, -x]; }).join(","), "1,-1,2,-2,3,-3");
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3).join(","), "4,5,3,4,5");
assert.sameValue([0,0,0].fill(0).join(","), "0,0,0");
assert.sameValue([1, 2, 3].concat([4, 5], 6).join(","), "1,2,3,4,5,6");
