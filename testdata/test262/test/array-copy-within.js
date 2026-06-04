/*---
description: copyWithin and multi-level flat
esid: sec-array.prototype.copywithin
---*/
assert.sameValue([1, 2, 3, 4, 5].copyWithin(0, 3).join(","), "4,5,3,4,5");
assert.sameValue([1, [2, [3, [4]]]].flat(2).join(","), "1,2,3,4");
assert.sameValue([1, [2], [3]].flat().length, 3);
