/*---
description: Array at with negatives, flat with various depths
esid: sec-array.prototype.at
---*/
assert.sameValue([1, 2, 3].at(0), 1);
assert.sameValue([1, 2, 3].at(-1), 3);
assert.sameValue([1, 2, 3].at(-3), 1);
assert.sameValue([1, 2, 3].at(5), undefined);
assert.sameValue([1, 2, 3].at(-5), undefined);
assert.sameValue([1, [2, [3, [4]]]].flat(Infinity).join(","), "1,2,3,4");
assert.sameValue([1, [2, [3]]].flat(0).length, 2, "depth 0 no flatten");
assert.sameValue([[1], [2], [3]].flat().join(","), "1,2,3");
assert.sameValue([1, 2, [3, 4, [5]]].flat().length, 5, "flat depth 1 leaves inner array");
assert.sameValue([].flat().length, 0);
assert.sameValue([1, [2], 3].flatMap(function (x) { return Array.isArray(x) ? x : [x]; }).join(","), "1,2,3");
