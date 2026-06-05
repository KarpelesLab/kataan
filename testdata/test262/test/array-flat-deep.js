/*---
description: Array flat and flatMap at various depths
esid: sec-array.prototype.flat
---*/
assert.sameValue([1, [2, [3, [4]]]].flat(Infinity).join(","), "1,2,3,4");
assert.sameValue([1, [2, [3]]].flat(1).join(","), "1,2,3", "flat(1) leaves [3] flattened once");
assert.sameValue([1, [2, [3]]].flat(0).length, 2);
assert.sameValue([[1], [2], [3]].flat().join(","), "1,2,3");
assert.sameValue([1, 2].flatMap(function (x) { return [x, x * 2]; }).join(","), "1,2,2,4");
assert.sameValue([1, 2, 3].flatMap(function (x) { return x % 2 === 0 ? [] : [x]; }).join(","), "1,3", "flatMap filters");
assert.sameValue(["a b", "c d"].flatMap(function (s) { return s.split(" "); }).join(","), "a,b,c,d");
assert.sameValue([1, [2], 3].flat().length, 3);
assert.sameValue([].flat().length, 0);
assert.sameValue([[[[1]]]].flat(Infinity).join(""), "1", "deeply nested");
