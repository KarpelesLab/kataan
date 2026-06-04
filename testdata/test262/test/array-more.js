/*---
description: Array.isArray, flat, fill, findIndex
esid: sec-properties-of-the-array-prototype-object
---*/
assert.sameValue(Array.isArray([1, 2]), true);
assert.sameValue(Array.isArray("no"), false);
assert.sameValue([1, [2, [3]]].flat().length, 3);
assert.sameValue([0, 0, 0].fill(7).join(","), "7,7,7");
assert.sameValue([5, 10, 15].findIndex(function (x) { return x === 10; }), 1);
