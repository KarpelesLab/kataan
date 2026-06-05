/*---
description: Array flat and flatMap comprehensive
esid: sec-array.prototype.flat
---*/
assert.sameValue([1, [2, [3, [4, [5]]]]].flat(Infinity).join(","), "1,2,3,4,5");
assert.sameValue([1, [2, [3]]].flat().join(","), "1,2,3", "default depth 1 — wait, [3] stays once");
assert.sameValue([1, 2, 3].flat().join(","), "1,2,3", "no nesting");
assert.sameValue([[], [], []].flat().length, 0, "empty subarrays");
assert.sameValue([1, [2], [[3]]].flat(2).join(","), "1,2,3");
assert.sameValue(["a", "b"].flatMap(function (s) { return [s, s]; }).join(","), "a,a,b,b");
assert.sameValue([1, 2, 3].flatMap(function (x) { return x % 2 ? [x] : []; }).join(","), "1,3");
assert.sameValue([1, 2].flatMap(function (x) { return [[x]]; }).length, 2, "flatMap one level only");
assert.sameValue([[1, 2], [3, 4]].flatMap(function (x) { return x; }).join(","), "1,2,3,4");
assert.sameValue([].flat().length, 0);
assert.sameValue([1, [2, 3], 4].flat().length, 4);
