/*---
description: flat and flatMap with mixed element types
esid: sec-array.prototype.flat
---*/
assert.sameValue([1, "a", [2, "b"], [3, [4]]].flat().join(","), "1,a,2,b,3,4");
assert.sameValue([[1], [2, 3], [], [4]].flat().join(","), "1,2,3,4");
assert.sameValue([true, [false, [true]]].flat(Infinity).join(","), "true,false,true");
assert.sameValue([null, [undefined, [1]]].flat(2).length, 3);
assert.sameValue(["a", "b"].flatMap(function (s) { return s.split(""); }).join(","), "a,b");
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, x * 10]; }).join(","), "1,10,2,20,3,30");
assert.sameValue([{ items: [1, 2] }, { items: [3] }].flatMap(function (o) { return o.items; }).join(","), "1,2,3");
assert.sameValue([1, 2].flatMap(function (x) { return x % 2 ? [x] : []; }).join(","), "1");
assert.sameValue(["hello world", "foo bar"].flatMap(function (s) { return s.split(" "); }).join(","), "hello,world,foo,bar");
assert.sameValue([[1, 2], 3, [4, [5]]].flat().length, 5, "flat depth 1 leaves [5]");
assert.sameValue([].flatMap(function (x) { return [x]; }).length, 0);
