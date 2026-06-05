/*---
description: Array.from over strings, sets, maps, and iterables with mapFn
esid: sec-array.from
---*/
assert.sameValue(Array.from("abc").join(","), "a,b,c", "from a string");
assert.sameValue(Array.from(new Set([1, 2, 2, 3])).join(","), "1,2,3", "from a Set");
assert.sameValue(Array.from([1, 2, 3], function (x) { return x * x; }).join(","), "1,4,9", "with mapFn");
var m = new Map([["a", 1], ["b", 2]]);
assert.sameValue(Array.from(m.keys()).join(","), "a,b");
assert.sameValue(Array.from(m.values()).join(","), "1,2");
assert.sameValue(Array.of(1, 2, 3).join(","), "1,2,3");
assert.sameValue(Array.of(7).length, 1, "Array.of(7) is [7] not empty*7");
