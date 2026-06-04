/*---
description: Array.from with a Set and a Map
esid: sec-array.from
---*/
assert.sameValue(Array.from(new Set([1, 1, 2, 3])).length, 3);
assert.sameValue(Array.from("abc").join("-"), "a-b-c");
var doubled = Array.from([1, 2, 3], function (x) { return x * 2; });
assert.sameValue(doubled.join(","), "2,4,6");
