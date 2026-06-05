/*---
description: Array() called without new, Array.of, Array.isArray
esid: sec-array-constructor
---*/
assert.sameValue(Array(3).length, 3, "Array(3) is a 3-length array");
assert.sameValue(Array(1, 2, 3).join(","), "1,2,3", "Array with elements");
assert.sameValue(Array().length, 0);
assert.sameValue(Array.of(7).length, 1, "Array.of(7) is [7]");
assert.sameValue(Array.of(1, 2, 3).join(","), "1,2,3");
assert.sameValue(Array.isArray(Array(3)), true);
assert.sameValue(Array.isArray([]), true);
assert.sameValue(Array.isArray("not"), false);
var filled = Array(3).fill(0).map(function (_, i) { return i; });
assert.sameValue(filled.join(","), "0,1,2");
