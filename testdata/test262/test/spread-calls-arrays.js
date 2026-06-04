/*---
description: Spread in array literals and function calls
esid: sec-array-initializer
---*/
var parts = [2, 3];
assert.sameValue([1, ...parts, 4].join(","), "1,2,3,4");
function sum3(a, b, c) { return a + b + c; }
assert.sameValue(sum3(...[1, 2, 3]), 6, "spread into call");
assert.sameValue(Math.max(...[5, 2, 9, 1]), 9);
var merged = [...[1, 2], ...[3, 4]];
assert.sameValue(merged.length, 4);
assert.sameValue([..."abc"].join("-"), "a-b-c", "spread a string");
