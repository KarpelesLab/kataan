/*---
description: Spread over various iterables into arrays and calls
esid: sec-array-initializer
---*/
assert.sameValue([...new Set([1, 2, 3])].join(","), "1,2,3", "spread a Set");
assert.sameValue([...new Map([["a", 1]])].length, 1, "spread a Map yields pairs");
assert.sameValue([..."hello"].length, 5, "spread a string");
function sum() { var t = 0; for (var i = 0; i < arguments.length; i++) t += arguments[i]; return t; }
assert.sameValue(sum(...[1, 2, 3, 4]), 10, "spread into call");
var combined = [0, ...[1, 2], ...new Set([3, 4]), 5];
assert.sameValue(combined.join(","), "0,1,2,3,4,5");
function* gen() { yield 1; yield 2; yield 3; }
assert.sameValue([...gen()].join(","), "1,2,3", "spread a generator");
assert.sameValue(Math.max(...[3, 1, 4, 1, 5]), 5);
