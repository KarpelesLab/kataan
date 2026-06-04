/*---
description: Rest parameters and argument spreading interplay
esid: sec-function-definitions
---*/
function variadic(first, ...rest) { return first + ":" + rest.length + ":" + rest.join(","); }
assert.sameValue(variadic(1, 2, 3, 4), "1:3:2,3,4");
assert.sameValue(variadic(1), "1:0:");
function sum(...nums) { return nums.reduce(function (a, b) { return a + b; }, 0); }
assert.sameValue(sum(1, 2, 3, 4, 5), 15);
assert.sameValue(sum(...[10, 20, 30]), 60, "spread into rest");
var args = [1, 2, 3];
assert.sameValue(sum(0, ...args, 4), 10, "mixed spread");
