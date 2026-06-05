/*---
description: Spread in function call arguments
esid: sec-function-calls
---*/
function sum(a, b, c) { return a + b + c; }
assert.sameValue(sum(...[1, 2, 3]), 6, "spread array into args");
assert.sameValue(sum(1, ...[2, 3]), 6, "partial spread");
assert.sameValue(sum(...[1], ...[2], ...[3]), 6, "multiple spreads");
assert.sameValue(Math.max(...[5, 2, 8, 1]), 8, "spread into Math.max");
assert.sameValue(Math.min(...[5, 2, 8, 1]), 1);
function collect(...args) { return args.length; }
assert.sameValue(collect(...[1, 2], ...[3, 4, 5]), 5);
var arr = [1, 2, 3];
var combined = [0, ...arr, 4];
assert.sameValue(combined.join(","), "0,1,2,3,4");
assert.sameValue([...new Set([1, 2, 3])].length, 3, "spread Set into array");
assert.sameValue(sum(...[1, 2, 3, 4]), 6, "extra args ignored");
