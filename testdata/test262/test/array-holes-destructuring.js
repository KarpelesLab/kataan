/*---
description: Array destructuring with holes, defaults, and swapping
esid: sec-destructuring-binding-patterns
---*/
var [, second, , fourth] = [1, 2, 3, 4];
assert.sameValue(second, 2);
assert.sameValue(fourth, 4);
var [a = 1, b = 2, c = 3] = [10, undefined];
assert.sameValue(a, 10);
assert.sameValue(b, 2, "default fills undefined");
assert.sameValue(c, 3, "default fills missing");
var [x, y] = [1, 2];
[x, y] = [y, x];
assert.sameValue(x + "," + y, "2,1");
var [[p], [q, r]] = [[1], [2, 3]];
assert.sameValue(p + q + r, 6);
var [first, ...tail] = [1, 2, 3, 4];
assert.sameValue(tail.length, 3);
