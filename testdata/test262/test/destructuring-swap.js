/*---
description: Destructuring assignment, swap, nested, and holes
esid: sec-destructuring-assignment
---*/
var a = 1, b = 2;
[a, b] = [b, a];
assert.sameValue(a, 2);
assert.sameValue(b, 1);
var [, second, , fourth] = [10, 20, 30, 40];
assert.sameValue(second, 20);
assert.sameValue(fourth, 40);
var { p: { q } } = { p: { q: 7 } };
assert.sameValue(q, 7);
var [x = 5, y = 6] = [1];
assert.sameValue(x, 1);
assert.sameValue(y, 6);
