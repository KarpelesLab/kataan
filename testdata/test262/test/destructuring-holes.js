/*---
description: Array destructuring with holes and nested patterns
esid: sec-destructuring-binding-patterns
---*/
var [, second, , fourth] = [1, 2, 3, 4];
assert.sameValue(second, 2, "skip with holes");
assert.sameValue(fourth, 4);
var [a, [b, c], d] = [1, [2, 3], 4];
assert.sameValue(a, 1);
assert.sameValue(b, 2, "nested destructuring");
assert.sameValue(c, 3);
assert.sameValue(d, 4);
var [first, ...others] = [10, 20, 30, 40];
assert.sameValue(first, 10);
assert.sameValue(others.length, 3);
var { x: { y } } = { x: { y: 42 } };
assert.sameValue(y, 42, "nested object destructuring");
var [[p], [q]] = [[1], [2]];
assert.sameValue(p + q, 3);
var { a: arr } = { a: [1, 2, 3] };
assert.sameValue(arr.length, 3, "destructure array value");
