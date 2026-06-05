/*---
description: Array and object destructuring with default values
esid: sec-destructuring-assignment
---*/
var [a = 1, b = 2, c = 3] = [10, undefined];
assert.sameValue(a, 10);
assert.sameValue(b, 2, "default for undefined");
assert.sameValue(c, 3, "default for missing");
var { x = 5, y = 10, z = 15 } = { x: 1, z: 3 };
assert.sameValue(x, 1);
assert.sameValue(y, 10, "default for missing object property");
assert.sameValue(z, 3);
var [p = 1, [q = 2, r = 3] = []] = [10];
assert.sameValue(p, 10);
assert.sameValue(q, 2, "nested default");
assert.sameValue(r, 3);
function f({ a = 1, b = 2 } = {}) { return a + b; }
assert.sameValue(f(), 3, "param default");
assert.sameValue(f({ a: 10 }), 12);
var { m: { n = 99 } = {} } = {};
assert.sameValue(n, 99, "deep nested default");
var [first = "x", ...rest] = [];
assert.sameValue(first, "x", "default with rest");
assert.sameValue(rest.length, 0);
var { count = 0 } = { count: null };
assert.sameValue(count, null, "null does not trigger default");
