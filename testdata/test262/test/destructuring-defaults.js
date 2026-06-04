/*---
description: Array/object destructuring with defaults, nesting, and rest
esid: sec-destructuring-binding-patterns
---*/
var [a, b = 10, ...rest] = [1, undefined, 3, 4];
assert.sameValue(a, 1);
assert.sameValue(b, 10, "default fills undefined");
assert.sameValue(rest.join(","), "3,4");

var { x, y = 5, z: { w } } = { x: 1, z: { w: 9 } };
assert.sameValue(x, 1);
assert.sameValue(y, 5);
assert.sameValue(w, 9, "nested destructuring");

function f({ p = 1, q = 2 } = {}) { return p + q; }
assert.sameValue(f(), 3, "defaulted parameter object");
assert.sameValue(f({ p: 10 }), 12);

var [m, n] = [n, m] = [1, 2];
assert.sameValue(m, 1);
