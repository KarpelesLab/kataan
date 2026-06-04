/*---
description: Array and object destructuring with defaults and rest
esid: sec-destructuring-assignment
---*/
var [a, b, ...rest] = [1, 2, 3, 4, 5];
assert.sameValue(a, 1);
assert.sameValue(b, 2);
assert.sameValue(rest.length, 3);
assert.sameValue(rest[0], 3);

var { x, y, z = 10 } = { x: 7, y: 8 };
assert.sameValue(x, 7);
assert.sameValue(y, 8);
assert.sameValue(z, 10, "default applies when key is absent");

var { p: renamed } = { p: 42 };
assert.sameValue(renamed, 42, "rename binding");
