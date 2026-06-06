/*---
description: Array and object destructuring with defaults, rest, and renaming
esid: sec-destructuring-assignment
---*/
// Array: default applies to `undefined`, rest gathers the tail.
var [a, b = 9, ...rest] = [1, undefined, 3, 4];
assert.sameValue(a, 1, "first element");
assert.sameValue(b, 9, "default fills an undefined hole");
assert.sameValue(rest.length, 2, "rest length");
assert.sameValue(rest[0], 3, "rest[0]");
assert.sameValue(rest[1], 4, "rest[1]");

// Object: rename and default.
var { x: px, y: py = 5, z: pz = 7 } = { x: 10, y: 20 };
assert.sameValue(px, 10, "renamed x");
assert.sameValue(py, 20, "present y keeps its value");
assert.sameValue(pz, 7, "missing z takes the default");

// Nested destructuring.
var { p: { q } } = { p: { q: 42 } };
assert.sameValue(q, 42, "nested destructuring");
