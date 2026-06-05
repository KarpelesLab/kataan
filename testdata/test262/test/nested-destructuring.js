/*---
description: Deeply nested destructuring with defaults and renaming
esid: sec-destructuring-binding-patterns
---*/
var { a: { b: { c } } } = { a: { b: { c: 42 } } };
assert.sameValue(c, 42, "three levels deep");
var [[x], [y, z]] = [[1], [2, 3]];
assert.sameValue(x + y + z, 6, "nested arrays");
var { p: { q = 10 } = {} } = {};
assert.sameValue(q, 10, "default for missing nested object");
var { items: [first, ...others] = [] } = { items: [1, 2, 3] };
assert.sameValue(first + ":" + others.join(","), "1:2,3");
var { m: renamed = 5 } = { m: 7 };
assert.sameValue(renamed, 7, "rename with default");
