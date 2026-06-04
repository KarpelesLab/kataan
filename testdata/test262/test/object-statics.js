/*---
description: Object static methods
esid: sec-properties-of-the-object-constructor
---*/
var o = { a: 1, b: 2, c: 3 };
assert.sameValue(Object.keys(o).length, 3);
assert.sameValue(Object.values(o).join(","), "1,2,3");
assert.sameValue(Object.entries(o).length, 3);
var merged = Object.assign({}, o, { d: 4 });
assert.sameValue(Object.keys(merged).length, 4);
