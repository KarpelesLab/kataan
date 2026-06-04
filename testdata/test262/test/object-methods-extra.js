/*---
description: Object entries, fromEntries, assign, and values
esid: sec-object.entries
---*/
var o = { a: 1, b: 2 };
assert.sameValue(Object.entries(o).map(function (e) { return e[0] + e[1]; }).join(","), "a1,b2");
assert.sameValue(Object.values(o).join(","), "1,2");
var back = Object.fromEntries([["x", 10], ["y", 20]]);
assert.sameValue(back.x, 10);
assert.sameValue(back.y, 20);
var merged = Object.assign({}, { a: 1 }, { b: 2, a: 9 });
assert.sameValue(merged.a, 9);
assert.sameValue(merged.b, 2);
assert.sameValue(Object.keys(o).length, 2);
