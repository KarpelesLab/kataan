/*---
description: Object.assign, keys/values/entries ordering, and property copying
esid: sec-object.assign
---*/
var target = { a: 1 };
var result = Object.assign(target, { b: 2 }, { c: 3, a: 10 });
assert.sameValue(result, target, "assign returns the target");
assert.sameValue(target.a, 10, "later sources override");
assert.sameValue(target.b + target.c, 5);
var o = { x: 1, y: 2, z: 3 };
assert.sameValue(Object.keys(o).join(","), "x,y,z", "insertion order");
assert.sameValue(Object.values(o).join(","), "1,2,3");
assert.sameValue(Object.entries(o).map(function (e) { return e.join(":"); }).join(","), "x:1,y:2,z:3");
var copy = Object.assign({}, o);
copy.x = 99;
assert.sameValue(o.x, 1, "assign makes a shallow copy");
