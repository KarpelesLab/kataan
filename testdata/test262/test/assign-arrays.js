/*---
description: Object.assign with arrays and multiple sources
esid: sec-object.assign
---*/
var target = { a: 1 };
var result = Object.assign(target, { b: 2 }, { c: 3 });
assert.sameValue(result, target, "assign returns target");
assert.sameValue(target.a + target.b + target.c, 6);
var merged = Object.assign({}, { x: 1, y: 2 }, { y: 3, z: 4 });
assert.sameValue(merged.y, 3, "later source overrides");
assert.sameValue(merged.x + merged.z, 5);
var fromArray = Object.assign({}, ["a", "b", "c"]);
assert.sameValue(fromArray[0], "a", "array indices become keys");
assert.sameValue(fromArray[2], "c");
assert.sameValue(Object.keys(fromArray).join(","), "0,1,2");
var clone = Object.assign({}, { nested: { deep: 1 } });
assert.sameValue(clone.nested.deep, 1, "shallow copy shares nested");
var withNull = Object.assign({ a: 1 }, null, undefined, { b: 2 });
assert.sameValue(withNull.a + withNull.b, 3, "null/undefined sources skipped");
