/*---
description: Computed property names in object literals
esid: sec-object-initializer
---*/
var key = "dynamic";
var obj = { [key]: 1, ["a" + "b"]: 2 };
assert.sameValue(obj.dynamic, 1);
assert.sameValue(obj.ab, 2);
