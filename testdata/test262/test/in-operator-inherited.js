/*---
description: the `in` operator checks own and inherited properties
esid: sec-relational-operators
---*/
assert.sameValue("a" in { a: 1 }, true, "own property");
assert.sameValue("z" in { a: 1 }, false, "missing property");
var proto = { inherited: 1 };
var obj = Object.create(proto);
obj.own = 2;
assert.sameValue("own" in obj, true, "own");
assert.sameValue("inherited" in obj, true, "inherited via prototype");
assert.sameValue("missing" in obj, false);
var grandparent = { deep: 1 };
var parent = Object.create(grandparent);
var child = Object.create(parent);
assert.sameValue("deep" in child, true, "inherited through a multi-level chain");
var shadow = Object.create({ v: 1 });
shadow.v = 2;
assert.sameValue("v" in shadow, true, "shadowed key still present");
assert.sameValue(0 in [10, 20], true, "array index in bounds");
assert.sameValue(5 in [10, 20], false, "array index out of bounds");
assert.sameValue("length" in [], true, "array length");
assert.sameValue(obj.hasOwnProperty("own"), true, "hasOwnProperty: own");
assert.sameValue(obj.hasOwnProperty("inherited"), false, "hasOwnProperty excludes inherited");
