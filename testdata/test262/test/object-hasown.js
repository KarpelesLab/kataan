/*---
description: Object.hasOwn and hasOwnProperty distinctions
esid: sec-object.hasown
---*/
var o = { a: 1, b: undefined };
assert.sameValue(Object.hasOwn(o, "a"), true);
assert.sameValue(Object.hasOwn(o, "b"), true, "undefined value still owned");
assert.sameValue(Object.hasOwn(o, "c"), false);
var proto = { inherited: 1 };
var child = Object.create(proto);
child.own = 2;
assert.sameValue(Object.hasOwn(child, "own"), true);
assert.sameValue(Object.hasOwn(child, "inherited"), false, "inherited is not own");
assert.sameValue(child.inherited, 1, "but still accessible");
assert.sameValue(Object.hasOwn([1, 2, 3], 0), true, "array index");
assert.sameValue(Object.hasOwn([1, 2, 3], 5), false);
