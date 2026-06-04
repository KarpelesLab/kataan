/*---
description: Object.freeze prevents mutation; isFrozen and getOwnPropertyNames
esid: sec-object.freeze
---*/
var o = Object.freeze({ a: 1, b: 2 });
o.a = 99;
o.c = 3;
assert.sameValue(o.a, 1, "frozen property is not mutated");
assert.sameValue(o.c, undefined, "no new property on a frozen object");
assert.sameValue(Object.isFrozen(o), true);
assert.sameValue(Object.isFrozen({}), false);
assert.sameValue(Object.getOwnPropertyNames(o).length, 2);
