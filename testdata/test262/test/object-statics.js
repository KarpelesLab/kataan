/*---
description: Object static methods comprehensively
esid: sec-properties-of-the-object-constructor
---*/
var o = { a: 1, b: 2, c: 3 };
assert.sameValue(Object.keys(o).join(","), "a,b,c");
assert.sameValue(Object.values(o).join(","), "1,2,3");
assert.sameValue(Object.entries(o).length, 3);
var frozen = Object.freeze({ x: 1 });
assert.sameValue(Object.isFrozen(frozen), true);
var sealed = Object.seal({ y: 2 });
assert.sameValue(Object.isSealed(sealed), true);
var combined = Object.assign({}, { a: 1 }, { b: 2 });
assert.sameValue(combined.a + combined.b, 3);
var fromEntries = Object.fromEntries([["k1", "v1"], ["k2", "v2"]]);
assert.sameValue(fromEntries.k1, "v1");
var created = Object.create({ inherited: true });
assert.sameValue(created.inherited, true);
assert.sameValue(Object.getPrototypeOf(created).inherited, true);
assert.sameValue(Object.is(NaN, NaN), true, "Object.is distinguishes NaN");
assert.sameValue(Object.is(0, -0), false, "Object.is distinguishes signed zero");
assert.sameValue(Object.is(1, 1), true);
var names = Object.getOwnPropertyNames({ a: 1, b: 2 });
assert.sameValue(names.length, 2);
