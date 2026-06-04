/*---
description: Reflect.get/set/has/ownKeys/apply
esid: sec-reflect.get
---*/
var o = { a: 1, b: 2 };
assert.sameValue(Reflect.get(o, "a"), 1);
Reflect.set(o, "c", 3);
assert.sameValue(o.c, 3);
assert.sameValue(Reflect.has(o, "b"), true);
assert.sameValue(Reflect.has(o, "z"), false);
assert.sameValue(Reflect.ownKeys(o).length, 3);
function add(a, b) { return a + b + this.base; }
assert.sameValue(Reflect.apply(add, { base: 10 }, [1, 2]), 13);
