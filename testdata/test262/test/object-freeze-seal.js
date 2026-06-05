/*---
description: Object.freeze, isFrozen, and write/delete prevention
esid: sec-object.freeze
---*/
var o = { a: 1, b: 2 };
Object.freeze(o);
assert.sameValue(Object.isFrozen(o), true);
o.a = 99;
assert.sameValue(o.a, 1, "frozen ignores writes");
o.c = 3;
assert.sameValue(o.c, undefined, "frozen ignores new props");
delete o.b;
assert.sameValue(o.b, 2, "frozen ignores deletes");
var nested = { inner: { x: 1 } };
Object.freeze(nested);
nested.inner.x = 5;
assert.sameValue(nested.inner.x, 5, "freeze is shallow");
