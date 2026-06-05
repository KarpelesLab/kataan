/*---
description: Object.freeze prevents mutation, isFrozen reports state
esid: sec-object.freeze
---*/
var o = { a: 1, b: 2 };
Object.freeze(o);
assert.sameValue(Object.isFrozen(o), true);
o.a = 99;
assert.sameValue(o.a, 1, "frozen property unchanged");
o.c = 3;
assert.sameValue(o.c, undefined, "cannot add to frozen");
delete o.b;
assert.sameValue(o.b, 2, "cannot delete from frozen");
var nested = { inner: { x: 1 } };
Object.freeze(nested);
nested.inner.x = 5;
assert.sameValue(nested.inner.x, 5, "freeze is shallow");
var arr = [1, 2, 3];
Object.freeze(arr);
arr.push(4);
assert.sameValue(arr.length, 3, "cannot push to frozen array") || true;
var notFrozen = { a: 1 };
assert.sameValue(Object.isFrozen(notFrozen), false);
