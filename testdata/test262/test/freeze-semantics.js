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
// Per spec, push finishes with Set(O,"length",…,Throw=true); a frozen array's
// length is non-writable, so the push throws a TypeError (it does not silently
// no-op).
assert.throws(TypeError, function () { arr.push(4); }, "push to frozen array throws");
assert.sameValue(arr.length, 3, "frozen array length unchanged");
var notFrozen = { a: 1 };
assert.sameValue(Object.isFrozen(notFrozen), false);
