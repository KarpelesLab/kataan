/*---
description: Object.seal and preventExtensions semantics
esid: sec-object.seal
---*/
var o = { a: 1, b: 2 };
Object.seal(o);
assert.sameValue(Object.isSealed(o), true);
o.c = 3;
assert.sameValue(o.c, undefined, "sealed: cannot add");
o.a = 10;
assert.sameValue(o.a, 10, "sealed: can still modify existing");
delete o.b;
assert.sameValue(o.b, 2, "sealed: cannot delete");
var p = { x: 1 };
Object.preventExtensions(p);
assert.sameValue(Object.isExtensible(p), false);
p.y = 2;
assert.sameValue(p.y, undefined, "non-extensible: cannot add");
p.x = 5;
assert.sameValue(p.x, 5, "non-extensible: can modify");
delete p.x;
assert.sameValue(p.x, undefined, "non-extensible: can delete");
var fresh = {};
assert.sameValue(Object.isExtensible(fresh), true);
assert.sameValue(Object.isSealed(fresh), false);
