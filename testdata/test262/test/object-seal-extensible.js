/*---
description: Object.seal, preventExtensions, isSealed, isExtensible
esid: sec-object.seal
---*/
var o = { a: 1 };
Object.preventExtensions(o);
o.b = 2;
assert.sameValue(o.b, undefined, "no new props after preventExtensions");
assert.sameValue(Object.isExtensible(o), false);
o.a = 9;
assert.sameValue(o.a, 9, "existing props still writable");
var s = { x: 1 };
Object.seal(s);
s.y = 2;
assert.sameValue(s.y, undefined, "sealed: no new props");
s.x = 5;
assert.sameValue(s.x, 5, "sealed: existing still writable");
assert.sameValue(Object.isSealed(s), true);
delete s.x;
assert.sameValue(s.x, 5, "sealed: cannot delete");
