/*---
description: Object.defineProperties and property descriptors
esid: sec-object.defineproperties
---*/
var o = {};
Object.defineProperties(o, {
  x: { value: 1, enumerable: true },
  y: { get: function () { return 2; }, enumerable: true }
});
assert.sameValue(o.x, 1);
assert.sameValue(o.y, 2);
var d = Object.getOwnPropertyDescriptor(o, "x");
assert.sameValue(d.value, 1);
var frozen = Object.freeze({ a: 1 });
frozen.a = 99;
assert.sameValue(frozen.a, 1, "frozen object ignores writes");
assert.sameValue(Object.isFrozen(frozen), true);
