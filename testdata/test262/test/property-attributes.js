/*---
description: Property attribute enforcement and descriptor round-tripping
esid: sec-object.getownpropertydescriptor
---*/
var o = {};
Object.defineProperty(o, "ro", { value: 1, writable: false });
o.ro = 99;
assert.sameValue(o.ro, 1, "non-writable ignores assignment");
Object.defineProperty(o, "rw", { value: 2, writable: true });
o.rw = 99;
assert.sameValue(o.rw, 99, "writable accepts assignment");
var d1 = Object.getOwnPropertyDescriptor(o, "ro");
assert.sameValue(d1.writable, false);
assert.sameValue(d1.configurable, false, "defineProperty defaults non-configurable");
assert.sameValue(d1.enumerable, false, "defineProperty defaults non-enumerable");
assert.sameValue(d1.value, 1);
var plain = { a: 5 };
var d2 = Object.getOwnPropertyDescriptor(plain, "a");
assert.sameValue(d2.writable, true, "literal property is writable");
assert.sameValue(d2.enumerable, true);
assert.sameValue(d2.configurable, true, "literal property is configurable");
assert.sameValue(d2.value, 5);
var frozen = Object.freeze({ x: 1 });
var d3 = Object.getOwnPropertyDescriptor(frozen, "x");
assert.sameValue(d3.writable, false, "frozen is non-writable");
assert.sameValue(d3.configurable, false, "frozen is non-configurable");
assert.sameValue(Object.isFrozen(frozen), true);
assert.sameValue(Object.isFrozen({}), false);
assert.sameValue(Object.isSealed(Object.seal({})), true);
var ext = {};
Object.preventExtensions(ext);
ext.y = 1;
assert.sameValue(ext.y, undefined, "non-extensible rejects new properties");
assert.sameValue(Object.isExtensible(ext), false);
assert.sameValue(Object.isExtensible({}), true);
var conf = {};
Object.defineProperty(conf, "c", { value: 1, configurable: true });
assert.sameValue(Object.getOwnPropertyDescriptor(conf, "c").configurable, true);
