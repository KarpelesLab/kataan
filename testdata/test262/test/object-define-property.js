/*---
description: Object.defineProperty with a value descriptor and getOwnPropertyDescriptor
esid: sec-object.defineproperty
---*/
var o = {};
Object.defineProperty(o, "x", { value: 42 });
assert.sameValue(o.x, 42);
Object.defineProperty(o, "y", { get: function () { return this.x + 1; } });
assert.sameValue(o.y, 43, "accessor descriptor");
var d = Object.getOwnPropertyDescriptor(o, "x");
assert.sameValue(d.value, 42);
