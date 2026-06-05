/*---
description: Object.create with property descriptors and prototype
esid: sec-object.create
---*/
var proto = { greet: function () { return "hi from " + this.name; } };
var obj = Object.create(proto, {
  name: { value: "test", enumerable: true },
  id: { value: 42, enumerable: false }
});
assert.sameValue(obj.name, "test");
assert.sameValue(obj.id, 42);
assert.sameValue(obj.greet(), "hi from test", "inherited method");
assert.sameValue(Object.keys(obj).join(","), "name", "only enumerable in keys");
assert.sameValue(Object.getPrototypeOf(obj), proto);
var bare = Object.create(null);
assert.sameValue(Object.getPrototypeOf(bare), null);
