/*---
description: Object.create with full property descriptors
esid: sec-object.create
---*/
var proto = { type: "base" };
var obj = Object.create(proto, {
  name: { value: "test", writable: true, enumerable: true, configurable: true },
  id: { value: 42, enumerable: false },
  computed: { get: function () { return this.id * 2; }, enumerable: true }
});
assert.sameValue(obj.name, "test");
assert.sameValue(obj.id, 42);
assert.sameValue(obj.computed, 84, "getter in descriptor");
assert.sameValue(obj.type, "base", "inherited from proto");
assert.sameValue(Object.keys(obj).join(","), "name,computed", "enumerable only");
assert.sameValue(Object.getPrototypeOf(obj), proto);
var bare = Object.create(null, { x: { value: 1, enumerable: true } });
assert.sameValue(bare.x, 1);
assert.sameValue(Object.getPrototypeOf(bare), null);
obj.name = "changed";
assert.sameValue(obj.name, "changed", "writable");
var readonly = Object.create({}, { c: { value: "const", writable: false } });
readonly.c = "new";
assert.sameValue(readonly.c, "const", "non-writable");
