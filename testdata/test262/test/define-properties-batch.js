/*---
description: Object.defineProperties with multiple descriptors
esid: sec-object.defineproperties
---*/
var o = {};
Object.defineProperties(o, {
  a: { value: 1, enumerable: true, writable: true },
  b: { value: 2, enumerable: true },
  c: { get: function () { return this.a + this.b; }, enumerable: true },
  hidden: { value: "secret", enumerable: false }
});
assert.sameValue(o.a, 1);
assert.sameValue(o.b, 2);
assert.sameValue(o.c, 3, "computed getter");
assert.sameValue(o.hidden, "secret");
assert.sameValue(Object.keys(o).join(","), "a,b,c", "enumerable only");
o.a = 10;
assert.sameValue(o.c, 12, "getter reflects change");
var point = {};
Object.defineProperties(point, {
  x: { value: 0, writable: true, enumerable: true },
  y: { value: 0, writable: true, enumerable: true }
});
point.x = 3;
point.y = 4;
assert.sameValue(point.x * point.x + point.y * point.y, 25);
var descs = Object.getOwnPropertyDescriptors(o);
assert.sameValue(descs.a.value, 10, "current value after mutation");
assert.sameValue(descs.c.get !== undefined, true, "has getter descriptor");
