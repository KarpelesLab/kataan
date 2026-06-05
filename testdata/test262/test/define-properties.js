/*---
description: Object.defineProperties with multiple descriptors
esid: sec-object.defineproperties
---*/
var o = {};
Object.defineProperties(o, {
  a: { value: 1, enumerable: true, writable: true },
  b: { value: 2, enumerable: false },
  c: { get: function () { return this.a + this.b; }, enumerable: true }
});
assert.sameValue(o.a, 1);
assert.sameValue(o.b, 2);
assert.sameValue(o.c, 3, "computed getter");
assert.sameValue(Object.keys(o).join(","), "a,c", "only enumerable keys");
o.a = 10;
assert.sameValue(o.c, 12, "getter reflects update");
var point = {};
Object.defineProperties(point, {
  x: { value: 3, writable: false, enumerable: true },
  y: { value: 4, writable: false, enumerable: true }
});
point.x = 99;
assert.sameValue(point.x, 3, "non-writable");
assert.sameValue(point.y, 4);
