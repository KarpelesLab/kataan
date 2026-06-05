/*---
description: Object shorthand properties, methods, and computed names
esid: sec-object-initializer
---*/
var x = 1, y = 2;
var point = { x, y };
assert.sameValue(point.x + point.y, 3, "shorthand properties");
var obj = {
  value: 10,
  getValue() { return this.value; },
  double() { return this.value * 2; }
};
assert.sameValue(obj.getValue(), 10, "method shorthand");
assert.sameValue(obj.double(), 20);
var key = "dynamic";
var computed = { [key]: 42, [`${key}2`]: 43 };
assert.sameValue(computed.dynamic, 42);
assert.sameValue(computed.dynamic2, 43);
var n = 1;
var nested = { a: { b: { c: n } } };
assert.sameValue(nested.a.b.c, 1);
