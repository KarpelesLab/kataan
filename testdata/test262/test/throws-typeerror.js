/*---
description: Reading a property of null throws a TypeError
esid: sec-property-accessors
---*/
assert.throws(TypeError, function () {
  var x = null;
  return x.field;
}, "property access on null");
