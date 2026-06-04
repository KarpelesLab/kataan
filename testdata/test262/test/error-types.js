/*---
description: Error constructors, message, name, and rethrow
esid: sec-error-objects
---*/
var e = new Error("boom");
assert.sameValue(e.message, "boom");
assert.sameValue(e.name, "Error");
var t = new TypeError("bad type");
assert.sameValue(t.message, "bad type");
assert.sameValue(t.name, "TypeError");
assert.sameValue(t instanceof TypeError, true);
assert.sameValue(t instanceof Error, true, "TypeError is an Error");

var caught = "";
try {
  try { throw new RangeError("range"); }
  catch (inner) { throw inner; }
} catch (outer) {
  caught = outer.name + ":" + outer.message;
}
assert.sameValue(caught, "RangeError:range");
