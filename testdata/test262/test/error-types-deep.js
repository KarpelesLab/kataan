/*---
description: Error types, instanceof, custom errors with prototype chains
esid: sec-error-objects
---*/
var e = new RangeError("out of range");
assert.sameValue(e instanceof RangeError, true);
assert.sameValue(e instanceof Error, true, "RangeError is an Error");
assert.sameValue(e.name, "RangeError");
assert.sameValue(e.message, "out of range");
function tryIt(fn) {
  try { fn(); return "no error"; }
  catch (err) {
    if (err instanceof TypeError) return "type";
    if (err instanceof RangeError) return "range";
    return "other";
  }
}
assert.sameValue(tryIt(function () { throw new TypeError("x"); }), "type");
assert.sameValue(tryIt(function () { throw new RangeError("x"); }), "range");
assert.sameValue(tryIt(function () { throw new Error("x"); }), "other");
assert.sameValue(tryIt(function () { return 1; }), "no error");
var caught;
try { undefined.x; } catch (e2) { caught = e2 instanceof TypeError; }
assert.sameValue(caught, true, "property access on undefined throws TypeError");
