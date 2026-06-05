/*---
description: Error properties and behavior
esid: sec-error-objects
---*/
var e = new Error("base");
assert.sameValue(e.message, "base");
assert.sameValue(e.name, "Error");
assert.sameValue(typeof e.toString(), "string");
assert.sameValue(e.toString(), "Error: base");
var errors = [new TypeError("t"), new RangeError("r"), new SyntaxError("s")];
assert.sameValue(errors[0].name, "TypeError");
assert.sameValue(errors[1].name, "RangeError");
assert.sameValue(errors[2].name, "SyntaxError");
assert.sameValue(errors.every(function (err) { return err instanceof Error; }), true);
var custom = new Error("custom");
custom.code = "E_CUSTOM";
assert.sameValue(custom.code, "E_CUSTOM", "custom property");
function risky() { throw new Error("risky failed"); }
var caught;
try { risky(); } catch (err) { caught = err.message; }
assert.sameValue(caught, "risky failed");
var rethrown;
try {
  try { throw new TypeError("original"); }
  catch (e1) { throw new Error("wrapped: " + e1.message); }
} catch (e2) { rethrown = e2.message; }
assert.sameValue(rethrown, "wrapped: original");
var withName = new RangeError("out");
assert.sameValue(withName.toString(), "RangeError: out");
