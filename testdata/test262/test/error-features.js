/*---
description: Error object properties, names, and toString
esid: sec-error-objects
---*/
var e = new Error("base message");
assert.sameValue(e.message, "base message");
assert.sameValue(e.name, "Error");
assert.sameValue(e.toString(), "Error: base message", "toString format");
var te = new TypeError("type issue");
assert.sameValue(te.toString(), "TypeError: type issue");
assert.sameValue(te.name, "TypeError");
assert.sameValue(te instanceof Error, true, "TypeError is an Error");
var emptyErr = new Error();
assert.sameValue(emptyErr.message, "", "no message");
assert.sameValue(emptyErr.toString(), "Error", "no message toString");
var re = new RangeError("range");
assert.sameValue(re.toString(), "RangeError: range");
assert.sameValue(re instanceof RangeError, true);
assert.sameValue(re instanceof Error, true);
var caught;
try { null.x; } catch (err) { caught = err instanceof TypeError; }
assert.sameValue(caught, true, "null property access throws TypeError");
var thrown;
try { throw new SyntaxError("bad syntax"); } catch (err) { thrown = err.name; }
assert.sameValue(thrown, "SyntaxError");
