/*---
description: Error name/message are non-enumerable; toString handles empty name/message
esid: sec-error.prototype.tostring
---*/
// name and message are non-enumerable own/inherited properties.
assert.sameValue(Object.keys(new TypeError("t")).length, 0, "TypeError has no enumerable keys");
assert.sameValue(Object.keys(new Error("m")).length, 0, "Error has no enumerable keys");
assert.sameValue(JSON.stringify(new Error("m")), "{}", "error stringifies to {}");
// But they are readable.
assert.sameValue(new TypeError("t").name, "TypeError", "name readable");
assert.sameValue(new TypeError("t").message, "t", "message readable");

// toString: empty name -> message; empty message -> name; else "name: message".
var e = new Error("m"); e.name = "";
assert.sameValue(e.toString(), "m", "empty name -> message only");
assert.sameValue(new Error().toString(), "Error", "empty message -> name only");
assert.sameValue(new TypeError("t").toString(), "TypeError: t", "name: message");
assert.sameValue(new RangeError("r").toString(), "RangeError: r", "RangeError toString");

// A subclass: own assigned props stay enumerable; inherited error props do not.
class AppError extends Error {
  constructor(m) { super(m); this.name = "AppError"; this.code = 5; }
}
var a = new AppError("boom");
assert.sameValue(Object.keys(a).sort().join(","), "code", "only the user prop enumerates");
assert.sameValue(a.toString(), "AppError: boom", "subclass toString");
assert.sameValue(a instanceof Error, true, "subclass instanceof Error");

// An engine-thrown error has the same shape.
var caught;
try { null.x; } catch (err) { caught = err; }
assert.sameValue(Object.keys(caught).length, 0, "thrown TypeError no enumerable keys");
assert.sameValue(caught.name, "TypeError", "thrown error name");
