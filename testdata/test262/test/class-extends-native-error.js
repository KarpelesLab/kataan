/*---
description: a user class with an explicit super() extending a native Error works
features: [class]
---*/
class MyError extends Error {
  constructor(m) { super(m); this.name = "MyError"; }
}
var e = new MyError("oops");
assert.sameValue(e.name, "MyError", "own name");
assert.sameValue(e.message, "oops", "message via super()");
assert.sameValue(e instanceof Error, true, "instanceof Error");
assert.sameValue(e instanceof MyError, true, "instanceof MyError");
assert.sameValue(e.toString(), "MyError: oops", "Error toString");

// Catchable as an Error.
var caught = null;
try { throw new MyError("boom"); } catch (x) { caught = x; }
assert.sameValue(caught instanceof Error, true, "thrown subclass is an Error");
assert.sameValue(caught.message, "boom", "thrown message");

// A typed-error subclass with an explicit super() forwards the message too.
class MyRange extends RangeError {
  constructor(m) { super(m); }
}
assert.sameValue(new MyRange("r") instanceof RangeError, true, "extends RangeError");
assert.sameValue(new MyRange("r").message, "r", "RangeError subclass message");
