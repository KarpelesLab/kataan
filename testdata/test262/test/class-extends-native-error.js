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

// A constructor-less subclass auto-forwards args through the implicit default
// constructor into the native error.
class Plain extends RangeError {}
assert.sameValue(new Plain("p").message, "p", "implicit constructor forwards message");
assert.sameValue(new Plain("p") instanceof RangeError, true, "implicit subclass instanceof");

// Deep chain of constructor-less subclasses still forwards.
class L1 extends Error {}
class L2 extends L1 {}
var deep = new L2("deep");
assert.sameValue(deep.message, "deep", "message through a 2-level constructor-less chain");
assert.sameValue(deep instanceof Error, true, "deep instanceof Error");
assert.sameValue(deep instanceof L2, true, "deep instanceof L2");
