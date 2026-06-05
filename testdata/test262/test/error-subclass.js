/*---
description: Classes extending native Error constructors
esid: sec-native-error-types-used-in-this-standard
---*/
class CustomError extends Error {
  constructor(msg, code) { super(msg); this.name = "CustomError"; this.code = code; }
}
var ce = new CustomError("custom message", 42);
assert.sameValue(ce.message, "custom message", "super sets message");
assert.sameValue(ce.code, 42, "subclass field");
assert.sameValue(ce.name, "CustomError", "overridden name");
assert.sameValue(ce instanceof CustomError, true);
assert.sameValue(ce instanceof Error, true, "subclass is an Error");
assert.sameValue(ce instanceof TypeError, false, "not a TypeError");
class ValidationError extends RangeError {
  constructor(field) { super("invalid " + field); this.field = field; }
}
var ve = new ValidationError("age");
assert.sameValue(ve.message, "invalid age");
assert.sameValue(ve.field, "age");
assert.sameValue(ve instanceof RangeError, true);
assert.sameValue(ve instanceof Error, true, "RangeError subclass is an Error");
var caught;
try { throw new CustomError("thrown", 1); }
catch (e) { caught = (e instanceof Error) + ":" + e.message + ":" + e.code; }
assert.sameValue(caught, "true:thrown:1", "thrown custom error");
class Base extends Error {}
var b = new Base("base");
assert.sameValue(b instanceof Error, true, "no-constructor subclass");
