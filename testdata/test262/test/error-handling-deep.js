/*---
description: Error properties, custom errors, finally ordering, nested try
esid: sec-error-objects
---*/
function CustomError(msg) { this.message = msg; this.name = "CustomError"; }
CustomError.prototype = Object.create(Error.prototype);
var e = new CustomError("boom");
assert.sameValue(e.message, "boom");
assert.sameValue(e.name, "CustomError");

var order = [];
function f() {
  try {
    try { order.push("inner-try"); throw new Error("x"); }
    finally { order.push("inner-finally"); }
  } catch (err) { order.push("outer-catch"); }
  finally { order.push("outer-finally"); }
}
f();
assert.sameValue(order.join(","), "inner-try,inner-finally,outer-catch,outer-finally");

var caught;
try { null.x; } catch (e) { caught = e instanceof TypeError; }
assert.sameValue(caught, true, "property access on null is a TypeError");
