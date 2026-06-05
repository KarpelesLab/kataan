/*---
description: Error.prototype.stack and the AggregateError constructor
esid: sec-error-objects
---*/
var e = new Error("boom");
assert.sameValue(typeof e.stack, "string", "errors have a string stack");
assert.sameValue(e.stack.indexOf("Error") >= 0, true, "stack mentions the error");
assert.sameValue(e.stack.indexOf("boom") >= 0, true, "stack mentions the message");
var te = new TypeError("bad");
assert.sameValue(te.stack.indexOf("TypeError") >= 0, true, "typed error stack");
assert.sameValue(Object.keys(e).indexOf("stack"), -1, "stack is non-enumerable");
var agg = new AggregateError([new Error("a"), new TypeError("b")], "multiple errors");
assert.sameValue(agg.message, "multiple errors", "AggregateError message is the second argument");
assert.sameValue(agg.name, "AggregateError");
assert.sameValue(agg.errors.length, 2, "errors array");
assert.sameValue(agg.errors[0].message, "a");
assert.sameValue(agg.errors[1] instanceof TypeError, true);
assert.sameValue(agg instanceof Error, true, "AggregateError is an Error");
assert.sameValue(agg instanceof AggregateError, true);
var withCause = new AggregateError([], "msg", { cause: 42 });
assert.sameValue(withCause.cause, 42, "AggregateError cause option (third arg)");
var fromSet = new AggregateError(new Set([new Error("x")]), "set");
assert.sameValue(fromSet.errors.length, 1, "errors from any iterable");
