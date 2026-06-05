/*---
description: Error cause option (ES2022) and generator yield* delegation
esid: sec-error-message
---*/
var e = new Error("failure", { cause: "root reason" });
assert.sameValue(e.cause, "root reason", "Error cause option");
assert.sameValue(e.message, "failure", "message preserved");
var te = new TypeError("bad", { cause: 42 });
assert.sameValue(te.cause, 42, "TypeError cause");
var plain = new Error("no opts");
assert.sameValue(plain.cause, undefined, "no cause when options omitted");
var noCause = new Error("msg", {});
assert.sameValue(noCause.cause, undefined, "no cause key");
var inner = new Error("inner");
var outer = new Error("outer", { cause: inner });
assert.sameValue(outer.cause, inner, "cause can be an Error");
assert.sameValue(outer.cause.message, "inner", "chained cause message");
var caught;
try {
  try { throw new RangeError("low-level"); }
  catch (err) { throw new Error("high-level", { cause: err }); }
} catch (e2) { caught = e2.cause.message; }
assert.sameValue(caught, "low-level", "rethrow with cause");
function* inner2() { yield 1; yield 2; }
function* outer2() { yield* inner2(); yield 3; }
assert.sameValue([...outer2()].join(","), "1,2,3", "yield* delegates to a generator");
function* g() { yield* [1, 2]; yield* [3, 4]; }
assert.sameValue([...g()].join(","), "1,2,3,4", "yield* delegates to an array");
