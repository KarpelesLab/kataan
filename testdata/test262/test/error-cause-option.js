/*---
description: the ES2022 Error cause option installs a non-enumerable cause property
esid: sec-error-message
features: [error-cause]
---*/
// new Error(message, { cause }) records the cause.
assert.sameValue(new Error("outer", { cause: "root" }).cause, "root", "string cause");
assert.sameValue(new Error("outer", { cause: new Error("inner") }).cause.message, "inner", "object cause");

// cause is non-enumerable (like name/message).
var e = new Error("x", { cause: "c" });
assert.sameValue(Object.keys(e).length, 0, "cause not enumerable");
assert.sameValue(e.propertyIsEnumerable("cause"), false, "propertyIsEnumerable false");

// Present even when undefined; absent when no options / no cause key.
assert.sameValue("cause" in new Error("x", { cause: undefined }), true, "explicit undefined cause exists");
assert.sameValue("cause" in new Error("x"), false, "no options -> no cause");
assert.sameValue("cause" in new Error("x", { other: 1 }), false, "options without cause -> no cause");

// Works for the derived error constructors.
assert.sameValue(new TypeError("t", { cause: 42 }).cause, 42, "TypeError cause");
assert.sameValue(new RangeError("r", { cause: "rc" }).cause, "rc", "RangeError cause");
assert.sameValue(new AggregateError([], "m", { cause: "ac" }).cause, "ac", "AggregateError cause");

// A subclass forwarding options through super gets the cause too.
class MyErr extends Error {
  constructor(m, o) { super(m, o); this.name = "MyErr"; }
}
var sub = new MyErr("x", { cause: "sc" });
assert.sameValue(sub.cause, "sc", "subclass cause via super");
assert.sameValue(Object.keys(sub).length, 1, "only the explicit name is enumerable");

// A nested cause chain is navigable.
var chain = new Error("a", { cause: new Error("b", { cause: new Error("c") }) });
assert.sameValue(chain.cause.cause.message, "c", "nested cause chain");
