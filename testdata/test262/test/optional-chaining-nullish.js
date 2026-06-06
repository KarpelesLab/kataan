/*---
description: Optional chaining (?.) and nullish coalescing (??) short-circuiting
features: [optional-chaining, logical-assignment-operators]
---*/
var o = { a: { b: 42 } };
assert.sameValue(o?.a?.b, 42, "chained optional access reaches the value");
assert.sameValue(o?.x?.y, undefined, "a nullish link short-circuits to undefined");

// `??` falls back only on null/undefined (not on 0 or "").
assert.sameValue(o?.x?.y ?? "def", "def", "nullish falls back on undefined");
assert.sameValue(o.a.b ?? 99, 42, "nullish keeps a present value");
assert.sameValue(0 ?? 7, 0, "?? does not fall back on 0");
assert.sameValue("" ?? "x", "", "?? does not fall back on empty string");
assert.sameValue(false ?? true, false, "?? does not fall back on false");

var n = null;
assert.sameValue(n?.foo, undefined, "optional access on null is undefined");
assert.sameValue(n ?? "fallback", "fallback", "?? falls back on null");
var u;
assert.sameValue(u?.foo, undefined, "optional access on undefined is undefined");
assert.sameValue(u ?? "fb", "fb", "?? falls back on undefined");

// Optional call: only invoked when the callee is present.
var api = { ping() { return "pong"; } };
assert.sameValue(api.ping?.(), "pong", "optional call invokes a present method");
assert.sameValue(api.missing?.(), undefined, "optional call on a missing method is undefined");
