/*---
description: Optional chaining (?.) and nullish coalescing (??) short-circuiting
features: [optional-chaining, logical-assignment-operators]
---*/
var o = { a: { b: 42 } };
assert.sameValue(o?.a?.b, 42, "chained optional access reaches the value");
assert.sameValue(o?.x?.y, undefined, "a nullish link short-circuits to undefined");

// `??` falls back only on null/undefined (not on 0 or "" or false).
assert.sameValue(o?.x?.y ?? "def", "def", "nullish falls back on undefined");
assert.sameValue(o.a.b ?? 99, 42, "nullish keeps a present value");
assert.sameValue(0 ?? 7, 0, "?? does not fall back on 0");
assert.sameValue("" ?? "x", "", "?? does not fall back on empty string");
assert.sameValue(false ?? true, false, "?? does not fall back on false");

var n = null;
assert.sameValue(n?.foo, undefined, "optional access on null is undefined");
assert.sameValue(n ?? "fallback", "fallback", "?? falls back on null");

// A nullish base short-circuits the WHOLE chain, skipping later non-optional links.
assert.sameValue(n?.foo.bar.baz, undefined, "short-circuit skips the rest of the chain");
assert.sameValue(o?.missing?.deep.deeper, undefined, "mid-chain nullish skips the tail");

// But a value that merely *happens* to be nullish still throws on a plain access.
var threw = false;
try { var bad = o.a?.zzz.qqq; } catch (e) { threw = true; }
assert.sameValue(threw, true, "non-optional access on a real undefined still throws");

// Optional call: only invoked when the callee is present.
var api = { ping() { return "pong"; } };
assert.sameValue(api.ping?.(), "pong", "optional call invokes a present method");
assert.sameValue(api.missing?.(), undefined, "optional call on a missing method is undefined");
