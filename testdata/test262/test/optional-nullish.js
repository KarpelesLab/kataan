/*---
description: Optional chaining and nullish coalescing
esid: sec-optional-chains
---*/
var obj = { a: { b: { c: 42 } } };
assert.sameValue(obj?.a?.b?.c, 42);
assert.sameValue(obj?.x?.y?.z, undefined, "short-circuits on undefined");
assert.sameValue(obj?.a?.missing, undefined);
var fn = { go: function () { return "went"; } };
assert.sameValue(fn.go?.(), "went");
assert.sameValue(fn.stop?.(), undefined, "optional call on missing method");
assert.sameValue(null ?? "default", "default");
assert.sameValue(0 ?? "default", 0, "nullish only on null/undefined");
assert.sameValue("" ?? "default", "");
