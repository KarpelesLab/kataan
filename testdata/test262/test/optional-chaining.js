/*---
description: Optional chaining and nullish coalescing
esid: sec-optional-chains
---*/
var o = { a: { b: { c: 42 } } };
assert.sameValue(o?.a?.b?.c, 42);
assert.sameValue(o?.x?.y?.z, undefined, "short-circuits on undefined");
assert.sameValue(o?.a?.missing, undefined);
var nul = null;
assert.sameValue(nul?.anything, undefined, "null short-circuits");
assert.sameValue(nul?.deep?.path, undefined);
var fn = { method: function () { return "called"; } };
assert.sameValue(fn.method?.(), "called", "optional call");
assert.sameValue(fn.missing?.(), undefined, "optional call on missing");
assert.sameValue(o?.["a"]?.["b"]?.["c"], 42, "optional computed");
var arr = [1, 2, 3];
assert.sameValue(arr?.[0], 1);
assert.sameValue(arr?.[10], undefined);
assert.sameValue((null ?? undefined ?? "fallback"), "fallback");
assert.sameValue((o?.a?.b?.c ?? "default"), 42);
assert.sameValue((o?.x?.y ?? "default"), "default");
