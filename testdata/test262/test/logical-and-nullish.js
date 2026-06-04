/*---
description: Short-circuit logical operators and nullish coalescing
esid: sec-binary-logical-operators
---*/
assert.sameValue(0 || "fallback", "fallback");
assert.sameValue(1 && 2, 2);
assert.sameValue(null ?? "default", "default");
assert.sameValue(0 ?? "default", 0, "?? only on null/undefined");
var obj = { a: { b: 5 } };
assert.sameValue(obj?.a?.b, 5, "optional chaining hit");
assert.sameValue(obj?.x?.y, undefined, "optional chaining miss");
