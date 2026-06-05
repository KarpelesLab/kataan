/*---
description: Truthiness and boolean coercion in conditionals
esid: sec-toboolean
---*/
assert.sameValue(!!0, false);
assert.sameValue(!!1, true);
assert.sameValue(!!"", false);
assert.sameValue(!!"false", true, "non-empty string is truthy");
assert.sameValue(!!null, false);
assert.sameValue(!!undefined, false);
assert.sameValue(!!NaN, false);
assert.sameValue(!![], true);
assert.sameValue(!!{}, true);
assert.sameValue(!!0n, false, "0n is falsy");
assert.sameValue(!!1n, true);
var result = "" || "default";
assert.sameValue(result, "default");
var nullish = null ?? "fallback";
assert.sameValue(nullish, "fallback");
var kept = 0 ?? "x";
assert.sameValue(kept, 0, "?? keeps 0");
