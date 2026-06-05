/*---
description: Nullish coalescing and logical operators with short-circuit
esid: sec-binary-logical-operators
---*/
assert.sameValue(null ?? "default", "default");
assert.sameValue(undefined ?? "default", "default");
assert.sameValue(0 ?? "default", 0, "0 is not nullish");
assert.sameValue("" ?? "default", "", "empty string is not nullish");
assert.sameValue(false ?? "default", false);
assert.sameValue(null ?? undefined ?? "last", "last", "chained nullish");
assert.sameValue(1 && 2 && 3, 3, "&& returns last truthy");
assert.sameValue(0 && 2, 0, "&& short-circuits on falsy");
assert.sameValue(null || 0 || "found", "found", "|| returns first truthy");
assert.sameValue("a" || "b", "a");
var calls = 0;
function sideEffect() { calls++; return true; }
false && sideEffect();
assert.sameValue(calls, 0, "&& does not eval right on false");
true || sideEffect();
assert.sameValue(calls, 0, "|| does not eval right on true");
var x = null;
x ??= "assigned";
assert.sameValue(x, "assigned", "nullish assignment");
var y = "exists";
y ??= "not used";
assert.sameValue(y, "exists");
