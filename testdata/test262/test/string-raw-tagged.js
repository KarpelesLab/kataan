/*---
description: Cooked tagged templates and String.raw with a raw-bearing object
esid: sec-string.raw
---*/
function cook(strings, ...values) {
  return strings.join("|") + "#" + values.join(",");
}
assert.sameValue(cook`a${10}b${20}c`, "a|b|c#10,20", "cooked strings array");
// String.raw interleaves the `raw` strings with the substitutions.
assert.sameValue(String.raw({ raw: ["a", "b", "c"] }, 1, 2), "a1b2c");
assert.sameValue(String.raw({ raw: ["x"] }), "x", "no substitutions");
