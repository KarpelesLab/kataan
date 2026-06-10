/*---
description: an array replacer emits keys in allowlist order, deduplicated, at every level
esid: sec-json.stringify
---*/
// Keys follow the allowlist order, not the object's own order.
assert.sameValue(JSON.stringify({ a: 1, b: 2, c: 3 }, ["c", "a"]), '{"c":3,"a":1}', "allowlist order");
// Duplicates in the allowlist are ignored.
assert.sameValue(JSON.stringify({ a: 1, b: 2 }, ["a", "a", "b"]), '{"a":1,"b":2}', "deduplicated");
// A key not present on the object is skipped.
assert.sameValue(JSON.stringify({ a: 1 }, ["a", "z"]), '{"a":1}', "missing key skipped");
// The allowlist applies at every nesting level, in its order.
assert.sameValue(JSON.stringify({ a: 1, b: { c: 2, a: 3 } }, ["a", "c", "b"]), '{"a":1,"b":{"a":3,"c":2}}', "nested allowlist order");
// Non-allowlisted keys are dropped entirely.
assert.sameValue(JSON.stringify({ x: 1, y: 2 }, ["x"]), '{"x":1}', "drop non-listed");
