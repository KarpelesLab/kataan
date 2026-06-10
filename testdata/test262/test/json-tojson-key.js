/*---
description: JSON.stringify calls toJSON(key) with the property key (or "" at top level)
esid: sec-serializejsonproperty
---*/
// The property name is passed to toJSON.
assert.sameValue(
  JSON.stringify({ when: { toJSON(k) { return "key:" + k; } } }),
  '{"when":"key:when"}',
  "object property key"
);
// The top-level value gets the empty-string key.
assert.sameValue(
  JSON.stringify({ toJSON(k) { return "top[" + k + "]"; } }),
  '"top[]"',
  "top-level key is empty string"
);
// Array elements get their index (as a string).
assert.sameValue(
  JSON.stringify([{ toJSON(k) { return "i" + k; } }, { toJSON(k) { return "i" + k; } }]),
  '["i0","i1"]',
  "array index keys"
);
// Nested.
assert.sameValue(
  JSON.stringify({ a: { b: { toJSON(k) { return "k=" + k; } } } }),
  '{"a":{"b":"k=b"}}',
  "nested key"
);
// A toJSON that ignores the key still works (and Date's built-in toJSON).
assert.sameValue(JSON.stringify({ x: { v: 5, toJSON() { return this.v * 2; } } }), '{"x":10}', "keyless toJSON");
assert.sameValue(JSON.stringify({ d: new Date(0) }), '{"d":"1970-01-01T00:00:00.000Z"}', "Date toJSON");
