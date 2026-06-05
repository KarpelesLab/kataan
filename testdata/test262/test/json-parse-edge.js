/*---
description: JSON.parse with various valid inputs and structures
esid: sec-json.parse
---*/
assert.sameValue(JSON.parse("42"), 42);
assert.sameValue(JSON.parse('"hello"'), "hello");
assert.sameValue(JSON.parse("true"), true);
assert.sameValue(JSON.parse("null"), null);
assert.sameValue(JSON.parse("[1,2,3]").length, 3);
assert.sameValue(JSON.parse('{"a":1,"b":[2,3]}').b[1], 3);
assert.sameValue(JSON.parse('{"nested":{"deep":{"value":42}}}').nested.deep.value, 42);
assert.sameValue(JSON.parse('"\\u0041"'), "A", "unicode escape");
assert.sameValue(JSON.parse('"line\\nbreak"'), "line\nbreak", "escaped newline");
assert.sameValue(JSON.parse("[]").length, 0);
assert.sameValue(Object.keys(JSON.parse("{}")).length, 0);
assert.sameValue(JSON.parse("-3.14"), -3.14);
assert.sameValue(JSON.parse("1e3"), 1000);
