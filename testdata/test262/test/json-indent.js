/*---
description: JSON.stringify with a numeric or string indentation argument
esid: sec-json.stringify
---*/
assert.sameValue(JSON.stringify({ a: 1, b: 2 }, null, 2), '{\n  "a": 1,\n  "b": 2\n}');
assert.sameValue(JSON.stringify([1, 2], null, "  "), "[\n  1,\n  2\n]");
assert.sameValue(JSON.stringify({ x: 1 }), '{"x":1}', "no indent without the arg");
assert.sameValue(JSON.stringify({}, null, 2), "{}", "empty object stays inline");
assert.sameValue(JSON.stringify([], null, 2), "[]", "empty array stays inline");
assert.sameValue(JSON.stringify({ a: [1] }, null, 1), '{\n "a": [\n  1\n ]\n}');
