/*---
description: JSON.stringify/parse round trips and edge cases
esid: sec-json.stringify
---*/
var obj = { name: "test", values: [1, 2, 3], nested: { a: true, b: null } };
var json = JSON.stringify(obj);
var parsed = JSON.parse(json);
assert.sameValue(parsed.name, "test");
assert.sameValue(parsed.values.length, 3);
assert.sameValue(parsed.nested.a, true);
assert.sameValue(parsed.nested.b, null);
assert.sameValue(JSON.stringify([1, "two", true, null]), '[1,"two",true,null]');
assert.sameValue(JSON.stringify({ a: undefined, b: 1 }), '{"b":1}', "undefined omitted");
assert.sameValue(JSON.stringify({ f: function () {} }), "{}", "functions omitted");
assert.sameValue(JSON.stringify("string"), '"string"');
assert.sameValue(JSON.stringify(42), "42");
assert.sameValue(JSON.stringify(null), "null");
assert.sameValue(JSON.stringify({ a: 1 }, null, 2), '{\n  "a": 1\n}', "indentation");
assert.sameValue(JSON.parse('{"x": [1, {"y": 2}]}').x[1].y, 2);
