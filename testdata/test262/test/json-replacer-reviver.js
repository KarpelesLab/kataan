/*---
description: JSON.stringify with indentation and array, parse of nested
esid: sec-json.stringify
---*/
assert.sameValue(JSON.stringify({ a: 1, b: [2, 3] }), '{"a":1,"b":[2,3]}');
assert.sameValue(JSON.stringify([1, "two", true, null]), '[1,"two",true,null]');
assert.sameValue(JSON.stringify("quote\"d"), '"quote\\"d"');
var o = JSON.parse('{"x":{"y":[1,2,3]},"z":true}');
assert.sameValue(o.x.y[2], 3);
assert.sameValue(o.z, true);
assert.sameValue(JSON.stringify(undefined), undefined);
