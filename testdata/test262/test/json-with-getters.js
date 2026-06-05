/*---
description: JSON.stringify invokes getters and handles toJSON
esid: sec-json.stringify
---*/
var obj = { a: 1, get b() { return 2; } };
assert.sameValue(JSON.stringify(obj), '{"a":1,"b":2}', "getter invoked");
var withToJSON = { value: 42, toJSON: function () { return { wrapped: this.value }; } };
assert.sameValue(JSON.stringify(withToJSON), '{"wrapped":42}', "toJSON used");
var nested = { d: { toJSON: function () { return "custom"; } } };
assert.sameValue(JSON.stringify(nested), '{"d":"custom"}');
assert.sameValue(JSON.stringify({ a: undefined, b: 1 }), '{"b":1}', "undefined dropped");
assert.sameValue(JSON.stringify([undefined, 1]), "[null,1]", "undefined in array is null");
