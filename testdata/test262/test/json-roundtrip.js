/*---
description: JSON.stringify and JSON.parse round-trip an object
esid: sec-json.parse
---*/
var obj = { a: 1, b: [2, 3], c: "x" };
var s = JSON.stringify(obj);
assert.sameValue(s, '{"a":1,"b":[2,3],"c":"x"}', "stringify");
var back = JSON.parse(s);
assert.sameValue(back.a, 1);
assert.sameValue(back.b[1], 3);
assert.sameValue(back.c, "x");
