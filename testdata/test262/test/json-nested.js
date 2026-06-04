/*---
description: JSON.stringify and parse round-trip nested structures
esid: sec-json-object
---*/
var data = { name: "x", nums: [1, 2, 3], nested: { ok: true, val: null } };
var s = JSON.stringify(data);
var back = JSON.parse(s);
assert.sameValue(back.name, "x");
assert.sameValue(back.nums.length, 3);
assert.sameValue(back.nested.ok, true);
assert.sameValue(back.nested.val, null);
assert.sameValue(JSON.stringify([1, "a", true, null]), '[1,"a",true,null]');
