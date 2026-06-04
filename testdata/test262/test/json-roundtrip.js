/*---
description: JSON.parse/stringify round-trips with nested structures
esid: sec-json.parse
---*/
var obj = { name: "test", nums: [1, 2, 3], nested: { a: true, b: null }, count: 42 };
var s = JSON.stringify(obj);
var back = JSON.parse(s);
assert.sameValue(back.name, "test");
assert.sameValue(back.nums.length, 3);
assert.sameValue(back.nums[1], 2);
assert.sameValue(back.nested.a, true);
assert.sameValue(back.nested.b, null);
assert.sameValue(back.count, 42);
assert.sameValue(JSON.stringify([1, "two", true, null]), '[1,"two",true,null]');
assert.sameValue(JSON.parse('{"escaped":"a\\nb"}').escaped, "a\nb", "escaped newline");
assert.sameValue(JSON.stringify({ undef: undefined, fn: function () {} }), "{}", "undefined and functions dropped");
