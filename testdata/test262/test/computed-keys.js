/*---
description: Computed property keys in objects and assignment
esid: sec-object-initializer
---*/
var k = "dynamic";
var o = { [k]: 1, [k + "2"]: 2, ["a" + "b"]: 3 };
assert.sameValue(o.dynamic, 1);
assert.sameValue(o.dynamic2, 2);
assert.sameValue(o.ab, 3);
var n = 5;
var nums = { [n]: "five", [n * 2]: "ten" };
assert.sameValue(nums[5], "five");
assert.sameValue(nums[10], "ten");
var prefix = "get";
var methods = { [prefix + "Name"]: function () { return "name"; } };
assert.sameValue(methods.getName(), "name", "computed method-ish key");
