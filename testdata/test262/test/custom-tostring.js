/*---
description: Custom toString/valueOf in coercion contexts
esid: sec-tostring
---*/
var obj = { toString: function () { return "custom"; } };
assert.sameValue("" + obj, "custom", "toString in concatenation");
assert.sameValue(`${obj}`, "custom", "toString in template");
assert.sameValue([obj, obj].join(","), "custom,custom", "toString in join");
var valued = { valueOf: function () { return 42; } };
assert.sameValue(valued + 8, 50, "valueOf in arithmetic");
assert.sameValue(valued * 2, 84);
var both = {
  toString: function () { return "str"; },
  valueOf: function () { return 10; }
};
assert.sameValue(both + 5, 15, "valueOf preferred in arithmetic");
assert.sameValue(`${both}`, "str", "toString in string context");
assert.sameValue(String(both), "str");
assert.sameValue(Number(valued), 42);
