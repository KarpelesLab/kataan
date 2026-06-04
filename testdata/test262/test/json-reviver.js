/*---
description: JSON.parse with a reviver function
esid: sec-json.parse
---*/
var doubled = JSON.parse('{"a":1,"b":2}', function (key, value) {
  return typeof value === "number" ? value * 2 : value;
});
assert.sameValue(doubled.a, 2);
assert.sameValue(doubled.b, 4);

var filtered = JSON.parse('{"keep":1,"drop":2}', function (key, value) {
  if (key === "drop") return undefined;
  return value;
});
assert.sameValue(filtered.keep, 1);
assert.sameValue("drop" in filtered, false, "reviver returning undefined deletes the key");

var arr = JSON.parse('[1,2,3]', function (key, value) {
  return typeof value === "number" ? value + 10 : value;
});
assert.sameValue(arr.join(","), "11,12,13");

// The root is revived with an empty key.
var rootKey = "";
JSON.parse('5', function (key, value) { rootKey = key; return value; });
assert.sameValue(rootKey, "");
