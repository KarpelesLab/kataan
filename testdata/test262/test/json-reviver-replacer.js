/*---
description: JSON.parse reviver and JSON.stringify replacer
esid: sec-json.parse
---*/
var parsed = JSON.parse('{"a":1,"b":2,"c":3}', function (key, value) {
  if (typeof value === "number") return value * 2;
  return value;
});
assert.sameValue(parsed.a, 2, "reviver doubles numbers");
assert.sameValue(parsed.b, 4);
var dateRevived = JSON.parse('{"n":5}', function (key, value) {
  return key === "n" ? value + 100 : value;
});
assert.sameValue(dateRevived.n, 105);
var filtered = JSON.stringify({ a: 1, b: 2, c: 3 }, function (key, value) {
  if (key === "b") return undefined;
  return value;
});
assert.sameValue(filtered, '{"a":1,"c":3}', "replacer omits b");
var arrayReplacer = JSON.stringify({ a: 1, b: 2, c: 3 }, ["a", "c"]);
assert.sameValue(arrayReplacer, '{"a":1,"c":3}', "array replacer whitelists");
var transformed = JSON.stringify({ x: 5 }, function (key, value) {
  return typeof value === "number" ? value * 10 : value;
});
assert.sameValue(transformed, '{"x":50}');
