/*---
description: JSON.stringify with a function or array replacer
esid: sec-json.stringify
---*/
var fn = JSON.stringify({ a: 1, b: 2, c: 3 }, function (k, v) {
  return k === "b" ? undefined : v;
});
assert.sameValue(fn, '{"a":1,"c":3}', "function replacer omits keys returning undefined");

var doubled = JSON.stringify({ x: { a: 1 }, y: 2 }, function (k, v) {
  return typeof v === "number" ? v * 10 : v;
});
assert.sameValue(doubled, '{"x":{"a":10},"y":20}', "function replacer recurses");

var allow = JSON.stringify({ a: 1, b: 2, c: 3 }, ["a", "c"]);
assert.sameValue(allow, '{"a":1,"c":3}', "array replacer is a key allowlist");

var nested = JSON.stringify({ keep: { a: 1, b: 2 }, drop: 9 }, ["keep", "a"]);
assert.sameValue(nested, '{"keep":{"a":1}}', "array replacer filters at every level");
