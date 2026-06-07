/*---
description: JSON.stringify omits functions (undefined top-level, omitted in objects, null in arrays)
esid: sec-json.stringify
---*/
// A function alone has no JSON representation.
assert.sameValue(JSON.stringify(function () {}), undefined, "top-level function");
assert.sameValue(JSON.stringify(() => 1), undefined, "top-level arrow");

// A function-valued object property is omitted.
assert.sameValue(JSON.stringify({ fn: function () {}, x: 1 }), '{"x":1}', "function property omitted");
assert.sameValue(JSON.stringify({ a: () => 1, arr: [1, 2] }), '{"arr":[1,2]}', "arrow property omitted");

// A function in an array becomes null.
assert.sameValue(JSON.stringify([1, function () {}, 2]), "[1,null,2]", "function array element -> null");

// Pretty-printing skips functions too.
assert.sameValue(JSON.stringify({ a: () => 1, b: 2 }, null, 2), '{\n  "b": 2\n}', "pretty omits function");

// Plain data is unaffected.
assert.sameValue(JSON.stringify({ a: 1, b: [2, 3] }), '{"a":1,"b":[2,3]}', "data unchanged");
