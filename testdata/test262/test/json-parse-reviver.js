/*---
description: JSON.parse honors the reviver and throws SyntaxError on malformed input (every engine)
esid: sec-json.parse
---*/
// The reviver transforms each value bottom-up.
assert.sameValue(JSON.parse('{"a":1,"b":2}', function (k, v) { return typeof v === "number" ? v * 10 : v; }).a, 10, "reviver transforms");
// Returning undefined deletes an object property.
assert.sameValue(JSON.stringify(JSON.parse('{"x":1,"y":2}', function (k, v) { return k === "x" ? undefined : v; })), '{"y":2}', "reviver deletes");
// Array elements are revived by index.
assert.sameValue(JSON.parse('[1,2,3]', function (k, v) { return typeof v === "number" ? v + 100 : v; })[0], 101, "reviver on array");
// Nested objects are revived bottom-up.
assert.sameValue(JSON.stringify(JSON.parse('{"o":{"n":5}}', function (k, v) { return typeof v === "number" ? v + 1 : v; })), '{"o":{"n":6}}', "nested reviver");
// The reviver sees the key (top-level key is "").
var topKey;
JSON.parse('{"a":1}', function (k, v) { if (typeof v === "object") topKey = k; return v; });
assert.sameValue(topKey, "", "top-level key is empty string");

// Malformed input throws a SyntaxError (not a silent undefined).
assert.throws(SyntaxError, function () { return JSON.parse("{bad}"); }, "object syntax error");
assert.throws(SyntaxError, function () { return JSON.parse("[1,2,"); }, "truncated array");

// Plain parse (no reviver) is unaffected.
assert.sameValue(JSON.stringify(JSON.parse('{"a":[1,2],"b":"x"}')), '{"a":[1,2],"b":"x"}', "plain parse");
assert.sameValue(JSON.parse("42"), 42, "primitive parse");
