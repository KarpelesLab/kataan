/*---
description: JSON.stringify honors toJSON, getters, and the replacer on every engine
esid: sec-json.stringify
---*/
// toJSON replaces the value (top-level and nested).
assert.sameValue(JSON.stringify({ toJSON() { return "custom"; } }), '"custom"', "toJSON top-level");
assert.sameValue(JSON.stringify({ d: { toJSON() { return "c"; } } }), '{"d":"c"}', "toJSON nested");

// A getter is invoked during serialization.
assert.sameValue(JSON.stringify({ get x() { return 42; } }), '{"x":42}', "getter");

// A function replacer transforms each value (object and array members).
assert.sameValue(JSON.stringify({ a: 1, b: 2, c: 3 }, function (k, v) { return k === "b" ? undefined : v; }), '{"a":1,"c":3}', "replacer function drops a key");
assert.sameValue(JSON.stringify([1, 2, 3], function (k, v) { return (typeof v === "number" && v > 1) ? v * 100 : v; }), "[1,200,300]", "replacer on array elements");

// An array replacer allowlists object keys.
assert.sameValue(JSON.stringify({ a: 1, b: 2, c: 3 }, ["a", "c"]), '{"a":1,"c":3}', "array replacer");

// Unaffected: plain data, nesting, Dates, methods/undefined omitted, indentation, cycles.
assert.sameValue(JSON.stringify({ a: 1, b: [2, 3], c: "x" }), '{"a":1,"b":[2,3],"c":"x"}', "plain");
assert.sameValue(JSON.stringify({ d: new Date(0) }), '{"d":"1970-01-01T00:00:00.000Z"}', "Date toJSON preserved");
assert.sameValue(JSON.stringify({ a: 1, m() {} }), '{"a":1}', "methods omitted");
assert.sameValue(JSON.stringify([NaN, Infinity, undefined, function () {}]), "[null,null,null,null]", "array holes");
assert.sameValue(JSON.stringify({ a: 1 }, null, 2), '{\n  "a": 1\n}', "indentation");
var threw = false;
try { var c = {}; c.s = c; JSON.stringify(c); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "circular throws a TypeError");
