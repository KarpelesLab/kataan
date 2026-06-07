/*---
description: JSON.parse reviver and JSON.stringify replacer/toJSON/special values
esid: sec-json.parse
---*/
// reviver transforms each value bottom-up.
var o = JSON.parse('{"a":1,"b":[2,3]}', function (k, v) {
  return typeof v === "number" ? v * 10 : v;
});
assert.sameValue(o.a, 10, "reviver scales a");
assert.sameValue(o.b[1], 30, "reviver scales nested");

// reviver returning undefined deletes the key.
var d = JSON.parse('{"keep":1,"drop":2}', function (k, v) {
  return k === "drop" ? undefined : v;
});
assert.sameValue(JSON.stringify(d), '{"keep":1}', "reviver deletes a key");

// replacer function omits keys when it returns undefined.
assert.sameValue(
  JSON.stringify({ a: 1, b: 2, c: 3 }, function (k, v) { return k === "b" ? undefined : v; }),
  '{"a":1,"c":3}',
  "replacer function"
);
// replacer array whitelists keys.
assert.sameValue(JSON.stringify({ a: 1, b: 2, c: 3 }, ["a", "c"]), '{"a":1,"c":3}', "replacer array");

// indentation.
assert.sameValue(JSON.stringify({ a: 1 }, null, 2), '{\n  "a": 1\n}', "indent");

// toJSON is honored.
assert.sameValue(JSON.stringify({ d: { toJSON() { return "custom"; } } }), '{"d":"custom"}', "toJSON");

// Non-serializable values: undefined/function omitted in objects; NaN/Infinity -> null.
assert.sameValue(
  JSON.stringify({ a: undefined, b: function () {}, c: null, d: NaN, e: Infinity }),
  '{"c":null,"d":null,"e":null}',
  "special values"
);

// String escaping and number edges.
assert.sameValue(JSON.stringify("a\"b\\c\nd\te"), '"a\\"b\\\\c\\nd\\te"', "escaping");
assert.sameValue(JSON.stringify([0, -0, 1.5, 1e21, 1e-7]), "[0,0,1.5,1e+21,1e-7]", "number edges");

// A circular structure and a BigInt both throw TypeError.
var circular = {};
circular.self = circular;
var threwC = false, threwB = false;
try { JSON.stringify(circular); } catch (e) { threwC = e instanceof TypeError; }
try { JSON.stringify(1n); } catch (e) { threwB = e instanceof TypeError; }
assert.sameValue(threwC, true, "circular -> TypeError");
assert.sameValue(threwB, true, "BigInt -> TypeError");
