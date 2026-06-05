/*---
description: JSON.stringify serializes Date as ISO and throws on BigInt
esid: sec-json.stringify
---*/
assert.sameValue(JSON.stringify(new Date(0)), '"1970-01-01T00:00:00.000Z"', "Date serializes to ISO string");
assert.sameValue(JSON.stringify({ created: new Date(0) }), '{"created":"1970-01-01T00:00:00.000Z"}', "Date in an object");
assert.sameValue(JSON.stringify([new Date(0)]), '["1970-01-01T00:00:00.000Z"]', "Date in an array");
var threw = false;
try { JSON.stringify(10n); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "BigInt cannot be serialized");
var threw2 = false;
try { JSON.stringify({ big: 1n }); } catch (e) { threw2 = e instanceof TypeError; }
assert.sameValue(threw2, true, "BigInt in an object throws");
assert.sameValue(JSON.stringify({ a: 1, b: "x", c: true, d: null }), '{"a":1,"b":"x","c":true,"d":null}', "ordinary values unaffected");
assert.sameValue(JSON.stringify({ toJSON: function () { return "custom"; } }), '"custom"', "toJSON still honored");
assert.sameValue(JSON.stringify(NaN), "null", "NaN serializes to null");
assert.sameValue(JSON.stringify(Infinity), "null");
