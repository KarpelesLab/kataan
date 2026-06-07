/*---
description: a function has no enumerable own keys (its VM backing array doesn't leak)
esid: sec-object.keys
---*/
function f() {}
var arrow = () => 1;

// Object.keys / values / entries of a function are empty.
assert.sameValue(Object.keys(f).length, 0, "Object.keys(function)");
assert.sameValue(Object.values(arrow).length, 0, "Object.values(arrow)");
assert.sameValue(JSON.stringify(Object.entries(f)), "[]", "Object.entries(function)");

// for-in over a function yields nothing.
var seen = "";
for (var k in f) seen += k;
assert.sameValue(seen, "", "for-in over a function");

// Real arrays and objects are unaffected.
assert.sameValue(Object.keys([10, 20, 30]).join(","), "0,1,2", "array indices");
assert.sameValue(Object.keys({ a: 1, b: 2 }).join(","), "a,b", "object keys");
var inObj = "";
for (var k2 in { x: 1, y: 2 }) inObj += k2;
assert.sameValue(inObj, "xy", "for-in over an object");
