/*---
description: named properties on arrays and function-expression values enumerate
esid: sec-array-exotic-objects
---*/
// An array carries integer indices first, then named own properties.
var a = [1, 2];
a.foo = "bar";
assert.sameValue(Object.keys(a).join(","), "0,1,foo", "Object.keys includes named prop");
assert.sameValue(Object.values(a).join(","), "1,2,bar", "Object.values");
assert.sameValue(JSON.stringify(Object.entries(a)), '[["0",1],["1",2],["foo","bar"]]', "Object.entries");

// for-in over the array yields indices then named props.
var seen = [];
for (var k in a) seen.push(k);
assert.sameValue(seen.join(","), "0,1,foo", "for-in");

// A function-expression value carries assigned named properties too.
var g = function () {};
g.x = 1;
g.y = 2;
assert.sameValue(Object.keys(g).join(","), "x,y", "function expression named props");

// Plain arrays/objects/functions are unaffected.
assert.sameValue(Object.keys([10, 20, 30]).join(","), "0,1,2", "plain array");
assert.sameValue(Object.keys({ p: 1, q: 2 }).join(","), "p,q", "plain object");
assert.sameValue(Object.keys(function () {}).length, 0, "bare function has no own keys");
