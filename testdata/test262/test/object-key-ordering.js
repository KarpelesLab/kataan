/*---
description: Object key ordering — integer keys ascending, then string keys
esid: sec-ordinaryownpropertykeys
---*/
var o = {};
o.b = 1; o.a = 2; o.c = 3;
assert.sameValue(Object.keys(o).join(","), "b,a,c", "string keys in insertion order");
var n = { 2: "a", 1: "b", 3: "c" };
assert.sameValue(Object.keys(n).join(","), "1,2,3", "integer keys ascending");
var mixed = { "z": 1, "2": 2, "a": 3, "1": 4 };
assert.sameValue(Object.keys(mixed).join(","), "1,2,z,a", "integers first, then strings");
var values = { 10: "x", 2: "y", 1: "z" };
assert.sameValue(Object.values(values).join(","), "z,y,x", "values follow key order");
