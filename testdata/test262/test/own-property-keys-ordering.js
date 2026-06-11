/*---
description: getOwnPropertyNames / Reflect.ownKeys use [[OwnPropertyKeys]] order (integers first)
esid: sec-ordinaryownpropertykeys
---*/
var o = {};
o.b = 1; o[2] = 2; o.a = 3; o[1] = 4; o.c = 5; o[10] = 6;

// Integer-index keys ascending, then string keys in insertion order — for ALL of these.
var expected = "1,2,10,b,a,c";
assert.sameValue(Object.keys(o).join(","), expected, "Object.keys");
assert.sameValue(Object.getOwnPropertyNames(o).join(","), expected, "getOwnPropertyNames");
assert.sameValue(Reflect.ownKeys(o).join(","), expected, "Reflect.ownKeys");
assert.sameValue(Object.keys(Object.getOwnPropertyDescriptors(o)).join(","), expected, "getOwnPropertyDescriptors");

// Non-enumerable keys are included by getOwnPropertyNames in the same order.
var o2 = {};
o2.b = 1; o2[2] = 2;
Object.defineProperty(o2, "hidden", { value: 9, enumerable: false });
o2[1] = 3;
assert.sameValue(Object.getOwnPropertyNames(o2).join(","), "1,2,b,hidden", "non-enumerable included, ordered");
assert.sameValue(Object.keys(o2).join(","), "1,2,b", "Object.keys excludes non-enumerable");

// Reflect.ownKeys: integer string keys, then string keys, then symbols (insertion order).
var s1 = Symbol("a"), s2 = Symbol("b");
var o3 = {};
o3[s1] = 1; o3.str = 2; o3[3] = 3; o3[s2] = 4;
var keys = Reflect.ownKeys(o3);
assert.sameValue(keys[0], "3", "integer key first");
assert.sameValue(keys[1], "str", "string key next");
assert.sameValue(keys[2], s1, "first symbol");
assert.sameValue(keys[3], s2, "second symbol");

// Non-canonical integer-like strings ("02") are ordinary string keys.
var o4 = {};
o4["02"] = 1; o4["2"] = 2; o4["1"] = 3;
assert.sameValue(Object.getOwnPropertyNames(o4).join(","), "1,2,02", "non-canonical stays a string key");
