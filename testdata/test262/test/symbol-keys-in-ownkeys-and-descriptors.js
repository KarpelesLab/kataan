/*---
description: Reflect.ownKeys includes non-enumerable symbols; getOwnPropertyDescriptors includes symbol keys
esid: sec-reflect.ownkeys
---*/
var s1 = Symbol("a"), s2 = Symbol("b");

// Reflect.ownKeys lists every own symbol, including non-enumerable ones.
var o = {};
o[s1] = 1;
Object.defineProperty(o, s2, { value: 2, enumerable: false });
var symKeys = Reflect.ownKeys(o).filter(function (k) { return typeof k === "symbol"; });
assert.sameValue(symKeys.length, 2, "Reflect.ownKeys symbol count");
assert.sameValue(Object.getOwnPropertySymbols(o).length, 2, "getOwnPropertySymbols agrees");

// Ordering: string keys (integer-first), then symbols.
var o2 = {};
o2[s1] = 1; o2.str = 2; o2[3] = 3;
var keys = Reflect.ownKeys(o2);
assert.sameValue(keys[0], "3", "integer string key first");
assert.sameValue(keys[1], "str", "string key");
assert.sameValue(keys[2], s1, "symbol last");

// getOwnPropertyDescriptors returns descriptors for symbol keys too.
var descs = Object.getOwnPropertyDescriptors({ [s1]: 1, a: 2 });
assert.sameValue(Object.getOwnPropertySymbols(descs).length, 1, "one symbol descriptor");
assert.sameValue(Object.keys(descs).length, 1, "one string descriptor");
assert.sameValue(descs[s1].value, 1, "symbol descriptor value");
assert.sameValue(descs[s1].enumerable, true, "symbol descriptor enumerable");

// A non-enumerable, non-writable symbol descriptor is reported accurately.
var o3 = {};
Object.defineProperty(o3, s1, { value: 5, enumerable: false, writable: false });
var d3 = Object.getOwnPropertyDescriptors(o3);
assert.sameValue(d3[s1].value, 5, "value");
assert.sameValue(d3[s1].enumerable, false, "non-enumerable");
assert.sameValue(d3[s1].writable, false, "non-writable");

// String-only objects are unchanged.
var d4 = Object.getOwnPropertyDescriptors({ a: 1, b: 2 });
assert.sameValue(Object.keys(d4).join(","), "a,b", "string keys");
assert.sameValue(d4.a.value, 1, "string descriptor value");
