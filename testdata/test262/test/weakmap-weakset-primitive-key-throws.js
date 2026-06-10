/*---
description: WeakMap/WeakSet reject primitive keys with a TypeError
esid: sec-weakmap.prototype.set
features: [WeakMap, WeakSet]
---*/
var wm = new WeakMap();
var key = {};
assert.sameValue(wm.set(key, 1), wm, "set returns the map");
assert.sameValue(wm.get(key), 1, "object key works");

// Primitive keys throw.
assert.throws(TypeError, function () { wm.set("s", 1); }, "string key");
assert.throws(TypeError, function () { wm.set(5, 1); }, "number key");
assert.throws(TypeError, function () { wm.set(null, 1); }, "null key");
assert.throws(TypeError, function () { wm.set(true, 1); }, "boolean key");
assert.throws(TypeError, function () { wm.set(undefined, 1); }, "undefined key");

// A symbol is a valid weak key (ES2023+).
var sym = Symbol("x");
assert.sameValue(wm.set(sym, 9), wm, "symbol key allowed");
assert.sameValue(wm.get(sym), 9, "symbol key stored");

// WeakSet too.
var ws = new WeakSet();
assert.sameValue(ws.add({}), ws, "object add");
assert.throws(TypeError, function () { ws.add("x"); }, "WeakSet string");

// The constructor's seeding validates as well.
assert.throws(TypeError, function () { return new WeakMap([["str", 1]]); }, "seed with primitive key");
assert.sameValue(new WeakMap([[{}, 1]]) instanceof WeakMap, true, "seed with object key");

// get/has with a primitive do NOT throw — they just miss.
assert.sameValue(wm.get("s"), undefined, "get primitive -> undefined");
assert.sameValue(wm.has(5), false, "has primitive -> false");

// A plain Map/Set still accepts primitive keys.
var m = new Map();
m.set("s", 1);
assert.sameValue(m.get("s"), 1, "Map allows string key");
var s = new Set();
s.add("x");
assert.sameValue(s.has("x"), true, "Set allows string");
