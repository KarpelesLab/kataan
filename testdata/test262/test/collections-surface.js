/*---
description: Map/Set/WeakMap/WeakSet method surface and semantics
features: [Map, Set, WeakMap, WeakSet]
---*/
// Map: set is chainable, get/has/delete/size, insertion order, clear.
var m = new Map();
assert.sameValue(m.set("a", 1).set("b", 2), m, "set is chainable and returns the map");
assert.sameValue(m.size, 2, "size");
assert.sameValue(m.get("a"), 1, "get");
assert.sameValue(m.has("b"), true, "has");
assert.sameValue(m.delete("a"), true, "delete returns true");
assert.sameValue(m.has("a"), false, "deleted");
assert.sameValue([...new Map([["x", 1], ["y", 2]]).keys()].join(","), "x,y", "keys in insertion order");
assert.sameValue([...new Map([["x", 1], ["y", 2]]).values()].join(","), "1,2", "values");
var fe = [];
new Map([["k", 9]]).forEach(function (v, k) { fe.push(k + "=" + v); });
assert.sameValue(fe.join(","), "k=9", "forEach(value, key)");
var mc = new Map([["z", 1]]); mc.clear();
assert.sameValue(mc.size, 0, "clear");
// Object keys use identity (not structural equality).
var key = {};
var mk = new Map(); mk.set(key, 1); mk.set({}, 2);
assert.sameValue(mk.get(key), 1, "object key identity");
assert.sameValue(mk.size, 2, "distinct object keys");
// Map from an iterable of entries.
assert.sameValue(new Map(Object.entries({ a: 1, b: 2 })).get("b"), 2, "Map from entries");

// Set: dedup (SameValueZero), chainable add, has/delete/size.
var s = new Set([1, 2, 2, 3]);
assert.sameValue(s.size, 3, "dedup on construction");
assert.sameValue(s.add(4).add(5), s, "add is chainable");
assert.sameValue([...s].join(","), "1,2,3,4,5", "iteration order");
assert.sameValue(s.delete(2), true, "delete");
assert.sameValue(s.has(2), false, "deleted");
assert.sameValue(new Set([NaN]).has(NaN), true, "SameValueZero: NaN");

// WeakMap / WeakSet with object keys.
var wm = new WeakMap(); var wk = {};
wm.set(wk, "v");
assert.sameValue(wm.get(wk), "v", "WeakMap get");
assert.sameValue(wm.has(wk), true, "WeakMap has");
assert.sameValue(wm.delete(wk), true, "WeakMap delete");
var ws = new WeakSet(); var wo = {}; ws.add(wo);
assert.sameValue(ws.has(wo), true, "WeakSet has");
assert.sameValue(ws.has({}), false, "WeakSet identity");
