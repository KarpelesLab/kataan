/*---
description: WeakMap/WeakSet are not iterable and have no size property
esid: sec-weakmap-objects
---*/
function notIterable(fn) { try { fn(); return false; } catch (e) { return e instanceof TypeError; } }

// Spreading or for-of over a weak collection is a TypeError (no Symbol.iterator).
assert.sameValue(notIterable(function () { return [...new WeakMap()]; }), true, "spread WeakMap");
assert.sameValue(notIterable(function () { return [...new WeakSet()]; }), true, "spread WeakSet");
assert.sameValue(notIterable(function () { for (var x of new WeakMap()) {} }), true, "for-of WeakMap");
assert.sameValue(notIterable(function () { for (var x of new WeakSet()) {} }), true, "for-of WeakSet");

// They have no size property.
assert.sameValue(new WeakMap().size, undefined, "WeakMap has no size");
assert.sameValue(new WeakSet().size, undefined, "WeakSet has no size");

// A WeakMap/WeakSet is still functional.
var wm = new WeakMap();
var key = {};
wm.set(key, 42);
assert.sameValue(wm.get(key), 42, "WeakMap.get");
assert.sameValue(wm.has(key), true, "WeakMap.has");
var ws = new WeakSet();
var o = {};
ws.add(o);
assert.sameValue(ws.has(o), true, "WeakSet.has");

// Map and Set remain iterable and carry size.
var m = new Map([["a", 1], ["b", 2]]);
assert.sameValue(m.size, 2, "Map size");
assert.sameValue([...m.keys()].join(","), "a,b", "Map iteration");
var s = new Set([1, 2, 3]);
assert.sameValue(s.size, 3, "Set size");
assert.sameValue([...s].join(","), "1,2,3", "Set iteration");
