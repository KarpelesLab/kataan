/*---
description: WeakMap and WeakSet basic operations
esid: sec-weakmap-objects
---*/
var wm = new WeakMap();
var k1 = {};
var k2 = {};
wm.set(k1, "one");
wm.set(k2, "two");
assert.sameValue(wm.get(k1), "one");
assert.sameValue(wm.get(k2), "two");
assert.sameValue(wm.has(k1), true);
assert.sameValue(wm.has({}), false, "different object");
wm.delete(k1);
assert.sameValue(wm.has(k1), false);
assert.sameValue(wm.get(k1), undefined);
var ws = new WeakSet();
var o1 = {};
ws.add(o1);
assert.sameValue(ws.has(o1), true);
assert.sameValue(ws.has({}), false);
ws.delete(o1);
assert.sameValue(ws.has(o1), false);
var chained = new WeakMap();
assert.sameValue(chained.set(k1, 1) instanceof WeakMap, true, "set returns the map");
