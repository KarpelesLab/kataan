/*---
description: WeakMap and WeakSet basic operations
esid: sec-weakmap-objects
---*/
var key = {};
var wm = new WeakMap();
wm.set(key, "value");
assert.sameValue(wm.get(key), "value");
assert.sameValue(wm.has(key), true);
wm.delete(key);
assert.sameValue(wm.has(key), false);
var ws = new WeakSet();
var obj = {};
ws.add(obj);
assert.sameValue(ws.has(obj), true);
