/*---
description: Map and Set with object keys and identity
esid: sec-map-objects
---*/
var key1 = { id: 1 };
var key2 = { id: 2 };
var m = new Map();
m.set(key1, "first");
m.set(key2, "second");
assert.sameValue(m.get(key1), "first", "object key identity");
assert.sameValue(m.get(key2), "second");
assert.sameValue(m.get({ id: 1 }), undefined, "different object not found");
assert.sameValue(m.size, 2);
m.set(key1, "updated");
assert.sameValue(m.get(key1), "updated", "update by same key");
assert.sameValue(m.size, 2, "no new entry");
var s = new Set();
var obj = {};
s.add(obj);
s.add(obj);
assert.sameValue(s.size, 1, "same object added once");
s.add({});
assert.sameValue(s.size, 2, "distinct objects");
var mixed = new Map();
mixed.set(1, "num");
mixed.set("1", "str");
mixed.set(true, "bool");
assert.sameValue(mixed.size, 3, "distinct key types");
assert.sameValue(mixed.get(1), "num");
assert.sameValue(mixed.get("1"), "str");
