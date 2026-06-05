/*---
description: Map and Set with object keys, forEach, and chaining
esid: sec-map-objects
---*/
var m = new Map();
var key1 = {}, key2 = {};
m.set(key1, "a").set(key2, "b");
assert.sameValue(m.get(key1), "a", "object keys are distinct");
assert.sameValue(m.get(key2), "b");
assert.sameValue(m.get({}), undefined, "different object is a different key");
assert.sameValue(m.size, 2);
var collected = [];
m.forEach(function (v, k) { collected.push(v); });
assert.sameValue(collected.join(","), "a,b");
m.delete(key1);
assert.sameValue(m.has(key1), false);
assert.sameValue(m.size, 1);
var s = new Set();
s.add(1).add(2).add(2).add(3);
assert.sameValue(s.size, 3, "Set dedupes");
var sum = 0;
s.forEach(function (v) { sum += v; });
assert.sameValue(sum, 6);
