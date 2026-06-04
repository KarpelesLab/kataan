/*---
description: Map and Set basic operations
esid: sec-map-objects
---*/
var m = new Map();
m.set("a", 1);
m.set("b", 2);
assert.sameValue(m.get("a"), 1);
assert.sameValue(m.size, 2);
assert.sameValue(m.has("b"), true);
m.delete("a");
assert.sameValue(m.has("a"), false);

var s = new Set([1, 2, 2, 3]);
assert.sameValue(s.size, 3, "Set dedups");
s.add(4);
assert.sameValue(s.has(4), true);
