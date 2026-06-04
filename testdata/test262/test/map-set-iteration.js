/*---
description: Map and Set iteration order and for-of
esid: sec-map-objects
---*/
var m = new Map();
m.set("a", 1); m.set("b", 2); m.set("c", 3);
var keys = [];
for (var k of m.keys()) keys.push(k);
assert.sameValue(keys.join(","), "a,b,c", "Map preserves insertion order");
var entries = [];
for (var e of m) entries.push(e[0] + "=" + e[1]);
assert.sameValue(entries.join(","), "a=1,b=2,c=3");
assert.sameValue(m.size, 3);
var s = new Set([3, 1, 2, 1, 3]);
assert.sameValue([...s].join(","), "3,1,2", "Set dedupes, keeps first-seen order");
assert.sameValue(s.size, 3);
