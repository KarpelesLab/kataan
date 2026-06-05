/*---
description: Map preserves insertion order across operations
esid: sec-map-objects
---*/
var m = new Map();
m.set("c", 3); m.set("a", 1); m.set("b", 2);
assert.sameValue([...m.keys()].join(","), "c,a,b", "insertion order");
m.set("a", 99);
assert.sameValue([...m.keys()].join(","), "c,a,b", "update keeps position");
assert.sameValue(m.get("a"), 99);
m.delete("a");
m.set("a", 1);
assert.sameValue([...m.keys()].join(","), "c,b,a", "re-add goes to end");
var entries = [...m.entries()].map(function (e) { return e[0] + e[1]; });
assert.sameValue(entries.join(","), "c3,b2,a1");
var keyTypes = new Map();
keyTypes.set(1, "num");
keyTypes.set("1", "str");
assert.sameValue(keyTypes.size, 2, "1 and '1' are distinct keys");
