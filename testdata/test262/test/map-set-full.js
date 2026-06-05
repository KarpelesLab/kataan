/*---
description: Map/Set clear, has, iteration, and entries
esid: sec-map.prototype.clear
---*/
var m = new Map();
m.set("a", 1).set("b", 2).set("c", 3);
assert.sameValue(m.size, 3);
m.clear();
assert.sameValue(m.size, 0, "clear empties the map");
assert.sameValue(m.has("a"), false);
m.set("x", 10);
m.set("x", 20);
assert.sameValue(m.get("x"), 20, "set overwrites");
assert.sameValue(m.size, 1, "same key not duplicated");
var s = new Set([1, 2, 3]);
s.clear();
assert.sameValue(s.size, 0);
s.add(5).add(5).add(6);
assert.sameValue(s.size, 2);
var entries = [];
new Map([["k", "v"]]).forEach(function (v, k) { entries.push(k + "=" + v); });
assert.sameValue(entries.join(","), "k=v");
