/*---
description: Map and Set forEach and iteration details
esid: sec-map.prototype.foreach
---*/
var m = new Map([["a", 1], ["b", 2], ["c", 3]]);
var pairs = [];
m.forEach(function (value, key) { pairs.push(key + "=" + value); });
assert.sameValue(pairs.join(","), "a=1,b=2,c=3", "forEach order and args");
var sum = 0;
m.forEach(function (v) { sum += v; });
assert.sameValue(sum, 6);
var s = new Set([10, 20, 30]);
var collected = [];
s.forEach(function (value) { collected.push(value); });
assert.sameValue(collected.join(","), "10,20,30");
s.forEach(function (value, key) { assert.sameValue(value, key, "Set forEach value === key"); });
var keys = [...m.keys()];
assert.sameValue(keys.join(","), "a,b,c");
var values = [...m.values()];
assert.sameValue(values.join(","), "1,2,3");
var entries = [...m.entries()];
assert.sameValue(entries.length, 3);
assert.sameValue(entries[0].join("="), "a=1");
assert.sameValue([...s].join(","), "10,20,30", "Set spread");
