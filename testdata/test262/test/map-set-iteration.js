/*---
description: Map/Set keys/values/entries iterators, forEach, and for-of
esid: sec-map.prototype.entries
---*/
var m = new Map([["a", 1], ["b", 2]]);
assert.sameValue([...m.keys()].join(","), "a,b");
assert.sameValue([...m.values()].join(","), "1,2");
assert.sameValue(m.entries().length, 2);
var vals = [];
m.forEach(function (v) { vals.push(v); });
assert.sameValue(vals.join(","), "1,2");
var pairs = [];
for (var e of m) { pairs.push(e[0] + ":" + e[1]); }
assert.sameValue(pairs.join(","), "a:1,b:2");

var s = new Set([1, 2, 3, 2]);
assert.sameValue([...s.values()].join(","), "1,2,3");
assert.sameValue([...s.keys()].join(","), "1,2,3");
var sum = 0;
s.forEach(function (x) { sum += x; });
assert.sameValue(sum, 6);
