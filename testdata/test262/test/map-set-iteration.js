/*---
description: Map/Set construction, lookup, forEach, and for-of iteration
esid: sec-map.prototype.foreach
---*/
var m = new Map([["a", 1], ["b", 2]]);
assert.sameValue(m.size, 2);
assert.sameValue(m.get("a"), 1);
assert.sameValue(m.has("b"), true);
var vals = [];
m.forEach(function (v) { vals.push(v); });
assert.sameValue(vals.join(","), "1,2");
var pairs = [];
for (var e of m) { pairs.push(e[0] + ":" + e[1]); }
assert.sameValue(pairs.join(","), "a:1,b:2");

var s = new Set([1, 2, 3, 2]);
assert.sameValue(s.size, 3);
var sum = 0;
s.forEach(function (x) { sum += x; });
assert.sameValue(sum, 6);
var seen = [];
for (var v of s) { seen.push(v); }
assert.sameValue(seen.join(","), "1,2,3");
