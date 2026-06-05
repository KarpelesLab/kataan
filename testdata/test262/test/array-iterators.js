/*---
description: Array entries, keys, values iterators
esid: sec-array.prototype.entries
---*/
var a = ["x", "y", "z"];
var ks = [];
for (var k of a.keys()) ks.push(k);
assert.sameValue(ks.join(","), "0,1,2", "keys");
var vs = [];
for (var v of a.values()) vs.push(v);
assert.sameValue(vs.join(","), "x,y,z", "values");
var es = [];
for (var e of a.entries()) es.push(e[0] + ":" + e[1]);
assert.sameValue(es.join(","), "0:x,1:y,2:z", "entries");
assert.sameValue([...a.keys()].join(","), "0,1,2", "spread keys");
assert.sameValue(Array.from(a.entries()).length, 3);
var it = a.values();
assert.sameValue(it.next().value, "x", "iterator next");
assert.sameValue(it.next().value, "y");
