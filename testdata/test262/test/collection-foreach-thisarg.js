/*---
description: Map/Set forEach with thisArg and iteration
esid: sec-map.prototype.foreach
---*/
var ctx = { multiplier: 10 };
var m = new Map([["a", 1], ["b", 2]]);
var results = [];
m.forEach(function (v, k) { results.push(k + ":" + v * this.multiplier); }, ctx);
assert.sameValue(results.join(","), "a:10,b:20", "Map forEach thisArg");
var s = new Set([1, 2, 3]);
var sum = 0;
s.forEach(function (v) { sum += v * this.multiplier; }, ctx);
assert.sameValue(sum, 60, "Set forEach thisArg");
var keys = [];
new Map([["x", 1], ["y", 2], ["z", 3]]).forEach(function (v, k) { keys.push(k); });
assert.sameValue(keys.join(""), "xyz", "iteration order");
var entries = [];
new Map([[1, "a"], [2, "b"]]).forEach(function (v, k) { entries.push(k + "=" + v); });
assert.sameValue(entries.join(","), "1=a,2=b");
var setValues = [];
new Set(["p", "q"]).forEach(function (v) { setValues.push(v); });
assert.sameValue(setValues.join(","), "p,q");
