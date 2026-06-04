/*---
description: for-in enumerates own and inherited enumerable string keys
esid: sec-for-in-and-for-of-statements
---*/
var o = { a: 1, b: 2, c: 3 };
var keys = [];
for (var k in o) keys.push(k);
assert.sameValue(keys.join(","), "a,b,c", "insertion order");
var sum = 0;
for (var k in o) sum += o[k];
assert.sameValue(sum, 6);
function Base() {}
Base.prototype.inherited = "x";
function Obj() { this.own = "y"; }
Obj.prototype = Object.create(Base.prototype);
var inst = new Obj();
var found = [];
for (var p in inst) found.push(p);
assert.sameValue(found.indexOf("own") >= 0, true, "own property enumerated");
assert.sameValue(inst.hasOwnProperty("own"), true);
assert.sameValue(inst.hasOwnProperty("inherited"), false, "inherited is not own");
