/*---
description: for-in enumerates own then inherited enumerable properties
esid: sec-for-in-and-for-of-statements
---*/
var proto = { inherited: 1, shared: 2 };
var obj = Object.create(proto);
obj.own = 3;
var keys = [];
for (var k in obj) keys.push(k);
assert.sameValue(keys.sort().join(","), "inherited,own,shared", "own + inherited enumerable");
var collected = {};
for (var k2 in obj) collected[k2] = obj[k2];
assert.sameValue(collected.own, 3);
assert.sameValue(collected.inherited, 1, "inherited value readable");
var empty = [];
for (var e in {}) empty.push(e);
assert.sameValue(empty.length, 0, "non-enumerable prototype methods are not enumerated");
var grandparent = { g: 1 };
var parent = Object.create(grandparent);
parent.p = 2;
var child = Object.create(parent);
child.c = 3;
var chain = [];
for (var x in child) chain.push(x);
assert.sameValue(chain.sort().join(","), "c,g,p", "three-level chain");
var shadow = Object.create({ value: "proto" });
shadow.value = "own";
var seen = [];
for (var s in shadow) seen.push(s);
assert.sameValue(seen.length, 1, "shadowed key enumerated once");
assert.sameValue(shadow.value, "own", "own shadows inherited");
var arr = [10, 20, 30];
var indices = [];
for (var i in arr) indices.push(i);
assert.sameValue(indices.join(","), "0,1,2", "array indices");
