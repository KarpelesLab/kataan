/*---
description: Symbols as object property keys keep identity and stay non-enumerable
esid: sec-object-type
---*/
var a = Symbol("k");
var b = Symbol("k");
var o = {};
o[a] = "A";
o[b] = "B";
o.plain = "P";
assert.sameValue(o[a], "A");
assert.sameValue(o[b], "B");
assert.sameValue(o[a] !== o[b], true, "distinct symbols are distinct keys");
assert.sameValue(a in o, true);
assert.sameValue(Object.keys(o).join(","), "plain", "symbol keys are not enumerable");

var wk = Symbol.iterator;
o[wk] = "iter";
assert.sameValue(o[wk], "iter", "well-known symbol as key");
delete o[a];
assert.sameValue(o[a], undefined, "symbol key deletes");
assert.sameValue(o[b], "B", "other symbol key intact");
