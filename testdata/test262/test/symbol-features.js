/*---
description: Symbol creation, description, and uniqueness
esid: sec-symbol-objects
---*/
var s1 = Symbol("desc");
var s2 = Symbol("desc");
assert.sameValue(s1 === s2, false, "symbols are unique");
assert.sameValue(s1.description, "desc");
assert.sameValue(typeof s1, "symbol");
var noDesc = Symbol();
assert.sameValue(noDesc.description, undefined);
var forA = Symbol.for("shared");
var forB = Symbol.for("shared");
assert.sameValue(forA === forB, true, "Symbol.for returns the same symbol");
assert.sameValue(Symbol.keyFor(forA), "shared");
var obj = {};
var key = Symbol("key");
obj[key] = "value";
assert.sameValue(obj[key], "value", "symbol as property key");
assert.sameValue(Object.keys(obj).length, 0, "symbol keys excluded from Object.keys");
