/*---
description: Object.isFrozen/isSealed return true for non-object arguments; isExtensible false
esid: sec-object.isfrozen
---*/
// A primitive is reported as frozen and sealed (it has no mutable properties).
[5, "x", true, null, undefined, Symbol(), 10n].forEach(function (v) {
  assert.sameValue(Object.isFrozen(v), true, "isFrozen primitive");
  assert.sameValue(Object.isSealed(v), true, "isSealed primitive");
  assert.sameValue(Object.isExtensible(v), false, "isExtensible primitive");
});

// Ordinary objects are not frozen/sealed by default but are extensible.
assert.sameValue(Object.isFrozen({}), false, "plain object not frozen");
assert.sameValue(Object.isSealed({}), false, "plain object not sealed");
assert.sameValue(Object.isExtensible({}), true, "plain object extensible");

// Freezing/sealing is reflected.
var fr = {};
Object.freeze(fr);
assert.sameValue(Object.isFrozen(fr), true, "frozen");
assert.sameValue(Object.isSealed(fr), true, "frozen implies sealed");
assert.sameValue(Object.isExtensible(fr), false, "frozen not extensible");
var se = { a: 1 };
Object.seal(se);
assert.sameValue(Object.isSealed(se), true, "sealed");
assert.sameValue(Object.isFrozen(se), false, "sealed with a writable value is not frozen");

// Arrays and functions follow the object rules.
assert.sameValue(Object.isFrozen([]), false, "array not frozen");
assert.sameValue(Object.isFrozen(Object.freeze([1, 2])), true, "frozen array");
assert.sameValue(Object.isFrozen(Object.freeze([])), true, "frozen empty array");
assert.sameValue(Object.isExtensible(function () {}), true, "function extensible");
