/*---
description: seal/preventExtensions/isExtensible/isSealed behave correctly on arrays
esid: sec-object.seal
---*/
// A plain array is extensible and neither sealed nor frozen.
assert.sameValue(Object.isExtensible([1, 2, 3]), true, "plain array extensible");
assert.sameValue(Object.isSealed([1, 2, 3]), false, "plain array not sealed");

// seal: existing elements stay writable, but the array cannot grow.
var s = [1, 2, 3];
assert.sameValue(Object.seal(s), s, "seal returns the array");
s[0] = 9;
s[5] = 99;
assert.sameValue(s.join(","), "9,2,3", "existing element written, extension blocked");
assert.sameValue(s.length, 3, "length unchanged");
assert.sameValue(Object.isSealed(s), true, "isSealed");
assert.sameValue(Object.isExtensible(s), false, "sealed -> not extensible");
assert.sameValue(Object.isFrozen(s), false, "sealed != frozen");
try { s.push(4); } catch (e) {}
assert.sameValue(s.length, 3, "push blocked on sealed");

// preventExtensions: blocks growth but is not "sealed".
var p = [1, 2, 3];
Object.preventExtensions(p);
p[0] = 9;
p[5] = 99;
assert.sameValue(p.join(","), "9,2,3", "extension blocked");
assert.sameValue(Object.isExtensible(p), false, "not extensible");
assert.sameValue(Object.isSealed(p), false, "preventExtensions alone is not sealed");

// freeze implies sealed and non-extensible.
var f = Object.freeze([1, 2, 3]);
assert.sameValue(Object.isFrozen(f), true, "frozen");
assert.sameValue(Object.isSealed(f), true, "frozen -> sealed");
assert.sameValue(Object.isExtensible(f), false, "frozen -> not extensible");

// Plain objects are unaffected.
assert.sameValue(Object.isExtensible({}), true, "object extensible");
assert.sameValue(Object.isSealed(Object.seal({})), true, "object seal");
