/*---
description: Object.freeze on an array makes its elements non-writable and blocks extension
esid: sec-object.freeze
---*/
var fa = [1, 2, 3];
assert.sameValue(Object.freeze(fa), fa, "freeze returns the array");
assert.sameValue(Object.isFrozen(fa), true, "array reports frozen");

// Element writes are rejected (value unchanged), in/out of range.
fa[0] = 9;
fa[5] = 99;
assert.sameValue(fa.join(","), "1,2,3", "element writes rejected");
assert.sameValue(fa.length, 3, "length unchanged (no extension)");

// push on a frozen array does not grow it.
try { fa.push(4); } catch (e) {}
assert.sameValue(fa.length, 3, "push blocked");
assert.sameValue(fa.join(","), "1,2,3", "still unchanged");

// A non-frozen array is unaffected.
var na = [1, 2, 3];
na[0] = 9;
na.push(4);
assert.sameValue(na.join(","), "9,2,3,4", "normal array writable");

// Frozen objects continue to work (regression guard).
var o = { a: 1 };
Object.freeze(o);
o.a = 9;
o.b = 2;
assert.sameValue(o.a, 1, "frozen object property unchanged");
assert.sameValue("b" in o, false, "frozen object not extended");
