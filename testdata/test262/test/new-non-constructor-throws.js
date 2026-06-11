/*---
description: new on a non-constructor native throws a catchable TypeError
esid: sec-evaluatenew
---*/
function throwsType(fn) {
  try { fn(); return false; } catch (e) { return e instanceof TypeError; }
}

// Symbol and BigInt are callable but not constructors.
assert.sameValue(throwsType(function () { return new Symbol(); }), true, "new Symbol()");
assert.sameValue(throwsType(function () { return new BigInt(1); }), true, "new BigInt()");

// Plain global functions are not constructors either.
assert.sameValue(throwsType(function () { return new isNaN(1); }), true, "new isNaN()");
assert.sameValue(throwsType(function () { return new parseInt("1"); }), true, "new parseInt()");

// The error is catchable (it does not abort) and the surrounding code continues.
var reached = false;
try { new Symbol("x"); } catch (e) { reached = true; }
assert.sameValue(reached, true, "catch block runs");

// Symbol() / BigInt() as ordinary calls still work.
assert.sameValue(typeof Symbol("x"), "symbol", "Symbol() call");
assert.sameValue(typeof BigInt(5), "bigint", "BigInt() call");

// Real constructors are unaffected.
assert.sameValue(new Map().size, 0, "new Map");
assert.sameValue(new Set().size, 0, "new Set");
assert.sameValue(new WeakMap() instanceof WeakMap, true, "new WeakMap");
assert.sameValue(new Array(3).length, 3, "new Array");
assert.sameValue(new Error("x").message, "x", "new Error");
