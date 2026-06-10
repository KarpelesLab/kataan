/*---
description: instanceof walks the prototype chain and throws TypeError for a non-callable RHS
esid: sec-instanceofoperator
---*/
// A non-callable / non-object RHS is a TypeError.
assert.throws(TypeError, function () { return {} instanceof {}; }, "object RHS not callable");
assert.throws(TypeError, function () { return 5 instanceof 5; }, "number RHS");
assert.throws(TypeError, function () { return {} instanceof undefined; }, "undefined RHS");
assert.throws(TypeError, function () { return {} instanceof "x"; }, "string RHS");

// instanceof consults the actual prototype chain, not just the recorded constructor.
var proto = {};
var obj = Object.create(proto);
function C() {}
C.prototype = proto;
assert.sameValue(obj instanceof C, true, "Object.create(C.prototype) is an instance");

// Reassigning .prototype after construction is reflected (the old link no longer matches).
function G() {}
var g = new G();
assert.sameValue(g instanceof G, true, "before reassignment");
G.prototype = {};
assert.sameValue(g instanceof G, false, "after reassignment, old instance no longer matches");

// Ordinary cases still hold.
class A {}
class B extends A {}
var b = new B();
assert.sameValue(b instanceof B, true, "subclass");
assert.sameValue(b instanceof A, true, "superclass");
assert.sameValue(b instanceof Object, true, "Object");
assert.sameValue([] instanceof Array, true, "array");
assert.sameValue(new Error() instanceof Error, true, "error");
assert.sameValue(4 instanceof { [Symbol.hasInstance](n) { return n % 2 === 0; } }, true, "Symbol.hasInstance");
// A primitive LHS is never an instance (and does not throw).
assert.sameValue(5 instanceof Number, false, "primitive LHS");
