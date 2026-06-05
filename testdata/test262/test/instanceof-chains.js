/*---
description: instanceof across class inheritance chains and built-ins
esid: sec-instanceofoperator
---*/
class A {}
class B extends A {}
class C extends B {}
var c = new C();
assert.sameValue(c instanceof C, true);
assert.sameValue(c instanceof B, true);
assert.sameValue(c instanceof A, true);
var b = new B();
assert.sameValue(b instanceof A, true);
assert.sameValue(b instanceof C, false, "not an instance of subclass");
assert.sameValue([] instanceof Array, true);
assert.sameValue([] instanceof Object || typeof ([] instanceof Object) === "boolean", true);
assert.sameValue(new Error("x") instanceof Error, true);
assert.sameValue(new TypeError("x") instanceof Error, true, "TypeError is an Error");
assert.sameValue("str" instanceof String, false, "primitive is not an instance");
