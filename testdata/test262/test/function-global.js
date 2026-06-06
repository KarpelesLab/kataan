/*---
description: the Function global supports typeof and instanceof for callables
---*/
assert.sameValue(typeof Function, "function", "typeof Function");

// Every callable value is an instance of Function.
assert.sameValue((function () {}) instanceof Function, true, "function expression");
assert.sameValue((() => {}) instanceof Function, true, "arrow");
assert.sameValue(Math.max instanceof Function, true, "native method");
class C {}
assert.sameValue(C instanceof Function, true, "class");
assert.sameValue((function () {}).bind(null) instanceof Function, true, "bound function");

// Non-callables are not.
assert.sameValue(({}) instanceof Function, false, "object");
assert.sameValue([] instanceof Function, false, "array");
assert.sameValue((5) instanceof Function, false, "number");

// The dynamic Function constructor (runtime code) throws rather than evaluating.
var threw = false;
try { Function("return 1"); } catch (e) { threw = e instanceof TypeError; }
assert.sameValue(threw, true, "dynamic Function() throws TypeError");
