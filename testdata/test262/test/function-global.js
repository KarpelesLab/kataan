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

// The dynamic Function constructor builds a callable from runtime source.
var add = Function("a", "b", "return a + b");
assert.sameValue(typeof add, "function", "Function() returns a function");
assert.sameValue(add(2, 3), 5, "Function()-built function runs");
assert.sameValue(add instanceof Function, true, "result is a Function instance");
assert.sameValue(add.name, "anonymous", "Function()-built function is named 'anonymous'");
assert.sameValue(add.length, 2, "length reflects the formal parameter count");
