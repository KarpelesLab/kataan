/*---
description: Function.prototype.call, apply, bind
esid: sec-function.prototype.call
---*/
function greet(greeting, punct) { return greeting + ", " + this.name + punct; }
var obj = { name: "World" };
assert.sameValue(greet.call(obj, "Hello", "!"), "Hello, World!", "call with this and args");
assert.sameValue(greet.apply(obj, ["Hi", "."]), "Hi, World.", "apply with array");
var bound = greet.bind(obj, "Hey");
assert.sameValue(bound("?"), "Hey, World?", "bind partial");
function sum() { var t = 0; for (var i = 0; i < arguments.length; i++) t += arguments[i]; return t; }
assert.sameValue(sum.apply(null, [1, 2, 3, 4]), 10, "apply spreads array");
assert.sameValue(sum.call(null, 5, 6, 7), 18, "call passes args");
var counter = { count: 0, inc: function () { this.count++; return this.count; } };
var incFn = counter.inc.bind(counter);
assert.sameValue(incFn(), 1);
assert.sameValue(incFn(), 2, "bound this persists");
function multiply(a, b, c) { return a * b * c; }
var double = multiply.bind(null, 2);
assert.sameValue(double(3, 4), 24, "bind partial application");
assert.sameValue(Math.max.apply(null, [3, 1, 4, 1, 5]), 5, "apply to Math.max");
