/*---
description: The arguments object behavior
esid: sec-arguments-exotic-objects
---*/
function f() { return arguments.length; }
assert.sameValue(f(), 0);
assert.sameValue(f(1, 2, 3), 3);
function g() { return arguments[0] + arguments[1]; }
assert.sameValue(g(10, 20), 30);
function toArray() { return Array.prototype.slice.call(arguments); }
function variadic() {
  var args = [];
  for (var i = 0; i < arguments.length; i++) args.push(arguments[i] * 2);
  return args;
}
assert.sameValue(variadic(1, 2, 3).join(","), "2,4,6");
function spread() { return [...arguments].join("-"); }
assert.sameValue(spread("a", "b", "c"), "a-b-c", "spread arguments");
function reduceArgs() { return Array.prototype.reduce.call(arguments, function (a, b) { return a + b; }, 0); }
function namedAndArgs(first) { return first + ":" + arguments.length; }
assert.sameValue(namedAndArgs("x", "y", "z"), "x:3", "named param plus arguments");
function modifyArg(a) { a = 99; return arguments[0]; }
assert.sameValue(typeof modifyArg(1), "number");
