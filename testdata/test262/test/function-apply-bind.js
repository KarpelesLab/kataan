/*---
description: Function.prototype.apply with array-likes, bind names, call/apply
esid: sec-function.prototype.apply
---*/
function sum() {
  var t = 0;
  for (var i = 0; i < arguments.length; i++) t += arguments[i];
  return t;
}
assert.sameValue(sum.apply(null, [1, 2, 3]), 6, "apply with array");
assert.sameValue(sum.apply(null, { length: 3, 0: 10, 1: 20, 2: 30 }), 60, "apply with array-like");
assert.sameValue(sum.apply(null, { length: 0 }), 0, "empty array-like");
function f() { return arguments.length; }
assert.sameValue(f.apply(null, { length: 5 }), 5, "length-only array-like");
assert.sameValue(Math.max.apply(null, [3, 1, 4, 1, 5]), 5, "apply to a native");
assert.sameValue(sum.call(null, 1, 2, 3), 6, "call with args");
function greet(greeting) { return greeting + " " + this.name; }
assert.sameValue(greet.call({ name: "X" }, "Hi"), "Hi X", "call this + args");
assert.sameValue(greet.apply({ name: "Y" }, ["Hey"]), "Hey Y", "apply this + args");
function foo() {}
assert.sameValue(foo.bind(null).name, "bound foo", "bound function name");
assert.sameValue(foo.bind(null).bind(null).name, "bound bound foo", "double bound name");
function withArgs(a, b, c) { return a + b + c; }
var bound = withArgs.bind(null, 1, 2);
assert.sameValue(bound(3), 6, "bind partial application");
assert.sameValue(bound.bind(null, 99) !== bound, true, "bind returns a new function");
var ctx = { m: 10 };
assert.sameValue([1, 2, 3].map(function (x) { return x * this.m; }, ctx).join(","), "10,20,30", "map thisArg");
