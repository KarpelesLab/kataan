/*---
description: Function.prototype.bind with partial application and this
esid: sec-function.prototype.bind
---*/
function greet(greeting, punct) { return greeting + ", " + this.name + punct; }
var bound = greet.bind({ name: "World" }, "Hello");
assert.sameValue(bound("!"), "Hello, World!");
function add(a, b, c) { return a + b + c; }
var add5 = add.bind(null, 5);
assert.sameValue(add5(10, 20), 35);
var add5and10 = add.bind(null, 5, 10);
assert.sameValue(add5and10(20), 35);
var obj = { x: 10, getX: function () { return this.x; } };
var unbound = obj.getX;
var rebound = unbound.bind(obj);
assert.sameValue(rebound(), 10);
assert.sameValue(typeof bound, "function");
