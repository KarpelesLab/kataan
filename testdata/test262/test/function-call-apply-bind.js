/*---
description: Function.prototype.call, apply, and bind
esid: sec-function.prototype.call
---*/
function greet(greeting, punct) { return greeting + ", " + this.name + punct; }
var who = { name: "World" };
assert.sameValue(greet.call(who, "Hello", "!"), "Hello, World!");
assert.sameValue(greet.apply(who, ["Hi", "."]), "Hi, World.");

var bound = greet.bind(who, "Hey");
assert.sameValue(bound("?"), "Hey, World?", "bind fixes this and leading args");
assert.sameValue(typeof bound, "function");

function add(a, b, c) { return a + b + c; }
assert.sameValue(add.apply(null, [1, 2, 3]), 6);
var add10 = add.bind(null, 10);
assert.sameValue(add10(20, 30), 60, "bind supports partial application");
assert.sameValue(Math.max.apply(null, [3, 7, 2]), 7);
