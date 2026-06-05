/*---
description: Destructuring with default values
esid: sec-destructuring-assignment
---*/
function greet({ name = "Anonymous", greeting = "Hello" } = {}) {
  return greeting + ", " + name;
}
assert.sameValue(greet(), "Hello, Anonymous");
assert.sameValue(greet({ name: "Alice" }), "Hello, Alice");
assert.sameValue(greet({ name: "Bob", greeting: "Hi" }), "Hi, Bob");
var [a = 1, b = 2, c = 3] = [10, undefined];
assert.sameValue(a, 10);
assert.sameValue(b, 2, "default for undefined");
assert.sameValue(c, 3, "default for missing");
var { x = 5, y = 10 } = { x: 1 };
assert.sameValue(x, 1);
assert.sameValue(y, 10);
var { p: { q = 99 } = {} } = {};
assert.sameValue(q, 99, "nested default");
function sum([first = 0, second = 0] = []) { return first + second; }
assert.sameValue(sum([3, 4]), 7);
assert.sameValue(sum(), 0);
