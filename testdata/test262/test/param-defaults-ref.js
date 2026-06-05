/*---
description: Default parameters referencing earlier parameters
esid: sec-function-definitions
---*/
function f(a, b = a * 2, c = a + b) { return a + b + c; }
assert.sameValue(f(1), 1 + 2 + 3, "defaults reference earlier params");
assert.sameValue(f(1, 5), 1 + 5 + 6);
assert.sameValue(f(1, 5, 10), 16);
function greet(name, greeting = "Hello", message = greeting + ", " + name) {
  return message;
}
assert.sameValue(greet("World"), "Hello, World");
assert.sameValue(greet("World", "Hi"), "Hi, World");
function withArray(arr = [], val = arr.length) { return val; }
assert.sameValue(withArray(), 0);
assert.sameValue(withArray([1, 2, 3]), 3);
function counter(start = 0, step = 1, end = start + step * 3) { return end; }
assert.sameValue(counter(), 3);
assert.sameValue(counter(10, 5), 25);
