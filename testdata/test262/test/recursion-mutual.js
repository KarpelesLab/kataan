/*---
description: Recursion, mutual recursion, and hoisted forward references
esid: sec-function-definitions
---*/
function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); }
assert.sameValue(fact(5), 120);
function isEven(n) { return n === 0 ? true : isOdd(n - 1); }
function isOdd(n) { return n === 0 ? false : isEven(n - 1); }
assert.sameValue(isEven(10), true, "mutual recursion via hoisting");
assert.sameValue(isOdd(7), true);
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
assert.sameValue(fib(10), 55);
var acc = function self(n) { return n === 0 ? 0 : n + self(n - 1); };
assert.sameValue(acc(5), 15, "named function expression self-reference");
