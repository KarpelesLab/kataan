/*---
description: Recursion — factorial, fibonacci, mutual, deep
esid: sec-function-definitions
---*/
function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); }
assert.sameValue(factorial(5), 120);
assert.sameValue(factorial(0), 1);
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
assert.sameValue(fib(10), 55);
function isEven(n) { return n === 0 ? true : isOdd(n - 1); }
function isOdd(n) { return n === 0 ? false : isEven(n - 1); }
assert.sameValue(isEven(10), true, "mutual recursion");
assert.sameValue(isOdd(7), true);
function sumTo(n) { return n === 0 ? 0 : n + sumTo(n - 1); }
assert.sameValue(sumTo(100), 5050);
function ackermann(m, n) {
  if (m === 0) return n + 1;
  if (n === 0) return ackermann(m - 1, 1);
  return ackermann(m - 1, ackermann(m, n - 1));
}
assert.sameValue(ackermann(2, 3), 9, "ackermann");
function flatten(arr) {
  return arr.reduce(function (acc, x) {
    return acc.concat(Array.isArray(x) ? flatten(x) : x);
  }, []);
}
assert.sameValue(flatten([1, [2, [3, [4, 5]]]]).join(","), "1,2,3,4,5", "recursive flatten");
var depth = function count(n) { return n === 0 ? 0 : 1 + count(n - 1); };
assert.sameValue(depth(500), 500, "named function expression recursion");
