/*---
description: Closures, counters, and private state patterns
esid: sec-closure
---*/
function createCounter(start) {
  var count = start || 0;
  return { increment: function () { return ++count; }, decrement: function () { return --count; }, value: function () { return count; } };
}
var c = createCounter(10);
assert.sameValue(c.increment(), 11);
assert.sameValue(c.increment(), 12);
assert.sameValue(c.decrement(), 11);
assert.sameValue(c.value(), 11);
var counters = [];
for (let i = 0; i < 3; i++) { counters.push(function () { return i; }); }
assert.sameValue(counters.map(function (f) { return f(); }).join(","), "0,1,2", "let binding per iteration");
function memoize(fn) {
  var cache = {};
  return function (n) { if (n in cache) return cache[n]; return cache[n] = fn(n); };
}
var calls = 0;
var square = memoize(function (n) { calls++; return n * n; });
assert.sameValue(square(5), 25);
assert.sameValue(square(5), 25);
assert.sameValue(calls, 1, "memoized, called once");
function once(fn) { var called = false, result; return function () { if (!called) { called = true; result = fn.apply(this, arguments); } return result; }; }
var init = once(function () { return Math.random(); });
assert.sameValue(init() === init(), true, "once returns same value");
