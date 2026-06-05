/*---
description: IIFEs and module-pattern closures
esid: sec-function-definitions
---*/
var result = (function () { return 42; })();
assert.sameValue(result, 42, "IIFE returns value");
var counter = (function () {
  var count = 0;
  return { increment: function () { return ++count; }, get: function () { return count; } };
})();
counter.increment();
counter.increment();
assert.sameValue(counter.get(), 2, "module pattern private state");
var memoized = (function () {
  var cache = {};
  return function (n) {
    if (n in cache) return cache[n];
    return cache[n] = n * n;
  };
})();
assert.sameValue(memoized(5), 25);
assert.sameValue(memoized(5), 25, "cached result");
var x = (function (a, b) { return a + b; })(3, 4);
assert.sameValue(x, 7, "IIFE with args");
var namespace = (function () {
  var privateVar = "secret";
  return { reveal: function () { return privateVar; } };
})();
assert.sameValue(namespace.reveal(), "secret");
var sum = (function () { return [1, 2, 3].reduce(function (a, b) { return a + b; }, 0); })();
assert.sameValue(sum, 6);
