/*---
description: Array method chaining and fluent transformations
esid: sec-array.prototype.map
---*/
var result = [1, 2, 3, 4, 5, 6]
  .filter(function (x) { return x % 2 === 0; })
  .map(function (x) { return x * x; })
  .reduce(function (a, b) { return a + b; }, 0);
assert.sameValue(result, 56, "4 + 16 + 36");
var words = ["hello", "world", "foo"]
  .map(function (w) { return w.toUpperCase(); })
  .filter(function (w) { return w.length > 3; })
  .join("-");
assert.sameValue(words, "HELLO-WORLD");
var sum = Array.from({ length: 5 }, function (_, i) { return i + 1; })
  .reduce(function (a, b) { return a + b; }, 0);
assert.sameValue(sum, 15);
assert.sameValue([3, 1, 2].slice().sort().join(""), "123", "slice clones before sort");
