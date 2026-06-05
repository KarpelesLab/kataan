/*---
description: Array.of and reduce-based grouping
esid: sec-array.of
---*/
assert.sameValue(Array.of(7).length, 1, "Array.of(7) is [7]");
assert.sameValue(Array.of(1, 2, 3).join(","), "1,2,3");
assert.sameValue(Array.of().length, 0);
assert.sameValue(Array.of(undefined).length, 1);
var grouped = ["apple", "banana", "cherry", "avocado", "blueberry"].reduce(function (acc, fruit) {
  var key = fruit[0];
  (acc[key] = acc[key] || []).push(fruit);
  return acc;
}, {});
assert.sameValue(grouped.a.join(","), "apple,avocado");
assert.sameValue(grouped.b.join(","), "banana,blueberry");
assert.sameValue(grouped.c.join(","), "cherry");
var counts = "mississippi".split("").reduce(function (acc, c) {
  acc[c] = (acc[c] || 0) + 1;
  return acc;
}, {});
assert.sameValue(counts.s, 4);
assert.sameValue(counts.i, 4);
assert.sameValue(counts.p, 2);
assert.sameValue(counts.m, 1);
