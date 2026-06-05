/*---
description: Object.groupBy groups iterable items by a callback key
esid: sec-object.groupby
---*/
var byParity = Object.groupBy([1, 2, 3, 4, 5, 6], function (x) { return x % 2 === 0 ? "even" : "odd"; });
assert.sameValue(byParity.odd.join(","), "1,3,5");
assert.sameValue(byParity.even.join(","), "2,4,6");
var people = [
  { name: "Alice", age: 30 },
  { name: "Bob", age: 25 },
  { name: "Carol", age: 30 }
];
var byAge = Object.groupBy(people, function (p) { return p.age; });
assert.sameValue(byAge["30"].length, 2, "two people aged 30");
assert.sameValue(byAge["25"].length, 1);
assert.sameValue(byAge["30"][0].name, "Alice");
var byFirst = Object.groupBy(["apple", "avocado", "banana", "cherry"], function (s) { return s[0]; });
assert.sameValue(byFirst.a.join(","), "apple,avocado");
assert.sameValue(byFirst.b.join(","), "banana");
assert.sameValue(byFirst.c.join(","), "cherry");
var withIndex = Object.groupBy([10, 20, 30, 40], function (v, i) { return i < 2 ? "first" : "second"; });
assert.sameValue(withIndex.first.join(","), "10,20");
assert.sameValue(withIndex.second.join(","), "30,40");
var empty = Object.groupBy([], function (x) { return x; });
assert.sameValue(Object.keys(empty).length, 0);
var fromString = Object.groupBy("aabbc", function (c) { return c; });
assert.sameValue(fromString.a.length, 2);
