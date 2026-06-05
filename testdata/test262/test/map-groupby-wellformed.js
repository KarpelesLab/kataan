/*---
description: Map.groupBy and String isWellFormed/toWellFormed
esid: sec-map.groupby
---*/
var g = Map.groupBy([1, 2, 3, 4, 5], function (x) { return x % 2 ? "odd" : "even"; });
assert.sameValue(g instanceof Map, true, "returns a Map");
assert.sameValue(g.get("odd").join(","), "1,3,5");
assert.sameValue(g.get("even").join(","), "2,4");
assert.sameValue(g.size, 2);
var keyA = { type: "a" };
var keyB = { type: "b" };
var byObj = Map.groupBy([1, 2, 3, 4], function (x) { return x <= 2 ? keyA : keyB; });
assert.sameValue(byObj.get(keyA).join(","), "1,2", "object keys preserved");
assert.sameValue(byObj.get(keyB).join(","), "3,4");
assert.sameValue(byObj.size, 2);
var empty = Map.groupBy([], function (x) { return x; });
assert.sameValue(empty.size, 0);
assert.sameValue("hello".isWellFormed(), true, "ascii is well-formed");
assert.sameValue("café".isWellFormed(), true, "accented is well-formed");
assert.sameValue("😀".isWellFormed(), true, "emoji is well-formed");
assert.sameValue("abc".toWellFormed(), "abc", "well-formed unchanged");
assert.sameValue("😀x".toWellFormed(), "😀x");
