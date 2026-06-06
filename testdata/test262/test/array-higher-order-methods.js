/*---
description: Array iteration/search/transform methods
esid: sec-array.prototype-methods
---*/
// sort with a comparator (does not mutate the slice's source).
var a = [3, 1, 4, 1, 5, 9, 2, 6];
var sorted = a.slice().sort(function (x, y) { return x - y; });
assert.sameValue(sorted.join(","), "1,1,2,3,4,5,6,9", "numeric sort");

// find / findIndex.
assert.sameValue([1, 2, 3, 4].find(function (x) { return x > 2; }), 3, "find first > 2");
assert.sameValue([1, 2, 3, 4].findIndex(function (x) { return x > 2; }), 2, "findIndex");

// some / every.
assert.sameValue([1, 2, 3].some(function (x) { return x > 2; }), true, "some > 2");
assert.sameValue([1, 2, 3].every(function (x) { return x > 0; }), true, "every > 0");
assert.sameValue([1, 2, 3].every(function (x) { return x > 1; }), false, "not every > 1");

// includes / indexOf.
assert.sameValue([1, 2, 3].includes(2), true, "includes 2");
assert.sameValue([1, 2, 3].indexOf(3), 2, "indexOf 3");
assert.sameValue([1, 2, 3].indexOf(9), -1, "indexOf absent is -1");

// flat / flatMap.
assert.sameValue([[1, 2], [3, [4]]].flat().length, 4, "flat one level leaves the inner array");
  assert.sameValue([[1, 2], [3, [4]]].flat(2).length, 4, "flat depth 2 fully flattens");
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, x * 10]; }).join(","), "1,10,2,20,3,30", "flatMap");

// filter / map / reduce chain.
var r = [1, 2, 3, 4, 5].filter(function (x) { return x % 2; }).map(function (x) { return x * 10; }).reduce(function (s, x) { return s + x; }, 0);
assert.sameValue(r, 90, "filter-map-reduce: 10+30+50");
