/*---
description: Array callbacks receive (element, index, array)
esid: sec-array.prototype.foreach
---*/
var indices = [];
["a", "b", "c"].forEach(function (el, i, arr) { indices.push(i + ":" + el + ":" + arr.length); });
assert.sameValue(indices.join(","), "0:a:3,1:b:3,2:c:3", "forEach args");
var mapped = [10, 20, 30].map(function (el, i) { return el + i; });
assert.sameValue(mapped.join(","), "10,21,32", "map index");
var filtered = [1, 2, 3, 4, 5].filter(function (el, i) { return i % 2 === 0; });
assert.sameValue(filtered.join(","), "1,3,5", "filter by index");
var found = [5, 10, 15].find(function (el, i) { return i === 1; });
assert.sameValue(found, 10);
var reduced = [1, 2, 3].reduce(function (acc, el, i) { return acc + el * i; }, 0);
assert.sameValue(reduced, 0 * 1 + 1 * 2 + 2 * 3);
var someResult = ["x", "y", "z"].some(function (el, i) { return i === 2 && el === "z"; });
assert.sameValue(someResult, true);
var everyResult = [0, 1, 2].every(function (el, i) { return el === i; });
assert.sameValue(everyResult, true, "element equals index");
