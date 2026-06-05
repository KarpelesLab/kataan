/*---
description: Array search methods indexOf, lastIndexOf, includes, find
esid: sec-array.prototype.indexof
---*/
var a = [10, 20, 30, 20, 10];
assert.sameValue(a.indexOf(20), 1);
assert.sameValue(a.lastIndexOf(20), 3);
assert.sameValue(a.indexOf(99), -1);
assert.sameValue(a.includes(30), true);
assert.sameValue(a.includes(99), false);
assert.sameValue(a.find(function (x) { return x > 15; }), 20);
assert.sameValue(a.findIndex(function (x) { return x > 15; }), 1);
assert.sameValue(a.findLast(function (x) { return x > 15; }), 20);
assert.sameValue(a.findLastIndex(function (x) { return x > 15; }), 3);
assert.sameValue([1, 2, 3].find(function (x) { return x > 10; }), undefined);
assert.sameValue([1, 2, 3].findIndex(function (x) { return x > 10; }), -1);
assert.sameValue(["a", "b", "c"].indexOf("b"), 1);
assert.sameValue([1, 2, 3, 4, 5].filter(function (x) { return x % 2 === 0; }).length, 2);
assert.sameValue([1, 2, 3].some(function (x) { return x === 2; }), true);
assert.sameValue([1, 2, 3].every(function (x) { return x > 0; }), true);
var objs = [{ id: 1 }, { id: 2 }, { id: 3 }];
assert.sameValue(objs.find(function (o) { return o.id === 2; }).id, 2);
assert.sameValue(objs.findIndex(function (o) { return o.id === 3; }), 2);
