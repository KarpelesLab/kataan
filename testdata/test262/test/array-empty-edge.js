/*---
description: Array methods on empty arrays and edge counts
esid: sec-array.prototype
---*/
assert.sameValue([].every(function () { return false; }), true, "every on empty is true");
assert.sameValue([].some(function () { return true; }), false, "some on empty is false");
assert.sameValue([].filter(function () { return true; }).length, 0);
assert.sameValue([].map(function (x) { return x; }).length, 0);
assert.sameValue([].reduce(function (a, b) { return a + b; }, 100), 100, "reduce empty with initial");
assert.sameValue([].forEach(function () { throw new Error("never"); }), undefined);
assert.sameValue([].join(","), "");
assert.sameValue([].concat([]).length, 0);
assert.sameValue([].indexOf(1), -1);
assert.sameValue([].includes(1), false);
assert.sameValue([5].reduce(function (a, b) { return a + b; }), 5, "single element no initial");
assert.sameValue([].flat().length, 0);
var threw = false;
try { [].reduce(function (a, b) { return a + b; }); } catch (e) { threw = true; }
assert.sameValue(threw, true, "reduce empty no initial throws");
