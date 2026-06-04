/*---
description: Core Array.prototype methods
esid: sec-properties-of-the-array-prototype-object
---*/
assert.sameValue([1, 2, 3, 4].reduce(function (a, b) { return a + b; }, 0), 10);
assert.sameValue([1, 2, 3].find(function (x) { return x > 1; }), 2);
assert.sameValue([1, 2, 3].some(function (x) { return x === 2; }), true);
assert.sameValue([1, 2, 3].every(function (x) { return x > 0; }), true);
assert.sameValue([3, 1, 2].sort().join(","), "1,2,3");
assert.sameValue([1, 2, 3].reverse().join(","), "3,2,1");
assert.sameValue([1, 2].concat([3, 4]).length, 4);
assert.sameValue([1, 2, 3].includes(2), true);
assert.sameValue([1, 2, 3].indexOf(3), 2);
assert.sameValue([1, 2, 3, 4].slice(1, 3).join(","), "2,3");
