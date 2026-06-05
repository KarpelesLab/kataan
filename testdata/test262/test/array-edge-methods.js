/*---
description: Array reduce edge cases, flat with depth, and join with undefined
esid: sec-array.prototype.reduce
---*/
assert.sameValue([].reduce(function (a, b) { return a + b; }, 100), 100, "empty with initial");
var threw = false;
try { [].reduce(function (a, b) { return a + b; }); } catch (e) { threw = true; }
assert.sameValue(threw, true, "empty without initial throws");
assert.sameValue([1, [2, [3, [4]]]].flat(Infinity).join(","), "1,2,3,4");
assert.sameValue([1, null, 2, undefined, 3].join("-"), "1--2--3", "null/undefined become empty");
assert.sameValue([1, 2, 3].reduceRight(function (a, b) { return a + "" + b; }), "321");
assert.sameValue(Array.from([1, 2, 3].entries()).length, 3, "entries iterator");
assert.sameValue([5, 4, 3, 2, 1].sort().join(","), "1,2,3,4,5");
