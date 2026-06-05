/*---
description: reduceRight processing order and edge cases
esid: sec-array.prototype.reduceright
---*/
assert.sameValue([1, 2, 3, 4].reduceRight(function (a, b) { return a + b; }), 10);
assert.sameValue(["a", "b", "c"].reduceRight(function (a, b) { return a + b; }), "cba", "right to left");
assert.sameValue([1, 2, 3].reduceRight(function (a, b) { return a + b; }, 10), 16, "with initial");
var order = [];
[1, 2, 3].reduceRight(function (acc, x, i) { order.push(i); return acc; }, 0);
assert.sameValue(order.join(","), "2,1,0", "processes right to left");
assert.sameValue([5].reduceRight(function (a, b) { return a + b; }), 5, "single element");
assert.sameValue([].reduceRight(function (a, b) { return a + b; }, 100), 100, "empty with initial");
var nested = [[1, 2], [3, 4], [5]].reduceRight(function (acc, x) { return acc.concat(x); }, []);
assert.sameValue(nested.join(","), "5,3,4,1,2", "flatten right to left");
var threw = false;
try { [].reduceRight(function (a, b) { return a + b; }); } catch (e) { threw = true; }
assert.sameValue(threw, true, "empty no initial throws");
