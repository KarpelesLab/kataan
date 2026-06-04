/*---
description: reduce without initial value, flatMap, and chaining
esid: sec-array.prototype.reduce
---*/
assert.sameValue([1, 2, 3, 4].reduce(function (a, b) { return a + b; }), 10);
assert.sameValue([5].reduce(function (a, b) { return a + b; }), 5, "single element, no initial");
assert.sameValue([1, 2, 3].map(function (x) { return x * 2; }).filter(function (x) { return x > 2; }).join(","), "4,6");
assert.sameValue([[1, 2], [3]].reduce(function (a, b) { return a.concat(b); }, []).join(","), "1,2,3");
