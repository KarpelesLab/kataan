/*---
description: Array findLast, at, flatMap, includes(NaN), and fill
esid: sec-array.prototype.findlast
---*/
assert.sameValue([1, 2, 3, 4].findLast(function (x) { return x % 2 === 1; }), 3);
assert.sameValue([1, 2, 3].at(-1), 3);
assert.sameValue([1, 2, 3].at(-4), undefined);
assert.sameValue([1, 2, 3].flatMap(function (x) { return [x, x * 10]; }).join(","), "1,10,2,20,3,30");
assert.sameValue([1, NaN, 3].includes(NaN), true, "includes finds NaN");
assert.sameValue([1, 2, 3].indexOf(NaN), -1, "indexOf does not find NaN");
assert.sameValue([0, 0, 0].fill(7, 1).join(","), "0,7,7");
assert.sameValue(Array.from({ length: 3 }, function (_, i) { return i * i; }).join(","), "0,1,4");
