/*---
description: Array.from and Array.of
esid: sec-array.from
---*/
assert.sameValue(Array.from("abc").length, 3);
assert.sameValue(Array.from([1, 2, 3], function (x) { return x * 2; }).join(","), "2,4,6");
assert.sameValue(Array.of(1, 2, 3).length, 3);
