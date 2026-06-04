/*---
description: Array reduceRight and at
esid: sec-properties-of-the-array-prototype-object
---*/
assert.sameValue(["a", "b", "c"].reduceRight(function (acc, x) { return acc + x; }), "cba");
assert.sameValue([10, 20, 30].at(-1), 30);
assert.sameValue([10, 20, 30].at(0), 10);
