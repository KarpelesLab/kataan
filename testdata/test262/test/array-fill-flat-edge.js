/*---
description: Array indexOf from index, lastIndexOf, includes fromIndex
esid: sec-properties-of-the-array-prototype-object
---*/
assert.sameValue([1, 2, 3, 2, 1].indexOf(2, 2), 3);
assert.sameValue([1, 2, 3, 2, 1].lastIndexOf(2), 3);
assert.sameValue([1, 2, 3].includes(2, 2), false);
assert.sameValue(["a", "b", "c"].indexOf("z"), -1);
