/*---
description: Sparse arrays, holes, length manipulation
esid: sec-array-exotic-objects
---*/
var a = [1, , 3];
assert.sameValue(a.length, 3);
assert.sameValue(a[1], undefined);
var b = [];
b[5] = "x";
assert.sameValue(b.length, 6, "assigning past end extends length");
var c = [1, 2, 3, 4, 5];
c.length = 3;
assert.sameValue(c.join(","), "1,2,3", "shrinking length truncates");
var d = [1, 2, 3];
assert.sameValue(d.reverse().join(","), "3,2,1");
assert.sameValue([3, 1, 2].slice(1).join(","), "1,2");
