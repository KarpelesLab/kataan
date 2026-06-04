/*---
description: Array splice, unshift, shift, and copy semantics
esid: sec-properties-of-the-array-prototype-object
---*/
var a = [1, 2, 3, 4];
var removed = a.splice(1, 2, "x", "y", "z");
assert.sameValue(removed.join(","), "2,3");
assert.sameValue(a.join(","), "1,x,y,z,4");
var b = [2, 3];
b.unshift(1);
assert.sameValue(b.join(","), "1,2,3");
assert.sameValue(b.shift(), 1);
assert.sameValue(b.join(","), "2,3");
