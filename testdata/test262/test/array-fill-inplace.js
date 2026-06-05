/*---
description: fill and copyWithin mutate and return the same array
esid: sec-array.prototype.fill
---*/
var a = [1, 2, 3, 4];
var fa = a.fill(0, 1, 3);
assert.sameValue(a.join(","), "1,0,0,4", "fill mutates");
assert.sameValue(a === fa, true, "fill returns same array");
var b = [1, 2, 3, 4, 5];
var cb = b.copyWithin(0, 3);
assert.sameValue(b.join(","), "4,5,3,4,5", "copyWithin mutates");
assert.sameValue(b === cb, true, "copyWithin returns same array");
var c = new Array(3).fill(7);
assert.sameValue(c.join(","), "7,7,7", "Array(n).fill");
