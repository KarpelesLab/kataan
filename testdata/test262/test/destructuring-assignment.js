/*---
description: Destructuring assignment (not declaration), swaps, defaults
esid: sec-destructuring-assignment
---*/
var a, b, c;
[a, b] = [1, 2];
assert.sameValue(a + "," + b, "1,2");
[a, b] = [b, a];
assert.sameValue(a + "," + b, "2,1", "swap via destructuring");
({ x: a, y: b } = { x: 10, y: 20 });
assert.sameValue(a + "," + b, "10,20", "object destructuring assignment");
[a, b, c = 99] = [1, 2];
assert.sameValue(c, 99, "default in assignment");
var arr = [1, 2, 3, 4];
var [first, ...others] = arr;
assert.sameValue(first, 1);
assert.sameValue(others.join(","), "2,3,4");
