/*---
description: Logical assignment operators
esid: sec-assignment-operators
---*/
var a = null;
a ??= 5;
assert.sameValue(a, 5);
var b = 0;
b ||= 10;
assert.sameValue(b, 10);
var c = 1;
c &&= 20;
assert.sameValue(c, 20);
