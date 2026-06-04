/*---
description: The exponentiation operator
esid: sec-exp-operator
---*/
assert.sameValue(2 ** 10, 1024);
assert.sameValue(2 ** 0, 1);
var b = 3;
b **= 4;
assert.sameValue(b, 81, "compound exponentiation assignment");
