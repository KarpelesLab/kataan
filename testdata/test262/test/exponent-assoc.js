/*---
description: Exponentiation right-associativity and unary precedence
esid: sec-exp-operator
---*/
assert.sameValue(2 ** 3 ** 2, 512, "right associative");
assert.sameValue((2 ** 3) ** 2, 64);
assert.sameValue(-(2 ** 2), -4);
assert.sameValue(2 ** -1, 0.5);
var b = 2; b **= 3; b **= 2;
assert.sameValue(b, 64);
