/*---
description: Exponentiation operator associativity and precedence
esid: sec-exp-operator
---*/
assert.sameValue(2 ** 3, 8);
assert.sameValue(2 ** 3 ** 2, 512, "right associative");
assert.sameValue((2 ** 3) ** 2, 64, "explicit left grouping");
assert.sameValue(2 ** 10, 1024);
assert.sameValue(-(2 ** 2), -4);
assert.sameValue((-2) ** 2, 4);
assert.sameValue(2 ** -1, 0.5, "negative exponent");
assert.sameValue(4 ** 0.5, 2, "fractional exponent");
assert.sameValue(2 ** 0, 1);
assert.sameValue(0 ** 0, 1);
assert.sameValue(2 + 3 ** 2, 11, "** before +");
assert.sameValue(3 ** 2 * 2, 18, "** before *");
var x = 2;
x **= 3;
assert.sameValue(x, 8, "exponentiation assignment");
assert.sameValue(Math.pow(2, 3), 2 ** 3);
