/*---
description: Operator precedence and associativity
esid: sec-operator-precedence
---*/
assert.sameValue(2 + 3 * 4, 14);
assert.sameValue((2 + 3) * 4, 20);
assert.sameValue(2 ** 3 ** 2, 512, "** is right-associative");
assert.sameValue(-(2 ** 2), -4, "unary minus must be parenthesized with **");
assert.sameValue(10 - 5 - 2, 3, "- is left-associative");
assert.sameValue(true || false && false, true, "&& before ||");
assert.sameValue(1 + 2 + "3", "33", "+ left-to-right");
assert.sameValue("3" + 1 + 2, "312");
assert.sameValue(5 & 3 | 8, 9, "& before |");
assert.sameValue(2 < 3 === true, true);
assert.sameValue(typeof typeof 5, "string", "typeof of typeof");
