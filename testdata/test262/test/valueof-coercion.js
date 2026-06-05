/*---
description: User valueOf/toString honored in operators; increment is numeric
esid: sec-toprimitive
---*/
var money = { amount: 5, valueOf: function () { return this.amount; } };
assert.sameValue(money - 2, 3, "valueOf in subtraction");
assert.sameValue(money * 2, 10);
assert.sameValue(money + 1, 6, "valueOf in addition");
assert.sameValue(money < 10, true, "valueOf in comparison");
assert.sameValue(money > 3, true);
assert.sameValue(~money, -6, "valueOf in bitwise not");
assert.sameValue(-money, -5, "valueOf in negation");
assert.sameValue(money & 3, 1, "valueOf in bitwise and");
assert.sameValue(money ** 2, 25, "valueOf in exponent");
var label = { toString: function () { return "tag"; } };
assert.sameValue(label + "!", "tag!", "toString in concatenation");
assert.sameValue("" + label, "tag");
assert.sameValue(`${label}`, "tag", "toString in template");
var s = "5";
assert.sameValue(++s, 6, "prefix increment is numeric");
assert.sameValue(s, 6);
var t = "3";
var u = t++;
assert.sameValue(u, 3, "postfix returns the numeric old value");
assert.sameValue(t, 4);
var arr = [10];
assert.sameValue(--arr, 9, "decrement coerces array");
var n = 0;
for (var i = 0; i < 5; i++) n += i;
assert.sameValue(n, 0 + 1 + 2 + 3 + 4, "numeric loop counter");
