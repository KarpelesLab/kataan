/*---
description: ToPrimitive (valueOf/toString) in + and string coercion
esid: sec-toprimitive
---*/
var money = { amount: 42, valueOf: function () { return this.amount; } };
assert.sameValue(money + 8, 50, "valueOf drives numeric +");
assert.sameValue(money * 2, 84, "valueOf in *");
var label = { toString: function () { return "tag"; } };
assert.sameValue("" + label, "tag", "toString in string +");
assert.sameValue(`${label}`, "tag", "toString in template");
assert.sameValue(label + "!", "tag!");
var both = { valueOf: function () { return 5; }, toString: function () { return "five"; } };
assert.sameValue(both + 1, 6, "valueOf preferred for default hint");
assert.sameValue(`${both}`, "five", "toString preferred for string hint");
