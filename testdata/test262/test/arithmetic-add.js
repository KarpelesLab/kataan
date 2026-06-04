/*---
description: Addition operator over numbers and string concatenation
esid: sec-addition-operator-plus
---*/
assert.sameValue(1 + 2, 3, "number addition");
assert.sameValue("a" + "b", "ab", "string concatenation");
assert.sameValue(1 + "2", "12", "number to string coercion");
assert.sameValue([1, 2] + "", "1,2", "array ToPrimitive");
