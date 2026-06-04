/*---
description: Type coercion edge cases in comparisons and arithmetic
esid: sec-abstract-equality-comparison
---*/
assert.sameValue(1 + "2", "12", "number + string");
assert.sameValue("3" * 2, 6, "string * number");
assert.sameValue(true + 1, 2, "boolean coerces to number");
assert.sameValue([] + [], "", "array to string");
assert.sameValue([1, 2] + [3], "1,23");
assert.sameValue(null == undefined, true);
assert.sameValue(null === undefined, false);
assert.sameValue(0 == false, true);
assert.sameValue("" == false, true);
assert.sameValue(NaN === NaN, false);
assert.sameValue(+"", 0);
assert.sameValue(+"  12  ", 12);
