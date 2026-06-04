/*---
description: Coercion in conditionals, equality, and unary operators
esid: sec-abstract-equality-comparison
---*/
assert.sameValue(1 + "2", "12");
assert.sameValue("3" * 2, 6);
assert.sameValue(!!"", false);
assert.sameValue(!!"x", true);
assert.sameValue(+"42", 42);
assert.sameValue(-"5", -5);
assert.sameValue(0 == false, true);
assert.sameValue("" == false, true);
assert.sameValue(null == undefined, true);
assert.sameValue(NaN === NaN, false);
assert.sameValue(void 0, undefined);
