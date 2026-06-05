/*---
description: Type coercion in operators and comparisons
esid: sec-abstract-operations
---*/
assert.sameValue(1 + "2", "12", "number + string");
assert.sameValue("3" * 2, 6, "string * number");
assert.sameValue("5" - 1, 4, "string - number");
assert.sameValue([1, 2] + [3], "1,23", "array concat coercion");
assert.sameValue(true + 1, 2, "boolean to number");
assert.sameValue(0 == "", true, "loose equality");
assert.sameValue(0 == false, true);
assert.sameValue(null == undefined, true);
assert.sameValue(null === undefined, false);
assert.sameValue(1 < "2", true, "numeric string comparison");
assert.sameValue("10" < "9", true, "lexicographic string comparison");
assert.sameValue(!!"", false, "empty string is falsy");
assert.sameValue(!!"a", true);
assert.sameValue(!!0, false);
assert.sameValue(!!null, false);
assert.sameValue(+"42", 42, "unary plus");
assert.sameValue(+"", 0);
assert.sameValue(+true, 1);
assert.sameValue(String(123), "123");
assert.sameValue(Number("3.14"), 3.14);
assert.sameValue([] == false, true, "empty array loose equals false");
assert.sameValue("" + null, "null");
assert.sameValue("" + undefined, "undefined");
assert.sameValue("" + [1, 2, 3], "1,2,3");
assert.sameValue("" + {}, "[object Object]");
