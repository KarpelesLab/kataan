/*---
description: Abstract relational and equality comparison coercion
esid: sec-abstract-relational-comparison
---*/
assert.sameValue("10" < "9", true, "string comparison is lexicographic");
assert.sameValue(10 < 9, false);
assert.sameValue("10" < 9, false, "string vs number coerces to number");
assert.sameValue(null == 0, false, "null only equals undefined");
assert.sameValue(undefined == null, true);
assert.sameValue("" == 0, true);
assert.sameValue("0" == 0, true);
assert.sameValue([] == 0, true, "empty array coerces to 0");
assert.sameValue([1] == 1, true);
assert.sameValue(true == 1, true);
assert.sameValue(false == "", true);
assert.sameValue(NaN == NaN, false);
