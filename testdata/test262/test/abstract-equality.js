/*---
description: Abstract equality (==) coercion rules
esid: sec-abstract-equality-comparison
---*/
assert.sameValue(1 == "1", true, "number to string");
assert.sameValue(0 == false, true);
assert.sameValue(1 == true, true);
assert.sameValue(null == undefined, true, "null and undefined");
assert.sameValue(null == 0, false, "null is not 0");
assert.sameValue(undefined == 0, false);
assert.sameValue("" == 0, true, "empty string to 0");
assert.sameValue("  " == 0, true, "whitespace to 0");
assert.sameValue([] == "", true, "empty array to empty string");
assert.sameValue([] == 0, true, "empty array to 0");
assert.sameValue([1] == 1, true, "single element array");
assert.sameValue([1, 2] == "1,2", true, "array to joined string");
assert.sameValue(NaN == NaN, false);
assert.sameValue(null == false, false, "null is not false");
assert.sameValue("0" == false, true);
assert.sameValue(true == 1, true);
assert.sameValue("1" == true, true);
