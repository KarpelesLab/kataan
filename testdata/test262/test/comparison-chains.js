/*---
description: Comparison results, NaN handling, and ordering
esid: sec-relational-operators
---*/
assert.sameValue(1 < 2, true);
assert.sameValue(2 <= 2, true);
assert.sameValue(NaN < 1, false);
assert.sameValue(NaN > 1, false);
assert.sameValue(NaN >= NaN, false, "NaN comparisons are always false");
assert.sameValue("apple" < "banana", true, "string ordering");
assert.sameValue("Z" < "a", true, "uppercase before lowercase");
assert.sameValue("10" < "9", true, "string lexicographic");
assert.sameValue(10 < 9, false, "numeric");
assert.sameValue(Infinity > 1e308, true);
assert.sameValue(-Infinity < -1e308, true);
assert.sameValue(0 < 0.0000001, true);
