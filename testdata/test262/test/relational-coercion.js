/*---
description: Relational operators apply ToPrimitive to object operands
esid: sec-abstract-relational-comparison
---*/
assert.sameValue([5] < 10, true, "array coerces to number");
assert.sameValue([20] > 10, true);
assert.sameValue([1] < [2], true, "two arrays compare as strings");
assert.sameValue([10] < [9], true, "lexicographic: '10' < '9'");
assert.sameValue([5] <= 5, true);
assert.sameValue([5] >= 5, true);
assert.sameValue(({} < 1), false, "plain object is NaN");
assert.sameValue("abc" < "abd", true, "string comparison");
assert.sameValue("10" < "9", true, "string lexicographic");
assert.sameValue(5 < 10, true, "plain numbers");
assert.sameValue("5" < 10, true, "string vs number is numeric");
var d1 = new Date(1000), d2 = new Date(2000);
assert.sameValue(d1 < d2, true, "dates compare by timestamp, not string");
assert.sameValue(d2 > d1, true);
assert.sameValue([3] < [20], false, "lexicographic: '3' > '20'");
assert.sameValue([100] > [99], false, "lexicographic: '100' < '99'");
var obj = { valueOf: function () { return 5; } };
assert.sameValue(obj < 10, true, "valueOf used in comparison");
assert.sameValue([1, 2] < [1, 3], true, "comma-joined string compare");
