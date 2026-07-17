/*---
description: Array.prototype.toLocaleString and BigInt.prototype.toLocaleString
esid: sec-array.prototype.tolocalestring
---*/
// Array.prototype.toLocaleString joins elements by ",", each via its locale form.
assert.sameValue([1, 2, 3].toLocaleString(), "1,2,3", "simple array");
assert.sameValue([1234567, 89].toLocaleString(), "1,234,567,89", "numbers get thousands grouping");
assert.sameValue([1, "a", true].toLocaleString(), "1,a,true", "mixed");
assert.sameValue([null, undefined, 5].toLocaleString(), ",,5", "null/undefined render empty");
assert.sameValue([].toLocaleString(), "", "empty array");
assert.sameValue([[1, 2], [3, 4]].toLocaleString(), "1,2,3,4", "nested arrays flatten via toString");

// It is a first-class method (readable / generic via call).
assert.sameValue(typeof [].toLocaleString, "function", "readable method");
assert.sameValue(Array.prototype.toLocaleString.call([7, 8, 9]), "7,8,9", "generic call");

// BigInt.prototype.toLocaleString groups the base-10 digits.
assert.sameValue((1234567n).toLocaleString(), "1,234,567", "bigint grouping");
assert.sameValue((5n).toLocaleString(), "5", "small bigint");
assert.sameValue((-1234567n).toLocaleString(), "-1,234,567", "negative bigint");
assert.sameValue(
  (123456789012345678901234567890n).toLocaleString(),
  "123,456,789,012,345,678,901,234,567,890",
  "huge bigint keeps full precision"
);
var bigArr = [1234n, 5n];
assert.sameValue(bigArr.toLocaleString(), "1,234,5", "bigint elements group too");

// Number.prototype.toLocaleString is unchanged, and -0 renders as "-0".
assert.sameValue((1234567).toLocaleString(), "1,234,567", "number grouping");
assert.sameValue((1234.56).toLocaleString(), "1,234.56", "number with fraction");
// ECMA-402 PartitionNumberPattern step 1 routes -0 through the NegativePattern
// (test262 intl402/.../format-negative-numbers.js asserts format(0) !== format(-0)),
// so Number.prototype.toLocaleString renders -0 as "-0", not "0".
assert.sameValue((-0).toLocaleString(), "-0", "negative zero renders as -0");
