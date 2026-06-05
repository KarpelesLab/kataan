/*---
description: String comparison and ordering
esid: sec-relational-operators
---*/
assert.sameValue("a" < "b", true);
assert.sameValue("apple" < "banana", true);
assert.sameValue("Z" < "a", true, "uppercase before lowercase");
assert.sameValue("abc" === "abc", true);
assert.sameValue("abc" < "abd", true);
assert.sameValue("ab" < "abc", true, "prefix is less");
assert.sameValue("10" < "9", true, "lexicographic not numeric");
assert.sameValue("" < "a", true, "empty is least");
assert.sameValue("a".localeCompare("a"), 0);
assert.sameValue("a".localeCompare("b") < 0, true);
assert.sameValue("b".localeCompare("a") > 0, true);
assert.sameValue(["banana", "apple", "cherry"].sort().join(","), "apple,banana,cherry");
assert.sameValue("hello" > "hell", true);
assert.sameValue("Hello".toLowerCase() === "hello", true);
assert.sameValue("ABC" === "abc".toUpperCase(), true);
var strs = ["delta", "alpha", "charlie", "bravo"];
assert.sameValue(strs.sort()[0], "alpha");
