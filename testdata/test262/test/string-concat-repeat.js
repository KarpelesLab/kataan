/*---
description: String concat, repeat, and building
esid: sec-string.prototype.concat
---*/
assert.sameValue("a".concat("b", "c"), "abc");
assert.sameValue("hello".concat(" ", "world"), "hello world");
assert.sameValue("".concat("x"), "x");
assert.sameValue("a".concat(), "a", "no args");
assert.sameValue("=".repeat(5), "=====");
assert.sameValue("ab".repeat(3), "ababab");
assert.sameValue("x".repeat(0), "");
assert.sameValue("".repeat(10), "");
var parts = ["a", "b", "c", "d"];
assert.sameValue(parts.reduce(function (acc, p) { return acc.concat(p); }, ""), "abcd");
var built = "";
for (var i = 0; i < 3; i++) built += "x";
assert.sameValue(built, "xxx");
assert.sameValue("abc".concat("def").concat("ghi"), "abcdefghi", "chained");
assert.sameValue(["-"].join("").repeat(5), "-----");
assert.sameValue("ab".repeat(2).length, 4);
assert.sameValue("0".repeat(8), "00000000");
