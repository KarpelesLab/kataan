/*---
description: String split with limit, regex, and edge cases
esid: sec-string.prototype.split
---*/
assert.sameValue("a,b,c,d".split(",", 2).join("|"), "a|b", "limit");
assert.sameValue("a1b2c3".split(/\d/).join("|"), "a|b|c|", "split on regex");
assert.sameValue("hello".split("").join("-"), "h-e-l-l-o", "split into chars");
assert.sameValue("".split(",").length, 1, "empty string splits to one empty");
assert.sameValue("a,,b".split(",").length, 3, "consecutive separators");
assert.sameValue("abc".split("x").join("|"), "abc", "no separator found");
assert.sameValue("2024-06-05".split("-").length, 3);
assert.sameValue("a b  c".split(/\s+/).join(","), "a,b,c", "regex whitespace");
