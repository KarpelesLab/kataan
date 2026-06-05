/*---
description: String split with limits, regex, and edge cases
esid: sec-string.prototype.split
---*/
assert.sameValue("a,b,c,d".split(",", 2).join("|"), "a|b", "limit");
assert.sameValue("a,b,c".split(",", 0).length, 0, "limit 0");
assert.sameValue("hello".split("").length, 5, "empty separator");
assert.sameValue("".split(",").length, 1, "empty string");
assert.sameValue("".split(",")[0], "", "empty string yields empty");
assert.sameValue("abc".split().length, 1, "no separator");
assert.sameValue("a-b-c".split("-").length, 3);
assert.sameValue("a,,b".split(",").length, 3, "consecutive separators");
assert.sameValue("a,,b".split(",")[1], "", "empty between");
assert.sameValue("one two  three".split(/\s+/).length, 3, "regex whitespace");
assert.sameValue("2024-06-15".split("-").join("/"), "2024/06/15");
assert.sameValue("a1b2c3".split(/\d/).join(""), "abc", "split on digits");
assert.sameValue(",a,b,".split(",").length, 4, "leading/trailing");
assert.sameValue("camelCase".split(/(?=[A-Z])/).join("-"), "camel-Case", "lookahead split");
