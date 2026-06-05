/*---
description: String split with regex capture groups in the separator
esid: sec-string.prototype.split
---*/
assert.sameValue("a1b2c3".split(/(\d)/).join("|"), "a|1|b|2|c|3|", "captures included");
assert.sameValue("2024-06-05".split("-").join("/"), "2024/06/05");
assert.sameValue("aXbYc".split(/[XY]/).join(","), "a,b,c");
assert.sameValue("hello".split("").length, 5);
assert.sameValue("a,b,c".split(",", 2).join("|"), "a|b", "limit");
assert.sameValue("".split("").length, 0, "empty string empty separator");
assert.sameValue("abc".split().length, 1, "no args is the whole string");
