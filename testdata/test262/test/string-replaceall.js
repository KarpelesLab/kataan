/*---
description: replaceAll with strings, regex, and function replacers
esid: sec-string.prototype.replaceall
---*/
assert.sameValue("a-b-c".replaceAll("-", "_"), "a_b_c");
assert.sameValue("aaa".replaceAll("a", "b"), "bbb");
assert.sameValue("hello".replaceAll("l", ""), "heo");
assert.sameValue("a.b.c".replaceAll(".", "/"), "a/b/c", "literal dot not regex");
assert.sameValue("x1y2z3".replaceAll(/\d/g, "#"), "x#y#z#", "global regex");
assert.sameValue("abc".replaceAll("", "-"), "-a-b-c-", "empty string inserts between");
assert.sameValue("no match".replaceAll("xyz", "abc"), "no match");
assert.sameValue("AAA".replaceAll("A", function () { return "B"; }), "BBB", "function replacer");
