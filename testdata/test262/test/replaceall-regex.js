/*---
description: replaceAll with global regex and function replacers
esid: sec-string.prototype.replaceall
---*/
assert.sameValue("a.b.c".replaceAll(".", "-"), "a-b-c", "literal dot");
assert.sameValue("a1b2c3".replaceAll(/\d/g, "#"), "a#b#c#", "global regex");
assert.sameValue("hello".replaceAll("l", "L"), "heLLo");
assert.sameValue("aaa".replaceAll("a", function () { return "b"; }), "bbb", "function replacer");
assert.sameValue("x1y2".replaceAll(/(\d)/g, "[$1]"), "x[1]y[2]", "capture group");
var count = 0;
"a-b-c".replaceAll("-", function () { count++; return "_"; });
assert.sameValue(count, 2, "function called per match");
assert.sameValue("Hello World".replaceAll(/o/g, "0"), "Hell0 W0rld");
assert.sameValue("  spaces  ".replaceAll(" ", "."), "..spaces..");
assert.sameValue("test".replaceAll("", "-"), "-t-e-s-t-", "empty pattern");
