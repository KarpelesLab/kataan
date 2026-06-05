/*---
description: String replace with empty matches and global flags
esid: sec-string.prototype.replace
---*/
assert.sameValue("abc".replace(/x*/g, "-"), "-a-b-c-", "empty matches between chars");
assert.sameValue("abc".replace("", "X"), "Xabc", "empty string pattern");
assert.sameValue("hello".replace(/l/g, ""), "heo", "remove matches");
assert.sameValue("aaa".replace(/a/g, "bb"), "bbbbbb");
assert.sameValue("test".replace(/(?:)/, "X"), "Xtest", "empty group");
assert.sameValue("a.b.c".replace(/\./g, "/"), "a/b/c");
assert.sameValue("123".replace(/\d/g, function (d) { return "[" + d + "]"; }), "[1][2][3]");
assert.sameValue("  trim  ".replace(/^\s+|\s+$/g, ""), "trim", "trim via regex");
assert.sameValue("camelCase".replace(/([A-Z])/g, " $1"), "camel Case");
assert.sameValue("hello world".replace(/(\w+)/g, function (m) { return m.toUpperCase(); }), "HELLO WORLD");
