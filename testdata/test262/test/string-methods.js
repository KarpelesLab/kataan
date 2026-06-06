/*---
description: String prototype methods (split/trim/pad/repeat/replace/slice)
esid: sec-string.prototype-methods
---*/
assert.sameValue("a,b,c".split(",").join("-"), "a-b-c", "split then join");
assert.sameValue("  hi  ".trim(), "hi", "trim both ends");
assert.sameValue("abc".padStart(5, "*"), "**abc", "padStart");
assert.sameValue("abc".padEnd(5, "*"), "abc**", "padEnd");
assert.sameValue("ab".repeat(3), "ababab", "repeat");
assert.sameValue("Hello".replace("l", "L"), "HeLlo", "replace first occurrence");
assert.sameValue("HELLO".toLowerCase(), "hello", "toLowerCase");
assert.sameValue("hello".toUpperCase(), "HELLO", "toUpperCase");
assert.sameValue("hello".charAt(1), "e", "charAt");
assert.sameValue("hello".charCodeAt(0), 104, "charCodeAt");
assert.sameValue("hello".slice(-3), "llo", "negative slice");
assert.sameValue("hello".substring(1, 3), "el", "substring");
assert.sameValue("hello".includes("ell"), true, "includes substring");
assert.sameValue("hello".startsWith("he"), true, "startsWith");
assert.sameValue("hello".endsWith("lo"), true, "endsWith");
assert.sameValue("hello".indexOf("l"), 2, "indexOf");
