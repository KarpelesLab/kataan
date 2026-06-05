/*---
description: Strings are immutable; methods return new strings
esid: sec-string-objects
---*/
var s = "hello";
var upper = s.toUpperCase();
assert.sameValue(s, "hello", "original unchanged");
assert.sameValue(upper, "HELLO");
assert.sameValue("abc".concat("def"), "abcdef");
assert.sameValue("a-b-c".replace("-", "+"), "a+b-c", "first only");
assert.sameValue("  trim  ".trim(), "trim");
assert.sameValue("hello"[0], "h", "index access");
assert.sameValue("hello".slice(1, 3), "el");
s += " world";
assert.sameValue(s, "hello world", "reassignment");
assert.sameValue("aaa".length, 3);
assert.sameValue("hello world".split(" ").length, 2);
