/*---
description: String at and codePointAt and concat
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("hello".at(-1), "o");
assert.sameValue("hello".at(0), "h");
assert.sameValue("abc".concat("def"), "abcdef");
assert.sameValue("a".repeat(0), "");
