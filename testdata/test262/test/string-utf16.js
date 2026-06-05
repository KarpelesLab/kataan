/*---
description: Strings are UTF-16; length and indexing count code units
esid: sec-ecmascript-language-types-string-type
---*/
assert.sameValue("abc".length, 3);
assert.sameValue("café".length, 4, "BMP accented char is one unit");
assert.sameValue("\u{1F600}".length, 2, "astral char is two UTF-16 units");
assert.sameValue("a\u{1F600}b".length, 4);
assert.sameValue("\u{1F600}".charCodeAt(0), 0xD83D, "high surrogate");
assert.sameValue("\u{1F600}".charCodeAt(1), 0xDE00, "low surrogate");
assert.sameValue("\u{1F600}".codePointAt(0), 0x1F600, "combined code point");
assert.sameValue("a\u{1F600}b".codePointAt(1), 0x1F600);
assert.sameValue("a\u{1F600}b".charCodeAt(3), "b".charCodeAt(0), "index past the surrogate pair");
assert.sameValue("hello".charCodeAt(0), 104);
assert.sameValue(Number.isNaN("x".charCodeAt(5)), true, "out of range is NaN");
