/*---
description: String unicode, code points, and iteration
esid: sec-string-objects
---*/
assert.sameValue("café".length, 4);
assert.sameValue("abc".length, 3);
assert.sameValue([..."abc"].length, 3);
assert.sameValue("abc".codePointAt(0), 97);
assert.sameValue(String.fromCodePoint(97, 98), "ab");
assert.sameValue("Hello".split("").reverse().join(""), "olleH");
assert.sameValue("  x  ".trimStart(), "x  ");
assert.sameValue("  x  ".trimEnd(), "  x");
