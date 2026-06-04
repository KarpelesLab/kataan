/*---
description: More String.prototype methods
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("5".padEnd(3, "-"), "5--");
assert.sameValue("abcabc".replaceAll("a", "X"), "XbcXbc");
assert.sameValue("hello".charAt(1), "e");
assert.sameValue("hello".substring(1, 3), "el");
assert.sameValue("a-b-c".lastIndexOf("-"), 3);
assert.sameValue("HELLO".toLowerCase(), "hello");
assert.sameValue(String(42), "42");
assert.sameValue("café".length, 4);
