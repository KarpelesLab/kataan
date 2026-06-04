/*---
description: String search, slice, and case methods
esid: sec-string.prototype.indexof
---*/
assert.sameValue("hello world".indexOf("o"), 4);
assert.sameValue("hello world".lastIndexOf("o"), 7);
assert.sameValue("hello".slice(1, 3), "el");
assert.sameValue("hello".slice(-3), "llo");
assert.sameValue("hello".substring(1, 3), "el");
assert.sameValue("Hello".toUpperCase(), "HELLO");
assert.sameValue("Hello".toLowerCase(), "hello");
assert.sameValue("a,b,c".split(",").length, 3);
assert.sameValue("hello".charAt(1), "e");
assert.sameValue("hello".charCodeAt(0), 104);
assert.sameValue(String.fromCharCode(104, 105), "hi");
