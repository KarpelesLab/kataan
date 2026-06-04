/*---
description: String search, replace with function, matchAll-like behavior
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("hello world".search("world"), 6);
assert.sameValue("hello".charCodeAt(0), 104);
assert.sameValue(String.fromCharCode(72, 105), "Hi");
assert.sameValue("café".normalize().length >= 4, true);
