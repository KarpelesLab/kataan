/*---
description: replaceAll with string and regex, and replace global flag
esid: sec-string.prototype.replaceall
---*/
assert.sameValue("a.b.c".replaceAll(".", "-"), "a-b-c");
assert.sameValue("a1b2c3".replace(/\d/g, "#"), "a#b#c#");
assert.sameValue("aaa".replaceAll("a", "bb"), "bbbbbb");
