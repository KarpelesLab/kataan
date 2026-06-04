/*---
description: String replace with capture group references and patterns
esid: sec-string.prototype.replace
---*/
assert.sameValue("2026-06-05".replace(/(\d{4})-(\d{2})-(\d{2})/, "$3/$2/$1"), "05/06/2026");
assert.sameValue("John Smith".replace(/(\w+)\s(\w+)/, "$2 $1"), "Smith John");
assert.sameValue("hello".replace(/l/g, "L"), "heLLo");
assert.sameValue("a1b2".replace(/(\d)/g, "[$1]"), "a[1]b[2]");
assert.sameValue("test".replace(/t/, "$&$&"), "ttest", "$& is the match");
assert.sameValue("abcabc".match(/a/g).length, 2);
