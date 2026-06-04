/*---
description: String matchAll-like collection and replace with named patterns
esid: sec-string.prototype.matchall
---*/
var nums = "a1b22c333".match(/\d+/g);
assert.sameValue(nums.join(","), "1,22,333");
assert.sameValue("HELLO world".replace(/[A-Z]/g, function (m) { return m.toLowerCase(); }), "hello world");
assert.sameValue("a,b;c".split(/[,;]/).join("-"), "a-b-c");
