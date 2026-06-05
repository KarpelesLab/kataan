/*---
description: String iteration, spread, and character access
esid: sec-string.prototype-@@iterator
---*/
var chars = [];
for (var c of "abc") chars.push(c);
assert.sameValue(chars.join("-"), "a-b-c", "for-of over a string");
assert.sameValue([..."hello"].length, 5, "spread");
assert.sameValue([..."hello"].reverse().join(""), "olleh");
assert.sameValue(Array.from("xyz").join(","), "x,y,z");
assert.sameValue("test".split("").map(function (c) { return c.toUpperCase(); }).join(""), "TEST");
var count = {};
for (var ch of "banana") count[ch] = (count[ch] || 0) + 1;
assert.sameValue(count.a, 3);
assert.sameValue(count.n, 2);
assert.sameValue(count.b, 1);
