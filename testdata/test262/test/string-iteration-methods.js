/*---
description: String character iteration and code point methods
esid: sec-string.prototype
---*/
assert.sameValue("abc".split("").reverse().join(""), "cba");
assert.sameValue([..."hello"].filter(function (c) { return c !== "l"; }).join(""), "heo");
assert.sameValue("hello".split("").map(function (c) { return c.charCodeAt(0); }).join(","), "104,101,108,108,111");
var vowels = "education".split("").filter(function (c) { return "aeiou".includes(c); });
assert.sameValue(vowels.join(""), "euaio");
assert.sameValue(Array.from("abc").map(function (c) { return c.toUpperCase(); }).join(""), "ABC");
assert.sameValue("hello".charAt(0), "h");
assert.sameValue("hello".at(-1), "o");
assert.sameValue("hello".codePointAt(0), 104);
assert.sameValue(String.fromCharCode(104, 105), "hi");
assert.sameValue("a-b-c".split("-").map(function (s) { return s.toUpperCase(); }).join("."), "A.B.C");
var counted = "hello".split("").reduce(function (acc, c) { acc[c] = (acc[c] || 0) + 1; return acc; }, {});
assert.sameValue(counted.l, 2);
