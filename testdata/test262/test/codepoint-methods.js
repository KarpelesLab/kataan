/*---
description: codePointAt, fromCodePoint, and charCodeAt
esid: sec-string.prototype.codepointat
---*/
assert.sameValue("A".charCodeAt(0), 65);
assert.sameValue("A".codePointAt(0), 65);
assert.sameValue(String.fromCharCode(65), "A");
assert.sameValue(String.fromCodePoint(65), "A");
assert.sameValue(String.fromCharCode(104, 105), "hi");
assert.sameValue("abc".charCodeAt(1), 98);
assert.sameValue("hello".codePointAt(0), 104);
assert.sameValue("".charCodeAt(0) !== "".charCodeAt(0), true, "out of range is NaN");
assert.sameValue(Number.isNaN("x".charCodeAt(5)), true);
assert.sameValue("x".codePointAt(5), undefined, "codePointAt out of range");
var emoji = String.fromCodePoint(0x1F600);
assert.sameValue(emoji.length, 2, "astral char");
assert.sameValue(emoji.codePointAt(0), 0x1F600);
assert.sameValue(String.fromCodePoint(0x1F600, 0x1F601).length, 4);
assert.sameValue("ABC".split("").map(function (c) { return c.codePointAt(0); }).join(","), "65,66,67");
