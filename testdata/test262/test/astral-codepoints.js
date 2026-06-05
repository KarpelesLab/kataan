/*---
description: Astral (supplementary plane) code point handling
esid: sec-string.prototype.codepointat
---*/
var emoji = "😀";  // U+1F600, a surrogate pair in UTF-16
assert.sameValue(emoji.length, 2, "astral char is 2 UTF-16 units");
assert.sameValue(emoji.codePointAt(0), 0x1F600, "full code point");
assert.sameValue([...emoji].length, 1, "iterator yields one code point");
assert.sameValue(String.fromCodePoint(0x1F600), emoji);
assert.sameValue("a😀b".length, 4);
assert.sameValue([..."a😀b"].length, 3, "spread counts code points");
assert.sameValue("a😀b".charCodeAt(0), 97);
var high = emoji.charCodeAt(0);
assert.sameValue(high >= 0xD800 && high <= 0xDBFF, true, "high surrogate");
assert.sameValue(Array.from("😀😁").length, 2, "Array.from counts code points");
var count = 0;
for (var ch of "😀x😁") count++;
assert.sameValue(count, 3, "for-of counts code points");
