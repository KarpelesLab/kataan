/*---
description: String search, match, matchAll-like behavior
esid: sec-string.prototype.search
---*/
assert.sameValue("hello world".search(/world/), 6);
assert.sameValue("hello".search(/xyz/), -1);
assert.sameValue("hello".search("ll"), 2, "search with string");
var m = "2024-06-05".match(/(\d{4})-(\d{2})-(\d{2})/);
assert.sameValue(m[0], "2024-06-05", "full match");
assert.sameValue(m[1], "2024");
assert.sameValue(m.length, 4, "match plus captures");
assert.sameValue("a1b2c3".match(/\d/g).length, 3, "global match all");
assert.sameValue("no digits".match(/\d/), null);
assert.sameValue("aaa".match(/a/g).join(""), "aaa");
assert.sameValue("Hello World".match(/\w+/g).join(","), "Hello,World");
