/*---
description: String match, matchAll, and search with regexes
esid: sec-string.prototype.match
---*/
var m = "2024-06-05".match(/(\d+)-(\d+)-(\d+)/);
assert.sameValue(m[1], "2024");
assert.sameValue(m[2], "06");
assert.sameValue("a1b2c3".match(/\d/g).join(","), "1,2,3", "global match");
assert.sameValue("hello".match(/z/), null, "no match returns null");
assert.sameValue("hello world".search(/world/), 6);
assert.sameValue("hello".search(/z/), -1);
assert.sameValue("a,b,,c".split(",").length, 4, "empty fields preserved");
assert.sameValue("a1b2c3".replace(/[a-z]/g, "_"), "_1_2_3");
