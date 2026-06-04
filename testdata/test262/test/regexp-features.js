/*---
description: RegExp groups, replace with callback, and global match
esid: sec-regexp-pattern
---*/
var m = "2026-06-05".match(/(\d{4})-(\d{2})-(\d{2})/);
assert.sameValue(m[1], "2026");
assert.sameValue(m[3], "05");
assert.sameValue("a1b2c3".replace(/\d/g, function (d) { return "[" + d + "]"; }), "a[1]b[2]c[3]");
assert.sameValue("hello world".replace(/o/g, "0"), "hell0 w0rld");
assert.sameValue(/^\d+$/.test("12345"), true);
assert.sameValue(/^\d+$/.test("12a45"), false);
assert.sameValue("a,b;c d".split(/[,; ]/).join("|"), "a|b|c|d");
