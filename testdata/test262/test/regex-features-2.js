/*---
description: RegExp features — anchors, quantifiers, character classes, groups
esid: sec-regexp-pattern
---*/
assert.sameValue(/^\d+$/.test("12345"), true);
assert.sameValue(/^\d+$/.test("12a45"), false);
assert.sameValue("hello world".replace(/\w+/g, function (w) { return w.length; }), "5 5");
assert.sameValue("a-b_c.d".split(/[-_.]/).join(","), "a,b,c,d");
assert.sameValue(/colou?r/.test("color"), true);
assert.sameValue(/colou?r/.test("colour"), true);
assert.sameValue("aaa".match(/a+/)[0], "aaa", "greedy");
assert.sameValue("aaa".match(/a+?/)[0], "a", "lazy");
assert.sameValue(/[A-Z]/.test("hello"), false);
assert.sameValue(/[A-Z]/.test("Hello"), true);
assert.sameValue("2024-06-05".replace(/(\d{4})-(\d{2})-(\d{2})/, "$3.$2.$1"), "05.06.2024");
var m = "key=value".match(/(\w+)=(\w+)/);
assert.sameValue(m[1] + ":" + m[2], "key:value");
