/*---
description: slice, substring, substr with negative and out-of-range args
esid: sec-string.prototype.slice
---*/
var s = "hello world";
assert.sameValue(s.slice(0, 5), "hello");
assert.sameValue(s.slice(-5), "world", "negative start");
assert.sameValue(s.slice(-5, -1), "worl");
assert.sameValue(s.slice(6), "world");
assert.sameValue(s.substring(0, 5), "hello");
assert.sameValue(s.substring(5, 0), "hello", "substring swaps args");
assert.sameValue(s.substring(-3, 5), "hello", "negative clamps to 0");
assert.sameValue(s.slice(100), "", "out of range");
assert.sameValue(s.slice(3, 1), "", "start after end");
assert.sameValue("abc".charAt(1), "b");
assert.sameValue("abc".charAt(10), "", "out of range charAt");
assert.sameValue("abcdef".slice(2, 4), "cd");
