/*---
description: String matchAll and global match iteration
esid: sec-string.prototype.matchall
---*/
var matches = [...("a1b2c3".matchAll(/([a-z])(\d)/g))];
assert.sameValue(matches.length, 3, "three matches");
assert.sameValue(matches[0][0], "a1", "full match");
assert.sameValue(matches[0][1], "a", "group 1");
assert.sameValue(matches[0][2], "1", "group 2");
assert.sameValue(matches[1][1], "b");
assert.sameValue(matches[2][2], "3");
var words = [...("hello world foo".matchAll(/\w+/g))].map(function (m) { return m[0]; });
assert.sameValue(words.join(","), "hello,world,foo");
var dates = "2024-01 2024-02".match(/\d{4}-\d{2}/g);
assert.sameValue(dates.length, 2);
assert.sameValue(dates[0], "2024-01");
var noMatch = [...("abc".matchAll(/\d/g))];
assert.sameValue(noMatch.length, 0, "no matches");
