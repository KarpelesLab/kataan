/*---
description: String match/matchAll/search coerce a non-RegExp argument to a RegExp pattern
esid: sec-string.prototype.match
---*/
// A plain string argument becomes a RegExp pattern (regex metacharacters apply).
var m = "hello world".match("o");
assert.sameValue(m[0], "o", "match value");
assert.sameValue(m.index, 4, "match index");
assert.sameValue("axb".match("a.b")[0], "axb", "'.' is any-char in the coerced pattern");
assert.sameValue("abc".match("z"), null, "no match -> null");

// A number argument is coerced too.
assert.sameValue("a5b".match(5)[0], "5", "number coerced to pattern");

// search returns the index (or -1), coercing a string argument.
assert.sameValue("hello".search("ll"), 2, "search string");
assert.sameValue("abc".search("."), 0, "search '.' matches first char");
assert.sameValue("abc".search("z"), -1, "search no match -> -1");

// matchAll over a coerced (global) pattern.
assert.sameValue([..."a1b1c".matchAll("1")].length, 2, "matchAll string -> global");

// A RegExp argument and a custom Symbol.match still work.
assert.sameValue("a1b2".match(/\d/g).join(","), "1,2", "regex argument");
var custom = { [Symbol.match]: function () { return ["c"]; } };
assert.sameValue("x".match(custom)[0], "c", "custom Symbol.match");

// split and replace keep treating a string argument literally (no regex coercion).
assert.sameValue("a.b.c".split(".").join("|"), "a|b|c", "split literal '.'");
assert.sameValue("a.b".replace(".", "X"), "aXb", "replace literal '.'");
