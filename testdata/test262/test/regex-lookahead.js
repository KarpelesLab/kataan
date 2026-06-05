/*---
description: Regex lookahead, backreferences, anchors
esid: sec-regexp-pattern-semantics
---*/
assert.sameValue(/foo(?=bar)/.test("foobar"), true, "positive lookahead match");
assert.sameValue(/foo(?=bar)/.test("foobaz"), false, "positive lookahead fail");
assert.sameValue(/foo(?!bar)/.test("foobaz"), true, "negative lookahead");
assert.sameValue(/foo(?!bar)/.test("foobar"), false);
assert.sameValue("foobar".replace(/foo(?=bar)/, "X"), "Xbar", "lookahead not consumed");
assert.sameValue(/(\w)\1/.test("hello"), true, "backreference matches ll");
assert.sameValue(/(\w)\1/.test("abc"), false, "no repeated char");
assert.sameValue("hello".match(/(.)\1/)[0], "ll", "backreference capture");
assert.sameValue(/^abc$/.test("abc"), true, "anchors");
assert.sameValue(/^abc$/.test("xabc"), false);
assert.sameValue(/\bword\b/.test("a word here"), true, "word boundary");
assert.sameValue(/\bword\b/.test("password"), false, "no boundary");
assert.sameValue("a1b2c3".replace(/\d/g, "#"), "a#b#c#");
