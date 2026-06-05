/*---
description: Regex Unicode property escapes \p{...} and \P{...}
esid: sec-regexp-pattern-semantics
---*/
assert.sameValue(/\p{L}/.test("a"), true, "letter");
assert.sameValue(/\p{L}/.test("5"), false, "digit is not a letter");
assert.sameValue(/^\p{L}+$/.test("hello"), true);
assert.sameValue(/^\p{N}+$/.test("12345"), true, "all numbers");
assert.sameValue(/^\p{N}+$/.test("12a45"), false);
assert.sameValue(/\p{Lu}/.test("A"), true, "uppercase");
assert.sameValue(/\p{Lu}/.test("a"), false);
assert.sameValue(/\p{Ll}/.test("a"), true, "lowercase");
assert.sameValue(/\p{L}/.test("Ω"), true, "Greek omega is a letter");
assert.sameValue(/\p{L}/.test("é"), true, "accented e is a letter");
assert.sameValue(/\P{L}/.test("5"), true, "negated: digit is non-letter");
assert.sameValue(/\P{L}/.test("a"), false, "negated: letter is not non-letter");
assert.sameValue(/^[\p{L}\p{N}]+$/.test("abc123"), true, "property in class");
assert.sameValue(/^[\p{L}\p{N}]+$/.test("abc 123"), false, "space excluded");
assert.sameValue("a1b2c3".replace(/\p{N}/g, "#"), "a#b#c#", "replace numbers");
assert.sameValue("Hello World".replace(/\p{Lu}/g, "_"), "_ello _orld", "replace uppercase");
assert.sameValue("foo123bar".match(/\p{N}+/)[0], "123");
