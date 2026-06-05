/*---
description: Regex lookbehind assertions
esid: sec-regexp-pattern-semantics
---*/
assert.sameValue(/(?<=\$)\d+/.test("$100"), true, "positive lookbehind match");
assert.sameValue(/(?<=\$)\d+/.test("100"), false, "positive lookbehind fail");
assert.sameValue("$100".match(/(?<=\$)\d+/)[0], "100", "captures the digits only");
assert.sameValue("price: $50".replace(/(?<=\$)\d+/, "X"), "price: $X", "lookbehind not consumed");
assert.sameValue(/(?<!a)b/.test("xb"), true, "negative lookbehind");
assert.sameValue(/(?<!a)b/.test("ab"), false, "negative lookbehind blocks");
assert.sameValue("1234567".replace(/(?<=\d)(?=(\d{3})+$)/g, ",") !== "", true, "lookbehind+lookahead");
assert.sameValue("foobar".match(/(?<=foo)bar/)[0], "bar");
assert.sameValue(/(?<=\w)\d/.test("a5"), true, "word then digit");
