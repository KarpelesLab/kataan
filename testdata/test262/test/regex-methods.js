/*---
description: RegExp test/match/replace
esid: sec-regexp-regular-expression-objects
---*/
assert.sameValue(/\d+/.test("abc123"), true);
assert.sameValue("abc123def".match(/\d+/)[0], "123");
assert.sameValue("a1b2c3".replace(/\d/g, "#"), "a#b#c#");
assert.sameValue("2024-01-15".split(/-/).length, 3);
