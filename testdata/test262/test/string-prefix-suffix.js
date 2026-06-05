/*---
description: startsWith, endsWith, includes with positions
esid: sec-string.prototype.startswith
---*/
assert.sameValue("hello world".startsWith("hello"), true);
assert.sameValue("hello world".startsWith("world"), false);
assert.sameValue("hello world".startsWith("world", 6), true, "with position");
assert.sameValue("hello world".endsWith("world"), true);
assert.sameValue("hello world".endsWith("hello", 5), true, "endsWith with endPosition");
assert.sameValue("hello".includes("ell"), true);
assert.sameValue("hello".includes("xyz"), false);
assert.sameValue("hello".includes("lo", 3), true);
assert.sameValue("hello".includes("he", 1), false, "position excludes start");
assert.sameValue("café".startsWith("café"), true);
