/*---
description: String comparison, localeCompare sign, and normalize idempotence
esid: sec-string.prototype.localecompare
---*/
assert.sameValue("a" < "b", true);
assert.sameValue("apple".localeCompare("banana") < 0, true);
assert.sameValue("b".localeCompare("a") > 0, true);
assert.sameValue("x".localeCompare("x"), 0);
assert.sameValue("abc".normalize() === "abc".normalize().normalize(), true);
