/*---
description: Core String.prototype methods
esid: sec-properties-of-the-string-prototype-object
---*/
assert.sameValue("hello world".slice(0, 5), "hello");
assert.sameValue("hello".indexOf("l"), 2);
assert.sameValue("a,b,c".split(",").length, 3);
assert.sameValue("abc".toUpperCase(), "ABC");
assert.sameValue("  trim  ".trim(), "trim");
assert.sameValue("ab".repeat(3), "ababab");
assert.sameValue("hello".includes("ell"), true);
assert.sameValue("hello".startsWith("he"), true);
assert.sameValue("hello".endsWith("lo"), true);
assert.sameValue("5".padStart(3, "0"), "005");
