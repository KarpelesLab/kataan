/*---
description: Modern String methods (padStart, replaceAll, at, includes)
esid: sec-string.prototype.padstart
---*/
assert.sameValue("5".padStart(3, "0"), "005");
assert.sameValue("5".padEnd(3, "-"), "5--");
assert.sameValue("a-b-c".replaceAll("-", "+"), "a+b+c");
assert.sameValue("hello".at(-1), "o");
assert.sameValue("hello".at(0), "h");
assert.sameValue("hello world".includes("world"), true);
assert.sameValue("hello".repeat(3), "hellohellohello");
assert.sameValue("  trim  ".trim(), "trim");
