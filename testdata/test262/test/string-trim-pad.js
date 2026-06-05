/*---
description: String trim, trimStart, trimEnd, padStart, padEnd
esid: sec-string.prototype.trim
---*/
assert.sameValue("  hello  ".trim(), "hello");
assert.sameValue("  hello  ".trimStart(), "hello  ");
assert.sameValue("  hello  ".trimEnd(), "  hello");
assert.sameValue("\t\n hello \n\t".trim(), "hello", "trims various whitespace");
assert.sameValue("".trim(), "");
assert.sameValue("nospaces".trim(), "nospaces");
assert.sameValue("5".padStart(3, "0"), "005");
assert.sameValue("5".padEnd(3, "0"), "500");
assert.sameValue("abc".padStart(10, "123").length, 10);
assert.sameValue("x".padStart(5), "    x", "default space");
assert.sameValue("hello".padStart(3), "hello", "already long enough");
assert.sameValue("ab".padStart(5, "xy"), "xyxab", "pad truncated to fit");
assert.sameValue("1".padStart(4, "0"), "0001");
assert.sameValue("100".padEnd(6, "-"), "100---");
assert.sameValue("  trim me  ".trim().length, 7);
