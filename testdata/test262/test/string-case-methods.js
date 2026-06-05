/*---
description: String case conversion, trim variants, and padding
esid: sec-string.prototype.touppercase
---*/
assert.sameValue("Hello World".toUpperCase(), "HELLO WORLD");
assert.sameValue("Hello World".toLowerCase(), "hello world");
assert.sameValue("MiXeD".toLowerCase(), "mixed");
assert.sameValue("  spaces  ".trim(), "spaces");
assert.sameValue("  left".trimStart(), "left");
assert.sameValue("right  ".trimEnd(), "right");
assert.sameValue("5".padStart(3, "0"), "005");
assert.sameValue("5".padEnd(3, "0"), "500");
assert.sameValue("abc".padStart(2), "abc", "no padding needed");
assert.sameValue("x".repeat(5), "xxxxx");
assert.sameValue("ab".repeat(0), "");
assert.sameValue("Hello".charAt(0).toLowerCase() + "Hello".slice(1), "hello");
