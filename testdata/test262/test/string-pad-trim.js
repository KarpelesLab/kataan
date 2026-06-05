/*---
description: String padStart/padEnd/trim/repeat edge cases
esid: sec-string.prototype.padstart
---*/
assert.sameValue("5".padStart(3, "0"), "005");
assert.sameValue("5".padStart(3), "  5", "default pad is space");
assert.sameValue("abc".padStart(2), "abc", "no pad when already long enough");
assert.sameValue("ab".padEnd(5, "xy"), "abxyx", "pad pattern truncated");
assert.sameValue("x".repeat(0), "", "repeat 0");
assert.sameValue("ab".repeat(3), "ababab");
assert.sameValue("\t hi \n".trim(), "hi");
assert.sameValue("aaa".replaceAll("a", "b"), "bbb");
assert.sameValue("a.b.c".split(".").join("/"), "a/b/c");
