/*---
description: padStart, padEnd, repeat edge cases
esid: sec-string.prototype.padstart
---*/
assert.sameValue("5".padStart(3, "0"), "005");
assert.sameValue("5".padEnd(3, "0"), "500");
assert.sameValue("abc".padStart(2), "abc", "already long enough");
assert.sameValue("abc".padStart(6), "   abc", "default space pad");
assert.sameValue("1".padStart(5, "ab"), "abab1", "multi-char pad truncated");
assert.sameValue("1".padEnd(5, "xy"), "1xyxy");
assert.sameValue("x".repeat(0), "");
assert.sameValue("ab".repeat(3), "ababab");
assert.sameValue("".repeat(5), "");
assert.sameValue("-".repeat(10).length, 10);
assert.sameValue("abc".padStart(0), "abc");
assert.sameValue("7".padStart(3, "0").padEnd(5, "_"), "007__");
