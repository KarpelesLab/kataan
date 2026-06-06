/*---
description: charAt/charCodeAt/codePointAt treat a negative index as out of range
esid: sec-string.prototype.charat
---*/
// A negative index is out of range; NaN/no-arg coerce to index 0 (ToInteger).
assert.sameValue("hello".charAt(-1), "", "charAt(-1)");
assert.sameValue("hello".charAt(-5), "", "charAt(-5)");
assert.sameValue("hello".charAt(0), "h", "charAt(0)");
assert.sameValue("hello".charAt(99), "", "charAt past end");
assert.sameValue("hello".charAt(NaN), "h", "charAt(NaN) -> 0");
assert.sameValue("hello".charAt(), "h", "charAt() -> 0");

assert.sameValue("hello".charCodeAt(-1), NaN, "charCodeAt(-1)");
assert.sameValue("hello".charCodeAt(0), 104, "charCodeAt(0)");
assert.sameValue("hello".charCodeAt(99), NaN, "charCodeAt past end");
assert.sameValue("hello".charCodeAt(NaN), 104, "charCodeAt(NaN) -> 0");

assert.sameValue("a\u{1F4A9}".codePointAt(-1), undefined, "codePointAt(-1)");
assert.sameValue("a\u{1F4A9}".codePointAt(1), 0x1F4A9, "codePointAt astral");
assert.sameValue("a\u{1F4A9}".codePointAt(99), undefined, "codePointAt past end");
