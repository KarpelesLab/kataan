/*---
description: String.prototype.lastIndexOf honors its fromIndex (the match may start at or before it)
esid: sec-string.prototype.lastindexof
---*/
var s = "Hello World"; // o's at index 4 and 7

// The match start must be at or before fromIndex.
assert.sameValue(s.lastIndexOf("o", 4), 4, "from 4 -> 4");
assert.sameValue(s.lastIndexOf("o", 7), 7, "from 7 -> 7");
assert.sameValue(s.lastIndexOf("o", 6), 4, "from 6 -> 4");
assert.sameValue(s.lastIndexOf("o", 3), -1, "from 3 -> none");
assert.sameValue(s.lastIndexOf("o"), 7, "no fromIndex -> last");
assert.sameValue(s.lastIndexOf("o", 100), 7, "huge fromIndex -> last");

// fromIndex is clamped to [0, len]; negatives become 0.
assert.sameValue("abc".lastIndexOf("a", -5), 0, "negative clamps to 0");
assert.sameValue("abcabc".lastIndexOf("a", 2), 0, "from 2 -> 0");
assert.sameValue("abcabc".lastIndexOf("a", 0), 0, "from 0 -> 0");

// NaN / undefined fromIndex search the whole string (+Infinity).
assert.sameValue("hello".lastIndexOf("l", NaN), 3, "NaN -> whole string");
assert.sameValue("hello".lastIndexOf("l"), 3, "undefined -> whole string");

// Multi-character needles: the *start* index is what is bounded.
assert.sameValue("ababab".lastIndexOf("ab"), 4, "last 'ab'");
assert.sameValue("ababab".lastIndexOf("ab", 3), 2, "last 'ab' starting <= 3");
assert.sameValue("ababab".lastIndexOf("ab", 2), 2, "last 'ab' starting <= 2");

// Empty needle returns min(fromIndex, len); a not-found / too-long needle is -1.
assert.sameValue("abc".lastIndexOf(""), 3, "empty needle");
assert.sameValue("abc".lastIndexOf("", 1), 1, "empty needle from 1");
assert.sameValue("abc".lastIndexOf("z"), -1, "not found");
assert.sameValue("abc".lastIndexOf("abcd"), -1, "needle longer than string");
assert.sameValue("".lastIndexOf("x"), -1, "needle in empty string");

// indexOf (forward) is unaffected.
assert.sameValue(s.indexOf("o", 5), 7, "indexOf with fromIndex");
