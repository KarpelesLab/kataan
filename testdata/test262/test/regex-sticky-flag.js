/*---
description: Regex sticky (y) flag anchors matches at the start position
esid: sec-regexp-pattern-semantics
---*/
assert.sameValue(/\d/y.test("a1"), false, "sticky: no match when start is not a digit");
assert.sameValue(/\d/.test("a1"), true, "non-sticky scans forward");
assert.sameValue(/\d/y.test("1a"), true, "sticky matches at position 0");
assert.sameValue(/abc/y.test("abc"), true);
assert.sameValue(/abc/y.test("xabc"), false, "sticky cannot skip a prefix");
assert.sameValue("abc".replace(/./y, "X"), "Xbc", "sticky replace at start");
assert.sameValue("123abc".match(/\d+/y)[0], "123", "sticky matches the run at start");
assert.sameValue("abc123".match(/\d+/y), null, "sticky null when start mismatches");
var g = [..."aaa".matchAll(/a/gy)];
assert.sameValue(g.length, 3, "global+sticky matches contiguous run");
var stopped = [..."aXa".matchAll(/a/gy)];
assert.sameValue(stopped.length, 1, "global+sticky stops at first gap");
