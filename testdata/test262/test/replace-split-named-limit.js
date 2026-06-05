/*---
description: replace function receives groups; regex split honors limit and empty matches
esid: sec-string.prototype.replace
---*/
var r = "2024-06".replace(/(?<y>\d+)-(?<m>\d+)/, function (m, y, mo, off, str, groups) {
  return groups.y + "/" + groups.m;
});
assert.sameValue(r, "2024/06", "replace fn receives the named groups object");
var args;
"abc".replace(/b/, function (m, off, str) { args = [m, off, str]; return m; });
assert.sameValue(args[0], "b");
assert.sameValue(args[1], 1, "offset");
assert.sameValue(args[2], "abc", "whole string");
assert.sameValue("a1b2c3".split(/(\d)/, 3).join("|"), "a|1|b", "regex split honors limit");
assert.sameValue("a,b,c,d".split(/,/, 2).join("|"), "a|b", "string-regex limit");
assert.sameValue("abc".split(/(?:)/).join(","), "a,b,c", "empty regex: no trailing empty");
assert.sameValue("a1b2c3".split(/(\d)/).length, 7, "capture split keeps trailing empty");
assert.sameValue("a,b,".split(/,/).join("|"), "a|b|", "trailing separator keeps empty");
assert.sameValue("camelCase".split(/(?=[A-Z])/).join("-"), "camel-Case", "lookahead split unaffected");
assert.sameValue("".split(/x/).length, 1, "split of empty string");
assert.sameValue("a,b,c".split(/,/, 0).length, 0, "limit 0");
