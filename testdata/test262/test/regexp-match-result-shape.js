/*---
description: a regex match result exposes index/input/groups (enumerable) and a readable, non-enumerable length
features: [regexp-named-groups]
---*/
var m = "abc".match(/b/);
assert.sameValue(m[0], "b", "whole match");
assert.sameValue(m.length, 1, "length (one group: the whole match)");
assert.sameValue(m.index, 1, "index");
assert.sameValue(m.input, "abc", "input");
assert.sameValue(m.groups, undefined, "groups is undefined with no named groups");
// Enumerable own keys: the numbered captures plus index/input/groups (NOT length).
assert.sameValue(Object.keys(m).join(","), "0,index,input,groups", "enumerable keys");

// Capture groups and a named-groups object.
var e = /(\w)(\w)/.exec("ab");
assert.sameValue(e.length, 3, "whole + two capture groups");
assert.sameValue(e[1] + e[2], "ab", "captures");

var n = "2024-01".match(/(?<y>\d+)-(?<mo>\d+)/);
assert.sameValue(n.groups.y, "2024", "named group y");
assert.sameValue(n.groups.mo, "01", "named group mo");
assert.sameValue(Object.keys(n).join(","), "0,1,2,index,input,groups", "named-match keys");
