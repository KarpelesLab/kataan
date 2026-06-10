/*---
description: a regex match result is an Array with index/input/groups as enumerable own props
features: [regexp-named-groups]
---*/
var m = "abc".match(/b/);
assert.sameValue(Array.isArray(m), true, "match result is a real Array");
assert.sameValue(m[0], "b", "whole match");
assert.sameValue(m.length, 1, "length is the capture count");
assert.sameValue(m.index, 1, "index");
assert.sameValue(m.input, "abc", "input");
assert.sameValue(m.groups, undefined, "groups undefined with no named groups");
assert.sameValue(JSON.stringify(m), '["b"]', "stringifies as an array (named props ignored)");
// Enumerable own keys: numbered captures then index/input/groups (length is intrinsic, not listed).
assert.sameValue(Object.keys(m).join(","), "0,index,input,groups", "enumerable keys");

// Capture groups, named groups, and array methods.
var e = /(\w)(\w)/.exec("ab");
assert.sameValue(e.length, 3, "whole + two captures");
assert.sameValue(JSON.stringify(e), '["ab","a","b"]', "exec result is an array");
assert.sameValue(e.map(function (x) { return x.toUpperCase(); }).join(","), "AB,A,B", "array methods work");

var n = "2024-01".match(/(?<y>\d+)-(?<mo>\d+)/);
assert.sameValue(n.groups.y, "2024", "named group y");
assert.sameValue(n.groups.mo, "01", "named group mo");
assert.sameValue(Object.keys(n).join(","), "0,1,2,index,input,groups", "named-match keys");

// Destructuring and spread treat it as an array of captures.
var parts = "xy".match(/(.)(.)/);
assert.sameValue([parts[0], parts[1], parts[2]].join(","), "xy,x,y", "indexed access");
assert.sameValue([...parts].join(","), "xy,x,y", "spread");
