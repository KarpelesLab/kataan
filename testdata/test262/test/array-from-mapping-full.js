/*---
description: Array.from with mapping over various sources
esid: sec-array.from
---*/
assert.sameValue(Array.from([1, 2, 3], function (x) { return x + 10; }).join(","), "11,12,13");
assert.sameValue(Array.from("abc", function (c, i) { return c + i; }).join(","), "a0,b1,c2");
var m = new Map([["a", 1], ["b", 2]]);
assert.sameValue(Array.from(m, function (entry) { return entry[0] + entry[1]; }).join(","), "a1,b2");
var s = new Set([1, 2, 3]);
assert.sameValue(Array.from(s, function (x) { return x * x; }).join(","), "1,4,9");
assert.sameValue(Array.from({ length: 3 }, function (_, i) { return i; }).join(","), "0,1,2");
assert.sameValue(Array.from([1, 2, 3].entries(), function (e) { return e[0] + ":" + e[1]; }).join(","), "0:1,1:2,2:3");
assert.sameValue(Array.from(new Set("hello")).join(""), "helo", "Set dedups string chars");
assert.sameValue(Array.from([]).length, 0);
