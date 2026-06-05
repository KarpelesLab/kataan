/*---
description: Array.from with various iterables and mapping
esid: sec-array.from
---*/
assert.sameValue(Array.from([1, 2, 3]).join(","), "1,2,3");
assert.sameValue(Array.from("abc").join(","), "a,b,c");
assert.sameValue(Array.from(new Set([1, 1, 2, 3])).join(","), "1,2,3");
assert.sameValue(Array.from(new Map([["a", 1]])).length, 1);
assert.sameValue(Array.from({ length: 3 }).length, 3);
assert.sameValue(Array.from({ length: 3 }, function (_, i) { return i * 2; }).join(","), "0,2,4");
assert.sameValue(Array.from([1, 2, 3], function (x) { return x * x; }).join(","), "1,4,9");
function* gen() { yield 1; yield 2; yield 3; }
assert.sameValue(Array.from(gen()).join(","), "1,2,3");
assert.sameValue(Array.from("hello", function (c) { return c.toUpperCase(); }).join(""), "HELLO");
assert.sameValue(Array.from([1, 2, 3].entries(), function (e) { return e[0] + ":" + e[1]; }).join(","), "0:1,1:2,2:3");
assert.sameValue(Array.from(new Set("aabbc")).join(""), "abc");
assert.sameValue(Array.from([]).length, 0);
assert.sameValue(Array.from({ 0: "x", 1: "y", length: 2 }).join(","), "x,y", "array-like");
