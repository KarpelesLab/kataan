/*---
description: Array.prototype read-only methods work on a plain-object array-like via call
esid: sec-array.prototype.slice
---*/
var al = { length: 3, 0: "a", 1: "b", 2: "c" };

// Generic application via .call to a plain object with a numeric length.
assert.sameValue(Array.prototype.slice.call(al).join(","), "a,b,c", "slice");
assert.sameValue(Array.prototype.slice.call(al, 1, 3).join(","), "b,c", "slice range");
assert.sameValue(Array.prototype.map.call(al, function (x) { return x.toUpperCase(); }).join(","), "A,B,C", "map");
assert.sameValue(Array.prototype.join.call(al, "-"), "a-b-c", "join");
assert.sameValue(Array.prototype.indexOf.call(al, "b"), 1, "indexOf");
assert.sameValue(Array.prototype.includes.call(al, "c"), true, "includes");

var nums = { length: 4, 0: 1, 1: 2, 2: 3, 3: 4 };
assert.sameValue(Array.prototype.filter.call(nums, function (x) { return x % 2 === 0; }).join(","), "2,4", "filter");
assert.sameValue(Array.prototype.reduce.call(nums, function (a, b) { return a + b; }, 0), 10, "reduce");
assert.sameValue(Array.prototype.some.call(nums, function (x) { return x > 3; }), true, "some");
assert.sameValue(Array.prototype.every.call(nums, function (x) { return x > 0; }), true, "every");
assert.sameValue(Array.prototype.find.call(nums, function (x) { return x > 2; }), 3, "find");

// forEach collects via its callback.
var seen = [];
Array.prototype.forEach.call(al, function (x) { seen.push(x); });
assert.sameValue(seen.join(","), "a,b,c", "forEach");

// A missing/absent length is treated as 0 (an empty array-like).
assert.sameValue(JSON.stringify(Array.prototype.slice.call({ a: 1 })), "[]", "no length -> empty");
assert.sameValue(JSON.stringify(Array.prototype.slice.call({ length: 0 })), "[]", "zero length");

// A missing index reads as undefined.
assert.sameValue(Array.prototype.join.call({ length: 3, 0: "x", 2: "z" }, ","), "x,,z", "sparse array-like");

// Real arrays (and arguments, which is array-like) are unaffected.
assert.sameValue(Array.prototype.slice.call([1, 2, 3, 4], 1, 3).join(","), "2,3", "real array");
assert.sameValue((function () { return Array.prototype.slice.call(arguments).join(","); })(7, 8, 9), "7,8,9", "arguments");
