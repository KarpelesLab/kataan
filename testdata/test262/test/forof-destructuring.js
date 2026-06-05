/*---
description: for-of with destructuring over arrays, entries, and Maps
esid: sec-for-in-and-for-of-statements
---*/
var pairs = [[1, "a"], [2, "b"], [3, "c"]];
var keys = [];
for (var [k, v] of pairs) keys.push(k + v);
assert.sameValue(keys.join(","), "1a,2b,3c", "array destructuring in for-of");
var sum = 0;
for (var [i, val] of [10, 20, 30].entries()) sum += i * val;
assert.sameValue(sum, 0 * 10 + 1 * 20 + 2 * 30, "entries destructuring");
var m = new Map([["x", 1], ["y", 2]]);
var out = [];
for (var [mk, mv] of m) out.push(mk + "=" + mv);
assert.sameValue(out.join(","), "x=1,y=2");
var nums = [];
for (var { n } of [{ n: 1 }, { n: 2 }]) nums.push(n);
assert.sameValue(nums.join(","), "1,2", "object destructuring in for-of");
