/*---
description: Generators with parameters and accumulated state (bounded)
esid: sec-generator-function-definitions
---*/
function* take(arr, n) { for (var i = 0; i < n && i < arr.length; i++) yield arr[i]; }
assert.sameValue([...take([10, 20, 30, 40], 2)].join(","), "10,20");
function* counter(limit) { var c = 0; while (c < limit) yield c++; }
var g = counter(101);
assert.sameValue(g.next().value, 0);
assert.sameValue(g.next().value, 1);
assert.sameValue(g.next().value, 2);
function* fibGen(count) { var a = 0, b = 1; for (var i = 0; i < count; i++) { yield a; var t = a + b; a = b; b = t; } }
assert.sameValue([...fibGen(7)].join(","), "0,1,1,2,3,5,8");
