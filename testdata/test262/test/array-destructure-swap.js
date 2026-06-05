/*---
description: Array destructuring assignment and swaps
esid: sec-destructuring-assignment
---*/
var a = 1, b = 2, c = 3;
[a, b, c] = [c, a, b];
assert.sameValue(a + "," + b + "," + c, "3,1,2", "rotate");
var x = 10, y = 20;
[x, y] = [y, x];
assert.sameValue(x + "," + y, "20,10", "swap");
var arr = [1, 2, 3, 4, 5];
var [first, , third, ...rest] = arr;
assert.sameValue(first, 1);
assert.sameValue(third, 3, "skip with hole");
assert.sameValue(rest.join(","), "4,5");
var [p = 100, q = 200] = [1];
assert.sameValue(p, 1);
assert.sameValue(q, 200, "default");
var nested = [[1, 2], [3, 4]];
var [[m, n], [o, r]] = nested;
assert.sameValue(m + n + o + r, 10);
var obj = { a: 1, b: 2 };
var { a: aa, b: bb } = obj;
assert.sameValue(aa + bb, 3);
function getCoords() { return [10, 20]; }
var [lat, lng] = getCoords();
assert.sameValue(lat * 100 + lng, 1020);
