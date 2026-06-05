/*---
description: Chained and destructuring assignment expressions
esid: sec-assignment-operators
---*/
var a, b, c;
a = b = c = 5;
assert.sameValue(a, 5);
assert.sameValue(b, 5);
assert.sameValue(c, 5);
var x = 1, y = 2;
[x, y] = [y, x];
assert.sameValue(x, 2, "swap via destructuring");
assert.sameValue(y, 1);
var obj = {};
obj.a = obj.b = 10;
assert.sameValue(obj.a, 10);
assert.sameValue(obj.b, 10);
var earr = [0, 0];
var ei = 0;
earr[ei] = ei = 1;
assert.sameValue(earr[0], 1, "computed member key resolved before RHS");
var nested = {};
({ p: nested.x, q: nested.y } = { p: 7, q: 8 });
assert.sameValue(nested.x, 7);
assert.sameValue(nested.y, 8);
