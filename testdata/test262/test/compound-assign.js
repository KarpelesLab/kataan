/*---
description: Compound assignment operators on properties and elements
esid: sec-assignment-operators
---*/
var o = { n: 10 };
o.n += 5; assert.sameValue(o.n, 15);
o.n *= 2; assert.sameValue(o.n, 30);
o.n -= 10; assert.sameValue(o.n, 20);
o.n **= 2; assert.sameValue(o.n, 400);
var a = [1, 2, 3];
a[1] += 10; assert.sameValue(a[1], 12);
var s = "x"; s += "y" + "z"; assert.sameValue(s, "xyz");
