/*---
description: Spread arguments into new
esid: sec-new-operator
---*/
function Vec(a, b, c) { this.sum = a + b + c; }
var parts = [1, 2, 3];
var v = new Vec(...parts);
assert.sameValue(v.sum, 6);
