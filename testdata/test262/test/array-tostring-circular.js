/*---
description: Array.prototype.toString/join render a self-reference as empty (no crash)
esid: sec-array.prototype.join
---*/
// A self-referential array renders the cycle as an empty string (and must not
// overflow the stack).
var a = [];
a.push(a);
assert.sameValue(a.toString(), "", "self-only array");
assert.sameValue(a.join(","), "", "self-only join");

// The cyclic element is empty; other elements render normally.
var b = [1, 2];
b.push(b);
b.push(3);
assert.sameValue(b.join(","), "1,2,,3", "self-reference element is empty");
assert.sameValue(b.toString(), "1,2,,3", "toString matches");

// A normal nested array still renders.
assert.sameValue([1, [2, 3], 4].join("-"), "1-2,3-4", "nested array");
assert.sameValue([1, [2, 3], 4].toString(), "1,2,3,4", "nested toString");
