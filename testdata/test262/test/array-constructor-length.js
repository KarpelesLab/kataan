/*---
description: new Array(n) sets the length; invalid lengths throw RangeError
esid: sec-array-len
---*/
// A single number argument is the array length (a sparse array; holes read as
// undefined here).
assert.sameValue(new Array(5).length, 5, "new Array(5) has length 5");
assert.sameValue(new Array(0).length, 0, "new Array(0)");
assert.sameValue(new Array(3).fill(7).join(","), "7,7,7", "new Array(3).fill(7)");

// Multiple arguments are the elements.
assert.sameValue(new Array(1, 2, 3).join(","), "1,2,3", "elements");
// A single non-number argument is the sole element (length 1).
assert.sameValue(new Array("x").length, 1, "new Array('x') is one element");
assert.sameValue(new Array("x")[0], "x", "the element");

// Invalid lengths throw RangeError (and must not crash).
function r(n) { try { new Array(n); return "ok"; } catch (e) { return e.name; } }
assert.sameValue(r(-1), "RangeError", "negative length");
assert.sameValue(r(2.5), "RangeError", "fractional length");
assert.sameValue(r(4294967296), "RangeError", "length >= 2^32");

// Array(...) without new behaves the same.
assert.sameValue(Array(4).length, 4, "Array(4) without new");
