/*---
description: TypedArray.prototype.set copies a source array in, coercing elements
features: [TypedArray]
---*/
// Copy into a typed array at an offset.
var base = new Int32Array(5);
base.set([10, 20], 1);
assert.sameValue(base.join(","), "0,10,20,0,0", "set at offset 1");

// Source values are coerced to the element type (Int8 wraps).
var i8 = new Int8Array(3);
i8.set([300, -5], 0);
assert.sameValue(i8[0], 44, "300 wraps to 44 in Int8");
assert.sameValue(i8[1], -5, "-5 stays");

// Default offset is 0.
var f = new Float64Array(4);
f.set([1.5, 2.5]);
assert.sameValue(f.join(","), "1.5,2.5,0,0", "default offset 0");

// Writing past the end is a RangeError.
var threw = false;
try { new Int32Array(2).set([1, 2, 3]); } catch (e) { threw = e instanceof RangeError; }
assert.sameValue(threw, true, "out-of-bounds set throws RangeError");

// Plain arrays do not have `set`.
assert.sameValue(typeof [1, 2, 3].set, "undefined", "Array has no set method");
