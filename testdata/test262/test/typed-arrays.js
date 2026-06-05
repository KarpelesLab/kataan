/*---
description: Typed arrays — construction, element coercion, length/byteLength
esid: sec-typedarray-objects
---*/
// Construction from a length (zero-filled).
var a = new Uint8Array(3);
assert.sameValue(a.length, 3, "length");
assert.sameValue(a[0], 0, "zero-filled");
a[0] = 255;
assert.sameValue(a[0], 255, "set/get");
// Unsigned 8-bit wraps modulo 256.
a[1] = 256;
assert.sameValue(a[1], 0, "Uint8 wraps 256 -> 0");
a[2] = -1;
assert.sameValue(a[2], 255, "Uint8 wraps -1 -> 255");
// Signed 8-bit.
var i8 = new Int8Array(1);
i8[0] = 200;
assert.sameValue(i8[0], -56, "Int8 200 -> -56");
// Uint8Clamped clamps instead of wrapping.
var c = new Uint8ClampedArray([300, -5, 100]);
assert.sameValue(c.join(","), "255,0,100", "Uint8Clamped clamps");
// 16-bit wrapping.
assert.sameValue(new Int16Array([70000])[0], 4464, "Int16 wraps");
// Floats keep precision (Float64) / narrow (Float32).
var f = new Float64Array(1);
f[0] = 3.14;
assert.sameValue(f[0], 3.14, "Float64 keeps value");
// Construction from an array-like.
var fromArr = new Uint8Array([1, 2, 3]);
assert.sameValue(fromArr.length, 3, "from array length");
assert.sameValue(fromArr[1], 2, "from array element");
// byteLength and BYTES_PER_ELEMENT.
assert.sameValue(new Uint16Array(4).byteLength, 8, "Uint16 byteLength");
assert.sameValue(new Float64Array(2).byteLength, 16, "Float64 byteLength");
assert.sameValue(new Uint8Array(1).BYTES_PER_ELEMENT, 1, "BYTES_PER_ELEMENT");
// instanceof and array-style methods.
assert.sameValue(fromArr instanceof Uint8Array, true, "instanceof");
assert.sameValue(fromArr.map(function (x) { return x * 2; }).join(","), "2,4,6", "map");
var filled = new Uint8Array(3);
filled.fill(7);
assert.sameValue(filled[2], 7, "fill");
assert.sameValue(Array.from(new Uint8Array([5, 6, 7])).join(","), "5,6,7", "Array.from");
assert.sameValue([...new Uint8Array([8, 9])].join(","), "8,9", "spread");
