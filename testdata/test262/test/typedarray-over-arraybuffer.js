/*---
description: constructing a typed array over an ArrayBuffer (length, byteOffset, byte decode)
esid: sec-typedarray-buffer-byteoffset-length
features: [TypedArray, DataView]
---*/
// new T(buffer): length is byteLength / BYTES_PER_ELEMENT, offset 0.
assert.sameValue(new Int32Array(new ArrayBuffer(8)).length, 2, "Int32 over 8 bytes -> 2");
assert.sameValue(new Int16Array(new ArrayBuffer(8)).length, 4, "Int16 over 8 bytes -> 4");
assert.sameValue(new Uint8Array(new ArrayBuffer(8)).length, 8, "Uint8 over 8 bytes -> 8");
assert.sameValue(new Float64Array(new ArrayBuffer(16)).length, 2, "Float64 over 16 bytes -> 2");
assert.sameValue(new Int32Array(new ArrayBuffer(8)).byteOffset, 0, "default byteOffset 0");

// new T(buffer, byteOffset): the offset is honored and the length spans the remainder.
var v = new Int32Array(new ArrayBuffer(16), 4);
assert.sameValue(v.byteOffset, 4, "byteOffset honored");
assert.sameValue(v.length, 3, "length spans (16-4)/4");

// new T(buffer, byteOffset, length): explicit length.
var w = new Int32Array(new ArrayBuffer(16), 4, 2);
assert.sameValue(w.length, 2, "explicit length");
assert.sameValue(w.byteOffset, 4, "explicit offset");
assert.sameValue(w.byteLength, 8, "byteLength = length * BYTES_PER_ELEMENT");

// Element values are decoded (little-endian) from the buffer bytes.
var b = new ArrayBuffer(8);
var dv = new DataView(b);
dv.setInt32(0, 42, true);
dv.setInt32(4, 99, true);
var ta = new Int32Array(b);
assert.sameValue(ta[0], 42, "decoded element 0");
assert.sameValue(ta[1], 99, "decoded element 1");

var fb = new ArrayBuffer(8);
new DataView(fb).setFloat64(0, 3.5, true);
assert.sameValue(new Float64Array(fb)[0], 3.5, "float decode");

var sb = new ArrayBuffer(2);
new DataView(sb).setInt8(0, -5);
assert.sameValue(new Int8Array(sb)[0], -5, "signed int8 decode");

// Construction from an array or a length is unaffected; plain arrays have offset 0.
assert.sameValue(new Uint8Array([1, 2, 3]).join(","), "1,2,3", "from array");
assert.sameValue(new Int32Array(3).length, 3, "from length");
assert.sameValue(new Uint8Array([1, 2, 3]).byteOffset, 0, "plain byteOffset 0");
