/*---
description: typed array .buffer returns a backing ArrayBuffer; ArrayBuffer/DataView instanceof
esid: sec-get-%typedarray%.prototype.buffer
features: [TypedArray, DataView]
---*/
// A view over a buffer reports that exact buffer.
var b = new ArrayBuffer(8);
assert.sameValue(new Int32Array(b).buffer, b, "view .buffer is the source buffer");

// An array/length-constructed typed array materializes a backing buffer.
assert.sameValue(new Int32Array(4).buffer.byteLength, 16, "materialized byteLength");
assert.sameValue(new Uint8Array([1, 2, 3]).buffer instanceof ArrayBuffer, true, ".buffer is an ArrayBuffer");

// The materialized buffer holds the encoded element bytes (little-endian).
var ta = new Int32Array([42, 99]);
var dv = new DataView(ta.buffer);
assert.sameValue(dv.getInt32(0, true), 42, "encoded element 0");
assert.sameValue(dv.getInt32(4, true), 99, "encoded element 1");
assert.sameValue(new DataView(new Uint8Array([1, 2, 3, 4]).buffer).byteLength, 4, "DataView over .buffer");
assert.sameValue(new DataView(new Float64Array([3.5]).buffer).getFloat64(0, true), 3.5, "float encode/decode");

// Uint8Clamped values are stored clamped, then encoded.
var c = new Uint8ClampedArray([300, 128]);
assert.sameValue(new DataView(c.buffer).getUint8(0), 255, "clamped high");
assert.sameValue(new DataView(c.buffer).getUint8(1), 128, "clamped mid");

// .buffer is cached (same identity on repeated access).
var t2 = new Uint8Array([1, 2]);
assert.sameValue(t2.buffer, t2.buffer, "buffer identity stable");

// instanceof for ArrayBuffer / DataView (and the negatives).
assert.sameValue(new ArrayBuffer(8) instanceof ArrayBuffer, true, "ArrayBuffer instanceof");
assert.sameValue(new DataView(new ArrayBuffer(8)) instanceof DataView, true, "DataView instanceof");
assert.sameValue(new Uint8Array([1]) instanceof ArrayBuffer, false, "typed array is not an ArrayBuffer");
assert.sameValue(new Uint8Array([1]) instanceof DataView, false, "typed array is not a DataView");
assert.sameValue([] instanceof ArrayBuffer, false, "array is not an ArrayBuffer");
