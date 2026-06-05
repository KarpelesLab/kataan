/*---
description: ArrayBuffer and DataView read/write with endianness and signedness
esid: sec-dataview-objects
---*/
var b = new ArrayBuffer(8);
assert.sameValue(b.byteLength, 8, "ArrayBuffer byteLength");
var v = new DataView(b);
assert.sameValue(v.byteLength, 8, "DataView byteLength");
assert.sameValue(v.buffer, b, "DataView buffer");
// Int32 round-trip and signedness.
v.setInt32(0, 42);
assert.sameValue(v.getInt32(0), 42, "Int32 round-trip");
v.setInt32(0, -1);
assert.sameValue(v.getInt32(0), -1, "Int32 negative");
assert.sameValue(v.getUint32(0), 4294967295, "same bytes as Uint32");
// 8-bit.
v.setUint8(0, 255);
assert.sameValue(v.getUint8(0), 255, "Uint8");
assert.sameValue(v.getInt8(0), -1, "Int8 of 0xFF");
// 16-bit with endianness.
v.setInt16(0, 1000, true);
assert.sameValue(v.getInt16(0, true), 1000, "Int16 little-endian");
assert.sameValue(v.getInt16(0, false), -6141, "Int16 big-endian reads byte-swapped");
// Floats.
v.setFloat64(0, 3.14159);
assert.sameValue(v.getFloat64(0), 3.14159, "Float64 round-trip");
v.setFloat32(0, 1.5);
assert.sameValue(v.getFloat32(0), 1.5, "Float32 round-trip");
// Wrapping on set (Int8 truncates).
v.setInt8(0, 300);
assert.sameValue(v.getInt8(0), 44, "Int8 set wraps (300 & 0xFF = 44)");
// A byte offset on the DataView.
var v2 = new DataView(b, 2);
v2.setInt32(0, 7);
assert.sameValue(v2.getInt32(0), 7, "offset DataView reads its own region");
assert.sameValue(v.getInt32(2), 7, "the base view sees it at offset 2");
