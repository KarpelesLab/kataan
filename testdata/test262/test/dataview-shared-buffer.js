/*---
description: multiple DataViews over one ArrayBuffer share its bytes (offset/endianness)
features: [DataView, ArrayBuffer]
---*/
var buf = new ArrayBuffer(16);
var a = new DataView(buf);
var b = new DataView(buf);

// A write through one view is visible through another (same backing bytes).
a.setInt32(0, 0x12345678);
assert.sameValue(b.getInt32(0), 0x12345678, "Int32 shared");
a.setUint8(4, 0xFF);
assert.sameValue(b.getUint8(4), 255, "byte shared");

// A view with a byteOffset addresses the same underlying bytes.
var c = new DataView(buf, 8);
c.setInt16(0, 0x0102);
assert.sameValue(a.getInt16(8), 0x0102, "offset view aliases base[8]");

// Endianness: little-endian writes the low byte first.
a.setUint16(10, 0x0102, true);
assert.sameValue(a.getUint8(10), 2, "LE low byte");
assert.sameValue(a.getUint8(11), 1, "LE high byte");
// Big-endian (default) writes high byte first.
a.setUint16(12, 0x0304);
assert.sameValue(a.getUint8(12), 3, "BE high byte");
assert.sameValue(a.getUint8(13), 4, "BE low byte");

assert.sameValue(buf.byteLength, 16, "buffer length");
