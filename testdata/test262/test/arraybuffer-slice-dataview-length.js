/*---
description: ArrayBuffer.prototype.slice copies bytes; DataView honors an explicit length
esid: sec-arraybuffer.prototype.slice
---*/
// Write bytes via a DataView (which shares the buffer), slice, read back via a DataView.
var b = new ArrayBuffer(8);
var dv = new DataView(b);
for (var i = 0; i < 8; i++) dv.setUint8(i, i * 10);

var s = b.slice(2, 5);
assert.sameValue(s.byteLength, 3, "slice length");
var sdv = new DataView(s);
assert.sameValue(sdv.getUint8(0), 20, "slice byte 0");
assert.sameValue(sdv.getUint8(1), 30, "slice byte 1");
assert.sameValue(sdv.getUint8(2), 40, "slice byte 2");

// The slice is an independent copy.
dv.setUint8(2, 99);
assert.sameValue(sdv.getUint8(0), 20, "slice unaffected by source mutation");
assert.sameValue(s === b, false, "slice is a new buffer");

// Negative and default bounds.
assert.sameValue(b.slice(-3).byteLength, 3, "slice(-3)");
assert.sameValue(b.slice(0).byteLength, 8, "slice(0) is a full copy");
assert.sameValue(new DataView(b.slice(-3)).getUint8(0), 50, "negative slice content");

// DataView honors an explicit byteLength; absent it spans the rest of the buffer.
var view = new DataView(b, 2, 4);
assert.sameValue(view.byteLength, 4, "explicit length");
assert.sameValue(view.byteOffset, 2, "byteOffset");
assert.sameValue(new DataView(b, 4).byteLength, 4, "rest-of-buffer length");
assert.sameValue(new DataView(b).byteLength, 8, "full buffer");
