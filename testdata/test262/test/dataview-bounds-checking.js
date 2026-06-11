/*---
description: DataView get/set bounds-check the access and throw RangeError out of range
esid: sec-dataview.prototype.getint32
---*/
function throwsRange(fn) { try { fn(); return false; } catch (e) { return e instanceof RangeError; } }

var buf = new ArrayBuffer(16);
var dv = new DataView(buf);

// In-bounds access works; the last 4-byte slot is offset 12.
dv.setInt32(0, 42, true);
assert.sameValue(dv.getInt32(0, true), 42, "in-bounds");
dv.setInt32(12, 99, true);
assert.sameValue(dv.getInt32(12, true), 99, "last valid Int32 slot");

// An access running past the end is a RangeError, for both get and set.
assert.sameValue(throwsRange(function () { dv.getInt32(13); }), true, "getInt32 13");
assert.sameValue(throwsRange(function () { dv.getInt32(16); }), true, "getInt32 16");
assert.sameValue(throwsRange(function () { dv.setInt32(13, 0); }), true, "setInt32 13");
assert.sameValue(throwsRange(function () { dv.setInt8(16, 0); }), true, "setInt8 16");

// A negative offset is a RangeError.
assert.sameValue(throwsRange(function () { dv.getInt8(-1); }), true, "negative get");
assert.sameValue(throwsRange(function () { dv.setInt8(-1, 0); }), true, "negative set");

// An out-of-bounds write must NOT silently grow the buffer.
throwsRange(function () { dv.setInt32(15, 0); });
assert.sameValue(dv.byteLength, 16, "byteLength unchanged after OOB attempt");

// The last byte is valid for Int8.
dv.setInt8(15, 7);
assert.sameValue(dv.getInt8(15), 7, "last Int8 byte");

// A DataView with an explicit offset/length bounds within its own window.
var dv2 = new DataView(buf, 4, 8);
dv2.setInt32(0, 1, true);
assert.sameValue(dv2.getInt32(0, true), 1, "sub-view in-bounds");
assert.sameValue(throwsRange(function () { dv2.getInt32(5); }), true, "sub-view OOB");
assert.sameValue(throwsRange(function () { dv2.getInt8(8); }), true, "sub-view past length");

// Float64 needs 8 bytes.
assert.sameValue(throwsRange(function () { dv.getFloat64(9); }), true, "float64 OOB");
var dv3 = new DataView(new ArrayBuffer(16));
dv3.setFloat64(8, 1.5, true);
assert.sameValue(dv3.getFloat64(8, true), 1.5, "float64 last slot");
