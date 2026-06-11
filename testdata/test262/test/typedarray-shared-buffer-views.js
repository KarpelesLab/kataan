/*---
description: typed-array views over a shared ArrayBuffer observe each other's (and a DataView's) writes
features: [TypedArray, DataView, ArrayBuffer]
---*/
// Two Uint8Array views over one buffer see each other's writes (both indexed and via methods).
var buf = new ArrayBuffer(8);
var a = new Uint8Array(buf);
var b = new Uint8Array(buf);
a[0] = 42;
assert.sameValue(b[0], 42, "sibling indexed read");
assert.sameValue(b.join(","), "42,0,0,0,0,0,0,0", "sibling method read");
b[7] = 9;
assert.sameValue(a[7], 9, "write through the other view");

// A DataView write is visible to a typed-array view.
var dv = new DataView(buf);
dv.setUint8(1, 99);
assert.sameValue(a[1], 99, "DataView write -> typed array indexed read");
assert.sameValue([...a].join(","), "42,99,0,0,0,0,0,9", "DataView write -> spread");

// And a typed-array write is visible through the DataView (the write direction).
a[2] = 7;
assert.sameValue(dv.getUint8(2), 7, "typed array write -> DataView read");

// Different element types over the same buffer share the bytes (little-endian).
var buf2 = new ArrayBuffer(4);
var u32 = new Uint32Array(buf2);
var u8 = new Uint8Array(buf2);
u32[0] = 0x04030201;
assert.sameValue(u8[0], 1, "u32 write -> u8[0] (LE)");
assert.sameValue(u8[1], 2, "u8[1]");
assert.sameValue(u8[2], 3, "u8[2]");
assert.sameValue(u8[3], 4, "u8[3]");
assert.sameValue(u8.reduce(function (s, x) { return s + x; }, 0), 10, "u8 method sees the bytes");

// A view's own .buffer is shared with a later sibling created over it.
var st = new Uint8Array(4);
st[0] = 1;
var shared = st.buffer;
var v = new Uint8Array(shared);
v[1] = 88;
assert.sameValue(st[1], 88, "write via sibling over the materialized buffer");
new DataView(shared).setUint8(2, 55);
assert.sameValue(st[2], 55, "DataView over the materialized buffer");

// An ordinary array and a standalone typed array are completely unaffected.
var ra = [1, 2, 3];
ra[1] = 99;
assert.sameValue(ra.join(","), "1,99,3", "ordinary array");
var lone = new Int16Array(3);
lone[0] = 1000;
assert.sameValue(lone.join(","), "1000,0,0", "standalone typed array");
